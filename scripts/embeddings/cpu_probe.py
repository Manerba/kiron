"""Unprivileged halves of smoke-cpu.py; never run against production ports."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
from types import SimpleNamespace

PORT = 18096
MODEL = 'mankei-326m-embedder'
PROFILE = 'kiron-mankei-dense-v1'
TEXT = 'Berlin ist die Hauptstadt von Deutschland.'
SHORT_TEXT = 'x'
ROLES = ('search_query', 'search_document')


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write('\n')


def imports(report):
    root = report / 'source/services'
    for name in ('kiron-common', 'kiron-proxy', 'kiron-embeddings'):
        sys.path.insert(0, str(root / name))


def check_identity(report):
    expected = json.loads((report / 'prepared.json').read_text())
    if os.geteuid() != expected['uid'] or os.getegid() != expected['gid'] or os.getgroups():
        raise RuntimeError('probe must run as dedicated foreign UID without supplementary groups')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('probe requires CPU-only offline environment')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    if status['NoNewPrivs'].strip() != '1':
        raise RuntimeError('probe requires no_new_privs')


async def native(report):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('CUDA must be invisible')
    cache = report / 'cache/models--keyvan-ai--Mankei-326M-Embedder'
    if not os.statvfs(cache).f_flag & os.ST_RDONLY:
        raise RuntimeError('model cache must be mounted read-only')
    import main as service
    import native_api
    from native_runtime import generation
    from token_usage import ForwardUsage
    import uvicorn
    from contextlib import nullcontext
    # Evidence-only observer. Counted tensors and encode behavior are unchanged.
    observe = ForwardUsage.observe
    trace = report / 'work/forward-inputs.ndjson'
    def observed(counter, features):
        value = {name: features[name].detach().cpu().tolist() for name in ('input_ids', 'attention_mask')}
        with trace.open('a') as output:
            output.write(json.dumps(value) + '\n')
        observe(counter, features)
    ForwardUsage.observe = observed
    async with service.lifespan(service.app):
        await service.model_worker.load(MODEL)
        references = {}
        for label, text in (('long', TEXT), ('short', SHORT_TEXT)):
            references[label] = {}
            for role in ROLES:
                before = len(trace.read_text().splitlines()) if trace.exists() else 0
                result = await service.model_worker.encode(MODEL, [text], role)
                records = [json.loads(line) for line in trace.read_text().splitlines()[before:]]
                actual = sum(sum(row) for record in records for row in record['attention_mask'])
                if actual != result.prompt_eval_count or actual <= 0:
                    raise RuntimeError('forward tensor evidence disagrees with native token count')
                references[label][role] = {'text': text, 'embeddings': result.embeddings,
                    'prompt_eval_count': result.prompt_eval_count, 'forward_batches': records}
        references['batch'] = {}
        for role in ROLES:
            texts = batch_texts(role)
            before = len(trace.read_text().splitlines())
            result = await service.model_worker.encode(MODEL, texts, role)
            records = [json.loads(line) for line in trace.read_text().splitlines()[before:]]
            reference = {'texts': texts, 'embeddings': result.embeddings,
                'prompt_eval_count': result.prompt_eval_count, 'forward_batches': records}
            batch_check(role, references, reference, result.embeddings, [0, 1],
                        result.prompt_eval_count, records)
            references['batch'][role] = reference
        snapshot = service.model_worker.snapshot()
        if snapshot['device'] != 'cpu' or MODEL not in snapshot['verified_artifacts']:
            raise RuntimeError('CPU artifact identity was not established')
        config = uvicorn.Config(service.app, host='127.0.0.1', port=PORT, lifespan='off',
            log_level='warning', timeout_keep_alive=1, timeout_graceful_shutdown=5)
        class ProbeServer(uvicorn.Server):
            # Own this test subprocess' stop signal so Uvicorn cannot re-raise
            # SIGTERM before the surrounding worker lifespan has completed.
            def capture_signals(self):
                return nullcontext()
        server = ProbeServer(config)
        signal.signal(signal.SIGTERM, lambda *_: setattr(server, 'should_exit', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(server, 'should_exit', True))
        serving = asyncio.create_task(server.serve())
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError('native listener failed')
            await asyncio.sleep(.02)
        write(report / 'work/native-ready.json', {'pid': os.getpid(), 'uid': os.geteuid(),
            'gid': os.getegid(), 'groups': os.getgroups(), 'port': PORT, 'device': 'cpu',
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'cache_read_only': True,
            'generation': generation(snapshot), 'service_revision': native_api.revision(),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'references': references})
        await serving
    write(report / 'work/native-stopped.json', {'worker_alive': service.model_worker.snapshot()['worker_thread_alive']})


def f32(values):
    return struct.pack('<' + 'f' * len(values), *values)


def batch_texts(role):
    return [TEXT, SHORT_TEXT] if role == 'search_query' else [SHORT_TEXT, TEXT]


def batch_check(role, singles, reference, vectors, indexes, tokens, records):
    """Require ordered native byte parity and actual unpadded per-input token counts."""
    assert role in ROLES and reference['texts'] == batch_texts(role)
    assert indexes == [0, 1] and len(vectors) == len(reference['embeddings']) == 2
    labels = ('long', 'short') if role == 'search_query' else ('short', 'long')
    expected_counts = [singles[label][role]['prompt_eval_count'] for label in labels]
    # These two deliberately short inputs must reach one padded native batch.
    assert len(records) == 1 and records == reference['forward_batches']
    ids, masks = records[0]['input_ids'], records[0]['attention_mask']
    assert len(ids) == len(masks) == 2 and len(ids[0]) == len(ids[1])
    assert all(len(row) == len(mask) and all(type(x) is int and x in (0, 1) for x in mask)
               for row, mask in zip(ids, masks))
    counts = [sum(mask) for mask in masks]
    for label, row, mask in zip(labels, ids, masks):
        single_records = singles[label][role]['forward_batches']
        assert len(single_records) == 1
        single_ids, single_mask = single_records[0]['input_ids'], single_records[0]['attention_mask']
        assert len(single_ids) == len(single_mask) == 1
        assert [token for token, used in zip(row, mask) if used] == [
            token for token, used in zip(single_ids[0], single_mask[0]) if used]
    padding = sum(len(mask) - sum(mask) for mask in masks)
    assert counts == expected_counts and padding > 0
    assert tokens == reference['prompt_eval_count'] == sum(counts)
    differences, hashes = [], []
    for label, vector, expected in zip(labels, vectors, reference['embeddings']):
        single = singles[label][role]['embeddings'][0]
        assert len(vector) == len(expected) == len(single) == 960
        assert all(math.isfinite(x) for x in (*vector, *single))
        assert f32(vector) == f32(expected)
        # Padding changes CPU matrix shapes; it may alter last Float32 bits.
        assert all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(vector, single))
        differences.append(max(abs(a - b) for a, b in zip(vector, single)))
        hashes.append(hashlib.sha256(f32(vector)).hexdigest())
    return {'role': role, 'encoding': 'float', 'tokens': tokens, 'batch_size': 2,
        'input_characters': [len(text) for text in batch_texts(role)], 'dimensions': 960,
        'vector_float32_sha256': hashes, 'per_input_tokens': counts,
        'padding_tokens_excluded': padding, 'single_vector_max_abs_delta': differences}


def candidate_capability(deployment, implementation, reference):
    from datetime import datetime, timezone
    from kiron_common.local_inference import Capability, CapabilityEvidence, CapabilityStatus, ParameterConstraint
    evidence = CapabilityEvidence(implementation.provider_revision, deployment.artifact_identity.fingerprint,
        deployment.artifact_identity.sha256, None, None, implementation.parser_revision,
        deployment.configuration_fingerprint, reference, datetime.now(timezone.utc))
    return Capability(CapabilityStatus.SUPPORTED, {
        'profiles': ParameterConstraint(allowed_values=(PROFILE,)),
        'roles': ParameterConstraint(allowed_values=ROLES),
        'dimensions': ParameterConstraint(allowed_values=(960,)),
        'devices': ParameterConstraint(allowed_values=('cpu',)),
        'encoding_formats': ParameterConstraint(allowed_values=('float', 'base64')),
        'max_batch_size': ParameterConstraint(minimum=1, maximum=2),
        'max_input_characters': ParameterConstraint(minimum=1, maximum=len(TEXT)),
    }, (evidence,))


def export_candidate(report, deployment, implementation, checks):
    """Review candidate only; final parent PASS/cleanup and independent pins remain required."""
    from kiron_common.local_inference import CapabilityName, CapabilitySet
    from kiron_common.model_catalog import BackendType
    from kiron_common.embedding_registry import MODEL_CATALOG
    from runtime_capabilities import encode_provider_evidence, decode_provider_evidence
    expected = {(role, encoding, (len(text),)) for role in ROLES
        for encoding, text in (('float', TEXT), ('float', SHORT_TEXT), ('base64', TEXT))}
    expected.update((role, 'float', tuple(map(len, batch_texts(role)))) for role in ROLES)
    def lengths(row):
        value = row['input_characters']
        return tuple(value) if isinstance(value, list) else (value,)
    def valid(row):
        count = row['batch_size']
        hashes = row['vector_float32_sha256'] if count == 2 else [row['vector_float32_sha256']]
        return (type(row['tokens']) is int and row['tokens'] > 0 and row['dimensions'] == 960
            and count == len(lengths(row)) and count in (1, 2) and len(hashes) == count
            and all(type(h) is str and len(h) == 64 and all(c in '0123456789abcdef' for c in h) for h in hashes)
            and (count == 1 or (len(row.get('per_input_tokens', ())) == 2
                and all(type(n) is int and n > 0 for n in row['per_input_tokens'])
                and sum(row['per_input_tokens']) == row['tokens']
                and type(row.get('padding_tokens_excluded')) is int and row['padding_tokens_excluded'] > 0
                and len(row.get('single_vector_max_abs_delta', ())) == 2
                and all(type(n) in (int, float) and math.isfinite(n) and n >= 0
                        for n in row['single_vector_max_abs_delta']))))
    if (len(checks) != len(expected)
            or {(row['role'], row['encoding'], lengths(row)) for row in checks} != expected
            or not all(valid(row) for row in checks)):
        raise ValueError('candidate needs the complete measured input/role/encoding matrix')
    cap = candidate_capability(deployment, implementation, 'isolated-cpu-candidate:' + report.name)
    typed = {deployment.id: CapabilitySet({CapabilityName.EMBEDDINGS: cap})}
    value = encode_provider_evidence(BackendType.KIRON_EMBEDDINGS, typed)
    decoded = decode_provider_evidence(value, BackendType.KIRON_EMBEDDINGS)
    if decoded != typed or not decoded[deployment.id].supports(CapabilityName.EMBEDDINGS, deployment, implementation):
        raise ValueError('candidate identity or codec roundtrip differs')
    path = report / 'work/capability-candidate.json'
    write(path, value)
    prepared = json.loads((report / 'prepared.json').read_text())
    provenance = {'scope': 'review candidate; requires overall PASS, cleanup and independent archive review',
        'production_capability_granted': False, 'deployment_id': deployment.id, 'profile_id': PROFILE,
        'catalog_digest': MODEL_CATALOG.catalog_digest,
        'capability_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'prepared_sha256': hashlib.sha256((report / 'prepared.json').read_bytes()).hexdigest(),
        'native_ready_sha256': hashlib.sha256((report / 'work/native-ready.json').read_bytes()).hexdigest(),
        'forward_inputs_sha256': hashlib.sha256((report / 'work/forward-inputs.ndjson').read_bytes()).hexdigest(),
        'source_pins': prepared['sources'], 'hf_revision': prepared['hf_revision'],
        'hf_artifacts': prepared['hf_artifacts'], 'implementation': asdict(implementation), 'checks': checks}
    provenance['measurement_scope'] = {
        'sdk_checks': 8, 'native_forward_batches': 14,
        'measured': 'per role: float long/short singles and ordered mixed batch; base64 long single',
        'derived': 'base64 short and batch2 combine separately measured input/batch and encoding behavior',
        'input_length_scope': '1..42 is a profile bound with measured endpoints, not every intermediate length',
        'single_batch_comparison': {'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-6},
        'public_native_batch_comparison': 'exact Float32 bytes in original input order'}
    write(report / 'work/capability-provenance.json', provenance)
    return provenance['capability_sha256']


async def sdk_probe(report):
    import base64
    import httpx
    import openai
    from kiron_common.embedding_registry import MODEL_CATALOG
    from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
    from kiron_common.local_inference import (CapabilityName, CapabilitySet,
        RuntimeGeneration, RuntimeImplementation, RuntimeTimeouts,
        build_resolver_snapshot)
    from kiron_common.model_catalog import BackendType
    from embedding_provider import KironEmbeddingProvider
    from openai_api import create_openai_api_app
    from runtime_composition import adapter_revision
    from runtime_service import RuntimeService, generation_key
    if openai.__version__ != '2.29.0':
        raise RuntimeError('unreviewed SDK version')
    ready = json.loads((report / 'work/native-ready.json').read_text())
    snapshot = build_resolver_snapshot(MODEL_CATALOG, ())
    model = snapshot.resolve(PROFILE + '.query')
    deployment = model.deployment
    implementation = RuntimeImplementation(ready['service_revision'], None, adapter_revision('embedding_provider.py'))
    # A test-only candidate declaration. It is never persisted as production
    # evidence; assertions below test the claim against real native observations.
    cap = candidate_capability(deployment, implementation, 'isolated-cpu-candidate:' + report.name)
    native_client = httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', trust_env=False,
                                     follow_redirects=False, timeout=90)
    resolver = SimpleNamespace(snapshot=lambda: resolved(snapshot))
    provider = KironEmbeddingProvider(client=native_client, resolver=resolver, implementation=implementation,
        catalog_digest=MODEL_CATALOG.catalog_digest, capabilities={deployment.id: CapabilitySet({CapabilityName.EMBEDDINGS: cap})})
    admission = AdmissionStore(report / 'admission', security=RuntimeSecurity(
        os.geteuid(), os.getegid(), frozenset({os.geteuid()})))
    def measure():
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return MemorySnapshot(0, int(values['MemAvailable'].split()[0]) * 1024, time.monotonic())
    generation = RuntimeGeneration(**ready['generation'])
    admission.reserve(operation_id='isolated-resident', owner='kiron-embeddings', generation=generation_key(generation),
        deployment_id=deployment.id, kind='load', gpu_bytes=0, host_bytes=ready['peak_rss_bytes'], measure=measure)
    admission.transition('isolated-resident', owner='kiron-embeddings', expected_generation=generation_key(generation), phase='resident')
    runtime = RuntimeService(resolver=resolver, providers={BackendType.KIRON_EMBEDDINGS: provider},
        admission=admission, measure=measure, timeouts=RuntimeTimeouts(90, 90, 90, 90, 180, 2, 2))
    class Records:
        async def add_request(self, *args, **kwargs):
            pass
        async def update_request(self, *args, **kwargs):
            pass
    app = create_openai_api_app(Records(), SimpleNamespace(validate_key=lambda key: {'fixture': True}
                                                          if key == 'isolated-cpu-key' else None))
    app.state.local_inference = runtime
    client = openai.AsyncOpenAI(api_key='isolated-cpu-key', base_url='http://asgi.invalid/v1', max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app), trust_env=False),
        _strict_response_validation=True)
    checks = []
    trace = report / 'work/forward-inputs.ndjson'
    try:
        models = await client.models.list()
        assert {item.id for item in models.data} == {PROFILE + '.query', PROFILE + '.document'}
        for label, text in (('long', TEXT), ('short', SHORT_TEXT)):
            for suffix, role in (('query', 'search_query'), ('document', 'search_document')):
                reference = ready['references'][label][role]
                result = await client.embeddings.create(model=PROFILE + '.' + suffix, input=text, encoding_format='float')
                assert len(result.data) == 1
                assert result.usage.prompt_tokens == reference['prompt_eval_count'] == result.usage.total_tokens
                actual, expected = f32(result.data[0].embedding), f32(reference['embeddings'][0])
                assert actual == expected and len(actual) == 960 * 4
                checks.append({'role': role, 'encoding': 'float', 'tokens': result.usage.prompt_tokens,
                    'input_characters': len(text), 'batch_size': 1, 'dimensions': 960,
                    'vector_float32_sha256': hashlib.sha256(actual).hexdigest()})
        for suffix, role in (('query', 'search_query'), ('document', 'search_document')):
            before_batch = len(trace.read_text().splitlines())
            result = await client.embeddings.create(model=PROFILE + '.' + suffix,
                input=batch_texts(role), encoding_format='float')
            records = [json.loads(line) for line in trace.read_text().splitlines()[before_batch:]]
            assert result.usage.prompt_tokens == result.usage.total_tokens
            checks.append(batch_check(role, ready['references'], ready['references']['batch'][role],
                [item.embedding for item in result.data], [item.index for item in result.data],
                result.usage.prompt_tokens, records))
        # SDK 2.29's strict List[float] response type cannot accept wire base64;
        # its ordinary supported base64 parser is tested separately, unchanged.
        normal = openai.AsyncOpenAI(
            api_key='isolated-cpu-key', base_url='http://asgi.invalid/v1', max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app), trust_env=False))
        try:
            for suffix, role in (('query', 'search_query'), ('document', 'search_document')):
                encoded = await normal.embeddings.create(model=PROFILE + '.' + suffix, input=TEXT, encoding_format='base64')
                assert len(encoded.data) == 1
                actual = base64.b64decode(encoded.data[0].embedding, validate=True)
                assert actual == f32(ready['references']['long'][role]['embeddings'][0])
                assert encoded.usage.prompt_tokens == encoded.usage.total_tokens == ready['references']['long'][role]['prompt_eval_count']
                checks.append({'role': role, 'encoding': 'base64', 'tokens': encoded.usage.prompt_tokens,
                    'input_characters': len(TEXT), 'batch_size': 1, 'dimensions': 960,
                    'vector_float32_sha256': hashlib.sha256(actual).hexdigest()})
        finally:
            await normal.close()
        before = trace.read_bytes()
        counts = [sum(sum(mask) for mask in json.loads(line)['attention_mask']) for line in before.splitlines()]
        q, d = (ready['references']['long'][role]['prompt_eval_count'] for role in ROLES)
        sq, sd = (ready['references']['short'][role]['prompt_eval_count'] for role in ROLES)
        assert counts == [q, d, sq, sd, q + sq, d + sd, q, d, sq, sd, q + sq, d + sd, q, d]
        for parameters in ({'model': PROFILE + '.query', 'input': TEXT, 'dimensions': 3},
                           {'model': PROFILE, 'input': TEXT},
                           {'model': PROFILE + '.query', 'input': TEXT + '!'},
                           {'model': PROFILE + '.query', 'input': [TEXT, SHORT_TEXT, TEXT]},
                           {'model': PROFILE + '.query', 'input': ''}):
            try:
                await client.embeddings.create(**parameters)
                raise AssertionError('negative request unexpectedly succeeded')
            except openai.APIStatusError as exc:
                assert exc.status_code in (400, 404)
        assert trace.read_bytes() == before
        assert not [ticket for ticket in admission.snapshot() if ticket.kind == 'request']
        candidate_sha = export_candidate(report, deployment, implementation, checks)
        write(report / 'work/sdk-result.json', {'status': 'passed', 'uid': os.geteuid(), 'groups': os.getgroups(),
            'sdk': openai.__version__, 'model_ids': sorted(item.id for item in models.data), 'checks': checks,
            'negative_requests_without_encode': 5, 'native_forward_batches': len(before.splitlines()),
            'capability_candidate_sha256': candidate_sha, 'deployment_id': deployment.id,
            'profile_id': PROFILE, 'catalog_digest': MODEL_CATALOG.catalog_digest,
            'resident_admission_host_bytes': ready['peak_rss_bytes'], 'generation': ready['generation'],
            'artifact_fingerprint': deployment.artifact_identity.fingerprint,
            'configuration_fingerprint': deployment.configuration_fingerprint,
            'implementation': asdict(implementation), 'production_capability_granted': False})
    finally:
        await client.close()
        await runtime.aclose()


async def resolved(value):
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--native', action='store_true')
    group.add_argument('--sdk', action='store_true')
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    check_identity(args.report)
    imports(args.report)
    asyncio.run(native(args.report) if args.native else sdk_probe(args.report))
