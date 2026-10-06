#!/usr/bin/env python3
"""Reconstruct one independently reviewed CPU-v10 candidate; never install evidence.

This is a fixed review recipe, not an exporter for arbitrary Ollama archives.
Changing the evidence pin requires another independent archive/code review.
"""
import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

WORKSPACE = Path(__file__).resolve().parents[2]
REPORT = Path('/usr/lib/kiron/test-runtimes/ollama/reports/cpu-contract-v10')
OUTPUT_ROOT = Path('/usr/lib/kiron/test-runtimes/ollama/capability-candidates')
REVIEW_SHA256 = 'ddf07bd850b39c17f2f9cd570d50f1844dc896c3c452d0e458dd9a799c250418'
MAX_FILE, MAX_TOTAL, MAX_ENTRIES, MAX_DEPTH = 4 * 1024**2, 32 * 1024**2, 1024, 16
CASES = (('text-False', 8, False), ('text-True', 8, True), ('tools-single', 96, False),
         ('tools-parallel-stream', 96, True), ('tools-replay', 24, False),
         ('json-object', 24, False), ('json-schema', 24, True))
CONTEXT_CASES = ('context-single-JSON', 'context-single-SSE', 'context-history-JSON', 'context-history-SSE')
CONTEXT_OVERFLOW = 'the input length exceeds the context length'
SCHEMA = {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['ok']}},
          'required': ['status'], 'additionalProperties': False}
TOOL_SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Berlin', 'Paris']}},
               'required': ['city'], 'additionalProperties': False}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    require(path.is_absolute() and path.resolve() == path, 'noncanonical input')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_FILE,
                'input is not a bounded single-link file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(MAX_FILE + 1)
        after = os.fstat(fd)
        require(len(raw) <= MAX_FILE and all(getattr(before, key) == getattr(after, key) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_gid', 'st_mode')),
            'input changed while reading')
        return raw
    finally:
        os.close(fd)


def decode(raw):
    from provider_transport import decode_provider_json
    return decode_provider_json(raw)


def value(path):
    return decode(read(path))


def inventory(root):
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'invalid archive root')
    result, total, count = {}, 0, 0
    def walk(directory, depth):
        nonlocal total, count
        require(depth <= MAX_DEPTH, 'archive depth exceeded')
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                require(count <= MAX_ENTRIES, 'archive entry limit exceeded')
                path, info = Path(entry.path), entry.stat(follow_symlinks=False)
                row = {'uid': info.st_uid, 'gid': info.st_gid, 'mode': stat.S_IMODE(info.st_mode)}
                if stat.S_ISDIR(info.st_mode):
                    row['type'] = 'directory'
                    walk(path, depth + 1)
                else:
                    raw = read(path)
                    total += len(raw)
                    require(total <= MAX_TOTAL, 'archive byte limit exceeded')
                    row.update(type='file', size=len(raw), sha256=digest(raw))
                result[str(path.relative_to(root))] = row
    walk(root, 0)
    require(result, 'empty archive')
    return dict(sorted(result.items()))


def current_harness():
    # This fixed current checkout module contains no import-time runtime action.
    spec = importlib.util.spec_from_file_location('ollama_review_harness', Path(__file__).with_name('smoke-runtime.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_cases(report, model_id):
    results = value(report / 'results/result.json')
    require(results['status'] == 'passed' and results['unloaded'] is True and results['admission_empty'] is True,
            'SDK workflow or unload failed')
    calls = []
    for index, file in enumerate(sorted((report / 'results').glob('native-*.request.json')), 1):
        require(file.name == f'native-{index:03d}.request.json', 'native trace numbering differs')
        stem = str(file).removesuffix('.request.json')
        request = value(file)
        status = value(Path(stem + '.status.json'))
        require(read(Path(stem + '.eof')) == b'', 'native EOF differs')
        if request['path'] == '/api/chat':
            require(request['method'] == 'POST', 'native chat method differs')
            body = request['body']
            raw = read(Path(stem + '.response.raw'))
            frames = [decode(line) for line in raw.splitlines()] if body['stream'] else [decode(raw)]
            previous = report / 'results' / f'native-{index - 1:03d}'
            require(value(previous.with_suffix('.request.json')) == {'method': 'GET', 'path': '/api/ps', 'body': None}
                    and read(previous.with_suffix('.eof')) == b'', 'fresh resident check missing')
            models = value(previous.with_suffix('.response.raw'))['models']
            require(len(models) == 1 and models[0]['name'] == 'qwen3:8b'
                    and models[0]['digest'].removeprefix('sha256:') == current_harness().MODEL_SHA
                    and type(models[0]['context_length']) is int and models[0]['context_length'] == 1024
                    and type(models[0]['size_vram']) is int and models[0]['size_vram'] == 0,
                    'native resident execution profile differs')
            require(body.get('truncate') is False and body.get('shift') is False and 'keep_alive' not in body,
                    'native context controls differ')
            calls.append((body, frames, status, Path(stem).name, raw))
        else:
            require(status == {'status': 200}, 'native setup or health status differs')
    require(len(calls) == len(CASES) + len(CONTEXT_CASES) + 1, 'native chat case count differs')
    for name, call in zip(CONTEXT_CASES, calls[len(CASES):-1]):
        validate_context_case(report, results, name, call)
    measured = []
    positives = calls[:len(CASES)] + calls[-1:]
    for (name, budget, stream), (body, frames, status, _, _) in zip(CASES + (('context-recovery', 8, False),), positives):
        require(status == {'status': 200}, 'native positive status differs')
        require(body['model'] == 'qwen3:8b' and body['think'] is False and body['stream'] is stream
                and body['options'] == {'num_predict': budget, 'temperature': 0, 'num_gpu': 0, 'num_ctx': 1024},
                'native CPU/sampling/budget controls differ')
        require(all(frame['model'] == 'qwen3:8b' and type(frame['done']) is bool for frame in frames)
                and sum(frame['done'] for frame in frames) == 1 and frames[-1]['done'] is True
                and frames[-1]['done_reason'] == 'stop', 'native completion order differs')
        public = results['cases'][name]
        require(public == value(report / ('results/sdk-' + name + '.json')) and public['model'] == model_id,
                'public result identity differs')
        usage, last = public['usage'], frames[-1]
        require((usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']) ==
                (last['prompt_eval_count'], last['eval_count'], last['prompt_eval_count'] + last['eval_count'])
                and 0 < last['eval_count'] <= budget and usage['completion_tokens_details'] is None
                and usage['prompt_tokens_details'] is None, 'native/public usage differs')
        message = public['choices'][0]['message']
        require((message['content'] or '') == ''.join(frame['message'].get('content', '') for frame in frames)
                and not any(frame['message'].get('thinking') for frame in frames), 'native/public text differs')
        native_tools = [call for frame in frames for call in frame['message'].get('tool_calls', [])]
        public_tools = message['tool_calls'] or []
        require(len(native_tools) == len(public_tools) == len({call['id'] for call in public_tools}), 'tool count/IDs differ')
        cities = ['Berlin'] if name == 'tools-single' else ['Berlin', 'Paris'] if name == 'tools-parallel-stream' else []
        require(sorted(call['function']['arguments']['city'] for call in native_tools) == cities, 'tool cases differ')
        if cities:
            require(body['tools'] == [{'type': 'function', 'function': {'name': 'lookup_value',
                'description': 'Look up the recorded value for a city.', 'parameters': TOOL_SCHEMA}}],
                'native automatic tool declaration differs')
        else:
            require(body['tools'] == [], 'unexpected native tools')
        for native, public_call in zip(native_tools, public_tools):
            require(native['function']['name'] == public_call['function']['name'] == 'lookup_value'
                    and native['function']['arguments'] == decode(public_call['function']['arguments'].encode()),
                    'tool arguments differ')
        require(public['choices'][0]['finish_reason'] == ('tool_calls' if native_tools else 'stop'), 'finish differs')
        measured.append({'case': name, 'budget': budget, 'stream': stream, 'usage': usage})
    require(calls[5][0]['format'] == 'json' and calls[6][0]['format'] == SCHEMA, 'native structured control differs')
    replay = calls[4][0]
    require(replay['tools'] == [] and results['cases']['tools-replay']['choices'][0]['message']['content']
            == 'Berlin=ALPHA;Paris=BETA', 'distinct replay failed')
    require([decode(item['content'].encode()) for item in replay['messages'] if item['role'] == 'tool']
            == [{'value': 'ALPHA'}, {'value': 'BETA'}]
            and not any('ALPHA' in canonical(item).decode() or 'BETA' in canonical(item).decode()
                        for item in replay['messages'] if item['role'] != 'tool'), 'replay leaks expected values')
    for name in ('reasoning', 'required', 'named', 'strict-tool', 'vision'):
        negative = results['cases']['unsupported-' + name]
        require(negative == value(report / ('results/sdk-unsupported-' + name + '.json'))
                and negative['native_requests'] == 0 and negative['code'] in ('unsupported_capability', 'unsupported_value'),
                'negative feature proof differs')
    return measured


def validate_context_case(report, results, name, call):
    body, frames, status, prefix, raw = call
    streaming = name.endswith('-SSE')
    words = 'alpha beta gamma delta '
    messages = ([{'role': role, 'content': words * 32} for _ in range(16)
                 for role in ('user', 'assistant')] + [{'role': 'user', 'content': 'Reply only OK.'}]
                if '-history-' in name else [{'role': 'user', 'content': words * 512}])
    require(body['model'] == 'qwen3:8b' and body['messages'] == messages and body['stream'] is streaming
            and body['think'] is False and body.get('truncate') is False and body.get('shift') is False
            and 'keep_alive' not in body and body['tools'] == []
            and body['options'] == {'num_predict': 8, 'temperature': 0, 'num_gpu': 0, 'num_ctx': 1024},
            'overlong native input or execution controls differ')
    expected = {'error': CONTEXT_OVERFLOW}
    if streaming and status == {'status': 200}:
        expected['status'] = 400
        require(raw.endswith(b'\n'), 'context error frame incomplete')
    else:
        require(status == {'status': 400}, 'native context error status differs')
    require(frames == [expected], 'native context rejection differs')
    public = results['cases'][name]
    require(public == value(report / ('results/sdk-' + name + '.json'))
            and public['code'] == 'context_length_exceeded' and public['request_tickets'] == 0
            and public['native'] == {'prefix': prefix, 'status': status['status'], 'eof': True}
            and public['sdk_error'] == ('APIError' if streaming else 'BadRequestError')
            and public['http_status'] == (None if streaming else 400), 'public context rejection differs')
    require(type(public['chunks']) is list and (streaming or not public['chunks']), 'unexpected context chunks')
    for chunk in public['chunks']:
        require(chunk['usage'] is None and all(choice['finish_reason'] is None
                and set(k for k, v in choice['delta'].items() if v is not None) <= {'role'}
                for choice in chunk['choices']), 'context rejection exposed success output')


def candidate(model, implementation, observed_at):
    from kiron_common.local_inference import Capability, CapabilityEvidence, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    from kiron_common.local_inference.json_schema import compile_schema, schema_features
    from kiron_common.model_catalog import BackendType
    from runtime_capabilities import encode_provider_evidence, decode_provider_evidence
    from runtime_service import chat_profile
    deployment = model.deployment
    evidence = CapabilityEvidence(implementation.provider_revision, deployment.artifact_identity.fingerprint,
        deployment.artifact_identity.sha256, None, implementation.template_revision, implementation.parser_revision,
        deployment.configuration_fingerprint, 'reviewed-archive-sha256:' + REVIEW_SHA256, observed_at)
    def cap(constraints):
        return Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))
    chat = cap({'roles': P(allowed_values=('user', 'assistant', 'tool')),
        'context_tokens': P(allowed_values=(1024,)), 'device': P(allowed_values=('cpu',)),
        'max_output_tokens': P(allowed_values=(8, 24, 96)), 'default_max_output_tokens': P(allowed_values=(8,)),
        'token_budget': P(allowed_values=('max_completion_tokens',)), 'temperature': P(allowed_values=(0,)),
        'reasoning_effort': P(allowed_values=('none',))})
    features = schema_features(compile_schema(SCHEMA, strict=True))
    caps = CapabilitySet({N.CHAT: chat, N.STREAMING: chat,
        N.FUNCTION_TOOLS: cap({'tool_choice': P(allowed_values=('none', 'auto')),
            'max_tools': P(allowed_values=(1,)), 'strict': P(allowed_values=(False,))}),
        N.PARALLEL_TOOLS: cap({}), N.STRUCTURED_OUTPUT: cap({'formats': P(allowed_values=('json_object', 'json_schema')),
            'strict': P(allowed_values=(True,)), **{'schema_' + key: P(allowed_values=tuple(items)) for key, items in features.items()}})})
    chat_profile(chat, deployment)
    encoded = encode_provider_evidence(BackendType.OLLAMA, {deployment.id: caps})
    decoded = decode_provider_evidence(encoded, BackendType.OLLAMA)
    require(decoded == {deployment.id: caps} and all(caps.supports(name, deployment, implementation)
            for name in (N.CHAT, N.STREAMING, N.FUNCTION_TOOLS, N.PARALLEL_TOOLS, N.STRUCTURED_OUTPUT)), 'evidence identity differs')
    return encoded


def assemble():
    from kiron_common.local_inference import build_resolver_snapshot, RuntimeImplementation
    from kiron_common.local_model_registry.codec import decode_registry
    from kiron_common.model_catalog import ModelCatalog
    from runtime_composition import adapter_revision
    files = inventory(REPORT)
    require(digest(canonical(files)) == REVIEW_SHA256, 'archive differs from the fixed independent review pin')
    plan, result = value(REPORT / 'plan.json'), value(REPORT / 'result.json')
    harness = current_harness()
    source = harness.inventory(WORKSPACE)
    require(source == plan['source_sha256'], 'current sources differ from reviewed run')
    harness.verify_source(REPORT, source)
    require(result['status'] == 'passed' and all(result[key] is True for key in
            ('container_removed', 'port_closed', 'sdk_reaped', 'source_verified_after')), 'cleanup not confirmed')
    require(plan['image']['Id'] == harness.IMAGE and harness.REPO_DIGEST in plan['image']['RepoDigests']
            and harness.model_inventory() == plan['model_sha256'], 'image or cache pin differs')
    raw_registry = read(REPORT / 'registry/models.json')
    require(digest(raw_registry) == plan['registry_sha256'], 'registry changed')
    entries = decode_registry(raw_registry)
    require(len(entries) == 1, 'unexpected registry entries')
    model = build_resolver_snapshot(ModelCatalog(), entries).resolve(plan['model_id'])
    require(model.deployment.resource_profile is None and model.deployment.artifact_identity.sha256 == harness.MODEL_SHA,
            'this review has no coldload resource profile')
    implementation = RuntimeImplementation(harness.REPO_DIGEST, None, adapter_revision('ollama_provider.py'))
    measured = validate_cases(REPORT, model.api_model_id)
    encoded = candidate(model, implementation, datetime.fromisoformat(result['container_state']['FinishedAt']))
    provenance = {'scope': 'isolated CPU-v10 review candidate only; not a production grant',
        'archive': str(REPORT), 'archive_sha256': REVIEW_SHA256, 'archive_inventory': files,
        'recipe_sha256': digest(read(Path(__file__).resolve())), 'candidate_sha256': digest(canonical(encoded)),
        'source_sha256': source, 'snapshot_revision': model.snapshot_revision,
        'deployment_id': model.deployment.id, 'configuration_fingerprint': model.deployment.configuration_fingerprint,
        'artifact_fingerprint': model.deployment.artifact_identity.fingerprint, 'implementation': asdict(implementation),
        'production_observations': {label: value(REPORT / ('production-' + label + '.json'))
                                    for label in ('before', 'after')},
        'measured': measured,
        'context_rejections': {name: value(REPORT / ('results/sdk-' + name + '.json')) for name in CONTEXT_CASES},
        'limits': {'device': 'cpu', 'context': 1024, 'cpu_quota': 2, 'memory_bytes': 9 * 1024**3,
            'parallel_slots': 1, 'resident_only': True, 'default_budget': 8},
        'limitations': ['RepoDigest/Template=None matches the production implementation contract; isolated CPU residency is mandatory.',
            'Default 8 is explicit local policy selected from measured values.',
            'No coldload/GPU, Vision, active Reasoning, Responses, cache/reasoning details or Embeddings grant.',
            'No system/developer role, strict/required/named tools or unmeasured sampling values.',
            'Four overlong single/history JSON/SSE inputs were rejected natively; no silent truncation was accepted.',
            'Eight positive cases and four context rejections do not guarantee arbitrary model output quality.']}
    require(inventory(REPORT) == files and harness.inventory(WORKSPACE) == source, 'inputs changed during assembly')
    return encoded, provenance


def output_path(path):
    require(path.is_absolute() and path.parent == OUTPUT_ROOT and path.resolve() == path
            and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', path.name), 'output must be a new isolated review directory')
    return path


def publish(path, encoded, provenance):
    output_path(path)
    require(os.geteuid() == 0, 'root ownership required')
    for parent in OUTPUT_ROOT.parents:
        info = parent.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022, 'unsafe output parent')
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o750)
        os.chown(OUTPUT_ROOT, 0, 982)
    info = OUTPUT_ROOT.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022, 'unsafe output root')
    stage = Path(tempfile.mkdtemp(prefix='.review-', dir=OUTPUT_ROOT))
    try:
        for name, content in (('ollama.json', encoded), ('provenance.json', provenance)):
            with (stage / name).open('xb') as stream:
                stream.write(canonical(content) + b'\n')
                os.fchown(stream.fileno(), 0, 982)
                os.fchmod(stream.fileno(), 0o640)
                stream.flush()
                os.fsync(stream.fileno())
        os.chown(stage, 0, 982)
        stage.chmod(0o750)
        directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(path), 1) != 0:
            raise OSError(ctypes.get_errno(), 'review publication failed without overwrite')
        directory = os.open(OUTPUT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify(path):
    output_path(path)
    require({item.name for item in path.iterdir()} == {'ollama.json', 'provenance.json'}, 'candidate file set differs')
    for item in (path, path / 'ollama.json', path / 'provenance.json'):
        info = item.lstat()
        require((info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (0, 982, 0o750 if item == path else 0o640),
                'candidate ownership/mode differs')
    encoded, provenance = assemble()
    require(read(path / 'ollama.json') == canonical(encoded) + b'\n'
            and read(path / 'provenance.json') == canonical(provenance) + b'\n', 'candidate content differs')
    return provenance['candidate_sha256']


def main():
    for service in ('kiron-common', 'kiron-proxy'):
        sys.path.insert(0, str(WORKSPACE / 'services' / service))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('inspect')
    commands.add_parser('export').add_argument('--output', type=Path, required=True)
    commands.add_parser('verify').add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps({'archive': str(REPORT), 'sha256': digest(canonical(inventory(REPORT)))}))
    elif args.command == 'export':
        output_path(args.output)
        encoded, provenance = assemble()
        publish(args.output, encoded, provenance)
        print(json.dumps({'candidate': str(args.output), 'sha256': provenance['candidate_sha256']}))
    else:
        print(json.dumps({'status': 'verified', 'sha256': verify(args.candidate)}))


if __name__ == '__main__':
    main()
