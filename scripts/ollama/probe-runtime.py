#!/usr/bin/env python3
"""Unprivileged SDK/RuntimeService probe against the supervisor's isolated CPU server."""
import argparse
import asyncio
from dataclasses import fields
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import secrets
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
PORT, UID, GID = 18095, 65534, 982
MODEL_SHA = '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
CONTEXT_OVERFLOW = 'the input length exceeds the context length'
SCHEMA = {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['ok']}},
          'required': ['status'], 'additionalProperties': False}
TOOL_SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Berlin', 'Paris']}},
               'required': ['city'], 'additionalProperties': False}
TOOL = {'type': 'function', 'function': {'name': 'lookup_value', 'description': 'Look up the recorded value for a city.',
                                       'parameters': TOOL_SCHEMA, 'strict': False}}


def imports():
    for service in ('kiron-common', 'kiron-proxy', 'kiron-prism'):
        sys.path.insert(0, str(WORKSPACE / 'services' / service))


def context(name, seconds=30):
    from kiron_common.local_inference import RequestContext
    return RequestContext(name, time.monotonic() + seconds, asyncio.Event())


def memory():
    from kiron_common.gpu_admission import MemorySnapshot
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return MemorySnapshot(0, int(values['MemAvailable'].split()[0]) * 1024, time.monotonic())


def resident(value):
    models = value.get('models')
    if type(models) is not list or len(models) != 1:
        raise ValueError('exactly the isolated qwen model must be resident')
    model = models[0]
    if (model.get('name') != 'qwen3:8b' or model.get('digest', '').removeprefix('sha256:') != MODEL_SHA
            or type(model.get('size_vram')) is not int or model['size_vram'] != 0
            or model.get('context_length') != 1024 or type(model.get('context_length')) is not int
            or type(model.get('size')) is not int or not 0 < model['size'] <= 9 * 1024**3):
        raise ValueError('CPU residency/identity/context/size not verified')
    return model


def capabilities(evidence):
    from kiron_common.local_inference import Capability, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    from kiron_common.local_inference.json_schema import compile_schema, schema_features
    def cap(constraints):
        return Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))
    text = cap({'context_tokens': P(allowed_values=(1024,)), 'device': P(allowed_values=('cpu',)),
        'roles': P(allowed_values=('system', 'user', 'assistant', 'tool')),
        'max_output_tokens': P(minimum=1, maximum=128), 'default_max_output_tokens': P(allowed_values=(64,)),
        'token_budget': P(allowed_values=('max_tokens', 'max_completion_tokens')),
        'temperature': P(allowed_values=(0,)), 'reasoning_effort': P(allowed_values=('none',))})
    schema = schema_features(compile_schema(SCHEMA, strict=True))
    return CapabilitySet({N.CHAT: text, N.STREAMING: text,
        N.FUNCTION_TOOLS: cap({'tool_choice': P(allowed_values=('none', 'auto')),
            'max_tools': P(minimum=0, maximum=1), 'strict': P(allowed_values=(False,))}),
        N.PARALLEL_TOOLS: cap({}),
        N.STRUCTURED_OUTPUT: cap({'formats': P(allowed_values=('json_object', 'json_schema')),
            'strict': P(allowed_values=(True,)),
            **{'schema_' + key: P(allowed_values=tuple(items)) for key, items in schema.items()}})})


def check_calls(completion, cities):
    choice = completion.choices[0]
    calls = choice.message.tool_calls or []
    if (choice.finish_reason != 'tool_calls' or len(calls) != len(cities)
            or len({call.id for call in calls}) != len(calls)
            or any(call.function.name != 'lookup_value' for call in calls)
            or {json.dumps(json.loads(call.function.arguments), sort_keys=True) for call in calls}
            != {json.dumps({'city': city}, sort_keys=True) for city in cities}):
        raise ValueError('auto tools did not produce the exact requested calls')
    return calls


def check_text(completion, model_id, max_tokens=8):
    """Validate the API result contract, not the model's punctuation choices."""
    usage = completion.usage
    if completion.model != model_id or len(completion.choices) != 1:
        raise ValueError('text probe model or choice identity failed')
    choice = completion.choices[0]
    if (choice.finish_reason != 'stop' or choice.message.tool_calls
            or type(choice.message.content) is not str or not choice.message.content.strip()
            or usage is None or any(type(value) is not int for value in
                (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens))
            or usage.prompt_tokens < 1 or not 1 <= usage.completion_tokens <= max_tokens
            or usage.total_tokens != usage.prompt_tokens + usage.completion_tokens):
        raise ValueError('text probe content, finish or usage failed')


def measured_implementation(plan, compat, parser_revision):
    from kiron_common.local_inference import RuntimeImplementation
    if (type(compat.get('image_digest')) is not str
            or compat['image_digest'] not in plan['image']['RepoDigests']
            or compat.get('report_status') != 'passed'):
        raise ValueError('compatibility evidence not bound to exact planned RepoDigest')
    return RuntimeImplementation(compat['image_digest'], None, parser_revision)


def context_messages(history):
    # Fixed, bounded fixtures substantially exceed context 1024. No tokenizer
    # approximation is used as evidence: the native rejection must prove it.
    words = 'alpha beta gamma delta '
    if not history:
        return [{'role': 'user', 'content': words * 512}]
    return [{'role': role, 'content': words * 32} for _ in range(16)
            for role in ('user', 'assistant')] + [{'role': 'user', 'content': 'Reply only OK.'}]


def context_trace(report, first, last, messages, streaming):
    """Require one untruncated native POST, its exact rejection and clean EOF."""
    posts = []
    for index in range(first + 1, last + 1):
        prefix = report / f'native-{index:03d}'
        request = json.loads(prefix.with_suffix('.request.json').read_text())
        if request['method'] == 'POST':
            posts.append((index, prefix, request))
    if len(posts) != 1:
        raise ValueError('context case must execute exactly one native POST')
    index, prefix, request = posts[0]
    body = request['body']
    if (request['path'] != '/api/chat' or body.get('messages') != messages
            or body.get('stream') is not streaming or body.get('truncate') is not False
            or body.get('shift') is not False or body.get('think') is not False
            or body.get('options', {}).get('num_ctx') != 1024
            or body['options'].get('num_gpu') != 0 or body['options'].get('num_predict') != 8
            or 'keep_alive' in body):
        raise ValueError('context case changed native inputs or execution controls')
    previous = report / f'native-{index - 1:03d}'
    if json.loads(previous.with_suffix('.request.json').read_text()) != {
            'method': 'GET', 'path': '/api/ps', 'body': None}:
        raise ValueError('fresh residency check must immediately precede native POST')
    resident(json.loads(previous.with_suffix('.response.raw').read_bytes()))
    if not previous.with_suffix('.eof').is_file() or not prefix.with_suffix('.eof').is_file():
        raise ValueError('context or residency response EOF missing')
    status = json.loads(prefix.with_suffix('.status.json').read_text())['status']
    raw = prefix.with_suffix('.response.raw').read_bytes()
    expected = {'error': CONTEXT_OVERFLOW}
    if status == 200 and streaming:
        expected['status'] = 400
        if not raw.endswith(b'\n'):
            raise ValueError('unterminated native context error frame')
    elif status != 400:
        raise ValueError('native context error status differs')
    if json.loads(raw) != expected:
        raise ValueError('native context rejection differs or contains success output')
    return {'prefix': prefix.name, 'status': status, 'eof': True}


async def context_case(client, *, model_id, messages, streaming, store, trace, report):
    import openai
    before, chunks = trace.counter, []
    try:
        response = await client.chat.completions.create(model=model_id, messages=messages,
            max_completion_tokens=8, temperature=0, stream=streaming,
            **({'stream_options': {'include_usage': True}} if streaming else {}))
        if streaming:
            async with response:
                async for chunk in response:
                    chunks.append(chunk.model_dump())
    except openai.APIError as error:
        if (error.code != 'context_length_exceeded'
                or (not streaming and (not isinstance(error, openai.BadRequestError) or error.status_code != 400))
                or (streaming and isinstance(error, openai.APIStatusError))):
            raise ValueError('context case produced the wrong public error boundary') from error
        for chunk in chunks:
            if chunk['usage'] is not None or any(choice['finish_reason'] is not None
                    or set(k for k, v in choice['delta'].items() if v is not None) - {'role'}
                    for choice in chunk['choices']):
                raise ValueError('context error exposed content, tools, usage or a success finish')
        if any(ticket.kind == 'request' for ticket in store.snapshot()):
            raise ValueError('confirmed context rejection left a request ticket')
        native = context_trace(report, before, trace.counter, messages, streaming)
        return {'code': error.code, 'sdk_error': type(error).__name__,
                'http_status': getattr(error, 'status_code', None), 'chunks': chunks,
                'native': native, 'request_tickets': 0}
    raise ValueError('overlong input was silently accepted or truncated')


async def cases(service, model, store, trace, report, verify_resident):
    import httpx
    import openai
    from openai_api import create_openai_api_app
    if openai.__version__ != '2.29.0':
        raise ValueError('pinned SDK required')
    key = secrets.token_hex(32)
    class Keys:
        def validate_key(self, value):
            return secrets.compare_digest(value, key)
    class Records:
        async def add_request(self, record):
            pass
        async def update_request(self, request_id, **changes):
            pass
    app = create_openai_api_app(Records(), Keys())
    app.state.local_inference = service
    values = {}
    def save(name, value):
        values[name] = value
        with (report / ('sdk-' + name + '.json')).open('x') as output:
            json.dump(value, output, indent=2)
    async with openai.AsyncOpenAI(api_key=key, base_url='http://isolated-ollama/v1', max_retries=0,
            _strict_response_validation=True,
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app), timeout=None)) as client:
        async def complete(name, messages, *, stream=False, max_tokens=96, **options):
            await verify_resident()
            args = dict(model=model.api_model_id, messages=messages, max_completion_tokens=max_tokens,
                        temperature=0, **options)
            if stream:
                events = []
                if options.get('tools'):
                    # The public .stream helper requires strict tools for auto-
                    # parsing; Ollama deliberately does not claim that capability.
                    # Consume public SDK chunks and use its pinned accumulator
                    # without auto-parsing. No control is silently made strict.
                    from openai.lib.streaming.chat import ChatCompletionStreamState
                    state, chunks = ChatCompletionStreamState(), []
                    response = await client.chat.completions.create(**args, stream=True,
                        stream_options={'include_usage': True})
                    async with response:
                        async for chunk in response:
                            chunks.append(chunk.model_dump())
                            events.extend(event.type for event in state.handle_chunk(chunk))
                    save(name + '-chunks', chunks)
                    final = state.get_final_completion()
                else:
                    async with client.chat.completions.stream(**args, stream_options={'include_usage': True}) as response:
                        async for event in response:
                            events.append(event.type)
                        final = await response.get_final_completion()
                save(name + '-events', events)
            else:
                final = await client.chat.completions.create(**args)
            save(name, final.model_dump())
            if (final.usage is None or final.usage.prompt_tokens < 1
                    or not 1 <= final.usage.completion_tokens <= max_tokens
                    or final.usage.total_tokens != final.usage.prompt_tokens + final.usage.completion_tokens
                    or any(ticket.kind == 'request' for ticket in store.snapshot())):
                raise ValueError('exact usage or request-end proof missing')
            return final
        for streaming in (False, True):
            final = await complete('text-' + str(streaming), [{'role': 'user', 'content': 'Reply with exactly OK.'}],
                                   stream=streaming, max_tokens=8)
            check_text(final, model.api_model_id)
        history = [{'role': 'user', 'content': 'Call lookup_value for Berlin. Do not answer from memory.'}]
        single = await complete('tools-single', history, tools=[TOOL], tool_choice='auto', parallel_tool_calls=False)
        check_calls(single, {'Berlin'})
        history = [{'role': 'user', 'content': 'Call lookup_value for both Berlin and Paris now. Make two function calls.'}]
        parallel = await complete('tools-parallel-stream', history, stream=True,
                                  tools=[TOOL], tool_choice='auto', parallel_tool_calls=True)
        calls = check_calls(parallel, {'Berlin', 'Paris'})
        history.append(parallel.choices[0].message.model_dump(exclude_none=True))
        answers = {'Berlin': 'ALPHA', 'Paris': 'BETA'}
        history.extend({'role': 'tool', 'tool_call_id': call.id,
                        'content': json.dumps({'value': answers[json.loads(call.function.arguments)['city']]})}
                       for call in reversed(calls))
        history.append({'role': 'user', 'content':
            'Use the returned values. Reply only Berlin=<returned value>;Paris=<returned value>, with no spaces.'})
        replay = await complete('tools-replay', history, tools=[TOOL], tool_choice='none', max_tokens=24)
        if replay.choices[0].message.content.strip() != 'Berlin=ALPHA;Paris=BETA':
            raise ValueError('distinct tool result correlation failed')
        for name, format in (('json-object', {'type': 'json_object'}),
                             ('json-schema', {'type': 'json_schema', 'json_schema': {'name': 'status', 'strict': True, 'schema': SCHEMA}})):
            final = await complete(name, [{'role': 'user', 'content': 'Return a JSON object with exactly status set to ok.'}],
                                   stream=name == 'json-schema', max_tokens=24, response_format=format)
            if json.loads(final.choices[0].message.content) != {'status': 'ok'}:
                raise ValueError('structured JSON output failed')
        for history in (False, True):
            for streaming in (False, True):
                await verify_resident()
                name = 'context-' + ('history' if history else 'single') + ('-SSE' if streaming else '-JSON')
                save(name, await context_case(client, model_id=model.api_model_id,
                    messages=context_messages(history), streaming=streaming, store=store, trace=trace, report=report))
        recovery = await complete('context-recovery', [{'role': 'user', 'content': 'Reply only OK.'}], max_tokens=8)
        check_text(recovery, model.api_model_id)
        negatives = [('reasoning', {'reasoning_effort': 'low'}),
            ('required', {'tools': [TOOL], 'tool_choice': 'required'}),
            ('named', {'tools': [TOOL], 'tool_choice': {'type': 'function', 'function': {'name': 'lookup_value'}}}),
            ('strict-tool', {'tools': [{**TOOL, 'function': {**TOOL['function'], 'strict': True}}]}),
            ('vision', {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC'}}]}]})]
        for name, kwargs in negatives:
            before = trace.counter
            try:
                await client.chat.completions.create(**{'model': model.api_model_id,
                    'messages': [{'role': 'user', 'content': 'Hello'}], 'max_completion_tokens': 8, **kwargs})
            except openai.BadRequestError as error:
                if error.code not in ('unsupported_capability', 'unsupported_value') or trace.counter != before:
                    raise ValueError('unsupported feature touched native backend or failed unexpectedly') from error
                save('unsupported-' + name, {'code': error.code, 'native_requests': 0})
            else:
                raise ValueError('unsupported feature was accepted')
    return values


async def execute(report):
    imports()
    import httpx
    from composition import RegistryResolver
    from kiron_common.local_model_registry import RuntimeModelRegistry, RegistryFilePolicy
    from kiron_common.model_catalog import BackendType, ModelCatalog
    from kiron_common.local_inference import CapabilityEvidence, RuntimeTimeouts
    from kiron_common.model_state import RuntimeState
    from kiron_common.gpu_admission import AdmissionStore, RuntimeSecurity
    from kiron_common.ollama_compat import CompatResult, OllamaCapabilities
    from ollama_provider import OllamaProvider
    from provider_transport import bounded_request, decode_provider_json
    from runtime_composition import adapter_revision
    from runtime_service import RuntimeService, generation_key
    plan = json.loads((report / 'plan.json').read_text())
    if (os.geteuid() != UID or os.getegid() != GID or os.getgroups()
            or Path(__file__).resolve() != report / 'source/scripts/ollama/probe-runtime.py'
            or not sys.flags.isolated or not sys.dont_write_bytecode):
        raise ValueError('probe requires unprivileged immutable snapshot')
    spec = importlib.util.spec_from_file_location('ollama_native_trace', WORKSPACE / 'scripts/prism/probe-openai-features.py')
    trace_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace_module)
    results = report / 'results'
    trace = trace_module.NativeTrace(results)
    client = httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', timeout=None, trust_env=False, transport=trace)
    async def native(path, body=None, seconds=30):
        status, raw = await bounded_request(client, 'GET' if body is None else 'POST', path,
            context('setup-' + path, seconds), limit=16 * 1024**2, **({'json': body} if body is not None else {}))
        if status != 200:
            raise ValueError('isolated native setup/status failed')
        return decode_provider_json(raw)
    result, service = {'status': 'failed', 'scope': plan['scope']}, None
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if (await native('/api/version'))['version'] != '0.18.0':
                    raise ValueError('unexpected native version')
                break
            except (httpx.ConnectError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(.2)
        if (await native('/api/ps')).get('models') != []:
            raise ValueError('fresh isolated native server is not empty')
        loaded = await native('/api/generate', {'model': 'qwen3:8b', 'stream': False, 'keep_alive': '10m',
            'think': False, 'options': {'num_gpu': 0, 'num_ctx': 1024}}, 180)
        if loaded.get('done') is not True:
            raise ValueError('controlled CPU preload did not complete')
        observed = resident(await native('/api/ps'))
        result['observed_preload'] = observed
        with (results / 'preload.json').open('x') as output:
            json.dump({'native': loaded, 'resident': observed, 'scope': 'measured existing residency, not a coldload profile'}, output, indent=2)
        registry = RuntimeModelRegistry(report / 'registry/models.json', readonly=True, file_policy=RegistryFilePolicy(0, GID))
        resolver = RegistryResolver(registry, {}, catalog=ModelCatalog())
        model = (await resolver.snapshot()).resolve(plan['model_id'])
        if model.deployment.resource_profile is not None:
            raise ValueError('this measurement must not manufacture a coldload profile')
        compat = json.loads((report / 'compat.json').read_text())
        implementation = measured_implementation(plan, compat, adapter_revision('ollama_provider.py'))
        evidence = CapabilityEvidence(implementation.provider_revision, model.deployment.artifact_identity.fingerprint,
            MODEL_SHA, None, None, implementation.parser_revision, model.deployment.configuration_fingerprint,
            'Isolated CPU candidate measurement only: ' + str(report), datetime.now(timezone.utc))
        flags = {field.name: CompatResult(compat['capabilities'][field.name]['ok']) for field in fields(OllamaCapabilities)}
        if not all(flag.ok is True for flag in flags.values()):
            raise ValueError('missing compatibility evidence')
        provider = OllamaProvider(client=client, resolver=resolver, implementation=implementation,
            capabilities={model.deployment.id: capabilities(evidence)}, compatibility=OllamaCapabilities(**flags),
            expected_version='0.18.0')
        health = await provider.health(context('measured-health'))
        if health.models[model.deployment.id].state is not RuntimeState.LOADED:
            raise ValueError('adapter did not confirm native identity/residency')
        store = AdmissionStore(report / 'admission', security=RuntimeSecurity(0, GID, frozenset({0, UID})))
        generation = generation_key(health.generation)
        store.reserve(operation_id='measured-cpu-resident', owner='kiron-proxy', generation=generation,
            deployment_id=model.deployment.id, kind='load', gpu_bytes=0, host_bytes=observed['size'],
            measure=memory, ttl_seconds=850, allow_existing=False,
            lifecycle_domain='ollama', lifecycle_model=MODEL_SHA)
        store.transition('measured-cpu-resident', owner='kiron-proxy', expected_generation=generation, phase='resident')
        service = RuntimeService(resolver=resolver, providers={BackendType.OLLAMA: provider}, admission=store,
            measure=memory, timeouts=RuntimeTimeouts(180, 30, 180, 120, 600, 10, 30))
        async def verify_resident():
            resident(await native('/api/ps'))
        async with asyncio.timeout(600):
            result['cases'] = await cases(service, model, store, trace, results, verify_resident)
        resident(await native('/api/ps'))
        await service.unload(model, context('isolated-unload', 60))
        if (await native('/api/ps')).get('models') != [] or store.snapshot():
            raise ValueError('unload or admission cleanup not confirmed')
        result['status'], result['unloaded'], result['admission_empty'] = 'passed', True, True
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            if service is not None:
                await asyncio.wait_for(service.aclose(), 20)
            else:
                await client.aclose()
        finally:
            with (results / 'result.json').open('x') as output:
                json.dump(result, output, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.report_dir))
