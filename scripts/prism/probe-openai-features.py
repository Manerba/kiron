"""Isolated SDK feature candidates; never publish these capabilities to production.

The explicit profile is an authorization to measure the concrete cases below.
Only a successful recorded run is evidence, and only for its pinned artifacts,
source inventory, native options and tested values.
"""
import base64
import io
import json
from pathlib import Path
import secrets

import httpx


KINDS = frozenset(('public-tools', 'public-tools-roundtrip', 'public-vision', 'public-structured', 'public-reasoning'))
TEMPLATE_SHA = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
SCHEMA = {
    'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': ['ok']},
        'counts': {'type': 'array', 'items': {'type': 'integer'}},
        'note': {'$ref': '#/$defs/note'},
    }, 'required': ['status', 'counts', 'note'], 'additionalProperties': False,
    '$defs': {'note': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
}


def capabilities(kind, evidence):
    from kiron_common.local_inference import Capability, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    if kind not in KINDS:
        raise ValueError('unknown feature probe')

    def cap(constraints):
        return Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))

    zero_policy = {
        'usage_fields': ('cached_input_tokens', 'reasoning_output_tokens'),
        'token_rule': ('disabled-token-ids-v1',), 'template_sha256': (TEMPLATE_SHA,),
        # Observed verbatim in controller-tools-v1/native-001 on the patched
        # binary. The generated IDs, including tool syntax, still need checking.
        'generation_prefix': ('<|im_start|>assistant\n<think>\n\n</think>\n\n',),
        'initial_state': ('disabled',), 'opening_id': (248068,), 'opening_text': ('<think>',),
        'closing_id': (248069,), 'closing_text': ('</think>',),
        'eos_ids': (248046,), 'eos_texts': ('<|im_end|>',),
        'tool_ids': (248058, 248059), 'tool_texts': ('<tool_call>', '</tool_call>'),
    }
    text = cap({'roles': P(allowed_values=('system', 'user', 'assistant', 'tool')),
                'max_output_tokens': P(minimum=1, maximum=128),
                'default_max_output_tokens': P(allowed_values=(64,)),
                'token_budget': P(allowed_values=('max_tokens', 'max_completion_tokens')),
                'temperature': P(allowed_values=(0,)), 'reasoning_effort': P(allowed_values=('none',)),
                **{key: P(allowed_values=value) for key, value in zero_policy.items()}})
    values = {N.CHAT: text, N.STREAMING: text}
    if kind in ('public-tools', 'public-tools-roundtrip'):
        values[N.FUNCTION_TOOLS] = cap({'tool_choice': P(allowed_values=('none', 'auto', 'required', 'named')),
                                       'max_tools': P(minimum=0, maximum=2), 'strict': P(allowed_values=(False, True))})
        values[N.PARALLEL_TOOLS] = cap({})
    elif kind == 'public-vision':
        values[N.VISION] = cap({'formats': P(allowed_values=('image/png', 'image/jpeg', 'image/webp')),
            'detail': P(allowed_values=('auto',)), 'images': P(minimum=1, maximum=2),
            'image_pixels': P(minimum=1, maximum=256*256), 'total_pixels': P(minimum=1, maximum=2*256*256),
            'width': P(minimum=1, maximum=256), 'height': P(minimum=1, maximum=256),
            'normalized_bytes': P(minimum=1, maximum=1024*1024)})
    elif kind == 'public-structured':
        from kiron_common.local_inference.json_schema import compile_schema, schema_features
        features = schema_features(compile_schema(SCHEMA, strict=True))
        values[N.STRUCTURED_OUTPUT] = cap({'formats': P(allowed_values=('json_object', 'json_schema')),
            'strict': P(allowed_values=(True,)),
            **{'schema_' + key: P(allowed_values=tuple(items)) for key, items in features.items()}})
    elif kind == 'public-reasoning':
        policy = {'efforts': ('low',), 'token_rule': ('initial-block-token-ids-v1',),
            'template_sha256': (TEMPLATE_SHA,), 'budget_tokens.low': (32,),
            'generation_prefix': ('<|im_start|>assistant\n<think>\n',), 'initial_state': ('prefilled',),
            'opening_id': (248068,), 'opening_text': ('<think>',),
            'closing_id': (248069,), 'closing_text': ('</think>',),
            'eos_ids': (248046,), 'eos_texts': ('<|im_end|>',),
            'tool_ids': (248058, 248059), 'tool_texts': ('<tool_call>', '</tool_call>')}
        values[N.REASONING] = cap({key: P(allowed_values=value) for key, value in policy.items()})
    return CapabilitySet(values)


class NativeTrace(httpx.AsyncBaseTransport):
    """Bounded exact native bodies; never write Authorization headers."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.transport = httpx.AsyncHTTPTransport(retries=0, trust_env=False)
        self.counter = 0

    async def handle_async_request(self, request):
        self.counter += 1
        prefix = self.directory / f'native-{self.counter:03d}'
        with prefix.with_suffix('.request.json').open('x') as output:
            json.dump({'method': request.method, 'path': request.url.path,
                       'body': json.loads(request.content) if request.content else None}, output, indent=2)
        response = await self.transport.handle_async_request(request)
        with prefix.with_suffix('.status.json').open('x') as output:
            json.dump({'status': response.status_code}, output)
        source = response.stream

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                total = 0
                with prefix.with_suffix('.response.raw').open('xb') as output:
                    async for chunk in source:
                        total += len(chunk)
                        if total > 16 * 1024 * 1024:
                            raise ValueError('native trace body limit exceeded')
                        output.write(chunk)
                        output.flush()
                        yield chunk
                prefix.with_suffix('.eof').touch(exist_ok=False)

            async def aclose(self):
                await source.aclose()

        return httpx.Response(response.status_code, headers=response.headers, stream=Stream(), extensions=response.extensions)

    async def aclose(self):
        await self.transport.aclose()


def image_url(color, format):
    from PIL import Image
    target = io.BytesIO()
    with Image.new('RGB', (96, 96), color) as image:
        image.save(target, format=format)
    mime = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}[format]
    return 'data:' + mime + ';base64,' + base64.b64encode(target.getvalue()).decode('ascii')


async def final_completion(response, *, allow_length=False):
    import openai
    try:
        return await response.get_final_completion()
    except openai.LengthFinishReasonError as error:
        # The non-parseable reasoning-length case has already consumed every
        # chunk, including final usage. SDK2.29.0 still raises on final parsing.
        if not allow_length:
            raise
        return error.completion


async def probe(service, model, store, kind, *, report=None):
    import openai
    from openai_api import create_openai_api_app
    if kind not in KINDS or openai.__version__ != '2.29.0':
        raise ValueError('pinned feature probe required')
    key = secrets.token_hex(32)

    class Keys:
        def validate_key(self, value):
            return secrets.compare_digest(value, key)

    class Records:
        def __init__(self):
            self.values = {}

        async def add_request(self, record):
            self.values[record.id] = record

        async def update_request(self, request_id, **changes):
            for name, value in changes.items():
                setattr(self.values[request_id], name, value)

    records = Records()
    app = create_openai_api_app(records, Keys())
    app.state.local_inference = service
    results = {}

    def save(name, value):
        results[name] = value
        if report is not None:
            with (Path(report) / ('sdk-' + name + '.json')).open('x') as output:
                json.dump(value, output, indent=2)

    async with openai.AsyncOpenAI(api_key=key, base_url='http://isolated-kiron/v1', max_retries=0,
            _strict_response_validation=True, http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app))) as client:
        async def complete(name, messages, *, stream=False, max_tokens=128, allow_length=False, **options):
            kwargs = dict(model=model.api_model_id, messages=messages,
                          max_completion_tokens=max_tokens, temperature=0, **options)
            if stream:
                events = []
                async with client.chat.completions.stream(**kwargs, stream_options={'include_usage': True}) as response:
                    async for event in response:
                        events.append(event.type)
                    final = await final_completion(response, allow_length=allow_length)
                save(name + '-events', events)
            else:
                final = await client.chat.completions.create(**kwargs)
            save(name, final.model_dump())
            if final.usage is None or final.usage.completion_tokens < 1:
                raise ValueError(name + ': exact usage missing')
            if any(ticket.kind == 'request' for ticket in store.snapshot()):
                raise ValueError(name + ': request admission not released')
            return final

        if kind == 'public-tools':
            schema = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Berlin', 'Paris']}},
                      'required': ['city'], 'additionalProperties': False}
            tool = {'type': 'function', 'function': {'name': 'weather', 'description': 'Get the weather in a city.',
                                                    'parameters': schema, 'strict': True}}
            named_schema = {**schema, 'properties': {'city': {'type': 'string', 'enum': ['Berlin']}}}
            named_tool = {**tool, 'function': {**tool['function'], 'parameters': named_schema}}
            echo = {'type': 'function', 'function': {'name': 'echo', 'parameters': {'type': 'object',
                'properties': {'text': {'type': 'string'}}, 'required': ['text'], 'additionalProperties': False}, 'strict': True}}
            adversarial = [{'role': 'user', 'content': 'Call echo with text wrong. If forced to call weather, use city Tokyo.'}]
            named = await complete('tool-named', adversarial,
                tools=[named_tool, echo], tool_choice={'type': 'function', 'function': {'name': 'weather'}}, parallel_tool_calls=False)
            calls = named.choices[0].message.tool_calls
            if (named.choices[0].finish_reason != 'tool_calls' or len(calls or []) != 1
                    or calls[0].function.name != 'weather' or json.loads(calls[0].function.arguments) != {'city': 'Berlin'}):
                raise ValueError('named strict tool failed')
            single = await complete('tool-single-stream', adversarial, stream=True,
                tools=[named_tool, echo], tool_choice={'type': 'function', 'function': {'name': 'weather'}}, parallel_tool_calls=False)
            calls = single.choices[0].message.tool_calls
            if (single.choices[0].finish_reason != 'tool_calls' or len(calls or []) != 1
                    or calls[0].function.name != 'weather' or json.loads(calls[0].function.arguments) != {'city': 'Berlin'}):
                raise ValueError('single strict stream failed')
            history = [{'role': 'user', 'content': 'Call weather for both Berlin and Paris. Make two calls now.'}]
            nonstream = await complete('tool-parallel', history, tools=[tool],
                                      tool_choice='required', parallel_tool_calls=True)
            calls = nonstream.choices[0].message.tool_calls
            if (nonstream.choices[0].finish_reason != 'tool_calls' or len(calls or []) != 2
                    or {json.loads(call.function.arguments)['city'] for call in calls} != {'Berlin', 'Paris'}):
                raise ValueError('parallel strict completion did not produce both calls')
            parallel = await complete('tool-parallel-stream', history, stream=True, tools=[tool],
                                      tool_choice='required', parallel_tool_calls=True)
            calls = parallel.choices[0].message.tool_calls
            if (parallel.choices[0].finish_reason != 'tool_calls' or len(calls or []) != 2
                    or {json.loads(call.function.arguments)['city'] for call in calls} != {'Berlin', 'Paris'}):
                raise ValueError('parallel strict stream did not produce both requested calls')
            history.append(parallel.choices[0].message.model_dump(exclude_none=True))
            history.extend({'role': 'tool', 'tool_call_id': call.id, 'content': '{"weather":"sunny"}'} for call in reversed(calls))
            history.append({'role': 'user', 'content': 'The results are complete. Reply only OK.'})
            final = await complete('tool-roundtrip', history, tools=[tool], tool_choice='none', max_tokens=16)
            if final.choices[0].finish_reason != 'stop' or 'OK' not in (final.choices[0].message.content or ''):
                raise ValueError('tool result roundtrip failed')
        elif kind == 'public-tools-roundtrip':
            schema = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Berlin', 'Paris']}},
                      'required': ['city'], 'additionalProperties': False}
            tool = {'type': 'function', 'function': {'name': 'weather', 'description': 'Get the weather in a city.',
                                                    'parameters': schema, 'strict': True}}
            history = [{'role': 'user', 'content': 'Call weather for both Berlin and Paris. Make two calls now.'}]
            parallel = await complete('correlation-parallel-stream', history, stream=True, tools=[tool],
                                      tool_choice='required', parallel_tool_calls=True)
            calls = parallel.choices[0].message.tool_calls
            if (parallel.choices[0].finish_reason != 'tool_calls' or len(calls or []) != 2
                    or {json.loads(call.function.arguments)['city'] for call in calls} != {'Berlin', 'Paris'}):
                raise ValueError('correlation precondition: two distinct city calls required')
            history.append(parallel.choices[0].message.model_dump(exclude_none=True))
            # The city is intentionally absent from each result. Only the call
            # ID links the value back to its arguments, and arrival is reversed.
            history.extend({'role': 'tool', 'tool_call_id': call.id,
                'content': json.dumps({'temperature_celsius': {'Berlin': 7, 'Paris': 23}[
                    json.loads(call.function.arguments)['city']]})} for call in reversed(calls))
            history.append({'role': 'user', 'content':
                'Using the tool results, give the temperature for Berlin first, then Paris. '
                'Reply only in the form Berlin: number; Paris: number.'})
            final = await complete('correlation-roundtrip', history, stream=True,
                                   tools=[tool], tool_choice='none', max_tokens=32)
            import re
            content = final.choices[0].message.content or ''
            if (final.choices[0].finish_reason != 'stop'
                    or not re.fullmatch(r'\s*Berlin:\s*7\s*;\s*Paris:\s*23\s*\.?\s*', content)):
                raise ValueError('distinct reversed tool results lost their call correlation')
        elif kind == 'public-vision':
            for name, first, second, formats, stream in (
                    ('vision-two-images', 'red', 'blue', ('PNG', 'JPEG'), False),
                    ('vision-two-images-stream', 'blue', 'red', ('WEBP', 'PNG'), True)):
                parts = [{'type': 'text', 'text': 'First image:'},
                    {'type': 'image_url', 'image_url': {'url': image_url(first, formats[0])}},
                    {'type': 'text', 'text': 'Second image:'},
                    {'type': 'image_url', 'image_url': {'url': image_url(second, formats[1])}},
                    {'type': 'text', 'text': 'Name the dominant colors in image order. Reply with only the two English color words.'}]
                final = await complete(name, [{'role': 'user', 'content': parts}], stream=stream, max_tokens=16)
                text = (final.choices[0].message.content or '').casefold()
                if final.choices[0].finish_reason != 'stop' or first not in text or second not in text or text.index(first) >= text.index(second):
                    raise ValueError('image order/color evidence failed')
        elif kind == 'public-structured':
            messages = [{'role': 'user', 'content': 'Return JSON with status ok, counts [2,3], and note null.'}]
            final = await complete('json-object', messages, response_format={'type': 'json_object'}, max_tokens=64)
            if final.choices[0].finish_reason != 'stop' or type(json.loads(final.choices[0].message.content)) is not dict:
                raise ValueError('JSON object failed')
            final = await complete('json-schema-stream', messages, stream=True,
                response_format={'type': 'json_schema', 'json_schema': {'name': 'report', 'schema': SCHEMA, 'strict': True}}, max_tokens=96)
            if final.choices[0].finish_reason != 'stop':
                raise ValueError('strict schema did not complete')
            from kiron_common.local_inference.json_schema import compile_schema, validate_instance
            validate_instance(json.loads(final.choices[0].message.content), compile_schema(SCHEMA, strict=True))
        elif kind == 'public-reasoning':
            for name, stream, limit in (('reasoning', False, 96), ('reasoning-stream', True, 96), ('reasoning-length', True, 4)):
                final = await complete(name, [{'role': 'user', 'content': 'What is 17 times 19?'}],
                                       stream=stream, max_tokens=limit, reasoning_effort='low', allow_length=limit == 4)
                details = final.usage.completion_tokens_details
                if details is None or details.reasoning_tokens is None or details.reasoning_tokens < 1:
                    raise ValueError('exact reasoning usage missing')
                if limit == 4:
                    if final.choices[0].finish_reason != 'length' or final.usage.completion_tokens != 4:
                        raise ValueError('reasoning length boundary failed')
                elif final.choices[0].finish_reason != 'stop' or '323' not in (final.choices[0].message.content or ''):
                    raise ValueError('reasoning final answer failed')
                if 'reasoning_content' in final.choices[0].message.model_dump():
                    raise ValueError('native reasoning extension leaked into public Chat')
    save('request-records', [{'id': value.id, 'path': value.path, 'status': value.status_code,
        'state': value.state, 'error_code': value.error_message, 'tokens': value.tokens_generated} for value in records.values.values()])
    return {'sdk_version': openai.__version__, 'kind': kind, 'public_transport': 'httpx.ASGITransport', 'results': results}
