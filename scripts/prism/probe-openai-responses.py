"""Bounded Responses measurement candidate for the isolated controller harness.

No capability here is production evidence. The harness owns the 600s workflow,
resident model and cleanup; this module only calls its authenticated ASGI API.
"""
import importlib.util
import json
from pathlib import Path
import secrets

import httpx


KINDS = frozenset(('public-responses', 'public-responses-budget'))
# Six bounded requests total 160 output tokens, including reasoning. The outer
# controller deadline remains 600 seconds; this adds no retry or extra load.
BUDGETS = {'text': 8, 'text-stream': 8, 'tools-stream': 64, 'tools-replay': 16,
           'tools-ordered-replay': 16, 'reasoning-stream': 48}
ORDERED_HISTORY_TEXT = 'I will use both tool results to answer the city question.'
BUDGET_PROMPT = ('First count from 1 to 100 in your private reasoning, writing every number. '
    'Then compute 17 times 19. The final answer must be only the result of the multiplication.')
TOOL = {'type': 'function', 'name': 'weather', 'description': 'Get the value for a city.', 'strict': True,
    'parameters': {'type': 'object', 'properties': {'city': {'type':'string','enum':['Berlin','Paris']}},
                   'required': ['city'], 'additionalProperties': False}}


def _features():
    spec = importlib.util.spec_from_file_location('responses_probe_features', Path(__file__).with_name('probe-openai-features.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NativeTrace = _features().NativeTrace


def capabilities(kind, evidence):
    from kiron_common.local_inference import CapabilityName, CapabilitySet
    if kind not in KINDS:
        raise ValueError('unknown Responses probe')
    module = _features()
    values = dict(module.capabilities('public-tools', evidence).by_name)
    values[CapabilityName.REASONING] = module.capabilities('public-reasoning', evidence).by_name[CapabilityName.REASONING]
    return CapabilitySet(values)


def validate_usage(input_tokens, output_tokens, total_tokens, cached, reasoning, *, budget):
    """SDK shape validation does not establish token arithmetic or subcounts."""
    if (any(type(value) is not int for value in (input_tokens, output_tokens, total_tokens, cached, reasoning))
            or not 0 <= cached <= input_tokens or not 0 <= reasoning <= output_tokens
            or not 1 <= output_tokens <= budget or total_tokens != input_tokens + output_tokens):
        raise ValueError('usage bounds or arithmetic failed')


def validate_budget_events(events):
    terminals = [event for event in events if event['type'] in
                 ('response.completed', 'response.incomplete', 'response.failed')]
    if (len(terminals) != 1 or terminals[0]['type'] != 'response.incomplete'
            or terminals[0] is not events[-1]
            or any(type(event.get('sequence_number')) is not int for event in events)
            or [event['sequence_number'] for event in events] != list(range(len(events)))):
        raise ValueError('budget terminal identity or sequence failed')
    final = terminals[0]['response']
    usage = final['usage']
    validate_usage(usage['input_tokens'], usage['output_tokens'], usage['total_tokens'],
                   usage['input_tokens_details']['cached_tokens'],
                   usage['output_tokens_details']['reasoning_tokens'], budget=48)
    reasoning = [item for item in final['output'] if item['type'] == 'reasoning']
    if (final['status'] != 'incomplete' or final['incomplete_details'] != {'reason': 'max_output_tokens'}
            or usage['output_tokens'] != 48 or usage['output_tokens_details']['reasoning_tokens'] != 32
            or not any(type(part.get('text')) is str and bool(part['text'])
                       for item in reasoning for part in item.get('content', []))
            or any(item.get('summary') != [] for item in reasoning)):
        raise ValueError('exact forced-budget LENGTH evidence failed')
    return final


def validate_result(value, name):
    """Exact native details are required even when the observed value is zero."""
    usage = value.usage
    if (value.status != 'completed' or usage is None or usage.input_tokens_details is None
            or usage.output_tokens_details is None):
        raise ValueError(name + ': incomplete result or missing usage')
    cached, reasoning = usage.input_tokens_details.cached_tokens, usage.output_tokens_details.reasoning_tokens
    validate_usage(usage.input_tokens, usage.output_tokens, usage.total_tokens, cached, reasoning, budget=BUDGETS[name])
    if name == 'reasoning-stream':
        thoughts = [part.text for item in value.output if item.type == 'reasoning' for part in (item.content or [])]
        if (reasoning < 1 or not any(thoughts) or '323' not in value.output_text
                or any(item.summary for item in value.output if item.type == 'reasoning')):
            raise ValueError('reasoning output or exact usage failed')
    elif reasoning != 0:
        raise ValueError(name + ': disabled reasoning count was not zero')


def replay_input(history, result):
    """Keep every SDK helper output item, including parsed_arguments annotation."""
    calls = [item for item in result.output if item.type == 'function_call']
    if (len(calls) != 2 or len({call.call_id for call in calls}) != 2
            or any(call.name != 'weather' or call.status != 'completed' for call in calls)):
        raise ValueError('parallel Responses calls missing or ambiguous')
    cities = {call.call_id: json.loads(call.arguments)['city'] for call in calls}
    if set(cities.values()) != {'Berlin', 'Paris'}:
        raise ValueError('parallel Responses city arguments differ')
    # Results deliberately arrive in the opposite call order and carry distinct
    # values, so the final answer must follow call_id associations, not position.
    result_values = {'Berlin':'ALPHA', 'Paris':'BETA'}
    return [*history, *[item.model_dump(exclude_none=True) for item in result.output],
        *[{'type':'function_call_output', 'call_id':call.call_id,
           'output':json.dumps({'value':result_values[cities[call.call_id]]})} for call in reversed(calls)],
        {'role':'user','content':'Use the returned values. Reply only Berlin=<returned value>;Paris=<returned value>, with no spaces.'}]


def ordered_replay_input(history, result):
    """Construct a mixed history; never label its inserted text as generated."""
    replay = replay_input(history, result)
    calls = [item.model_dump(exclude_none=True) for item in result.output if item.type == 'function_call']
    return [*history, calls[0], {'type': 'message', 'role': 'assistant', 'content': [
        {'type': 'output_text', 'text': ORDERED_HISTORY_TEXT, 'annotations': []}]}, calls[1],
        *[item for item in replay[len(history):] if item.get('type') == 'function_call_output'], replay[-1]]


async def probe(service, model, store, kind, *, report=None):
    import openai
    from openai_api import create_openai_api_app
    if kind not in KINDS or openai.__version__ != '2.29.0':
        raise ValueError('pinned Responses probe required')
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

    records, results = Records(), {}
    app = create_openai_api_app(records, Keys())
    app.state.local_inference = service

    def save(name, value):
        results[name] = value
        if report is not None:
            with (Path(report) / ('sdk-responses-' + name + '.json')).open('x') as output:
                json.dump(value, output, indent=2)

    async with openai.AsyncOpenAI(api_key=key, base_url='http://isolated-kiron/v1', max_retries=0,
            _strict_response_validation=True, http_client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app), timeout=None)) as client:
        if kind == 'public-responses-budget':
            # This is intentionally a LENGTH case. SDK 2.29.0's final helper
            # requires response.completed, so consume its typed raw events and
            # inspect the complete incomplete terminal instead.
            events = []
            stream = await client.responses.create(model=model.api_model_id, input=BUDGET_PROMPT,
                max_output_tokens=48, temperature=0, reasoning={'effort': 'low'}, stream=True)
            async for event in stream:
                events.append(event.model_dump(exclude_none=True))
            save('budget-events', events)
            final = validate_budget_events(events)
            save('budget-terminal', final)
            if any(ticket.kind == 'request' for ticket in store.snapshot()):
                raise ValueError('budget request admission remains active')
            return {'sdk_version': openai.__version__, 'kind': kind, 'public_transport': 'httpx.ASGITransport',
                    'output_token_budget': 48, 'results': results}

        async def complete(name, input, *, stream=False, **options):
            kwargs = dict(model=model.api_model_id, input=input, max_output_tokens=BUDGETS[name], temperature=0, **options)
            save(name + '-request', kwargs)
            if stream:
                events = []
                async with client.responses.stream(**kwargs) as response:
                    async for event in response:
                        events.append(event.model_dump(exclude_none=True))
                        # Preserve a failed/incomplete terminal if the SDK final
                        # helper subsequently rejects it; evidence must survive.
                        if event.type in ('response.completed', 'response.incomplete', 'response.failed'):
                            save(name + '-terminal', event.model_dump(exclude_none=True))
                    save(name + '-events', events)
                    final = await response.get_final_response()
                if ([event['sequence_number'] for event in events] != list(range(len(events)))
                        or sum(event['type'] in ('response.completed','response.incomplete','response.failed') for event in events) != 1):
                    raise ValueError(name + ': event identity or terminal count failed')
            else:
                final = await client.responses.create(**kwargs)
            save(name, final.model_dump(exclude_none=True))
            validate_result(final, name)
            if any(ticket.kind == 'request' for ticket in store.snapshot()):
                raise ValueError(name + ': request admission remains active')
            return final

        for name, stream in (('text',False), ('text-stream',True)):
            final = await complete(name, 'Reply only OK.', stream=stream)
            if final.output_text.strip() != 'OK':
                raise ValueError(name + ': expected exact OK response')
        history = [{'role':'user','content':'Call weather for both Berlin and Paris. Make two calls now.'}]
        tools = await complete('tools-stream', history, stream=True,
                               tools=[TOOL], tool_choice='required', parallel_tool_calls=True)
        replay = replay_input(history, tools)
        final = await complete('tools-replay', replay, tools=[TOOL], tool_choice='none', parallel_tool_calls=False)
        if ''.join(final.output_text.split()).strip('`') != 'Berlin=ALPHA;Paris=BETA':
            raise ValueError('distinct tool results were not correctly associated by call_id')
        save('tools-ordered-replay-scope', {'history': 'constructed',
            'output_item_order': ['function_call', 'message', 'function_call'],
            'calls_from': 'tools-stream', 'text': ORDERED_HISTORY_TEXT})
        final = await complete('tools-ordered-replay', ordered_replay_input(history, tools),
                               tools=[TOOL], tool_choice='none', parallel_tool_calls=False)
        if ''.join(final.output_text.split()).strip('`') != 'Berlin=ALPHA;Paris=BETA':
            raise ValueError('constructed mixed-turn history lost call/result association')
        await complete('reasoning-stream', 'What is 17 times 19?', stream=True, reasoning={'effort':'low'})
    save('request-records', [{'id': value.id, 'path': value.path, 'status': value.status_code,
        'state': value.state, 'error_code': value.error_message, 'tokens': value.tokens_generated} for value in records.values.values()])
    return {'sdk_version':openai.__version__, 'kind':kind, 'public_transport':'httpx.ASGITransport',
            'output_token_budget':sum(BUDGETS.values()), 'results':results}
