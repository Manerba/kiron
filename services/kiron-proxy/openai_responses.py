"""Stateless Responses normalization over the existing Chat/Tools/Vision contract.

No provider, HTTP, session store or inference is accessed here. Reasoning replay
and summary requests remain explicitly closed until a provider proves them.
"""
from dataclasses import dataclass
import copy

from kiron_common.local_inference import (
    FinishReason, InferenceResult, ReasoningKind, TextPart, TokenUsage,
)
from openai_generation import validate_output
from kiron_common.local_inference.json_schema import InstanceValidationError
from openai_tools import plain
from openai_wire import (
    ApiError, ParsedChat, MAX_OUTPUT_BYTES, _checked_calls, _closed, _encoded,
    _finish, _json_values, _model_id, parse_chat,
)

FIELDS = frozenset(('model', 'input', 'instructions', 'stream', 'tools', 'tool_choice',
    'parallel_tool_calls', 'temperature', 'top_p', 'max_output_tokens', 'text', 'reasoning',
    'store', 'background', 'truncation', 'include'))


def invalid(param=None):
    return ApiError('Invalid Responses request value or structure.', 'invalid_request', param)


def unsupported(param=None):
    return ApiError('This Responses capability is not verified.', 'unsupported_capability', param)


def protocol():
    return ApiError('The backend returned an invalid Responses result.', 'backend_protocol_error',
                    status=502, error_type='server_error')


def _identity(value, param):
    if type(value) is not str or not value:
        raise invalid(param)


def _status(item, param):
    if 'id' in item:
        _identity(item['id'], param + '.id')
    if 'status' in item and item['status'] not in ('in_progress', 'completed', 'incomplete'):
        raise invalid(param + '.status')


def _content(value, role, param):
    if type(value) is str:
        return value
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise invalid(param)
    result = []
    for index, part in enumerate(value):
        path = f'{param}[{index}]'
        if type(part) is not dict:
            raise invalid(path)
        kind = part.get('type')
        if kind == 'input_text':
            _closed(part, ('type', 'text'), ('type', 'text'), path)
            result.append({'type': 'text', 'text': part['text']})
        elif kind == 'output_text' and role == 'assistant':
            _closed(part, ('type', 'text', 'annotations', 'logprobs'), ('type', 'text', 'annotations'), path)
            if part['annotations'] != [] or part.get('logprobs', []) != []:
                raise unsupported(path)
            result.append({'type': 'text', 'text': part['text']})
        elif kind == 'input_image':
            if 'file_id' in part:
                raise unsupported(path + '.file_id')
            _closed(part, ('type', 'image_url', 'detail'), ('type', 'image_url'), path)
            if role != 'user':
                raise invalid(path)
            result.append({'type': 'image_url', 'image_url': {
                'url': part['image_url'], 'detail': part.get('detail', 'auto')}})
        else:
            raise unsupported(path + '.type')
    return result


def _reasoning_replay(item, path):
    if 'encrypted_content' in item:
        raise unsupported(path + '.encrypted_content')
    _closed(item, ('type', 'id', 'summary', 'content', 'status'), ('type', 'id', 'summary'), path)
    _status(item, path)
    for name, kind in (('summary', 'summary_text'), ('content', 'reasoning_text')):
        parts = item.get(name, [])
        if type(parts) is not list or len(parts) > 64:
            raise invalid(path + '.' + name)
        for part in parts:
            _closed(part, ('type', 'text'), ('type', 'text'), path + '.' + name)
            if part['type'] != kind or type(part['text']) is not str:
                raise invalid(path + '.' + name)
    raise unsupported(path)  # Never discard native thoughts to fake replay support.


@dataclass(frozen=True, slots=True)
class PreparedResponse:
    chat_data: dict
    message_paths: tuple[str, ...]
    assistant_turns: tuple[tuple[int, int], ...]

    @property
    def stream(self):
        return self.chat_data.get('stream', False)

    def parameter(self, value):
        if value is None:
            return None
        for index, path in reversed(tuple(enumerate(self.message_paths))):
            prefix = f'messages[{index}]'
            if value.startswith(prefix):
                return path + value[len(prefix):].replace('.image_url.url', '.image_url')
        return (value.replace('max_completion_tokens', 'max_output_tokens')
            .replace('response_format.json_schema', 'text.format').replace('response_format', 'text.format')
            .replace('reasoning_effort', 'reasoning.effort').replace('.function.', '.'))


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    chat: ParsedChat

    @property
    def model_id(self):
        return self.chat.model_id

    @property
    def stream(self):
        return self.chat.stream

    def to_request(self, model, context, default_max_output_tokens):
        return self.chat.to_request(model, context, default_max_output_tokens)


def prepare_response(data):
    _json_values(data)
    _closed(data, FIELDS, ('model', 'input'))
    _model_id(data['model'])
    for key in ('store', 'background'):
        if key in data and (type(data[key]) is not bool or data[key]):
            raise unsupported(key) if type(data[key]) is bool else invalid(key)
    if 'truncation' in data and data['truncation'] != 'disabled':
        raise unsupported('truncation')
    if 'include' in data and data['include'] != []:
        raise unsupported('include')
    chat = {key: copy.deepcopy(data[key]) for key in
        ('model', 'stream', 'temperature', 'top_p', 'parallel_tool_calls') if key in data}
    if 'max_output_tokens' in data:
        chat['max_completion_tokens'] = data['max_output_tokens']
    if 'text' in data:
        _closed(data['text'], ('format',), ('format',), 'text')
        value = data['text']['format']
        if type(value) is not dict:
            raise invalid('text.format')
        if value.get('type') == 'json_schema':
            _closed(value, ('type', 'name', 'description', 'schema', 'strict'), ('type', 'name', 'schema'), 'text.format')
            chat['response_format'] = {'type': 'json_schema', 'json_schema': {k: v for k, v in value.items() if k != 'type'}}
        else:
            chat['response_format'] = copy.deepcopy(value)
    if 'reasoning' in data:
        _closed(data['reasoning'], ('effort', 'summary'), (), 'reasoning')
        if 'summary' in data['reasoning']:
            summary = data['reasoning']['summary']
            if type(summary) is not str or summary not in ('auto', 'concise', 'detailed'):
                raise invalid('reasoning.summary')
            raise unsupported('reasoning.summary')
        if 'effort' in data['reasoning']:
            chat['reasoning_effort'] = data['reasoning']['effort']
    if 'tools' in data:
        if type(data['tools']) is not list:
            raise invalid('tools')
        chat['tools'] = []
        for index, tool in enumerate(data['tools']):
            path = f'tools[{index}]'
            if type(tool) is not dict or tool.get('type') != 'function':
                raise unsupported(path)
            _closed(tool, ('type', 'name', 'description', 'parameters', 'strict'), ('type', 'name', 'parameters'), path)
            chat['tools'].append({'type': 'function', 'function': {k: v for k, v in tool.items() if k != 'type'}})
    if 'tool_choice' in data:
        choice = data['tool_choice']
        if type(choice) is dict:
            _closed(choice, ('type', 'name'), ('type', 'name'), 'tool_choice')
            if choice['type'] != 'function':
                raise unsupported('tool_choice')
            choice = {'type': 'function', 'function': {'name': choice['name']}}
        chat['tool_choice'] = choice
    messages, paths = [], []
    if 'instructions' in data:
        if type(data['instructions']) is not str:
            raise invalid('instructions')
        messages.append({'role': 'developer', 'content': data['instructions']})
        paths.append('instructions')
    value = data['input']
    if type(value) is str:
        messages.append({'role': 'user', 'content': value})
        paths.append('input')
    elif type(value) is list and 1 <= len(value) <= 512:
        call_group = None
        for index, item in enumerate(value):
            path = f'input[{index}]'
            if type(item) is not dict:
                raise invalid(path)
            kind = item.get('type', 'message')
            if kind == 'reasoning':
                _reasoning_replay(item, path)
            if kind == 'function_call':
                _closed(item, ('type', 'call_id', 'name', 'arguments', 'id', 'status', 'parsed_arguments'), ('type', 'call_id', 'name', 'arguments'), path)
                _status(item, path)
                if item.get('status', 'completed') != 'completed':
                    raise unsupported(path + '.status')
                if call_group is None:
                    call_group = {'role': 'assistant', 'content': None, 'tool_calls': []}
                    messages.append(call_group)
                    paths.append(path)
                call_group['tool_calls'].append({'id': item['call_id'], 'type': 'function',
                    'function': {'name': item['name'], 'arguments': item['arguments'],
                        **({'parsed_arguments': item['parsed_arguments']} if 'parsed_arguments' in item else {})}})
                continue
            call_group = None
            if kind == 'function_call_output':
                _closed(item, ('type', 'call_id', 'output'), ('type', 'call_id', 'output'), path)
                if type(item['output']) is not str:
                    raise invalid(path + '.output')
                messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': item['output']})
            elif kind == 'message':
                role = item.get('role')
                if role not in ('system', 'developer', 'user', 'assistant'):
                    raise invalid(path + '.role')
                allowed = ('type', 'role', 'content', 'id', 'status') if role == 'assistant' else ('type', 'role', 'content')
                _closed(item, allowed, ('role', 'content'), path)
                _status(item, path)
                messages.append({'role': role, 'content': _content(item['content'], role, path + '.content')})
            else:
                raise unsupported(path + '.type')
            paths.append(path)
    else:
        raise invalid('input')
    chat['messages'] = messages
    # An output array may alternate calls and assistant messages. Its contiguous
    # assistant items form one complete turn; only a user/system/tool item ends
    # that turn. Keep each original message/part position for image/error mapping.
    turns, start = [], None
    for index, message in enumerate(messages):
        if message['role'] == 'assistant':
            if start is None:
                start = index
        elif start is not None:
            turns.append((start, index))
            start = None
    if start is not None:
        turns.append((start, len(messages)))
    return PreparedResponse(chat, tuple(paths), tuple(turns))


def parse_response(prepared, *, image_parts=None):
    if not isinstance(prepared, PreparedResponse):
        raise TypeError('prepare_response must run before decoding image parts')
    try:
        return ParsedResponse(parse_chat(prepared.chat_data, image_parts=image_parts,
                                         assistant_turns=prepared.assistant_turns))
    except ApiError as error:
        raise ApiError(error.message, error.code, prepared.parameter(error.param), error.status, error.error_type) from None


def response_usage(value):
    if not isinstance(value, TokenUsage) or value.cached_input_tokens is None or value.reasoning_output_tokens is None:
        raise protocol()
    try:
        TokenUsage(value.input_tokens, value.output_tokens, value.cached_input_tokens, value.reasoning_output_tokens)
    except (TypeError, ValueError):
        raise protocol() from None
    return {'input_tokens': value.input_tokens, 'input_tokens_details': {'cached_tokens': value.cached_input_tokens},
        'output_tokens': value.output_tokens, 'output_tokens_details': {'reasoning_tokens': value.reasoning_output_tokens},
        'total_tokens': value.total_tokens}


def response_base(request, response_id, created, *, status='in_progress', output=None, usage=None, error=None):
    if type(response_id) is not str or not response_id or type(created) is not int or created < 0:
        raise protocol()
    choice = request.tool_choice
    tools = [{'type': 'function', 'name': tool.name, 'description': tool.description,
              'parameters': plain(tool.parameters), 'strict': tool.strict} for tool in request.tools]
    return {'id': response_id, 'created_at': created, 'object': 'response', 'model': request.model.api_model_id,
        'output': output or [], 'parallel_tool_calls': request.options.parallel_tool_calls,
        'tools': tools, 'tool_choice': {'type': 'function', 'name': choice.name} if choice.name is not None else choice.kind.value,
        'status': status, 'error': error, 'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
        'usage': usage, 'store': False, 'background': False, 'truncation': 'disabled',
        'max_output_tokens': request.options.max_output_tokens,
        'temperature': request.options.sampling.temperature, 'top_p': request.options.sampling.top_p}


def serialize_response(result, *, request, response_id, created):
    if not isinstance(result, InferenceResult) or result.request_id != request.context.request_id:
        raise protocol()
    finish = _finish(result.finish_reason)
    _checked_calls(result.tool_calls, result.finish_reason, request)
    content = ''.join(part.text for part in result.content)
    try:
        validate_output(content, request.options.output_format, result.finish_reason)
    except InstanceValidationError:
        raise protocol() from None
    status = 'incomplete' if finish == 'length' else 'completed'
    output = []
    if result.reasoning:
        if request.options.reasoning.effort in (None, 'none'):
            raise protocol()
        item = {'id': response_id + '_reasoning_' + str(len(output)), 'type': 'reasoning', 'status': status, 'summary': [], 'content': []}
        for part in result.reasoning:
            if part.kind is ReasoningKind.SUMMARY and request.options.reasoning.summary is None:
                raise protocol()
            key, kind = ('summary', 'summary_text') if part.kind is ReasoningKind.SUMMARY else ('content', 'reasoning_text')
            item[key].append({'type': kind, 'text': part.text})
        output.append(item)
    if content or not result.tool_calls:
        output.append({'id': response_id + '_message_' + str(len(output)), 'type': 'message', 'role': 'assistant', 'status': status,
            'content': [{'type': 'output_text', 'text': content, 'annotations': [], 'logprobs': []}]})
    for call in result.tool_calls:
        output.append({'id': response_id + '_call_' + str(call.index), 'type': 'function_call',
            'call_id': call.id, 'name': call.name, 'arguments': call.arguments,
            'status': 'completed' if call.complete else 'incomplete'})
    value = response_base(request, response_id, created, status=status, output=output, usage=response_usage(result.usage))
    if len(_encoded(value)) > MAX_OUTPUT_BYTES:
        raise protocol()
    return value


async def response_events(events, *, request, response_id, created, state=None):
    from openai_responses_stream import response_events as serialize
    async for frame in serialize(events, request=request, response_id=response_id, created=created, state=state):
        yield frame
