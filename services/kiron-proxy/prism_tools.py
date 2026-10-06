"""Pure, bounded translation of canonical tools for the pinned Qwen parser.

Strict schemas use an object parameter to avoid its unconstrained XML string
argument branch. This implementation does not publish capability evidence.
"""
from dataclasses import dataclass
import json

from kiron_common.local_inference import (
    AssistantTextItem, ErrorCode, EventKind, FinishReason, InferenceEvent, LocalInferenceError,
    MessageRole, OutputEventLayout, RuntimeFailure, TextPart, ToolCall, ToolChoiceKind,
)
from openai_tools import (
    MAX_ARGUMENT_BYTES, MAX_CALLS, ToolError, arguments_object, compiled_tools,
    plain, validate_calls, validate_history,
)


def protocol():
    return LocalInferenceError(RuntimeFailure(ErrorCode.PROVIDER_ERROR, "Invalid native tool response"))


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ToolPlan:
    request: object
    wrapped: frozenset[str]
    payload: dict
    messages: list[dict]


def _assistant_segments(message, base, wrapped):
    """Project one ordered turn into the native content-before-calls form.

    Text after a call starts a new native assistant segment, never a new
    canonical turn. The pending result set remains owned by the complete turn.
    """
    segments, current = [], None
    for item in message.assistant_items:
        if isinstance(item, AssistantTextItem):
            if current is not None:
                segments.append(current)
            current = {**base, "content": "".join(part.text for part in item.content)}
        elif isinstance(item, ToolCall):
            if current is None:
                current = {**base, "content": ""}
            current.setdefault("tool_calls", []).append({"id": item.id, "type": "function", "function": {
                "name": item.name,
                "arguments": '{"payload":' + item.arguments + '}' if item.name in wrapped else item.arguments,
            }})
        else:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_CAPABILITY,
                "Native reasoning history is not verified"))
    if current is not None:
        segments.append(current)
    return segments or [dict(base)]


def prepare_tools(request, *, messages=None):
    """Build only tool fields/history; caller owns all transport and sampling."""
    try:
        validate_history(request.messages)
        schemas = compiled_tools(request.tools)
        wrapped = frozenset(tool.name for tool in request.tools if tool.strict)
        definitions = []
        for tool, compiled in zip(request.tools, schemas):
            schema = plain(compiled.expanded)
            if tool.strict:
                schema = {"type": "object", "properties": {"payload": schema},
                          "required": ["payload"], "additionalProperties": False}
            definitions.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description, "parameters": schema}})
        choice = request.tool_choice
        native_choice, parallel = choice.kind.value, request.options.parallel_tool_calls
        if choice.kind is ToolChoiceKind.NAMED:
            definitions = [d for d in definitions if d["function"]["name"] == choice.name]
            native_choice, parallel = "required", False
        if choice.kind is ToolChoiceKind.NONE:
            definitions = []
        payload = {"tools": definitions, "tool_choice": native_choice,
                   "parallel_tool_calls": parallel}
        if messages is None:
            if any(not isinstance(part, TextPart) for m in request.messages for part in m.content):
                raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_CAPABILITY, "Image mapping requires its verified helper"))
            messages = [{"role": m.role.value, "content": "".join(p.text for p in m.content)}
                        for m in request.messages]
        if len(messages) != len(request.messages):
            raise ToolError()
        native, pending, answers = [], (), {}
        for message, base in zip(request.messages, messages):
            value = dict(base)
            if message.role is MessageRole.ASSISTANT:
                pending = message.tool_calls
                native.extend(_assistant_segments(message, base, wrapped))
                continue
            if message.role is MessageRole.TOOL:
                value["tool_call_id"] = message.tool_call_id
                answers[message.tool_call_id] = value
                # The pinned Qwen template emits bare <tool_response> blocks;
                # neither their call IDs nor function names survive rendering.
                # Resolve IDs here, then preserve the assistant's call order.
                # validate_history has already required each result exactly once.
                if len(answers) == len(pending):
                    native.extend(answers[call.id] for call in pending)
                    pending, answers = (), {}
                continue
            native.append(value)
        return ToolPlan(request, wrapped, payload, native)
    except (ValueError, TypeError) as exc:
        raise LocalInferenceError(RuntimeFailure(ErrorCode.INVALID_REQUEST, "Invalid canonical tool request")) from exc


def _unwrapped(source, *, incomplete=False):
    """Extract the exact inner JSON text, including a length-limited prefix.

    A small structural scanner removes the known wrapper, never repairs JSON.
    Definitive responses must also pass the full JSON and schema validators.
    """
    if len(source.encode("utf-8")) > MAX_ARGUMENT_BYTES + 64:
        raise ToolError()
    position = 0
    def space():
        nonlocal position
        while position < len(source) and source[position] in " \t\r\n":
            position += 1
    def literal(value):
        nonlocal position
        space()
        if position == len(source) and incomplete:
            return False
        if not source.startswith(value, position):
            raise ToolError()
        position += len(value)
        return True
    if not literal("{"):
        return ""
    space()
    try:
        key, end = json.JSONDecoder().raw_decode(source, position)
    except json.JSONDecodeError:
        if incomplete and ('"payload"'.startswith(source[position:]) or not source[position:]):
            return ""
        raise ToolError() from None
    if key != "payload":
        raise ToolError()
    position = end
    if not literal(":"):
        return ""
    space()
    start = position
    if position == len(source) and incomplete:
        return ""
    if position == len(source) or source[position] != "{":
        raise ToolError()
    stack, quoted, escaped = [], False, False
    end = None
    for index in range(start, len(source)):
        char = source[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
            if len(stack) > 32:
                raise ToolError()
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                raise ToolError()
            if not stack:
                end = index + 1
                break
    if end is None:
        if incomplete:
            return source[start:]
        raise ToolError()
    suffix = source[end:].strip()
    if suffix != "}" and not (incomplete and not suffix):
        raise ToolError()
    return source[start:end]


def _call(index, raw, plan, *, incomplete=False):
    if (type(raw) is not dict or raw.get("type") != "function"
            or type(raw.get("id")) is not str or not raw["id"]):
        raise ToolError()
    function = raw.get("function")
    if type(function) is not dict or type(function.get("name")) is not str:
        raise ToolError()
    source = function.get("arguments")
    if type(source) is not str:
        raise ToolError()
    if function["name"] in plan.wrapped:
        source = _unwrapped(source, incomplete=incomplete)
    if len(source.encode("utf-8")) > MAX_ARGUMENT_BYTES:
        raise ToolError()
    if not incomplete:
        arguments_object(source)
    return ToolCall(index, raw["id"], function["name"], source, complete=not incomplete)


def decode_tool_calls(raw_calls, plan, *, finish_reason):
    try:
        finish = FinishReason(finish_reason)
        if type(raw_calls) is not list or len(raw_calls) > MAX_CALLS:
            raise ToolError()
        calls = tuple(_call(index, raw, plan, incomplete=finish is FinishReason.LENGTH)
                      for index, raw in enumerate(raw_calls))
        request = plan.request
        validate_calls(calls, request.tools, request.tool_choice, request.options.parallel_tool_calls, finish,
                       history=request.messages)
        return calls
    except (ToolError, TypeError, ValueError, UnicodeError):
        raise protocol() from None


class ToolStreamDecoder:
    """Retain native argument fragments; unwrap strict calls at their terminal.

    Wrapper fragments are never exposed publicly. At most 64 bounded argument
    buffers exist. Length termination emits their original inner prefixes and
    deliberately emits no tool_call_completed event.
    """
    def __init__(self, plan, *, layout=None):
        self.plan, self.calls, self.finished = plan, {}, False
        self.layout = layout if layout is not None else OutputEventLayout()

    def feed(self, raw_delta_calls):
        try:
            if self.finished or type(raw_delta_calls) is not list:
                raise ToolError()
            events = []
            for raw in raw_delta_calls:
                if type(raw) is not dict or type(raw.get("index")) is not int or not 0 <= raw["index"] < MAX_CALLS:
                    raise ToolError()
                index = raw["index"]
                function = raw.get("function")
                if type(function) is not dict:
                    raise ToolError()
                if index not in self.calls:
                    name, call_id = function.get("name"), raw.get("id")
                    if (type(name) is not str or name not in {t.name for t in self.plan.request.tools}
                            or raw.get("type") != "function" or type(call_id) is not str or not call_id
                            or call_id in {c["id"] for c in self.calls.values()}):
                        raise ToolError()
                    self.calls[index] = {"id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}
                    events.append(InferenceEvent(EventKind.TOOL_CALL_STARTED, self.plan.request.context.request_id,
                                                  call_index=index, call_id=call_id, name=name, **self.layout.tool(index)))
                elif (raw.get("id") is not None or raw.get("type") is not None or function.get("name") is not None):
                    raise ToolError()
                fragment = function.get("arguments", "")
                if type(fragment) is not str:
                    raise ToolError()
                native = self.calls[index]["function"]
                native["arguments"] += fragment
                cap = MAX_ARGUMENT_BYTES + (64 if native["name"] in self.plan.wrapped else 0)
                if len(native["arguments"].encode("utf-8")) > cap:
                    raise ToolError()
                if native["name"] not in self.plan.wrapped:
                    events.append(InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA, self.plan.request.context.request_id,
                                                  call_index=index, text=fragment, **self.layout.tool(index)))
            return tuple(events)
        except (ToolError, TypeError, ValueError, UnicodeError):
            raise protocol() from None

    def finish(self, finish_reason):
        if self.finished or sorted(self.calls) != list(range(len(self.calls))):
            raise protocol()
        self.finished = True
        calls = decode_tool_calls([self.calls[i] for i in range(len(self.calls))], self.plan,
                                  finish_reason=finish_reason)
        result = []
        for call in calls:
            if call.name in self.plan.wrapped:
                result.append(InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA, self.plan.request.context.request_id,
                                              call_index=call.index, text=call.arguments, **self.layout.tool(call.index)))
            if call.complete:
                result.append(InferenceEvent(EventKind.TOOL_CALL_COMPLETED, self.plan.request.context.request_id,
                                              tool_call=call, **self.layout.tool(call.index)))
        return tuple(result)
