"""Closed public chat wire profile; no HTTP, resolver or provider access.

The caller owns the source iterator and operation cleanup, including when the
serializer rejects a malformed event sequence before consuming its remainder.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from types import MappingProxyType

from kiron_common.local_inference import (
    ErrorCode, EventIndexState, EventKind, FinishReason, GenerationOptions, InferenceEvent,
    InferenceRequest, InferenceResult, LocalInferenceError, Message, MessageRole,
    ReasoningOptions, SamplingOptions, TextPart, TokenUsage, ToolChoice,
    ToolChoiceKind, ToolCall, OutputFormat, OutputFormatKind, ImagePart, ReasoningPart, ReasoningKind,
)

from openai_tools import (MAX_ARGUMENT_BYTES, MAX_CALLS, NAME, ToolError, arguments_object,
                          assistant_turn, parse_calls, parse_tools, validate_calls, validate_history)

from openai_generation import parse_output_format, validate_request_schemas, validate_output
from kiron_common.local_inference.json_schema import SchemaError, InstanceValidationError

MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 32
DONE = b"data: [DONE]\n\n"
_FIELDS = frozenset("model messages stream stream_options n temperature top_p "
    "max_completion_tokens max_tokens stop seed presence_penalty frequency_penalty "
    "tools tool_choice parallel_tool_calls functions function_call response_format "
    "reasoning_effort logprobs top_logprobs logit_bias store".split())


class ApiError(Exception):
    def __init__(self, message, code, param=None, status=400,
                 error_type="invalid_request_error"):
        super().__init__(message)
        self.message, self.code, self.param = message, code, param
        self.status, self.error_type = status, error_type

    def envelope(self):
        return {"error": {"message": self.message, "type": self.error_type,
                          "param": self.param, "code": self.code}}


def _invalid(param=None):
    return ApiError("Invalid request value or structure.", "invalid_request", param)


def _unsupported(param, capability=False):
    return ApiError("This request feature is not supported.",
                    "unsupported_capability" if capability else "unsupported_parameter", param)


def _protocol():
    return ApiError("The backend returned an invalid response.",
                    "backend_protocol_error", status=502, error_type="server_error")


def _closed(value, allowed, required=(), param=None):
    if type(value) is not dict or not set(required) <= value.keys():
        raise _invalid(param)
    if value.keys() - set(allowed):
        # Do not echo attacker-controlled keys in error text or parameter paths.
        raise _unsupported(param)


def _json_values(value, depth=0):
    if type(value) in (dict, list):
        depth += 1
        if depth > MAX_DEPTH:
            raise _invalid()
        if type(value) is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise _invalid()
                _json_values(key, depth)
                _json_values(child, depth)
        else:
            for child in value:
                _json_values(child, depth)
    elif type(value) is str:
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise _invalid() from None
    elif type(value) is float:
        if not math.isfinite(value):
            raise _invalid()
    elif value is not None and type(value) not in (bool, int):
        raise _invalid()


def decode_json(raw: bytes) -> dict:
    if type(raw) is not bytes:
        raise _invalid()
    if len(raw) > MAX_BODY_BYTES:
        raise ApiError("Request body is too large.", "request_too_large", status=413)
    try:
        source = raw.decode("utf-8", errors="strict")
        # Check nesting before the JSON decoder recurses; braces in strings do
        # not contribute to depth, including escaped quotes and backslashes.
        depth, quoted, escaped = 0, False, False
        for char in source:
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
                depth += 1
                if depth > MAX_DEPTH:
                    raise _invalid()
            elif char in "]}":
                depth -= 1

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise _invalid()
                result[key] = value
            return result

        def constant(_):
            raise _invalid()

        value = json.loads(source, object_pairs_hook=pairs, parse_constant=constant)
        _json_values(value)
    except (UnicodeError, ValueError, RecursionError):
        raise _invalid() from None
    if type(value) is not dict:
        raise _invalid()
    return value


def _model_id(value):
    if (type(value) is not str or not 1 <= len(value) <= 256
            or any(not 33 <= ord(c) <= 126 for c in value)
            or value.startswith("/") or "\\" in value or "://" in value
            or any(part in (".", "..") for part in value.split("/"))):
        raise _invalid("model")
    return value


def _text_content(value, role, param, *, message_index=None, image_parts=None, used_images=None):
    if type(value) is str:
        return (TextPart(value),)
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise _invalid(param)
    parts = []
    for index, part in enumerate(value):
        path = f"{param}[{index}]"
        if type(part) is not dict:
            raise _invalid(path)
        if part.get("type") == "text":
            _closed(part, ("type", "text"), ("type", "text"), path)
            if type(part["text"]) is not str:
                raise _invalid(path + ".text")
            parts.append(TextPart(part["text"]))
        elif part.get("type") == "image_url":
            _closed(part, ("type", "image_url"), ("type", "image_url"), path)
            image = part["image_url"]
            _closed(image, ("url", "detail"), ("url",), path + ".image_url")
            if (role is not MessageRole.USER or type(image["url"]) is not str
                    or not image["url"] or image.get("detail", "auto") not in ("auto", "low", "high")):
                raise _invalid(path)
            key = (message_index, index)
            if image_parts is None or key not in image_parts:
                raise _unsupported(path, capability=True)
            decoded = image_parts[key]
            if not isinstance(decoded, ImagePart) or decoded.detail != image.get("detail", "auto"):
                raise _invalid(path)
            used_images.add(key)
            parts.append(decoded)
        else:
            raise _invalid(path + ".type")
    return tuple(parts)


def _messages(value, *, image_parts=None, assistant_turns=()):
    if type(value) is not list or not 1 <= len(value) <= 512:
        raise _invalid("messages")
    result, used_images = [], set()
    for index, item in enumerate(value):
        path = f"messages[{index}]"
        if type(item) is not dict or type(item.get("role")) is not str:
            raise _invalid(path)
        try:
            role = MessageRole(item["role"])
        except ValueError:
            raise _invalid(path + ".role") from None
        fields = {"role", "content"}
        if role is MessageRole.ASSISTANT:
            fields.add("tool_calls")
        elif role is MessageRole.TOOL:
            fields.add("tool_call_id")
        _closed(item, fields, ("role",), path)
        calls = parse_calls(item.get("tool_calls", []), param=path + ".tool_calls")
        if role is MessageRole.TOOL:
            if type(item.get("tool_call_id")) is not str or not item["tool_call_id"]:
                raise _invalid(path + ".tool_call_id")
            content = _text_content(item.get("content"), role, path + ".content", message_index=index, image_parts=image_parts, used_images=used_images)
            result.append(Message(role, content, tool_call_id=item["tool_call_id"]))
        else:
            content = () if calls and item.get("content") is None else _text_content(item.get("content"), role, path + ".content", message_index=index, image_parts=image_parts, used_images=used_images)
            result.append(Message(role, content, tool_calls=calls))
    if image_parts is not None and used_images != set(image_parts):
        raise _invalid("messages")
    # Responses item runs explicitly identify a single assistant turn. Ordinary
    # Chat messages never receive this grouping and retain strict turn boundaries.
    grouped, origins, cursor = [], [], 0
    for start, stop in assistant_turns:
        if (type(start) is not int or type(stop) is not int
                or not cursor <= start < stop <= len(result)):
            raise _invalid("messages")
        grouped.extend(result[cursor:start])
        origins.extend(range(cursor, start))
        grouped.append(assistant_turn(result[start:stop], param=f"messages[{start}]"))
        origins.append(start)
        cursor = stop
    grouped.extend(result[cursor:])
    origins.extend(range(cursor, len(result)))
    try:
        validate_history(grouped)
    except ToolError as exc:
        if exc.param is not None:
            for index, origin in enumerate(origins):
                prefix = f"messages[{index}]"
                if exc.param.startswith(prefix):
                    raise ToolError(exc.code, f"messages[{origin}]" + exc.param[len(prefix):]) from None
        raise
    return tuple(grouped)


@dataclass(frozen=True, slots=True)
class ParsedChat:
    model_id: str
    messages: tuple[Message, ...]
    stream: bool
    include_usage: bool
    max_tokens: int | None
    max_completion_tokens: int | None
    sampling: SamplingOptions
    reasoning_effort: str | None
    explicit_parameters: Mapping
    tools: tuple = ()
    tool_choice: ToolChoice = ToolChoice(ToolChoiceKind.NONE)
    parallel_tool_calls: bool = False
    output_format: OutputFormat = OutputFormat()

    def __post_init__(self):
        object.__setattr__(self, "explicit_parameters", MappingProxyType(dict(self.explicit_parameters)))

    def to_request(self, model, context, default_max_output_tokens):
        budget = self.max_tokens or self.max_completion_tokens or default_max_output_tokens
        if type(budget) is not int or not 1 <= budget <= 100000:
            raise _invalid("max_completion_tokens")
        return InferenceRequest(model, self.messages,
            GenerationOptions(budget, sampling=self.sampling,
                              reasoning=ReasoningOptions(effort=self.reasoning_effort),
                              parallel_tool_calls=self.parallel_tool_calls,
                              output_format=self.output_format),
            context, tools=self.tools, tool_choice=self.tool_choice)


def parse_chat(data: dict, *, image_parts=None, assistant_turns=()) -> ParsedChat:
    try:
        return _parse_chat(data, image_parts=image_parts, assistant_turns=assistant_turns)
    except ToolError as exc:
        raise ApiError("Invalid or unsupported function-tool value.", exc.code, exc.param) from None
    except SchemaError as exc:
        raise ApiError("Invalid or unsupported output schema.", exc.code.value, "response_format") from None


def _parse_chat(data: dict, *, image_parts=None, assistant_turns=()) -> ParsedChat:
    _json_values(data)
    _closed(data, _FIELDS, ("model", "messages"))
    model_id = _model_id(data["model"])
    stream = data.get("stream", False)
    if type(stream) is not bool:
        raise _invalid("stream")
    include_usage = False
    if "stream_options" in data:
        if not stream:
            raise _invalid("stream_options")
        options = data["stream_options"]
        _closed(options, ("include_usage",), param="stream_options")
        include_usage = options.get("include_usage", False)
        if type(include_usage) is not bool:
            raise _invalid("stream_options.include_usage")
    n = data.get("n", 1)
    if type(n) is not int:
        raise _invalid("n")
    if n != 1:
        raise _unsupported("n")
    for key, expected in (("logprobs", False), ("top_logprobs", 0), ("logit_bias", {}), ("store", False)):
        if key in data:
            if type(data[key]) is not type(expected):
                raise _invalid(key)
            if data[key] != expected:
                raise _unsupported(key)
    explicit, sampling = {}, {}
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                            ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        value = data.get(key)
        if value is not None:
            if type(value) not in (int, float) or not low <= value <= high:
                raise _invalid(key)
            sampling[key] = explicit[key] = value
    seed = data.get("seed")
    if seed is not None:
        if type(seed) is not int or not -(2**63) <= seed < 2**63:
            raise _invalid("seed")
        sampling["seed"] = explicit["seed"] = seed
    if "stop" in data:
        stop = [data["stop"]] if type(data["stop"]) is str else data["stop"]
        if type(stop) is not list or not 1 <= len(stop) <= 4 or any(type(s) is not str or not s for s in stop):
            raise _invalid("stop")
        sampling["stop"] = explicit["stop"] = tuple(stop)
    for key in ("max_tokens", "max_completion_tokens"):
        value = data.get(key)
        if value is not None:
            if type(value) is not int or not 1 <= value <= 100000:
                raise _invalid(key)
            explicit[key] = value
    if data.get("max_tokens") is not None and data.get("max_completion_tokens") is not None:
        raise _invalid("max_completion_tokens")
    effort = data.get("reasoning_effort")
    if effort is not None:
        if type(effort) is not str or effort not in ("none", "minimal", "low", "medium", "high", "xhigh"):
            raise _invalid("reasoning_effort")
        explicit["reasoning_effort"] = effort
    tool_options = parse_tools(data)
    output_format = parse_output_format(data.get("response_format", {"type": "text"}))
    validate_request_schemas(tool_options.tools, output_format)
    return ParsedChat(model_id, _messages(data["messages"], image_parts=image_parts, assistant_turns=assistant_turns), stream, include_usage,
                      data.get("max_tokens"), data.get("max_completion_tokens"),
                      SamplingOptions(**sampling), effort, explicit, tool_options.tools,
                      tool_options.choice, tool_options.parallel, output_format)


def _usage(value):
    if not isinstance(value, TokenUsage):
        raise _protocol()
    # Validate even forged/mutated frozen objects at the public boundary.
    try:
        TokenUsage(value.input_tokens, value.output_tokens, value.cached_input_tokens, value.reasoning_output_tokens)
    except (TypeError, ValueError):
        raise _protocol() from None
    result = {"prompt_tokens": value.input_tokens, "completion_tokens": value.output_tokens,
              "total_tokens": value.total_tokens}
    if value.cached_input_tokens is not None:
        result["prompt_tokens_details"] = {"cached_tokens": value.cached_input_tokens}
    if value.reasoning_output_tokens is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": value.reasoning_output_tokens}
    return result


def _encoded(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise _protocol() from None


def _metadata(model_id, completion_id, created, streaming=False):
    if (type(model_id) is not str or not model_id or len(model_id) > 256
            or type(completion_id) is not str or not completion_id or len(completion_id) > 256
            or type(created) is not int or created < 0):
        raise _protocol()
    return {"id": completion_id, "object": "chat.completion.chunk" if streaming else "chat.completion",
            "created": created, "model": model_id}


def _finish(reason):
    if reason not in (FinishReason.STOP, FinishReason.LENGTH, FinishReason.TOOL_CALLS) or not isinstance(reason, FinishReason):
        raise _protocol()
    return reason.value


def _checked_calls(calls, finish, request=None):
    try:
        if (len(calls) > MAX_CALLS or [c.index for c in calls] != list(range(len(calls)))
                or len({c.id for c in calls}) != len(calls)):
            raise ToolError()
        if finish is not FinishReason.LENGTH and bool(calls) != (finish is FinishReason.TOOL_CALLS):
            raise ToolError()
        for call in calls:
            if (not isinstance(call, ToolCall) or type(call.id) is not str or not call.id
                    or type(call.name) is not str or not NAME.fullmatch(call.name)
                    or type(call.arguments) is not str or len(call.arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES):
                raise ToolError()
            if call.complete:
                arguments_object(call.arguments)
            elif finish is not FinishReason.LENGTH:
                raise ToolError()
        if request is not None:
            validate_calls(calls, request.tools, request.tool_choice, request.options.parallel_tool_calls, finish,
                           history=request.messages)
    except (ToolError, TypeError, ValueError, AttributeError, UnicodeError):
        raise _protocol() from None
    return calls


def _call_value(call):
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}


def _reasoning_enabled(request):
    return request is not None and request.options.reasoning.effort not in (None, "none")


def serialize_completion(result, model_id, completion_id, created, *, request=None):
    if (not isinstance(result, InferenceResult)
            or any(not isinstance(part, TextPart) or type(part.text) is not str for part in result.content)):
        raise _protocol()
    if result.reasoning:
        if not _reasoning_enabled(request) or any(not isinstance(part, ReasoningPart)
                or part.kind is not ReasoningKind.TEXT or type(part.text) is not str for part in result.reasoning):
            raise _protocol()
        try:
            if sum(len(part.text.encode("utf-8")) for part in result.reasoning) > MAX_OUTPUT_BYTES:
                raise _protocol()
        except UnicodeError:
            raise _protocol() from None
    if _reasoning_enabled(request) and (result.usage is None or result.usage.reasoning_output_tokens is None):
        raise _protocol()
    finish = _finish(result.finish_reason)
    calls = _checked_calls(result.tool_calls, result.finish_reason, request)
    content = "".join(p.text for p in result.content)
    if request is not None:
        try:
            validate_output(content, request.options.output_format, result.finish_reason)
        except InstanceValidationError:
            raise _protocol() from None
    message = {"role": "assistant", "content": content if content or not calls else None}
    if calls:
        message["tool_calls"] = [_call_value(call) for call in calls]
    value = _metadata(model_id, completion_id, created)
    value.update(choices=[{"index": 0, "message": message, "finish_reason": finish}], usage=_usage(result.usage))
    if len(_encoded(value)) > MAX_OUTPUT_BYTES:
        raise _protocol()
    return value


def _runtime_error(failure):
    mapping = {
        ErrorCode.INVALID_CONFIGURATION: (503, "model_unavailable"),
        ErrorCode.MODEL_NOT_FOUND: (404, "model_not_found"),
        ErrorCode.PROVIDER_UNAVAILABLE: (503, "provider_unavailable"),
        ErrorCode.UNSUPPORTED_CAPABILITY: (400, "unsupported_capability"),
        ErrorCode.UNSUPPORTED_PARAMETER: (400, "unsupported_parameter"),
        ErrorCode.UNSUPPORTED_VALUE: (400, "unsupported_value"),
        ErrorCode.CONFLICT: (503, "resource_busy"),
        ErrorCode.OVERLOADED: (429, "overloaded"),
        ErrorCode.INVALID_REQUEST: (400, "invalid_request"),
        ErrorCode.CONTEXT_LENGTH_EXCEEDED: (400, "context_length_exceeded"),
        ErrorCode.PROVIDER_ERROR: (502, "backend_protocol_error"),
        ErrorCode.TIMEOUT: (504, "timeout"),
        ErrorCode.CANCELLED: (503, "resource_busy"),
    }
    status, code = mapping.get(failure.code, (500, "internal_error"))
    kind = "rate_limit_error" if status == 429 else "server_error" if status >= 500 else "invalid_request_error"
    # Neither arbitrary provider error messages nor parameter strings cross the
    # public boundary; both may contain paths, prompts or credentials.
    return ApiError("The inference request could not be completed.", code, status=status, error_type=kind)


async def chat_events(events, model_id, completion_id, created, include_usage, *, request=None):
    """Emit text chunks; publish success only after clean canonical iterator EOF."""
    emitted = 0
    # All generated errors are fixed-size generic envelopes. Reserve their tail
    # before emitting any payload; a cap error must itself still fit the cap.
    error_reserve = 1024
    request_id, usage, terminal = None, None, None
    indices = EventIndexState()
    calls, ids = {}, set()
    structured_text = []
    reasoning_bytes = 0
    try:
        base = _metadata(model_id, completion_id, created, streaming=True)
        if type(include_usage) is not bool:
            raise _protocol()

        def chunk(delta, finish=None, usage_only=False):
            value = dict(base)
            value["choices"] = [] if usage_only else [{"index": 0, "delta": delta, "finish_reason": finish}]
            if include_usage:
                value["usage"] = usage if usage_only else None
            return b"data: " + _encoded(value) + b"\n\n"

        async for event in _source_events(events):
            if not isinstance(event, InferenceEvent) or terminal is not None:
                raise _protocol()
            try:
                indices.accept(event)
            except (TypeError, ValueError):
                raise _protocol() from None
            if request_id is None:
                if event.kind is not EventKind.STARTED:
                    raise _protocol()
                request_id = event.request_id
                frame = chunk({"role": "assistant"})
            else:
                if event.request_id != request_id or event.kind is EventKind.STARTED:
                    raise _protocol()
                if event.kind is EventKind.TEXT_DELTA and usage is None:
                    if type(event.text) is not str:
                        raise _protocol()
                    if request is not None and request.options.output_format.kind is not OutputFormatKind.TEXT:
                        structured_text.append(event.text)
                    frame = chunk({"content": event.text})
                elif event.kind is EventKind.REASONING_DELTA and usage is None:
                    if (not _reasoning_enabled(request) or event.reasoning_kind is not ReasoningKind.TEXT
                            or type(event.text) is not str):
                        raise _protocol()
                    try:
                        reasoning_bytes += len(event.text.encode("utf-8"))
                    except UnicodeError:
                        raise _protocol() from None
                    if reasoning_bytes > MAX_OUTPUT_BYTES:
                        raise _protocol()
                    continue  # Chat exposes verified usage/control, not thought text.
                elif event.kind in (EventKind.TOOL_CALL_STARTED, EventKind.TOOL_ARGUMENTS_DELTA, EventKind.TOOL_CALL_COMPLETED) and usage is None:
                    if event.kind is EventKind.TOOL_CALL_STARTED:
                        index = event.call_index
                        if (type(index) is not int or not 0 <= index < MAX_CALLS or index in calls
                                or type(event.call_id) is not str or not event.call_id or event.call_id in ids
                                or type(event.name) is not str or not NAME.fullmatch(event.name)):
                            raise _protocol()
                        import io
                        calls[index] = [event.call_id, event.name, io.StringIO(), 0, False]
                        ids.add(event.call_id)
                    elif event.kind is EventKind.TOOL_ARGUMENTS_DELTA:
                        if event.call_index not in calls or calls[event.call_index][4] or type(event.text) is not str:
                            raise _protocol()
                        state = calls[event.call_index]
                        state[3] += len(event.text.encode("utf-8"))
                        if state[3] > MAX_ARGUMENT_BYTES:
                            raise _protocol()
                        state[2].write(event.text)
                    else:
                        call = event.tool_call
                        if not isinstance(call, ToolCall) or call.index not in calls:
                            raise _protocol()
                        state = calls[call.index]
                        if state[4] or not call.complete or [call.id, call.name, call.arguments] != [state[0], state[1], state[2].getvalue()]:
                            raise _protocol()
                        _checked_calls((ToolCall(0, call.id, call.name, call.arguments),), FinishReason.TOOL_CALLS)
                        state[4] = True
                    continue
                elif event.kind is EventKind.USAGE and usage is None:
                    usage = _usage(event.usage)
                    if _reasoning_enabled(request) and event.usage.reasoning_output_tokens is None:
                        raise _protocol()
                    continue
                elif event.kind is EventKind.COMPLETED and usage is not None:
                    terminal = _finish(event.finish_reason)
                    continue
                elif event.kind is EventKind.FAILED:
                    raise _runtime_error(event.error)
                elif event.kind is EventKind.CANCELLED:
                    raise ApiError("The inference request was cancelled.", "resource_busy", status=503, error_type="server_error")
                else:
                    raise _protocol()
            if emitted + len(frame) + error_reserve > MAX_OUTPUT_BYTES:
                raise _protocol()
            emitted += len(frame)
            yield frame
        if terminal is None:
            raise _protocol()
        if sorted(calls) != list(range(len(calls))):
            raise _protocol()
        assembled = tuple(ToolCall(index, state[0], state[1], state[2].getvalue(), complete=state[4])
                          for index, state in sorted(calls.items()))
        _checked_calls(assembled, FinishReason(terminal), request)
        if request is not None:
            try:
                validate_output("".join(structured_text), request.options.output_format, FinishReason(terminal))
            except InstanceValidationError:
                raise _protocol() from None
        # Hold tool buffers until canonical EOF and validate before exposing a
        # completed call. Native interleaving never leaks into SDK helper order.
        tail = []
        for call in assembled:
            tail.append(chunk({"tool_calls": [{"index": call.index, "id": call.id, "type": "function",
                                               "function": {"name": call.name, "arguments": ""}}]}))
            tail.append(chunk({"tool_calls": [{"index": call.index, "function": {"arguments": call.arguments}}]}))
        tail.append(chunk({}, terminal))
        if include_usage:
            tail.append(chunk({}, usage_only=True))
        tail.append(DONE)
        if emitted + sum(map(len, tail)) > MAX_OUTPUT_BYTES:
            raise _protocol()
        for frame in tail:
            yield frame
    except Exception as exc:
        if isinstance(exc, LocalInferenceError):
            error = _runtime_error(exc.failure)
        elif isinstance(exc, ApiError):
            error = exc
        else:
            error = ApiError("The inference request could not be completed.", "internal_error", status=500, error_type="server_error")
        yield b"data: " + _encoded(error.envelope()) + b"\n\n"
        yield DONE


async def _source_events(events):
    """Only typed runtime failures may influence the public error code."""
    try:
        async for event in events:
            yield event
    except LocalInferenceError:
        raise
    except TimeoutError:
        raise ApiError("The inference deadline was exceeded.", "timeout",
                       status=504, error_type="server_error") from None
    except Exception:
        raise ApiError("The inference request could not be completed.", "internal_error",
                       status=500, error_type="server_error") from None
