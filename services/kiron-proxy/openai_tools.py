"""Pure function-tool validation shared by public decoding and native adapters.

No name is dispatched and no reference is fetched. History is supplied entirely
by the caller; this module has no conversation state or network dependencies.
"""
from dataclasses import dataclass, replace
import json
import math
import re

from kiron_common.local_inference import (
    FinishReason, Message, MessageRole, ToolCall, ToolChoice, ToolChoiceKind, ToolDefinition,
)
from kiron_common.local_inference.json_schema import (
    InstanceValidationError, SchemaError, compile_schemas, validate_instance,
)

MAX_CALLS = 64
MAX_ARGUMENT_BYTES = 256 * 1024
NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class ToolError(ValueError):
    def __init__(self, code="invalid_request", param=None):
        super().__init__("Invalid or unsupported function-tool value.")
        self.code, self.param = code, param


def closed(value, allowed, required=(), param=None):
    if type(value) is not dict or not set(required) <= value.keys():
        raise ToolError(param=param)
    if value.keys() - set(allowed):
        raise ToolError("unsupported_parameter", param)


def plain(value):
    """Create transport JSON from immutable canonical mappings/tuples."""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(child) for child in value]
    return value


def arguments_object(source, *, param=None):
    if type(source) is not str:
        raise ToolError(param=param)
    try:
        if len(source.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            raise ToolError(param=param)
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
                if depth > 32:
                    raise ToolError(param=param)
            elif char in "]}":
                depth -= 1

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ToolError(param=param)
                result[key] = value
            return result

        def constant(_):
            raise ToolError(param=param)

        value = json.loads(source, object_pairs_hook=pairs, parse_constant=constant)
        def check(item):
            if type(item) is str:
                item.encode("utf-8")
            elif type(item) is float and not math.isfinite(item):
                raise ToolError(param=param)
            elif type(item) is dict:
                for key, child in item.items():
                    check(key)
                    check(child)
            elif type(item) is list:
                for child in item:
                    check(child)
        check(value)
    except (UnicodeError, ValueError, RecursionError):
        raise ToolError(param=param) from None
    if type(value) is not dict:
        raise ToolError(param=param)
    return value


@dataclass(frozen=True, slots=True)
class ParsedTools:
    tools: tuple[ToolDefinition, ...]
    choice: ToolChoice
    parallel: bool


def compiled_tools(tools):
    return compile_schemas(tuple((tool.parameters, tool.strict) for tool in tools))


def _same_json(left, right):
    # JSON numbers share a value domain; Booleans do not alias 0/1.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is dict and type(right) is dict:
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def parse_tools(data):
    old = bool({"functions", "function_call"} & data.keys())
    if old and {"tools", "tool_choice", "parallel_tool_calls"} & data.keys():
        raise ToolError(param="functions")
    key = "functions" if old else "tools"
    definitions = data.get(key, [])
    if type(definitions) is not list or len(definitions) > MAX_CALLS:
        raise ToolError(param=key)
    tools, names = [], set()
    for index, definition in enumerate(definitions):
        path = f"{key}[{index}]"
        if not old:
            closed(definition, ("type", "function"), ("type", "function"), path)
            if definition["type"] != "function":
                raise ToolError(param=path + ".type")
            definition = definition["function"]
            path += ".function"
        closed(definition, ("name", "description", "parameters", "strict"), ("name", "parameters"), path)
        name = definition["name"]
        if (type(name) is not str or not NAME.fullmatch(name) or name in names
                or type(definition["parameters"]) is not dict
                or type(definition.get("description", "")) is not str
                or type(definition.get("strict", False)) is not bool):
            raise ToolError(param=path)
        names.add(name)
        tools.append(ToolDefinition(name, definition["parameters"], definition.get("description", ""),
                                    definition.get("strict", False)))
    try:
        compiled_tools(tools)
    except SchemaError as exc:
        index = exc.schema_index if exc.schema_index is not None else 0
        path = f"{key}[{index}]" + ("" if old else ".function") + ".parameters"
        raise ToolError(exc.code.value, path) from None
    choice_key = "function_call" if old else "tool_choice"
    choice = data.get(choice_key, "auto" if tools else "none")
    if type(choice) is dict:
        if old:
            closed(choice, ("name",), ("name",), choice_key)
            name = choice["name"]
        else:
            closed(choice, ("type", "function"), ("type", "function"), choice_key)
            closed(choice["function"], ("name",), ("name",), choice_key)
            if choice["type"] != "function":
                raise ToolError(param=choice_key)
            name = choice["function"]["name"]
        if type(name) is not str or name not in names:
            raise ToolError(param=choice_key)
        selected = ToolChoice(ToolChoiceKind.NAMED, name)
    else:
        allowed = ("none", "auto") if old else ("none", "auto", "required")
        if type(choice) is not str or choice not in allowed or (choice == "required" and not tools):
            raise ToolError(param=choice_key)
        selected = ToolChoice(ToolChoiceKind(choice))
    parallel = data.get("parallel_tool_calls", False)
    if type(parallel) is not bool:
        raise ToolError(param="parallel_tool_calls")
    return ParsedTools(tuple(tools), selected, parallel)


def parse_calls(value, *, param):
    if type(value) is not list or len(value) > MAX_CALLS:
        raise ToolError(param=param)
    calls = []
    for index, raw in enumerate(value):
        path = f"{param}[{index}]"
        closed(raw, ("id", "type", "function", "index"), ("id", "type", "function"), path)
        function = raw["function"]
        closed(function, ("name", "arguments", "parsed_arguments"), ("name", "arguments"), path + ".function")
        if (raw["type"] != "function" or type(raw["id"]) is not str or not raw["id"]
                or type(function["name"]) is not str or not NAME.fullmatch(function["name"])):
            raise ToolError(param=path)
        arguments = arguments_object(function["arguments"], param=path + ".function.arguments")
        # Official SDK 2.29.0 stream helpers decorate their final call objects.
        # Accept exactly these checked replay annotations, never trust them as
        # independent input and never emit them on our public response wire.
        if "index" in raw and (type(raw["index"]) is not int or raw["index"] != index):
            raise ToolError(param=path + ".index")
        if "parsed_arguments" in function and (type(function["parsed_arguments"]) is not dict
                or not _same_json(function["parsed_arguments"], arguments)):
            raise ToolError(param=path + ".function.parsed_arguments")
        calls.append(ToolCall(index, raw["id"], function["name"], function["arguments"]))
    if len({call.id for call in calls}) != len(calls):
        raise ToolError(param=param)
    return tuple(calls)


def assistant_turn(messages, *, param):
    """Normalize output items of one turn; never loosen pending-result checks."""
    items, seen, count = [], set(), 0
    for message in messages:
        if message.role is not MessageRole.ASSISTANT:
            raise ToolError(param=param)
        for item in message.assistant_items:
            if isinstance(item, ToolCall):
                if item.id in seen or count >= MAX_CALLS:
                    raise ToolError(param=param + ".tool_calls")
                seen.add(item.id)
                item = replace(item, index=count)
                count += 1
            items.append(item)
    return Message(MessageRole.ASSISTANT, assistant_items=tuple(items))


def validate_history(messages):
    seen, pending = set(), set()
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if message.role is MessageRole.TOOL:
            if message.tool_call_id not in pending:
                raise ToolError(param=path + ".tool_call_id")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ToolError(param=path)
            for call in message.tool_calls:
                if call.id in seen or not call.complete:
                    raise ToolError(param=path + ".tool_calls")
                arguments_object(call.arguments, param=path + ".tool_calls")
                seen.add(call.id)
                pending.add(call.id)
    if pending:
        raise ToolError(param="messages")


def validate_calls(calls, tools, choice, parallel, finish, *, history=()):
    """Validate definitive backend semantics; caller maps failures to 502."""
    if (len(calls) > MAX_CALLS or len({c.id for c in calls}) != len(calls)
            or [c.index for c in calls] != list(range(len(calls)))):
        raise ToolError()
    previous_ids = {call.id for message in history for call in message.tool_calls}
    if any(call.id in previous_ids for call in calls):
        raise ToolError()
    if choice.kind is ToolChoiceKind.NONE and calls:
        raise ToolError()
    if (not parallel or choice.kind is ToolChoiceKind.NAMED) and len(calls) > 1:
        raise ToolError()
    if finish is not FinishReason.LENGTH:
        if choice.kind in (ToolChoiceKind.REQUIRED, ToolChoiceKind.NAMED) and not calls:
            raise ToolError()
        if bool(calls) != (finish is FinishReason.TOOL_CALLS):
            raise ToolError()
    schemas = dict(zip((t.name for t in tools), compiled_tools(tools)))
    for call in calls:
        if (call.name not in schemas or (choice.kind is ToolChoiceKind.NAMED and call.name != choice.name)
                or type(call.arguments) is not str or len(call.arguments.encode("utf-8")) > MAX_ARGUMENT_BYTES):
            raise ToolError()
        if not call.complete:
            if finish is not FinishReason.LENGTH:
                raise ToolError()
            continue
        value = arguments_object(call.arguments)
        if schemas[call.name].strict:
            try:
                validate_instance(value, schemas[call.name])
            except InstanceValidationError:
                raise ToolError() from None
