"""Pure native Ollama tool mapping with explicit unsupported control modes.

Native call objects carry parsed arguments. Stable KIron IDs bind these calls to
their client history; result messages are sent in original call order because
the native tool-result format correlates by function name/position, not IDs.
"""
from dataclasses import replace
import json
from uuid import NAMESPACE_URL, uuid5

from kiron_common.local_inference import (
    ErrorCode, FinishReason, LocalInferenceError, MessageRole, RuntimeFailure,
    ToolChoiceKind,
)
from openai_tools import MAX_CALLS, arguments_object
from prism_tools import ToolStreamDecoder as PrismStreamDecoder
from prism_tools import decode_tool_calls as decode_prism_calls
from prism_tools import prepare_tools as prepare_prism_tools
from prism_tools import protocol


def prepare_tools(request, *, messages=None):
    if (request.tool_choice.kind in (ToolChoiceKind.REQUIRED, ToolChoiceKind.NAMED)
            or any(tool.strict for tool in request.tools)):
        raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_CAPABILITY,
            "Native Ollama tool selection/strict grammar is not verified"))
    plan = prepare_prism_tools(request, messages=messages)
    history = []
    names = {call.id: call.name for message in request.messages for call in message.tool_calls}
    # A complete assistant turn can project to several ordered native segments.
    # The shared plan already orders results by the turn's unique call IDs.
    for base in plan.messages:
        if base["role"] == MessageRole.TOOL.value:
            history.append({"role": "tool", "tool_name": names[base["tool_call_id"]], "content": base["content"]})
            continue
        value = dict(base)
        if "tool_calls" in base:
            value["tool_calls"] = [{"function": {"name": call["function"]["name"],
                "arguments": arguments_object(call["function"]["arguments"])}} for call in base["tool_calls"]]
        history.append(value)
    return replace(plan, payload={"tools": plan.payload["tools"]}, messages=history)


def canonical_finish(native_reason, has_calls):
    if native_reason not in ("stop", "length"):
        raise protocol()
    return FinishReason.TOOL_CALLS if native_reason == "stop" and has_calls else FinishReason(native_reason)


def _native_call(raw, plan, index):
    if type(raw) is not dict or type(raw.get("function")) is not dict:
        raise protocol()
    function = raw["function"]
    if type(function.get("name")) is not str or type(function.get("arguments")) is not dict:
        raise protocol()
    try:
        source = json.dumps(function["arguments"], allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        arguments_object(source)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise protocol() from None
    # IDs are local correlation values, never fabricated usage or provider IDs.
    call_id = "call_" + uuid5(NAMESPACE_URL, plan.request.context.request_id + ":" + str(index)).hex
    return {"type": "function", "id": call_id,
            "function": {"name": function["name"], "arguments": source}}


def decode_tool_calls(raw_calls, plan, *, finish_reason):
    if type(raw_calls) is not list or len(raw_calls) > MAX_CALLS:
        raise protocol()
    finish = canonical_finish(finish_reason, bool(raw_calls))
    return decode_prism_calls([_native_call(raw, plan, i) for i, raw in enumerate(raw_calls)], plan,
                             finish_reason=finish)


class ToolStreamDecoder:
    """Native Ollama emits complete argument objects, not JSON-string patches."""
    def __init__(self, plan, *, layout=None):
        self.decoder, self.count = PrismStreamDecoder(plan, layout=layout), 0

    @property
    def has_calls(self):
        return bool(self.count)

    def feed(self, raw_delta_calls):
        if type(raw_delta_calls) is not list:
            raise protocol()
        translated = []
        for raw in raw_delta_calls:
            if self.count >= MAX_CALLS:
                raise protocol()
            function = raw.get("function") if type(raw) is dict else None
            if type(function) is not dict:
                raise protocol()
            if "index" in function and (type(function["index"]) is not int or function["index"] != self.count):
                raise protocol()
            translated.append({"index": self.count, **_native_call(raw, self.decoder.plan, self.count)})
            self.count += 1
        return self.decoder.feed(translated)

    def finish(self, finish_reason):
        return self.decoder.finish(canonical_finish(finish_reason, self.has_calls))
