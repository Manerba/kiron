"""Closed canonical event payloads and a finite-sequence contract check."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from ._values import integer, text
from .content import FinishReason, ReasoningKind, TokenUsage, ToolCall
from .lifecycle import RuntimeFailure


class EventKind(str, Enum):
    STARTED = "started"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_ARGUMENTS_DELTA = "tool_arguments_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    USAGE = "usage"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_EVENTS = frozenset((EventKind.COMPLETED, EventKind.FAILED, EventKind.CANCELLED))
_PAYLOADS = {
    EventKind.STARTED: frozenset(),
    EventKind.TEXT_DELTA: frozenset(("text", "output_item_index", "part_index")),
    EventKind.REASONING_DELTA: frozenset(("text", "reasoning_kind", "output_item_index", "part_index")),
    EventKind.TOOL_CALL_STARTED: frozenset(("call_index", "call_id", "name", "output_item_index")),
    EventKind.TOOL_ARGUMENTS_DELTA: frozenset(("call_index", "text", "output_item_index")),
    EventKind.TOOL_CALL_COMPLETED: frozenset(("tool_call", "output_item_index")),
    EventKind.USAGE: frozenset(("usage",)),
    EventKind.COMPLETED: frozenset(("finish_reason",)),
    EventKind.FAILED: frozenset(("error",)),
    EventKind.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    kind: EventKind
    request_id: str
    text: str | None = None
    reasoning_kind: ReasoningKind | None = None
    call_index: int | None = None
    call_id: str | None = None
    name: str | None = None
    tool_call: ToolCall | None = None
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None
    error: RuntimeFailure | None = None
    output_item_index: int | None = None
    part_index: int | None = None
    summary_index: int | None = None

    def __post_init__(self):
        text(self.request_id, "request ID")
        if not isinstance(self.kind, EventKind):
            raise TypeError("event kind must be normalized")
        payload_names = ("text", "reasoning_kind", "call_index", "call_id", "name",
                         "tool_call", "usage", "finish_reason", "error", "output_item_index",
                         "part_index", "summary_index")
        present = frozenset(name for name in payload_names if getattr(self, name) is not None)
        expected = _PAYLOADS[self.kind]
        if self.kind is EventKind.REASONING_DELTA and self.reasoning_kind is ReasoningKind.SUMMARY:
            expected = (expected - {"part_index"}) | {"summary_index"}
        if present != expected:
            raise ValueError(f"invalid payload for {self.kind.value}")
        for name, expected in (("text", str), ("reasoning_kind", ReasoningKind), ("tool_call", ToolCall),
                               ("usage", TokenUsage), ("finish_reason", FinishReason), ("error", RuntimeFailure)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"invalid event {name}")
        for name in ("call_index", "output_item_index", "part_index", "summary_index"):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)
        for name in ("call_id", "name"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)
        if self.tool_call is not None and not self.tool_call.complete:
            raise ValueError("tool_call_completed requires complete arguments")

    @property
    def terminal(self) -> bool:
        return self.kind in TERMINAL_EVENTS


class OutputEventLayout:
    """Allocate a native Chat turn's one text/reasoning item and tool items.

    The provider shares this allocator with its tool decoder. First appearance,
    not call index, determines output-item order. It does no legacy decoding.
    """
    def __init__(self):
        self._items = {}

    def _item(self, key):
        if key not in self._items:
            self._items[key] = len(self._items)
        return self._items[key]

    def text(self):
        return {"output_item_index": self._item(("text",)), "part_index": 0}

    def reasoning(self, kind):
        if not isinstance(kind, ReasoningKind):
            raise TypeError("reasoning kind must be normalized")
        return {"output_item_index": self._item(("reasoning",)),
                "summary_index" if kind is ReasoningKind.SUMMARY else "part_index": 0}

    def tool(self, call_index):
        integer(call_index, "call index")
        return {"output_item_index": self._item(("tool", call_index))}


class EventIndexState:
    """Incremental index validation shared by runtime and wire serializers.

    First appearances are contiguous; fragments may revisit an open item/part.
    Tools close at tool_call_completed; all items close at usage or terminal.
    """
    def __init__(self):
        self._items, self._calls, self._closed = [], {}, False

    def accept(self, event):
        if not isinstance(event, InferenceEvent):
            raise TypeError("noncanonical event")
        index = event.output_item_index
        if index is None:
            if event.kind is EventKind.USAGE or event.terminal:
                self._closed = True
            return
        if self._closed:
            raise ValueError("output after item closure")
        kind = ("message" if event.kind is EventKind.TEXT_DELTA else "reasoning"
                if event.kind is EventKind.REASONING_DELTA else "tool")
        if index == len(self._items):
            if kind == "tool" and event.kind is not EventKind.TOOL_CALL_STARTED:
                raise ValueError("tool item must start before its fragments")
            self._items.append({"kind": kind, "parts": [0, 0], "closed": False})
        if index >= len(self._items):
            raise ValueError("output item indices must start at zero without gaps")
        item = self._items[index]
        if item["kind"] != kind or item["closed"]:
            raise ValueError("output item changed kind or reopened")
        if kind == "tool":
            call_index = event.tool_call.index if event.kind is EventKind.TOOL_CALL_COMPLETED else event.call_index
            if event.kind is EventKind.TOOL_CALL_STARTED:
                if call_index in self._calls or "call_index" in item:
                    raise ValueError("duplicate tool item")
                self._calls[call_index] = index
                item["call_index"] = call_index
            elif self._calls.get(call_index) != index:
                raise ValueError("tool changed output item index")
            if event.kind is EventKind.TOOL_CALL_COMPLETED:
                item["closed"] = True
        else:
            summary = event.summary_index is not None
            part = event.summary_index if summary else event.part_index
            count = item["parts"][summary]
            if part > count:
                raise ValueError("part indices must start at zero without gaps")
            if part == count:
                item["parts"][summary] += 1


def validate_event_sequence(events: Iterable[InferenceEvent]) -> None:
    """Check complete recorded sequences; live ownership stays in RuntimeService.

    Provider tool fragments may interleave. Public emission ordering belongs
    to the API serializer, not to this transport-neutral contract.
    """
    request_id, terminal, usage_seen = None, False, False
    calls, completed, ids = {}, set(), set()
    indices = EventIndexState()
    for event in events:
        if not isinstance(event, InferenceEvent):
            raise TypeError("sequence contains a noncanonical event")
        if terminal:
            raise ValueError("event after terminal state")
        indices.accept(event)
        if request_id is None:
            if event.kind is not EventKind.STARTED:
                raise ValueError("sequence must begin with started")
            request_id = event.request_id
            continue
        if event.request_id != request_id or event.kind is EventKind.STARTED:
            raise ValueError("request identity or start event changed")
        if event.kind is EventKind.TOOL_CALL_STARTED:
            if event.call_index in calls or event.call_id in ids:
                raise ValueError("duplicate tool identity")
            calls[event.call_index] = [event.call_id, event.name, ""]
            ids.add(event.call_id)
        elif event.kind is EventKind.TOOL_ARGUMENTS_DELTA:
            if event.call_index not in calls or event.call_index in completed:
                raise ValueError("arguments require an active tool call")
            calls[event.call_index][2] += event.text
        elif event.kind is EventKind.TOOL_CALL_COMPLETED:
            call = event.tool_call
            if call.index in completed or calls.get(call.index) != [call.id, call.name, call.arguments]:
                raise ValueError("completed call differs from its accumulated fragments")
            completed.add(call.index)
        elif event.kind is EventKind.USAGE:
            if usage_seen:
                raise ValueError("duplicate final usage")
            usage_seen = True
        elif event.kind is EventKind.COMPLETED:
            if event.finish_reason is not FinishReason.LENGTH and set(calls) != completed:
                raise ValueError("successful completion has unfinished tool calls")
            if event.finish_reason is FinishReason.TOOL_CALLS and not completed:
                raise ValueError("tool_calls finish without calls")
        terminal = event.terminal
    if request_id is None or not terminal:
        raise ValueError("sequence has no terminal event")
