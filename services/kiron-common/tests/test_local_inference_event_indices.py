"""Index coordinates are a required canonical contract, not serializer guesses."""
from dataclasses import replace

import pytest

from kiron_common.local_inference import (
    EventIndexState, EventKind, InferenceEvent, OutputEventLayout, ReasoningKind, ToolCall,
)


def test_closed_payloads_require_exact_nonnegative_index_coordinates():
    for fields in ({}, {"output_item_index": 0}, {"output_item_index": True, "part_index": 0},
                   {"output_item_index": -1, "part_index": 0},
                   {"output_item_index": 0, "part_index": False},
                   {"output_item_index": 0, "summary_index": 0},
                   {"output_item_index": 0, "part_index": 0, "summary_index": 0}):
        with pytest.raises((TypeError, ValueError)):
            InferenceEvent(EventKind.TEXT_DELTA, "r", text="x", **fields)
    with pytest.raises(ValueError):
        InferenceEvent(EventKind.REASONING_DELTA, "r", text="summary", reasoning_kind=ReasoningKind.SUMMARY,
                       output_item_index=0, part_index=0)
    InferenceEvent(EventKind.REASONING_DELTA, "r", text="summary", reasoning_kind=ReasoningKind.SUMMARY,
                   output_item_index=0, summary_index=0)


def test_allocator_shares_item_order_without_confusing_call_indices():
    layout = OutputEventLayout()
    assert layout.reasoning(ReasoningKind.TEXT) == {"output_item_index": 0, "part_index": 0}
    assert layout.text() == {"output_item_index": 1, "part_index": 0}
    assert layout.tool(1) == {"output_item_index": 2}
    assert layout.tool(0) == {"output_item_index": 3}
    assert layout.reasoning(ReasoningKind.SUMMARY) == {"output_item_index": 0, "summary_index": 0}
    assert layout.tool(1) == {"output_item_index": 2}


def test_item_part_gaps_kind_changes_and_closed_call_reopening_are_rejected():
    text = InferenceEvent(EventKind.TEXT_DELTA, "r", text="x", output_item_index=0, part_index=0)
    for invalid in (replace(text, output_item_index=1), replace(text, part_index=1)):
        with pytest.raises(ValueError):
            EventIndexState().accept(invalid)
    state = EventIndexState()
    state.accept(text)
    with pytest.raises(ValueError):
        state.accept(InferenceEvent(EventKind.REASONING_DELTA, "r", text="x", reasoning_kind=ReasoningKind.TEXT,
                                    output_item_index=0, part_index=0))
    started = InferenceEvent(EventKind.TOOL_CALL_STARTED, "r", call_index=1, call_id="a", name="f", output_item_index=1)
    state.accept(started)
    with pytest.raises(ValueError):
        state.accept(InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA, "r", call_index=1, text="{}", output_item_index=2))
    state.accept(InferenceEvent(EventKind.TOOL_CALL_COMPLETED, "r", tool_call=ToolCall(1, "a", "f", "{}"), output_item_index=1))
    with pytest.raises(ValueError):
        state.accept(InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA, "r", call_index=1, text="x", output_item_index=1))
