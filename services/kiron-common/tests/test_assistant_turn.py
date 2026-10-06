"""An assistant turn has authoritative ordered items and exact projections."""
from dataclasses import FrozenInstanceError, replace
import pytest
from kiron_common.local_inference import AssistantTextItem, Message, MessageRole, TextPart, ToolCall


def test_ordered_turn_is_immutable_and_preserves_parts_and_call_order():
    calls = [ToolCall(i, str(i), 'weather', '{}') for i in range(2)]
    text = AssistantTextItem([TextPart(''), TextPart('A'), TextPart('B')])
    items = [calls[0], text, calls[1]]
    turn = Message(MessageRole.ASSISTANT, assistant_items=items)
    items.clear()
    assert turn.assistant_items == (calls[0], text, calls[1])
    assert turn.content == text.content and turn.tool_calls == tuple(calls)
    assert replace(turn) == turn
    with pytest.raises(FrozenInstanceError):
        turn.assistant_items = ()


def test_conflicting_projections_wrong_roles_and_duplicates_are_rejected():
    call = ToolCall(0, 'a', 'weather', '{}')
    for kwargs in ({'content': (TextPart('lost'),)}, {'tool_calls': (ToolCall(0, 'b', 'weather', '{}'),)}):
        with pytest.raises(ValueError):
            Message(MessageRole.ASSISTANT, assistant_items=(call,), **kwargs)
    with pytest.raises(ValueError):
        Message(MessageRole.USER, assistant_items=(call,))
    with pytest.raises(ValueError):
        Message(MessageRole.ASSISTANT, assistant_items=(call, call))
    with pytest.raises(TypeError):
        AssistantTextItem(())
