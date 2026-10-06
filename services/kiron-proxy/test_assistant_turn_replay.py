"""Own decoder/serializer outputs replay as one ordered, complete assistant turn."""
from copy import deepcopy
import json
import unittest

from kiron_common.local_inference import (
    AssistantTextItem, EventKind, FinishReason, OutputEventLayout, TextPart, TokenUsage, ToolCall,
)
from openai_responses import parse_response, prepare_response
from openai_wire import ApiError, parse_chat
from prism_tools import ToolStreamDecoder, prepare_tools
from ollama_tools import prepare_tools as ollama_plan
from test_openai_responses import body, event, frames, request_for_response, wire
from test_openai_tools import definition, raw_call, raw_request


def tool_definition(strict=False):
    tool = {'type': 'function', **definition(strict=strict)['function']}
    tool['parameters']['properties']['city']['enum'] = ['Berlin', 'Paris']
    return tool


async def mixed_output(order, *, strict=False):
    request = request_for_response(tools=[tool_definition(strict)], parallel_tool_calls=True)
    layout = OutputEventLayout()
    decoder = ToolStreamDecoder(prepare_tools(request), layout=layout)
    events = [event(EventKind.STARTED)]
    for fragment in order:
        if fragment == 'text':
            events.append(event(EventKind.TEXT_DELTA, text='Checking both cities.', **layout.text()))
        else:
            index = int(fragment)
            arguments = {'city': ('Berlin', 'Paris')[index]}
            if strict:
                arguments = {'payload': arguments}
            events.extend(decoder.feed([{'index': index, 'id': f'call_{index}', 'type': 'function',
                'function': {'name': 'weather', 'arguments': json.dumps(arguments)}}]))
    events.extend(decoder.finish(FinishReason.TOOL_CALLS))
    events.extend([event(EventKind.USAGE, usage=TokenUsage(10, 20, 0, 0)),
                   event(EventKind.COMPLETED, finish_reason=FinishReason.TOOL_CALLS)])
    output = await wire(events, request)
    assert frames(output)[-1]['type'] == 'response.completed'
    return request, output


def replay_items(output):
    calls = [item for item in output if item['type'] == 'function_call']
    return [{'role': 'user', 'content': 'Both cities?'}, *deepcopy(output),
        *[{'type': 'function_call_output', 'call_id': item['call_id'],
           'output': {'Berlin': 'ALPHA', 'Paris': 'BETA'}[json.loads(item['arguments'])['city']]}
          for item in reversed(calls)]]


class AssistantTurnReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_decoder_outputs_preserve_order_parts_and_id_correlations(self):
        for order in (['text', '0'], ['0', 'text'], ['0', 'text', '1']):
            for strict in (False, True):
                with self.subTest(order=order, strict=strict):
                    request, encoded = await mixed_output(order, strict=strict)
                    output = frames(encoded)[-1]['response']['output']
                    replay = replay_items(output)
                    original = deepcopy(replay)
                    parsed = parse_response(prepare_response(body(tools=[tool_definition(strict)], input=replay)))
                    self.assertEqual(replay, original)
                    turn = parsed.chat.messages[1]
                    self.assertEqual([m.role.value for m in parsed.chat.messages],
                                     ['user', 'assistant'] + ['tool'] * len(turn.tool_calls))
                    self.assertEqual(['text' if isinstance(item, AssistantTextItem) else str(item.index)
                                      for item in turn.assistant_items], order)
                    self.assertEqual(turn.content, (TextPart('Checking both cities.'),))
                    self.assertEqual([c.id for c in turn.tool_calls], [i['call_id'] for i in output if i['type'] == 'function_call'])
                    native_request = parsed.to_request(request.model, request.context, 128)
                    plan = prepare_tools(native_request)
                    flattened = []
                    for message in plan.messages:
                        if message['role'] == 'assistant':
                            if message['content']:
                                flattened.append('text')
                            flattened.extend(call['id'].split('_')[-1] for call in message.get('tool_calls', []))
                    self.assertEqual(flattened, order)
                    self.assertEqual([(m['tool_call_id'], m['content']) for m in plan.messages if m['role'] == 'tool'],
                                     [(c.id, ('ALPHA', 'BETA')[c.index]) for c in turn.tool_calls])
                    for m in plan.messages:
                        for call in m.get('tool_calls', []):
                            args = json.loads(call['function']['arguments'])
                            self.assertEqual(args.get('payload', args)['city'], ('Berlin', 'Paris')[int(call['id'][-1])])
                    if not strict:
                        native = ollama_plan(native_request).messages
                        self.assertEqual([(m['tool_name'], m['content']) for m in native if m['role'] == 'tool'],
                                         [('weather', ('ALPHA', 'BETA')[c.index]) for c in turn.tool_calls])
                        self.assertEqual([m['content'] for m in native if m['role'] == 'assistant'],
                                         [m['content'] for m in plan.messages if m['role'] == 'assistant'])

    async def test_missing_duplicate_foreign_or_cross_turn_results_stay_invalid(self):
        _, encoded = await mixed_output(['0', 'text', '1'])
        source = replay_items(frames(encoded)[-1]['response']['output'])
        variants = [source[:-1], source + [source[-1]],
                    source[:-1] + [{**source[-1], 'call_id': 'foreign'}],
                    source[:2] + [{'role': 'user', 'content': 'interrupt'}] + source[2:],
                    source[:-1] + [{'role': 'assistant', 'content': 'next turn'}] + source[-1:]]
        duplicate_call = deepcopy(source)
        duplicate_call[3]['call_id'] = duplicate_call[1]['call_id']
        variants.append(duplicate_call)
        for replay in variants:
            with self.subTest(replay=replay), self.assertRaises(ApiError) as caught:
                parse_response(prepare_response(body(input=replay)))
            self.assertEqual((caught.exception.status, caught.exception.code), (400, 'invalid_request'))

    async def test_completed_turns_remain_separate_and_call_ids_cannot_be_reused(self):
        _, encoded = await mixed_output(['0', 'text'])
        source = replay_items(frames(encoded)[-1]['response']['output'])
        next_turn = deepcopy(source[1:])
        next_turn[0]['call_id'] = 'next'
        next_turn[-1]['call_id'] = 'next'
        parsed = parse_response(prepare_response(body(input=source + next_turn)))
        turns = [m for m in parsed.chat.messages if m.role.value == 'assistant']
        self.assertEqual([[c.id for c in turn.tool_calls] for turn in turns], [['call_0'], ['next']])
        with self.assertRaises(ApiError):
            parse_response(prepare_response(body(input=source + source[1:])))

    async def test_multiple_message_items_keep_part_boundaries_and_error_paths(self):
        _, encoded = await mixed_output(['0', 'text', '1'])
        output = frames(encoded)[-1]['response']['output']
        output[1]['content'] = [{'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}
                                for text in ('', 'A', 'B')]
        output.insert(2, {**deepcopy(output[1]), 'id': 'second-message', 'content': [
            {'type': 'output_text', 'text': 'C', 'annotations': []}]})
        source = replay_items(output)
        parsed = parse_response(prepare_response(body(input=source)))
        self.assertEqual([item.content for item in parsed.chat.messages[1].assistant_items if isinstance(item, AssistantTextItem)],
                         [(TextPart(''), TextPart('A'), TextPart('B')), (TextPart('C'),)])
        source[-1]['call_id'] = 'foreign'
        with self.assertRaises(ApiError) as caught:
            parse_response(prepare_response(body(input=source)))
        self.assertEqual(caught.exception.param, f'input[{len(source)-1}].tool_call_id')

    def test_chat_does_not_infer_assistant_continuation_across_pending_calls(self):
        with self.assertRaises(ApiError):
            parse_chat(raw_request(messages=[{'role': 'assistant', 'tool_calls': [raw_call()]},
                {'role': 'assistant', 'content': 'not a declared output item'},
                {'role': 'tool', 'tool_call_id': 'a', 'content': 'answer'}]))
