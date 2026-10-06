"""Pure Responses normalization and complete-event aggregation; no service I/O."""
import asyncio
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

from kiron_common.local_inference import (
    EventKind, FinishReason, ImagePart, InferenceEvent, InferenceResult,
    ReasoningKind, ReasoningPart, TextPart, TokenUsage, ToolCall,
)
from openai_responses import (parse_response, prepare_response, serialize_response,
                              response_events, response_usage)
from openai_responses_stream import ResponseStreamState
from openai_wire import ApiError
from test_openai_tools import definition, request_for


def body(**fields):
    return {'model': 'public', 'input': 'hello', **fields}


def request_for_response(**fields):
    base = request_for()
    return parse_response(prepare_response(body(**fields))).to_request(base.model, base.context, 128)


def event(kind, **fields):
    return InferenceEvent(kind, 'tool-request', **fields)


def text_events(text='Hallo 世界', finish=FinishReason.STOP, usage=None, *, output_item_index=0):
    return [event(EventKind.STARTED), event(EventKind.TEXT_DELTA, text=text, output_item_index=output_item_index, part_index=0),
        event(EventKind.USAGE, usage=usage or TokenUsage(17, 4, 2, 1)),
        event(EventKind.COMPLETED, finish_reason=finish)]


async def source(values, error=None):
    for value in values:
        yield value
    if error:
        raise error


async def wire(values, request=None, error=None):
    return b''.join([frame async for frame in response_events(source(values, error),
        request=request or request_for_response(), response_id='resp_test', created=123)])


def frames(data):
    result = []
    for block in data.split(b'\n\n'):
        if not block:
            continue
        event_line, data_line = block.split(b'\n')
        value = json.loads(data_line.removeprefix(b'data: '))
        assert event_line == b'event: ' + value['type'].encode()
        result.append(value)
    assert [v['sequence_number'] for v in result] == list(range(len(result)))
    assert b'[DONE]' not in data
    return result


class ParseTests(unittest.TestCase):
    def test_shared_parameters_instructions_and_noop_fields(self):
        parsed = parse_response(prepare_response(body(instructions='be precise', stream=True,
            temperature=0, top_p=1, max_output_tokens=19, store=False, background=False,
            truncation='disabled', include=[], reasoning={'effort': 'none'}, text={'format': {'type': 'text'}})))
        self.assertTrue(parsed.stream)
        self.assertEqual(parsed.model_id, 'public')
        self.assertEqual([m.role.value for m in parsed.chat.messages], ['developer', 'user'])
        self.assertEqual(parsed.chat.max_completion_tokens, 19)
        self.assertIn('max_completion_tokens', parsed.chat.explicit_parameters)
        self.assertEqual(parsed.chat.sampling.temperature, 0)
        with self.assertRaises(TypeError):
            parsed.chat.explicit_parameters['temperature'] = 1

    def test_closed_fields_and_unsupported_cloud_state(self):
        cases = [body(**{field: value}) for field, value in (
            ('previous_response_id', 'resp_x'), ('conversation', 'c'), ('metadata', {}),
            ('stream_options', {}), ('user', 'a'), ('service_tier', 'default'),
            ('max_tool_calls', 1), ('store', True), ('background', True), ('include', ['reasoning.encrypted_content']),
            ('truncation', 'auto'), ('functions', []), ('max_output_tokens', True),
            ('store', 0), ('background', 'false'), ('instructions', []))]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ApiError):
                parse_response(prepare_response(data))

    def test_shared_schema_and_flat_function_normalization(self):
        function = {'type': 'function', **definition(strict=True)['function']}
        parsed = parse_response(prepare_response(body(tools=[function],
            tool_choice={'type': 'function', 'name': 'weather'}, parallel_tool_calls=True,
            text={'format': {'type': 'json_schema', 'name': 'out', 'strict': True,
                'schema': function['parameters']}})))
        self.assertEqual(parsed.chat.tools[0].name, 'weather')
        self.assertTrue(parsed.chat.tools[0].strict)
        self.assertEqual(parsed.chat.tool_choice.name, 'weather')
        self.assertEqual(parsed.chat.output_format.name, 'out')
        for data in (body(tools=[{'type': 'web_search'}]), body(text={'verbosity': 'low'}),
                     body(text={'format': {'type': 'json_schema', 'name': 's', 'schema': {'$ref': 'https://bad'}}})):
            with self.subTest(data=data), self.assertRaises(ApiError):
                parse_response(prepare_response(data))

    def test_images_keep_exact_positions_after_instructions(self):
        image = ImagePart('image/png', b'validated', hashlib.sha256(b'validated').hexdigest(), 1, 1)
        prepared = prepare_response(body(instructions='rules', input=[{'role': 'user', 'content': [
            {'type': 'input_text', 'text': 'before'}, {'type': 'input_image', 'image_url': 'data:image/png;base64,AA=='},
            {'type': 'input_text', 'text': 'after'}]}]))
        parsed = parse_response(prepared, image_parts={(1, 1): image})
        self.assertEqual(parsed.chat.messages[1].content, (TextPart('before'), image, TextPart('after')))
        self.assertEqual(prepared.parameter('messages[1].content[1].image_url.url'), 'input[0].content[1].image_url')
        with self.assertRaises(ApiError) as caught:
            parse_response(prepared)
        self.assertTrue(caught.exception.param.startswith('input[0]'))
        for role in ('assistant', 'developer'):
            with self.assertRaises(ApiError):
                prepare_response(body(input=[{'role': role, 'content': [{'type': 'input_image', 'image_url': 'data:bad'}]}]))

    def test_full_function_output_replay_matches_by_id_without_loss(self):
        function = {'type': 'function', **definition()['function']}
        request = request_for_response(tools=[function], parallel_tool_calls=True)
        calls = (ToolCall(0, 'a', 'weather', '{"city":"Berlin"}'), ToolCall(1, 'b', 'weather', '{"city":"Berlin"}'))
        result = InferenceResult(request.context.request_id, (TextPart('checking'),), (), calls,
                                 TokenUsage(7, 3, 0, 0), FinishReason.TOOL_CALLS)
        output = serialize_response(result, request=request, response_id='resp_test', created=123)['output']
        parsed = parse_response(prepare_response(body(tools=[function], input=[{'role':'user','content':'both'}, *output,
            {'type': 'function_call_output', 'call_id': 'b', 'output': 'second'},
            {'type': 'function_call_output', 'call_id': 'a', 'output': 'first'}])))
        self.assertEqual(parsed.chat.messages[1].content, (TextPart('checking'),))
        self.assertEqual(parsed.chat.messages[1].tool_calls, calls)
        self.assertEqual([m.tool_call_id for m in parsed.chat.messages[-2:]], ['b', 'a'])
        for outputs in ([{'type':'function_call_output','call_id':'unknown','output':'x'}],
                        [{'type':'function_call_output','call_id':'a','output':'x'}] * 2):
            with self.assertRaises(ApiError):
                parse_response(prepare_response(body(tools=[function], input=[*output, *outputs])))

    def test_sdk_parsed_arguments_annotation_is_typed_equal_and_not_emitted(self):
        for arguments, parsed in (('{"v":1}', {'v':True}),
                ('{"v":1}', {'v':2}), ('{}', None), ('{}', [])):
            with self.subTest(parsed=parsed), self.assertRaises(ApiError):
                parse_response(prepare_response(body(input=[
                    {'type':'function_call','call_id':'a','name':'weather','arguments':arguments,'parsed_arguments':parsed},
                    {'type':'function_call_output','call_id':'a','output':'x'}])))

    def test_replay_status_annotations_reasoning_and_partial_calls_are_not_discarded(self):
        bad = [
            {'type': 'reasoning', 'id': 'r', 'summary': [], 'content': [{'type':'reasoning_text','text':'thought'}]},
            {'type': 'reasoning', 'id': 'r', 'summary': [], 'encrypted_content': 'secret'},
            {'type': 'function_call', 'call_id': 'a', 'name': 'weather', 'arguments': '{', 'status': 'incomplete'},
            {'type': 'message', 'id':'m','role':'assistant','content':[{'type':'output_text','text':'x','annotations':[{}]}]},
        ]
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ApiError):
                parse_response(prepare_response(body(input=[item])))
        with self.assertRaises(ApiError) as caught:
            prepare_response(body(reasoning={'effort':'low','summary':'auto'}))
        self.assertEqual(caught.exception.code, 'unsupported_capability')


class OutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_items_parts_summaries_and_independent_call_indices(self):
        request = request_for_response(reasoning={'effort': 'low'},
            tools=[{'type': 'function', **definition()['function']}], parallel_tool_calls=True)
        # Summary support remains capability-gated; this pure serializer fixture
        # deliberately supplies canonical verified-summary options directly.
        request = replace(request, options=replace(request.options,
            reasoning=replace(request.options.reasoning, summary='concise')))
        sequence = [event(EventKind.STARTED)]
        for index, text in enumerate(('think A', 'think B')):
            sequence.append(event(EventKind.REASONING_DELTA, text=text, reasoning_kind=ReasoningKind.TEXT,
                                  output_item_index=0, part_index=index))
        for index, text in enumerate(('summary A', 'summary B')):
            sequence.append(event(EventKind.REASONING_DELTA, text=text, reasoning_kind=ReasoningKind.SUMMARY,
                                  output_item_index=0, summary_index=index))
        for index, text in enumerate(('before ', 'tools')):
            sequence.append(event(EventKind.TEXT_DELTA, text=text, output_item_index=1, part_index=index))
        calls = [ToolCall(i, chr(97+i), 'weather', '{"city":"Berlin"}') for i in range(2)]
        for output_index, call in zip((2, 3), reversed(calls)):
            sequence.extend([event(EventKind.TOOL_CALL_STARTED, call_index=call.index, call_id=call.id,
                                   name=call.name, output_item_index=output_index),
                event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=call.index, text=call.arguments, output_item_index=output_index),
                event(EventKind.TOOL_CALL_COMPLETED, tool_call=call, output_item_index=output_index)])
        sequence.extend([event(EventKind.TEXT_DELTA, text='after', output_item_index=4, part_index=0),
            event(EventKind.USAGE, usage=TokenUsage(10, 12, 0, 4)),
            event(EventKind.COMPLETED, finish_reason=FinishReason.TOOL_CALLS)])
        values = frames(await wire(sequence, request))
        final = values[-1]['response']
        self.assertEqual(values[-1]['type'], 'response.completed')
        self.assertEqual([i['type'] for i in final['output']],
                         ['reasoning', 'message', 'function_call', 'function_call', 'message'])
        self.assertEqual([p['text'] for p in final['output'][0]['content']], ['think A', 'think B'])
        self.assertEqual([p['text'] for p in final['output'][0]['summary']], ['summary A', 'summary B'])
        self.assertEqual([p['text'] for p in final['output'][1]['content']], ['before ', 'tools'])
        self.assertEqual([v['output_index'] for v in values if v['type'] == 'response.output_item.added'], list(range(5)))
        self.assertEqual([(v['output_index'], v['content_index']) for v in values if v['type'] == 'response.output_text.delta'],
                         [(1, 0), (1, 1), (4, 0)])
        self.assertEqual([v['summary_index'] for v in values if v['type'] == 'response.reasoning_summary_text.delta'], [0, 1])
        self.assertEqual([final['output'][i]['call_id'] for i in (2, 3)], ['b', 'a'])

    async def test_invalid_index_change_fails_without_mutating_valid_output(self):
        good = text_events()[:2]
        for invalid in (replace(good[1], output_item_index=2), replace(good[1], part_index=2),
                        event(EventKind.REASONING_DELTA, text='secret', reasoning_kind=ReasoningKind.TEXT,
                              output_item_index=0, part_index=0)):
            with self.subTest(invalid=invalid):
                result = frames(await wire([*good, invalid]))
                self.assertEqual(result[-1]['type'], 'response.failed')
                self.assertEqual(result[-1]['response']['output'][0]['content'][0]['text'], 'Hallo 世界')
                self.assertNotIn('secret', json.dumps(result))

    async def test_full_text_sse_terminal_and_done_match_nonstream(self):
        request = request_for_response()
        values = frames(await wire(text_events(), request))
        self.assertEqual([v['type'] for v in values[:4]], ['response.created', 'response.in_progress',
            'response.output_item.added', 'response.content_part.added'])
        self.assertEqual(values[-1]['type'], 'response.completed')
        result = InferenceResult(request.context.request_id, (TextPart('Hallo 世界'),), (), (),
                                 TokenUsage(17, 4, 2, 1), FinishReason.STOP)
        self.assertEqual(values[-1]['response'], serialize_response(result, request=request, response_id='resp_test', created=123))
        self.assertEqual(next(v['text'] for v in values if v['type']=='response.output_text.done'), 'Hallo 世界')
        self.assertEqual(values[-1]['response']['usage']['input_tokens_details'], {'cached_tokens':2})

    async def test_active_reasoning_is_explicit_and_fully_aggregated(self):
        request = request_for_response(reasoning={'effort':'low'})
        seq = [event(EventKind.STARTED), event(EventKind.REASONING_DELTA, text='think ', reasoning_kind=ReasoningKind.TEXT, output_item_index=0, part_index=0),
            event(EventKind.REASONING_DELTA,text='carefully',reasoning_kind=ReasoningKind.TEXT, output_item_index=0, part_index=0), *text_events(output_item_index=1)[1:]]
        values = frames(await wire(seq, request))
        thought = values[-1]['response']['output'][0]
        self.assertEqual(thought['content'], [{'type':'reasoning_text','text':'think carefully'}])
        self.assertEqual(thought['summary'], [])
        self.assertEqual(next(v['text'] for v in values if v['type']=='response.reasoning_text.done'), 'think carefully')
        for invalid_request, kind in ((request_for_response(), ReasoningKind.TEXT), (request, ReasoningKind.SUMMARY)):
            bad = [event(EventKind.STARTED), event(EventKind.REASONING_DELTA,text='hidden',reasoning_kind=kind,
                output_item_index=0, **({'summary_index': 0} if kind is ReasoningKind.SUMMARY else {'part_index': 0}))]
            data = await wire(bad, invalid_request)
            self.assertEqual(frames(data)[-1]['type'], 'response.failed')
            self.assertNotIn(b'hidden', data)
        result = InferenceResult(request.context.request_id, (), (ReasoningPart(ReasoningKind.TEXT,'thought'),), (),
                                 TokenUsage(7,4,0,4), FinishReason.LENGTH)
        value = serialize_response(result,request=request,response_id='resp_test',created=123)
        self.assertEqual(value['status'], 'incomplete')
        with self.assertRaises(ApiError):
            serialize_response(result,request=request_for_response(),response_id='resp_test',created=123)

    async def test_interleaved_function_stream_preserves_all_call_identity_and_exact_arguments(self):
        request = request_for_response(tools=[{'type':'function', **definition()['function']}], parallel_tool_calls=True)
        calls = [ToolCall(i, chr(97+i), 'weather', '{"city":"Berlin"}') for i in range(2)]
        seq = [event(EventKind.STARTED)]
        for index in (1,0):
            seq += [event(EventKind.TOOL_CALL_STARTED,call_index=index,call_id=calls[index].id,name='weather', output_item_index=1-index),
                    event(EventKind.TOOL_ARGUMENTS_DELTA,call_index=index,text='{"city":', output_item_index=1-index)]
        for index in (0,1):
            seq += [event(EventKind.TOOL_ARGUMENTS_DELTA,call_index=index,text='"Berlin"}', output_item_index=1-index),
                    event(EventKind.TOOL_CALL_COMPLETED,tool_call=calls[index], output_item_index=1-index)]
        seq += [event(EventKind.USAGE,usage=TokenUsage(9,8,0,0)),event(EventKind.COMPLETED,finish_reason=FinishReason.TOOL_CALLS)]
        values = frames(await wire(seq, request))
        output = values[-1]['response']['output']
        self.assertEqual([c['call_id'] for c in output], ['b','a'])
        self.assertEqual([c['arguments'] for c in output], [c.arguments for c in calls])
        self.assertEqual([v['name'] for v in values if v['type']=='response.function_call_arguments.done'], ['weather','weather'])
        self.assertTrue(all(item['status']=='completed' for item in output))

    async def test_length_keeps_partial_call_incomplete(self):
        request = request_for_response(tools=[{'type':'function', **definition()['function']}])
        seq = [event(EventKind.STARTED), event(EventKind.TOOL_CALL_STARTED,call_index=0,call_id='a',name='weather', output_item_index=0),
            event(EventKind.TOOL_ARGUMENTS_DELTA,call_index=0,text='{"city":', output_item_index=0),
            event(EventKind.USAGE,usage=TokenUsage(3,2,0,0)),event(EventKind.COMPLETED,finish_reason=FinishReason.LENGTH)]
        value = frames(await wire(seq, request))[-1]
        self.assertEqual(value['type'], 'response.incomplete')
        self.assertEqual(value['response']['output'][0]['status'], 'incomplete')
        self.assertEqual(value['response']['output'][0]['arguments'], '{"city":')

    async def test_usage_is_never_fabricated_and_success_waits_for_clean_eof(self):
        valid = text_events()
        cases = [valid[:-1], valid + [valid[-1]], [replace(valid[0],request_id='foreign'), *valid[1:]],
            [*valid[:2], event(EventKind.USAGE,usage=TokenUsage(17,4)), valid[-1]],
            [valid[0], valid[2], valid[1], valid[3]]]
        for seq in cases:
            with self.subTest(seq=seq):
                values = frames(await wire(seq))
                self.assertEqual(values[-1]['type'], 'response.failed')
                self.assertNotIn('response.completed', [v['type'] for v in values])
        values = frames(await wire(valid,error=RuntimeError('/private/token')))
        self.assertEqual(values[-1]['type'], 'response.failed')
        self.assertNotIn('/private/token',json.dumps(values))
        with self.assertRaises(ApiError):
            response_usage(TokenUsage(4,2,0))

    async def test_structured_output_validated_before_done_or_success(self):
        request = request_for_response(text={'format':{'type':'json_object'}})
        values = frames(await wire(text_events('not json'),request))
        self.assertEqual(values[-1]['type'], 'response.failed')
        self.assertFalse(any(v['type'].endswith('.done') for v in values))
        values = frames(await wire(text_events('{',FinishReason.LENGTH),request))
        self.assertEqual(values[-1]['type'], 'response.incomplete')

    async def test_external_timeout_initializes_and_fences_all_later_events(self):
        request = request_for_response()
        state = ResponseStreamState(request,'resp_test',123)
        failed = state.fail(ApiError('deadline','timeout',status=504))
        self.assertEqual([v['type'] for v in frames(failed)], ['response.created','response.in_progress','response.failed'])
        self.assertEqual(state.fail(ApiError('again','timeout',status=504)),b'')
        before = state.sequence
        self.assertEqual(state.accept(text_events()[0]),b'')
        self.assertEqual(state.sequence,before)
        state = ResponseStreamState(request,'resp_test',123)
        data = state.start()
        for value in text_events()[:2]:
            data += state.accept(value)
        data += state.fail(ApiError('secret','timeout',status=504))
        values = frames(data)
        self.assertEqual(values[-1]['response']['output'][0]['content'][0]['text'],'Hallo 世界')
        self.assertEqual(values[-1]['response']['output'][0]['status'],'incomplete')
        self.assertIsNone(values[-1]['response']['usage'])

    async def test_disconnect_does_not_invent_a_terminal_and_completed_state_stays_terminal(self):
        request = request_for_response()
        state = ResponseStreamState(request,'resp_test',123)
        async def disconnected():
            yield text_events()[0]
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            async for _ in response_events(disconnected(),request=request,response_id='resp_test',created=123,state=state):
                pass
        self.assertFalse(state.terminal)
        state = ResponseStreamState(request,'resp_test',123)
        async for _ in response_events(source(text_events()),request=request,response_id='resp_test',created=123,state=state):
            pass
        self.assertEqual(state.fail(ApiError('cleanup','timeout',status=504)),b'')

    async def test_large_exact_usage_cannot_exhaust_reserved_failure_terminal(self):
        sequence = text_events('x'*12000,usage=TokenUsage(10**4000,3,0,0))
        with patch('openai_responses_stream.MAX_OUTPUT_BYTES',32768):
            data = await wire(sequence)
        values = frames(data)
        self.assertEqual(values[-1]['type'],'response.failed')
        self.assertIsNone(values[-1]['response']['usage'])
        self.assertEqual(len(values[-1]['response']['output'][0]['content'][0]['text']),12000)
        self.assertLessEqual(len(data),32768)

    async def test_output_cap_reserves_full_partial_state_after_multiple_accepted_deltas(self):
        sequence = [event(EventKind.STARTED), *[event(EventKind.TEXT_DELTA,text='x'*5000, output_item_index=0, part_index=0) for _ in range(2)],
            event(EventKind.USAGE,usage=TokenUsage(7,3,0,0)),event(EventKind.COMPLETED,finish_reason=FinishReason.STOP)]
        with patch('openai_responses_stream.MAX_OUTPUT_BYTES',32768):
            data = await wire(sequence)
        values = frames(data)
        self.assertEqual(values[-1]['type'],'response.failed')
        text = values[-1]['response']['output'][0]['content'][0]['text']
        deltas = ''.join(v['delta'] for v in values if v['type']=='response.output_text.delta')
        self.assertEqual(text,deltas)
        self.assertEqual(len(text),10000)
        self.assertLessEqual(len(data),32768)

    async def test_bounded_output_retains_valid_failed_state_even_on_invalid_unicode(self):
        with patch('openai_responses_stream.MAX_OUTPUT_BYTES',32768):
            data = await wire(text_events('x'*20000))
        values = frames(data)
        self.assertLessEqual(len(data),32768)
        self.assertEqual(values[-1]['type'],'response.failed')
        self.assertNotIn('response.completed',[v['type'] for v in values])
        data = await wire(text_events('\ud800'))
        values = frames(data)
        self.assertEqual(values[-1]['type'],'response.failed')
        # Newly opened empty items are emitted before their failure, never leave
        # sequence gaps when a subsequent fragment fails validation.
        self.assertIn('response.output_item.added',[v['type'] for v in values])


if __name__ == '__main__':
    unittest.main()
