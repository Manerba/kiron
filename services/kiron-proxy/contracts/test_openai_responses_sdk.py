"""Actual pinned OpenAI Responses SDK over MockTransport; no network/inference."""
import json
import unittest

import httpx
from openai import AsyncOpenAI
from kiron_common.local_inference import EventKind, FinishReason, InferenceResult, TextPart, TokenUsage, ToolCall
from openai_responses import parse_response, prepare_response, serialize_response
from test_openai_responses import body, event, frames, request_for_response, text_events, wire
from test_openai_tools import definition


class ResponsesSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_decoder_mixed_turn_helper_dump_replays_unchanged_with_reversed_results(self):
        from test_assistant_turn_replay import mixed_output, replay_items, tool_definition
        from prism_tools import prepare_tools
        from openai_wire import ApiError
        for order in (['text', '0'], ['0', 'text'], ['0', 'text', '1']):
            with self.subTest(order=order):
                original, output = await mixed_output(order, strict=True)
                replayed = []
                async def handle(http_request):
                    data = json.loads(http_request.content)
                    if data.get('stream'):
                        return httpx.Response(200, content=output, headers={'content-type': 'text/event-stream'})
                    try:
                        parsed = parse_response(prepare_response(data))
                    except ApiError as error:
                        return httpx.Response(error.status, json=error.envelope())
                    request = parsed.to_request(original.model, original.context, 128)
                    replayed.append((data['input'], request, prepare_tools(request)))
                    result = InferenceResult(request.context.request_id, (TextPart('ALPHA/BETA'),), (), (),
                                             TokenUsage(20, 4, 0, 0), FinishReason.STOP)
                    return httpx.Response(200, json=serialize_response(result, request=request, response_id='resp_next', created=124))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
                    async with AsyncOpenAI(api_key='fixture', base_url='http://fixture/v1', http_client=http,
                                           _strict_response_validation=True, max_retries=0) as client:
                        async with client.responses.stream(model='public', input='Both cities?',
                                tools=[tool_definition(True)], parallel_tool_calls=True) as stream:
                            async for _ in stream:
                                pass
                            first = await stream.get_final_response()
                        dumped = [item.model_dump(exclude_none=True) for item in first.output]
                        self.assertTrue(all('parsed_arguments' in item for item in dumped if item['type'] == 'function_call'))
                        replay = replay_items(dumped)
                        second = await client.responses.create(model='public', input=replay,
                            tools=[tool_definition(True)], tool_choice='none')
                        self.assertEqual(second.output_text, 'ALPHA/BETA')
                        self.assertEqual(replayed[0][0][1:1+len(dumped)], dumped)
                        turn = replayed[0][1].messages[1]
                        self.assertEqual([call.id for call in turn.tool_calls],
                                         [item['call_id'] for item in dumped if item['type'] == 'function_call'])
                        self.assertEqual([m['content'] for m in replayed[0][2].messages if m['role'] == 'tool'],
                                         ['ALPHA', 'BETA'][:len(turn.tool_calls)])

    async def test_multiple_explicit_items_and_parts_survive_raw_and_helper_parsing(self):
        sequence = [event(EventKind.STARTED),
            event(EventKind.TEXT_DELTA, text='first ', output_item_index=0, part_index=0),
            event(EventKind.TEXT_DELTA, text='part', output_item_index=0, part_index=1),
            event(EventKind.TEXT_DELTA, text='second item', output_item_index=1, part_index=0),
            event(EventKind.USAGE, usage=TokenUsage(5, 4, 0, 0)),
            event(EventKind.COMPLETED, finish_reason=FinishReason.STOP)]
        output = await wire(sequence)
        async def check(client):
            raw = await client.responses.create(model='public', input='hello', stream=True)
            events = [value async for value in raw]
            self.assertEqual([(v.output_index, v.content_index) for v in events if v.type == 'response.output_text.delta'],
                             [(0, 0), (0, 1), (1, 0)])
            async with client.responses.stream(model='public', input='hello') as stream:
                async for _ in stream:
                    pass
                result = await stream.get_final_response()
            self.assertEqual([[part.text for part in item.content] for item in result.output],
                             [['first ', 'part'], ['second item']])
        await self.client_case(output, check)

    async def client_case(self, output, callback):
        async def handle(request):
            if json.loads(request.content).get('stream'):
                return httpx.Response(200, content=output, headers={'content-type':'text/event-stream'})
            return httpx.Response(200,json=frames(output)[-1]['response'])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(api_key='fixture',base_url='http://fixture/v1',http_client=http,
                                   _strict_response_validation=True,max_retries=0) as client:
                await callback(client)

    async def test_strict_nonstream_raw_events_and_helper_complete_state(self):
        output = await wire(text_events())
        async def check(client):
            result = await client.responses.create(model='public',input='hello')
            self.assertEqual(result.output_text,'Hallo 世界')
            self.assertEqual(result.usage.input_tokens_details.cached_tokens,2)
            self.assertEqual(result.usage.output_tokens_details.reasoning_tokens,1)
            raw = await client.responses.create(model='public',input='hello',stream=True)
            events = [value async for value in raw]
            self.assertEqual(events[-1].type,'response.completed')
            async with client.responses.stream(model='public',input='hello') as stream:
                events = [value async for value in stream]
                result = await stream.get_final_response()
            self.assertEqual(result.output_text,'Hallo 世界')
            replay = [item.model_dump(exclude_none=True) for item in result.output]
            parsed = parse_response(prepare_response(body(input=[{'role':'user','content':'hello'},*replay])))
            self.assertEqual(parsed.chat.messages[-1].content,(TextPart('Hallo 世界'),))
            self.assertEqual([v.sequence_number for v in events],list(range(len(events))))
        await self.client_case(output,check)

    async def test_reasoning_events_and_terminal_preserve_native_text_without_summary(self):
        from kiron_common.local_inference import ReasoningKind
        request = request_for_response(reasoning={'effort':'low'})
        output = await wire([event(EventKind.STARTED),event(EventKind.REASONING_DELTA,text='native thought',
            reasoning_kind=ReasoningKind.TEXT, output_item_index=0, part_index=0),*text_events(output_item_index=1)[1:]],request)
        async def check(client):
            async with client.responses.stream(model='public',input='hello',reasoning={'effort':'low'}) as stream:
                events = [value async for value in stream]
                result = await stream.get_final_response()
            self.assertEqual(result.output[0].type,'reasoning')
            self.assertEqual(result.output[0].content[0].text,'native thought')
            self.assertEqual(result.output[0].summary,[])
            self.assertIn('response.reasoning_text.done',[v.type for v in events])
        await self.client_case(output,check)

    async def test_parallel_tools_have_strict_sdk_events_and_full_replay(self):
        tool = {'type':'function',**definition(strict=True)['function']}
        request = request_for_response(tools=[tool],parallel_tool_calls=True)
        calls = [ToolCall(index,chr(97+index),'weather','{"city":"Berlin"}') for index in range(2)]
        sequence = [event(EventKind.STARTED)]
        for call in calls:
            sequence += [event(EventKind.TOOL_CALL_STARTED,call_index=call.index,call_id=call.id,name=call.name, output_item_index=call.index)]
        for call in reversed(calls):
            sequence += [event(EventKind.TOOL_ARGUMENTS_DELTA,call_index=call.index,text=call.arguments, output_item_index=call.index),
                         event(EventKind.TOOL_CALL_COMPLETED,tool_call=call, output_item_index=(call).index)]
        sequence += [event(EventKind.USAGE,usage=TokenUsage(9,8,0,0)),event(EventKind.COMPLETED,finish_reason=FinishReason.TOOL_CALLS)]
        output = await wire(sequence,request)
        async def check(client):
            raw = await client.responses.create(model='public',input='both',tools=[tool],parallel_tool_calls=True,stream=True)
            events = [value async for value in raw]
            replay = [item.model_dump(exclude_none=True) for item in events[-1].response.output]
            self.assertEqual([v.call_id for v in events[-1].response.output],['a','b'])
            parsed = parse_response(prepare_response(body(tools=[tool],input=[{'role':'user','content':'both'},*replay,
                {'type':'function_call_output','call_id':'b','output':'second'},
                {'type':'function_call_output','call_id':'a','output':'first'}])))
            self.assertEqual(parsed.chat.messages[1].tool_calls,tuple(calls))
            async with client.responses.stream(model='public',input='both',tools=[tool],parallel_tool_calls=True) as stream:
                helper_events = [value async for value in stream]
                result = await stream.get_final_response()
            self.assertEqual([item.parsed_arguments for item in result.output],[{'city':'Berlin'}]*2)
            self.assertEqual([item.arguments for item in result.output],[call.arguments for call in calls])
            replay = [item.model_dump(exclude_none=True) for item in result.output]
            self.assertIn('parsed_arguments', replay[0])
            parsed = parse_response(prepare_response(body(tools=[tool], input=[{'role':'user','content':'both'}, *replay,
                {'type':'function_call_output','call_id':'b','output':'second'},
                {'type':'function_call_output','call_id':'a','output':'first'}])))
            self.assertEqual(parsed.chat.messages[1].tool_calls, tuple(calls))
        await self.client_case(output,check)

    async def test_sdk_vision_payload_uses_shared_bounded_decoder_and_exact_part_order(self):
        import base64
        import io
        from PIL import Image
        from kiron_common.local_inference import ImagePart
        from openai_vision import decode_chat_images
        target = io.BytesIO()
        with Image.new('RGB',(2,2),'red') as image:
            image.save(target,format='PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(target.getvalue()).decode()
        base = request_for_response()
        seen = []
        async def handle(http_request):
            prepared = prepare_response(json.loads(http_request.content))
            images = await decode_chat_images(prepared.chat_data,base.context)
            parsed = parse_response(prepared,image_parts=images)
            request = parsed.to_request(base.model,base.context,128)
            seen.extend(request.messages)
            result = InferenceResult(request.context.request_id,(TextPart('red'),),(),(),
                                     TokenUsage(11,2,0,0),FinishReason.STOP)
            return httpx.Response(200,json=serialize_response(result,request=request,response_id='resp_image',created=123))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(api_key='fixture',base_url='http://fixture/v1',http_client=http,
                                   _strict_response_validation=True,max_retries=0) as client:
                result = await client.responses.create(model='public',instructions='Inspect the image.',input=[
                    {'role':'user','content':[{'type':'input_text','text':'before'},
                        {'type':'input_image','image_url':uri,'detail':'auto'},
                        {'type':'input_text','text':'after'}]}])
        self.assertEqual(result.output_text,'red')
        self.assertEqual([message.role.value for message in seen],['developer','user'])
        self.assertEqual((seen[1].content[0],seen[1].content[2]),(TextPart('before'),TextPart('after')))
        self.assertIsInstance(seen[1].content[1],ImagePart)
        self.assertEqual(seen[1].content[1].data,target.getvalue())

    async def test_sdk_structured_parse_consumes_the_same_validated_schema_result(self):
        from pydantic import BaseModel, ConfigDict
        class Weather(BaseModel):
            model_config = ConfigDict(extra='forbid')
            city: str
        request = request_for_response(text={'format':{'type':'json_schema','name':'Weather',
            'strict':True,'schema':definition(strict=True)['function']['parameters']}})
        output = await wire(text_events('{"city":"Berlin"}'),request)
        async def check(client):
            result = await client.responses.parse(model='public',input='hello',text_format=Weather)
            self.assertEqual(result.output_parsed,Weather(city='Berlin'))
            self.assertEqual(result.usage.output_tokens_details.reasoning_tokens,1)
        await self.client_case(output,check)

    async def test_incomplete_and_failed_are_valid_sdk_terminal_types(self):
        for sequence, error, terminal in ((text_events('partial',FinishReason.LENGTH),None,'response.incomplete'),
                (text_events()[:2],RuntimeError('private'),'response.failed')):
            with self.subTest(terminal=terminal):
                output = await wire(sequence,error=error)
                async def check(client):
                    events = await client.responses.create(model='public',input='hello',stream=True)
                    values = [value async for value in events]
                    self.assertEqual(values[-1].type,terminal)
                    self.assertEqual(values[-1].response.output[0].content[0].text,
                                     'partial' if terminal.endswith('incomplete') else 'Hallo 世界')
                    self.assertEqual(values[-1].response.output[0].status,'incomplete')
                    if terminal.endswith('failed'):
                        self.assertEqual(values[-1].response.error.code,'server_error')
                        self.assertIsNone(values[-1].response.usage)
                await self.client_case(output,check)


if __name__ == '__main__':
    unittest.main()
