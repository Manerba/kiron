"""Actual KIron tool serialization parsed by the pinned official SDK helper."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import openai

import httpx
from openai import AsyncOpenAI, LengthFinishReasonError

PROXY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROXY))
sys.path.insert(0, str(PROXY.parent / "kiron-common"))

from kiron_common.local_inference import (CapabilityName, CapabilitySet, EventKind, FinishReason,
    InferenceEvent, InferenceResult, MessageRole, OutputEventLayout, ParameterConstraint, TextPart, TokenUsage, ToolCall)
from kiron_common.model_catalog import BackendType
from openai_wire import chat_events, serialize_completion, parse_chat
from test_openai_tools import definition, raw_request, request_for
import test_openai_api_sdk as api_sdk


def event(kind, **fields):
    return InferenceEvent(kind, "sdk-tools", **fields)


class ToolSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_parallel_helper_assembles_interleaved_canonical_events_and_usage(self):
        request = request_for(tools=[definition(strict=True)], tool_choice="required", parallel_tool_calls=True)
        calls = (ToolCall(0,"a","weather",'{"city":"Berlin"}'), ToolCall(1,"b","weather",'{"city":"Berlin"}'))
        async def canonical():
            yield event(EventKind.STARTED)
            for index in (1, 0):
                yield event(EventKind.TOOL_CALL_STARTED, call_index=index, call_id=calls[index].id, name="weather", output_item_index=1-index)
                yield event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=index, text='{"city":', output_item_index=1-index)
            for index in (0, 1):
                yield event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=index, text='"Berlin"}', output_item_index=1-index)
                yield event(EventKind.TOOL_CALL_COMPLETED, tool_call=calls[index], output_item_index=1-index)
            yield event(EventKind.USAGE, usage=TokenUsage(31, 18, 0))
            yield event(EventKind.COMPLETED, finish_reason=FinishReason.TOOL_CALLS)
        output = b"".join([frame async for frame in chat_events(canonical(), "public", "chatcmpl-tools", 123, True, request=request)])
        async def handle(req):
            return httpx.Response(200, content=output, headers={"content-type":"text/event-stream"})
        async with AsyncOpenAI(api_key="fixture", base_url="http://fixture/v1", max_retries=0,
                _strict_response_validation=True, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            async with client.chat.completions.stream(model="public", messages=[{"role":"user","content":"both"}],
                    tools=[definition(strict=True)], parallel_tool_calls=True, stream_options={"include_usage":True}) as stream:
                events = [item async for item in stream]
                final = await stream.get_final_completion()
        self.assertEqual(final.choices[0].finish_reason, "tool_calls")
        self.assertEqual([c.id for c in final.choices[0].message.tool_calls], ["a","b"])
        self.assertEqual([c.function.parsed_arguments for c in final.choices[0].message.tool_calls], [{"city":"Berlin"}]*2)
        self.assertEqual(sum(e.type == "tool_calls.function.arguments.done" for e in events), 2)
        self.assertEqual(final.usage.total_tokens, 49)

    async def test_sdk_completion_dump_replays_full_roundtrip_with_ids_and_client_results(self):
        requests = []
        tools = [definition(strict=True)]
        first = InferenceResult("sdk-tools", (), (), (ToolCall(0,"call_1","weather",'{"city":"Berlin"}'),), TokenUsage(5,6), FinishReason.TOOL_CALLS)
        last = InferenceResult("sdk-tools", (TextPart("It is 20 degrees."),), (), (), TokenUsage(10,5), FinishReason.STOP)
        async def handle(req):
            body = json.loads(req.content)
            parsed = parse_chat(body)
            requests.append(parsed)
            result = first if len(requests) == 1 else last
            return httpx.Response(200, json=serialize_completion(result,"public","chatcmpl-roundtrip",123))
        async with AsyncOpenAI(api_key="fixture", base_url="http://fixture/v1", max_retries=0,
                _strict_response_validation=True, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            messages = [{"role":"user","content":"weather"}]
            result = await client.chat.completions.create(model="public", messages=messages, tools=tools)
            messages += [result.choices[0].message.model_dump(exclude_none=True),
                         {"role":"tool", "tool_call_id":"call_1", "content":"20 degrees"}]
            result = await client.chat.completions.create(model="public", messages=messages, tools=tools)
        self.assertEqual(result.choices[0].message.content,"It is 20 degrees.")
        self.assertEqual(requests[1].messages[1].tool_calls[0].id,"call_1")
        self.assertEqual(requests[1].messages[2].tool_call_id,"call_1")

    async def test_length_keeps_partial_arguments_and_strict_helper_does_not_execute(self):
        async def canonical():
            yield event(EventKind.STARTED)
            yield event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="weather", output_item_index=0)
            yield event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text='{"city":', output_item_index=0)
            yield event(EventKind.USAGE, usage=TokenUsage(5,2))
            yield event(EventKind.COMPLETED, finish_reason=FinishReason.LENGTH)
        output = b"".join([part async for part in chat_events(canonical(),"public","chatcmpl-length",123,True)])
        async def handle(req):
            return httpx.Response(200,content=output,headers={"content-type":"text/event-stream"})
        async with AsyncOpenAI(api_key="fixture",base_url="http://fixture/v1",max_retries=0,
                _strict_response_validation=True,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            raw = await client.chat.completions.create(model="public",messages=[{"role":"user","content":"weather"}],stream=True)
            chunks = [chunk async for chunk in raw]
            self.assertEqual(chunks[-2].choices[0].finish_reason,"length")
            self.assertEqual(chunks[2].choices[0].delta.tool_calls[0].function.arguments,'{"city":')
            with self.assertRaises(LengthFinishReasonError):
                async with client.chat.completions.stream(model="public",messages=[{"role":"user","content":"weather"}],tools=[definition(strict=True)]) as stream:
                    [item async for item in stream]


class ToolASGISdkTests(unittest.IsolatedAsyncioTestCase):
    """Real HTTP boundary/RuntimeService/admission with fixture-only execution."""
    request_tickets = api_sdk.OpenAISDKContractTests.request_tickets
    assert_no_provider_io = api_sdk.OpenAISDKContractTests.assert_no_provider_io

    async def asyncSetUp(self):
        await api_sdk.OpenAISDKContractTests.asyncSetUp(self)
        for provider in self.providers.values():
            chat = provider.chat_capability
            provider.chat_capability = replace(chat, constraints={**chat.constraints,
                "roles": ParameterConstraint(allowed_values=("system","user","assistant","tool"))})
            tools = replace(chat, constraints={"tool_choice":ParameterConstraint(allowed_values=("none","auto","required","named")),
                "strict":ParameterConstraint(allowed_values=(False,True)), "max_tools":ParameterConstraint(minimum=0,maximum=64)})
            async def capabilities(deployment, p=provider, tool_cap=tools):
                return CapabilitySet({CapabilityName.CHAT:p.chat_capability, CapabilityName.STREAMING:p.chat_capability,
                                      CapabilityName.FUNCTION_TOOLS:tool_cap, CapabilityName.PARALLEL_TOOLS:tool_cap})
            provider.capabilities = capabilities
            def result(request):
                if request.messages[-1].role is MessageRole.TOOL:
                    return InferenceResult(request.context.request_id,(TextPart("20 degrees"),),(),(),TokenUsage(10,3),FinishReason.STOP)
                count = 2 if request.options.parallel_tool_calls else 1
                calls = tuple(ToolCall(i,"call_"+str(i),"weather",'{"city":"Berlin"}') for i in range(count))
                return InferenceResult(request.context.request_id,(),(),calls,TokenUsage(5,6),FinishReason.TOOL_CALLS)
            async def chat(request, p=provider):
                p._execution("chat",request)
                return result(request)
            async def stream(request, p=provider):
                p._execution("stream",request)
                value = result(request)
                rid = request.context.request_id
                layout = OutputEventLayout()
                yield InferenceEvent(EventKind.STARTED,rid)
                for call in reversed(value.tool_calls):
                    yield InferenceEvent(EventKind.TOOL_CALL_STARTED,rid,call_index=call.index,call_id=call.id,name=call.name, **layout.tool(call.index))
                    yield InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA,rid,call_index=call.index,text='{"city":', **layout.tool(call.index))
                for call in value.tool_calls:
                    yield InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA,rid,call_index=call.index,text='"Berlin"}', **layout.tool(call.index))
                    yield InferenceEvent(EventKind.TOOL_CALL_COMPLETED,rid,tool_call=call, **layout.tool(call.index))
                for part in value.content:
                    yield InferenceEvent(EventKind.TEXT_DELTA,rid,text=part.text, **layout.text())
                yield InferenceEvent(EventKind.USAGE,rid,usage=value.usage)
                yield InferenceEvent(EventKind.COMPLETED,rid,finish_reason=value.finish_reason)
            provider.chat, provider.stream = chat, stream

    async def test_parallel_strict_roundtrip_uses_same_public_contract_for_each_provider(self):
        for provider_id, provider in self.providers.items():
            with self.subTest(provider=provider_id.value):
                messages = [{"role":"user","content":"weather twice"}]
                async with self.sdk.chat.completions.stream(model=provider_id.value+".chat",messages=messages,
                        tools=[definition(strict=True)],parallel_tool_calls=True,stream_options={"include_usage":True}) as stream:
                    [item async for item in stream]
                    completion = await stream.get_final_completion()
                self.assertEqual(len(completion.choices[0].message.tool_calls),2)
                messages.append(completion.choices[0].message.model_dump(exclude_none=True))
                messages += [{"role":"tool","tool_call_id":"call_1","content":"second"},
                             {"role":"tool","tool_call_id":"call_0","content":"first"}]
                final = await self.sdk.chat.completions.create(model=provider_id.value+".chat",messages=messages,tools=[definition(strict=True)])
                self.assertEqual(final.choices[0].message.content,"20 degrees")
                self.assertEqual([m.tool_call_id for m in provider.requests[-1].messages[-2:]], ["call_1","call_0"])
                self.assertEqual(self.request_tickets(),[])

    async def test_schema_history_and_unverified_tool_capability_fail_before_any_provider_io(self):
        provider = self.providers[BackendType.PRISM]
        async def no_tools(deployment):
            return CapabilitySet({CapabilityName.CHAT:provider.chat_capability})
        provider.capabilities = no_tools
        for fields, expected in (({"tools":[definition()]},"unsupported_capability"),
            ({"tools":[definition()],"functions":[]},"invalid_request"),
            ({"messages":[{"role":"tool","tool_call_id":"unknown","content":"x"}]},"invalid_request")):
            with self.subTest(fields=fields), self.assertRaises(openai.BadRequestError) as caught:
                await self.sdk.chat.completions.create(**{"model":"prism.chat","messages":[{"role":"user","content":"hi"}],**fields})
            self.assertEqual(caught.exception.code,expected)
            self.assert_no_provider_io()


if __name__ == "__main__":
    unittest.main()
