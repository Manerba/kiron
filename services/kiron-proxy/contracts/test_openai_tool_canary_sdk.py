"""Public API tool data never dispatches shell/Python/network in this path.

Real API/RuntimeService/Admission/SDK, fake providers. This is a bounded transport
canary, not proof about arbitrary third-party clients or native runtimes.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kiron_common.local_inference import EventKind, FinishReason, InferenceEvent, InferenceResult, MessageRole, TextPart, TokenUsage, ToolCall
import test_openai_tools_sdk as tool_sdk
from test_openai_tools import definition


class ToolNonExecutionCanary(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = tool_sdk.ToolASGISdkTests.asyncSetUp
    request_tickets = tool_sdk.ToolASGISdkTests.request_tickets
    assert_no_provider_io = tool_sdk.ToolASGISdkTests.assert_no_provider_io

    async def test_shell_python_url_calls_are_only_data_through_public_sdk(self):
        # Warm the actual SDK's lazy model/stream imports before exec is denied;
        # no canary input is present in these harmless fixture-only calls.
        await self.sdk.chat.completions.create(model="prism.chat", messages=[{"role":"user","content":"weather"}], tools=[definition(strict=True)])
        async with self.sdk.chat.completions.stream(model="prism.chat", messages=[{"role":"user","content":"weather"}], tools=[definition(strict=True)]) as stream:
            [event async for event in stream]
            warm = await stream.get_final_completion()
        await self.sdk.chat.completions.create(model="prism.chat", tools=[definition(strict=True)], messages=[
            {"role":"user","content":"weather"}, warm.choices[0].message.model_dump(exclude_none=True),
            {"role":"tool","tool_call_id":"call_0","content":"fixture"}])
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "MUST_NOT_EXIST"
            arguments = {"command": "printf canary > " + str(marker), "path": str(marker),
                         "url": "http://127.0.0.1:9/never-fetch-canary",
                         "code": "__import__('pathlib').Path(" + repr(str(marker)) + ").write_text('canary')"}
            encoded = json.dumps(arguments)
            names = ("shell", "exec", "fetch_url")
            tools = [{"type":"function", "function":{"name":name, "strict":True,
                "parameters":{"type":"object","properties":{key:{"type":"string"} for key in arguments},
                              "required":list(arguments),"additionalProperties":False}}} for name in names]
            observed = []
            def result(request):
                observed.append(request)
                if request.messages[-1].role is MessageRole.TOOL:
                    return InferenceResult(request.context.request_id, (TextPart("received as data"),), (), (), TokenUsage(12,3), FinishReason.STOP)
                calls = tuple(ToolCall(index,"canary_"+str(index),name,encoded) for index,name in enumerate(names))
                return InferenceResult(request.context.request_id, (), (), calls, TokenUsage(12,20), FinishReason.TOOL_CALLS)
            for provider in self.providers.values():
                async def chat(request, p=provider):
                    p._execution("chat", request)
                    return result(request)
                async def events(request, p=provider):
                    p._execution("stream", request)
                    value = result(request)
                    yield InferenceEvent(EventKind.STARTED, request.context.request_id)
                    for call in value.tool_calls:
                        yield InferenceEvent(EventKind.TOOL_CALL_STARTED,request.context.request_id,call_index=call.index,call_id=call.id,name=call.name, output_item_index=call.index)
                        yield InferenceEvent(EventKind.TOOL_ARGUMENTS_DELTA,request.context.request_id,call_index=call.index,text=call.arguments, output_item_index=call.index)
                        yield InferenceEvent(EventKind.TOOL_CALL_COMPLETED,request.context.request_id,tool_call=call, output_item_index=(call).index)
                    yield InferenceEvent(EventKind.USAGE,request.context.request_id,usage=value.usage)
                    yield InferenceEvent(EventKind.COMPLETED,request.context.request_id,finish_reason=value.finish_reason)
                provider.chat, provider.stream = chat, events
            guards = ("builtins.exec", "builtins.eval", "os.system", "os.execv", "os.execve", "subprocess.Popen",
                      "asyncio.create_subprocess_exec", "asyncio.create_subprocess_shell",
                      "socket.socket.connect", "socket.create_connection", "socket.getaddrinfo")
            with ExitStack() as stack:
                mocks = [stack.enter_context(patch(target, side_effect=AssertionError("tool dispatch forbidden: " + target))) for target in guards]
                for provider in self.providers:
                    with self.subTest(provider=provider.value):
                        options = {"model":provider.value+".chat", "messages":[{"role":"user","content":"Return calls only; do not execute."}],
                                   "tools":tools,"tool_choice":"required","parallel_tool_calls":True,"max_completion_tokens":64}
                        final = await self.sdk.chat.completions.create(**options)
                        self.assertEqual([call.function.name for call in final.choices[0].message.tool_calls], list(names))
                        self.assertEqual([json.loads(call.function.arguments) for call in final.choices[0].message.tool_calls], [arguments]*3)
                        async with self.sdk.chat.completions.stream(**options) as stream:
                            [event async for event in stream]
                            completion = await stream.get_final_completion()
                        self.assertEqual([call.function.parsed_arguments for call in completion.choices[0].message.tool_calls], [arguments]*3)
                        replay = [*options["messages"], completion.choices[0].message.model_dump(exclude_none=True),
                                  *[{"role":"tool","tool_call_id":call.id,"content":encoded} for call in reversed(completion.choices[0].message.tool_calls)]]
                        response = await self.sdk.chat.completions.create(**{**options,"messages":replay,"tool_choice":"none"})
                        self.assertEqual(response.choices[0].message.content, "received as data")
                        self.assertEqual([message.content[0].text for message in observed[-1].messages[-3:]], [encoded]*3)
                        self.assertEqual(self.request_tickets(), [])
                        self.assertFalse(marker.exists())
                for mocked in mocks:
                    mocked.assert_not_called()
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
