"""Pure wire contract checks without an OpenAI SDK dependency."""

import asyncio
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

from kiron_common.local_inference import (
    ArtifactIdentity, ErrorCode, EventKind, FinishReason, InferenceEvent, InferenceResult,
    LocalInferenceError, MessageRole, ReasoningKind, ReasoningPart, RequestContext,
    ResolvedDeployment, ResolvedModel, RuntimeFailure, TextPart, TokenUsage, ToolCall,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
import openai_wire as wire


def request(**fields):
    return {"model": "public/model:tag", "messages": [{"role": "user", "content": "hello"}], **fields}


def event(kind, **fields):
    return InferenceEvent(kind, "internal-request", **fields)


def sequence():
    return [event(EventKind.STARTED), event(EventKind.TEXT_DELTA, text="Hallo ", output_item_index=0, part_index=0),
            event(EventKind.TEXT_DELTA, text="世界", output_item_index=0, part_index=0), event(EventKind.USAGE, usage=TokenUsage(17, 2, 0, 0)),
            event(EventKind.COMPLETED, finish_reason=FinishReason.STOP)]


async def source(values, error=None):
    for value in values:
        yield value
    if error is not None:
        raise error


async def stream(values, *, error=None, include_usage=True):
    return b"".join([part async for part in wire.chat_events(
        source(values, error), "public/model:tag", "chatcmpl-fixed", 123, include_usage)])


def frames(data):
    assert data.endswith(wire.DONE)
    return [json.loads(line[6:]) for line in data.split(b"\n\n") if line and line != b"data: [DONE]"]


class JsonTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_and_invalid_utf8_rejected_at_any_level(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}', b'{"n":NaN}',
                    b'{"n":Infinity}', b'{"n":-Infinity}', b'{"n":1e999}',
                    b'{"x":"\xff"}', b'{"x":"\\ud800"}', b'[]', b'null', b'{'):
            with self.subTest(raw=raw), self.assertRaises(wire.ApiError) as caught:
                wire.decode_json(raw)
            self.assertEqual(caught.exception.code, "invalid_request")

    def test_depth_and_size_before_decoder_recursion(self):
        valid = b'{"a":' + b'[' * 31 + b'0' + b']' * 31 + b'}'
        wire.decode_json(valid)
        for raw in (b'{"a":' + b'[' * 32 + b'0' + b']' * 32 + b'}',
                    b'{"a":' + b'[' * 10000 + b'0' + b']' * 10000 + b'}'):
            with self.assertRaises(wire.ApiError):
                wire.decode_json(raw)
        with self.assertRaises(wire.ApiError) as caught:
            wire.decode_json(b" " * (wire.MAX_BODY_BYTES + 1))
        self.assertEqual(caught.exception.status, 413)
        self.assertEqual(wire.decode_json(json.dumps({"a": '[{"\\' * 80}).encode()), {"a": '[{"\\' * 80})


class ParseTests(unittest.TestCase):
    def assert_invalid(self, value, code="invalid_request"):
        with self.assertRaises(wire.ApiError) as caught:
            wire.parse_chat(value)
        self.assertEqual(caught.exception.code, code)

    def test_order_roles_empty_text_and_immutable_parameters(self):
        data = request(messages=[{"role": "system", "content": "s"},
                                 {"role": "developer", "content": [{"type": "text", "text": "d"}]},
                                 {"role": "user", "content": [{"type": "text", "text": ""},
                                                               {"type": "text", "text": "a\nb"}]}],
                       temperature=0, top_p=1, frequency_penalty=-2, presence_penalty=2,
                       seed=-(2**63), stop=["END", "!"], max_tokens=8, reasoning_effort="none")
        parsed = wire.parse_chat(data)
        self.assertEqual([m.role for m in parsed.messages], [MessageRole.SYSTEM, MessageRole.DEVELOPER, MessageRole.USER])
        self.assertEqual(parsed.messages[-1].content, (TextPart(""), TextPart("a\nb")))
        self.assertEqual(parsed.sampling.stop, ("END", "!"))

        self.assertEqual(parsed.explicit_parameters["max_tokens"], 8)
        self.assertNotIn("max_output_tokens", parsed.explicit_parameters)
        with self.assertRaises(TypeError):
            parsed.explicit_parameters["temperature"] = 1
        data["stop"].append("changed")
        self.assertEqual(parsed.explicit_parameters["stop"], ("END", "!"))

    def test_decoded_images_keep_exact_positions_and_all_map_entries_are_consumed(self):
        from kiron_common.local_inference import ImagePart
        from hashlib import sha256
        data = b"already decoded by isolated image worker"
        image = ImagePart("image/png", data, sha256(data).hexdigest(), 1, 1)
        body = request(messages=[{"role":"user","content":[{"type":"text","text":"before"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},
            {"type":"text","text":"between"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},
            {"type":"text","text":"after"}]}])
        parsed = wire.parse_chat(body, image_parts={(0,1):image,(0,3):image})
        self.assertEqual(parsed.messages[0].content,(TextPart("before"),image,TextPart("between"),image,TextPart("after")))
        for images in ({(0,1):image}, {(0,1):image,(0,3):image,(0,4):image},
                       {(0,1):replace(image,detail="low"),(0,3):image}):
            with self.subTest(images=images), self.assertRaises(wire.ApiError):
                wire.parse_chat(body,image_parts=images)

    def test_null_optional_scalars_and_explicit_noops(self):
        parsed = wire.parse_chat(request(temperature=None, top_p=None, seed=None,
            max_tokens=None, max_completion_tokens=None, presence_penalty=None,
            frequency_penalty=None, reasoning_effort=None, n=1, tools=[], tool_choice="none",
            parallel_tool_calls=False, response_format={"type": "text"},
            logprobs=False, top_logprobs=0, logit_bias={}, store=False))
        self.assertEqual(dict(parsed.explicit_parameters), {})
        self.assertFalse(parsed.stream)
        self.assertFalse(parsed.include_usage)
        wire.parse_chat(request(functions=[], function_call="auto"))

    def test_numeric_bool_and_range_boundaries(self):
        invalid = {"temperature": [True, -0.1, 2.1, float("inf")], "top_p": [False, -1, 1.01],
                   "presence_penalty": [True, -2.1], "frequency_penalty": [False, 2.1],
                   "seed": [True, 1.5, -(2**63)-1, 2**63],
                   "max_tokens": [False, 0, -1, 1.2, 100001],
                   "max_completion_tokens": [True, 0, 100001],
                   "n": [True, None, 1.0], "top_logprobs": [False, None],
                   "logprobs": [0, None], "store": [0, None]}
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assert_invalid(request(**{name: value}))
        self.assert_invalid(request(max_tokens=8, max_completion_tokens=8))
        for name, value in (("n", 2), ("logprobs", True), ("top_logprobs", 1),
                            ("store", True), ("logit_bias", {"1": 2})):
            self.assert_invalid(request(**{name: value}), "unsupported_parameter")

    def test_every_protocol_object_is_closed_even_for_false_null_empty(self):
        for value in (False, None, {}):
            cases = [request(options=value), request(messages=[{"role": "user", "content": "x", "name": value}]),
                request(messages=[{"role": "user", "content": [{"type": "text", "text": "x", "extra": value}]}]),
                request(stream=True, stream_options={"include_usage": True, "extra": value}),
                request(response_format={"type": "text", "extra": value}),
                request(tool_choice={"type": "function", "function": {"name": "x", "extra": value}})]
            for case in cases:
                with self.subTest(case=case):
                    self.assert_invalid(case, "unsupported_parameter")

    def test_stream_options_and_stop(self):
        parsed = wire.parse_chat(request(stream=True, stream_options={"include_usage": True}, stop="END"))
        self.assertTrue(parsed.stream and parsed.include_usage)
        self.assertEqual(parsed.sampling.stop, ("END",))
        for value in (None, 1, "true"):
            self.assert_invalid(request(stream=value))
        for value in (None, [], {"include_usage": None}, {"include_usage": 1}):
            self.assert_invalid(request(stream=True, stream_options=value))
        self.assert_invalid(request(stream_options={}))
        for value in (None, "", [], [""], ["a"] * 5, [False]):
            self.assert_invalid(request(stop=value))

    def test_message_and_model_boundaries(self):
        for model in (None, "", "/models/a.gguf", "../a", "a/../b", "https://host/a", "a\\b", "a b", "ä", "x" * 257):
            with self.subTest(model=model):
                self.assert_invalid(request(model=model))
        for messages in ([], None, [{}], [{"role": "function", "content": "x"}],
                         [{"role": "assistant", "content": None}], [{"role": "user", "content": []}],
                         [{"role": "user", "content": "x", "tool_calls": []}],
                         [{"role": "user", "content": "x"}] * 513):
            with self.subTest(messages=messages):
                with self.assertRaises(wire.ApiError):
                    wire.parse_chat(request(messages=messages))
        wire.parse_chat(request(messages=[{"role": "assistant", "content": "", "tool_calls": []}]))
        self.assert_invalid(request(messages=[{"role": "user", "content": [{"type": "text", "text": "x"}] * 65}]))

    def test_future_capabilities_are_explicitly_rejected(self):
        cases = [request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])]
        for case in cases:
            with self.subTest(case=case):
                self.assert_invalid(case, "unsupported_capability")
        self.assert_invalid(request(tools=[], functions=[]))
        self.assert_invalid(request(tool_choice="required"))
        self.assert_invalid(request(reasoning_effort="invented"))
        self.assert_invalid(request(response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": {}, "extra": False}}), "unsupported_parameter")
        self.assert_invalid(request(functions=[{"name": "tool", "parameters": {}}], function_call="required"))

    def test_disabled_capability_objects_still_reject_unknown_protocol_fields(self):
        cases = [request(tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}, "extra": None}]),
                 request(tools=[{"type": "function", "function": {"name": "tool", "parameters": {}, "extra": False}}]),
                 request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a", "extra": {}}}]}]),
                 request(messages=[{"role": "assistant", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}", "extra": None}}]}])]
        for case in cases:
            with self.subTest(case=case):
                self.assert_invalid(case, "unsupported_parameter")

    def test_to_request_preserves_budget_semantics_and_canonical_content(self):
        deployment = ResolvedDeployment("deployment", BackendType.OLLAMA, "tag:latest",
            ArtifactIdentity(ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST), LoaderType.OLLAMA, None, "a" * 64)
        model = ResolvedModel("public/model:tag", deployment, None, None, "b" * 64)
        context = RequestContext("request", time.monotonic() + 2, asyncio.Event())
        for fields, budget in (({}, 16), ({"max_tokens": 7}, 7), ({"max_completion_tokens": 9}, 9)):
            parsed = wire.parse_chat(request(**fields))
            normalized = parsed.to_request(model, context, 16)
            self.assertEqual(normalized.options.max_output_tokens, budget)
            self.assertIs(normalized.model, model)
            self.assertIs(normalized.context, context)
            self.assertEqual(normalized.messages, parsed.messages)
            self.assertFalse(normalized.tools)


class SerializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_reasoning_keeps_thoughts_private_and_requires_exact_usage(self):
        from test_openai_tools import request_for
        request = request_for(reasoning_effort="low")
        result = InferenceResult("internal",(TextPart("answer"),),(ReasoningPart(ReasoningKind.TEXT,"private thought"),),
                                 (),TokenUsage(5,7,reasoning_output_tokens=4),FinishReason.STOP)
        value = wire.serialize_completion(result,"public","id",123,request=request)
        self.assertNotIn("private thought",json.dumps(value))
        self.assertEqual(value["usage"]["completion_tokens_details"],{"reasoning_tokens":4})
        with self.assertRaises(wire.ApiError):
            wire.serialize_completion(replace(result,usage=TokenUsage(5,7)),"public","id",123,request=request)
        values = [event(EventKind.STARTED),event(EventKind.REASONING_DELTA,text="private thought",reasoning_kind=ReasoningKind.TEXT, output_item_index=0, part_index=0),
                  event(EventKind.TEXT_DELTA,text="answer", output_item_index=1, part_index=0),event(EventKind.USAGE,usage=result.usage),
                  event(EventKind.COMPLETED,finish_reason=FinishReason.STOP)]
        output = b"".join([part async for part in wire.chat_events(source(values),"public","id",123,True,request=request)])
        self.assertNotIn(b"private thought",output)
        self.assertEqual(frames(output)[-1]["usage"]["completion_tokens_details"]["reasoning_tokens"],4)
    async def test_interleaved_tools_emit_contiguous_index_blocks_and_exact_empty_arguments(self):
        first, second = ToolCall(0, "a", "weather", '{}'), ToolCall(1, "b", "weather", '{"city":"Berlin"}')
        values = [event(EventKind.STARTED),
            event(EventKind.TOOL_CALL_STARTED, call_index=1, call_id="b", name="weather", output_item_index=0),
            event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=1, text='{"city":', output_item_index=0),
            event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="weather", output_item_index=1),
            event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text="", output_item_index=1),
            event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text="{}", output_item_index=1),
            event(EventKind.TOOL_CALL_COMPLETED, tool_call=first, output_item_index=1),
            event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=1, text='"Berlin"}', output_item_index=0),
            event(EventKind.TOOL_CALL_COMPLETED, tool_call=second, output_item_index=0),
            event(EventKind.USAGE, usage=TokenUsage(19, 5)),
            event(EventKind.COMPLETED, finish_reason=FinishReason.TOOL_CALLS)]
        output = frames(await stream(values))
        fragments = [call for value in output for choice in value.get("choices", []) for call in choice["delta"].get("tool_calls", [])]
        self.assertEqual([f["index"] for f in fragments], [0, 0, 1, 1])
        self.assertEqual([f["function"]["arguments"] for f in fragments], ["", "{}", "", '{"city":"Berlin"}'])
        self.assertEqual(output[-2]["choices"][0]["finish_reason"], "tool_calls")

    async def test_incomplete_tools_survive_length_but_never_fake_success_or_complete(self):
        values = [event(EventKind.STARTED), event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="weather", output_item_index=0),
                  event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text='{"city":', output_item_index=0),
                  event(EventKind.USAGE, usage=TokenUsage(19, 2)), event(EventKind.COMPLETED, finish_reason=FinishReason.LENGTH)]
        output = frames(await stream(values))
        self.assertEqual(output[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(output[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"], '{"city":')
        for reason in (FinishReason.STOP, FinishReason.TOOL_CALLS):
            invalid = values[:-1] + [event(EventKind.COMPLETED, finish_reason=reason)]
            self.assertEqual(frames(await stream(invalid))[-1]["error"]["code"], "backend_protocol_error")

    async def test_tool_complete_mismatch_duplicate_id_and_post_terminal_fail_closed(self):
        first = event(EventKind.TOOL_CALL_STARTED, call_index=0, call_id="a", name="weather", output_item_index=0)
        cases = [[first, first], [first, event(EventKind.TOOL_CALL_STARTED, call_index=1, call_id="a", name="weather", output_item_index=1)],
                 [first, event(EventKind.TOOL_ARGUMENTS_DELTA, call_index=0, text="{}", output_item_index=0),
                  event(EventKind.TOOL_CALL_COMPLETED, tool_call=ToolCall(0, "a", "weather", '{"x":1}'), output_item_index=(ToolCall(0, "a", "weather", '{"x":1}')).index)]]
        for suffix in cases:
            with self.subTest(suffix=suffix):
                output = frames(await stream([event(EventKind.STARTED), *suffix]))
                self.assertEqual(output[-1]["error"]["code"], "backend_protocol_error")

    def test_tool_completion_and_structured_output_checked_against_request(self):
        from test_openai_tools import definition, request_for
        request = request_for(tools=[definition(strict=True)], tool_choice="required")
        result = InferenceResult("internal", (), (), (ToolCall(0,"a","weather",'{"city":"Berlin"}'),), TokenUsage(3,4), FinishReason.TOOL_CALLS)
        value = wire.serialize_completion(result, "public", "id", 123, request=request)
        self.assertIsNone(value["choices"][0]["message"]["content"])
        self.assertEqual(value["choices"][0]["message"]["tool_calls"][0]["id"], "a")
        with self.assertRaises(wire.ApiError):
            wire.serialize_completion(replace(result, tool_calls=(ToolCall(0,"a","weather",'{"city":"Paris"}'),)), "public", "id", 123, request=request)
        structured = request_for(response_format={"type":"json_object"})
        bad = InferenceResult("internal", (TextPart('{"x":1,"x":2}'),), (), (), TokenUsage(3,4), FinishReason.STOP)
        with self.assertRaises(wire.ApiError):
            wire.serialize_completion(bad, "public", "id", 123, request=structured)

    def test_completion_exact_usage_content_and_finish(self):
        result = InferenceResult("internal", (TextPart("a"), TextPart("世界")), (), (), TokenUsage(17, 2, 0, 0), FinishReason.LENGTH)
        value = wire.serialize_completion(result, "public", "chatcmpl-1", 123)
        self.assertEqual(value["choices"][0]["message"]["content"], "a世界")
        self.assertEqual(value["choices"][0]["finish_reason"], "length")
        self.assertEqual(value["usage"], {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19,
            "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}})
        for invalid in (replace(result, usage=None), replace(result, reasoning=(ReasoningPart(ReasoningKind.TEXT, "private"),))):
            with self.assertRaises(wire.ApiError) as caught:
                wire.serialize_completion(invalid, "public", "chatcmpl-1", 123)
            self.assertEqual(caught.exception.code, "backend_protocol_error")
        object.__setattr__(result, "finish_reason", "filter")
        with self.assertRaises(wire.ApiError):
            wire.serialize_completion(result, "public", "chatcmpl-1", 123)

    async def test_stream_exact_stable_metadata_role_finish_and_usage(self):
        data = await stream(sequence())
        values = frames(data)
        self.assertEqual(len(values), 5)
        for value in values:
            self.assertEqual((value["id"], value["model"], value["created"], value["object"]),
                ("chatcmpl-fixed", "public/model:tag", 123, "chat.completion.chunk"))
        self.assertEqual(values[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual([v["choices"][0]["delta"].get("content") for v in values[1:3]], ["Hallo ", "世界"])
        self.assertEqual(values[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(values[-1]["choices"], [])
        self.assertEqual(values[-1]["usage"]["total_tokens"], 19)
        self.assertTrue(all(v["usage"] is None for v in values[:-1]))
        without = frames(await stream(sequence(), include_usage=False))
        self.assertEqual(len(without), 4)
        self.assertTrue(all("usage" not in v for v in without))

    async def test_malformed_sequences_never_emit_success_finish(self):
        good = sequence()
        bad = [[], good[1:], good[:-1], good[:3] + good[-1:],
               good[:1] + good, good + good[-1:], good[:4] + [good[1]] + good[-1:],
               good[:4] + [good[3]] + good[-1:],
               good[:1] + [InferenceEvent(EventKind.TEXT_DELTA, "other", text="x", output_item_index=0, part_index=0)],
               good[:1] + [event(EventKind.REASONING_DELTA, text="private", reasoning_kind=ReasoningKind.TEXT, output_item_index=0, part_index=0)],
               good[:1] + [object()]]
        for values in bad:
            with self.subTest(values=values):
                output = frames(await stream(values))
                self.assertEqual(sum("error" in value for value in output), 1)
                self.assertEqual(output[-1]["error"]["code"], "backend_protocol_error")
                self.assertTrue(all(not c["finish_reason"] for v in output if "choices" in v for c in v["choices"]))
                self.assertTrue(all(v.get("usage") is None for v in output))

    async def test_error_after_completed_is_not_success_and_messages_are_sanitized(self):
        for error in (RuntimeError("secret /models/x"), wire.ApiError("secret", "secret"),
                      LocalInferenceError(RuntimeFailure(ErrorCode.TIMEOUT, "secret", "/models/private"))):
            output = await stream(sequence(), error=error)
            self.assertNotIn(b"secret", output)
            self.assertNotIn(b"/models/", output)
            values = frames(output)
            self.assertIn("error", values[-1])
            self.assertTrue(all(not c["finish_reason"] for v in values if "choices" in v for c in v["choices"]))

    async def test_typed_failures_and_cancellation_have_one_safe_error(self):
        for code, public in ((ErrorCode.PROVIDER_ERROR, "backend_protocol_error"),
                             (ErrorCode.INVALID_CONFIGURATION, "model_unavailable"),
                             (ErrorCode.OVERLOADED, "overloaded"), (ErrorCode.CONFLICT, "resource_busy"),
                             (ErrorCode.TIMEOUT, "timeout")):
            values = frames(await stream([event(EventKind.STARTED), event(EventKind.FAILED,
                error=RuntimeFailure(code, "secret", "secret"))]))
            self.assertEqual(values[-1]["error"]["code"], public)
            self.assertIsNone(values[-1]["error"]["param"])
        values = frames(await stream([event(EventKind.STARTED), event(EventKind.CANCELLED)]))
        self.assertEqual(values[-1]["error"]["code"], "resource_busy")
        with self.assertRaises(asyncio.CancelledError):
            await stream([event(EventKind.STARTED)], error=asyncio.CancelledError())
        values = frames(await stream([event(EventKind.STARTED)], error=TimeoutError("private")))
        self.assertEqual(values[-1]["error"]["code"], "timeout")

    async def test_usage_remains_required_without_public_usage_and_forged_counts_fail(self):
        missing = sequence()[:3] + sequence()[-1:]
        values = frames(await stream(missing, include_usage=False))
        self.assertEqual(values[-1]["error"]["code"], "backend_protocol_error")
        bad = TokenUsage(2, 1)
        object.__setattr__(bad, "output_tokens", -1)
        values = frames(await stream([event(EventKind.STARTED), event(EventKind.USAGE, usage=bad),
                                     event(EventKind.COMPLETED, finish_reason=FinishReason.STOP)]))
        self.assertEqual(values[-1]["error"]["code"], "backend_protocol_error")
        empty = [event(EventKind.STARTED), event(EventKind.USAGE, usage=TokenUsage(2, 0)),
                 event(EventKind.COMPLETED, finish_reason=FinishReason.LENGTH)]
        values = frames(await stream(empty))
        self.assertEqual(values[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(values[-1]["usage"]["completion_tokens"], 0)

    async def test_output_caps_count_utf8_and_reserve_error_tail(self):
        with patch.object(wire, "MAX_OUTPUT_BYTES", 1800):
            result = InferenceResult("internal", (TextPart("世" * 600),), (), (), TokenUsage(1, 1), FinishReason.STOP)
            with self.assertRaises(wire.ApiError):
                wire.serialize_completion(result, "public", "id", 123)
            data = await stream([event(EventKind.STARTED), event(EventKind.TEXT_DELTA, text="世" * 600, output_item_index=0, part_index=0)])
            self.assertLessEqual(len(data), 1800)
            self.assertEqual(frames(data)[-1]["error"]["code"], "backend_protocol_error")

    async def test_iterator_completion_is_observed_before_success_published(self):
        gate = asyncio.Event()
        async def delayed():
            for value in sequence():
                yield value
            await gate.wait()
        output = wire.chat_events(delayed(), "public", "id", 123, True)
        for _ in range(3):
            self.assertNotIn(b'"finish_reason":"stop"', await anext(output))
        pending = asyncio.create_task(anext(output))
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        gate.set()
        self.assertIn(b'"finish_reason":"stop"', await pending)
        await output.aclose()

if __name__ == "__main__":
    unittest.main()
