"""Tool request/history semantics independent of backends and tool execution."""
import asyncio
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

from kiron_common.local_inference import (
    ArtifactIdentity, FinishReason, RequestContext, ResolvedDeployment, ResolvedModel,
    ToolCall, ToolChoiceKind, ResourceProfile,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
import openai_tools as tools
import openai_wire as wire


def definition(name="weather", strict=False):
    return {"type": "function", "function": {"name": name, "strict": strict,
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Berlin"]}},
                       "required": ["city"], "additionalProperties": False}}}


def raw_call(call_id="a", name="weather", arguments='{"city":"Berlin"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def raw_request(**fields):
    return {"model": "public", "messages": [{"role": "user", "content": "weather"}], **fields}


def request_for(**fields):
    deployment = ResolvedDeployment("deployment", BackendType.PRISM, "/model.gguf",
        ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 1),
        LoaderType.PRISM_GGUF, ResourceProfile("test",1024,128,128,1,2,0,False,0,0), "b" * 64)
    model = ResolvedModel("public", deployment, None, None, "c" * 64)
    context = RequestContext("tool-request", time.monotonic() + 60, asyncio.Event())
    return wire.parse_chat(raw_request(**fields)).to_request(model, context, 128)


class ToolParserTests(unittest.TestCase):
    def test_modes_defaults_and_one_legacy_normalization(self):
        self.assertIs(wire.parse_chat(raw_request()).tool_choice.kind, ToolChoiceKind.NONE)
        for mode in ("none", "auto", "required", {"type": "function", "function": {"name": "weather"}}):
            parsed = wire.parse_chat(raw_request(tools=[definition()], tool_choice=mode, parallel_tool_calls=True))
            self.assertEqual(parsed.tools[0].name, "weather")
            self.assertTrue(parsed.parallel_tool_calls)
        modern = wire.parse_chat(raw_request(tools=[definition()], tool_choice={"type": "function", "function": {"name": "weather"}}))
        legacy = wire.parse_chat(raw_request(functions=[definition()["function"]], function_call={"name": "weather"}))
        self.assertEqual((legacy.tools, legacy.tool_choice), (modern.tools, modern.tool_choice))
        self.assertIs(wire.parse_chat(raw_request(tools=[definition()])).tool_choice.kind, ToolChoiceKind.AUTO)

    def test_history_roundtrips_by_id_allow_reordered_same_name_results(self):
        history = [{"role": "user", "content": "both"},
            {"role": "assistant", "content": None, "tool_calls": [raw_call("a"), raw_call("b")]},
            {"role": "tool", "tool_call_id": "b", "content": "second"},
            {"role": "tool", "tool_call_id": "a", "content": "first"},
            {"role": "assistant", "content": "continue", "tool_calls": [raw_call("c")]},
            {"role": "tool", "tool_call_id": "c", "content": "third"}]
        result = wire.parse_chat(raw_request(messages=history))
        self.assertEqual([m.tool_call_id for m in result.messages if m.tool_call_id], ["b", "a", "c"])
        self.assertEqual(result.messages[1].content, ())

    def test_unknown_duplicate_and_open_history_are_invalid(self):
        assistant = {"role": "assistant", "tool_calls": [raw_call()]}
        answer = {"role": "tool", "tool_call_id": "a", "content": "x"}
        for history in ([assistant], [answer], [assistant, answer, answer],
                        [assistant, {"role": "user", "content": "next"}],
                        [assistant, answer, assistant, answer],
                        [assistant, {**answer, "tool_call_id": "unknown"}]):
            with self.subTest(history=history), self.assertRaises(wire.ApiError) as caught:
                wire.parse_chat(raw_request(messages=history))
            self.assertEqual(caught.exception.code, "invalid_request")

    def test_closed_shapes_names_legacy_and_schema_fail_before_any_execution(self):
        bad = [raw_request(functions=[], tools=[]), raw_request(function_call="auto", parallel_tool_calls=False),
            raw_request(tools=[definition(), definition()]), raw_request(tools=[definition("run.shell;rm")]),
            raw_request(tool_choice="required"), raw_request(tools=[definition()], tool_choice={"type": "function", "function": {"name": "other"}}),
            raw_request(functions=[definition()["function"]], function_call="required"),
            raw_request(tools=[{**definition(), "extra": False}])]
        with patch("subprocess.Popen", side_effect=AssertionError("no execution")):
            for value in bad:
                with self.subTest(value=value), self.assertRaises(wire.ApiError):
                    wire.parse_chat(value)
            wire.parse_chat(raw_request(tools=[definition("exec")]))

    def test_argument_json_is_an_object_without_duplicate_keys_nonfinite_or_surrogates(self):
        for source in ('', '[]', 'null', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}',
                       '{"x":"\\ud800"}', '{"x":' + '[' * 33 + '0' + ']' * 33 + '}'):
            with self.subTest(source=source), self.assertRaises(tools.ToolError):
                tools.arguments_object(source)
        self.assertEqual(tools.arguments_object('{}'), {})
        with patch.object(tools, "MAX_ARGUMENT_BYTES", 8):
            with self.assertRaises(tools.ToolError):
                tools.arguments_object('{"x":"世"}')

    def test_sdk_helper_replay_annotations_are_checked_not_trusted(self):
        call = raw_call(arguments='{"n":1,"nested":[true,null,{"x":"y"}]}')
        call["index"] = 0
        call["function"]["parsed_arguments"] = {"nested":[True,None,{"x":"y"}],"n":1.0}
        parsed = tools.parse_calls([call],param="messages[1].tool_calls")
        self.assertEqual(parsed[0].arguments,call["function"]["arguments"])
        for index, parsed_value in ((True,{}),(1,{}),(0,{"n":True,"nested":[True,None,{"x":"y"}]}),
                                    (0,None),(0,{"n":1,"nested":[False,None,{"x":"y"}]})):
            bad = {**call,"index":index,"function":{**call["function"],"parsed_arguments":parsed_value}}
            with self.subTest(index=index,parsed=parsed_value), self.assertRaises(tools.ToolError):
                tools.parse_calls([bad],param="messages[1].tool_calls")
        with self.assertRaises(tools.ToolError) as caught:
            tools.parse_calls([{**call,"future_sdk_field":False}],param="messages[1].tool_calls")
        self.assertEqual(caught.exception.code,"unsupported_parameter")

    def test_strict_schema_complete_vs_length_and_call_choice_validation(self):
        request = request_for(tools=[definition(strict=True)], tool_choice="required")
        good = ToolCall(0, "a", "weather", '{"city":"Berlin"}')
        tools.validate_calls((good,), request.tools, request.tool_choice, False, FinishReason.TOOL_CALLS)
        for calls, finish in (((replace(good, arguments='{"city":"Paris"}'),), FinishReason.TOOL_CALLS),
                              ((), FinishReason.STOP), ((good,), FinishReason.STOP),
                              ((good, replace(good, index=1, id="b")), FinishReason.TOOL_CALLS)):
            with self.subTest(calls=calls), self.assertRaises(tools.ToolError):
                tools.validate_calls(calls, request.tools, request.tool_choice, False, finish)
        partial = ToolCall(0, "a", "weather", '{"city":', complete=False)
        tools.validate_calls((partial,), request.tools, request.tool_choice, False, FinishReason.LENGTH)

    def test_request_preserves_all_canonical_tool_values_and_immutable_schema(self):
        raw = definition(strict=True)
        request = request_for(tools=[raw], tool_choice="required", parallel_tool_calls=True)
        raw["function"]["parameters"]["required"].clear()
        self.assertEqual(request.tools[0].parameters["required"], ("city",))
        self.assertTrue(request.options.parallel_tool_calls)
        self.assertIs(request.tool_choice.kind, ToolChoiceKind.REQUIRED)


if __name__ == "__main__":
    unittest.main()
