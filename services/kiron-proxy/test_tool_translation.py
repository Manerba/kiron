"""Pinned native tool shapes and conservative translator failure boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kiron_common.local_inference import EventKind, FinishReason, LocalInferenceError, validate_event_sequence, InferenceEvent, TokenUsage
import prism_tools as prism
import ollama_tools as ollama
from test_openai_tools import definition, raw_call, request_for


class PrismToolsTests(unittest.TestCase):
    def test_parallel_history_maps_distinct_reversed_results_by_id_before_qwen_rendering(self):
        for second_name in ("weather","clock"):
            with self.subTest(second_name=second_name):
                history = [{"role":"user","content":"Compare both cities"},
                    {"role":"assistant","tool_calls":[raw_call("berlin"),
                        raw_call("paris",second_name,'{"city":"Paris"}')]},
                    {"role":"tool","tool_call_id":"paris","content":"PARIS_VALUE_29"},
                    {"role":"tool","tool_call_id":"berlin","content":"BERLIN_VALUE_11"}]
                request = request_for(tools=[definition(),definition("clock")], messages=history,
                                      parallel_tool_calls=True)
                plan = prism.prepare_tools(request)
                self.assertEqual([m["tool_call_id"] for m in plan.messages[2:]],["berlin","paris"])
                self.assertEqual([m["content"] for m in plan.messages[2:]],["BERLIN_VALUE_11","PARIS_VALUE_29"])
                calls = plan.messages[1]["tool_calls"]
                self.assertEqual([(c["id"],c["function"]["name"],json.loads(c["function"]["arguments"])["city"])
                                  for c in calls],[("berlin","weather","Berlin"),("paris",second_name,"Paris")])
                self.assertEqual([m.tool_call_id for m in request.messages[2:]],["paris","berlin"])

    def test_tool_result_ordering_is_scoped_to_each_round_and_keeps_pretranslated_content(self):
        history = [{"role":"user","content":"first query"},
            {"role":"assistant","tool_calls":[raw_call("a"),raw_call("b","clock")]},
            {"role":"tool","tool_call_id":"b","content":"second"},
            {"role":"tool","tool_call_id":"a","content":"first"},
            {"role":"assistant","content":"first answer"},
            {"role":"user","content":"next query"},
            {"role":"assistant","tool_calls":[raw_call("c","clock")]},
            {"role":"tool","tool_call_id":"c","content":"third"}]
        request = request_for(tools=[definition(strict=True),definition("clock")],messages=history)
        rendered = [{"role":m.role.value,"content":"".join(p.text for p in m.content)} for m in request.messages]
        rendered[0]["content"] = [{"type":"text","text":"first query"}]
        original = json.loads(json.dumps(rendered))
        plan = prism.prepare_tools(request,messages=rendered)
        self.assertEqual([m.get("tool_call_id") for m in plan.messages],[None,None,"a","b",None,None,None,"c"])
        self.assertEqual([m["content"] for m in plan.messages[2:4]],["first","second"])
        self.assertEqual(plan.messages[0]["content"],rendered[0]["content"])
        self.assertEqual(rendered,original)
        self.assertEqual(plan.messages[1]["tool_calls"][0]["function"]["arguments"],'{"payload":{"city":"Berlin"}}')
        self.assertEqual(plan.messages[6]["tool_calls"][0]["function"]["arguments"],'{"city":"Berlin"}')

    def test_named_strict_plan_narrows_grammar_and_wraps_history_without_changing_ids(self):
        history = [{"role": "assistant", "tool_calls": [raw_call()]},
                   {"role": "tool", "tool_call_id": "a", "content": "result"}]
        request = request_for(tools=[definition(strict=True), definition("echo")], messages=history,
            tool_choice={"type": "function", "function": {"name": "weather"}}, parallel_tool_calls=True)
        plan = prism.prepare_tools(request)
        self.assertEqual([t["function"]["name"] for t in plan.payload["tools"]], ["weather"])
        self.assertEqual(plan.payload["tool_choice"], "required")
        self.assertFalse(plan.payload["parallel_tool_calls"])
        wrapped = plan.payload["tools"][0]["function"]["parameters"]
        self.assertEqual(wrapped["properties"]["payload"]["properties"]["city"]["enum"], ["Berlin"])
        self.assertEqual(plan.messages[0]["tool_calls"][0]["id"], "a")
        self.assertEqual(plan.messages[0]["tool_calls"][0]["function"]["arguments"], '{"payload":{"city":"Berlin"}}')
        self.assertEqual(plan.messages[1]["tool_call_id"], "a")
        self.assertEqual(request.messages[0].tool_calls[0].arguments, '{"city":"Berlin"}')

    def test_strict_result_unwraps_and_checks_actual_schema(self):
        plan = prism.prepare_tools(request_for(tools=[definition(strict=True)], tool_choice="required"))
        raw = raw_call(arguments='{"payload":{"city":"Berlin"}}')
        result = prism.decode_tool_calls([raw], plan, finish_reason="tool_calls")
        self.assertEqual(result[0].arguments, '{"city":"Berlin"}')
        for arguments in ('{"payload":{"city":"Paris"}}', '{"payload":{"city":"Berlin"},"other":0}',
                          '{"payload":{"city":"Berlin"},"payload":{"city":"Paris"}}', '{"city":"Berlin"}',
                          '{"payload":[]}', '{"payload":{"city":"Berlin","city":"Paris"}}'):
            with self.subTest(arguments=arguments), self.assertRaises(LocalInferenceError):
                prism.decode_tool_calls([raw_call(arguments=arguments)], plan, finish_reason="tool_calls")

    def test_local_refs_are_expanded_under_wrapper(self):
        tool = definition(strict=True)
        schema = tool["function"]["parameters"]
        schema["$defs"] = {"city": schema["properties"]["city"]}
        schema["properties"]["city"] = {"$ref": "#/$defs/city"}
        plan = prism.prepare_tools(request_for(tools=[tool]))
        self.assertNotIn("$ref", json.dumps(plan.payload))
        self.assertNotIn("$defs", json.dumps(plan.payload))
        self.assertEqual(plan.payload["tools"][0]["function"]["parameters"]["properties"]["payload"]["properties"]["city"]["enum"], ["Berlin"])

    def test_partial_wrapper_at_every_boundary_is_lossless_and_never_complete(self):
        plan = prism.prepare_tools(request_for(tools=[definition(strict=True)], tool_choice="required"))
        source = '{"payload":{"city":"Berlin"}}'
        inner_start, inner_end = source.index('{', 1), len(source)-1
        for end in range(len(source) + 1):
            with self.subTest(end=end):
                call = prism.decode_tool_calls([raw_call(arguments=source[:end])], plan, finish_reason="length")[0]
                self.assertFalse(call.complete)
                self.assertEqual(call.arguments, source[inner_start:min(end, inner_end)] if end > inner_start else "")

    def test_interleaved_strict_stream_keeps_ids_and_validates_before_completed(self):
        request = request_for(tools=[definition(strict=True)], tool_choice="required", parallel_tool_calls=True)
        decoder = prism.ToolStreamDecoder(prism.prepare_tools(request))
        events = [InferenceEvent(EventKind.STARTED, request.context.request_id)]
        for index, call_id in ((1, "second"), (0, "first")):
            events.extend(decoder.feed([{"index": index, **raw_call(call_id, arguments='{"payload":')}]))
        for index in (0, 1):
            events.extend(decoder.feed([{"index": index, "function": {"arguments": '{"city":"Berlin"}}'}}]))
        self.assertFalse(any(e.kind is EventKind.TOOL_ARGUMENTS_DELTA for e in events))
        events.extend(decoder.finish("tool_calls"))
        events.extend([InferenceEvent(EventKind.USAGE, request.context.request_id, usage=TokenUsage(3, 8)),
                       InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.TOOL_CALLS)])
        validate_event_sequence(events)
        self.assertEqual([e.tool_call.id for e in events if e.kind is EventKind.TOOL_CALL_COMPLETED], ["first", "second"])

    def test_stream_rejects_changed_identity_unknown_name_duplicate_indices_and_bounds(self):
        plan = prism.prepare_tools(request_for(tools=[definition()]))
        for initial, extra in (([{"index": 0, **raw_call(arguments="{")}], [{"index": 0, **raw_call(arguments="}")}]),
            ([{"index": 0, **raw_call(arguments="{")}], [{"index": 1, **raw_call(arguments="{}")}]),
            ([{"index": 0, **raw_call(name="unknown")}], []),
            ([{"index": True, **raw_call()}], [])):
            with self.subTest(initial=initial), self.assertRaises(LocalInferenceError):
                decoder = prism.ToolStreamDecoder(plan)
                decoder.feed(initial)
                decoder.feed(extra)
                decoder.finish("tool_calls")
        with patch.object(prism, "MAX_ARGUMENT_BYTES", 8), self.assertRaises(LocalInferenceError):
            prism.ToolStreamDecoder(plan).feed([{"index": 0, **raw_call()}])

    def test_none_required_named_and_parallel_enforced_for_definitive_outputs(self):
        for mode, raw, parallel in (("none", [raw_call()], False), ("required", [], False),
            ({"type": "function", "function": {"name": "weather"}}, [raw_call(), raw_call("b")], True),
            ("auto", [raw_call(), raw_call("b")], False)):
            with self.subTest(mode=mode), self.assertRaises(LocalInferenceError):
                plan = prism.prepare_tools(request_for(tools=[definition()], tool_choice=mode, parallel_tool_calls=parallel))
                prism.decode_tool_calls(raw, plan, finish_reason="tool_calls" if raw else "stop")

    def test_new_backend_call_cannot_reuse_a_historical_id(self):
        history = [{"role":"assistant","tool_calls":[raw_call("previous")]},
                   {"role":"tool","tool_call_id":"previous","content":"done"}]
        plan = prism.prepare_tools(request_for(tools=[definition()],messages=history))
        with self.assertRaises(LocalInferenceError):
            prism.decode_tool_calls([raw_call("previous")],plan,finish_reason="tool_calls")
        self.assertEqual(prism.decode_tool_calls([raw_call("next")],plan,finish_reason="tool_calls")[0].id,"next")

    def test_recorded_native_fragment_shapes_replay_without_claiming_strict_enforcement(self):
        root = Path(__file__).parent / "contracts/fixtures/prism-b10709"
        for name in ("tool_stream", "parallel_tools_stream"):
            request = request_for(tools=[definition()], parallel_tool_calls=name.startswith("parallel"))
            # These historical samples carry ordinary, unwrapped arguments;
            # the fixture proves shape only, not the recorded strict flag.
            decoder = prism.ToolStreamDecoder(prism.prepare_tools(request))
            calls = []
            for line in (root / f"{name}.response.raw").read_text().splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:])["choices"]:
                    if choice["delta"].get("tool_calls"):
                        decoder.feed(choice["delta"]["tool_calls"])
                    if choice["finish_reason"]:
                        calls.extend(e.tool_call for e in decoder.finish(choice["finish_reason"]) if e.tool_call)
            self.assertEqual([json.loads(c.arguments)["city"] for c in calls], ["Berlin", "Paris"] if name.startswith("parallel") else ["Berlin"])


class OllamaToolsTests(unittest.TestCase):
    def test_unsupported_native_modes_are_explicit_and_no_prompt_is_rewritten(self):
        for fields in ({"tool_choice": "required"}, {"tool_choice": {"type": "function", "function": {"name": "weather"}}},
                       {"tools": [definition(strict=True)]}):
            with self.subTest(fields=fields), self.assertRaises(LocalInferenceError) as caught:
                ollama.prepare_tools(request_for(**{"tools": [definition()], **fields}))
            self.assertEqual(caught.exception.failure.code.value, "unsupported_capability")
        request = request_for(tools=[definition()])
        self.assertEqual(ollama.prepare_tools(request).messages[0]["content"], "weather")

    def test_native_history_reorders_results_by_call_id_and_uses_exact_names(self):
        history = [{"role": "assistant", "tool_calls": [raw_call("a"), raw_call("b","clock",'{"city":"Paris"}')]},
                   {"role": "tool", "tool_call_id": "b", "content": "second"},
                   {"role": "tool", "tool_call_id": "a", "content": "first"}]
        plan = ollama.prepare_tools(request_for(tools=[definition(),definition("clock")], messages=history))
        self.assertEqual([m["content"] for m in plan.messages[1:]], ["first", "second"])
        self.assertEqual([m["tool_name"] for m in plan.messages[1:]], ["weather", "clock"])
        self.assertEqual(plan.messages[0]["tool_calls"][0]["function"]["arguments"], {"city": "Berlin"})
        self.assertEqual(plan.messages[0]["tool_calls"][1]["function"]["arguments"], {"city": "Paris"})

    def test_nonstream_and_stream_ids_are_stable_and_distinguish_same_name_calls(self):
        request = request_for(tools=[definition()], parallel_tool_calls=True)
        plan = ollama.prepare_tools(request)
        calls = [{"function": {"name": "weather", "arguments": {"city": "Berlin"}}}] * 2
        decoded = ollama.decode_tool_calls(calls, plan, finish_reason="stop")
        decoder = ollama.ToolStreamDecoder(plan)
        decoder.feed(calls)
        finished = decoder.finish("stop")
        self.assertEqual([e.tool_call for e in finished], list(decoded))
        self.assertNotEqual(decoded[0].id, decoded[1].id)
        self.assertIs(ollama.canonical_finish("stop", decoder.has_calls), FinishReason.TOOL_CALLS)

    def test_native_argument_strings_or_repeated_indices_are_not_guessed(self):
        plan = ollama.prepare_tools(request_for(tools=[definition()], parallel_tool_calls=True))
        for raw in ({"function": {"name": "weather", "arguments": "{}"}},
                    {"function": {"index": 1, "name": "weather", "arguments": {}}}):
            with self.subTest(raw=raw), self.assertRaises(LocalInferenceError):
                ollama.ToolStreamDecoder(plan).feed([raw])


if __name__ == "__main__":
    unittest.main()
