"""No-network tests for direct upstream evidence and failure handling."""
import argparse
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("prism_smoke", Path(__file__).with_name("smoke-runtime.py"))
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
USAGE = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}


def tool(name="weather", arguments='{"city":"Berlin"}', identity="call_1"):
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


def completion(content="OK", calls=None):
    return {"object": "chat.completion", "id": "chat_1", "model": "fixture", "usage": USAGE,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}}]}


def wire(payload, status=200, sse=False):
    return {"status": status, "headers": {"Content-Type": "text/event-stream" if sse else "application/json"},
            "raw": payload if sse else json.dumps(payload).encode(), "elapsed_s": 0.01, "error": None}


def chunk(delta=None, finish=None, usage=None):
    return {"object": "chat.completion.chunk", "id": "chat_1", "usage": usage,
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}


def sse(chunks, done=True):
    data = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks)
    return (data + ("data: [DONE]\n\n" if done else "")).encode()


class SmokeTests(unittest.TestCase):
    def runner(self, replies, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        values = dict(base_url="http://127.0.0.1:18089", model="fixture", output_dir=directory.name,
                      timeout=1, max_tokens=64, max_output_bytes=smoke.OUTPUT_LIMIT,
                      reasoning_effort=None, reasoning_budget_tokens=32, vision_image=None, vision_expect=None, case=None)
        values.update(overrides)
        send = Mock(side_effect=deepcopy(replies))
        return smoke.Runner(argparse.Namespace(**values), send), send

    def test_loopback_and_production_port_guard(self):
        for address in ("http://127.0.0.1:18089", "http://[::1]:18089"):
            smoke.validate_base_url(address)
        invalid = ["http://localhost:18089", "https://127.0.0.1:18089", "http://192.0.2.1:18089",
                   "http://" "user:password@127.0.0.1:18089", "http://127.0.0.1:18089/v1",
                   "http://127.0.0.1:18089?x=1", "http://127.0.0.1", "http://127.0.0.1:80"]
        invalid += [f"http://127.0.0.1:{port}" for port in smoke.PRODUCTION_PORTS]
        for address in invalid:
            with self.subTest(address=address), self.assertRaises(ValueError):
                smoke.validate_base_url(address)

    def test_mixed_outcomes_preserve_every_raw_response(self):
        runner, _ = self.runner([wire({"status": "ok"}), wire({"error": "not found"}, 404),
                                 wire({"error": "crashed"}, 500), wire(completion())])
        summary = runner.run(["health", "responses", "embeddings", "text"])
        self.assertEqual([row["status"] for row in summary["results"]], ["pass", "gap", "fail", "pass"])
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(json.loads((runner.directory / "02-responses.response.raw").read_bytes()), {"error": "not found"})
        self.assertEqual(json.loads((runner.directory / "summary.json").read_text())["request_count"], 4)

    def test_interleaved_tools_assemble_by_index(self):
        chunks = [chunk({"role": "assistant", "tool_calls": [
            {"index": index, **tool(arguments="", identity=f"call_{index}")} for index in range(2)]})]
        for index, fragment in [(0, '{"city":'), (1, '{"city":'), (0, '"Berlin"}'), (1, '"Paris"}')]:
            chunks.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}))
        chunks += [chunk({}, "tool_calls"), chunk(usage=USAGE)]
        message, finish, events, usage = smoke.stream_chat(sse(chunks))
        self.assertEqual(len(smoke.validate_tools(message, finish, parallel=True)), 2)
        self.assertEqual(events, len(chunks))
        self.assertEqual(usage, USAGE)

    def test_parallel_tools_stream_runner_handles_interleaved_calls(self):
        chunks = [chunk({"role": "assistant", "tool_calls": [
            {"index": index, **tool(arguments="", identity=f"call_{index}")} for index in range(2)]})]
        for index, fragment in [(0, '{"city":'), (1, '{"city":'), (0, '"Berlin"}'), (1, '"Paris"}')]:
            chunks.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]}))
        chunks += [chunk({}, "tool_calls"), chunk(usage=USAGE)]
        runner, send = self.runner([wire(sse(chunks), sse=True)])
        result = runner.run(["parallel_tools_stream"])["results"][0]
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["observed"]["calls"], 2)
        self.assertEqual(result["observed"]["stream_events"], len(chunks))
        request = send.call_args.args[2]
        self.assertTrue(request["stream"])
        self.assertTrue(request["parallel_tool_calls"])
        self.assertTrue(request["stream_options"]["include_usage"])
        self.assertEqual(request["tool_choice"], "required")
        self.assertIn("Paris", request["messages"][0]["content"])

    def test_malformed_truncated_error_and_unfinished_streams_fail(self):
        cases = [b"data: not JSON\n\ndata: [DONE]\n\n",
                 sse([chunk({"content": "OK"}), chunk(usage=USAGE)]),
                 sse([chunk({"content": "OK"}, "stop"), chunk(usage=USAGE)], done=False),
                 sse([chunk({"content": "OK"}, "stop")]),
                 sse([{"error": {"message": "upstream failure"}}])]
        for payload in cases:
            with self.subTest(payload=payload):
                runner, _ = self.runner([wire(payload, sse=True), wire({"status": "ok"})])
                results = runner.run(["text_stream", "health"])["results"]
                self.assertEqual([r["status"] for r in results], ["fail", "pass"])

    def test_invalid_tools_named_selection_and_schema_are_not_passes(self):
        cases = [("tool", tool(arguments="not JSON")), ("tool", tool(identity="")),
                 ("named_tool_choice", tool(name="echo")), ("tool", tool(arguments='{"city":"Rome"}'))]
        for case, call in cases:
            with self.subTest(case=case, call=call):
                runner, send = self.runner([wire(completion(None, [call]))])
                self.assertEqual(runner.run([case])["results"][0]["status"], "fail")
                if case == "named_tool_choice":
                    self.assertEqual(send.call_args.args[2]["tool_choice"]["function"]["name"], "weather")
        runner, _ = self.runner([wire(completion('{"ok":1}'))])
        self.assertEqual(runner.run(["json_schema"])["results"][0]["status"], "fail")

    def test_roundtrip_uses_only_a_fixed_fixture_result(self):
        runner, send = self.runner([wire(completion(None, [tool()])), wire(completion("It is 20 degrees."))])
        self.assertEqual(runner.run(["tool_roundtrip"])["results"][0]["status"], "pass")
        request = send.call_args_list[1].args[2]
        self.assertEqual(request["messages"][-1], {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature_c":20}'})
        self.assertEqual(request["tool_choice"], "none")

    def test_explicit_reasoning_controls_and_required_native_output(self):
        reply = completion("323")
        reply["choices"][0]["message"]["reasoning_content"] = "17*20 minus 17."
        reply["__verbose"] = {"tokens": [1, 2, 3, 4], "tokens_predicted": 4, "content": "17*20 minus 17.</think>323"}
        runner, send = self.runner([wire(reply), wire(completion())], max_tokens=128)
        results = runner.run(["reasoning", "text"])["results"]
        self.assertEqual([r["status"] for r in results], ["pass", "pass"])
        evidence = results[0]["observed"]["native_token_evidence"]
        self.assertTrue(evidence["id_count_matches_completion_tokens"])
        self.assertEqual(evidence["reasoning_token_boundary"], "not_established")
        self.assertIsNone(evidence["reasoning_tokens"])
        request = send.call_args_list[0].args[2]
        self.assertTrue(request["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(request["return_tokens"])
        self.assertTrue(request["verbose"])
        self.assertEqual(request["reasoning_budget_tokens"], 32)
        self.assertNotIn("reasoning_effort", request)
        self.assertEqual(send.call_args_list[1].args[2]["reasoning_effort"], "none")
        runner, _ = self.runner([wire(completion("323"))])
        self.assertEqual(runner.run(["reasoning"])["results"][0]["status"], "fail")
        reply["__verbose"]["tokens"].pop()
        runner, _ = self.runner([wire(reply)])
        evidence = runner.run(["reasoning"])["results"][0]["observed"]["native_token_evidence"]
        self.assertFalse(evidence["id_count_matches_completion_tokens"])

    def test_adversarial_enum_is_only_a_sample_not_an_enforcement_guarantee(self):
        for city, expected in [("Berlin", "pass"), ("Tokyo", "fail")]:
            with self.subTest(city=city):
                runner, send = self.runner([wire(completion(None, [tool(arguments=json.dumps({"city": city}))]))])
                result = runner.run(["strict_tool_enum"])["results"][0]
                self.assertEqual(result["status"], expected)
                request = send.call_args.args[2]
                self.assertIn("Tokyo", request["messages"][0]["content"])
                self.assertEqual(request["tools"][0]["function"]["parameters"]["properties"]["city"]["enum"], ["Berlin"])
                if expected == "pass":
                    self.assertTrue(result["observed"]["sample_only"])
                    self.assertEqual(result["observed"]["constraint_enforcement"], "not_established")
                    self.assertIn("drops strict", result["observed"]["source_gap"])

    def test_only_exact_known_vision_projector_error_is_a_gap(self):
        error = {"error": {"code": 500, "type": "server_error", "message": smoke.MISSING_PROJECTOR}}
        runner, _ = self.runner([wire(error, 500), wire(error, 500), wire({"error": "unrelated mmproj failure"}, 500)])
        with self.assertRaises(smoke.Gap):
            runner.ask("vision", "/v1/chat/completions", {})
        with self.assertRaises(ValueError):
            runner.ask("text", "/v1/chat/completions", {})
        with self.assertRaises(ValueError):
            runner.ask("vision", "/v1/chat/completions", {})

    def test_responses_incomplete_schema_cannot_pass_on_http_200(self):
        runner, _ = self.runner([wire({"object": "response", "status": "completed", "output": []})])
        self.assertEqual(runner.run(["responses"])["results"][0]["status"], "fail")

    def test_vision_requires_an_observed_fixture_marker(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            image.write(smoke.base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jA1cAAAAASUVORK5CYII="))
            image.flush()
            for marker, answer, expected in [
                (None, "A white pixel.", "gap"), ("white", "white", "pass"),
                ("red", " Red\n", "pass"), ("blue", "BLUE", "pass"),
                ("red", "not red", "fail"), ("red", "red or blue", "fail"),
                ("red", "inferred", "fail"), ("blue", "red", "fail"),
            ]:
                with self.subTest(marker=marker, answer=answer):
                    runner, send = self.runner([wire(completion(answer))], vision_image=image.name, vision_expect=marker)
                    self.assertEqual(runner.run(["vision"])["results"][0]["status"], expected)
                    parts = send.call_args.args[2]["messages"][0]["content"]
                    self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
                    if marker:
                        self.assertIn("exactly one English color word", parts[0]["text"])
                        self.assertNotIn(marker, parts[0]["text"].casefold())

    def test_transport_response_limit_and_no_proxy_auth_or_redirect(self):
        with patch.object(smoke.http.client, "HTTPConnection") as connection:
            reply = connection.return_value.getresponse.return_value
            reply.status = 302
            reply.getheaders.return_value = [("Location", "http://192.0.2.1/"), ("Content-Type", "application/json")]
            reply.read1.side_effect = io.BytesIO(b"oversized response").read1
            result = smoke.transport("http://127.0.0.1:18089", "/health", None, 1, 4)
            self.assertIn("byte limit", result["error"])
            self.assertEqual(result["raw"], b"overs")
            self.assertEqual(result["status"], 302)
            connection.assert_called_once_with("127.0.0.1", 18089, timeout=1)
            self.assertNotIn("Authorization", connection.return_value.request.call_args.kwargs["headers"])
            self.assertEqual(connection.return_value.request.call_count, 1)

    def test_total_timeout_aborts_socket_and_retains_partial_evidence(self):
        with patch.object(smoke.http.client, "HTTPConnection") as connection, patch.object(smoke.threading, "Timer") as timer:
            timer.side_effect = lambda timeout, callback: Mock(start=callback)
            reply = connection.return_value.getresponse.return_value
            reply.status = 200
            reply.getheaders.return_value = []
            reply.read1.side_effect = io.BytesIO(b"partial").read1
            result = smoke.transport("http://127.0.0.1:18089", "/health", None, 1, 100)
            self.assertEqual(result["error"], "total request timeout")
            self.assertEqual(result["raw"], b"partial")
            connection.return_value.sock.shutdown.assert_called_once()
            connection.return_value.close.assert_called_once()

    def test_request_limit_prevents_transport(self):
        with patch.object(smoke.http.client, "HTTPConnection") as connection, self.assertRaisesRegex(ValueError, "request byte limit"):
            smoke.transport("http://127.0.0.1:18089", "/v1/chat/completions", {"input": "x" * smoke.REQUEST_LIMIT}, 1, 100)
        connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
