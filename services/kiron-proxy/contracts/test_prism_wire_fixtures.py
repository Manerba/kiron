"""Replay real direct-upstream responses; never a KIron conformance claim."""

import hashlib
import json
from pathlib import Path
import unittest
from urllib.parse import urlsplit

import httpx
import openai


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prism-b10709"


class PrismWireFixtures(unittest.TestCase):
    def replay(self, name, *, strict=False, helper=False):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        self.assertEqual(openai.__version__, manifest["sdk_version"])
        fixture = manifest["fixtures"][name]
        files = {}
        for kind, record in fixture["files"].items():
            raw = (FIXTURES / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"], kind)
            files[kind] = raw
        recorded_request = json.loads(files["request.json"])
        metadata = json.loads(files["response.json"])
        expected = fixture["expected"]
        self.assertEqual(metadata["status"], 200)
        self.assertIsNone(metadata["error"])
        self.assertEqual(len(files["response.raw"]), metadata["response_bytes"])
        body = recorded_request["json"]
        is_stream = body.get("stream", False)
        if is_stream:
            lines = files["response.raw"].decode().splitlines()
            self.assertEqual(lines.count("data: [DONE]"), 1)
            self.assertEqual(next(line for line in reversed(lines) if line), "data: [DONE]")
            self.assertEqual(sum(line.startswith(":") for line in lines),
                             expected["heartbeat_comment_lines"])
        requests = []

        def respond(request):
            # No socket transport exists, even if the recorded URL is loopback.
            self.assertEqual(request.url.host, "prism-fixture.invalid")
            self.assertEqual(request.url.path, urlsplit(recorded_request["url"]).path)
            self.assertEqual(request.method, recorded_request["method"])
            self.assertEqual(request.headers["authorization"], "Bearer fixture-key")
            self.assertEqual(json.loads(request.content), body)
            requests.append(request)
            return httpx.Response(metadata["status"], headers=metadata["headers"],
                                  content=files["response.raw"])

        with openai.OpenAI(
            api_key="fixture-key", base_url="http://prism-fixture.invalid/v1",
            max_retries=0, _strict_response_validation=strict,
            http_client=httpx.Client(transport=httpx.MockTransport(respond), trust_env=False),
        ) as client:
            arguments = dict(body)
            # Native request field is passed through without changing its wire value.
            arguments["extra_body"] = {"reasoning_format": arguments.pop("reasoning_format")}
            if helper:
                self.assertTrue(arguments.pop("stream"))
                for tool in arguments.get("tools", []):
                    self.assertIs(tool["function"]["strict"], True)
                with client.chat.completions.stream(**arguments) as stream:
                    events = list(stream)
                    final = stream.get_final_completion()
                self.assertEqual(len(events), expected["helper_events"])
                done = [event for event in events
                        if event.type == "tool_calls.function.arguments.done"]
                self.assertEqual([event.parsed_arguments for event in done],
                                 [call["arguments"] for call in expected["calls"]])
                self.check_completion(final, expected, parsed=True)
            elif is_stream:
                with client.chat.completions.create(**arguments) as stream:
                    chunks = list(stream)
                self.check_chunks(chunks, expected, manifest["upstream"]["system_fingerprint"])
            else:
                final = client.chat.completions.create(**arguments)
                self.assertEqual(final.system_fingerprint, manifest["upstream"]["system_fingerprint"])
                self.check_completion(final, expected)
        self.assertEqual(len(requests), 1)

    def check_usage(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertEqual(actual.model_dump(exclude_none=True), expected)
        self.assertEqual(actual.prompt_tokens + actual.completion_tokens, actual.total_tokens)

    def check_completion(self, result, expected, *, parsed=False):
        self.assertEqual(result.id, expected["completion_id"])
        self.assertEqual(result.model, expected["model"])
        self.assertEqual(len(result.choices), 1)
        choice = result.choices[0]
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.finish_reason, expected["finish_reason"])
        self.assertEqual(choice.message.role, "assistant")
        self.assertEqual(choice.message.content or "", expected["content"])
        calls = choice.message.tool_calls or []
        self.assertEqual(len(calls), len(expected["calls"]))
        for call, wanted in zip(calls, expected["calls"]):
            self.assertEqual(call.type, "function")
            self.assertEqual(call.id, wanted["id"])
            self.assertEqual(call.function.name, wanted["name"])
            self.assertEqual(json.loads(call.function.arguments), wanted["arguments"])
            if parsed:
                self.assertEqual(call.function.parsed_arguments, wanted["arguments"])
        self.check_usage(result.usage, expected["usage"])

    def check_chunks(self, chunks, expected, fingerprint):
        self.assertEqual(len(chunks), expected["chunks"])
        self.assertEqual({chunk.id for chunk in chunks}, {expected["completion_id"]})
        self.assertEqual({chunk.model for chunk in chunks}, {expected["model"]})
        self.assertEqual({chunk.system_fingerprint for chunk in chunks}, {fingerprint})
        content, finishes, roles, indices = [], [], [], []
        calls = {}
        for chunk in chunks:
            for choice in chunk.choices:
                self.assertEqual(choice.index, 0)
                if choice.finish_reason:
                    finishes.append(choice.finish_reason)
                delta = choice.delta
                if delta.role:
                    roles.append(delta.role)
                content.append(delta.content or "")
                for fragment in delta.tool_calls or []:
                    indices.append(fragment.index)
                    call = calls.setdefault(fragment.index, {"id": None, "name": "", "arguments": ""})
                    if fragment.id:
                        self.assertIsNone(call["id"])
                        call["id"] = fragment.id
                        self.assertEqual(fragment.type, "function")
                    if fragment.function:
                        call["name"] += fragment.function.name or ""
                        call["arguments"] += fragment.function.arguments or ""
        self.assertEqual(roles, ["assistant"])
        self.assertEqual("".join(content), expected["content"])
        self.assertEqual(finishes, [expected["finish_reason"]])
        self.assertEqual(indices, expected["fragment_indices"])
        self.assertEqual(indices, sorted(indices), "native calls must form ascending contiguous blocks")
        self.assertEqual(sorted(calls), list(range(len(expected["calls"]))))
        assembled = [{**call, "arguments": json.loads(call["arguments"])}
                     for _, call in sorted(calls.items())]
        self.assertEqual(assembled, expected["calls"])
        self.assertEqual(sum(chunk.usage is not None for chunk in chunks), 1)
        self.assertEqual(chunks[-1].choices, [])
        self.check_usage(chunks[-1].usage, expected["usage"])

    def test_text_create(self):
        self.replay("text")

    def test_text_create_strict(self):
        self.replay("text", strict=True)

    def test_text_stream_create(self):
        self.replay("text_stream")

    def test_text_stream_create_strict(self):
        self.replay("text_stream", strict=True)

    def test_text_stream_helper_strict(self):
        self.replay("text_stream", strict=True, helper=True)

    def test_tool_create(self):
        self.replay("tool")

    def test_tool_create_strict(self):
        self.replay("tool", strict=True)

    def test_tool_stream_create(self):
        self.replay("tool_stream")

    def test_tool_stream_create_strict(self):
        self.replay("tool_stream", strict=True)

    def test_tool_stream_helper_strict(self):
        self.replay("tool_stream", strict=True, helper=True)

    def test_parallel_tools_stream_create(self):
        self.replay("parallel_tools_stream")

    def test_parallel_tools_stream_create_strict(self):
        self.replay("parallel_tools_stream", strict=True)

    def test_parallel_tools_stream_helper_strict(self):
        self.replay("parallel_tools_stream", strict=True, helper=True)


if __name__ == "__main__":
    unittest.main()
