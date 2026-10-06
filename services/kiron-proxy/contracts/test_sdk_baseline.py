"""Offline SDK/schema fixtures, not tests of KIron's API implementation."""

import base64
from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import struct
import unittest

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import TypeAdapter, ValidationError


HERE = Path(__file__).resolve().parent
MODEL = "fixture-local-model"
FUNCTION = {
    "name": "lookup", "description": "Client-owned fixture function",
    "parameters": {
        "type": "object", "properties": {"key": {"type": "string"}},
        "required": ["key"], "additionalProperties": False,
    },
    "strict": True,
}
CHAT_TOOL = {"type": "function", "function": FUNCTION}
RESPONSE_TOOL = {"type": "function", **FUNCTION}
CHAT_USAGE = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
RESPONSE_USAGE = {
    "input_tokens": 8, "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 1},
    "total_tokens": 12,
}


def chat(message, finish="stop"):
    return {
        "id": "chatcmpl_fixture", "object": "chat.completion", "created": 1,
        "model": MODEL, "choices": [{"index": 0, "message": message,
                                    "finish_reason": finish, "logprobs": None}],
        "usage": CHAT_USAGE,
    }


def chat_chunk(delta=None, finish=None, usage=None):
    return {
        "id": "chatcmpl_fixture", "object": "chat.completion.chunk", "created": 1,
        "model": MODEL,
        "choices": [] if delta is None else [{"index": 0, "delta": delta,
                                              "finish_reason": finish, "logprobs": None}],
        "usage": usage,
    }


def response(output=(), status="completed"):
    return {
        "id": "resp_fixture", "object": "response", "created_at": 1.0,
        "status": status, "error": None, "incomplete_details": None,
        "model": MODEL, "output": list(output), "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": [RESPONSE_TOOL],
        "usage": RESPONSE_USAGE if status == "completed" else None,
    }


def response_items():
    return [
        {"id": "rs_fixture", "type": "reasoning", "status": "completed",
         "summary": [{"type": "summary_text", "text": "Lookup needed."}]},
        {"id": "msg_fixture", "type": "message", "role": "assistant",
         "status": "completed", "content": [
             {"type": "output_text", "text": "Checking.", "annotations": [], "logprobs": []}]},
        {"id": "fc_fixture", "type": "function_call", "status": "completed",
         "call_id": "call_fixture", "name": "lookup", "arguments": '{"key":"a"}'},
    ]


def response_events():
    """An explicit valid item lifecycle, with sequence numbers across all items."""
    events = []

    def add(kind, **fields):
        events.append({"type": kind, "sequence_number": len(events), **deepcopy(fields)})

    add("response.created", response=response(status="in_progress"))
    add("response.in_progress", response=response(status="in_progress"))
    reasoning, message, call = response_items()
    for index, item in enumerate((reasoning, message, call)):
        initial = {**item, "status": "in_progress"}
        initial[{"reasoning": "summary", "message": "content", "function_call": "arguments"}[item["type"]]] = (
            "" if item["type"] == "function_call" else [])
        add("response.output_item.added", output_index=index, item=initial)
        location = {"output_index": index, "item_id": item["id"]}
        if item["type"] == "reasoning":
            location["summary_index"] = 0
            add("response.reasoning_summary_part.added", **location,
                part={"type": "summary_text", "text": ""})
            for fragment in ("Lookup ", "needed."):
                add("response.reasoning_summary_text.delta", **location, delta=fragment)
            add("response.reasoning_summary_text.done", **location, text="Lookup needed.")
            add("response.reasoning_summary_part.done", **location, part=item["summary"][0])
        elif item["type"] == "message":
            location["content_index"] = 0
            add("response.content_part.added", **location,
                part={**item["content"][0], "text": ""})
            for fragment in ("Check", "ing."):
                add("response.output_text.delta", **location, delta=fragment, logprobs=[])
            add("response.output_text.done", **location, text="Checking.", logprobs=[])
            add("response.content_part.done", **location, part=item["content"][0])
        else:
            for fragment in ('{"key":', '"a"}'):
                add("response.function_call_arguments.delta", **location, delta=fragment)
            add("response.function_call_arguments.done", **location,
                arguments=item["arguments"], name=item["name"])
        add("response.output_item.done", output_index=index, item=item)
    add("response.completed", response=response(response_items()))
    return events


def sse(items, *, responses=False):
    frames = [
        (f'event: {item["type"]}\n' if responses else "") + f"data: {json.dumps(item)}\n\n"
        for item in items
    ]
    if not responses:
        frames.append("data: [DONE]\n\n")
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                          content="".join(frames).encode())


class SDKBaseline(unittest.TestCase):
    def client(self, replies, *, strict=True):
        """Record requests; MockTransport makes external network access impossible."""
        pending = iter(replies)
        requests = []

        def handle(request):
            self.assertEqual(request.url.host, "sdk-fixture.invalid")
            self.assertEqual(request.headers["authorization"], "Bearer fixture-key")
            if request.content:
                self.assertEqual(request.headers["content-type"], "application/json")
            requests.append(request)
            reply = next(pending)
            return reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)

        client = openai.OpenAI(
            api_key="fixture-key", base_url="http://sdk-fixture.invalid/v1",
            max_retries=0, _strict_response_validation=strict,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        self.addCleanup(client.close)
        return client, requests

    def test_snapshot_and_dependency_lock(self):
        snapshot = json.loads((HERE / "openai-sdk-snapshot.json").read_text())
        self.assertEqual(openai.__version__, snapshot["sdk_version"])
        package = Path(openai.__file__).parent
        for path, expected in snapshot["source_sha256"].items():
            with self.subTest(source=path):
                self.assertEqual(hashlib.sha256((package / path).read_bytes()).hexdigest(), expected)
        for line in (HERE.parent / "requirements-contract.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                package_name, version = line.split("==")
                with self.subTest(dependency=package_name):
                    self.assertEqual(metadata.version(package_name), version)

    def test_models_list_and_retrieve(self):
        model = {"id": MODEL, "object": "model", "created": 1, "owned_by": "kiron"}
        client, requests = self.client([{"object": "list", "data": [model]}, model])
        self.assertEqual([item.id for item in client.models.list()], [MODEL])
        self.assertEqual(client.models.retrieve(MODEL).model_dump(), model)
        self.assertEqual([r.url.path for r in requests], ["/v1/models", f"/v1/models/{MODEL}"])

    def test_strict_schema_and_json_content_type_rejection(self):
        invalid_model = {"id": MODEL, "object": "model", "created": 1}  # missing owned_by
        client, _ = self.client([
            invalid_model,
            httpx.Response(200, headers={"Content-Type": "text/plain"}, content="not-json"),
        ])
        for _ in range(2):
            with self.assertRaises(openai.APIResponseValidationError):
                client.models.retrieve(MODEL)
        with self.assertRaises(ValidationError):
            TypeAdapter(ResponseStreamEvent).validate_python({
                "type": "response.output_text.delta", "content_index": 0,
                "item_id": "msg_fixture", "output_index": 0, "sequence_number": 0,
                "delta": "missing required logprobs",
            })

    def test_chat_tool_roundtrip(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_fixture", "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"a"}'},
        }]}
        payload = chat(message, "tool_calls")
        ChatCompletion.model_validate(payload, strict=True)
        client, requests = self.client([payload, chat({"role": "assistant", "content": "Found."})])
        result = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Find a"}],
                                                tools=[CHAT_TOOL], tool_choice="required")
        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        call = result.choices[0].message.tool_calls[0]
        self.assertEqual(json.loads(call.function.arguments), {"key": "a"})
        answer = client.chat.completions.create(model=MODEL, messages=[
            {"role": "user", "content": "Find a"},
            result.choices[0].message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "Found."},
        ])
        self.assertEqual(answer.choices[0].message.content, "Found.")
        self.assertEqual(json.loads(requests[1].content)["messages"][-1]["tool_call_id"], call.id)

    def test_chat_interleaved_tool_stream_and_usage(self):
        # Indices are introduced in order, but argument fragments alternate calls.
        chunks = [chat_chunk({"role": "assistant", "content": None, "tool_calls": [
            {"index": index, "id": f"call_{index}", "type": "function",
             "function": {"name": "lookup", "arguments": ""}} for index in range(2)
        ]})]
        for index, fragment in [(0, '{"key":'), (1, '{"key":'), (0, '"a"}'), (1, '"b"}')]:
            chunks.append(chat_chunk({"tool_calls": [
                {"index": index, "function": {"arguments": fragment}},
            ]}))
        chunks.extend([chat_chunk({}, "tool_calls"), chat_chunk(usage=CHAT_USAGE)])
        for chunk in chunks:
            ChatCompletionChunk.model_validate(chunk, strict=True)
        client, requests = self.client([sse(chunks), sse(chunks)])
        raw = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Find a and b"}],
            tools=[CHAT_TOOL], parallel_tool_calls=True, stream=True,
            stream_options={"include_usage": True},
        )
        with raw:
            parsed_chunks = list(raw)
        self.assertEqual(len(parsed_chunks), len(chunks))
        self.assertEqual(parsed_chunks[-1].usage.total_tokens, 12)
        for index, key in enumerate(("a", "b")):
            fragments = [
                call.function.arguments for chunk in parsed_chunks for choice in chunk.choices
                for call in choice.delta.tool_calls or [] if call.index == index
            ]
            self.assertEqual(json.loads("".join(fragments)), {"key": key})
        # Ordinary tools avoid auto-parsing partial JSON on interleaved done callbacks.
        with client.chat.completions.stream(
            model=MODEL, messages=[{"role": "user", "content": "Find a and b"}],
            tools=[{"type": "function", "function": {**FUNCTION, "strict": False}}],
            parallel_tool_calls=True,
            stream_options={"include_usage": True},
        ) as stream:
            result = stream.get_final_completion()
        calls = result.choices[0].message.tool_calls
        self.assertEqual([call.id for call in calls], ["call_0", "call_1"])
        self.assertEqual([json.loads(call.function.arguments) for call in calls], [{"key": "a"}, {"key": "b"}])
        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertTrue(json.loads(requests[0].content)["stream_options"]["include_usage"])

    def test_chat_strict_parallel_tool_stream_in_public_emission_order(self):
        # Public emission finishes each call's argument block before the next
        # index appears, allowing the SDK's ordinary strict auto-parser to work.
        chunks = [chat_chunk({"role": "assistant", "content": None})]
        for index, key in enumerate(("a", "b")):
            chunks.append(chat_chunk({"tool_calls": [{
                "index": index, "id": f"call_{index}", "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            }]}))
            for fragment in ('{"key":', json.dumps(key), '}'):
                chunks.append(chat_chunk({"tool_calls": [
                    {"index": index, "function": {"arguments": fragment}},
                ]}))
        chunks.extend([chat_chunk({}, "tool_calls"), chat_chunk(usage=CHAT_USAGE)])
        for chunk in chunks:
            ChatCompletionChunk.model_validate(chunk, strict=True)
        client, requests = self.client([sse(chunks)])
        with client.chat.completions.stream(
            model=MODEL, messages=[{"role": "user", "content": "Find a and b"}],
            tools=[CHAT_TOOL], parallel_tool_calls=True,
            stream_options={"include_usage": True},
        ) as stream:
            events = list(stream)
            result = stream.get_final_completion()
        calls = result.choices[0].message.tool_calls
        self.assertEqual([call.id for call in calls], ["call_0", "call_1"])
        self.assertEqual([call.function.parsed_arguments for call in calls], [{"key": "a"}, {"key": "b"}])
        done = [event for event in events if event.type == "tool_calls.function.arguments.done"]
        self.assertEqual([event.parsed_arguments for event in done], [{"key": "a"}, {"key": "b"}])
        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        self.assertEqual(result.usage.total_tokens, 12)
        body = json.loads(requests[0].content)
        self.assertTrue(body["tools"][0]["function"]["strict"])
        self.assertTrue(body["parallel_tool_calls"])

    def test_responses_full_items_and_stateless_tool_roundtrip(self):
        payload = response(response_items())
        Response.model_validate(payload, strict=True)
        client, requests = self.client([payload, response([response_items()[1]])])
        first = client.responses.create(model=MODEL, input="Find a", store=False, tools=[RESPONSE_TOOL])
        self.assertEqual(first.output_text, "Checking.")
        self.assertEqual(first.output[0].summary[0].text, "Lookup needed.")
        self.assertEqual(first.usage.output_tokens_details.reasoning_tokens, 1)
        call = first.output[2]
        client.responses.create(model=MODEL, store=False, input=[
            {"role": "user", "content": "Find a"},
            *[item.model_dump(exclude_none=True) for item in first.output],
            {"type": "function_call_output", "call_id": call.call_id, "output": "Found."},
        ])
        body = json.loads(requests[1].content)
        self.assertFalse(body["store"])
        self.assertNotIn("previous_response_id", body)
        self.assertEqual(body["input"][-1]["call_id"], "call_fixture")

    def test_responses_stream_assembly_and_complete_reasoning(self):
        events = response_events()
        adapter = TypeAdapter(ResponseStreamEvent)
        for event in events:
            adapter.validate_python(event, strict=True)
        client, requests = self.client([sse(events, responses=True)])
        with client.responses.stream(model=MODEL, input="Find a", store=False) as stream:
            observed = list(stream)
            final = stream.get_final_response()
        self.assertEqual([event.sequence_number for event in observed], list(range(len(events))))
        text_deltas = [event for event in observed if event.type == "response.output_text.delta"]
        tool_deltas = [event for event in observed if event.type == "response.function_call_arguments.delta"]
        self.assertEqual(text_deltas[-1].snapshot, "Checking.")
        self.assertEqual(tool_deltas[-1].snapshot, '{"key":"a"}')
        summary = "".join(event.delta for event in observed if event.type == "response.reasoning_summary_text.delta")
        self.assertEqual(final.output[0].summary[0].text, summary)
        self.assertEqual(final.output_text, "Checking.")
        self.assertEqual(json.loads(final.output[2].arguments), {"key": "a"})
        self.assertEqual(final.usage.total_tokens, 12)
        self.assertEqual(requests[0].url.path, "/v1/responses")

    def test_embeddings_float_strict(self):
        payload = self.embedding_payload([0.25, -0.5])
        client, requests = self.client([payload])
        result = client.embeddings.create(model=MODEL, input=["a"], encoding_format="float", dimensions=2)
        self.assertEqual(result.data[0].embedding, [0.25, -0.5])
        self.assertEqual(result.usage.total_tokens, 1)
        self.assertEqual(json.loads(requests[0].content)["dimensions"], 2)

    @staticmethod
    def embedding_payload(vector):
        return {"object": "list", "model": MODEL, "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "data": [{"object": "embedding", "index": 0, "embedding": vector}]}

    def test_embeddings_base64_and_sdk_default(self):
        encoded = base64.b64encode(struct.pack("<2f", 0.25, -0.5)).decode("ascii")
        payload = self.embedding_payload(encoded)
        client, requests = self.client([payload, payload], strict=False)
        explicit = client.embeddings.create(model=MODEL, input="a", encoding_format="base64")
        self.assertEqual(explicit.data[0].embedding, encoded)
        self.assertEqual(struct.unpack("<2f", base64.b64decode(encoded, validate=True)), (0.25, -0.5))
        default = client.embeddings.create(model=MODEL, input="a")
        self.assertEqual(default.data[0].embedding, [0.25, -0.5])
        self.assertEqual([json.loads(r.content)["encoding_format"] for r in requests], ["base64", "base64"])

    def test_embeddings_base64_strict_sdk_schema_limitation(self):
        encoded = base64.b64encode(struct.pack("<2f", 0.25, -0.5)).decode("ascii")
        client, _ = self.client([self.embedding_payload(encoded)])
        with self.assertRaises(openai.APIResponseValidationError):
            client.embeddings.create(model=MODEL, input="a", encoding_format="base64")

    def test_standard_error_envelope_and_status_mapping(self):
        for status, exception in [(400, openai.BadRequestError), (401, openai.AuthenticationError),
                                  (404, openai.NotFoundError), (503, openai.InternalServerError)]:
            with self.subTest(status=status):
                error = {"message": "Fixture failure", "type": "invalid_request_error",
                         "param": "model", "code": "fixture_error"}
                client, _ = self.client([httpx.Response(status, json={"error": error})])
                with self.assertRaises(exception) as caught:
                    client.models.retrieve(MODEL)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.body, error)
                self.assertEqual(caught.exception.code, "fixture_error")


if __name__ == "__main__":
    unittest.main()
