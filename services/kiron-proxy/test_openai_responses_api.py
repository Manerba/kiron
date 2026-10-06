"""Responses shares authentication, cancellation and the canonical runtime."""
import asyncio
from dataclasses import replace
import json
from unittest import mock

from kiron_common.local_inference import (
    ErrorCode, EventKind, FinishReason, InferenceEvent, InferenceResult, MessageRole, TextPart, TokenUsage,
)
from kiron_common.model_catalog import BackendType
import openai_api
from test_openai_runtime_api import RuntimeApiFixture, FakeOperation, failure, resolved_model


REQUEST = {"model": "alias", "input": "Hello"}


def frames(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]


class ResponseOperation(FakeOperation):
    async def events(self):
        async for event in super().events():
            if event.kind is EventKind.USAGE and self.runtime.complete_usage:
                event = replace(event, usage=TokenUsage(3, 2, 0, 0))
            yield event


class ResponsesApiTests(RuntimeApiFixture):
    async def asyncSetUp(self):
        self.runtime.complete_usage = True
        chat = self.runtime.chat
        async def complete(request):
            result = await chat(request)
            return replace(result, usage=TokenUsage(3, 2, 0, 0)) if self.runtime.complete_usage else result
        async def validate(parsed, model, context):
            self.runtime.check("validate_response", context)
            return parsed.to_request(model, context, 64)
        async def prepare(request, *, streaming):
            self.runtime.check("prepare", request.context)
            self.assertTrue(streaming)
            self.runtime.requests.append(request)
            operation = ResponseOperation(self.runtime, request)
            self.runtime.operations.append(operation)
            return operation
        self.runtime.chat, self.runtime.validate_response, self.runtime.prepare = complete, validate, prepare

    async def test_nonstream_uses_same_runtime_and_canonical_model_with_exact_usage(self):
        for provider in (BackendType.OLLAMA, BackendType.PRISM):
            with self.subTest(provider=provider):
                self.runtime.calls.clear()
                self.runtime.model = resolved_model(provider)
                response = await self.client.post("/v1/responses", json={**REQUEST, "instructions": "Be concise"})
                self.assertEqual(response.status_code, 200, response.text)
                value, request = response.json(), self.runtime.requests[-1]
                self.assertEqual(self.runtime.calls, ["resolve", "validate_response", "public_model_for", "chat"])
                self.assertEqual((value["object"], value["model"], value["status"]), ("response", "canonical", "completed"))
                self.assertEqual(value["output"][0]["content"][0]["text"], "Hello")
                self.assertEqual(value["usage"], {"input_tokens":3, "output_tokens":2, "total_tokens":5,
                    "input_tokens_details":{"cached_tokens":0}, "output_tokens_details":{"reasoning_tokens":0}})
                self.assertIs(request.messages[0].role, MessageRole.DEVELOPER)
                self.assertEqual(request.messages[0].content, (TextPart("Be concise"),))
                self.assertEqual(response.headers["x-request-id"], request.context.request_id)
                self.assertEqual(self.records.updates[-1][1]["tokens_generated"], 2)

    async def test_context_and_timeout_have_exact_json_and_sdk_stream_failure_classes(self):
        for code, status, public, response_code in (
            (ErrorCode.CONTEXT_LENGTH_EXCEEDED, 400, "context_length_exceeded", "invalid_prompt"),
            (ErrorCode.TIMEOUT, 504, "timeout", "server_error"),
        ):
            with self.subTest(code=code):
                self.runtime.errors["chat"] = failure(code)
                response = await self.client.post("/v1/responses", json=REQUEST)
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(response.json()["error"]["code"], public)
                self.assertNotIn("secret", response.text)
                self.runtime.errors.pop("chat")
                self.runtime.errors["events"] = failure(code)
                response = await self.client.post("/v1/responses", json={**REQUEST, "stream": True})
                values = frames(response)
                self.assertEqual(values[-1]["type"], "response.failed")
                self.assertEqual(values[-1]["response"]["error"]["code"], response_code)
                self.assertEqual(values[-1]["response"]["output"][0]["content"][0]["text"], "Hello")
                self.assertIsNone(values[-1]["response"]["usage"])
                self.assertNotIn("secret", response.text)
                self.assertNotIn("[DONE]", response.text)
                self.assertFalse(any(v["type"] in {"response.completed", "response.incomplete"} for v in values))
                self.runtime.errors.pop("events")

    async def test_stream_has_one_aggregated_terminal_sequential_events_and_no_chat_done(self):
        response = await self.client.post("/v1/responses", json={**REQUEST, "stream":True})
        self.assertEqual(response.status_code, 200, response.text)
        events = frames(response)
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))
        self.assertEqual([event["type"] for event in events[:2]], ["response.created", "response.in_progress"])
        self.assertEqual([event["type"] for event in events if event["type"] in
                         {"response.completed", "response.failed", "response.incomplete"}], ["response.completed"])
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "Hello")
        self.assertNotIn("[DONE]", response.text)
        self.assertEqual(self.runtime.operations[-1].closed, 1)
        self.assertEqual(self.records.updates[-1][1]["tokens_generated"], 2)

    async def test_missing_usage_details_are_protocol_errors_never_invented_zero(self):
        self.runtime.complete_usage = False
        response = await self.client.post("/v1/responses", json=REQUEST)
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["error"]["code"], "backend_protocol_error")
        response = await self.client.post("/v1/responses", json={**REQUEST, "stream":True})
        events = frames(response)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.completed", [event["type"] for event in events])
        self.assertNotIn("[DONE]", response.text)

    async def test_completed_is_not_published_before_clean_source_eof(self):
        original = ResponseOperation.events
        for mode in ("trailing_event", "trailing_exception"):
            async def broken(operation):
                async for event in original(operation):
                    yield event
                if mode == "trailing_event":
                    yield InferenceEvent(EventKind.TEXT_DELTA,operation.request.context.request_id,text="late", output_item_index=0, part_index=0)
                else:
                    raise RuntimeError("private provider failure")
            with self.subTest(mode=mode), mock.patch.object(ResponseOperation,"events",broken):
                response = await self.client.post("/v1/responses",json={**REQUEST,"stream":True})
                events = frames(response)
                self.assertEqual(events[-1]["type"],"response.failed")
                self.assertNotIn("response.completed",[event["type"] for event in events])
                self.assertNotIn("private",response.text)
                self.assertNotIn("[DONE]",response.text)
                self.assertEqual(self.runtime.operations[-1].closed,1)

    async def test_runtime_error_before_headers_stays_json_and_omits_private_details(self):
        self.runtime.errors["validate_response"] = failure(ErrorCode.UNSUPPORTED_CAPABILITY)
        response = await self.client.post("/v1/responses", json={**REQUEST, "stream":True})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], "unsupported_capability")
        self.assertNotIn("private", response.text)
        self.assertEqual(self.runtime.calls, ["resolve", "validate_response"])

    async def test_image_errors_refer_to_public_response_input_paths(self):
        from openai_vision import VisionError
        body = {**REQUEST,"instructions":"Be concise","input":[{"role":"user","content":[
            {"type":"input_image","image_url":"data:image/png;base64,eA==","detail":"low"}]}]}
        with mock.patch.object(openai_api,"decode_chat_images",side_effect=VisionError(
                "invalid_request","messages[1].content[0].image_url.url",400)):
            response = await self.client.post("/v1/responses",json=body)
        self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(response.json()["error"]["param"],"input[0].content[0].image_url")
        self.assertEqual(self.runtime.calls,[])

    async def test_stateless_cloud_fields_fail_before_resolver_and_auth_wraps_missing_routes(self):
        for field, value in (("previous_response_id","resp_old"), ("conversation","conv_old"),
                              ("background",True), ("store",True), ("include",["encrypted_content"])):
            with self.subTest(field=field):
                response = await self.client.post("/v1/responses", json={**REQUEST, field:value})
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.runtime.calls, [])
        for method, path in (("POST","/v1/responses"), ("GET","/v1/responses/id"), ("DELETE","/v1/responses/id")):
            response = await self.client.request(method, path, headers={"Authorization":"Bearer wrong"})
            self.assertEqual(response.status_code, 401, response.text)
        response = await self.client.get("/v1/responses/id")
        self.assertEqual(response.status_code, 404, response.text)

    async def test_deadline_after_headers_uses_responses_failure_and_closes_operation(self):
        self.runtime.pause = True
        with mock.patch.object(openai_api,"REQUEST_TIMEOUT",.03):
            response = await self.client.post("/v1/responses", json={**REQUEST,"stream":True})
        self.assertEqual(response.status_code,200,response.text)
        events = frames(response)
        self.assertEqual(events[-1]["type"],"response.failed")
        self.assertEqual(events[-1]["response"]["status"],"failed")
        self.assertEqual([event["sequence_number"] for event in events],list(range(len(events))))
        self.assertNotIn("[DONE]",response.text)
        self.assertEqual(self.runtime.operations[-1].closed,1)

    async def raw_request(self, *, send_failure=False, disconnect=False):
        messages = asyncio.Queue()
        await messages.put({"type":"http.request","body":json.dumps({**REQUEST,"stream":True}).encode(),"more_body":False})
        sent = []
        async def send(message):
            if send_failure:
                raise OSError("closed socket")
            sent.append(message)
        scope = {"type":"http","asgi":{"version":"3.0","spec_version":"2.4"},"http_version":"1.1",
            "method":"POST","scheme":"http","path":"/v1/responses","raw_path":b"/v1/responses",
            "query_string":b"","root_path":"","client":("test",1),"server":("test",80),
            "headers":[(b"authorization",b"Bearer test-key"),(b"content-type",b"application/json")]}
        task = asyncio.create_task(self.app(scope,messages.get,send))
        if disconnect:
            await asyncio.wait_for(self.runtime.started.wait(),1)
            await messages.put({"type":"http.disconnect"})
        await asyncio.wait_for(task,1)
        return sent

    async def test_header_send_failure_and_disconnect_close_without_fabricated_terminal(self):
        sent = await self.raw_request(send_failure=True)
        self.assertEqual(sent,[])
        self.assertEqual(self.runtime.operations[-1].closed,1)
        self.assertNotIn("events",self.runtime.calls)
        self.runtime.pause = True
        sent = await self.raw_request(disconnect=True)
        output = b"".join(message.get("body",b"") for message in sent)
        self.assertNotIn(b"response.failed",output)
        self.assertNotIn(b"response.completed",output)
        self.assertNotIn(b"[DONE]",output)
        self.assertEqual(self.runtime.operations[-1].closed,1)

    async def test_resistant_stream_cannot_delay_timeout_or_write_after_failure(self):
        release, exited = asyncio.Event(), asyncio.Event()
        late_tasks = []
        async def resistant(operation):
            late_tasks.append(asyncio.current_task())
            rid = operation.request.context.request_id
            self.runtime.started.set()
            yield InferenceEvent(EventKind.STARTED,rid)
            yield InferenceEvent(EventKind.TEXT_DELTA,rid,text="partial", output_item_index=0, part_index=0)
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                yield InferenceEvent(EventKind.TEXT_DELTA,rid,text="late", output_item_index=0, part_index=0)
                yield InferenceEvent(EventKind.USAGE,rid,usage=TokenUsage(3,2,0,0))
                yield InferenceEvent(EventKind.COMPLETED,rid,finish_reason=FinishReason.STOP)
            finally:
                exited.set()
        try:
            with mock.patch.object(ResponseOperation,"events",resistant), \
                    mock.patch.object(openai_api,"REQUEST_TIMEOUT",.05), mock.patch.object(openai_api,"CANCEL_WAIT",.01):
                sent = await self.raw_request()
                output = b"".join(message.get("body",b"") for message in sent)
                events = [json.loads(line[6:]) for line in output.splitlines() if line.startswith(b"data: {")]
                self.assertEqual(events[-1]["type"],"response.failed")
                self.assertEqual([event["sequence_number"] for event in events],list(range(len(events))))
                self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"],"partial")
                self.assertNotIn(b"[DONE]",output)
                sent_count = len(sent)
                release.set()
                await asyncio.wait_for(exited.wait(),.3)
                await asyncio.sleep(0)
                self.assertEqual(len(sent),sent_count)
                self.assertTrue(late_tasks[0].done())
        finally:
            release.set()

    async def test_resistant_stream_disconnect_returns_without_terminal_or_late_sends(self):
        release, exited = asyncio.Event(), asyncio.Event()
        async def resistant(operation):
            self.runtime.started.set()
            rid = operation.request.context.request_id
            yield InferenceEvent(EventKind.STARTED,rid)
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                yield InferenceEvent(EventKind.TEXT_DELTA,rid,text="late", output_item_index=0, part_index=0)
            finally:
                exited.set()
        try:
            with mock.patch.object(ResponseOperation,"events",resistant), mock.patch.object(openai_api,"CANCEL_WAIT",.01):
                sent = await self.raw_request(disconnect=True)
                output = b"".join(message.get("body",b"") for message in sent)
                self.assertNotIn(b"response.failed",output)
                self.assertNotIn(b"response.completed",output)
                self.assertNotIn(b"[DONE]",output)
                sent_count = len(sent)
                release.set()
                await asyncio.wait_for(exited.wait(),.3)
                await asyncio.sleep(0)
                self.assertEqual(len(sent),sent_count)
        finally:
            release.set()

    async def test_resistant_logger_does_not_hold_responses_stream_or_change_terminal(self):
        release, exited = asyncio.Event(), asyncio.Event()
        async def blocked(record):
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
            finally:
                exited.set()
        self.records.add_request = blocked
        try:
            with mock.patch.object(openai_api,"LOG_TIMEOUT",.01):
                response = await asyncio.wait_for(self.client.post("/v1/responses",json={**REQUEST,"stream":True}),.5)
            events = frames(response)
            self.assertEqual(events[-1]["type"],"response.completed")
            self.assertEqual(sum(event["type"]=="response.completed" for event in events),1)
            self.assertNotIn("[DONE]",response.text)
            release.set()
            await asyncio.wait_for(exited.wait(),.3)
        finally:
            release.set()
