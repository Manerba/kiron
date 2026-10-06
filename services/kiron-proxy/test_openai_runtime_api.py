"""Provider-neutral HTTP/ASGI contract with canonical fake runtime values."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openai_api
from kiron_common.local_inference import (
    ArtifactIdentity, ErrorCode, EventKind, FinishReason, InferenceEvent, InferenceResult,
    LocalInferenceError, ResolvedDeployment, ResolvedModel, ResourceProfile, RuntimeFailure,
    TextPart, TokenUsage,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType, ModelTask


CHAT = {"model": "alias", "messages": [{"role": "user", "content": "Hi"}]}


def failure(code):
    return LocalInferenceError(RuntimeFailure(code, "secret /private/model/path token=secret", "temperature"))


def resolved_model(provider=BackendType.PRISM):
    deployment = ResolvedDeployment("deployment", provider, "/private/model.gguf",
        ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100), LoaderType.PRISM_GGUF,
        ResourceProfile("measured", 1024, 128, 128, 1, 4, 40, False, 40, 20), "b" * 64)
    return ResolvedModel("alias", deployment, None, None, "c" * 64, task=ModelTask.CHAT,
                         created=1234567890, canonical_model_id="canonical")


class Records:
    def __init__(self):
        self.records, self.updates = [], []
        self.fail = False

    async def add_request(self, record):
        if self.fail:
            raise RuntimeError("logging unavailable")
        self.records.append(record)

    async def update_request(self, identifier, **kwargs):
        if self.fail:
            raise RuntimeError("logging unavailable")
        self.updates.append((identifier, kwargs))


class Keys:
    error = False
    def validate_key(self, token):
        if self.error:
            raise RuntimeError("secret database error")
        return {"id": "key-id"} if token == "test-key" else None


class FakeOperation:
    def __init__(self, runtime, request):
        self.runtime, self.request, self.closed = runtime, request, 0

    async def events(self):
        self.runtime.calls.append("events")
        self.runtime.started.set()
        rid = self.request.context.request_id
        if self.runtime.pause:
            await asyncio.Event().wait()
        yield InferenceEvent(EventKind.STARTED, rid)
        yield InferenceEvent(EventKind.TEXT_DELTA, rid, text="Hello", output_item_index=0, part_index=0)
        if "events" in self.runtime.errors:
            raise self.runtime.errors["events"]
        yield InferenceEvent(EventKind.USAGE, rid, usage=TokenUsage(3, 2))
        yield InferenceEvent(EventKind.COMPLETED, rid, finish_reason=FinishReason.STOP)
        if self.runtime.trailing:
            yield InferenceEvent(EventKind.TEXT_DELTA, rid, text="invalid trailing data", output_item_index=0, part_index=0)

    async def close(self):
        self.closed = 1


class FakeRuntime:
    def __init__(self, provider=BackendType.PRISM):
        self.model = resolved_model(provider)
        self.calls, self.requests, self.operations = [], [], []
        self.errors, self.contexts = {}, []
        self.started = asyncio.Event()
        self.pause = self.trailing = self.chat_closed = False

    def check(self, name, context):
        self.calls.append(name)
        self.contexts.append(context)
        if name in self.errors:
            raise self.errors[name]

    def public(self):
        return {"id": "canonical", "object": "model", "created": self.model.created,
                "owned_by": self.model.deployment.provider.value}

    async def public_models(self, context):
        self.check("public_models", context)
        return [self.public()]

    async def public_model(self, name, context):
        self.check("public_model", context)
        if name not in {"alias", "canonical"}:
            raise failure(ErrorCode.MODEL_NOT_FOUND)
        return self.model, self.public()

    async def resolve(self, name, context=None):
        self.calls.append("resolve")
        if name not in {"alias", "canonical"}:
            raise failure(ErrorCode.MODEL_NOT_FOUND)
        return self.model

    async def public_model_for(self, model, context):
        self.check("public_model_for", context)
        assert model is self.model
        return self.public()

    async def validate_chat(self, parsed, model, context):
        self.check("validate_chat", context)
        return parsed.to_request(model, context, 64)

    async def chat(self, request):
        self.check("chat", request.context)
        self.requests.append(request)
        self.started.set()
        try:
            if self.pause:
                await asyncio.Event().wait()
            return InferenceResult(request.context.request_id, (TextPart("Hello"),), (), (), TokenUsage(3, 2), FinishReason.STOP)
        finally:
            self.chat_closed = True

    async def prepare(self, request, *, streaming):
        self.check("prepare", request.context)
        assert streaming
        self.requests.append(request)
        operation = FakeOperation(self, request)
        self.operations.append(operation)
        return operation


class RuntimeApiFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.records, self.keys, self.runtime = Records(), Keys(), FakeRuntime()
        self.app = openai_api.create_openai_api_app(self.records, self.keys)
        self.app.state.local_inference = self.runtime
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://api.test",
                                       headers={"Authorization": "Bearer test-key"})
        self.addAsyncCleanup(self.client.aclose)


class RuntimeApiTests(RuntimeApiFixture):
    async def test_two_providers_use_identical_canonical_request_and_wire_contract(self):
        for provider in (BackendType.PRISM, BackendType.OLLAMA):
            with self.subTest(provider=provider):
                self.runtime.model = resolved_model(provider)
                response = await self.client.post("/v1/chat/completions", json=CHAT)
                self.assertEqual(response.status_code, 200, response.text)
                value, request = response.json(), self.runtime.requests[-1]
                self.assertEqual(value["model"], "canonical")
                self.assertEqual(value["choices"][0]["message"], {"role":"assistant", "content":"Hello"})
                self.assertEqual(value["usage"], {"prompt_tokens":3, "completion_tokens":2, "total_tokens":5})
                self.assertEqual(request.model.deployment.provider, provider)
                self.assertEqual(request.messages[0].content, (TextPart("Hi"),))
                self.assertIsNone(request.execution_generation)
                self.assertEqual(response.headers["x-request-id"], request.context.request_id)
                self.assertEqual(self.records.updates[-1][1]["state"], "completed")
                self.assertEqual(self.records.updates[-1][1]["tokens_generated"], 2)

    async def test_discovery_detail_alias_and_creation_are_stable_and_do_not_infer(self):
        listing = await self.client.get("/v1/models")
        detail = await self.client.get("/v1/models/alias")
        again = await self.client.get("/v1/models/canonical")
        self.assertEqual(listing.json(), {"object":"list", "data":[detail.json()]})
        self.assertEqual(detail.json(), again.json())
        self.assertEqual(set(detail.json()), {"id", "object", "created", "owned_by"})
        self.assertEqual(self.runtime.calls, ["public_models", "public_model", "public_model"])

    async def test_auth_surrounds_models_chat_unknown_routes_and_wrong_methods(self):
        for method, path in [("GET","/v1/models"), ("GET","/v1/models/alias"), ("POST","/v1/chat/completions"),
                             ("GET","/missing"), ("DELETE","/v1/models"), ("OPTIONS","/v1/models")]:
            with self.subTest(method=method,path=path):
                result = await self.client.request(method, path, headers={"Authorization":"Bearer wrong"})
                self.assertEqual(result.status_code, 401)
                self.assertEqual(set(result.json()["error"]), {"message","type","param","code"})
                self.assertEqual(result.headers["www-authenticate"], 'Bearer realm="api"')
                self.assertIn("x-request-id", result.headers)
        self.assertEqual(self.runtime.calls, [])

    async def test_authenticated_404_405_are_json_and_slashes_do_not_redirect(self):
        for method,path,status,code in [("GET","/missing",404,"endpoint_not_found"),
            ("GET","/v1/chat/completions",405,"method_not_allowed"),
            ("POST","/v1/models",405,"method_not_allowed"),
            ("POST","/v1/chat/completions/",404,"endpoint_not_found")]:
            with self.subTest(method=method,path=path):
                result = await self.client.request(method,path)
                self.assertEqual(result.status_code,status)
                self.assertEqual(result.json()["error"]["code"],code)
                self.assertIn("x-request-id",result.headers)
        head = await self.client.head("/v1/models")
        self.assertEqual(head.status_code,405)
        self.assertEqual(self.runtime.calls,[])

    async def test_runtime_errors_are_mapped_before_stream_and_do_not_leak_details(self):
        cases = [(ErrorCode.MODEL_NOT_FOUND,404,"model_not_found"),
                 (ErrorCode.PROVIDER_UNAVAILABLE,503,"provider_unavailable"),
                 (ErrorCode.CONFLICT,503,"resource_busy"), (ErrorCode.TIMEOUT,504,"timeout"),
                 (ErrorCode.CONTEXT_LENGTH_EXCEEDED,400,"context_length_exceeded"),
                 (ErrorCode.PROVIDER_ERROR,502,"backend_protocol_error"),
                 (ErrorCode.UNSUPPORTED_VALUE,400,"unsupported_value")]
        for code,status,public in cases:
            with self.subTest(code=code):
                self.runtime.errors["validate_chat"] = failure(code)
                result = await self.client.post("/v1/chat/completions",json={**CHAT,"stream":True})
                self.assertEqual(result.status_code,status,result.text)
                self.assertEqual(result.json()["error"]["code"],public)
                self.assertNotIn("secret",result.text)
                self.assertNotIn("private",result.text)
        self.assertNotIn("prepare",self.runtime.calls)
        self.assertNotIn("public_model_for",self.runtime.calls)

    async def test_admission_overload_has_retry_header_and_safe_runtime_parameter(self):
        self.runtime.errors["prepare"] = LocalInferenceError(RuntimeFailure(
            ErrorCode.OVERLOADED,"secret backend path","/private/secret"))
        response = await self.client.post("/v1/chat/completions",json={**CHAT,"stream":True})
        self.assertEqual(response.status_code,429)
        self.assertEqual(response.headers["retry-after"],"1")
        self.assertEqual(response.json()["error"]["type"],"rate_limit_error")
        self.assertEqual(response.json()["error"]["code"],"overloaded")
        self.assertIsNone(response.json()["error"]["param"])
        self.assertNotIn("secret",response.text)

    async def test_auth_backend_error_and_unbound_service_are_503(self):
        self.keys.error = True
        result = await self.client.get("/v1/models")
        self.assertEqual(result.status_code,503)
        self.assertEqual(result.json()["error"]["code"],"auth_backend_unavailable")
        self.assertNotIn("secret",result.text)
        self.keys.error = False
        del self.app.state.local_inference
        result = await self.client.get("/v1/models")
        self.assertEqual(result.status_code,503)
        self.assertEqual(result.json()["error"]["code"],"provider_unavailable")

    async def test_stream_identity_usage_done_and_operation_cleanup(self):
        response = await self.client.post("/v1/chat/completions",json={**CHAT,"stream":True,"stream_options":{"include_usage":True}})
        self.assertEqual(response.status_code,200,response.text)
        lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1],"[DONE]")
        chunks = [json.loads(value) for value in lines[:-1]]
        self.assertEqual(len({(v["id"],v["model"],v["created"]) for v in chunks}),1)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"],"assistant")
        self.assertEqual(chunks[-1]["usage"]["total_tokens"],5)
        self.assertEqual(self.runtime.operations[0].closed,1)
        self.assertEqual(response.headers["cache-control"],"no-cache")
        self.assertEqual(response.headers["x-accel-buffering"],"no")

    async def test_poststream_failure_and_trailing_data_never_emit_success_finish(self):
        for trailing in (False,True):
            with self.subTest(trailing=trailing):
                self.runtime.trailing = trailing
                self.runtime.errors = {} if trailing else {"events":failure(ErrorCode.PROVIDER_ERROR)}
                response = await self.client.post("/v1/chat/completions",json={**CHAT,"stream":True})
                self.assertEqual(response.status_code,200)
                frames = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith("data: {")]
                self.assertTrue(any("error" in v for v in frames))
                self.assertFalse(any(c.get("finish_reason") is not None for v in frames for c in v.get("choices",[])))
                self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
                self.assertEqual(self.runtime.operations[-1].closed,1)

    async def test_logging_failure_cannot_mask_success_or_error_and_no_auth_is_recorded(self):
        self.records.fail = True
        result = await self.client.post("/v1/chat/completions",json=CHAT)
        self.assertEqual(result.status_code,200)
        self.runtime.errors["chat"] = failure(ErrorCode.PROVIDER_ERROR)
        result = await self.client.post("/v1/chat/completions",json=CHAT)
        self.assertEqual(result.status_code,502)
        self.records.fail = False
        self.runtime.errors = {}
        await self.client.post("/v1/chat/completions",json=CHAT)
        self.assertNotIn("test-key",repr(self.records.records)+repr(self.records.updates))
        self.assertEqual(self.records.records[-1].request_body,"")

    async def test_deadline_cancels_nonstream_and_returns_sanitized_504(self):
        self.runtime.pause = True
        with mock.patch.object(openai_api,"REQUEST_TIMEOUT",0.03):
            response = await self.client.post("/v1/chat/completions",json=CHAT)
        self.assertEqual(response.status_code,504,response.text)
        self.assertEqual(response.json()["error"]["code"],"timeout")
        self.assertTrue(self.runtime.chat_closed)
        self.assertTrue(self.runtime.requests[0].context.cancellation.is_set())

    async def test_deadline_after_stream_headers_emits_error_done_and_closes_operation(self):
        self.runtime.pause = True
        with mock.patch.object(openai_api,"REQUEST_TIMEOUT",0.03):
            response = await self.client.post("/v1/chat/completions",json={**CHAT,"stream":True})
        self.assertEqual(response.status_code,200)
        self.assertIn('"code":"timeout"',response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        self.assertEqual(self.runtime.operations[0].closed,1)

    async def test_cancellation_resistant_runtime_cannot_delay_or_write_after_timeout(self):
        release, exited = asyncio.Event(), asyncio.Event()
        late_tasks = []
        async def resistant(request):
            late_tasks.append(asyncio.current_task())
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                return InferenceResult(request.context.request_id,(TextPart("late"),),(),(),TokenUsage(3,2),FinishReason.STOP)
            finally:
                exited.set()
        self.runtime.chat = resistant
        try:
            with mock.patch.object(openai_api,"REQUEST_TIMEOUT",0.05), mock.patch.object(openai_api,"CANCEL_WAIT",0.01):
                sent = await self.raw_request(body=CHAT)
            self.assertTrue(late_tasks and not late_tasks[0].done())
            self.assertEqual([v["status"] for v in sent if v["type"]=="http.response.start"],[504])
            message_count = len(sent)
            release.set()
            await asyncio.wait_for(exited.wait(),0.3)
            await asyncio.sleep(0)
            self.assertEqual(len(sent),message_count)
            self.assertTrue(late_tasks[0].done())
        finally:
            release.set()

    async def test_cancellation_resistant_logger_does_not_delay_inference(self):
        release, exited = asyncio.Event(), asyncio.Event()
        async def slow_add(record):
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
            finally:
                exited.set()
        self.records.add_request = slow_add
        try:
            with mock.patch.object(openai_api,"LOG_TIMEOUT",0.01):
                response = await asyncio.wait_for(self.client.post("/v1/chat/completions",json=CHAT),0.5)
            self.assertEqual(response.status_code,200)
            release.set()
            await asyncio.wait_for(exited.wait(),0.3)
        finally:
            release.set()

    async def test_response_output_limit_is_a_json_protocol_error(self):
        with mock.patch.object(openai_api,"MAX_RESPONSE_SIZE",16):
            response = await self.client.post("/v1/chat/completions",json=CHAT)
        self.assertEqual(response.status_code,502)
        self.assertEqual(response.json()["error"]["code"],"backend_protocol_error")

    async def raw_request(self, *, body=None, send_failure=False, disconnect=False):
        payload = json.dumps(body or {**CHAT,"stream":True}).encode()
        messages = asyncio.Queue()
        await messages.put({"type":"http.request","body":payload,"more_body":False})
        sent = []
        async def send(value):
            if send_failure:
                raise OSError("closed client socket")
            sent.append(value)
        scope = {"type":"http","asgi":{"version":"3.0","spec_version":"2.4"},"http_version":"1.1",
                 "method":"POST","scheme":"http","path":"/v1/chat/completions","raw_path":b"/v1/chat/completions",
                 "query_string":b"","root_path":"","client":("test",1),"server":("test",80),
                 "headers":[(b"authorization",b"Bearer test-key"),(b"content-type",b"application/json")]}
        task = asyncio.create_task(self.app(scope,messages.get,send))
        if disconnect:
            await asyncio.wait_for(self.runtime.started.wait(),1)
            await messages.put({"type":"http.disconnect"})
        await asyncio.wait_for(task,1)
        return sent

    async def test_header_failure_closes_operation_without_entering_generator(self):
        await self.raw_request(send_failure=True)
        self.assertEqual(self.runtime.operations[0].closed,1)
        self.assertNotIn("events",self.runtime.calls)
        self.assertTrue(self.runtime.requests[0].context.cancellation.is_set())

    async def test_disconnect_cancels_nonstream_and_stream_even_before_first_token(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                self.runtime = FakeRuntime()
                self.runtime.pause = True
                self.app.state.local_inference = self.runtime
                await self.raw_request(body={**CHAT,"stream":stream},disconnect=True)
                self.assertTrue(self.runtime.requests[0].context.cancellation.is_set())
                if stream:
                    self.assertEqual(self.runtime.operations[0].closed,1)
                else:
                    self.assertTrue(self.runtime.chat_closed)

    async def test_disconnect_is_bounded_even_when_chat_or_stream_ignores_cancellation(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                self.runtime = FakeRuntime()
                self.app.state.local_inference = self.runtime
                release, exited = asyncio.Event(), asyncio.Event()
                async def ignore_cancel():
                    self.runtime.started.set()
                    try:
                        while not release.is_set():
                            try:
                                await release.wait()
                            except asyncio.CancelledError:
                                pass
                    finally:
                        exited.set()
                async def chat(request):
                    self.runtime.requests.append(request)
                    await ignore_cancel()
                    return InferenceResult(request.context.request_id,(TextPart("late"),),(),(),TokenUsage(3,2),FinishReason.STOP)
                async def events(operation):
                    await ignore_cancel()
                    yield InferenceEvent(EventKind.STARTED,operation.request.context.request_id)
                self.runtime.chat = chat
                try:
                    with mock.patch.object(FakeOperation,"events",events), \
                         mock.patch.object(openai_api,"REQUEST_TIMEOUT",20), \
                         mock.patch.object(openai_api,"CANCEL_WAIT",0.01):
                        sent = await asyncio.wait_for(self.raw_request(body={**CHAT,"stream":stream},disconnect=True),0.5)
                    self.assertFalse(exited.is_set())
                    self.assertTrue(self.runtime.requests[0].context.cancellation.is_set())
                    message_count = len(sent)
                    release.set()
                    await asyncio.wait_for(exited.wait(),0.3)
                    for _ in range(4):
                        await asyncio.sleep(0)
                    self.assertEqual(len(sent),message_count)
                    if stream:
                        self.assertEqual(self.runtime.operations[0].closed,1)
                finally:
                    release.set()


if __name__ == "__main__":
    unittest.main()
