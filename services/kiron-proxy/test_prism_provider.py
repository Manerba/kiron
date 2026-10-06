import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    ParameterConstraint, ErrorCode, EventKind, GenerationOptions, InferenceRequest,
    LocalInferenceError, Message, MessageRole, RequestContext, ResolvedDeployment,
    ProviderHealth, ResolvedModel, ResourceProfile, RuntimeGeneration, RuntimeImplementation, TextPart,
    validate_event_sequence,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

from prism_provider import PrismProvider


FIXTURES = Path(__file__).parent / "contracts/fixtures/prism-b10709"


class PrismProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = RequestContext("request", time.monotonic() + 3, asyncio.Event())
        self.deployment = ResolvedDeployment("model", BackendType.PRISM, "/models/model.gguf",
            ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100), LoaderType.PRISM_GGUF,
            ResourceProfile("measured", 1024, 128, 128, 1, 4, 40, False, 40, 20), "b" * 64)
        self.model = ResolvedModel("public", self.deployment, None, None, "c" * 64)
        self.request = InferenceRequest(self.model, (Message(MessageRole.USER, (TextPart("Say OK"),)),),
                                        GenerationOptions(8), self.context, execution_generation=RuntimeGeneration("boot", "child"))
        self.calls = []
        self.observation = dict(provider="prism", generation=dict(boot_id="boot", process_id="child"),
            state="loaded", health="healthy", deployment_id="model", configuration_fingerprint="b" * 64,
            snapshot_revision="c" * 64, backend_model="bonsai-probe", observed_at=time.time(), error=None)

    async def make(self, fixture="text", control_override=None, inference_handler=None):
        async def control(request):
            self.calls.append((request.url.path, json.loads(request.content) if request.content else None))
            return httpx.Response(200, json=self.observation if control_override is None else control_override)
        async def inference(request):
            self.calls.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, content=(FIXTURES / f"{fixture}.response.raw").read_bytes())
        adapter = PrismProvider(control=httpx.AsyncClient(base_url="http://control", transport=httpx.MockTransport(control)),
            inference=httpx.AsyncClient(base_url="http://backend", transport=httpx.MockTransport(inference_handler or inference)),
            resolver=SimpleNamespace(), implementation=RuntimeImplementation("revision", None, "parser-v1"))
        evidence = CapabilityEvidence("revision", self.deployment.artifact_identity.fingerprint,
            "a" * 64, None, None, "parser-v1", "b" * 64, "recorded-text-fixture", datetime.now(timezone.utc))
        capability = Capability(CapabilityStatus.SUPPORTED, {
            "roles": ParameterConstraint(allowed_values=("user",)),
            "max_output_tokens": ParameterConstraint(minimum=1, maximum=128),
            "default_max_output_tokens": ParameterConstraint(allowed_values=(8,)),
        }, (evidence,))
        adapter._capabilities[self.deployment.id] = CapabilitySet({CapabilityName.CHAT: capability,
                                                                 CapabilityName.STREAMING: capability})
        self.addAsyncCleanup(adapter.aclose)
        await adapter.health(self.context)
        return adapter

    async def test_recorded_native_text_becomes_canonical_result_and_uses_owned_alias(self):
        adapter = await self.make()
        result = await adapter.chat(self.request)
        self.assertEqual(result.content, (TextPart("OK"),))
        self.assertEqual(result.usage.input_tokens, 19)
        self.assertEqual(result.usage.output_tokens, 2)
        self.assertIsNone(result.usage.reasoning_output_tokens)
        self.assertEqual(self.calls[-1][1]["model"], "bonsai-probe")

    async def test_stopped_service_health_is_read_only_and_requires_verified_startable_state(self):
        adapter = await self.make()
        async def offline(request):
            raise httpx.ConnectError("not listening", request=request)
        await adapter.control.aclose()
        adapter.control = httpx.AsyncClient(base_url="http://control", transport=httpx.MockTransport(offline))
        adapter.service_control = SimpleNamespace(status=mock.AsyncMock(return_value=ProviderHealth.STARTABLE),
            start=mock.AsyncMock())
        observed = await adapter.health(self.context)
        self.assertEqual(observed.health, ProviderHealth.STARTABLE)
        self.assertIsNone(observed.generation)
        adapter.service_control.start.assert_not_called()
        adapter.service_control.status.return_value = ProviderHealth.UNAVAILABLE
        with self.assertRaises(LocalInferenceError) as raised:
            await adapter.health(self.context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_UNAVAILABLE)

    async def test_start_waits_for_healthy_control_and_is_deadline_bounded(self):
        adapter = await self.make()
        ready = await adapter.health(self.context)
        adapter.service_control = SimpleNamespace(start=mock.AsyncMock(return_value=True))
        not_ready = replace(ready, health=ProviderHealth.UNAVAILABLE, models={})
        adapter.health = mock.AsyncMock(side_effect=[not_ready, ready])
        result = await adapter.start(self.context)
        self.assertEqual(result.observation, ready)
        self.assertTrue(result.changed)
        self.assertEqual(adapter.health.await_count, 2)
        adapter.health = mock.AsyncMock(return_value=not_ready)
        context = RequestContext("start-deadline", time.monotonic() + .02, asyncio.Event())
        with self.assertRaises(LocalInferenceError) as raised:
            await adapter.start(context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.TIMEOUT)

    async def test_recorded_native_stream_has_exact_usage_and_one_terminal(self):
        adapter = await self.make("text_stream")
        events = [event async for event in adapter.stream(self.request)]
        validate_event_sequence(events)
        self.assertEqual("".join(e.text for e in events if e.kind is EventKind.TEXT_DELTA), "OK")
        self.assertEqual(events[-2].usage.cached_input_tokens, 15)
        self.assertEqual(events[-1].kind, EventKind.COMPLETED)
        self.assertEqual(sum(e.terminal for e in events), 1)

    async def test_lifecycle_sends_only_identifiers_and_expected_generation(self):
        adapter = await self.make()
        await adapter.load(self.deployment, snapshot_revision="c" * 64,
                           expected_generation=RuntimeGeneration("boot"), context=self.context)
        path, body = self.calls[-1]
        self.assertEqual(path, "/load")
        self.assertEqual(set(body), {"deployment_id", "snapshot_revision", "operation_id", "expected_generation"})
        self.assertNotIn("reference", body)

    async def test_native_request_uses_its_own_generation_token(self):
        headers = []
        async def native(request):
            headers.append(request.headers.get("authorization"))
            return httpx.Response(200, content=(FIXTURES / "text.response.raw").read_bytes())
        adapter = await self.make(inference_handler=native)
        await adapter.chat(self.request)
        self.assertEqual(headers, ["Bearer kiron-prism-child"])

    async def test_failed_transport_never_proves_request_ended(self):
        adapter = await self.make()
        self.assertFalse(await adapter.wait_request_end(self.deployment, generation=RuntimeGeneration("boot", "child"), context=self.context))

    async def delayed_tcp_adapter(self, *, stage, streaming=False):
        """Hold native headers/body until released, using real HTTPX timeouts."""
        received, release = asyncio.Event(), asyncio.Event()
        handlers = set()
        raw = (FIXTURES / ("text_stream.response.raw" if streaming else "text.response.raw")).read_bytes()

        async def serve(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                size = next(int(line.split(b":", 1)[1]) for line in headers.lower().split(b"\r\n")
                            if line.startswith(b"content-length:"))
                await reader.readexactly(size)
                received.set()
                if stage == "headers":
                    await release.wait()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(raw)).encode()
                             + b"\r\nConnection: close\r\n\r\n")
                await writer.drain()
                if stage == "body":
                    await release.wait()
                writer.write(raw)
                await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                pass  # Expected when the request deadline closes the socket.
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass

        server = await asyncio.start_server(serve, "127.0.0.1", 0)
        adapter = await self.make()
        await adapter.inference.aclose()
        adapter.inference = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", trust_env=False,
            timeout=httpx.Timeout(.03, connect=1, write=1, pool=1))

        async def cleanup():
            release.set()
            await adapter.inference.aclose()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.wait_for(asyncio.gather(*handlers), 1)
        self.addAsyncCleanup(cleanup)
        return adapter, received, release

    async def test_buffered_tcp_response_may_exceed_stream_idle_within_remaining_deadline(self):
        for stage in ("headers", "body"):
            with self.subTest(stage=stage):
                adapter, received, release = await self.delayed_tcp_adapter(stage=stage)
                context = RequestContext("buffered", time.monotonic() + 2, asyncio.Event())
                pending = asyncio.create_task(adapter.chat(replace(self.request, context=context)))
                try:
                    await asyncio.wait_for(received.wait(), 1)
                    done, _ = await asyncio.wait({pending}, timeout=.09)
                    self.assertFalse(done, "stream idle timeout must not cut off buffered inference")
                    release.set()
                    result = await asyncio.wait_for(pending, 1)
                    self.assertEqual(result.content, (TextPart("OK"),))
                    self.assertEqual(adapter.inference.timeout.read, .03)
                finally:
                    release.set()
                    if not pending.done():
                        pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

    async def test_buffered_tcp_total_deadline_is_not_reset_and_cannot_confirm_backend_end(self):
        for stage in ("headers", "body"):
            with self.subTest(stage=stage):
                adapter, received, release = await self.delayed_tcp_adapter(stage=stage)
                context = RequestContext("expired", time.monotonic() + .15, asyncio.Event())
                started = time.monotonic()
                with self.assertRaises(LocalInferenceError) as raised:
                    await asyncio.wait_for(adapter.chat(replace(self.request, context=context)), 1)
                self.assertTrue(received.is_set())
                self.assertEqual(raised.exception.failure.code, ErrorCode.TIMEOUT)
                self.assertLess(time.monotonic() - started, .8)
                self.assertFalse(await adapter.wait_request_end(self.deployment,
                    generation=self.request.execution_generation, context=self.context))
                release.set()

    async def test_stream_tcp_idle_protection_is_unchanged_before_headers_and_during_body(self):
        for stage in ("headers", "body"):
            with self.subTest(stage=stage):
                adapter, received, release = await self.delayed_tcp_adapter(stage=stage, streaming=True)
                events = [event async for event in adapter.stream(self.request)]
                self.assertTrue(received.is_set())
                self.assertEqual(events[-1].kind, EventKind.FAILED)
                self.assertEqual(events[-1].error.code, ErrorCode.TIMEOUT)
                self.assertFalse(any(event.kind is EventKind.COMPLETED for event in events))
                release.set()

    async def test_buffered_read_timeout_uses_remaining_budget_and_preserves_other_limits(self):
        observed = []
        async def native(request):
            observed.append(request.extensions["timeout"])
            return httpx.Response(200, content=(FIXTURES / "text.response.raw").read_bytes())
        adapter = await self.make(inference_handler=native)
        adapter.inference.timeout = httpx.Timeout(.03, connect=1, write=2, pool=3)
        context = RequestContext("remaining", time.monotonic() + .5, asyncio.Event())
        await adapter.chat(replace(self.request, context=context))
        self.assertEqual({k: v for k, v in observed[0].items() if k != "read"},
                         {"connect": 1, "write": 2, "pool": 3})
        self.assertGreater(observed[0]["read"], .03)
        self.assertLessEqual(observed[0]["read"], .5)
        self.assertEqual(adapter.inference.timeout.read, .03)

    async def test_changed_configuration_or_unknown_generation_cannot_be_loaded(self):
        observation = {**self.observation, "generation": {"boot_id": "boot", "process_id": None}}
        with self.assertRaises(LocalInferenceError):
            await self.make(control_override=observation)
        adapter = await self.make()
        request = replace(self.request, model=replace(self.model, deployment=replace(self.deployment, configuration_fingerprint="d" * 64)))
        with self.assertRaises(LocalInferenceError):
            await adapter.chat(request)
        self.assertEqual([p for p, _ in self.calls].count("/v1/chat/completions"), 0)

    async def test_truncated_stream_fails_without_false_completion(self):
        async def truncated(request):
            raw = (FIXTURES / "text_stream.response.raw").read_bytes().replace(b"data: [DONE]\n\n", b"")
            return httpx.Response(200, content=raw)
        adapter = await self.make(inference_handler=truncated)
        events = [event async for event in adapter.stream(self.request)]
        validate_event_sequence(events)
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertFalse(any(e.kind is EventKind.COMPLETED for e in events))

    async def test_native_frame_after_done_cannot_be_hidden_by_provider_iterator_eof(self):
        async def tailed(request):
            raw = (FIXTURES / "text_stream.response.raw").read_bytes()
            return httpx.Response(200, content=raw + b'data: {"late":true}\n\n')
        adapter = await self.make(inference_handler=tailed)
        events = [event async for event in adapter.stream(self.request)]
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertFalse(any(event.kind is EventKind.COMPLETED for event in events))

    async def test_cancel_during_response_headers_is_bounded(self):
        entered = asyncio.Event()
        async def slow(request):
            entered.set()
            await asyncio.Event().wait()
        adapter = await self.make(inference_handler=slow)
        async def collect():
            return [event async for event in adapter.stream(self.request)]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 1)
        self.context.cancellation.set()
        events = await asyncio.wait_for(task, 1)
        self.assertEqual(events[-1].kind, EventKind.CANCELLED)
        validate_event_sequence(events)

    async def test_invalid_usage_is_rejected_instead_of_estimated(self):
        async def invalid(request):
            value = json.loads((FIXTURES / "text.response.raw").read_bytes())
            value["usage"]["total_tokens"] = 999
            return httpx.Response(200, json=value)
        adapter = await self.make(inference_handler=invalid)
        with self.assertRaises(LocalInferenceError):
            await adapter.chat(self.request)

    async def test_malformed_native_shapes_have_controlled_provider_errors(self):
        for field, malformed in (("usage", []), ("details", None), ("details", []),
                                 ("choices", {}), ("choice", []), ("index", False), ("message", [])):
            with self.subTest(field=field, malformed=malformed):
                async def invalid(request):
                    value = json.loads((FIXTURES / "text.response.raw").read_bytes())
                    if field == "details":
                        value["usage"]["prompt_tokens_details"] = malformed
                    elif field == "choice":
                        value["choices"] = [malformed]
                    elif field in {"index", "message"}:
                        value["choices"][0][field] = malformed
                    else:
                        value[field] = malformed
                    return httpx.Response(200, json=value)
                adapter = await self.make(inference_handler=invalid)
                with self.assertRaises(LocalInferenceError) as raised:
                    await adapter.chat(self.request)
                self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)

        for malformed in (None, [], "text", 1, False):
            with self.subTest(delta=malformed):
                async def invalid_stream(request):
                    value = {"model": "bonsai-probe", "choices": [{"index": 0, "delta": malformed}]}
                    return httpx.Response(200, content=('data: ' + json.dumps(value) + '\n\n').encode())
                adapter = await self.make(inference_handler=invalid_stream)
                events = [event async for event in adapter.stream(self.request)]
                self.assertEqual(events[-1].kind, EventKind.FAILED)
                self.assertEqual(events[-1].error.code, ErrorCode.PROVIDER_ERROR)

    async def test_unterminated_huge_frame_is_bounded_and_fails(self):
        async def huge(request):
            return httpx.Response(200, content=b"data: " + b"x" * (1024 * 1024 + 1))
        adapter = await self.make(inference_handler=huge)
        events = [event async for event in adapter.stream(self.request)]
        self.assertEqual(events[-1].kind, EventKind.FAILED)

    async def test_refresh_to_new_generation_cannot_rebind_an_existing_request(self):
        adapter = await self.make()
        self.observation["generation"]["process_id"] = "replacement"
        self.observation["backend_model"] = "replacement-alias"
        await adapter.health(self.context)
        with self.assertRaises(LocalInferenceError):
            await adapter.chat(self.request)
        self.assertFalse(any(path == "/v1/chat/completions" for path, _ in self.calls))

    async def test_unload_requires_healthy_generation_bound_confirmation(self):
        adapter = await self.make()
        self.observation.update(state="unloaded", health="unhealthy", deployment_id=None)
        with self.assertRaises(LocalInferenceError):
            await adapter.unload(self.deployment, snapshot_revision="c" * 64,
                expected_generation=RuntimeGeneration("boot", "child"), context=self.context)

    async def test_complete_native_context_error_is_typed_and_public_message_is_sanitized(self):
        from openai_api import _runtime_error
        from openai_wire import chat_events
        value = {"error": {"code": 400, "type": "exceed_context_size_error",
            "message": "private prompt /models/secret", "n_prompt_tokens": 2048, "n_ctx": 1024}}
        async def rejected(request):
            return httpx.Response(400, json=value)
        adapter = await self.make(inference_handler=rejected)
        with self.assertRaises(LocalInferenceError) as caught:
            await adapter.chat(self.request)
        error = _runtime_error(caught.exception)
        self.assertEqual((error.status, error.code, error.error_type),
                         (400, "context_length_exceeded", "invalid_request_error"))
        self.assertNotIn("secret", json.dumps(error.envelope()))
        events = [e async for e in adapter.stream(self.request)]
        self.assertEqual(events[-1].error.code, ErrorCode.CONTEXT_LENGTH_EXCEEDED)
        async def source():
            for e in events:
                yield e
        raw = b''.join([chunk async for chunk in chat_events(source(), "public", "id", 123, True)])
        self.assertIn(b'"code":"context_length_exceeded"', raw)
        self.assertNotIn(b'secret', raw)
        self.assertNotIn(b'"finish_reason":"stop"', raw)
        self.assertTrue(raw.endswith(b'data: [DONE]\n\n'))

    async def test_context_error_requires_exact_envelope_and_complete_transport(self):
        from prism_provider import _http_failure
        base = {"code": 400, "type": "exceed_context_size_error", "message": "private",
                "n_prompt_tokens": 2048, "n_ctx": 1024}
        invalid = [{**base, "n_ctx": True}, {**base, "n_prompt_tokens": 10},
                   {**base, "type": "invalid_request_error"}, {**base, "extra": 1},
                   {key: value for key, value in base.items() if key != "n_ctx"}]
        for error in invalid:
            with self.subTest(error=error):
                self.assertNotEqual(_http_failure(400, json.dumps({"error": error})).failure.code,
                                    ErrorCode.CONTEXT_LENGTH_EXCEEDED)
        for transport_failure in (httpx.ReadTimeout("private"), httpx.ReadError("private")):
            class InterruptedBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield json.dumps({"error": base}).encode()
                    raise transport_failure
            async def interrupted(request):
                return httpx.Response(400, stream=InterruptedBody())
            adapter = await self.make(inference_handler=interrupted)
            with self.assertRaises(LocalInferenceError) as caught:
                await adapter.chat(self.request)
            self.assertNotEqual(caught.exception.failure.code, ErrorCode.CONTEXT_LENGTH_EXCEEDED)
            events = [e async for e in adapter.stream(self.request)]
            self.assertNotEqual(events[-1].error.code, ErrorCode.CONTEXT_LENGTH_EXCEEDED)

    async def test_timeout_headers_first_byte_and_partial_body_keep_timeout_class(self):
        from openai_wire import chat_events
        for phase in ("headers", "first_byte", "partial_body"):
            with self.subTest(phase=phase):
                class TimeoutBody(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        if phase == "partial_body":
                            # bounded_lines reads 64KiB chunks; spaces remain legal SSE comments.
                            frame = b'data: {"model":"bonsai-probe","choices":[{"index":0,"delta":{"content":"part"}}]}\n\n'
                            yield frame + b':' + b' ' * (65536 - len(frame) - 2) + b'\n'
                        raise httpx.ReadTimeout("private transport detail")
                async def timeout(request):
                    if phase == "headers":
                        raise httpx.ReadTimeout("private transport detail", request=request)
                    return httpx.Response(200, stream=TimeoutBody())
                adapter = await self.make(inference_handler=timeout)
                events = [e async for e in adapter.stream(self.request)]
                validate_event_sequence(events)
                self.assertEqual(events[-1].error.code, ErrorCode.TIMEOUT)
                self.assertFalse(any(e.kind in (EventKind.USAGE, EventKind.COMPLETED) for e in events))
                self.assertEqual(any(e.kind is EventKind.TEXT_DELTA for e in events), phase == "partial_body")
                async def source():
                    for e in events:
                        yield e
                raw = b''.join([p async for p in chat_events(source(), "public", "id", 123, True)])
                self.assertIn(b'"code":"timeout"', raw)
                self.assertNotIn(b'private', raw)
                self.assertNotIn(b'backend_protocol_error', raw)

    async def test_context_rejection_does_not_hide_response_close_failure(self):
        class CloseTimeout(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield json.dumps({"error": {"code": 400, "type": "exceed_context_size_error",
                    "message": "private", "n_prompt_tokens": 1024, "n_ctx": 1024}}).encode()
            async def aclose(self):
                raise httpx.ReadTimeout("close was not confirmed")
        async def rejected(request):
            return httpx.Response(400, stream=CloseTimeout())
        adapter = await self.make(inference_handler=rejected)
        with self.assertRaises(LocalInferenceError) as caught:
            await adapter.chat(self.request)
        self.assertEqual(caught.exception.failure.code, ErrorCode.TIMEOUT)
        events = [e async for e in adapter.stream(self.request)]
        self.assertEqual(events[-1].error.code, ErrorCode.TIMEOUT)
        self.assertFalse(any(e.error and e.error.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED for e in events))

    async def test_total_stream_deadline_remains_timeout_not_cancel_or_protocol_error(self):
        async def slow(request):
            await asyncio.Event().wait()
        adapter = await self.make(inference_handler=slow)
        context = replace(self.context, deadline_monotonic=time.monotonic() + .01)
        events = [e async for e in adapter.stream(replace(self.request, context=context))]
        self.assertEqual(events[-1].error.code, ErrorCode.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
