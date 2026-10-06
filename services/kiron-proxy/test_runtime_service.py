import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import time
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import (
    ArtifactIdentity, DeploymentObservation, ErrorCode, EventKind, FinishReason, GenerationOptions,
    InferenceEvent, InferenceRequest, InferenceResult, LifecycleResult, LocalInferenceError,
    Message, MessageRole, ModelLifecycleOperation, ProviderHealth, ProviderObservation, RequestContext, ResolvedDeployment,
    ResolvedModel, ResolverSnapshot, ResourceProfile, RuntimeFailure, RuntimeGeneration,
    RuntimeTimeouts, TextPart, TokenUsage,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.model_state import RuntimeState

from runtime_service import RuntimeService, RuntimeStreamingResponse, generation_key


class FakeProvider:
    provider = BackendType.PRISM
    implementation = object()
    model_lifecycle_operations = frozenset((ModelLifecycleOperation.LOAD, ModelLifecycleOperation.UNLOAD))

    def __init__(self, model, generation):
        self.model, self.generation = model, generation
        self.supported = True
        self.health_error = False
        self.confirm_end = False
        self.mode = "complete"
        self.calls = []
        self.closed = 0

    async def capabilities(self, deployment):
        return SimpleNamespace(supports=lambda *args: self.supported)

    def validate_request(self, request, capabilities):
        pass

    async def health(self, context):
        self.calls.append("health")
        if self.health_error:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.PROVIDER_UNAVAILABLE, "offline"))
        return ProviderObservation(self.provider, self.generation, datetime.now(timezone.utc), ProviderHealth.AVAILABLE, {
            self.model.deployment.id: DeploymentObservation(self.model.deployment.id, RuntimeState.LOADED,
                self.generation, self.model.deployment.configuration_fingerprint)})

    async def chat(self, request):
        self.calls.append("chat")
        if self.mode == "fail":
            raise LocalInferenceError(RuntimeFailure(ErrorCode.PROVIDER_ERROR, "connection lost"))
        return InferenceResult(request.context.request_id, (TextPart("OK"),), (), (), TokenUsage(2, 1), FinishReason.STOP)

    async def stream(self, request):
        self.calls.append("stream")
        yield InferenceEvent(EventKind.STARTED, request.context.request_id)
        yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id, text="OK", output_item_index=0, part_index=0)
        if self.mode == "wait":
            await asyncio.Event().wait()
        yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.STOP)

    async def wait_request_end(self, deployment, *, generation, context):
        self.calls.append("wait_end")
        return self.confirm_end

    async def aclose(self):
        self.closed += 1


class RuntimeServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        root.chmod(0o2770)
        self.store = AdmissionStore(root, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        self.generation = RuntimeGeneration("boot", "child")
        deployment = ResolvedDeployment("model", BackendType.PRISM, "/models/model.gguf",
            ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100), LoaderType.PRISM_GGUF,
            ResourceProfile("measured", 1024, 128, 128, 1, 4, 40, False, 40, 20), "b" * 64)
        self.model = ResolvedModel("public", deployment, None, None, "c" * 64)
        snapshot = ResolverSnapshot("c" * 64, {deployment.id: deployment}, {"public": self.model})
        class Resolver:
            async def snapshot(self):
                return snapshot
        self.provider = FakeProvider(self.model, self.generation)
        self.measure = lambda: MemorySnapshot(100, 100, time.monotonic())
        self.service = RuntimeService(resolver=Resolver(), providers={BackendType.PRISM: self.provider},
            admission=self.store, measure=self.measure, timeouts=RuntimeTimeouts(1, 1, 1, 1, 1, .1, .1))
        self.addAsyncCleanup(self.service.aclose)
        self.store.reserve(operation_id="resident", owner="kiron-proxy", generation=generation_key(self.generation),
            deployment_id="model", kind="load", gpu_bytes=40, host_bytes=20, measure=self.measure, resident_slot="prism")
        self.store.transition("resident", owner="kiron-proxy", expected_generation=generation_key(self.generation), phase="resident")

    def request(self, request_id="request"):
        return InferenceRequest(self.model, (Message(MessageRole.USER, (TextPart("Hi"),)),), GenerationOptions(8),
            RequestContext(request_id, time.monotonic() + 3, asyncio.Event()))

    def requests(self):
        return [ticket for ticket in self.store.snapshot() if ticket.kind == "request"]

    async def test_capability_rejected_before_health_load_or_reservation(self):
        self.provider.supported = False
        with self.assertRaises(LocalInferenceError):
            await self.service.prepare(self.request())
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.requests(), [])

    async def test_undeclared_model_mutations_leave_tickets_and_provider_untouched(self):
        self.provider.model_lifecycle_operations = frozenset()
        before = self.store.snapshot()
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.unload(self.model, self.request().context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.store.snapshot(), before)
        self.provider.start = mock.AsyncMock()
        self.provider.load = mock.AsyncMock()
        for status in (ProviderHealth.AVAILABLE, ProviderHealth.STARTABLE):
            with self.subTest(health=status):
                self.provider.health = mock.AsyncMock(return_value=ProviderObservation(
                    BackendType.PRISM, self.generation, datetime.now(timezone.utc), status, {}))
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.load(self.model, self.request().context)
                self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
                self.provider.start.assert_not_awaited()
                self.provider.load.assert_not_awaited()
                self.assertEqual(self.store.snapshot(), before)

    async def test_read_only_resident_provider_needs_no_load_permission(self):
        self.provider.model_lifecycle_operations = frozenset()
        self.provider.load = mock.AsyncMock()
        before = self.store.snapshot()
        loaded = await self.service.load(self.model, self.request().context)
        self.assertFalse(loaded.changed)
        self.provider.load.assert_not_awaited()
        self.assertEqual(self.store.snapshot(), before)
        result = await self.service.chat(self.request())
        self.assertEqual(result.content, (TextPart("OK"),))
        self.provider.load.assert_not_awaited()
        self.assertEqual(self.store.snapshot(), before)

    async def test_request_slot_is_reserved_before_backend_and_prevents_concurrency(self):
        operation = await self.service.prepare(self.request(), streaming=True)
        self.assertEqual(self.provider.calls, ["health"])
        self.assertEqual(len(self.requests()), 1)
        with self.assertRaises(LocalInferenceError):
            await self.service.prepare(self.request("other"))
        await operation.close()
        await operation.close()
        self.assertEqual(self.requests(), [])

    async def test_nonstream_success_releases_only_request_not_resident_model(self):
        result = await self.service.chat(self.request())
        self.assertEqual(result.content, (TextPart("OK"),))
        self.assertEqual(self.requests(), [])
        self.assertEqual(self.store.snapshot()[0].phase, "resident")
        self.assertNotIn("wait_end", self.provider.calls)

    async def test_nonstream_foreign_request_id_is_rejected_and_does_not_prove_end(self):
        self.provider.chat = mock.AsyncMock(return_value=InferenceResult("foreign",
            (TextPart("wrong turn"),), (), (), TokenUsage(2, 1), FinishReason.STOP))
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.chat(self.request())
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        self.assertEqual(self.requests()[0].phase, "unknown")

    async def test_failed_backend_keeps_unknown_ticket_until_reconciliation(self):
        self.provider.mode = "fail"
        with self.assertRaises(LocalInferenceError):
            await self.service.chat(self.request())
        self.assertEqual(self.requests()[0].phase, "unknown")
        self.assertIn("wait_end", self.provider.calls)

    async def test_deadline_failure_keeps_unknown_and_blocks_the_next_gpu_request(self):
        self.provider.chat = mock.AsyncMock(side_effect=LocalInferenceError(
            RuntimeFailure(ErrorCode.TIMEOUT, "Request deadline exceeded")))
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.chat(self.request())
        self.assertEqual(raised.exception.failure.code, ErrorCode.TIMEOUT)
        self.assertEqual(self.requests()[0].phase, "unknown")
        self.assertIn("wait_end", self.provider.calls)
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.prepare(self.request("next"))
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_UNAVAILABLE)
        self.assertEqual(self.requests()[0].phase, "unknown")
        self.assertEqual(len(self.requests()), 1)

    async def test_nonstream_exact_output_budget_passes_excess_fails_without_clamping(self):
        count = 8
        async def reply(request):
            return InferenceResult(request.context.request_id, (TextPart("fixture"),), (), (),
                                   TokenUsage(2, count), FinishReason.LENGTH)
        self.provider.chat = reply
        result = await self.service.chat(self.request('exact-budget'))
        self.assertEqual(result.usage.output_tokens, 8)
        self.assertEqual(self.requests(), [])
        count = 9
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.chat(self.request('excess-budget'))
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        self.assertEqual(self.requests()[0].phase, 'unknown')

    async def test_stream_excess_usage_is_never_emitted_or_confirmed_complete(self):
        count = 8
        async def events(request):
            yield InferenceEvent(EventKind.STARTED, request.context.request_id)
            yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id, text='partial', output_item_index=0, part_index=0)
            yield InferenceEvent(EventKind.USAGE, request.context.request_id, usage=TokenUsage(2, count))
            yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.LENGTH)
        self.provider.stream = events
        operation = await self.service.prepare(self.request('exact-stream'), streaming=True)
        emitted = [event async for event in operation.events()]
        await operation.close()
        self.assertEqual(emitted[-2].usage.output_tokens, 8)
        self.assertEqual(self.requests(), [])
        count = 9
        operation = await self.service.prepare(self.request('excess-stream'), streaming=True)
        emitted = []
        with self.assertRaises(LocalInferenceError) as raised:
            async for event in operation.events():
                emitted.append(event)
        await operation.close()
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        self.assertEqual([event.kind for event in emitted], [EventKind.STARTED, EventKind.TEXT_DELTA])
        self.assertEqual(self.requests()[0].phase, 'unknown')

    async def test_failed_backend_with_independent_end_proof_releases_ticket(self):
        self.provider.mode, self.provider.confirm_end = "fail", True
        with self.assertRaises(LocalInferenceError):
            await self.service.chat(self.request())
        self.assertEqual(self.requests(), [])

    async def test_header_failure_before_body_runs_cleans_the_unstarted_ticket(self):
        operation = await self.service.prepare(self.request(), streaming=True)
        async def content():
            async for event in operation.events():
                yield event.kind.value.encode()
        response = RuntimeStreamingResponse(content(), operation=operation)
        async def send(event):
            raise RuntimeError("header send failed")
        async def receive():
            await asyncio.Event().wait()
        with self.assertRaisesRegex(RuntimeError, "header send failed"):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        self.assertEqual(self.requests(), [])
        self.assertNotIn("stream", self.provider.calls)

    async def test_disconnect_after_output_keeps_unknown_work_and_does_not_unload(self):
        operation = await self.service.prepare(self.request(), streaming=True)
        self.provider.mode = "wait"
        iterator = operation.events()
        await iterator.__anext__()
        await iterator.__anext__()
        await iterator.aclose()
        await operation.close()
        self.assertEqual(self.requests()[0].phase, "unknown")
        self.assertEqual(next(t for t in self.store.snapshot() if t.kind == "load").phase, "resident")

    async def test_exception_after_terminal_cannot_release_request_without_end_proof(self):
        async def broken(request):
            yield InferenceEvent(EventKind.STARTED, request.context.request_id)
            yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.STOP)
            raise RuntimeError("late provider failure")
        self.provider.stream = broken
        operation = await self.service.prepare(self.request(), streaming=True)
        with self.assertRaisesRegex(RuntimeError, "late provider failure"):
            _ = [event async for event in operation.events()]
        await operation.close()
        self.assertEqual(self.requests()[0].phase, "unknown")

    async def test_closing_after_terminal_before_eof_does_not_prove_backend_end(self):
        operation = await self.service.prepare(self.request(), streaming=True)
        events = operation.events()
        async for event in events:
            if event.kind is EventKind.COMPLETED:
                break
        await events.aclose()
        await operation.close()
        self.assertEqual(self.requests()[0].phase, "unknown")

    async def test_clean_eof_after_completion_releases_request(self):
        operation = await self.service.prepare(self.request(), streaming=True)
        _ = [event async for event in operation.events()]
        await operation.close()
        self.assertEqual(self.requests(), [])
        self.assertNotIn("wait_end", self.provider.calls)

    async def test_provider_failure_isolated_in_health_collection(self):
        self.provider.health_error = True
        good = FakeProvider(self.model, self.generation)
        good.provider = BackendType.OLLAMA
        self.service.providers[BackendType.OLLAMA] = good
        observations = await self.service.observations(self.request().context)
        self.assertEqual(observations[BackendType.PRISM].health, ProviderHealth.UNAVAILABLE)
        self.assertEqual(observations[BackendType.OLLAMA].health, ProviderHealth.AVAILABLE)

    async def test_shutdown_cleans_unstarted_operations_and_closes_provider_once(self):
        await self.service.prepare(self.request(), streaming=True)
        await self.service.aclose()
        await self.service.aclose()
        self.assertEqual(self.requests(), [])
        self.assertEqual(self.provider.closed, 1)
        with self.assertRaises(LocalInferenceError):
            await self.service.prepare(self.request("new"))

    async def test_contended_lifecycle_lock_obeys_deadline_without_backend_io(self):
        lock = self.service._locks[BackendType.PRISM]
        await lock.acquire()
        context = RequestContext("short", time.monotonic() + .02, asyncio.Event())
        try:
            with self.assertRaises(LocalInferenceError) as raised:
                await self.service.load(self.model, context)
            self.assertEqual(raised.exception.failure.code, ErrorCode.TIMEOUT)
            self.assertEqual(self.provider.calls, [])
            self.assertTrue(lock.locked())
        finally:
            lock.release()
        await self.service.load(self.model, self.request().context)
        self.assertFalse(lock.locked())

    async def test_contended_lifecycle_lock_obeys_cancellation(self):
        lock = self.service._locks[BackendType.PRISM]
        await lock.acquire()
        context = self.request().context
        pending = asyncio.create_task(self.service.load(self.model, context))
        try:
            await asyncio.sleep(.01)
            context.cancellation.set()
            with self.assertRaises(LocalInferenceError) as raised:
                await pending
            self.assertEqual(raised.exception.failure.code, ErrorCode.CANCELLED)
            self.assertEqual(self.provider.calls, [])
        finally:
            lock.release()

    async def test_unavailable_unload_result_does_not_release_resident_ticket(self):
        result = LifecycleResult("unload", ProviderObservation(BackendType.PRISM, self.generation,
            datetime.now(timezone.utc), ProviderHealth.UNAVAILABLE, {}), True)
        self.provider.unload = mock.AsyncMock(return_value=result)
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.unload(self.model, self.request("unload").context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        self.assertEqual(next(t for t in self.store.snapshot() if t.kind == "load").phase, "draining")

    async def test_cancel_during_admission_transaction_removes_unstarted_request(self):
        entered, complete = threading.Event(), threading.Event()
        original = self.store.reserve
        def delayed(**kwargs):
            entered.set()
            if not complete.wait(2):
                raise RuntimeError("test transaction was not released")
            return original(**kwargs)
        with mock.patch.object(self.store, "reserve", side_effect=delayed):
            pending = asyncio.create_task(self.service.prepare(self.request()))
            await asyncio.to_thread(entered.wait, 1)
            self.assertTrue(entered.is_set())
            pending.cancel()
            complete.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertEqual(self.requests(), [])
        self.assertNotIn("chat", self.provider.calls)

    async def test_duplicate_request_id_cannot_own_or_release_running_request(self):
        first = await self.service.prepare(self.request())
        first.backend_started = True
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.prepare(self.request())
        self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
        self.assertEqual(len(self.requests()), 1)
        first.confirmed_end = True
        await first.close()

    async def test_confirmed_context_rejection_releases_request_but_preserves_resident(self):
        self.provider.chat = mock.AsyncMock(side_effect=LocalInferenceError(RuntimeFailure(
            ErrorCode.CONTEXT_LENGTH_EXCEEDED, "Input exceeds the model context")))
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.chat(self.request())
        self.assertEqual(raised.exception.failure.code, ErrorCode.CONTEXT_LENGTH_EXCEEDED)
        self.assertEqual(self.requests(), [])
        self.assertEqual(self.store.snapshot()[0].kind, "load")
        self.assertNotIn("wait_end", self.provider.calls)

    async def test_context_rejection_stream_requires_clean_eof_for_ticket_completion(self):
        for suffix in ("clean", "exception", "frame", "close"):
            with self.subTest(suffix=suffix):
                async def events(request):
                    yield InferenceEvent(EventKind.STARTED, request.context.request_id)
                    yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                        error=RuntimeFailure(ErrorCode.CONTEXT_LENGTH_EXCEEDED, "Input exceeds the model context"))
                    if suffix == "exception":
                        raise OSError("late transport failure")
                    if suffix == "frame":
                        yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.STOP)
                self.provider.stream = events
                operation = await self.service.prepare(self.request(suffix), streaming=True)
                iterator = operation.events()
                if suffix == "close":
                    await iterator.__anext__()
                    await iterator.aclose()
                elif suffix in {"exception", "frame"}:
                    with self.assertRaises((OSError, LocalInferenceError)):
                        _ = [event async for event in iterator]
                else:
                    result = [event async for event in iterator]
                    self.assertEqual(result[-1].kind, EventKind.FAILED)
                await operation.close()
                if suffix == "clean":
                    self.assertEqual(self.requests(), [])
                else:
                    self.assertEqual([(t.operation_id, t.phase) for t in self.requests()], [(suffix, "unknown")])
                    self.store.release(suffix, owner="kiron-proxy", generation=generation_key(self.generation),
                                       confirmed_terminated=True)

    async def test_context_failure_terminal_is_published_only_with_end_proof(self):
        async def events(request):
            yield InferenceEvent(EventKind.STARTED, request.context.request_id)
            yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                error=RuntimeFailure(ErrorCode.CONTEXT_LENGTH_EXCEEDED, "Input exceeds the model context"))
        self.provider.stream = events
        operation = await self.service.prepare(self.request(), streaming=True)
        iterator = operation.events()
        await iterator.__anext__()
        terminal = await iterator.__anext__()
        self.assertEqual(terminal.kind, EventKind.FAILED)
        self.assertTrue(operation.confirmed_end)
        await iterator.aclose()
        await operation.close()
        self.assertEqual(self.requests(), [])

    async def test_invalid_canonical_indices_leave_request_unknown(self):
        async def events(request):
            yield InferenceEvent(EventKind.STARTED, request.context.request_id)
            yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id, text="invalid first index",
                                 output_item_index=1, part_index=0)
            yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=FinishReason.STOP)
        self.provider.stream = events
        operation = await self.service.prepare(self.request(), streaming=True)
        with self.assertRaises(LocalInferenceError) as raised:
            _ = [event async for event in operation.events()]
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        await operation.close()
        self.assertEqual(self.requests()[0].phase, "unknown")


if __name__ == "__main__":
    unittest.main()
