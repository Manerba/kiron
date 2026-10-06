"""Real runtime/adapter/admission lifecycle tests with native HTTP mocked only."""
import asyncio
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    ErrorCode, EventKind, GenerationOptions, InferenceRequest, LocalInferenceError, Message, MessageRole,
    ParameterConstraint, RequestContext, ResourceProfile, ResolvedDeployment, ResolvedModel, ResolverSnapshot,
    RuntimeFailure, RuntimeImplementation, RuntimeTimeouts, TextPart,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.ollama_compat import CompatResult, OllamaCapabilities

from native_admission import NativeRequestOperation
from ollama_provider import OllamaProvider
from runtime_service import RuntimeService, generation_key


class OllamaRuntimeLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        root.chmod(0o2770)
        self.store = AdmissionStore(root, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        self.deployments = {}
        models = {}
        for name, digest in (("target", "a"), ("other", "b")):
            deployment = ResolvedDeployment(name, BackendType.OLLAMA, name + ":latest",
                ArtifactIdentity(ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST,
                                 sha256=digest * 64, manifest_digest=digest * 64),
                LoaderType.OLLAMA, None, digest * 64)
            self.deployments[name] = deployment
            models[name] = ResolvedModel(name, deployment, None, None, "c" * 64)
        self.snapshot = ResolverSnapshot("c" * 64, self.deployments, models)
        self.model = models["target"]
        self.resolver = SimpleNamespace(snapshot=mock.AsyncMock(side_effect=lambda: self.snapshot))
        self.loaded = {"target"}
        self.calls, self.post_checks = [], []
        self.fail_post = self.fail_health = self.bad_digest = False
        self.resident_vram, self.resident_context = 1000, 4096
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False,
                                       transport=httpx.MockTransport(self.native))
        compatibility = OllamaCapabilities(**{f.name: CompatResult(True) for f in fields(OllamaCapabilities)})
        self.provider = OllamaProvider(client=self.client, resolver=self.resolver,
            implementation=RuntimeImplementation("offline-fixture", None, "offline-fixture"),
            compatibility=compatibility, expected_version="0.18.0")
        self.measure = lambda: MemorySnapshot(10000, 10000, time.monotonic())
        self.service = RuntimeService(resolver=self.resolver, providers={BackendType.OLLAMA: self.provider},
            admission=self.store, measure=self.measure, timeouts=RuntimeTimeouts(1, 1, 1, 1, 1, 1, 1))
        self.addAsyncCleanup(self.service.aclose)
        self.generation = generation_key(self.provider.generation)

    def context(self, request_id="unload"):
        return RequestContext(request_id, time.monotonic() + 3, asyncio.Event())

    def native(self, request):
        self.calls.append((request.method, request.url.path))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.18.0"})
        if request.url.path == "/api/ps":
            if self.fail_health:
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200, json={"models": [
                {"name": item.reference, "model": item.reference,
                 "digest": "f" * 64 if self.bad_digest else item.artifact_identity.sha256,
                 "size": 1000, "size_vram": self.resident_vram, "context_length": self.resident_context}
                for key, item in self.deployments.items() if key in self.loaded]})
        self.assertEqual((request.method, request.url.path), ("POST", "/api/generate"))
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], self.model.deployment.reference)
        self.assertEqual(payload["keep_alive"], 0)
        self.assertIs(payload["stream"], False)
        for check in self.post_checks:
            check()
        if self.fail_post:
            raise httpx.ReadTimeout("fixture lost reply", request=request)
        self.loaded.discard("target")
        return httpx.Response(200, json={"model": payload["model"], "done": True})

    def reserve(self, operation, *, deployment="target", owner="kiron-proxy", generation=None, kind="load"):
        ticket = self.store.reserve(operation_id=operation, owner=owner, generation=generation or self.generation,
            deployment_id=deployment, kind=kind, gpu_bytes=100 if kind == "load" else 0,
            host_bytes=100 if kind == "load" else 0, measure=self.measure, ttl_seconds=60,
            **self.service._lifecycle_identity(self.deployments[deployment]))
        if kind == "load":
            ticket = self.store.transition(operation, owner=owner, expected_generation=ticket.generation, phase="resident")
        return ticket

    def assert_no_post(self):
        self.assertFalse(any(method == "POST" for method, _ in self.calls))

    async def test_external_resident_unloads_without_fake_load_ticket_and_second_call_is_noop(self):
        def check_fence():
            tickets = self.store.snapshot()
            self.assertEqual(len(tickets), 1)
            self.assertEqual((tickets[0].kind, tickets[0].gpu_bytes, tickets[0].host_bytes), ("unload", 0, 0))
            with self.assertRaisesRegex(AdmissionError, "draining"):
                self.reserve("concurrent-request", kind="request")
        self.post_checks.append(check_fence)
        result = await self.service.unload(self.model, self.context())
        self.assertTrue(result.changed)
        self.assertEqual(self.loaded, set())
        self.assertEqual(self.store.snapshot(), ())
        again = await self.service.unload(self.model, self.context("again"))
        self.assertFalse(again.changed)
        self.assertEqual(self.calls.count(("POST", "/api/generate")), 1)
        self.assertEqual(self.store.snapshot(), ())

    async def test_target_end_releases_only_its_tickets_with_other_residents_preserved(self):
        self.loaded.add("other")
        self.reserve("target-load")
        other = self.reserve("other-load", deployment="other", owner="foreign")
        def check_draining():
            tickets = {t.operation_id: t for t in self.store.snapshot()}
            self.assertEqual(tickets["target-load"].phase, "resident")
            self.assertEqual(tickets["unload"].kind, "unload")
            self.assertEqual(tickets["other-load"], other)
        self.post_checks.append(check_draining)
        result = await self.service.unload(self.model, self.context())
        self.assertTrue(result.changed)
        self.assertEqual(self.loaded, {"other"})
        self.assertEqual(self.store.snapshot(), (other,))

    async def test_already_unloaded_is_noop_even_when_other_model_is_resident(self):
        self.loaded = {"other"}
        other = self.reserve("other-load", deployment="other")
        result = await self.service.unload(self.model, self.context())
        self.assertFalse(result.changed)
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), (other,))

    async def test_health_only_noop_does_not_release_unknown_queued_work(self):
        self.loaded.clear()
        self.reserve("request", kind="request")
        self.store.release("request", owner="kiron-proxy", generation=self.generation, confirmed_terminated=False)
        before = self.store.snapshot()
        result = await self.service.unload(self.model, self.context())
        self.assertFalse(result.changed)
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)

    async def test_foreign_owner_or_generation_prevents_native_mutation(self):
        for changes in ({"owner": "foreign"}, {"generation": "previous-generation"}):
            with self.subTest(changes=changes):
                ticket = self.reserve("foreign-load", **changes)
                before = self.store.snapshot()
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.unload(self.model, self.context())
                self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
                self.assert_no_post()
                self.assertEqual(self.store.snapshot(), before)
                self.store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                                   confirmed_terminated=True)

    async def test_transport_uncertainty_keeps_target_tickets_and_other_residents(self):
        self.reserve("load")
        other = self.reserve("other-load", deployment="other")
        self.fail_post = True
        with self.assertRaises(LocalInferenceError):
            await self.service.unload(self.model, self.context())
        tickets = {t.operation_id: t for t in self.store.snapshot()}
        self.assertEqual(tickets["load"].phase, "resident")
        self.assertEqual(tickets["unload"].phase, "unknown")
        self.assertEqual(tickets["other-load"], other)
        self.assertIn("target", self.loaded)

    async def test_active_or_unknown_request_blocks_unload_and_only_new_fence_is_released(self):
        self.service.timeouts = replace(self.service.timeouts, drain=.03)
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                self.reserve("request", kind="request")
                if unknown:
                    self.store.release("request", owner="kiron-proxy", generation=self.generation,
                                       confirmed_terminated=False)
                before = self.store.snapshot()
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.unload(self.model, self.context())
                self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
                self.assert_no_post()
                self.assertEqual(self.store.snapshot(), before)
                self.store.release("request", owner="kiron-proxy", generation=self.generation,
                                   confirmed_terminated=True)

    async def test_drain_cancellation_does_not_change_original_load_or_request(self):
        self.reserve("load")
        self.reserve("request", kind="request")
        before = self.store.snapshot()
        context = self.context()
        pending = asyncio.create_task(self.service.unload(self.model, context))
        try:
            for _ in range(100):
                if any(t.kind == "unload" for t in self.store.snapshot()):
                    break
                await asyncio.sleep(.005)
            else:
                self.fail("unload did not acquire its fence")
            context.cancellation.set()
            with self.assertRaises(LocalInferenceError) as raised:
                await pending
            self.assertEqual(raised.exception.failure.code, ErrorCode.CANCELLED)
            self.assert_no_post()
            self.assertEqual(self.store.snapshot(), before)
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def prepared_operation(self, request_id="prepared", model=None):
        model = model or self.model
        deployment = model.deployment
        self.resident_vram, self.resident_context = 0, 1024
        evidence = CapabilityEvidence("offline-fixture", deployment.artifact_identity.fingerprint, "a" * 64,
            None, None, "offline-fixture", deployment.configuration_fingerprint, "offline-fixture", datetime.now(timezone.utc))
        self.provider._capabilities[deployment.id] = CapabilitySet({CapabilityName.CHAT: Capability(
            CapabilityStatus.SUPPORTED, {"max_output_tokens": ParameterConstraint(allowed_values=(8,)),
                "roles": ParameterConstraint(allowed_values=("user",)),
                "context_tokens": ParameterConstraint(allowed_values=(1024,)),
                "device": ParameterConstraint(allowed_values=("cpu",))}, (evidence,))})
        request = InferenceRequest(model, (Message(MessageRole.USER, (TextPart("Hello"),)),),
                                   GenerationOptions(8), self.context(request_id))
        return await self.service.prepare(request)

    async def test_prepared_request_prevents_unload_until_its_own_confirmed_close(self):
        operation = await self.prepared_operation()
        self.assertFalse(operation.backend_started)
        self.assert_no_post()
        before = self.store.snapshot()
        self.service.timeouts = replace(self.service.timeouts, drain=.03)
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.unload(self.model, self.context())
        self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)
        await operation.close()
        self.assertEqual(self.store.snapshot(), ())
        result = await self.service.unload(self.model, self.context("after-close"))
        self.assertTrue(result.changed)
        self.assertEqual(self.calls.count(("POST", "/api/generate")), 1)

    async def test_disconnected_canonical_stream_drains_only_until_proven_end_or_deadline(self):
        from kiron_common.gpu_admission.ollama_backend import BackendLock
        for ending in ("complete", "late_error", "timeout"):
            with self.subTest(ending=ending):
                ready, finish = asyncio.Event(), asyncio.Event()
                class Stream(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield b'{"model":"target:latest","message":{"role":"assistant","content":"hello"},"done":false}\n'
                        ready.set()
                        await finish.wait()
                        yield b'{"model":"target:latest","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}\n'
                        if ending == "late_error":
                            raise httpx.ReadError("fixture lost EOF")
                async def native(request):
                    if request.url.path == "/api/chat":
                        return httpx.Response(200, stream=Stream())
                    return self.native(request)
                self.provider.client = httpx.AsyncClient(base_url="http://127.0.0.1:11435",
                    transport=httpx.MockTransport(native))
                self.addAsyncCleanup(self.provider.client.aclose)
                self.service.timeouts = replace(self.service.timeouts, drain=.04)
                operation = await self.prepared_operation(ending)
                capabilities = self.provider._capabilities[self.model.deployment.id]
                self.provider._capabilities[self.model.deployment.id] = CapabilitySet({
                    **capabilities.by_name,
                    CapabilityName.STREAMING: Capability(CapabilityStatus.SUPPORTED, {},
                        capabilities.by_name[CapabilityName.CHAT].evidence)})
                events = operation.events()
                async for event in events:
                    if event.kind is EventKind.TEXT_DELTA:
                        break
                await asyncio.wait_for(ready.wait(), 1)
                operation.request.context.cancellation.set()
                await events.aclose()
                closing = asyncio.create_task(operation.close())
                await asyncio.sleep(0)
                with self.assertRaises(AdmissionError):
                    BackendLock(self.store, exclusive=True).acquire()
                if ending != "timeout":
                    finish.set()
                await asyncio.wait_for(closing, 1)
                if ending == "complete":
                    self.assertEqual(self.store.snapshot(), ())
                else:
                    ticket, = self.store.snapshot()
                    self.assertEqual(ticket.phase, "unknown")
                    self.store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                                       confirmed_terminated=True)
                lock = BackendLock(self.store, exclusive=True).acquire()
                lock.close()
        self.assertEqual(self.store.snapshot(), ())

    async def test_cancel_during_unload_transaction_releases_only_new_unstarted_fence(self):
        self.reserve("load")
        before = self.store.snapshot()
        entered, complete = threading.Event(), threading.Event()
        original = self.store.begin_unload
        def delayed(*args, **kwargs):
            entered.set()
            if not complete.wait(2):
                raise RuntimeError("test transaction did not finish")
            return original(*args, **kwargs)
        with mock.patch.object(self.store, "begin_unload", side_effect=delayed):
            pending = asyncio.create_task(self.service.unload(self.model, self.context()))
            await asyncio.to_thread(entered.wait, 1)
            self.assertTrue(entered.is_set())
            pending.cancel()
            complete.set()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)

    async def test_duplicate_unload_operation_cannot_own_or_release_an_existing_fence(self):
        self.store.begin_unload("unload", owner="kiron-proxy", generation=self.generation,
                                deployment_id="target", require_resident=False)
        before = self.store.snapshot()
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.unload(self.model, self.context())
        self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)
        with self.assertRaises(LocalInferenceError):
            await self.service.unload(self.model, self.context("another-unload"))
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)

    async def test_unavailable_health_and_unknown_target_identity_never_mutate(self):
        for flag in ("fail_health", "bad_digest"):
            with self.subTest(flag=flag):
                setattr(self, flag, True)
                with self.assertRaises(LocalInferenceError):
                    await self.service.unload(self.model, self.context())
                self.assert_no_post()
                self.assertEqual(self.store.snapshot(), ())
                setattr(self, flag, False)

    async def test_stale_resolver_revision_fails_before_health_or_ticket(self):
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.unload(replace(self.model, snapshot_revision="d" * 64), self.context())
        self.assertEqual(raised.exception.failure.code, ErrorCode.CONFLICT)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.snapshot(), ())

    def native_operation(self, *, cpu=False, unload=False, name="arbitrary-alias"):
        body = {"model": name, "stream": False if unload else True}
        if cpu:
            body["options"] = {"num_gpu": 0}
        if unload:
            body["keep_alive"] = 0
        return NativeRequestOperation(backend="ollama", endpoint="/api/generate" if unload else "/api/chat",
            model=name, body=json.dumps(body).encode(), store=self.store, measure=self.measure)

    def native_client(self, handler=None):
        calls = []
        def native(request):
            calls.append(request)
            return httpx.Response(200, content=b'{"done":true,"message":{"content":"OK"}}\n')
        client = httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False,
                                   transport=httpx.MockTransport(handler or native))
        self.addAsyncCleanup(client.aclose)
        return client, calls

    async def native_blocks_unload(self, *, cpu=False, unknown=False):
        self.service.timeouts = replace(self.service.timeouts, drain=.03)
        client, native_calls = self.native_client()
        with mock.patch("native_admission.vram_lease.num_gpu_zero_effective", return_value=True):
            operation = self.native_operation(cpu=cpu)
        self.addAsyncCleanup(operation.close)
        await operation.send(client, client.build_request("POST", "/api/chat"))
        if unknown:
            # A clean, complete backend reply is now drained after disconnect.
            # Model genuinely unconfirmed work with a truncated native reply.
            operation.response = httpx.Response(200, content=b'{"done":false}\n')
            await operation.close()
        before = self.store.snapshot()
        self.assertEqual(before[0].gpu_guard, not cpu)
        with self.assertRaises(LocalInferenceError):
            await self.service.unload(self.model, self.context())
        self.assert_no_post()
        self.assertEqual(len(native_calls), 1)
        self.assertEqual(self.store.snapshot(), before)

    async def test_native_active_alias_protects_canonical_unload(self):
        await self.native_blocks_unload()

    async def test_native_unknown_alias_protects_canonical_unload(self):
        await self.native_blocks_unload(unknown=True)

    async def test_native_cpu_active_alias_protects_canonical_unload(self):
        await self.native_blocks_unload(cpu=True)

    async def test_native_cpu_unknown_alias_protects_canonical_unload(self):
        await self.native_blocks_unload(cpu=True, unknown=True)

    async def test_native_owner_eof_drains_then_canonical_unload_can_proceed(self):
        client, _ = self.native_client()
        operation = self.native_operation(name="target")
        response = await operation.send(client, client.build_request("POST", "/api/chat"))
        pending = asyncio.create_task(self.service.unload(self.model, self.context()))
        try:
            for _ in range(100):
                if any(t.kind == "unload" for t in self.store.snapshot()):
                    break
                await asyncio.sleep(.005)
            else:
                self.fail("missing fence")
            self.assert_no_post()
            self.assertTrue([chunk async for chunk in operation.chunks(response)])
            await operation.close()
            self.assertTrue((await pending).changed)
            self.assertEqual(self.store.snapshot(), ())
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await operation.close()

    async def test_native_cpu_and_gpu_cannot_send_inside_canonical_unload_fence(self):
        native_client, native_calls = self.native_client()
        async def native(request):
            if request.method == "POST":
                for cpu in (False, True):
                    with mock.patch("native_admission.vram_lease.num_gpu_zero_effective", return_value=True):
                        operation = self.native_operation(cpu=cpu)
                    with self.assertRaises(AdmissionError):
                        await operation.send(native_client, native_client.build_request("POST", "/api/chat"))
                    await operation.close()
            return self.native(request)
        self.provider.client = self.native_client(native)[0]
        self.assertTrue((await self.service.unload(self.model, self.context())).changed)
        self.assertEqual(native_calls, [])
        self.assertEqual(self.store.snapshot(), ())

    async def test_canonical_prepared_request_blocks_native_implicit_mutations_and_unload(self):
        prepared = await self.prepared_operation()
        before = self.store.snapshot()
        client, calls = self.native_client()
        for cpu in (False, True):
            with mock.patch("native_admission.vram_lease.num_gpu_zero_effective", return_value=True):
                operation = self.native_operation(cpu=cpu)
            with self.assertRaises(AdmissionError):
                await operation.send(client, client.build_request("POST", "/api/chat"))
            await operation.close()
        unload = self.native_operation(unload=True)
        pending = asyncio.create_task(unload.send(client, client.build_request("POST", "/api/generate")))
        try:
            for _ in range(100):
                if any(t.kind == "unload" for t in self.store.snapshot()):
                    break
                await asyncio.sleep(.005)
            else:
                self.fail("native unload did not fence")
            self.assertEqual(calls, [])
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            await unload.close()
            self.assertEqual(self.store.snapshot(), before)
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await prepared.close()

    async def test_native_unload_fence_blocks_new_canonical_prepared_request(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        async def native(request):
            entered.set()
            await finish.wait()
            return httpx.Response(200, content=b'{"done":true}')
        client, _ = self.native_client(native)
        operation = self.native_operation(unload=True)
        pending = asyncio.create_task(operation.send(client, client.build_request("POST", "/api/generate")))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            with self.assertRaises(LocalInferenceError):
                await self.prepared_operation()
            self.assert_no_post()
            self.assertEqual([t.kind for t in self.store.snapshot()], ["unload"])
            finish.set()
            response = await pending
            self.assertTrue([part async for part in operation.chunks(response)])
            await operation.close()
            self.assertEqual(self.store.snapshot(), ())
        finally:
            finish.set()
            await pending
            await operation.close()

    async def test_unknown_canonical_request_prevents_native_unload_without_ticket_adoption(self):
        prepared = await self.prepared_operation()
        prepared.backend_started = True  # No native end proof exists.
        await prepared.close()
        before = self.store.snapshot()
        self.assertEqual(before[0].phase, "unknown")
        client, calls = self.native_client()
        operation = self.native_operation(unload=True)
        with self.assertRaises(AdmissionError):
            await operation.send(client, client.build_request("POST", "/api/generate"))
        await operation.close()
        self.assertEqual(calls, [])
        self.assertEqual(self.store.snapshot(), before)

    async def test_other_profile_same_artifact_resident_cannot_be_adopted_for_unload(self):
        first = self.reserve("resident")
        alias = replace(self.model.deployment, id="profile-two", reference="alias:latest",
            artifact_identity=replace(self.model.deployment.artifact_identity,
                                      manifest_digest="sha256:" + "a" * 64))
        alias_model = replace(self.model, public_model_id="profile-two", deployment=alias)
        self.deployments[alias.id] = alias
        self.loaded.add(alias.id)
        self.snapshot = ResolverSnapshot(self.snapshot.revision, self.deployments,
                                        {"target": self.model, "profile-two": alias_model})
        with self.assertRaises(LocalInferenceError):
            await self.service.unload(alias_model, self.context())
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), (first,))

    async def test_native_request_blocks_measured_canonical_coldload_before_backend_post(self):
        client, _ = self.native_client()
        operation = self.native_operation()
        await operation.send(client, client.build_request("POST", "/api/chat"))
        self.addAsyncCleanup(operation.close)
        self.loaded.clear()
        profile = ResourceProfile("measured-fixture", 1024, 128, 128, 1, 2, 0, False, 100, 100)
        deployment = replace(self.model.deployment, resource_profile=profile)
        model = replace(self.model, deployment=deployment)
        self.deployments[deployment.id] = deployment
        self.snapshot = ResolverSnapshot(self.snapshot.revision, self.deployments, {"target": model})
        before = self.store.snapshot()
        with self.assertRaises(LocalInferenceError):
            await self.service.load(model, self.context())
        self.assert_no_post()
        self.assertEqual(self.store.snapshot(), before)

    async def test_two_profiles_share_one_canonical_request_slot(self):
        first = await self.prepared_operation()
        second_deployment = replace(self.model.deployment, id="second-deployment")
        second_model = replace(self.model, public_model_id="second-public-profile", deployment=second_deployment)
        self.deployments[second_deployment.id] = second_deployment
        self.snapshot = ResolverSnapshot(self.snapshot.revision, self.deployments,
            {self.model.public_model_id: self.model, second_model.public_model_id: second_model})
        try:
            with self.assertRaises(LocalInferenceError) as raised:
                await self.prepared_operation("second", second_model)
            self.assertEqual(raised.exception.failure.code, ErrorCode.OVERLOADED)
            self.assertEqual(len(self.store.snapshot()), 1)
            self.assert_no_post()
        finally:
            await first.close()
        self.assertEqual(self.store.snapshot(), ())

    async def test_native_cannot_invalidate_a_managed_resident_via_request_or_unload(self):
        resident = self.reserve("resident")
        client, calls = self.native_client()
        for unload in (False, True):
            operation = self.native_operation(unload=unload, name="target")
            with self.assertRaises(AdmissionError):
                await operation.send(client, client.build_request("POST", operation.endpoint))
            await operation.close()
            self.assertEqual(self.store.snapshot(), (resident,))
        self.assertEqual(calls, [])

    async def test_missing_service_controller_is_unsupported_before_health_or_reservation(self):
        for method in (self.service.start, self.service.stop):
            with self.assertRaises(LocalInferenceError) as raised:
                await method(BackendType.OLLAMA, self.context())
            self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.store.snapshot(), ())

    async def test_native_work_prevents_injected_service_start_and_stop(self):
        self.provider.service_control = SimpleNamespace(start=mock.AsyncMock(), stop=mock.AsyncMock())
        client, _ = self.native_client()
        operation = self.native_operation(cpu=True)
        await operation.send(client, client.build_request("POST", "/api/chat"))
        self.addAsyncCleanup(operation.close)
        before = self.store.snapshot()
        for method in (self.service.start, self.service.stop):
            context = replace(self.context(), deadline_monotonic=time.monotonic() + .04)
            with self.assertRaises(LocalInferenceError):
                await method(BackendType.OLLAMA, context)
            self.assertEqual(self.store.snapshot(), before)
        self.provider.service_control.start.assert_not_awaited()
        self.provider.service_control.stop.assert_not_awaited()

    async def test_service_transitions_fence_native_send_and_release_only_after_confirmation(self):
        client, calls = self.native_client()
        async def transition():
            operation = self.native_operation()
            with self.assertRaises(AdmissionError):
                await operation.send(client, client.build_request("POST", "/api/chat"))
            await operation.close()
            self.assertEqual([t.kind for t in self.store.snapshot()], ["unload"])
            return True
        self.provider.service_control = SimpleNamespace(start=mock.AsyncMock(side_effect=transition),
                                                        stop=mock.AsyncMock(side_effect=transition))
        for method in (self.service.start, self.service.stop):
            result = await method(BackendType.OLLAMA, self.context())
            self.assertTrue(result.changed)
            self.assertEqual(self.store.snapshot(), ())
        self.assertEqual(calls, [])
        self.provider.service_control.start.assert_awaited_once()
        self.provider.service_control.stop.assert_awaited_once()

    async def test_uncertain_service_stop_keeps_its_own_mutation_fence_unknown(self):
        async def uncertain():
            raise LocalInferenceError(RuntimeFailure(ErrorCode.TIMEOUT, "fixed controller has no end proof"))
        self.provider.service_control = SimpleNamespace(stop=mock.AsyncMock(side_effect=uncertain))
        with self.assertRaises(LocalInferenceError):
            await self.service.stop(BackendType.OLLAMA, self.context())
        ticket, = self.store.snapshot()
        self.assertEqual((ticket.owner, ticket.kind, ticket.phase, ticket.lifecycle_domain),
                         ("kiron-proxy", "unload", "unknown", "ollama"))
        self.provider.service_control.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
