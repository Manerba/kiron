"""Offline lifecycle tests: real tiny immutable files, fake resolver/admission/child."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from kiron_common.local_inference import ArtifactIdentity, ResolvedDeployment, ResourceProfile, RuntimeGeneration
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

from controller import Command, Controller, ControlError
from main import ControlApp
from kiron_common.prism_runtime_policy import Policy, PolicyError, Profile


PROFILE_ID = "prism-bonsai27b-cuda40-c1024-v1"


def gguf(architecture="qwen35"):
    def text(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded
    return (b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
            + text("general.architecture") + struct.pack("<I", 8) + text(architecture)
            + text(architecture + ".embedding_length") + struct.pack("<II", 4, 1024))


class FakeChild:
    def __init__(self):
        self.running, self.is_ready, self.stop_result = True, True, True
        self.stops = 0

    def alive(self):
        return self.running

    async def ready(self, alias):
        await asyncio.sleep(0)
        return self.is_ready

    async def stop(self):
        self.stops += 1
        if self.stop_result:
            self.running = False
        return self.stop_result

    async def slots(self):
        return {"active_requests": 0, "slot_task_id": 5, "slots_observed_at": 1}


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        bundle = self.root / "bundle"
        bundle.mkdir()
        binary = bundle / "llama-server"
        binary.write_bytes(b"non-executable fixture content")
        binary.chmod(0o755)
        models = self.root / "models"
        models.mkdir()
        model = models / "tiny.gguf"
        model.write_bytes(gguf())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        profile = Profile(PROFILE_ID, 40, 4, digest, "qwen35", 4608 << 20, 8192 << 20, 512 << 20)
        self.policy = Policy(bundle, binary, {"llama-server": {"sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}},
                             (bundle,), (models,), {PROFILE_ID: profile}, anchor=self.root,
                             startup_timeout=0.3, health_timeout=0.1, drain_timeout=0.1,
                             term_timeout=0.1, kill_timeout=0.1)
        artifact = ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, digest, model.stat().st_size)
        resources = ResourceProfile(PROFILE_ID, 1024, 128, 128, 1, 4, 40, False,
                                    4608 << 20, 8192 << 20, 512 << 20)
        self.deployment = ResolvedDeployment("fixture.deployment", BackendType.PRISM, str(model), artifact,
                                             LoaderType.PRISM_GGUF, resources, "a" * 64)
        self.snapshot = SimpleNamespace(revision="b" * 64, resolve_deployment=lambda value: self.deployment)
        self.resolver = SimpleNamespace(snapshot=mock.AsyncMock(side_effect=lambda: self.snapshot))
        self.admission = SimpleNamespace(**{name: mock.AsyncMock() for name in
                            ("validate", "loaded", "drain", "terminated", "heartbeat", "rejected")})
        self.child = FakeChild()
        self.factory = mock.Mock(return_value=self.child)
        self.controller = Controller(self.policy, self.resolver, self.admission,
                                     child_factory=self.factory, port_check=mock.Mock())
        self.addAsyncCleanup(self.controller.close)

    def command(self, operation="load-1", **kwargs):
        return Command(self.deployment.id, self.snapshot.revision,
                       kwargs.get("generation", self.controller.generation), operation)

    async def test_load_idempotence_and_no_stale_loaded_replay_after_unload(self):
        command = self.command()
        loaded = await self.controller.mutate("load", command)
        self.assertEqual(loaded["state"], "loaded")
        self.assertNotEqual(loaded["generation"], asdict(command.expected_generation))
        await self.controller.mutate("load", command)
        self.factory.assert_called_once()
        self.admission.validate.assert_awaited_once_with(command.operation_id, self.deployment,
                                                        command.expected_generation, "load")
        self.admission.loaded.assert_awaited_once()
        await self.controller.mutate("unload", self.command("unload-1"))
        replay = await self.controller.mutate("load", command)
        self.assertEqual(replay["state"], "unloaded")
        self.assertEqual(self.child.stops, 1)
        self.admission.terminated.assert_awaited_once()

    async def test_generation_and_operation_conflicts_do_not_spawn_or_stop(self):
        old = self.command()
        await self.controller.mutate("load", old)
        for command in (self.command("old", generation=old.expected_generation),
                        replace(old, deployment_id="different")):
            with self.assertRaises(ControlError):
                await self.controller.mutate("unload", command)
        self.assertEqual(self.child.stops, 0)
        self.factory.assert_called_once()

    async def test_registered_alias_resolves_to_same_canonical_deployment_on_unload(self):
        original = self.snapshot.resolve_deployment
        self.snapshot.resolve_deployment = mock.Mock(side_effect=original)
        await self.controller.mutate("load", self.command())
        result = await self.controller.mutate("unload", replace(self.command("alias-unload"),
                                                               deployment_id="registry.alias"))
        self.assertEqual(result["state"], "unloaded")
        self.snapshot.resolve_deployment.assert_called_with("registry.alias")
        self.admission.validate.assert_awaited_with("alias-unload", self.deployment,
                                                   self.controller.generation, "unload")
        self.assertEqual(self.child.stops, 1)

    async def test_concurrent_same_operation_joins_other_operation_conflicts(self):
        self.child.is_ready = False
        first = asyncio.create_task(self.controller.mutate("load", self.command()))
        while not self.factory.called:
            await asyncio.sleep(0.001)
        duplicate = asyncio.create_task(self.controller.mutate("load", self.command(generation=RuntimeGeneration(
                         self.controller.generation.boot_id, None))))
        with self.assertRaisesRegex(ControlError, "resource_busy"):
            await self.controller.mutate("load", self.command("competing"))
        self.child.is_ready = True
        results = await asyncio.gather(first, duplicate)
        self.assertTrue(all(result["state"] == "loaded" for result in results))
        self.factory.assert_called_once()

    async def test_other_deployment_and_changed_profile_are_rejected(self):
        await self.controller.mutate("load", self.command())
        self.deployment = replace(self.deployment, id="second")
        with self.assertRaisesRegex(ControlError, "resource_busy"):
            await self.controller.mutate("load", self.command("second"))
        self.factory.assert_called_once()

    async def test_snapshot_change_during_validation_rejects_before_admission(self):
        original = self.snapshot
        changed = SimpleNamespace(revision="c" * 64)
        self.resolver.snapshot.side_effect = [original, changed]
        with self.assertRaisesRegex(ControlError, "snapshot_conflict"):
            await self.controller.mutate("load", self.command())
        self.factory.assert_not_called()
        self.admission.validate.assert_not_awaited()

    async def test_foreign_port_prevents_spawn(self):
        self.controller.port_check.side_effect = OSError("occupied")
        command = self.command()
        with self.assertRaisesRegex(ControlError, "backend_port_in_use"):
            await self.controller.mutate("load", command)
        self.factory.assert_not_called()
        self.admission.validate.assert_not_awaited()
        self.admission.rejected.assert_awaited_once_with(command.operation_id, self.deployment.id,
                                                       command.expected_generation)

    async def test_admission_rejection_does_not_spawn_or_release_someone_elses_reservation(self):
        self.admission.validate.side_effect = RuntimeError("not admitted")
        with self.assertRaisesRegex(ControlError, "provider_unavailable"):
            await self.controller.mutate("load", self.command())
        self.factory.assert_not_called()
        self.admission.terminated.assert_not_awaited()
        self.assertEqual(self.controller.state, "unloaded")

    async def test_readiness_timeout_stops_child_then_releases(self):
        self.child.is_ready = False
        with self.assertRaisesRegex(ControlError, "timeout"):
            await self.controller.mutate("load", self.command())
        self.assertEqual(self.child.stops, 1)
        self.assertEqual(self.controller.state, "unloaded")
        self.admission.loaded.assert_not_awaited()
        self.admission.terminated.assert_awaited_once()

    async def test_uncertain_cleanup_never_releases_resident(self):
        await self.controller.mutate("load", self.command())
        self.child.stop_result = False
        with self.assertRaisesRegex(ControlError, "process_cleanup_unconfirmed"):
            await self.controller.mutate("unload", self.command("stop"))
        self.assertEqual(self.controller.state, "unknown")
        self.admission.terminated.assert_not_awaited()
        self.child.stop_result = True

    async def test_crash_is_reconciled_and_reaped_without_false_loaded(self):
        await self.controller.mutate("load", self.command())
        self.child.running = False
        state = await self.controller.health()
        self.assertEqual(state["state"], "unloaded")
        self.assertIsNone(state["deployment_id"])
        self.assertEqual(self.child.stops, 1)
        self.admission.terminated.assert_awaited_once()

    async def test_health_loss_is_unknown_and_never_renews_lease(self):
        await self.controller.mutate("load", self.command())
        self.child.is_ready = False
        result = await self.controller.health()
        self.assertEqual(result["state"], "unknown")
        self.admission.heartbeat.assert_not_awaited()
        self.child.is_ready = True
        self.assertEqual((await self.controller.health())["state"], "loaded")
        self.admission.heartbeat.assert_awaited_once()

    async def test_slot_evidence_requires_current_healthy_generation(self):
        await self.controller.mutate("load", self.command())
        result = await self.controller.health()
        self.assertEqual(result["active_requests"], 0)
        self.assertEqual(result["slot_task_id"], 5)
        self.child.is_ready = False
        result = await self.controller.health()
        self.assertIsNone(result["active_requests"])
        self.assertIsNone(result["slots_observed_at"])

    async def test_restart_changes_boot_and_does_not_clear_existing_shared_residents(self):
        other = Controller(self.policy, self.resolver, self.admission, child_factory=self.factory)
        self.assertNotEqual(other.generation.boot_id, self.controller.generation.boot_id)
        self.assertEqual((await other.health())["state"], "unloaded")
        self.admission.terminated.assert_not_awaited()
        self.factory.assert_not_called()

    async def test_client_cancellation_does_not_orphan_operation(self):
        self.child.is_ready = False
        request = asyncio.create_task(self.controller.mutate("load", self.command()))
        while not self.factory.called:
            await asyncio.sleep(0.001)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.child.is_ready = True
        await self.controller._active
        self.assertEqual(self.controller.state, "loaded")

    async def test_drain_timeout_leaves_owned_child_and_resident_intact(self):
        await self.controller.mutate("load", self.command())
        async def blocked(*args):
            await asyncio.sleep(1)
        self.admission.drain.side_effect = blocked
        with self.assertRaisesRegex(ControlError, "timeout"):
            await self.controller.mutate("unload", self.command("stop"))
        self.assertEqual(self.child.stops, 0)
        self.admission.terminated.assert_not_awaited()
        self.admission.drain.side_effect = None

    async def test_profile_digest_resource_and_architecture_mismatches_reject_before_spawn(self):
        original = self.deployment
        variants = [replace(original, artifact_identity=replace(original.artifact_identity, sha256="f" * 64)),
                    replace(original, resource_profile=replace(original.resource_profile, gpu_memory_bytes=0))]
        for index, variant in enumerate(variants):
            self.deployment = variant
            with self.assertRaisesRegex(ControlError, "unsupported_profile"):
                await self.controller.mutate("load", self.command(f"bad-{index}"))
        self.deployment = original
        self.controller.policy = replace(self.policy, profiles={PROFILE_ID:
                                        replace(self.policy.profiles[PROFILE_ID], architecture="wrong")})
        with self.assertRaisesRegex(ControlError, "model_unavailable"):
            await self.controller.mutate("load", self.command("bad-architecture"))
        self.factory.assert_not_called()

    async def test_argv_uses_verified_descriptors_fixed_flags_and_clean_environment(self):
        with mock.patch.dict(os.environ, {"LLAMA_ARG_AGENT": "1", "HF_TOKEN": "fixture", "LD_PRELOAD": "evil"}):
            await self.controller.mutate("load", self.command())
        argv, env, descriptors, policy = self.factory.call_args.args
        self.assertTrue(argv[argv.index("--model") + 1].startswith("/proc/self/fd/"))
        for flag in ("--no-agent", "--no-webui", "--no-ui-mcp-proxy", "--no-mmproj"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--parallel") + 1], "1")
        self.assertEqual(argv[argv.index("--threads-batch") + 1], "4")
        self.assertEqual(set(env), {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH"})
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    async def test_wire_rejects_paths_flags_extra_fields_and_oversize_before_lifecycle(self):
        app = ControlApp(self.controller)
        value = asdict(self.command())
        for body, expected_status in ((json.dumps({**value, "argv": ["--agent"]}).encode(), 400),
                                      (json.dumps({**value, "deployment_id": "/tmp/model"}).encode(), 400),
                                      (b"x" * 4097, 413)):
            sent = []
            await app({"type": "http", "method": "POST", "path": "/load"},
                      mock.AsyncMock(return_value={"type": "http.request", "body": body}),
                      mock.AsyncMock(side_effect=sent.append))
            self.assertEqual(sent[0]["status"], expected_status)
        self.factory.assert_not_called()

    async def test_runtime_provider_rejected_load_releases_ticket_and_retry_succeeds(self):
        await self._runtime_rejection(transport_lost=False)

    async def test_lost_load_transport_retains_unknown_reservation(self):
        await self._runtime_rejection(transport_lost=True)

    async def _runtime_rejection(self, *, transport_lost):
        import httpx
        from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
        from kiron_common.local_inference import (
            LocalInferenceError, RequestContext, ResolvedModel, ResolverSnapshot, RuntimeImplementation, RuntimeTimeouts,
        )
        from admission import ControllerAdmission
        from prism_provider import PrismProvider
        from runtime_service import RuntimeService
        # This integration path uses real flock/fsync transactions, unlike the
        # instant admission mocks above. Their 100-ms budgets are not disk I/O
        # guarantees on a shared test host; production deadlines stay unchanged.
        self.controller.policy = replace(self.policy, startup_timeout=2, health_timeout=2,
                                         drain_timeout=2, kill_timeout=2)
        directory = self.root / 'admission'
        directory.mkdir(mode=0o2770)
        directory.chmod(0o2770)
        store = AdmissionStore(directory, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        self.controller.admission = ControllerAdmission(store)
        model = ResolvedModel('public', self.deployment, None, None, self.snapshot.revision)
        self.snapshot = ResolverSnapshot(self.snapshot.revision, {self.deployment.id: self.deployment}, {'public': model})
        transport = httpx.ASGITransport(app=ControlApp(self.controller))
        class LostLoad(httpx.AsyncBaseTransport):
            async def handle_async_request(inner, request):
                if request.url.path == '/load':
                    raise httpx.ReadError('unknown whether controller accepted', request=request)
                return await transport.handle_async_request(request)
            async def aclose(inner):
                await transport.aclose()
        provider = PrismProvider(control=httpx.AsyncClient(base_url='http://prism-control', trust_env=False,
                transport=LostLoad() if transport_lost else transport),
            inference=httpx.AsyncClient(base_url='http://127.0.0.1:18089', trust_env=False,
                transport=httpx.MockTransport(lambda _: self.fail('unexpected inference I/O'))),
            resolver=self.resolver, implementation=RuntimeImplementation('fixture', None, None))
        service = RuntimeService(resolver=self.resolver, providers={BackendType.PRISM: provider}, admission=store,
            measure=lambda: MemorySnapshot(16 << 30, 32 << 30, time.monotonic()),
            timeouts=RuntimeTimeouts(1, 1, 1, 1, 1, 1, 1))
        self.addAsyncCleanup(service.aclose)
        def context(name):
            return RequestContext(name, time.monotonic() + 5, asyncio.Event())
        self.controller.port_check.side_effect = OSError('occupied')
        with self.assertRaises(LocalInferenceError):
            await service.load(model, context('rejected-load'))
        self.factory.assert_not_called()
        if transport_lost:
            ticket, = store.snapshot()
            self.assertEqual((ticket.operation_id, ticket.phase), ('rejected-load', 'unknown'))
        else:
            self.assertEqual(store.snapshot(), ())
            self.controller.port_check.side_effect = None
            result = await service.load(model, context('retry-load'))
            self.assertEqual(result.observation.models[self.deployment.id].state.value, 'loaded')
            self.assertEqual(store.snapshot()[0].phase, 'resident')
            await service.unload(model, context('unload'))
            self.assertEqual(store.snapshot(), ())


if __name__ == "__main__":
    unittest.main()
