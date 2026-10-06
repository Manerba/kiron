import asyncio
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import RuntimeGeneration

from admission import ControllerAdmission, generation_key


class ControllerAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        root.chmod(0o2770)
        self.store = AdmissionStore(root, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        self.admission = ControllerAdmission(self.store)
        self.initial = RuntimeGeneration("boot")
        self.child = RuntimeGeneration("boot", "child")
        self.deployment = SimpleNamespace(id="model", resource_profile=SimpleNamespace(gpu_memory_bytes=40, host_memory_bytes=20))

    def reserve(self, operation="load", **kwargs):
        args = dict(operation_id=operation, owner="kiron-proxy", generation=generation_key(self.initial),
                    deployment_id="model", kind="load", gpu_bytes=40, host_bytes=20, resident_slot="prism",
                    measure=lambda: MemorySnapshot(100, 100, time.monotonic()))
        args.update(kwargs)
        return self.store.reserve(**args)

    async def loaded(self):
        self.reserve()
        await self.admission.validate("load", self.deployment, self.initial, "load")
        await self.admission.loaded("load", self.deployment, self.child)

    async def test_load_requires_owned_reservation_and_sufficient_resources(self):
        with self.assertRaises(AdmissionError):
            await self.admission.validate("load", self.deployment, self.initial, "load")
        self.reserve(gpu_bytes=39)
        with self.assertRaises(AdmissionError):
            await self.admission.validate("load", self.deployment, self.initial, "load")

    async def test_loaded_changes_generation_and_confirmed_end_clears(self):
        await self.loaded()
        await self.admission.heartbeat("load", self.deployment, self.child)
        self.assertEqual(self.store.snapshot()[0].generation, generation_key(self.child))
        await self.admission.terminated("load", self.deployment, self.child)
        self.assertEqual(self.store.snapshot(), ())

    async def test_failed_load_cleans_original_generation_only_after_confirmed_end(self):
        self.reserve()
        await self.admission.validate("load", self.deployment, self.initial, "load")
        await self.admission.terminated("load", self.deployment, self.child)
        self.assertEqual(self.store.snapshot(), ())

    async def test_unload_stops_new_requests_and_drains_existing_requests(self):
        await self.loaded()
        self.reserve("request", generation=generation_key(self.child), kind="request", gpu_bytes=0, host_bytes=0)
        self.store.begin_unload("unload", owner="kiron-proxy", generation=generation_key(self.child), deployment_id="model")
        await self.admission.validate("unload", self.deployment, self.child, "unload")
        with self.assertRaisesRegex(AdmissionError, "runtime slot"):
            self.reserve("next", generation=generation_key(self.child), kind="request", gpu_bytes=0, host_bytes=0)
        with self.assertRaises(AdmissionError):
            await self.admission.drain("unload", self.deployment, self.child, time.monotonic())
        self.store.release("request", owner="kiron-proxy", generation=generation_key(self.child), confirmed_terminated=True)
        await self.admission.drain("unload", self.deployment, self.child, time.monotonic())
        await self.admission.terminated("unload", self.deployment, self.child)
        self.assertEqual(self.store.snapshot(), ())

    async def test_controller_restart_cannot_clear_previous_controller_work(self):
        await self.loaded()
        replacement = ControllerAdmission(self.store)
        with self.assertRaises(AdmissionError):
            await replacement.terminated("load", self.deployment, RuntimeGeneration("new-boot", "new-child"))
        self.assertEqual(len(self.store.snapshot()), 1)

    async def test_old_process_generation_cannot_clear_current_work(self):
        await self.loaded()
        with self.assertRaises(AdmissionError):
            await self.admission.terminated("load", self.deployment, RuntimeGeneration("boot", "old-child"))
        self.assertEqual(len(self.store.snapshot()), 1)

    async def test_shutdown_termination_clears_unconfirmed_requests_of_own_generation(self):
        await self.loaded()
        self.reserve("request", generation=generation_key(self.child), kind="request", gpu_bytes=0, host_bytes=0)
        self.store.release("request", owner="kiron-proxy", generation=generation_key(self.child), confirmed_terminated=False)
        await self.admission.terminated("load", self.deployment, self.child)
        self.assertEqual(self.store.snapshot(), ())

    async def test_pre_spawn_rejection_clears_only_exact_reserved_or_unknown_load(self):
        for phase in ("reserved", "unknown"):
            with self.subTest(phase=phase):
                self.reserve()
                if phase == "unknown":
                    self.store.release("load", owner="kiron-proxy", generation=generation_key(self.initial),
                                       confirmed_terminated=False)
                await self.admission.rejected("load", "model", self.initial)
                self.assertEqual(self.store.snapshot(), ())
                await self.admission.rejected("load", "model", self.initial)

    async def test_rejection_cannot_clear_foreign_identity_request_or_resident(self):
        self.reserve()
        for operation, deployment, generation in (("load", "other", self.initial),
                ("load", "model", RuntimeGeneration("other"))):
            with self.subTest(deployment=deployment, generation=generation), self.assertRaises(AdmissionError):
                await self.admission.rejected(operation, deployment, generation)
        self.assertEqual(len(self.store.snapshot()), 1)
        await self.admission.validate("load", self.deployment, self.initial, "load")
        await self.admission.loaded("load", self.deployment, self.child)
        with self.assertRaises(AdmissionError):
            await self.admission.rejected("load", "model", self.child)
        # Even a fresh callback instance cannot bypass the atomic phase guard.
        with self.assertRaises(AdmissionError):
            await ControllerAdmission(self.store).rejected("load", "model", self.child)
        self.reserve("request", kind="request", generation=generation_key(self.child), gpu_bytes=0, host_bytes=0)
        with self.assertRaises(AdmissionError):
            await self.admission.rejected("request", "model", self.child)
        self.assertEqual(len(self.store.snapshot()), 2)


if __name__ == "__main__":
    unittest.main()
