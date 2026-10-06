"""Targeted unload through the real serial worker and HTTP API; no model loads."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import httpx

import main
from model_worker import SerialModelWorker, StaleGenerationError, WorkerQueueFullError, WorkerStoppedError
from kiron_common.gpu_admission import AdmissionError


class Model:
    pass


def resident_manager():
    manager = main.ModelManager(model_slots=2)
    manager.device = "cpu"
    for name in ("mankei-326m-embedder", "nomic-embed-text"):
        model = Model()
        manager.models[name] = model
        manager._artifact_proofs[name] = (id(model), SimpleNamespace(fingerprint=name))
    manager.current_model_name = "nomic-embed-text"
    manager.model = manager.models[manager.current_model_name]
    manager._model_epoch = 7
    return manager


class ManagerUnloadTests(unittest.TestCase):
    def setUp(self):
        self.manager = resident_manager()
        self.manager.set_worker_thread()

    def test_target_is_destroyed_before_cuda_cache_cleanup_and_other_model_survives(self):
        manager = self.manager
        manager.device = "cuda"
        target = manager.models["mankei-326m-embedder"]
        target.cycle = target
        reference = weakref.ref(target)
        del target
        other = manager.model
        with mock.patch.object(main.torch.cuda, "synchronize") as sync, \
             mock.patch.object(main.torch.cuda, "empty_cache", side_effect=lambda: self.assertIsNone(reference())) as empty:
            self.assertTrue(manager._unload_model_sync("mankei-326m-embedder"))
        sync.assert_called_once()
        empty.assert_called_once()
        self.assertEqual(list(manager.models), ["nomic-embed-text"])
        self.assertIs(manager.model, other)
        self.assertEqual(manager.snapshot()["verified_artifacts"], {"nomic-embed-text": "nomic-embed-text"})
        self.assertEqual(manager.snapshot()["model_epoch"], 8)

    def test_current_then_last_model_unload_and_repeated_unload_is_noop(self):
        manager = self.manager
        self.assertTrue(manager._unload_model_sync("nomic-embed-text"))
        self.assertEqual(manager.current_model_name, "mankei-326m-embedder")
        self.assertIs(manager.model, manager.models["mankei-326m-embedder"])
        self.assertTrue(manager._unload_model_sync("mankei-326m-embedder"))
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["loaded_models"], [])
        self.assertIsNone(manager.model)
        self.assertIsNone(manager.current_model_name)
        self.assertFalse(manager._unload_model_sync("mankei-326m-embedder"))
        self.assertEqual(manager.snapshot(), snapshot)

    def test_stale_request_cannot_reload_after_unload(self):
        self.manager._unload_model_sync("mankei-326m-embedder")
        with mock.patch.object(self.manager, "_ensure_model_sync") as load:
            with self.assertRaises(StaleGenerationError):
                self.manager._encode_sync("mankei-326m-embedder", ["x"], None,
                    expected_epoch=7, expected_artifact="mankei-326m-embedder")
            load.assert_not_called()

    def test_cuda_sync_failure_preserves_resident_and_proof(self):
        self.manager.device = "cuda"
        before = self.manager.snapshot()
        with mock.patch.object(main.torch.cuda, "synchronize", side_effect=RuntimeError("CUDA failure")):
            with self.assertRaises(RuntimeError):
                self.manager._unload_model_sync("mankei-326m-embedder")
        self.assertEqual(self.manager.snapshot(), before)

    def test_unload_requires_worker_thread(self):
        self.manager.owner_thread_id = None
        with self.assertRaises(RuntimeError):
            self.manager._unload_model_sync("mankei-326m-embedder")


class WorkerUnloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = resident_manager()
        self.worker = SerialModelWorker(self.manager)
        self.worker.start()
        self.addAsyncCleanup(self.worker.stop)

    async def test_unload_waits_for_running_encode_and_cancellation_skips_queued_unload(self):
        for cancelled in (True, False):
            with self.subTest(cancelled=cancelled):
                started, release = threading.Event(), threading.Event()
                def encode(*args, **kwargs):
                    started.set()
                    if not release.wait(3):
                        raise TimeoutError("fixture encode blocked")
                    self.assertIn("mankei-326m-embedder", self.manager.models)
                    return "encoded"
                with mock.patch.object(self.manager, "_encode_sync", side_effect=encode):
                    encoding = asyncio.create_task(self.worker.encode("mankei-326m-embedder", ["x"], None))
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        unloading = asyncio.create_task(self.worker.unload("mankei-326m-embedder"))
                        await asyncio.sleep(0)
                        self.assertFalse(unloading.done())
                        self.assertIn("mankei-326m-embedder", self.manager.models)
                        if cancelled:
                            unloading.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await unloading
                    finally:
                        release.set()
                    self.assertEqual(await encoding, "encoded")
                    if not cancelled:
                        self.assertTrue(await unloading)
                self.assertEqual("mankei-326m-embedder" in self.manager.models, cancelled)
                self.assertIn("nomic-embed-text", self.manager.models)

    async def test_residency_boundary_completes_before_unload_response(self):
        reconciled = []
        with mock.patch.object(self.manager, "residency_boundary",
                               side_effect=lambda: reconciled.append(self.manager.snapshot())):
            self.assertTrue(await self.worker.unload("mankei-326m-embedder"))
            self.assertEqual(reconciled[0]["loaded_models"], ["nomic-embed-text"])
            self.assertIsNone(self.worker.snapshot()["current_job"])

    async def test_residency_failure_is_not_reported_as_success(self):
        with mock.patch.object(self.manager, "residency_boundary",
                               side_effect=AdmissionError("resource_unknown", "fixture")):
            with self.assertRaises(AdmissionError):
                await self.worker.unload("mankei-326m-embedder")


class UnloadApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = SimpleNamespace(unload=mock.AsyncMock(return_value=True))
        patch = mock.patch.object(main, "model_worker", self.worker)
        patch.start()
        self.addCleanup(patch.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://fixture")
        self.addAsyncCleanup(self.client.aclose)

    async def test_alias_is_resolved_and_repeated_unload_is_idempotent(self):
        for removed in (True, False):
            self.worker.unload.return_value = removed
            response = await self.client.post("/api/unload", json={"model": "mankei-326m-embedder:latest"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "unloaded", "model": "mankei-326m-embedder", "already": not removed})
        self.worker.unload.assert_awaited_with("mankei-326m-embedder")

    async def test_unknown_missing_and_extra_fields_never_enqueue(self):
        for body, status in (({"model": "unknown"}, 400), ({}, 422), ({"model": ""}, 422),
                             ({"model": 5}, 422), ({"model": "mankei-326m-embedder", "force": True}, 422)):
            response = await self.client.post("/api/unload", json=body)
            self.assertEqual(response.status_code, status, response.text)
        self.worker.unload.assert_not_awaited()

    async def test_failures_never_confirm_unloaded(self):
        for error, status in ((WorkerStoppedError("stopped"), 503), (WorkerQueueFullError("full"), 503),
                              (AdmissionError("resource_unknown", "fixture"), 409), (RuntimeError("fixture"), 500)):
            self.worker.unload.side_effect = error
            response = await self.client.post("/api/unload", json={"model": "mankei-326m-embedder"})
            self.assertEqual(response.status_code, status)
            self.assertIn("error", response.json())
