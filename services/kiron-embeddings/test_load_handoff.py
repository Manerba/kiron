"""Dashboard load handoff and receipts at the serial worker boundary."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

import main
from kiron_common.gpu_admission import AdmissionError
from kiron_common.gpu_admission.native_contract import COMPLETION_HEADER, OPERATION_HEADER, OVERLAY_HEADER
from model_worker import SerialModelWorker, WorkerQueueFullError


class LoadHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.parent = ("a" * 32, "b" * 32)
        self.headers = dict(zip((OPERATION_HEADER, OVERLAY_HEADER), self.parent))
        self.worker = SimpleNamespace(load=mock.AsyncMock())
        patch = mock.patch.object(main, "model_worker", self.worker)
        patch.start()
        self.addCleanup(patch.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://fixture")
        self.addAsyncCleanup(self.client.aclose)

    async def test_native_load_passes_parent_and_confirms_completed_work(self):
        response = await self.client.post("/api/load", json={"model": "mankei-326m-embedder:latest"},
                                          headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers[COMPLETION_HEADER], self.parent[0])
        self.worker.load.assert_awaited_once_with("mankei-326m-embedder", load_parent=self.parent)

    async def test_direct_load_needs_no_parent_or_receipt(self):
        response = await self.client.post("/api/load", json={"model": "mankei-326m-embedder"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(COMPLETION_HEADER, response.headers)
        self.worker.load.assert_awaited_once_with("mankei-326m-embedder", load_parent=None)

    async def test_incomplete_or_invalid_parent_never_enqueues(self):
        for headers in ({OPERATION_HEADER: self.parent[0]}, {OVERLAY_HEADER: self.parent[1]},
                        {**self.headers, OPERATION_HEADER: "invalid"}):
            response = await self.client.post("/api/load", json={"model": "mankei-326m-embedder"}, headers=headers)
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(COMPLETION_HEADER, response.headers)
        self.worker.load.assert_not_awaited()

    async def test_completed_errors_and_predispatch_rejection_get_receipts(self):
        response = await self.client.post("/api/load", json={"model": "unknown"}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers[COMPLETION_HEADER], self.parent[0])
        self.worker.load.assert_not_awaited()
        for error, status in ((AdmissionError("resource_conflict", "fixture"), 409),
                              (WorkerQueueFullError("full"), 503), (RuntimeError("fixture"), 500)):
            self.worker.load.side_effect = error
            response = await self.client.post("/api/load", json={"model": "mankei-326m-embedder"}, headers=self.headers)
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.headers[COMPLETION_HEADER], self.parent[0])

    async def test_cancelled_wait_never_returns_a_completion_receipt(self):
        started = asyncio.Event()

        async def load(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        self.worker.load.side_effect = load
        task = asyncio.create_task(self.client.post("/api/load", json={"model": "mankei-326m-embedder"},
                                                    headers=self.headers))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_real_worker_forwards_parent_and_finishes_residency_before_returning(self):
        manager = main.ModelManager()
        reconciled = []
        worker = SerialModelWorker(manager)
        with mock.patch.object(manager, "_load_model_sync", return_value=object()) as load, \
             mock.patch.object(manager, "residency_boundary", side_effect=lambda: reconciled.append(True)):
            worker.start()
            try:
                await worker.load("mankei-326m-embedder", load_parent=self.parent)
                load.assert_called_once_with("mankei-326m-embedder", load_parent=self.parent)
                self.assertEqual(reconciled, [True])
            finally:
                await worker.stop()
