"""Native admission receipts and memory checks without touching CUDA."""
import asyncio
import threading
import unittest
from unittest import mock

import httpx
from fastapi import FastAPI

import main
from catalog_view import build_deberta_service_view
from operation_tracking import OperationLedger, tracked_route_class
from kiron_common.gpu_admission.native_contract import (
    COMPLETION_HEADER, OPERATION_HEADER, OVERLAY_HEADER, OPERATION_PATH,
)
from kiron_common.model_catalog import LoaderType
from test_cuda_runtime_errors import _FakeModelManager


class MemoryPreflightTests(unittest.TestCase):
    def test_cold_load_refuses_external_pressure_without_loading_or_dropping_resident(self):
        loader = mock.Mock()
        view = build_deberta_service_view(main.SHARED_MODEL_CATALOG,
            {LoaderType.CROSS_ENCODER: loader, LoaderType.MANKEI_LAST_TOKEN: loader})
        manager = main.ModelManager(view)
        old_model = object()
        manager.model = old_model
        manager.current_model_name = "ms-marco-MiniLM-L-6-v2"
        manager.config = view.require_runtime_model(manager.current_model_name)
        with mock.patch.object(main.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(main.torch.cuda, "mem_get_info", return_value=(211 * 1024**2, 12 * 1024**3)):
            with self.assertRaises(main.GPUCapacityError) as caught:
                manager._load_model("bge-reranker-v2-m3")
        self.assertGreater(caught.exception.required_bytes, caught.exception.free_bytes)
        loader.assert_not_called()
        self.assertIs(manager.model, old_model)
        self.assertIsNone(manager.loading_model)

    def test_warm_request_needs_workspace_but_does_not_count_resident_twice(self):
        manager = main.ModelManager()
        config = manager._service_view.require_runtime_model("bge-reranker-v2-m3")
        budget = config.gpu_memory
        with mock.patch.object(main.torch.cuda, "mem_get_info",
                               return_value=(budget.request_bytes + budget.headroom_bytes, 12 * 1024**3)):
            manager._check_memory(config, loading=False)
            with self.assertRaises(main.GPUCapacityError):
                manager._check_memory(config, loading=True)


class NativeReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_errors_have_receipts_without_starting_inference(self):
        manager = _FakeModelManager()
        main.operation_ledger.records.clear()
        cases = [
            ({"json": {"query": "q", "documents": ["d"], "top_k": -1}}, "greater_than_equal"),
            ({"json": {"query": "q", "documents": "d"}}, "list_type"),
            ({"content": b'{"query":'}, "json_invalid"),
        ]
        with mock.patch.object(main, "model_manager", manager), \
             mock.patch.object(manager, "get_model", new_callable=mock.AsyncMock) as get_model, \
             mock.patch.object(main.torch.cuda, "is_initialized", return_value=False):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url="http://backend") as client:
                for index, (payload, error_type) in enumerate(cases, start=1):
                    with self.subTest(error_type=error_type):
                        operation_id = f"{index:032x}"
                        response = await client.post("/api/rerank", **payload,
                            headers={OPERATION_HEADER: operation_id, "Content-Type": "application/json"})
                        self.assertEqual(response.status_code, 422)
                        self.assertEqual(response.json()["detail"][0]["type"], error_type)
                        self.assertEqual(response.headers[COMPLETION_HEADER], operation_id)
                        self.assertEqual(main.operation_ledger.snapshot(operation_id)["state"], "terminated")
            get_model.assert_not_awaited()

    async def test_validation_error_with_failed_cleanup_has_no_receipt(self):
        manager = _FakeModelManager()
        operation_id = "e" * 32
        main.operation_ledger.records.clear()
        with mock.patch.object(main, "model_manager", manager), \
             mock.patch.object(main.torch.cuda, "is_initialized", return_value=True), \
             mock.patch.object(main.torch.cuda, "synchronize", side_effect=RuntimeError("device lost")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url="http://backend") as client:
                response = await client.post("/api/rerank", json={"query": "q", "documents": ["d"], "top_k": -1},
                                             headers={OPERATION_HEADER: operation_id})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(COMPLETION_HEADER, response.headers)
        self.assertEqual(main.operation_ledger.snapshot(operation_id)["state"], "unknown")

    async def test_handled_oom_has_operation_receipt_and_later_status_evidence(self):
        manager = _FakeModelManager(load_error=main.torch.cuda.OutOfMemoryError("OOM"))
        operation_id = "a" * 32
        overlay = "b" * 32
        main.operation_ledger.records.clear()
        with mock.patch.object(main, "model_manager", manager), \
             mock.patch.object(main.torch.cuda, "is_initialized", return_value=False):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url="http://backend") as client:
                response = await client.post("/api/rerank", json={"query": "q", "documents": ["d"]},
                    headers={OPERATION_HEADER: operation_id, OVERLAY_HEADER: overlay})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers[COMPLETION_HEADER], operation_id)
                status = (await client.get(OPERATION_PATH + operation_id)).json()
                self.assertEqual(status["state"], "terminated")
                self.assertEqual(status["overlay_token"], overlay)
                replay = await client.post("/api/rerank", json={"query": "q", "documents": ["d"]},
                                           headers={OPERATION_HEADER: operation_id})
                self.assertNotIn(COMPLETION_HEADER, replay.headers)
        self.assertEqual(manager.resets, 1)

    async def test_failed_cuda_synchronization_produces_no_end_evidence(self):
        manager = _FakeModelManager(load_error=main.GPUCapacityError(1000, 100))
        operation_id = "c" * 32
        main.operation_ledger.records.clear()
        with mock.patch.object(main, "model_manager", manager), \
             mock.patch.object(main.torch.cuda, "is_initialized", return_value=True), \
             mock.patch.object(main.torch.cuda, "synchronize", side_effect=RuntimeError("device lost")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(main.app), base_url="http://backend") as client:
                response = await client.post("/api/rerank", json={"query": "q", "documents": ["d"]},
                                             headers={OPERATION_HEADER: operation_id})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "resource_exhausted")
        self.assertNotIn(COMPLETION_HEADER, response.headers)
        self.assertEqual(main.operation_ledger.snapshot(operation_id)["state"], "unknown")

    async def test_cancellation_receipt_waits_for_actual_worker_end(self):
        ledger = OperationLedger()
        entered, release = threading.Event(), threading.Event()
        async def confirm():
            return release.is_set()
        app = FastAPI()
        app.router.route_class = tracked_route_class(ledger, confirm)
        @app.post("/work")
        async def work():
            def worker():
                entered.set()
                release.wait(5)
            await main._shielded_to_thread(worker)
            return {"ok": True}
        operation_id = "d" * 32
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://backend") as client:
            task = asyncio.create_task(client.post("/work", headers={OPERATION_HEADER: operation_id}))
            try:
                for _ in range(100):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.005)
                self.assertTrue(entered.is_set())
                task.cancel()
                await asyncio.sleep(0.02)
                task.cancel()
                self.assertEqual(ledger.snapshot(operation_id)["state"], "active")
                self.assertFalse(task.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(ledger.snapshot(operation_id)["state"], "terminated")

    def test_bounded_ledger_never_evicts_active_or_unconfirmed_work(self):
        ledger = OperationLedger(limit=2)
        self.assertTrue(ledger.begin("a" * 32, None))
        self.assertTrue(ledger.begin("b" * 32, None))
        ledger.finish("a" * 32, False)
        self.assertFalse(ledger.begin("c" * 32, None))
        ledger.finish("b" * 32, True)
        self.assertTrue(ledger.begin("c" * 32, None))
        self.assertIsNone(ledger.snapshot("b" * 32))
        self.assertEqual(ledger.snapshot("a" * 32)["state"], "unknown")
