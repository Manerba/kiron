"""Targeted native unload: model identity and inference-lock serialization."""
import asyncio
import threading
import unittest
from unittest import mock
import weakref

import httpx

import main


MODEL = "bge-reranker-v2-m3"
OTHER = "mankei-326m-reranker"


class Model:
    pass


class UnloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = main.ModelManager()
        self.manager.model = object()
        self.manager.current_model_name = MODEL
        self.manager.config = main.DEBERTA_CATALOG_VIEW.require_runtime_model(MODEL)
        for patch in (mock.patch.object(main, "model_manager", self.manager),
                      mock.patch.object(main.torch.cuda, "empty_cache")):
            result = patch.start()
            self.addCleanup(patch.stop)
        self.empty_cache = result
        patch = mock.patch.object(main.torch.cuda, "is_initialized", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://fixture")
        self.addAsyncCleanup(self.client.aclose)

    async def test_alias_target_and_repeated_unload(self):
        for already in (False, True):
            response = await self.client.post("/api/unload", json={"model": "BAAI/bge-reranker-v2-m3"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "unloaded", "model": MODEL, "already": already})
        self.assertEqual(self.manager.snapshot()["loaded_models"], [])
        self.assertIsNone(self.manager.model)
        self.assertIsNone(self.manager.config)
        self.empty_cache.assert_called_once()

    async def test_unloading_absent_target_keeps_another_model(self):
        before = self.manager.snapshot()
        model = self.manager.model
        response = await self.client.post("/api/unload", json={"model": OTHER})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "unloaded", "model": OTHER, "already": True})
        self.assertEqual(self.manager.snapshot(), before)
        self.assertIs(self.manager.model, model)
        self.empty_cache.assert_not_called()

    async def test_invalid_or_missing_target_never_unloads(self):
        for body, status in (({}, 422), ({"model": ""}, 422), ({"model": None}, 422),
                             ({"model": "unknown"}, 400), ({"model": MODEL, "force": True}, 422)):
            response = await self.client.post("/api/unload", json=body)
            self.assertEqual(response.status_code, status, response.text)
        response = await self.client.post("/api/unload")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.manager.snapshot()["loaded_models"], [MODEL])
        self.empty_cache.assert_not_called()

    async def test_target_is_checked_after_waiting_for_model_switch(self):
        async with self.manager._lock:
            task = asyncio.create_task(self.manager.unload(expected_model=MODEL))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.manager.current_model_name = OTHER
            self.manager.config = main.DEBERTA_CATALOG_VIEW.require_runtime_model(OTHER)
        self.assertFalse(await task)
        self.assertEqual(self.manager.snapshot()["loaded_models"], [OTHER])
        self.empty_cache.assert_not_called()

    async def test_cancelled_queued_unload_keeps_model(self):
        async with self.manager._lock:
            task = asyncio.create_task(self.manager.unload(expected_model=MODEL))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.manager.snapshot()["loaded_models"], [MODEL])
        self.empty_cache.assert_not_called()

    async def test_unload_waits_for_cancelled_inference_thread_to_end(self):
        entered, release = threading.Event(), threading.Event()

        def predict():
            entered.set()
            if not release.wait(3):
                raise TimeoutError("fixture blocked")
            self.assertIsNotNone(self.manager.model)

        async def inference():
            async with self.manager._lock:
                await main._shielded_to_thread(predict)

        pending = asyncio.create_task(inference())
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            pending.cancel()
            unloading = asyncio.create_task(self.manager.unload(expected_model=MODEL))
            await asyncio.sleep(0)
            self.assertFalse(unloading.done())
            self.assertIsNotNone(self.manager.model)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(await unloading)

    async def test_service_shutdown_can_release_the_current_model(self):
        self.assertTrue(await self.manager.unload())
        self.assertEqual(self.manager.snapshot()["loaded_models"], [])

    async def test_unload_error_does_not_confirm_success(self):
        with mock.patch.object(self.manager, "unload", side_effect=RuntimeError("fixture")):
            response = await self.client.post("/api/unload", json={"model": MODEL})
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())

    async def test_cyclic_model_is_destroyed_before_cuda_cache_release(self):
        model = Model()
        model.cycle = model
        reference = weakref.ref(model)
        self.manager.model = model
        del model
        self.empty_cache.side_effect = lambda: self.assertIsNone(reference())
        with mock.patch.object(main.torch.cuda, "is_initialized", return_value=True), \
             mock.patch.object(main.torch.cuda, "synchronize") as sync:
            self.assertTrue(await self.manager.unload(expected_model=MODEL))
        sync.assert_called_once()
        self.empty_cache.assert_called_once()

    async def test_failed_cuda_synchronization_preserves_model(self):
        before = self.manager.snapshot()
        with mock.patch.object(main.torch.cuda, "is_initialized", return_value=True), \
             mock.patch.object(main.torch.cuda, "synchronize", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                await self.manager.unload(expected_model=MODEL)
        self.assertEqual(self.manager.snapshot(), before)
        self.empty_cache.assert_not_called()

    async def test_cancelled_unload_holds_lock_until_cleanup_thread_ends(self):
        entered, release = threading.Event(), threading.Event()

        def cleanup():
            entered.set()
            if not release.wait(3):
                raise TimeoutError("fixture blocked")
            self.assertTrue(self.manager._lock.locked())

        self.empty_cache.side_effect = cleanup
        pending = asyncio.create_task(self.manager.unload(expected_model=MODEL))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            pending.cancel()
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.assertTrue(self.manager._lock.locked())
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(self.manager._lock.locked())
