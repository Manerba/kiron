"""Tests fuer Cache-Stampede-Schutz im Tags-Endpoint."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as app_module  # noqa: E402


_EMPTY_DISK = {"registry": {"data": None, "timestamp": 0}, "tags": {}}


class TagsCoalescingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_fetch = app_module._fetch_registry_tags_uncached
        self._orig_disk = app_module._load_registry_disk_cache
        self._orig_cache = dict(app_module._tags_cache)
        self._orig_inflight = dict(app_module._tags_inflight)
        app_module._tags_cache.clear()
        app_module._tags_inflight.clear()
        app_module._load_registry_disk_cache = lambda: _EMPTY_DISK

    def tearDown(self):
        app_module._fetch_registry_tags_uncached = self._orig_fetch
        app_module._load_registry_disk_cache = self._orig_disk
        app_module._tags_cache.clear()
        app_module._tags_cache.update(self._orig_cache)
        app_module._tags_inflight.clear()
        app_module._tags_inflight.update(self._orig_inflight)

    async def test_concurrent_misses_share_single_fetch(self):
        call_count = 0

        async def fake_fetch(base_name, disk_tags, now):
            nonlocal call_count
            call_count += 1
            # Yield, damit die anderen Aufrufer den Inflight-Check erreichen.
            await asyncio.sleep(0.01)
            result = {"tags": [{"tag": f"{base_name}:1"}], "model": base_name}
            app_module._tags_cache_set(base_name, {"data": result, "timestamp": now})
            return result

        app_module._fetch_registry_tags_uncached = fake_fetch

        results = await asyncio.gather(*[
            app_module.get_registry_model_tags("qwen3") for _ in range(5)
        ])

        self.assertEqual(call_count, 1, f"expected single coalesced fetch, got {call_count}")
        for r in results:
            self.assertEqual(r, results[0])

    async def test_inflight_cleared_after_exception(self):
        call_count = 0

        async def failing_fetch(base_name, disk_tags, now):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("upstream blew up")

        app_module._fetch_registry_tags_uncached = failing_fetch

        with self.assertRaises(RuntimeError):
            await app_module.get_registry_model_tags("qwen3")
        self.assertNotIn("qwen3", app_module._tags_inflight)

        # Naechster Aufruf darf nicht haengen, sondern muss neu fetchen.
        with self.assertRaises(RuntimeError):
            await app_module.get_registry_model_tags("qwen3")
        self.assertEqual(call_count, 2)

    async def test_first_caller_cancel_does_not_break_waiters(self):
        call_count = 0
        gate = asyncio.Event()

        async def gated_fetch(base_name, disk_tags, now):
            nonlocal call_count
            call_count += 1
            await gate.wait()
            result = {"tags": [{"tag": f"{base_name}:1"}], "model": base_name}
            app_module._tags_cache_set(base_name, {"data": result, "timestamp": now})
            return result

        app_module._fetch_registry_tags_uncached = gated_fetch

        first = asyncio.create_task(app_module.get_registry_model_tags("qwen3"))
        # Wait until first caller registered the inflight entry.
        for _ in range(50):
            await asyncio.sleep(0)
            if "qwen3" in app_module._tags_inflight:
                break
        self.assertIn("qwen3", app_module._tags_inflight)

        waiter = asyncio.create_task(app_module.get_registry_model_tags("qwen3"))
        # Let waiter reach its await before cancelling first.
        for _ in range(5):
            await asyncio.sleep(0)

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        gate.set()
        result = await waiter
        self.assertEqual(result["model"], "qwen3")
        self.assertEqual(call_count, 1)

    async def test_different_base_names_do_not_block_each_other(self):
        in_flight = []
        gate = asyncio.Event()

        async def fake_fetch(base_name, disk_tags, now):
            in_flight.append(base_name)
            await gate.wait()
            return {"tags": [], "model": base_name}

        app_module._fetch_registry_tags_uncached = fake_fetch

        t1 = asyncio.create_task(app_module.get_registry_model_tags("qwen3"))
        t2 = asyncio.create_task(app_module.get_registry_model_tags("gemma3"))
        # Beide muessen unabhaengig den Fetch starten (kein globaler Lock).
        for _ in range(20):
            await asyncio.sleep(0)
            if len(in_flight) >= 2:
                break
        self.assertEqual(sorted(in_flight), ["gemma3", "qwen3"])
        gate.set()
        await asyncio.gather(t1, t2)


if __name__ == "__main__":
    unittest.main()
