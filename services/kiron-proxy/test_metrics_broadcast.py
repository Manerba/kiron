"""Stdlib-Unittest fuer den zentralen WebSocket-Metrics-Broadcast-Cache."""

import asyncio
import inspect
import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402
import app as app_module  # noqa: E402
import main as main_module  # noqa: E402


class _BaseBroadcastTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_cached_payload = metrics._cached_payload
        self._old_interval = metrics._PRODUCER_INTERVAL_S
        self._old_initial_timeout = metrics._PRODUCER_INITIAL_TIMEOUT_S
        self._old_maintenance_transition = app_module._maintenance_transition
        metrics._cached_payload = None
        metrics._PRODUCER_INTERVAL_S = 0.01
        metrics._PRODUCER_INITIAL_TIMEOUT_S = 0.01
        app_module._maintenance_transition = None

    def tearDown(self):
        metrics._cached_payload = self._old_cached_payload
        metrics._PRODUCER_INTERVAL_S = self._old_interval
        metrics._PRODUCER_INITIAL_TIMEOUT_S = self._old_initial_timeout
        app_module._maintenance_transition = self._old_maintenance_transition


class _Store:
    def __init__(self, summary=None, exc=None):
        self.summary = summary if summary is not None else {"requests_today": 1}
        self.exc = exc
        self.calls = 0

    async def get_summary(self):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.summary


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


class CacheProducerTests(_BaseBroadcastTest):
    async def test_cache_empty_default(self):
        self.assertIsNone(metrics.get_cached_payload())

    async def test_producer_tick_success_sets_reference(self):
        system = {"cpu": {"usage_percent": 12.0}, "timestamp": time.time()}
        summary = {"requests_today": 5}

        async def fake_get_all_metrics():
            return system

        with mock.patch.object(metrics, "get_all_metrics", fake_get_all_metrics):
            result = await metrics._run_producer_tick(_Store(summary))

        self.assertIs(metrics.get_cached_payload(), result)
        self.assertIs(result["system"], system)
        self.assertIs(result["summary"], summary)
        self.assertIsInstance(result["produced_at_monotonic"], float)

    async def test_producer_tick_summary_exception_keeps_system_snapshot(self):
        old = {"system": {"old": True}, "summary": {}, "produced_at_monotonic": 1.0}
        metrics._cached_payload = old
        system = {"cpu": {"usage_percent": 12.0}, "timestamp": time.time()}

        async def fake_get_all_metrics():
            return system

        with mock.patch.object(metrics, "get_all_metrics", fake_get_all_metrics):
            with self.assertNoLogs("metrics.producer", level="WARNING"):
                result = await metrics._run_producer_tick(_Store(exc=RuntimeError("boom")))

        self.assertIs(metrics.get_cached_payload(), result)
        self.assertIs(result["system"], system)
        self.assertEqual(result["summary"], {})
        self.assertGreater(result["produced_at_monotonic"], old["produced_at_monotonic"])

    async def test_producer_tick_metrics_exception_keeps_old_snapshot(self):
        old = {"system": {"old": True}, "summary": {}, "produced_at_monotonic": 1.0}
        metrics._cached_payload = old

        async def fake_get_all_metrics():
            raise RuntimeError("boom")

        with mock.patch.object(metrics, "get_all_metrics", fake_get_all_metrics):
            result = await metrics._run_producer_tick(_Store())

        self.assertIsNone(result)
        self.assertIs(metrics.get_cached_payload(), old)

    async def test_cache_reference_identity(self):
        payload = {"system": {}, "summary": {}, "produced_at_monotonic": 1.0}
        metrics._cached_payload = payload
        self.assertIs(metrics.get_cached_payload(), payload)
        self.assertIs(metrics.get_cached_payload(), metrics.get_cached_payload())

    async def test_consumer_shallow_copy_isolates_top_level_mutation(self):
        payload = {
            "system": {"cpu": {"usage_percent": 1.0}},
            "summary": {},
            "produced_at_monotonic": 1.0,
        }
        metrics._cached_payload = payload
        system_copy = dict(metrics.get_cached_payload()["system"])
        system_copy["maintenance"] = {"active": True}
        self.assertNotIn("maintenance", metrics.get_cached_payload()["system"])


class AppConsumerTests(_BaseBroadcastTest):
    async def test_rest_system_metrics_empty_cache_returns_503(self):
        with mock.patch.object(app_module, "get_cached_payload", return_value=None):
            response = await app_module.get_system_metrics()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            json.loads(response.body.decode("utf-8")),
            {"error": "metrics cache not ready", "retry_after_s": 2},
        )

    async def test_rest_system_metrics_reads_cache_and_does_not_mutate_it(self):
        system = {"cpu": {"usage_percent": 1.0}, "timestamp": 123.0}
        payload = {
            "system": system,
            "summary": {"requests_today": 1},
            "produced_at_monotonic": 1.0,
        }

        with mock.patch.object(app_module, "get_cached_payload", return_value=payload):
            with mock.patch.object(app_module, "_get_maintenance_state", return_value=True):
                result = await app_module.get_system_metrics()

        self.assertEqual(result["cpu"], system["cpu"])
        self.assertEqual(result["maintenance"]["active"], True)
        self.assertEqual(result["maintenance"]["transitioning"], False)
        self.assertNotIn("maintenance", system)

    async def test_rest_system_metrics_exposes_maintenance_transition(self):
        system = {"cpu": {"usage_percent": 1.0}, "timestamp": 123.0}
        payload = {
            "system": system,
            "summary": {"requests_today": 1},
            "produced_at_monotonic": 1.0,
        }
        app_module._set_maintenance_transition(False, True)

        with mock.patch.object(app_module, "get_cached_payload", return_value=payload):
            with mock.patch.object(app_module, "_get_maintenance_state", return_value=False):
                result = await app_module.get_system_metrics()

        self.assertEqual(result["maintenance"]["active"], False)
        self.assertEqual(result["maintenance"]["transitioning"], True)
        self.assertEqual(result["maintenance"]["pending"], "activating")
        self.assertEqual(result["maintenance"]["target_active"], True)

    async def test_toggle_maintenance_exposes_transition_during_apply(self):
        snapshots = []
        written = []

        async def fake_apply(_new_state):
            snapshots.append(app_module._maintenance_status_snapshot())
            return {"errors": []}

        with mock.patch.object(app_module, "_get_maintenance_state", return_value=False):
            with mock.patch.object(app_module, "_set_maintenance_state",
                                   side_effect=lambda active: written.append(active)):
                with mock.patch.object(app_module, "_apply_iptables_rules", fake_apply):
                    result = await app_module.toggle_maintenance()

        self.assertEqual(result["active"], True)
        self.assertEqual(written, [True])
        self.assertEqual(snapshots[0]["active"], False)
        self.assertEqual(snapshots[0]["transitioning"], True)
        self.assertEqual(snapshots[0]["pending"], "activating")
        self.assertEqual(snapshots[0]["target_active"], True)
        self.assertIsNone(app_module._maintenance_transition)

    async def test_websocket_single_tick_empty_cache_sends_nothing(self):
        websocket = _FakeWebSocket()
        with mock.patch.object(app_module, "get_cached_payload", return_value=None):
            sent = await app_module._send_cached_metrics_frame(websocket)

        self.assertFalse(sent)
        self.assertEqual(websocket.sent, [])

    async def test_websocket_single_tick_uses_cache_shape(self):
        summary = {"active_now": 2}
        system = {"cpu": {"usage_percent": 1.0}, "timestamp": 123.0}
        payload = {
            "system": system,
            "summary": summary,
            "produced_at_monotonic": 42.0,
        }
        websocket = _FakeWebSocket()

        with mock.patch.object(app_module, "get_cached_payload", return_value=payload):
            with mock.patch.object(app_module, "_get_maintenance_state", return_value=False):
                sent = await app_module._send_cached_metrics_frame(websocket)

        self.assertTrue(sent)
        self.assertEqual(len(websocket.sent), 1)
        frame = json.loads(websocket.sent[0])
        self.assertEqual(frame["type"], "metrics")
        self.assertEqual(frame["data"]["summary"], summary)
        self.assertEqual(frame["data"]["system"]["maintenance"]["active"], False)
        self.assertEqual(frame["data"]["system"]["maintenance"]["transitioning"], False)
        self.assertNotIn("produced_at_monotonic", frame["data"])
        self.assertNotIn("maintenance", system)


class MainProducerTests(_BaseBroadcastTest):
    async def test_metrics_producer_logs_warning_once_then_recovery(self):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        calls = 0
        snapshot = {"system": {}, "summary": {}, "produced_at_monotonic": 1.0}

        async def fake_tick(_store):
            nonlocal calls
            calls += 1
            if calls <= 2:
                return None
            return snapshot

        handler = Capture()
        logger = logging.getLogger("metrics.producer")
        old_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            with mock.patch.object(metrics, "_run_producer_tick", fake_tick):
                task = asyncio.create_task(main_module.metrics_producer(object()))
                await asyncio.sleep(0.05)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        warnings = [r for r in records if r.levelno == logging.WARNING and "tick failed" in r.getMessage()]
        recoveries = [r for r in records if r.levelno == logging.INFO and "recovered" in r.getMessage()]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(recoveries), 1)

    async def test_initial_tick_timeout_returns_pending_uncancelled_task(self):
        async def slow_tick(_store):
            await asyncio.sleep(0.05)
            return {"system": {}, "summary": {}, "produced_at_monotonic": time.monotonic()}

        with mock.patch.object(metrics, "_run_producer_tick", slow_tick):
            ready, pending = await main_module._run_initial_producer_tick(object())

        self.assertFalse(ready)
        self.assertIsNotNone(pending)
        self.assertFalse(pending.cancelled())
        await pending

    async def test_initial_tick_none_is_failed_without_pending_task(self):
        async def none_tick(_store):
            return None

        with mock.patch.object(metrics, "_run_producer_tick", none_tick):
            ready, pending = await main_module._run_initial_producer_tick(object())

        self.assertFalse(ready)
        self.assertIsNone(pending)

    async def test_initial_tick_exception_is_failed_without_pending_task(self):
        async def boom_tick(_store):
            raise RuntimeError("boom")

        with mock.patch.object(metrics, "_run_producer_tick", boom_tick):
            ready, pending = await main_module._run_initial_producer_tick(object())

        self.assertFalse(ready)
        self.assertIsNone(pending)

    async def test_pending_initial_task_adoption_is_bounded(self):
        release = asyncio.Event()
        normal_calls = 0
        first_normal_call = asyncio.Event()

        async def pending_tick():
            await release.wait()
            return {"system": {}, "summary": {}, "produced_at_monotonic": 1.0}

        async def normal_tick(_store):
            nonlocal normal_calls
            normal_calls += 1
            first_normal_call.set()
            return {"system": {}, "summary": {}, "produced_at_monotonic": 2.0}

        metrics._PRODUCER_INITIAL_TIMEOUT_S = 0.03
        metrics._PRODUCER_INTERVAL_S = 0.05
        pending = asyncio.create_task(pending_tick())
        with mock.patch.object(metrics, "_run_producer_tick", normal_tick):
            task = asyncio.create_task(main_module.metrics_producer(object(), pending))
            await asyncio.sleep(0.02)
            self.assertEqual(normal_calls, 0)
            await asyncio.wait_for(first_normal_call.wait(), timeout=1.0)
            self.assertGreaterEqual(normal_calls, 1)
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await pending

    async def test_producer_cancellation_drains_in_flight_tick(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def blocking_tick(_store):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"system": {}, "summary": {}, "produced_at_monotonic": 1.0}

        with mock.patch.object(metrics, "_run_producer_tick", blocking_tick):
            task = asyncio.create_task(main_module.metrics_producer(object()))
            await asyncio.wait_for(started.wait(), timeout=1.0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(calls, 1)

    async def test_producer_waits_interval_before_first_periodic_tick(self):
        calls = 0
        first_call = asyncio.Event()

        async def tick(_store):
            nonlocal calls
            calls += 1
            first_call.set()
            return {"system": {}, "summary": {}, "produced_at_monotonic": 1.0}

        metrics._PRODUCER_INTERVAL_S = 0.05
        with mock.patch.object(metrics, "_run_producer_tick", tick):
            task = asyncio.create_task(main_module.metrics_producer(object()))
            await asyncio.sleep(0.02)
            self.assertEqual(calls, 0)
            await asyncio.wait_for(first_call.wait(), timeout=1.0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class MetricsWriterTests(_BaseBroadcastTest):
    async def test_metrics_writer_empty_cache_skips_insert_and_sleeps(self):
        sleeps = []
        inserts = []

        class DB:
            def insert_metrics(self, value):
                inserts.append(value)

        async def fake_sleep(delay):
            sleeps.append(delay)
            raise asyncio.CancelledError

        with mock.patch.object(metrics, "get_cached_payload", return_value=None):
            with mock.patch.object(main_module.asyncio, "sleep", fake_sleep):
                with self.assertRaises(asyncio.CancelledError):
                    await main_module.metrics_writer(DB())

        self.assertEqual(inserts, [])
        self.assertTrue(sleeps)

    async def test_metrics_writer_inserts_system_dict_only(self):
        inserts = []
        system = {"cpu": {"usage_percent": 3.0}, "timestamp": 123.0}
        payload = {"system": system, "summary": {"x": 1}, "produced_at_monotonic": 1.0}

        class DB:
            def insert_metrics(self, value):
                inserts.append(value)

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        async def fake_sleep(_delay):
            raise asyncio.CancelledError

        with mock.patch.object(metrics, "get_cached_payload", return_value=payload):
            with mock.patch.object(main_module.asyncio, "to_thread", fake_to_thread):
                with mock.patch.object(main_module.asyncio, "sleep", fake_sleep):
                    with self.assertRaises(asyncio.CancelledError):
                        await main_module.metrics_writer(DB())

        self.assertEqual(inserts, [system])

    async def test_metrics_writer_skips_stale_snapshots(self):
        inserts = []
        systems = [
            {"timestamp": 1, "cpu": {}},
            {"timestamp": 2, "cpu": {}},
            {"timestamp": 3, "cpu": {}},
            {"timestamp": 4, "cpu": {}},
        ]
        payloads = [
            {"system": systems[0], "summary": {}, "produced_at_monotonic": 1.0},
            {"system": systems[1], "summary": {}, "produced_at_monotonic": 1.0},
            {"system": systems[2], "summary": {}, "produced_at_monotonic": 0.5},
            {"system": systems[3], "summary": {}, "produced_at_monotonic": 2.0},
        ]
        payload_iter = iter(payloads)

        class DB:
            def insert_metrics(self, value):
                inserts.append(value)

        async def fake_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        sleep_calls = 0

        async def fake_sleep(_delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= len(payloads):
                raise asyncio.CancelledError

        with mock.patch.object(metrics, "get_cached_payload", side_effect=lambda: next(payload_iter)):
            with mock.patch.object(main_module.asyncio, "to_thread", fake_to_thread):
                with mock.patch.object(main_module.asyncio, "sleep", fake_sleep):
                    with self.assertRaises(asyncio.CancelledError):
                        await main_module.metrics_writer(DB())

        self.assertEqual(inserts, [systems[0], systems[3]])


class StaticContractTests(_BaseBroadcastTest):
    async def test_background_task_order_places_producer_between_sampler_and_writer(self):
        source = inspect.getsource(main_module.main)
        sampler_idx = source.index("asyncio.create_task(disk_io_sampler())")
        cpu_sampler_idx = source.index("asyncio.create_task(cpu_sampler())")
        producer_idx = source.index("asyncio.create_task(metrics_producer(store, pending_metrics_initial_task))")
        writer_idx = source.index("asyncio.create_task(metrics_writer(db, db_work))")
        self.assertLess(sampler_idx, producer_idx)
        self.assertLess(cpu_sampler_idx, producer_idx)
        self.assertLess(producer_idx, writer_idx)

    async def test_frontend_metrics_503_handling_is_transient(self):
        root = Path(__file__).resolve().parent
        dashboard = (root / "static/js/tab_dashboard.js").read_text()
        system = (root / "static/js/tab_system.js").read_text()

        self.assertIn("!metricsResp.ok", dashboard)
        self.assertIn("metricsResp.status === 503", dashboard)
        self.assertIn("retry_after_s", dashboard)
        self.assertIn("!resp.ok", system)
        self.assertIn("resp.status === 503", system)
        self.assertIn("retry_after_s", system)


if __name__ == "__main__":
    unittest.main()
