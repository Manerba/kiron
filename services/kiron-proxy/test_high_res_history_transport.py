"""Backend-Vertrag fuer die inkrementellen 1m-/10m-History-Streams."""

import inspect
import json
import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app as app_module  # noqa: E402
import high_res_history as high_res  # noqa: E402
import main as main_module  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402


class _FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, value):
        self.sent.append(json.loads(value))


class _FiniteHistoryService:
    def __init__(self):
        self.running = True
        self.waits = 0

    def snapshot_frame(self, range_key):
        window_seconds = {"1m": 60, "10m": 600}[range_key]
        return {
            "type": f"history_{range_key}_snapshot",
            "version": 1,
            "interval_ms": 250,
            "window_seconds": window_seconds,
            "samples": [
                {
                    "sequence": 3,
                    "timestamp": 100.0,
                    "values": {"cpu_usage": 1.0},
                    "states": {"cpu": "ok"},
                }
            ],
            "status": {"running": True},
        }

    async def wait_for_next(self, sequence):
        self.waits += 1
        if self.waits == 1:
            if sequence != 3:
                raise AssertionError(f"unexpected sequence {sequence}")
            return {
                "sequence": 7,
                "timestamp": 101.0,
                "values": {"cpu_usage": 2.0},
                "states": {"cpu": "ok"},
            }
        raise high_res.HighResHistoryClosed("done")

    @staticmethod
    def point_frame(sample, range_key):
        return {
            "type": f"history_{range_key}_point",
            "version": 1,
            "sample": dict(sample),
        }


class HistoryStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_sends_range_snapshot_then_incremental_latest_point(self):
        for range_key in ("1m", "10m"):
            with self.subTest(range_key=range_key):
                websocket = _FakeWebSocket()
                service = _FiniteHistoryService()

                with self.assertRaises(high_res.HighResHistoryClosed):
                    await app_module._stream_high_res_history(
                        websocket,
                        service,
                        range_key,
                    )

                self.assertEqual([frame["type"] for frame in websocket.sent], [
                    f"history_{range_key}_snapshot",
                    f"history_{range_key}_point",
                ])
                self.assertEqual(websocket.sent[1]["sample"]["sequence"], 7)
                self.assertEqual(service.waits, 2)


def _basic_auth_header():
    token = base64.b64encode(b"admin:admin").decode("ascii")
    return {"Authorization": f"Basic {token}"}


class HistoryWebSocketIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old_service = app_module.high_res_history_service
        self.client = TestClient(app_module.app)

    def tearDown(self):
        app_module.set_high_res_history_service(self.old_service)

    def test_unavailable_service_closes_both_ranges_with_try_again_later(self):
        app_module.set_high_res_history_service(None)
        for range_key in ("1m", "10m"):
            with self.subTest(range_key=range_key):
                with self.client.websocket_connect(
                    f"/ws/history/{range_key}",
                    headers=_basic_auth_header(),
                ) as websocket:
                    with self.assertRaises(WebSocketDisconnect) as raised:
                        websocket.receive_text()
                self.assertEqual(raised.exception.code, 1013)

    def test_authenticated_client_receives_each_range_snapshot_and_increment(self):
        for range_key, window_seconds in (("1m", 60), ("10m", 600)):
            with self.subTest(range_key=range_key):
                app_module.set_high_res_history_service(_FiniteHistoryService())
                with self.client.websocket_connect(
                    f"/ws/history/{range_key}",
                    headers=_basic_auth_header(),
                ) as websocket:
                    snapshot = websocket.receive_json()
                    point = websocket.receive_json()

                self.assertEqual(snapshot["type"], f"history_{range_key}_snapshot")
                self.assertEqual(snapshot["window_seconds"], window_seconds)
                self.assertEqual(snapshot["samples"][-1]["sequence"], 3)
                self.assertEqual(point["type"], f"history_{range_key}_point")
                self.assertEqual(point["sample"]["sequence"], 7)


class LifecycleContractTests(unittest.TestCase):
    def test_main_owns_exactly_one_service_and_stops_it_before_db_close(self):
        source = inspect.getsource(main_module.main)
        create_at = source.index("high_res_history = HighResHistoryService()")
        set_at = source.index("set_high_res_history_service(high_res_history)")
        start_at = source.index("high_res_history.start()")
        stop_at = source.index("await high_res_history.stop()")
        clear_at = source.index("set_high_res_history_service(None)")
        db_close_at = source.index("db.close()")

        self.assertEqual(source.count("high_res_history = HighResHistoryService()"), 1)
        self.assertLess(create_at, set_at)
        self.assertLess(set_at, start_at)
        self.assertLess(start_at, stop_at)
        self.assertLess(stop_at, clear_at)
        self.assertLess(clear_at, db_close_at)

    def test_history_db_and_existing_live_websocket_contracts_are_untouched(self):
        app_source = inspect.getsource(app_module)
        main_source = inspect.getsource(main_module)

        self.assertIn('@app.websocket("/ws/history/1m")', app_source)
        self.assertIn('@app.websocket("/ws/history/10m")', app_source)
        self.assertIn('@app.websocket("/ws/live")', app_source)
        self.assertIn("interval = 10.0", inspect.getsource(main_module.metrics_writer))
        self.assertNotIn("high_res", inspect.getsource(main_module.metrics_writer))
        self.assertNotIn("high_res", inspect.getsource(app_module.get_history_metrics))
        self.assertIn("asyncio.create_task(metrics_producer", main_source)


if __name__ == "__main__":
    unittest.main()
