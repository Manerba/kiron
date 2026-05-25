"""Stdlib-Unittest fuer services/kiron-docling/proxy.py.

Ausfuehren:
    cd services/kiron-docling && python -m unittest test_proxy

Deckt State-Maschine, Streaming-Fehlerpfade, VRAM-Freigabe und
Shutdown-Gates ab. Keine Docker-Calls — alles gemockt.
"""

import asyncio
import concurrent.futures
import contextlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

import httpx


TEST_RUNTIME_DIR = Path(os.environ.get(
    "KIRON_TEST_RUNTIME_DIR",
    "/tmp/kiron-docling-test-runtime",
))


def _reload_module():
    """Frischer Import von proxy — State und Globals werden zurueckgesetzt."""
    os.environ["KIRON_RUNTIME_DIR"] = str(TEST_RUNTIME_DIR)
    os.environ["KIRON_ACTIVE_REQUESTS_URL"] = ""
    TEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for path in TEST_RUNTIME_DIR.glob("docling-vram-*.json"):
        try:
            path.unlink()
        except OSError:
            pass
    module_dir = Path(__file__).resolve().parent
    current = sys.modules.get("proxy")
    current_file = Path(getattr(current, "__file__", "")).resolve() if current else None
    if current_file != module_dir / "proxy.py":
        sys.modules.pop("proxy", None)
    sys.path.insert(0, str(module_dir))
    import proxy as _proxy  # noqa: F401
    try:
        module = importlib.reload(_proxy)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(module_dir))

    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    module._to_thread = inline_to_thread
    return module


class StateMachineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    async def _start_supervisor_result(self, started=True, healthy=True):
        p = self.proxy

        async def fake_free_vram():
            return None

        def fake_start():
            return started

        async def fake_wait_healthy(timeout=None):
            return healthy

        async def fake_run_stop_once():
            return True

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram), \
             mock.patch.object(p, "_start", fake_start), \
             mock.patch.object(p, "_wait_healthy", fake_wait_healthy), \
             mock.patch.object(p, "_run_stop_once", fake_run_stop_once):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            return await p._start_supervisor()

    async def test_cold_start_success_promotes_running(self):
        p = self.proxy
        ok = await self._start_supervisor_result(started=True, healthy=True)
        self.assertIsNotNone(ok)
        self.assertIs(p._state, p.State.RUNNING)
        self.assertGreater(p._last_request_time, 0.0)
        self.assertEqual(p._active_requests, 0)

    async def test_cold_start_start_fails_stays_stopped(self):
        p = self.proxy
        ok = await self._start_supervisor_result(started=False, healthy=False)
        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED)

    async def test_cold_start_health_fails_stops_and_stays_stopped(self):
        p = self.proxy

        stop_calls = []

        async def fake_run_stop_once():
            stop_calls.append(1)
            return True

        async def fake_free_vram():
            return None

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(False)), \
             mock.patch.object(p, "_run_stop_once", fake_run_stop_once):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            ok = await p._start_supervisor()
        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(len(stop_calls), 1, "Health-Failure soll Stop ausloesen")

    async def test_shutdown_during_starting_wins(self):
        """SHUTDOWN setzen waehrend Health laeuft -> finaler State bleibt SHUTDOWN."""
        p = self.proxy

        health_started = asyncio.Event()
        health_release = asyncio.Event()

        async def slow_wait_healthy(timeout=None):
            health_started.set()
            await health_release.wait()
            return True

        async def fake_free_vram():
            return None

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy", slow_wait_healthy), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            task = asyncio.create_task(p._start_supervisor())
            await health_started.wait()
            async with p._state_lock:
                p._state = p.State.SHUTDOWN
                p._state_changed.notify_all()
            health_release.set()
            ok = await task
        self.assertFalse(ok)
        self.assertIs(p._state, p.State.SHUTDOWN)

    async def test_ensure_running_returns_false_during_shutdown(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.SHUTDOWN
        ok = await p.ensure_running()
        self.assertFalse(ok)

    async def test_warm_reuse_triggers_regate_on_first_slot(self):
        p = self.proxy
        calls = []

        async def fake_free_vram():
            calls.append(1)

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 1.0

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram):
            ok = await p.ensure_running()

        self.assertIsNotNone(ok)
        self.assertEqual(p._active_requests, 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(p._warm_regate_done.is_set())

    async def test_warm_reuse_skips_regate_on_subsequent_slot(self):
        p = self.proxy
        calls = []

        async def fake_free_vram():
            calls.append(1)

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 1.0
            p._warm_regate_done.set()

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram):
            ok = await p.ensure_running()

        self.assertIsNotNone(ok)
        self.assertEqual(p._active_requests, 2)
        self.assertEqual(calls, [])

    async def test_warm_regate_parallel_callers_single_flight(self):
        p = self.proxy
        calls = []
        regate_started = asyncio.Event()
        regate_release = asyncio.Event()

        async def slow_free_vram():
            calls.append(1)
            regate_started.set()
            await regate_release.wait()

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 1.0

        with mock.patch.object(p, "_prepare_vram_for_docling", slow_free_vram):
            callers = [asyncio.create_task(p.ensure_running())
                       for _ in range(3)]
            await asyncio.wait_for(regate_started.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
            self.assertEqual(len(calls), 1)
            async with p._state_lock:
                self.assertEqual(p._active_requests, 1)
                self.assertTrue(await p._vram_lease_active())
            self.assertFalse(p._warm_regate_done.is_set())
            regate_release.set()
            results = await asyncio.gather(*callers)

        self.assertTrue(all(token is not None for token in results))
        self.assertEqual(len(calls), 1)
        self.assertEqual(p._active_requests, 3)

    async def test_warm_regate_failure_releases_slot(self):
        p = self.proxy

        async def failing_free_vram():
            raise p.VramGateError("blocked")

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 1.0

        with mock.patch.object(p, "_prepare_vram_for_docling", failing_free_vram):
            ok = await p.ensure_running()

        self.assertFalse(ok)
        self.assertEqual(p._active_requests, 0)
        self.assertTrue(p._warm_regate_done.is_set())

    async def test_warm_regate_refreshes_existing_own_startup_marker(self):
        p = self.proxy
        p.GPU_SERVICE_MARKER_SHORT_TTL_S = 0.2
        p._write_vram_marker("startup", ttl_s=0.02)
        marker_seen_active = []

        async def slow_free_vram():
            await asyncio.sleep(0.05)
            _, active = p._marker_payload(p.STARTUP_MARKER_PATH)
            marker_seen_active.append(active)

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 1.0

        with mock.patch.object(p, "_prepare_vram_for_docling", slow_free_vram):
            ok = await p.ensure_running()

        self.assertIsNotNone(ok)
        self.assertEqual(marker_seen_active, [True])
        _, active_after = p._marker_payload(p.STARTUP_MARKER_PATH)
        self.assertTrue(active_after)

    async def test_warm_regate_shutdown_releases_waiters(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 1.0
            p._warm_regate_done.clear()

        waiter = asyncio.create_task(p.ensure_running())
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())

        with mock.patch.object(p, "_is_running_for_dirty", lambda: False), \
             mock.patch.object(p.http_client, "aclose", lambda: _future(None)):
            await p.on_shutdown()

        ok = await asyncio.wait_for(waiter, timeout=2.0)
        self.assertFalse(ok)
        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertTrue(p._warm_regate_done.is_set())

    async def test_warm_regate_owner_releases_slot_if_shutdown_wins(self):
        p = self.proxy
        regate_started = asyncio.Event()
        regate_release = asyncio.Event()

        async def slow_free_vram():
            regate_started.set()
            await regate_release.wait()

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 1.0

        with mock.patch.object(p, "_prepare_vram_for_docling", slow_free_vram):
            owner = asyncio.create_task(p.ensure_running())
            await asyncio.wait_for(regate_started.wait(), timeout=2.0)
            async with p._state_lock:
                p._state = p.State.SHUTDOWN
                p._state_changed.notify_all()
            regate_release.set()
            ok = await asyncio.wait_for(owner, timeout=2.0)

        self.assertFalse(ok)
        self.assertEqual(p._active_requests, 0)
        self.assertTrue(p._warm_regate_done.is_set())

    async def test_ensure_running_backend_failed_waits_when_slots_active(self):
        """Bei _backend_failed und active > 0 wartet neuer Request."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._backend_failed = True
            p._active_requests = 1
            p._last_request_time = 1.0

        task = asyncio.create_task(p.ensure_running())
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())

        # Release the blocker slot — triggers drain
        async def fake_run_stop_once():
            return True

        with mock.patch.object(p, "_run_stop_once", fake_run_stop_once), \
             mock.patch.object(p, "_is_running", lambda: False), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(True)), \
             mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)):
            await p._release_slot(ok=False, backend_failed=False)
            ok = await asyncio.wait_for(task, timeout=5.0)
        self.assertTrue(ok)
        self.assertEqual(p._active_requests, 1)

    async def test_parallel_cold_start_single_flight(self):
        p = self.proxy
        start_calls = 0
        gate = asyncio.Event()

        async def slow_health(timeout=None):
            await gate.wait()
            return True

        def counting_start():
            nonlocal start_calls
            start_calls += 1
            return True

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", counting_start), \
             mock.patch.object(p, "_wait_healthy", slow_health), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STOPPED

            callers = [asyncio.create_task(p.ensure_running())
                       for _ in range(4)]
            await asyncio.sleep(0.05)
            gate.set()
            results = await asyncio.gather(*callers)

        self.assertTrue(all(token is not None for token in results))
        self.assertEqual(start_calls, 1, "genau ein _start() im single-flight")
        self.assertEqual(p._active_requests, 4)


class StartFailureCooldownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    async def _start_supervisor_result(self, started=True, healthy=True):
        p = self.proxy

        async def fake_free_vram():
            return None

        def fake_start():
            return started

        async def fake_wait_healthy(timeout=None):
            return healthy

        async def fake_run_stop_once():
            return True

        with mock.patch.object(p, "_prepare_vram_for_docling", fake_free_vram), \
             mock.patch.object(p, "_start", fake_start), \
             mock.patch.object(p, "_wait_healthy", fake_wait_healthy), \
             mock.patch.object(p, "_run_stop_once", fake_run_stop_once):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            return await p._start_supervisor()

    async def test_counter_increments_on_failed_start(self):
        p = self.proxy
        ok = await self._start_supervisor_result(started=True, healthy=False)

        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(p._start_failure_count, 1)
        self.assertEqual(p._start_cooldown_until, 0.0)

    async def test_counter_resets_on_successful_start(self):
        p = self.proxy
        async with p._state_lock:
            p._start_failure_count = 2
            p._start_failure_window_start = time.monotonic()
            p._start_cooldown_until = time.monotonic() + 300.0

        ok = await self._start_supervisor_result(started=True, healthy=True)

        self.assertTrue(ok)
        self.assertIs(p._state, p.State.RUNNING)
        self.assertEqual(p._start_failure_count, 0)
        self.assertEqual(p._start_failure_window_start, 0.0)
        self.assertEqual(p._start_cooldown_until, 0.0)

    async def test_cooldown_triggers_after_threshold(self):
        p = self.proxy

        for _ in range(p.START_FAILURE_THRESHOLD):
            ok = await self._start_supervisor_result(
                started=True, healthy=False)
            self.assertFalse(ok)

        self.assertGreater(p._start_cooldown_until, time.monotonic())
        self.assertEqual(p._start_failure_count, p.START_FAILURE_THRESHOLD)

    async def test_ensure_running_blocked_in_cooldown(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED
            p._start_cooldown_until = time.monotonic() + 300.0

        async def unexpected_start():
            raise AssertionError("Cooldown must skip _run_start_once")

        async def unexpected_free_vram():
            raise AssertionError("Cooldown must skip VRAM gate")

        with mock.patch.object(p, "_run_start_once", unexpected_start), \
             mock.patch.object(p, "_prepare_vram_for_docling",
                               unexpected_free_vram):
            ok = await p.ensure_running()

        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(p._active_requests, 0)

    async def test_cooldown_expires_allows_restart(self):
        p = self.proxy
        seen_states = []
        async with p._state_lock:
            p._state = p.State.STOPPED
            p._start_cooldown_until = time.monotonic() - 1.0

        async def fake_run_start_once():
            async with p._state_lock:
                seen_states.append(p._state)
                p._state = p.State.STOPPED
                p._state_changed.notify_all()
            return False

        with mock.patch.object(p, "_run_start_once", fake_run_start_once):
            ok = await p.ensure_running()

        self.assertFalse(ok)
        self.assertEqual(seen_states, [p.State.STARTING])
        self.assertIs(p._state, p.State.STOPPED)

    async def test_failure_window_rollover(self):
        p = self.proxy
        old_window_start = (
            time.monotonic() - p.START_FAILURE_WINDOW_S - 10.0
        )
        async with p._state_lock:
            p._start_failure_window_start = old_window_start
            p._start_failure_count = 2
            p._start_cooldown_until = 0.0
            p._record_start_failure_locked()

        self.assertEqual(p._start_failure_count, 1)
        self.assertEqual(p._start_cooldown_until, 0.0)
        self.assertGreater(p._start_failure_window_start, old_window_start)


class VramTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()
        # Verify-Polling auf 2s kuerzen, damit Tests schnell laufen.
        self.proxy.VRAM_VERIFY_TIMEOUT_S = 2

    async def test_cumulative_budget_unloads_until_fits(self):
        """3 kleine Modelle, Summe > BUDGET -> mindestens eines entladen."""
        p = self.proxy
        # 3x 2 GiB = 6 GiB, BUDGET 4.5 GiB -> eines muss raus
        gib = 1024**3
        models = [
            {"name": "m1", "size_vram": 2 * gib},
            {"name": "m2", "size_vram": 2 * gib},
            {"name": "m3", "size_vram": 2 * gib},
        ]

        calls = {"ps": 0, "unload": [], "verify": 0}

        class FakeResp:
            def __init__(self, data, status=200):
                self._data = data
                self.status_code = status

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError("err", request=None, response=None)

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self, base_url, timeout):
                self.base_url = base_url

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, path):
                if path == "/api/ps":
                    calls["ps"] += 1
                    if calls["ps"] == 1:
                        return FakeResp({"models": models})
                    # Verify-Polling: unloaded model still present first, then gone
                    calls["verify"] += 1
                    remaining = [m for m in models
                                  if m["name"] not in calls["unload"]]
                    return FakeResp({"models": remaining})
                return FakeResp({})

            async def post(self, path, json=None, timeout=None):
                if path == "/api/generate":
                    calls["unload"].append(json["model"])
                return FakeResp({})

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            await p._free_vram_for_docling()

        # Budget 4.5 GiB, sort ascending: kept m1 (2) + m2 (2) = 4. Adding
        # m3 would make 6 > 4.5, so m3 is unloaded.
        self.assertEqual(calls["unload"], ["m3"])
        self.assertGreaterEqual(calls["verify"], 1)

    async def test_unload_post_sends_stream_false(self):
        p = self.proxy
        captured = []

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"models": [{"name": "big", "size_vram": 10 * 1024**3}]}

        class FakeEmptyResp(FakeResp):
            def json(self):
                return {"models": []}

        class FakeClient:
            def __init__(self, base_url, timeout):
                self.get_calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, path):
                self.get_calls += 1
                if self.get_calls > 1:
                    return FakeEmptyResp()
                return FakeResp()

            async def post(self, path, json=None, timeout=None):
                captured.append({"path": path, "json": json,
                                  "timeout": timeout})
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            await p._free_vram_for_docling()
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0]["json"]["stream"], False)
        self.assertEqual(captured[0]["json"]["keep_alive"], 0)

    async def test_ollama_down_swallowed(self):
        """/api/ps wirft ConnectError -> fail-closed vor Docling-Start."""
        p = self.proxy

        class FakeClient:
            def __init__(self, base_url, timeout):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, path):
                raise httpx.ConnectError("Ollama down", request=None)

            async def post(self, path, json=None, timeout=None):
                return None

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._free_vram_for_docling()

    async def test_unload_timeout_does_not_abort_loop(self):
        """Timeout bei einem Unload blockiert den Docling-Start fail-closed."""
        p = self.proxy
        gib = 1024**3
        models = [
            {"name": "big1", "size_vram": 10 * gib},
            {"name": "big2", "size_vram": 10 * gib},
        ]
        unloaded = []

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"models": models}

        class FakeClient:
            def __init__(self, base_url, timeout):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, path):
                return FakeResp()

            async def post(self, path, json=None, timeout=None):
                if json["model"] == "big1":
                    raise httpx.TimeoutException("unload timeout")
                unloaded.append(json["model"])
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._free_vram_for_docling()
        self.assertEqual(unloaded, [])


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    async def test_release_slot_success_updates_time(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 100.0
        await p._release_slot(ok=True, backend_failed=False)
        self.assertEqual(p._active_requests, 0)
        self.assertGreater(p._last_request_time, 100.0)
        self.assertIs(p._state, p.State.RUNNING)

    async def test_release_slot_during_shutdown_keeps_shutdown(self):
        """F91: Erfolg waehrend SHUTDOWN aktualisiert Warm-Timestamp NICHT."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.SHUTDOWN
            p._active_requests = 1
            p._last_request_time = 100.0
        await p._release_slot(ok=True, backend_failed=False)
        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertEqual(p._last_request_time, 100.0)
        self.assertEqual(p._active_requests, 0)

    async def test_release_slot_backend_failure_drain(self):
        """F136: letzter Slot mit backend_failed triggert _run_stop_once."""
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = False
            p._last_request_time = 100.0

        with mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running", lambda: False):
            await p._release_slot(ok=False, backend_failed=True)
        self.assertEqual(len(stops), 1)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertFalse(p._backend_failed)
        self.assertEqual(p._last_request_time, 0.0)

    async def test_release_slot_backend_failure_with_active_slots_waits(self):
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 2
            p._backend_failed = False
            p._last_request_time = 100.0

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._release_slot(ok=False, backend_failed=True)
        self.assertEqual(len(stops), 0)
        self.assertTrue(p._backend_failed)
        self.assertIs(p._state, p.State.RUNNING)

    async def test_release_slot_shutdown_with_backend_failure_no_drain(self):
        """F90: Im SHUTDOWN nie State-Overwrite, aber _backend_failed resetten."""
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        async with p._state_lock:
            p._state = p.State.SHUTDOWN
            p._active_requests = 1
            p._backend_failed = True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._release_slot(ok=False, backend_failed=True)
        self.assertEqual(len(stops), 0)
        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertFalse(p._backend_failed)

    async def test_response_iter_complete_releases_slot_ok(self):
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                for chunk in [b"hello", b"world"]:
                    yield chunk

            async def aclose(self):
                pass

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 100.0

        chunks = []
        async for c in p._response_iter(FakeResp()):
            chunks.append(c)
        self.assertEqual(chunks, [b"hello", b"world"])
        self.assertEqual(p._active_requests, 0)
        self.assertGreater(p._last_request_time, 100.0)

    async def test_response_iter_read_error_invalidates_backend(self):
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                yield b"partial"
                raise httpx.ReadError("mid-stream", request=None)

            async def aclose(self):
                pass

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1

        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        chunks = []
        with mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running", lambda: False):
            with self.assertRaises(httpx.ReadError):
                async for c in p._response_iter(FakeResp()):
                    chunks.append(c)
        self.assertEqual(chunks, [b"partial"])
        # Backend-Failure-Drain hat Stop ausgeloest
        self.assertEqual(len(stops), 1)
        self.assertIs(p._state, p.State.STOPPED)

    async def test_response_iter_cancelled_no_backend_invalidation(self):
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                yield b"x"
                raise asyncio.CancelledError()

            async def aclose(self):
                pass

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 100.0

        with self.assertRaises(asyncio.CancelledError):
            async for _ in p._response_iter(FakeResp()):
                pass
        self.assertEqual(p._active_requests, 0)
        self.assertFalse(p._backend_failed)
        # Kein SHUTDOWN wird gesetzt; State bleibt RUNNING
        self.assertIs(p._state, p.State.RUNNING)

    async def test_response_iter_aclose_error_does_not_mask_slot_release(self):
        """F81: aclose-Fehler darf Slot-Release nicht verschlucken."""
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                yield b"done"

            async def aclose(self):
                raise RuntimeError("close failed")

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = 100.0

        async for _ in p._response_iter(FakeResp()):
            pass
        self.assertEqual(p._active_requests, 0)
        self.assertGreater(p._last_request_time, 100.0)

    async def test_response_iter_release_survives_cancel_during_release(self):
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                yield b"x"

            async def aclose(self):
                pass

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1

        original_release = p._release_slot
        release_started = asyncio.Event()
        release_continue = asyncio.Event()

        async def slow_release(*args, **kwargs):
            release_started.set()
            await release_continue.wait()
            return await original_release(*args, **kwargs)

        agen = p._response_iter(FakeResp())
        self.assertEqual(await agen.__anext__(), b"x")

        with mock.patch.object(p, "_release_slot", slow_release):
            close_task = asyncio.create_task(agen.aclose())
            await asyncio.wait_for(release_started.wait(), timeout=1.0)
            close_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await close_task

            self.assertEqual(p._active_requests, 1)
            release_continue.set()

            for _ in range(20):
                if p._active_requests == 0:
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(p._active_requests, 0)


class IdleWatcherTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    async def test_idle_stop_transitions_to_stopped(self):
        import time as real_time
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = real_time.monotonic() - p.IDLE_TIMEOUT_S - 10

        with mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            await p._try_stop_if_idle()
        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(p._last_request_time, 0.0)

    async def test_idle_stop_skips_if_active_requests(self):
        import time as real_time
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._last_request_time = real_time.monotonic() - p.IDLE_TIMEOUT_S - 10

        with mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            await p._try_stop_if_idle()
        self.assertIs(p._state, p.State.RUNNING)

    async def test_idle_stop_shutdown_wins(self):
        """F32: parallel SHUTDOWN -> Stop-Failure-Recovery beruehrt SHUTDOWN nicht."""
        import time as real_time
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = real_time.monotonic() - p.IDLE_TIMEOUT_S - 10

        async def simulate_shutdown_during_stop():
            async with p._state_lock:
                p._state = p.State.SHUTDOWN
                p._state_changed.notify_all()
            return False

        with mock.patch.object(p, "_run_stop_once",
                                simulate_shutdown_during_stop), \
             mock.patch.object(p, "_is_running", lambda: False):
            await p._try_stop_if_idle()
        self.assertIs(p._state, p.State.SHUTDOWN)

    async def test_idle_stop_writes_shutdown_marker_during_stopping(self):
        """#771: Shutdown-Overlay-Marker liegt waehrend STOPPING vor."""
        import time as real_time
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = real_time.monotonic() - p.IDLE_TIMEOUT_S - 10

        seen = {"during": False}

        async def fake_stop():
            seen["during"] = p.SHUTDOWN_MARKER_PATH.exists()
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._try_stop_if_idle()
        self.assertTrue(seen["during"])
        self.assertIs(p._state, p.State.STOPPED)
        self.assertFalse(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_idle_stop_keeps_marker_on_dirty(self):
        """#771: Stop-Fehler -> STOPPED_DIRTY behaelt Shutdown-Marker bis TTL."""
        import time as real_time
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = real_time.monotonic() - p.IDLE_TIMEOUT_S - 10

        with mock.patch.object(p, "_run_stop_once", lambda: _future(False)), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._try_stop_if_idle()
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertTrue(p.SHUTDOWN_MARKER_PATH.exists())


class ProxyHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()
        self._vram_gate_patcher = mock.patch.object(
            self.proxy, "_prepare_vram_for_docling", lambda: _future(None))
        self._vram_gate_patcher.start()

    def tearDown(self):
        self._vram_gate_patcher.stop()

    async def _make_request(self, method="GET", path="/health"):
        # Minimal starlette Request mock — proxy_handler nutzt nur
        # method, url.path, url.query, headers, stream().
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"host", b"localhost"), (b"accept", b"*/*")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        return Request(scope, receive=receive)

    async def _prime_running(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._last_request_time = 100.0
            p._active_requests = 0

    async def test_ensure_running_failure_returns_503(self):
        p = self.proxy
        req = await self._make_request()
        async with p._state_lock:
            p._state = p.State.SHUTDOWN
        resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 503)

    async def test_connect_error_returns_502_and_invalidates(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request()

        async def fake_send(*a, **kw):
            raise httpx.ConnectError("down", request=None)

        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        with mock.patch.object(p.http_client, "send", fake_send), \
             mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running", lambda: False):
            resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 502)
        self.assertEqual(len(stops), 1, "letzter Slot triggert drain cleanup")
        self.assertIs(p._state, p.State.STOPPED)

    async def test_timeout_returns_504_without_backend_invalidation(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request()

        async def fake_send(*a, **kw):
            raise httpx.TimeoutException("timeout", request=None)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 504)
        self.assertFalse(p._backend_failed)

    async def test_head_releases_slot_and_returns_body_less(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request(method="HEAD")

        class FakeResp:
            def __init__(self):
                self.status_code = 200
                self.headers = {"x-custom": "1",
                                 "transfer-encoding": "chunked"}

            async def aclose(self):
                pass

        async def fake_send(req, stream=True):
            return FakeResp()

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(p._active_requests, 0)
        self.assertNotIn("transfer-encoding",
                          {k.lower(): v for k, v in resp.headers.items()})

    async def test_head_preserves_backend_content_length(self):
        """#504: HEAD soll Backend-Content-Length weiterreichen (RFC 7231 §4.3.2)."""
        p = self.proxy
        await self._prime_running()
        req = await self._make_request(method="HEAD")

        class FakeResp:
            def __init__(self):
                self.status_code = 200
                self.headers = {"content-length": "12345", "x-custom": "1"}

            async def aclose(self):
                pass

        async def fake_send(req, stream=True):
            return FakeResp()

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 200)
        resp_headers_ci = {k.lower(): v for k, v in resp.headers.items()}
        self.assertEqual(resp_headers_ci.get("content-length"), "12345")

    async def test_forward_preserves_percent_encoded_path(self):
        """#505: raw_path wird byte-identisch ans Backend durchgereicht (keine Decoding-Normalisierung)."""
        p = self.proxy
        await self._prime_running()
        # Starlette decoded path waere "/files/my doc%.pdf" — raw_path ist das Wire-Original.
        encoded_path = "/files/my%20doc%25.pdf"
        scope = {
            "type": "http",
            "method": "HEAD",
            "path": "/files/my doc%.pdf",
            "raw_path": encoded_path.encode(),
            "query_string": b"",
            "headers": [(b"host", b"localhost")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)

        seen_paths = []

        class FakeResp:
            def __init__(self):
                self.status_code = 200
                self.headers = {}

            async def aclose(self):
                pass

        async def fake_send(outgoing, stream=True):
            seen_paths.append(str(outgoing.url))
            return FakeResp()

        with mock.patch.object(p.http_client, "send", fake_send):
            await p.proxy_handler(req)
        self.assertTrue(seen_paths, "Backend nicht aufgerufen")
        self.assertIn("%20", seen_paths[0])
        self.assertIn("%25", seen_paths[0])


class AsyncTaskRegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()
        self._vram_gate_patcher = mock.patch.object(
            self.proxy, "_prepare_vram_for_docling", lambda: _future(None))
        self._vram_gate_patcher.start()

    def tearDown(self):
        self._vram_gate_patcher.stop()

    async def _make_request(self, method="GET", path="/health"):
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"host", b"localhost")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        return Request(scope, receive=receive)

    async def _prime_running(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 100.0
            p._backend_failed = False

    class _BufferedResp:
        def __init__(self, body: bytes, status_code: int = 200):
            self.status_code = status_code
            self.headers = {"content-type": "application/json"}
            self._body = body
            self.closed = False

        async def aread(self):
            return self._body

        async def aiter_raw(self):
            yield self._body

        async def aclose(self):
            self.closed = True

    async def test_trigger_registers_task(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request(
            "POST", "/v1alpha/convert/source/async")

        body = json.dumps({"task_id": "abc123"}).encode()

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("abc123", p._tasks)
        self.assertFalse(p._tasks["abc123"].terminal)
        self.assertEqual(p._active_requests, 0)

    async def test_poll_updates_status_nonterminal(self):
        p = self.proxy
        await self._prime_running()
        now = time.monotonic()
        async with p._tasks_lock:
            p._tasks["abc"] = p._TaskEntry(now, now)
        req = await self._make_request(
            "GET", "/v1alpha/status/poll/abc")
        body = json.dumps({"task_status": "pending"}).encode()

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)

        self.assertEqual(resp.status_code, 200)
        self.assertFalse(p._tasks["abc"].terminal)
        self.assertEqual(p._tasks["abc"].status, "pending")

    async def test_poll_updates_status_terminal(self):
        p = self.proxy
        await self._prime_running()
        now = time.monotonic()
        async with p._tasks_lock:
            p._tasks["abc"] = p._TaskEntry(now, now)
        req = await self._make_request(
            "GET", "/v1alpha/status/poll/abc")
        body = json.dumps({"task_status": "success"}).encode()

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(p._tasks["abc"].terminal)
        self.assertEqual(p._tasks["abc"].status, "success")
        self.assertEqual(p._active_requests, 0)

    async def test_idle_stop_blocked_by_nonterminal_task(self):
        p = self.proxy
        now = time.monotonic()
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = now - p.IDLE_TIMEOUT_S - 10
        async with p._tasks_lock:
            p._tasks["abc"] = p._TaskEntry(now, now)
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._try_stop_if_idle()

        self.assertIs(p._state, p.State.RUNNING)
        self.assertEqual(stops, [])

    async def test_idle_stop_proceeds_when_all_terminal(self):
        p = self.proxy
        now = time.monotonic()
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = now - p.IDLE_TIMEOUT_S - 10
        async with p._tasks_lock:
            p._tasks["abc"] = p._TaskEntry(
                now, now, terminal=True, status="success")
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._try_stop_if_idle()

        self.assertEqual(stops, [1])
        self.assertIs(p._state, p.State.STOPPED)

    async def test_task_gc_evicts_stale_nonterminal(self):
        p = self.proxy
        now = time.monotonic()
        old = now - p.TASK_MAX_AGE_S - 1
        async with p._tasks_lock:
            p._tasks["stale"] = p._TaskEntry(old, old)

        self.assertFalse(await p._has_nonterminal_tasks())
        self.assertEqual(p._tasks, {})

    async def test_unknown_poll_inserts_entry(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request(
            "GET", "/v1alpha/status/poll/xyz")
        body = json.dumps({"task_status": "success"}).encode()

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)

        self.assertEqual(resp.status_code, 200)
        self.assertIn("xyz", p._tasks)
        self.assertTrue(p._tasks["xyz"].terminal)
        self.assertEqual(p._tasks["xyz"].status, "success")

    async def test_result_download_marks_task_terminal(self):
        # #876: /result/<id> liefert Markdown/JSON ohne task_status; ohne
        # Sonderbehandlung bliebe der Task nonterminal und blockierte Idle-Stop.
        p = self.proxy
        await self._prime_running()
        now = time.monotonic()
        async with p._tasks_lock:
            p._tasks["abc"] = p._TaskEntry(now, now)
        req = await self._make_request("GET", "/v1/result/abc")
        body = b"# Markdown ohne task_status\n\nDokument-Inhalt..."

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(chunks), body)
        self.assertTrue(p._tasks["abc"].terminal)
        self.assertEqual(p._tasks["abc"].status, "success")
        self.assertEqual(p._active_requests, 0)

    async def test_result_download_unknown_task_inserts_terminal(self):
        # #876: /result/<id> ohne vorherigen Task-Eintrag (z.B. nach
        # Slot-Generation-Wechsel) muss als terminal eingefuegt werden.
        p = self.proxy
        await self._prime_running()
        req = await self._make_request("GET", "/v1alpha/result/zzz")
        body = json.dumps({"document": {"md_content": "..."}}).encode()

        async def fake_send(req, stream=True):
            return self._BufferedResp(body)

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(chunks), body)
        self.assertIn("zzz", p._tasks)
        self.assertTrue(p._tasks["zzz"].terminal)
        self.assertEqual(p._tasks["zzz"].status, "success")
        self.assertEqual(p._active_requests, 0)

    async def test_result_download_stays_streaming(self):
        # #875: /result/<id> kann grosse Markdown/JSON-Antworten liefern und
        # darf daher nicht wie status/poll per aread() voll gepuffert werden.
        p = self.proxy
        await self._prime_running()
        req = await self._make_request("GET", "/v1/result/no-buffer")

        class StreamingOnlyResp:
            status_code = 200
            headers = {"content-type": "application/json"}

            async def aread(self):
                raise AssertionError("result download must not be buffered")

            async def aiter_raw(self):
                yield b"one"
                yield b"two"

            async def aclose(self):
                pass

        async def fake_send(req, stream=True):
            return StreamingOnlyResp()

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(chunks, [b"one", b"two"])
        self.assertTrue(p._tasks["no-buffer"].terminal)
        self.assertEqual(p._tasks["no-buffer"].status, "success")
        self.assertEqual(p._active_requests, 0)

    async def test_non_async_convert_source_stays_streaming(self):
        p = self.proxy
        await self._prime_running()
        req = await self._make_request("POST", "/v1alpha/convert/source")

        class StreamResp:
            status_code = 200
            headers = {"content-type": "application/json"}

            async def aread(self):
                raise AssertionError("non-async conversion must not be buffered")

            async def aiter_raw(self):
                yield b"one"
                yield b"two"

            async def aclose(self):
                pass

        async def fake_send(req, stream=True):
            return StreamResp()

        with mock.patch.object(p.http_client, "send", fake_send):
            resp = await p.proxy_handler(req)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk)

        self.assertEqual(chunks, [b"one", b"two"])
        self.assertEqual(p._tasks, {})
        self.assertEqual(p._active_requests, 0)


class StartupShutdownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    async def test_startup_takeover_healthy(self):
        p = self.proxy
        created_tasks = []

        real_create_task = asyncio.create_task

        def capture_task(coro):
            t = real_create_task(coro)
            created_tasks.append(t)
            return t

        try:
            with mock.patch.object(p, "_is_running", lambda: True), \
                 mock.patch.object(p, "_has_expected_image", lambda: True), \
                 mock.patch.object(p, "_has_expected_port_binding",
                                    lambda: True), \
                 mock.patch.object(p, "_remediate_restart_policy",
                                    lambda: True), \
                 mock.patch.object(p, "_wait_healthy",
                                    lambda timeout=None: _future(True)), \
                 mock.patch.object(asyncio, "create_task", capture_task):
                await p.on_startup()
            self.assertIs(p._state, p.State.RUNNING)
        finally:
            # idle_watcher-Task abbrechen, damit der Loop sauber endet
            if p._idle_watcher_task is not None:
                p._idle_watcher_task.cancel()
                try:
                    await p._idle_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def test_startup_takeover_dirty_stops_and_stays_stopped(self):
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        try:
            with mock.patch.object(p, "_is_running", lambda: True), \
                 mock.patch.object(p, "_has_expected_image", lambda: True), \
                 mock.patch.object(p, "_has_expected_port_binding",
                                    lambda: True), \
                 mock.patch.object(p, "_remediate_restart_policy",
                                    lambda: True), \
                 mock.patch.object(p, "_wait_healthy",
                                    lambda timeout=None: _future(False)), \
                 mock.patch.object(p, "_run_stop_once", fake_stop):
                await p.on_startup()
            self.assertIs(p._state, p.State.STOPPED)
            self.assertEqual(len(stops), 1)
        finally:
            if p._idle_watcher_task is not None:
                p._idle_watcher_task.cancel()
                try:
                    await p._idle_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def test_startup_takeover_wrong_image_stops_without_health(self):
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        def unexpected_policy():
            raise AssertionError("wrong image must skip RestartPolicy")

        def unexpected_port():
            raise AssertionError("wrong image must skip PortBinding")

        async def unexpected_health(timeout=None):
            raise AssertionError("wrong image must skip health check")

        try:
            with mock.patch.object(p, "_is_running", lambda: True), \
                 mock.patch.object(p, "_has_expected_image", lambda: False), \
                 mock.patch.object(p, "_has_expected_port_binding",
                                    unexpected_port), \
                 mock.patch.object(p, "_remediate_restart_policy",
                                    unexpected_policy), \
                 mock.patch.object(p, "_wait_healthy", unexpected_health), \
                 mock.patch.object(p, "_run_stop_once", fake_stop):
                await p.on_startup()
            self.assertIs(p._state, p.State.STOPPED)
            self.assertEqual(len(stops), 1)
        finally:
            if p._idle_watcher_task is not None:
                p._idle_watcher_task.cancel()
                try:
                    await p._idle_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def test_startup_takeover_wrong_port_binding_stops_without_health(self):
        """#877: PortBinding-Mismatch beim Takeover verhindert _wait_healthy."""
        p = self.proxy
        stops = []

        async def fake_stop():
            stops.append(1)
            return True

        async def unexpected_health(timeout=None):
            raise AssertionError("bad PortBinding must skip health check")

        try:
            with mock.patch.object(p, "_is_running", lambda: True), \
                 mock.patch.object(p, "_has_expected_image", lambda: True), \
                 mock.patch.object(p, "_remediate_restart_policy",
                                    lambda: True), \
                 mock.patch.object(p, "_has_expected_port_binding",
                                    lambda: False), \
                 mock.patch.object(p, "_wait_healthy", unexpected_health), \
                 mock.patch.object(p, "_run_stop_once", fake_stop):
                await p.on_startup()
            self.assertIs(p._state, p.State.STOPPED)
            self.assertEqual(len(stops), 1)
        finally:
            if p._idle_watcher_task is not None:
                p._idle_watcher_task.cancel()
                try:
                    await p._idle_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    async def test_shutdown_sets_terminal_state(self):
        p = self.proxy
        # Simuliere laufenden Watcher
        p._idle_watcher_task = asyncio.create_task(p.idle_watcher())
        # Kleiner Sleep, damit Watcher wirklich laeuft
        await asyncio.sleep(0)

        with mock.patch.object(p, "_is_running_for_dirty", lambda: False), \
             mock.patch.object(p.http_client, "aclose",
                                lambda: _future(None)):
            await p.on_shutdown()
        self.assertIs(p._state, p.State.SHUTDOWN)

    async def test_shutdown_clears_marker_on_clean_stop(self):
        p = self.proxy

        with mock.patch.object(p, "_is_running_for_dirty", lambda: False), \
             mock.patch.object(p.http_client, "aclose",
                                lambda: _future(None)):
            await p.on_shutdown()

        self.assertFalse(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_shutdown_keeps_marker_when_final_stop_fails(self):
        p = self.proxy

        async def fake_stop():
            return False

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p.http_client, "aclose",
                                lambda: _future(None)):
            await p.on_shutdown()

        self.assertTrue(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_shutdown_keeps_marker_on_stop_task_timeout(self):
        p = self.proxy
        p.SHUTDOWN_STOP_GRACE_S = 0.01
        p._stop_task = asyncio.create_task(asyncio.sleep(10.0))

        try:
            with mock.patch.object(p, "_is_running_for_dirty", lambda: False), \
                 mock.patch.object(p.http_client, "aclose",
                                    lambda: _future(None)):
                await p.on_shutdown()
        finally:
            p._stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await p._stop_task
            p._stop_task = None

        self.assertTrue(p.SHUTDOWN_MARKER_PATH.exists())


class IsRunningBestEffortTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    def _fake_proc(self, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_stdout_true_returns_true(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "true\n")):
            self.assertTrue(p._is_running())

    def test_stdout_false_returns_false(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "false\n")):
            self.assertFalse(p._is_running())

    def test_empty_stdout_returns_false(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "")):
            self.assertFalse(p._is_running())

    def test_unexpected_stdout_returns_false(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "maybe")):
            self.assertFalse(p._is_running())

    def test_timeout_returns_false(self):
        p = self.proxy

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

        with mock.patch.object(subprocess, "run", boom):
            self.assertFalse(p._is_running())

    def test_file_not_found_returns_false(self):
        p = self.proxy

        def boom(*a, **kw):
            raise FileNotFoundError("no docker")

        with mock.patch.object(subprocess, "run", boom):
            self.assertFalse(p._is_running())

    def test_unexpected_exception_returns_false(self):
        p = self.proxy

        def boom(*a, **kw):
            raise RuntimeError("unexpected")

        with mock.patch.object(subprocess, "run", boom):
            self.assertFalse(p._is_running())


class ContainerImageValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxy = _reload_module()

    def _fake_proc(self, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_expected_image_returns_true(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(
                0, f"{p.EXPECTED_CONTAINER_IMAGE}\n"),
        ):
            self.assertTrue(p._has_expected_image())

    def test_wrong_image_returns_false(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, "example.com/wrong:latest\n"),
        ):
            self.assertFalse(p._has_expected_image())

    def test_inspect_failure_returns_false(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(1, "", "No such object"),
        ):
            self.assertFalse(p._has_expected_image())

    def test_timeout_returns_false(self):
        p = self.proxy

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

        with mock.patch.object(subprocess, "run", boom):
            self.assertFalse(p._has_expected_image())

    def test_start_rejects_wrong_image_before_side_effects(self):
        p = self.proxy

        def unexpected_policy():
            raise AssertionError("wrong image must skip RestartPolicy")

        def unexpected_port():
            raise AssertionError("wrong image must skip PortBinding")

        with mock.patch.object(p, "_has_expected_image", lambda: False), \
             mock.patch.object(p, "_has_expected_port_binding",
                                unexpected_port), \
             mock.patch.object(p, "_remediate_restart_policy",
                                unexpected_policy):
            self.assertFalse(p._start())


class ContainerPortBindingValidationTests(unittest.IsolatedAsyncioTestCase):
    """Pruefungen fuer _has_expected_port_binding (#877).

    Garantiert, dass Container ohne erwartetes 5001/tcp -> 127.0.0.1:5002
    Mapping nicht als gesund akzeptiert werden, damit _wait_healthy nicht
    auf einen Drittprozess oder einen falsch gebundenen Container faellt.
    """

    def setUp(self):
        self.proxy = _reload_module()

    def _fake_proc(self, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_expected_binding_returns_true(self):
        p = self.proxy
        payload = json.dumps({
            p.EXPECTED_CONTAINER_PORT_SPEC: [
                {"HostIp": p.EXPECTED_HOST_BIND_IP,
                 "HostPort": p.EXPECTED_HOST_BIND_PORT},
            ],
        })
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, payload + "\n"),
        ):
            self.assertTrue(p._has_expected_port_binding())

    def test_missing_port_spec_returns_false(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, "{}\n"),
        ):
            self.assertFalse(p._has_expected_port_binding())

    def test_wrong_host_port_returns_false(self):
        p = self.proxy
        payload = json.dumps({
            p.EXPECTED_CONTAINER_PORT_SPEC: [
                {"HostIp": p.EXPECTED_HOST_BIND_IP, "HostPort": "9999"},
            ],
        })
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, payload + "\n"),
        ):
            self.assertFalse(p._has_expected_port_binding())

    def test_wrong_host_ip_returns_false(self):
        """Public-Bind (0.0.0.0) muss abgelehnt werden — by-design localhost-only."""
        p = self.proxy
        payload = json.dumps({
            p.EXPECTED_CONTAINER_PORT_SPEC: [
                {"HostIp": "0.0.0.0",
                 "HostPort": p.EXPECTED_HOST_BIND_PORT},
            ],
        })
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, payload + "\n"),
        ):
            self.assertFalse(p._has_expected_port_binding())

    def test_inspect_failure_returns_false(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(1, "", "No such object"),
        ):
            self.assertFalse(p._has_expected_port_binding())

    def test_invalid_json_returns_false(self):
        p = self.proxy
        with mock.patch.object(
            subprocess, "run",
            lambda *a, **kw: self._fake_proc(0, "not-json"),
        ):
            self.assertFalse(p._has_expected_port_binding())

    def test_timeout_returns_false(self):
        p = self.proxy

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

        with mock.patch.object(subprocess, "run", boom):
            self.assertFalse(p._has_expected_port_binding())

    def test_start_rejects_wrong_port_binding_before_docker_start(self):
        p = self.proxy

        run_calls = []

        def fake_run(args, **kw):
            run_calls.append(args)
            raise AssertionError("docker start must not run on bad port binding")

        with mock.patch.object(p, "_has_expected_image", lambda: True), \
             mock.patch.object(p, "_remediate_restart_policy",
                                lambda: True), \
             mock.patch.object(p, "_has_expected_port_binding",
                                lambda: False), \
             mock.patch.object(subprocess, "run", fake_run):
            self.assertFalse(p._start())
        self.assertEqual(run_calls, [])


class IsRunningForDirtyTests(unittest.IsolatedAsyncioTestCase):
    """Pinnt den konservativen Dirty-Inspect-Vertrag (Gegenstueck zu
    IsRunningBestEffortTests). Unklare Ausgaben/Fehler -> True.
    """

    def setUp(self):
        self.proxy = _reload_module()

    def _fake_proc(self, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_stdout_true_returns_true(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "true\n")):
            self.assertTrue(p._is_running_for_dirty())

    def test_stdout_false_returns_false(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "false\n")):
            self.assertFalse(p._is_running_for_dirty())

    def test_empty_stdout_returns_true_conservative(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "")):
            self.assertTrue(p._is_running_for_dirty())

    def test_unexpected_stdout_returns_true_conservative(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(0, "maybe")):
            self.assertTrue(p._is_running_for_dirty())

    def test_no_such_object_returns_false(self):
        p = self.proxy
        stderr = "Error: No such object: docling-serve"
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(
                                    1, "", stderr)):
            self.assertFalse(p._is_running_for_dirty())

    def test_daemon_error_returns_true_conservative(self):
        p = self.proxy
        stderr = "Error response from daemon: Cannot connect"
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(
                                    1, "", stderr)):
            self.assertTrue(p._is_running_for_dirty())

    def test_nonzero_empty_output_returns_true(self):
        p = self.proxy
        with mock.patch.object(subprocess, "run",
                                lambda *a, **kw: self._fake_proc(1, "", "")):
            self.assertTrue(p._is_running_for_dirty())

    def test_timeout_returns_true(self):
        p = self.proxy

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

        with mock.patch.object(subprocess, "run", boom):
            self.assertTrue(p._is_running_for_dirty())

    def test_file_not_found_returns_true(self):
        p = self.proxy

        def boom(*a, **kw):
            raise FileNotFoundError("no docker")

        with mock.patch.object(subprocess, "run", boom):
            self.assertTrue(p._is_running_for_dirty())

    def test_os_error_returns_true(self):
        p = self.proxy

        def boom(*a, **kw):
            raise OSError("broken")

        with mock.patch.object(subprocess, "run", boom):
            self.assertTrue(p._is_running_for_dirty())

    def test_unexpected_exception_returns_true(self):
        p = self.proxy

        def boom(*a, **kw):
            raise RuntimeError("unexpected")

        with mock.patch.object(subprocess, "run", boom):
            self.assertTrue(p._is_running_for_dirty())


class DirtyStateTests(unittest.IsolatedAsyncioTestCase):
    """Abdeckung fuer STOPPED_DIRTY-Uebergaenge, _finalize_stop,
    Retry-Scheduler, DIRTY-Gate und SHUTDOWN-Interaktionen.
    """

    def setUp(self):
        self.proxy = _reload_module()
        # Schnelle Retry-Iterationen fuer Tests.
        self.proxy._DIRTY_RETRY_INTERVAL_S = 0.05

    async def asyncTearDown(self):
        p = self.proxy
        task = p._dirty_retry_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            p._dirty_retry_task = None

    # --- _finalize_stop direkt ---

    async def test_finalize_stop_dirty_invariants(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPING
            p._active_requests = 0
            p._backend_failed = True
            p._last_request_time = 42.0

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._finalize_stop(stop_ok=False)

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertEqual(p._active_requests, 0)
        self.assertEqual(p._last_request_time, 0.0)
        self.assertFalse(p._backend_failed)
        self.assertIsNotNone(p._dirty_retry_task)
        self.assertFalse(p._dirty_retry_task.done())

    async def test_finalize_stop_ok_to_stopped(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPING
            p._active_requests = 0
            p._backend_failed = True
            p._last_request_time = 42.0

        await p._finalize_stop(stop_ok=True)

        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(p._last_request_time, 0.0)
        self.assertFalse(p._backend_failed)
        self.assertIsNone(p._dirty_retry_task)

    async def test_finalize_stop_fail_but_inspect_false_to_stopped(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPING
            p._active_requests = 0

        with mock.patch.object(p, "_is_running_for_dirty", lambda: False):
            await p._finalize_stop(stop_ok=False)

        self.assertIs(p._state, p.State.STOPPED)
        self.assertIsNone(p._dirty_retry_task)

    async def test_finalize_stop_shutdown_gate(self):
        """Direkt: im SHUTDOWN darf weder STOPPED noch DIRTY gesetzt werden."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.SHUTDOWN
            p._active_requests = 0

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._finalize_stop(stop_ok=False)

        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertIsNone(p._dirty_retry_task)

    async def test_finalize_stop_reentry_idempotent(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPING
            p._active_requests = 0

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._finalize_stop(stop_ok=False)
            first_task = p._dirty_retry_task
            self.assertIsNotNone(first_task)
            self.assertFalse(first_task.done())
            await p._finalize_stop(stop_ok=False)

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertIs(p._dirty_retry_task, first_task,
                      "Retry-Task muss identisch sein (Idempotenz)")
        self.assertFalse(p._backend_failed)

    # --- ensure_running DIRTY-Gate ---

    async def test_ensure_running_in_dirty_returns_false(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        stops = []

        async def counting_stop():
            stops.append(1)
            return True

        with mock.patch.object(p, "_run_stop_once", counting_stop), \
             mock.patch.object(p, "_start", lambda: True):
            ok = await p.ensure_running()

        self.assertFalse(ok)
        self.assertEqual(p._active_requests, 0)
        self.assertEqual(len(stops), 0, "DIRTY darf keinen Cold-Start triggern")

    async def test_proxy_handler_503_in_dirty(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        scope = {
            "type": "http", "method": "GET", "path": "/h",
            "raw_path": b"/h", "query_string": b"",
            "headers": [(b"host", b"x")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)
        resp = await p.proxy_handler(req)
        self.assertEqual(resp.status_code, 503)

    # --- Dirty-Transition aus Cleanup-Pfaden ---

    async def test_ensure_running_drain_to_dirty(self):
        """F136-Drain-Pfad: Stop-Fehler + Dirty-Inspect True -> DIRTY."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = True

        async def fake_stop():
            return False

        task = asyncio.create_task(p.ensure_running())
        await asyncio.sleep(0.01)

        # Letzter Slot triggert Drain
        with mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._release_slot(ok=False, backend_failed=False)
            # DIRTY-Gate in ensure_running -> task muss False liefern
            result = await asyncio.wait_for(task, timeout=2.0)
        self.assertFalse(result)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertIsNotNone(p._dirty_retry_task)

    async def test_release_slot_drain_to_dirty(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = False

        async def fake_stop():
            return False

        with mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._release_slot(ok=False, backend_failed=True)

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertFalse(p._backend_failed)
        self.assertEqual(p._active_requests, 0)
        self.assertIsNotNone(p._dirty_retry_task)

    async def test_release_slot_drain_writes_shutdown_marker(self):
        """#771: Drain schreibt Shutdown-Overlay vor STOPPING; clean stop loescht."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = False

        seen = {"during": False}

        async def fake_stop():
            seen["during"] = p.SHUTDOWN_MARKER_PATH.exists()
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            await p._release_slot(ok=False, backend_failed=True)

        self.assertTrue(seen["during"])
        self.assertIs(p._state, p.State.STOPPED)
        self.assertFalse(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_release_slot_drain_keeps_marker_on_dirty(self):
        """#771: Stop-Fehler -> STOPPED_DIRTY behaelt Shutdown-Marker."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = False

        with mock.patch.object(p, "_run_stop_once", lambda: _future(False)), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            await p._release_slot(ok=False, backend_failed=True)

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertTrue(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_response_iter_mid_stream_dirty(self):
        """F35: mid-stream ReadError + Stop-Fehler + Dirty -> DIRTY."""
        p = self.proxy

        class FakeResp:
            async def aiter_raw(self):
                yield b"partial"
                raise httpx.ReadError("mid-stream", request=None)

            async def aclose(self):
                pass

        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1

        async def fake_stop():
            return False

        with mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            with self.assertRaises(httpx.ReadError):
                async for _ in p._response_iter(FakeResp()):
                    pass

        self.assertEqual(p._active_requests, 0)
        self.assertFalse(p._backend_failed)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertIsNotNone(p._dirty_retry_task)

    async def test_send_phase_connect_error_dirty(self):
        """F56: proxy_handler send-phase ConnectError + Stop-Fehler + Dirty."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0
            p._last_request_time = 100.0

        async def fake_send(*a, **kw):
            raise httpx.ConnectError("down", request=None)

        async def fake_stop():
            return False

        scope = {
            "type": "http", "method": "GET", "path": "/h",
            "raw_path": b"/h", "query_string": b"",
            "headers": [(b"host", b"x")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)
        with mock.patch.object(p.http_client, "send", fake_send), \
             mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_prepare_vram_for_docling",
                               lambda: _future(None)):
            resp = await p.proxy_handler(req)

        self.assertEqual(resp.status_code, 502)
        self.assertEqual(p._active_requests, 0)
        self.assertFalse(p._backend_failed)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertIsNotNone(p._dirty_retry_task)

    # --- _start_supervisor Dirty ---

    async def test_start_supervisor_health_failure_dirty(self):
        """Health-Failure + Stop-Fehler + Dirty-Inspect True -> DIRTY."""
        p = self.proxy

        finalize_calls = []
        original_finalize = p._finalize_stop

        async def counting_finalize(stop_ok):
            finalize_calls.append(stop_ok)
            await original_finalize(stop_ok)

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(False)), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(False)), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_finalize_stop", counting_finalize):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            ok = await p._start_supervisor()

        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertEqual(len(finalize_calls), 0,
                          "_start_supervisor darf _finalize_stop NICHT aus "
                          "dem try-Block rufen (F26)")
        self.assertTrue(p.STARTUP_MARKER_PATH.exists())
        self.assertIsNotNone(p._dirty_retry_task)

    async def test_start_supervisor_health_failure_not_dirty_when_stop_ok(self):
        """Health-Failure aber Stop-Erfolg -> STOPPED (nicht DIRTY)."""
        p = self.proxy

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(False)), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            ok = await p._start_supervisor()

        self.assertFalse(ok)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertIsNone(p._dirty_retry_task)

    async def test_race_f26_start_supervisor_vs_ensure_running(self):
        """Race: Health-Failure + Dirty vs. paralleler ensure_running.

        Ein zweiter Caller muss entweder 503 bekommen (DIRTY-Gate) oder in
        einen neuen Start gehen, NIEMALS aber an einen bereits beendeten
        `_start_task` haengen bleiben. Finaler State deterministisch DIRTY.
        """
        p = self.proxy
        # Lang genug, dass Retry-Scheduler den Counter nicht stoert
        p._DIRTY_RETRY_INTERVAL_S = 5.0

        finalize_counter = {"try_calls": 0}
        original_finalize = p._finalize_stop

        async def counting_finalize(stop_ok):
            finalize_counter["try_calls"] += 1
            await original_finalize(stop_ok)

        gate = asyncio.Event()

        async def slow_wait_healthy(timeout=None):
            await gate.wait()
            return False

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy", slow_wait_healthy), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(False)), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_finalize_stop", counting_finalize):
            async with p._state_lock:
                p._state = p.State.STOPPED

            caller1 = asyncio.create_task(p.ensure_running())
            await asyncio.sleep(0.02)
            caller2 = asyncio.create_task(p.ensure_running())
            await asyncio.sleep(0.02)
            gate.set()
            res1 = await asyncio.wait_for(caller1, timeout=3.0)
            res2 = await asyncio.wait_for(caller2, timeout=3.0)

        self.assertFalse(res1)
        self.assertFalse(res2)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertEqual(finalize_counter["try_calls"], 0,
                          "_finalize_stop darf nicht aus "
                          "_start_supervisor-try gerufen werden")

    async def test_start_supervisor_base_exception_normal_cleanup_dirty(self):
        """BaseException-Zweig: Cleanup liefert stop_ok=False + Dirty -> DIRTY.

        Cancellation setzt ein, nachdem _start() erfolgreich war; der Cleanup
        laeuft normal durch und liefert stop_ok=False.
        """
        p = self.proxy

        async def cancelling_wait_healthy(timeout=None):
            raise asyncio.CancelledError()

        async def fake_stop():
            return False

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy", cancelling_wait_healthy), \
             mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            async with p._state_lock:
                p._state = p.State.STARTING
                p._state_changed.notify_all()
            with self.assertRaises(asyncio.CancelledError):
                await p._start_supervisor()

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertIsNotNone(p._dirty_retry_task)

    async def test_start_supervisor_shutdown_wins_over_dirty(self):
        """SHUTDOWN gewinnt immer, auch bei target_dirty=True."""
        p = self.proxy

        async with p._state_lock:
            p._state = p.State.STARTING

        async def cancelling_wait_healthy(timeout=None):
            async with p._state_lock:
                p._state = p.State.SHUTDOWN
                p._state_changed.notify_all()
            return False

        async def fake_stop():
            return False

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy", cancelling_wait_healthy), \
             mock.patch.object(p, "_run_stop_once", fake_stop), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            ok = await p._start_supervisor()

        self.assertFalse(ok)
        self.assertIs(p._state, p.State.SHUTDOWN)

    # --- on_startup-Takeover ---

    async def test_on_startup_takeover_to_dirty(self):
        p = self.proxy

        async def fake_stop():
            return False

        try:
            with mock.patch.object(p, "_is_running", lambda: True), \
                 mock.patch.object(p, "_has_expected_image", lambda: True), \
                 mock.patch.object(p, "_has_expected_port_binding",
                                    lambda: True), \
                 mock.patch.object(p, "_remediate_restart_policy",
                                    lambda: True), \
                 mock.patch.object(p, "_wait_healthy",
                                    lambda timeout=None: _future(False)), \
                 mock.patch.object(p, "_run_stop_once", fake_stop), \
                 mock.patch.object(p, "_is_running_for_dirty", lambda: True):
                await p.on_startup()
            self.assertIs(p._state, p.State.STOPPED_DIRTY)
            self.assertIsNotNone(p._dirty_retry_task)
        finally:
            if p._idle_watcher_task is not None:
                p._idle_watcher_task.cancel()
                try:
                    await p._idle_watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    # --- Retry-Scheduler ---

    async def test_retry_recovery_via_inspect(self):
        """Inspect liefert not-running -> Retry-Loop finalisiert zu STOPPED."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        to_thread_calls = []

        async def counting_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(fn)
            return fn(*args, **kwargs)

        stop_calls = []

        async def should_not_stop():
            stop_calls.append(1)
            return True

        patched_inspect = lambda: False  # noqa: E731
        with mock.patch.object(p, "_is_running_for_dirty", patched_inspect), \
             mock.patch.object(p, "_to_thread", counting_to_thread), \
             mock.patch.object(p, "_run_stop_once", should_not_stop):
            p._dirty_retry_task = asyncio.create_task(p._dirty_retry_loop())
            await asyncio.wait_for(p._dirty_retry_task, timeout=2.0)
            saw_inspect_via_to_thread = any(
                fn is patched_inspect for fn in to_thread_calls)

        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(len(stop_calls), 0,
                          "Inspect=not-running muss vor _run_stop_once terminieren")
        self.assertTrue(saw_inspect_via_to_thread,
                        "Retry-Loop muss _to_thread fuer Dirty-Inspect verwenden")

    async def test_retry_stop_success_recovers_stopped(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        inspects = {"n": 0}

        def inspect():
            inspects["n"] += 1
            return True

        with mock.patch.object(p, "_is_running_for_dirty", inspect), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            p._dirty_retry_task = asyncio.create_task(p._dirty_retry_loop())
            await asyncio.wait_for(p._dirty_retry_task, timeout=2.0)

        self.assertIs(p._state, p.State.STOPPED)

    async def test_retry_stop_failure_stays_dirty(self):
        p = self.proxy
        p._DIRTY_RETRY_MAX_ATTEMPTS = 2
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(False)):
            p._dirty_retry_task = asyncio.create_task(p._dirty_retry_loop())
            await asyncio.wait_for(p._dirty_retry_task, timeout=2.0)

        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertTrue(p._dirty_retry_task.done())

    async def test_retry_max_attempts_terminates_and_stays_dirty(self):
        p = self.proxy
        p._DIRTY_RETRY_MAX_ATTEMPTS = 3
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        attempts = {"n": 0}

        async def counting_stop():
            attempts["n"] += 1
            return False

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_run_stop_once", counting_stop):
            p._dirty_retry_task = asyncio.create_task(p._dirty_retry_loop())
            await asyncio.wait_for(p._dirty_retry_task, timeout=3.0)

        self.assertEqual(attempts["n"], 3)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertTrue(p._dirty_retry_task.done())

    # --- Shutdown-Interaktionen ---

    async def test_shutdown_cancels_dirty_retry(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        # Langsamer Retry, damit on_shutdown ihn tatsaechlich canceln kann
        p._DIRTY_RETRY_INTERVAL_S = 5.0

        with mock.patch.object(p, "_is_running_for_dirty", lambda: True), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(False)):
            async with p._state_lock:
                p._ensure_retry_scheduler()
            task = p._dirty_retry_task
            self.assertIsNotNone(task)
            await asyncio.sleep(0.05)

            with mock.patch.object(p, "_is_running", lambda: False), \
                 mock.patch.object(p.http_client, "aclose",
                                    lambda: _future(None)):
                await p.on_shutdown()

        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertIsNone(p._dirty_retry_task)
        self.assertTrue(task.done())

    async def test_shutdown_shares_dirty_stop_task(self):
        """F57: Retry-Scheduler mid-stop + on_shutdown -> geteilter _stop_task."""
        p = self.proxy
        p._DIRTY_RETRY_INTERVAL_S = 0.01

        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        stop_started = threading.Event()
        stop_release = threading.Event()
        stop_runs = []

        def blocking_stop():
            stop_started.set()
            stop_release.wait(timeout=5.0)
            stop_runs.append(1)
            return False

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        inspect_calls = 0

        async def threaded_to_thread(func, /, *args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor, lambda: func(*args, **kwargs))

        def inspect_running_for_dirty():
            nonlocal inspect_calls
            inspect_calls += 1
            return inspect_calls == 1

        try:
            with mock.patch.object(p, "_to_thread", threaded_to_thread), \
                 mock.patch.object(p, "_is_running_for_dirty",
                                   inspect_running_for_dirty), \
                 mock.patch.object(p, "_stop", blocking_stop), \
                 mock.patch.object(p, "_is_running", lambda: False), \
                 mock.patch.object(p.http_client, "aclose",
                                    lambda: _future(None)):
                async with p._state_lock:
                    p._ensure_retry_scheduler()
                retry_task = p._dirty_retry_task
                deadline = time.monotonic() + 2.0
                while not stop_started.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                self.assertTrue(stop_started.is_set())
                shutdown_task = asyncio.create_task(p.on_shutdown())
                await asyncio.sleep(0.05)
                stop_release.set()
                await asyncio.wait_for(shutdown_task, timeout=3.0)
        finally:
            stop_release.set()
            executor.shutdown(wait=True)

        self.assertIs(p._state, p.State.SHUTDOWN)
        self.assertEqual(len(stop_runs), 1,
                          "Genau ein _stop()-Aufruf (single-flight)")
        self.assertIsNone(p._dirty_retry_task)
        self.assertTrue(retry_task.done())

    # --- Keine Duplikat-Drains ---

    async def test_no_duplicate_drains_in_dirty(self):
        p = self.proxy
        p._DIRTY_RETRY_INTERVAL_S = 5.0  # Kein Retry-Stop im Testfenster
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY

        stop_calls = []

        async def counting_stop():
            stop_calls.append(1)
            return True

        with mock.patch.object(p, "_run_stop_once", counting_stop):
            # Mehrere ensure_running-Caller in DIRTY -> keine Slots, kein Drain
            results = await asyncio.gather(
                *(p.ensure_running() for _ in range(5)))

        self.assertFalse(any(results))
        self.assertEqual(len(stop_calls), 0)


class LifecycleEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Interner Lifecycle-Snapshot-Endpoint `GET /_internal/lifecycle` (#278).

    Dashboard-Drain-Pfad liest diesen Endpoint vor `docker stop`.
    Snapshot ist atomar (alle Werte unter `_state_lock`), ohne Side-Effects,
    liefert `Cache-Control: no-store`.
    """

    def setUp(self):
        self.proxy = _reload_module()

    async def _lifecycle_snapshot(self) -> dict:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/_internal/lifecycle",
            "raw_path": b"/_internal/lifecycle",
            "query_string": b"",
            "headers": [(b"host", b"localhost")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)
        return await self.proxy.lifecycle_handler(req)

    async def test_state_all_values(self):
        p = self.proxy
        for state in (p.State.STOPPED, p.State.STARTING, p.State.RUNNING,
                      p.State.STOPPING, p.State.STOPPED_DIRTY,
                      p.State.SHUTDOWN):
            async with p._state_lock:
                p._state = state
                p._active_requests = 0
                p._last_request_time = 0.0
                p._backend_failed = False
            resp = await self._lifecycle_snapshot()
            self.assertEqual(resp.status_code, 200)
            import json as _json
            body = _json.loads(bytes(resp.body).decode())
            self.assertEqual(body["state"], state.value)
            self.assertIsInstance(body["state"], str)

    async def test_snapshot_schema_types(self):
        p = self.proxy
        import time as _time
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 2
            p._last_request_time = _time.monotonic()
            p._backend_failed = False
        resp = await self._lifecycle_snapshot()
        import json as _json
        body = _json.loads(bytes(resp.body).decode())
        self.assertEqual(body["state"], "running")
        self.assertIsInstance(body["active_requests"], int)
        self.assertEqual(body["active_requests"], 2)
        self.assertIsInstance(body["last_request_age_s"], float)
        self.assertGreaterEqual(body["last_request_age_s"], 0.0)
        self.assertIs(body["backend_failed"], False)

    async def test_last_request_age_null_when_no_request(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED
            p._active_requests = 0
            p._last_request_time = 0.0
            p._backend_failed = False
        resp = await self._lifecycle_snapshot()
        import json as _json
        body = _json.loads(bytes(resp.body).decode())
        self.assertIsNone(body["last_request_age_s"])

    async def test_backend_failed_flag_visible(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 1
            p._backend_failed = True
        resp = await self._lifecycle_snapshot()
        import json as _json
        body = _json.loads(bytes(resp.body).decode())
        self.assertIs(body["backend_failed"], True)

    async def test_snapshot_no_side_effects(self):
        """Snapshot darf State/Counter/Timestamp nicht veraendern."""
        p = self.proxy
        import time as _time
        now = _time.monotonic()
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 3
            p._last_request_time = now
            p._backend_failed = False
        await self._lifecycle_snapshot()
        self.assertIs(p._state, p.State.RUNNING)
        self.assertEqual(p._active_requests, 3)
        self.assertEqual(p._last_request_time, now)
        self.assertFalse(p._backend_failed)

    async def test_cache_control_no_store_header(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
        resp = await self._lifecycle_snapshot()
        self.assertEqual(resp.headers.get("cache-control"), "no-store")

    async def test_snapshot_atomic_under_lock(self):
        """Snapshot sieht nie einen Mix-Zustand: entweder komplett vor der
        Mutation oder komplett danach.
        """
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 5
            p._last_request_time = 100.0
            p._backend_failed = False

        mut_release = asyncio.Event()

        async def mutator():
            async with p._state_lock:
                p._active_requests = 0
                p._last_request_time = 200.0
                p._backend_failed = True
                await mut_release.wait()

        mut_task = asyncio.create_task(mutator())
        await asyncio.sleep(0.01)  # Mutator haelt den Lock
        snap_task = asyncio.create_task(self._lifecycle_snapshot())
        await asyncio.sleep(0.01)
        self.assertFalse(snap_task.done(),
                         "Snapshot muss auf Lock warten")
        mut_release.set()
        await mut_task
        resp = await snap_task
        import json as _json
        body = _json.loads(bytes(resp.body).decode())
        # Nach Mutator-Release: alle Werte aus dem mutated-Snapshot
        self.assertEqual(body["active_requests"], 0)
        self.assertIs(body["backend_failed"], True)

    async def test_vram_lease_active_matrix(self):
        """`_vram_lease_active()` deckt die Matrix aus TODO_VRAM_LEASE_V3 ab.

        Lease aktiv in STARTING, STOPPED_DIRTY, STOPPING (#503), RUNNING+active>0.
        Lease NICHT aktiv in STOPPED, RUNNING+active=0, SHUTDOWN.
        """
        p = self.proxy
        cases = [
            (p.State.STARTING, 0, True),
            (p.State.STOPPED_DIRTY, 0, True),
            (p.State.RUNNING, 1, True),
            (p.State.RUNNING, 5, True),
            (p.State.RUNNING, 0, False),
            (p.State.STOPPED, 0, False),
            # #503: STOPPING haelt Lease bis _finalize_stop in STOPPED/STOPPED_DIRTY wechselt.
            (p.State.STOPPING, 0, True),
            (p.State.SHUTDOWN, 0, False),
        ]
        for state, active, expected in cases:
            async with p._state_lock:
                p._state = state
                p._active_requests = active
                got = await p._vram_lease_active()
            self.assertEqual(
                got, expected,
                f"state={state.value} active={active} expected={expected}",
            )

    async def test_lifecycle_includes_vram_lease_active(self):
        """Response-Schema enthaelt vram_lease_active (bool) in jedem State."""
        p = self.proxy
        cases = [
            (p.State.STARTING, 0, True),
            (p.State.STOPPED_DIRTY, 0, True),
            (p.State.RUNNING, 2, True),
            (p.State.RUNNING, 0, False),
            (p.State.STOPPED, 0, False),
            # #503: STOPPING haelt Lease bis zum finalize_stop.
            (p.State.STOPPING, 0, True),
            (p.State.SHUTDOWN, 0, False),
        ]
        import json as _json
        for state, active, expected in cases:
            async with p._state_lock:
                p._state = state
                p._active_requests = active
                p._last_request_time = 0.0
                p._backend_failed = False
            resp = await self._lifecycle_snapshot()
            body = _json.loads(bytes(resp.body).decode())
            self.assertIn("vram_lease_active", body)
            self.assertIsInstance(body["vram_lease_active"], bool)
            self.assertEqual(
                body["vram_lease_active"], expected,
                f"state={state.value} active={active}",
            )

    async def test_lifecycle_schema_contains_proxy_fields(self):
        """Schema enthaelt die Felder, die kiron-proxy aktuell auswertet.
        """
        p = self.proxy
        import json as _json
        import time as _time
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 3
            p._last_request_time = _time.monotonic()
            p._backend_failed = False
        resp = await self._lifecycle_snapshot()
        body = _json.loads(bytes(resp.body).decode())
        self.assertEqual(body["state"], "running")
        self.assertEqual(body["active_requests"], 3)
        self.assertIsInstance(body["last_request_age_s"], float)
        self.assertIs(body["backend_failed"], False)
        # Simulierter V5-Consumer: _drain_done-Logik aus app.py
        TRANSIENT = {"starting", "stopping"}

        def drain_done(snap: dict) -> bool:
            if snap.get("active_requests", 0) != 0:
                return False
            return snap.get("state") not in TRANSIENT

        self.assertFalse(drain_done(body))
        async with p._state_lock:
            p._active_requests = 0
        resp2 = await self._lifecycle_snapshot()
        body2 = _json.loads(bytes(resp2.body).decode())
        self.assertTrue(drain_done(body2))

    async def test_lease_under_lock(self):
        """Lease-Compute sieht immer konsistenten Snapshot, auch wenn
        _active_requests parallel mutiert wird.
        """
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0

        mut_release = asyncio.Event()

        async def mutator():
            async with p._state_lock:
                p._active_requests = 7
                await mut_release.wait()

        mut_task = asyncio.create_task(mutator())
        await asyncio.sleep(0.01)

        async def read_lease():
            async with p._state_lock:
                return await p._vram_lease_active()

        read_task = asyncio.create_task(read_lease())
        await asyncio.sleep(0.01)
        self.assertFalse(read_task.done(),
                         "Lease-Read muss auf Lock warten")
        mut_release.set()
        await mut_task
        got = await read_task
        self.assertTrue(got,
                        "Nach Mutator-Release: active_requests=7 → Lease aktiv")

    async def test_lifecycle_route_bypasses_catch_all(self):
        """`/_internal/lifecycle` landet beim Lifecycle-Handler und NICHT
        beim catch_all -> Backend. Kontrolle via `backend_called`-Liste:
        /_internal/lifecycle darf das Backend nicht erreichen, /health schon.
        """
        p = self.proxy

        backend_called: list[str] = []

        class FakeResp:
            def __init__(self):
                self.status_code = 200
                self.headers: dict[str, str] = {}

            async def aclose(self):
                pass

            async def aiter_raw(self):
                if False:
                    yield b""

        async def fake_send(req, stream=True):
            backend_called.append(req.url.path)
            return FakeResp()

        with mock.patch.object(p.http_client, "send", fake_send), \
             mock.patch.object(p, "_prepare_vram_for_docling",
                               lambda: _future(None)):
            # State so setzen, dass proxy_handler fuer /health durchlaeuft
            async with p._state_lock:
                p._state = p.State.RUNNING
                p._last_request_time = 100.0
                p._active_requests = 0
                p._backend_failed = False

            transport = httpx.ASGITransport(app=p.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                r = await client.get("/_internal/lifecycle")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.headers.get("cache-control"), "no-store")
                body = r.json()
                self.assertIn("state", body)
                self.assertIn("active_requests", body)
                # Kontrolle: /health muss durch catch_all an das Backend gehen
                r2 = await client.get("/health")
                self.assertEqual(r2.status_code, 200)

        self.assertNotIn("/_internal/lifecycle", backend_called,
                         "Lifecycle-Route darf NICHT via catch_all an das "
                         "Backend weitergereicht werden")
        self.assertIn("/health", backend_called,
                      "Kontrolle: normale Pfade landen via catch_all beim "
                      "Backend")


class StartEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Interner Coldstart-Endpoint `POST /_internal/start` (#560).

    Dashboard delegiert den Coldstart an den docling-proxy, damit der
    VRAM-Marker waehrend des kompletten Modellladevorgangs gehalten wird
    statt nach dem ~1-3s `docker start` schon wieder gecleart zu sein.
    """

    def setUp(self):
        self.proxy = _reload_module()

    async def _post_start(self):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/_internal/start",
            "raw_path": b"/_internal/start",
            "query_string": b"",
            "headers": [(b"host", b"localhost")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)
        return await self.proxy.start_handler(req)

    async def test_already_running_returns_200_started_false(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
        resp = await self._post_start()
        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["started"], False)
        self.assertEqual(body["state"], "running")

    async def test_shutdown_returns_503(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.SHUTDOWN
        resp = await self._post_start()
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["started"], False)

    async def test_stopped_dirty_returns_409(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY
        resp = await self._post_start()
        self.assertEqual(resp.status_code, 409)

    async def test_cooldown_returns_503_without_start_attempt(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED
            p._start_cooldown_until = time.monotonic() + 300.0

        async def unexpected_start():
            raise AssertionError("Cooldown must skip _run_start_once")

        with mock.patch.object(p, "_run_start_once", unexpected_start):
            resp = await self._post_start()

        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["started"], False)
        self.assertEqual(body["state"], "stopped")
        self.assertEqual(body["error"], "start_cooldown")

    async def test_cold_start_drives_to_running(self):
        p = self.proxy

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(True)), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STOPPED
            resp = await self._post_start()
        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["started"], True)
        self.assertEqual(body["state"], "running")
        self.assertIs(p._state, p.State.RUNNING)
        # Wichtig: Endpoint reserviert KEINEN Slot.
        self.assertEqual(p._active_requests, 0)

    async def test_cold_start_health_fails_returns_503(self):
        p = self.proxy

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                                lambda timeout=None: _future(False)), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STOPPED
            resp = await self._post_start()
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["started"], False)
        self.assertEqual(body["error"], "start_failed")

    async def test_concurrent_start_calls_share_single_flight(self):
        """Mehrere parallele POST /_internal/start triggern genau einen
        Cold-Start im Single-Flight-Supervisor (`_run_start_once`).
        """
        p = self.proxy
        start_calls = 0
        gate = asyncio.Event()

        async def slow_health(timeout=None):
            await gate.wait()
            return True

        def counting_start():
            nonlocal start_calls
            start_calls += 1
            return True

        with mock.patch.object(p, "_prepare_vram_for_docling",
                                lambda: _future(None)), \
             mock.patch.object(p, "_start", counting_start), \
             mock.patch.object(p, "_wait_healthy", slow_health), \
             mock.patch.object(p, "_run_stop_once",
                                lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STOPPED
            callers = [asyncio.create_task(self._post_start())
                       for _ in range(3)]
            await asyncio.sleep(0.05)
            gate.set()
            results = await asyncio.gather(*callers)

        self.assertEqual(start_calls, 1,
                         "Nur ein _start() im Single-Flight")
        for resp in results:
            self.assertEqual(resp.status_code, 200)
        # Genau einer meldet started=True (der STOPPED->STARTING owner),
        # die anderen sehen RUNNING und melden started=False.
        starteds = [json.loads(bytes(r.body).decode())["started"]
                    for r in results]
        self.assertEqual(sum(1 for s in starteds if s), 1)
        # Alle Caller landen auf state=running
        for resp in results:
            body = json.loads(bytes(resp.body).decode())
            self.assertEqual(body["state"], "running")
        self.assertEqual(p._active_requests, 0)


class StopEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Interner Stop-Endpoint `POST /_internal/stop` (#617)."""

    def setUp(self):
        self.proxy = _reload_module()

    async def _post_stop(self, body=b"{}"):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/_internal/stop",
            "raw_path": b"/_internal/stop",
            "query_string": b"",
            "headers": [(b"host", b"localhost"),
                        (b"content-type", b"application/json")],
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        from starlette.requests import Request
        req = Request(scope, receive=receive)
        return await self.proxy.stop_handler(req)

    async def test_already_stopped_returns_200(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED
        resp = await self._post_stop()
        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["stopped"], False)
        self.assertEqual(body["state"], "stopped")

    async def test_running_without_active_requests_stops_and_updates_state(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0

        with mock.patch.object(p, "_run_stop_once", lambda: _future(True)):
            resp = await self._post_stop()

        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertIs(body["stopped"], True)
        self.assertEqual(body["state"], "stopped")
        self.assertIs(p._state, p.State.STOPPED)

    async def test_active_requests_without_force_returns_409(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 2

        with mock.patch.object(p, "_run_stop_once") as stop_mock:
            resp = await self._post_stop()

        self.assertEqual(resp.status_code, 409)
        self.assertFalse(stop_mock.called)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["active_requests"], 2)

    async def test_force_stops_with_active_requests(self):
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 2

        with mock.patch.object(p, "_run_stop_once", lambda: _future(True)):
            resp = await self._post_stop(b'{"force": true}')

        self.assertEqual(resp.status_code, 200)
        self.assertIs(p._state, p.State.STOPPED)
        self.assertEqual(p._active_requests, 0)

    async def test_running_writes_shutdown_marker_during_stopping(self):
        """#771: Shutdown-Overlay-Marker liegt waehrend STOPPING vor."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0

        seen = {"during": False}

        async def fake_stop():
            seen["during"] = p.SHUTDOWN_MARKER_PATH.exists()
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            resp = await self._post_stop()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(seen["during"])
        self.assertFalse(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_dirty_path_writes_shutdown_marker_during_stopping(self):
        """#771: STOPPED_DIRTY -> STOPPING auch mit Marker."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.STOPPED_DIRTY
            p._active_requests = 0

        seen = {"during": False}

        async def fake_stop():
            seen["during"] = p.SHUTDOWN_MARKER_PATH.exists()
            return True

        with mock.patch.object(p, "_run_stop_once", fake_stop):
            resp = await self._post_stop()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(seen["during"])
        self.assertFalse(p.SHUTDOWN_MARKER_PATH.exists())

    async def test_keeps_marker_on_dirty(self):
        """#771: Stop-Fehler -> STOPPED_DIRTY behaelt Shutdown-Marker bis TTL."""
        p = self.proxy
        async with p._state_lock:
            p._state = p.State.RUNNING
            p._active_requests = 0

        with mock.patch.object(p, "_run_stop_once", lambda: _future(False)), \
             mock.patch.object(p, "_is_running_for_dirty", lambda: True):
            resp = await self._post_stop()
        self.assertEqual(resp.status_code, 503)
        self.assertIs(p._state, p.State.STOPPED_DIRTY)
        self.assertTrue(p.SHUTDOWN_MARKER_PATH.exists())


class InflightGpuDrainTests(unittest.IsolatedAsyncioTestCase):
    """Cross-Service Drain vor Docling-Start (#620)."""

    def setUp(self):
        self.proxy = _reload_module()
        self.proxy.KIRON_ACTIVE_REQUESTS_URL = "http://kiron.test/active"

    async def test_drain_without_active_requests_url_raises(self):
        p = self.proxy
        p.KIRON_ACTIVE_REQUESTS_URL = ""

        with self.assertRaises(p.VramGateError):
            await p._drain_inflight_gpu_requests()

    async def test_drain_http_error_raises(self):
        p = self.proxy

        class FakeResp:
            status_code = 503

            def json(self):
                return {"requests": []}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._drain_inflight_gpu_requests()

    async def test_drain_request_error_raises(self):
        p = self.proxy

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                raise httpx.ConnectError("down", request=None)

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._drain_inflight_gpu_requests()

    async def test_start_drains_inflight_before_free_vram(self):
        p = self.proxy
        order = []

        async def drain():
            order.append("drain")

        async def free():
            order.append("free")

        with mock.patch.object(p, "_drain_inflight_gpu_requests", drain), \
             mock.patch.object(p, "_free_vram_for_docling", free), \
             mock.patch.object(p, "_start", lambda: True), \
             mock.patch.object(p, "_wait_healthy",
                               lambda timeout=None: _future(True)), \
             mock.patch.object(p, "_run_stop_once",
                               lambda: _future(True)):
            async with p._state_lock:
                p._state = p.State.STOPPED
            ok = await p._run_start_once()

        self.assertTrue(ok)
        self.assertEqual(order[:2], ["drain", "free"])

    async def test_drain_waits_until_active_requests_clear(self):
        p = self.proxy
        p.GPU_INFLIGHT_DRAIN_TIMEOUT_S = 1.0
        p.GPU_INFLIGHT_DRAIN_POLL_S = 0.01
        calls = {"n": 0}
        urls = []

        class FakeResp:
            status_code = 200

            def json(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return {"requests": [
                        {"path": "/api/generate", "state": "active",
                         "model": "qwen"}
                    ]}
                return {"requests": []}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                urls.append(url)
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            await p._drain_inflight_gpu_requests()
        self.assertEqual(calls["n"], 2)
        self.assertEqual(urls, [p.KIRON_ACTIVE_REQUESTS_URL,
                                p.KIRON_ACTIVE_REQUESTS_URL])

    async def test_drain_treats_deberta_paths_as_gpu_active(self):
        p = self.proxy
        p.GPU_INFLIGHT_DRAIN_TIMEOUT_S = 0.02
        p.GPU_INFLIGHT_DRAIN_POLL_S = 0.01

        class FakeResp:
            status_code = 200

            def json(self):
                return {"requests": [
                    {"path": "/api/rerank", "state": "active",
                     "model": "deberta"}
                ]}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._drain_inflight_gpu_requests()

    async def test_drain_timeout_raises_vram_gate_error(self):
        p = self.proxy
        p.GPU_INFLIGHT_DRAIN_TIMEOUT_S = 0.02
        p.GPU_INFLIGHT_DRAIN_POLL_S = 0.01

        class FakeResp:
            status_code = 200

            def json(self):
                return {"requests": [
                    {"path": "/api/chat", "state": "active", "model": "m"}
                ]}

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return FakeResp()

        with mock.patch.object(httpx, "AsyncClient", FakeClient):
            with self.assertRaises(p.VramGateError):
                await p._drain_inflight_gpu_requests()


class CorruptMarkerRecoveryTests(unittest.TestCase):
    """Korrupte Marker-Dateien duerfen Writes nicht indefinite blockieren."""

    def setUp(self):
        self.proxy = _reload_module()

    def _write_raw(self, path, content):
        path.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(path, 0o600)

    def test_invalid_json_does_not_block_new_marker(self):
        p = self.proxy
        self._write_raw(p.STARTUP_MARKER_PATH, "{not-json")
        data, active = p._marker_payload(p.STARTUP_MARKER_PATH)
        self.assertIsNone(data)
        self.assertFalse(active)
        token = p._write_vram_marker("startup", ttl_s=1.0)
        self.assertIsInstance(token, str)
        _, active_after = p._marker_payload(p.STARTUP_MARKER_PATH)
        self.assertTrue(active_after)

    def test_non_dict_json_does_not_block_new_marker(self):
        p = self.proxy
        self._write_raw(p.STARTUP_MARKER_PATH, "[1,2,3]")
        data, active = p._marker_payload(p.STARTUP_MARKER_PATH)
        self.assertIsNone(data)
        self.assertFalse(active)
        token = p._write_vram_marker("startup", ttl_s=1.0)
        self.assertIsInstance(token, str)

    def test_dict_without_ttl_fields_does_not_block_new_marker(self):
        p = self.proxy
        self._write_raw(p.STARTUP_MARKER_PATH, json.dumps({"token": "abc"}))
        data, active = p._marker_payload(p.STARTUP_MARKER_PATH)
        self.assertIsNone(data)
        self.assertFalse(active)
        token = p._write_vram_marker("startup", ttl_s=1.0)
        self.assertIsInstance(token, str)


def _future(value):
    """Hilfsfunktion: Gibt eine Coroutine zurueck, die value liefert."""
    async def _c():
        return value
    return _c()


if __name__ == "__main__":
    unittest.main()
