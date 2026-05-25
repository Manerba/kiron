"""Stdlib-Unittest fuer set_docling_device und Docling-Control-Fehlerpfade.

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_docling_device

Mockt Docker-Inspect/Create. Deckt Port-Migration (F96/F102) und
JSON-Fehlervertrag (F126) fuer /api/docling/device|start|stop ab.
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
from unittest import mock

import httpx


TEST_RUNTIME_DIR = Path(os.environ.get(
    "KIRON_TEST_RUNTIME_DIR",
    "/tmp/kiron-proxy-test-runtime",
))


class _FakeRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _load_app():
    import importlib
    import sys
    sys.path.insert(0, "/opt/kiron/services/kiron-proxy")
    import app as _app  # noqa
    module = importlib.reload(_app)
    TEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for path in TEST_RUNTIME_DIR.glob("docling-vram-*.json"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in TEST_RUNTIME_DIR.glob("gpu-service-loading.json"):
        try:
            path.unlink()
        except OSError:
            pass
    module.vram_lease.RUNTIME_MARKER_DIR = TEST_RUNTIME_DIR
    module.vram_lease.STARTUP_MARKER_PATH = (
        TEST_RUNTIME_DIR / "docling-vram-startup.json")
    module.vram_lease.SHUTDOWN_MARKER_PATH = (
        TEST_RUNTIME_DIR / "docling-vram-shutdown.json")
    module.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH = (
        TEST_RUNTIME_DIR / "gpu-service-loading.json")

    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    module._to_thread = inline_to_thread
    module._trigger_docling_proxy_stop = _async_return(
        (502, {"error": "test fallback"})
    )
    return module


def _async_return(value):
    """Liefert eine AsyncMock-aehnliche Coroutine-Factory mit festem Wert."""
    async def _coro(*args, **kwargs):
        return value
    return _coro


def _async_sequence(values):
    """Liefert eine Coroutine-Factory, die pro Aufruf den naechsten Wert
    aus `values` zurueckgibt. Letzter Wert wird wiederholt wenn die
    Sequenz erschoepft ist."""
    iterator = iter(values)
    last = values[-1] if values else None

    async def _coro(*args, **kwargs):
        nonlocal last
        try:
            v = next(iterator)
            last = v
            return v
        except StopIteration:
            return last
    return _coro


class PortInspectionTests(unittest.TestCase):
    def setUp(self):
        self.app_mod = _load_app()

    def test_localhost_only_returns_true(self):
        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })
        with mock.patch.object(subprocess, "run",
                                return_value=_FakeRun(stdout=ports_json)):
            self.assertIs(
                self.app_mod._docling_port_bound_localhost_only(), True)

    def test_wildcard_returns_false(self):
        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "5002"}],
        })
        with mock.patch.object(subprocess, "run",
                                return_value=_FakeRun(stdout=ports_json)):
            self.assertIs(
                self.app_mod._docling_port_bound_localhost_only(), False)

    def test_no_bindings_returns_false(self):
        ports_json = json.dumps({"5001/tcp": None})
        with mock.patch.object(subprocess, "run",
                                return_value=_FakeRun(stdout=ports_json)):
            self.assertIs(
                self.app_mod._docling_port_bound_localhost_only(), False)

    def test_null_output_returns_none(self):
        with mock.patch.object(subprocess, "run",
                                return_value=_FakeRun(stdout="null")):
            self.assertIsNone(
                self.app_mod._docling_port_bound_localhost_only())

    def test_timeout_returns_none(self):
        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=5)
        with mock.patch.object(subprocess, "run", boom):
            self.assertIsNone(
                self.app_mod._docling_port_bound_localhost_only())

    def test_file_not_found_returns_none(self):
        with mock.patch.object(subprocess, "run",
                                side_effect=FileNotFoundError()):
            self.assertIsNone(
                self.app_mod._docling_port_bound_localhost_only())


class SetDoclingDeviceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_app()
        # Drain-Helper patchen: sonst schlaegt jeder Testaufruf den 2s
        # httpx-Timeout zum nicht laufenden Docling-Proxy zu Buche. None =
        # Lifecycle-Endpoint unerreichbar -> Fallback "direkt stoppen".
        patcher = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(None))
        patcher.start()
        self.addCleanup(patcher.stop)
        restart_patcher = mock.patch.object(
            self.app_mod, "_docling_restart_policy_safe",
            lambda: True)
        restart_patcher.start()
        self.addCleanup(restart_patcher.stop)

    async def test_unchanged_when_device_and_port_match(self):
        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cuda\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        # FastAPI-dict-Response
        self.assertEqual(resp, {"status": "unchanged", "device": "cuda"})

    async def test_recreate_when_device_matches_but_port_wildcard(self):
        """F96: Device gleich aber Port wildcard -> Recreate, nicht unchanged."""
        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "5002"}],
        })
        commands_run = []

        def fake_run(cmd, **kw):
            commands_run.append(cmd)
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cuda\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                # Port-Bind muss 127.0.0.1:5002:5001 sein
                self.assertIn("127.0.0.1:5002:5001", cmd)
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        # FastAPI dict-Response
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})

    async def test_recreate_when_port_unknown(self):
        """Nicht pruefbare Portbindung -> nicht unchanged, sondern Recreate."""
        commands_run = []

        def fake_run(cmd, **kw):
            commands_run.append(cmd)
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cuda\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(returncode=1, stderr="No such object")
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                self.assertIn("127.0.0.1:5002:5001", cmd)
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})

    async def test_device_change_recreates(self):
        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cuda\n")
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cpu"})
        self.assertEqual(resp, {"status": "recreated", "device": "cpu"})

    async def test_device_invalid_returns_400_json(self):
        resp = await self.app_mod.set_docling_device({"device": "tpu"})
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_device_inspect_timeout_json_error(self):
        def boom(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)
        with mock.patch.object(subprocess, "run", side_effect=boom):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 504)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_device_inspect_file_not_found_json_error(self):
        def boom(cmd, **kw):
            raise FileNotFoundError("no docker")
        with mock.patch.object(subprocess, "run", side_effect=boom):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 500)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_device_inspect_unexpected_exception_json_error(self):
        def boom(cmd, **kw):
            raise RuntimeError("oops")
        with mock.patch.object(subprocess, "run", side_effect=boom):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 500)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_foreign_startup_marker_returns_409(self):
        """Fremder startup-Marker -> 409 statt 500 (Issue #870)."""
        with mock.patch.object(
            self.app_mod.vram_lease, "write_overlay_marker",
            side_effect=self.app_mod.vram_lease.MarkerOwnershipError("foreign"),
        ):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)


class StartStopControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_app()
        patcher = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(None))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _run_start(self, status, body):
        helper = _async_return((status, body))
        with mock.patch.object(
                self.app_mod, "_trigger_docling_proxy_start", new=helper):
            return await self.app_mod.start_docling()

    async def test_start_ok_returns_json(self):
        resp = await self._run_start(
            200, {"started": True, "state": "running"})
        self.assertEqual(resp, {"status": "started"})

    async def test_start_already_running_returns_already(self):
        resp = await self._run_start(
            200, {"started": False, "state": "running"})
        self.assertEqual(resp, {"status": "started", "already": True})

    async def test_start_proxy_timeout_returns_504(self):
        resp = await self._run_start(
            504, {"error": "Start: docling-proxy Timeout"})
        self.assertEqual(resp.status_code, 504)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_start_proxy_unreachable_returns_502(self):
        resp = await self._run_start(
            502, {"error": "Start: docling-proxy nicht erreichbar"})
        self.assertEqual(resp.status_code, 502)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_start_failure_returns_503(self):
        resp = await self._run_start(
            503, {"started": False, "state": "stopped",
                  "error": "start_failed"})
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["error"], "start_failed")

    async def test_start_dirty_returns_409(self):
        resp = await self._run_start(
            409, {"started": False, "state": "stopped_dirty",
                  "error": "stopped_dirty"})
        self.assertEqual(resp.status_code, 409)

    async def test_stop_timeout_json_error(self):
        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=30)
        with mock.patch.object(subprocess, "run", side_effect=boom):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp.status_code, 504)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)

    async def test_stop_ok_returns_json(self):
        with mock.patch.object(subprocess, "run",
                                return_value=_FakeRun()):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp, {"status": "stopped"})

    async def test_stop_file_not_found_json_error(self):
        with mock.patch.object(subprocess, "run",
                                side_effect=FileNotFoundError("nope")):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp.status_code, 500)

    async def test_stop_unexpected_json_error(self):
        with mock.patch.object(subprocess, "run",
                                side_effect=RuntimeError("boom")):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp.status_code, 500)

    async def test_stop_foreign_shutdown_marker_returns_409(self):
        """Fremder shutdown-Marker -> 409 statt 500 (Issue #870)."""
        with mock.patch.object(
            self.app_mod.vram_lease, "write_overlay_marker",
            side_effect=self.app_mod.vram_lease.MarkerOwnershipError("foreign"),
        ):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)


class GpuServiceLeaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_app()

    def _blocked_decision(self, service="GPU-Service"):
        return self.app_mod.vram_lease.GPUGateDecision(
            allowed=False,
            reason="gpu_overlay_active",
            status_code=409,
            marker_kind="gpu_service_loading",
            lease_state="overlay",
            service_name=service,
        )

    async def test_embedding_start_is_lifecycle_only_when_docling_lease_active(self):
        with mock.patch.object(
                self.app_mod.vram_lease, "effective_snapshot",
                new=_async_return(True)), \
             mock.patch.object(self.app_mod, "_json_service_health",
                               new=_async_return((True, {"status": "no_model"}, 503))), \
             mock.patch.object(subprocess, "run") as run_mock:
            resp = await self.app_mod.start_embedding()
        self.assertEqual(resp["status"], "started")
        self.assertTrue(resp["already"])
        self.assertFalse(run_mock.called)

    async def test_embedding_load_blocks_when_docling_lease_active(self):
        with mock.patch.object(
                self.app_mod.vram_lease, "gpu_gate_decision",
                new=_async_return(self._blocked_decision("Embedding-Service"))):
            resp = await self.app_mod.load_embedding_model(
                {"model": "nomic-embed-text"})
        self.assertEqual(resp.status_code, 409)

    async def test_deberta_start_is_lifecycle_only_when_docling_lease_active(self):
        with mock.patch.object(
                self.app_mod.vram_lease, "effective_snapshot",
                new=_async_return(True)), \
             mock.patch.object(self.app_mod, "_json_service_health",
                               new=_async_return((True, {"status": "no_model"}, 503))), \
             mock.patch.object(subprocess, "run") as run_mock:
            resp = await self.app_mod.start_deberta()
        self.assertEqual(resp["status"], "started")
        self.assertTrue(resp["already"])
        self.assertFalse(run_mock.called)

    async def test_deberta_load_blocks_when_docling_lease_active(self):
        with mock.patch.object(
                self.app_mod.vram_lease, "gpu_gate_decision",
                new=_async_return(self._blocked_decision("DeBERTa-Service"))):
            resp = await self.app_mod.load_deberta_model(
                {"model": "cross-encoder-test"})
        self.assertEqual(resp.status_code, 409)

    async def test_force_bypasses_gpu_service_lease_guard(self):
        with mock.patch.object(
                self.app_mod.vram_lease, "effective_snapshot",
                new=_async_return(True)):
            guard = await self.app_mod._gpu_service_lease_guard(
                True, "Embedding-Service")
        self.assertIsNone(guard)


class DoclingDrainTests(unittest.IsolatedAsyncioTestCase):
    """Drain-Pfad fuer /api/docling/device und /api/docling/stop (#278).

    Liest den Lifecycle-Snapshot vom Docling-Proxy und wartet bis
    active_requests==0 und state NOT in {starting, stopping}. Bei Timeout
    ohne force -> 409. Mit force -> direkt stoppen.
    """

    def setUp(self):
        self.app_mod = _load_app()
        # Drain-Konstanten patchen, damit Timeout-Tests nicht 60s laufen.
        # 0.2s Deadline + 0.02s Poll = max ~10 Iterationen.
        patch_deadline = mock.patch.object(
            self.app_mod, "DRAIN_DEADLINE_S", 0.2)
        patch_poll = mock.patch.object(
            self.app_mod, "DRAIN_POLL_INTERVAL_S", 0.02)
        patch_deadline.start()
        patch_poll.start()
        self.addCleanup(patch_deadline.stop)
        self.addCleanup(patch_poll.stop)
        restart_patcher = mock.patch.object(
            self.app_mod, "_docling_restart_policy_safe",
            lambda: True)
        restart_patcher.start()
        self.addCleanup(restart_patcher.stop)

    # --- set_docling_device Drain ---

    async def test_drain_skip_when_active_zero(self):
        """Lifecycle liefert active=0, state=running -> kein Warten, Stop
        laeuft direkt.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cpu\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        start = time.monotonic()
        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        elapsed = time.monotonic() - start
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})
        # Kein Sleep — muss deutlich unter einer Poll-Dauer bleiben
        self.assertLess(elapsed, 0.1)

    async def test_drain_waits_until_active_zero(self):
        """Lifecycle liefert zuerst active=2, dann 0 -> Drain exittet und
        Stop laeuft danach durch.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_sequence([
                {"state": "running", "active_requests": 2,
                 "last_request_age_s": 1.0, "backend_failed": False},
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.5, "backend_failed": False},
            ]))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cpu\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})

    async def test_drain_timeout_409_without_force(self):
        """Lifecycle liefert dauerhaft active=1 -> 409, KEIN docker stop."""
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 1,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        docker_calls = []

        def fake_run(cmd, **kw):
            docker_calls.append(cmd)
            raise AssertionError(
                f"docker darf vor 409 nicht aufgerufen werden: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("error", body)
        self.assertEqual(body["active_requests"], 1)
        self.assertEqual(body["state"], "running")
        self.assertEqual(docker_calls, [])

    async def test_drain_timeout_with_force_stops(self):
        """Lifecycle dauerhaft active=1, force=true -> Stop laeuft durch."""
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 1,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        create_called = []

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cpu\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                create_called.append(cmd)
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device(
                {"device": "cuda", "force": True})
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})
        self.assertEqual(len(create_called), 1,
                         "force=true soll docker create erreichen")

    async def test_drain_lifecycle_unreachable_fallback(self):
        """Lifecycle-Endpoint wirft ConnectError -> kein Drain, Stop direkt."""
        async def unreachable(*a, **kw):
            raise httpx.ConnectError("refused", request=None)

        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=unreachable)
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cpu\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        # _docling_lifecycle_snapshot fangt Exceptions selbst ab und liefert
        # None — hier simulieren wir den Wrapper direkt (snapshot returned
        # None bei ConnectError).
        with mock.patch.object(
                self.app_mod, "_docling_lifecycle_snapshot",
                new=_async_return(None)):
            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                resp = await self.app_mod.set_docling_device(
                    {"device": "cuda"})
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})

    async def test_drain_waits_on_starting_state(self):
        """Lifecycle liefert erst state=starting,active=0, dann running,0.
        Drain muss mindestens einmal loopen und darf nicht bei active=0
        waehrend STARTING exiten — sonst wird der Cold-Start gesprengt.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_sequence([
                {"state": "starting", "active_requests": 0,
                 "last_request_age_s": None, "backend_failed": False},
                {"state": "starting", "active_requests": 0,
                 "last_request_age_s": None, "backend_failed": False},
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.0, "backend_failed": False},
            ]))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        stop_calls = []

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cpu\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                stop_calls.append(cmd)
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp, {"status": "recreated", "device": "cuda"})
        self.assertEqual(len(stop_calls), 1,
                         "docker stop laeuft erst nach state=running")

    async def test_drain_stopping_state_timeout_409(self):
        """Lifecycle liefert dauerhaft state=stopping, active=0 -> Drain
        wartet bis Deadline und liefert 409 (ohne force). Beweist, dass
        transiente States `active_requests==0` nicht als Drain-Done
        interpretieren.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "stopping", "active_requests": 0,
                 "last_request_age_s": None, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        docker_calls = []

        def fake_run(cmd, **kw):
            docker_calls.append(cmd)
            raise AssertionError(
                f"docker darf vor 409 nicht aufgerufen werden: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["state"], "stopping")
        self.assertEqual(docker_calls, [])

    async def test_drain_unchanged_path(self):
        """Drain laeuft VOR dem Lock; wenn nach dem Drain Device und Port
        schon stimmen, bleibt die unchanged-Antwort erhalten. Erwartet:
        `docker inspect` (Env+Ports), KEINE stop/rm/create-Calls.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        docker_calls = []

        def fake_run(cmd, **kw):
            docker_calls.append(cmd)
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout="DOCLING_DEVICE=cuda\n")
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] in (["docker", "stop"], ["docker", "rm"],
                            ["docker", "create"]):
                raise AssertionError(
                    f"unchanged-Pfad darf kein {cmd[:2]} rufen")
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.set_docling_device({"device": "cuda"})
        self.assertEqual(resp, {"status": "unchanged", "device": "cuda"})
        # Beide Inspects erwartet: Config.Env und NetworkSettings.Ports
        self.assertTrue(any(
            "inspect" in cmd and ".Config.Env" in " ".join(cmd)
            for cmd in docker_calls))
        self.assertTrue(any(
            "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd)
            for cmd in docker_calls))

    async def test_concurrent_device_calls_drain_once_per_call(self):
        """Zwei parallele set_docling_device: erster Call geht in Drain und
        recreatet, zweiter wartet vor `_docling_device_lock`, sieht danach
        den bereits korrekten State und gibt `unchanged`. Drain-Helper wird
        pro Call einmal gerufen (akzeptabel).
        """
        state = {"device_env": "DOCLING_DEVICE=cpu\n"}

        # Lifecycle: immer active=0 — Drain exittet sofort
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        ports_json = json.dumps({
            "5001/tcp": [{"HostIp": "127.0.0.1", "HostPort": "5002"}],
        })

        create_calls = []

        def fake_run(cmd, **kw):
            if "inspect" in cmd and ".Config.Env" in " ".join(cmd):
                return _FakeRun(stdout=state["device_env"])
            if "inspect" in cmd and ".NetworkSettings.Ports" in " ".join(cmd):
                return _FakeRun(stdout=ports_json)
            if cmd[:2] == ["docker", "stop"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "rm"]:
                return _FakeRun()
            if cmd[:2] == ["docker", "create"]:
                create_calls.append(cmd)
                # Nach erstem Create: Device ist jetzt cuda
                state["device_env"] = "DOCLING_DEVICE=cuda\n"
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            r1, r2 = await asyncio.gather(
                self.app_mod.set_docling_device({"device": "cuda"}),
                self.app_mod.set_docling_device({"device": "cuda"}),
            )

        statuses = sorted([r1.get("status"), r2.get("status")]
                          if isinstance(r1, dict) and isinstance(r2, dict)
                          else [])
        self.assertEqual(statuses, ["recreated", "unchanged"])
        self.assertEqual(len(create_calls), 1,
                         "genau ein docker create-Aufruf ueber beide Calls")

    # --- stop_docling Drain ---

    async def test_stop_drain_skip_when_active_zero(self):
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        with mock.patch.object(subprocess, "run", return_value=_FakeRun()):
            resp = await self.app_mod.stop_docling()
        self.assertEqual(resp, {"status": "stopped"})

    async def test_stop_drain_waits_until_zero(self):
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_sequence([
                {"state": "running", "active_requests": 3,
                 "last_request_age_s": 1.0, "backend_failed": False},
                {"state": "running", "active_requests": 0,
                 "last_request_age_s": 2.0, "backend_failed": False},
            ]))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        with mock.patch.object(subprocess, "run", return_value=_FakeRun()):
            resp = await self.app_mod.stop_docling({})
        self.assertEqual(resp, {"status": "stopped"})

    async def test_stop_drain_timeout_409_without_force(self):
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 2,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        docker_calls = []

        def fake_run(cmd, **kw):
            docker_calls.append(cmd)
            raise AssertionError(
                f"docker darf vor 409 nicht aufgerufen werden: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.stop_docling({})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["active_requests"], 2)
        self.assertEqual(docker_calls, [])

    async def test_stop_drain_timeout_with_force_stops(self):
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(
                {"state": "running", "active_requests": 1,
                 "last_request_age_s": 1.0, "backend_failed": False}))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        stop_calls = []

        def fake_run(cmd, **kw):
            if cmd[:2] == ["docker", "stop"]:
                stop_calls.append(cmd)
                return _FakeRun()
            raise AssertionError(f"Unerwartete Docker-Command: {cmd}")

        with mock.patch.object(subprocess, "run", side_effect=fake_run):
            resp = await self.app_mod.stop_docling({"force": True})
        self.assertEqual(resp, {"status": "stopped"})
        self.assertEqual(len(stop_calls), 1)

    async def test_stop_drain_lifecycle_unreachable(self):
        """Lifecycle unerreichbar (snapshot=None) -> kein Drain, Stop direkt.
        """
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(None))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        with mock.patch.object(subprocess, "run", return_value=_FakeRun()):
            resp = await self.app_mod.stop_docling({})
        self.assertEqual(resp, {"status": "stopped"})

    async def test_stop_drain_body_none_compatible(self):
        """Dashboard-JS ruft /api/docling/stop ohne Body — Direktaufruf mit
        body=None darf nicht 422 werfen und muss default {} behandeln."""
        lifecycle = mock.patch.object(
            self.app_mod, "_docling_lifecycle_snapshot",
            new=_async_return(None))
        lifecycle.start()
        self.addCleanup(lifecycle.stop)

        with mock.patch.object(subprocess, "run", return_value=_FakeRun()):
            resp = await self.app_mod.stop_docling(body=None)
        self.assertEqual(resp, {"status": "stopped"})


if __name__ == "__main__":
    unittest.main()
