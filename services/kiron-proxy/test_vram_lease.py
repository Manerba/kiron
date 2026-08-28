"""Stdlib-Unittest fuer services/kiron-proxy/vram_lease.py.

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_vram_lease

Deckt Policy-Resolver, Cache-TTL, Thundering-Herd-Lock, bytes-Intercept,
dict-Intercept, Lifecycle-Fallback und Log-Throttle ab. Kein HTTP-
Call — `_shared_client` wird durch ein Fake ersetzt.
"""

import asyncio
import grp
import importlib
import json
import os
from pathlib import Path
import pwd
import tempfile
import time
import unittest
from unittest import mock


def _reload_module(env_overrides: dict | None = None):
    """Frischer Import von vram_lease mit optionalen Env-Overrides.

    Notwendig weil VRAM_LEASE_POLICY beim Import resolved wird.
    """
    if env_overrides is not None:
        for k, v in env_overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    import vram_lease as _vl  # noqa: F401
    return importlib.reload(_vl)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Httpx-Client-Stub mit Call-Counter und programmierbarer Antwort."""

    def __init__(self, response=None, exc: Exception | None = None,
                 delay: float = 0.0):
        self.response = response
        self.exc = exc
        self.delay = delay
        self.calls = 0

    async def get(self, url):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.response

    async def aclose(self):
        pass


def _reset_cache(vl, *, active: bool = False):
    """Cache auf frischen Zustand setzen."""
    vl._lease_cache["active"] = active
    vl._lease_cache["ts"] = 0.0
    vl._last_unreachable_log_ts = 0.0
    vl._last_pass_warning_log_ts = 0.0
    vl.RUNTIME_CAPABILITY_PATH = "/opt/kiron/data/kiron-test-ollama-compat-runtime.json"
    try:
        Path(vl.RUNTIME_CAPABILITY_PATH).unlink()
    except FileNotFoundError:
        pass
    runtime_dir = Path("/tmp/kiron-proxy-test-vram-lease")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for path in runtime_dir.iterdir():
        if path.is_file():
            path.unlink()
    _configure_test_runtime_contract(vl, runtime_dir)
    if hasattr(vl, "invalidate_runtime_capability_cache"):
        vl.invalidate_runtime_capability_cache()


def _write_safe_handoff(vl):
    path = Path(vl.RUNTIME_CAPABILITY_PATH)
    path.write_text(json.dumps({
        "image_digest": "ollama/ollama@sha256:test",
        "report_path": "/opt/kiron/data/ollama_compat_reports/test.json",
        "num_gpu_zero_effective": True,
    }))
    path.chmod(0o644)
    if hasattr(vl, "invalidate_runtime_capability_cache"):
        vl.invalidate_runtime_capability_cache()


def _configure_test_runtime_contract(vl, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(runtime_dir, 0o2770)
    vl.RUNTIME_MARKER_DIR = runtime_dir
    vl.STARTUP_MARKER_PATH = runtime_dir / "docling-vram-startup.json"
    vl.SHUTDOWN_MARKER_PATH = runtime_dir / "docling-vram-shutdown.json"
    vl.GPU_SERVICE_LOADING_MARKER_PATH = runtime_dir / "gpu-service-loading.json"
    vl.RUNTIME_MARKER_GROUP = grp.getgrgid(os.getgid()).gr_name
    vl.RUNTIME_MARKER_FILE_OWNER_NAMES = frozenset({pwd.getpwuid(os.getuid()).pw_name})
    vl._runtime_marker_dir_owner_uid = lambda: os.getuid()


def _write_valid_runtime_file(path: Path, payload: dict | None = None, mode: int = 0o660) -> None:
    data = payload or {
        "token": "test",
        "deadline_monotonic": time.monotonic() + 60.0,
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            json.dump(data, fh)
    finally:
        if fd is not None:
            os.close(fd)
    os.chmod(path, mode)


class PolicyResolverTests(unittest.TestCase):
    def test_default_is_force_cpu(self):
        vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": None})
        self.assertEqual(vl.VRAM_LEASE_POLICY, "force_cpu")

    def test_known_values_kept(self):
        for val in ("block", "force_cpu", "pass"):
            vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": val})
            self.assertEqual(vl.VRAM_LEASE_POLICY, val)

    def test_unknown_value_falls_back_with_warning(self):
        with self.assertLogs("vram_lease", level="WARNING") as cm:
            vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "lol"})
        self.assertEqual(vl.VRAM_LEASE_POLICY, "force_cpu")
        joined = "\n".join(cm.output)
        self.assertIn("ungueltig", joined)
        self.assertIn("force_cpu", joined)


class SnapshotCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        _reset_cache(self.vl)

    async def test_snapshot_without_client_returns_false(self):
        """Vor lifespan_client-Startup liefert snapshot() False ohne Exception."""
        self.vl._shared_client = None
        with self.assertNoLogs("vram_lease", level="WARNING"):
            got = await self.vl.snapshot()
        self.assertFalse(got)

    async def test_first_call_fetches_second_cached(self):
        fake = _FakeClient(_FakeResponse(200, {"vram_lease_active": True}))
        self.vl._shared_client = fake
        first = await self.vl.snapshot()
        second = await self.vl.snapshot()
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(fake.calls, 1, "2. Call muss aus Cache kommen")

    async def test_cache_ttl_expires(self):
        fake = _FakeClient(_FakeResponse(200, {"vram_lease_active": True}))
        self.vl._shared_client = fake
        await self.vl.snapshot()
        # Cache-Timestamp zurueckdatieren statt realer sleep
        self.vl._lease_cache["ts"] = time.monotonic() - (
            self.vl._LEASE_CACHE_TTL_S + 0.1
        )
        await self.vl.snapshot()
        self.assertEqual(fake.calls, 2)

    async def test_thundering_herd_single_fetch(self):
        """N parallele snapshot()-Calls bei kaltem Cache → genau 1 HTTP-Call."""
        fake = _FakeClient(
            _FakeResponse(200, {"vram_lease_active": True}),
            delay=0.05,
        )
        self.vl._shared_client = fake
        results = await asyncio.gather(
            *[self.vl.snapshot() for _ in range(8)]
        )
        self.assertTrue(all(results))
        self.assertEqual(fake.calls, 1,
                         "Nur 1 HTTP-Call trotz 8 paralleler Aufrufer")

    async def test_connect_error_fallback_false(self):
        import httpx
        fake = _FakeClient(exc=httpx.ConnectError("nope"))
        self.vl._shared_client = fake
        with self.assertLogs("vram_lease", level="WARNING") as cm:
            got = await self.vl.snapshot()
        self.assertFalse(got)
        self.assertIn("unreachable", "\n".join(cm.output))

    async def test_http_error_fallback_false(self):
        fake = _FakeClient(_FakeResponse(500, {}))
        self.vl._shared_client = fake
        with self.assertLogs("vram_lease", level="WARNING"):
            got = await self.vl.snapshot()
        self.assertFalse(got)

    async def test_unreachable_log_throttled(self):
        import httpx
        fake = _FakeClient(exc=httpx.ConnectError("nope"))
        self.vl._shared_client = fake

        # Erster Fehler → Warning
        with self.assertLogs("vram_lease", level="WARNING"):
            await self.vl.snapshot()

        # TTL abgelaufen faken — zweiter Fetch innerhalb 60s loggt NICHT
        self.vl._lease_cache["ts"] = time.monotonic() - 10.0
        with self.assertNoLogs("vram_lease", level="WARNING"):
            await self.vl.snapshot()

        # Throttle-Fenster manuell in der Vergangenheit → naechster Fetch
        # loggt wieder
        self.vl._last_unreachable_log_ts = (
            time.monotonic() - self.vl._LEASE_UNREACHABLE_LOG_INTERVAL_S - 1.0
        )
        self.vl._lease_cache["ts"] = time.monotonic() - 10.0
        with self.assertLogs("vram_lease", level="WARNING"):
            await self.vl.snapshot()


class ApplyBytesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        _reset_cache(self.vl)
        # Dummy-Client: snapshot() liefert sonst immer False vor Startup
        self.vl._shared_client = _FakeClient(_FakeResponse(200, {}))

    async def _set_lease(self, active: bool):
        self.vl._lease_cache["active"] = active
        self.vl._lease_cache["ts"] = time.monotonic()

    async def test_chat_lease_inactive_pass(self):
        await self._set_lease(False)
        body = json.dumps({"model": "qwen3:8b", "messages": []}).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_chat_force_cpu_injects_num_gpu(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        body = json.dumps({"model": "qwen3:8b", "messages": []}).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")
        self.assertEqual(outcome, self.vl.LeaseOutcome.FORCE_CPU)
        data = json.loads(out)
        self.assertEqual(data["options"]["num_gpu"], 0)

    async def test_chat_force_cpu_blocks_explicit_nonzero_num_gpu(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        body = json.dumps({
            "model": "qwen3:8b", "messages": [],
            "options": {"num_gpu": 99},
        }).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
        self.assertEqual(out, body)

    async def test_chat_block_policy(self):
        self.vl.VRAM_LEASE_POLICY = "block"
        await self._set_lease(True)
        body = json.dumps({"model": "qwen3:8b", "messages": []}).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
        self.assertEqual(out, body, "Body bleibt unveraendert bei BLOCK")

    async def test_chat_pass_policy_logs_warning_once(self):
        self.vl.VRAM_LEASE_POLICY = "pass"
        await self._set_lease(True)
        body = json.dumps({"model": "qwen3:8b", "messages": []}).encode()
        with self.assertLogs("vram_lease", level="WARNING") as cm:
            _, outcome = await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        joined = "\n".join(cm.output)
        self.assertIn("Policy=pass", joined)

        # Zweiter Call innerhalb 60s → kein zusaetzliches Log
        with self.assertNoLogs("vram_lease", level="WARNING"):
            await self.vl.apply_bytes(body, "/api/chat", "qwen3:8b")

    async def test_chat_non_intercepted_path_pass(self):
        await self._set_lease(True)
        body = json.dumps({"model": "x"}).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/tags", "x")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_chat_invalid_json_pass(self):
        await self._set_lease(True)
        body = b"{not json"
        out, outcome = await self.vl.apply_bytes(body, "/api/chat", "")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_embed_ollama_model_lease_active_blocks_by_default(self):
        """Catalog-routed Ollama embedding + active lease stays blocked.

        Ollama cannot reliably honor ``num_gpu=0`` on ``/api/embed``.
        """
        await self._set_lease(True)
        body = json.dumps({"model": "e5-mistral-7b-instruct"}).encode()
        _, outcome = await self.vl.apply_bytes(
            body, "/api/embed", "e5-mistral-7b-instruct",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)

    async def test_every_exact_ollama_embedding_name_blocks_when_active(self):
        await self._set_lease(True)
        routes = self.vl.PROXY_ROUTING_VIEW.routes_for_endpoint(
            self.vl.ModelEndpoint.EMBED,
            backend=self.vl.BackendType.OLLAMA,
        )
        self.assertEqual(len(routes), 2)
        for route in routes:
            for name in route.input_names:
                with self.subTest(name=name):
                    body = json.dumps({"model": name}).encode()
                    out, outcome = await self.vl.apply_bytes(
                        body,
                        "/api/embed",
                        name,
                    )
                    self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
                    self.assertEqual(out, body)

    async def test_ollama_embedding_pass_policy_stays_pass(self):
        self.vl.VRAM_LEASE_POLICY = "pass"
        await self._set_lease(True)
        route = self.vl.PROXY_ROUTING_VIEW.routes_for_endpoint(
            self.vl.ModelEndpoint.EMBED,
            backend=self.vl.BackendType.OLLAMA,
        )[0]
        body = json.dumps({"model": route.canonical_model_id}).encode()
        with self.assertLogs("vram_lease", level="WARNING"):
            out, outcome = await self.vl.apply_bytes(
                body,
                "/api/embed",
                route.canonical_model_id,
            )
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_ollama_embedding_explicit_num_gpu_zero_remains_blocked(self):
        """Ollama /api/embed cannot safely honor num_gpu=0 (existing F1)."""
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        route = self.vl.PROXY_ROUTING_VIEW.routes_for_endpoint(
            self.vl.ModelEndpoint.EMBED,
            backend=self.vl.BackendType.OLLAMA,
        )[0]
        body = json.dumps(
            {
                "model": route.backend_model_name,
                "options": {"num_gpu": 0},
            }
        ).encode()
        out, outcome = await self.vl.apply_bytes(
            body,
            "/api/embed",
            route.backend_model_name,
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
        self.assertEqual(out, body)

    async def test_embed_service_model_lease_active_pass(self):
        """A kiron-embeddings Catalog route is not intercepted."""
        await self._set_lease(True)
        body = json.dumps({"model": "nomic-embed-text"}).encode()
        out, outcome = await self.vl.apply_bytes(
            body, "/api/embed", "nomic-embed-text",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_embed_unknown_model_lease_active_pass(self):
        """Unbekanntes Embed-Modell: Intercept passt durch; bestehender
        400-Handler in proxy.py:448-468 greift."""
        await self._set_lease(True)
        body = json.dumps({"model": "llama3"}).encode()
        out, outcome = await self.vl.apply_bytes(body, "/api/embed", "llama3")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, body)

    async def test_unknown_or_endpoint_foreign_embed_never_hits_hard_gate(self):
        await self._set_lease(True)
        for name in (
            "llama3",
            "acme/e5-mistral-7b-instruct:bogus",
            "mankei-326m-reranker",
        ):
            with self.subTest(name=name), mock.patch.object(
                self.vl,
                "_hard_overlay_marker_kind",
                side_effect=AssertionError("unknown model reached hard gate"),
            ), mock.patch.object(
                self.vl,
                "snapshot",
                side_effect=AssertionError("unknown model reached lease lookup"),
            ):
                body = json.dumps({"model": name}).encode()
                out, outcome = await self.vl.apply_bytes(
                    body,
                    "/api/embed",
                    name,
                )
            self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
            self.assertEqual(out, body)

    async def test_chat_explicit_num_gpu_zero_keeps_force_cpu_outcome(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        body = json.dumps(
            {
                "model": "qwen3:8b",
                "messages": [],
                "options": {"num_gpu": 0},
            }
        ).encode()
        out, outcome = await self.vl.apply_bytes(
            body,
            "/api/chat",
            "qwen3:8b",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.FORCE_CPU)
        self.assertEqual(json.loads(out)["options"]["num_gpu"], 0)

    async def test_embed_empty_body_pass(self):
        await self._set_lease(True)
        out, outcome = await self.vl.apply_bytes(b"", "/api/embed", "")
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(out, b"")


class ApplyOptionsDictTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        _reset_cache(self.vl)
        self.vl._shared_client = _FakeClient(_FakeResponse(200, {}))

    async def _set_lease(self, active: bool):
        self.vl._lease_cache["active"] = active
        self.vl._lease_cache["ts"] = time.monotonic()

    async def test_force_cpu_mutates_dict(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        opts = {"temperature": 0.7}
        outcome = await self.vl.apply_options_dict(opts)
        self.assertEqual(outcome, self.vl.LeaseOutcome.FORCE_CPU)
        self.assertEqual(opts["num_gpu"], 0)
        self.assertEqual(opts["temperature"], 0.7)

    async def test_explicit_nonzero_num_gpu_blocks_when_gate_active(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        opts = {"num_gpu": 99}
        outcome = await self.vl.apply_options_dict(opts)
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
        self.assertEqual(opts["num_gpu"], 99, "Client-Wert bleibt")

    async def test_lease_inactive_pass(self):
        await self._set_lease(False)
        opts = {"temperature": 0.7}
        outcome = await self.vl.apply_options_dict(opts)
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertNotIn("num_gpu", opts)

    async def test_block_policy(self):
        self.vl.VRAM_LEASE_POLICY = "block"
        await self._set_lease(True)
        opts = {}
        outcome = await self.vl.apply_options_dict(opts)
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)
        self.assertEqual(opts, {}, "Dict unveraendert bei BLOCK")

    async def test_pass_policy_logs_warning(self):
        self.vl.VRAM_LEASE_POLICY = "pass"
        await self._set_lease(True)
        opts = {}
        with self.assertLogs("vram_lease", level="WARNING"):
            outcome = await self.vl.apply_options_dict(opts)
        self.assertEqual(outcome, self.vl.LeaseOutcome.PASS)
        self.assertEqual(opts, {})


class LifespanClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vl = _reload_module()

    async def test_lifespan_sets_and_clears_shared_client(self):
        self.assertIsNone(self.vl._shared_client)
        async with self.vl.lifespan_client() as client:
            self.assertIsNotNone(self.vl._shared_client)
            self.assertIs(self.vl._shared_client, client)
        self.assertIsNone(self.vl._shared_client,
                          "Nach Exit muss _shared_client None sein")


class RuntimeMarkerContractTests(unittest.TestCase):
    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.tmp.name)
        _configure_test_runtime_contract(self.vl, self.runtime_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_runtime_dir_matches_contract(self):
        vl = _reload_module({"KIRON_RUNTIME_DIR": None})
        self.assertEqual(vl.RUNTIME_MARKER_DIR, Path("/run/kiron/vram"))

    def test_accepts_valid_kiron_runtime_marker(self):
        _write_valid_runtime_file(self.vl.STARTUP_MARKER_PATH)

        ok, reason, _ = self.vl._path_is_safe_runtime_file(
            self.vl.STARTUP_MARKER_PATH
        )

        self.assertTrue(ok, reason)

    def test_rejects_world_bits_and_special_bits(self):
        for mode in (0o664, 0o2660):
            with self.subTest(mode=oct(mode)):
                _write_valid_runtime_file(
                    self.vl.STARTUP_MARKER_PATH,
                    mode=mode,
                )

                ok, reason, _ = self.vl._path_is_safe_runtime_file(
                    self.vl.STARTUP_MARKER_PATH
                )

                self.assertFalse(ok)
                self.assertIn(reason, {"file_world_bits", "file_mode_not_0660"})
                self.vl.STARTUP_MARKER_PATH.unlink()

    def test_open_marker_lock_accepts_existing_valid_lock_without_chmod(self):
        lock_path = self.vl._marker_lock_path(self.vl.STARTUP_MARKER_PATH)
        _write_valid_runtime_file(lock_path)

        with mock.patch.object(
            self.vl.os,
            "fchmod",
            side_effect=AssertionError("existing peer lock must not chmod"),
        ):
            lock_fd = self.vl._open_marker_lock(self.vl.STARTUP_MARKER_PATH)

        os.close(lock_fd)

    def test_marker_payload_uses_fd_bound_open(self):
        _write_valid_runtime_file(self.vl.STARTUP_MARKER_PATH)

        with mock.patch.object(
            Path,
            "open",
            side_effect=AssertionError("marker reads must use os.open/fstat"),
        ):
            data, active = self.vl._marker_payload(self.vl.STARTUP_MARKER_PATH)

        self.assertIsInstance(data, dict)
        self.assertTrue(active)

    def test_clear_does_not_unlink_replaced_marker(self):
        old_payload = {
            "token": "old",
            "deadline_monotonic": time.monotonic() + 60.0,
        }
        replacement_payload = {
            "token": "replacement",
            "deadline_monotonic": time.monotonic() + 60.0,
        }
        _write_valid_runtime_file(self.vl.STARTUP_MARKER_PATH, payload=old_payload)
        original_payload_with_stat = self.vl._marker_payload_with_stat

        def replacing_payload(path):
            result = original_payload_with_stat(path)
            path.unlink()
            _write_valid_runtime_file(path, payload=replacement_payload)
            return result

        with mock.patch.object(
            self.vl,
            "_marker_payload_with_stat",
            replacing_payload,
        ):
            self.vl.clear_overlay_marker("startup", "old")

        data = json.loads(self.vl.STARTUP_MARKER_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["token"], "replacement")


class AppLoadModelIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration mit app.py /api/models/load — Block-Semantik bei aktivem
    Lease (unabhaengig von VRAM_LEASE_POLICY)."""

    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        _reset_cache(self.vl)
        self.vl._shared_client = _FakeClient(_FakeResponse(200, {}))
        import importlib
        import sys
        sys.path.insert(0, "/opt/kiron/services/kiron-proxy")
        import app as _app
        self.app_mod = importlib.reload(_app)

    async def _set_lease(self, active: bool):
        self.vl._lease_cache["active"] = active
        self.vl._lease_cache["ts"] = time.monotonic()

    async def _call_load(self, payload: dict):
        """Ruft load_model direkt — umgeht HTTP-Stack."""
        return await self.app_mod.load_model(payload)

    async def test_gpu_true_lease_active_returns_409(self):
        await self._set_lease(True)
        resp = await self._call_load({"name": "qwen3:8b", "gpu": True})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["vram_lease"], "active")

    async def test_gpu_true_force_true_bypasses_lease(self):
        await self._set_lease(True)

        posted: dict = {}

        class _FakeHttpResp:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {}

        class _FakeHttp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None):
                posted["url"] = url
                posted["json"] = json
                return _FakeHttpResp()

            async def get(self, url):
                return _FakePsResp()

        class _FakePsResp:
            status_code = 200

            def json(self):
                return {"models": [{"name": "qwen3:8b", "size_vram": 1, "size": 1}]}

        with mock.patch("httpx.AsyncClient", lambda **kw: _FakeHttp()):
            resp = await self._call_load(
                {"name": "qwen3:8b", "gpu": True, "force": True}
            )
        # Erfolgreicher Load → dict mit status=loaded
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp["status"], "loaded")
        self.assertEqual(posted["url"], "/api/generate")

    async def test_gpu_false_lease_active_blocks_without_safe_handoff(self):
        """gpu=false braucht einen sicheren Runtime-Handoff."""
        await self._set_lease(True)
        resp = await self._call_load({"name": "qwen3:8b", "gpu": False})
        self.assertEqual(resp.status_code, 409)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("CPU-Offload", body["error"])

    async def test_gpu_true_lease_inactive_loads_normally(self):
        await self._set_lease(False)

        class _FakeHttpResp:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {}

        class _FakeHttp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None):
                return _FakeHttpResp()

            async def get(self, url):
                return _FakePsResp()

        class _FakePsResp:
            status_code = 200

            def json(self):
                return {"models": [{"name": "qwen3:8b", "size_vram": 1, "size": 1}]}

        with mock.patch("httpx.AsyncClient", lambda **kw: _FakeHttp()):
            resp = await self._call_load(
                {"name": "qwen3:8b", "gpu": True}
            )
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp["status"], "loaded")


class ProxyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration mit kiron-proxy/proxy.py — apply_bytes-Aufruf im
    proxy_handler und Header-Propagation."""

    def setUp(self):
        self.vl = _reload_module({"KIRON_VRAM_LEASE_POLICY": "force_cpu"})
        _reset_cache(self.vl)
        self.vl._shared_client = _FakeClient(_FakeResponse(200, {}))

    async def _set_lease(self, active: bool):
        self.vl._lease_cache["active"] = active
        self.vl._lease_cache["ts"] = time.monotonic()

    async def test_apply_bytes_integration_force_cpu(self):
        """End-to-end-Smoke: apply_bytes mit realem JSON-Body, Outcome
        und Body-Mutation konsistent."""
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        body = json.dumps({
            "model": "qwen3:8b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode()
        new_body, outcome = await self.vl.apply_bytes(
            body, "/api/chat", "qwen3:8b",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.FORCE_CPU)
        data = json.loads(new_body)
        self.assertEqual(data["options"]["num_gpu"], 0)
        self.assertEqual(data["model"], "qwen3:8b")
        self.assertEqual(data["stream"], True)

    async def test_apply_bytes_generate_path(self):
        await self._set_lease(True)
        _write_safe_handoff(self.vl)
        body = json.dumps({"model": "qwen3:8b", "prompt": "hi"}).encode()
        new_body, outcome = await self.vl.apply_bytes(
            body, "/api/generate", "qwen3:8b",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.FORCE_CPU)
        data = json.loads(new_body)
        self.assertEqual(data["options"]["num_gpu"], 0)


if __name__ == "__main__":
    unittest.main()
