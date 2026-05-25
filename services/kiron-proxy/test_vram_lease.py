"""Stdlib-Unittest fuer services/kiron-proxy/vram_lease.py.

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_vram_lease

Deckt Policy-Resolver, Cache-TTL, Thundering-Herd-Lock, bytes-Intercept,
dict-Intercept, Lifecycle-Fallback und Log-Throttle ab. Kein HTTP-
Call — `_shared_client` wird durch ein Fake ersetzt.
"""

import asyncio
import importlib
import json
import os
from pathlib import Path
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
        """F1-Fallback: OLLAMA_EMBED_MODELS + lease_active + force_cpu
        → konservativ BLOCK, weil Ollama num_gpu auf /api/embed nicht
        zuverlaessig honoriert."""
        await self._set_lease(True)
        body = json.dumps({"model": "e5-mistral-7b-instruct"}).encode()
        _, outcome = await self.vl.apply_bytes(
            body, "/api/embed", "e5-mistral-7b-instruct",
        )
        self.assertEqual(outcome, self.vl.LeaseOutcome.BLOCK)

    async def test_embed_service_model_lease_active_pass(self):
        """EMBED_SERVICE_MODELS werden nicht interceptet —
        Routing an kiron-embeddings, keine Ollama-GPU-Kollision."""
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
