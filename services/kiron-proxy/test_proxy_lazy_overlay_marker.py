"""Regressionstests fuer Lazy-GPU-Fast-Paths gegen harte Overlay-Marker."""

import os
import contextlib
import grp
import importlib.util
from pathlib import Path
import pwd
import sys
import unittest
from unittest import mock

import httpx

PROXY_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def _load_local_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, PROXY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(PROXY_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(PROXY_DIR))
    return module


openai_api = _load_local_module("kiron_proxy_lazy_openai_api", "openai_api.py")
proxy = _load_local_module("kiron_proxy_lazy_proxy", "proxy.py")


TEST_RUNTIME_DIR = Path(os.environ.get(
    "KIRON_TEST_RUNTIME_DIR",
    "/tmp/kiron-proxy-lazy-overlay-test",
))


def _configure_runtime(vl):
    TEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(TEST_RUNTIME_DIR, 0o2770)
    for path in TEST_RUNTIME_DIR.iterdir():
        if path.is_file():
            path.unlink()
    vl.RUNTIME_MARKER_DIR = TEST_RUNTIME_DIR
    vl.STARTUP_MARKER_PATH = TEST_RUNTIME_DIR / "docling-vram-startup.json"
    vl.SHUTDOWN_MARKER_PATH = TEST_RUNTIME_DIR / "docling-vram-shutdown.json"
    vl.GPU_SERVICE_LOADING_MARKER_PATH = TEST_RUNTIME_DIR / "gpu-service-loading.json"
    vl._lease_cache["active"] = False
    vl._lease_cache["ts"] = 0.0
    vl._shared_client = None
    vl.RUNTIME_MARKER_GROUP = grp.getgrgid(os.getgid()).gr_name
    vl.RUNTIME_MARKER_FILE_OWNER_NAMES = frozenset({
        pwd.getpwuid(os.getuid()).pw_name,
    })
    vl._runtime_marker_dir_owner_uid = lambda: os.getuid()


class _Store:
    def __init__(self):
        self.records = []
        self.updates = []

    async def add_request(self, record):
        self.records.append(record)

    async def update_request(self, request_id, **kwargs):
        self.updates.append((request_id, kwargs))


class _ApiKeys:
    def validate_key(self, token):
        return {"token": token}


class _BackendClient:
    def __init__(self, *, health=None, ps=None):
        self.health = health
        self.ps = ps
        self.sent = []

    async def get(self, path):
        if path == "/health":
            return httpx.Response(200, json=self.health or {})
        if path == "/api/ps":
            return httpx.Response(200, json=self.ps or {"models": []})
        return httpx.Response(404, json={"error": "unexpected"})

    def build_request(self, method, url, headers=None, content=None):
        return {
            "method": method,
            "url": url,
            "headers": headers or {},
            "content": content or b"",
        }

    async def send(self, request, stream=False):
        self.sent.append(request)
        raise AssertionError("backend send must not be reached")

    async def aclose(self):
        pass


class _BackendResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield self.body

    async def aclose(self):
        pass


class LazyOverlayMarkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _configure_runtime(proxy.vram_lease)

    def tearDown(self):
        _configure_runtime(proxy.vram_lease)

    async def test_deberta_already_loaded_fast_path_blocks_startup_marker(self):
        token = proxy.vram_lease.write_overlay_marker("startup", ttl_s=60)
        store = _Store()
        ollama_client = _BackendClient()
        embed_client = _BackendClient()
        deberta_client = _BackendClient(health={
            "status": "ok",
            "current_model": "ms-marco-MiniLM-L-6-v2",
            "loading_model": None,
        })

        try:
            clients = iter([ollama_client, embed_client, deberta_client])
            with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=lambda **kw: next(clients)):
                app = proxy.create_proxy_app(store)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                resp = await client.post(
                    "/api/rerank",
                    json={
                        "model": "ms-marco-MiniLM-L-6-v2",
                        "query": "a",
                        "documents": ["b"],
                    },
                )

            self.assertEqual(resp.status_code, 409)
            self.assertEqual(deberta_client.sent, [])
        finally:
            proxy.vram_lease.clear_overlay_marker("startup", token)

    async def test_openai_resident_model_cannot_bypass_shared_admission(self):
        """The new API delegates admission to RuntimeService, without Ollama shortcuts."""
        import time
        from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot
        from kiron_common.local_inference import ErrorCode, LocalInferenceError, RuntimeFailure
        from kiron_common.model_catalog import BackendType
        from test_openai_runtime_api import FakeRuntime

        admission = AdmissionStore(proxy.vram_lease.RUNTIME_MARKER_DIR,
                                   security=proxy.vram_lease._admission_security())
        class RuntimeWithAdmission(FakeRuntime):
            async def chat(self, request):
                try:
                    admission.reserve(operation_id=request.context.request_id, owner="test-api",
                        generation="fixture-generation", deployment_id=request.model.deployment.id,
                        kind="request", gpu_bytes=0, host_bytes=0,
                        measure=lambda: MemorySnapshot(100,100,time.monotonic()))
                except AdmissionError:
                    raise LocalInferenceError(RuntimeFailure(ErrorCode.CONFLICT,"resource busy")) from None
                raise AssertionError("startup overlay must block before inference")

        runtime = RuntimeWithAdmission(BackendType.OLLAMA)
        token = proxy.vram_lease.write_overlay_marker("startup", ttl_s=60)
        try:
            app = openai_api.create_openai_api_app(_Store(), _ApiKeys())
            app.state.local_inference = runtime
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://testserver") as client:
                response = await client.post("/v1/chat/completions",
                    headers={"Authorization":"Bearer test"},
                    json={"model":"alias","messages":[{"role":"user","content":"hi"}]})
            self.assertEqual(response.status_code,503)
            self.assertEqual(response.json()["error"]["code"],"resource_busy")
            self.assertNotIn("events",runtime.calls)
            self.assertEqual(admission.snapshot(),())
            self.assertTrue(proxy.vram_lease.STARTUP_MARKER_PATH.exists())
        finally:
            proxy.vram_lease.clear_overlay_marker("startup",token)

    async def test_ollama_unreachable_does_not_leak_loading_marker(self):
        """Regression: Ollama-Transport-Fehler beim already_loaded-Check darf
        keinen gpu_service_loading-Marker hinterlassen.

        Vorher schluckte _ollama_model_loaded_gpu alle Exceptions und gab False
        zurueck. Der Caller schrieb dann einen 5min-TTL-Marker, der durch das
        nachfolgende send()-Fehlschlag nicht geclearet wurde und alle weiteren
        GPU-Operationen via gpu_gate_decision blockierte.
        """

        class _UnreachableOllama:
            def __init__(self):
                self.ps_calls = 0
                self.send_calls = 0

            async def get(self, path):
                if path == "/api/ps":
                    self.ps_calls += 1
                    raise httpx.ConnectError("connection refused")
                return httpx.Response(404, json={"error": "unexpected"})

            def build_request(self, method, url, headers=None, content=None):
                return {"method": method, "url": url, "headers": headers or {}, "content": content or b""}

            async def send(self, request, stream=False):
                self.send_calls += 1
                raise httpx.ConnectError("connection refused")

            async def aclose(self):
                pass

        store = _Store()
        ollama_client = _UnreachableOllama()
        embed_client = _BackendClient()
        deberta_client = _BackendClient()

        clients = iter([ollama_client, embed_client, deberta_client])
        with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=lambda **kw: next(clients)):
            app = proxy.create_proxy_app(store)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "model": "qwen3:8b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(resp.status_code, 502)
        self.assertGreaterEqual(ollama_client.ps_calls, 1)
        # Marker darf nicht geschrieben worden sein, sonst leakt er fuer 5 min.
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked despite Ollama transport error",
        )

    async def test_ollama_send_connect_error_clears_no_start_marker(self):
        class _OllamaConnectOnSend(_BackendClient):
            async def send(self, request, stream=False):
                self.sent.append(request)
                raise httpx.ConnectError("connection refused")

        store = _Store()
        ollama_client = _OllamaConnectOnSend(ps={"models": []})
        embed_client = _BackendClient()
        deberta_client = _BackendClient()

        clients = iter([ollama_client, embed_client, deberta_client])
        with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=lambda **kw: next(clients)):
            app = proxy.create_proxy_app(store)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": "hi",
                    "stream": False,
                },
            )

        self.assertEqual(resp.status_code, 502)
        self.assertEqual(len(ollama_client.sent), 1)
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked after no-start send ConnectError",
        )

    async def test_embed_service_connect_fail_closed_clears_embed_marker(self):
        class _EmbedConnectOnSend(_BackendClient):
            async def send(self, request, stream=False):
                self.sent.append(request)
                raise httpx.ConnectError("embedding service down")

        class _OllamaEmbedOk(_BackendClient):
            async def send(self, request, stream=False):
                self.sent.append(request)
                return _BackendResponse(b'{"embeddings":[[1.0,0.0]]}', 200)

        store = _Store()
        ollama_client = _OllamaEmbedOk()
        embed_client = _EmbedConnectOnSend(health={"status": "no_model"})
        deberta_client = _BackendClient()

        clients = iter([ollama_client, embed_client, deberta_client])
        with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=lambda **kw: next(clients)):
            app = proxy.create_proxy_app(store)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post(
                "/api/embed",
                json={
                    "model": "mxbai-embed-large",
                    "input": "hi",
                    "input_type": "search_document",
                },
            )

        self.assertEqual(resp.status_code, 502)
        self.assertEqual(len(embed_client.sent), 1)
        self.assertEqual(ollama_client.sent, [])
        self.assertEqual(
            resp.json()["error"]["code"],
            "embedding_backend_incompatible_or_unavailable",
        )
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "embedding gpu_service_loading marker leaked after fail-closed response",
        )


if __name__ == "__main__":
    unittest.main()
