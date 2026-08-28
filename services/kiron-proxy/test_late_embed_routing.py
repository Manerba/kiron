"""Routing-Tests fuer /api/embed_late im Proxy.

Pruft D2-Filter, Lazy-Op-Marker, ConnectError-Verhalten (kein Ollama-Fallback)
und D2-Pruefung vor Hard-Overlay-Gate.
"""

import contextlib
import grp
import importlib.util
import json
import os
import pwd
import sys
import unittest
from pathlib import Path
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


proxy = _load_local_module("kiron_proxy_late_embed_proxy", "proxy.py")


TEST_RUNTIME_DIR = Path(os.environ.get(
    "KIRON_TEST_RUNTIME_DIR_LATE",
    "/tmp/kiron-proxy-late-embed-test",
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


class _BackendClient:
    def __init__(self, *, health=None, ps=None, response_body=b'{}', status_code=200, send_exc=None):
        self.health = health
        self.ps = ps
        self.response_body = response_body
        self.status_code = status_code
        self.send_exc = send_exc
        self.sent = []
        self.gets = []

    async def get(self, path):
        self.gets.append(path)
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
        if self.send_exc is not None:
            raise self.send_exc
        return _BackendResponse(self.response_body, self.status_code)

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


def _build_app(*, ollama=None, embed=None, deberta=None):
    """Wire proxy.create_proxy_app with three fake clients (Ollama, Embed, DeBERTa)."""
    ollama = ollama or _BackendClient()
    embed = embed or _BackendClient()
    deberta = deberta or _BackendClient()
    clients = iter([ollama, embed, deberta])
    with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=lambda **kw: next(clients)):
        app = proxy.create_proxy_app(_Store())
    return app, ollama, embed, deberta


class LateEmbedRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _configure_runtime(proxy.vram_lease)

    def tearDown(self):
        _configure_runtime(proxy.vram_lease)

    async def _post_late_embed(self, app, *, model, payload=None):
        body = payload or {
            "model": model,
            "document": "hello world",
            "chunks": [{"text": "hello", "char_start": 0, "char_end": 5}],
            "input_type": "search_document",
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/embed_late", json=body)

    async def _post_colbert(self, app, *, model, payload=None):
        body = payload or {
            "model": model,
            "input": ["hello world"],
            "input_type": "search_document",
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/embed_colbert", json=body)

    async def test_proxy_late_embed_routes_to_embed_service(self):
        # nomic-embed-text -> embed-Client (request-recorder).
        embed_response = json.dumps({
            "model": "nomic-embed-text",
            "embeddings": [[0.5, 0.5]],
        }).encode()
        embed = _BackendClient(
            health={
                "status": "ok",
                "current_model": "nomic-embed-text",
                "loading_model": None,
            },
            response_body=embed_response,
            status_code=200,
        )
        app, ollama, _embed, _deberta = _build_app(embed=embed)

        resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(embed.sent), 1)
        self.assertEqual(ollama.sent, [])
        # Request wurde an /api/embed_late am Embed-Backend geschickt
        self.assertEqual(embed.sent[0]["url"], "/api/embed_late")

    async def test_proxy_colbert_routes_to_embed_service(self):
        embed_response = json.dumps({
            "model": "colbert-xm",
            "embeddings": [[[0.1, 0.2]]],
        }).encode()
        embed = _BackendClient(
            health={
                "status": "ok",
                "current_model": "colbert-xm",
                "loading_model": None,
            },
            response_body=embed_response,
            status_code=200,
        )
        app, ollama, _embed, _deberta = _build_app(embed=embed)

        resp = await self._post_colbert(app, model="colbert-xm")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(embed.sent), 1)
        self.assertEqual(embed.sent[0]["url"], "/api/embed_colbert")
        self.assertEqual(ollama.sent, [])

    async def test_missing_input_type_fails_before_gate_health_or_backend(self):
        for path, model, payload in (
            (
                "/api/embed_late",
                "nomic-embed-text",
                {
                    "model": "nomic-embed-text",
                    "document": "hello world",
                    "chunks": [
                        {"text": "hello", "char_start": 0, "char_end": 5}
                    ],
                },
            ),
            (
                "/api/embed_colbert",
                "colbert-xm",
                {"model": "colbert-xm", "input": ["hello world"]},
            ),
        ):
            with self.subTest(path=path):
                app, ollama, embed, _deberta = _build_app()
                with mock.patch.object(
                    proxy.vram_lease,
                    "apply_bytes",
                    side_effect=AssertionError("VRAM gate must not run"),
                ):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client:
                        response = await client.post(path, json=payload)

                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.json()["error"],
                    {
                        "code": "missing_required_input_type",
                        "model": f"{model}:latest",
                        "profile_id": (
                            "kiron-nomic-late-v1"
                            if path == "/api/embed_late"
                            else "kiron-colbert-xm-multivector-v1"
                        ),
                        "field_path": "/input_type",
                        "supported": [
                            "search_document",
                            "search_query",
                        ],
                    },
                )
                self.assertEqual(embed.sent, [])
                self.assertEqual(embed.gets, [])
                self.assertEqual(ollama.sent, [])

    async def test_proxy_colbert_rejects_unknown_model(self):
        app, ollama, embed, _deberta = _build_app()
        resp = await self._post_colbert(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["available_models"], ["colbert-xm"])
        self.assertEqual(embed.sent, [])
        self.assertEqual(ollama.sent, [])

    async def test_proxy_colbert_connect_error_has_no_ollama_fallback(self):
        embed = _BackendClient(
            health={"status": "no_model", "current_model": None, "loading_model": None},
            send_exc=httpx.ConnectError("embedding down"),
        )
        app, ollama, _embed, _deberta = _build_app(embed=embed)

        resp = await self._post_colbert(app, model="colbert-xm")

        self.assertEqual(resp.status_code, 503)
        self.assertEqual(len(embed.sent), 1)
        self.assertEqual(ollama.sent, [])

    async def test_proxy_late_embed_rejects_non_nomic(self):
        # D2: mxbai/snowflake/bge-m3 -> 400 mit available_models=["nomic-embed-text"].
        for non_nomic in ("mxbai-embed-large", "snowflake-arctic-embed", "bge-m3"):
            with self.subTest(model=non_nomic):
                app, ollama, embed, _deberta = _build_app()
                resp = await self._post_late_embed(app, model=non_nomic)

                self.assertEqual(resp.status_code, 400)
                body = resp.json()
                self.assertEqual(body["available_models"], ["nomic-embed-text"])
                self.assertEqual(embed.sent, [])
                self.assertEqual(ollama.sent, [])

    async def test_proxy_late_embed_rejects_ollama_models(self):
        # e5-mistral / gte-qwen sind Ollama-Modelle ohne /api/embed_late-Endpoint.
        for ollama_model in ("e5-mistral-7b-instruct", "gte-qwen2-7b-instruct"):
            with self.subTest(model=ollama_model):
                app, ollama, embed, _deberta = _build_app()
                resp = await self._post_late_embed(app, model=ollama_model)

                self.assertEqual(resp.status_code, 400)
                body = resp.json()
                self.assertEqual(body["available_models"], ["nomic-embed-text"])
                self.assertEqual(ollama.sent, [])
                self.assertEqual(embed.sent, [])

    async def test_proxy_late_embed_unknown_model(self):
        app, ollama, embed, _deberta = _build_app()
        resp = await self._post_late_embed(app, model="frob-embed")

        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertEqual(body["available_models"], ["nomic-embed-text"])
        self.assertEqual(ollama.sent, [])
        self.assertEqual(embed.sent, [])

    async def test_proxy_late_embed_lazy_op_marker(self):
        # Marker wird waehrend Operation gesetzt und nach 200 freigegeben.
        embed_response = json.dumps({
            "model": "nomic-embed-text",
            "embeddings": [[0.5, 0.5]],
        }).encode()
        embed = _BackendClient(
            # current_model=None -> lazy-op nimmt nicht den already_loaded-Pfad
            # und schreibt einen Marker.
            health={"status": "no_model", "current_model": None, "loading_model": None},
            response_body=embed_response,
            status_code=200,
        )
        app, _ollama, _embed, _deberta = _build_app(embed=embed)

        resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(embed.sent), 1)
        # Nach erfolgreicher Antwort ist der Marker freigegeben.
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked after 200 response",
        )

    async def test_proxy_late_embed_loaded_models_skip_lazy_marker(self):
        # current_model zeigt auf ColBERT, nomic ist aber ebenfalls resident.
        # Der Already-Loaded-Pfad muss loaded_models auswerten, sonst wird bei
        # Hybrid-Retrieval trotz warmem Dense-Modell ein Loading-Marker gesetzt.
        embed_response = json.dumps({
            "model": "nomic-embed-text",
            "embeddings": [[0.5, 0.5]],
        }).encode()
        embed = _BackendClient(
            health={
                "status": "ok",
                "current_model": "colbert-xm",
                "loaded_models": ["nomic-embed-text", "colbert-xm"],
                "loading_model": None,
            },
            response_body=embed_response,
            status_code=200,
        )
        app, _ollama, _embed, _deberta = _build_app(embed=embed)

        with mock.patch.object(
            proxy.vram_lease,
            "write_overlay_marker",
            wraps=proxy.vram_lease.write_overlay_marker,
        ) as write_marker:
            resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(embed.sent), 1)
        write_marker.assert_not_called()

    async def test_proxy_late_embed_current_model_without_loaded_models_is_not_loaded(self):
        embed_response = json.dumps({
            "model": "nomic-embed-text",
            "embeddings": [[0.5, 0.5]],
        }).encode()
        embed = _BackendClient(
            health={
                "status": "ok",
                "current_model": "nomic-embed-text",
                "loading_model": None,
            },
            response_body=embed_response,
            status_code=200,
        )
        app, _ollama, _embed, _deberta = _build_app(embed=embed)

        with mock.patch.object(
            proxy.vram_lease,
            "write_overlay_marker",
            wraps=proxy.vram_lease.write_overlay_marker,
        ) as write_marker:
            resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(embed.sent), 1)
        write_marker.assert_called_once()
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked after current_model-only health",
        )

    async def test_proxy_late_embed_lazy_op_marker_on_error(self):
        # 5xx vom Embed-Service -> Marker bleibt gesetzt (5xx = unklarer Loading-Zustand).
        # 4xx -> Marker wird freigegeben.
        embed_response = json.dumps({"error": "Late-Embed Length-Mismatch"}).encode()
        embed = _BackendClient(
            health={"status": "no_model", "current_model": None, "loading_model": None},
            response_body=embed_response,
            status_code=400,  # 4xx -> Marker freigeben
        )
        app, _ollama, _embed, _deberta = _build_app(embed=embed)

        resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 400)
        # 4xx zaehlt als "Loading-Status klar nicht verantwortlich" -> Marker geclearet.
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked after 4xx response",
        )

    async def test_proxy_late_embed_no_ollama_fallback_on_connect_error(self):
        # Embed-Service ConnectError fuer /api/embed_late -> 502 ohne Ollama-Fallback.
        embed = _BackendClient(
            health={"status": "no_model", "current_model": None, "loading_model": None},
            send_exc=httpx.ConnectError("embedding service down"),
        )
        ollama = _BackendClient(
            response_body=b'{"embeddings":[[0.0]]}',
            status_code=200,
        )
        app, _ollama, _embed, _deberta = _build_app(embed=embed, ollama=ollama)

        resp = await self._post_late_embed(app, model="nomic-embed-text")

        self.assertEqual(resp.status_code, 502)
        self.assertEqual(len(embed.sent), 1)
        # KEIN Ollama-Fallback (Ollama hat keinen /api/embed_late-Endpoint)
        self.assertEqual(ollama.sent, [])
        body = resp.json()
        self.assertEqual(
            body["error"]["code"],
            "embedding_backend_incompatible_or_unavailable",
        )
        self.assertEqual(body["error"]["profile_id"], "kiron-nomic-late-v1")
        self.assertEqual(body["error"]["field_path"], "/backend")
        # Marker geclearet, kein Leak
        self.assertFalse(
            proxy.vram_lease.overlay_marker_active("gpu_service_loading"),
            "gpu_service_loading marker leaked after embed ConnectError",
        )

    async def test_proxy_late_embed_d2_check_before_gate(self):
        # Bei aktivem Hard-Overlay-Gate liefert /api/embed_late mit non-nomic-Modell
        # 400 (D2-Filter), nicht 409 (Gate-Check kommt erst nach Modell-Check).
        token = proxy.vram_lease.write_overlay_marker("startup", ttl_s=60)
        try:
            app, _ollama, embed, _deberta = _build_app()
            resp = await self._post_late_embed(app, model="mxbai-embed-large")

            self.assertEqual(resp.status_code, 400)
            body = resp.json()
            self.assertEqual(body["available_models"], ["nomic-embed-text"])
            # Backend wurde nie kontaktiert
            self.assertEqual(embed.sent, [])
        finally:
            proxy.vram_lease.clear_overlay_marker("startup", token)


if __name__ == "__main__":
    unittest.main()
