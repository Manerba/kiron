"""Catalog-view, exact routing, rewrite, and fail-fast proxy tests."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import httpx
import pytest

PROXY_DIR = Path(__file__).resolve().parent

import proxy  # noqa: E402
from kiron_common.embedding_registry import MODEL_CATALOG  # noqa: E402
from kiron_common.model_catalog import (  # noqa: E402
    BackendType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
)
from routing_catalog import (  # noqa: E402
    PROXY_ROUTING_VIEW,
    ProxyRoutingCatalogError,
    build_proxy_routing_view,
)


EXPECTED_AVAILABLE = {
    ModelEndpoint.EMBED: (
        "bge-m3",
        "hellord/e5-mistral-7b-instruct:Q4_0",
        "mankei-326m-embedder",
        "mxbai-embed-large",
        "nomic-embed-text",
        "since2006/gte-Qwen2-7B-instruct:Q4_K_M",
        "snowflake-arctic-embed",
    ),
    ModelEndpoint.EMBED_LATE: ("nomic-embed-text",),
    ModelEndpoint.EMBED_COLBERT: ("colbert-xm",),
    ModelEndpoint.RERANK: (
        "bge-reranker-v2-m3",
        "mankei-326m-reranker",
        "mdeberta-v3-xnli",
        "ms-marco-MiniLM-L-6-v2",
        "nli-deberta-v3-base",
    ),
    ModelEndpoint.SCORE: (
        "bge-reranker-v2-m3",
        "mankei-326m-reranker",
        "mdeberta-v3-xnli",
        "ms-marco-MiniLM-L-6-v2",
        "nli-deberta-v3-base",
    ),
}


def _async_test(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _manifests() -> list[dict]:
    return [
        group.to_manifest_dict(schema_version=1)
        for group in MODEL_CATALOG.groups
    ]


def _manifest(documents: list[dict], canonical_model_id: str) -> dict:
    return next(
        item
        for item in documents
        if item["canonical_model_id"] == canonical_model_id
    )


class _Store:
    def __init__(self) -> None:
        self.records = []
        self.updates = []

    async def add_request(self, record):
        self.records.append(record)

    async def update_request(self, request_id, **kwargs):
        self.updates.append((request_id, kwargs))


class _BackendResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield b"{}"

    async def aclose(self):
        pass


class _BackendClient:
    def __init__(self, backend: BackendType) -> None:
        self.backend = backend
        self.sent: list[dict] = []
        self.gets: list[str] = []
        self.current_model: str | None = None

    def build_request(self, method, url, headers=None, content=None):
        return {
            "method": method,
            "url": url,
            "headers": headers or {},
            "content": content or b"",
        }

    async def send(self, request, stream=False):
        del stream
        self.sent.append(request)
        return _BackendResponse()

    async def get(self, url):
        self.gets.append(url)
        if self.backend is BackendType.OLLAMA and url == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": name, "size_vram": 1}
                        for name in EXPECTED_AVAILABLE[ModelEndpoint.EMBED]
                    ]
                },
            )
        if self.backend is BackendType.KIRON_EMBEDDINGS and url == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "loaded_models": list(
                        PROXY_ROUTING_VIEW.available_models(
                            ModelEndpoint.EMBED,
                            backend=BackendType.KIRON_EMBEDDINGS,
                        )
                    )
                    + list(EXPECTED_AVAILABLE[ModelEndpoint.EMBED_COLBERT]),
                    "loading_model": None,
                },
            )
        if self.backend is BackendType.KIRON_DEBERTA and url == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "current_model": self.current_model,
                    "loading_model": None,
                },
            )
        return httpx.Response(404, json={"error": "unexpected"})

    async def aclose(self):
        pass


def _payload(endpoint: ModelEndpoint, model_name: str | None) -> dict:
    if endpoint is ModelEndpoint.EMBED:
        payload = {
            "input": ["Dokument"],
            "input_type": "search_document",
        }
    elif endpoint is ModelEndpoint.EMBED_LATE:
        payload = {
            "document": "Dokument",
            "chunks": [
                {"text": "Dokument", "char_start": 0, "char_end": 8}
            ],
            "input_type": "search_document",
        }
    elif endpoint is ModelEndpoint.EMBED_COLBERT:
        payload = {
            "input": ["Dokument"],
            "input_type": "search_document",
        }
    elif endpoint is ModelEndpoint.RERANK:
        payload = {"query": "Frage", "documents": ["Dokument"]}
    else:
        payload = {"pairs": [["Frage", "Dokument"]]}
    if model_name is not None:
        payload["model"] = model_name
    return payload


async def _pass_vram(body, path, model, routing_view):
    assert routing_view is PROXY_ROUTING_VIEW
    route = routing_view.resolve(model, path)
    if path in {endpoint.value for endpoint in ModelEndpoint}:
        assert route is not None
        assert json.loads(body)["model"] == route.backend_model_name
    return body, proxy.vram_lease.LeaseOutcome.PASS


async def _allow_gpu_gate(*, force=False, service_name="GPU-Service"):
    del force
    return proxy.vram_lease.GPUGateDecision(
        allowed=True,
        reason="test",
        service_name=service_name,
    )


def _build_app():
    clients = {
        BackendType.OLLAMA: _BackendClient(BackendType.OLLAMA),
        BackendType.KIRON_EMBEDDINGS: _BackendClient(
            BackendType.KIRON_EMBEDDINGS
        ),
        BackendType.KIRON_DEBERTA: _BackendClient(BackendType.KIRON_DEBERTA),
    }
    with mock.patch.object(
        proxy.httpx,
        "AsyncClient",
        side_effect=[
            clients[BackendType.OLLAMA],
            clients[BackendType.KIRON_EMBEDDINGS],
            clients[BackendType.KIRON_DEBERTA],
        ],
    ):
        app = proxy.create_proxy_app(_Store(), PROXY_ROUTING_VIEW)
    return app, clients


def test_production_view_uses_shared_singleton_and_ordered_routes():
    assert PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG
    assert PROXY_ROUTING_VIEW.catalog_digest == (
        "sha256:c91229d7ea472b49d87f6344dbfb640fc760f43e8cace398421d5b364452e6f6"
    )
    assert PROXY_ROUTING_VIEW.endpoints == tuple(ModelEndpoint)
    for endpoint, expected in EXPECTED_AVAILABLE.items():
        assert PROXY_ROUTING_VIEW.available_models(endpoint) == expected

    assert PROXY_ROUTING_VIEW.available_models(
        ModelEndpoint.EMBED,
        backend=BackendType.KIRON_EMBEDDINGS,
    ) == (
        "bge-m3",
        "mankei-326m-embedder",
        "mxbai-embed-large",
        "nomic-embed-text",
        "snowflake-arctic-embed",
    )
    assert PROXY_ROUTING_VIEW.available_models(
        ModelEndpoint.EMBED_COLBERT,
        backend=BackendType.KIRON_EMBEDDINGS,
    ) == ("colbert-xm",)
    assert PROXY_ROUTING_VIEW.available_models(
        ModelEndpoint.EMBED,
        backend=BackendType.OLLAMA,
    ) == (
        "hellord/e5-mistral-7b-instruct:Q4_0",
        "since2006/gte-Qwen2-7B-instruct:Q4_K_M",
    )


def test_every_canonical_name_and_declared_alias_resolves_exactly():
    for route in PROXY_ROUTING_VIEW.routes:
        expected_task = {
            ModelEndpoint.EMBED: ModelTask.EMBEDDING,
            ModelEndpoint.EMBED_LATE: ModelTask.EMBEDDING,
            ModelEndpoint.EMBED_COLBERT: ModelTask.EMBEDDING,
            ModelEndpoint.RERANK: ModelTask.RERANK,
            ModelEndpoint.SCORE: ModelTask.NLI,
        }[route.endpoint]
        assert route.task is expected_task
        for name in route.input_names:
            assert PROXY_ROUTING_VIEW.resolve(name, route.endpoint) is route
            for altered in (f" {name}", f"{name} ", name.swapcase(), f"acme/{name}"):
                if altered not in route.input_names:
                    assert (
                        PROXY_ROUTING_VIEW.resolve(altered, route.endpoint)
                        is None
                    )


def test_endpoint_and_backend_foreign_names_do_not_resolve():
    assert PROXY_ROUTING_VIEW.resolve(
        "nomic-embed-text",
        ModelEndpoint.RERANK,
    ) is None
    assert PROXY_ROUTING_VIEW.resolve(
        "mankei-326m-reranker",
        ModelEndpoint.EMBED,
    ) is None
    assert PROXY_ROUTING_VIEW.resolve(
        "e5-mistral-7b-instruct",
        ModelEndpoint.EMBED,
        backend=BackendType.KIRON_EMBEDDINGS,
    ) is None
    assert PROXY_ROUTING_VIEW.resolve(
        "nomic-embed-text",
        ModelEndpoint.EMBED,
        backend=BackendType.OLLAMA,
    ) is None


def test_rerank_and_score_defaults_are_catalog_routes():
    for endpoint in (ModelEndpoint.RERANK, ModelEndpoint.SCORE):
        route = PROXY_ROUTING_VIEW.request_default(endpoint)
        assert route.backend is BackendType.KIRON_DEBERTA
        assert route.task is (
            ModelTask.RERANK
            if endpoint is ModelEndpoint.RERANK
            else ModelTask.NLI
        )
        assert route.backend_model_name == "ms-marco-MiniLM-L-6-v2"


def test_view_is_deeply_immutable_and_uses_injected_catalog():
    injected_catalog = ModelCatalog.from_manifests(_manifests())
    view = build_proxy_routing_view(injected_catalog)
    assert view.catalog is injected_catalog
    assert view.catalog is not MODEL_CATALOG
    assert view.catalog_digest == MODEL_CATALOG.catalog_digest
    with pytest.raises(FrozenInstanceError):
        view.routes = ()
    with pytest.raises(FrozenInstanceError):
        view.routes[0].backend_model_name = "mutated"
    with pytest.raises(TypeError):
        view._names_by_endpoint[ModelEndpoint.EMBED]["mutated"] = view.routes[0]


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    (
        ("loader", "/loader/type"),
        ("backend", "/backend/type"),
        ("missing_model_name", "/backend/parameters"),
        ("unknown_backend_parameter", "/backend/parameters"),
    ),
)
def test_unknown_loader_or_routing_data_fails_fast(mutation, path_fragment):
    documents = _manifests()
    if mutation == "loader":
        deployment = _manifest(
            documents, "bge-reranker-v2-m3"
        )["deployments"][0]
        deployment["loader"]["type"] = "sentence_transformers"
    else:
        deployment = _manifest(
            documents, "colbert-xm:latest"
        )["deployments"][0]
        if mutation == "backend":
            deployment["backend"]["type"] = "kiron_deberta"
        elif mutation == "missing_model_name":
            del deployment["backend"]["parameters"]["model_name"]
        else:
            deployment["backend"]["parameters"]["routing_guess"] = True
    catalog = ModelCatalog.from_manifests(documents)
    with pytest.raises(ProxyRoutingCatalogError) as exc_info:
        build_proxy_routing_view(catalog)
    assert path_fragment in exc_info.value.path


def test_missing_required_endpoint_fails_fast():
    documents = [
        item
        for item in _manifests()
        if item["canonical_model_id"] != "colbert-xm:latest"
    ]
    catalog = ModelCatalog.from_manifests(documents)
    with pytest.raises(ProxyRoutingCatalogError, match="no managed route"):
        build_proxy_routing_view(catalog)


@_async_test
async def test_all_canonical_names_and_aliases_rewrite_to_manifest_backend():
    app, clients = _build_app()
    transport = httpx.ASGITransport(app=app)
    with (
        mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=_pass_vram,
        ),
        mock.patch.object(
            proxy.vram_lease,
            "gpu_gate_decision",
            side_effect=_allow_gpu_gate,
        ),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for route in PROXY_ROUTING_VIEW.routes:
                for input_name in route.input_names:
                    before = {
                        backend: len(backend_client.sent)
                        for backend, backend_client in clients.items()
                    }
                    if route.backend is BackendType.KIRON_DEBERTA:
                        clients[route.backend].current_model = (
                            route.backend_model_name
                        )
                    response = await client.post(
                        route.endpoint.value,
                        json=_payload(route.endpoint, input_name),
                    )
                    assert response.status_code == 200, (
                        route.endpoint,
                        input_name,
                        response.text,
                    )
                    assert len(clients[route.backend].sent) == before[route.backend] + 1
                    for other_backend, backend_client in clients.items():
                        if other_backend is not route.backend:
                            assert len(backend_client.sent) == before[other_backend]
                    forwarded = json.loads(clients[route.backend].sent[-1]["content"])
                    assert forwarded["model"] == route.backend_model_name


@_async_test
async def test_both_missing_model_defaults_are_rewritten_before_forwarding():
    app, clients = _build_app()
    transport = httpx.ASGITransport(app=app)
    with (
        mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=_pass_vram,
        ),
        mock.patch.object(
            proxy.vram_lease,
            "gpu_gate_decision",
            side_effect=_allow_gpu_gate,
        ),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for endpoint in (ModelEndpoint.RERANK, ModelEndpoint.SCORE):
                route = PROXY_ROUTING_VIEW.request_default(endpoint)
                clients[BackendType.KIRON_DEBERTA].current_model = (
                    route.backend_model_name
                )
                response = await client.post(
                    endpoint.value,
                    json=_payload(endpoint, None),
                )
                assert response.status_code == 200
                forwarded = json.loads(
                    clients[BackendType.KIRON_DEBERTA].sent[-1]["content"]
                )
                assert forwarded["model"] == "ms-marco-MiniLM-L-6-v2"


@_async_test
async def test_unknown_endpoint_and_backend_foreign_models_are_400_before_vram():
    cases = (
        (ModelEndpoint.EMBED, "unknown-model"),
        (ModelEndpoint.EMBED, "mankei-326m-reranker"),
        (ModelEndpoint.EMBED_LATE, "mxbai-embed-large"),
        (ModelEndpoint.EMBED_COLBERT, "nomic-embed-text"),
        (ModelEndpoint.RERANK, "nomic-embed-text"),
        (ModelEndpoint.SCORE, "colbert-xm"),
        (
            ModelEndpoint.RERANK,
            "cross-encoder/ms-marco-MiniLM-L-6-v2:bogus",
        ),
        (ModelEndpoint.SCORE, " ms-marco-MiniLM-L-6-v2"),
    )
    app, clients = _build_app()
    transport = httpx.ASGITransport(app=app)
    with mock.patch.object(
        proxy.vram_lease,
        "apply_bytes",
        side_effect=AssertionError("VRAM must not run for invalid routing"),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            for endpoint, model_name in cases:
                response = await client.post(
                    endpoint.value,
                    json=_payload(endpoint, model_name),
                )
                assert response.status_code == 400
                assert response.json()["available_models"] == list(
                    PROXY_ROUTING_VIEW.available_models(endpoint)
                )
    assert all(not client.sent for client in clients.values())


@_async_test
async def test_explicit_empty_deberta_model_is_not_a_default():
    app, clients = _build_app()
    transport = httpx.ASGITransport(app=app)
    with mock.patch.object(
        proxy.vram_lease,
        "apply_bytes",
        side_effect=AssertionError("VRAM must not run for empty model"),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                ModelEndpoint.RERANK.value,
                json=_payload(ModelEndpoint.RERANK, ""),
            )
    assert response.status_code == 400
    assert all(not client.sent for client in clients.values())


def test_catalog_import_is_offline_and_does_not_import_model_loaders():
    code = r'''
import builtins
import socket

real_import = builtins.__import__
forbidden = {"huggingface_hub", "sentence_transformers", "torch", "transformers"}
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"model loader imported: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def no_network(*args, **kwargs):
    raise AssertionError("network access attempted")
socket.socket.connect = no_network
socket.create_connection = no_network
from kiron_common.embedding_registry import MODEL_CATALOG
from routing_catalog import PROXY_ROUTING_VIEW
assert PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG
assert len(PROXY_ROUTING_VIEW.routes) == 19
print(PROXY_ROUTING_VIEW.catalog_digest)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(PROXY_DIR), str(PROXY_DIR.parent / "kiron-common"))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY_DIR.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert MODEL_CATALOG.catalog_digest in result.stdout


def test_missing_kiron_common_fails_fast_without_source_path_fallback():
    code = r'''
import sys
class BlockKironCommon:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kiron_common" or fullname.startswith("kiron_common."):
            raise ModuleNotFoundError("blocked kiron_common for fail-fast test")
        return None
sys.meta_path.insert(0, BlockKironCommon())
import routing_catalog
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROXY_DIR)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROXY_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "blocked kiron_common for fail-fast test" in result.stderr
