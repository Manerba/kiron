"""Focused central discovery and fail-closed embedding routing tests."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from unittest import mock

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy  # noqa: E402

from kiron_common.embedding_contract import canonical_json  # noqa: E402
from kiron_common.embedding_registry import EMBEDDING_REGISTRY  # noqa: E402


class _Store:
    async def add_request(self, _record):
        pass

    async def update_request(self, _request_id, **_kwargs):
        pass


class _StreamResponse:
    def __init__(self, payload: object, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    async def aclose(self):
        pass


class _Client:
    def __init__(
        self,
        *,
        send_payloads: dict[str, object] | None = None,
        get_payloads: dict[str, object] | None = None,
        send_error: BaseException | None = None,
    ):
        self.send_payloads = send_payloads or {}
        self.get_payloads = get_payloads or {}
        self.send_error = send_error
        self.sent: list[dict] = []
        self.gets: list[str] = []

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
        if self.send_error is not None:
            raise self.send_error
        return _StreamResponse(self.send_payloads.get(request["url"], {}))

    async def get(self, url):
        self.gets.append(url)
        payload = self.get_payloads.get(url, {})
        return httpx.Response(200, json=payload)

    async def aclose(self):
        pass


def _service_tags() -> dict:
    return {
        "models": [
            {
                "name": group.service_model,
                "model": group.service_model,
                "service_extension": group.canonical_model_id,
                "kiron_capabilities": copy.deepcopy(group.capabilities),
            }
            for group in EMBEDDING_REGISTRY.service_groups
        ]
    }


def _ollama_tags() -> dict:
    return {
        "models": [
            {"name": "chat-model:latest", "model": "chat-model:latest"},
            *(
                {
                    "name": group.ollama_model,
                    "model": group.ollama_model,
                    "ollama_extension": group.canonical_model_id,
                }
                for group in EMBEDDING_REGISTRY.ollama_groups
            ),
        ]
    }


async def _pass_vram_lease(body, _path, _model, _routing_view):
    return body, proxy.vram_lease.LeaseOutcome.PASS


async def _allow_gpu_gate(*, force=False, service_name="GPU-Service"):
    del force
    return proxy.vram_lease.GPUGateDecision(
        allowed=True,
        reason="test",
        service_name=service_name,
    )


def _build_app(ollama: _Client, embedding: _Client):
    deberta = _Client()
    with mock.patch.object(
        proxy.httpx,
        "AsyncClient",
        side_effect=[ollama, embedding, deberta],
    ):
        return proxy.create_proxy_app(_Store())


class CentralDiscoveryContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_tags_merge_all_groups_and_show_bytes_are_identical(self):
        nomic = EMBEDDING_REGISTRY.require("nomic-embed-text")
        extended_capabilities = copy.deepcopy(nomic.capabilities)
        extended_capabilities["x_runtime_score"] = 0.5
        extended_capabilities["profiles"][0]["pipeline"]["tokenizer"][
            "x_ratio"
        ] = 0.5
        service_tags = _service_tags()
        next(
            row
            for row in service_tags["models"]
            if row["name"] == nomic.service_model
        )["kiron_capabilities"] = copy.deepcopy(extended_capabilities)
        ollama = _Client(send_payloads={"/api/tags": _ollama_tags()})
        embedding = _Client(
            send_payloads={
                "/api/show": {
                    "model": "nomic-embed-text",
                    "service_show_extension": True,
                    "kiron_capabilities": copy.deepcopy(extended_capabilities),
                }
            },
            get_payloads={"/api/tags": service_tags},
        )
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                tags_response = await client.get("/api/tags")
                show_response = await client.post(
                    "/api/show", json={"model": "nomic-embed-text:latest"}
                )

        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(show_response.status_code, 200)
        rows = tags_response.json()["models"]
        self.assertEqual(rows[0]["name"], "chat-model:latest")
        vector_rows = rows[1:]
        self.assertEqual(len(vector_rows), len(EMBEDDING_REGISTRY.groups))
        self.assertIn("colbert-xm:latest", {row["name"] for row in vector_rows})
        tags_nomic = next(
            row for row in vector_rows if row["name"] == nomic.canonical_model_id
        )
        self.assertEqual(
            {
                profile["profile_id"]
                for profile in tags_nomic["kiron_capabilities"]["profiles"]
            },
            {
                "kiron-nomic-dense-v1",
                "kiron-nomic-late-v1",
                "ollama-nomic-dense-v1",
            },
        )
        tags_capabilities_bytes = canonical_json(
            tags_nomic["kiron_capabilities"]
        ).encode("utf-8")
        show_capabilities_bytes = canonical_json(
            show_response.json()["kiron_capabilities"]
        ).encode("utf-8")
        self.assertEqual(tags_capabilities_bytes, show_capabilities_bytes)
        self.assertEqual(
            tags_capabilities_bytes,
            canonical_json(extended_capabilities).encode("utf-8"),
        )
        self.assertEqual(
            rows[0],
            {"name": "chat-model:latest", "model": "chat-model:latest"},
        )
        self.assertNotIn("kiron_capabilities", rows[0])
        self.assertEqual(
            tags_nomic["kiron_capabilities"]["x_runtime_score"], 0.5
        )
        self.assertEqual(
            tags_nomic["kiron_capabilities"]["profiles"][0]["pipeline"][
                "tokenizer"
            ]["x_ratio"],
            0.5,
        )
        self.assertTrue(show_response.json()["service_show_extension"])

    async def test_colbert_show_is_centrally_visible_and_routes_to_service(self):
        ollama = _Client()
        embedding = _Client(send_payloads={"/api/show": {"family": "xmod"}})
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/show", json={"model": "colbert-xm:latest"}
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(ollama.sent, [])
        self.assertEqual(len(embedding.sent), 1)
        profile = response.json()["kiron_capabilities"]["profiles"][0]
        self.assertEqual(profile["profile_id"], "kiron-colbert-xm-multivector-v1")
        self.assertEqual(profile["endpoint"], "/api/embed_colbert")

    async def test_ollama_only_show_is_explicitly_unverified_and_alias_is_rewritten(self):
        e5 = EMBEDDING_REGISTRY.require("e5-mistral-7b-instruct")
        ollama = _Client(send_payloads={"/api/show": {"native_extension": True}})
        embedding = _Client()
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/show", json={"model": "e5-mistral-7b-instruct"}
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["native_extension"])
        profile = response.json()["kiron_capabilities"]["profiles"][0]
        self.assertEqual(profile["verification"]["status"], "unverified")
        self.assertIsNone(profile["index_compatibility_id"])
        sent_body = json.loads(ollama.sent[0]["content"])
        self.assertEqual(sent_body["model"], e5.ollama_model)

    async def test_malformed_service_discovery_fails_complete_request_with_503(self):
        malformed = _service_tags()
        malformed["models"][0]["kiron_capabilities"]["schema_version"] = 2
        ollama = _Client(send_payloads={"/api/tags": _ollama_tags()})
        embedding = _Client(get_payloads={"/api/tags": malformed})
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.get("/api/tags")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"],
            "embedding_discovery_registry_invalid",
        )

    async def test_duplicate_or_incomplete_service_registry_fails_whole_discovery(self):
        complete = _service_tags()
        malformed_payloads = (
            {
                "models": [
                    *copy.deepcopy(complete["models"]),
                    copy.deepcopy(complete["models"][0]),
                ]
            },
            {"models": copy.deepcopy(complete["models"][:-1])},
        )

        for malformed in malformed_payloads:
            with self.subTest(rows=len(malformed["models"])):
                ollama = _Client(send_payloads={"/api/tags": _ollama_tags()})
                embedding = _Client(get_payloads={"/api/tags": malformed})
                app = _build_app(ollama, embedding)
                transport = httpx.ASGITransport(app=app)
                with mock.patch.object(
                    proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
                ):
                    async with httpx.AsyncClient(
                        transport=transport, base_url="http://testserver"
                    ) as client:
                        response = await client.get("/api/tags")

                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json()["error"]["code"],
                    "embedding_discovery_registry_invalid",
                )


class FailClosedEmbeddingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_dense_role_fails_before_vram_health_or_backend(self):
        ollama = _Client()
        embedding = _Client()
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=AssertionError("VRAM gate must not run"),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": "nomic-embed-text:latest",
                        "input": ["Dokument"],
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            {
                "code": "missing_required_input_type",
                "model": "nomic-embed-text:latest",
                "profile_id": "kiron-nomic-dense-v1",
                "field_path": "/input_type",
                "supported": ["search_document", "search_query"],
            },
        )
        self.assertEqual(embedding.gets, [])
        self.assertEqual(embedding.sent, [])
        self.assertEqual(ollama.sent, [])

    async def test_unsupported_dense_role_fails_before_backend(self):
        ollama = _Client()
        embedding = _Client()
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=AssertionError("VRAM gate must not run"),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": "mankei-326m-embedder",
                        "input": ["Dokument"],
                        "input_type": "classification",
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_input_type")
        self.assertEqual(response.json()["error"]["received"], "classification")
        self.assertEqual(embedding.gets, [])
        self.assertEqual(embedding.sent, [])
        self.assertEqual(ollama.sent, [])

    async def test_service_connect_error_never_falls_back_to_same_named_ollama(self):
        ollama = _Client()
        embedding = _Client(
            get_payloads={
                "/health": {
                    "status": "ok",
                    "loaded_models": ["nomic-embed-text"],
                    "loading_model": None,
                }
            },
            send_error=httpx.ConnectError("embedding unavailable"),
        )
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with (
            mock.patch.object(
                proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
            ),
            mock.patch.object(
                proxy.vram_lease,
                "gpu_gate_decision",
                side_effect=_allow_gpu_gate,
            ),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": "nomic-embed-text:latest",
                        "input": ["Dokument"],
                        "input_type": "search_document",
                    },
                )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(ollama.sent, [])
        self.assertEqual(
            response.json()["error"],
            {
                "code": "embedding_backend_incompatible_or_unavailable",
                "model": "nomic-embed-text:latest",
                "profile_id": "kiron-nomic-dense-v1",
                "field_path": "/backend",
            },
        )

    async def test_name_heuristics_cannot_select_an_embedding_backend(self):
        ollama = _Client()
        embedding = _Client()
        app = _build_app(ollama, embedding)
        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": "vendor/nomic-embed-text:latest",
                        "input": ["Dokument"],
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ollama.sent, [])
        self.assertEqual(embedding.sent, [])


if __name__ == "__main__":
    unittest.main()
