"""Discovery- und Routingtests fuer die Mankei-Modellfamilie."""

import copy
import json
import os
import sys
import unittest
from unittest import mock

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy  # noqa: E402


class _Store:
    async def add_request(self, _record):
        pass

    async def update_request(self, _request_id, **_kwargs):
        pass


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    async def aiter_bytes(self):
        yield json.dumps(self.payload).encode()

    async def aclose(self):
        pass

    def json(self):
        return self.payload


class _Client:
    def __init__(self, send_payload=None, get_payload=None):
        self.send_response = _Response(send_payload or {"models": []})
        self.get_response = _Response(get_payload or {"models": []})
        self.sent = []
        self.gets = []

    def build_request(self, method, url, headers, content):
        return {"method": method, "url": url, "headers": headers, "content": content}

    async def send(self, request, stream=False):
        self.sent.append((request, stream))
        return self.send_response

    async def get(self, url):
        self.gets.append(url)
        return self.get_response

    async def aclose(self):
        pass


async def _pass_vram_lease(body, _path, _model, _routing_view):
    return body, proxy.vram_lease.LeaseOutcome.PASS


class MankeiRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_registries_contain_mankei_services(self):
        self.assertIn(
            "mankei-326m-embedder",
            proxy.PROXY_ROUTING_VIEW.available_models(
                "/api/embed",
                backend=proxy.BackendType.KIRON_EMBEDDINGS,
            ),
        )
        self.assertIn(
            "mankei-326m-reranker",
            proxy.PROXY_ROUTING_VIEW.available_models("/api/rerank"),
        )

    def test_tag_merge_keeps_mankei_and_exposes_colbert_canonically(self):
        ollama_rows = [
            {"name": "mankei-1b-chat:Q4_K_M"},
            *(
                {"name": group.ollama_model}
                for group in proxy.EMBEDDING_REGISTRY.ollama_groups
            ),
        ]
        service_rows = [
            {
                "name": group.service_model,
                "kiron_capabilities": copy.deepcopy(group.capabilities),
            }
            for group in proxy.EMBEDDING_REGISTRY.service_groups
        ]
        merged = proxy.merge_model_tags(
            {"models": ollama_rows},
            {"models": service_rows},
        )
        names = [item["name"] for item in merged["models"]]
        self.assertEqual(names[0], "mankei-1b-chat:Q4_K_M")
        self.assertEqual(
            set(names[1:]),
            {group.canonical_model_id for group in proxy.EMBEDDING_REGISTRY.groups},
        )
        self.assertIn("mankei-326m-embedder:latest", names)
        self.assertIn("colbert-xm:latest", names)

    async def test_show_for_embedder_is_routed_to_embedding_service(self):
        ollama = _Client()
        embedding = _Client(send_payload={
            "model": "mankei-326m-embedder",
            "model_info": {"llama.embedding_length": 960},
        })
        deberta = _Client()
        with mock.patch.object(
            proxy.httpx,
            "AsyncClient",
            side_effect=[ollama, embedding, deberta],
        ):
            app = proxy.create_proxy_app(_Store())

        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=_pass_vram_lease,
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/show",
                    json={"model": "mankei-326m-embedder:latest"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_info"]["llama.embedding_length"], 960)
        self.assertEqual(len(embedding.sent), 1)
        self.assertEqual(embedding.sent[0][0]["url"], "/api/show")
        self.assertEqual(ollama.sent, [])

    async def test_main_tags_overlay_contains_embedder(self):
        ollama = _Client(send_payload={
            "models": [
                {"name": "mankei-1b-chat:Q4_K_M"},
                *(
                    {"name": group.ollama_model}
                    for group in proxy.EMBEDDING_REGISTRY.ollama_groups
                ),
            ],
        })
        embedding = _Client(get_payload={
            "models": [
                {
                    "name": group.service_model,
                    "kiron_capabilities": copy.deepcopy(group.capabilities),
                }
                for group in proxy.EMBEDDING_REGISTRY.service_groups
            ],
        })
        deberta = _Client()
        with mock.patch.object(
            proxy.httpx,
            "AsyncClient",
            side_effect=[ollama, embedding, deberta],
        ):
            app = proxy.create_proxy_app(_Store())

        transport = httpx.ASGITransport(app=app)
        with mock.patch.object(
            proxy.vram_lease,
            "apply_bytes",
            side_effect=_pass_vram_lease,
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/tags")

        self.assertEqual(response.status_code, 200)
        names = [item["name"] for item in response.json()["models"]]
        self.assertEqual(names[0], "mankei-1b-chat:Q4_K_M")
        self.assertEqual(
            set(names[1:]),
            {group.canonical_model_id for group in proxy.EMBEDDING_REGISTRY.groups},
        )
        self.assertEqual(embedding.gets, ["/api/tags"])


if __name__ == "__main__":
    unittest.main()
