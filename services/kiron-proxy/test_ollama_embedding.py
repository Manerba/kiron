"""The managed Ollama embedding path must execute the advertised contract."""

import copy
import json
from unittest import mock

import httpx
import pytest

import proxy
from kiron_common.embedding_registry import EMBEDDING_REGISTRY
from ollama_embedding import EmbeddingRequestError, prepare_request
from test_embedding_contract import _Client, _build_app, _pass_vram_lease, _allow_gpu_gate


MODEL = "hellord/e5-mistral-7b-instruct:Q4_0"
PROFILE = "ollama-e5-mistral-dense-v1"
PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


@pytest.mark.parametrize("role,prefix", [("search_document", ""), ("search_query", PREFIX)])
@pytest.mark.parametrize("raw", ["Berlin {text} 😀", "", ["Berlin", "", "</s>"]])
def test_catalog_formatting_handles_scalar_batch_empty_and_literal_placeholders(role, prefix, raw):
    profile = EMBEDDING_REGISTRY.profile(PROFILE)
    body = {"model": MODEL, "input_type": role, "input": raw}
    original = copy.deepcopy(body)
    prepared = prepare_request(body, profile, role)
    assert body == original
    expected = [prefix + text + "</s>" for text in raw] if isinstance(raw, list) else prefix + raw + "</s>"
    assert prepared["input"] == expected
    assert prepared["options"] == {"num_ctx": 4096, "num_batch": 512}
    assert prepared["truncate"] is False


@pytest.mark.parametrize("extra", [
    {"dimensions": 768}, {"dimensions": True}, {"dimensions": None},
    {"options": {"num_ctx": 8192}}, {"options": {"num_ctx": True}},
    {"options": {"num_batch": 4096}}, {"options": {"rope_frequency_scale": 2}},
    {"options": {"num_gpu": -1}}, {"options": {"num_thread": 0}},
    {"options": None}, {"options": []}, {"truncate": "false"}, {"truncate": True},
    {"input": None}, {"input": []}, {"input": [1]}, {"input": {"text": "A"}},
    {"template": "{text}"}, {"prompt": "A"},
])
def test_vector_affecting_overrides_and_bad_input_fail_before_any_backend(extra):
    body = {"model": MODEL, "input_type": "search_document", "input": "Berlin", **extra}
    with pytest.raises(EmbeddingRequestError):
        prepare_request(body, EMBEDDING_REGISTRY.profile(PROFILE), "search_document")


def test_explicit_same_dimensions_context_cpu_placement_and_reject_overflow_are_allowed():
    body = {"model": MODEL, "input": "A", "input_type": "search_query", "truncate": False,
            "dimensions": 4096, "options": {"num_ctx": 4096, "num_gpu": 0, "num_thread": 4}}
    prepared = prepare_request(body, EMBEDDING_REGISTRY.profile(PROFILE), "search_query")
    assert prepared["truncate"] is False
    assert prepared["options"] == {"num_ctx": 4096, "num_batch": 512, "num_gpu": 0, "num_thread": 4}


def test_unverified_ollama_profile_without_policy_is_untouched():
    body = {"model": "gte-qwen2-7b-instruct", "input": "A"}
    assert prepare_request(body, EMBEDDING_REGISTRY.profile("ollama-gte-qwen2-dense-v1"), None) is body


@pytest.mark.parametrize("alias", [MODEL, "e5-mistral-7b-instruct", "hellord/e5-mistral-7b-instruct"])
def test_public_route_formats_query_once_and_preserves_gpu_admission(alias):
    async def run():
        ollama = _Client(get_payloads={"/api/ps": {"models": [{"name": MODEL, "size_vram": 4096}]}})
        embedding = _Client()
        app = _build_app(ollama, embedding)
        with (
            mock.patch.object(proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease) as gate,
            mock.patch.object(proxy.vram_lease, "gpu_gate_decision", side_effect=_allow_gpu_gate) as operation,
        ):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/embed", json={"model": alias, "input_type": "search_query", "input": "Berlin?"})
        assert response.status_code == 200
        assert gate.call_count == 1
        assert operation.call_count == 1
        assert embedding.sent == []
        assert len(ollama.sent) == 1
        sent = json.loads(ollama.sent[0]["content"])
        assert sent["model"] == MODEL
        assert sent["input"] == PREFIX + "Berlin?</s>"
        assert sent["options"] == {"num_ctx": 4096, "num_batch": 512}
    import asyncio
    asyncio.run(run())


@pytest.mark.parametrize("extra", [
    {"input_type": None}, {"input_type": "document"},
    {"options": {"num_ctx": 8192}}, {"dimensions": 128}, {"input": [False]},
])
def test_public_route_rejects_invalid_contract_before_admission(extra):
    async def run():
        ollama, embedding = _Client(), _Client()
        app = _build_app(ollama, embedding)
        with mock.patch.object(proxy.vram_lease, "apply_bytes", side_effect=AssertionError("must reject before gate")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/embed", json={"model": MODEL, "input_type": "search_document", "input": "A", **extra})
        assert response.status_code == 400
        assert ollama.sent == embedding.sent == []
    import asyncio
    asyncio.run(run())
