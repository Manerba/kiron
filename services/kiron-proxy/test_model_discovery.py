from __future__ import annotations

import pytest

from kiron_common.embedding_registry import MODEL_STATE_VIEW
from kiron_common.local_model_registry import RegistryEntry
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.model_state import HuggingFaceRevision

from model_discovery import (
    InventoryShapeError,
    ServiceInventoryError,
    build_local_models_payload,
    service_huggingface_inventory,
)
from kiron_common.model_state import LocalModelInventory, RuntimeInventory


EXPECTED_DIGEST = (
    "sha256:be10f0dca76de099a561f53ba77adab9e4c9b9e7238ae10ac6d8b506206f4fc0"
)


def _health(*, loaded=(), loading=None):
    return {
        "running": True,
        "status": "loading" if loading else "ok" if loaded else "no_model",
        "current_model": loaded[-1] if loaded else None,
        "loaded_models": list(loaded),
        "loading_model": loading,
        "catalog_digest": EXPECTED_DIGEST,
    }


def _payload(
    *,
    hf=frozenset(),
    tags=(),
    ps=None,
    embedding=None,
    embedding_reachable=True,
    deberta=None,
    deberta_reachable=True,
    registrations=(),
    show=None,
    service_memory=None,
):
    return build_local_models_payload(
        state_view=MODEL_STATE_VIEW,
        huggingface_revisions=frozenset(hf),
        unavailable_huggingface_backends=frozenset(
            backend
            for backend, reachable in (
                (BackendType.KIRON_EMBEDDINGS, embedding_reachable),
                (BackendType.KIRON_DEBERTA, deberta_reachable),
            )
            if not reachable
        ),
        ollama_tag_rows=list(tags),
        ollama_ps={"models": []} if ps is None else ps,
        embedding_health=_health() if embedding is None else embedding,
        embedding_reachable=embedding_reachable,
        deberta_health=_health() if deberta is None else deberta,
        deberta_reachable=deberta_reachable,
        registrations=tuple(registrations),
        ollama_show_by_name={} if show is None else show,
        service_memory=service_memory,
    )


def test_managed_rows_use_catalog_metadata_and_exact_identity() -> None:
    payload = _payload(
        embedding=_health(loaded=("colbert-xm",)),
    )
    rows = {
        (item["backend"], item["name"]): item for item in payload["models"]
    }
    colbert = rows[(BackendType.KIRON_EMBEDDINGS.value, "colbert-xm")]

    assert colbert["canonical_model_id"] == "colbert-xm:latest"
    assert colbert["configured"] is True
    assert colbert["installed"] is False
    assert colbert["runtime_state"] == "loaded"
    assert colbert["family"] == "xmod"
    assert colbert["parameter_size"] == "0.9B"
    assert colbert["catalog_context_length"] == 256
    assert colbert["runtime_context_length"] is None
    assert colbert["native_context_length"] is None
    assert colbert["catalog_token_limits"] == [
        {
            "profile_id": "kiron-colbert-xm-multivector-v1",
            "unit": "tokenizer_tokens",
            "counting": "after_server_formatting_including_special_tokens",
            "truncation": "right",
            "overflow": "truncate",
            "by_role": {"search_document": 256, "search_query": 32},
        }
    ]
    assert colbert["catalog_digest"] == EXPECTED_DIGEST
    assert colbert["catalog_managed"] is True
    assert colbert["locally_registered"] is False


def test_hf_installation_is_injected_separately_from_runtime() -> None:
    definition = MODEL_STATE_VIEW.resolve(
        "mxbai-embed-large",
        BackendType.KIRON_EMBEDDINGS,
    )
    assert definition is not None
    revision = definition.huggingface_revision
    assert isinstance(revision, HuggingFaceRevision)
    payload = _payload(hf=frozenset((revision,)))
    row = next(
        item for item in payload["models"]
        if item["backend"] == BackendType.KIRON_EMBEDDINGS.value
        and item["name"] == "mxbai-embed-large"
    )

    assert row["configured"] is True
    assert row["installed"] is True
    assert row["runtime_state"] == "unloaded"


def test_embedding_profile_proofs_are_scoped_to_the_backend_and_independent_of_residency():
    rows = _payload()["models"]
    native = next(row for row in rows if row["name"] == "nomic-embed-text"
                  and row["backend"] == "kiron_embeddings")
    ollama = next(row for row in rows if row["name"] == "nomic-embed-text:latest"
                  and row["backend"] == "ollama")
    assert native["load_state"] == "unloaded" and native["installed"] is False
    assert {profile["profile_id"] for profile in native["embedding_profiles"]} == {
        "kiron-nomic-dense-v1", "kiron-nomic-late-v1",
    }
    for profile in native["embedding_profiles"]:
        assert profile["verification"] == {"status": "verified", "blocking_reasons": []}
        assert profile["index_compatibility_id"] and profile["query_compatibility_id"]
    assert len(ollama["embedding_profiles"]) == 1
    assert ollama["embedding_profiles"][0]["profile_id"] == "ollama-nomic-dense-v1"
    assert ollama["embedding_profiles"][0]["verification"]["status"] == "unverified"
    assert ollama["embedding_profiles"][0]["query_compatibility_id"] is None


def test_loading_and_unknown_runtime_are_not_guessed() -> None:
    loading = _payload(
        embedding=_health(loading="nomic-embed-text"),
    )
    unknown = _payload(
        embedding=None,
        embedding_reachable=False,
    )
    loading_row = next(
        item for item in loading["models"]
        if item["backend"] == BackendType.KIRON_EMBEDDINGS.value
        and item["name"] == "nomic-embed-text"
    )
    unknown_row = next(
        item for item in unknown["models"]
        if item["backend"] == BackendType.KIRON_EMBEDDINGS.value
        and item["name"] == "nomic-embed-text"
    )

    assert loading_row["runtime_state"] == "loading"
    assert unknown_row["runtime_state"] == "unknown"
    assert unknown_row["installed"] is None


def test_only_registered_unknown_ollama_rows_join_the_catalog_list() -> None:
    alias = "hellord/e5-mistral-7b-instruct"
    unknown = {
        "name": "acme/embed-lookalike:latest",
        "size": 123,
        "details": {"family": "bertish", "extension": {"kept": True}},
        "native_extension": {"kept": True},
    }
    registration = RegistryEntry.create(
        runtime_provider=BackendType.OLLAMA,
        artifact_origin=ArtifactType.OLLAMA, artifact_format=ArtifactFormat.OLLAMA_MANIFEST,
        reference=unknown["name"],
        display_name=unknown["name"],
        loader=LoaderType.OLLAMA,
    )
    payload = _payload(
        tags=({"name": alias}, unknown, {"name": "not-registered:latest"}),
        ps={
            "models": [
                {
                    "name": unknown["name"],
                    "size": 123,
                    "size_vram": 100,
                    "context_length": 8192,
                }
            ]
        },
        show={
            unknown["name"]: {
                "model_info": {
                    "general.architecture": "acme",
                    "acme.context_length": 32768,
                }
            }
        },
        registrations=(registration,),
    )
    managed = next(
        item for item in payload["models"]
        if item["canonical_model_id"]
        == "hellord/e5-mistral-7b-instruct:Q4_0"
    )
    generic = next(
        item for item in payload["models"]
        if item["name"] == unknown["name"]
    )

    assert managed["installed"] is True
    assert managed["name"] == "hellord/e5-mistral-7b-instruct:Q4_0"
    assert generic["configured"] is False
    assert generic["installed"] is True
    assert generic["catalog_managed"] is False
    assert generic["locally_registered"] is True
    assert generic["registry_id"] == registration.id
    assert generic["native_context_length"] == 32768
    assert generic["runtime_context_length"] == 8192
    assert generic["runtime_device"] == "gpu"  # Exact bytes, even when rounded GB is zero.
    assert generic["catalog_context_length"] is None
    assert generic["catalog_token_limits"] == []
    assert "canonical_model_id" not in generic
    assert "kiron_capabilities" not in generic
    assert not any(
        item["name"] == "not-registered:latest"
        for item in payload["models"]
    )


@pytest.mark.parametrize(
    "rows",
    [["not-an-object"], [{"name": "dup"}, {"name": "dup"}], [{}]],
)
def test_malformed_native_inventory_fails_closed(rows) -> None:
    with pytest.raises(InventoryShapeError):
        _payload(tags=rows)


def test_huggingface_registration_is_visible_without_changing_catalog_state(
    tmp_path,
) -> None:
    model_path = tmp_path / "dynamic-hf"
    model_path.mkdir()
    registration = RegistryEntry.create(
        runtime_provider=BackendType.KIRON_DEBERTA,
        artifact_origin=ArtifactType.LOCAL, artifact_format=ArtifactFormat.HF_WEIGHTS,
        reference=str(model_path),
        display_name="Dynamic <model>",
        loader=LoaderType.CROSS_ENCODER,
    )

    payload = _payload(registrations=(registration,))
    row = next(
        item for item in payload["models"]
        if item.get("registry_id") == registration.id
    )

    assert row["name"] == "Dynamic <model>"
    assert row["reference"] == str(model_path)
    assert row["backend"] == "kiron_deberta"
    assert row["installed"] is True
    assert row["catalog_managed"] is False
    assert row["locally_registered"] is True
    assert row["runtime_state"] == "unknown"


def _service_inventory_health(
    backend: BackendType,
    installed: frozenset[HuggingFaceRevision] = frozenset(),
) -> dict:
    states = MODEL_STATE_VIEW.states(
        LocalModelInventory(huggingface_revisions=installed),
        RuntimeInventory(),
        backends=(backend,),
    )
    return {
        "catalog_digest": EXPECTED_DIGEST,
        "model_states": [state.to_dict() for state in states],
    }


def test_service_health_provides_exact_hf_inventory_without_proxy_fs_access() -> None:
    embedding = MODEL_STATE_VIEW.resolve(
        "mxbai-embed-large",
        BackendType.KIRON_EMBEDDINGS,
    )
    deberta = MODEL_STATE_VIEW.resolve(
        "mankei-326m-reranker",
        BackendType.KIRON_DEBERTA,
    )
    assert embedding is not None and embedding.huggingface_revision is not None
    assert deberta is not None and deberta.huggingface_revision is not None
    expected = frozenset(
        (embedding.huggingface_revision, deberta.huggingface_revision)
    )

    inventory = service_huggingface_inventory(
        MODEL_STATE_VIEW,
        {
            BackendType.KIRON_EMBEDDINGS: _service_inventory_health(
                BackendType.KIRON_EMBEDDINGS,
                frozenset((embedding.huggingface_revision,)),
            ),
            BackendType.KIRON_DEBERTA: _service_inventory_health(
                BackendType.KIRON_DEBERTA,
                frozenset((deberta.huggingface_revision,)),
            ),
        },
        reachable_backends=(
            BackendType.KIRON_EMBEDDINGS,
            BackendType.KIRON_DEBERTA,
        ),
    )

    assert inventory == expected


def test_unreachable_service_inventory_is_unknown_not_inconsistent() -> None:
    embedding = MODEL_STATE_VIEW.resolve(
        "mxbai-embed-large",
        BackendType.KIRON_EMBEDDINGS,
    )
    assert embedding is not None and embedding.huggingface_revision is not None

    inventory = service_huggingface_inventory(
        MODEL_STATE_VIEW,
        {
            BackendType.KIRON_EMBEDDINGS: _service_inventory_health(
                BackendType.KIRON_EMBEDDINGS,
                frozenset((embedding.huggingface_revision,)),
            ),
            BackendType.KIRON_DEBERTA: {
                "catalog_digest": None,
                "model_states": None,
            },
        },
        reachable_backends=(BackendType.KIRON_EMBEDDINGS,),
    )

    assert inventory == frozenset((embedding.huggingface_revision,))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda health: health[BackendType.KIRON_EMBEDDINGS].pop("catalog_digest"),
        lambda health: health[BackendType.KIRON_EMBEDDINGS].__setitem__(
            "catalog_digest", "invalid"
        ),
        lambda health: health[BackendType.KIRON_DEBERTA].__setitem__(
            "catalog_digest", "sha256:" + "0" * 64
        ),
        lambda health: health[BackendType.KIRON_DEBERTA]["model_states"].pop(),
    ),
)
def test_service_inventory_fails_closed_for_digest_or_shape_errors(mutate) -> None:
    health = {
        backend: _service_inventory_health(backend)
        for backend in (
            BackendType.KIRON_EMBEDDINGS,
            BackendType.KIRON_DEBERTA,
        )
    }
    mutate(health)

    with pytest.raises(ServiceInventoryError):
        service_huggingface_inventory(
            MODEL_STATE_VIEW,
            health,
            reachable_backends=(
                BackendType.KIRON_EMBEDDINGS,
                BackendType.KIRON_DEBERTA,
            ),
        )
