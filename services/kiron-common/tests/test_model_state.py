from __future__ import annotations

from pathlib import Path

import pytest

from kiron_common.embedding_registry import MODEL_CATALOG, MODEL_STATE_VIEW
from kiron_common.model_catalog import BackendType
from kiron_common.model_state import (
    BackendRuntimeSnapshot,
    HuggingFaceRevision,
    LocalModelInventory,
    RuntimeInventory,
    RuntimeState,
    scan_huggingface_inventory,
)


def _state(
    *,
    installed: bool,
    known: bool,
    loaded: frozenset[str] = frozenset(),
    loading: frozenset[str] = frozenset(),
):
    definition = MODEL_STATE_VIEW.resolve(
        "mxbai-embed-large",
        BackendType.KIRON_EMBEDDINGS,
    )
    assert definition is not None
    revision = definition.huggingface_revision
    assert revision is not None
    local = LocalModelInventory(
        huggingface_revisions=frozenset((revision,)) if installed else frozenset()
    )
    runtime = RuntimeInventory({
        BackendType.KIRON_EMBEDDINGS: BackendRuntimeSnapshot(
            known=known,
            loaded_names=loaded,
            loading_names=loading,
        )
    })
    states = MODEL_STATE_VIEW.states(
        local,
        runtime,
        backends=(BackendType.KIRON_EMBEDDINGS,),
    )
    return next(item for item in states if item.definition is definition)


def test_state_view_uses_the_single_loaded_catalog() -> None:
    assert MODEL_STATE_VIEW.catalog is MODEL_CATALOG
    assert MODEL_STATE_VIEW.catalog_digest == MODEL_CATALOG.catalog_digest


def test_configured_but_not_installed_is_distinct() -> None:
    state = _state(installed=False, known=True)
    assert state.configured is True
    assert state.installed is False
    assert state.runtime_state is RuntimeState.UNLOADED


def test_installed_but_unloaded_is_distinct() -> None:
    state = _state(installed=True, known=True)
    assert state.configured is True
    assert state.installed is True
    assert state.runtime_state is RuntimeState.UNLOADED


def test_loading_is_derived_only_from_runtime_snapshot() -> None:
    state = _state(
        installed=True,
        known=True,
        loading=frozenset(("mxbai-embed-large",)),
    )
    assert state.runtime_state is RuntimeState.LOADING
    assert state.loading is True
    assert state.loaded is False


def test_loaded_is_derived_only_from_runtime_snapshot() -> None:
    state = _state(
        installed=True,
        known=True,
        loaded=frozenset(("mxbai-embed-large",)),
    )
    assert state.runtime_state is RuntimeState.LOADED
    assert state.loaded is True


def test_unknown_runtime_is_not_guessed_from_installation() -> None:
    state = _state(installed=True, known=False)
    assert state.installed is True
    assert state.runtime_state is RuntimeState.UNKNOWN
    assert state.loaded is False
    assert state.loading is False


def test_ollama_installation_uses_only_exact_native_tags() -> None:
    definition = MODEL_STATE_VIEW.resolve(
        "hellord/e5-mistral-7b-instruct:Q4_0",
        BackendType.OLLAMA,
    )
    assert definition is not None
    runtime = RuntimeInventory({
        BackendType.OLLAMA: BackendRuntimeSnapshot(known=True)
    })
    wrong = MODEL_STATE_VIEW.states(
        LocalModelInventory(ollama_tags=frozenset(("e5-mistral-7b-instruct:latest",))),
        runtime,
        backends=(BackendType.OLLAMA,),
    )
    exact = MODEL_STATE_VIEW.states(
        LocalModelInventory(ollama_tags=frozenset((definition.backend_model_name,))),
        runtime,
        backends=(BackendType.OLLAMA,),
    )
    assert next(item for item in wrong if item.definition is definition).installed is False
    assert next(item for item in exact if item.definition is definition).installed is True


def test_huggingface_scan_requires_exact_revision_and_declared_files(
    tmp_path: Path,
) -> None:
    definition = MODEL_STATE_VIEW.resolve(
        "mxbai-embed-large",
        BackendType.KIRON_EMBEDDINGS,
    )
    assert definition is not None
    revision = definition.huggingface_revision
    assert revision is not None
    repo_dir = f"models--{revision.repository.replace('/', '--')}"
    wrong = tmp_path / repo_dir / "snapshots" / ("0" * 40)
    wrong.mkdir(parents=True)
    assert revision not in scan_huggingface_inventory(MODEL_STATE_VIEW, tmp_path)

    exact = tmp_path / repo_dir / "snapshots" / revision.revision
    exact.mkdir(parents=True)
    assert revision not in scan_huggingface_inventory(MODEL_STATE_VIEW, tmp_path)
    for relative in definition.required_files:
        target = exact / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    assert revision in scan_huggingface_inventory(MODEL_STATE_VIEW, tmp_path)


def test_unknown_runtime_snapshot_cannot_smuggle_names() -> None:
    with pytest.raises(ValueError, match="unknown runtime"):
        BackendRuntimeSnapshot(
            known=False,
            loaded_names=frozenset(("some-model",)),
        )
