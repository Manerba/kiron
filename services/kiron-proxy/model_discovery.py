"""Pure rendering of Catalog-managed and generic local model inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kiron_common.catalog_consistency import check_catalog_digests
from kiron_common.local_model_registry import RegistryEntry
from kiron_common.local_model_registry.models import canonical_ollama_reference
from kiron_common.embedding_registry import build_embedding_registry
from kiron_common.model_catalog import ArtifactFormat, BackendType, LoaderType, ModelTask
from kiron_common.model_state import (
    BackendRuntimeSnapshot,
    HuggingFaceRevision,
    LocalModelInventory,
    ManagedModelDefinition,
    ManagedModelState,
    ModelStateView,
    RuntimeInventory,
    RuntimeState,
)
from kiron_common.ollama_compat import is_real_int


class InventoryShapeError(ValueError):
    """An inventory payload cannot safely describe local/runtime state."""


class ServiceInventoryError(InventoryShapeError):
    """Managed-service inventory is missing or Catalog-inconsistent."""


_SERVICE_BACKENDS = (
    BackendType.KIRON_EMBEDDINGS,
    BackendType.KIRON_DEBERTA,
)


def _gb_or_none(value: object) -> float | None:
    return round(value / (1024 ** 3), 2) if is_real_int(value) else None


def _native_rows(rows: object, *, path: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, list):
        raise InventoryShapeError(f"{path} must be an array")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InventoryShapeError(f"{path}/{index} must be an object")
        name = row.get("name")
        if type(name) is not str or not name:
            raise InventoryShapeError(f"{path}/{index}/name must be a non-empty string")
        if name in names:
            raise InventoryShapeError(f"{path}/{index}/name duplicates {name!r}")
        names.add(name)
        result.append(dict(row))
    return tuple(result)


def _ollama_runtime_snapshot(
    payload: object,
) -> tuple[BackendRuntimeSnapshot, Mapping[str, dict[str, Any]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return BackendRuntimeSnapshot(known=False), {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in payload["models"]:
        if not isinstance(row, dict):
            return BackendRuntimeSnapshot(known=False), {}
        name = row.get("name")
        if type(name) is not str or not name or name in by_name:
            return BackendRuntimeSnapshot(known=False), {}
        by_name[name] = dict(row)
    return (
        BackendRuntimeSnapshot(
            known=True,
            loaded_names=frozenset(by_name),
        ),
        by_name,
    )


def _service_runtime_snapshot(
    health: object,
    *,
    reachable: bool,
) -> BackendRuntimeSnapshot:
    if not reachable or not isinstance(health, dict):
        return BackendRuntimeSnapshot(known=False)
    loaded = health.get("loaded_models")
    if not isinstance(loaded, list) or any(
        type(name) is not str or not name for name in loaded
    ):
        return BackendRuntimeSnapshot(known=False)
    loading_value = health.get("loading_model")
    if loading_value is not None and (
        type(loading_value) is not str or not loading_value
    ):
        return BackendRuntimeSnapshot(known=False)
    if health.get("status") == "loading" and loading_value is None:
        return BackendRuntimeSnapshot(known=False)
    return BackendRuntimeSnapshot(
        known=True,
        loaded_names=frozenset(loaded),
        loading_names=(
            frozenset((loading_value,))
            if loading_value is not None
            else frozenset()
        ),
    )


def build_runtime_inventory(
    *,
    ollama_ps: object,
    embedding_health: object,
    embedding_reachable: bool,
    deberta_health: object,
    deberta_reachable: bool,
) -> tuple[RuntimeInventory, Mapping[str, dict[str, Any]]]:
    ollama, ollama_rows = _ollama_runtime_snapshot(ollama_ps)
    return (
        RuntimeInventory({
            BackendType.OLLAMA: ollama,
            BackendType.KIRON_EMBEDDINGS: _service_runtime_snapshot(
                embedding_health,
                reachable=embedding_reachable,
            ),
            BackendType.KIRON_DEBERTA: _service_runtime_snapshot(
                deberta_health,
                reachable=deberta_reachable,
            ),
        }),
        ollama_rows,
    )


def service_huggingface_inventory(
    state_view: ModelStateView,
    health_by_backend: Mapping[BackendType, object],
    *,
    reachable_backends: Sequence[BackendType],
) -> frozenset[HuggingFaceRevision]:
    """Validate exact HF inventory reported by currently reachable services."""

    if not isinstance(health_by_backend, Mapping):
        raise TypeError("health_by_backend must be a mapping")
    selected_backends = tuple(reachable_backends)
    if any(backend not in _SERVICE_BACKENDS for backend in selected_backends):
        raise ValueError("reachable_backends contains an unsupported backend")
    if len(set(selected_backends)) != len(selected_backends):
        raise ValueError("reachable_backends must not contain duplicates")
    if not selected_backends:
        return frozenset()

    payloads: dict[BackendType, Mapping[str, Any]] = {}
    reported_digests: dict[str, object] = {}
    for backend in selected_backends:
        raw = health_by_backend.get(backend)
        payload = raw if isinstance(raw, Mapping) else {}
        payloads[backend] = payload
        reported_digests[backend.value] = payload.get("catalog_digest")

    digest_report = check_catalog_digests(
        state_view.catalog_digest,
        reported_digests,
        required_services=tuple(backend.value for backend in selected_backends),
    )
    if not digest_report.consistent:
        raise ServiceInventoryError("; ".join(digest_report.errors))

    installed: set[HuggingFaceRevision] = set()
    for backend in selected_backends:
        path = f"/{backend.value}/model_states"
        rows = payloads[backend].get("model_states")
        if not isinstance(rows, list):
            raise ServiceInventoryError(f"{path} must be an array")
        expected = {
            definition.backend_model_name: definition
            for definition in state_view.for_backend(backend)
        }
        observed: set[str] = set()
        for index, row in enumerate(rows):
            row_path = f"{path}/{index}"
            if not isinstance(row, Mapping):
                raise ServiceInventoryError(f"{row_path} must be an object")
            name = row.get("name")
            if type(name) is not str or name not in expected:
                raise ServiceInventoryError(
                    f"{row_path}/name is not an exact configured runtime name: {name!r}"
                )
            if name in observed:
                raise ServiceInventoryError(f"{row_path}/name duplicates {name!r}")
            observed.add(name)
            definition = expected[name]
            if row.get("backend") != backend.value:
                raise ServiceInventoryError(
                    f"{row_path}/backend must equal {backend.value!r}"
                )
            if row.get("canonical_model_id") != definition.canonical_model_id:
                raise ServiceInventoryError(
                    f"{row_path}/canonical_model_id differs from the local Catalog"
                )
            if row.get("configured") is not True:
                raise ServiceInventoryError(f"{row_path}/configured must be true")
            installed_value = row.get("installed")
            if type(installed_value) is not bool:
                raise ServiceInventoryError(f"{row_path}/installed must be a bool")
            if installed_value:
                revision = definition.huggingface_revision
                if revision is None:  # pragma: no cover - Catalog invariant
                    raise ServiceInventoryError(
                        f"{row_path} reports a non-HuggingFace service artifact"
                    )
                installed.add(revision)

        missing = sorted(set(expected) - observed)
        if missing:
            raise ServiceInventoryError(
                f"{path} lacks configured runtime names: {missing!r}"
            )
    return frozenset(installed)


def _generic_model_type(name: str, family: str) -> str:
    """UI-only classification for inventory outside KIron's Catalog."""

    name_lower = name.lower()
    family_lower = family.lower()
    if (
        "embed" in name_lower
        or "bert" in family_lower
        or "/gte-" in name_lower
        or "/e5-" in name_lower
    ):
        return "embedding"
    if "vl" in family_lower or "vision" in name_lower:
        return "vlm"
    return "llm"


def _matching_runtime_row(
    definition: ManagedModelDefinition,
    rows: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for name in definition.input_names:
        if name in rows:
            return rows[name]
    return None


def _catalog_token_limits(
    state_view: ModelStateView,
    definition: ManagedModelDefinition,
) -> list[dict[str, Any]]:
    group = next(
        (
            item
            for item in state_view.catalog.groups
            if item.canonical_model_id == definition.canonical_model_id
        ),
        None,
    )
    if group is None:  # pragma: no cover - ModelStateView invariant
        raise AssertionError("managed definition lost its Catalog group")

    profile_ids = frozenset(definition.profile_ids)
    limits: list[dict[str, Any]] = []
    for profile in group.profiles:
        if profile.id not in profile_ids:
            continue
        raw = profile.metadata.get("max_input_tokens")
        if not isinstance(raw, Mapping):
            continue
        by_role = raw.get("by_role")
        if not isinstance(by_role, Mapping):
            continue
        limits.append(
            {
                "profile_id": profile.id,
                "unit": raw.get("unit"),
                "counting": raw.get("counting"),
                "truncation": raw.get("truncation"),
                "overflow": raw.get("overflow"),
                "by_role": {
                    str(role): value if is_real_int(value) else None
                    for role, value in sorted(by_role.items())
                },
            }
        )
    return limits


def _embedding_profiles(registry, definition: ManagedModelDefinition) -> list[dict[str, Any]]:
    group = registry.resolve(definition.canonical_model_id)
    if group is None:
        return []
    return [
        {
            "profile_id": profile["profile_id"],
            "kind": profile["kind"],
            "endpoint": profile["endpoint"],
            "verification": {
                "status": profile["verification"]["status"],
                "blocking_reasons": list(profile["verification"]["blocking_reasons"]),
            },
            "index_compatibility_id": profile["index_compatibility_id"],
            "query_compatibility_id": profile["query_compatibility_id"],
        }
        for profile in group.capabilities["profiles"]
        if profile["profile_id"] in definition.profile_ids
    ]


def _ollama_native_context_length(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    model_info = payload.get("model_info")
    if not isinstance(model_info, Mapping):
        return None

    architecture = model_info.get("general.architecture")
    if type(architecture) is str and architecture:
        exact = model_info.get(f"{architecture}.context_length")
        if is_real_int(exact) and exact > 0:
            return exact

    candidates = {
        value
        for key, value in model_info.items()
        if type(key) is str
        and key.endswith(".context_length")
        and is_real_int(value)
        and value > 0
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _managed_local_row(
    state: ManagedModelState,
    ollama_runtime_rows: Mapping[str, dict[str, Any]],
    ollama_show_rows: Mapping[str, Mapping[str, Any]],
    *,
    catalog_token_limits: list[dict[str, Any]],
    embedding_profiles: list[dict[str, Any]],
    catalog_digest: str,
    installation_known: bool,
    service_memory: object,
) -> dict[str, Any]:
    definition = state.definition
    runtime_row = (
        _matching_runtime_row(definition, ollama_runtime_rows)
        if definition.backend is BackendType.OLLAMA
        else None
    )
    show_row = None
    if definition.backend is BackendType.OLLAMA:
        for input_name in definition.input_names:
            canonical = canonical_ollama_reference(input_name)
            if canonical is not None:
                show_row = ollama_show_rows.get(canonical.lower())
            if show_row is not None:
                break
    total = runtime_row.get("size") if runtime_row is not None else None
    vram = runtime_row.get("size_vram") if runtime_row is not None else None
    if state.runtime_state is RuntimeState.UNKNOWN:
        vram_gb = None
        ram_gb = None
    elif runtime_row is not None and is_real_int(total) and is_real_int(vram):
        vram_gb = _gb_or_none(vram)
        ram_gb = round(max(0, total - vram) / (1024 ** 3), 2)
    elif state.runtime_state is RuntimeState.UNLOADED:
        vram_gb = 0.0
        ram_gb = 0.0
    else:
        vram_gb = None
        ram_gb = None
    memory = service_memory.get(definition.backend.value) if isinstance(service_memory, dict) else None
    if (not state.loaded or not isinstance(memory, dict)
            or definition.backend_model_name not in memory.get("loaded_models", [])):
        memory = None
    service_embedding = definition.backend is BackendType.KIRON_EMBEDDINGS
    embedding_capable = (
        service_embedding and ModelTask.EMBEDDING in definition.tasks
    )
    return {
        "name": definition.backend_model_name,
        "canonical_model_id": definition.canonical_model_id,
        "aliases": list(definition.aliases),
        "backend": definition.backend.value,
        "deployment_ids": list(definition.deployment_ids),
        "profile_ids": list(definition.profile_ids),
        "tasks": [task.value for task in definition.tasks],
        "endpoints": [endpoint.value for endpoint in definition.endpoints],
        "catalog_digest": catalog_digest,
        "configured": state.configured,
        "installed": state.installed if installation_known else None,
        "runtime_state": state.runtime_state.value,
        "loaded": state.loaded,
        "loading": state.loading,
        "load_state": state.runtime_state.value,
        "parameter_size": definition.parameter_size or "",
        "quantization_level": definition.precision or "",
        "family": definition.family or "",
        "format": definition.format or "",
        "model_type": definition.model_type,
        "size_gb": _gb_or_none(definition.size_bytes),
        "embedding_capable": embedding_capable,
        "embedding_active": service_embedding and state.loaded,
        "embedding_loading": service_embedding and state.loading,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "service_memory": memory,
        "native_context_length": _ollama_native_context_length(show_row),
        "runtime_context_length": (
            runtime_row.get("context_length")
            if runtime_row is not None
            and is_real_int(runtime_row.get("context_length"))
            and runtime_row["context_length"] > 0
            else None
        ),
        "runtime_device": ("cpu" if runtime_row["size_vram"] == 0 else "gpu")
        if runtime_row is not None and is_real_int(runtime_row.get("size_vram")) and runtime_row["size_vram"] >= 0 else None,
        "catalog_context_length": definition.context_length,
        "catalog_token_limits": catalog_token_limits,
        "embedding_profiles": embedding_profiles,
        "expires_at": runtime_row.get("expires_at") if runtime_row else None,
        "embedding_only": service_embedding,
        "managed_service": definition.backend is not BackendType.OLLAMA,
        "embedding_kind": definition.embedding_kind,
        "source": definition.backend.value,
        "catalog_managed": True,
        "locally_registered": False,
        "registry_id": None,
        "runtime_provider": None,
        "reference": None,
        "loader": None,
    }


def _generic_local_row(
    row: Mapping[str, Any],
    runtime: BackendRuntimeSnapshot,
    runtime_rows: Mapping[str, dict[str, Any]],
    show_row: Mapping[str, Any] | None,
    registration: RegistryEntry,
) -> dict[str, Any]:
    name = row["name"]
    details_value = row.get("details")
    details = details_value if isinstance(details_value, dict) else {}
    family = details.get("family") if type(details.get("family")) is str else ""
    canonical = name if ":" in name else f"{name}:latest"
    loaded_row = runtime_rows.get(name) or runtime_rows.get(canonical)
    if not runtime.known:
        runtime_state = RuntimeState.UNKNOWN
    elif loaded_row is not None:
        runtime_state = RuntimeState.LOADED
    else:
        runtime_state = RuntimeState.UNLOADED
    total = loaded_row.get("size") if loaded_row is not None else None
    vram = loaded_row.get("size_vram") if loaded_row is not None else None
    if runtime_state is RuntimeState.UNKNOWN:
        vram_gb = None
        ram_gb = None
    elif is_real_int(total) and is_real_int(vram):
        vram_gb = _gb_or_none(vram)
        ram_gb = round(max(0, total - vram) / (1024 ** 3), 2)
    elif runtime_state is RuntimeState.UNLOADED:
        vram_gb = 0.0
        ram_gb = 0.0
    else:
        vram_gb = None
        ram_gb = None
    return {
        "name": name,
        "configured": False,
        "installed": True,
        "backend": BackendType.OLLAMA.value,
        "runtime_state": runtime_state.value,
        "loaded": runtime_state is RuntimeState.LOADED,
        "loading": False,
        "load_state": runtime_state.value,
        "parameter_size": details.get("parameter_size", ""),
        "quantization_level": details.get("quantization_level", ""),
        "family": family,
        "format": details.get("format", ""),
        "model_type": _generic_model_type(name, family),
        "tasks": [],
        "endpoints": [],
        "profile_ids": [],
        "deployment_ids": [],
        "size_gb": _gb_or_none(row.get("size")),
        "runtime_device": ("cpu" if vram == 0 else "gpu") if is_real_int(vram) and vram >= 0 else None,
        "embedding_capable": False,
        "embedding_active": False,
        "embedding_loading": False,
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "native_context_length": _ollama_native_context_length(show_row),
        "runtime_context_length": (
            loaded_row.get("context_length")
            if loaded_row is not None
            and is_real_int(loaded_row.get("context_length"))
            and loaded_row["context_length"] > 0
            else None
        ),
        "catalog_context_length": None,
        "catalog_token_limits": [],
        "expires_at": loaded_row.get("expires_at") if loaded_row else None,
        "embedding_only": False,
        "managed_service": False,
        "embedding_kind": None,
        "source": "ollama",
        "catalog_managed": False,
        "locally_registered": True,
        "registry_id": registration.id,
        "runtime_provider": registration.runtime_provider.value,
        "artifact_origin": registration.artifact_origin.value,
        "artifact_format": registration.artifact_format.value,
        "reference": registration.reference,
        "display_name": registration.display_name,
        "loader": registration.loader.value,
    }


def _registered_huggingface_row(registration: RegistryEntry) -> dict[str, Any]:
    model_type = (
        "colbert"
        if registration.loader is LoaderType.COLBERT_XMOD
        else "rerank"
        if registration.loader is LoaderType.CROSS_ENCODER
        else "embedding"
    )
    return {
        "name": registration.display_name,
        "configured": False,
        "installed": True,
        "backend": registration.runtime_provider.value,
        "runtime_state": RuntimeState.UNKNOWN.value,
        "loaded": False,
        "loading": False,
        "load_state": RuntimeState.UNKNOWN.value,
        "parameter_size": "",
        "quantization_level": "",
        "family": "",
        "format": "",
        "model_type": model_type,
        "tasks": [],
        "endpoints": [],
        "profile_ids": [],
        "deployment_ids": [],
        "size_gb": None,
        "embedding_capable": model_type == "embedding",
        "embedding_active": False,
        "embedding_loading": False,
        "vram_gb": None,
        "ram_gb": None,
        "native_context_length": None,
        "runtime_context_length": None,
        "catalog_context_length": None,
        "catalog_token_limits": [],
        "expires_at": None,
        "embedding_only": False,
        "managed_service": False,
        "embedding_kind": None,
        "source": "local_registry",
        "catalog_managed": False,
        "locally_registered": True,
        "registry_id": registration.id,
        "runtime_provider": registration.runtime_provider.value,
        "artifact_origin": registration.artifact_origin.value,
        "artifact_format": registration.artifact_format.value,
        "reference": registration.reference,
        "display_name": registration.display_name,
        "loader": registration.loader.value,
    }


def _apply_registration(row: dict[str, Any], registration: RegistryEntry) -> None:
    row.update(
        {
            "locally_registered": True,
            "registry_id": registration.id,
            "runtime_provider": registration.runtime_provider.value,
        "artifact_origin": registration.artifact_origin.value,
        "artifact_format": registration.artifact_format.value,
            "reference": registration.reference,
            "display_name": registration.display_name,
            "loader": registration.loader.value,
        }
    )


def build_local_models_payload(
    *,
    state_view: ModelStateView,
    huggingface_revisions: frozenset[HuggingFaceRevision],
    unavailable_huggingface_backends: frozenset[BackendType],
    ollama_tag_rows: object,
    ollama_ps: object,
    embedding_health: object,
    embedding_reachable: bool,
    deberta_health: object,
    deberta_reachable: bool,
    registrations: Sequence[RegistryEntry],
    ollama_show_by_name: Mapping[str, Mapping[str, Any]],
    service_memory: object,
) -> dict[str, Any]:
    """Build the complete local model wire from injected observations."""

    if type(unavailable_huggingface_backends) is not frozenset or any(
        backend not in _SERVICE_BACKENDS
        for backend in unavailable_huggingface_backends
    ):
        raise TypeError(
            "unavailable_huggingface_backends must be a frozenset of service backends"
        )
    native_rows = _native_rows(ollama_tag_rows, path="/ollama/tags/models")
    native_names = frozenset(row["name"] for row in native_rows)
    if not isinstance(registrations, Sequence) or any(
        type(entry) is not RegistryEntry for entry in registrations
    ):
        raise TypeError("registrations must contain RegistryEntry values")
    local = LocalModelInventory(
        huggingface_revisions=huggingface_revisions,
        ollama_tags=native_names,
    )
    runtime, ollama_runtime_rows = build_runtime_inventory(
        ollama_ps=ollama_ps,
        embedding_health=embedding_health,
        embedding_reachable=embedding_reachable,
        deberta_health=deberta_health,
        deberta_reachable=deberta_reachable,
    )
    managed_states = state_view.states(local, runtime)
    if not isinstance(ollama_show_by_name, Mapping) or any(
        type(name) is not str or not isinstance(payload, Mapping)
        for name, payload in ollama_show_by_name.items()
    ):
        raise TypeError("ollama_show_by_name must map model names to objects")
    ollama_show_rows: dict[str, Mapping[str, Any]] = {}
    for name, payload in ollama_show_by_name.items():
        canonical = canonical_ollama_reference(name)
        if canonical is None:
            continue
        key = canonical.lower()
        if key in ollama_show_rows:
            raise InventoryShapeError(
                "/ollama/show contains duplicate canonical names"
            )
        ollama_show_rows[key] = payload
    embedding_registry = build_embedding_registry(state_view.catalog)
    rows = [
        _managed_local_row(
            state,
            ollama_runtime_rows,
            ollama_show_rows,
            catalog_token_limits=_catalog_token_limits(
                state_view,
                state.definition,
            ),
            catalog_digest=state_view.catalog_digest,
            embedding_profiles=_embedding_profiles(embedding_registry, state.definition),
            installation_known=(
                state.definition.backend not in unavailable_huggingface_backends
            ),
            service_memory=service_memory,
        )
        for state in managed_states
    ]
    managed_by_id = {
        row["canonical_model_id"]: row
        for row in rows
    }
    native_by_reference: dict[str, dict[str, Any]] = {}
    for native_row in native_rows:
        canonical = canonical_ollama_reference(native_row["name"])
        if canonical is None:
            continue
        key = canonical.lower()
        if key in native_by_reference:
            raise InventoryShapeError(
                "/ollama/tags/models contains duplicate canonical names"
            )
        native_by_reference[key] = native_row

    ollama_runtime = runtime.for_backend(BackendType.OLLAMA)
    for registration in registrations:
        if registration.artifact_format is ArtifactFormat.HF_WEIGHTS:
            rows.append(_registered_huggingface_row(registration))
            continue
        if registration.runtime_provider is BackendType.PRISM:
            row = _registered_huggingface_row(registration)
            row.update(model_type="llm", embedding_capable=False, installed=None,
                       installation_known=False, deployment_ids=[registration.id],
                       size_gb=_gb_or_none(registration.size_bytes),
                       format=registration.artifact_format.value)
            rows.append(row)
            continue
        if registration.runtime_provider is not BackendType.OLLAMA:
            raise InventoryShapeError("unsupported registration provider")
        definition = state_view.resolve(
            registration.reference,
            BackendType.OLLAMA,
        )
        if definition is not None:
            managed = managed_by_id.get(definition.canonical_model_id)
            if managed is not None:
                _apply_registration(managed, registration)
            continue
        native_row = native_by_reference.get(registration.reference.lower())
        if native_row is not None:
            canonical = canonical_ollama_reference(native_row["name"])
            show_row = (
                ollama_show_rows.get(canonical.lower())
                if canonical is not None
                else None
            )
            rows.append(
                _generic_local_row(
                    native_row,
                    ollama_runtime,
                    ollama_runtime_rows,
                    show_row,
                    registration,
                )
            )

    def _summary(backend: BackendType, health: object, reachable: bool) -> dict[str, Any]:
        definitions = state_view.for_backend(backend)
        backend_states = [
            state for state in managed_states if state.definition.backend is backend
        ]
        health_data = health if isinstance(health, dict) else {}
        return {
            "running": reachable,
            "status": health_data.get("status") if reachable else "down",
            "current_model": health_data.get("current_model") if reachable else None,
            "loaded_models": sorted(
                state.definition.backend_model_name
                for state in backend_states
                if state.loaded
            ),
            "loading_model": health_data.get("loading_model") if reachable else None,
            "available_models": [item.backend_model_name for item in definitions],
            "catalog_digest": health_data.get("catalog_digest") if reachable else None,
        }

    embedding_summary = _summary(
        BackendType.KIRON_EMBEDDINGS,
        embedding_health,
        embedding_reachable,
    )
    embedding_summary["available_models"] = [
        item.backend_model_name
        for item in state_view.for_backend(BackendType.KIRON_EMBEDDINGS)
        if item.embedding_kind != "colbert"
    ]
    if isinstance(embedding_health, dict):
        slots = embedding_health.get("model_slots")
        embedding_summary["model_slots"] = slots if is_real_int(slots) else None
    else:
        embedding_summary["model_slots"] = None
    embedding_summary["available_colbert_models"] = [
        item.backend_model_name
        for item in state_view.for_backend(BackendType.KIRON_EMBEDDINGS)
        if item.embedding_kind == "colbert"
    ]

    return {
        "models": rows,
        "models_loaded_state": "known" if ollama_runtime.known else "unknown",
        "catalog_digest": state_view.catalog_digest,
        "embedding_service": embedding_summary,
        "deberta_service": _summary(
            BackendType.KIRON_DEBERTA,
            deberta_health,
            deberta_reachable,
        ),
    }


__all__ = [
    "InventoryShapeError",
    "ServiceInventoryError",
    "build_local_models_payload",
    "build_runtime_inventory",
    "service_huggingface_inventory",
]
