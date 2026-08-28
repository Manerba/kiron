"""Immutable Catalog-derived model discovery and state projection.

The view contains configuration only.  Installation and runtime state enter
through explicit immutable inventories, keeping filesystem, Ollama, and service
health observations independently mockable.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .model_catalog import (
    Artifact,
    ArtifactType,
    BackendType,
    ModelCatalog,
    ModelEndpoint,
    ModelTask,
)


class ModelStateCatalogError(ValueError):
    """A malformed Catalog projection for model discovery."""


class RuntimeState(str, Enum):
    UNKNOWN = "unknown"
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"


@dataclass(frozen=True, slots=True, order=True)
class HuggingFaceRevision:
    repository: str
    revision: str


@dataclass(frozen=True, slots=True)
class LocalModelInventory:
    """Exact local artifacts observed without any network access."""

    huggingface_revisions: frozenset[HuggingFaceRevision] = frozenset()
    ollama_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "huggingface_revisions",
            frozenset(self.huggingface_revisions),
        )
        object.__setattr__(self, "ollama_tags", frozenset(self.ollama_tags))
        if any(type(tag) is not str or not tag for tag in self.ollama_tags):
            raise ValueError("Ollama inventory tags must be non-empty strings")


@dataclass(frozen=True, slots=True)
class BackendRuntimeSnapshot:
    """One atomic backend observation."""

    known: bool
    loaded_names: frozenset[str] = frozenset()
    loading_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if type(self.known) is not bool:
            raise TypeError("known must be a bool")
        loaded = frozenset(self.loaded_names)
        loading = frozenset(self.loading_names)
        if any(type(name) is not str or not name for name in (*loaded, *loading)):
            raise ValueError("runtime model names must be non-empty strings")
        if not self.known and (loaded or loading):
            raise ValueError("unknown runtime snapshots must not carry model names")
        object.__setattr__(self, "loaded_names", loaded)
        object.__setattr__(self, "loading_names", loading)


@dataclass(frozen=True, slots=True)
class RuntimeInventory:
    by_backend: Mapping[BackendType, BackendRuntimeSnapshot] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        normalized: dict[BackendType, BackendRuntimeSnapshot] = {}
        for backend, snapshot in self.by_backend.items():
            if not isinstance(backend, BackendType):
                raise TypeError("runtime inventory keys must be BackendType values")
            if not isinstance(snapshot, BackendRuntimeSnapshot):
                raise TypeError("runtime inventory values must be BackendRuntimeSnapshot")
            normalized[backend] = snapshot
        object.__setattr__(self, "by_backend", MappingProxyType(normalized))

    def for_backend(self, backend: BackendType) -> BackendRuntimeSnapshot:
        return self.by_backend.get(backend, BackendRuntimeSnapshot(known=False))


@dataclass(frozen=True, slots=True)
class ManagedModelDefinition:
    canonical_model_id: str
    aliases: tuple[str, ...]
    backend: BackendType
    backend_model_name: str
    deployment_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    tasks: tuple[ModelTask, ...]
    endpoints: tuple[ModelEndpoint, ...]
    artifact: Artifact
    required_files: tuple[str, ...]
    family: str | None
    format: str | None
    parameter_size: str | None
    precision: str | None
    size_bytes: int | None
    context_length: int | None
    model_type: str
    embedding_kind: str | None

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.canonical_model_id, *self.aliases)

    @property
    def huggingface_revision(self) -> HuggingFaceRevision | None:
        if self.artifact.type is not ArtifactType.HUGGINGFACE:
            return None
        repository = self.artifact.repository
        revision = self.artifact.revision
        if repository is None or revision is None:
            raise AssertionError("validated HuggingFace artifact lost its coordinates")
        return HuggingFaceRevision(repository, revision)


@dataclass(frozen=True, slots=True)
class ManagedModelState:
    definition: ManagedModelDefinition
    configured: bool
    installed: bool
    runtime_state: RuntimeState

    @property
    def loaded(self) -> bool:
        return self.runtime_state is RuntimeState.LOADED

    @property
    def loading(self) -> bool:
        return self.runtime_state is RuntimeState.LOADING

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition
        return {
            "canonical_model_id": definition.canonical_model_id,
            "aliases": list(definition.aliases),
            "name": definition.backend_model_name,
            "backend": definition.backend.value,
            "deployment_ids": list(definition.deployment_ids),
            "profile_ids": list(definition.profile_ids),
            "tasks": [task.value for task in definition.tasks],
            "endpoints": [endpoint.value for endpoint in definition.endpoints],
            "configured": self.configured,
            "installed": self.installed,
            "runtime_state": self.runtime_state.value,
            "loaded": self.loaded,
            "loading": self.loading,
            "family": definition.family,
            "format": definition.format,
            "parameter_size": definition.parameter_size,
            "precision": definition.precision,
            "size": definition.size_bytes,
            "context_length": definition.context_length,
            "model_type": definition.model_type,
            "embedding_kind": definition.embedding_kind,
        }


@dataclass(frozen=True, slots=True)
class ModelStateView:
    """Pure, read-only model definitions derived from one Catalog instance."""

    catalog: ModelCatalog
    models: tuple[ManagedModelDefinition, ...]
    _names_by_backend: Mapping[BackendType, Mapping[str, ManagedModelDefinition]] = field(
        repr=False
    )

    @property
    def catalog_digest(self) -> str:
        return self.catalog.catalog_digest

    def for_backend(self, backend: BackendType) -> tuple[ManagedModelDefinition, ...]:
        return tuple(model for model in self.models if model.backend is backend)

    def resolve(
        self,
        name: object,
        backend: BackendType,
    ) -> ManagedModelDefinition | None:
        if type(name) is not str:
            return None
        return self._names_by_backend.get(backend, {}).get(name)

    def states(
        self,
        local_inventory: LocalModelInventory,
        runtime_inventory: RuntimeInventory,
        *,
        backends: Iterable[BackendType] | None = None,
    ) -> tuple[ManagedModelState, ...]:
        if not isinstance(local_inventory, LocalModelInventory):
            raise TypeError("local_inventory must be LocalModelInventory")
        if not isinstance(runtime_inventory, RuntimeInventory):
            raise TypeError("runtime_inventory must be RuntimeInventory")
        selected = frozenset(backends) if backends is not None else None
        states: list[ManagedModelState] = []
        for definition in self.models:
            if selected is not None and definition.backend not in selected:
                continue
            installed = _is_installed(definition, local_inventory)
            runtime = runtime_inventory.for_backend(definition.backend)
            runtime_state = _runtime_state(definition, runtime)
            states.append(
                ManagedModelState(
                    definition=definition,
                    configured=True,
                    installed=installed,
                    runtime_state=runtime_state,
                )
            )
        return tuple(states)


def _is_installed(
    definition: ManagedModelDefinition,
    inventory: LocalModelInventory,
) -> bool:
    hf_revision = definition.huggingface_revision
    if hf_revision is not None:
        return hf_revision in inventory.huggingface_revisions
    if definition.artifact.type is ArtifactType.OLLAMA:
        return any(name in inventory.ollama_tags for name in definition.input_names)
    raise AssertionError(f"unsupported artifact type: {definition.artifact.type!r}")


def _runtime_state(
    definition: ManagedModelDefinition,
    snapshot: BackendRuntimeSnapshot,
) -> RuntimeState:
    if not snapshot.known:
        return RuntimeState.UNKNOWN
    names = frozenset(definition.input_names)
    if names & snapshot.loading_names:
        return RuntimeState.LOADING
    if names & snapshot.loaded_names:
        return RuntimeState.LOADED
    return RuntimeState.UNLOADED


def _backend_model_name(parameters: Mapping[str, Any], *, path: str) -> str:
    value = parameters.get("model_name")
    if type(value) is not str or not value:
        raise ModelStateCatalogError(f"{path}/model_name must be a non-empty string")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _declared_values(values: Iterable[object], *, path: str) -> object | None:
    declared = [value for value in values if value is not None]
    if not declared:
        return None
    first = declared[0]
    if any(value != first for value in declared[1:]):
        raise ModelStateCatalogError(f"{path} has conflicting declared values")
    return first


def _definition_metadata(
    deployments: tuple[object, ...],
    profiles: tuple[object, ...],
) -> dict[str, object | None]:
    discovery = [
        _mapping(_mapping(deployment.metadata).get("discovery"))
        for deployment in deployments
    ]
    service = [
        _mapping(_mapping(deployment.metadata).get("service"))
        for deployment in deployments
    ]
    loader_parameters = [
        _mapping(deployment.loader.parameters) for deployment in deployments
    ]
    profile_metadata = [_mapping(profile.metadata) for profile in profiles]

    context_candidates: list[object] = []
    observed_precision: list[object] = []
    for metadata in profile_metadata:
        max_input = _mapping(metadata.get("max_input_tokens"))
        by_role = _mapping(max_input.get("by_role"))
        context_candidates.extend(
            value for value in by_role.values() if type(value) is int
        )
        parameters = _mapping(_mapping(metadata.get("pipeline")).get("parameters"))
        context_candidates.extend(
            parameters.get(key)
            for key in (
                "observed_context_length",
                "observed_num_ctx",
            )
            if type(parameters.get(key)) is int
        )
        if type(parameters.get("observed_quantization")) is str:
            observed_precision.append(parameters["observed_quantization"])

    kinds = {
        metadata.get("kind")
        for metadata in profile_metadata
        if type(metadata.get("kind")) is str
    }
    endpoints = {
        route.endpoint for deployment in deployments for route in deployment.routes
    }
    tasks = {route.task for deployment in deployments for route in deployment.routes}
    if ModelEndpoint.EMBED_COLBERT in endpoints or "multi_vector" in kinds:
        model_type = "colbert"
        embedding_kind = "colbert"
    elif ModelTask.EMBEDDING in tasks:
        model_type = "embedding"
        embedding_kind = "dense" if "dense" in kinds else (
            "late_chunking" if "late_chunking" in kinds else None
        )
    elif ModelTask.RERANK in tasks:
        model_type = "reranker"
        embedding_kind = None
    elif ModelTask.NLI in tasks:
        model_type = "nli"
        embedding_kind = None
    else:
        model_type = "managed"
        embedding_kind = None

    discovery_precision = _declared_values(
        (item.get("precision") for item in discovery),
        path="/state/metadata/discovery/precision",
    )
    loader_precision = _declared_values(
        (item.get("torch_dtype") for item in loader_parameters),
        path="/state/metadata/loader/torch_dtype",
    )
    observed_quantization = _declared_values(
        observed_precision,
        path="/state/metadata/pipeline/observed_quantization",
    )
    precision = (
        discovery_precision
        if discovery_precision is not None
        else loader_precision
        if loader_precision is not None
        else observed_quantization
    )
    return {
        "family": _declared_values(
            (item.get("family") for item in discovery),
            path="/state/metadata/family",
        ),
        "format": _declared_values(
            (item.get("format") for item in discovery),
            path="/state/metadata/format",
        ),
        "parameter_size": _declared_values(
            (item.get("parameter_size") for item in discovery),
            path="/state/metadata/parameter_size",
        ),
        "precision": precision,
        "size_bytes": _declared_values(
            [
                *(item.get("size") for item in discovery),
                *(item.get("size") for item in service),
            ],
            path="/state/metadata/size",
        ),
        "context_length": max(context_candidates) if context_candidates else None,
        "model_type": model_type,
        "embedding_kind": embedding_kind,
    }


def build_model_state_view(catalog: ModelCatalog) -> ModelStateView:
    """Build discovery definitions solely from an already-loaded Catalog."""

    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")

    definitions: list[ManagedModelDefinition] = []
    names_by_backend: dict[BackendType, dict[str, ManagedModelDefinition]] = {
        backend: {} for backend in BackendType
    }
    for group in catalog.groups:
        grouped: dict[tuple[BackendType, str], list[object]] = {}
        for deployment in group.deployments:
            backend_name = _backend_model_name(
                deployment.backend.parameters,
                path=f"/catalog/{group.canonical_model_id}/{deployment.id}/backend",
            )
            if backend_name not in (group.canonical_model_id, *group.aliases):
                raise ModelStateCatalogError(
                    f"/catalog/{group.canonical_model_id}/{deployment.id}/backend/model_name "
                    "must be an exact canonical model ID or declared alias"
                )
            grouped.setdefault((deployment.backend.type, backend_name), []).append(
                deployment
            )

        for (backend, backend_name), raw_deployments in grouped.items():
            deployments = tuple(raw_deployments)
            artifact = deployments[0].artifact
            if any(deployment.artifact != artifact for deployment in deployments[1:]):
                raise ModelStateCatalogError(
                    f"/catalog/{group.canonical_model_id}/{backend.value}/{backend_name} "
                    "uses conflicting artifacts for one runtime model"
                )
            deployment_ids = tuple(sorted(deployment.id for deployment in deployments))
            profiles = tuple(
                profile
                for profile in group.profiles
                if profile.deployment_id in deployment_ids
            )
            metadata = _definition_metadata(deployments, profiles)
            definition = ManagedModelDefinition(
                canonical_model_id=group.canonical_model_id,
                aliases=tuple(group.aliases),
                backend=backend,
                backend_model_name=backend_name,
                deployment_ids=deployment_ids,
                profile_ids=tuple(sorted(profile.id for profile in profiles)),
                tasks=tuple(sorted(
                    {route.task for deployment in deployments for route in deployment.routes},
                    key=lambda item: item.value,
                )),
                endpoints=tuple(sorted(
                    {route.endpoint for deployment in deployments for route in deployment.routes},
                    key=lambda item: item.value,
                )),
                artifact=artifact,
                required_files=tuple(sorted(
                    {item.path for item in (*artifact.weights, *artifact.auxiliary)}
                )),
                family=metadata["family"],
                format=metadata["format"],
                parameter_size=metadata["parameter_size"],
                precision=metadata["precision"],
                size_bytes=metadata["size_bytes"],
                context_length=metadata["context_length"],
                model_type=metadata["model_type"],
                embedding_kind=metadata["embedding_kind"],
            )
            definitions.append(definition)
            for name in definition.input_names:
                previous = names_by_backend[backend].get(name)
                if previous is not None and previous != definition:
                    raise ModelStateCatalogError(
                        f"/state/names/{backend.value}: {name!r} resolves to "
                        "multiple runtime models"
                    )
                names_by_backend[backend][name] = definition

    ordered = tuple(sorted(
        definitions,
        key=lambda item: (
            item.canonical_model_id,
            item.backend.value,
            item.backend_model_name,
        ),
    ))
    frozen_names = MappingProxyType({
        backend: MappingProxyType(dict(index))
        for backend, index in names_by_backend.items()
    })
    return ModelStateView(
        catalog=catalog,
        models=ordered,
        _names_by_backend=frozen_names,
    )


def default_huggingface_hub_cache(
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    explicit = values.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = values.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else Path("/var/cache/kiron/huggingface/hub")


def scan_huggingface_inventory(
    view: ModelStateView,
    cache_dir: Path,
) -> frozenset[HuggingFaceRevision]:
    """Read exact pinned snapshots and declared files without hashing or downloads."""

    installed: set[HuggingFaceRevision] = set()
    for definition in view.models:
        revision = definition.huggingface_revision
        if revision is None or revision in installed:
            continue
        repo_dir = f"models--{revision.repository.replace('/', '--')}"
        snapshot = cache_dir / repo_dir / "snapshots" / revision.revision
        try:
            present = snapshot.is_dir() and all(
                (snapshot / relative).is_file()
                for relative in definition.required_files
            )
        except OSError:
            present = False
        if present:
            installed.add(revision)
    return frozenset(installed)


__all__ = [
    "BackendRuntimeSnapshot",
    "HuggingFaceRevision",
    "LocalModelInventory",
    "ManagedModelDefinition",
    "ManagedModelState",
    "ModelStateCatalogError",
    "ModelStateView",
    "RuntimeInventory",
    "RuntimeState",
    "build_model_state_view",
    "default_huggingface_hub_cache",
    "scan_huggingface_inventory",
]
