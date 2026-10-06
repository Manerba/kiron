"""Catalog-derived embedding profile registry shared by KIron producers.

The declarative model catalog owns every model-specific identity and fact.
This module contains only the strict ADR-0009 projection, immutable lookup
facade, request-contract helpers, and discovery adapters.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .embedding_contract import (
    EmbeddingContractError,
    canonical_hash_json,
    canonical_json,
    finalize_capabilities,
    validate_capabilities,
)
from .model_catalog import (
    ArtifactType,
    BackendType,
    LoaderType,
    ModelCatalog,
    ModelTask,
    WireProfile,
    load_catalog,
)
from .model_state import build_model_state_view
from .model_catalog._json import (
    FrozenJSONMapping,
    freeze_json_mapping,
    thaw_json,
)


_PROFILE_METADATA_FIELDS = frozenset(
    (
        "kind",
        "dimensions",
        "max_input_tokens",
        "output_normalized",
        "similarity",
        "input_type",
        "formatting_owner",
        "pipeline",
        "verification",
    )
)
_BACKEND_PARAMETER_FIELDS = frozenset(
    ("implementation", "implementation_revision", "model_name")
)
_ALLOWED_EMBEDDING_LOADERS = MappingProxyType(
    {
        BackendType.KIRON_EMBEDDINGS: frozenset(
            (
                LoaderType.SENTENCE_TRANSFORMERS,
                LoaderType.TRANSFORMERS_LAST_TOKEN,
                LoaderType.COLBERT_XMOD,
            )
        ),
        BackendType.OLLAMA: frozenset((LoaderType.OLLAMA,)),
    }
)


def _closed_mapping(
    value: object,
    *,
    path: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbeddingContractError(path, "must be an object")
    if any(type(key) is not str for key in value):
        raise EmbeddingContractError(path, "object names must be strings")
    actual = set(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing:
        raise EmbeddingContractError(
            path, f"missing required fields: {', '.join(missing)}"
        )
    if unknown:
        raise EmbeddingContractError(
            path, f"unknown fields: {', '.join(unknown)}"
        )
    return value


def _non_empty_string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise EmbeddingContractError(path, "must be a non-empty string")
    return value


def _backend_projection(
    wire: WireProfile,
    *,
    path: str,
) -> tuple[str, dict[str, Any]]:
    allowed_loaders = _ALLOWED_EMBEDDING_LOADERS.get(wire.backend.type)
    if allowed_loaders is None:
        raise EmbeddingContractError(
            f"{path}/backend/type",
            f"backend {wire.backend.type.value!r} cannot serve embedding profiles",
        )
    if wire.loader.type not in allowed_loaders:
        raise EmbeddingContractError(
            f"{path}/loader/type",
            f"loader {wire.loader.type.value!r} is invalid for "
            f"embedding backend {wire.backend.type.value!r}",
        )

    expected_artifact = (
        ArtifactType.OLLAMA
        if wire.backend.type is BackendType.OLLAMA
        else ArtifactType.HUGGINGFACE
    )
    if wire.artifact.type is not expected_artifact:
        raise EmbeddingContractError(
            f"{path}/artifact/type",
            f"backend {wire.backend.type.value!r} requires "
            f"artifact type {expected_artifact.value!r}",
        )

    parameters = thaw_json(wire.backend.parameters)
    parameters = _closed_mapping(
        parameters,
        path=f"{path}/backend/parameters",
        fields=_BACKEND_PARAMETER_FIELDS,
    )
    implementation = _non_empty_string(
        parameters["implementation"],
        f"{path}/backend/parameters/implementation",
    )
    implementation_revision = parameters["implementation_revision"]
    if implementation_revision is not None:
        implementation_revision = _non_empty_string(
            implementation_revision,
            f"{path}/backend/parameters/implementation_revision",
        )
    model_name = _non_empty_string(
        parameters["model_name"],
        f"{path}/backend/parameters/model_name",
    )
    return model_name, {
        "type": wire.backend.type.value,
        "implementation": implementation,
        "implementation_revision": implementation_revision,
    }


def _artifact_projection(wire: WireProfile) -> dict[str, Any]:
    return {
        "repository": wire.artifact.repository,
        "revision": wire.artifact.revision,
        "manifest_digest": wire.artifact.manifest_digest,
        "weights": [item.to_dict() for item in wire.artifact.weights],
        "auxiliary": [item.to_dict() for item in wire.artifact.auxiliary],
    }


def _profile_projection(
    wire: WireProfile,
    *,
    path: str,
) -> tuple[str, dict[str, Any]]:
    if wire.task is not ModelTask.EMBEDDING:
        raise EmbeddingContractError(path, "profile task must be embedding")
    metadata = thaw_json(wire.profile_metadata)
    # Stable OpenAI record metadata is not part of the vector compatibility
    # fingerprint. The existing embedding pipeline projection stays exact.
    if isinstance(metadata, dict) and "created" in metadata:
        created = metadata.pop("created")
        if type(created) is not int or created < 0:
            raise EmbeddingContractError(f"{path}/metadata/created", "must be a nonnegative integer")
    metadata = _closed_mapping(
        metadata,
        path=f"{path}/metadata",
        fields=_PROFILE_METADATA_FIELDS,
    )
    model_name, backend = _backend_projection(wire, path=path)
    request_options = metadata["pipeline"]["parameters"].get("request_options")
    if request_options is not None:
        policy_path = f"{path}/metadata/pipeline/parameters/request_options"
        if backend["type"] != "ollama" or metadata["kind"] != "dense":
            raise EmbeddingContractError(policy_path, "request options require an Ollama dense profile")
        if (not isinstance(request_options, dict)
                or set(request_options) != {"num_ctx", "num_batch"}
                or any(type(v) is not int or v <= 0 for v in request_options.values())):
            raise EmbeddingContractError(policy_path, "must pin positive integer context and batch size")
        limits = metadata["max_input_tokens"]
        if (limits["truncation"] != "none" or limits["overflow"] != "reject"
                or any(v != request_options["num_ctx"] for v in limits["by_role"].values())
                or metadata["pipeline"]["parameters"].get("truncate") is not False):
            raise EmbeddingContractError(policy_path, "request options must match the complete-input limit policy")
        if (metadata["input_type"]["required"] is not True
                or metadata["input_type"]["missing_role_behavior"] != "reject"):
            raise EmbeddingContractError(policy_path, "formatted Ollama requests require explicit roles")
        for role in ("search_document", "search_query"):
            template = metadata["pipeline"]["formatting"][role]["template"]
            if type(template) is not str or template.count("{text}") != 1:
                raise EmbeddingContractError(policy_path, "each role must declare one {text} placeholder")
    profile = {
        "profile_id": wire.profile_id,
        "kind": copy.deepcopy(metadata["kind"]),
        "endpoint": wire.endpoint.value,
        "default_for_endpoint": wire.default_for_endpoint,
        "backend": backend,
        "artifact": _artifact_projection(wire),
        "dimensions": copy.deepcopy(metadata["dimensions"]),
        "max_input_tokens": copy.deepcopy(metadata["max_input_tokens"]),
        "output_normalized": copy.deepcopy(metadata["output_normalized"]),
        "similarity": copy.deepcopy(metadata["similarity"]),
        "input_type": copy.deepcopy(metadata["input_type"]),
        "formatting_owner": copy.deepcopy(metadata["formatting_owner"]),
        "pipeline": copy.deepcopy(metadata["pipeline"]),
        "verification": copy.deepcopy(metadata["verification"]),
        "index_compatibility_id": None,
        "query_compatibility_id": None,
    }
    return model_name, profile


def _single_backend_model_name(
    names: set[str],
    *,
    path: str,
    backend: BackendType,
) -> str | None:
    if not names:
        return None
    if len(names) != 1:
        raise EmbeddingContractError(
            path,
            f"embedding backend {backend.value!r} declares multiple model names: "
            f"{sorted(names)!r}",
        )
    return next(iter(names))


def _build_group(
    *,
    canonical_model_id: str,
    aliases: tuple[str, ...],
    wires: tuple[WireProfile, ...],
) -> "EmbeddingModelGroup":
    profiles: list[dict[str, Any]] = []
    names_by_backend: dict[BackendType, set[str]] = {
        BackendType.KIRON_EMBEDDINGS: set(),
        BackendType.OLLAMA: set(),
    }
    for index, wire in enumerate(wires):
        path = f"/catalog/{canonical_model_id}/profiles/{index}"
        model_name, profile = _profile_projection(wire, path=path)
        names_by_backend[wire.backend.type].add(model_name)
        profiles.append(profile)

    service_model = _single_backend_model_name(
        names_by_backend[BackendType.KIRON_EMBEDDINGS],
        path=f"/catalog/{canonical_model_id}/deployments",
        backend=BackendType.KIRON_EMBEDDINGS,
    )
    ollama_model = _single_backend_model_name(
        names_by_backend[BackendType.OLLAMA],
        path=f"/catalog/{canonical_model_id}/deployments",
        backend=BackendType.OLLAMA,
    )
    capabilities = finalize_capabilities(
        {
            "schema_version": 1,
            "canonical_model_id": canonical_model_id,
            "aliases": sorted(aliases),
            "profiles": sorted(profiles, key=lambda profile: profile["profile_id"]),
        }
    )
    return EmbeddingModelGroup(
        canonical_model_id=canonical_model_id,
        aliases=tuple(sorted(aliases)),
        service_model=service_model,
        ollama_model=ollama_model,
        capabilities=capabilities,
    )


@dataclass(frozen=True, slots=True, init=False)
class EmbeddingModelGroup:
    canonical_model_id: str
    aliases: tuple[str, ...]
    service_model: str | None
    ollama_model: str | None
    _capabilities: FrozenJSONMapping

    def __init__(
        self,
        canonical_model_id: str,
        aliases: tuple[str, ...],
        service_model: str | None,
        ollama_model: str | None,
        capabilities: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "canonical_model_id", canonical_model_id)
        object.__setattr__(self, "aliases", tuple(aliases))
        object.__setattr__(self, "service_model", service_model)
        object.__setattr__(self, "ollama_model", ollama_model)
        object.__setattr__(
            self,
            "_capabilities",
            freeze_json_mapping(
                capabilities,
                path=f"/registry/{canonical_model_id}/kiron_capabilities",
            ),
        )

    @property
    def capabilities(self) -> dict[str, Any]:
        detached = thaw_json(self._capabilities)
        if not isinstance(detached, dict):
            raise AssertionError("capabilities must thaw to an object")
        return detached


class EmbeddingProfileRegistry:
    """Immutable lookup facade over a fully validated explicit registry."""

    __slots__ = (
        "_catalog_digest",
        "_defaults",
        "_groups",
        "_names",
        "_profile_count",
        "_profiles",
        "_sealed",
    )

    def __init__(
        self,
        groups: tuple[EmbeddingModelGroup, ...],
        *,
        catalog_digest: str | None = None,
    ) -> None:
        if not groups:
            raise EmbeddingContractError("/registry", "must contain groups")
        ordered = tuple(sorted(groups, key=lambda item: item.canonical_model_id))
        names: dict[str, EmbeddingModelGroup] = {}
        profiles: dict[str, FrozenJSONMapping] = {}
        defaults: dict[tuple[str, str], FrozenJSONMapping] = {}
        for group_index, group in enumerate(ordered):
            path = f"/registry/groups/{group_index}"
            capabilities = group.capabilities
            validate_capabilities(capabilities)
            if capabilities["canonical_model_id"] != group.canonical_model_id:
                raise EmbeddingContractError(path, "canonical model ID mismatch")
            if tuple(capabilities["aliases"]) != tuple(sorted(group.aliases)):
                raise EmbeddingContractError(path, "alias declaration mismatch")
            declared = (group.canonical_model_id, *group.aliases)
            for name in declared:
                if name in names:
                    raise EmbeddingContractError(
                        f"{path}/aliases", f"duplicate explicit model name {name!r}"
                    )
                names[name] = group
            if group.service_model is not None and group.service_model not in declared:
                raise EmbeddingContractError(
                    path, "service model must be canonical or an alias"
                )
            if group.ollama_model is not None and group.ollama_model not in declared:
                raise EmbeddingContractError(
                    path, "Ollama model must be canonical or an alias"
                )
            for profile in capabilities["profiles"]:
                profile_id = profile["profile_id"]
                if profile_id in profiles:
                    raise EmbeddingContractError(
                        f"{path}/profiles", f"duplicate profile ID {profile_id!r}"
                    )
                frozen = freeze_json_mapping(
                    profile,
                    path=f"{path}/profiles/{profile_id}",
                )
                profiles[profile_id] = frozen
                if profile["default_for_endpoint"]:
                    defaults[
                        (group.canonical_model_id, profile["endpoint"])
                    ] = frozen

        object.__setattr__(self, "_groups", ordered)
        object.__setattr__(self, "_names", MappingProxyType(names))
        object.__setattr__(self, "_profiles", MappingProxyType(profiles))
        object.__setattr__(self, "_defaults", MappingProxyType(defaults))
        object.__setattr__(self, "_profile_count", len(profiles))
        object.__setattr__(self, "_catalog_digest", catalog_digest)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("EmbeddingProfileRegistry is immutable")
        object.__setattr__(self, name, value)

    @property
    def catalog_digest(self) -> str | None:
        return self._catalog_digest

    @property
    def profile_count(self) -> int:
        return self._profile_count

    @property
    def groups(self) -> tuple[EmbeddingModelGroup, ...]:
        return self._groups

    @property
    def service_groups(self) -> tuple[EmbeddingModelGroup, ...]:
        return tuple(group for group in self._groups if group.service_model is not None)

    @property
    def ollama_groups(self) -> tuple[EmbeddingModelGroup, ...]:
        return tuple(group for group in self._groups if group.ollama_model is not None)

    def validate(self) -> None:
        for group in self._groups:
            validate_capabilities(group.capabilities)

    def resolve(self, model_name: object) -> EmbeddingModelGroup | None:
        if type(model_name) is not str:
            return None
        return self._names.get(model_name)

    def require(self, model_name: object) -> EmbeddingModelGroup:
        group = self.resolve(model_name)
        if group is None:
            raise EmbeddingContractError(
                "/model", f"model name is not in the explicit registry: {model_name!r}"
            )
        return group

    def capabilities_for(self, model_name: object) -> dict[str, Any]:
        return self.require(model_name).capabilities

    def default_profile(
        self, model_name: object, endpoint: str
    ) -> dict[str, Any] | None:
        group = self.resolve(model_name)
        if group is None or type(endpoint) is not str:
            return None
        profile = self._defaults.get((group.canonical_model_id, endpoint))
        if profile is None:
            return None
        detached = thaw_json(profile)
        if not isinstance(detached, dict):
            raise AssertionError("profile must thaw to an object")
        return detached

    def profile(self, profile_id: str) -> dict[str, Any] | None:
        if type(profile_id) is not str:
            return None
        profile = self._profiles.get(profile_id)
        if profile is None:
            return None
        detached = thaw_json(profile)
        if not isinstance(detached, dict):
            raise AssertionError("profile must thaw to an object")
        return detached


def build_embedding_registry(catalog: ModelCatalog) -> EmbeddingProfileRegistry:
    """Build the complete embedding view solely from a validated catalog."""

    if not isinstance(catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")
    embedding_wires: dict[str, list[WireProfile]] = {}
    for wire in catalog.for_task(ModelTask.EMBEDDING):
        embedding_wires.setdefault(wire.canonical_model_id, []).append(wire)

    groups = tuple(
        _build_group(
            canonical_model_id=group.canonical_model_id,
            aliases=group.aliases,
            wires=tuple(embedding_wires[group.canonical_model_id]),
        )
        for group in catalog.groups
        if group.canonical_model_id in embedding_wires
    )
    return EmbeddingProfileRegistry(
        groups,
        catalog_digest=catalog.catalog_digest,
    )


MODEL_CATALOG = load_catalog()
EMBEDDING_REGISTRY = build_embedding_registry(MODEL_CATALOG)
MODEL_STATE_VIEW = build_model_state_view(MODEL_CATALOG)


@dataclass(frozen=True)
class InputTypeResolution:
    """Contract-derived decision for one resolved vector-producing profile."""

    accepted: bool
    canonical_model_id: str
    profile_id: str
    canonical_input_type: str | None
    error_code: str | None
    received: object
    supported: tuple[str, ...]
    valid_input_types: tuple[str, ...]


def resolve_profile_input_type(
    model_name: object,
    endpoint: str,
    input_type: object,
    *,
    registry: EmbeddingProfileRegistry = EMBEDDING_REGISTRY,
) -> InputTypeResolution | None:
    """Resolve and validate input_type without contacting a model backend."""

    group = registry.resolve(model_name)
    if group is None:
        return None
    profile = registry.default_profile(model_name, endpoint)
    if profile is None:
        return None

    contract = profile["input_type"]
    aliases = {
        item["alias"]: item["canonical"] for item in contract["aliases"]
    }
    supported = tuple(contract["supported"])
    valid_input_types = tuple(
        sorted(
            set(supported)
            | set(contract["additional_task_roles"])
            | set(aliases)
        )
    )

    error_code: str | None = None
    canonical_input_type: str | None = None
    if input_type is None:
        if contract["required"] or contract["missing_role_behavior"] == "reject":
            error_code = "missing_required_input_type"
        elif contract["missing_role_behavior"] == "use_default":
            canonical_input_type = contract["default"]
    elif not isinstance(input_type, str):
        error_code = "unsupported_input_type"
    elif input_type in aliases:
        canonical_input_type = aliases[input_type]
    elif input_type in supported or input_type in contract["additional_task_roles"]:
        canonical_input_type = input_type
    else:
        error_code = "unsupported_input_type"

    return InputTypeResolution(
        accepted=error_code is None,
        canonical_model_id=group.canonical_model_id,
        profile_id=profile["profile_id"],
        canonical_input_type=canonical_input_type,
        error_code=error_code,
        received=input_type,
        supported=supported,
        valid_input_types=valid_input_types,
    )


def input_type_error_payload(decision: InputTypeResolution) -> dict[str, Any]:
    """Build the stable S1 request-contract error shape."""

    if decision.accepted or decision.error_code is None:
        raise ValueError("accepted input_type decision has no error payload")
    error: dict[str, Any] = {
        "code": decision.error_code,
        "model": decision.canonical_model_id,
        "profile_id": decision.profile_id,
        "field_path": "/input_type",
        "supported": list(decision.supported),
    }
    if decision.error_code == "unsupported_input_type":
        error["received"] = decision.received
    return {
        "error": error,
        "valid_input_types": list(decision.valid_input_types),
    }


def _row_name(row: dict[str, Any], path: str) -> str:
    name = row.get("name", row.get("model"))
    if not isinstance(name, str) or not name:
        raise EmbeddingContractError(path, "model row requires a non-empty name")
    return name


def _known_contract_fields_equal(
    candidate: object,
    reference: object,
    path: tuple[object, ...] = (),
) -> bool:
    """Compare V1-known fields while ignoring schema-open extensions."""

    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            return False
        if path and path[-1] == "parameters" and "pipeline" in path:
            return canonical_hash_json(candidate) == canonical_hash_json(reference)
        return all(
            key in candidate
            and _known_contract_fields_equal(
                candidate[key], item, (*path, key)
            )
            for key, item in reference.items()
        )
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(candidate) != len(reference):
            return False
        return all(
            _known_contract_fields_equal(item, expected, (*path, index))
            for index, (item, expected) in enumerate(zip(candidate, reference))
        )
    return canonical_json(candidate) == canonical_json(reference)


def _capabilities_match_registry(candidate: object, reference: object) -> bool:
    validate_capabilities(candidate)
    validate_capabilities(reference)
    return _known_contract_fields_equal(candidate, reference)


def _index_known_rows(
    rows: object,
    *,
    source: str,
    require_capabilities: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(rows, list):
        raise EmbeddingContractError(f"/{source}/models", "must be an array")
    known: dict[str, dict[str, Any]] = {}
    unknown: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        path = f"/{source}/models/{index}"
        if not isinstance(raw, dict):
            raise EmbeddingContractError(path, "must be an object")
        row = copy.deepcopy(raw)
        name = _row_name(row, path)
        group = EMBEDDING_REGISTRY.resolve(name)
        if group is None:
            if "kiron_capabilities" in row:
                raise EmbeddingContractError(
                    path, "unregistered row must not advertise kiron_capabilities"
                )
            unknown.append(row)
            continue
        if group.canonical_model_id in known:
            raise EmbeddingContractError(
                path, f"duplicate canonical group {group.canonical_model_id!r}"
            )
        supplied = row.get("kiron_capabilities")
        if require_capabilities and supplied is None:
            raise EmbeddingContractError(path, "service row lacks kiron_capabilities")
        if supplied is not None and not _capabilities_match_registry(
            supplied, group.capabilities
        ):
            raise EmbeddingContractError(path, "capability object differs from registry")
        known[group.canonical_model_id] = row
    return known, unknown


def merge_discovery_tags(
    ollama_payload: object,
    service_payload: object,
) -> dict[str, Any]:
    """Merge discovery by canonical group while preserving extension fields."""

    if not isinstance(ollama_payload, dict):
        raise EmbeddingContractError("/ollama", "tags payload must be an object")
    if not isinstance(service_payload, dict):
        raise EmbeddingContractError("/kiron_embeddings", "tags payload must be an object")
    ollama_known, ollama_unknown = _index_known_rows(
        ollama_payload.get("models"), source="ollama", require_capabilities=False
    )
    service_known, service_unknown = _index_known_rows(
        service_payload.get("models"),
        source="kiron_embeddings",
        require_capabilities=True,
    )
    if service_unknown:
        raise EmbeddingContractError(
            "/kiron_embeddings/models", "service advertised an unregistered model"
        )

    for group in EMBEDDING_REGISTRY.service_groups:
        if group.canonical_model_id not in service_known:
            raise EmbeddingContractError(
                "/kiron_embeddings/models",
                f"missing configured service group {group.canonical_model_id!r}",
            )
    for group in EMBEDDING_REGISTRY.ollama_groups:
        if group.canonical_model_id not in ollama_known:
            raise EmbeddingContractError(
                "/ollama/models",
                f"missing configured Ollama group {group.canonical_model_id!r}",
            )

    merged_rows = ollama_unknown
    for group in EMBEDDING_REGISTRY.groups:
        native = ollama_known.get(group.canonical_model_id, {})
        service = service_known.get(group.canonical_model_id, {})
        if not native and not service:
            continue
        merged = copy.deepcopy(native)
        merged.update(copy.deepcopy(service))
        merged["name"] = group.canonical_model_id
        merged["model"] = group.canonical_model_id
        supplied_capabilities = service.get("kiron_capabilities")
        if supplied_capabilities is None:
            supplied_capabilities = native.get("kiron_capabilities")
        if supplied_capabilities is None:
            supplied_capabilities = group.capabilities
        merged["kiron_capabilities"] = copy.deepcopy(supplied_capabilities)
        merged_rows.append(merged)

    result = copy.deepcopy(ollama_payload)
    result["models"] = merged_rows
    return result


def attach_show_capabilities(payload: object, model_name: object) -> dict[str, Any]:
    """Attach and consistency-check the selected canonical group's contract."""

    if not isinstance(payload, dict):
        raise EmbeddingContractError("/show", "payload must be an object")
    group = EMBEDDING_REGISTRY.require(model_name)
    supplied = payload.get("kiron_capabilities")
    if supplied is not None and not _capabilities_match_registry(
        supplied, group.capabilities
    ):
        raise EmbeddingContractError(
            "/show/kiron_capabilities", "capability object differs from registry"
        )
    result = copy.deepcopy(payload)
    result["model"] = group.canonical_model_id
    result["kiron_capabilities"] = copy.deepcopy(
        supplied if supplied is not None else group.capabilities
    )
    return result
