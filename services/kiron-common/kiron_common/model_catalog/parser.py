"""Closed V1 manifest parser and per-model invariant validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeVar

from ._json import freeze_json_mapping
from .errors import CatalogValidationError
from .models import (
    Artifact,
    ArtifactFile,
    ArtifactType,
    Backend,
    BackendType,
    Deployment,
    Loader,
    LoaderType,
    ModelEndpoint,
    ModelGroup,
    ModelTask,
    Profile,
    REQUEST_DEFAULT_ENDPOINTS,
    RequestDefault,
    Route,
)


SCHEMA_VERSION = 1

_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")

_ENDPOINTS_FOR_TASK = {
    ModelTask.EMBEDDING: frozenset(
        (
            ModelEndpoint.EMBED,
            ModelEndpoint.EMBED_LATE,
            ModelEndpoint.EMBED_COLBERT,
        )
    ),
    ModelTask.RERANK: frozenset((ModelEndpoint.RERANK,)),
    ModelTask.NLI: frozenset((ModelEndpoint.SCORE,)),
}

EnumT = TypeVar("EnumT")


def _fail(path: str, detail: str, source: str | None) -> None:
    raise CatalogValidationError(path, detail, source=source)


def _object(
    value: object,
    *,
    path: str,
    required: tuple[str, ...],
    source: str | None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object", source)
    if any(type(key) is not str for key in value):
        _fail(path, "object names must be strings", source)
    for key in value:
        _validate_unicode_scalar(key, path, source)
    keys = set(value)
    expected = set(required)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}", source)
    if unknown:
        _fail(path, f"unknown fields: {', '.join(unknown)}", source)
    return value


def _array(value: object, *, path: str, source: str | None) -> list[object]:
    if not isinstance(value, list):
        _fail(path, "must be an array", source)
    return value


def _literal_string(
    value: object,
    *,
    path: str,
    source: str | None,
) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a non-empty string", source)
    _validate_unicode_scalar(value, path, source)
    if value != value.strip():
        _fail(path, "must not contain leading or trailing whitespace", source)
    if any(ord(character) < 0x20 for character in value):
        _fail(path, "must not contain control characters", source)
    return value


def _validate_unicode_scalar(
    value: str,
    path: str,
    source: str | None,
) -> None:
    for code_point in map(ord, value):
        if 0xD800 <= code_point <= 0xDFFF:
            _fail(
                path,
                f"contains a lone Unicode surrogate U+{code_point:04X}",
                source,
            )


def _stable_id(
    value: object,
    *,
    path: str,
    source: str | None,
) -> str:
    identifier = _literal_string(value, path=path, source=source)
    if not _STABLE_ID.fullmatch(identifier):
        _fail(
            path,
            "must match ^[a-z0-9][a-z0-9._-]*$",
            source,
        )
    return identifier


def _enum(
    value: object,
    enum_type: type[EnumT],
    *,
    path: str,
    source: str | None,
) -> EnumT:
    if type(value) is not str:
        _fail(path, "must be a string enum value", source)
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(item.value for item in enum_type)  # type: ignore[attr-defined]
        _fail(path, f"unknown value {value!r}; allowed: {allowed}", source)
    raise AssertionError("_fail must raise")


def _boolean(
    value: object,
    *,
    path: str,
    source: str | None,
) -> bool:
    if type(value) is not bool:
        _fail(path, "must be a boolean", source)
    return value


def _artifact_files(
    value: object,
    *,
    path: str,
    source: str | None,
) -> tuple[ArtifactFile, ...]:
    raw_items = _array(value, path=path, source=source)
    items: list[ArtifactFile] = []
    paths: set[str] = set()
    for index, raw in enumerate(raw_items):
        item_path = f"{path}/{index}"
        item = _object(
            raw,
            path=item_path,
            required=("path", "sha256"),
            source=source,
        )
        artifact_path = _literal_string(
            item["path"], path=f"{item_path}/path", source=source
        )
        digest = item["sha256"]
        if type(digest) is not str or not _SHA256_HEX.fullmatch(digest):
            _fail(
                f"{item_path}/sha256",
                "must be 64 lowercase hexadecimal characters",
                source,
            )
        if artifact_path in paths:
            _fail(path, f"duplicate artifact path {artifact_path!r}", source)
        paths.add(artifact_path)
        items.append(ArtifactFile(path=artifact_path, sha256=digest))
    return tuple(sorted(items, key=lambda item: item.path))


def _parse_artifact(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Artifact:
    artifact = _object(
        value,
        path=path,
        required=(
            "type",
            "repository",
            "revision",
            "manifest_digest",
            "trust_remote_code",
            "weights",
            "auxiliary",
            "metadata",
        ),
        source=source,
    )
    artifact_type = _enum(
        artifact["type"], ArtifactType, path=f"{path}/type", source=source
    )
    repository = artifact["repository"]
    revision = artifact["revision"]
    manifest_digest = artifact["manifest_digest"]
    if repository is not None:
        repository = _literal_string(
            repository, path=f"{path}/repository", source=source
        )
    if revision is not None:
        revision = _literal_string(
            revision, path=f"{path}/revision", source=source
        )
    if manifest_digest is not None and (
        type(manifest_digest) is not str
        or not _SHA256_ID.fullmatch(manifest_digest)
    ):
        _fail(
            f"{path}/manifest_digest",
            "must be a sha256: URI with a lowercase digest or null",
            source,
        )
    trust_remote_code = _boolean(
        artifact["trust_remote_code"],
        path=f"{path}/trust_remote_code",
        source=source,
    )
    weights = _artifact_files(
        artifact["weights"], path=f"{path}/weights", source=source
    )
    auxiliary = _artifact_files(
        artifact["auxiliary"], path=f"{path}/auxiliary", source=source
    )
    overlap = {item.path for item in weights}.intersection(
        item.path for item in auxiliary
    )
    if overlap:
        _fail(
            path,
            f"weight and auxiliary paths overlap: {sorted(overlap)!r}",
            source,
        )

    immutable_revision = (
        isinstance(revision, str) and _HF_REVISION.fullmatch(revision) is not None
    )
    if trust_remote_code and not immutable_revision:
        _fail(
            f"{path}/trust_remote_code",
            "trust_remote_code requires an immutable 40-character revision",
            source,
        )
    if artifact_type is ArtifactType.HUGGINGFACE:
        if not isinstance(repository, str):
            _fail(
                f"{path}/repository",
                "HuggingFace artifacts require a repository",
                source,
            )
        if not immutable_revision:
            _fail(
                f"{path}/revision",
                "HuggingFace artifacts require an immutable 40-character lowercase revision",
                source,
            )
        if manifest_digest is not None:
            _fail(
                f"{path}/manifest_digest",
                "HuggingFace artifacts use repository plus revision, not a manifest digest",
                source,
            )
        if not weights:
            _fail(f"{path}/weights", "must contain at least one weight", source)
    else:
        if repository is not None or revision is not None:
            _fail(
                path,
                "Ollama artifacts must use null repository and revision",
                source,
            )
        if manifest_digest is None:
            _fail(
                f"{path}/manifest_digest",
                "Ollama artifacts require an immutable manifest digest",
                source,
            )
        if trust_remote_code:
            _fail(
                f"{path}/trust_remote_code",
                "is only meaningful for HuggingFace artifacts",
                source,
            )
        if not weights:
            _fail(f"{path}/weights", "must contain at least one model blob", source)

    metadata = freeze_json_mapping(
        artifact["metadata"], path=f"{path}/metadata", source=source
    )
    return Artifact(
        type=artifact_type,
        repository=repository,
        revision=revision,
        manifest_digest=manifest_digest,
        trust_remote_code=trust_remote_code,
        weights=weights,
        auxiliary=auxiliary,
        metadata=metadata,
    )


def _parse_backend(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Backend:
    backend = _object(
        value,
        path=path,
        required=("type", "parameters"),
        source=source,
    )
    return Backend(
        type=_enum(
            backend["type"], BackendType, path=f"{path}/type", source=source
        ),
        parameters=freeze_json_mapping(
            backend["parameters"],
            path=f"{path}/parameters",
            source=source,
        ),
    )


def _parse_loader(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Loader:
    loader = _object(
        value,
        path=path,
        required=("type", "parameters"),
        source=source,
    )
    return Loader(
        type=_enum(
            loader["type"], LoaderType, path=f"{path}/type", source=source
        ),
        parameters=freeze_json_mapping(
            loader["parameters"],
            path=f"{path}/parameters",
            source=source,
        ),
    )


def _parse_route(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Route:
    raw = _object(
        value,
        path=path,
        required=("task", "endpoint"),
        source=source,
    )
    task = _enum(raw["task"], ModelTask, path=f"{path}/task", source=source)
    endpoint = _enum(
        raw["endpoint"], ModelEndpoint, path=f"{path}/endpoint", source=source
    )
    allowed = _ENDPOINTS_FOR_TASK[task]
    if endpoint not in allowed:
        rendered = ", ".join(sorted(item.value for item in allowed))
        _fail(
            path,
            f"task {task.value!r} allows only endpoints: {rendered}",
            source,
        )
    return Route(task=task, endpoint=endpoint)


def _parse_deployment(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Deployment:
    raw = _object(
        value,
        path=path,
        required=("id", "backend", "artifact", "routes", "loader", "metadata"),
        source=source,
    )
    deployment_id = _stable_id(raw["id"], path=f"{path}/id", source=source)
    backend = _parse_backend(raw["backend"], path=f"{path}/backend", source=source)
    artifact = _parse_artifact(
        raw["artifact"], path=f"{path}/artifact", source=source
    )
    loader = _parse_loader(raw["loader"], path=f"{path}/loader", source=source)
    raw_routes = _array(raw["routes"], path=f"{path}/routes", source=source)
    if not raw_routes:
        _fail(f"{path}/routes", "must contain at least one route", source)
    routes = tuple(
        _parse_route(item, path=f"{path}/routes/{index}", source=source)
        for index, item in enumerate(raw_routes)
    )
    if len(routes) != len(set(routes)):
        _fail(f"{path}/routes", "must not contain duplicate routes", source)

    if backend.type is BackendType.OLLAMA:
        if artifact.type is not ArtifactType.OLLAMA:
            _fail(
                f"{path}/artifact/type",
                "the Ollama backend requires an Ollama artifact",
                source,
            )
        if loader.type is not LoaderType.OLLAMA:
            _fail(
                f"{path}/loader/type",
                "the Ollama backend requires the ollama loader ID",
                source,
            )
    else:
        if artifact.type is not ArtifactType.HUGGINGFACE:
            _fail(
                f"{path}/artifact/type",
                "KIron service backends require a HuggingFace artifact",
                source,
            )
        if loader.type is LoaderType.OLLAMA:
            _fail(
                f"{path}/loader/type",
                "the ollama loader ID is restricted to the Ollama backend",
                source,
            )

    return Deployment(
        id=deployment_id,
        backend=backend,
        artifact=artifact,
        routes=routes,
        loader=loader,
        metadata=freeze_json_mapping(
            raw["metadata"], path=f"{path}/metadata", source=source
        ),
    )


def _parse_profile(
    value: object,
    *,
    path: str,
    source: str | None,
) -> Profile:
    raw = _object(
        value,
        path=path,
        required=(
            "id",
            "deployment_id",
            "task",
            "endpoint",
            "default_for_endpoint",
            "metadata",
        ),
        source=source,
    )
    task = _enum(raw["task"], ModelTask, path=f"{path}/task", source=source)
    endpoint = _enum(
        raw["endpoint"], ModelEndpoint, path=f"{path}/endpoint", source=source
    )
    allowed = _ENDPOINTS_FOR_TASK[task]
    if endpoint not in allowed:
        rendered = ", ".join(sorted(item.value for item in allowed))
        _fail(
            path,
            f"task {task.value!r} allows only endpoints: {rendered}",
            source,
        )
    return Profile(
        id=_stable_id(raw["id"], path=f"{path}/id", source=source),
        deployment_id=_stable_id(
            raw["deployment_id"], path=f"{path}/deployment_id", source=source
        ),
        task=task,
        endpoint=endpoint,
        default_for_endpoint=_boolean(
            raw["default_for_endpoint"],
            path=f"{path}/default_for_endpoint",
            source=source,
        ),
        metadata=freeze_json_mapping(
            raw["metadata"], path=f"{path}/metadata", source=source
        ),
    )


def _parse_request_default(
    value: object,
    *,
    path: str,
    source: str | None,
) -> RequestDefault:
    raw = _object(
        value,
        path=path,
        required=("endpoint", "profile_id"),
        source=source,
    )
    endpoint = _enum(
        raw["endpoint"], ModelEndpoint, path=f"{path}/endpoint", source=source
    )
    if endpoint not in REQUEST_DEFAULT_ENDPOINTS:
        rendered = ", ".join(
            sorted(item.value for item in REQUEST_DEFAULT_ENDPOINTS)
        )
        _fail(
            f"{path}/endpoint",
            f"request defaults are allowed only for endpoints: {rendered}",
            source,
        )
    return RequestDefault(
        endpoint=endpoint,
        profile_id=_stable_id(
            raw["profile_id"], path=f"{path}/profile_id", source=source
        ),
    )


def parse_manifest(
    manifest: object,
    *,
    source: str | None = None,
) -> ModelGroup:
    """Parse one closed V1 document representing one canonical model group."""

    raw = _object(
        manifest,
        path="/",
        required=(
            "schema_version",
            "canonical_model_id",
            "aliases",
            "deployments",
            "profiles",
            "request_defaults",
            "metadata",
        ),
        source=source,
    )
    schema_version = raw["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _fail(
            "/schema_version",
            f"unsupported schema version {schema_version!r}; expected {SCHEMA_VERSION}",
            source,
        )
    canonical_model_id = _literal_string(
        raw["canonical_model_id"],
        path="/canonical_model_id",
        source=source,
    )
    raw_aliases = _array(raw["aliases"], path="/aliases", source=source)
    aliases = tuple(
        _literal_string(item, path=f"/aliases/{index}", source=source)
        for index, item in enumerate(raw_aliases)
    )
    if len(aliases) != len(set(aliases)):
        _fail("/aliases", "must not contain duplicates", source)
    if canonical_model_id in aliases:
        _fail("/aliases", "must not repeat the canonical model ID", source)

    raw_deployments = _array(
        raw["deployments"], path="/deployments", source=source
    )
    if not raw_deployments:
        _fail("/deployments", "must contain at least one deployment", source)
    deployments = tuple(
        _parse_deployment(
            item,
            path=f"/deployments/{index}",
            source=source,
        )
        for index, item in enumerate(raw_deployments)
    )
    deployment_ids = [item.id for item in deployments]
    if len(deployment_ids) != len(set(deployment_ids)):
        _fail("/deployments", "contains duplicate deployment IDs", source)

    raw_profiles = _array(raw["profiles"], path="/profiles", source=source)
    if not raw_profiles:
        _fail("/profiles", "must contain at least one profile", source)
    profiles = tuple(
        _parse_profile(item, path=f"/profiles/{index}", source=source)
        for index, item in enumerate(raw_profiles)
    )
    profile_ids = [item.id for item in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        _fail("/profiles", "contains duplicate profile IDs", source)

    raw_request_defaults = _array(
        raw["request_defaults"], path="/request_defaults", source=source
    )
    request_defaults = tuple(
        _parse_request_default(
            item,
            path=f"/request_defaults/{index}",
            source=source,
        )
        for index, item in enumerate(raw_request_defaults)
    )
    request_default_endpoints = [item.endpoint for item in request_defaults]
    if len(request_default_endpoints) != len(set(request_default_endpoints)):
        _fail(
            "/request_defaults",
            "contains multiple request defaults for the same endpoint",
            source,
        )

    deployment_by_id = {item.id: item for item in deployments}
    routes_in_use: dict[str, set[Route]] = {
        deployment.id: set() for deployment in deployments
    }
    for index, profile in enumerate(profiles):
        deployment = deployment_by_id.get(profile.deployment_id)
        if deployment is None:
            _fail(
                f"/profiles/{index}/deployment_id",
                f"references unknown deployment {profile.deployment_id!r}",
                source,
            )
        if profile.route not in deployment.routes:
            _fail(
                f"/profiles/{index}",
                f"route is not declared by deployment {profile.deployment_id!r}",
                source,
            )
        routes_in_use[deployment.id].add(profile.route)

    for deployment_index, deployment in enumerate(deployments):
        unused = set(deployment.routes) - routes_in_use[deployment.id]
        if unused:
            rendered = sorted(
                f"{route.task.value}:{route.endpoint.value}" for route in unused
            )
            _fail(
                f"/deployments/{deployment_index}/routes",
                f"contains routes without profiles: {rendered!r}",
                source,
            )

    endpoints = {profile.endpoint for profile in profiles}
    for endpoint in endpoints:
        defaults = [
            profile
            for profile in profiles
            if profile.endpoint is endpoint and profile.default_for_endpoint
        ]
        if len(defaults) != 1:
            _fail(
                "/profiles",
                f"endpoint {endpoint.value!r} requires exactly one default; found {len(defaults)}",
                source,
            )

    profile_by_id = {profile.id: profile for profile in profiles}
    for index, request_default in enumerate(request_defaults):
        profile = profile_by_id.get(request_default.profile_id)
        if profile is None:
            _fail(
                f"/request_defaults/{index}/profile_id",
                f"references unknown profile {request_default.profile_id!r}",
                source,
            )
        if profile.endpoint is not request_default.endpoint:
            _fail(
                f"/request_defaults/{index}",
                f"profile {profile.id!r} serves endpoint "
                f"{profile.endpoint.value!r}, not "
                f"{request_default.endpoint.value!r}",
                source,
            )

    return ModelGroup(
        canonical_model_id=canonical_model_id,
        aliases=aliases,
        deployments=deployments,
        profiles=profiles,
        request_defaults=request_defaults,
        metadata=freeze_json_mapping(
            raw["metadata"], path="/metadata", source=source
        ),
    )
