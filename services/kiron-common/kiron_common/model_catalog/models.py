"""Strict, frozen data models for KIron's normalized model manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ._json import FrozenJSONMapping, freeze_json_mapping, thaw_json


class BackendType(str, Enum):
    KIRON_EMBEDDINGS = "kiron_embeddings"
    KIRON_DEBERTA = "kiron_deberta"
    OLLAMA = "ollama"


class ArtifactType(str, Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class ModelTask(str, Enum):
    EMBEDDING = "embedding"
    RERANK = "rerank"
    NLI = "nli"


class ModelEndpoint(str, Enum):
    EMBED = "/api/embed"
    EMBED_LATE = "/api/embed_late"
    EMBED_COLBERT = "/api/embed_colbert"
    RERANK = "/api/rerank"
    SCORE = "/api/score"


REQUEST_DEFAULT_ENDPOINTS = frozenset(
    (ModelEndpoint.RERANK, ModelEndpoint.SCORE)
)


class LoaderType(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    TRANSFORMERS_LAST_TOKEN = "transformers_last_token"
    COLBERT_XMOD = "colbert_xmod"
    CROSS_ENCODER = "cross_encoder"
    MANKEI_LAST_TOKEN = "mankei_last_token"
    OLLAMA = "ollama"


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class Artifact:
    type: ArtifactType
    repository: str | None
    revision: str | None
    manifest_digest: str | None
    trust_remote_code: bool
    weights: tuple[ArtifactFile, ...]
    auxiliary: tuple[ArtifactFile, ...]
    metadata: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, path="/artifact/metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "repository": self.repository,
            "revision": self.revision,
            "manifest_digest": self.manifest_digest,
            "trust_remote_code": self.trust_remote_code,
            "weights": [item.to_dict() for item in self.weights],
            "auxiliary": [item.to_dict() for item in self.auxiliary],
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Backend:
    type: BackendType
    parameters: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            freeze_json_mapping(self.parameters, path="/backend/parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "parameters": thaw_json(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class Loader:
    type: LoaderType
    parameters: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            freeze_json_mapping(self.parameters, path="/loader/parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "parameters": thaw_json(self.parameters),
        }


@dataclass(frozen=True, slots=True, order=True)
class Route:
    task: ModelTask
    endpoint: ModelEndpoint

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task.value, "endpoint": self.endpoint.value}


@dataclass(frozen=True, slots=True)
class Deployment:
    id: str
    backend: Backend
    artifact: Artifact
    routes: tuple[Route, ...]
    loader: Loader
    metadata: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(sorted(self.routes)))
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, path="/deployment/metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend.to_dict(),
            "artifact": self.artifact.to_dict(),
            "routes": [route.to_dict() for route in self.routes],
            "loader": self.loader.to_dict(),
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Profile:
    id: str
    deployment_id: str
    task: ModelTask
    endpoint: ModelEndpoint
    default_for_endpoint: bool
    metadata: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, path="/profile/metadata"),
        )

    @property
    def route(self) -> Route:
        return Route(task=self.task, endpoint=self.endpoint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "deployment_id": self.deployment_id,
            "task": self.task.value,
            "endpoint": self.endpoint.value,
            "default_for_endpoint": self.default_for_endpoint,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RequestDefault:
    """Catalog-wide request default declared by one model group."""

    endpoint: ModelEndpoint
    profile_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "endpoint": self.endpoint.value,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True, slots=True)
class ModelGroup:
    canonical_model_id: str
    aliases: tuple[str, ...]
    deployments: tuple[Deployment, ...]
    profiles: tuple[Profile, ...]
    request_defaults: tuple[RequestDefault, ...]
    metadata: FrozenJSONMapping = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(sorted(self.aliases)))
        object.__setattr__(
            self,
            "deployments",
            tuple(sorted(self.deployments, key=lambda item: item.id)),
        )
        object.__setattr__(
            self,
            "profiles",
            tuple(sorted(self.profiles, key=lambda item: item.id)),
        )
        object.__setattr__(
            self,
            "request_defaults",
            tuple(
                sorted(
                    self.request_defaults,
                    key=lambda item: (item.endpoint.value, item.profile_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, path="/model/metadata"),
        )

    def to_manifest_dict(self, *, schema_version: int) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "canonical_model_id": self.canonical_model_id,
            "aliases": list(self.aliases),
            "deployments": [item.to_dict() for item in self.deployments],
            "profiles": [item.to_dict() for item in self.profiles],
            "request_defaults": [
                item.to_dict() for item in self.request_defaults
            ],
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class WireProfile:
    """Expanded immutable profile used by service-specific catalog views."""

    canonical_model_id: str
    aliases: tuple[str, ...]
    profile_id: str
    deployment_id: str
    task: ModelTask
    endpoint: ModelEndpoint
    default_for_endpoint: bool
    is_request_default: bool
    backend: Backend
    artifact: Artifact
    loader: Loader
    model_metadata: FrozenJSONMapping
    deployment_metadata: FrozenJSONMapping
    profile_metadata: FrozenJSONMapping

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(
            self,
            "model_metadata",
            freeze_json_mapping(self.model_metadata, path="/wire/metadata/model"),
        )
        object.__setattr__(
            self,
            "deployment_metadata",
            freeze_json_mapping(
                self.deployment_metadata,
                path="/wire/metadata/deployment",
            ),
        )
        object.__setattr__(
            self,
            "profile_metadata",
            freeze_json_mapping(
                self.profile_metadata,
                path="/wire/metadata/profile",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_model_id": self.canonical_model_id,
            "aliases": list(self.aliases),
            "profile_id": self.profile_id,
            "deployment_id": self.deployment_id,
            "task": self.task.value,
            "endpoint": self.endpoint.value,
            "default_for_endpoint": self.default_for_endpoint,
            "is_request_default": self.is_request_default,
            "backend": self.backend.to_dict(),
            "artifact": self.artifact.to_dict(),
            "loader": self.loader.to_dict(),
            "metadata": {
                "model": thaw_json(self.model_metadata),
                "deployment": thaw_json(self.deployment_metadata),
                "profile": thaw_json(self.profile_metadata),
            },
        }
