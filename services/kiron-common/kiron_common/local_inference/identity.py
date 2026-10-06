"""Immutable identities, measured resource profiles and resolver values."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from enum import Enum
from pathlib import PurePosixPath

from kiron_common.model_catalog import ArtifactFile, ArtifactFormat, ArtifactType, BackendType, LoaderType, ModelEndpoint, ModelTask

from ._values import fingerprint, integer, json_mapping, sha256, text


class ModelSource(str, Enum):
    CATALOG = "catalog"
    LOCAL_REGISTRY = "local_registry"


class EmbeddingRole(str, Enum):
    QUERY = "search_query"
    DOCUMENT = "search_document"


@dataclass(frozen=True, slots=True)
class ArtifactFileReference:
    reference: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        text(self.reference, "artifact reference")
        path = PurePosixPath(self.reference)
        if not path.is_absolute() or ".." in path.parts or str(path) != self.reference:
            raise ValueError("artifact reference must be a canonical absolute path")
        sha256(self.sha256, "artifact sha256")
        integer(self.size_bytes, "artifact size", 1)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    origin: ArtifactType
    format: ArtifactFormat
    sha256: str | None = None
    size_bytes: int | None = None
    projector: ArtifactFileReference | None = None
    repository: str | None = None
    revision: str | None = None
    manifest_digest: str | None = None
    files: tuple[ArtifactFile, ...] = ()

    def __post_init__(self):
        if not isinstance(self.origin, ArtifactType) or not isinstance(self.format, ArtifactFormat):
            raise TypeError("artifact origin/format require catalog enums")
        if self.sha256 is not None:
            sha256(self.sha256, "artifact sha256")
        if self.size_bytes is not None:
            integer(self.size_bytes, "artifact size", 1)
        if self.projector is not None and not isinstance(self.projector, ArtifactFileReference):
            raise TypeError("projector must be an ArtifactFileReference")
        for name in ("repository", "revision", "manifest_digest"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)
        files = tuple(self.files)
        if any(not isinstance(item, ArtifactFile) for item in files):
            raise TypeError("artifact files must be ArtifactFile values")
        if len({item.path for item in files}) != len(files):
            raise ValueError("duplicate artifact file path")
        for item in files:
            text(item.path, "artifact file path")
            sha256(item.sha256, "artifact file sha256")
            if item.size_bytes is not None:
                integer(item.size_bytes, "artifact file size", 1)
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda f: f.path)))
        if self.format is ArtifactFormat.GGUF and (self.sha256 is None or self.size_bytes is None):
            raise ValueError("GGUF requires a measured digest and size")

    @property
    def fingerprint(self) -> str:
        return fingerprint("kiron.local-inference.artifact.v1", self)

    @property
    def content_key(self) -> tuple | None:
        """Only measured content identity may merge a local registration."""
        projector = None if self.projector is None else (self.projector.sha256, self.projector.size_bytes)
        if self.sha256 is not None:
            return (self.format.value, self.sha256, self.size_bytes, projector)
        if self.manifest_digest:
            return (self.format.value, self.manifest_digest, projector)
        if self.files:
            return (self.format.value, tuple((f.path, f.sha256, f.size_bytes) for f in self.files), projector)
        return None


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Explicit measured configuration; construction does not grant admission."""

    id: str
    context_tokens: int
    batch_size: int
    ubatch_size: int
    parallel_slots: int
    threads: int
    gpu_layers: int
    projector_on_gpu: bool
    gpu_memory_bytes: int
    host_memory_bytes: int
    memory_headroom_bytes: int = 0

    def __post_init__(self):
        text(self.id, "resource profile ID")
        for name in ("context_tokens", "batch_size", "ubatch_size", "parallel_slots", "threads"):
            integer(getattr(self, name), name, 1)
        integer(self.gpu_layers, "gpu_layers")
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size exceeds batch_size")
        if type(self.projector_on_gpu) is not bool:
            raise TypeError("projector_on_gpu must be bool")
        for name in ("gpu_memory_bytes", "host_memory_bytes", "memory_headroom_bytes"):
            integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ResolvedDeployment:
    id: str
    provider: BackendType
    reference: str
    artifact_identity: ArtifactIdentity
    loader: LoaderType
    resource_profile: ResourceProfile | None
    configuration_fingerprint: str
    source: ModelSource = ModelSource.CATALOG
    registry_ids: tuple[str, ...] = ()
    backend_parameters: Mapping = field(default_factory=dict)
    loader_parameters: Mapping = field(default_factory=dict)
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ("id", "reference"):
            text(getattr(self, name), name)
        sha256(self.configuration_fingerprint, "configuration fingerprint")
        if not isinstance(self.provider, BackendType) or not isinstance(self.loader, LoaderType):
            raise TypeError("provider/loader require catalog enums")
        if not isinstance(self.artifact_identity, ArtifactIdentity) or not isinstance(self.source, ModelSource):
            raise TypeError("invalid artifact identity or model source")
        if self.resource_profile is not None and not isinstance(self.resource_profile, ResourceProfile):
            raise TypeError("invalid resource profile")
        if self.provider is BackendType.PRISM and self.resource_profile is None:
            raise ValueError("Prism deployment requires an explicit resource profile")
        object.__setattr__(self, "registry_ids", tuple(sorted(set(self.registry_ids))))
        for name in ("backend_parameters", "loader_parameters", "metadata"):
            object.__setattr__(self, name, json_mapping(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    public_model_id: str
    deployment: ResolvedDeployment
    profile_id: str | None
    embedding_role: EmbeddingRole | None
    snapshot_revision: str
    task: ModelTask | None = None
    endpoint: ModelEndpoint | None = None
    profile_metadata: Mapping = field(default_factory=dict)
    created: int | None = None
    canonical_model_id: str | None = None

    def __post_init__(self):
        text(self.public_model_id, "public model ID")
        if not isinstance(self.deployment, ResolvedDeployment):
            raise TypeError("model requires a resolved deployment")
        if self.profile_id is not None:
            text(self.profile_id, "profile ID")
        if self.embedding_role is not None and not isinstance(self.embedding_role, EmbeddingRole):
            raise TypeError("embedding role must be explicit")
        if self.embedding_role is not None and self.profile_id is None:
            raise ValueError("embedding role requires a profile")
        sha256(self.snapshot_revision, "snapshot revision")
        if self.task is not None and not isinstance(self.task, ModelTask):
            raise TypeError("invalid model task")
        if self.endpoint is not None and not isinstance(self.endpoint, ModelEndpoint):
            raise TypeError("invalid model endpoint")
        if self.created is not None:
            integer(self.created, "model creation timestamp")
        if self.canonical_model_id is not None:
            text(self.canonical_model_id, "canonical model ID")
        object.__setattr__(self, "profile_metadata", json_mapping(self.profile_metadata))

    @property
    def api_model_id(self) -> str:
        return self.canonical_model_id or self.public_model_id
