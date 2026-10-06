"""Small immutable values for the dynamic local-model registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType


_ID_PATTERN = re.compile(r"local\.[0-9a-f]{64}\Z")
_MAX_REFERENCE_LENGTH = 4096
_MAX_DISPLAY_LENGTH = 256
_OLLAMA_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OLLAMA_HOST = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?::[0-9]{1,5})?\Z"
)
_MAX_OLLAMA_REFERENCE = 255


def stable_registry_id(runtime_provider: BackendType, reference: str) -> str:
    if not isinstance(runtime_provider, BackendType) or type(reference) is not str:
        raise TypeError("runtime_provider and reference must be normalized")
    preimage = (b"kiron.local-model-registry.identity.v2\x00"
                + runtime_provider.value.encode("ascii") + b"\x00" + reference.encode("utf-8"))
    return "local." + hashlib.sha256(preimage).hexdigest()


def canonical_ollama_reference(value: object) -> str | None:
    """Return the canonical local tag form, without consulting a provider."""

    if type(value) is not str:
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_OLLAMA_REFERENCE
        or not normalized.isascii()
        or "://" in normalized
        or "@" in normalized
        or any(
            character.isspace() or ord(character) < 32
            for character in normalized
        )
    ):
        return None
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if any(_OLLAMA_HOST.fullmatch(segment) is None for segment in parts[:-1]):
        return None
    final = parts[-1]
    if ":" in final:
        name, tag = final.rsplit(":", 1)
    else:
        name, tag = final, "latest"
    if (
        _OLLAMA_SEGMENT.fullmatch(name) is None
        or _OLLAMA_SEGMENT.fullmatch(tag) is None
    ):
        return None
    parts[-1] = f"{name}:{tag}"
    return "/".join(parts)


def _valid_text(value: object, *, maximum: int) -> bool:
    return (
        type(value) is str
        and value == value.strip()
        and 0 < len(value) <= maximum
        and not any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


@dataclass(frozen=True, slots=True)
class LocalLoaderMetadata:
    """Only loader-verified presentation data that the registry consumes."""

    display_name: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is not None and not _valid_text(
            self.display_name,
            maximum=_MAX_DISPLAY_LENGTH,
        ):
            raise ValueError("display_name must be a short printable string")


def _canonical_path(value: object) -> bool:
    return (_valid_text(value, maximum=_MAX_REFERENCE_LENGTH)
            and Path(value).is_absolute() and Path(value) != Path("/")
            and os.path.normpath(value) == value)


def _digest(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True, slots=True)
class LocalArtifactFile:
    reference: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _canonical_path(self.reference) or not _digest(self.sha256):
            raise ValueError("artifact requires a canonical path and SHA256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("artifact size must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"reference": self.reference, "sha256": self.sha256, "size_bytes": self.size_bytes}


HF_LOADERS = frozenset((LoaderType.SENTENCE_TRANSFORMERS, LoaderType.TRANSFORMERS_LAST_TOKEN,
                       LoaderType.COLBERT_XMOD, LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN))


def loader_backend(loader: LoaderType) -> BackendType:
    if loader in (LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN):
        return BackendType.KIRON_DEBERTA
    if loader in HF_LOADERS:
        return BackendType.KIRON_EMBEDDINGS
    if loader is LoaderType.OLLAMA:
        return BackendType.OLLAMA
    if loader is LoaderType.PRISM_GGUF:
        return BackendType.PRISM
    raise ValueError("unsupported loader")


@dataclass(frozen=True, slots=True)
class RegistrationCandidate:
    runtime_provider: BackendType
    artifact_origin: ArtifactType
    artifact_format: ArtifactFormat
    reference: str
    display_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_provider, BackendType):
            raise ValueError("runtime_provider must be a BackendType")
        if not isinstance(self.artifact_origin, ArtifactType) or not isinstance(self.artifact_format, ArtifactFormat):
            raise ValueError("artifact identity enums are invalid")
        if not _valid_text(self.reference, maximum=_MAX_REFERENCE_LENGTH):
            raise ValueError("reference is invalid")
        if not _valid_text(self.display_name, maximum=_MAX_DISPLAY_LENGTH):
            raise ValueError("display_name is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"runtime_provider": self.runtime_provider.value, "artifact_origin": self.artifact_origin.value,
                "artifact_format": self.artifact_format.value, "reference": self.reference,
                "display_name": self.display_name}


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    id: str
    runtime_provider: BackendType
    artifact_origin: ArtifactType
    artifact_format: ArtifactFormat
    reference: str
    display_name: str
    loader: LoaderType
    sha256: str | None
    size_bytes: int | None
    projector: LocalArtifactFile | None
    runtime_profile: str | None
    configuration_fingerprint: str
    capability_fingerprint: str | None
    registered_at: datetime

    def __post_init__(self) -> None:
        if (type(self.registered_at) is not datetime
                or self.registered_at.tzinfo is None
                or self.registered_at.utcoffset() != timedelta(0)
                or self.registered_at < datetime(1970, 1, 1, tzinfo=timezone.utc)):
            raise ValueError("registered_at must be a UTC datetime at or after the Unix epoch")
        object.__setattr__(self, "registered_at", self.registered_at.astimezone(timezone.utc))
        RegistrationCandidate(self.runtime_provider, self.artifact_origin, self.artifact_format,
                              self.reference, self.display_name)
        if not isinstance(self.loader, LoaderType) or loader_backend(self.loader) is not self.runtime_provider:
            raise ValueError("loader and runtime_provider disagree")
        if self.sha256 is not None and not _digest(self.sha256):
            raise ValueError("invalid artifact SHA256")
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes <= 0):
            raise ValueError("invalid artifact size")
        if self.projector is not None and type(self.projector) is not LocalArtifactFile:
            raise ValueError("invalid projector")
        if self.runtime_provider is BackendType.OLLAMA:
            if (self.artifact_origin is not ArtifactType.OLLAMA
                    or self.artifact_format is not ArtifactFormat.OLLAMA_MANIFEST
                    or canonical_ollama_reference(self.reference) != self.reference):
                raise ValueError("invalid Ollama artifact identity")
        elif self.runtime_provider in (BackendType.KIRON_EMBEDDINGS, BackendType.KIRON_DEBERTA):
            if (self.artifact_origin not in (ArtifactType.LOCAL, ArtifactType.HUGGINGFACE)
                    or self.artifact_format is not ArtifactFormat.HF_WEIGHTS
                    or not _canonical_path(self.reference)):
                raise ValueError("invalid local weights identity")
        elif self.runtime_provider is BackendType.PRISM:
            if (self.artifact_origin not in (ArtifactType.LOCAL, ArtifactType.HUGGINGFACE)
                    or self.artifact_format is not ArtifactFormat.GGUF
                    or not _canonical_path(self.reference) or not self.reference.endswith(".gguf")
                    or self.sha256 is None or self.size_bytes is None
                    or type(self.runtime_profile) is not str
                    or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.runtime_profile) is None):
                raise ValueError("GGUF requires explicit file identity and runtime profile")
        else:
            raise ValueError("unsupported runtime provider")
        if self.runtime_provider is not BackendType.PRISM and (self.projector is not None or self.runtime_profile is not None):
            raise ValueError("projector and runtime_profile are only supported for Prism")
        if self.projector is not None and self.projector.reference == self.reference:
            raise ValueError("projector must be a separate artifact")
        if self.id != stable_registry_id(self.runtime_provider, self.reference):
            raise ValueError("id does not match runtime provider and reference")
        if self.configuration_fingerprint != self.compute_fingerprint():
            raise ValueError("configuration fingerprint does not match entry")
        if self.capability_fingerprint is not None and not _digest(self.capability_fingerprint):
            raise ValueError("invalid capability fingerprint")

    def definition(self) -> dict[str, object]:
        return {"runtime_provider": self.runtime_provider.value, "artifact_origin": self.artifact_origin.value,
                "artifact_format": self.artifact_format.value, "reference": self.reference,
                "loader": self.loader.value, "sha256": self.sha256, "size_bytes": self.size_bytes,
                "projector": self.projector.to_dict() if self.projector else None,
                "runtime_profile": self.runtime_profile}

    def compute_fingerprint(self) -> str:
        encoded = json.dumps(self.definition(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(b"kiron.registry.configuration.v2\x00" + encoded).hexdigest()

    @classmethod
    def create(cls, *, runtime_provider: BackendType, artifact_origin: ArtifactType,
               artifact_format: ArtifactFormat, reference: str, display_name: str, loader: LoaderType,
               sha256: str | None = None, size_bytes: int | None = None,
               projector: LocalArtifactFile | None = None, runtime_profile: str | None = None) -> "RegistryEntry":
        values = dict(runtime_provider=runtime_provider, artifact_origin=artifact_origin,
                      artifact_format=artifact_format, reference=reference, display_name=display_name,
                      loader=loader, sha256=sha256, size_bytes=size_bytes, projector=projector,
                      runtime_profile=runtime_profile)
        definition = {key: value.value if isinstance(value, (BackendType, ArtifactType, ArtifactFormat, LoaderType))
                      else value.to_dict() if isinstance(value, LocalArtifactFile) else value
                      for key, value in values.items() if key != "display_name"}
        encoded = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return cls(id=stable_registry_id(runtime_provider, reference), **values,
                   configuration_fingerprint=hashlib.sha256(b"kiron.registry.configuration.v2\x00" + encoded).hexdigest(),
                   capability_fingerprint=None, registered_at=datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, **self.definition(), "display_name": self.display_name,
                "configuration_fingerprint": self.configuration_fingerprint,
                "capability_fingerprint": self.capability_fingerprint,
                "registered_at": self.registered_at.isoformat(timespec="microseconds").replace("+00:00", "Z")}


@dataclass(frozen=True, slots=True)
class ValidatedLocalModel:
    runtime_provider: BackendType
    artifact_origin: ArtifactType
    artifact_format: ArtifactFormat
    reference: str
    display_name: str
    loader: LoaderType
    sha256: str | None = None
    size_bytes: int | None = None
    projector: LocalArtifactFile | None = None
    runtime_profile: str | None = None

    def to_entry(self) -> RegistryEntry:
        from dataclasses import asdict
        values = asdict(self)
        values["projector"] = self.projector
        return RegistryEntry.create(**values)
