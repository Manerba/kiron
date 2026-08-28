"""Small immutable values for the dynamic local-model registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import re

from kiron_common.model_catalog import LoaderType


_ID_PATTERN = re.compile(r"local\.[0-9a-f]{64}\Z")
_MAX_REFERENCE_LENGTH = 4096
_MAX_DISPLAY_LENGTH = 256
_OLLAMA_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OLLAMA_HOST = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?::[0-9]{1,5})?\Z"
)
_MAX_OLLAMA_REFERENCE = 255


class LocalModelProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


def stable_registry_id(provider: LocalModelProvider, reference: str) -> str:
    if not isinstance(provider, LocalModelProvider) or type(reference) is not str:
        raise TypeError("provider and reference must be normalized")
    preimage = (
        b"kiron.local-model-registry.identity.v1\x00"
        + provider.value.encode("ascii")
        + b"\x00"
        + reference.encode("utf-8")
    )
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


@dataclass(frozen=True, slots=True)
class ValidatedLocalModel:
    provider: LocalModelProvider
    reference: str
    display_name: str
    loader: LoaderType

    def to_entry(self) -> "RegistryEntry":
        return RegistryEntry.create(
            provider=self.provider,
            reference=self.reference,
            display_name=self.display_name,
            loader=self.loader,
        )


@dataclass(frozen=True, slots=True)
class RegistrationCandidate:
    """One already-local model that is not present in the runtime registry."""

    provider: LocalModelProvider
    reference: str
    display_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, LocalModelProvider):
            raise ValueError("provider must be a LocalModelProvider")
        if not _valid_text(self.reference, maximum=_MAX_REFERENCE_LENGTH):
            raise ValueError("reference is invalid")
        if not _valid_text(self.display_name, maximum=_MAX_DISPLAY_LENGTH):
            raise ValueError("display_name is invalid")
        if (
            self.provider is LocalModelProvider.OLLAMA
            and canonical_ollama_reference(self.reference) != self.reference
        ):
            raise ValueError("Ollama reference is not canonical")

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider.value,
            "reference": self.reference,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """One minimal, durable dynamic overlay entry."""

    id: str
    provider: LocalModelProvider
    reference: str
    display_name: str
    loader: LoaderType

    def __post_init__(self) -> None:
        if not isinstance(self.provider, LocalModelProvider):
            raise ValueError("provider must be a LocalModelProvider")
        if not isinstance(self.loader, LoaderType):
            raise ValueError("loader must be a LoaderType")
        if type(self.id) is not str or _ID_PATTERN.fullmatch(self.id) is None:
            raise ValueError("id has an invalid format")
        if not _valid_text(self.reference, maximum=_MAX_REFERENCE_LENGTH):
            raise ValueError("reference is invalid")
        if not _valid_text(self.display_name, maximum=_MAX_DISPLAY_LENGTH):
            raise ValueError("display_name is invalid")
        if self.provider is LocalModelProvider.OLLAMA:
            if self.loader is not LoaderType.OLLAMA:
                raise ValueError("Ollama entries require the Ollama loader")
            if canonical_ollama_reference(self.reference) != self.reference:
                raise ValueError("Ollama reference is not canonical")
        else:
            path = Path(self.reference)
            if (
                self.loader is LoaderType.OLLAMA
                or not path.is_absolute()
                or path == Path(path.anchor)
                or os.path.normpath(self.reference) != self.reference
            ):
                raise ValueError(
                    "Hugging Face entries require a canonical local loader path"
                )
        if self.id != stable_registry_id(self.provider, self.reference):
            raise ValueError("id does not match provider and reference")

    @classmethod
    def create(
        cls,
        *,
        provider: LocalModelProvider,
        reference: str,
        display_name: str,
        loader: LoaderType,
    ) -> "RegistryEntry":
        return cls(
            id=stable_registry_id(provider, reference),
            provider=provider,
            reference=reference,
            display_name=display_name,
            loader=loader,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "provider": self.provider.value,
            "reference": self.reference,
            "display_name": self.display_name,
            "loader": self.loader.value,
        }


__all__ = [
    "LocalLoaderMetadata",
    "LocalModelProvider",
    "RegistrationCandidate",
    "RegistryEntry",
    "ValidatedLocalModel",
    "canonical_ollama_reference",
    "stable_registry_id",
]
