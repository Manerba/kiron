"""Injected, local-only validators for Ollama and Hugging Face models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import re
import stat

from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

from .errors import (
    InvalidReferenceError,
    LoaderMetadataError,
    LocalModelNotFoundError,
    LocalValidationError,
)
from .models import (
    LocalLoaderMetadata,
    loader_backend,
    HF_LOADERS,
    ValidatedLocalModel,
    canonical_ollama_reference,
)


DEFAULT_HUGGINGFACE_MODEL_ROOT = Path("/usr/lib/kiron/data/local-models")


def normalize_ollama_reference(value: object) -> str:
    normalized = canonical_ollama_reference(value)
    if normalized is None:
        raise InvalidReferenceError()
    return normalized


class OllamaLocalValidator:
    """Validate solely with injected results from local list/show operations."""

    __slots__ = ("_list_models", "_show_model")

    def __init__(
        self,
        *,
        list_models: Callable[[], object],
        show_model: Callable[[str], object],
    ) -> None:
        if not callable(list_models) or not callable(show_model):
            raise TypeError("local Ollama callbacks must be callable")
        self._list_models = list_models
        self._show_model = show_model

    def available_references(self) -> tuple[str, ...]:
        """Return exact local names; Ollama cloud proxies are not local models."""
        return tuple(sorted((item[0] for item in self._listed_models().values()),
                            key=lambda item: (item.casefold(), item)))

    def _listed_models(self) -> dict[str, tuple[str, object]]:

        try:
            payload = self._list_models()
        except Exception:
            raise LocalValidationError() from None
        if not isinstance(payload, Mapping):
            raise LocalValidationError()
        rows = payload.get("models")
        if type(rows) is not list:
            raise LocalValidationError()
        available: dict[str, tuple[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise LocalValidationError()
            raw_name = row.get("name", row.get("model"))
            try:
                canonical = normalize_ollama_reference(raw_name)
            except InvalidReferenceError:
                raise LocalValidationError() from None
            key = canonical.lower()
            if key in available:
                raise LocalValidationError()
            if canonical.rsplit(":", 1)[1].lower() == "cloud":
                continue
            available[key] = (canonical, row.get("digest"))
        return available

    @staticmethod
    def _manifest_digest(value: object) -> str:
        match = re.fullmatch(r"(?:sha256:)?([0-9a-fA-F]{64})", value) if type(value) is str else None
        if match is None:
            raise LocalValidationError()
        return match.group(1).lower()

    def validate(self, reference: object) -> ValidatedLocalModel:
        requested = normalize_ollama_reference(reference)
        listed = self._listed_models().get(requested.lower())
        if listed is None:
            raise LocalModelNotFoundError()
        canonical, raw_digest = listed
        digest = self._manifest_digest(raw_digest)
        try:
            shown = self._show_model(canonical)
        except Exception:
            raise LocalValidationError() from None
        if not isinstance(shown, Mapping) or not shown:
            raise LocalValidationError()
        for field in ("name", "model"):
            if field in shown:
                try:
                    shown_name = normalize_ollama_reference(shown[field])
                except InvalidReferenceError:
                    raise LocalValidationError() from None
                if shown_name.lower() != canonical.lower():
                    raise LocalValidationError()
        # Detect observed tag replacement around show; neither names nor reported
        # model sizes are evidence of the manifest's content identity.
        current = self._listed_models().get(requested.lower())
        if (current is None or current[0] != canonical
                or self._manifest_digest(current[1]) != digest):
            raise LocalValidationError()
        return ValidatedLocalModel(
            runtime_provider=BackendType.OLLAMA,
            artifact_origin=ArtifactType.OLLAMA,
            artifact_format=ArtifactFormat.OLLAMA_MANIFEST,
            reference=canonical,
            display_name=canonical,
            loader=LoaderType.OLLAMA,
            sha256=digest,
        )


def _canonical_local_directory(value: object) -> Path:
    if type(value) is not str:
        raise InvalidReferenceError()
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > 4096
        or "://" in rendered
        or "\x00" in rendered
        or any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in rendered
        )
        or not Path(rendered).is_absolute()
    ):
        raise InvalidReferenceError()
    normalized = Path(os.path.normpath(rendered))
    if normalized == Path(normalized.anchor):
        raise InvalidReferenceError()
    current = Path(normalized.anchor)
    for component in normalized.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            raise LocalModelNotFoundError() from None
        except OSError:
            raise InvalidReferenceError() from None
        if stat.S_ISLNK(info.st_mode):
            raise InvalidReferenceError()
    if not stat.S_ISDIR(info.st_mode):
        raise InvalidReferenceError()
    try:
        canonical = normalized.resolve(strict=True)
    except OSError:
        raise InvalidReferenceError() from None
    if canonical != normalized:
        raise InvalidReferenceError()
    return canonical


class HuggingFaceLocalValidator:
    """Validate a canonical local directory through an injected loader probe."""

    __slots__ = ("_inspect_loader", "_model_root")

    def __init__(
        self,
        *,
        inspect_loader: Callable[[Path, LoaderType], LocalLoaderMetadata],
        model_root: Path = DEFAULT_HUGGINGFACE_MODEL_ROOT,
    ) -> None:
        if not callable(inspect_loader):
            raise TypeError("local loader inspector must be callable")
        if not isinstance(model_root, Path) or not model_root.is_absolute():
            raise TypeError("model_root must be an absolute Path")
        self._inspect_loader = inspect_loader
        self._model_root = model_root

    def _canonical_root(self) -> Path:
        return _canonical_local_directory(str(self._model_root))

    def available_references(self) -> tuple[str, ...]:
        """List direct, visible model directories below the fixed local root."""

        try:
            root = self._canonical_root()
        except LocalModelNotFoundError:
            return ()
        except InvalidReferenceError:
            raise LocalValidationError() from None
        try:
            with os.scandir(root) as scanner:
                entries = tuple(scanner)
        except OSError:
            raise LocalValidationError() from None
        references: list[str] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                candidate = _canonical_local_directory(str(root / entry.name))
            except (InvalidReferenceError, LocalModelNotFoundError, OSError):
                raise LocalValidationError() from None
            if candidate.parent != root:
                raise LocalValidationError()
            references.append(str(candidate))
        return tuple(sorted(
            references,
            key=lambda item: (item.casefold(), item),
        ))

    def validate(
        self,
        reference: object,
        loader: LoaderType,
    ) -> ValidatedLocalModel:
        if loader not in HF_LOADERS:
            raise LoaderMetadataError()
        path = _canonical_local_directory(reference)
        try:
            root = self._canonical_root()
        except (InvalidReferenceError, LocalModelNotFoundError):
            raise InvalidReferenceError() from None
        if path.parent != root:
            raise InvalidReferenceError()
        try:
            metadata = self._inspect_loader(path, loader)
        except Exception:
            raise LoaderMetadataError() from None
        if type(metadata) is not LocalLoaderMetadata:
            raise LoaderMetadataError()
        display_name = metadata.display_name or path.name
        return ValidatedLocalModel(
            runtime_provider=loader_backend(loader),
            artifact_origin=ArtifactType.LOCAL,
            artifact_format=ArtifactFormat.HF_WEIGHTS,
            reference=str(path),
            display_name=display_name,
            loader=loader,
        )


__all__ = [
    "DEFAULT_HUGGINGFACE_MODEL_ROOT",
    "HuggingFaceLocalValidator",
    "OllamaLocalValidator",
    "normalize_ollama_reference",
]
