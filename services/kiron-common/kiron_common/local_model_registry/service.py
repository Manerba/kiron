"""One local registration use case shared by dashboard and root CLI."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from .errors import InvalidLoaderError, InvalidProviderError, ModelRegistrationError, LoaderMetadataError
from .models import RegistrationCandidate, RegistryEntry, loader_backend, HF_LOADERS
from .registry import RuntimeModelRegistry
from .validators import HuggingFaceLocalValidator, OllamaLocalValidator
from .gguf import GGUFLocalValidator


@dataclass(frozen=True, slots=True)
class RegistrationValidators:
    ollama: OllamaLocalValidator
    huggingface: HuggingFaceLocalValidator
    gguf: GGUFLocalValidator

    def __post_init__(self):
        if not isinstance(self.ollama, OllamaLocalValidator) or not isinstance(self.huggingface, HuggingFaceLocalValidator) or not isinstance(self.gguf, GGUFLocalValidator):
            raise TypeError("invalid registration validators")


@dataclass(frozen=True, slots=True)
class CandidateDiscovery:
    candidates: tuple[RegistrationCandidate, ...]
    errors: Mapping[BackendType, str]

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))


def _provider(value):
    if isinstance(value, BackendType):
        return value
    try:
        if type(value) is not str:
            raise ValueError()
        return BackendType(value.strip().lower())
    except ValueError:
        raise InvalidProviderError() from None


def _loader(value):
    if isinstance(value, LoaderType):
        return value
    try:
        if type(value) is not str:
            raise ValueError()
        return LoaderType(value.strip().lower())
    except ValueError:
        raise InvalidLoaderError() from None


def register_model(registry, validators, *, runtime_provider, reference, loader=None,
                   projector_reference=None, runtime_profile=None, expected_sha256=None,
                   expected_projector_sha256=None):
    if not isinstance(registry, RuntimeModelRegistry) or not isinstance(validators, RegistrationValidators):
        raise TypeError("invalid registration dependencies")
    provider = _provider(runtime_provider)
    if provider is BackendType.OLLAMA:
        selected = LoaderType.OLLAMA if loader is None else _loader(loader)
    elif provider is BackendType.PRISM:
        selected = LoaderType.PRISM_GGUF if loader is None else _loader(loader)
    elif provider in (BackendType.KIRON_EMBEDDINGS, BackendType.KIRON_DEBERTA):
        selected = _loader(loader)
    else:
        raise InvalidProviderError()
    if loader_backend(selected) is not provider:
        raise InvalidLoaderError()
    if provider is not BackendType.PRISM and any(item is not None for item in
            (projector_reference, runtime_profile, expected_sha256, expected_projector_sha256)):
        raise LoaderMetadataError()
    if provider is BackendType.OLLAMA:
        candidate = validators.ollama.validate(reference)
    elif provider in (BackendType.KIRON_EMBEDDINGS, BackendType.KIRON_DEBERTA):
        if selected not in HF_LOADERS:
            raise InvalidLoaderError()
        candidate = validators.huggingface.validate(reference, selected)
    elif provider is BackendType.PRISM:
        candidate = validators.gguf.validate(reference, runtime_profile=runtime_profile,
                    projector_reference=projector_reference, expected_sha256=expected_sha256,
                    expected_projector_sha256=expected_projector_sha256)
    else:
        raise InvalidProviderError()
    try:
        entry = candidate.to_entry()
    except (TypeError, ValueError):
        raise LoaderMetadataError() from None
    return registry.add(entry)


def list_models(registry: RuntimeModelRegistry) -> tuple[RegistryEntry, ...]:
    if not isinstance(registry, RuntimeModelRegistry):
        raise TypeError("registry has the wrong type")
    return registry.list()


def read_model(
    registry: RuntimeModelRegistry,
    entry_id: object,
) -> RegistryEntry | None:
    if not isinstance(registry, RuntimeModelRegistry):
        raise TypeError("registry has the wrong type")
    return registry.get(entry_id)


def list_candidates(registry, validators):
    registered = registry.list()  # Corrupt persistent state remains a whole-snapshot failure.
    identities = {(entry.runtime_provider, entry.reference) for entry in registered}
    candidates, errors = [], {}
    sources = (
        (BackendType.OLLAMA, ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST, validators.ollama),
        (BackendType.KIRON_EMBEDDINGS, ArtifactType.LOCAL, ArtifactFormat.HF_WEIGHTS, validators.huggingface),
        (BackendType.KIRON_DEBERTA, ArtifactType.LOCAL, ArtifactFormat.HF_WEIGHTS, validators.huggingface),
        (BackendType.PRISM, ArtifactType.LOCAL, ArtifactFormat.GGUF, validators.gguf),
    )
    for provider, origin, format_, validator in sources:
        try:
            references = validator.available_references()
        except ModelRegistrationError as exc:
            errors[provider] = exc.code
            continue
        for reference in references:
            if (provider, reference) in identities:
                continue
            name = reference if provider is BackendType.OLLAMA else Path(reference).name
            candidates.append(RegistrationCandidate(provider, origin, format_, reference, name))
    return CandidateDiscovery(tuple(sorted(candidates, key=lambda item:
                             (item.runtime_provider.value, item.display_name.casefold(), item.reference))), errors)


class ModelRegistrationService:
    def __init__(self, registry, validators):
        if not isinstance(registry, RuntimeModelRegistry) or not isinstance(validators, RegistrationValidators):
            raise TypeError("invalid registration dependencies")
        self._registry, self._validators = registry, validators

    def register_model(self, *, runtime_provider, reference, loader=None, projector_reference=None,
                       runtime_profile=None, expected_sha256=None, expected_projector_sha256=None):
        return register_model(self._registry, self._validators, runtime_provider=runtime_provider,
                              reference=reference, loader=loader, projector_reference=projector_reference,
                              runtime_profile=runtime_profile, expected_sha256=expected_sha256,
                              expected_projector_sha256=expected_projector_sha256)

    def list_models(self):
        return list_models(self._registry)

    def read_model(self, entry_id):
        return read_model(self._registry, entry_id)

    def list_candidates(self):
        return list_candidates(self._registry, self._validators)

    def registration_profiles(self):
        """Immutable approved GGUF registration choices; never launch parameters."""
        return MappingProxyType(dict(self._validators.gguf.profiles))
