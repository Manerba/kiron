"""Single registration use case shared by future dashboard and root CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kiron_common.model_catalog import LoaderType

from .errors import InvalidLoaderError, InvalidProviderError
from .models import LocalModelProvider, RegistrationCandidate, RegistryEntry
from .registry import RuntimeModelRegistry
from .validators import HuggingFaceLocalValidator, OllamaLocalValidator


@dataclass(frozen=True, slots=True)
class RegistrationValidators:
    ollama: OllamaLocalValidator
    huggingface: HuggingFaceLocalValidator

    def __post_init__(self) -> None:
        if not isinstance(self.ollama, OllamaLocalValidator):
            raise TypeError("ollama validator has the wrong type")
        if not isinstance(self.huggingface, HuggingFaceLocalValidator):
            raise TypeError("huggingface validator has the wrong type")


def _provider(value: object) -> LocalModelProvider:
    if isinstance(value, LocalModelProvider):
        return value
    if type(value) is not str:
        raise InvalidProviderError()
    try:
        return LocalModelProvider(value.strip().lower())
    except ValueError:
        raise InvalidProviderError() from None


def _loader(value: object) -> LoaderType:
    if isinstance(value, LoaderType):
        return value
    if type(value) is not str:
        raise InvalidLoaderError()
    try:
        return LoaderType(value.strip().lower())
    except ValueError:
        raise InvalidLoaderError() from None


def register_model(
    registry: RuntimeModelRegistry,
    validators: RegistrationValidators,
    *,
    provider: object,
    reference: object,
    loader: object = None,
) -> RegistryEntry:
    """Validate one already-local model and atomically register it."""

    if not isinstance(registry, RuntimeModelRegistry):
        raise TypeError("registry has the wrong type")
    if not isinstance(validators, RegistrationValidators):
        raise TypeError("validators have the wrong type")
    resolved_provider = _provider(provider)
    if resolved_provider is LocalModelProvider.OLLAMA:
        resolved_loader = LoaderType.OLLAMA if loader is None else _loader(loader)
        if resolved_loader is not LoaderType.OLLAMA:
            raise InvalidLoaderError()
        candidate = validators.ollama.validate(reference)
    else:
        resolved_loader = _loader(loader)
        if resolved_loader is LoaderType.OLLAMA:
            raise InvalidLoaderError()
        candidate = validators.huggingface.validate(reference, resolved_loader)
    return registry.add(candidate.to_entry())


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


def list_candidates(
    registry: RuntimeModelRegistry,
    validators: RegistrationValidators,
) -> tuple[RegistrationCandidate, ...]:
    """Discover already-local models not yet present in the runtime registry."""

    if not isinstance(registry, RuntimeModelRegistry):
        raise TypeError("registry has the wrong type")
    if not isinstance(validators, RegistrationValidators):
        raise TypeError("validators have the wrong type")
    registered = registry.list()
    registered_ollama = {
        entry.reference.lower()
        for entry in registered
        if entry.provider is LocalModelProvider.OLLAMA
    }
    registered_huggingface = {
        entry.reference
        for entry in registered
        if entry.provider is LocalModelProvider.HUGGINGFACE
    }
    candidates = [
        RegistrationCandidate(
            provider=LocalModelProvider.OLLAMA,
            reference=reference,
            display_name=reference,
        )
        for reference in validators.ollama.available_references()
        if reference.lower() not in registered_ollama
    ]
    candidates.extend(
        RegistrationCandidate(
            provider=LocalModelProvider.HUGGINGFACE,
            reference=reference,
            display_name=Path(reference).name,
        )
        for reference in validators.huggingface.available_references()
        if reference not in registered_huggingface
    )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.provider.value,
                item.display_name.casefold(),
                item.reference,
            ),
        )
    )


class ModelRegistrationService:
    """Bound facade so every caller executes the same use case."""

    __slots__ = ("_registry", "_validators")

    def __init__(
        self,
        registry: RuntimeModelRegistry,
        validators: RegistrationValidators,
    ) -> None:
        if not isinstance(registry, RuntimeModelRegistry):
            raise TypeError("registry has the wrong type")
        if not isinstance(validators, RegistrationValidators):
            raise TypeError("validators have the wrong type")
        self._registry = registry
        self._validators = validators

    def register_model(
        self,
        *,
        provider: object,
        reference: object,
        loader: object = None,
    ) -> RegistryEntry:
        return register_model(
            self._registry,
            self._validators,
            provider=provider,
            reference=reference,
            loader=loader,
        )

    def list_models(self) -> tuple[RegistryEntry, ...]:
        return list_models(self._registry)

    def read_model(self, entry_id: object) -> RegistryEntry | None:
        return read_model(self._registry, entry_id)

    def list_candidates(self) -> tuple[RegistrationCandidate, ...]:
        return list_candidates(self._registry, self._validators)


__all__ = [
    "ModelRegistrationService",
    "RegistrationValidators",
    "list_candidates",
    "list_models",
    "read_model",
    "register_model",
]
