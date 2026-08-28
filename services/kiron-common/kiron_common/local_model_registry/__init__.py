"""Dynamic local-model registry, separate from the static release Catalog.

This package performs no provider or filesystem work at import time. Callers
inject local validators and use the same :func:`register_model` use case.
"""

from .errors import (
    DuplicateModelError,
    InvalidLoaderError,
    InvalidProviderError,
    InvalidReferenceError,
    LoaderMetadataError,
    LocalModelNotFoundError,
    LocalValidationError,
    ModelRegistrationError,
    RegistryAccessError,
    RegistryCorruptionError,
)
from .models import (
    LocalLoaderMetadata,
    LocalModelProvider,
    RegistrationCandidate,
    RegistryEntry,
    ValidatedLocalModel,
    stable_registry_id,
)
from .registry import (
    DEFAULT_REGISTRY_PATH,
    RegistryFilePolicy,
    RuntimeModelRegistry,
)
from .service import (
    ModelRegistrationService,
    RegistrationValidators,
    list_candidates,
    list_models,
    read_model,
    register_model,
)
from .validators import (
    DEFAULT_HUGGINGFACE_MODEL_ROOT,
    HuggingFaceLocalValidator,
    OllamaLocalValidator,
    normalize_ollama_reference,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_HUGGINGFACE_MODEL_ROOT",
    "DuplicateModelError",
    "HuggingFaceLocalValidator",
    "InvalidLoaderError",
    "InvalidProviderError",
    "InvalidReferenceError",
    "LoaderMetadataError",
    "LocalLoaderMetadata",
    "LocalModelNotFoundError",
    "LocalModelProvider",
    "LocalValidationError",
    "ModelRegistrationError",
    "ModelRegistrationService",
    "OllamaLocalValidator",
    "RegistrationCandidate",
    "RegistrationValidators",
    "RegistryAccessError",
    "RegistryCorruptionError",
    "RegistryEntry",
    "RegistryFilePolicy",
    "RuntimeModelRegistry",
    "ValidatedLocalModel",
    "list_models",
    "list_candidates",
    "normalize_ollama_reference",
    "read_model",
    "register_model",
    "stable_registry_id",
]
