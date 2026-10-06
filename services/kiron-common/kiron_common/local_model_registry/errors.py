"""Deterministic errors for local model registration and persistence."""

from __future__ import annotations


class ModelRegistrationError(ValueError):
    """A stable, presentation-safe registration failure."""

    code = "registration_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidProviderError(ModelRegistrationError):
    code = "invalid_provider"

    def __init__(self) -> None:
        super().__init__("runtime_provider must be ollama, prism, kiron_embeddings or kiron_deberta")


class InvalidReferenceError(ModelRegistrationError):
    code = "invalid_reference"

    def __init__(self) -> None:
        super().__init__("local model reference is invalid")


class LocalModelNotFoundError(ModelRegistrationError):
    code = "model_not_found"

    def __init__(self) -> None:
        super().__init__("model is not present locally")


class InvalidLoaderError(ModelRegistrationError):
    code = "invalid_loader"

    def __init__(self) -> None:
        super().__init__("loader is missing or unsupported for this provider")


class LoaderMetadataError(ModelRegistrationError):
    code = "loader_metadata_invalid"

    def __init__(self) -> None:
        super().__init__("local model metadata is not valid for the loader")


class LocalValidationError(ModelRegistrationError):
    code = "local_validation_failed"

    def __init__(self) -> None:
        super().__init__("local provider validation failed")


class DuplicateModelError(ModelRegistrationError):
    code = "duplicate_model"

    def __init__(self) -> None:
        super().__init__("model is already registered")


class RegistryCorruptionError(ModelRegistrationError):
    code = "registry_corrupt"

    def __init__(self) -> None:
        super().__init__("local model registry is corrupt")


class RegistryAccessError(ModelRegistrationError):
    code = "registry_access_failed"

    def __init__(self) -> None:
        super().__init__("local model registry access failed")


__all__ = [
    "DuplicateModelError",
    "InvalidLoaderError",
    "InvalidProviderError",
    "InvalidReferenceError",
    "LoaderMetadataError",
    "LocalModelNotFoundError",
    "LocalValidationError",
    "ModelRegistrationError",
    "RegistryAccessError",
    "RegistryCorruptionError",
]
