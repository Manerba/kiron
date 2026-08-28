"""Controlled errors raised by the declarative model catalog."""

from __future__ import annotations


class ModelCatalogError(ValueError):
    """Base class for fail-fast catalog construction and loading errors."""


class CatalogValidationError(ModelCatalogError):
    """A manifest or cross-manifest invariant is invalid."""

    reason_code = "model_catalog_invalid"

    def __init__(
        self,
        field_path: str,
        detail: str,
        *,
        source: str | None = None,
    ) -> None:
        self.field_path = field_path
        self.detail = detail
        self.source = source
        location = f" in {source}" if source is not None else ""
        super().__init__(
            f"{self.reason_code}{location} at {field_path}: {detail}"
        )


class CatalogResourceError(ModelCatalogError):
    """A packaged manifest resource cannot be enumerated or decoded."""

    reason_code = "model_catalog_resource_invalid"

    def __init__(self, resource: str, detail: str) -> None:
        self.resource = resource
        self.detail = detail
        super().__init__(f"{self.reason_code} for {resource}: {detail}")


class CatalogLookupError(LookupError):
    """A required literal model name is absent from the catalog."""

    reason_code = "model_catalog_name_not_found"

    def __init__(self, model_name: object) -> None:
        self.model_name = model_name
        super().__init__(
            f"{self.reason_code}: no exact canonical ID or alias for {model_name!r}"
        )
