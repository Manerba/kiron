"""Deterministic importlib.resources loading for packaged model manifests."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .catalog import ModelCatalog
from .errors import CatalogResourceError
from .parser import parse_manifest


DEFAULT_MANIFEST_PACKAGE = "kiron_common.model_catalog.manifests"
MANIFEST_SUFFIX = ".model.json"


class _DuplicateObjectName(ValueError):
    pass


def _reject_duplicate_names(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateObjectName(f"duplicate JSON object name {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _decode_manifest(payload: bytes, *, resource_name: str) -> object:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogResourceError(resource_name, "must be UTF-8") from exc
    if text.startswith("\ufeff"):
        raise CatalogResourceError(resource_name, "must not contain a UTF-8 BOM")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_names,
            parse_constant=_reject_non_finite,
        )
    except (json.JSONDecodeError, _DuplicateObjectName, ValueError) as exc:
        raise CatalogResourceError(resource_name, str(exc)) from exc


def load_catalog_from_package(
    package: str = DEFAULT_MANIFEST_PACKAGE,
) -> ModelCatalog:
    """Load every immediate ``*.model.json`` resource in filename order."""

    try:
        root = resources.files(package)
        manifest_resources = sorted(
            (
                item
                for item in root.iterdir()
                if item.is_file() and item.name.endswith(MANIFEST_SUFFIX)
            ),
            key=lambda item: item.name,
        )
    except (ModuleNotFoundError, TypeError, OSError) as exc:
        raise CatalogResourceError(package, f"cannot enumerate package: {exc}") from exc

    groups = []
    for item in manifest_resources:
        resource_name = f"{package}:{item.name}"
        try:
            payload = item.read_bytes()
        except OSError as exc:
            raise CatalogResourceError(resource_name, f"cannot read: {exc}") from exc
        document = _decode_manifest(payload, resource_name=resource_name)
        groups.append(parse_manifest(document, source=resource_name))
    return ModelCatalog(groups)


def load_catalog(
    package: str = DEFAULT_MANIFEST_PACKAGE,
) -> ModelCatalog:
    """Public shorthand for loading the immutable packaged catalog."""

    return load_catalog_from_package(package)
