"""Deeply immutable JSON values used by catalog parameter maps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

from .errors import CatalogValidationError


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FrozenJSONValue: TypeAlias = (
    JSONScalar
    | tuple["FrozenJSONValue", ...]
    | Mapping[str, "FrozenJSONValue"]
)
FrozenJSONMapping: TypeAlias = Mapping[str, FrozenJSONValue]

I_JSON_SAFE_INTEGER = 9_007_199_254_740_991


def _child_path(path: str, key: str | int) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _validate_string(value: str, path: str, source: str | None) -> None:
    for code_point in map(ord, value):
        if 0xD800 <= code_point <= 0xDFFF:
            raise CatalogValidationError(
                path,
                f"contains a lone Unicode surrogate U+{code_point:04X}",
                source=source,
            )


def freeze_json(
    value: object,
    *,
    path: str,
    source: str | None = None,
) -> FrozenJSONValue:
    """Validate an I-JSON-safe value and return a deeply immutable copy."""

    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > I_JSON_SAFE_INTEGER:
            raise CatalogValidationError(
                path,
                "integer is outside the I-JSON interoperable range",
                source=source,
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CatalogValidationError(
                path,
                "number must be finite",
                source=source,
            )
        return value
    if type(value) is str:
        _validate_string(value, path, source)
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(
                item,
                path=_child_path(path, index),
                source=source,
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJSONValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise CatalogValidationError(
                    path,
                    "JSON object names must be strings",
                    source=source,
                )
            _validate_string(key, _child_path(path, key), source)
            frozen[key] = freeze_json(
                item,
                path=_child_path(path, key),
                source=source,
            )
        return MappingProxyType(frozen)
    raise CatalogValidationError(
        path,
        f"value of type {type(value).__name__} is not JSON",
        source=source,
    )


def freeze_json_mapping(
    value: object,
    *,
    path: str,
    source: str | None = None,
) -> FrozenJSONMapping:
    if not isinstance(value, Mapping):
        raise CatalogValidationError(path, "must be an object", source=source)
    frozen = freeze_json(value, path=path, source=source)
    if not isinstance(frozen, Mapping):
        raise AssertionError("mapping input must freeze to a mapping")
    return frozen


def thaw_json(value: FrozenJSONValue) -> JSONValue:
    """Return an ordinary JSON tree detached from the immutable model."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
