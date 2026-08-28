"""Closed JSON codec for the persistent local-model registry."""

from __future__ import annotations

from collections.abc import Iterable
import json

from kiron_common.model_catalog import LoaderType

from .errors import RegistryCorruptionError
from .models import LocalModelProvider, RegistryEntry


REGISTRY_VERSION = 1
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_REGISTRY_ENTRIES = 4096
_ENTRY_KEYS = frozenset(
    ("id", "provider", "reference", "display_name", "loader")
)


class _DecodeFailure(ValueError):
    pass


def _object_without_duplicate_names(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DecodeFailure("duplicate object name")
        result[key] = value
    return result


def _reject_number(_value: str):
    raise _DecodeFailure("non-integer number")


def decode_registry(payload: bytes) -> tuple[RegistryEntry, ...]:
    try:
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_REGISTRY_BYTES
        ):
            raise _DecodeFailure("invalid size")
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_without_duplicate_names,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
        if type(document) is not dict or set(document) != {"version", "entries"}:
            raise _DecodeFailure("invalid document")
        if (
            type(document["version"]) is not int
            or document["version"] != REGISTRY_VERSION
        ):
            raise _DecodeFailure("invalid version")
        rows = document["entries"]
        if type(rows) is not list or len(rows) > MAX_REGISTRY_ENTRIES:
            raise _DecodeFailure("invalid entries")
        entries: list[RegistryEntry] = []
        identities: set[tuple[LocalModelProvider, str]] = set()
        for row in rows:
            if type(row) is not dict or set(row) != _ENTRY_KEYS:
                raise _DecodeFailure("invalid entry")
            if any(type(row[key]) is not str for key in _ENTRY_KEYS):
                raise _DecodeFailure("entry values must be strings")
            entry = RegistryEntry(
                id=row["id"],
                provider=LocalModelProvider(row["provider"]),
                reference=row["reference"],
                display_name=row["display_name"],
                loader=LoaderType(row["loader"]),
            )
            identity = (entry.provider, entry.reference)
            if identity in identities:
                raise _DecodeFailure("duplicate identity")
            identities.add(identity)
            entries.append(entry)
        if entries != sorted(entries, key=lambda entry: entry.id):
            raise _DecodeFailure("entries are not ordered")
        return tuple(entries)
    except RegistryCorruptionError:
        raise
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DecodeFailure,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise RegistryCorruptionError() from None


def encode_registry(entries: Iterable[RegistryEntry]) -> bytes:
    ordered = tuple(sorted(entries, key=lambda entry: entry.id))
    if len(ordered) > MAX_REGISTRY_ENTRIES or any(
        type(entry) is not RegistryEntry for entry in ordered
    ):
        raise ValueError("invalid registry entries")
    identities = {(entry.provider, entry.reference) for entry in ordered}
    if len(identities) != len(ordered):
        raise ValueError("duplicate registry identity")
    document = {
        "version": REGISTRY_VERSION,
        "entries": [entry.to_dict() for entry in ordered],
    }
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_REGISTRY_BYTES:
        raise ValueError("registry exceeds maximum size")
    return payload


__all__ = [
    "MAX_REGISTRY_BYTES",
    "MAX_REGISTRY_ENTRIES",
    "REGISTRY_VERSION",
    "decode_registry",
    "encode_registry",
]
