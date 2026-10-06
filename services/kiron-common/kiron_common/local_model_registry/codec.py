"""Closed JSON codec for the persistent local-model registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
from datetime import datetime
import json
import re

from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

from .errors import RegistryCorruptionError
from .models import LocalArtifactFile, RegistryEntry


REGISTRY_VERSION = 2
MAX_REGISTRY_BYTES = 1024 * 1024
MAX_REGISTRY_ENTRIES = 4096
_ENTRY_KEYS = frozenset(field.name for field in fields(RegistryEntry))

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
        identities: set[tuple[BackendType, str]] = set()
        for row in rows:
            if type(row) is not dict or set(row) != _ENTRY_KEYS:
                raise _DecodeFailure("invalid entry")
            values = dict(row)
            values["runtime_provider"] = BackendType(row["runtime_provider"])
            values["artifact_origin"] = ArtifactType(row["artifact_origin"])
            values["artifact_format"] = ArtifactFormat(row["artifact_format"])
            values["loader"] = LoaderType(row["loader"])
            registered_at = row["registered_at"]
            if (type(registered_at) is not str or re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
                    registered_at) is None):
                raise _DecodeFailure("invalid UTC registration timestamp")
            values["registered_at"] = datetime.fromisoformat(registered_at)
            if row["projector"] is not None:
                projector = row["projector"]
                if type(projector) is not dict or set(projector) != {"reference", "sha256", "size_bytes"}:
                    raise _DecodeFailure("invalid projector")
                values["projector"] = LocalArtifactFile(**projector)
            entry = RegistryEntry(**values)
            identity = (entry.runtime_provider, entry.reference)
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
    identities = {(entry.runtime_provider, entry.reference) for entry in ordered}
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
