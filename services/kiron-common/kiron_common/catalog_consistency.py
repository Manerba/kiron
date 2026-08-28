"""Pure fail-closed comparison of KIron Catalog digests."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


_CATALOG_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CatalogDigestResult:
    """One service's exact digest observation."""

    service: str
    reported_digest: object
    state: str
    error: str | None

    @property
    def valid(self) -> bool:
        return self.state == "match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "reported_digest": self.reported_digest,
            "state": self.state,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CatalogConsistencyReport:
    """Deeply immutable result of one digest comparison."""

    expected_digest: str
    consistent: bool
    services: Mapping[str, CatalogDigestResult]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "services",
            MappingProxyType(dict(self.services)),
        )
        object.__setattr__(self, "errors", tuple(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "consistent" if self.consistent else "inconsistent",
            "consistent": self.consistent,
            "expected_digest": self.expected_digest,
            "services": {
                name: result.to_dict() for name, result in self.services.items()
            },
            "errors": list(self.errors),
        }


def is_catalog_digest(value: object) -> bool:
    """Return whether *value* is one canonical lower-case SHA-256 ID."""

    return type(value) is str and _CATALOG_DIGEST_RE.fullmatch(value) is not None


def check_catalog_digests(
    expected_digest: str,
    reported_digests: Mapping[str, object],
    *,
    required_services: Sequence[str],
) -> CatalogConsistencyReport:
    """Compare required service digests without inference or fallback values.

    Missing keys, explicit nulls, malformed values, and mismatches are distinct
    fail-closed states.  The caller supplies the required service order so the
    diagnostic wire and error text are deterministic.
    """

    if not is_catalog_digest(expected_digest):
        raise ValueError(
            "expected_digest must match 'sha256:' followed by 64 lower-case hex digits"
        )
    if not isinstance(reported_digests, Mapping):
        raise TypeError("reported_digests must be a mapping")

    ordered_services = tuple(required_services)
    if not ordered_services:
        raise ValueError("required_services must not be empty")
    if any(type(name) is not str or not name for name in ordered_services):
        raise ValueError("required service names must be non-empty strings")
    if len(set(ordered_services)) != len(ordered_services):
        raise ValueError("required service names must be unique")

    results: dict[str, CatalogDigestResult] = {}
    errors: list[str] = []
    for service in ordered_services:
        if service not in reported_digests or reported_digests[service] is None:
            actual: object = None
            state = "missing"
            error = (
                f"{service}: catalog_digest missing; "
                f"expected {expected_digest}, reported {actual!r}"
            )
        else:
            actual = reported_digests[service]
            if not is_catalog_digest(actual):
                state = "malformed"
                error = (
                    f"{service}: catalog_digest malformed; "
                    f"expected {expected_digest}, reported {actual!r}"
                )
            elif actual != expected_digest:
                state = "mismatch"
                error = (
                    f"{service}: catalog_digest mismatch; "
                    f"expected {expected_digest}, reported {actual!r}"
                )
            else:
                state = "match"
                error = None
        result = CatalogDigestResult(
            service=service,
            reported_digest=actual,
            state=state,
            error=error,
        )
        results[service] = result
        if error is not None:
            errors.append(error)

    return CatalogConsistencyReport(
        expected_digest=expected_digest,
        consistent=not errors,
        services=results,
        errors=tuple(errors),
    )


__all__ = [
    "CatalogConsistencyReport",
    "CatalogDigestResult",
    "check_catalog_digests",
    "is_catalog_digest",
]
