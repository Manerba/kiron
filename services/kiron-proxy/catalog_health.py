"""Fail-closed live Catalog consistency gate for managed KIron services."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kiron_common.catalog_consistency import check_catalog_digests

from routing_catalog import PROXY_ROUTING_VIEW


CATALOG_HEALTH_URLS = {
    "kiron-embeddings": "http://127.0.0.1:11436/health",
    "kiron-deberta": "http://127.0.0.1:11437/health",
}
_REACHABLE_503_STATES = frozenset(("no_model", "loading"))


@dataclass(frozen=True, slots=True)
class HealthObservation:
    """One HTTP health observation, separated from comparison policy."""

    status_code: int | None
    payload: object
    error: str | None = None


def _decode_health_payload(raw: bytes) -> tuple[object, str | None]:
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"malformed health JSON: {type(exc).__name__}: {exc}"


def fetch_health(url: str, *, timeout: float) -> HealthObservation:
    """Fetch one local health endpoint, retaining HTTP 503 response bodies."""

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload, error = _decode_health_payload(response.read())
            return HealthObservation(response.status, payload, error)
    except HTTPError as exc:
        payload, error = _decode_health_payload(exc.read())
        return HealthObservation(exc.code, payload, error)
    except (URLError, TimeoutError, OSError) as exc:
        return HealthObservation(
            None,
            None,
            f"health request failed: {type(exc).__name__}: {exc}",
        )


def evaluate_catalog_health(
    expected_digest: str,
    observations: Mapping[str, HealthObservation],
    *,
    required_services: Sequence[str],
) -> dict[str, Any]:
    """Purely evaluate reachability and exact Catalog digests."""

    reported: dict[str, object] = {}
    health: dict[str, dict[str, Any]] = {}
    reachability_errors: list[str] = []

    for service in required_services:
        observation = observations.get(service)
        payload = (
            observation.payload
            if observation is not None and isinstance(observation.payload, Mapping)
            else None
        )
        actual = payload.get("catalog_digest") if payload is not None else None
        reported[service] = actual

        service_status = payload.get("status") if payload is not None else None
        reachable = bool(
            observation is not None
            and observation.error is None
            and payload is not None
            and (
                observation.status_code == 200
                or (
                    observation.status_code == 503
                    and service_status in _REACHABLE_503_STATES
                )
            )
        )
        error: str | None = None
        if not reachable:
            if observation is None:
                reason = "health observation missing"
                status_code = None
            elif observation.error is not None:
                reason = observation.error
                status_code = observation.status_code
            elif payload is None:
                reason = "health payload must be a JSON object"
                status_code = observation.status_code
            else:
                reason = (
                    f"unacceptable health response HTTP {observation.status_code} "
                    f"with status {service_status!r}"
                )
                status_code = observation.status_code
            error = (
                f"{service}: {reason}; expected {expected_digest}, "
                f"reported {actual!r}"
            )
            reachability_errors.append(error)
        else:
            status_code = observation.status_code

        health[service] = {
            "reachable": reachable,
            "status_code": status_code,
            "service_status": service_status,
            "error": error,
        }

    digest_report = check_catalog_digests(
        expected_digest,
        reported,
        required_services=required_services,
    ).to_dict()
    errors = [*reachability_errors, *digest_report["errors"]]
    consistent = digest_report["consistent"] and not reachability_errors
    return {
        **digest_report,
        "status": "consistent" if consistent else "inconsistent",
        "consistent": consistent,
        "proxy_digest": expected_digest,
        "health": health,
        "errors": errors,
    }


def check_live_catalog_consistency(
    *,
    fetcher: Callable[..., HealthObservation] = fetch_health,
    timeout: float = 2.0,
    service_urls: Mapping[str, str] = CATALOG_HEALTH_URLS,
) -> dict[str, Any]:
    """Fetch all managed services once and apply the shared comparison."""

    services = tuple(service_urls)
    observations = {
        service: fetcher(url, timeout=timeout)
        for service, url in service_urls.items()
    }
    return evaluate_catalog_health(
        PROXY_ROUTING_VIEW.catalog_digest,
        observations,
        required_services=services,
    )


def wait_for_live_catalog_consistency(
    *,
    wait_seconds: float,
    interval_seconds: float = 0.5,
    checker: Callable[[], dict[str, Any]] = check_live_catalog_consistency,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait only for restart reachability; a reached mismatch fails immediately."""

    if wait_seconds < 0 or interval_seconds <= 0:
        raise ValueError("wait_seconds must be >= 0 and interval_seconds must be > 0")
    deadline = monotonic() + wait_seconds
    while True:
        report = checker()
        if report["consistent"]:
            return report

        health = report.get("health", {})
        all_reachable = bool(health) and all(
            isinstance(item, Mapping) and item.get("reachable") is True
            for item in health.values()
        )
        if all_reachable or monotonic() >= deadline:
            return report
        sleep(min(interval_seconds, max(0.0, deadline - monotonic())))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check live managed-service Catalog digests fail-closed.",
    )
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = wait_for_live_catalog_consistency(
        wait_seconds=args.wait_seconds,
        interval_seconds=args.interval_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
