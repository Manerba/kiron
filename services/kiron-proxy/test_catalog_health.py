from __future__ import annotations

from catalog_health import (
    CATALOG_HEALTH_URLS,
    HealthObservation,
    check_live_catalog_consistency,
    evaluate_catalog_health,
    wait_for_live_catalog_consistency,
)
from routing_catalog import PROXY_ROUTING_VIEW


EXPECTED = PROXY_ROUTING_VIEW.catalog_digest
SERVICES = tuple(CATALOG_HEALTH_URLS)


def _observations(
    embedding_digest: object = EXPECTED,
    deberta_digest: object = EXPECTED,
) -> dict[str, HealthObservation]:
    return {
        "kiron-embeddings": HealthObservation(
            503,
            {"status": "no_model", "catalog_digest": embedding_digest},
        ),
        "kiron-deberta": HealthObservation(
            503,
            {"status": "loading", "catalog_digest": deberta_digest},
        ),
    }


def test_http_503_no_model_and_loading_are_reachable_and_checked() -> None:
    report = evaluate_catalog_health(
        EXPECTED,
        _observations(),
        required_services=SERVICES,
    )

    assert report["consistent"] is True
    assert all(row["reachable"] for row in report["health"].values())
    assert report["services"]["kiron-embeddings"]["state"] == "match"
    assert report["services"]["kiron-deberta"]["state"] == "match"


def test_reachable_digest_mismatch_fails_immediately_with_expected_and_actual() -> None:
    mismatch = "sha256:" + "0" * 64
    report = evaluate_catalog_health(
        EXPECTED,
        _observations(deberta_digest=mismatch),
        required_services=SERVICES,
    )

    assert report["consistent"] is False
    result = report["services"]["kiron-deberta"]
    assert result["state"] == "mismatch"
    assert EXPECTED in result["error"]
    assert mismatch in result["error"]


def test_missing_malformed_and_invalid_503_health_fail_closed() -> None:
    observations = _observations(embedding_digest=None, deberta_digest="not-a-digest")
    observations["kiron-deberta"] = HealthObservation(
        503,
        {"status": "stopping", "catalog_digest": "not-a-digest"},
    )
    report = evaluate_catalog_health(
        EXPECTED,
        observations,
        required_services=SERVICES,
    )

    assert report["consistent"] is False
    assert report["services"]["kiron-embeddings"]["state"] == "missing"
    assert report["services"]["kiron-deberta"]["state"] == "malformed"
    assert report["health"]["kiron-deberta"]["reachable"] is False


def test_live_checker_uses_injected_fetcher_without_network() -> None:
    seen: list[tuple[str, float]] = []

    def fetcher(url: str, *, timeout: float) -> HealthObservation:
        seen.append((url, timeout))
        status = "loading" if url.endswith("11437/health") else "no_model"
        return HealthObservation(503, {"status": status, "catalog_digest": EXPECTED})

    report = check_live_catalog_consistency(fetcher=fetcher, timeout=0.25)

    assert report["consistent"] is True
    assert seen == [(url, 0.25) for url in CATALOG_HEALTH_URLS.values()]


def test_restart_wait_does_not_retry_a_reachable_mismatch() -> None:
    mismatch_report = evaluate_catalog_health(
        EXPECTED,
        _observations(deberta_digest="sha256:" + "f" * 64),
        required_services=SERVICES,
    )
    calls = 0

    def checker():
        nonlocal calls
        calls += 1
        return mismatch_report

    report = wait_for_live_catalog_consistency(
        wait_seconds=30,
        checker=checker,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: (_ for _ in ()).throw(
            AssertionError("reachable mismatch must not sleep")
        ),
    )

    assert report is mismatch_report
    assert calls == 1
