from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kiron_common.catalog_consistency import (
    check_catalog_digests,
    is_catalog_digest,
)


EXPECTED = "sha256:" + "a" * 64
SERVICES = ("kiron-embeddings", "kiron-deberta")


def test_matching_digests_are_consistent_and_immutable() -> None:
    report = check_catalog_digests(
        EXPECTED,
        {name: EXPECTED for name in SERVICES},
        required_services=SERVICES,
    )

    assert report.consistent is True
    assert report.errors == ()
    assert report.to_dict()["status"] == "consistent"
    assert tuple(report.services) == SERVICES
    with pytest.raises(TypeError):
        report.services["other"] = report.services[SERVICES[0]]
    with pytest.raises(FrozenInstanceError):
        report.consistent = False


@pytest.mark.parametrize(
    ("reported", "state", "needle"),
    [
        ({"kiron-deberta": EXPECTED}, "missing", "reported None"),
        (
            {"kiron-embeddings": "a" * 64, "kiron-deberta": EXPECTED},
            "malformed",
            "reported 'aaaaaaaa",
        ),
        (
            {
                "kiron-embeddings": "sha256:" + "b" * 64,
                "kiron-deberta": EXPECTED,
            },
            "mismatch",
            "expected sha256:aaaaaaaa",
        ),
    ],
)
def test_missing_malformed_and_mismatched_digests_fail_closed(
    reported: dict[str, object],
    state: str,
    needle: str,
) -> None:
    report = check_catalog_digests(
        EXPECTED,
        reported,
        required_services=SERVICES,
    )

    result = report.services["kiron-embeddings"]
    assert report.consistent is False
    assert result.state == state
    assert result.error is not None
    assert needle in result.error
    assert EXPECTED in result.error


@pytest.mark.parametrize(
    "value",
    [None, True, "", "sha256:" + "A" * 64, "sha256:" + "a" * 63],
)
def test_digest_format_is_exact(value: object) -> None:
    assert is_catalog_digest(value) is False


def test_invalid_expected_digest_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="expected_digest"):
        check_catalog_digests(
            "a" * 64,
            {name: EXPECTED for name in SERVICES},
            required_services=SERVICES,
        )
