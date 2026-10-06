from __future__ import annotations

import asyncio
from unittest import mock

import pytest

import metrics


EXPECTED = (
    "sha256:be10f0dca76de099a561f53ba77adab9e4c9b9e7238ae10ac6d8b506206f4fc0"
)


def _status(digest=EXPECTED):
    return {
        "running": True,
        "status": "no_model",
        "catalog_digest": digest,
    }


def test_proxy_diagnostic_reports_all_three_matching_digests() -> None:
    result = metrics.catalog_consistency_diagnostic(_status(), _status())

    assert result["status"] == "consistent"
    assert result["proxy_digest"] == EXPECTED
    assert result["expected_digest"] == EXPECTED
    assert result["services"]["kiron-embeddings"]["reported_digest"] == EXPECTED
    assert result["services"]["kiron-deberta"]["reported_digest"] == EXPECTED


@pytest.mark.parametrize(
    ("digest", "state"),
    [
        (None, "missing"),
        ("c91229d7", "malformed"),
        ("sha256:" + "0" * 64, "mismatch"),
    ],
)
def test_proxy_diagnostic_fails_closed_for_bad_service_digest(
    digest: object,
    state: str,
) -> None:
    result = metrics.catalog_consistency_diagnostic(_status(digest), _status())

    service = result["services"]["kiron-embeddings"]
    assert result["status"] == "inconsistent"
    assert result["consistent"] is False
    assert service["state"] == state
    assert EXPECTED in service["error"]
    assert "reported" in service["error"]


def test_periodic_503_probes_preserve_digest_and_state_inventory() -> None:
    model_states = [{"name": "exact", "installed": True}]

    class Response:
        status_code = 503

        def json(self):
            return {
                "status": "no_model",
                "catalog_digest": EXPECTED,
                "loaded_models": [],
                "model_states": model_states,
            }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, _url):
            return Response()

    with mock.patch.object(metrics.httpx, "AsyncClient", Client):
        embedding = asyncio.run(metrics.get_embedding_status())
        deberta = asyncio.run(metrics.get_deberta_status())

    assert embedding["running"] is True
    assert deberta["running"] is True
    assert embedding["catalog_digest"] == deberta["catalog_digest"] == EXPECTED
    assert embedding["model_states"] == deberta["model_states"] == model_states
