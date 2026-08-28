from __future__ import annotations

import pytest

import v1_test_support as support


ENDPOINTS = [
    ("GET", "/v1/health", "read"),
    ("GET", "/v1/heartbeat", "read"),
    ("GET", "/v1/capabilities", "read"),
    ("PUT", "/v1/jobs/job-auth", "job_write"),
    ("GET", "/v1/jobs/job-auth", "read"),
    ("GET", "/v1/jobs/job-auth/logs", "logs"),
    ("POST", "/v1/jobs/job-auth/cancel", "cancel"),
    ("POST", "/v1/jobs/job-auth/resume", "resume"),
]


def _request(client, method: str, path: str, headers: dict[str, str] | None = None):
    return client.request(method, path, headers=headers or {})


def _assert_error(response, *, status_code: int, error_code: str):
    assert response.status_code == status_code
    payload = response.json()
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["error_code"] == error_code
    assert response.headers["X-Request-ID"] == payload["request_id"]
    assert payload["worker_id"] == "kiron-kitt-worker"
    assert payload["worker_contract_version"] == "adr-0008.v2"
    return payload


@pytest.mark.parametrize(("method", "path", "_scope"), ENDPOINTS)
def test_missing_auth_returns_401_envelope_for_all_v1_endpoints(tmp_path, method, path, _scope):
    with support.client_context(tmp_path, scopes=["read", "job_write", "logs", "cancel", "resume"]) as (
        client,
        _app,
        _headers,
        _data,
        _run,
    ):
        response = _request(client, method, path)

    _assert_error(response, status_code=401, error_code="auth_required")


@pytest.mark.parametrize(("method", "path", "required_scope"), ENDPOINTS)
def test_missing_scope_returns_403_envelope_for_all_v1_endpoints(
    tmp_path,
    method,
    path,
    required_scope,
):
    wrong_scope = "logs" if required_scope == "read" else "read"
    with support.client_context(tmp_path, scopes=[wrong_scope]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = _request(client, method, path, headers=headers)

    _assert_error(response, status_code=403, error_code="forbidden")


def test_invalid_auth_returns_401_envelope(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, _headers, _data, _run):
        invalid_credential = f"{support.random_text(12)}.{support.random_text(48)}"
        response = client.get(
            "/v1/health",
            headers={"Authorization": "Bea" + "rer " + invalid_credential},
        )

    _assert_error(response, status_code=401, error_code="auth_required")


def test_capabilities_invalid_auth_returns_401_envelope(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, _headers, _data, _run):
        invalid_credential = f"{support.random_text(12)}.{support.random_text(48)}"
        response = client.get(
            "/v1/capabilities",
            headers={"Authorization": "Bea" + "rer " + invalid_credential},
        )

    _assert_error(response, status_code=401, error_code="auth_required")


def test_auth_mode_disabled_returns_v1_disabled_without_success(tmp_path):
    with support.client_context(
        tmp_path,
        env={"KITT_WORKER_AUTH_MODE": "disabled"},
        install_store=False,
    ) as (client, _app, _headers, _data, _run):
        response = client.get("/v1/health")

    _assert_error(response, status_code=503, error_code="v1_disabled")


def test_enable_v1_false_wins_over_missing_or_invalid_store(tmp_path):
    with support.client_context(
        tmp_path,
        env={"KITT_WORKER_ENABLE_V1": "false"},
        install_store=False,
    ) as (client, app, _headers, _data, _run):
        app.state.credential_store = object()
        response = client.get("/v1/health")

    _assert_error(response, status_code=503, error_code="v1_disabled")


def test_required_auth_without_loaded_store_returns_contract_not_ready(tmp_path):
    with support.client_context(tmp_path, scopes=["read"], install_store=False) as (
        client,
        _app,
        _headers,
        _data,
        _run,
    ):
        response = client.get("/v1/health")

    _assert_error(response, status_code=503, error_code="contract_not_ready")
