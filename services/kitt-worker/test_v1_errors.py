from __future__ import annotations

import json

import pytest

import v1_test_support as support


VALID_SPEC = {
    "schema_version": "kitt_job_spec_v1",
    "run_uid": "kitt-run-1",
    "run_type": "sft",
    "training_profile": {
        "profile_uid": "profile-1",
        "version_label": "v1",
        "profile_hash_sha256": "a" * 64,
    },
    "input_reference": {"mode": "dataset_version", "dataset_version_uid": "5136e167-dbb7-49a6-a752-6fe1e962adb7"},
    "base_or_parent_model": {"external_parent_ref": "Qwen/Qwen2.5-7B-Instruct"},
    "hyperparameters": {"learning_rate": 0.0002},
    "output_roles": [
        "adapter",
        "merged_weights",
        "gguf",
        "ollama_modelfile",
        "training_log",
        "metrics_jsonl",
        "run_lock",
    ],
    "capability_snapshot_hash_sha256": "b" * 64,
    "labels": {"kitt_job": "job.v1"},
    "metadata": {"operator": "kiron"},
}


def _join(*parts: str) -> str:
    return "".join(parts)


def _set_store(app, scopes: list[str]) -> dict[str, str]:
    store, credential = support.credential_store(scopes)
    app.state.credential_store = store
    return support.authorization_header(credential)


def _assert_error(response, *, status_code: int, error_code: str):
    assert response.status_code == status_code
    payload = response.json()
    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["worker_id"] == "kiron-kitt-worker"
    assert payload["worker_contract_version"] == "adr-0008.v2"
    assert payload["request_id"] == response.headers["X-Request-ID"]
    assert payload["error_code"] == error_code
    assert payload["diagnostic"]
    return payload


def test_unknown_v1_route_and_wrong_method_return_envelope(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, headers, _data, _run):
        missing = client.get("/v1/does-not-exist", headers=headers)
        wrong_method = client.post("/v1/health", headers=headers)

    _assert_error(missing, status_code=404, error_code="not_found")
    _assert_error(wrong_method, status_code=405, error_code="method_not_allowed")


def test_invalid_job_uid_and_query_validation_return_422(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        invalid_job = client.put(
            "/v1/jobs/bad%20id",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        create = client.put(
            "/v1/jobs/job-logs-validation",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert create.status_code == 202
        logs_headers = _set_store(app, ["logs"])
        invalid_query = client.get(
            "/v1/jobs/job-logs-validation/logs?limit=0",
            headers=logs_headers,
        )

    _assert_error(invalid_job, status_code=422, error_code="validation_failed")
    _assert_error(invalid_query, status_code=422, error_code="validation_failed")


@pytest.mark.parametrize(
    ("method", "path", "scopes", "json_payload"),
    (
        ("PUT", "/v1/jobs/bad%2Fid", ["job_write"], {"job_spec": VALID_SPEC}),
        ("GET", "/v1/jobs/bad%2Fid", ["read"], None),
        ("GET", "/v1/jobs/bad%2Fid/logs", ["logs"], None),
        ("POST", "/v1/jobs/bad%2Fid/cancel", ["cancel"], {}),
        ("POST", "/v1/jobs/bad%2Fid/resume", ["resume"], {}),
    ),
)
def test_url_encoded_slash_job_uid_returns_422(
    tmp_path,
    method,
    path,
    scopes,
    json_payload,
):
    with support.client_context(tmp_path, scopes=scopes) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.request(method, path, json=json_payload, headers=headers)

    _assert_error(response, status_code=422, error_code="validation_failed")


def test_put_body_errors_return_planned_status_codes(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        bad_json = client.put(
            "/v1/jobs/job-bad-json",
            data=b"{broken",
            headers={**headers, "Content-Type": "application/json"},
        )
        too_large = client.put(
            "/v1/jobs/job-too-large",
            data=b"x" * 262145,
            headers={**headers, "Content-Type": "application/json"},
        )
        wrong_media = client.put(
            "/v1/jobs/job-wrong-media",
            data=json.dumps({"job_spec": VALID_SPEC}),
            headers={**headers, "Content-Type": "text/plain"},
        )
        wrong_top_level = client.put(
            "/v1/jobs/job-wrong-top",
            json=[],
            headers=headers,
        )
        missing_job_spec = client.put(
            "/v1/jobs/job-missing-spec",
            json={"idempotency_key": "client-key"},
            headers=headers,
        )

    _assert_error(bad_json, status_code=400, error_code="invalid_request")
    _assert_error(too_large, status_code=413, error_code="payload_too_large")
    _assert_error(wrong_media, status_code=415, error_code="unsupported_media_type")
    _assert_error(wrong_top_level, status_code=400, error_code="invalid_request")
    _assert_error(missing_job_spec, status_code=422, error_code="validation_failed")


def test_cancel_and_resume_body_errors_return_planned_status_codes(tmp_path):
    with support.client_context(tmp_path, scopes=["cancel"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        bad_json = client.post(
            "/v1/jobs/job-action/cancel",
            data=b"{broken",
            headers={**headers, "Content-Type": "application/json"},
        )
        too_large = client.post(
            "/v1/jobs/job-action/cancel",
            data=b"x" * 8193,
            headers={**headers, "Content-Type": "application/json"},
        )
        wrong_media = client.post(
            "/v1/jobs/job-action/cancel",
            data=json.dumps({"reason_code": "operator_requested"}),
            headers={**headers, "Content-Type": "text/plain"},
        )
        invalid_reason = client.post(
            "/v1/jobs/job-action/cancel",
            json={"reason_code": "Bad Reason"},
            headers=headers,
        )

    _assert_error(bad_json, status_code=400, error_code="invalid_request")
    _assert_error(too_large, status_code=413, error_code="payload_too_large")
    _assert_error(wrong_media, status_code=415, error_code="unsupported_media_type")
    _assert_error(invalid_reason, status_code=422, error_code="validation_failed")


def test_put_and_action_unknown_fields_return_validation_failed(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        unknown_put = client.put(
            "/v1/jobs/job-unknown-field",
            json={"job_spec": VALID_SPEC, "unexpected": "value"},
            headers=headers,
        )
        create = client.put(
            "/v1/jobs/job-action-unknown-field",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert create.status_code == 202
        cancel_headers = _set_store(app, ["cancel"])
        unknown_cancel = client.post(
            "/v1/jobs/job-action-unknown-field/cancel",
            json={"reason_code": "operator_requested", "reason": "not_allowed"},
            headers=cancel_headers,
        )
        resume_headers = _set_store(app, ["resume"])
        unknown_resume = client.post(
            "/v1/jobs/job-action-unknown-field/resume",
            json={"last_checkpoint_ref": "checkpoint:one", "extra_ref": "not_allowed"},
            headers=resume_headers,
        )

    _assert_error(unknown_put, status_code=422, error_code="validation_failed")
    _assert_error(unknown_cancel, status_code=422, error_code="validation_failed")
    _assert_error(unknown_resume, status_code=422, error_code="validation_failed")


def test_resume_rejects_db_blocked_checkpoint_ref_as_validation(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        create = client.put(
            "/v1/jobs/job-invalid-checkpoint-ref",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert create.status_code == 202
        resume_headers = _set_store(app, ["resume"])
        response = client.post(
            "/v1/jobs/job-invalid-checkpoint-ref/resume",
            json={"last_checkpoint_ref": _join("checkpoint:", "to", "ken", "ized")},
            headers=resume_headers,
        )

    _assert_error(response, status_code=422, error_code="validation_failed")


def test_resume_after_cancel_returns_conflict(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        create = client.put(
            "/v1/jobs/job-cancel-then-resume",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert create.status_code == 202
        cancel_headers = _set_store(app, ["cancel"])
        cancel = client.post(
            "/v1/jobs/job-cancel-then-resume/cancel",
            json={"reason_code": "operator_requested"},
            headers=cancel_headers,
        )
        assert cancel.status_code == 202
        resume_headers = _set_store(app, ["resume"])
        resume = client.post(
            "/v1/jobs/job-cancel-then-resume/resume",
            json={"reason_code": "retry"},
            headers=resume_headers,
        )

    _assert_error(resume, status_code=409, error_code="job_conflict")


def test_queue_config_invalid_keeps_internal_health_but_jobs_fail_closed(tmp_path):
    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        env={"KITT_WORKER_QUEUE_MAX_JOBS": "0"},
    ) as (client, _app, headers, _data, _run):
        internal = client.get("/internal/health")
        response = client.put(
            "/v1/jobs/job-queue-unavailable",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        read_headers = _set_store(_app, ["read"])
        capabilities = client.get("/v1/capabilities", headers=read_headers)

    assert internal.status_code == 200
    payload = _assert_error(response, status_code=503, error_code="queue_unavailable")
    cap_payload = capabilities.json()["data"]
    assert cap_payload["operational_status"]["queue_status"] == "unavailable"
    assert cap_payload["operational_status"]["queue_degraded_reason"] == "config_invalid"


def test_v1_disabled_has_priority_over_missing_credential_store(tmp_path):
    with support.client_context(
        tmp_path,
        env={"KITT_WORKER_ENABLE_V1": "false"},
        install_store=False,
    ) as (client, _app, _headers, _data, _run):
        response = client.get("/v1/health")
        internal = client.get("/internal/health")

    _assert_error(response, status_code=503, error_code="v1_disabled")
    assert internal.status_code == 200


def test_unexpected_queue_exception_degrades_without_leaking_diagnostics(tmp_path):
    leaked_store, leaked_credential = support.credential_store(["job_write"])
    leaked_hash = leaked_store.current.token_hash_sha256
    url_user = support.random_text(10)
    url_password = support.random_text(18)
    dsn_user = support.random_text(10)
    dsn_password = support.random_text(18)
    signed_url_value = support.random_text(24)
    signed_query_value = support.random_text(24)
    plain_access_value = support.random_text(24)
    plain_client_value = support.random_text(24)
    plain_refresh_value = support.random_text(24)
    bearer = "Bea" + "rer"
    access_key = "access_" + "to" + "ken"
    client_key = "client_" + "secret"
    refresh_key = "refresh_" + "to" + "ken"
    signature_key = "sig" + "nature"
    runtime_path = "runtime" + "/" + "artifact" + "/" + "redaction-marker"
    dsn = "post" + "gresql" + "://"
    url_scheme = "http" + "s" + "://"

    class ExplodingQueue:
        def queue_stats(self):
            raise RuntimeError(self._message())

        def put_job(self, **_kwargs):
            raise RuntimeError(self._message())

        def _message(self):
            return (
                f"Traceback leak {bearer} {leaked_credential} "
                f"hash {leaked_hash} "
                f"{access_key}={plain_access_value} "
                f"{client_key}={plain_client_value} "
                f"{refresh_key}={plain_refresh_value} "
                f"url {url_scheme}{url_user}:{url_password}@api.invalid/object"
                f"?{signature_key}={signed_url_value}&{access_key}={signed_query_value} "
                f"dsn {dsn}{dsn_user}:{dsn_password}@db.invalid/kitt "
                f"path {runtime_path}/redacted-test-marker.txt"
            )

    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        raise_server_exceptions=False,
    ) as (client, app, headers, _data, _run):
        app.state.queue_store = ExplodingQueue()
        app.state.queue_unavailable = None
        response = client.put(
            "/v1/jobs/job-queue-explodes",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )

    payload = _assert_error(response, status_code=503, error_code="queue_unavailable")
    body = response.text
    for forbidden in (
        leaked_credential,
        leaked_hash,
        url_user,
        url_password,
        dsn_user,
        dsn_password,
        signed_url_value,
        signed_query_value,
        plain_access_value,
        plain_client_value,
        plain_refresh_value,
        runtime_path,
        "Traceback",
        "RuntimeError",
    ):
        assert forbidden not in body
    assert len(payload["diagnostic"]) <= 256
