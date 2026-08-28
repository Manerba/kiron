from __future__ import annotations

import json

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


def _set_store(app, scopes: list[str]) -> dict[str, str]:
    store, credential = support.credential_store(scopes)
    app.state.credential_store = store
    return support.authorization_header(credential)


def _assert_error(response, *, status_code: int, error_code: str):
    assert response.status_code == status_code
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == error_code
    return payload


def _queue_file_names(data_dir):
    return {path.name for path in data_dir.iterdir()}


def test_put_is_idempotent_by_job_uid_and_canonical_job_spec_hash_only(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        first = client.put(
            "/v1/jobs/job-idempotent",
            json={"job_spec": VALID_SPEC, "idempotency_key": "client-key-1"},
            headers=headers,
        )
        second = client.put(
            "/v1/jobs/job-idempotent",
            json={
                "job_spec": dict(reversed(list(VALID_SPEC.items()))),
                "idempotency_key": "client-key-2",
            },
            headers=headers,
        )

    assert first.status_code == 202
    assert second.status_code == 202
    first_data = first.json()["data"]
    second_data = second.json()["data"]
    assert first_data["run_uid"] == second_data["run_uid"]
    assert first_data["state"] == "queued"
    assert first_data["job_spec_hash_sha256"] == second_data["job_spec_hash_sha256"]
    assert "idempotency_key" not in json.dumps(first_data, sort_keys=True)
    assert "idempotency_key" not in json.dumps(second_data, sort_keys=True)


def test_put_accepts_boolean_training_hyperparameters(tmp_path):
    spec = {
        **VALID_SPEC,
        "hyperparameters": {
            **VALID_SPEC["hyperparameters"],
            "gradient_checkpointing": True,
        },
    }
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-bool-hyperparameter",
            json={"job_spec": spec},
            headers=headers,
        )

    assert response.status_code == 202


def test_put_same_job_uid_with_different_hash_conflicts(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        first = client.put(
            "/v1/jobs/job-conflict",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        conflict = client.put(
            "/v1/jobs/job-conflict",
            json={"job_spec": {**VALID_SPEC, "metadata": {"operator": "other"}}},
            headers=headers,
        )

    assert first.status_code == 202
    _assert_error(conflict, status_code=409, error_code="job_conflict")


def test_get_unknown_job_returns_job_not_found(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, headers, _data, _run):
        response = client.get("/v1/jobs/missing-job", headers=headers)

    _assert_error(response, status_code=404, error_code="job_not_found")


def test_logs_cancel_resume_and_runtime_dirs_remain_queue_only(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        data_dir,
        run_dir,
    ):
        create = client.put(
            "/v1/jobs/job-actions",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert create.status_code == 202
        resume_create = client.put(
            "/v1/jobs/job-actions-resume",
            json={
                "job_spec": {
                    **VALID_SPEC,
                    "labels": {"kitt_job": "job.actions.resume"},
                }
            },
            headers=headers,
        )
        assert resume_create.status_code == 202

        logs_headers = _set_store(app, ["logs"])
        logs = client.get("/v1/jobs/job-actions/logs", headers=logs_headers)
        assert logs.status_code == 200
        assert logs.json()["data"] == {
            "job_uid": "job-actions",
            "items": [],
            "next_cursor": None,
            "truncated": False,
            "execution_enabled": False,
        }

        cancel_headers = _set_store(app, ["cancel"])
        cancel = client.post(
            "/v1/jobs/job-actions/cancel",
            json={"reason_code": "operator_requested"},
            headers=cancel_headers,
        )
        assert cancel.status_code == 202
        assert cancel.json()["data"]["state"] == "cancel_requested"
        assert cancel.json()["data"]["cancel_requested"] is True
        assert cancel.json()["data"]["cancel_reason_code"] == "operator_requested"

        resume_headers = _set_store(app, ["resume"])
        resume = client.post(
            "/v1/jobs/job-actions-resume/resume",
            json={
                "reason_code": "checkpoint_available",
                "last_checkpoint_ref": "checkpoint:one",
            },
            headers=resume_headers,
        )
        assert resume.status_code == 202
        assert resume.json()["data"]["state"] == "resume_requested"
        assert resume.json()["data"]["resume_requested"] is True
        assert resume.json()["data"]["resume_reason_code"] == "checkpoint_available"
        assert resume.json()["data"]["resume_count"] == 1
        assert resume.json()["data"]["last_checkpoint_ref"] == "checkpoint:one"

        assert _queue_file_names(data_dir).issubset(
            {"queue.sqlite3", "queue.sqlite3-wal", "queue.sqlite3-shm"}
        )
        assert list(run_dir.iterdir()) == []


def test_new_testclient_reads_persisted_jobs_from_same_runtime_root(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        created = client.put(
            "/v1/jobs/job-persistent",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        assert created.status_code == 202
        run_uid = created.json()["data"]["run_uid"]

    with support.client_context(tmp_path, scopes=["read"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        persisted = client.get("/v1/jobs/job-persistent", headers=headers)

    assert persisted.status_code == 200
    assert persisted.json()["data"]["run_uid"] == run_uid


def test_queue_full_keeps_idempotent_put_allowed(tmp_path):
    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        env={"KITT_WORKER_QUEUE_MAX_JOBS": "1"},
    ) as (client, _app, headers, _data, _run):
        first = client.put(
            "/v1/jobs/job-full",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        idempotent = client.put(
            "/v1/jobs/job-full",
            json={"job_spec": dict(reversed(list(VALID_SPEC.items())))},
            headers=headers,
        )
        full = client.put(
            "/v1/jobs/job-full-other",
            json={"job_spec": {**VALID_SPEC, "labels": {"kitt_job": "job.other"}}},
            headers=headers,
        )

    assert first.status_code == 202
    assert idempotent.status_code == 202
    _assert_error(full, status_code=503, error_code="queue_full")
