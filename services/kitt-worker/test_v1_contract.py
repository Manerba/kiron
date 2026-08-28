from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest

import capabilities
import contract
import publish_test_support
import queue_store
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
    "input_reference": {
        "mode": "dataset_version",
        "dataset_version_uid": "5136e167-dbb7-49a6-a752-6fe1e962adb7",
    },
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


class OpenCapabilityProbes:
    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "cuda",
                    "count": 1,
                    "vram_total_bytes": 12 * 1024 * 1024 * 1024,
                    "vram_available_bytes": 10 * 1024 * 1024 * 1024,
                    "vram_budget_bytes": 0,
                    "capability_status": "unavailable",
                },
            ),
            host_ram_total_bytes=64 * 1024 * 1024 * 1024,
            host_ram_available_bytes=48 * 1024 * 1024 * 1024,
            staging_free_bytes=100 * 1024 * 1024 * 1024,
            measurement_status="complete",
        )

    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(
                *(
                    {"name": name, "version": "2.7.0", "available": True}
                    for name in (
                        "torch",
                        "transformers",
                        "datasets",
                        "peft",
                        "trl",
                        "accelerate",
                        "bitsandbytes",
                    )
                ),
            ),
            measurement_status="complete",
        )

    def probe_inference_conflict(self) -> capabilities.ConflictProbeResult:
        return capabilities.ConflictProbeResult(
            status="none",
            source_status="complete",
            maintenance="inactive",
        )


def _set_store(app, scopes: list[str]) -> dict[str, str]:
    store, credential = support.credential_store(scopes)
    app.state.credential_store = store
    return support.authorization_header(credential)


def _assert_envelope(response, *, status_code: int, ok: bool = True):
    assert response.status_code == status_code
    payload = response.json()
    assert payload["ok"] is ok
    assert payload["worker_id"] == "kiron-kitt-worker"
    assert payload["worker_contract_version"] == "adr-0008.v2"
    assert payload["server_time"].endswith("Z")
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]
    if ok:
        assert isinstance(payload["data"], dict)
        assert "error_code" not in payload
        assert "diagnostic" not in payload
    else:
        assert payload["data"] is None
        assert payload["error_code"]
        assert payload["diagnostic"]
    return payload


def _create_job(client, app, job_uid: str = "job-contract-1"):
    headers = _set_store(app, ["job_write"])
    response = client.put(
        f"/v1/jobs/{job_uid}",
        json={"job_spec": VALID_SPEC},
        headers=headers,
    )
    _assert_envelope(response, status_code=202)


def _write_sft_trainer(tmp_path: Path) -> Path:
    return Path(__file__).resolve().with_name("sft_trainer.py")


def _sft_command(tmp_path: Path) -> str:
    return f"{sys.executable} {_write_sft_trainer(tmp_path)}"


def _write_resource_catalog(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sft_resources.json").write_text(
        json.dumps(
            {
                "schema_version": "kiron_sft_resources_v1",
                "datasets": {
                    "5136e167-dbb7-49a6-a752-6fe1e962adb7": {"format": "jsonl"}
                },
                "models": {"Qwen/Qwen2.5-7B-Instruct": {"revision": "test"}},
                "training_profiles": {
                    "00000000-0000-4000-8000-000000000901:v1": {
                        "profile_hash_sha256": (
                            "0acbaef1d68b6085394669788b203878853d20c29217932506fd60ca5ea8c123"
                        ),
                        "model_limits": {
                            "max_model_parameters_b": 8,
                            "max_context_tokens": 4096,
                            "quantization_modes": ["qlora_4bit"],
                            "max_batch_size": 1,
                        },
                        "training_args": {
                            "max_seq_length": 4096,
                            "per_device_train_batch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "quantization_mode": "qlora_4bit",
                        },
                        "output_artifacts": {
                            "allow_merge": True,
                            "gguf_converter_path": sys.executable,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _open_gate_env(tmp_path: Path) -> dict[str, str]:
    _write_resource_catalog(tmp_path)
    staging = tmp_path / "data" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging.chmod(0o750)
    return {
        "KITT_WORKER_DISPATCH_GATE": "open",
        "KITT_WORKER_EXECUTOR_ENABLED": "true",
        "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
        "KITT_WORKER_SFT_COMMAND": _sft_command(tmp_path),
        "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging),
    }


@pytest.mark.parametrize(
    ("path", "scope", "expected"),
    [
        ("/v1/health", "read", {"status": "ok", "mode": "adr-0008-metadata-queue"}),
        ("/v1/heartbeat", "read", {"status": "ok", "execution_enabled": False}),
        (
            "/v1/capabilities",
            "read",
            {
                "schema_version": "kitt_worker_capabilities_v2",
                "valid_for_scheduling": False,
                "execution_enabled": False,
            },
        ),
    ],
)
def test_read_endpoints_return_adr_0008_envelope(tmp_path, path, scope, expected):
    with support.client_context(tmp_path, scopes=[scope]) as (client, _app, headers, _data, _run):
        response = client.get(path, headers=headers)

    payload = _assert_envelope(response, status_code=200)
    for key, value in expected.items():
        assert payload["data"][key] == value


def test_read_endpoints_report_open_operator_gate(tmp_path):
    env = _open_gate_env(tmp_path)
    with support.client_context(
        tmp_path,
        scopes=["read"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        health = client.get("/v1/health", headers=headers)
        heartbeat = client.get("/v1/heartbeat", headers=headers)
        capabilities = client.get("/v1/capabilities", headers=headers)

    assert _assert_envelope(health, status_code=200)["data"]["dispatch_enabled"] is True
    assert _assert_envelope(heartbeat, status_code=200)["data"]["execution_enabled"] is True
    data = _assert_envelope(capabilities, status_code=200)["data"]
    assert data["valid_for_scheduling"] is True
    assert data["execution_enabled"] is True
    assert data["scheduling_gate"]["status"] == "open"
    assert data["parallelism"]["gpu_slots"] == 1
    assert data["hardware"]["accelerators"][0]["kind"] == "cuda"
    assert data["hardware"]["accelerators"][0]["vram_budget_bytes"] > 0


def test_read_endpoints_keep_execution_closed_when_capabilities_block(tmp_path):
    env = {
        "KITT_WORKER_DISPATCH_GATE": "open",
        "KITT_WORKER_EXECUTOR_ENABLED": "true",
        "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
        "KITT_WORKER_SFT_COMMAND": _sft_command(tmp_path),
    }
    with support.client_context(tmp_path, scopes=["read"], env=env) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        health = client.get("/v1/health", headers=headers)
        heartbeat = client.get("/v1/heartbeat", headers=headers)
        capabilities = client.get("/v1/capabilities", headers=headers)

    assert _assert_envelope(health, status_code=200)["data"]["dispatch_enabled"] is False
    assert _assert_envelope(heartbeat, status_code=200)["data"]["execution_enabled"] is False
    data = _assert_envelope(capabilities, status_code=200)["data"]
    assert data["valid_for_scheduling"] is False
    assert data["execution_enabled"] is False
    assert data["scheduling_gate"]["status"] == "closed"
    assert "resource_unknown" in data["operational_status"]["blocking_reasons"]


def test_put_job_reports_execution_enabled_when_operator_gate_is_open(tmp_path):
    env = _open_gate_env(tmp_path)
    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-open-gate",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )

    payload = _assert_envelope(response, status_code=202)
    assert payload["data"]["execution_enabled"] is True


def test_put_job_rejects_legacy_spec_when_operator_gate_is_open(tmp_path):
    env = _open_gate_env(tmp_path)
    legacy_spec = {
        "job_kind": "metadata_only",
        "model_ref": "model.v1",
        "dataset_ref": "dataset.v1",
        "trainer_profile_ref": "trainer.v1",
        "limits": {"max_steps": 0},
        "labels": {"kitt_job": "job.v1"},
        "metadata": {"operator": "kiron"},
    }
    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-legacy-open-gate",
            json={"job_spec": legacy_spec},
            headers=headers,
        )

    _assert_envelope(response, status_code=422, ok=False)


def test_put_job_rejects_unresolvable_empty_kitt_required_objects(tmp_path):
    env = _open_gate_env(tmp_path)
    bad_spec = {
        **VALID_SPEC,
        "training_profile": {},
        "input_reference": {},
        "base_or_parent_model": {},
        "hyperparameters": {},
    }
    with support.client_context(
        tmp_path,
        scopes=["job_write"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-empty-required-objects",
            json={"job_spec": bad_spec},
            headers=headers,
        )

    _assert_envelope(response, status_code=422, ok=False)


def test_put_job_returns_accepted_envelope_with_hash(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-contract-put",
            json={"job_spec": VALID_SPEC, "idempotency_key": "client-key-1"},
            headers=headers,
        )

    payload = _assert_envelope(response, status_code=202)
    assert payload["data"]["job_uid"] == "job-contract-put"
    assert payload["data"]["state"] == "queued"
    assert payload["data"]["job_type"] == "sft"
    assert payload["data"]["run_uid"].startswith("run_")
    assert payload["data"]["execution_state_version"] == 2
    assert payload["data"]["execution_enabled"] is False
    assert payload["data"]["started_at"] is None
    assert payload["data"]["finished_at"] is None
    assert payload["data"]["terminal_at"] is None
    assert payload["data"]["runner_kind"] == "none"
    assert payload["data"]["last_failure_class"] is None
    assert len(payload["data"]["job_spec_hash_sha256"]) == 64
    assert "idempotency_key" not in payload["data"]
    assert not any("artifact" in key for key in payload["data"])
    assert set(payload["data"]) == {
        "job_uid",
        "run_uid",
        "job_type",
        "state",
        "execution_state_version",
        "execution_enabled",
        "job_spec_hash_sha256",
        "capability_hash_sha256",
        "created_at",
        "updated_at",
        "accepted_at",
        "queued_at",
        "started_at",
        "finished_at",
        "terminal_at",
        "cancel_requested",
        "cancel_reason_code",
        "resume_requested",
        "resume_reason_code",
        "resume_count",
        "resume_limit",
        "last_checkpoint_ref",
        "attempt_count",
        "last_failure_code",
        "last_failure_class",
        "runner_kind",
        "lease_status",
        "lease_expires_at",
    }


def test_get_job_includes_staged_artifact_refs_without_paths(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-contract-artifacts",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        _assert_envelope(response, status_code=202)
        reserved = app.state.queue_store.reserve_artifact(
            job_uid="job-contract-artifacts",
            role="ollama_modelfile",
            expected_size_bytes=40,
            quotas=queue_store.ArtifactQuotaSettings(
                max_bytes=1024,
                job_quota_bytes=4096,
                total_quota_bytes=8192,
                min_free_bytes=0,
            ),
            staging_free_bytes=1000,
        )
        staged = app.state.queue_store.mark_artifact_staged(
            artifact_uid=reserved.artifact_uid,
            sha256="a" * 64,
            size_bytes=40,
        )
        read_headers = _set_store(app, ["read"])
        payload = _assert_envelope(
            client.get("/v1/jobs/job-contract-artifacts", headers=read_headers),
            status_code=200,
        )

    assert payload["data"]["artifacts"] == [
        {
            "role": "ollama_modelfile",
            "artifact_ref": staged.artifact_ref,
            "sha256": "a" * 64,
            "size_bytes": 40,
            "status": "staged",
            "verified_at": staged.verified_at,
        }
    ]
    assert "/" not in payload["data"]["artifacts"][0]["artifact_ref"]
    assert "://" not in payload["data"]["artifacts"][0]["artifact_ref"]


def test_put_job_artifact_stages_small_publish_artifact(tmp_path):
    data = b"FROM ./model.gguf\nPARAMETER temperature 0\n"
    digest = hashlib.sha256(data).hexdigest()
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        data_dir,
        _run,
    ):
        staging_dir = data_dir / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.chmod(0o750)
        response = client.put(
            "/v1/jobs/job-contract-artifact-upload",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        _assert_envelope(response, status_code=202)
        app.state.queue_store.next_runnable_job("worker.01")
        app.state.queue_store.mark_running(
            job_uid="job-contract-artifact-upload",
            owner="worker.01",
        )
        app.state.queue_store.mark_succeeded(
            job_uid="job-contract-artifact-upload",
            owner="worker.01",
        )

        upload = client.put(
            "/v1/jobs/job-contract-artifact-upload/artifacts/ollama_modelfile",
            json={
                "sha256": digest,
                "size_bytes": len(data),
                "content_base64": base64.b64encode(data).decode("ascii"),
            },
            headers=headers,
        )

    payload = _assert_envelope(upload, status_code=202)
    artifact = payload["data"]["artifact"]
    assert artifact["role"] == "ollama_modelfile"
    assert artifact["sha256"] == digest
    assert artifact["size_bytes"] == len(data)
    assert artifact["status"] == "staged"
    assert "/" not in artifact["artifact_ref"]
    assert "://" not in artifact["artifact_ref"]


def test_put_job_artifact_rejects_hash_mismatch(tmp_path):
    data = b"FROM ./model.gguf\n"
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        data_dir,
        _run,
    ):
        staging_dir = data_dir / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.chmod(0o750)
        response = client.put(
            "/v1/jobs/job-contract-artifact-hash",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        _assert_envelope(response, status_code=202)
        app.state.queue_store.next_runnable_job("worker.01")
        app.state.queue_store.mark_running(
            job_uid="job-contract-artifact-hash",
            owner="worker.01",
        )
        app.state.queue_store.mark_succeeded(
            job_uid="job-contract-artifact-hash",
            owner="worker.01",
        )

        upload = client.put(
            "/v1/jobs/job-contract-artifact-hash/artifacts/ollama_modelfile",
            json={
                "sha256": "a" * 64,
                "size_bytes": len(data),
                "content_base64": base64.b64encode(data).decode("ascii"),
            },
            headers=headers,
        )

    payload = _assert_envelope(upload, status_code=422, ok=False)
    assert payload["error_code"] == "artifact_hash_mismatch"


def test_put_publish_job_requires_signature_keyring(tmp_path):
    spec, _keyring = publish_test_support.signed_publish_spec()
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        response = client.put(
            "/v1/jobs/job-publish-no-key",
            json={"job_spec": spec},
            headers=headers,
        )

    payload = _assert_envelope(response, status_code=422, ok=False)
    assert payload["error_code"] == "publish_spec_signature_key_unavailable"


def test_put_publish_job_rejects_wrong_signature_purpose(tmp_path):
    spec, keyring = publish_test_support.signed_publish_spec()
    spec["signature_envelope"] = {
        **spec["signature_envelope"],
        "purpose": "wrong-purpose",
    }
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        app.state.publish_verify_keyring = keyring
        response = client.put(
            "/v1/jobs/job-publish-wrong-purpose",
            json={"job_spec": spec},
            headers=headers,
        )

    payload = _assert_envelope(response, status_code=422, ok=False)
    assert payload["error_code"] == "publish_spec_signature_shape_invalid"


def test_put_publish_job_accepts_signed_spec_and_returns_publish_shape(tmp_path):
    spec, keyring = publish_test_support.signed_publish_spec()
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        app.state.publish_verify_keyring = keyring
        response = client.put(
            "/v1/jobs/job-publish-contract",
            json={"job_spec": spec},
            headers=headers,
        )
        read_headers = _set_store(app, ["read"])
        read_response = client.get("/v1/jobs/job-publish-contract", headers=read_headers)

    payload = _assert_envelope(response, status_code=202)
    assert payload["data"]["job_type"] == "ollama_publish"
    assert payload["data"]["runner_kind"] == "none"
    assert payload["data"]["publish_status"] == "pending"
    assert payload["data"]["target_ref"] == spec["target"]["target_ref"]
    assert payload["data"]["publish_result"] is None
    read_payload = _assert_envelope(read_response, status_code=200)
    assert read_payload["data"]["job_type"] == "ollama_publish"
    assert read_payload["data"]["target_ref"] == spec["target"]["target_ref"]


def test_configured_publish_keyring_makes_publish_ready_and_accepts_spec(tmp_path):
    env = _open_gate_env(tmp_path)
    spec, keyring = publish_test_support.signed_publish_spec()
    keyring_path = tmp_path / "data" / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": keyring.current.key_id,
                    "public_key_hex": keyring.current.public_key.hex(),
                    "active": True,
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)
    with support.client_context(
        tmp_path,
        scopes=["job_write", "read"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        capabilities_response = client.get("/v1/capabilities", headers=headers)
        response = client.put(
            "/v1/jobs/job-publish-configured-keyring",
            json={"job_spec": spec},
            headers=headers,
        )

    capabilities_payload = _assert_envelope(capabilities_response, status_code=200)
    assert capabilities_payload["data"]["operations"]["publish"]["ready"] is True
    payload = _assert_envelope(response, status_code=202)
    assert payload["data"]["job_type"] == "ollama_publish"


def test_put_publish_job_reloads_configured_keyring_for_rotation(tmp_path):
    env = _open_gate_env(tmp_path)
    old_spec, old_keyring = publish_test_support.signed_publish_spec()
    new_spec, new_keyring = publish_test_support.signed_publish_spec()
    keyring_path = tmp_path / "data" / "publish-verify.json"
    _write_publish_keyring_file(keyring_path, old_keyring)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)

    with support.client_context(
        tmp_path,
        scopes=["job_write", "read"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        keyring_path.write_text("{}", encoding="utf-8")
        invalid_response = client.put(
            "/v1/jobs/job-publish-invalid-rotated-keyring",
            json={"job_spec": old_spec},
            headers=headers,
        )
        _write_publish_keyring_file(keyring_path, new_keyring)
        old_response = client.put(
            "/v1/jobs/job-publish-old-rotated-keyring",
            json={"job_spec": old_spec},
            headers=headers,
        )
        new_response = client.put(
            "/v1/jobs/job-publish-new-rotated-keyring",
            json={"job_spec": new_spec},
            headers=headers,
        )

    invalid_payload = _assert_envelope(invalid_response, status_code=422, ok=False)
    old_payload = _assert_envelope(old_response, status_code=422, ok=False)
    new_payload = _assert_envelope(new_response, status_code=202)

    assert invalid_payload["error_code"] == "publish_spec_signature_key_unavailable"
    assert old_payload["error_code"] == "publish_spec_signature_invalid"
    assert new_payload["data"]["job_type"] == "ollama_publish"


def _write_publish_keyring_file(path: Path, keyring) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": keyring.current.key_id,
                    "public_key_hex": keyring.current.public_key.hex(),
                    "active": True,
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def test_capabilities_and_heartbeat_report_active_worker_job(tmp_path):
    env = _open_gate_env(tmp_path)
    _spec, keyring = publish_test_support.signed_publish_spec()
    keyring_path = tmp_path / "data" / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": keyring.current.key_id,
                    "public_key_hex": keyring.current.public_key.hex(),
                    "active": True,
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)
    with support.client_context(
        tmp_path,
        scopes=["read"],
        env=env,
        capability_probes=OpenCapabilityProbes(),
    ) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        store = app.state.queue_store
        store.put_job(
            job_uid="job-active-sft",
            job_spec=VALID_SPEC,
            capability_hash_sha256="a" * 64,
        )
        store.next_runnable_job("worker.01")
        store.mark_running(job_uid="job-active-sft", owner="worker.01")
        heartbeat_response = client.get("/v1/heartbeat", headers=headers)
        capabilities_response = client.get("/v1/capabilities", headers=headers)

    heartbeat_payload = _assert_envelope(heartbeat_response, status_code=200)
    capabilities_payload = _assert_envelope(capabilities_response, status_code=200)

    assert heartbeat_payload["data"]["active_jobs"] == 1
    data = capabilities_payload["data"]
    assert data["operational_status"]["active_jobs"] == 1
    assert "worker_active_job" in data["operational_status"]["blocking_reasons"]
    assert data["operations"]["publish"]["ready"] is False
    assert data["operations"]["publish"]["blocked_reason"] == "worker_active_job"


def test_job_get_logs_cancel_and_resume_return_envelopes(tmp_path):
    with support.client_context(tmp_path, scopes=["job_write"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        create_response = client.put(
            "/v1/jobs/job-contract-flow",
            json={"job_spec": VALID_SPEC},
            headers=headers,
        )
        _assert_envelope(create_response, status_code=202)
        resume_create_response = client.put(
            "/v1/jobs/job-contract-resume",
            json={
                "job_spec": {
                    **VALID_SPEC,
                    "labels": {"kitt_job": "job.contract.resume"},
                }
            },
            headers=headers,
        )
        _assert_envelope(resume_create_response, status_code=202)

        read_headers = _set_store(app, ["read"])
        get_payload = _assert_envelope(
            client.get("/v1/jobs/job-contract-flow", headers=read_headers),
            status_code=200,
        )
        assert get_payload["data"]["job_uid"] == "job-contract-flow"
        assert not any("artifact" in key for key in get_payload["data"])

        logs_headers = _set_store(app, ["logs"])
        logs_payload = _assert_envelope(
            client.get("/v1/jobs/job-contract-flow/logs", headers=logs_headers),
            status_code=200,
        )
        assert logs_payload["data"]["items"] == []
        assert logs_payload["data"]["next_cursor"] is None

        cancel_headers = _set_store(app, ["cancel"])
        cancel_payload = _assert_envelope(
            client.post(
                "/v1/jobs/job-contract-flow/cancel",
                json={"reason_code": "operator_requested"},
                headers=cancel_headers,
            ),
            status_code=202,
        )
        assert cancel_payload["data"]["cancel_requested"] is True
        assert cancel_payload["data"]["cancel_reason_code"] == "operator_requested"

        resume_headers = _set_store(app, ["resume"])
        resume_payload = _assert_envelope(
            client.post(
                "/v1/jobs/job-contract-resume/resume",
                json={
                    "reason_code": "checkpoint_available",
                    "last_checkpoint_ref": "checkpoint:contract",
                },
                headers=resume_headers,
            ),
            status_code=202,
        )
        assert resume_payload["data"]["resume_requested"] is True
        assert resume_payload["data"]["resume_reason_code"] == "checkpoint_available"
        assert resume_payload["data"]["last_checkpoint_ref"] == "checkpoint:contract"


def test_request_id_header_is_reused_or_generated(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, headers, _data, _run):
        valid = client.get(
            "/v1/health",
            headers={**headers, "X-Request-ID": "trace_1:abc.def"},
        )
        missing = client.get("/v1/health", headers=headers)
        invalid = client.get(
            "/v1/health",
            headers={**headers, "X-Request-ID": "bad/path"},
        )
        credential_like = client.get(
            "/v1/health",
            headers={**headers, "X-Request-ID": f"kidvalue01.{'s' * 40}"},
        )
        credential_substring = client.get(
            "/v1/health",
            headers={**headers, "X-Request-ID": f"trace:kidvalue01.{'s' * 40}"},
        )

    valid_payload = _assert_envelope(valid, status_code=200)
    assert valid_payload["request_id"] == "trace_1:abc.def"
    missing_payload = _assert_envelope(missing, status_code=200)
    assert missing_payload["request_id"].startswith("req_")
    invalid_payload = _assert_envelope(invalid, status_code=200)
    assert invalid_payload["request_id"].startswith("req_")
    assert invalid_payload["request_id"] != "bad/path"
    credential_payload = _assert_envelope(credential_like, status_code=200)
    assert credential_payload["request_id"].startswith("req_")
    assert credential_payload["request_id"] != f"kidvalue01.{'s' * 40}"
    assert credential_like.headers["X-Request-ID"] == credential_payload["request_id"]
    credential_substring_payload = _assert_envelope(
        credential_substring,
        status_code=200,
    )
    assert credential_substring_payload["request_id"].startswith("req_")
    assert credential_substring_payload["request_id"] != f"trace:kidvalue01.{'s' * 40}"
    assert (
        credential_substring.headers["X-Request-ID"]
        == credential_substring_payload["request_id"]
    )


def test_contract_diagnostic_redacts_dsns_signed_urls_and_raw_payload_markers():
    diagnostic = (
        "dsn=postgresql://db.internal/kitt "
        "url=https://example.invalid/object?signature=abc&access_key=def "
        "raw_prompt=hello raw_response=world cp_raw_text=body "
        "canonical_job_spec_json={} artifact_bytes=deadbeef "
        '"raw_prompt": "hello secret", "artifact_bytes": "cafebabe" '
        'canonical_job_spec_json={"cp_raw_text":"nested body",'
        '"nested":{"raw_response":"nested world"}} '
        "cp_raw_text: multi word payload"
    )

    redacted = contract.sanitize_diagnostic(diagnostic)

    for forbidden in (
        "postgresql://",
        "signature=abc",
        "access_key=def",
        "raw_prompt",
        "raw_response",
        "cp_raw_text",
        "canonical_job_spec_json",
        "artifact_bytes",
        "hello",
        "world",
        "cafebabe",
        "nested body",
        "nested world",
        "multi word payload",
        "deadbeef",
    ):
        assert forbidden not in redacted


def test_capabilities_returns_sprint_5_snapshot_in_adr_0008_envelope(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (
        client,
        _app,
        headers,
        data_dir,
        run_dir,
    ):
        response = client.get(
            "/v1/capabilities",
            headers={**headers, "X-Request-ID": "capability-contract-1"},
        )

    payload = _assert_envelope(response, status_code=200)
    assert payload["request_id"] == "capability-contract-1"
    data = payload["data"]
    assert data["snapshot_version"] == 1
    assert data["snapshot_time"] == "2026-06-15T12:00:00.000Z"
    assert data["valid_until"] == "2026-06-15T12:01:00.000Z"
    assert data["ttl_seconds"] == 60
    assert len(data["capability_hash_sha256"]) == 64
    assert data["capability_mode"] == "policy_snapshot"
    assert data["valid_for_scheduling"] is False
    assert data["execution_enabled"] is False
    assert data["scheduling_gate"]["reason_code"] == "tareas_dispatch_gate_closed"
    assert data["parallelism"]["max_concurrent_jobs"] == 0
    assert data["parallelism"]["gpu_slots"] == 0
    assert data["parallelism"]["queue_enabled"] is True
    assert data["parallelism"]["lease_enabled"] is True
    assert data["parallelism"]["resume_persistence_enabled"] is True
    assert data["parallelism"]["inference_conflict_policy"] == "block_training"
    assert data["operational_status"]["queue_status"] == "ready"
    assert data["operational_status"]["queue_depth"] == 0
    assert data["operational_status"]["queue_degraded_reason"] is None
    assert data["operational_status"]["inference_conflict"]["policy_decision"] == "block_training"
    assert "tareas_dispatch_gate_closed" in data["operational_status"]["blocking_reasons"]
    for section in (
        "hardware",
        "operations",
        "model_limits",
        "trainer_stack",
        "parallelism",
        "operational_status",
    ):
        assert section in data
    body = json.dumps(data, sort_keys=True)
    for forbidden in (
        "/" + "opt" + "/kiron",
        "/" + "usr" + "/lib" + "/kiron",
        "/" + "run" + "/kiron",
        "/" + "etc" + "/kiron",
        "Authori" + "zation",
        "Bea" + "rer ",
        "credential_hash",
        "pass" + "word",
    ):
        assert forbidden not in body
    assert {path.name for path in data_dir.iterdir()}.issubset(
        {"queue.sqlite3", "queue.sqlite3-wal", "queue.sqlite3-shm"}
    )
    assert list(run_dir.iterdir()) == []


def test_internal_health_remains_internal_plain_response(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (client, _app, _headers, _data, _run):
        response = client.get("/internal/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "kitt-worker",
        "mode": "skeleton",
    }
    assert "X-Request-ID" not in response.headers
