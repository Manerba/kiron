from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import artifact_staging
import publish_contract
import queue_store
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


CAPABILITY_HASH = "a" * 64
SOURCE_JOB_UID = "job-source-publish"
TRAINING_RUN_UID = "training-run-publish"
MODEL_VERSION_UID = "model-version-publish"
PUBLISH_JOB_UID = "publish-job-1"
PUBLISH_TARGET_REF = "kiron/test-model:sprint13"
PUBLISH_KEY_ID = "testkey01"

SFT_SOURCE_SPEC = {
    "schema_version": "kitt_job_spec_v1",
    "run_uid": TRAINING_RUN_UID,
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
}


def publish_keyring() -> tuple[publish_contract.PublishVerifyKeyring, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.from_private_bytes(os.urandom(32))
    key = publish_contract.PublishVerifyKey(
        key_id=PUBLISH_KEY_ID,
        public_key=publish_contract.public_key_bytes(private_key),
    )
    return publish_contract.PublishVerifyKeyring(current=key), private_key


def signed_publish_spec(
    *,
    artifacts: list[dict[str, Any]] | None = None,
    private_key: Ed25519PrivateKey | None = None,  # gitleaks:allow -- generated test key
) -> tuple[dict[str, Any], publish_contract.PublishVerifyKeyring]:
    keyring, generated_private_key = publish_keyring()
    signing_key = generated_private_key if private_key is None else private_key
    if private_key is not None:
        keyring = publish_contract.PublishVerifyKeyring(
            current=publish_contract.PublishVerifyKey(
                key_id=PUBLISH_KEY_ID,
                public_key=publish_contract.public_key_bytes(private_key),
            )
        )
    spec = {
        "schema_version": publish_contract.PUBLISH_JOB_SPEC_SCHEMA_VERSION,
        "job_type": publish_contract.PUBLISH_JOB_TYPE,
        "publish_job_uid": PUBLISH_JOB_UID,
        "publish_intent_uid": "publish-intent-1",
        "model_version_uid": MODEL_VERSION_UID,
        "training_run_uid": TRAINING_RUN_UID,
        "source_worker_job_uid": SOURCE_JOB_UID,
        "target": {
            "target_ref": PUBLISH_TARGET_REF,
            "ollama_model_name": "kiron/test-model",
            "ollama_tag": "sprint13",
        },
        "approval": {
            "approved": True,
            "approval_status": "approved",
            "eval_gate_status": "passed",
            "approval_ref": "approval-1",
        },
        "artifacts": artifacts or _default_artifact_declarations(),
        "lineage": {
            "parent_model_ref": "Qwen/Qwen2.5-7B-Instruct",
            "lineage_kind": "sft",
        },
        "capability_snapshot_hash_sha256": "c" * 64,
        "publish_spec_hash_sha256": "0" * 64,
        "signature_envelope": {},
    }
    envelope = publish_contract.build_signature_envelope(
        spec,
        key_id=PUBLISH_KEY_ID,
        private_key=signing_key,
    )
    spec["publish_spec_hash_sha256"] = envelope["publish_spec_hash_sha256"]
    spec["signature_envelope"] = envelope
    return spec, keyring


def publish_cfg(tmp_path: Path, *, ollama_bin: str | None = None) -> SimpleNamespace:
    data_dir = tmp_path
    staging = data_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging.chmod(0o750)
    return SimpleNamespace(
        data_dir=str(data_dir),
        artifact_staging_dir=str(staging),
        artifact_max_bytes=1024 * 1024,
        artifact_job_quota_bytes=2 * 1024 * 1024,
        artifact_total_quota_bytes=4 * 1024 * 1024,
        artifact_min_free_bytes=0,
        artifact_cleanup_after_seconds=0,
        artifact_config_error_code=None,
        ollama_bin=ollama_bin or "ollama",
    )


def stage_publish_artifacts(
    *,
    tmp_path: Path,
    store: queue_store.QueueStore,
    cfg: object,
    complete_source_job: bool = True,
    bind_run_lock: bool = True,
    run_lock_schema_version: str | None = "kitt_run_lock_v1",
) -> list[dict[str, Any]]:
    store.put_job(
        job_uid=SOURCE_JOB_UID,
        job_spec=SFT_SOURCE_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    if complete_source_job:
        store.next_runnable_job("worker.01")
        store.mark_running(job_uid=SOURCE_JOB_UID, owner="worker.01")
        store.mark_succeeded(job_uid=SOURCE_JOB_UID, owner="worker.01")
    stager = artifact_staging.ArtifactStager.from_config(cfg=cfg, store=store)
    gguf = b"GGUF" + bytes((index % 251) + 1 for index in range(256))
    modelfile = b"FROM ./model.gguf\nPARAMETER temperature 0\n"
    gguf_record = _stage_bytes(stager, "gguf", gguf)
    modelfile_record = _stage_bytes(stager, "ollama_modelfile", modelfile)
    run_lock_payload: dict[str, Any] = {
        "complete": True,
        "release_eligible": True,
        "blockers": [],
        "missing_fields": [],
    }
    if run_lock_schema_version is not None:
        run_lock_payload["schema_version"] = run_lock_schema_version
    if bind_run_lock:
        run_lock_payload.update(
            {
                "job_uid": SOURCE_JOB_UID,
                "training_run_uid": TRAINING_RUN_UID,
                "model_version_uid": MODEL_VERSION_UID,
                "parent_model_ref": "Qwen/Qwen2.5-7B-Instruct",
                "artifacts": {
                    "gguf": {
                        "sha256": gguf_record.sha256,
                        "size_bytes": gguf_record.size_bytes,
                    },
                    "ollama_modelfile": {
                        "sha256": modelfile_record.sha256,
                        "size_bytes": modelfile_record.size_bytes,
                    },
                },
            },
        )
    run_lock = json.dumps(
        run_lock_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_lock_record = _stage_bytes(stager, "run_lock", run_lock)
    return [
        _artifact_declaration("gguf", gguf_record),
        _artifact_declaration("ollama_modelfile", modelfile_record),
        _artifact_declaration("run_lock", run_lock_record),
    ]


def _stage_bytes(
    stager: artifact_staging.ArtifactStager,
    role: str,
    data: bytes,
) -> queue_store.ArtifactRecord:
    result = stager.stage_bytes(
        job_uid=SOURCE_JOB_UID,
        role=role,
        data=data,
        expected_size_bytes=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
    )
    assert result.artifact is not None
    assert result.artifact.status == "staged"
    return result.artifact


def _artifact_declaration(
    role: str,
    artifact: queue_store.ArtifactRecord,
) -> dict[str, Any]:
    return {
        "role": role,
        "artifact_ref": artifact.artifact_ref,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "source_worker_job_uid": SOURCE_JOB_UID,
        "training_run_uid": TRAINING_RUN_UID,
        "model_version_uid": MODEL_VERSION_UID,
    }


def _default_artifact_declarations() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for role in sorted(publish_contract.PUBLISH_REQUIRED_ARTIFACT_ROLES):
        payload = role.encode("ascii")
        result.append(
            {
                "role": role,
                "artifact_ref": f"artifact:{TRAINING_RUN_UID}:{role}:fixture",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "source_worker_job_uid": SOURCE_JOB_UID,
                "training_run_uid": TRAINING_RUN_UID,
                "model_version_uid": MODEL_VERSION_UID,
            }
        )
    return result
