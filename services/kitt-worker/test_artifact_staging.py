from __future__ import annotations

from dataclasses import asdict
import errno
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import artifact_staging
import config
import queue_store


CAPABILITY_HASH = "a" * 64
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


def _runtime(tmp_path: Path):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    staging_dir = data_dir / "staging"
    data_dir.mkdir()
    run_dir.mkdir()
    staging_dir.mkdir()
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    staging_dir.chmod(0o750)
    env = {
        "KITT_WORKER_DATA_DIR": str(data_dir),
        "KITT_WORKER_RUN_DIR": str(run_dir),
        "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
        "KITT_WORKER_ARTIFACT_MAX_BYTES": "100",
        "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "100",
        "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "200",
        "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "0",
        "KITT_WORKER_ARTIFACT_CLEANUP_AFTER_SECONDS": "0",
    }
    cfg = config.load_config(env, data_root=data_dir, run_root=run_dir)
    return cfg, data_dir, run_dir, staging_dir


def _store_with_job(data_dir: Path, job_uid: str = "job-artifact"):
    store = queue_store.QueueStore.open(
        data_dir,
        queue_store.QueueSettings(
            max_jobs=10,
            lease_ttl_seconds=30,
            resume_limit=2,
            sqlite_busy_timeout_ms=1000,
        ),
    )
    store.put_job(
        job_uid=job_uid,
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    return store


def _stager(cfg, store, free_bytes: int = 10_000):
    return artifact_staging.ArtifactStager.from_config(
        cfg=cfg,
        store=store,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        disk_usage_fn=lambda _path: SimpleNamespace(free=free_bytes),
    )


def test_staging_root_validation_rejects_missing_mode_symlink_and_escape(tmp_path):
    cfg, data_dir, run_dir, staging_dir = _runtime(tmp_path)
    validated = artifact_staging.validate_staging_root(
        cfg,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        disk_usage_fn=lambda _path: SimpleNamespace(free=1234),
    )
    assert validated.root == staging_dir.resolve()
    assert validated.free_bytes == 1234

    staging_dir.chmod(0o755)
    with pytest.raises(artifact_staging.ArtifactStagingError) as excinfo:
        artifact_staging.validate_staging_root(
            cfg,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )
    assert excinfo.value.reason_code == "artifact_staging_unavailable"
    assert str(tmp_path) not in str(excinfo.value)

    staging_dir.chmod(0o750)
    staging_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o750)
    staging_dir.symlink_to(outside, target_is_directory=True)
    symlink_cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
        },
        data_root=data_dir,
        run_root=run_dir,
    )
    with pytest.raises(artifact_staging.ArtifactStagingError):
        artifact_staging.validate_staging_root(symlink_cfg)
    assert symlink_cfg.artifact_config_error_code == "artifact_staging_unavailable"


def test_staging_root_validation_rejects_symlink_to_internal_target(tmp_path):
    _cfg, data_dir, run_dir, staging_dir = _runtime(tmp_path)
    staging_dir.rmdir()
    internal_target = data_dir / "internal-staging-target"
    internal_target.mkdir()
    internal_target.chmod(0o750)
    staging_dir.symlink_to(internal_target, target_is_directory=True)

    symlink_cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
        },
        data_root=data_dir,
        run_root=run_dir,
    )

    assert symlink_cfg.artifact_config_error_code == "artifact_staging_unavailable"
    with pytest.raises(artifact_staging.ArtifactStagingError):
        artifact_staging.validate_staging_root(symlink_cfg)


def test_staging_root_validation_rejects_broken_symlink(tmp_path):
    _cfg, data_dir, run_dir, staging_dir = _runtime(tmp_path)
    staging_dir.rmdir()
    staging_dir.symlink_to(data_dir / "missing-target", target_is_directory=True)

    symlink_cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
        },
        data_root=data_dir,
        run_root=run_dir,
    )

    assert symlink_cfg.artifact_config_error_code == "artifact_staging_unavailable"


def test_direct_staging_root_validation_rejects_raw_symlink_path(tmp_path):
    _cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    staging_dir.rmdir()
    internal_target = data_dir / "direct-internal-target"
    internal_target.mkdir()
    internal_target.chmod(0o750)
    staging_dir.symlink_to(internal_target, target_is_directory=True)
    direct_cfg = SimpleNamespace(
        artifact_config_error_code=None,
        artifact_staging_dir=staging_dir,
        data_dir=data_dir,
    )

    with pytest.raises(artifact_staging.ArtifactStagingError) as excinfo:
        artifact_staging.validate_staging_root(direct_cfg)
    assert excinfo.value.reason_code == "artifact_staging_unavailable"


def test_hash_size_verification_finalizes_atomically_without_path_metadata(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    data = b'{"loss":0.1}\n'
    expected_sha = hashlib.sha256(data).hexdigest()

    result = stager.stage_bytes(
        job_uid="job-artifact",
        role="metrics_jsonl",
        data=data,
        expected_size_bytes=len(data),
        expected_sha256=expected_sha,
    )

    assert result.status == "staged"
    assert result.artifact is not None
    assert result.artifact.sha256 == expected_sha
    assert result.artifact.size_bytes == len(data)
    assert result.artifact.reserved_size_bytes == 0
    assert result.artifact.artifact_ref is not None
    assert "/" not in result.artifact.artifact_ref
    assert "://" not in result.artifact.artifact_ref
    final_path = staging_dir / "objects" / result.artifact.artifact_uid
    assert final_path.read_bytes() == data
    assert oct(final_path.stat().st_mode & 0o777) == "0o640"
    metadata = json.dumps(asdict(result.artifact), sort_keys=True)
    assert str(tmp_path) not in metadata
    assert "http://" not in metadata
    assert "token" not in metadata.lower()


def test_stage_file_moves_local_runner_artifact_without_copy_space(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(tmp_path / "run"),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
            "KITT_WORKER_ARTIFACT_MAX_BYTES": "100",
            "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "100",
            "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "100",
            "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "50",
        },
        data_root=data_dir,
        run_root=tmp_path / "run",
    )
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store, free_bytes=60)
    source = data_dir / "work" / "job-artifact" / "model.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * 80)
    expected_sha = hashlib.sha256(b"x" * 80).hexdigest()

    result = stager.stage_file(
        job_uid="job-artifact",
        role="gguf",
        source_path=source,
        expected_size_bytes=80,
        expected_sha256=expected_sha,
    )

    assert result.status == "staged"
    assert result.artifact is not None
    assert result.artifact.sha256 == expected_sha
    assert result.artifact.size_bytes == 80
    assert not source.exists()
    final_path = staging_dir / "objects" / result.artifact.artifact_uid
    assert final_path.read_bytes() == b"x" * 80
    assert oct(final_path.stat().st_mode & 0o777) == "0o640"


def test_hash_and_size_mismatch_release_reservation_and_do_not_publish_file(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)

    size_result = stager.stage_bytes(
        job_uid="job-artifact",
        role="metrics_jsonl",
        data=b"abc",
        expected_size_bytes=4,
    )
    hash_result = stager.stage_bytes(
        job_uid="job-artifact",
        role="manifest",
        data=b"abcd",
        expected_size_bytes=4,
        expected_sha256="0" * 64,
    )

    assert size_result.status == "verification_failed"
    assert size_result.artifact is not None
    assert size_result.artifact.failure_code == "artifact_size_mismatch"
    assert size_result.artifact.reserved_size_bytes == 0
    assert hash_result.status == "verification_failed"
    assert hash_result.artifact is not None
    assert hash_result.artifact.failure_code == "artifact_hash_mismatch"
    assert not list((staging_dir / "objects").glob("*"))
    assert store.artifact_quota_usage(job_uid="job-artifact")["job_bytes"] == 0


def test_quota_reservations_block_parallel_stages_and_min_free(tmp_path):
    cfg, data_dir, _run_dir, _staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store, free_bytes=10_000)

    first = stager.store.reserve_artifact(
        job_uid="job-artifact",
        role="metrics_jsonl",
        expected_size_bytes=60,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    second = stager.store.reserve_artifact(
        job_uid="job-artifact",
        role="manifest",
        expected_size_bytes=60,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    min_free_blocked = stager.store.reserve_artifact(
        job_uid="job-artifact",
            role="merged_weights",
        expected_size_bytes=1,
        quotas=queue_store.ArtifactQuotaSettings(
            max_bytes=100,
            job_quota_bytes=200,
            total_quota_bytes=200,
            min_free_bytes=10_000,
        ),
        staging_free_bytes=60,
    )

    assert first.status == "staging"
    assert second.status == "quota_blocked"
    assert second.failure_code == "artifact_job_quota_exceeded"
    assert min_free_blocked.status == "quota_blocked"
    assert min_free_blocked.failure_code == "artifact_min_free_blocked"


def test_internal_staging_dir_failure_does_not_reserve_quota(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    blocking_file = staging_dir / "tmp"
    blocking_file.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(artifact_staging.ArtifactStagingError) as excinfo:
        stager.stage_bytes(
            job_uid="job-artifact",
            role="metrics_jsonl",
            data=b"abc",
            expected_size_bytes=3,
        )

    assert excinfo.value.reason_code == "artifact_staging_unavailable"
    assert store.list_artifacts_for_cleanup() == []
    assert store.artifact_quota_usage(job_uid="job-artifact")["job_bytes"] == 0


def test_disk_full_recovery_clears_partial_truth_and_temp_file(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)

    class DiskFullStager(artifact_staging.ArtifactStager):
        def _write_stream(self, temp_path, chunks, *, expected_size_bytes):
            temp_path.write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "disk full /usr/lib/kiron/leak")

    stager = DiskFullStager(
        store=store,
        root=artifact_staging.validate_staging_root(
            cfg,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            disk_usage_fn=lambda _path: SimpleNamespace(free=10_000),
        ),
        quotas=artifact_staging.quota_settings_from_config(cfg),
    )
    result = stager.stage_bytes(
        job_uid="job-artifact",
        role="metrics_jsonl",
        data=b"abc",
        expected_size_bytes=3,
    )

    assert result.status == "disk_full"
    assert result.artifact is not None
    assert result.artifact.sha256 is None
    assert result.artifact.size_bytes is None
    assert result.artifact.expected_size_bytes is None
    assert result.artifact.reserved_size_bytes == 0
    assert not list((staging_dir / "tmp").glob("*"))
    assert str(tmp_path) not in json.dumps(asdict(result.artifact), sort_keys=True)


def test_finalize_db_failure_removes_visible_object_before_cleanup(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    real_store = _store_with_job(data_dir)

    class FinalizeFailingStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def mark_artifact_staged(self, **_kwargs):
            raise queue_store.QueueUnavailable("sqlite_unavailable")

    stager = artifact_staging.ArtifactStager(
        store=FinalizeFailingStore(real_store),
        root=artifact_staging.validate_staging_root(
            cfg,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
            disk_usage_fn=lambda _path: SimpleNamespace(free=10_000),
        ),
        quotas=artifact_staging.quota_settings_from_config(cfg),
    )
    with pytest.raises(queue_store.QueueUnavailable):
        stager.stage_bytes(
            job_uid="job-artifact",
            role="metrics_jsonl",
            data=b"abc",
            expected_size_bytes=3,
        )

    artifact = real_store.list_artifacts_for_cleanup()[0]
    assert artifact.status == "staging"
    assert artifact.reserved_size_bytes == 3
    assert not (staging_dir / "objects" / artifact.artifact_uid).exists()

    cleanup = _stager(cfg, real_store).cleanup()
    recovered = real_store.get_artifact(artifact.artifact_uid)
    assert cleanup.failed_deletes == 0
    assert recovered.status == "verification_failed"
    assert recovered.reserved_size_bytes == 0


def test_restaging_after_deleted_creates_new_staged_artifact(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    data = b"same-content"

    first = stager.stage_bytes(
        job_uid="job-artifact",
        role="metrics_jsonl",
        data=data,
        expected_size_bytes=len(data),
    ).artifact
    assert first is not None
    store.mark_artifact_cleanup_pending(artifact_uid=first.artifact_uid)
    cleanup = stager.cleanup()
    assert cleanup.failed_deletes == 0
    deleted = store.get_artifact(first.artifact_uid)
    assert deleted.status == "deleted"
    assert deleted.artifact_ref == first.artifact_ref

    second = stager.stage_bytes(
        job_uid="job-artifact",
        role="metrics_jsonl",
        data=data,
        expected_size_bytes=len(data),
    ).artifact

    assert second is not None
    assert second.status == "staged"
    assert second.artifact_uid != first.artifact_uid
    assert second.artifact_ref is not None
    assert second.artifact_ref != first.artifact_ref
    assert (staging_dir / "objects" / second.artifact_uid).read_bytes() == data


def test_cleanup_removes_object_for_inactive_staging_before_releasing_quota(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    artifact = store.reserve_artifact(
        job_uid="job-artifact",
        role="metrics_jsonl",
        expected_size_bytes=3,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    leaked_object = staging_dir / "objects" / artifact.artifact_uid
    leaked_object.parent.mkdir(mode=0o750)
    leaked_object.write_bytes(b"abc")

    cleanup = stager.cleanup()
    recovered = store.get_artifact(artifact.artifact_uid)

    assert cleanup.failed_deletes == 0
    assert not leaked_object.exists()
    assert recovered.status == "verification_failed"
    assert recovered.reserved_size_bytes == 0


def test_cleanup_keeps_staging_reservation_when_delete_fails(tmp_path, monkeypatch):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    artifact = store.reserve_artifact(
        job_uid="job-artifact",
        role="metrics_jsonl",
        expected_size_bytes=3,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    temp_path = staging_dir / "tmp" / f"{artifact.artifact_uid}.tmp"
    temp_path.parent.mkdir(mode=0o750)
    temp_path.write_bytes(b"abc")

    def fail_for_temp(path):
        if path == temp_path:
            return "failed"
        return "missing"

    monkeypatch.setattr(artifact_staging, "_remove_file", fail_for_temp)
    cleanup = stager.cleanup()
    current = store.get_artifact(artifact.artifact_uid)

    assert cleanup.failed_deletes == 1
    assert current.status == "staging"
    assert current.reserved_size_bytes == 3
    assert store.artifact_quota_usage(job_uid="job-artifact")["job_bytes"] == 3


def test_stream_over_expected_size_aborts_before_overwriting_quota(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)

    result = stager.stage_iterable(
        job_uid="job-artifact",
        role="metrics_jsonl",
        chunks=(b"abc", b"d"),
        expected_size_bytes=3,
    )

    assert result.status == "verification_failed"
    assert result.artifact is not None
    assert result.artifact.failure_code == "artifact_size_mismatch"
    assert result.artifact.sha256 is None
    assert result.artifact.size_bytes is None
    assert result.artifact.reserved_size_bytes == 0
    assert not list((staging_dir / "tmp").glob("*"))
    assert not list((staging_dir / "objects").glob("*"))


def test_recent_staging_row_is_not_cleaned_by_another_stager(tmp_path):
    cfg, data_dir, run_dir, staging_dir = _runtime(tmp_path)
    cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
            "KITT_WORKER_ARTIFACT_MAX_BYTES": "100",
            "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "100",
            "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "200",
            "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "0",
            "KITT_WORKER_ARTIFACT_CLEANUP_AFTER_SECONDS": "86400",
        },
        data_root=data_dir,
        run_root=run_dir,
    )
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    artifact = store.reserve_artifact(
        job_uid="job-artifact",
        role="metrics_jsonl",
        expected_size_bytes=3,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    temp_path = staging_dir / "tmp" / f"{artifact.artifact_uid}.tmp"
    temp_path.parent.mkdir(mode=0o750)
    temp_path.write_bytes(b"abc")

    cleanup = stager.cleanup()
    current = store.get_artifact(artifact.artifact_uid)

    assert cleanup.failed_deletes == 0
    assert current.status == "staging"
    assert current.reserved_size_bytes == 3
    assert temp_path.exists()


def test_cleanup_deletes_orphans_recovers_staging_and_is_idempotent(tmp_path):
    cfg, data_dir, _run_dir, staging_dir = _runtime(tmp_path)
    store = _store_with_job(data_dir)
    stager = _stager(cfg, store)
    data = b"manifest"
    staged = stager.stage_bytes(
        job_uid="job-artifact",
        role="manifest",
        data=data,
        expected_size_bytes=len(data),
    ).artifact
    assert staged is not None
    store.mark_artifact_cleanup_pending(artifact_uid=staged.artifact_uid)
    orphan_temp = staging_dir / "tmp" / ("artifact_" + "f" * 32 + ".tmp")
    orphan_temp.write_bytes(b"orphan")
    stale = store.reserve_artifact(
        job_uid="job-artifact",
        role="metrics_jsonl",
        expected_size_bytes=1,
        quotas=stager.quotas,
        staging_free_bytes=10_000,
    )
    orphan_object = staging_dir / "objects" / ("artifact_" + "e" * 32)
    orphan_object.write_bytes(b"orphan-object")

    first = stager.cleanup()
    second = stager.cleanup()

    assert first.failed_deletes == 0
    assert second.failed_deletes == 0
    assert not orphan_temp.exists()
    assert not orphan_object.exists()
    assert store.get_artifact(staged.artifact_uid).status == "deleted"
    recovered = store.get_artifact(stale.artifact_uid)
    assert recovered.status == "verification_failed"
    assert recovered.reserved_size_bytes == 0
