from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

import job_stubs
import publish_test_support
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


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value = self.value + timedelta(seconds=seconds)


def _settings(*, max_jobs: int = 10, resume_limit: int = 2, busy_ms: int = 100):
    return queue_store.QueueSettings(
        max_jobs=max_jobs,
        lease_ttl_seconds=30,
        resume_limit=resume_limit,
        sqlite_busy_timeout_ms=busy_ms,
    )


def _store(tmp_path, **kwargs):
    return queue_store.QueueStore.open(tmp_path, _settings(**kwargs))


def _artifact_quotas(
    *,
    max_bytes: int = 100,
    job_quota_bytes: int = 100,
    total_quota_bytes: int = 200,
    min_free_bytes: int = 0,
):
    return queue_store.ArtifactQuotaSettings(
        max_bytes=max_bytes,
        job_quota_bytes=job_quota_bytes,
        total_quota_bytes=total_quota_bytes,
        min_free_bytes=min_free_bytes,
    )


def _join(*parts: str) -> str:
    return "".join(parts)


def _install_schema_version(conn: sqlite3.Connection, version: int) -> None:
    migrations = [
        (1, queue_store.MIGRATION_1_SQL, queue_store.MIGRATION_1_CHECKSUM_SHA256),
        (2, queue_store.MIGRATION_2_SQL, queue_store.MIGRATION_2_CHECKSUM_SHA256),
        (3, queue_store.MIGRATION_3_SQL, queue_store.MIGRATION_3_CHECKSUM_SHA256),
        (4, queue_store.MIGRATION_4_SQL, queue_store.MIGRATION_4_CHECKSUM_SHA256),
        (5, queue_store.MIGRATION_5_SQL, queue_store.MIGRATION_5_CHECKSUM_SHA256),
    ]
    for migration_version, sql, checksum in migrations[:version]:
        for statement in queue_store._sql_statements(sql):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, '2026-06-15T12:00:00.000Z', ?)
            """,
            (migration_version, checksum),
        )
    conn.execute(f"PRAGMA user_version={version}")


def test_schema_migration_indices_and_wal_sidecars_are_bounded(tmp_path):
    store = _store(tmp_path)
    job = store.put_job(
        job_uid="job-schema",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    assert job.state == "queued"
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        migrations = conn.execute(
            "select version, checksum_sha256 from schema_migrations"
        ).fetchall()
        indices = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type = 'index'"
            ).fetchall()
        }
        job_columns = {
            row[1]
            for row in conn.execute("pragma table_info(jobs)").fetchall()
        }
        attempt_columns = {
            row[1]
            for row in conn.execute("pragma table_info(attempts)").fetchall()
        }
        artifact_columns = {
            row[1]
            for row in conn.execute("pragma table_info(artifacts)").fetchall()
        }
        publish_result_columns = {
            row[1]
            for row in conn.execute("pragma table_info(publish_results)").fetchall()
        }

    assert migrations == [
        (1, queue_store.MIGRATION_1_CHECKSUM_SHA256),
        (2, queue_store.MIGRATION_2_CHECKSUM_SHA256),
        (3, queue_store.MIGRATION_3_CHECKSUM_SHA256),
        (4, queue_store.MIGRATION_4_CHECKSUM_SHA256),
        (5, queue_store.MIGRATION_5_CHECKSUM_SHA256),
        (6, queue_store.MIGRATION_6_CHECKSUM_SHA256),
    ]
    assert {
        "idx_jobs_state_created_at",
        "idx_jobs_updated_at",
        "idx_jobs_active_lease_expiry",
        "idx_jobs_cancel_requested",
        "idx_jobs_resume_requested",
        "idx_jobs_job_type_state_created_at",
        "idx_attempts_job_uid_attempt_number",
        "idx_jobs_job_uid_run_uid",
        "idx_artifacts_job_status_created_at",
        "idx_artifacts_run_uid",
        "idx_artifacts_cleanup",
        "idx_artifacts_role_status",
        "idx_artifacts_ref_unique",
        "idx_artifacts_staged_job_role_hash_size",
        "idx_publish_results_target_ref",
        "idx_publish_results_publish_job_uid",
    }.issubset(indices)
    assert {
        "started_at",
        "finished_at",
        "last_failure_class",
        "runner_kind",
        "job_type",
    }.issubset(job_columns)
    assert {"started_at", "finished_at", "failure_class"}.issubset(
        attempt_columns
    )
    assert {
        "artifact_uid",
        "job_uid",
        "run_uid",
        "role",
        "artifact_ref",
        "sha256",
        "size_bytes",
        "expected_size_bytes",
        "reserved_size_bytes",
        "status",
        "verified_at",
        "deleted_at",
        "failure_code",
    }.issubset(artifact_columns)
    assert {
        "job_uid",
        "publish_job_uid",
        "target_ref",
        "publish_spec_hash_sha256",
        "source_artifact_fingerprint_sha256",
        "provenance_fingerprint_sha256",
        "ollama_digest",
        "idempotent",
        "completed_at",
    }.issubset(publish_result_columns)
    assert {path.name for path in tmp_path.iterdir()}.issubset(
        {"queue.sqlite3", "queue.sqlite3-wal", "queue.sqlite3-shm"}
    )


def test_queue_rejects_legacy_metadata_only_job_spec(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(job_stubs.JobValidationError):
        store.put_job(
            job_uid="job-legacy-spec",
            job_spec={
                "job_kind": "metadata_only",
                "model_ref": "model.v1",
                "dataset_ref": "dataset.v1",
                "trainer_profile_ref": "trainer.v1",
                "limits": {"max_steps": 0},
                "labels": {"kitt_job": "job.v1"},
                "metadata": {"operator": "kiron"},
            },
            capability_hash_sha256=CAPABILITY_HASH,
        )


def test_schema_v6_drains_existing_active_sprint12_jobs(tmp_path):
    db_path = tmp_path / "queue.sqlite3"
    canonical, spec_hash = job_stubs.canonicalize_job_spec(VALID_SPEC)
    now = "2026-06-15T12:00:00.000Z"
    with sqlite3.connect(db_path) as conn:
        _install_schema_version(conn, 5)
        conn.execute(
            """
            insert into jobs (
                job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
                capability_hash_sha256, state, created_at, updated_at,
                accepted_at, queued_at, resume_limit, runner_kind
            )
            values (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, 2, 'none')
            """,
            (
                "job-sprint12-active",
                "run-sprint12-active",
                spec_hash,
                canonical,
                CAPABILITY_HASH,
                now,
                now,
                now,
                now,
            ),
        )

    store = _store(tmp_path)
    drained = store.get_job("job-sprint12-active")

    assert drained.job_type == "sft"
    assert drained.state == "failed_metadata_only"
    assert drained.terminal_at == now
    assert drained.lease_owner is None
    assert drained.last_failure_code == "sprint12_queue_drained"


def test_unknown_higher_schema_version_fails_closed(tmp_path):
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute(
            "create table schema_migrations(version integer primary key, applied_at text not null, checksum_sha256 text not null)"
        )
        conn.execute(
            "insert into schema_migrations values (7, '2026-06-15T12:00:00.000Z', ?)",
            ("b" * 64,),
        )

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"


def test_unknown_higher_user_version_fails_closed(tmp_path):
    _store(tmp_path)
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute("PRAGMA user_version=7")

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"


def test_schema_user_version_drift_fails_closed(tmp_path):
    _store(tmp_path)
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute("PRAGMA user_version=2")

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"


def test_corrupt_database_fails_closed_without_repair(tmp_path):
    db_path = tmp_path / "queue.sqlite3"
    db_path.write_bytes(b"not a sqlite database")

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code in {"db_corrupt", "migration_failed"}
    assert db_path.read_bytes() == b"not a sqlite database"


def test_schema_with_matching_migration_checksum_but_missing_checks_fails_closed(tmp_path):
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute(
            "create table schema_migrations(version integer primary key, applied_at text not null, checksum_sha256 text not null)"
        )
        conn.execute(
            "insert into schema_migrations values (1, '2026-06-15T12:00:00.000Z', ?)",
            (queue_store.MIGRATION_1_CHECKSUM_SHA256,),
        )
        conn.execute(
            "insert into schema_migrations values (2, '2026-06-15T12:00:00.000Z', ?)",
            (queue_store.MIGRATION_2_CHECKSUM_SHA256,),
        )
        conn.execute(
            "insert into schema_migrations values (3, '2026-06-15T12:00:00.000Z', ?)",
            (queue_store.MIGRATION_3_CHECKSUM_SHA256,),
        )
        conn.execute("create table jobs(job_uid text primary key, run_uid text, state text)")
        conn.execute("create table attempts(attempt_uid text primary key, job_uid text)")
        conn.execute("create index idx_jobs_state_created_at on jobs(job_uid)")
        conn.execute("create index idx_jobs_updated_at on jobs(job_uid)")
        conn.execute("create index idx_jobs_active_lease_expiry on jobs(job_uid)")
        conn.execute("create index idx_jobs_cancel_requested on jobs(job_uid)")
        conn.execute("create index idx_jobs_resume_requested on jobs(job_uid)")
        conn.execute("create unique index idx_attempts_job_uid_attempt_number on attempts(attempt_uid)")
        conn.execute("create unique index idx_jobs_job_uid_run_uid on jobs(job_uid, run_uid)")

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"


def test_role_migration_rolls_back_when_existing_foreign_keys_are_broken(tmp_path):
    _store(tmp_path)
    db_path = tmp_path / "queue.sqlite3"
    now = "2026-06-15T12:00:00.000Z"
    with sqlite3.connect(db_path) as conn:
        conn.execute("pragma foreign_keys=off")
        for name in (
            "idx_artifacts_job_status_created_at",
            "idx_artifacts_run_uid",
            "idx_artifacts_cleanup",
            "idx_artifacts_role_status",
            "idx_artifacts_ref_unique",
            "idx_artifacts_staged_job_role_hash_size",
        ):
            conn.execute(f"drop index if exists {name}")
        conn.execute("drop table artifacts")
        conn.execute("delete from schema_migrations where version >= 4")
        conn.execute("pragma user_version=3")
        conn.execute(
            """
            insert into attempts (
                attempt_uid, job_uid, attempt_number, state, created_at, updated_at
            )
            values ('attempt_orphan', 'job-missing', 1, 'metadata_created', ?, ?)
            """,
            (now, now),
        )

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"
    with sqlite3.connect(db_path) as conn:
        versions = [
            row[0]
            for row in conn.execute(
                "select version from schema_migrations order by version"
            )
        ]
        user_version = conn.execute("pragma user_version").fetchone()[0]
        artifacts_exists = conn.execute(
            """
            select 1
            from sqlite_master
            where type = 'table' and name = 'artifacts'
            """
        ).fetchone()
        conn.execute("pragma foreign_keys=on")
        fk_rows = conn.execute("pragma foreign_key_check").fetchall()

    assert versions == [1, 2, 3]
    assert user_version == 3
    assert artifacts_exists is None
    assert fk_rows


@pytest.mark.parametrize("extra_object_type", ["table", "view", "trigger"])
def test_schema_with_extra_user_object_fails_closed(tmp_path, extra_object_type):
    _store(tmp_path)
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        if extra_object_type == "table":
            conn.execute("create table extra_metadata(value text)")
        elif extra_object_type == "view":
            conn.execute("create view extra_schema_view as select job_uid from jobs")
        else:
            conn.execute(
                """
                create trigger extra_schema_trigger
                after insert on jobs
                begin
                    select 1;
                end
                """
            )

    with pytest.raises(queue_store.QueueUnavailable) as excinfo:
        _store(tmp_path)

    assert excinfo.value.reason_code == "migration_failed"


def test_check_constraints_reject_invalid_hashes_counters_and_terminality(tmp_path):
    _store(tmp_path)
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        base = {
            "job_uid": "job-check",
            "run_uid": "run_check",
            "job_spec_hash_sha256": "a" * 64,
            "canonical_job_spec_json": "{}",
            "capability_hash_sha256": "b" * 64,
            "state": "queued",
            "created_at": "2026-06-15T12:00:00.000Z",
            "updated_at": "2026-06-15T12:00:00.000Z",
            "accepted_at": "2026-06-15T12:00:00.000Z",
            "queued_at": "2026-06-15T12:00:00.000Z",
            "resume_limit": 1,
        }

        def insert_with(**overrides):
            data = {**base, **overrides}
            conn.execute(
                """
                insert into jobs (
                    job_uid, run_uid, job_spec_hash_sha256,
                    canonical_job_spec_json, capability_hash_sha256, state,
                    created_at, updated_at, accepted_at, queued_at,
                    terminal_at, resume_count, resume_limit, last_checkpoint_ref,
                    lease_owner, lease_acquired_at, lease_expires_at
                )
                values (
                    :job_uid, :run_uid, :job_spec_hash_sha256,
                    :canonical_job_spec_json, :capability_hash_sha256, :state,
                    :created_at, :updated_at, :accepted_at, :queued_at,
                    :terminal_at, :resume_count, :resume_limit, :last_checkpoint_ref,
                    :lease_owner, :lease_acquired_at, :lease_expires_at
                )
                """,
                {
                    **data,
                    "terminal_at": data.get("terminal_at"),
                    "resume_count": data.get("resume_count", 0),
                    "last_checkpoint_ref": data.get("last_checkpoint_ref"),
                    "lease_owner": data.get("lease_owner"),
                    "lease_acquired_at": data.get("lease_acquired_at"),
                    "lease_expires_at": data.get("lease_expires_at"),
                },
            )

        with pytest.raises(sqlite3.IntegrityError):
            insert_with(job_spec_hash_sha256="A" * 64)
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(job_uid="bad/job", run_uid="run_check_bad_job")
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(job_uid="job-check-2", run_uid="run_check_2", resume_count=2)
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                job_uid="job-check-3",
                run_uid="run_check_3",
                state="failed_metadata_only",
                terminal_at=None,
            )
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                job_uid="job-check-4",
                run_uid="run_check_4",
                last_checkpoint_ref="-bad",
            )
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                job_uid="job-check-5",
                run_uid="run_check_5",
                last_checkpoint_ref="checkpoint-" + "to" + "ken",
            )
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                job_uid="job-check-6",
                run_uid="run_check_6",
                lease_owner="worker-" + "secret",
                lease_acquired_at="2026-06-15T12:00:00.000Z",
                lease_expires_at="2026-06-15T12:05:00.000Z",
            )


def test_artifact_constraints_reject_paths_urls_secrets_and_bad_status_shapes(tmp_path):
    store = _store(tmp_path)
    job = store.put_job(
        job_uid="job-artifact-check",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        base = {
            "artifact_uid": "artifact_" + "1" * 32,
            "job_uid": job.job_uid,
            "run_uid": job.run_uid,
            "role": "metrics_jsonl",
            "created_at": "2026-06-15T12:00:00.000Z",
            "updated_at": "2026-06-15T12:00:00.000Z",
        }

        def insert_with(**overrides):
            data = {**base, **overrides}
            conn.execute(
                """
                insert into artifacts (
                    artifact_uid, job_uid, run_uid, role, artifact_ref,
                    sha256, size_bytes, expected_size_bytes,
                    reserved_size_bytes, status, created_at, updated_at,
                    verified_at, deleted_at, failure_code
                )
                values (
                    :artifact_uid, :job_uid, :run_uid, :role, :artifact_ref,
                    :sha256, :size_bytes, :expected_size_bytes,
                    :reserved_size_bytes, :status, :created_at, :updated_at,
                    :verified_at, :deleted_at, :failure_code
                )
                """,
                {
                    **data,
                    "artifact_ref": data.get("artifact_ref"),
                    "sha256": data.get("sha256"),
                    "size_bytes": data.get("size_bytes"),
                    "expected_size_bytes": data.get("expected_size_bytes"),
                    "reserved_size_bytes": data.get("reserved_size_bytes", 0),
                    "status": data.get("status", "quota_blocked"),
                    "verified_at": data.get("verified_at"),
                    "deleted_at": data.get("deleted_at"),
                    "failure_code": data.get("failure_code", "artifact_too_large"),
                },
            )

        with pytest.raises(sqlite3.IntegrityError):
            insert_with(artifact_uid="artifact_bad/path")
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(artifact_uid="artifact_" + "2" * 32, role="diagnostic_bundle")
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                artifact_uid="artifact_" + "3" * 32,
                status="staged",
                artifact_ref="artifact/bad-ref",
                sha256="a" * 64,
                size_bytes=1,
                expected_size_bytes=1,
                verified_at="2026-06-15T12:00:00.000Z",
                failure_code=None,
            )
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                artifact_uid="artifact_" + "4" * 32,
                status="disk_full",
                expected_size_bytes=1,
                failure_code="disk_full",
            )
        with pytest.raises(sqlite3.IntegrityError):
            insert_with(
                artifact_uid="artifact_" + "5" * 32,
                status="staging",
                expected_size_bytes=5,
                reserved_size_bytes=4,
                failure_code=None,
            )


def test_artifact_run_uid_must_match_referenced_job_uid(tmp_path):
    store = _store(tmp_path)
    first = store.put_job(
        job_uid="job-artifact-pair-one",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    second = store.put_job(
        job_uid="job-artifact-pair-two",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.artifact.pair.two"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )

    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute("pragma foreign_keys=on")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                insert into artifacts (
                    artifact_uid, job_uid, run_uid, role,
                    expected_size_bytes, reserved_size_bytes, status,
                    created_at, updated_at
                )
                values (?, ?, ?, 'metrics_jsonl', 1, 1, 'staging', ?, ?)
                """,
                (
                    "artifact_" + "6" * 32,
                    first.job_uid,
                    second.run_uid,
                    "2026-06-15T12:00:00.000Z",
                    "2026-06-15T12:00:00.000Z",
                ),
            )


def test_artifact_ref_remains_unique_after_delete(tmp_path):
    store = _store(tmp_path)
    job = store.put_job(
        job_uid="job-artifact-ref-unique",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    now = "2026-06-15T12:00:00.000Z"
    artifact_ref = (
        "artifact:run_ref_unique:metrics_jsonl:aaaaaaaaaaaaaaaa:artifact_" + "7" * 32
    )

    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        conn.execute("pragma foreign_keys=on")
        conn.execute(
            """
            insert into artifacts (
                artifact_uid, job_uid, run_uid, role, artifact_ref,
                sha256, size_bytes, expected_size_bytes, reserved_size_bytes,
                status, created_at, updated_at, verified_at, deleted_at
            )
            values (?, ?, ?, 'metrics_jsonl', ?, ?, 1, 1, 0, 'deleted', ?, ?, ?, ?)
            """,
            (
                "artifact_" + "7" * 32,
                job.job_uid,
                job.run_uid,
                artifact_ref,
                "a" * 64,
                now,
                now,
                now,
                now,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                insert into artifacts (
                    artifact_uid, job_uid, run_uid, role, artifact_ref,
                    sha256, size_bytes, expected_size_bytes, reserved_size_bytes,
                    status, created_at, updated_at, verified_at
                )
                values (?, ?, ?, 'metrics_jsonl', ?, ?, 1, 1, 0, 'staged', ?, ?, ?)
                """,
                (
                    "artifact_" + "8" * 32,
                    job.job_uid,
                    job.run_uid,
                    artifact_ref,
                    "b" * 64,
                    now,
                    now,
                    now,
                ),
            )


def test_artifact_live_content_dedupe_for_staged_and_cleanup_pending(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-live-dedupe",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    quotas = _artifact_quotas()
    sha256 = "a" * 64

    first = store.reserve_artifact(
        job_uid="job-artifact-live-dedupe",
        role="metrics_jsonl",
        expected_size_bytes=2,
        quotas=quotas,
        staging_free_bytes=1000,
    )
    staged = store.mark_artifact_staged(
        artifact_uid=first.artifact_uid,
        sha256=sha256,
        size_bytes=2,
    )
    duplicate_staged = store.reserve_artifact(
        job_uid="job-artifact-live-dedupe",
        role="metrics_jsonl",
        expected_size_bytes=2,
        quotas=quotas,
        staging_free_bytes=1000,
    )
    returned_staged = store.mark_artifact_staged(
        artifact_uid=duplicate_staged.artifact_uid,
        sha256=sha256,
        size_bytes=2,
    )
    failed_staged_duplicate = store.get_artifact(duplicate_staged.artifact_uid)

    assert returned_staged.artifact_uid == staged.artifact_uid
    assert returned_staged.status == "staged"
    assert failed_staged_duplicate.status == "verification_failed"
    assert failed_staged_duplicate.failure_code == "artifact_duplicate"
    assert failed_staged_duplicate.reserved_size_bytes == 0

    cleanup_pending = store.mark_artifact_cleanup_pending(
        artifact_uid=staged.artifact_uid
    )
    duplicate_cleanup_pending = store.reserve_artifact(
        job_uid="job-artifact-live-dedupe",
        role="metrics_jsonl",
        expected_size_bytes=2,
        quotas=quotas,
        staging_free_bytes=1000,
    )
    returned_cleanup_pending = store.mark_artifact_staged(
        artifact_uid=duplicate_cleanup_pending.artifact_uid,
        sha256=sha256,
        size_bytes=2,
    )
    failed_cleanup_duplicate = store.get_artifact(
        duplicate_cleanup_pending.artifact_uid
    )

    assert returned_cleanup_pending.artifact_uid == cleanup_pending.artifact_uid
    assert returned_cleanup_pending.status == "cleanup_pending"
    assert failed_cleanup_duplicate.status == "verification_failed"
    assert failed_cleanup_duplicate.failure_code == "artifact_duplicate"
    assert failed_cleanup_duplicate.reserved_size_bytes == 0


def test_artifact_reservation_and_cleanup_pending_quota_accounting(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifacts",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    first = store.reserve_artifact(
        job_uid="job-artifacts",
        role="metrics_jsonl",
        expected_size_bytes=40,
        quotas=_artifact_quotas(
            max_bytes=80,
            job_quota_bytes=80,
            total_quota_bytes=100,
        ),
        staging_free_bytes=1000,
    )
    assert first.status == "staging"
    usage = store.artifact_quota_usage(job_uid="job-artifacts")
    assert usage["job_bytes"] == 40
    assert usage["total_bytes"] == 40
    staged = store.mark_artifact_staged(
        artifact_uid=first.artifact_uid,
        sha256="a" * 64,
        size_bytes=40,
    )
    assert staged.status == "staged"
    cleanup_pending = store.mark_artifact_cleanup_pending(
        artifact_uid=staged.artifact_uid
    )
    assert cleanup_pending.status == "cleanup_pending"
    assert store.artifact_quota_usage(job_uid="job-artifacts")["job_bytes"] == 40

    blocked = store.reserve_artifact(
        job_uid="job-artifacts",
        role="manifest",
        expected_size_bytes=41,
        quotas=_artifact_quotas(
            max_bytes=80,
            job_quota_bytes=80,
            total_quota_bytes=100,
        ),
        staging_free_bytes=1000,
    )
    assert blocked.status == "quota_blocked"
    assert blocked.failure_code == "artifact_job_quota_exceeded"
    deleted = store.mark_artifact_deleted(artifact_uid=staged.artifact_uid)
    assert deleted.status == "deleted"
    assert store.artifact_quota_usage(job_uid="job-artifacts")["job_bytes"] == 0


def test_artifact_deleted_requires_cleanup_pending_state(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-delete",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    artifact = store.reserve_artifact(
        job_uid="job-artifact-delete",
        role="metrics_jsonl",
        expected_size_bytes=1,
        quotas=_artifact_quotas(),
        staging_free_bytes=1000,
    )
    staged = store.mark_artifact_staged(
        artifact_uid=artifact.artifact_uid,
        sha256="a" * 64,
        size_bytes=1,
    )

    with pytest.raises(queue_store.ArtifactConflict):
        store.mark_artifact_deleted(artifact_uid=staged.artifact_uid)

    cleanup_pending = store.mark_artifact_cleanup_pending(
        artifact_uid=staged.artifact_uid
    )
    deleted = store.mark_artifact_deleted(artifact_uid=cleanup_pending.artifact_uid)
    assert deleted.status == "deleted"


def test_artifact_quota_reservation_is_visible_to_parallel_precheck(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-race",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    first = store.reserve_artifact(
        job_uid="job-artifact-race",
        role="metrics_jsonl",
        expected_size_bytes=60,
        quotas=_artifact_quotas(job_quota_bytes=100, total_quota_bytes=100),
        staging_free_bytes=1000,
    )
    second = store.reserve_artifact(
        job_uid="job-artifact-race",
        role="manifest",
        expected_size_bytes=60,
        quotas=_artifact_quotas(job_quota_bytes=100, total_quota_bytes=100),
        staging_free_bytes=1000,
    )

    assert first.status == "staging"
    assert second.status == "quota_blocked"
    assert second.failure_code == "artifact_job_quota_exceeded"


def test_artifact_status_summary_ignores_terminal_job_history(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-terminal-summary",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    artifact = store.reserve_artifact(
        job_uid="job-artifact-terminal-summary",
        role="gguf",
        expected_size_bytes=200,
        quotas=_artifact_quotas(max_bytes=100),
        staging_free_bytes=1000,
    )
    assert artifact.status == "quota_blocked"
    assert store.artifact_status_summary()["failure_counts"] == {
        "artifact_too_large": 1
    }
    leased = store.next_runnable_job("worker.01")
    assert leased is not None
    store.mark_failed(
        job_uid="job-artifact-terminal-summary",
        owner="worker.01",
        failure_code="runner_failed",
        failure_class="runner",
    )

    assert store.artifact_status_summary() == {
        "status_counts": {},
        "failure_counts": {},
    }


def test_list_staged_artifacts_for_job_returns_public_refs(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-list",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    reserved = store.reserve_artifact(
        job_uid="job-artifact-list",
        role="ollama_modelfile",
        expected_size_bytes=40,
        quotas=_artifact_quotas(),
        staging_free_bytes=1000,
    )
    staged = store.mark_artifact_staged(
        artifact_uid=reserved.artifact_uid,
        sha256="a" * 64,
        size_bytes=40,
    )

    artifacts = store.list_staged_artifacts_for_job("job-artifact-list")

    assert artifacts == [staged]
    assert artifacts[0].artifact_ref is not None
    assert artifacts[0].role == "ollama_modelfile"


def test_artifact_reservation_can_account_for_local_move_without_copy_space(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-artifact-move-space",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    moved = store.reserve_artifact(
        job_uid="job-artifact-move-space",
        role="gguf",
        expected_size_bytes=80,
        quotas=_artifact_quotas(
            max_bytes=100,
            job_quota_bytes=100,
            total_quota_bytes=100,
            min_free_bytes=50,
        ),
        staging_free_bytes=60,
        additional_staging_bytes=0,
    )
    copied = store.reserve_artifact(
        job_uid="job-artifact-move-space",
        role="merged_weights",
        expected_size_bytes=1,
        quotas=_artifact_quotas(
            max_bytes=100,
            job_quota_bytes=100,
            total_quota_bytes=100,
            min_free_bytes=50,
        ),
        staging_free_bytes=60,
    )

    assert moved.status == "staging"
    assert copied.status == "quota_blocked"
    assert copied.failure_code == "artifact_min_free_blocked"


def test_put_idempotency_conflict_queue_full_and_terminal_exclusion(tmp_path):
    store = _store(tmp_path, max_jobs=1)

    with pytest.raises(job_stubs.JobValidationError):
        store.put_job(
            job_uid="bad/job",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )

    first = store.put_job(
        job_uid="job-one",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    second = store.put_job(
        job_uid="job-one",
        job_spec=dict(reversed(list(VALID_SPEC.items()))),
        capability_hash_sha256=CAPABILITY_HASH,
    )

    assert first.run_uid == second.run_uid
    with pytest.raises(job_stubs.JobConflictError):
        store.put_job(
            job_uid="job-one",
            job_spec={**VALID_SPEC, "metadata": {"operator": "other"}},
            capability_hash_sha256=CAPABILITY_HASH,
        )
    with pytest.raises(queue_store.QueueFullError):
        store.put_job(
            job_uid="job-two",
            job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.two"}},
            capability_hash_sha256=CAPABILITY_HASH,
        )

    failed = store.mark_failed_metadata_only(
        job_uid="job-one",
        failure_code="metadata_only_failure",
    )
    assert failed.state == "failed_metadata_only"
    assert failed.terminal_at is not None
    replacement = store.put_job(
        job_uid="job-two",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.two"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    assert replacement.state == "queued"


def test_lease_acquire_renew_expire_and_release_without_execution(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    store.put_job(
        job_uid="job-lease",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    acquired = store.acquire_lease(job_uid="job-lease", owner="worker.01")
    assert acquired.state == "lease_acquired"
    assert acquired.lease_owner == "worker.01"
    assert acquired.attempt_count == 1
    with pytest.raises(queue_store.LeaseUnavailable):
        store.acquire_lease(job_uid="job-lease", owner="worker.02")

    clock.advance(5)
    renewed = store.renew_lease(job_uid="job-lease", owner="worker.01")
    assert renewed.lease_expires_at > acquired.lease_expires_at
    released = store.release_lease(job_uid="job-lease", owner="worker.01")
    assert released.state == "queued"
    assert released.lease_owner is None

    reacquired = store.acquire_lease(job_uid="job-lease", owner="worker.01")
    assert reacquired.attempt_count == 2
    clock.advance(31)
    assert store.expire_stale_leases() == 1
    expired = store.get_job("job-lease")
    assert expired.state == "lease_expired"
    assert expired.lease_owner is None
    assert store.next_runnable_job("worker.01") is None
    with pytest.raises(queue_store.InvalidStateTransition):
        store.acquire_lease(job_uid="job-lease", owner="worker.01")


def test_next_runnable_job_can_filter_by_job_type_without_failing_sft(tmp_path):
    store = _store(tmp_path)
    publish_spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-sft-first",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.put_job(
        job_uid="job-publish-second",
        job_spec=publish_spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    leased = store.next_runnable_job(
        "worker.01",
        job_types=("ollama_publish",),
    )

    assert leased is not None
    assert leased.job_uid == "job-publish-second"
    assert leased.job_type == "ollama_publish"
    sft = store.get_job("job-sft-first")
    assert sft.state == "queued"
    assert sft.lease_owner is None


def test_active_worker_job_prevents_second_lease_and_is_counted(tmp_path):
    store = _store(tmp_path)
    publish_spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-sft-active",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.put_job(
        job_uid="job-publish-waiting",
        job_spec=publish_spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(job_uid="job-sft-active", owner="worker.01")

    assert (
        store.next_runnable_job(
            "worker.02",
            job_types=("ollama_publish",),
        )
        is None
    )
    with pytest.raises(queue_store.LeaseUnavailable):
        store.acquire_lease(job_uid="job-publish-waiting", owner="worker.02")
    stats = store.queue_stats()
    assert stats["queue_depth"] == 2
    assert stats["active_jobs"] == 1


def test_execution_transitions_set_timestamps_and_attempts(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    store.put_job(
        job_uid="job-execution",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    leased = store.next_runnable_job("worker.01")
    assert leased is not None
    assert leased.state == "lease_acquired"
    assert leased.started_at is None
    clock.advance(1)
    running = store.mark_running(job_uid="job-execution", owner="worker.01")
    assert running.state == "running"
    assert running.started_at is not None
    assert running.finished_at is None
    assert running.terminal_at is None
    assert running.runner_kind == "sft_subprocess"
    clock.advance(1)
    succeeded = store.mark_succeeded(job_uid="job-execution", owner="worker.01")
    assert succeeded.state == "succeeded"
    assert succeeded.started_at == running.started_at
    assert succeeded.finished_at == succeeded.terminal_at
    assert succeeded.lease_owner is None
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        attempt = conn.execute(
            """
            select state, started_at, finished_at, failure_code, failure_class
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-execution", 1),
        ).fetchone()
    assert attempt[0] == "succeeded"
    assert attempt[1] == running.started_at
    assert attempt[2] == succeeded.finished_at
    assert attempt[3:] == (None, None)


def test_publish_success_inserts_result_and_succeeds_atomically(tmp_path):
    store = _store(tmp_path)
    spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-publish-atomic",
        job_spec=spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    leased = store.next_runnable_job("worker.01")
    assert leased is not None
    running = store.mark_running(
        job_uid="job-publish-atomic",
        owner="worker.01",
        runner_kind="ollama_publish",
    )
    assert running.job_type == "ollama_publish"
    assert running.runner_kind == "ollama_publish"

    job, result = store.mark_publish_succeeded(
        job_uid="job-publish-atomic",
        owner="worker.01",
        publish_job_uid=spec["publish_job_uid"],
        target_ref=spec["target"]["target_ref"],
        publish_spec_hash_sha256=spec["publish_spec_hash_sha256"],
        source_artifact_fingerprint_sha256="d" * 64,
        provenance_fingerprint_sha256="e" * 64,
        ollama_digest="sha256:abc123",
        idempotent=False,
    )

    assert job.state == "succeeded"
    assert job.finished_at == job.terminal_at
    assert job.lease_owner is None
    assert result.job_uid == "job-publish-atomic"
    assert result.target_ref == spec["target"]["target_ref"]
    assert result.as_public_dict()["idempotent"] is False
    assert store.get_publish_result("job-publish-atomic") == result
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        attempt = conn.execute(
            "select state, lease_owner from attempts where job_uid = ?",
            ("job-publish-atomic",),
        ).fetchone()
    assert attempt == ("succeeded", None)


def test_publish_result_insert_failure_does_not_mark_job_succeeded(tmp_path):
    store = _store(tmp_path)
    spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-publish-invalid-result",
        job_spec=spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(
        job_uid="job-publish-invalid-result",
        owner="worker.01",
        runner_kind="ollama_publish",
    )

    with pytest.raises(job_stubs.JobValidationError):
        store.mark_publish_succeeded(
            job_uid="job-publish-invalid-result",
            owner="worker.01",
            publish_job_uid=spec["publish_job_uid"],
            target_ref=spec["target"]["target_ref"],
            publish_spec_hash_sha256=spec["publish_spec_hash_sha256"],
            source_artifact_fingerprint_sha256="d" * 64,
            provenance_fingerprint_sha256="e" * 64,
            ollama_digest="bad/value",
            idempotent=False,
        )

    current = store.get_job("job-publish-invalid-result")
    assert current.state == "running"
    assert store.get_publish_result("job-publish-invalid-result") is None


def test_failed_metadata_only_rejects_active_or_started_execution(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-metadata-only-leased",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.metadata.leased"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    with pytest.raises(queue_store.InvalidStateTransition):
        store.mark_failed_metadata_only(
            job_uid="job-metadata-only-leased",
            failure_code="metadata_only_failure",
        )
    leased = store.get_job("job-metadata-only-leased")
    assert leased.state == "lease_acquired"
    assert leased.lease_owner == "worker.01"
    assert leased.started_at is None
    store.release_lease(job_uid="job-metadata-only-leased", owner="worker.01")
    store.mark_failed_metadata_only(
        job_uid="job-metadata-only-leased",
        failure_code="metadata_only_failure",
    )

    store.put_job(
        job_uid="job-metadata-only-active",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(job_uid="job-metadata-only-active", owner="worker.01")

    with pytest.raises(queue_store.InvalidStateTransition):
        store.mark_failed_metadata_only(
            job_uid="job-metadata-only-active",
            failure_code="metadata_only_failure",
        )

    job = store.get_job("job-metadata-only-active")
    assert job.state == "running"
    assert job.lease_owner == "worker.01"
    assert job.started_at is not None
    assert job.finished_at is None
    assert job.terminal_at is None


def test_policy_blocked_and_runner_failure_terminalize_owner_bound(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-policy-block",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    blocked = store.mark_policy_blocked(job_uid="job-policy-block", owner="worker.01")
    assert blocked.state == "failed"
    assert blocked.started_at is None
    assert blocked.finished_at == blocked.terminal_at
    assert blocked.last_failure_code == "policy_blocked"
    assert blocked.last_failure_class == "policy"
    assert blocked.runner_kind == "none"

    store.put_job(
        job_uid="job-runner-fail",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.runner.fail"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(job_uid="job-runner-fail", owner="worker.01")
    failed = store.mark_failed(
        job_uid="job-runner-fail",
        owner="worker.01",
        failure_code="runner_failed",
        failure_class="runner",
    )
    assert failed.state == "failed"
    assert failed.started_at is not None
    assert failed.finished_at == failed.terminal_at
    assert failed.last_failure_code == "runner_failed"
    assert failed.last_failure_class == "runner"


def test_queued_cancel_and_resume_gate_closed_maintenance_paths(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-queued-cancel",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.request_cancel(job_uid="job-queued-cancel", reason_code="operator_requested")
    selected = store.next_queued_cancel_job_without_lease()
    assert selected is not None
    assert selected.job_uid == "job-queued-cancel"
    canceled = store.mark_queued_canceled(job_uid="job-queued-cancel")
    assert canceled.state == "canceled"
    assert canceled.started_at is None
    assert canceled.finished_at == canceled.terminal_at
    assert canceled.attempt_count == 0

    store.put_job(
        job_uid="job-resume-gate",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.resume.gate"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    first = store.request_resume(
        job_uid="job-resume-gate",
        reason_code="checkpoint_available",
        last_checkpoint_ref="checkpoint:one",
    )
    second = store.request_resume(job_uid="job-resume-gate", reason_code="retry")
    assert first.resume_count == second.resume_count == 1
    assert store.next_runnable_job("worker.01") is None
    selected_resume = store.next_gate_closed_resume_job_without_lease()
    assert selected_resume is not None
    assert selected_resume.job_uid == "job-resume-gate"
    failed = store.mark_resume_gate_closed(job_uid="job-resume-gate")
    assert failed.state == "failed"
    assert failed.started_at is None
    assert failed.finished_at == failed.terminal_at
    assert failed.last_failure_code == "resume_not_supported"
    assert failed.last_failure_class == "resume"
    assert failed.attempt_count == 0


def test_running_cancel_and_lease_lost_are_owner_bound(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-running-cancel",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(job_uid="job-running-cancel", owner="worker.01")
    store.request_cancel(job_uid="job-running-cancel", reason_code="operator_requested")
    with pytest.raises(queue_store.LeaseUnavailable):
        store.mark_running_canceled(job_uid="job-running-cancel", owner="worker.02")
    canceled = store.mark_running_canceled(
        job_uid="job-running-cancel",
        owner="worker.01",
    )
    assert canceled.state == "canceled"
    assert canceled.finished_at == canceled.terminal_at

    store.put_job(
        job_uid="job-lease-lost",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.lease.lost"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.next_runnable_job("worker.01")
    store.mark_running(job_uid="job-lease-lost", owner="worker.01")
    lost = store.mark_lease_lost(job_uid="job-lease-lost", owner="worker.01")
    assert lost.state == "failed"
    assert lost.last_failure_code == "lease_lost"
    assert lost.last_failure_class == "lease"


def test_get_job_expires_stale_lease_before_reading_status(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    store.put_job(
        job_uid="job-get-stale",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-get-stale", owner="worker.01")

    clock.advance(31)
    job = store.get_job("job-get-stale")

    assert job.state == "lease_expired"
    assert job.lease_owner is None
    assert job.lease_acquired_at is None
    assert job.lease_expires_at is None
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        attempt = conn.execute(
            """
            select state, lease_owner, failure_code
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-get-stale", 1),
        ).fetchone()
    assert attempt == ("lease_expired", None, "stale_lease")


def test_publish_stale_lease_records_verification_unknown(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-publish-stale",
        job_spec=spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-publish-stale", owner="worker.01")
    store.mark_running(
        job_uid="job-publish-stale",
        owner="worker.01",
        runner_kind="ollama_publish",
    )

    clock.advance(31)
    job = store.get_job("job-publish-stale")

    assert job.state == "lease_expired"
    assert job.last_failure_code == "publish_verification_unknown"
    assert job.last_failure_class == "publish"
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        attempt = conn.execute(
            """
            select state, lease_owner, failure_code, failure_class
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-publish-stale", 1),
        ).fetchone()
    assert attempt == (
        "lease_expired",
        None,
        "publish_verification_unknown",
        "publish",
    )


def test_publish_cancel_requested_stale_lease_records_verification_unknown(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    spec, _keyring = publish_test_support.signed_publish_spec()
    store.put_job(
        job_uid="job-publish-cancel-stale",
        job_spec=spec,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-publish-cancel-stale", owner="worker.01")
    store.mark_running(
        job_uid="job-publish-cancel-stale",
        owner="worker.01",
        runner_kind="ollama_publish",
    )
    store.request_cancel(
        job_uid="job-publish-cancel-stale",
        reason_code="operator_requested",
    )

    clock.advance(31)
    job = store.get_job("job-publish-cancel-stale")

    assert job.state == "lease_expired"
    assert job.cancel_requested is True
    assert job.last_failure_code == "publish_verification_unknown"
    assert job.last_failure_class == "publish"
    assert store.next_queued_cancel_job_without_lease() is None
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        attempt = conn.execute(
            """
            select state, lease_owner, failure_code, failure_class
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-publish-cancel-stale", 1),
        ).fetchone()
    assert attempt == (
        "lease_expired",
        None,
        "publish_verification_unknown",
        "publish",
    )


def test_queue_stats_expires_stale_lease_before_counting(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    store.put_job(
        job_uid="job-stats-stale",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-stats-stale", owner="worker.01")

    clock.advance(31)
    stats = store.queue_stats()

    assert stats["queue_depth"] == 1
    assert stats["queue_status"] == "ready"
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        job_row = conn.execute(
            """
            select state, lease_owner, lease_acquired_at, lease_expires_at
            from jobs
            where job_uid = ?
            """,
            ("job-stats-stale",),
        ).fetchone()
        attempt = conn.execute(
            """
            select state, lease_owner, failure_code
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-stats-stale", 1),
        ).fetchone()
    assert job_row == ("lease_expired", None, None, None)
    assert attempt == ("lease_expired", None, "stale_lease")


def test_cancel_keeps_active_lease_and_resume_rejects_active_lease(tmp_path):
    clock = Clock()
    store = queue_store.QueueStore.open(tmp_path, _settings(), now_fn=clock)
    store.put_job(
        job_uid="job-cancel-lease",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    store.acquire_lease(job_uid="job-cancel-lease", owner="worker.01")
    canceled = store.request_cancel(
        job_uid="job-cancel-lease",
        reason_code="operator_requested",
    )

    assert canceled.state == "cancel_requested"
    assert canceled.lease_owner == "worker.01"
    assert canceled.lease_acquired_at is not None
    assert canceled.lease_expires_at is not None
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        cancel_attempt = conn.execute(
            """
            select state, lease_owner
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-cancel-lease", 1),
        ).fetchone()
    assert cancel_attempt == ("lease_acquired", "worker.01")
    store.mark_running_canceled(
        job_uid="job-cancel-lease",
        owner="worker.01",
    )

    store.put_job(
        job_uid="job-resume-lease",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.resume.lease"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-resume-lease", owner="worker.01")
    with pytest.raises(queue_store.InvalidStateTransition):
        store.request_resume(
            job_uid="job-resume-lease",
            reason_code="checkpoint_available",
            last_checkpoint_ref="checkpoint:one",
        )
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        resume_attempt = conn.execute(
            """
            select state, lease_owner
            from attempts
            where job_uid = ? and attempt_number = ?
            """,
            ("job-resume-lease", 1),
        ).fetchone()
    assert resume_attempt == ("lease_acquired", "worker.01")


def test_lease_owner_values_rejected_before_db_constraint(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-invalid-owner",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    for owner in (
        _join("worker.", "to", "ken", "ized"),
        _join("worker.", "sec", "ret", "ive"),
        _join("worker.", "key", "id", "ed"),
    ):
        with pytest.raises(job_stubs.JobValidationError):
            store.acquire_lease(job_uid="job-invalid-owner", owner=owner)

    assert store.get_job("job-invalid-owner").state == "queued"


def test_resume_limit_cancel_reason_and_checkpoint_metadata(tmp_path):
    store = _store(tmp_path, resume_limit=2)
    store.put_job(
        job_uid="job-cancel-actions",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    canceled = store.request_cancel(
        job_uid="job-cancel-actions",
        reason_code="operator_requested",
    )
    assert canceled.cancel_requested is True
    assert canceled.cancel_reason_code == "operator_requested"

    store.put_job(
        job_uid="job-resume-actions",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.resume.actions"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )

    first = store.request_resume(
        job_uid="job-resume-actions",
        reason_code="checkpoint_available",
        last_checkpoint_ref="checkpoint:one",
    )
    second = store.request_resume(job_uid="job-resume-actions", reason_code="retry")
    assert first.resume_count == 1
    assert second.resume_count == 1
    assert second.resume_reason_code == "checkpoint_available"
    assert second.last_checkpoint_ref == "checkpoint:one"
    assert second.state == "resume_requested"


def test_failure_message_text_is_not_persisted_in_sprint_6(tmp_path):
    store = _store(tmp_path)
    store.put_job(
        job_uid="job-failure-message",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )

    with pytest.raises(job_stubs.JobValidationError):
        store.mark_failed_metadata_only(
            job_uid="job-failure-message",
            failure_code="metadata_only_failure",
            failure_message_redacted="diagnostic_message",
        )

    failed = store.mark_failed_metadata_only(
        job_uid="job-failure-message",
        failure_code="metadata_only_failure",
    )
    assert failed.state == "failed_metadata_only"
    with sqlite3.connect(tmp_path / "queue.sqlite3") as conn:
        row = conn.execute(
            "select last_failure_message_redacted from jobs where job_uid = ?",
            ("job-failure-message",),
        ).fetchone()
    assert row[0] is None


def test_invalid_wish_state_transitions_are_rejected(tmp_path):
    store = _store(tmp_path, resume_limit=2)
    store.put_job(
        job_uid="job-cancel-then-resume",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.request_cancel(
        job_uid="job-cancel-then-resume",
        reason_code="operator_requested",
    )
    with pytest.raises(queue_store.InvalidStateTransition):
        store.request_resume(job_uid="job-cancel-then-resume", reason_code="retry")

    store.put_job(
        job_uid="job-resume-then-cancel",
        job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.resume"}},
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.request_resume(job_uid="job-resume-then-cancel", reason_code="retry")
    with pytest.raises(queue_store.InvalidStateTransition):
        store.request_cancel(
            job_uid="job-resume-then-cancel",
            reason_code="operator_requested",
        )


def test_busy_writer_returns_redacted_queue_unavailable(tmp_path):
    store = _store(tmp_path, busy_ms=100)
    store.put_job(
        job_uid="job-busy-base",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    conn = sqlite3.connect(tmp_path / "queue.sqlite3", isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(queue_store.QueueUnavailable) as excinfo:
            store.put_job(
                job_uid="job-busy",
                job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.busy"}},
                capability_hash_sha256=CAPABILITY_HASH,
            )
    finally:
        conn.rollback()
        conn.close()

    assert excinfo.value.reason_code == "sqlite_unavailable"
