"""Local SQLite queue and artifact metadata store for kitt-worker Sprint 8."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Callable, Iterator

import job_stubs
import publish_contract


QUEUE_FILENAME = "queue.sqlite3"
SCHEMA_VERSION = 6
TERMINAL_STATES = {"succeeded", "failed", "canceled", "failed_metadata_only"}
ARTIFACT_ROLES = {
    "adapter",
    "gguf",
    "manifest",
    "merged_weights",
    "metrics_jsonl",
    "ollama_modelfile",
    "run_lock",
    "training_log",
}
ARTIFACT_VISIBLE_STATUSES = {"staged", "cleanup_pending"}
ARTIFACT_QUOTA_STATUSES = {"staging", "staged", "cleanup_pending"}
JOB_TYPES = {"sft", "ollama_publish"}
RUNNER_KINDS = {"fake", "sft_subprocess", "ollama_publish"}
STATE_TRANSITIONS = {
    "queued": {
        "cancel_requested",
        "resume_requested",
        "lease_acquired",
        "failed_metadata_only",
    },
    "lease_acquired": {
        "lease_expired",
        "cancel_requested",
        "running",
        "failed",
        "canceled",
        "queued",
    },
    "lease_expired": {
        "cancel_requested",
        "resume_requested",
        "failed_metadata_only",
    },
    "running": {"cancel_requested", "succeeded", "failed", "canceled", "lease_expired"},
    "cancel_requested": {"canceled", "failed_metadata_only"},
    "resume_requested": {"failed", "failed_metadata_only"},
}
VISIBLE_QUEUE_ERROR_REASONS = {
    "migration_failed",
    "db_corrupt",
    "disk_full",
    "permission_denied",
    "config_invalid",
    "sqlite_unavailable",
}


MIGRATION_1_SQL = """
CREATE TABLE schema_migrations (
    version integer primary key,
    applied_at text not null,
    checksum_sha256 text not null
);

CREATE TABLE jobs (
    job_uid text primary key,
    run_uid text not null unique,
    job_spec_hash_sha256 text not null,
    canonical_job_spec_json text not null,
    capability_hash_sha256 text not null,
    state text not null,
    created_at text not null,
    updated_at text not null,
    accepted_at text not null,
    queued_at text not null,
    terminal_at text,
    cancel_requested integer not null default 0,
    cancel_reason_code text,
    resume_requested integer not null default 0,
    resume_reason_code text,
    resume_count integer not null default 0,
    resume_limit integer not null,
    last_checkpoint_ref text,
    attempt_count integer not null default 0,
    last_failure_code text,
    last_failure_message_redacted text,
    last_failure_at text,
    lease_owner text,
    lease_acquired_at text,
    lease_expires_at text,
    row_version integer not null default 1,
    CHECK(length(job_uid) between 1 and 128),
    CHECK(substr(job_uid, 1, 1) glob '[A-Za-z0-9]'),
    CHECK(job_uid not glob '*[^A-Za-z0-9_.-]*'),
    CHECK(length(job_spec_hash_sha256) = 64),
    CHECK(job_spec_hash_sha256 = lower(job_spec_hash_sha256) and job_spec_hash_sha256 not glob '*[^0-9a-f]*'),
    CHECK(length(capability_hash_sha256) = 64),
    CHECK(capability_hash_sha256 = lower(capability_hash_sha256) and capability_hash_sha256 not glob '*[^0-9a-f]*'),
    CHECK(state in ('queued', 'cancel_requested', 'resume_requested', 'lease_acquired', 'lease_expired', 'failed_metadata_only')),
    CHECK((state = 'failed_metadata_only' and terminal_at is not null) or (state <> 'failed_metadata_only' and terminal_at is null)),
    CHECK(cancel_requested in (0, 1)),
    CHECK(cancel_reason_code is null or (length(cancel_reason_code) between 1 and 64 and cancel_reason_code not glob '*[^a-z0-9_.-]*' and substr(cancel_reason_code, 1, 1) glob '[a-z0-9]')),
    CHECK(resume_requested in (0, 1)),
    CHECK(resume_reason_code is null or (length(resume_reason_code) between 1 and 64 and resume_reason_code not glob '*[^a-z0-9_.-]*' and substr(resume_reason_code, 1, 1) glob '[a-z0-9]')),
    CHECK(resume_count >= 0),
    CHECK(resume_limit >= 0 and resume_limit <= 10),
    CHECK(resume_count <= resume_limit),
    CHECK(last_checkpoint_ref is null or (
        length(last_checkpoint_ref) between 1 and 192
        and last_checkpoint_ref not glob '*[^A-Za-z0-9_.:-]*'
        and substr(last_checkpoint_ref, 1, 1) glob '[A-Za-z0-9]'
        and last_checkpoint_ref not like '%..%'
        and lower(last_checkpoint_ref) not like '%authorization%'
        and lower(last_checkpoint_ref) not like '%bearer%'
        and lower(last_checkpoint_ref) not like '%token%'
        and lower(last_checkpoint_ref) not like '%secret%'
        and lower(last_checkpoint_ref) not like '%credential%'
        and lower(last_checkpoint_ref) not like '%password%'
        and lower(last_checkpoint_ref) not like '%passwd%'
        and lower(last_checkpoint_ref) not like '%private_key%'
        and lower(last_checkpoint_ref) not like '%private-key%'
        and lower(last_checkpoint_ref) not like '%privatekey%'
        and lower(last_checkpoint_ref) not like '%api_key%'
        and lower(last_checkpoint_ref) not like '%api-key%'
        and lower(last_checkpoint_ref) not like '%apikey%'
        and lower(last_checkpoint_ref) not like '%access_key%'
        and lower(last_checkpoint_ref) not like '%access-key%'
        and lower(last_checkpoint_ref) not like '%accesskey%'
        and lower(last_checkpoint_ref) not like '%session_key%'
        and lower(last_checkpoint_ref) not like '%session-key%'
        and lower(last_checkpoint_ref) not like '%sessionkey%'
        and lower(last_checkpoint_ref) not like '%key_id%'
        and lower(last_checkpoint_ref) not like '%key-id%'
        and lower(last_checkpoint_ref) not like '%keyid%'
    )),
    CHECK(attempt_count >= 0),
    CHECK(last_failure_code is null or (length(last_failure_code) between 1 and 64 and last_failure_code not glob '*[^a-z0-9_.-]*' and substr(last_failure_code, 1, 1) glob '[a-z0-9]')),
    CHECK(lease_owner is null or (
        length(lease_owner) between 3 and 128
        and lease_owner = lower(lease_owner)
        and lease_owner not glob '*[^a-z0-9._:-]*'
        and substr(lease_owner, 1, 1) glob '[a-z0-9]'
        and lease_owner not like '%..%'
        and lease_owner not like '%authorization%'
        and lease_owner not like '%bearer%'
        and lease_owner not like '%token%'
        and lease_owner not like '%secret%'
        and lease_owner not like '%credential%'
        and lease_owner not like '%password%'
        and lease_owner not like '%passwd%'
        and lease_owner not like '%private_key%'
        and lease_owner not like '%private-key%'
        and lease_owner not like '%privatekey%'
        and lease_owner not like '%api_key%'
        and lease_owner not like '%api-key%'
        and lease_owner not like '%apikey%'
        and lease_owner not like '%access_key%'
        and lease_owner not like '%access-key%'
        and lease_owner not like '%accesskey%'
        and lease_owner not like '%session_key%'
        and lease_owner not like '%session-key%'
        and lease_owner not like '%sessionkey%'
        and lease_owner not like '%key_id%'
        and lease_owner not like '%key-id%'
        and lease_owner not like '%keyid%'
    )),
    CHECK(row_version >= 1),
    CHECK((lease_owner is null and lease_acquired_at is null and lease_expires_at is null) or (lease_owner is not null and lease_acquired_at is not null and lease_expires_at is not null))
);

CREATE TABLE attempts (
    attempt_uid text primary key,
    job_uid text not null references jobs(job_uid),
    attempt_number integer not null,
    state text not null,
    created_at text not null,
    updated_at text not null,
    lease_owner text,
    failure_code text,
    failure_message_redacted text,
    CHECK(attempt_number >= 1),
    CHECK(state in ('metadata_created', 'lease_acquired', 'lease_expired', 'released', 'failed_metadata_only')),
    CHECK(lease_owner is null or (
        length(lease_owner) between 3 and 128
        and lease_owner = lower(lease_owner)
        and lease_owner not glob '*[^a-z0-9._:-]*'
        and substr(lease_owner, 1, 1) glob '[a-z0-9]'
        and lease_owner not like '%..%'
        and lease_owner not like '%authorization%'
        and lease_owner not like '%bearer%'
        and lease_owner not like '%token%'
        and lease_owner not like '%secret%'
        and lease_owner not like '%credential%'
        and lease_owner not like '%password%'
        and lease_owner not like '%passwd%'
        and lease_owner not like '%private_key%'
        and lease_owner not like '%private-key%'
        and lease_owner not like '%privatekey%'
        and lease_owner not like '%api_key%'
        and lease_owner not like '%api-key%'
        and lease_owner not like '%apikey%'
        and lease_owner not like '%access_key%'
        and lease_owner not like '%access-key%'
        and lease_owner not like '%accesskey%'
        and lease_owner not like '%session_key%'
        and lease_owner not like '%session-key%'
        and lease_owner not like '%sessionkey%'
        and lease_owner not like '%key_id%'
        and lease_owner not like '%key-id%'
        and lease_owner not like '%keyid%'
    )),
    CHECK(failure_code is null or (length(failure_code) between 1 and 64 and failure_code not glob '*[^a-z0-9_.-]*' and substr(failure_code, 1, 1) glob '[a-z0-9]'))
);

CREATE INDEX idx_jobs_state_created_at ON jobs(state, created_at);
CREATE INDEX idx_jobs_updated_at ON jobs(updated_at);
CREATE INDEX idx_jobs_active_lease_expiry ON jobs(lease_expires_at) WHERE lease_expires_at is not null;
CREATE INDEX idx_jobs_cancel_requested ON jobs(cancel_requested, updated_at);
CREATE INDEX idx_jobs_resume_requested ON jobs(resume_requested, updated_at);
CREATE UNIQUE INDEX idx_attempts_job_uid_attempt_number ON attempts(job_uid, attempt_number);
"""

MIGRATION_1_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_1_SQL.encode("utf-8")
).hexdigest()

MIGRATION_2_SQL = """
CREATE TABLE jobs_new (
    job_uid text primary key,
    run_uid text not null unique,
    job_spec_hash_sha256 text not null,
    canonical_job_spec_json text not null,
    capability_hash_sha256 text not null,
    state text not null,
    created_at text not null,
    updated_at text not null,
    accepted_at text not null,
    queued_at text not null,
    started_at text,
    finished_at text,
    terminal_at text,
    cancel_requested integer not null default 0,
    cancel_reason_code text,
    resume_requested integer not null default 0,
    resume_reason_code text,
    resume_count integer not null default 0,
    resume_limit integer not null,
    last_checkpoint_ref text,
    attempt_count integer not null default 0,
    last_failure_code text,
    last_failure_class text,
    last_failure_message_redacted text,
    last_failure_at text,
    runner_kind text not null default 'none',
    lease_owner text,
    lease_acquired_at text,
    lease_expires_at text,
    row_version integer not null default 1,
    CHECK(length(job_uid) between 1 and 128),
    CHECK(substr(job_uid, 1, 1) glob '[A-Za-z0-9]'),
    CHECK(job_uid not glob '*[^A-Za-z0-9_.-]*'),
    CHECK(length(job_spec_hash_sha256) = 64),
    CHECK(job_spec_hash_sha256 = lower(job_spec_hash_sha256) and job_spec_hash_sha256 not glob '*[^0-9a-f]*'),
    CHECK(length(capability_hash_sha256) = 64),
    CHECK(capability_hash_sha256 = lower(capability_hash_sha256) and capability_hash_sha256 not glob '*[^0-9a-f]*'),
    CHECK(state in ('queued', 'cancel_requested', 'resume_requested', 'lease_acquired', 'lease_expired', 'running', 'succeeded', 'failed', 'canceled', 'failed_metadata_only')),
    CHECK(
        (
            state in ('succeeded', 'failed', 'canceled')
            and terminal_at is not null
            and finished_at is not null
            and finished_at = terminal_at
        )
        or (
            state = 'failed_metadata_only'
            and terminal_at is not null
            and started_at is null
            and finished_at is null
        )
        or (
            state not in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
            and terminal_at is null
            and finished_at is null
        )
    ),
    CHECK(cancel_requested in (0, 1)),
    CHECK(cancel_reason_code is null or (length(cancel_reason_code) between 1 and 64 and cancel_reason_code not glob '*[^a-z0-9_.-]*' and substr(cancel_reason_code, 1, 1) glob '[a-z0-9]')),
    CHECK(resume_requested in (0, 1)),
    CHECK(resume_reason_code is null or (length(resume_reason_code) between 1 and 64 and resume_reason_code not glob '*[^a-z0-9_.-]*' and substr(resume_reason_code, 1, 1) glob '[a-z0-9]')),
    CHECK(resume_count >= 0),
    CHECK(resume_limit >= 0 and resume_limit <= 10),
    CHECK(resume_count <= resume_limit),
    CHECK(last_checkpoint_ref is null or (
        length(last_checkpoint_ref) between 1 and 192
        and last_checkpoint_ref not glob '*[^A-Za-z0-9_.:-]*'
        and substr(last_checkpoint_ref, 1, 1) glob '[A-Za-z0-9]'
        and last_checkpoint_ref not like '%..%'
        and lower(last_checkpoint_ref) not like '%authorization%'
        and lower(last_checkpoint_ref) not like '%bearer%'
        and lower(last_checkpoint_ref) not like '%token%'
        and lower(last_checkpoint_ref) not like '%secret%'
        and lower(last_checkpoint_ref) not like '%credential%'
        and lower(last_checkpoint_ref) not like '%password%'
        and lower(last_checkpoint_ref) not like '%passwd%'
        and lower(last_checkpoint_ref) not like '%private_key%'
        and lower(last_checkpoint_ref) not like '%private-key%'
        and lower(last_checkpoint_ref) not like '%privatekey%'
        and lower(last_checkpoint_ref) not like '%api_key%'
        and lower(last_checkpoint_ref) not like '%api-key%'
        and lower(last_checkpoint_ref) not like '%apikey%'
        and lower(last_checkpoint_ref) not like '%access_key%'
        and lower(last_checkpoint_ref) not like '%access-key%'
        and lower(last_checkpoint_ref) not like '%accesskey%'
        and lower(last_checkpoint_ref) not like '%session_key%'
        and lower(last_checkpoint_ref) not like '%session-key%'
        and lower(last_checkpoint_ref) not like '%sessionkey%'
        and lower(last_checkpoint_ref) not like '%key_id%'
        and lower(last_checkpoint_ref) not like '%key-id%'
        and lower(last_checkpoint_ref) not like '%keyid%'
    )),
    CHECK(attempt_count >= 0),
    CHECK(last_failure_code is null or (length(last_failure_code) between 1 and 64 and last_failure_code not glob '*[^a-z0-9_.-]*' and substr(last_failure_code, 1, 1) glob '[a-z0-9]')),
    CHECK(last_failure_class is null or (length(last_failure_class) between 1 and 64 and last_failure_class not glob '*[^a-z0-9_.-]*' and substr(last_failure_class, 1, 1) glob '[a-z0-9]')),
    CHECK(runner_kind in ('none', 'fake')),
    CHECK(lease_owner is null or (
        length(lease_owner) between 3 and 128
        and lease_owner = lower(lease_owner)
        and lease_owner not glob '*[^a-z0-9._:-]*'
        and substr(lease_owner, 1, 1) glob '[a-z0-9]'
        and lease_owner not like '%..%'
        and lease_owner not like '%authorization%'
        and lease_owner not like '%bearer%'
        and lease_owner not like '%token%'
        and lease_owner not like '%secret%'
        and lease_owner not like '%credential%'
        and lease_owner not like '%password%'
        and lease_owner not like '%passwd%'
        and lease_owner not like '%private_key%'
        and lease_owner not like '%private-key%'
        and lease_owner not like '%privatekey%'
        and lease_owner not like '%api_key%'
        and lease_owner not like '%api-key%'
        and lease_owner not like '%apikey%'
        and lease_owner not like '%access_key%'
        and lease_owner not like '%access-key%'
        and lease_owner not like '%accesskey%'
        and lease_owner not like '%session_key%'
        and lease_owner not like '%session-key%'
        and lease_owner not like '%sessionkey%'
        and lease_owner not like '%key_id%'
        and lease_owner not like '%key-id%'
        and lease_owner not like '%keyid%'
    )),
    CHECK(row_version >= 1),
    CHECK((lease_owner is null and lease_acquired_at is null and lease_expires_at is null) or (lease_owner is not null and lease_acquired_at is not null and lease_expires_at is not null)),
    CHECK(state <> 'running' or (lease_owner is not null and lease_acquired_at is not null and lease_expires_at is not null)),
    CHECK(state not in ('succeeded', 'failed', 'canceled', 'failed_metadata_only') or (lease_owner is null and lease_acquired_at is null and lease_expires_at is null))
);

INSERT INTO jobs_new (
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, state, created_at, updated_at, accepted_at,
    queued_at, started_at, finished_at, terminal_at, cancel_requested,
    cancel_reason_code, resume_requested, resume_reason_code, resume_count,
    resume_limit, last_checkpoint_ref, attempt_count, last_failure_code,
    last_failure_class, last_failure_message_redacted, last_failure_at,
    runner_kind, lease_owner, lease_acquired_at, lease_expires_at, row_version
)
SELECT
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, state, created_at, updated_at, accepted_at,
    queued_at, null, null, terminal_at, cancel_requested,
    cancel_reason_code, resume_requested, resume_reason_code, resume_count,
    resume_limit, last_checkpoint_ref, attempt_count, last_failure_code,
    null, last_failure_message_redacted, last_failure_at,
    'none', lease_owner, lease_acquired_at, lease_expires_at, row_version
FROM jobs;

CREATE TABLE attempts_new (
    attempt_uid text primary key,
    job_uid text not null references jobs_new(job_uid),
    attempt_number integer not null,
    state text not null,
    created_at text not null,
    updated_at text not null,
    started_at text,
    finished_at text,
    lease_owner text,
    failure_code text,
    failure_class text,
    failure_message_redacted text,
    CHECK(attempt_number >= 1),
    CHECK(state in ('metadata_created', 'lease_acquired', 'running', 'succeeded', 'failed', 'canceled', 'policy_blocked', 'lease_expired', 'released', 'failed_metadata_only')),
    CHECK(lease_owner is null or (
        length(lease_owner) between 3 and 128
        and lease_owner = lower(lease_owner)
        and lease_owner not glob '*[^a-z0-9._:-]*'
        and substr(lease_owner, 1, 1) glob '[a-z0-9]'
        and lease_owner not like '%..%'
        and lease_owner not like '%authorization%'
        and lease_owner not like '%bearer%'
        and lease_owner not like '%token%'
        and lease_owner not like '%secret%'
        and lease_owner not like '%credential%'
        and lease_owner not like '%password%'
        and lease_owner not like '%passwd%'
        and lease_owner not like '%private_key%'
        and lease_owner not like '%private-key%'
        and lease_owner not like '%privatekey%'
        and lease_owner not like '%api_key%'
        and lease_owner not like '%api-key%'
        and lease_owner not like '%apikey%'
        and lease_owner not like '%access_key%'
        and lease_owner not like '%access-key%'
        and lease_owner not like '%accesskey%'
        and lease_owner not like '%session_key%'
        and lease_owner not like '%session-key%'
        and lease_owner not like '%sessionkey%'
        and lease_owner not like '%key_id%'
        and lease_owner not like '%key-id%'
        and lease_owner not like '%keyid%'
    )),
    CHECK(failure_code is null or (length(failure_code) between 1 and 64 and failure_code not glob '*[^a-z0-9_.-]*' and substr(failure_code, 1, 1) glob '[a-z0-9]')),
    CHECK(failure_class is null or (length(failure_class) between 1 and 64 and failure_class not glob '*[^a-z0-9_.-]*' and substr(failure_class, 1, 1) glob '[a-z0-9]'))
);

INSERT INTO attempts_new (
    attempt_uid, job_uid, attempt_number, state, created_at, updated_at,
    started_at, finished_at, lease_owner, failure_code, failure_class,
    failure_message_redacted
)
SELECT
    attempt_uid, job_uid, attempt_number, state, created_at, updated_at,
    null, null, lease_owner, failure_code, null, failure_message_redacted
FROM attempts;

DROP TABLE attempts;
DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
ALTER TABLE attempts_new RENAME TO attempts;
CREATE INDEX idx_jobs_state_created_at ON jobs(state, created_at);
CREATE INDEX idx_jobs_updated_at ON jobs(updated_at);
CREATE INDEX idx_jobs_active_lease_expiry ON jobs(lease_expires_at) WHERE lease_expires_at is not null;
CREATE INDEX idx_jobs_cancel_requested ON jobs(cancel_requested, updated_at);
CREATE INDEX idx_jobs_resume_requested ON jobs(resume_requested, updated_at);
CREATE UNIQUE INDEX idx_attempts_job_uid_attempt_number ON attempts(job_uid, attempt_number);
"""

MIGRATION_2_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_2_SQL.encode("utf-8")
).hexdigest()

MIGRATION_3_SQL = """
CREATE UNIQUE INDEX idx_jobs_job_uid_run_uid ON jobs(job_uid, run_uid);

CREATE TABLE artifacts (
    artifact_uid text primary key,
    job_uid text not null,
    run_uid text not null,
    role text not null,
    artifact_ref text,
    sha256 text,
    size_bytes integer,
    expected_size_bytes integer,
    reserved_size_bytes integer not null default 0,
    status text not null,
    created_at text not null,
    updated_at text not null,
    verified_at text,
    deleted_at text,
    failure_code text,
    row_version integer not null default 1,
    CHECK(length(artifact_uid) = 41),
    CHECK(substr(artifact_uid, 1, 9) = 'artifact_'),
    CHECK(substr(artifact_uid, 10) not glob '*[^0-9a-f]*'),
    CHECK(role in ('checkpoint', 'model_candidate', 'metrics', 'manifest')),
    CHECK(status in ('staging', 'staged', 'verification_failed', 'quota_blocked', 'disk_full', 'cleanup_pending', 'deleted')),
    CHECK(sha256 is null or (length(sha256) = 64 and sha256 = lower(sha256) and sha256 not glob '*[^0-9a-f]*')),
    CHECK(size_bytes is null or size_bytes >= 0),
    CHECK(expected_size_bytes is null or expected_size_bytes >= 0),
    CHECK(reserved_size_bytes >= 0),
    CHECK(artifact_ref is null or (
        length(artifact_ref) between 1 and 192
        and artifact_ref not glob '*[^A-Za-z0-9_.:-]*'
        and substr(artifact_ref, 1, 1) glob '[A-Za-z0-9]'
        and artifact_ref not like '%..%'
        and artifact_ref not like '%/%'
        and artifact_ref not like '%\\%'
        and artifact_ref not like '%://%'
        and artifact_ref not like '%?%'
        and artifact_ref not like '%#%'
        and lower(artifact_ref) not like '%authorization%'
        and lower(artifact_ref) not like '%bearer%'
        and lower(artifact_ref) not like '%token%'
        and lower(artifact_ref) not like '%secret%'
        and lower(artifact_ref) not like '%credential%'
        and lower(artifact_ref) not like '%password%'
        and lower(artifact_ref) not like '%passwd%'
        and lower(artifact_ref) not like '%private_key%'
        and lower(artifact_ref) not like '%private-key%'
        and lower(artifact_ref) not like '%privatekey%'
        and lower(artifact_ref) not like '%api_key%'
        and lower(artifact_ref) not like '%api-key%'
        and lower(artifact_ref) not like '%apikey%'
        and lower(artifact_ref) not like '%access_key%'
        and lower(artifact_ref) not like '%access-key%'
        and lower(artifact_ref) not like '%accesskey%'
        and lower(artifact_ref) not like '%session_key%'
        and lower(artifact_ref) not like '%session-key%'
        and lower(artifact_ref) not like '%sessionkey%'
        and lower(artifact_ref) not like '%key_id%'
        and lower(artifact_ref) not like '%key-id%'
        and lower(artifact_ref) not like '%keyid%'
    )),
    CHECK(failure_code is null or (length(failure_code) between 1 and 64 and failure_code not glob '*[^a-z0-9_.-]*' and substr(failure_code, 1, 1) glob '[a-z0-9]')),
    CHECK(row_version >= 1),
    FOREIGN KEY(job_uid, run_uid) REFERENCES jobs(job_uid, run_uid),
    CHECK(
        (
            status = 'staging'
            and artifact_ref is null
            and sha256 is null
            and size_bytes is null
            and expected_size_bytes is not null
            and reserved_size_bytes = expected_size_bytes
            and verified_at is null
            and deleted_at is null
            and failure_code is null
        )
        or (
            status = 'staged'
            and artifact_ref is not null
            and sha256 is not null
            and size_bytes is not null
            and expected_size_bytes is not null
            and verified_at is not null
            and reserved_size_bytes = 0
            and deleted_at is null
            and failure_code is null
        )
        or (
            status = 'verification_failed'
            and artifact_ref is null
            and verified_at is null
            and deleted_at is null
            and reserved_size_bytes = 0
            and failure_code is not null
        )
        or (
            status in ('quota_blocked', 'disk_full')
            and artifact_ref is null
            and sha256 is null
            and size_bytes is null
            and expected_size_bytes is null
            and reserved_size_bytes = 0
            and verified_at is null
            and deleted_at is null
            and failure_code is not null
        )
        or (
            status = 'cleanup_pending'
            and artifact_ref is not null
            and sha256 is not null
            and size_bytes is not null
            and expected_size_bytes is not null
            and verified_at is not null
            and reserved_size_bytes = 0
            and deleted_at is null
            and failure_code is null
        )
        or (
            status = 'deleted'
            and artifact_ref is not null
            and sha256 is not null
            and size_bytes is not null
            and expected_size_bytes is not null
            and verified_at is not null
            and deleted_at is not null
            and reserved_size_bytes = 0
            and failure_code is null
        )
    )
);
CREATE INDEX idx_artifacts_job_status_created_at ON artifacts(job_uid, status, created_at);
CREATE INDEX idx_artifacts_run_uid ON artifacts(run_uid);
CREATE INDEX idx_artifacts_cleanup ON artifacts(status, updated_at) WHERE status in ('staging', 'verification_failed', 'quota_blocked', 'disk_full', 'cleanup_pending');
CREATE INDEX idx_artifacts_role_status ON artifacts(role, status);
CREATE UNIQUE INDEX idx_artifacts_ref_unique ON artifacts(artifact_ref) WHERE artifact_ref is not null;
CREATE UNIQUE INDEX idx_artifacts_staged_job_role_hash_size ON artifacts(job_uid, role, sha256, size_bytes) WHERE status in ('staged', 'cleanup_pending');
"""

MIGRATION_3_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_3_SQL.encode("utf-8")
).hexdigest()

_MIGRATION_4_JOBS_CREATE_SQL = (
    "CREATE TABLE jobs_new"
    + MIGRATION_2_SQL.split("CREATE TABLE jobs_new", 1)[1].split(
        "INSERT INTO jobs_new",
        1,
    )[0]
).replace(
    "CHECK(runner_kind in ('none', 'fake'))",
    "CHECK(runner_kind in ('none', 'fake', 'sft_subprocess'))",
)

MIGRATION_4_SQL = _MIGRATION_4_JOBS_CREATE_SQL + """
INSERT INTO jobs_new (
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, state, created_at, updated_at, accepted_at,
    queued_at, started_at, finished_at, terminal_at, cancel_requested,
    cancel_reason_code, resume_requested, resume_reason_code, resume_count,
    resume_limit, last_checkpoint_ref, attempt_count, last_failure_code,
    last_failure_class, last_failure_message_redacted, last_failure_at,
    runner_kind, lease_owner, lease_acquired_at, lease_expires_at, row_version
)
SELECT
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, state, created_at, updated_at, accepted_at,
    queued_at, started_at, finished_at, terminal_at, cancel_requested,
    cancel_reason_code, resume_requested, resume_reason_code, resume_count,
    resume_limit, last_checkpoint_ref, attempt_count, last_failure_code,
    last_failure_class, last_failure_message_redacted, last_failure_at,
    runner_kind, lease_owner, lease_acquired_at, lease_expires_at, row_version
FROM jobs;

DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
CREATE INDEX idx_jobs_state_created_at ON jobs(state, created_at);
CREATE INDEX idx_jobs_updated_at ON jobs(updated_at);
CREATE INDEX idx_jobs_active_lease_expiry ON jobs(lease_expires_at) WHERE lease_expires_at is not null;
CREATE INDEX idx_jobs_cancel_requested ON jobs(cancel_requested, updated_at);
CREATE INDEX idx_jobs_resume_requested ON jobs(resume_requested, updated_at);
CREATE UNIQUE INDEX idx_jobs_job_uid_run_uid ON jobs(job_uid, run_uid);
"""

MIGRATION_4_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_4_SQL.encode("utf-8")
).hexdigest()

_MIGRATION_5_ARTIFACTS_CREATE_SQL = (
    "CREATE TABLE artifacts_new"
    + MIGRATION_3_SQL.split("CREATE TABLE artifacts", 1)[1].split(
        "CREATE INDEX idx_artifacts_job_status_created_at",
        1,
    )[0]
).replace(
    "CHECK(role in ('checkpoint', 'model_candidate', 'metrics', 'manifest'))",
    (
        "CHECK(role in ('adapter', 'gguf', 'manifest', 'merged_weights', "
        "'metrics_jsonl', 'ollama_modelfile', 'run_lock', 'training_log'))"
    ),
)

MIGRATION_5_SQL = _MIGRATION_5_ARTIFACTS_CREATE_SQL + """
INSERT INTO artifacts_new (
    artifact_uid, job_uid, run_uid, role, artifact_ref, sha256, size_bytes,
    expected_size_bytes, reserved_size_bytes, status, created_at, updated_at,
    verified_at, deleted_at, failure_code, row_version
)
SELECT
    artifact_uid, job_uid, run_uid,
    case role
        when 'model_candidate' then 'adapter'
        when 'metrics' then 'metrics_jsonl'
        when 'checkpoint' then 'merged_weights'
        else role
    end,
    artifact_ref, sha256, size_bytes,
    expected_size_bytes, reserved_size_bytes, status, created_at, updated_at,
    verified_at, deleted_at, failure_code, row_version
FROM artifacts;

DROP INDEX IF EXISTS idx_artifacts_job_status_created_at;
DROP INDEX IF EXISTS idx_artifacts_run_uid;
DROP INDEX IF EXISTS idx_artifacts_cleanup;
DROP INDEX IF EXISTS idx_artifacts_role_status;
DROP INDEX IF EXISTS idx_artifacts_ref_unique;
DROP INDEX IF EXISTS idx_artifacts_staged_job_role_hash_size;
DROP TABLE artifacts;
ALTER TABLE artifacts_new RENAME TO artifacts;
CREATE INDEX idx_artifacts_job_status_created_at ON artifacts(job_uid, status, created_at);
CREATE INDEX idx_artifacts_run_uid ON artifacts(run_uid);
CREATE INDEX idx_artifacts_cleanup ON artifacts(status, updated_at) WHERE status in ('staging', 'verification_failed', 'quota_blocked', 'disk_full', 'cleanup_pending');
CREATE INDEX idx_artifacts_role_status ON artifacts(role, status);
CREATE UNIQUE INDEX idx_artifacts_ref_unique ON artifacts(artifact_ref) WHERE artifact_ref is not null;
CREATE UNIQUE INDEX idx_artifacts_staged_job_role_hash_size ON artifacts(job_uid, role, sha256, size_bytes) WHERE status in ('staged', 'cleanup_pending');
"""

MIGRATION_5_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_5_SQL.encode("utf-8")
).hexdigest()

_MIGRATION_6_JOBS_CREATE_SQL = _MIGRATION_4_JOBS_CREATE_SQL.replace(
    "capability_hash_sha256 text not null,",
    "capability_hash_sha256 text not null,\n    job_type text not null default 'sft',",
).replace(
    "CHECK(length(capability_hash_sha256) = 64),",
    (
        "CHECK(length(capability_hash_sha256) = 64),\n"
        "    CHECK(job_type in ('sft', 'ollama_publish')),"
    ),
).replace(
    "CHECK(runner_kind in ('none', 'fake', 'sft_subprocess'))",
    "CHECK(runner_kind in ('none', 'fake', 'sft_subprocess', 'ollama_publish'))",
)

MIGRATION_6_SQL = _MIGRATION_6_JOBS_CREATE_SQL + """
INSERT INTO jobs_new (
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, job_type, state, created_at, updated_at, accepted_at,
    queued_at, started_at, finished_at, terminal_at, cancel_requested,
    cancel_reason_code, resume_requested, resume_reason_code, resume_count,
    resume_limit, last_checkpoint_ref, attempt_count, last_failure_code,
    last_failure_class, last_failure_message_redacted, last_failure_at,
    runner_kind, lease_owner, lease_acquired_at, lease_expires_at, row_version
)
SELECT
    job_uid, run_uid, job_spec_hash_sha256, canonical_job_spec_json,
    capability_hash_sha256, 'sft',
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then state
        else 'failed_metadata_only'
    end,
    created_at,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then updated_at
        else updated_at
    end,
    accepted_at, queued_at,
    case
        when state in ('succeeded', 'failed', 'canceled')
        then started_at
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled')
        then finished_at
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then terminal_at
        else updated_at
    end,
    cancel_requested, cancel_reason_code, resume_requested, resume_reason_code,
    resume_count, resume_limit, last_checkpoint_ref, attempt_count,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then last_failure_code
        else 'sprint12_queue_drained'
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then last_failure_class
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then last_failure_message_redacted
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then last_failure_at
        else updated_at
    end,
    runner_kind,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then lease_owner
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then lease_acquired_at
        else null
    end,
    case
        when state in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
        then lease_expires_at
        else null
    end,
    row_version
FROM jobs;

DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
UPDATE attempts
set state = 'failed_metadata_only',
    lease_owner = null,
    failure_code = coalesce(failure_code, 'sprint12_queue_drained'),
    failure_class = null,
    failure_message_redacted = null,
    finished_at = null,
    updated_at = (
        select jobs.updated_at from jobs where jobs.job_uid = attempts.job_uid
    )
where job_uid in (
    select job_uid
    from jobs
    where state = 'failed_metadata_only'
      and last_failure_code = 'sprint12_queue_drained'
);
CREATE INDEX idx_jobs_state_created_at ON jobs(state, created_at);
CREATE INDEX idx_jobs_updated_at ON jobs(updated_at);
CREATE INDEX idx_jobs_active_lease_expiry ON jobs(lease_expires_at) WHERE lease_expires_at is not null;
CREATE INDEX idx_jobs_cancel_requested ON jobs(cancel_requested, updated_at);
CREATE INDEX idx_jobs_resume_requested ON jobs(resume_requested, updated_at);
CREATE UNIQUE INDEX idx_jobs_job_uid_run_uid ON jobs(job_uid, run_uid);
CREATE INDEX idx_jobs_job_type_state_created_at ON jobs(job_type, state, created_at);

CREATE TABLE publish_results (
    job_uid text primary key references jobs(job_uid),
    publish_job_uid text not null,
    target_ref text not null,
    publish_spec_hash_sha256 text not null,
    source_artifact_fingerprint_sha256 text not null,
    provenance_fingerprint_sha256 text not null,
    ollama_digest text,
    idempotent integer not null,
    created_at text not null,
    completed_at text not null,
    CHECK(length(job_uid) between 1 and 128),
    CHECK(length(publish_job_uid) between 1 and 192),
    CHECK(publish_job_uid not glob '*[^A-Za-z0-9_.:-]*'),
    CHECK(length(target_ref) between 1 and 192),
    CHECK(substr(target_ref, 1, 1) glob '[A-Za-z0-9]'),
    CHECK(target_ref not like '/%'),
    CHECK(target_ref not like '%\\%'),
    CHECK(target_ref not like '%://%'),
    CHECK(target_ref not like '%?%'),
    CHECK(target_ref not like '%#%'),
    CHECK(target_ref not like '%..%'),
    CHECK(length(publish_spec_hash_sha256) = 64),
    CHECK(publish_spec_hash_sha256 = lower(publish_spec_hash_sha256) and publish_spec_hash_sha256 not glob '*[^0-9a-f]*'),
    CHECK(length(source_artifact_fingerprint_sha256) = 64),
    CHECK(source_artifact_fingerprint_sha256 = lower(source_artifact_fingerprint_sha256) and source_artifact_fingerprint_sha256 not glob '*[^0-9a-f]*'),
    CHECK(length(provenance_fingerprint_sha256) = 64),
    CHECK(provenance_fingerprint_sha256 = lower(provenance_fingerprint_sha256) and provenance_fingerprint_sha256 not glob '*[^0-9a-f]*'),
    CHECK(ollama_digest is null or (
        length(ollama_digest) between 1 and 128
        and ollama_digest not glob '*[^A-Za-z0-9_.:-]*'
        and lower(ollama_digest) not like '%authorization%'
        and lower(ollama_digest) not like '%bearer%'
        and lower(ollama_digest) not like '%token%'
        and lower(ollama_digest) not like '%secret%'
        and lower(ollama_digest) not like '%credential%'
        and lower(ollama_digest) not like '%password%'
    )),
    CHECK(idempotent in (0, 1))
);
CREATE INDEX idx_publish_results_target_ref ON publish_results(target_ref);
CREATE INDEX idx_publish_results_publish_job_uid ON publish_results(publish_job_uid);
"""

MIGRATION_6_CHECKSUM_SHA256 = hashlib.sha256(
    MIGRATION_6_SQL.encode("utf-8")
).hexdigest()
_EXPECTED_SCHEMA_SQL: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class QueueSettings:
    max_jobs: int
    lease_ttl_seconds: int
    resume_limit: int
    sqlite_busy_timeout_ms: int


@dataclass(frozen=True, slots=True)
class QueueUnavailableState:
    reason_code: str


@dataclass(frozen=True, slots=True)
class ArtifactQuotaSettings:
    max_bytes: int
    job_quota_bytes: int
    total_quota_bytes: int
    min_free_bytes: int


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_uid: str
    run_uid: str
    job_spec_hash_sha256: str
    canonical_job_spec_json: str
    capability_hash_sha256: str
    job_type: str
    state: str
    created_at: str
    updated_at: str
    accepted_at: str
    queued_at: str
    started_at: str | None
    finished_at: str | None
    terminal_at: str | None
    cancel_requested: bool
    cancel_reason_code: str | None
    resume_requested: bool
    resume_reason_code: str | None
    resume_count: int
    resume_limit: int
    last_checkpoint_ref: str | None
    attempt_count: int
    last_failure_code: str | None
    last_failure_class: str | None
    runner_kind: str
    lease_owner: str | None
    lease_acquired_at: str | None
    lease_expires_at: str | None
    row_version: int


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_uid: str
    job_uid: str
    run_uid: str
    role: str
    artifact_ref: str | None
    sha256: str | None
    size_bytes: int | None
    expected_size_bytes: int | None
    reserved_size_bytes: int
    status: str
    created_at: str
    updated_at: str
    verified_at: str | None
    deleted_at: str | None
    failure_code: str | None
    row_version: int


@dataclass(frozen=True, slots=True)
class PublishResultRecord:
    job_uid: str
    publish_job_uid: str
    target_ref: str
    publish_spec_hash_sha256: str
    source_artifact_fingerprint_sha256: str
    provenance_fingerprint_sha256: str
    ollama_digest: str | None
    idempotent: bool
    created_at: str
    completed_at: str

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": publish_contract.PUBLISH_RESULT_SCHEMA_VERSION,
            "publish_job_uid": self.publish_job_uid,
            "target_ref": self.target_ref,
            "publish_spec_hash_sha256": self.publish_spec_hash_sha256,
            "source_artifact_fingerprint_sha256": (
                self.source_artifact_fingerprint_sha256
            ),
            "provenance_fingerprint_sha256": self.provenance_fingerprint_sha256,
            "ollama_digest": self.ollama_digest,
            "idempotent": self.idempotent,
            "completed_at": self.completed_at,
        }


class QueueUnavailable(RuntimeError):
    def __init__(self, reason_code: str = "sqlite_unavailable") -> None:
        self.reason_code = _safe_reason(reason_code)
        super().__init__(self.reason_code)


class QueueFullError(RuntimeError):
    pass


class ResumeLimitExceeded(RuntimeError):
    pass


class LeaseUnavailable(RuntimeError):
    pass


class InvalidStateTransition(RuntimeError):
    pass


class ArtifactConflict(RuntimeError):
    pass


class QueueStore:
    def __init__(
        self,
        *,
        db_path: Path,
        settings: QueueSettings,
        now_fn=None,
        lease_expiry_sink: Callable[[int], None] | None = None,
    ) -> None:
        self.db_path = db_path
        self.settings = settings
        self._now_fn = now_fn
        self._lease_expiry_sink = lease_expiry_sink
        self._lease_expiry_pending_counts: dict[int, int] = {}

    @classmethod
    def open(
        cls,
        data_dir: Path,
        settings: QueueSettings,
        *,
        now_fn=None,
        lease_expiry_sink: Callable[[int], None] | None = None,
    ) -> "QueueStore":
        db_path = Path(data_dir) / QUEUE_FILENAME
        store = cls(
            db_path=db_path,
            settings=settings,
            now_fn=now_fn,
            lease_expiry_sink=lease_expiry_sink,
        )
        store.migrate()
        return store

    @classmethod
    def open_from_config(
        cls,
        cfg: object,
        *,
        now_fn=None,
        lease_expiry_sink: Callable[[int], None] | None = None,
    ) -> "QueueStore":
        reason = getattr(cfg, "queue_config_error_code", None)
        if reason is not None:
            raise QueueUnavailable(reason)
        return cls.open(
            Path(getattr(cfg, "data_dir")),
            QueueSettings(
                max_jobs=int(getattr(cfg, "queue_max_jobs")),
                lease_ttl_seconds=int(getattr(cfg, "lease_ttl_seconds")),
                resume_limit=int(getattr(cfg, "resume_limit")),
                sqlite_busy_timeout_ms=int(getattr(cfg, "sqlite_busy_timeout_ms")),
            ),
            now_fn=now_fn,
            lease_expiry_sink=lease_expiry_sink,
        )

    def set_lease_expiry_sink(self, sink: Callable[[int], None] | None) -> None:
        self._lease_expiry_sink = sink

    def migrate(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=False, exist_ok=True)
        except OSError as exc:
            raise QueueUnavailable(_classify_os_error(exc)) from exc
        try:
            with self._connect() as conn:
                self._integrity_check(conn)
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if not self._table_exists(conn, "schema_migrations"):
                        if self._user_tables(conn):
                            raise QueueUnavailable("migration_failed")
                        self._apply_migration_1(conn)
                    else:
                        self._validate_existing_schema(conn)
                    self._foreign_key_check(conn)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.execute("PRAGMA foreign_keys=ON")
        except QueueUnavailable:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc, default="migration_failed")) from exc

    def put_job(
        self,
        *,
        job_uid: str,
        job_spec: dict[str, Any],
        capability_hash_sha256: str,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        canonical, job_hash = job_stubs.canonicalize_job_spec(job_spec)
        job_type = job_stubs.job_type_from_spec(job_spec)
        if not _is_hash(capability_hash_sha256):
            raise job_stubs.JobValidationError("capability_hash_sha256 is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                existing = self._fetch_job(conn, job_uid)
                if existing is not None:
                    if existing.job_spec_hash_sha256 != job_hash:
                        raise job_stubs.JobConflictError()
                    return existing
                active_count = self._active_job_count(conn)
                if active_count >= self.settings.max_jobs:
                    raise QueueFullError()
                run_uid = f"run_{uuid.uuid4().hex}"
                conn.execute(
                    """
                    insert into jobs (
                        job_uid, run_uid, job_spec_hash_sha256,
                        canonical_job_spec_json, capability_hash_sha256, job_type, state,
                        created_at, updated_at, accepted_at, queued_at,
                        resume_limit
                    )
                    values (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        job_uid,
                        run_uid,
                        job_hash,
                        canonical,
                        capability_hash_sha256,
                        job_type,
                        now,
                        now,
                        now,
                        now,
                        self.settings.resume_limit,
                    ),
                )
                return self._require_job(conn, job_uid)
        except (QueueFullError, job_stubs.JobConflictError, job_stubs.JobValidationError):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def get_job(self, job_uid: str) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._fetch_job(conn, job_uid)
                if job is None:
                    raise job_stubs.JobNotFoundError()
                return job
        except job_stubs.JobNotFoundError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def request_cancel(
        self,
        *,
        job_uid: str,
        reason_code: str | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if reason_code is not None and not job_stubs.validate_reason_code(reason_code):
            raise job_stubs.JobValidationError("cancel reason_code is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_non_terminal(job)
                if job.state not in {
                    "queued",
                    "lease_expired",
                    "lease_acquired",
                    "running",
                    "cancel_requested",
                }:
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    update jobs
                    set state = 'cancel_requested',
                        cancel_requested = 1,
                        cancel_reason_code = coalesce(?, cancel_reason_code),
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (reason_code, now, job_uid),
                )
                return self._require_job(conn, job_uid)
        except (job_stubs.JobNotFoundError, job_stubs.JobValidationError, InvalidStateTransition):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def request_resume(
        self,
        *,
        job_uid: str,
        reason_code: str | None = None,
        last_checkpoint_ref: str | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if reason_code is not None and not job_stubs.validate_reason_code(reason_code):
            raise job_stubs.JobValidationError("resume reason_code is invalid")
        if last_checkpoint_ref is not None and not job_stubs.validate_last_checkpoint_ref(last_checkpoint_ref):
            raise job_stubs.JobValidationError("last_checkpoint_ref is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_non_terminal(job)
                if _has_any_lease_metadata(job):
                    raise InvalidStateTransition()
                if job.state == "resume_requested":
                    return job
                if job.state not in {"queued", "lease_expired"}:
                    raise InvalidStateTransition()
                if job.resume_count >= job.resume_limit:
                    raise ResumeLimitExceeded()
                conn.execute(
                    """
                    update jobs
                    set state = 'resume_requested',
                        resume_requested = 1,
                        resume_reason_code = coalesce(?, resume_reason_code),
                        resume_count = resume_count + 1,
                        last_checkpoint_ref = coalesce(?, last_checkpoint_ref),
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (reason_code, last_checkpoint_ref, now, job_uid),
                )
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            InvalidStateTransition,
            ResumeLimitExceeded,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def acquire_lease(
        self,
        *,
        job_uid: str,
        owner: str,
        ttl_seconds: int | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if not job_stubs.validate_lease_owner(owner):
            raise job_stubs.JobValidationError("lease_owner is invalid")
        ttl = self.settings.lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 30 or ttl > 86400:
            raise job_stubs.JobValidationError("lease ttl is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_non_terminal(job)
                if self._active_worker_job_count(conn, now) > 0:
                    raise LeaseUnavailable()
                if _has_active_lease(job, now):
                    raise LeaseUnavailable()
                if job.state != "queued":
                    raise InvalidStateTransition()
                self._acquire_lease_in_conn(conn, job, owner, ttl, now)
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            LeaseUnavailable,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def next_runnable_job(
        self,
        owner: str,
        *,
        job_types: set[str] | tuple[str, ...] | list[str] | None = None,
    ) -> JobRecord | None:
        if not job_stubs.validate_lease_owner(owner):
            raise job_stubs.JobValidationError("lease_owner is invalid")
        job_type_filter = _normalize_job_type_filter(job_types)
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                if self._active_worker_job_count(conn, now) > 0:
                    return None
                params: list[str] = []
                job_type_clause = ""
                if job_type_filter is not None:
                    job_type_clause = (
                        " and job_type in ("
                        + ",".join("?" for _ in job_type_filter)
                        + ")"
                    )
                    params.extend(job_type_filter)
                row = conn.execute(
                    f"""
                    select *
                    from jobs
                    where state = 'queued'
                      and lease_owner is null
                      and lease_acquired_at is null
                      and lease_expires_at is null
                      {job_type_clause}
                    order by queued_at, created_at, job_uid
                    limit 1
                    """,
                    params,
                ).fetchone()
                if row is None:
                    return None
                job = _row_to_job(row)
                self._acquire_lease_in_conn(
                    conn,
                    job,
                    owner,
                    self.settings.lease_ttl_seconds,
                    now,
                )
                return self._require_job(conn, job.job_uid)
        except job_stubs.JobValidationError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def next_queued_cancel_job_without_lease(self) -> JobRecord | None:
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                row = conn.execute(
                    """
                    select *
                    from jobs
                    where state = 'cancel_requested'
                      and lease_owner is null
                      and lease_acquired_at is null
                      and lease_expires_at is null
                    order by updated_at, created_at, job_uid
                    limit 1
                    """
                ).fetchone()
                return None if row is None else _row_to_job(row)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def next_gate_closed_resume_job_without_lease(self) -> JobRecord | None:
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                row = conn.execute(
                    """
                    select *
                    from jobs
                    where state = 'resume_requested'
                      and lease_owner is null
                      and lease_acquired_at is null
                      and lease_expires_at is null
                    order by updated_at, created_at, job_uid
                    limit 1
                    """
                ).fetchone()
                return None if row is None else _row_to_job(row)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def renew_lease(
        self,
        *,
        job_uid: str,
        owner: str,
        ttl_seconds: int | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if not job_stubs.validate_lease_owner(owner):
            raise job_stubs.JobValidationError("lease_owner is invalid")
        ttl = self.settings.lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 30 or ttl > 86400:
            raise job_stubs.JobValidationError("lease ttl is invalid")
        now = self._now()
        expires_at = self._format_now_plus(ttl)
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                if job.lease_owner != owner or not _has_active_lease(job, now):
                    raise LeaseUnavailable()
                conn.execute(
                    """
                    update jobs
                    set lease_expires_at = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (expires_at, now, job_uid),
                )
                conn.execute(
                    """
                    update attempts
                    set updated_at = ?
                    where job_uid = ? and attempt_number = ?
                    """,
                    (now, job_uid, job.attempt_count),
                )
                return self._require_job(conn, job_uid)
        except (job_stubs.JobNotFoundError, job_stubs.JobValidationError, LeaseUnavailable):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def release_lease(self, *, job_uid: str, owner: str) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if not job_stubs.validate_lease_owner(owner):
            raise job_stubs.JobValidationError("lease_owner is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                if job.lease_owner is None:
                    return job
                if job.lease_owner != owner:
                    raise LeaseUnavailable()
                next_state = _state_after_clean_release(job)
                _require_transition(job.state, next_state)
                conn.execute(
                    """
                    update jobs
                    set state = ?,
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (next_state, now, job_uid),
                )
                conn.execute(
                    """
                    update attempts
                    set state = 'released',
                        lease_owner = null,
                        updated_at = ?
                    where job_uid = ? and attempt_number = ?
                    """,
                    (now, job_uid, job.attempt_count),
                )
                return self._require_job(conn, job_uid)
        except (job_stubs.JobNotFoundError, job_stubs.JobValidationError, LeaseUnavailable):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def expire_stale_leases(self) -> int:
        now = self._now()
        try:
            with self._transaction() as conn:
                return self._expire_stale_leases_in_conn(conn, now)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_running(
        self,
        *,
        job_uid: str,
        owner: str,
        runner_kind: str = "sft_subprocess",
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        _validate_owner_or_raise(owner)
        if runner_kind not in RUNNER_KINDS:
            raise job_stubs.JobValidationError("runner_kind is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_owned_active_lease(job, owner, now)
                if job.state != "lease_acquired":
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    update jobs
                    set state = 'running',
                        started_at = ?,
                        finished_at = null,
                        terminal_at = null,
                        runner_kind = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (now, runner_kind, now, job_uid),
                )
                self._update_attempt(
                    conn,
                    job,
                    now,
                    state="running",
                    started_at=now,
                )
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            LeaseUnavailable,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_succeeded(self, *, job_uid: str, owner: str) -> JobRecord:
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="succeeded",
            allowed_states={"running"},
            attempt_state="succeeded",
        )

    def mark_publish_succeeded(
        self,
        *,
        job_uid: str,
        owner: str,
        publish_job_uid: str,
        target_ref: str,
        publish_spec_hash_sha256: str,
        source_artifact_fingerprint_sha256: str,
        provenance_fingerprint_sha256: str,
        ollama_digest: str | None = None,
        idempotent: bool = False,
    ) -> tuple[JobRecord, PublishResultRecord]:
        _validate_job_uid_or_raise(job_uid)
        _validate_owner_or_raise(owner)
        _validate_publish_result_fields(
            publish_job_uid=publish_job_uid,
            target_ref=target_ref,
            publish_spec_hash_sha256=publish_spec_hash_sha256,
            source_artifact_fingerprint_sha256=source_artifact_fingerprint_sha256,
            provenance_fingerprint_sha256=provenance_fingerprint_sha256,
            ollama_digest=ollama_digest,
        )
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_owned_active_lease(job, owner, now)
                if job.job_type != publish_contract.PUBLISH_JOB_TYPE:
                    raise InvalidStateTransition()
                if job.state != "running":
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    insert into publish_results (
                        job_uid, publish_job_uid, target_ref,
                        publish_spec_hash_sha256,
                        source_artifact_fingerprint_sha256,
                        provenance_fingerprint_sha256, ollama_digest, idempotent,
                        created_at, completed_at
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_uid,
                        publish_job_uid,
                        target_ref,
                        publish_spec_hash_sha256,
                        source_artifact_fingerprint_sha256,
                        provenance_fingerprint_sha256,
                        ollama_digest,
                        1 if idempotent else 0,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    update jobs
                    set state = 'succeeded',
                        finished_at = ?,
                        terminal_at = ?,
                        last_failure_code = null,
                        last_failure_class = null,
                        last_failure_message_redacted = null,
                        last_failure_at = null,
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (now, now, now, job_uid),
                )
                self._update_attempt(
                    conn,
                    job,
                    now,
                    state="succeeded",
                    finished_at=now,
                    clear_lease=True,
                )
                return self._require_job(conn, job_uid), self._require_publish_result(
                    conn,
                    job_uid,
                )
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            LeaseUnavailable,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.IntegrityError as exc:
            raise InvalidStateTransition() from exc
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_failed(
        self,
        *,
        job_uid: str,
        owner: str,
        failure_code: str,
        failure_class: str,
    ) -> JobRecord:
        _validate_failure_code_or_raise(failure_code)
        _validate_failure_code_or_raise(failure_class, label="failure_class")
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="failed",
            allowed_states={"lease_acquired", "running"},
            failure_code=failure_code,
            failure_class=failure_class,
            attempt_state="failed",
        )

    def mark_publish_verification_unknown(
        self,
        *,
        job_uid: str,
        owner: str,
    ) -> JobRecord:
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="failed",
            allowed_states={"lease_acquired", "running", "cancel_requested"},
            failure_code="publish_verification_unknown",
            failure_class="publish",
            attempt_state="failed",
        )

    def mark_policy_blocked(
        self,
        *,
        job_uid: str,
        owner: str,
        reason_code: str = "policy_blocked",
    ) -> JobRecord:
        _validate_failure_code_or_raise(reason_code)
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="failed",
            allowed_states={"lease_acquired"},
            failure_code=reason_code,
            failure_class="policy",
            attempt_state="policy_blocked",
        )

    def mark_running_canceled(
        self,
        *,
        job_uid: str,
        owner: str,
        reason_code: str | None = None,
    ) -> JobRecord:
        if reason_code is not None and not job_stubs.validate_reason_code(reason_code):
            raise job_stubs.JobValidationError("cancel reason_code is invalid")
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="canceled",
            allowed_states={"lease_acquired", "running", "cancel_requested"},
            cancel_reason_code=reason_code,
            attempt_state="canceled",
        )

    def mark_queued_canceled(
        self,
        *,
        job_uid: str,
        reason_code: str | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if reason_code is not None and not job_stubs.validate_reason_code(reason_code):
            raise job_stubs.JobValidationError("cancel reason_code is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                if job.state != "cancel_requested" or _has_any_lease_metadata(job):
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    update jobs
                    set state = 'canceled',
                        started_at = null,
                        finished_at = ?,
                        terminal_at = ?,
                        cancel_requested = 1,
                        cancel_reason_code = coalesce(?, cancel_reason_code),
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (now, now, reason_code, now, job_uid),
                )
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_resume_gate_closed(self, *, job_uid: str) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                if job.state != "resume_requested" or _has_any_lease_metadata(job):
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    update jobs
                    set state = 'failed',
                        started_at = null,
                        finished_at = ?,
                        terminal_at = ?,
                        last_failure_code = 'resume_not_supported',
                        last_failure_class = 'resume',
                        last_failure_message_redacted = null,
                        last_failure_at = ?,
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (now, now, now, now, job_uid),
                )
                return self._require_job(conn, job_uid)
        except (job_stubs.JobNotFoundError, InvalidStateTransition):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_lease_lost(self, *, job_uid: str, owner: str) -> JobRecord:
        return self._mark_terminal_from_owned_execution(
            job_uid=job_uid,
            owner=owner,
            target_state="failed",
            allowed_states={"running"},
            failure_code="lease_lost",
            failure_class="lease",
            attempt_state="failed",
        )

    def _mark_terminal_from_owned_execution(
        self,
        *,
        job_uid: str,
        owner: str,
        target_state: str,
        allowed_states: set[str],
        failure_code: str | None = None,
        failure_class: str | None = None,
        cancel_reason_code: str | None = None,
        attempt_state: str | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        _validate_owner_or_raise(owner)
        if target_state not in {"succeeded", "failed", "canceled"}:
            raise InvalidStateTransition()
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_owned_active_lease(job, owner, now)
                if job.state not in allowed_states:
                    raise InvalidStateTransition()
                conn.execute(
                    """
                    update jobs
                    set state = ?,
                        finished_at = ?,
                        terminal_at = ?,
                        cancel_reason_code = coalesce(?, cancel_reason_code),
                        last_failure_code = ?,
                        last_failure_class = ?,
                        last_failure_message_redacted = null,
                        last_failure_at = ?,
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (
                        target_state,
                        now,
                        now,
                        cancel_reason_code,
                        failure_code,
                        failure_class,
                        now if failure_code else None,
                        now,
                        job_uid,
                    ),
                )
                if attempt_state is not None:
                    self._update_attempt(
                        conn,
                        job,
                        now,
                        state=attempt_state,
                        finished_at=now,
                        failure_code=failure_code,
                        failure_class=failure_class,
                        clear_lease=True,
                    )
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            LeaseUnavailable,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_failed_metadata_only(
        self,
        *,
        job_uid: str,
        failure_code: str | None = None,
        failure_message_redacted: str | None = None,
    ) -> JobRecord:
        _validate_job_uid_or_raise(job_uid)
        if failure_message_redacted is not None:
            raise job_stubs.JobValidationError("failure messages are not persisted in sprint 6")
        if failure_code is not None and not job_stubs.validate_reason_code(failure_code):
            raise job_stubs.JobValidationError("failure_code is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                job = self._require_job(conn, job_uid)
                _require_non_terminal(job)
                if _has_any_lease_metadata(job) or job.started_at is not None:
                    raise InvalidStateTransition()
                _require_transition(job.state, "failed_metadata_only")
                conn.execute(
                    """
                    update jobs
                    set state = 'failed_metadata_only',
                        started_at = null,
                        finished_at = null,
                        terminal_at = ?,
                        lease_owner = null,
                        lease_acquired_at = null,
                        lease_expires_at = null,
                        last_failure_code = ?,
                        last_failure_class = null,
                        last_failure_message_redacted = ?,
                        last_failure_at = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where job_uid = ?
                    """,
                    (
                        now,
                        failure_code,
                        None,
                        now if failure_code else None,
                        now,
                        job_uid,
                    ),
                )
                conn.execute(
                    """
            update attempts
            set state = 'failed_metadata_only',
                failure_code = coalesce(?, failure_code),
                failure_class = null,
                failure_message_redacted = null,
                updated_at = ?
            where job_uid = ? and attempt_number = ?
                    """,
                    (
                        failure_code,
                        now,
                        job_uid,
                        job.attempt_count,
                    ),
                )
                return self._require_job(conn, job_uid)
        except (
            job_stubs.JobNotFoundError,
            job_stubs.JobValidationError,
            InvalidStateTransition,
        ):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def queue_stats(self) -> dict[str, Any]:
        now = self._now()
        try:
            with self._transaction() as conn:
                self._expire_stale_leases_in_conn(conn, now)
                depth = self._active_job_count(conn)
                active_jobs = self._active_worker_job_count(conn, now)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc
        full = depth >= self.settings.max_jobs
        return {
            "queue_enabled": True,
            "lease_enabled": True,
            "resume_persistence_enabled": True,
            "queue_depth": depth,
            "active_jobs": active_jobs,
            "queue_status": "full" if full else "ready",
            "queue_degraded_reason": "queue_full" if full else None,
            "max_jobs": self.settings.max_jobs,
        }

    def reserve_artifact(
        self,
        *,
        job_uid: str,
        role: str,
        expected_size_bytes: int,
        quotas: ArtifactQuotaSettings,
        staging_free_bytes: int,
        additional_staging_bytes: int | None = None,
    ) -> ArtifactRecord:
        _validate_job_uid_or_raise(job_uid)
        _validate_artifact_role_or_raise(role)
        _validate_artifact_expected_size_or_raise(expected_size_bytes)
        _validate_artifact_quotas_or_raise(quotas)
        if (
            isinstance(staging_free_bytes, bool)
            or not isinstance(staging_free_bytes, int)
            or staging_free_bytes < 0
        ):
            raise job_stubs.JobValidationError("staging_free_bytes is invalid")
        if additional_staging_bytes is None:
            additional_staging_bytes = expected_size_bytes
        if (
            isinstance(additional_staging_bytes, bool)
            or not isinstance(additional_staging_bytes, int)
            or additional_staging_bytes < 0
            or additional_staging_bytes > expected_size_bytes
        ):
            raise job_stubs.JobValidationError("additional_staging_bytes is invalid")
        now = self._now()
        try:
            with self._transaction() as conn:
                job = self._require_job(conn, job_uid)
                artifact_uid = _new_artifact_uid()
                reason = self._artifact_quota_blocker(
                    conn,
                    job_uid=job_uid,
                    expected_size_bytes=expected_size_bytes,
                    quotas=quotas,
                    staging_free_bytes=staging_free_bytes,
                    additional_staging_bytes=additional_staging_bytes,
                )
                if reason is not None:
                    self._insert_artifact_failure(
                        conn,
                        artifact_uid=artifact_uid,
                        job=job,
                        role=role,
                        status="quota_blocked",
                        failure_code=reason,
                        now=now,
                    )
                    return self._require_artifact(conn, artifact_uid)
                conn.execute(
                    """
                    insert into artifacts (
                        artifact_uid, job_uid, run_uid, role,
                        expected_size_bytes, reserved_size_bytes, status,
                        created_at, updated_at
                    )
                    values (?, ?, ?, ?, ?, ?, 'staging', ?, ?)
                    """,
                    (
                        artifact_uid,
                        job.job_uid,
                        job.run_uid,
                        role,
                        expected_size_bytes,
                        expected_size_bytes,
                        now,
                        now,
                    ),
                )
                return self._require_artifact(conn, artifact_uid)
        except job_stubs.JobValidationError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def get_artifact(self, artifact_uid: str) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        try:
            with self._transaction() as conn:
                return self._require_artifact(conn, artifact_uid)
        except job_stubs.JobNotFoundError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def get_artifact_by_ref(self, artifact_ref: str) -> ArtifactRecord:
        if not _is_safe_artifact_ref(artifact_ref):
            raise job_stubs.JobValidationError("artifact_ref is invalid")
        try:
            with self._transaction() as conn:
                row = conn.execute(
                    "select * from artifacts where artifact_ref = ?",
                    (artifact_ref,),
                ).fetchone()
                if row is None:
                    raise job_stubs.JobNotFoundError()
                return _row_to_artifact(row)
        except job_stubs.JobNotFoundError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def list_staged_artifacts_for_job(self, job_uid: str) -> list[ArtifactRecord]:
        _validate_job_uid_or_raise(job_uid)
        try:
            with self._transaction() as conn:
                self._require_job(conn, job_uid)
                rows = conn.execute(
                    """
                    select *
                    from artifacts
                    where job_uid = ?
                      and status = 'staged'
                      and artifact_ref is not null
                      and sha256 is not null
                      and size_bytes is not null
                    order by role, verified_at desc, artifact_uid
                    """,
                    (job_uid,),
                ).fetchall()
                return [_row_to_artifact(row) for row in rows]
        except job_stubs.JobNotFoundError:
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def get_publish_result(self, job_uid: str) -> PublishResultRecord | None:
        _validate_job_uid_or_raise(job_uid)
        try:
            with self._transaction() as conn:
                row = conn.execute(
                    "select * from publish_results where job_uid = ?",
                    (job_uid,),
                ).fetchone()
                return None if row is None else _row_to_publish_result(row)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def find_publish_result_by_target(
        self,
        *,
        target_ref: str,
        publish_spec_hash_sha256: str,
        source_artifact_fingerprint_sha256: str,
    ) -> PublishResultRecord | None:
        _validate_publish_result_fields(
            publish_job_uid="lookup",
            target_ref=target_ref,
            publish_spec_hash_sha256=publish_spec_hash_sha256,
            source_artifact_fingerprint_sha256=source_artifact_fingerprint_sha256,
            provenance_fingerprint_sha256="0" * 64,
            ollama_digest=None,
        )
        try:
            with self._transaction() as conn:
                row = conn.execute(
                    """
                    select *
                    from publish_results
                    where target_ref = ?
                      and publish_spec_hash_sha256 = ?
                      and source_artifact_fingerprint_sha256 = ?
                    order by completed_at desc
                    limit 1
                    """,
                    (
                        target_ref,
                        publish_spec_hash_sha256,
                        source_artifact_fingerprint_sha256,
                    ),
                ).fetchone()
                return None if row is None else _row_to_publish_result(row)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def find_artifact_by_content(
        self,
        *,
        job_uid: str,
        role: str,
        sha256: str,
        size_bytes: int,
    ) -> ArtifactRecord | None:
        _validate_job_uid_or_raise(job_uid)
        _validate_artifact_role_or_raise(role)
        _validate_hash_or_raise(sha256, label="sha256")
        _validate_artifact_expected_size_or_raise(size_bytes, label="size_bytes")
        try:
            with self._transaction() as conn:
                row = self._find_artifact_by_content_in_conn(
                    conn,
                    job_uid=job_uid,
                    role=role,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
                return None if row is None else _row_to_artifact(row)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_artifact_staged(
        self,
        *,
        artifact_uid: str,
        sha256: str,
        size_bytes: int,
    ) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        _validate_hash_or_raise(sha256, label="sha256")
        _validate_artifact_expected_size_or_raise(size_bytes, label="size_bytes")
        now = self._now()
        try:
            with self._transaction() as conn:
                artifact = self._require_artifact(conn, artifact_uid)
                if artifact.status != "staging":
                    raise ArtifactConflict()
                if artifact.expected_size_bytes != size_bytes:
                    conn.execute(
                        """
                        update artifacts
                        set status = 'verification_failed',
                            sha256 = ?,
                            size_bytes = ?,
                            reserved_size_bytes = 0,
                            failure_code = 'artifact_size_mismatch',
                            updated_at = ?,
                            row_version = row_version + 1
                        where artifact_uid = ?
                        """,
                        (sha256, size_bytes, now, artifact_uid),
                    )
                    return self._require_artifact(conn, artifact_uid)
                existing = self._find_artifact_by_content_in_conn(
                    conn,
                    job_uid=artifact.job_uid,
                    role=artifact.role,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
                if existing is not None and existing["artifact_uid"] != artifact_uid:
                    conn.execute(
                        """
                        update artifacts
                        set status = 'verification_failed',
                            sha256 = ?,
                            size_bytes = ?,
                            reserved_size_bytes = 0,
                            failure_code = 'artifact_duplicate',
                            updated_at = ?,
                            row_version = row_version + 1
                        where artifact_uid = ?
                        """,
                        (sha256, size_bytes, now, artifact_uid),
                    )
                    return _row_to_artifact(existing)
                artifact_ref = _artifact_ref(
                    artifact_uid=artifact.artifact_uid,
                    run_uid=artifact.run_uid,
                    role=artifact.role,
                    sha256=sha256,
                )
                conn.execute(
                    """
                    update artifacts
                    set status = 'staged',
                        artifact_ref = ?,
                        sha256 = ?,
                        size_bytes = ?,
                        reserved_size_bytes = 0,
                        verified_at = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where artifact_uid = ?
                    """,
                    (artifact_ref, sha256, size_bytes, now, now, artifact_uid),
                )
                return self._require_artifact(conn, artifact_uid)
        except (job_stubs.JobNotFoundError, ArtifactConflict):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_artifact_verification_failed(
        self,
        *,
        artifact_uid: str,
        failure_code: str,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        _validate_failure_code_or_raise(failure_code)
        if sha256 is not None:
            _validate_hash_or_raise(sha256, label="sha256")
        if size_bytes is not None:
            _validate_artifact_expected_size_or_raise(size_bytes, label="size_bytes")
        now = self._now()
        try:
            with self._transaction() as conn:
                artifact = self._require_artifact(conn, artifact_uid)
                if artifact.status != "staging":
                    raise ArtifactConflict()
                conn.execute(
                    """
                    update artifacts
                    set status = 'verification_failed',
                        sha256 = ?,
                        size_bytes = ?,
                        reserved_size_bytes = 0,
                        failure_code = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where artifact_uid = ?
                    """,
                    (sha256, size_bytes, failure_code, now, artifact_uid),
                )
                return self._require_artifact(conn, artifact_uid)
        except (job_stubs.JobNotFoundError, ArtifactConflict):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_artifact_disk_full(self, *, artifact_uid: str) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        now = self._now()
        try:
            with self._transaction() as conn:
                artifact = self._require_artifact(conn, artifact_uid)
                if artifact.status != "staging":
                    raise ArtifactConflict()
                conn.execute(
                    """
                    update artifacts
                    set status = 'disk_full',
                        sha256 = null,
                        size_bytes = null,
                        expected_size_bytes = null,
                        reserved_size_bytes = 0,
                        failure_code = 'disk_full',
                        updated_at = ?,
                        row_version = row_version + 1
                    where artifact_uid = ?
                    """,
                    (now, artifact_uid),
                )
                return self._require_artifact(conn, artifact_uid)
        except (job_stubs.JobNotFoundError, ArtifactConflict):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_artifact_cleanup_pending(self, *, artifact_uid: str) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        now = self._now()
        try:
            with self._transaction() as conn:
                artifact = self._require_artifact(conn, artifact_uid)
                if artifact.status == "cleanup_pending":
                    return artifact
                if artifact.status != "staged":
                    raise ArtifactConflict()
                conn.execute(
                    """
                    update artifacts
                    set status = 'cleanup_pending',
                        updated_at = ?,
                        row_version = row_version + 1
                    where artifact_uid = ?
                    """,
                    (now, artifact_uid),
                )
                return self._require_artifact(conn, artifact_uid)
        except (job_stubs.JobNotFoundError, ArtifactConflict):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def mark_artifact_deleted(self, *, artifact_uid: str) -> ArtifactRecord:
        _validate_artifact_uid_or_raise(artifact_uid)
        now = self._now()
        try:
            with self._transaction() as conn:
                artifact = self._require_artifact(conn, artifact_uid)
                if artifact.status == "deleted":
                    return artifact
                if artifact.status != "cleanup_pending":
                    raise ArtifactConflict()
                conn.execute(
                    """
                    update artifacts
                    set status = 'deleted',
                        deleted_at = ?,
                        updated_at = ?,
                        row_version = row_version + 1
                    where artifact_uid = ?
                    """,
                    (now, now, artifact_uid),
                )
                return self._require_artifact(conn, artifact_uid)
        except (job_stubs.JobNotFoundError, ArtifactConflict):
            raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def list_artifacts_for_cleanup(self) -> list[ArtifactRecord]:
        try:
            with self._transaction() as conn:
                rows = conn.execute(
                    """
                    select *
                    from artifacts
                    where status in (
                        'staging',
                        'verification_failed',
                        'quota_blocked',
                        'disk_full',
                        'cleanup_pending'
                    )
                    order by updated_at, artifact_uid
                    """
                ).fetchall()
                return [_row_to_artifact(row) for row in rows]
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def artifact_quota_usage(self, *, job_uid: str | None = None) -> dict[str, int]:
        if job_uid is not None:
            _validate_job_uid_or_raise(job_uid)
        try:
            with self._transaction() as conn:
                return self._artifact_quota_usage(conn, job_uid=job_uid)
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc

    def artifact_status_summary(self) -> dict[str, dict[str, int]]:
        try:
            with self._transaction() as conn:
                status_rows = conn.execute(
                    """
                    select artifacts.status, count(*) as count
                    from artifacts
                    join jobs on jobs.job_uid = artifacts.job_uid
                    where jobs.state not in (
                        'succeeded',
                        'failed',
                        'canceled',
                        'failed_metadata_only'
                    )
                    group by artifacts.status
                    """
                ).fetchall()
                failure_rows = conn.execute(
                    """
                    select artifacts.failure_code, count(*) as count
                    from artifacts
                    join jobs on jobs.job_uid = artifacts.job_uid
                    where artifacts.failure_code is not null
                      and jobs.state not in (
                          'succeeded',
                          'failed',
                          'canceled',
                          'failed_metadata_only'
                      )
                    group by artifacts.failure_code
                    """
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc
        return {
            "status_counts": {
                str(row["status"]): int(row["count"]) for row in status_rows
            },
            "failure_counts": {
                str(row["failure_code"]): int(row["count"]) for row in failure_rows
            },
        }

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.settings.sqlite_busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute(f"PRAGMA busy_timeout={self.settings.sqlite_busy_timeout_ms}")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        pending_lease_expiries = 0
        try:
            with self._connect() as conn:
                conn_key = id(conn)
                self._lease_expiry_pending_counts[conn_key] = 0
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    yield conn
                    conn.commit()
                    pending_lease_expiries = self._lease_expiry_pending_counts.pop(
                        conn_key,
                        0,
                    )
                except Exception:
                    self._lease_expiry_pending_counts.pop(conn_key, None)
                    conn.rollback()
                    raise
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc)) from exc
        self._notify_lease_expiry_sink(pending_lease_expiries)

    def _apply_migration_1(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_1_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (1, self._now(), MIGRATION_1_CHECKSUM_SHA256),
        )
        conn.execute("PRAGMA user_version=1")
        self._apply_migration_2(conn)

    def _apply_migration_2(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_2_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (2, self._now(), MIGRATION_2_CHECKSUM_SHA256),
        )
        conn.execute("PRAGMA user_version=2")
        self._apply_migration_3(conn)

    def _apply_migration_3(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_3_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (3, self._now(), MIGRATION_3_CHECKSUM_SHA256),
        )
        conn.execute("PRAGMA user_version=3")
        self._apply_migration_4(conn)

    def _apply_migration_4(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_4_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (4, self._now(), MIGRATION_4_CHECKSUM_SHA256),
        )
        conn.execute("PRAGMA user_version=4")
        self._apply_migration_5(conn)

    def _apply_migration_5(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_5_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (5, self._now(), MIGRATION_5_CHECKSUM_SHA256),
        )
        conn.execute("PRAGMA user_version=5")
        self._apply_migration_6(conn)

    def _apply_migration_6(self, conn: sqlite3.Connection) -> None:
        for statement in _sql_statements(MIGRATION_6_SQL):
            conn.execute(statement)
        conn.execute(
            """
            insert into schema_migrations(version, applied_at, checksum_sha256)
            values (?, ?, ?)
            """,
            (6, self._now(), MIGRATION_6_CHECKSUM_SHA256),
        )
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self._validate_required_objects(conn)

    def _validate_existing_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "select version, checksum_sha256 from schema_migrations order by version"
        ).fetchall()
        if not rows:
            raise QueueUnavailable("migration_failed")
        max_version = max(int(row["version"]) for row in rows)
        if max_version > SCHEMA_VERSION:
            raise QueueUnavailable("migration_failed")
        user_version = self._user_version(conn)
        if user_version > SCHEMA_VERSION or user_version != max_version:
            raise QueueUnavailable("migration_failed")
        version_1 = [row for row in rows if int(row["version"]) == 1]
        if len(version_1) != 1:
            raise QueueUnavailable("migration_failed")
        if version_1[0]["checksum_sha256"] != MIGRATION_1_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        version_2 = [row for row in rows if int(row["version"]) == 2]
        if max_version == 1:
            self._apply_migration_2(conn)
            return
        if len(version_2) != 1:
            raise QueueUnavailable("migration_failed")
        if version_2[0]["checksum_sha256"] != MIGRATION_2_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        version_3 = [row for row in rows if int(row["version"]) == 3]
        if max_version == 2:
            self._apply_migration_3(conn)
            return
        if len(version_3) != 1:
            raise QueueUnavailable("migration_failed")
        if version_3[0]["checksum_sha256"] != MIGRATION_3_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        version_4 = [row for row in rows if int(row["version"]) == 4]
        if max_version == 3:
            self._apply_migration_4(conn)
            return
        if len(version_4) != 1:
            raise QueueUnavailable("migration_failed")
        if version_4[0]["checksum_sha256"] != MIGRATION_4_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        version_5 = [row for row in rows if int(row["version"]) == 5]
        if max_version == 4:
            self._apply_migration_5(conn)
            return
        if len(version_5) != 1:
            raise QueueUnavailable("migration_failed")
        if version_5[0]["checksum_sha256"] != MIGRATION_5_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        version_6 = [row for row in rows if int(row["version"]) == 6]
        if max_version == 5:
            self._apply_migration_6(conn)
            return
        if len(version_6) != 1:
            raise QueueUnavailable("migration_failed")
        if version_6[0]["checksum_sha256"] != MIGRATION_6_CHECKSUM_SHA256:
            raise QueueUnavailable("migration_failed")
        self._validate_required_objects(conn)

    def _validate_required_objects(self, conn: sqlite3.Connection) -> None:
        expected_objects = {
            ("table", "schema_migrations"),
            ("table", "jobs"),
            ("table", "attempts"),
            ("table", "artifacts"),
            ("table", "publish_results"),
            ("index", "idx_jobs_state_created_at"),
            ("index", "idx_jobs_updated_at"),
            ("index", "idx_jobs_active_lease_expiry"),
            ("index", "idx_jobs_cancel_requested"),
            ("index", "idx_jobs_resume_requested"),
            ("index", "idx_jobs_job_type_state_created_at"),
            ("index", "idx_attempts_job_uid_attempt_number"),
            ("index", "idx_jobs_job_uid_run_uid"),
            ("index", "idx_artifacts_job_status_created_at"),
            ("index", "idx_artifacts_run_uid"),
            ("index", "idx_artifacts_cleanup"),
            ("index", "idx_artifacts_role_status"),
            ("index", "idx_artifacts_ref_unique"),
            ("index", "idx_artifacts_staged_job_role_hash_size"),
            ("index", "idx_publish_results_target_ref"),
            ("index", "idx_publish_results_publish_job_uid"),
        }
        actual_objects = {
            (row["type"], row["name"])
            for row in conn.execute(
                """
                select type, name
                from sqlite_master
                where name not like 'sqlite_%'
                """
            ).fetchall()
        }
        if actual_objects != expected_objects:
            raise QueueUnavailable("migration_failed")
        actual_sql = {
            row["name"]: _normalize_sql(row["sql"])
            for row in conn.execute(
                """
                select name, sql
                from sqlite_master
                where name in (
                    'schema_migrations',
                    'jobs',
                    'attempts',
                    'artifacts',
                    'publish_results',
                    'idx_jobs_state_created_at',
                    'idx_jobs_updated_at',
                    'idx_jobs_active_lease_expiry',
                    'idx_jobs_cancel_requested',
                    'idx_jobs_resume_requested',
                    'idx_jobs_job_type_state_created_at',
                    'idx_attempts_job_uid_attempt_number',
                    'idx_jobs_job_uid_run_uid',
                    'idx_artifacts_job_status_created_at',
                    'idx_artifacts_run_uid',
                    'idx_artifacts_cleanup',
                    'idx_artifacts_role_status',
                    'idx_artifacts_ref_unique',
                    'idx_artifacts_staged_job_role_hash_size',
                    'idx_publish_results_target_ref',
                    'idx_publish_results_publish_job_uid'
                )
                """
            ).fetchall()
        }
        if actual_sql != _expected_schema_sql():
            raise QueueUnavailable("migration_failed")

    def _user_version(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("PRAGMA user_version").fetchone()
        if row is None:
            raise QueueUnavailable("migration_failed")
        try:
            value = int(row[0])
        except (TypeError, ValueError) as exc:
            raise QueueUnavailable("migration_failed") from exc
        if value < 0:
            raise QueueUnavailable("migration_failed")
        return value

    def _integrity_check(self, conn: sqlite3.Connection) -> None:
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc, default="db_corrupt")) from exc
        if result is None or result[0] != "ok":
            raise QueueUnavailable("db_corrupt")

    def _foreign_key_check(self, conn: sqlite3.Connection) -> None:
        try:
            rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.DatabaseError as exc:
            raise QueueUnavailable(_classify_sqlite_error(exc, default="db_corrupt")) from exc
        if rows:
            raise QueueUnavailable("migration_failed")

    def _table_exists(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            """
            select 1
            from sqlite_master
            where type = 'table' and name = ?
            """,
            (name,),
        ).fetchone()
        return row is not None

    def _user_tables(self, conn: sqlite3.Connection) -> set[str]:
        return {
            row["name"]
            for row in conn.execute(
                """
                select name
                from sqlite_master
                where type = 'table' and name not like 'sqlite_%'
                """
            ).fetchall()
        }

    def _fetch_job(self, conn: sqlite3.Connection, job_uid: str) -> JobRecord | None:
        row = conn.execute("select * from jobs where job_uid = ?", (job_uid,)).fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    def _require_job(self, conn: sqlite3.Connection, job_uid: str) -> JobRecord:
        job = self._fetch_job(conn, job_uid)
        if job is None:
            raise job_stubs.JobNotFoundError()
        return job

    def _fetch_artifact(
        self,
        conn: sqlite3.Connection,
        artifact_uid: str,
    ) -> ArtifactRecord | None:
        row = conn.execute(
            "select * from artifacts where artifact_uid = ?",
            (artifact_uid,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_artifact(row)

    def _require_artifact(
        self,
        conn: sqlite3.Connection,
        artifact_uid: str,
    ) -> ArtifactRecord:
        artifact = self._fetch_artifact(conn, artifact_uid)
        if artifact is None:
            raise job_stubs.JobNotFoundError()
        return artifact

    def _require_publish_result(
        self,
        conn: sqlite3.Connection,
        job_uid: str,
    ) -> PublishResultRecord:
        row = conn.execute(
            "select * from publish_results where job_uid = ?",
            (job_uid,),
        ).fetchone()
        if row is None:
            raise job_stubs.JobNotFoundError()
        return _row_to_publish_result(row)

    def _insert_artifact_failure(
        self,
        conn: sqlite3.Connection,
        *,
        artifact_uid: str,
        job: JobRecord,
        role: str,
        status: str,
        failure_code: str,
        now: str,
    ) -> None:
        if status not in {"quota_blocked", "disk_full"}:
            raise ArtifactConflict()
        conn.execute(
            """
            insert into artifacts (
                artifact_uid, job_uid, run_uid, role, status,
                created_at, updated_at, failure_code
            )
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_uid,
                job.job_uid,
                job.run_uid,
                role,
                status,
                now,
                now,
                failure_code,
            ),
        )

    def _artifact_quota_blocker(
        self,
        conn: sqlite3.Connection,
        *,
        job_uid: str,
        expected_size_bytes: int,
        quotas: ArtifactQuotaSettings,
        staging_free_bytes: int,
        additional_staging_bytes: int,
    ) -> str | None:
        usage = self._artifact_quota_usage(conn, job_uid=job_uid)
        if expected_size_bytes > quotas.max_bytes:
            return "artifact_too_large"
        if usage["job_bytes"] + expected_size_bytes > quotas.job_quota_bytes:
            return "artifact_job_quota_exceeded"
        if usage["total_bytes"] + expected_size_bytes > quotas.total_quota_bytes:
            return "artifact_total_quota_exceeded"
        if (
            staging_free_bytes
            - usage["active_reserved_bytes"]
            - additional_staging_bytes
            < quotas.min_free_bytes
        ):
            return "artifact_min_free_blocked"
        return None

    def _artifact_quota_usage(
        self,
        conn: sqlite3.Connection,
        *,
        job_uid: str | None = None,
    ) -> dict[str, int]:
        total = conn.execute(
            """
            select
                coalesce(sum(
                    case
                        when status = 'staging' then reserved_size_bytes
                        when status in ('staged', 'cleanup_pending') then size_bytes
                        else 0
                    end
                ), 0) as total_bytes,
                coalesce(sum(
                    case
                        when status = 'staging' then reserved_size_bytes
                        else 0
                    end
                ), 0) as active_reserved_bytes
            from artifacts
            """
        ).fetchone()
        if job_uid is None:
            job_bytes = 0
        else:
            job_row = conn.execute(
                """
                select coalesce(sum(
                    case
                        when status = 'staging' then reserved_size_bytes
                        when status in ('staged', 'cleanup_pending') then size_bytes
                        else 0
                    end
                ), 0) as job_bytes
                from artifacts
                where job_uid = ?
                """,
                (job_uid,),
            ).fetchone()
            job_bytes = int(job_row["job_bytes"])
        return {
            "total_bytes": int(total["total_bytes"]),
            "job_bytes": job_bytes,
            "active_reserved_bytes": int(total["active_reserved_bytes"]),
        }

    def _find_artifact_by_content_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        job_uid: str,
        role: str,
        sha256: str,
        size_bytes: int,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            select *
            from artifacts
            where job_uid = ?
              and role = ?
              and sha256 = ?
              and size_bytes = ?
              and status in ('staged', 'cleanup_pending')
            order by verified_at, artifact_uid
            limit 1
            """,
            (job_uid, role, sha256, size_bytes),
        ).fetchone()

    def _acquire_lease_in_conn(
        self,
        conn: sqlite3.Connection,
        job: JobRecord,
        owner: str,
        ttl_seconds: int,
        now: str,
    ) -> None:
        expires_at = self._format_now_plus(ttl_seconds)
        attempt_number = job.attempt_count + 1
        attempt_uid = f"attempt_{uuid.uuid4().hex}"
        conn.execute(
            """
            insert into attempts (
                attempt_uid, job_uid, attempt_number, state,
                created_at, updated_at, lease_owner
            )
            values (?, ?, ?, 'lease_acquired', ?, ?, ?)
            """,
            (attempt_uid, job.job_uid, attempt_number, now, now, owner),
        )
        conn.execute(
            """
            update jobs
            set state = 'lease_acquired',
                lease_owner = ?,
                lease_acquired_at = ?,
                lease_expires_at = ?,
                attempt_count = ?,
                updated_at = ?,
                row_version = row_version + 1
            where job_uid = ?
            """,
            (owner, now, expires_at, attempt_number, now, job.job_uid),
        )

    def _update_attempt(
        self,
        conn: sqlite3.Connection,
        job: JobRecord,
        now: str,
        *,
        state: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        failure_code: str | None = None,
        failure_class: str | None = None,
        clear_lease: bool = False,
    ) -> None:
        if job.attempt_count <= 0:
            return
        conn.execute(
            """
            update attempts
            set state = ?,
                started_at = coalesce(?, started_at),
                finished_at = ?,
                lease_owner = case when ? then null else lease_owner end,
                failure_code = ?,
                failure_class = ?,
                failure_message_redacted = null,
                updated_at = ?
            where job_uid = ? and attempt_number = ?
            """,
            (
                state,
                started_at,
                finished_at,
                1 if clear_lease else 0,
                failure_code,
                failure_class,
                now,
                job.job_uid,
                job.attempt_count,
            ),
        )

    def _active_job_count(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """
            select count(*) as count
            from jobs
            where state not in ('succeeded', 'failed', 'canceled', 'failed_metadata_only')
            """
        ).fetchone()
        return int(row["count"])

    def _active_worker_job_count(self, conn: sqlite3.Connection, now: str) -> int:
        row = conn.execute(
            """
            select count(*) as count
            from jobs
            where state in ('lease_acquired', 'running', 'cancel_requested')
              and lease_owner is not null
              and lease_expires_at is not null
              and lease_expires_at >= ?
            """,
            (now,),
        ).fetchone()
        return int(row["count"])

    def _expire_stale_leases_in_conn(self, conn: sqlite3.Connection, now: str) -> int:
        rows = conn.execute(
            """
            select job_uid
            from jobs
            where lease_expires_at is not null
              and lease_expires_at < ?
            """,
            (now,),
        ).fetchall()
        for row in rows:
            self._mark_lease_expired(conn, row["job_uid"], now)
        count = len(rows)
        self._record_lease_expiry_count(conn, count)
        return count

    def _record_lease_expiry_count(self, conn: sqlite3.Connection, count: int) -> None:
        if count <= 0:
            return
        conn_key = id(conn)
        self._lease_expiry_pending_counts[conn_key] = (
            self._lease_expiry_pending_counts.get(conn_key, 0) + count
        )

    def _notify_lease_expiry_sink(self, count: int) -> None:
        if count <= 0 or self._lease_expiry_sink is None:
            return
        try:
            self._lease_expiry_sink(count)
        except Exception:
            pass

    def _mark_lease_expired(self, conn: sqlite3.Connection, job_uid: str, now: str) -> None:
        job = self._require_job(conn, job_uid)
        publish_unknown = (
            job.job_type == publish_contract.PUBLISH_JOB_TYPE
            and (
                job.state == "running"
                or (job.state == "cancel_requested" and job.started_at is not None)
            )
        )
        next_state = (
            "lease_expired"
            if publish_unknown
            else "cancel_requested"
            if job.state == "cancel_requested"
            else "lease_expired"
        )
        conn.execute(
            """
            update jobs
            set state = ?,
                lease_owner = null,
                lease_acquired_at = null,
                lease_expires_at = null,
                last_failure_code = case when ? then 'publish_verification_unknown' else last_failure_code end,
                last_failure_class = case when ? then 'publish' else last_failure_class end,
                last_failure_at = case when ? then ? else last_failure_at end,
                updated_at = ?,
                row_version = row_version + 1
            where job_uid = ?
            """,
            (
                next_state,
                1 if publish_unknown else 0,
                1 if publish_unknown else 0,
                1 if publish_unknown else 0,
                now,
                now,
                job_uid,
            ),
        )
        conn.execute(
            """
            update attempts
            set state = 'lease_expired',
                lease_owner = null,
                failure_code = ?,
                failure_class = ?,
                finished_at = null,
                updated_at = ?
            where job_uid = ? and attempt_number = ?
            """,
            (
                "publish_verification_unknown" if publish_unknown else "stale_lease",
                "publish" if publish_unknown else None,
                now,
                job_uid,
                job.attempt_count,
            ),
        )

    def _release_attempt_metadata(
        self,
        conn: sqlite3.Connection,
        job: JobRecord,
        now: str,
    ) -> None:
        if job.lease_owner is None or job.attempt_count <= 0:
            return
        conn.execute(
            """
            update attempts
            set state = 'released',
                lease_owner = null,
                updated_at = ?
            where job_uid = ? and attempt_number = ?
            """,
            (now, job.job_uid, job.attempt_count),
        )

    def _now(self) -> str:
        if self._now_fn is None:
            value = datetime.now(timezone.utc)
        else:
            value = self._now_fn()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _format_now_plus(self, seconds: int) -> str:
        if self._now_fn is None:
            value = datetime.now(timezone.utc)
        else:
            value = self._now_fn()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        value = value + timedelta(seconds=seconds)
        return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def unavailable_stats(reason_code: str) -> dict[str, Any]:
    reason = _safe_reason(reason_code)
    return {
        "queue_enabled": False,
        "lease_enabled": False,
        "resume_persistence_enabled": False,
        "queue_depth": 0,
        "active_jobs": 0,
        "queue_status": "unavailable",
        "queue_degraded_reason": reason,
        "max_jobs": 0,
    }


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_uid=row["job_uid"],
        run_uid=row["run_uid"],
        job_spec_hash_sha256=row["job_spec_hash_sha256"],
        canonical_job_spec_json=row["canonical_job_spec_json"],
        capability_hash_sha256=row["capability_hash_sha256"],
        job_type=row["job_type"],
        state=row["state"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        accepted_at=row["accepted_at"],
        queued_at=row["queued_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        terminal_at=row["terminal_at"],
        cancel_requested=bool(row["cancel_requested"]),
        cancel_reason_code=row["cancel_reason_code"],
        resume_requested=bool(row["resume_requested"]),
        resume_reason_code=row["resume_reason_code"],
        resume_count=int(row["resume_count"]),
        resume_limit=int(row["resume_limit"]),
        last_checkpoint_ref=row["last_checkpoint_ref"],
        attempt_count=int(row["attempt_count"]),
        last_failure_code=row["last_failure_code"],
        last_failure_class=row["last_failure_class"],
        runner_kind=row["runner_kind"],
        lease_owner=row["lease_owner"],
        lease_acquired_at=row["lease_acquired_at"],
        lease_expires_at=row["lease_expires_at"],
        row_version=int(row["row_version"]),
    )


def _row_to_publish_result(row: sqlite3.Row) -> PublishResultRecord:
    return PublishResultRecord(
        job_uid=row["job_uid"],
        publish_job_uid=row["publish_job_uid"],
        target_ref=row["target_ref"],
        publish_spec_hash_sha256=row["publish_spec_hash_sha256"],
        source_artifact_fingerprint_sha256=row["source_artifact_fingerprint_sha256"],
        provenance_fingerprint_sha256=row["provenance_fingerprint_sha256"],
        ollama_digest=row["ollama_digest"],
        idempotent=bool(row["idempotent"]),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_uid=row["artifact_uid"],
        job_uid=row["job_uid"],
        run_uid=row["run_uid"],
        role=row["role"],
        artifact_ref=row["artifact_ref"],
        sha256=row["sha256"],
        size_bytes=None if row["size_bytes"] is None else int(row["size_bytes"]),
        expected_size_bytes=(
            None
            if row["expected_size_bytes"] is None
            else int(row["expected_size_bytes"])
        ),
        reserved_size_bytes=int(row["reserved_size_bytes"]),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        verified_at=row["verified_at"],
        deleted_at=row["deleted_at"],
        failure_code=row["failure_code"],
        row_version=int(row["row_version"]),
    )


def _require_non_terminal(job: JobRecord) -> None:
    if job.state in TERMINAL_STATES:
        raise InvalidStateTransition()


def _require_transition(current: str, target: str) -> None:
    if current == target:
        return
    if current in TERMINAL_STATES:
        raise InvalidStateTransition()
    if target not in STATE_TRANSITIONS.get(current, set()):
        raise InvalidStateTransition()


def _state_after_clean_release(job: JobRecord) -> str:
    if job.resume_requested:
        return "resume_requested"
    if job.cancel_requested:
        return "cancel_requested"
    return "queued"


def _has_active_lease(job: JobRecord, now: str) -> bool:
    return job.lease_owner is not None and job.lease_expires_at is not None and job.lease_expires_at >= now


def _has_any_lease_metadata(job: JobRecord) -> bool:
    return (
        job.lease_owner is not None
        or job.lease_acquired_at is not None
        or job.lease_expires_at is not None
    )


def _require_owned_active_lease(job: JobRecord, owner: str, now: str) -> None:
    if job.lease_owner != owner or not _has_active_lease(job, now):
        raise LeaseUnavailable()


def _validate_job_uid_or_raise(job_uid: str) -> None:
    if not job_stubs.validate_job_uid(job_uid):
        raise job_stubs.JobValidationError("job_uid is invalid")


def _validate_owner_or_raise(owner: str) -> None:
    if not job_stubs.validate_lease_owner(owner):
        raise job_stubs.JobValidationError("lease_owner is invalid")


def _validate_failure_code_or_raise(value: str, *, label: str = "failure_code") -> None:
    if not job_stubs.validate_reason_code(value):
        raise job_stubs.JobValidationError(f"{label} is invalid")


def _normalize_job_type_filter(
    job_types: set[str] | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    if job_types is None:
        return None
    if not isinstance(job_types, (set, tuple, list)):
        raise job_stubs.JobValidationError("job_types is invalid")
    result = tuple(sorted(set(job_types)))
    if not result or any(job_type not in JOB_TYPES for job_type in result):
        raise job_stubs.JobValidationError("job_types is invalid")
    return result


def _validate_artifact_uid_or_raise(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 41
        or not value.startswith("artifact_")
        or any(char not in "0123456789abcdef" for char in value[9:])
    ):
        raise job_stubs.JobValidationError("artifact_uid is invalid")


def _validate_artifact_role_or_raise(value: str) -> None:
    if value not in ARTIFACT_ROLES:
        raise job_stubs.JobValidationError("artifact role is invalid")


def _is_safe_artifact_ref(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= 192
        and value.startswith("artifact:")
        and "/" not in value
        and "\\" not in value
        and "://" not in value
        and "?" not in value
        and "#" not in value
        and ".." not in value
    )


def _validate_publish_result_fields(
    *,
    publish_job_uid: str,
    target_ref: str,
    publish_spec_hash_sha256: str,
    source_artifact_fingerprint_sha256: str,
    provenance_fingerprint_sha256: str,
    ollama_digest: str | None,
) -> None:
    if not job_stubs.validate_last_checkpoint_ref(publish_job_uid):
        raise job_stubs.JobValidationError("publish_job_uid is invalid")
    if not _is_safe_target_ref(target_ref):
        raise job_stubs.JobValidationError("target_ref is invalid")
    for label, value in (
        ("publish_spec_hash_sha256", publish_spec_hash_sha256),
        ("source_artifact_fingerprint_sha256", source_artifact_fingerprint_sha256),
        ("provenance_fingerprint_sha256", provenance_fingerprint_sha256),
    ):
        if not _is_hash(value):
            raise job_stubs.JobValidationError(f"{label} is invalid")
    if ollama_digest is not None and not _is_safe_digest_ref(ollama_digest):
        raise job_stubs.JobValidationError("ollama_digest is invalid")


def _is_safe_target_ref(value: object) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 192:
        return False
    if value.startswith("/") or any(part in value for part in ("\\", "://", "?", "#", "..")):
        return False
    lowered = value.lower()
    return not any(
        part in lowered
        for part in (
            "authorization",
            "bearer",
            "token",
            "secret",
            "credential",
            "password",
            "passwd",
            "private_key",
            "api_key",
        )
    )


def _is_safe_digest_ref(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(char.isalnum() or char in "_.:-" for char in value)
        and not any(
            part in value.lower()
            for part in (
                "authorization",
                "bearer",
                "token",
                "secret",
                "credential",
                "password",
            )
        )
    )


def _validate_artifact_expected_size_or_raise(
    value: object,
    *,
    label: str = "expected_size_bytes",
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 9_223_372_036_854_775_807
    ):
        raise job_stubs.JobValidationError(f"{label} is invalid")


def _validate_artifact_quotas_or_raise(quotas: ArtifactQuotaSettings) -> None:
    for label, value, minimum in (
        ("max_bytes", quotas.max_bytes, 1),
        ("job_quota_bytes", quotas.job_quota_bytes, 1),
        ("total_quota_bytes", quotas.total_quota_bytes, 1),
        ("min_free_bytes", quotas.min_free_bytes, 0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value > 9_223_372_036_854_775_807
        ):
            raise job_stubs.JobValidationError(f"{label} is invalid")
    if quotas.max_bytes > quotas.job_quota_bytes:
        raise job_stubs.JobValidationError("artifact quota ordering is invalid")
    if quotas.job_quota_bytes > quotas.total_quota_bytes:
        raise job_stubs.JobValidationError("artifact quota ordering is invalid")


def _validate_hash_or_raise(value: object, *, label: str) -> None:
    if not _is_hash(value):
        raise job_stubs.JobValidationError(f"{label} is invalid")


def _new_artifact_uid() -> str:
    return f"artifact_{uuid.uuid4().hex}"


def _artifact_ref(*, artifact_uid: str, run_uid: str, role: str, sha256: str) -> str:
    return f"artifact:{run_uid}:{role}:{sha256[:16]}:{artifact_uid}"


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _sql_statements(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _expected_schema_sql() -> dict[str, str]:
    global _EXPECTED_SCHEMA_SQL
    if _EXPECTED_SCHEMA_SQL is not None:
        return _EXPECTED_SCHEMA_SQL
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        for statement in _sql_statements(MIGRATION_1_SQL):
            conn.execute(statement)
        for statement in _sql_statements(MIGRATION_2_SQL):
            conn.execute(statement)
        for statement in _sql_statements(MIGRATION_3_SQL):
            conn.execute(statement)
        for statement in _sql_statements(MIGRATION_4_SQL):
            conn.execute(statement)
        for statement in _sql_statements(MIGRATION_5_SQL):
            conn.execute(statement)
        for statement in _sql_statements(MIGRATION_6_SQL):
            conn.execute(statement)
        rows = conn.execute(
            """
            select name, sql
            from sqlite_master
            where name in (
                'schema_migrations',
                'jobs',
                'attempts',
                'artifacts',
                'publish_results',
                'idx_jobs_state_created_at',
                'idx_jobs_updated_at',
                'idx_jobs_active_lease_expiry',
                'idx_jobs_cancel_requested',
                'idx_jobs_resume_requested',
                'idx_jobs_job_type_state_created_at',
                'idx_attempts_job_uid_attempt_number',
                'idx_jobs_job_uid_run_uid',
                'idx_artifacts_job_status_created_at',
                'idx_artifacts_run_uid',
                'idx_artifacts_cleanup',
                'idx_artifacts_role_status',
                'idx_artifacts_ref_unique',
                'idx_artifacts_staged_job_role_hash_size',
                'idx_publish_results_target_ref',
                'idx_publish_results_publish_job_uid'
            )
            """
        ).fetchall()
    finally:
        conn.close()
    _EXPECTED_SCHEMA_SQL = {
        row["name"]: _normalize_sql(row["sql"])
        for row in rows
    }
    return _EXPECTED_SCHEMA_SQL


def _normalize_sql(value: str | None) -> str:
    return " ".join((value or "").split())


def _safe_reason(reason_code: str) -> str:
    if reason_code in VISIBLE_QUEUE_ERROR_REASONS:
        return reason_code
    return "sqlite_unavailable"


def _classify_os_error(exc: OSError) -> str:
    if exc.errno == errno.ENOSPC:
        return "disk_full"
    if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
        return "permission_denied"
    return "sqlite_unavailable"


def _classify_sqlite_error(
    exc: sqlite3.DatabaseError,
    *,
    default: str = "sqlite_unavailable",
) -> str:
    name = str(getattr(exc, "sqlite_errorname", "") or "").upper()
    text = str(exc).lower()
    if "SQLITE_FULL" in name or "database or disk is full" in text:
        return "disk_full"
    if "SQLITE_CORRUPT" in name or "SQLITE_NOTADB" in name or "database disk image is malformed" in text or "file is not a database" in text:
        return "db_corrupt"
    if "SQLITE_PERM" in name or "SQLITE_CANTOPEN" in name or "SQLITE_READONLY" in name:
        return "permission_denied"
    if "SQLITE_BUSY" in name or "SQLITE_LOCKED" in name or "database is locked" in text:
        return "sqlite_unavailable"
    return default
