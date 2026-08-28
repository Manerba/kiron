from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import artifact_staging
import capabilities
import job_stubs
import monitoring
import queue_store
import v1_test_support as support


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
READY_QUEUE = {
    "queue_enabled": True,
    "lease_enabled": True,
    "resume_persistence_enabled": True,
    "queue_depth": 0,
    "queue_status": "ready",
    "queue_degraded_reason": None,
    "max_jobs": 10,
}


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value = self.value + timedelta(seconds=seconds)


class CompleteProbes:
    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "cuda",
                    "count": 1,
                    "vram_total_bytes": 12,
                    "vram_available_bytes": 10,
                    "vram_budget_bytes": 0,
                    "capability_status": "unavailable",
                },
            ),
            host_ram_total_bytes=64,
            host_ram_available_bytes=48,
            staging_free_bytes=1024,
            measurement_status="complete",
        )

    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(),
            measurement_status="complete",
        )

    def probe_inference_conflict(self) -> capabilities.ConflictProbeResult:
        return capabilities.ConflictProbeResult(
            status="none",
            source_status="complete",
            maintenance="inactive",
        )


def _cfg(**overrides):
    values = {
        "monitoring_enabled": True,
        "alerts_enabled": True,
        "monitoring_interval_seconds": 30.0,
        "heartbeat_stale_seconds": 300,
        "alert_repeat_seconds": 3600,
        "monitoring_config_error_code": None,
        "artifact_config_error_code": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manager(clock: Clock | None = None, **cfg_overrides):
    events: list[dict] = []
    clock = Clock() if clock is None else clock
    app_state = SimpleNamespace(
        worker_config=_cfg(**cfg_overrides),
        queue_store=None,
        queue_unavailable=None,
        artifact_staging_unavailable=None,
        capability_probes=CompleteProbes(),
        capability_clock=clock,
    )
    manager = monitoring.MonitoringManager(
        cfg=app_state.worker_config,
        app_state=app_state,
        event_sink=events.append,
        now_fn=clock,
    )
    app_state.monitoring_manager = manager
    return manager, app_state, events, clock


def _queue_settings() -> queue_store.QueueSettings:
    return queue_store.QueueSettings(
        max_jobs=10,
        lease_ttl_seconds=30,
        resume_limit=2,
        sqlite_busy_timeout_ms=100,
    )


def test_alert_code_allowlist_covers_sprint_9_scope():
    assert {
        "stale_heartbeat",
        "queue_full",
        "stale_lease",
        "capability_expired",
        "gpu_policy_blocked",
        "disk_full",
        "redaction_failure",
        "artifact_staging_failed",
        "artifact_cleanup_failed",
    }.issubset(monitoring.ALERT_CODES)


def test_signal_to_alert_code_mapping():
    assert monitoring.alert_codes_from_queue_stats(
        {**READY_QUEUE, "queue_status": "full", "queue_degraded_reason": "queue_full"}
    ) == ("queue_full",)
    assert monitoring.alert_codes_from_queue_stats(
        queue_store.unavailable_stats("disk_full")
    ) == ("disk_full",)
    assert monitoring.alert_codes_from_capability_snapshot(
        {"operational_status": {"blocking_reasons": ["ollama_active"]}}
    ) == ("gpu_policy_blocked",)
    assert (
        monitoring.alert_code_from_cleanup_result(
            artifact_staging.CleanupResult(failed_deletes=1)
        )
        == "artifact_cleanup_failed"
    )
    assert monitoring.alert_codes_from_artifact_state(
        unavailable_reason=None,
        config_error_code=None,
        summary={
            "status_counts": {"quota_blocked": 1},
            "failure_counts": {"artifact_job_quota_exceeded": 1},
        },
    ) == ("artifact_staging_failed",)
    assert monitoring.alert_codes_from_artifact_state(
        unavailable_reason=None,
        config_error_code=None,
        summary={
            "status_counts": {"quota_blocked": 1},
            "failure_counts": {"artifact_min_free_blocked": 1},
        },
    ) == ("artifact_staging_failed", "disk_full")
    assert monitoring.alert_codes_from_stage_result(
        artifact_staging.StageResult(
            status="quota_blocked",
            artifact=None,
            failure_code="artifact_too_large",
        )
    ) == ("artifact_staging_failed",)
    assert monitoring.alert_codes_from_stage_result(
        artifact_staging.StageResult(
            status="quota_blocked",
            artifact=None,
            failure_code="artifact_job_quota_exceeded",
        )
    ) == ("artifact_staging_failed",)
    assert monitoring.alert_codes_from_stage_result(
        artifact_staging.StageResult(
            status="quota_blocked",
            artifact=None,
            failure_code="artifact_total_quota_exceeded",
        )
    ) == ("artifact_staging_failed",)
    assert monitoring.alert_codes_from_stage_result(
        artifact_staging.StageResult(
            status="quota_blocked",
            artifact=None,
            failure_code="artifact_min_free_blocked",
        )
    ) == ("artifact_staging_failed", "disk_full")


def test_alert_state_transitions_and_no_storm_behavior():
    manager, _state, events, _clock = _manager()

    manager.set_alert("queue_full", True, reason_code="queue_full")
    manager.set_alert("queue_full", True, reason_code="queue_full")
    manager.set_alert("queue_full", False, reason_code="queue_ready")

    assert [
        (event["alert_code"], event["state"], event["reason_code"])
        for event in events
    ] == [
        ("queue_full", "active", "queue_full"),
        ("queue_full", "inactive", "queue_ready"),
    ]


def test_disk_full_alert_aggregates_sources_without_last_writer_wins():
    manager, _state, events, _clock = _manager()

    manager.set_disk_full_source("queue", True, reason_code="disk_full")
    manager.set_disk_full_source(
        "artifact_staging",
        False,
        reason_code="artifact_disk_ok",
    )

    assert manager.alert_states["disk_full"].state == "active"
    assert [
        (event["alert_code"], event["state"], event["reason_code"], event["source"])
        for event in events
    ] == [("disk_full", "active", "disk_full", "queue")]

    manager.set_disk_full_source("queue", False, reason_code="disk_ok")

    assert manager.alert_states["disk_full"].state == "inactive"
    assert [
        (event["alert_code"], event["state"], event["reason_code"], event["source"])
        for event in events
    ] == [
        ("disk_full", "active", "disk_full", "queue"),
        ("disk_full", "inactive", "disk_ok", "queue"),
    ]


def test_executor_failure_monitoring_uses_alert_state_without_poll_storm():
    manager, app_state, events, _clock = _manager()

    class Store:
        def queue_stats(self):
            return READY_QUEUE

        def artifact_status_summary(self):
            return {"status_counts": {}, "failure_counts": {}}

    app_state.queue_store = Store()
    app_state.executor_state = SimpleNamespace(
        failed=True,
        last_error_code="executor_failed",
    )

    manager.poll_once()
    manager.poll_once()

    assert [
        (event["alert_code"], event["state"], event["reason_code"], event["source"])
        for event in events
    ] == [("capability_probe_failed", "active", "executor_failed", "executor")]

    app_state.executor_state = SimpleNamespace(failed=False, last_error_code=None)
    manager.poll_once()

    assert [
        (event["alert_code"], event["state"], event["reason_code"], event["source"])
        for event in events
    ] == [
        ("capability_probe_failed", "active", "executor_failed", "executor"),
        ("capability_probe_failed", "inactive", "complete", "capabilities"),
    ]


def test_monitoring_capability_snapshot_uses_active_jobs_from_queue_stats():
    manager, app_state, _events, _clock = _manager()

    class Store:
        def queue_stats(self):
            return {**READY_QUEUE, "active_jobs": 1}

        def artifact_status_summary(self):
            return {"status_counts": {}, "failure_counts": {}}

    app_state.queue_store = Store()

    manager.poll_once()

    assert "worker_active_job" in manager.capability_snapshot.blocking_reasons


def test_invalid_monitoring_config_logs_once_and_disables_monitoring():
    events: list[dict] = []
    cfg = _cfg(
        monitoring_enabled=True,
        alerts_enabled=True,
        monitoring_config_error_code="monitoring_config_invalid",
    )
    manager = monitoring.MonitoringManager(
        cfg=cfg,
        app_state=SimpleNamespace(worker_config=cfg),
        event_sink=events.append,
        now_fn=Clock(),
    )

    manager.emit_config_invalid_once()
    manager.poll_once()

    assert manager.enabled is False
    assert manager.alerts_enabled is False
    assert events == [
        {
            "event": "kitt_worker_monitoring_startup",
            "alert_code": "monitoring_config_invalid",
            "state": "active",
            "severity": "critical",
            "reason_code": "monitoring_config_invalid",
            "enabled": False,
        }
    ]


def test_queue_stats_preserves_stale_lease_count_for_monitoring(tmp_path):
    manager, _state, _events, clock = _manager()
    store = queue_store.QueueStore.open(
        tmp_path,
        _queue_settings(),
        now_fn=clock,
        lease_expiry_sink=manager.record_stale_leases,
    )
    store.put_job(
        job_uid="job-stale-stats",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(job_uid="job-stale-stats", owner="worker.test", ttl_seconds=30)
    clock.advance(31)

    stats = store.queue_stats()

    assert stats["queue_status"] == "ready"
    assert manager.lease_expiry_sink.drain() == 1


def test_stale_lease_sink_reports_only_after_committed_transaction(tmp_path):
    manager, _state, _events, clock = _manager()
    store = queue_store.QueueStore.open(
        tmp_path,
        _queue_settings(),
        now_fn=clock,
        lease_expiry_sink=manager.record_stale_leases,
    )
    store.put_job(
        job_uid="job-stale-rollback",
        job_spec=VALID_SPEC,
        capability_hash_sha256=CAPABILITY_HASH,
    )
    store.acquire_lease(
        job_uid="job-stale-rollback",
        owner="worker.test",
        ttl_seconds=30,
    )
    clock.advance(31)

    with pytest.raises(job_stubs.JobNotFoundError):
        store.get_job("job-missing-rollback")

    assert manager.lease_expiry_sink.drain() == 0

    store.get_job("job-stale-rollback")

    assert manager.lease_expiry_sink.drain() == 1


def test_heartbeat_and_capabilities_paths_preserve_stale_lease_counts(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (
        client,
        app,
        headers,
        _data,
        _run,
    ):
        clock = Clock()
        app.state.queue_store._now_fn = clock
        app.state.capability_clock = clock

        app.state.queue_store.put_job(
            job_uid="job-stale-heartbeat",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        app.state.queue_store.acquire_lease(
            job_uid="job-stale-heartbeat",
            owner="worker.test",
            ttl_seconds=30,
        )
        clock.advance(31)
        heartbeat = client.get("/v1/heartbeat", headers=headers)
        assert heartbeat.status_code == 200
        assert app.state.monitoring_manager.lease_expiry_sink.drain() == 1

        app.state.queue_store.put_job(
            job_uid="job-stale-capabilities",
            job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.stale.capabilities"}},
            capability_hash_sha256=CAPABILITY_HASH,
        )
        app.state.queue_store.acquire_lease(
            job_uid="job-stale-capabilities",
            owner="worker.test",
            ttl_seconds=30,
        )
        clock.advance(31)
        capability_response = client.get("/v1/capabilities", headers=headers)
        assert capability_response.status_code == 200
        assert app.state.monitoring_manager.lease_expiry_sink.drain() == 1


def test_capability_expired_uses_internal_snapshot_only():
    now = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    expired = monitoring.CapabilityMonitorSnapshot(
        last_success_at=now - timedelta(seconds=120),
        valid_until=now - timedelta(seconds=1),
        probe_status="complete",
    )
    valid = monitoring.CapabilityMonitorSnapshot(
        last_success_at=now,
        valid_until=now + timedelta(seconds=60),
        probe_status="complete",
    )
    failed_without_success = monitoring.CapabilityMonitorSnapshot(
        last_attempt_at=now,
        last_failure_code="capability_probe_failed",
    )

    assert monitoring.capability_snapshot_expired(expired, now) is True
    assert monitoring.capability_snapshot_expired(valid, now) is False
    assert monitoring.capability_snapshot_expired(failed_without_success, now) is True


def test_artifact_monitoring_does_not_run_cleanup_and_explicit_result_alerts():
    manager, app_state, events, _clock = _manager()
    cleanup_calls = {"count": 0}

    class Store:
        def queue_stats(self):
            return READY_QUEUE

        def artifact_status_summary(self):
            return {"status_counts": {}, "failure_counts": {}}

        def cleanup(self):
            cleanup_calls["count"] += 1
            raise AssertionError("monitoring must not cleanup")

    app_state.queue_store = Store()

    manager.poll_once()
    manager.record_artifact_cleanup_result(
        artifact_staging.CleanupResult(failed_deletes=1)
    )

    assert cleanup_calls["count"] == 0
    assert any(event["alert_code"] == "artifact_cleanup_failed" for event in events)


def test_stage_result_quota_alert_reason_is_failure_code():
    manager, _state, events, _clock = _manager()

    manager.record_artifact_stage_result(
        artifact_staging.StageResult(
            status="quota_blocked",
            artifact=None,
            failure_code="artifact_job_quota_exceeded",
        )
    )

    assert [
        (event["alert_code"], event["state"], event["reason_code"])
        for event in events
    ] == [
        (
            "artifact_staging_failed",
            "active",
            "artifact_job_quota_exceeded",
        )
    ]


def test_v1_contract_shapes_are_not_extended_by_monitoring(tmp_path):
    with support.client_context(tmp_path, scopes=["read"]) as (
        client,
        _app,
        headers,
        _data,
        _run,
    ):
        heartbeat = client.get("/v1/heartbeat", headers=headers).json()["data"]
        capabilities_payload = client.get("/v1/capabilities", headers=headers).json()[
            "data"
        ]

    assert set(heartbeat) == {
        "status",
        "heartbeat_time",
        "dispatch_enabled",
        "execution_enabled",
        "active_jobs",
        "queue_depth",
        "queue_status",
    }
    assert "alerts" not in capabilities_payload
    assert "monitoring" not in capabilities_payload
    assert capabilities_payload["parallelism"]["max_concurrent_jobs"] == 0
    assert capabilities_payload["parallelism"]["gpu_slots"] == 0
