from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
import os
from types import SimpleNamespace

import pytest
from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity

import executor
import gpu_policy
import publish_test_support
import queue_store
import runners


CAPABILITY_HASH = "a" * 64


@pytest.fixture(autouse=True)
def admission_runtime(tmp_path, monkeypatch):
    root = tmp_path / "vram"
    root.mkdir(mode=0o2770)
    root.chmod(0o2770)
    security = RuntimeSecurity(os.geteuid(), os.getegid(), frozenset({os.geteuid()}))
    store = AdmissionStore(root, security=security, boot_id="test-boot")
    monkeypatch.setattr(gpu_policy, "training_admission_store", lambda: store)
    monkeypatch.setattr(gpu_policy, "measure_training_memory",
                        lambda: MemorySnapshot(12 * 1024**3, 32 * 1024**3, time.monotonic()))
    return store
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
KITT_SPEC = VALID_SPEC


def _write_sft_trainer(tmp_path: Path) -> Path:
    return Path(__file__).resolve().with_name("sft_trainer.py")


def _cfg(tmp_path: Path | None = None, **overrides):
    values = {
        "worker_id": "kiron-kitt-worker",
        "dispatch_gate": "closed",
        "executor_runner": "sft_subprocess",
        "executor_enabled": True,
        "sft_command": "",
        "executor_config_error_code": None,
        "artifact_config_error_code": None,
        "executor_poll_interval_seconds": 0.1,
        "executor_renew_interval_seconds": 1,
        "executor_job_timeout_seconds": 5,
        "executor_shutdown_grace_seconds": 1,
    }
    if tmp_path is not None:
        values["dispatch_gate"] = "open"
        values["sft_command"] = f"{sys.executable} {_write_sft_trainer(tmp_path)}"
    values.update(overrides)
    return SimpleNamespace(**values)


def _store(tmp_path):
    return queue_store.QueueStore.open(
        tmp_path,
        queue_store.QueueSettings(
            max_jobs=10,
            lease_ttl_seconds=30,
            resume_limit=2,
            sqlite_busy_timeout_ms=1000,
        ),
    )


def _allow():
    return gpu_policy.ExecutionStartDecision(
        True,
        "allowed",
        "none",
        "complete",
        "inactive",
    )


def _deny():
    return gpu_policy.ExecutionStartDecision(
        False,
        "ollama_active",
        "ollama_active",
        "complete",
        "inactive",
    )


def _ready_snapshot():
    return {"valid_for_scheduling": True, "execution_enabled": True}


def _blocked_snapshot():
    return {"valid_for_scheduling": False, "execution_enabled": False}


def _publish_ready_snapshot():
    return {
        "valid_for_scheduling": True,
        "execution_enabled": True,
        "operations": {
            "publish": {
                "supported": True,
                "ready": True,
                "execution_enabled": True,
            }
        },
    }


def _publish_only_ready_snapshot():
    return {
        "valid_for_scheduling": False,
        "execution_enabled": False,
        "operations": {
            "publish": {
                "supported": True,
                "ready": True,
                "execution_enabled": True,
            }
        },
    }


def _publish_self_active_snapshot():
    return {
        "valid_for_scheduling": True,
        "execution_enabled": True,
        "operations": {
            "publish": {
                "supported": True,
                "ready": False,
                "execution_enabled": False,
                "blocked_reason": "worker_active_job",
            }
        },
        "operational_status": {
            "active_jobs": 1,
            "queue_status": "ready",
            "maintenance_mode": False,
            "inference_conflict": {
                "status": "none",
                "source_status": "complete",
            },
        },
    }


class _PublishSuccessRunner:
    def __init__(self, publish_result):
        self.publish_result = publish_result

    async def run(self, _job, _cancel_event):
        return runners.RunnerOutcome(
            "succeeded",
            publish_result=self.publish_result,
        )


class _PublishUnknownAfterCancelRunner:
    def __init__(self, store, job_uid: str):
        self.store = store
        self.job_uid = job_uid

    async def run(self, _job, _cancel_event):
        self.store.request_cancel(
            job_uid=self.job_uid,
            reason_code="operator_requested",
        )
        return runners.RunnerOutcome(
            "failed",
            failure_code="publish_verification_unknown",
            failure_class="publish",
        )


def test_open_dispatch_gate_uses_gpu_policy_instead_of_fake_bypass(tmp_path):
    decision_fn = executor._policy_decider_for_cfg(_cfg(tmp_path))

    assert decision_fn is gpu_policy.decide_execution_start


def test_start_executor_task_requires_open_dispatch_gate(tmp_path):
    app_state = SimpleNamespace(
        worker_config=_cfg(tmp_path, dispatch_gate="closed"),
        queue_store=_store(tmp_path),
    )

    assert executor.start_executor_task(app_state) is None
    assert app_state.executor_state.failed is False
    assert app_state.executor_state.last_error_code == "dispatch_gate_closed"


def test_executor_closed_dispatch_gate_does_not_lease_runnable_job(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-gate-closed",
            job_spec=KITT_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        runner_calls = 0

        def runner_factory():
            nonlocal runner_calls
            runner_calls += 1
            return runners.FakeRunner()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path, dispatch_gate="closed"),
            runner_factory=runner_factory,
            policy_decider=_allow,
            capability_snapshot_fn=_blocked_snapshot,
        )
        assert await worker.run_once() is False
        return store.get_job("job-gate-closed"), runner_calls

    job, runner_calls = asyncio.run(run())

    assert runner_calls == 0
    assert job.state == "queued"
    assert job.lease_owner is None


def test_executor_dispatches_publish_job_to_publish_runner_kind(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-publish-executor",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        policy_calls = 0

        def policy_decider():
            nonlocal policy_calls
            policy_calls += 1
            return _deny()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: _PublishSuccessRunner(
                {
                    "publish_job_uid": spec["publish_job_uid"],
                    "target_ref": spec["target"]["target_ref"],
                    "publish_spec_hash_sha256": spec["publish_spec_hash_sha256"],
                    "source_artifact_fingerprint_sha256": "d" * 64,
                    "provenance_fingerprint_sha256": "e" * 64,
                    "ollama_digest": "sha256:publishabc",
                    "idempotent": False,
                }
            ),
            policy_decider=policy_decider,
            capability_snapshot_fn=_publish_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-publish-executor"), store.get_publish_result(
            "job-publish-executor"
        ), policy_calls

    job, publish_result, policy_calls = asyncio.run(run())

    assert job.state == "succeeded"
    assert job.job_type == "ollama_publish"
    assert job.runner_kind == "ollama_publish"
    assert publish_result is not None
    assert publish_result.target_ref == publish_test_support.PUBLISH_TARGET_REF
    assert policy_calls == 0


def test_executor_dispatches_publish_when_only_self_active_after_lease(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-publish-self-active",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        snapshots = [_publish_ready_snapshot(), _publish_self_active_snapshot()]

        def snapshot_fn():
            if snapshots:
                return snapshots.pop(0)
            return _publish_self_active_snapshot()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: _PublishSuccessRunner(
                {
                    "publish_job_uid": spec["publish_job_uid"],
                    "target_ref": spec["target"]["target_ref"],
                    "publish_spec_hash_sha256": spec["publish_spec_hash_sha256"],
                    "source_artifact_fingerprint_sha256": "d" * 64,
                    "provenance_fingerprint_sha256": "e" * 64,
                    "ollama_digest": "sha256:publishabc",
                    "idempotent": False,
                }
            ),
            policy_decider=_deny,
            capability_snapshot_fn=snapshot_fn,
        )
        assert await worker.run_once() is True
        return store.get_job("job-publish-self-active")

    job = asyncio.run(run())

    assert job.state == "succeeded"
    assert job.last_failure_code is None
    assert job.runner_kind == "ollama_publish"


def test_executor_dispatches_publish_when_sft_gate_is_blocked(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-publish-only-ready",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: _PublishSuccessRunner(
                {
                    "publish_job_uid": spec["publish_job_uid"],
                    "target_ref": spec["target"]["target_ref"],
                    "publish_spec_hash_sha256": spec["publish_spec_hash_sha256"],
                    "source_artifact_fingerprint_sha256": "d" * 64,
                    "provenance_fingerprint_sha256": "e" * 64,
                    "ollama_digest": "sha256:publishabc",
                    "idempotent": False,
                }
            ),
            policy_decider=_deny,
            capability_snapshot_fn=_publish_only_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-publish-only-ready")

    job = asyncio.run(run())

    assert job.state == "succeeded"
    assert job.job_type == "ollama_publish"
    assert job.runner_kind == "ollama_publish"


def test_executor_publish_only_readiness_skips_queued_sft_job(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-sft-first",
            job_spec=KITT_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        store.put_job(
            job_uid="job-publish-second",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: _PublishSuccessRunner(
                {
                    "publish_job_uid": spec["publish_job_uid"],
                    "target_ref": spec["target"]["target_ref"],
                    "publish_spec_hash_sha256": spec["publish_spec_hash_sha256"],
                    "source_artifact_fingerprint_sha256": "d" * 64,
                    "provenance_fingerprint_sha256": "e" * 64,
                    "ollama_digest": "sha256:publishabc",
                    "idempotent": False,
                }
            ),
            policy_decider=_deny,
            capability_snapshot_fn=_publish_only_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-sft-first"), store.get_job("job-publish-second")

    sft, publish = asyncio.run(run())

    assert sft.state == "queued"
    assert sft.lease_owner is None
    assert sft.last_failure_code is None
    assert publish.state == "succeeded"
    assert publish.runner_kind == "ollama_publish"


def test_executor_publish_timeout_marks_verification_unknown(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-publish-timeout",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(
                tmp_path,
                executor_job_timeout_seconds=1,
                executor_shutdown_grace_seconds=1,
            ),
            runner_factory=lambda: runners.FakeRunner(delay_seconds=10),
            policy_decider=_allow,
            capability_snapshot_fn=_publish_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-publish-timeout")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.job_type == "ollama_publish"
    assert job.last_failure_code == "publish_verification_unknown"
    assert job.last_failure_class == "publish"


def test_executor_preserves_publish_unknown_after_cancel_request(tmp_path):
    async def run():
        store = _store(tmp_path)
        spec, _keyring = publish_test_support.signed_publish_spec()
        store.put_job(
            job_uid="job-publish-cancel-unknown",
            job_spec=spec,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: _PublishUnknownAfterCancelRunner(
                store,
                "job-publish-cancel-unknown",
            ),
            policy_decider=_allow,
            capability_snapshot_fn=_publish_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-publish-cancel-unknown")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.cancel_requested is True
    assert job.last_failure_code == "publish_verification_unknown"
    assert job.last_failure_class == "publish"


def test_executor_open_config_gate_requires_capability_readiness(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-capability-blocked",
            job_spec=KITT_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        runner_calls = 0

        def runner_factory():
            nonlocal runner_calls
            runner_calls += 1
            return runners.FakeRunner()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=runner_factory,
            policy_decider=_allow,
            capability_snapshot_fn=_blocked_snapshot,
        )
        assert await worker.run_once() is False
        return store.get_job("job-capability-blocked"), runner_calls

    job, runner_calls = asyncio.run(run())

    assert runner_calls == 0
    assert job.state == "queued"
    assert job.lease_owner is None


def test_executor_maintenance_processes_queued_cancel_before_runnable(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-cancel",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        store.request_cancel(job_uid="job-cancel", reason_code="operator_requested")
        store.put_job(
            job_uid="job-runnable",
            job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.runnable"}},
            capability_hash_sha256=CAPABILITY_HASH,
        )
        runner_calls = 0
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(),
            runner_factory=lambda: _counted_runner(),
            policy_decider=_allow,
        )

        def _counted_runner():
            nonlocal runner_calls
            runner_calls += 1
            return runners.FakeRunner()

        assert await worker.run_once() is True
        return store, runner_calls

    store, runner_calls = asyncio.run(run())

    assert store.get_job("job-cancel").state == "canceled"
    assert store.get_job("job-runnable").state == "queued"
    assert runner_calls == 0


def test_executor_maintenance_processes_gate_closed_resume(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-resume",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        store.request_resume(job_uid="job-resume", reason_code="retry")
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(),
            policy_decider=_allow,
        )
        assert await worker.run_once() is True
        return store.get_job("job-resume")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.last_failure_code == "resume_not_supported"
    assert job.last_failure_class == "resume"
    assert job.started_at is None


def test_executor_policy_deny_terminalizes_without_runner_start(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-policy",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        runner_calls = 0

        def runner_factory():
            nonlocal runner_calls
            runner_calls += 1
            return runners.FakeRunner()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=runner_factory,
            policy_decider=_deny,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-policy"), runner_calls

    job, runner_calls = asyncio.run(run())

    assert runner_calls == 0
    assert job.state == "failed"
    assert job.last_failure_code == "policy_blocked"
    assert job.last_failure_class == "policy"
    assert job.started_at is None


def test_executor_cancel_race_after_lease_acquire_cancels_before_running(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-cancel-race",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        runner_calls = 0

        def runner_factory():
            nonlocal runner_calls
            runner_calls += 1
            return runners.FakeRunner()

        def policy_decider():
            store.request_cancel(
                job_uid="job-cancel-race",
                reason_code="operator_requested",
            )
            return _allow()

        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=runner_factory,
            policy_decider=policy_decider,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-cancel-race"), runner_calls

    job, runner_calls = asyncio.run(run())

    assert runner_calls == 0
    assert job.state == "canceled"
    assert job.cancel_reason_code == "operator_requested"
    assert job.started_at is None
    assert job.finished_at == job.terminal_at
    assert job.lease_owner is None


def test_executor_fake_runner_success_and_timeout(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-success",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        success_worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: runners.FakeRunner(),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await success_worker.run_once() is True

        store.put_job(
            job_uid="job-timeout",
            job_spec={**VALID_SPEC, "labels": {"kitt_job": "job.timeout"}},
            capability_hash_sha256=CAPABILITY_HASH,
        )
        timeout_worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path, executor_job_timeout_seconds=1),
            runner_factory=lambda: runners.FakeRunner(delay_seconds=5),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await timeout_worker.run_once() is True
        return store.get_job("job-success"), store.get_job("job-timeout")

    success, timeout = asyncio.run(run())

    assert success.state == "succeeded"
    assert success.runner_kind == "sft_subprocess"
    assert success.started_at is not None
    assert success.finished_at == success.terminal_at
    assert timeout.state == "failed"
    assert timeout.last_failure_code == "runner_timeout"
    assert timeout.last_failure_class == "runner"


def test_executor_renews_lease_during_blocking_post_runner_work(tmp_path):
    async def run():
        store = _store(tmp_path)
        cfg = _cfg(tmp_path, executor_renew_interval_seconds=0.1)
        store.put_job(
            job_uid="job-renew-blocking-work",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        renewals = {"count": 0}

        class SpyStore:
            def __getattr__(self, name):
                return getattr(store, name)

            def renew_lease(self, **kwargs):
                renewals["count"] += 1
                return store.renew_lease(**kwargs)

        worker = executor.KittWorkerExecutor(
            store=SpyStore(),
            cfg=cfg,
            runner_factory=lambda: runners.FakeRunner(),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        leased = store.next_runnable_job(worker.owner)
        assert leased is not None
        store.mark_running(
            job_uid="job-renew-blocking-work",
            owner=worker.owner,
        )
        result = await worker._run_with_lease_renewal(
            "job-renew-blocking-work",
            lambda: (time.sleep(0.25), "done")[1],
        )
        return result, renewals["count"]

    result, renew_count = asyncio.run(run())

    assert result == "done"
    assert renew_count >= 3


def test_executor_rejects_empty_runner_artifact_before_success(tmp_path):
    class EmptyArtifactRunner:
        async def run(self, _job, _cancel_event):
            artifact_path = tmp_path / "empty-manifest.json"
            artifact_path.write_bytes(b"")
            return runners.RunnerOutcome(
                "succeeded",
                artifacts=(runners.RunnerArtifact("manifest", artifact_path),),
            )

    async def run():
        data_dir = tmp_path / "data"
        staging_dir = data_dir / "staging"
        staging_dir.mkdir(parents=True)
        staging_dir.chmod(0o750)
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-empty-artifact",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(
                tmp_path,
                data_dir=data_dir,
                artifact_staging_dir=staging_dir,
                artifact_max_bytes=1024,
                artifact_job_quota_bytes=4096,
                artifact_total_quota_bytes=8192,
                artifact_min_free_bytes=0,
                artifact_cleanup_after_seconds=60,
                artifact_config_error_code=None,
            ),
            runner_factory=lambda: EmptyArtifactRunner(),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-empty-artifact")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.last_failure_code == "artifact_empty"
    assert job.last_failure_class == "artifact"


def test_executor_rejects_nonempty_dummy_artifacts_before_success(tmp_path):
    class DummyArtifactRunner:
        async def run(self, _job, _cancel_event):
            manifest_path = tmp_path / "manifest.json"
            model_path = tmp_path / "adapter.zip"
            manifest_path.write_text("{}", encoding="utf-8")
            model_path.write_bytes(b"x")
            return runners.RunnerOutcome(
                "succeeded",
                artifacts=(
                    runners.RunnerArtifact("manifest", manifest_path),
                    runners.RunnerArtifact("adapter", model_path),
                ),
            )

    async def run():
        data_dir = tmp_path / "data"
        staging_dir = data_dir / "staging"
        staging_dir.mkdir(parents=True)
        staging_dir.chmod(0o750)
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-dummy-artifact",
            job_spec=KITT_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(
                tmp_path,
                data_dir=data_dir,
                artifact_staging_dir=staging_dir,
                artifact_max_bytes=1024,
                artifact_job_quota_bytes=4096,
                artifact_total_quota_bytes=8192,
                artifact_min_free_bytes=0,
                artifact_cleanup_after_seconds=60,
                artifact_config_error_code=None,
            ),
            runner_factory=lambda: DummyArtifactRunner(),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-dummy-artifact")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.last_failure_code == "runner_artifact_invalid"
    assert job.last_failure_class == "artifact"


def test_executor_runner_exception_becomes_redacted_runner_failure(tmp_path):
    class ExplodingRunner:
        async def run(self, _job, _cancel_event):
            raise RuntimeError("raw runner exception with /usr/lib/kiron/path")

    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-runner-exception",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: ExplodingRunner(),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        assert await worker.run_once() is True
        return store.get_job("job-runner-exception")

    job = asyncio.run(run())

    assert job.state == "failed"
    assert job.last_failure_code == "runner_failed"
    assert job.last_failure_class == "runner"
    assert job.finished_at == job.terminal_at
    assert job.lease_owner is None


def test_executor_running_cancel_request_becomes_canceled(tmp_path):
    async def run():
        store = _store(tmp_path)
        store.put_job(
            job_uid="job-cancel-running",
            job_spec=VALID_SPEC,
            capability_hash_sha256=CAPABILITY_HASH,
        )
        worker = executor.KittWorkerExecutor(
            store=store,
            cfg=_cfg(tmp_path),
            runner_factory=lambda: runners.FakeRunner(delay_seconds=5),
            policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot,
        )
        task = asyncio.create_task(worker.run_once())
        while store.get_job("job-cancel-running").state != "running":
            await asyncio.sleep(0.01)
        store.request_cancel(
            job_uid="job-cancel-running",
            reason_code="operator_requested",
        )
        assert await task is True
        return store.get_job("job-cancel-running")

    job = asyncio.run(run())

    assert job.state == "canceled"
    assert job.cancel_reason_code == "operator_requested"
    assert job.finished_at == job.terminal_at


def test_atomic_training_admission_blocks_resident_prism_before_runner(tmp_path, admission_runtime):
    async def run():
        store = _store(tmp_path)
        store.put_job(job_uid="job-admission", job_spec=VALID_SPEC, capability_hash_sha256=CAPABILITY_HASH)
        started = []

        def decision():
            ticket = admission_runtime.reserve(operation_id="prism-load", owner="prism", generation="child",
                deployment_id="bonsai", kind="load", gpu_bytes=0, host_bytes=0,
                measure=gpu_policy.measure_training_memory)
            admission_runtime.transition(ticket.operation_id, owner="prism", expected_generation="child", phase="resident")
            return _allow()

        worker = executor.KittWorkerExecutor(store=store, cfg=_cfg(tmp_path), policy_decider=decision,
            capability_snapshot_fn=_ready_snapshot, runner_factory=lambda: started.append(True))
        assert await worker.run_once()
        assert not started
        assert store.get_job("job-admission").state == "failed"
        assert store.get_job("job-admission").last_failure_code == "policy_blocked"
        assert [ticket.owner for ticket in admission_runtime.snapshot()] == ["prism"]
    asyncio.run(run())


def test_training_admission_heartbeats_and_releases_only_with_backend_proof(tmp_path, admission_runtime, monkeypatch):
    async def run():
        store = _store(tmp_path)
        store.put_job(job_uid="job-heartbeat", job_spec=VALID_SPEC, capability_hash_sha256=CAPABILITY_HASH)
        beats = []
        original = admission_runtime.heartbeat
        monkeypatch.setattr(admission_runtime, "heartbeat", lambda *args, **kwargs: (beats.append(True), original(*args, **kwargs))[1])
        worker = executor.KittWorkerExecutor(store=store, cfg=_cfg(tmp_path, executor_renew_interval_seconds=0),
            policy_decider=_allow, capability_snapshot_fn=_ready_snapshot,
            runner_factory=lambda: runners.FakeRunner(delay_seconds=0.12))
        assert await worker.run_once()
        assert beats
        assert admission_runtime.snapshot() == ()

        class UnknownRunner:
            async def run(self, _job, _cancel):
                assert admission_runtime.snapshot()[0].kind == "training"
                return runners.RunnerOutcome("succeeded")

        store.put_job(job_uid="job-unknown", job_spec=VALID_SPEC, capability_hash_sha256=CAPABILITY_HASH)
        worker.runner_factory = UnknownRunner
        assert await worker.run_once()
        assert admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())


def test_executor_cancellation_preserves_unknown_backend_ticket(tmp_path, admission_runtime):
    async def run():
        store = _store(tmp_path)
        store.put_job(job_uid="job-cancel-admission", job_spec=VALID_SPEC, capability_hash_sha256=CAPABILITY_HASH)
        started = asyncio.Event()

        class UnknownRunner:
            backend_terminated = False
            async def run(self, _job, cancel):
                started.set()
                await cancel.wait()
                return runners.RunnerOutcome("canceled")

        worker = executor.KittWorkerExecutor(store=store, cfg=_cfg(tmp_path), policy_decider=_allow,
            capability_snapshot_fn=_ready_snapshot, runner_factory=UnknownRunner)
        task = asyncio.create_task(worker.run_once())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission_runtime.snapshot()[0].phase == "unknown"
        assert worker.state.active_job_uid is None
    asyncio.run(run())
