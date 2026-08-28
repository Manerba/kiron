"""Async executor loop for kitt-worker job execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import logging
from pathlib import Path
import uuid
from typing import Any, Callable, Mapping

import artifact_staging
import capabilities
import gpu_policy
import job_stubs
import queue_store
import runners


logger = logging.getLogger(__name__)

RunnerFactory = Callable[[], object]
PolicyDecider = Callable[[], gpu_policy.ExecutionStartDecision]
CapabilitySnapshotFn = Callable[[], Mapping[str, Any]]


@dataclass(slots=True)
class ExecutorState:
    running: bool = False
    failed: bool = False
    last_error_code: str | None = None
    active_job_uid: str | None = None


class KittWorkerExecutor:
    def __init__(
        self,
        *,
        store: queue_store.QueueStore,
        cfg: object,
        runner_factory: RunnerFactory | None = None,
        policy_decider: PolicyDecider | None = None,
        capability_snapshot_fn: CapabilitySnapshotFn | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.owner = _owner_from_cfg(cfg)
        self.runner_factory = runner_factory
        self.policy_decider = (
            _policy_decider_for_cfg(cfg) if policy_decider is None else policy_decider
        )
        self.capability_snapshot_fn = (
            (lambda: _capability_snapshot_for_executor(cfg, store))
            if capability_snapshot_fn is None
            else capability_snapshot_fn
        )
        self.state = ExecutorState()
        self._stop_event = asyncio.Event()
        self._active_cancel_event: asyncio.Event | None = None

    async def run(self) -> None:
        self.state.running = True
        try:
            while not self._stop_event.is_set() and not self.state.failed:
                worked = await self.run_once()
                if not worked:
                    await self._sleep_poll_interval()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.state.failed = True
            self.state.last_error_code = "executor_failed"
            logger.error(
                "kitt-worker executor stopped error_code=executor_failed",
                exc_info=False,
            )
        finally:
            self.state.running = False

    async def stop(self) -> None:
        self._stop_event.set()
        if self._active_cancel_event is not None:
            self._active_cancel_event.set()

    async def run_once(self) -> bool:
        cancel_job = self.store.next_queued_cancel_job_without_lease()
        if cancel_job is not None:
            self.store.mark_queued_canceled(job_uid=cancel_job.job_uid)
            return True

        resume_job = self.store.next_gate_closed_resume_job_without_lease()
        if resume_job is not None:
            self.store.mark_resume_gate_closed(job_uid=resume_job.job_uid)
            return True

        dispatchable_job_types = _dispatchable_job_types(self.capability_snapshot_fn)
        if not dispatchable_job_types:
            return False

        acquired = self.store.next_runnable_job(
            self.owner,
            job_types=dispatchable_job_types,
        )
        if acquired is None:
            return False
        await self._execute(acquired)
        return True

    async def _execute(self, leased: queue_store.JobRecord) -> None:
        self.state.active_job_uid = leased.job_uid
        try:
            if self._cancel_if_requested_before_start(leased.job_uid):
                return

            job_type = job_stubs.job_type(leased)
            if job_type == "ollama_publish":
                dispatch_ready = _publish_dispatch_ready_after_lease(
                    self.capability_snapshot_fn
                )
            else:
                dispatch_ready = _job_dispatch_ready(self.capability_snapshot_fn, job_type)
            if not dispatch_ready:
                self.store.mark_policy_blocked(
                    job_uid=leased.job_uid,
                    owner=self.owner,
                    reason_code="publish_not_ready"
                    if job_type == "ollama_publish"
                    else "policy_blocked",
                )
                return

            if job_type == "sft":
                decision = self.policy_decider()
                if self._cancel_if_requested_before_start(leased.job_uid):
                    return
                if not decision.allowed:
                    self.store.mark_policy_blocked(
                        job_uid=leased.job_uid,
                        owner=self.owner,
                        reason_code="policy_blocked",
                    )
                    return
            if self._cancel_if_requested_before_start(leased.job_uid):
                return

            runner_kind = _runner_kind_for_job(leased)
            running_job = self.store.mark_running(
                job_uid=leased.job_uid,
                owner=self.owner,
                runner_kind=runner_kind,
            )
            cancel_event = asyncio.Event()
            self._active_cancel_event = cancel_event
            try:
                runner = self._runner_for_kind(runner_kind)
                task = asyncio.create_task(runner.run(running_job, cancel_event))
            except Exception:
                self.store.mark_failed(
                    job_uid=running_job.job_uid,
                    owner=self.owner,
                    failure_code="runner_failed",
                    failure_class="runner",
                )
                return
            try:
                outcome = await self._drive_runner(running_job.job_uid, task, cancel_event)
            finally:
                self._active_cancel_event = None
            await self._terminalize_runner_outcome(running_job.job_uid, outcome)
        finally:
            self.state.active_job_uid = None

    def _runner_for_kind(self, runner_kind: str) -> object:
        if self.runner_factory is not None:
            return self.runner_factory()
        return runners.build_runner(runner_kind, cfg=self.cfg, store=self.store)

    def _cancel_if_requested_before_start(self, job_uid: str) -> bool:
        current = self.store.get_job(job_uid)
        if current.state != "cancel_requested" or current.lease_owner != self.owner:
            return False
        self.store.mark_running_canceled(
            job_uid=job_uid,
            owner=self.owner,
            reason_code=current.cancel_reason_code or "runner_canceled",
        )
        return True

    async def _drive_runner(
        self,
        job_uid: str,
        task: asyncio.Task[runners.RunnerOutcome],
        cancel_event: asyncio.Event,
    ) -> runners.RunnerOutcome:
        loop = asyncio.get_running_loop()
        timeout_at = loop.time() + int(
            getattr(self.cfg, "executor_job_timeout_seconds", 300)
        )
        renew_interval = int(getattr(self.cfg, "executor_renew_interval_seconds", 60))
        renew_at = loop.time() + renew_interval
        while True:
            if self._stop_event.is_set():
                cancel_event.set()
            wait_for = min(
                0.05,
                max(0.0, renew_at - loop.time()),
                max(0.0, timeout_at - loop.time()),
            )
            done, _pending = await asyncio.wait({task}, timeout=wait_for)
            if done:
                try:
                    self.store.renew_lease(job_uid=job_uid, owner=self.owner)
                    return task.result()
                except queue_store.LeaseUnavailable:
                    return self._lease_lost_outcome(job_uid)
                except Exception:
                    return runners.RunnerOutcome(
                        "failed",
                        failure_code="runner_failed",
                        failure_class="runner",
                    )

            current = self.store.get_job(job_uid)
            if current.state == "cancel_requested" and current.lease_owner == self.owner:
                cancel_event.set()

            now = loop.time()
            if now >= timeout_at:
                cancel_event.set()
                await self._wait_for_runner_stop(task)
                return self._timeout_outcome(job_uid)
            if now >= renew_at:
                try:
                    self.store.renew_lease(job_uid=job_uid, owner=self.owner)
                except queue_store.LeaseUnavailable:
                    cancel_event.set()
                    await self._wait_for_runner_stop(task)
                    return self._lease_lost_outcome(job_uid)
                renew_at = now + renew_interval

    def _timeout_outcome(self, job_uid: str) -> runners.RunnerOutcome:
        if self._job_type_or_none(job_uid) == "ollama_publish":
            return runners.RunnerOutcome(
                "failed",
                failure_code="publish_verification_unknown",
                failure_class="publish",
            )
        return runners.RunnerOutcome(
            "failed",
            failure_code="runner_timeout",
            failure_class="runner",
        )

    def _lease_lost_outcome(self, job_uid: str) -> runners.RunnerOutcome:
        if self._job_type_or_none(job_uid) == "ollama_publish":
            return runners.RunnerOutcome(
                "failed",
                failure_code="publish_verification_unknown",
                failure_class="publish",
            )
        return runners.RunnerOutcome(
            "failed",
            failure_code="lease_lost",
            failure_class="lease",
        )

    def _job_type_or_none(self, job_uid: str) -> str | None:
        try:
            return self.store.get_job(job_uid).job_type
        except Exception:
            return None

    async def _terminalize_runner_outcome(
        self,
        job_uid: str,
        outcome: runners.RunnerOutcome,
    ) -> None:
        current = self.store.get_job(job_uid)
        if outcome.failure_code == "lease_lost":
            self._mark_lease_lost_if_possible(job_uid)
            return
        if (
            outcome.failure_code == "publish_verification_unknown"
            and current.job_type == "ollama_publish"
        ):
            if current.state == "lease_expired":
                return
            try:
                self.store.mark_publish_verification_unknown(
                    job_uid=job_uid,
                    owner=self.owner,
                )
            except queue_store.LeaseUnavailable:
                return
            return
        if current.state == "cancel_requested" or outcome.state == "canceled":
            self.store.mark_running_canceled(
                job_uid=job_uid,
                owner=self.owner,
                reason_code=current.cancel_reason_code or "runner_canceled",
            )
            return
        if outcome.state == "succeeded":
            if current.job_type == "ollama_publish":
                publish_result = outcome.publish_result
                if not publish_result:
                    self.store.mark_failed(
                        job_uid=job_uid,
                        owner=self.owner,
                        failure_code="publish_verification_failed",
                        failure_class="publish",
                    )
                    return
                try:
                    self.store.mark_publish_succeeded(
                        job_uid=job_uid,
                        owner=self.owner,
                        publish_job_uid=str(publish_result["publish_job_uid"]),
                        target_ref=str(publish_result["target_ref"]),
                        publish_spec_hash_sha256=str(
                            publish_result["publish_spec_hash_sha256"]
                        ),
                        source_artifact_fingerprint_sha256=str(
                            publish_result["source_artifact_fingerprint_sha256"]
                        ),
                        provenance_fingerprint_sha256=str(
                            publish_result["provenance_fingerprint_sha256"]
                        ),
                        ollama_digest=publish_result.get("ollama_digest"),
                        idempotent=publish_result.get("idempotent") is True,
                    )
                except (queue_store.LeaseUnavailable, queue_store.InvalidStateTransition):
                    self._mark_lease_lost_if_possible(job_uid)
                return

            artifact_failure = await self._stage_runner_artifacts(
                job_uid,
                outcome.artifacts,
                spec=_kitt_job_spec_or_none(current),
                runner_nonce_sha256=outcome.runner_nonce_sha256,
            )
            if artifact_failure is not None:
                if artifact_failure == "lease_lost":
                    self._mark_lease_lost_if_possible(job_uid)
                    return
                self.store.mark_failed(
                    job_uid=job_uid,
                    owner=self.owner,
                    failure_code=artifact_failure,
                    failure_class="artifact",
                )
                return
            try:
                self.store.mark_succeeded(job_uid=job_uid, owner=self.owner)
            except (queue_store.LeaseUnavailable, queue_store.InvalidStateTransition):
                self._mark_lease_lost_if_possible(job_uid)
            return
        failure_code = outcome.failure_code or "runner_failed"
        failure_class = outcome.failure_class or "runner"
        try:
            self.store.mark_failed(
                job_uid=job_uid,
                owner=self.owner,
                failure_code=failure_code,
                failure_class=failure_class,
            )
        except queue_store.LeaseUnavailable:
            if failure_code == "publish_verification_unknown":
                return
            raise

    async def _wait_for_runner_stop(
        self,
        task: asyncio.Task[runners.RunnerOutcome],
    ) -> None:
        if task.done():
            return
        grace = int(getattr(self.cfg, "executor_shutdown_grace_seconds", 10))
        try:
            await asyncio.wait_for(task, timeout=grace)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return
            except Exception:
                return
        except Exception:
            return

    async def _stage_runner_artifacts(
        self,
        job_uid: str,
        artifacts: tuple[runners.RunnerArtifact, ...],
        *,
        spec: dict | None = None,
        runner_nonce_sha256: str | None = None,
    ) -> str | None:
        if not artifacts:
            return None
        try:
            stager = artifact_staging.ArtifactStager.from_config(
                cfg=self.cfg,
                store=self.store,
            )
        except artifact_staging.ArtifactStagingError as exc:
            return exc.reason_code
        try:
            validation_error = await self._run_with_lease_renewal(
                job_uid,
                lambda: _validate_runner_artifacts(
                    artifacts,
                    spec=spec,
                    runner_nonce_sha256=runner_nonce_sha256,
                ),
            )
        except queue_store.LeaseUnavailable:
            return "lease_lost"
        if validation_error is not None:
            return validation_error
        for artifact in artifacts:
            try:
                size_bytes = artifact.path.stat().st_size
                if not _is_safe_runner_artifact_path(artifact.path, self.cfg):
                    return "artifact_path_invalid"
                result = await self._run_with_lease_renewal(
                    job_uid,
                    lambda artifact=artifact, size_bytes=size_bytes: stager.stage_file(
                        job_uid=job_uid,
                        role=artifact.role,
                        source_path=artifact.path,
                        expected_size_bytes=size_bytes,
                    ),
                )
            except queue_store.LeaseUnavailable:
                return "lease_lost"
            except (OSError, artifact_staging.ArtifactStagingError):
                return "artifact_staging_failed"
            if result.status != "staged":
                return result.failure_code or "artifact_staging_failed"
        return None

    async def _run_with_lease_renewal(self, job_uid: str, func):
        self.store.renew_lease(job_uid=job_uid, owner=self.owner)
        task = asyncio.create_task(asyncio.to_thread(func))
        renew_interval = max(
            0.05,
            float(getattr(self.cfg, "executor_renew_interval_seconds", 60)),
        )
        while True:
            done, _pending = await asyncio.wait({task}, timeout=renew_interval)
            if done:
                result = task.result()
                self.store.renew_lease(job_uid=job_uid, owner=self.owner)
                return result
            self.store.renew_lease(job_uid=job_uid, owner=self.owner)

    def _mark_lease_lost_if_possible(self, job_uid: str) -> None:
        with suppress(Exception):
            current = self.store.get_job(job_uid)
            if current.state == "lease_expired":
                return
            self.store.mark_lease_lost(job_uid=job_uid, owner=self.owner)

    async def _sleep_poll_interval(self) -> None:
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=float(getattr(self.cfg, "executor_poll_interval_seconds", 1.0)),
            )
        except asyncio.TimeoutError:
            return


def start_executor_task(app_state: object) -> asyncio.Task[None] | None:
    cfg = getattr(app_state, "worker_config", None)
    store = getattr(app_state, "queue_store", None)
    if store is None or cfg is None:
        return None
    if not getattr(cfg, "executor_enabled", False):
        return None
    if getattr(cfg, "executor_config_error_code", None) is not None:
        setattr(
            app_state,
            "executor_state",
            ExecutorState(failed=True, last_error_code="config_invalid"),
        )
        return None
    capability_snapshot_fn = lambda: _capability_snapshot_for_executor(
        cfg,
        store,
        probes=getattr(app_state, "capability_probes", None),
        now_fn=getattr(app_state, "capability_clock", None),
    )
    if not _any_dispatch_ready(capability_snapshot_fn):
        setattr(
            app_state,
            "executor_state",
            ExecutorState(failed=False, last_error_code="dispatch_gate_closed"),
        )
        return None
    worker = KittWorkerExecutor(
        store=store,
        cfg=cfg,
        capability_snapshot_fn=capability_snapshot_fn,
    )
    setattr(app_state, "executor", worker)
    setattr(app_state, "executor_state", worker.state)
    task: asyncio.Task[None] = asyncio.create_task(worker.run())
    setattr(app_state, "executor_task", task)
    return task


def _policy_decider_for_cfg(cfg: object) -> PolicyDecider:
    return gpu_policy.decide_execution_start


def _capability_snapshot_for_executor(
    cfg: object,
    store: queue_store.QueueStore,
    *,
    probes: capabilities.CapabilityProbes | None = None,
    now_fn: Callable[[], Any] | None = None,
) -> Mapping[str, Any]:
    try:
        queue_status = store.queue_stats()
    except queue_store.QueueUnavailable as exc:
        queue_status = queue_store.unavailable_stats(exc.reason_code)
    except Exception:
        queue_status = queue_store.unavailable_stats("sqlite_unavailable")
    return capabilities.build_capability_snapshot(
        cfg=cfg,
        active_jobs=_active_jobs_from_queue_status(queue_status),
        queue_status=queue_status,
        probes=probes,
        now_fn=now_fn,
    )


def _active_jobs_from_queue_status(queue_status: Mapping[str, Any]) -> int:
    value = queue_status.get("active_jobs", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _executor_dispatch_ready(snapshot_fn: CapabilitySnapshotFn) -> bool:
    try:
        snapshot = snapshot_fn()
    except Exception:
        return False
    return _executor_dispatch_ready_from_snapshot(snapshot)


def _executor_dispatch_ready_from_snapshot(snapshot: Mapping[str, Any]) -> bool:
    return (
        snapshot.get("valid_for_scheduling") is True
        and snapshot.get("execution_enabled") is True
    )


def _any_dispatch_ready(snapshot_fn: CapabilitySnapshotFn) -> bool:
    return bool(_dispatchable_job_types(snapshot_fn))


def _dispatchable_job_types(snapshot_fn: CapabilitySnapshotFn) -> tuple[str, ...]:
    try:
        snapshot = snapshot_fn()
    except Exception:
        return ()
    result: list[str] = []
    if _executor_dispatch_ready_from_snapshot(snapshot):
        result.append("sft")
    if _publish_dispatch_ready_from_snapshot(snapshot):
        result.append("ollama_publish")
    return tuple(result)


def _job_dispatch_ready(snapshot_fn: CapabilitySnapshotFn, job_type: str) -> bool:
    if job_type == "sft":
        return _executor_dispatch_ready(snapshot_fn)
    if job_type != "ollama_publish":
        return False
    try:
        snapshot = snapshot_fn()
    except Exception:
        return False
    return _publish_dispatch_ready_from_snapshot(snapshot)


def _publish_dispatch_ready_from_snapshot(snapshot: Mapping[str, Any]) -> bool:
    operations = snapshot.get("operations")
    publish = operations.get("publish") if isinstance(operations, Mapping) else None
    return (
        isinstance(publish, Mapping)
        and publish.get("supported") is True
        and publish.get("ready") is True
        and publish.get("execution_enabled") is True
    )


def _publish_dispatch_ready_after_lease(snapshot_fn: CapabilitySnapshotFn) -> bool:
    try:
        snapshot = snapshot_fn()
    except Exception:
        return False
    if _publish_dispatch_ready_from_snapshot(snapshot):
        return True
    operations = snapshot.get("operations")
    publish = operations.get("publish") if isinstance(operations, Mapping) else None
    operational = snapshot.get("operational_status")
    conflict = (
        operational.get("inference_conflict")
        if isinstance(operational, Mapping)
        else None
    )
    return (
        isinstance(publish, Mapping)
        and publish.get("supported") is True
        and publish.get("blocked_reason") == "worker_active_job"
        and isinstance(operational, Mapping)
        and operational.get("active_jobs") == 1
        and operational.get("queue_status") == "ready"
        and isinstance(conflict, Mapping)
        and conflict.get("status") == "none"
        and conflict.get("source_status") == "complete"
        and operational.get("maintenance_mode") is not True
    )


def _runner_kind_for_job(job: queue_store.JobRecord) -> str:
    job_type = job_stubs.job_type(job)
    if job_type == "sft":
        return "sft_subprocess"
    if job_type == "ollama_publish":
        return "ollama_publish"
    raise job_stubs.JobValidationError("job_type is invalid")


def _validate_runner_artifacts(
    artifacts: tuple[runners.RunnerArtifact, ...],
    *,
    spec: dict | None,
    runner_nonce_sha256: str | None,
) -> str | None:
    for artifact in artifacts:
        artifact_error = runners.validate_artifact_file(
            artifact,
            spec=spec,
            runner_nonce_sha256=runner_nonce_sha256,
        )
        if artifact_error is not None:
            return artifact_error
    return runners.validate_artifact_bundle(
        artifacts,
        spec=spec,
        runner_nonce_sha256=runner_nonce_sha256,
    )


def _kitt_job_spec_or_none(job: queue_store.JobRecord) -> dict | None:
    try:
        return job_stubs.kitt_job_spec(job)
    except job_stubs.JobValidationError:
        return None


def _is_safe_runner_artifact_path(path: Path, cfg: object) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    work_root = Path(getattr(cfg, "data_dir")) / "work"
    try:
        resolved_work_root = work_root.resolve(strict=False)
    except OSError:
        return False
    return resolved != resolved_work_root and resolved.is_relative_to(resolved_work_root)


async def stop_executor_task(app_state: object) -> None:
    worker = getattr(app_state, "executor", None)
    task = getattr(app_state, "executor_task", None)
    if worker is not None:
        await worker.stop()
    if task is None:
        return
    try:
        cfg = getattr(app_state, "worker_config", None)
        timeout = float(getattr(cfg, "executor_shutdown_grace_seconds", 10))
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _owner_from_cfg(cfg: object) -> str:
    worker_id = str(getattr(cfg, "worker_id", "kiron-kitt-worker")).lower()
    safe = "".join(
        char if char.isalnum() or char in "._:-" else "." for char in worker_id
    )
    return f"{safe}.executor.{uuid.uuid4().hex[:12]}"
