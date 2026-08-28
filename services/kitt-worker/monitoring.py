"""Internal redacted monitoring and journald-only alerts for kitt-worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import re
from typing import Any, Callable, Mapping

import artifact_staging
import auth
import capabilities
import queue_store


SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.]\d{1,6})?Z$"
)

ALERT_CODES = frozenset(
    {
        "stale_heartbeat",
        "queue_full",
        "queue_unavailable",
        "stale_lease",
        "capability_expired",
        "capability_probe_failed",
        "gpu_policy_blocked",
        "disk_full",
        "artifact_staging_failed",
        "artifact_cleanup_failed",
        "redaction_failure",
        "monitoring_config_invalid",
    }
)

ALERT_SEVERITY = {
    "stale_heartbeat": "warning",
    "queue_full": "warning",
    "queue_unavailable": "error",
    "stale_lease": "warning",
    "capability_expired": "warning",
    "capability_probe_failed": "warning",
    "gpu_policy_blocked": "warning",
    "disk_full": "error",
    "artifact_staging_failed": "warning",
    "artifact_cleanup_failed": "warning",
    "redaction_failure": "critical",
    "monitoring_config_invalid": "critical",
}

EVENT_FIELDS = frozenset(
    {
        "event",
        "alert_code",
        "state",
        "severity",
        "reason_code",
        "request_id",
        "count",
        "queue_depth",
        "queue_status",
        "source",
        "enabled",
    }
)

SAFE_EVENT_NAMES = frozenset(
    {
        "kitt_worker_alert",
        "kitt_worker_monitoring_startup",
        "kitt_worker_monitoring_event",
    }
)

_LOCAL_PATH_RE = re.compile(r"/(?:opt|usr/lib|run|etc|var|home)/[^\s'\"\)]*")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)
_TRACEBACK_FILE_RE = re.compile(r"\bFile \"[^\"]+\", line [0-9]+[^\n]*")
_EXCEPTION_CLASS_RE = re.compile(
    r"\b(?:Exception|RuntimeError|ValueError|TypeError|OSError|"
    r"ModuleNotFoundError|ImportError|ConfigError|RuntimeConfigError)\b"
)
_DSN_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:[+]srv)?|redis|amqp|amqps|"
    r"mqtt|kafka|sqlite|mssql|oracle|file)://[^\s'\"\)]*",
    re.IGNORECASE,
)
_SIGNED_URL_RE = re.compile(
    r"\b(?:https?|wss?)://[^\s'\"\)]*[?&;][^\s'\"\)]*"
    r"(?:signature|token|secret|credential|password|passwd|api[_-]?key|"
    r"access[_-]?key|session[_-]?key|x-amz-signature|kid|key[_-]?id)"
    r"[^\s'\"\)]*",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)([^/?#\s@]+)@")
_SENSITIVE_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&;][A-Za-z0-9_.:-]*"
    r"(?:signature|token|secret|credential|password|passwd|api[_-]?key|"
    r"access[_-]?key|session[_-]?key|kid|key[_-]?id)"
    r"[A-Za-z0-9_.:-]*=)([^&#;\s]*)"
)
_SENSITIVE_KEY_VALUE_RE = re.compile(
    r"(?i)([\"']?\b[A-Za-z0-9_.:-]*"
    r"(?:authorization|bearer|token|secret|credential|password|passwd|"
    r"private[_-]?key|api[_-]?key|access[_-]?key|session[_-]?key|"
    r"kid|key[_-]?id|prompt|response|canonical_job_spec_json|"
    r"cp[_-]?raw(?:[_-]?text)?|cp[_-]?text|"
    r"artifact[_-]?(?:bytes|data|path|filename))"
    r"[A-Za-z0-9_.:-]*\b[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^,\s;&)}\]]*)"
)
_RAW_CONTENT_FIELD_PATTERN = (
    r"(?:raw[_-]?prompt|raw[_-]?response|canonical_job_spec_json|"
    r"cp[_-]?(?:raw(?:[_-]?text)?|text)|"
    r"artifact[_-]?(?:bytes|data|path|filename))"
)
_RAW_CONTENT_KEY_RE = re.compile(
    r"(?i)[\"']?" + _RAW_CONTENT_FIELD_PATTERN + r"[\"']?\s*[:=]\s*"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\b[^\r\n,;]*")
_CREDENTIAL_RE = re.compile(r"\b[A-Za-z0-9_-]{8,64}\.[A-Za-z0-9_-]{32,256}\b")
_SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b")
_SECRET_WORD_RE = re.compile(
    r"(?i)\b(?:authorization|bearer|token|credential|password|private[_-]?key|"
    r"api[_-]?key|secret|kid|key[_-]?id)\b"
)
_RAW_CONTENT_WORD_RE = re.compile(
    r"(?i)\b" + _RAW_CONTENT_FIELD_PATTERN + r"\b"
)

logger = logging.getLogger("kitt_worker.monitoring")


@dataclass(slots=True)
class AlertRuntimeState:
    state: str = "inactive"
    last_event_at: datetime | None = None
    last_reason_code: str | None = None


@dataclass(slots=True)
class CapabilityMonitorSnapshot:
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    valid_until: datetime | None = None
    capability_hash_sha256: str | None = None
    blocking_reasons: tuple[str, ...] = ()
    probe_status: str = "unknown"
    redaction_reasons: tuple[str, ...] = ()
    last_failure_code: str | None = None


@dataclass(slots=True)
class LeaseExpirySink:
    total_seen: int = 0
    _pending: int = 0

    def record(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            return
        self.total_seen += count
        self._pending += count

    def drain(self) -> int:
        pending = self._pending
        self._pending = 0
        return pending


@dataclass(slots=True)
class MonitoringManager:
    cfg: object
    app_state: object | None = None
    event_sink: Callable[[dict[str, Any]], None] | None = None
    now_fn: Callable[[], datetime] | None = None
    enabled: bool = field(init=False)
    alerts_enabled: bool = field(init=False)
    interval_seconds: float = field(init=False)
    heartbeat_stale_seconds: int = field(init=False)
    alert_repeat_seconds: int = field(init=False)
    lease_expiry_sink: LeaseExpirySink = field(default_factory=LeaseExpirySink)
    capability_snapshot: CapabilityMonitorSnapshot = field(
        default_factory=CapabilityMonitorSnapshot
    )
    executor_failure_reason: str | None = None
    alert_states: dict[str, AlertRuntimeState] = field(default_factory=dict)
    disk_full_sources: dict[str, bool] = field(default_factory=dict)
    last_heartbeat_at: datetime | None = None
    started_at: datetime = field(init=False)
    _stop_event: asyncio.Event | None = field(default=None, init=False)
    _config_invalid_logged: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.enabled = bool(getattr(self.cfg, "monitoring_enabled", True))
        self.alerts_enabled = bool(getattr(self.cfg, "alerts_enabled", True))
        self.interval_seconds = float(
            getattr(self.cfg, "monitoring_interval_seconds", 30.0)
        )
        self.heartbeat_stale_seconds = int(
            getattr(self.cfg, "heartbeat_stale_seconds", 300)
        )
        self.alert_repeat_seconds = int(
            getattr(self.cfg, "alert_repeat_seconds", 3600)
        )
        self.started_at = self._now()
        if getattr(self.cfg, "monitoring_config_error_code", None) is not None:
            self.enabled = False
            self.alerts_enabled = False
            self.emit_config_invalid_once()
        if not self.enabled:
            self.alerts_enabled = False

    def _now(self) -> datetime:
        raw = self.now_fn() if self.now_fn is not None else datetime.now(timezone.utc)
        return _coerce_datetime(raw)

    def emit_config_invalid_once(self) -> None:
        if self._config_invalid_logged:
            return
        self._config_invalid_logged = True
        emit_event(
            "kitt_worker_monitoring_startup",
            {
                "alert_code": "monitoring_config_invalid",
                "state": "active",
                "severity": "critical",
                "reason_code": "monitoring_config_invalid",
                "enabled": False,
            },
            event_sink=self.event_sink,
        )

    def record_heartbeat(self, when: datetime | None = None) -> None:
        if not self.enabled:
            return
        self.last_heartbeat_at = _coerce_datetime(when) if when is not None else self._now()
        self.set_alert("stale_heartbeat", False, reason_code="heartbeat_seen")

    def record_stale_leases(self, count: int) -> None:
        if not self.enabled:
            return
        self.lease_expiry_sink.record(count)

    def record_artifact_cleanup_result(
        self,
        result: artifact_staging.CleanupResult,
    ) -> None:
        code = alert_code_from_cleanup_result(result)
        self.set_alert(
            "artifact_cleanup_failed",
            code == "artifact_cleanup_failed",
            reason_code=code or "artifact_cleanup_ok",
            count=max(0, int(getattr(result, "failed_deletes", 0))),
        )

    def record_artifact_stage_result(
        self,
        result: artifact_staging.StageResult,
    ) -> None:
        codes = alert_codes_from_stage_result(result)
        self.set_disk_full_source(
            "artifact_staging",
            "disk_full" in codes,
            reason_code=(
                _safe_reason(getattr(result, "failure_code", None))
                if "disk_full" in codes
                else "artifact_stage_ok"
            ),
        )
        self.set_alert(
            "artifact_staging_failed",
            "artifact_staging_failed" in codes,
            reason_code=(
                _safe_reason(getattr(result, "failure_code", None))
                if "artifact_staging_failed" in codes
                else "artifact_stage_ok"
            ),
        )

    def set_alert(
        self,
        alert_code: str,
        active: bool,
        *,
        reason_code: str | None = None,
        count: int | None = None,
        queue_depth: int | None = None,
        queue_status: str | None = None,
        source: str | None = None,
        request_id: str | None = None,
    ) -> None:
        if alert_code not in ALERT_CODES or alert_code == "monitoring_config_invalid":
            self._emit_redaction_failure()
            return
        state = self.alert_states.setdefault(alert_code, AlertRuntimeState())
        now = self._now()
        reason = _safe_reason(reason_code) or alert_code
        if active:
            should_emit = False
            next_state = "active"
            if state.state != "active":
                should_emit = True
            elif (
                state.last_event_at is not None
                and self.alert_repeat_seconds > 0
                and (now - state.last_event_at).total_seconds()
                >= self.alert_repeat_seconds
            ):
                should_emit = True
            if should_emit:
                self._emit_alert(
                    alert_code,
                    next_state,
                    reason_code=reason,
                    count=count,
                    queue_depth=queue_depth,
                    queue_status=queue_status,
                    source=source,
                    request_id=request_id,
                )
                state.last_event_at = now
                state.last_reason_code = reason
            state.state = next_state
            return

        if state.state == "active":
            self._emit_alert(
                alert_code,
                "inactive",
                reason_code=reason,
                count=count,
                queue_depth=queue_depth,
                queue_status=queue_status,
                source=source,
                request_id=request_id,
            )
            state.last_event_at = now
            state.last_reason_code = reason
            state.state = "inactive"

    def set_disk_full_source(
        self,
        source: str,
        active: bool,
        *,
        reason_code: str | None = None,
        count: int | None = None,
        queue_depth: int | None = None,
        queue_status: str | None = None,
        request_id: str | None = None,
    ) -> None:
        safe_source = _safe_reason(source) or "unknown"
        self.disk_full_sources[safe_source] = bool(active)
        aggregate_active = any(self.disk_full_sources.values())
        if aggregate_active and not active:
            state = self.alert_states.get("disk_full")
            if state is not None and state.state == "active":
                return
        if aggregate_active:
            active_source = next(
                (
                    name
                    for name, source_active in self.disk_full_sources.items()
                    if source_active
                ),
                safe_source,
            )
            self.set_alert(
                "disk_full",
                True,
                reason_code=(_safe_reason(reason_code) if active else None) or "disk_full",
                count=count,
                queue_depth=queue_depth,
                queue_status=queue_status,
                source=active_source,
                request_id=request_id,
            )
            return
        self.set_alert(
            "disk_full",
            False,
            reason_code=_safe_reason(reason_code) or "disk_ok",
            count=count,
            queue_depth=queue_depth,
            queue_status=queue_status,
            source=safe_source,
            request_id=request_id,
        )

    def _emit_alert(self, alert_code: str, state: str, **fields: Any) -> None:
        if not self.alerts_enabled:
            return
        payload = {
            "alert_code": alert_code,
            "state": state,
            "severity": ALERT_SEVERITY[alert_code],
        }
        payload.update({key: value for key, value in fields.items() if value is not None})
        emit_event("kitt_worker_alert", payload, event_sink=self.event_sink)

    def _emit_redaction_failure(self) -> None:
        emit_event(
            "kitt_worker_alert",
            {
                "alert_code": "redaction_failure",
                "state": "active",
                "severity": "critical",
                "reason_code": "redaction_failure",
            },
            event_sink=self.event_sink,
        )

    async def run(self) -> None:
        if not self.enabled:
            return
        self._stop_event = asyncio.Event()
        while not self._stop_event.is_set():
            self.poll_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def poll_once(self) -> None:
        if not self.enabled:
            return
        now = self._now()
        queue_stats = self._queue_stats()
        self._poll_queue(queue_stats)
        self._poll_leases()
        self._poll_executor()
        self._poll_capability(queue_stats, now)
        self._poll_artifacts()
        self._poll_heartbeat(now)

    def _queue_stats(self) -> Mapping[str, Any]:
        unavailable = getattr(self.app_state, "queue_unavailable", None)
        if unavailable is not None:
            return queue_store.unavailable_stats(unavailable.reason_code)
        store = getattr(self.app_state, "queue_store", None)
        if store is None:
            return queue_store.unavailable_stats("sqlite_unavailable")
        try:
            return store.queue_stats()
        except queue_store.QueueUnavailable as exc:
            setattr(
                self.app_state,
                "queue_unavailable",
                queue_store.QueueUnavailableState(exc.reason_code),
            )
            setattr(self.app_state, "queue_store", None)
            return queue_store.unavailable_stats(exc.reason_code)
        except Exception:
            setattr(
                self.app_state,
                "queue_unavailable",
                queue_store.QueueUnavailableState("sqlite_unavailable"),
            )
            setattr(self.app_state, "queue_store", None)
            return queue_store.unavailable_stats("sqlite_unavailable")

    def _poll_queue(self, stats: Mapping[str, Any]) -> None:
        codes = alert_codes_from_queue_stats(stats)
        queue_depth = _non_negative_int(stats.get("queue_depth"))
        queue_status = _safe_reason(stats.get("queue_status"))
        reason = _safe_reason(stats.get("queue_degraded_reason"))
        self.set_alert(
            "queue_full",
            "queue_full" in codes,
            reason_code=reason or "queue_ready",
            queue_depth=queue_depth,
            queue_status=queue_status,
            source="queue",
        )
        self.set_alert(
            "queue_unavailable",
            "queue_unavailable" in codes,
            reason_code=reason or "queue_ready",
            queue_depth=queue_depth,
            queue_status=queue_status,
            source="queue",
        )
        self.set_disk_full_source(
            "queue",
            "disk_full" in codes,
            reason_code=reason or "disk_ok",
            queue_depth=queue_depth,
            queue_status=queue_status,
        )

    def _poll_leases(self) -> None:
        count = self.lease_expiry_sink.drain()
        self.set_alert(
            "stale_lease",
            count > 0,
            reason_code="stale_lease" if count > 0 else "no_stale_lease",
            count=count,
            source="leases",
        )

    def _poll_executor(self) -> None:
        executor_state = getattr(self.app_state, "executor_state", None)
        if executor_state is None:
            self.executor_failure_reason = None
            return
        failed = getattr(executor_state, "failed", False) is True
        reason = _safe_reason(getattr(executor_state, "last_error_code", None))
        self.executor_failure_reason = (reason or "executor_failed") if failed else None

    def _poll_capability(self, queue_stats: Mapping[str, Any], now: datetime) -> None:
        snapshot, failed = self._build_capability_monitor_snapshot(queue_stats, now)
        if snapshot is not None:
            self.capability_snapshot = snapshot
        expired = capability_snapshot_expired(self.capability_snapshot, now)
        self.set_alert(
            "capability_expired",
            expired,
            reason_code="capability_expired" if expired else "capability_valid",
            source="capabilities",
        )
        executor_failed = self.executor_failure_reason is not None
        probe_failed = (
            failed
            or self.capability_snapshot.probe_status != "complete"
            or executor_failed
        )
        self.set_alert(
            "capability_probe_failed",
            probe_failed,
            reason_code=(
                self.executor_failure_reason
                if executor_failed
                else (
                self.capability_snapshot.last_failure_code
                if failed
                else self.capability_snapshot.probe_status
                )
            ),
            source="executor" if executor_failed else "capabilities",
        )
        gpu_blocked = "gpu_policy_blocked" in alert_codes_from_capability_snapshot_data(
            self.capability_snapshot.blocking_reasons
        )
        self.set_alert(
            "gpu_policy_blocked",
            gpu_blocked,
            reason_code=_gpu_policy_reason(self.capability_snapshot.blocking_reasons),
            source="gpu_policy",
        )

    def _build_capability_monitor_snapshot(
        self,
        queue_stats: Mapping[str, Any],
        now: datetime,
    ) -> tuple[CapabilityMonitorSnapshot | None, bool]:
        try:
            probes = getattr(self.app_state, "capability_probes", None)
            now_fn = getattr(self.app_state, "capability_clock", None)
            snapshot = capabilities.build_capability_snapshot(
                cfg=getattr(self.app_state, "worker_config", None),
                active_jobs=_active_jobs_from_queue_stats(queue_stats),
                queue_status=queue_stats,
                probes=probes,
                now_fn=now_fn,
            )
        except Exception:
            previous = self.capability_snapshot
            return (
                CapabilityMonitorSnapshot(
                    last_attempt_at=now,
                    last_success_at=previous.last_success_at,
                    valid_until=previous.valid_until,
                    capability_hash_sha256=previous.capability_hash_sha256,
                    blocking_reasons=previous.blocking_reasons,
                    probe_status=previous.probe_status,
                    redaction_reasons=previous.redaction_reasons,
                    last_failure_code="capability_probe_failed",
                ),
                True,
            )
        return capability_monitor_snapshot_from_public(snapshot, now), False

    def _poll_artifacts(self) -> None:
        store = getattr(self.app_state, "queue_store", None)
        summary = {}
        if store is not None:
            try:
                summary = store.artifact_status_summary()
            except queue_store.QueueUnavailable:
                summary = {}
        codes = alert_codes_from_artifact_state(
            unavailable_reason=getattr(self.app_state, "artifact_staging_unavailable", None),
            config_error_code=getattr(
                getattr(self.app_state, "worker_config", None),
                "artifact_config_error_code",
                None,
            ),
            summary=summary,
        )
        self.set_alert(
            "artifact_staging_failed",
            "artifact_staging_failed" in codes,
            reason_code=(
                _artifact_failure_reason(summary)
                or _safe_reason(getattr(self.app_state, "artifact_staging_unavailable", None))
                or "artifact_staging_ok"
            ),
            source="artifact_staging",
        )
        self.set_disk_full_source(
            "artifact_staging",
            "disk_full" in codes,
            reason_code="disk_full" if "disk_full" in codes else "artifact_disk_ok",
        )

    def _poll_heartbeat(self, now: datetime) -> None:
        if self.heartbeat_stale_seconds == 0:
            self.set_alert("stale_heartbeat", False, reason_code="heartbeat_disabled")
            return
        heartbeat_at = self.last_heartbeat_at or self.started_at
        stale = (now - heartbeat_at).total_seconds() >= self.heartbeat_stale_seconds
        self.set_alert(
            "stale_heartbeat",
            stale,
            reason_code="stale_heartbeat" if stale else "heartbeat_recent",
            source="heartbeat",
        )


class RedactingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        safe = sanitize_log_text(rendered)
        if contains_forbidden_text(safe):
            safe = (
                "event=kitt_worker_alert alert_code=redaction_failure "
                "state=active severity=critical reason_code=redaction_failure"
            )
        record.msg = safe
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


def install_logging_redaction(level: int | str = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    redactor = _shared_redacting_filter()
    root.addFilter(redactor)
    for handler in root.handlers:
        handler.addFilter(redactor)
    for name in ("uvicorn", "uvicorn.error", "kitt_worker", "kitt-worker"):
        log = logging.getLogger(name)
        log.setLevel(level)
        log.addFilter(redactor)
    access_log = logging.getLogger("uvicorn.access")
    access_log.disabled = True
    access_log.propagate = False
    access_log.handlers.clear()


def start_monitoring_task(app_state: object) -> asyncio.Task[None] | None:
    manager = getattr(app_state, "monitoring_manager", None)
    if manager is None or not getattr(manager, "enabled", False):
        return None
    task: asyncio.Task[None] = asyncio.create_task(manager.run())
    setattr(app_state, "monitoring_task", task)
    return task


async def stop_monitoring_task(app_state: object) -> None:
    manager = getattr(app_state, "monitoring_manager", None)
    task = getattr(app_state, "monitoring_task", None)
    if manager is not None:
        await manager.stop()
    if task is None:
        return
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def record_heartbeat(app_state: object) -> None:
    manager = getattr(app_state, "monitoring_manager", None)
    if manager is not None:
        manager.record_heartbeat()


def report_alert(
    app_state: object,
    alert_code: str,
    *,
    active: bool = True,
    reason_code: str | None = None,
    count: int | None = None,
    request_id: str | None = None,
    source: str | None = None,
) -> None:
    manager = getattr(app_state, "monitoring_manager", None)
    if manager is not None:
        if alert_code == "disk_full":
            manager.set_disk_full_source(
                source or "external",
                active,
                reason_code=reason_code,
                count=count,
                request_id=request_id,
            )
            return
        manager.set_alert(
            alert_code,
            active,
            reason_code=reason_code,
            count=count,
            request_id=request_id,
            source=source,
        )


def alert_codes_from_queue_stats(stats: Mapping[str, Any]) -> tuple[str, ...]:
    codes: list[str] = []
    status = stats.get("queue_status")
    reason = stats.get("queue_degraded_reason")
    if status == "full":
        codes.append("queue_full")
    elif status == "unavailable":
        codes.append("disk_full" if reason == "disk_full" else "queue_unavailable")
    if reason == "disk_full":
        codes.append("disk_full")
    return tuple(_unique_codes(codes))


def alert_codes_from_capability_snapshot(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    operational = snapshot.get("operational_status")
    if not isinstance(operational, Mapping):
        return ()
    reasons = operational.get("blocking_reasons", ())
    if not isinstance(reasons, (list, tuple)):
        reasons = ()
    return alert_codes_from_capability_snapshot_data(tuple(str(item) for item in reasons))


def alert_codes_from_capability_snapshot_data(
    blocking_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    codes: list[str] = []
    if any(
        reason in {
            "gpu_policy_blocked",
            "ollama_active",
            "maintenance",
            "inference_conflict_unknown",
        }
        for reason in blocking_reasons
    ):
        codes.append("gpu_policy_blocked")
    if "queue_full" in blocking_reasons:
        codes.append("queue_full")
    if "artifact_disk_full" in blocking_reasons or "disk_full" in blocking_reasons:
        codes.append("disk_full")
    return tuple(codes)


def alert_codes_from_artifact_state(
    *,
    unavailable_reason: str | None,
    config_error_code: str | None,
    summary: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    codes: list[str] = []
    reason = _safe_reason(unavailable_reason) or _safe_reason(config_error_code)
    if reason is not None:
        if reason in {"artifact_disk_full", "disk_full"}:
            codes.append("disk_full")
        else:
            codes.append("artifact_staging_failed")
    status_counts = _dict_ints((summary or {}).get("status_counts"))
    failure_counts = _dict_ints((summary or {}).get("failure_counts"))
    if status_counts.get("disk_full", 0) > 0 or failure_counts.get("disk_full", 0) > 0:
        codes.append("disk_full")
    staging_failure_codes = {
        "artifact_write_failed",
        "artifact_finalize_failed",
        "artifact_staging_recovered",
        "artifact_staging_unavailable",
        "artifact_quota_invalid",
        "artifact_too_large",
        "artifact_job_quota_exceeded",
        "artifact_total_quota_exceeded",
        "artifact_min_free_blocked",
    }
    if status_counts.get("quota_blocked", 0) > 0:
        codes.append("artifact_staging_failed")
    if any(failure_counts.get(code, 0) > 0 for code in staging_failure_codes):
        codes.append("artifact_staging_failed")
    if failure_counts.get("artifact_min_free_blocked", 0) > 0:
        codes.append("disk_full")
    return tuple(_unique_codes(codes))


def alert_codes_from_stage_result(
    result: artifact_staging.StageResult,
) -> tuple[str, ...]:
    codes: list[str] = []
    status = getattr(result, "status", None)
    failure_code = getattr(result, "failure_code", None)
    quota_failure_codes = {
        "artifact_too_large",
        "artifact_job_quota_exceeded",
        "artifact_total_quota_exceeded",
        "artifact_min_free_blocked",
    }
    if status == "disk_full" or failure_code in {"disk_full", "artifact_disk_full"}:
        codes.append("disk_full")
    if status == "quota_blocked" or failure_code in quota_failure_codes:
        codes.append("artifact_staging_failed")
    if failure_code == "artifact_min_free_blocked":
        codes.append("disk_full")
    if failure_code in {"artifact_write_failed", "artifact_finalize_failed"}:
        codes.append("artifact_staging_failed")
    return tuple(_unique_codes(codes))


def alert_code_from_cleanup_result(
    result: artifact_staging.CleanupResult,
) -> str | None:
    failed_deletes = getattr(result, "failed_deletes", 0)
    reason = getattr(result, "reason_code", None)
    if failed_deletes > 0 or reason == "artifact_cleanup_failed":
        return "artifact_cleanup_failed"
    return None


def capability_monitor_snapshot_from_public(
    snapshot: Mapping[str, Any],
    now: datetime,
) -> CapabilityMonitorSnapshot:
    operational = snapshot.get("operational_status", {})
    blocking_reasons = ()
    if isinstance(operational, Mapping):
        raw = operational.get("blocking_reasons", ())
        if isinstance(raw, (list, tuple)):
            blocking_reasons = tuple(
                item for item in (str(value) for value in raw) if SAFE_CODE_RE.fullmatch(item)
            )
    valid_until = _parse_utc(str(snapshot.get("valid_until", "")))
    capability_hash = snapshot.get("capability_hash_sha256")
    if not (isinstance(capability_hash, str) and re.fullmatch(r"[a-f0-9]{64}", capability_hash)):
        capability_hash = None
    redactions = tuple(
        reason for reason in blocking_reasons if reason == "capability_value_redacted"
    )
    return CapabilityMonitorSnapshot(
        last_attempt_at=now,
        last_success_at=now,
        valid_until=valid_until,
        capability_hash_sha256=capability_hash,
        blocking_reasons=blocking_reasons,
        probe_status=_probe_status(snapshot),
        redaction_reasons=redactions,
        last_failure_code=None,
        )


def _active_jobs_from_queue_stats(queue_stats: Mapping[str, Any]) -> int:
    value = queue_stats.get("active_jobs", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def capability_snapshot_expired(
    snapshot: CapabilityMonitorSnapshot,
    now: datetime,
) -> bool:
    if snapshot.last_success_at is None:
        return snapshot.last_failure_code is not None
    if snapshot.valid_until is None:
        return True
    return snapshot.valid_until <= now


def emit_event(
    event: str,
    fields: Mapping[str, Any],
    *,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    payload: dict[str, Any] = {"event": event}
    payload.update(dict(fields))
    safe_payload = _sanitize_event_payload(payload)
    if safe_payload is None:
        safe_payload = {
            "event": "kitt_worker_alert",
            "alert_code": "redaction_failure",
            "state": "active",
            "severity": "critical",
            "reason_code": "redaction_failure",
        }
    _write_event(safe_payload, event_sink=event_sink)


def sanitize_log_text(value: object) -> str:
    text = auth.redact_diagnostic(str(value))
    text = _redact_raw_content_values(text)
    text = _TRACEBACK_RE.sub("<redacted-traceback>", text)
    text = _TRACEBACK_FILE_RE.sub("<redacted-traceback-frame>", text)
    text = _EXCEPTION_CLASS_RE.sub("<redacted-exception>", text)
    text = _DSN_RE.sub("<redacted-dsn>", text)
    text = _SIGNED_URL_RE.sub("<redacted-url>", text)
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _SENSITIVE_QUERY_PARAM_RE.sub(r"\1<redacted>", text)
    text = _SENSITIVE_KEY_VALUE_RE.sub(r"\1<redacted>", text)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _CREDENTIAL_RE.sub("<redacted-token>", text)
    text = _SHA256_RE.sub("<redacted-hash>", text)
    text = _LOCAL_PATH_RE.sub("<redacted-path>", text)
    text = _RAW_CONTENT_WORD_RE.sub("<redacted-payload>", text)
    text = _SECRET_WORD_RE.sub("<redacted-secret-word>", text)
    return " ".join(text.split())[:1024]


def contains_forbidden_text(value: object) -> bool:
    text = str(value)
    return any(
        regex.search(text)
        for regex in (
            _LOCAL_PATH_RE,
            _TRACEBACK_RE,
            _TRACEBACK_FILE_RE,
            _EXCEPTION_CLASS_RE,
            _DSN_RE,
            _SIGNED_URL_RE,
            _BEARER_RE,
            _CREDENTIAL_RE,
            _SECRET_WORD_RE,
            _RAW_CONTENT_KEY_RE,
            _RAW_CONTENT_WORD_RE,
        )
    )


def _redact_raw_content_values(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    while True:
        match = _RAW_CONTENT_KEY_RE.search(text, cursor)
        if match is None:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor : match.start()])
        parts.append("<redacted-payload>")
        cursor = _consume_raw_content_value(text, match.end())
    return "".join(parts)


def _consume_raw_content_value(text: str, start: int) -> int:
    pos = start
    length = len(text)
    while pos < length and text[pos].isspace():
        pos += 1
    if pos >= length:
        return pos
    if text[pos] in {"'", '"'}:
        return _consume_quoted_value(text, pos)
    if text[pos] in {"{", "["}:
        return _consume_balanced_value(text, pos)
    next_key = _RAW_CONTENT_KEY_RE.search(text, pos + 1)
    stop = length
    for marker in (",", ";", "\n", ")", "}", "]"):
        candidate = text.find(marker, pos)
        if candidate != -1:
            stop = min(stop, candidate)
    if next_key is not None:
        stop = min(stop, next_key.start())
    return stop


def _consume_quoted_value(text: str, start: int) -> int:
    quote = text[start]
    escaped = False
    for pos in range(start + 1, len(text)):
        char = text[pos]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return pos + 1
    return len(text)


def _consume_balanced_value(text: str, start: int) -> int:
    closers = {"{": "}", "[": "]"}
    stack = [closers[text[start]]]
    pos = start + 1
    while pos < len(text):
        char = text[pos]
        if char in {"'", '"'}:
            pos = _consume_quoted_value(text, pos)
            continue
        if char in closers:
            stack.append(closers[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return pos + 1
        pos += 1
    return len(text)


def _sanitize_event_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    if set(payload) - EVENT_FIELDS:
        return None
    event = payload.get("event")
    if not isinstance(event, str) or event not in SAFE_EVENT_NAMES:
        return None
    alert_code = payload.get("alert_code")
    if not isinstance(alert_code, str) or alert_code not in ALERT_CODES:
        return None
    safe: dict[str, Any] = {"event": event, "alert_code": alert_code}
    for key, value in payload.items():
        if key in {"event", "alert_code"} or value is None:
            continue
        normalized = _normalize_event_value(key, value)
        if normalized is None:
            return None
        safe[key] = normalized
    rendered = _render_event(safe)
    if contains_forbidden_text(sanitize_log_text(rendered)):
        return None
    return safe


def _normalize_event_value(key: str, value: Any) -> str | int | bool | None:
    if key in {"count", "queue_depth"}:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
    if key == "enabled":
        return value if isinstance(value, bool) else None
    if key == "request_id":
        return value if _safe_request_id(value) else None
    if key == "state":
        return value if value in {"active", "inactive", "cooldown"} else None
    if key == "severity":
        return value if value in {"info", "warning", "error", "critical"} else None
    if key in {"reason_code", "queue_status", "source"}:
        return value if isinstance(value, str) and SAFE_CODE_RE.fullmatch(value) else None
    return None


def _write_event(
    payload: dict[str, Any],
    *,
    event_sink: Callable[[dict[str, Any]], None] | None,
) -> None:
    if event_sink is not None:
        event_sink(dict(payload))
        return
    severity = payload.get("severity", ALERT_SEVERITY.get(payload["alert_code"], "info"))
    level = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
    }.get(str(severity), logging.INFO)
    logger.log(level, _render_event(payload), exc_info=False)


def _render_event(payload: Mapping[str, Any]) -> str:
    return " ".join(f"{key}={_format_event_value(value)}" for key, value in payload.items())


def _format_event_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _coerce_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_utc(value: str) -> datetime | None:
    if not RFC3339_UTC_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _probe_status(snapshot: Mapping[str, Any]) -> str:
    statuses: list[str] = []
    hardware = snapshot.get("hardware", {})
    trainer = snapshot.get("trainer_stack", {})
    operational = snapshot.get("operational_status", {})
    if isinstance(hardware, Mapping):
        statuses.append(str(hardware.get("measurement_status", "unknown")))
    if isinstance(trainer, Mapping):
        statuses.append(str(trainer.get("measurement_status", "unknown")))
    if isinstance(operational, Mapping):
        conflict = operational.get("inference_conflict", {})
        if isinstance(conflict, Mapping):
            statuses.append(str(conflict.get("source_status", "unknown")))
    if statuses and all(status == "complete" for status in statuses):
        return "complete"
    if any(status == "partial" for status in statuses):
        return "partial"
    return "unknown"


def _gpu_policy_reason(blocking_reasons: tuple[str, ...]) -> str:
    for reason in (
        "gpu_policy_blocked",
        "ollama_active",
        "maintenance",
        "inference_conflict_unknown",
    ):
        if reason in blocking_reasons:
            return reason
    return "gpu_policy_ok"


def _artifact_failure_reason(summary: Mapping[str, Any]) -> str | None:
    failure_counts = _dict_ints(summary.get("failure_counts"))
    for reason in (
        "artifact_write_failed",
        "artifact_finalize_failed",
        "artifact_staging_recovered",
        "artifact_staging_unavailable",
        "artifact_quota_invalid",
        "artifact_too_large",
        "artifact_job_quota_exceeded",
        "artifact_total_quota_exceeded",
        "artifact_min_free_blocked",
    ):
        if failure_counts.get(reason, 0) > 0:
            return reason
    return None


def _dict_ints(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(key, str) and SAFE_CODE_RE.fullmatch(key):
            result[key] = _non_negative_int(raw)
    return result


def _safe_reason(value: object) -> str | None:
    if isinstance(value, str) and SAFE_CODE_RE.fullmatch(value):
        return value
    return None


def _safe_request_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and REQUEST_ID_RE.fullmatch(value) is not None
        and _CREDENTIAL_RE.search(value) is None
        and _SHA256_RE.search(value) is None
        and _SECRET_WORD_RE.search(value) is None
    )


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _unique_codes(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in ALERT_CODES and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _shared_redacting_filter() -> RedactingLogFilter:
    existing = getattr(_shared_redacting_filter, "_instance", None)
    if existing is None:
        existing = RedactingLogFilter()
        setattr(_shared_redacting_filter, "_instance", existing)
    return existing
