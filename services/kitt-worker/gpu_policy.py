"""Side-effect-free execution start policy for Sprint 7."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import time

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot

import capabilities


_CONFLICT_STATUSES = {
    "none",
    "ollama_active",
    "gpu_marker_active",
    "blocked_by_policy",
    "maintenance",
    "unknown",
}
_SOURCE_STATUSES = {"complete", "partial", "unknown"}
_MAINTENANCE_STATUSES = {"active", "inactive", "unknown"}
_UNREADABLE_WARNING_PARTS = (
    "unreadable",
    "unreachable",
    "unavailable",
    "shape_unknown",
    "http_error",
)


@dataclass(frozen=True, slots=True)
class ExecutionStartDecision:
    allowed: bool
    reason_code: str
    conflict_status: str
    source_status: str
    maintenance: str


def training_admission_store() -> AdmissionStore:
    return AdmissionStore()


def measure_training_memory() -> MemorySnapshot:
    """Fresh availability, without relaxing the separate training policy.

    Training receives exclusive admission, not a fabricated job-size estimate.
    The store calls this under its lock; no policy/store calls belong here.
    """
    warnings: list[str] = []
    _, host = capabilities._host_memory_bytes(warnings)
    devices, status, _ = capabilities._cuda_accelerators()
    if (status != "complete" or len(devices) != 1 or type(host) is not int or host <= 0
            or type(devices[0].get("vram_available_bytes")) is not int
            or devices[0]["vram_available_bytes"] <= 0):
        raise AdmissionError("resource_unknown", "fresh training memory unavailable")
    return MemorySnapshot(devices[0]["vram_available_bytes"], host, time.monotonic())


def decide_execution_start(
    probes: capabilities.CapabilityProbes | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> ExecutionStartDecision:
    """Measure the current conflict and resource state and fail closed."""

    if now_fn is not None:
        now_fn()
    probes = capabilities.DefaultCapabilityProbes() if probes is None else probes
    try:
        conflict = probes.probe_inference_conflict()
    except Exception:
        return ExecutionStartDecision(
            allowed=False,
            reason_code="source_unreadable",
            conflict_status="unknown",
            source_status="unknown",
            maintenance="unknown",
        )
    try:
        resource_blocked_reason = capabilities.execution_resource_blocked_reason(
            probes.measure_hardware()
        )
    except Exception:
        resource_blocked_reason = "resource_unknown"
    return _from_measurements(conflict, resource_blocked_reason)


def _from_measurements(
    conflict: capabilities.ConflictProbeResult,
    resource_blocked_reason: str | None,
) -> ExecutionStartDecision:
    status = conflict.status if conflict.status in _CONFLICT_STATUSES else "unknown"
    source_status = (
        conflict.source_status
        if conflict.source_status in _SOURCE_STATUSES
        else "unknown"
    )
    maintenance = (
        conflict.maintenance
        if conflict.maintenance in _MAINTENANCE_STATUSES
        else "unknown"
    )
    warnings = tuple(getattr(conflict, "measurement_warnings", ()) or ())

    allowed = (
        status in {"none", "ollama_active"}
        and source_status == "complete"
        and maintenance == "inactive"
        and resource_blocked_reason is None
    )
    if allowed:
        return ExecutionStartDecision(True, "allowed", status, source_status, maintenance)
    if status in {"blocked_by_policy", "gpu_marker_active"}:
        reason = "gpu_policy_blocked"
    elif status == "maintenance" or maintenance == "active":
        reason = "maintenance"
    elif _looks_unreadable(source_status, warnings):
        reason = "source_unreadable"
    elif resource_blocked_reason is not None:
        reason = resource_blocked_reason
    else:
        reason = "inference_conflict_unknown"
    return ExecutionStartDecision(False, reason, status, source_status, maintenance)


def _looks_unreadable(source_status: str, warnings: tuple[str, ...]) -> bool:
    if source_status != "complete":
        return True
    for warning in warnings:
        if any(part in warning for part in _UNREADABLE_WARNING_PARTS):
            return True
    return False
