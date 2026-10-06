"""Sprint 5 kitt-worker capability snapshot builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol
import urllib.error
import urllib.request

from kiron_common.gpu_admission import AdmissionStore

import artifact_staging
import config as worker_config
import sft_command


SCHEMA_VERSION = "kitt_worker_capabilities_v2"
SNAPSHOT_VERSION = 1
DEFAULT_TTL_SECONDS = 60
CAPABILITY_MODE = "policy_snapshot"
POLICY_DECISION = "block_training"
OPEN_POLICY_DECISION = POLICY_DECISION
SCHEDULING_GATE_REASON = "tareas_dispatch_gate_closed"
OPEN_SCHEDULING_GATE_REASON = "operator_dispatch_open"
SPRINT12_SFT_PROFILE_REF = "00000000-0000-4000-8000-000000000901:v1"
SPRINT12_SFT_PROFILE_HASH = "0acbaef1d68b6085394669788b203878853d20c29217932506fd60ca5ea8c123"
SPRINT12_REQUIRED_OUTPUT_ARTIFACTS = {"allow_merge", "gguf_converter_path"}

_SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_LOCAL_PATH_RE = re.compile(r"/(?:opt|usr/lib|run|etc|var|home)/[^\s'\"\)]*")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9+.-]:)(?<!/)/(?:[A-Za-z0-9._-]+/)+[^\s'\"\)]*"
)
_INTERNAL_URL_RE = re.compile(
    r"\b(?:https?|wss?)://"
    r"(?:[^/?#\s@]+@)?"
    r"(?:localhost|127(?:[.][0-9]{1,3}){3}|10[.]0[.]12[.]16)"
    r"(?::[0-9]{1,5})?"
    r"(?:/[^\s'\"\)]*)?"
)
_SIGNED_URL_RE = re.compile(
    r"\b(?:https?|wss?)://[^\s'\"\)]*[?&;][^\s'\"\)]*"
    r"(?:signature|token|secret|credential|password|passwd|api[_-]?key|"
    r"access[_-]?key|session[_-]?key|x-amz-signature)"
    r"[^\s'\"\)]*",
    re.IGNORECASE,
)
_DSN_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:[+]srv)?|redis|amqp|amqps|"
    r"mqtt|kafka|sqlite|mssql|oracle|file)://[^\s'\"\)]*",
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
    r"kid|key[_-]?id)"
    r"[A-Za-z0-9_.:-]*\b[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^,\s;&)}\]]*)"
)
_SECRET_WORD_RE = re.compile(
    r"(?i)\b(?:authorization|bearer|token|credential|password|private[_-]?key|"
    r"api[_-]?key|secret|kid|key[_-]?id)\b"
)

_ACCELERATOR_KINDS = {"cuda", "cpu", "unknown"}
_CAPABILITY_STATUSES = {"available", "unavailable", "unknown"}
_MEASUREMENT_STATUSES = {"complete", "partial", "unknown"}
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
_TRAINER_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
)
_TRAINING_PHASES = ("sft", "dpo", "cp")
_FEATURES = ("sft", "qlora", "dpo", "cp", "checkpoint_resume")
_DEFAULT_MARKER_PATHS = {
    "gpu_service_loading": Path("/run/kiron/vram/gpu-service-loading.json"),
    "startup": Path("/run/kiron/vram/docling-vram-startup.json"),
    "shutdown": Path("/run/kiron/vram/docling-vram-shutdown.json"),
}
_DEFAULT_MAINTENANCE_PATH = Path("/usr/lib/kiron/data/kiron-proxy/maintenance_mode.json")
_DEFAULT_OLLAMA_PS_URL = "http://127.0.0.1:11435/api/ps"


@dataclass(frozen=True, slots=True)
class HardwareProbeResult:
    accelerators: tuple[Mapping[str, Any], ...] = ()
    host_ram_total_bytes: int | None = None
    host_ram_available_bytes: int | None = None
    staging_free_bytes: int | None = None
    measurement_status: str = "unknown"
    measurement_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrainerProbeResult:
    python_runtime: str
    packages: tuple[Mapping[str, Any], ...]
    measurement_status: str
    measurement_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConflictProbeResult:
    status: str = "unknown"
    source_status: str = "unknown"
    maintenance: str = "unknown"
    measurement_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceCatalog:
    path: Path
    data: Mapping[str, Any]


class CapabilityProbes(Protocol):
    def measure_hardware(self) -> HardwareProbeResult:
        ...

    def measure_trainer_stack(self) -> TrainerProbeResult:
        ...

    def probe_inference_conflict(self) -> ConflictProbeResult:
        ...


class DefaultCapabilityProbes:
    """Side-effect-free best-effort probes; failures become unknown."""

    def __init__(self, cfg: object | None = None) -> None:
        self._cfg = cfg
        self._conflict_probe = InferenceConflictProbe()

    def measure_hardware(self) -> HardwareProbeResult:
        warnings: list[str] = []
        host_total, host_available = _host_memory_bytes(warnings)
        staging_free = _runtime_free_bytes(self._cfg, warnings)
        accelerators, accelerator_status, accelerator_warnings = _cuda_accelerators()
        warnings.extend(accelerator_warnings)
        statuses = [accelerator_status]
        statuses.append("complete" if host_total is not None and host_available is not None else "unknown")
        statuses.append("complete" if staging_free is not None else "unknown")
        return HardwareProbeResult(
            accelerators=tuple(accelerators),
            host_ram_total_bytes=host_total,
            host_ram_available_bytes=host_available,
            staging_free_bytes=staging_free,
            measurement_status=_combine_measurement_status(statuses),
            measurement_warnings=tuple(warnings),
        )

    def measure_trainer_stack(self) -> TrainerProbeResult:
        packages: list[dict[str, Any]] = []
        missing = 0
        for name in _TRAINER_PACKAGES:
            try:
                version = importlib.metadata.version(name)
                available = True
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
                available = False
                missing += 1
            except Exception:
                version = "unknown"
                available = False
                missing += 1
            packages.append(
                {
                    "name": name,
                    "version": version,
                    "available": available,
                }
            )
        if missing == 0:
            status = "complete"
        elif missing == len(_TRAINER_PACKAGES):
            status = "unknown"
        else:
            status = "partial"
        warnings = ("trainer_dependency_missing",) if missing else ()
        return TrainerProbeResult(
            python_runtime=platform.python_version(),
            packages=tuple(packages),
            measurement_status=status,
            measurement_warnings=warnings,
        )

    def probe_inference_conflict(self) -> ConflictProbeResult:
        return self._conflict_probe.probe()


class InferenceConflictProbe:
    """Read-only conflict detector with strict shape validation."""

    def __init__(
        self,
        *,
        marker_paths: Mapping[str, Path] | None = None,
        maintenance_path: Path = _DEFAULT_MAINTENANCE_PATH,
        ollama_ps_url: str = _DEFAULT_OLLAMA_PS_URL,
        timeout_seconds: float = 0.25,
        urlopen: Callable[..., Any] | None = None,
        wall_time_fn: Callable[[], float] = time.time,
        admission_store: AdmissionStore | None = None,
    ) -> None:
        self._marker_paths = dict(_DEFAULT_MARKER_PATHS if marker_paths is None else marker_paths)
        self._maintenance_path = maintenance_path
        self._ollama_ps_url = ollama_ps_url
        self._timeout_seconds = timeout_seconds
        self._urlopen = urllib.request.urlopen if urlopen is None else urlopen
        self._wall_time_fn = wall_time_fn
        self._admission_store = AdmissionStore() if admission_store is None else admission_store

    def probe(self) -> ConflictProbeResult:
        warnings: list[str] = []
        source_statuses: list[str] = []

        maintenance, maintenance_source, maintenance_warnings = self._read_maintenance()
        source_statuses.append(maintenance_source)
        warnings.extend(maintenance_warnings)
        if maintenance == "active":
            return ConflictProbeResult(
                status="maintenance",
                source_status=_combine_source_status(source_statuses),
                maintenance=maintenance,
                measurement_warnings=tuple(warnings),
            )

        marker_status, marker_source, marker_warnings = self._read_markers()
        source_statuses.append(marker_source)
        warnings.extend(marker_warnings)
        if marker_status == "active":
            return ConflictProbeResult(
                status="gpu_marker_active",
                source_status=_combine_source_status(source_statuses),
                maintenance=maintenance,
                measurement_warnings=tuple(warnings),
            )
        if marker_status == "unknown":
            return ConflictProbeResult(
                status="unknown",
                source_status=_combine_source_status(source_statuses),
                maintenance=maintenance,
                measurement_warnings=tuple(warnings),
            )

        try:
            tickets = self._admission_store.snapshot()
        except Exception:
            return ConflictProbeResult("unknown", "unknown", maintenance,
                                       (*warnings, "gpu_admission_unreadable"))
        if tickets:
            return ConflictProbeResult("gpu_marker_active", _combine_source_status(source_statuses),
                                       maintenance, (*warnings, "gpu_admission_reserved"))

        ollama_status, ollama_source, ollama_warnings = self._read_ollama_ps()
        source_statuses.append(ollama_source)
        warnings.extend(ollama_warnings)
        if ollama_status == "active":
            return ConflictProbeResult(
                status="ollama_active",
                source_status=_combine_source_status(source_statuses),
                maintenance=maintenance,
                measurement_warnings=tuple(warnings),
            )
        if ollama_status == "unknown" or maintenance == "unknown":
            return ConflictProbeResult(
                status="unknown",
                source_status=_combine_source_status(source_statuses),
                maintenance=maintenance,
                measurement_warnings=tuple(warnings),
            )
        return ConflictProbeResult(
            status="none",
            source_status=_combine_source_status(source_statuses),
            maintenance=maintenance,
            measurement_warnings=tuple(warnings),
        )

    def _read_maintenance(self) -> tuple[str, str, tuple[str, ...]]:
        try:
            if not self._maintenance_path.exists():
                return "unknown", "unknown", ("maintenance_source_unavailable",)
            data = json.loads(self._maintenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return "unknown", "unknown", ("maintenance_source_unreadable",)
        if not isinstance(data, dict) or not isinstance(data.get("active"), bool):
            return "unknown", "unknown", ("maintenance_shape_unknown",)
        return ("active" if data["active"] else "inactive"), "complete", ()

    def _read_markers(self) -> tuple[str, str, tuple[str, ...]]:
        warnings: list[str] = []
        for path in self._marker_paths.values():
            try:
                if not path.parent.exists():
                    warnings.append("gpu_marker_source_unavailable")
                    return "unknown", "unknown", tuple(warnings)
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                warnings.append("gpu_marker_unreadable")
                return "unknown", "unknown", tuple(warnings)
            if not isinstance(data, dict):
                warnings.append("gpu_marker_shape_unknown")
                return "unknown", "unknown", tuple(warnings)
            created_wall = data.get("created_wall")
            ttl_s = data.get("ttl_s")
            if (
                isinstance(created_wall, bool)
                or isinstance(ttl_s, bool)
                or not isinstance(created_wall, (int, float))
                or not isinstance(ttl_s, (int, float))
            ):
                warnings.append("gpu_marker_shape_unknown")
                return "unknown", "unknown", tuple(warnings)
            ttl_s = max(0.0, min(float(ttl_s), 900.0))
            if self._wall_time_fn() < float(created_wall) + ttl_s:
                return "active", "complete", tuple(warnings)
        return "none", "complete", tuple(warnings)

    def _read_ollama_ps(self) -> tuple[str, str, tuple[str, ...]]:
        try:
            with self._urlopen(self._ollama_ps_url, timeout=self._timeout_seconds) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                if status != 200:
                    return "unknown", "unknown", ("ollama_ps_http_error",)
                raw = response.read(1024 * 1024)
            data = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return "unknown", "unknown", ("ollama_ps_unreachable",)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            return "unknown", "unknown", ("ollama_ps_shape_unknown",)
        for item in data["models"]:
            if not isinstance(item, dict):
                return "unknown", "unknown", ("ollama_ps_shape_unknown",)
            size_vram = item.get("size_vram")
            if isinstance(size_vram, bool) or not isinstance(size_vram, int):
                return "unknown", "unknown", ("ollama_ps_shape_unknown",)
            if size_vram > 0:
                return "active", "complete", ()
        return "none", "complete", ()


def build_capability_snapshot(
    *,
    cfg: object | None = None,
    active_jobs: int = 0,
    queue_status: Mapping[str, Any] | None = None,
    probes: CapabilityProbes | None = None,
    now_fn: Callable[[], datetime] | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any]:
    now = _coerce_datetime(now_fn() if now_fn is not None else datetime.now(timezone.utc))
    ttl_seconds = _validate_ttl_seconds(ttl_seconds)
    probes = DefaultCapabilityProbes(cfg) if probes is None else probes

    hardware_probe = _call_probe(probes.measure_hardware, _unknown_hardware())
    trainer_probe = _call_probe(probes.measure_trainer_stack, _unknown_trainer())
    conflict_probe = _call_probe(probes.probe_inference_conflict, ConflictProbeResult())
    dispatch_open = dispatch_gate_open(cfg)
    publish_ready = _publish_ready(cfg, now=now)

    snapshot_time = _format_utc(now)
    valid_until = _format_utc(now + timedelta(seconds=ttl_seconds))
    snapshot: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_time": snapshot_time,
        "valid_until": valid_until,
        "ttl_seconds": ttl_seconds,
        "capability_mode": CAPABILITY_MODE,
        "valid_for_scheduling": dispatch_open,
        "execution_enabled": dispatch_open,
        "scheduling_gate": {
            "status": "open" if dispatch_open else "closed",
            "reason_code": (
                OPEN_SCHEDULING_GATE_REASON
                if dispatch_open
                else SCHEDULING_GATE_REASON
            ),
            "source": "operator_runtime_config" if dispatch_open else "tareas_subtask_79",
            "policy_decision": (
                OPEN_POLICY_DECISION if dispatch_open else POLICY_DECISION
            ),
        },
        "hardware": _hardware_section(hardware_probe, dispatch_open=dispatch_open),
        "operations": _operations_section(
            dispatch_open=dispatch_open,
            publish_ready=publish_ready,
            publish_blocked_reason=_publish_blocked_reason(cfg, now=now),
        ),
        "model_limits": _model_limits_section(cfg),
        "trainer_stack": _trainer_stack_section(
            trainer_probe,
            dispatch_open=dispatch_open,
        ),
        "parallelism": _parallelism_section(
            queue_status,
            dispatch_open=dispatch_open,
        ),
        "operational_status": _operational_status_section(
            conflict_probe,
            active_jobs=active_jobs,
            queue_status=queue_status,
            dispatch_open=dispatch_open,
        ),
    }
    snapshot, redaction_reasons = scrub_capability_snapshot(snapshot)
    blocking_reasons = _blocking_reasons(
        snapshot,
        redaction_reasons,
        cfg=cfg,
        dispatch_open=dispatch_open,
    )
    if blocking_reasons and dispatch_open:
        snapshot["valid_for_scheduling"] = False
        snapshot["execution_enabled"] = False
        snapshot["scheduling_gate"]["status"] = "closed"
        snapshot["scheduling_gate"]["reason_code"] = blocking_reasons[0]
        snapshot["scheduling_gate"]["source"] = "runtime_probe"
        _disable_dispatch_capacity(snapshot, blocking_reasons[0])
    publish_blocked_reason = _publish_blocked_reason(
        cfg,
        now=now,
    ) or _publish_runtime_blocked_reason(
        snapshot,
    )
    _set_publish_operation_state(snapshot, blocked_reason=publish_blocked_reason)
    snapshot["operational_status"]["blocking_reasons"] = blocking_reasons
    snapshot["operational_status"]["health"] = "degraded" if blocking_reasons else "healthy"
    snapshot, second_redactions = scrub_capability_snapshot(snapshot)
    if second_redactions:
        blocking_reasons = _unique_sorted([*blocking_reasons, *second_redactions])
        snapshot["operational_status"]["blocking_reasons"] = blocking_reasons
        snapshot["operational_status"]["health"] = "degraded"
    snapshot["capability_hash_sha256"] = capability_hash_sha256(snapshot)
    return snapshot


def dispatch_gate_open(cfg: object | None) -> bool:
    return (
        cfg is not None
        and getattr(cfg, "dispatch_gate", "closed") == "open"
        and getattr(cfg, "executor_enabled", False) is True
        and getattr(cfg, "executor_runner", None) == "sft_subprocess"
        and _sft_command_gate_ready(getattr(cfg, "sft_command", ""))
        and getattr(cfg, "executor_config_error_code", None) is None
        and getattr(cfg, "artifact_config_error_code", None) is None
    )


def _disable_dispatch_capacity(snapshot: dict[str, Any], reason_code: str) -> None:
    snapshot["parallelism"]["max_concurrent_jobs"] = 0
    snapshot["parallelism"]["gpu_slots"] = 0
    snapshot["operations"]["training"] = []
    for phase in snapshot["operations"]["training_phases"]:
        if phase["execution_enabled"] is True:
            phase["execution_enabled"] = False
            phase["blocked_reason"] = reason_code
    snapshot["trainer_stack"]["features"] = []
    for feature in snapshot["trainer_stack"].get("feature_status", []):
        if feature["available"] is True:
            feature["available"] = False
            feature["blocked_reason"] = reason_code


def _sft_command_gate_ready(command: object) -> bool:
    return sft_command.is_trusted_sft_command(command)


def capability_hash_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in snapshot.items()
        if key not in {"snapshot_time", "valid_until", "capability_hash_sha256"}
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scrub_capability_snapshot(data: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    redactions: list[str] = []

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): scrub(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(child) for child in value]
        if isinstance(value, str):
            cleaned = _scrub_string(value)
            if cleaned != value:
                redactions.append("capability_value_redacted")
            return cleaned
        return value

    return scrub(dict(data)), _unique_sorted(redactions)


def _hardware_section(
    probe: HardwareProbeResult,
    *,
    dispatch_open: bool = False,
) -> dict[str, Any]:
    accelerators = [
        _accelerator_entry(raw, dispatch_open=dispatch_open)
        for raw in probe.accelerators
    ]
    accelerators = [
        accelerator
        for accelerator in accelerators
        if accelerator["count"] > 0 and accelerator["vram_gib"] > 0
    ]
    if not accelerators:
        accelerators = []
    host_total = _nonnegative_int_or_none(probe.host_ram_total_bytes)
    host_available = _nonnegative_int_or_none(probe.host_ram_available_bytes)
    staging_free = _nonnegative_int_or_none(probe.staging_free_bytes)
    result = {
        "accelerators": accelerators,
        "host_ram_total_bytes": host_total,
        "host_ram_available_bytes": host_available,
        "staging_free_bytes": staging_free,
        "measurement_status": _measurement_status(probe.measurement_status),
        "measurement_warnings": _safe_codes(probe.measurement_warnings),
    }
    host_ram_gib = _bytes_to_gib(host_total)
    disk_free_gib = _bytes_to_gib(staging_free)
    if host_ram_gib > 0:
        result["host_ram_gib"] = host_ram_gib
    if disk_free_gib > 0:
        result["disk_free_gib"] = disk_free_gib
    return result


def _operations_section(
    *,
    dispatch_open: bool = False,
    publish_ready: bool = False,
    publish_blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "training": ["sft"] if dispatch_open else [],
        "eval": False,
        "publish": {
            "supported": True,
            "ready": publish_ready,
            "execution_enabled": publish_ready,
            "job_type": "ollama_publish",
            "runner_kind": "ollama_publish",
            "contract_version": "kitt_publish_job_spec_v1",
            "requires_signed_spec": True,
            "requires_operator_approval": True,
            "approval_plane": "kitt",
            "side_effect": "ollama_create_local_tag",
            "blocked_reason": publish_blocked_reason,
            "rollback": "operator_tag_remove_only",
        },
        "features": ["sft", "qlora"] if dispatch_open else [],
        "training_phases": [
            {
                "phase": phase,
                "declared": True,
                "execution_enabled": dispatch_open and phase == "sft",
                "blocked_reason": _phase_blocked_reason(
                    phase,
                    dispatch_open=dispatch_open,
                ),
            }
            for phase in _TRAINING_PHASES
        ],
        "eval_supported": False,
        "publish_supported": False,
        "cancel_supported": True,
        "resume_supported": True,
        "artifact_staging_supported": True,
    }


def _publish_ready(cfg: object | None, *, now: datetime | None = None) -> bool:
    return _publish_blocked_reason(cfg, now=now) is None


def _publish_blocked_reason(
    cfg: object | None,
    *,
    now: datetime | None = None,
) -> str | None:
    if cfg is None:
        return "execution_test_gate_closed"
    if getattr(cfg, "dispatch_gate", "closed") != "open":
        return "execution_test_gate_closed"
    if getattr(cfg, "executor_enabled", False) is not True:
        return "execution_test_gate_closed"
    reason = getattr(cfg, "executor_config_error_code", None)
    if isinstance(reason, str):
        return reason
    reason = getattr(cfg, "artifact_config_error_code", None)
    if isinstance(reason, str):
        return reason
    reason = worker_config.publish_verify_keyring_availability_error(cfg, now=now)
    return reason if isinstance(reason, str) else None


def _publish_runtime_blocked_reason(snapshot: Mapping[str, Any]) -> str | None:
    operational = snapshot.get("operational_status")
    if not isinstance(operational, Mapping):
        return "queue_unavailable"
    active_jobs = operational.get("active_jobs", 0)
    if isinstance(active_jobs, int) and not isinstance(active_jobs, bool) and active_jobs > 0:
        return "worker_active_job"
    queue_status = operational.get("queue_status")
    if queue_status == "full":
        return "queue_full"
    if queue_status != "ready":
        reason = operational.get("queue_degraded_reason")
        return reason if isinstance(reason, str) else "queue_unavailable"
    conflict = operational.get("inference_conflict")
    if not isinstance(conflict, Mapping):
        return "inference_conflict_unknown"
    status = conflict.get("status")
    source_status = conflict.get("source_status")
    if status == "ollama_active":
        return _ollama_active_resource_blocked_reason(snapshot)
    if status == "maintenance" or operational.get("maintenance_mode") is True:
        return "maintenance"
    if status in {"blocked_by_policy", "gpu_marker_active"}:
        return "gpu_policy_blocked"
    if status == "unknown" or source_status != "complete":
        return "inference_conflict_unknown"
    for warning in snapshot.get("hardware", {}).get("measurement_warnings", []):
        if warning in {
            "artifact_staging_unavailable",
            "artifact_quota_invalid",
            "artifact_cleanup_failed",
            "artifact_disk_full",
        }:
            return str(warning)
    return None


def _ollama_active_resource_blocked_reason(snapshot: Mapping[str, Any]) -> str | None:
    hardware = snapshot.get("hardware")
    if not isinstance(hardware, Mapping):
        return "resource_unknown"
    return _resource_blocked_reason_from_hardware_section(hardware)


def _set_publish_operation_state(
    snapshot: dict[str, Any],
    *,
    blocked_reason: str | None,
) -> None:
    operations = snapshot.get("operations")
    publish = operations.get("publish") if isinstance(operations, dict) else None
    if not isinstance(publish, dict):
        return
    ready = blocked_reason is None
    publish["ready"] = ready
    publish["execution_enabled"] = ready
    publish["blocked_reason"] = blocked_reason


def _model_limits_section(cfg: object | None = None) -> dict[str, Any]:
    limits = _catalog_model_limits(cfg)
    max_model = limits["max_model_parameters_b"]
    max_context = limits["max_context_tokens"]
    max_sequence = limits["max_sequence_length"]
    max_batch = limits["max_batch_size"]
    return {
        "max_model_parameters_b": max_model,
        "max_context_tokens": max_context,
        "quantization_modes": limits["quantization_modes"],
        "max_sequence_length": max_sequence,
        "max_batch_size": max_batch,
        "parameter_budget_b": max_model,
        "context_limit_tokens": max_context,
        "max_sequence_length_tokens": max_sequence,
        "batch_limits": {
            "max_batch_size": max_batch,
            "gradient_accumulation_steps": limits["gradient_accumulation_steps"],
        },
        "limit_source": limits["limit_source"],
        "measurement_status": limits["measurement_status"],
    }


def _trainer_stack_section(
    probe: TrainerProbeResult,
    *,
    dispatch_open: bool = False,
) -> dict[str, Any]:
    return {
        "python_runtime": _safe_python_runtime(probe.python_runtime),
        "packages": [_package_entry(raw) for raw in probe.packages],
        "features": ["sft", "qlora"] if dispatch_open else [],
        "feature_status": [
            {
                "name": feature,
                "available": dispatch_open and feature in {"sft", "qlora"},
                "blocked_reason": _feature_blocked_reason(
                    feature,
                    dispatch_open=dispatch_open,
                ),
            }
            for feature in _FEATURES
        ],
        "measurement_status": _measurement_status(probe.measurement_status),
        "measurement_warnings": _safe_codes(probe.measurement_warnings),
    }


def _phase_blocked_reason(phase: str, *, dispatch_open: bool) -> str | None:
    if not dispatch_open:
        return "execution_test_gate_closed"
    if phase != "sft":
        return "runner_phase_unsupported"
    return None


def _feature_blocked_reason(feature: str, *, dispatch_open: bool) -> str | None:
    if not dispatch_open:
        return "execution_test_gate_closed"
    if feature not in {"sft", "qlora"}:
        return "runner_feature_unsupported"
    return None


def _parallelism_section(
    queue_status: Mapping[str, Any] | None = None,
    *,
    dispatch_open: bool = False,
) -> dict[str, Any]:
    queue_status = _queue_status_defaults(queue_status)
    queue_ready = queue_status["queue_status"] == "ready"
    return {
        "max_concurrent_jobs": 1 if dispatch_open and queue_ready else 0,
        "gpu_slots": 1 if dispatch_open and queue_ready else 0,
        "exclusive_gpu_required": True,
        "inference_conflict_policy": (
            POLICY_DECISION
        ),
        "queue_enabled": queue_status["queue_enabled"] is True,
        "lease_enabled": queue_status["lease_enabled"] is True,
        "resume_persistence_enabled": queue_status["resume_persistence_enabled"] is True,
    }


def _operational_status_section(
    conflict: ConflictProbeResult,
    *,
    active_jobs: int,
    queue_status: Mapping[str, Any] | None = None,
    dispatch_open: bool = False,
) -> dict[str, Any]:
    queue_status = _queue_status_defaults(queue_status)
    status = conflict.status if conflict.status in _CONFLICT_STATUSES else "unknown"
    source_status = (
        conflict.source_status if conflict.source_status in _SOURCE_STATUSES else "unknown"
    )
    maintenance = (
        conflict.maintenance
        if conflict.maintenance in _MAINTENANCE_STATUSES
        else "unknown"
    )
    return {
        "health": "degraded",
        "maintenance_mode": maintenance == "active",
        "maintenance": maintenance,
        "active_jobs": max(0, int(active_jobs)),
        "queue_depth": queue_status["queue_depth"],
        "queue_status": queue_status["queue_status"],
        "queue_degraded_reason": queue_status["queue_degraded_reason"],
        "inference_conflict": {
            "status": status,
            "policy_decision": (
                POLICY_DECISION
            ),
            "source_status": source_status,
            "measurement_warnings": _safe_codes(conflict.measurement_warnings),
        },
        "inference_conflict_state": status,
        "blocking_reasons": [],
    }


def _blocking_reasons(
    snapshot: Mapping[str, Any],
    redaction_reasons: list[str],
    *,
    cfg: object | None = None,
    dispatch_open: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if not dispatch_open:
        reasons.extend(
            [
                SCHEDULING_GATE_REASON,
                "execution_test_gate_closed",
                "parallelism_disabled",
            ]
        )
    hardware = snapshot["hardware"]
    model_limits = snapshot["model_limits"]
    trainer_stack = snapshot["trainer_stack"]
    if hardware["measurement_status"] != "complete":
        reasons.append("resource_unknown")
    if dispatch_open and not _has_schedulable_cuda(hardware["accelerators"]):
        reasons.append("resource_unknown")
    for warning in hardware.get("measurement_warnings", []):
        if warning in {
            "artifact_staging_unavailable",
            "artifact_quota_invalid",
            "artifact_cleanup_failed",
            "artifact_disk_full",
        }:
            reasons.append(warning)
    if model_limits["measurement_status"] != "complete":
        reasons.append("resource_unknown")
    if dispatch_open and (
        model_limits["max_model_parameters_b"] <= 1
        or model_limits["max_context_tokens"] <= 1
        or not model_limits["quantization_modes"]
    ):
        reasons.append("resource_unknown")
    if trainer_stack["measurement_status"] != "complete":
        reasons.append("trainer_stack_unknown")
    elif dispatch_open and not _trainer_stack_ready(trainer_stack):
        reasons.append("trainer_dependency_missing")
    if dispatch_open:
        catalog_blocker = _resource_catalog_blocking_reason(cfg)
        if catalog_blocker is not None:
            reasons.append(catalog_blocker)

    active_jobs = snapshot["operational_status"].get("active_jobs", 0)
    if isinstance(active_jobs, int) and not isinstance(active_jobs, bool) and active_jobs > 0:
        reasons.append("worker_active_job")

    conflict = snapshot["operational_status"]["inference_conflict"]
    status = conflict["status"]
    if dispatch_open and status == "maintenance":
        reasons.append("maintenance")
    elif dispatch_open and status in {"blocked_by_policy", "gpu_marker_active"}:
        reasons.append("gpu_policy_blocked")
    elif dispatch_open and (
        status == "unknown" or conflict["source_status"] != "complete"
    ):
        reasons.append("inference_conflict_unknown")
    elif not dispatch_open and status == "maintenance":
        reasons.append("maintenance")
    elif not dispatch_open and status in {"blocked_by_policy", "gpu_marker_active"}:
        reasons.append("gpu_policy_blocked")
    elif not dispatch_open and (
        status == "unknown" or conflict["source_status"] != "complete"
    ):
        reasons.append("inference_conflict_unknown")
    queue_status = snapshot["operational_status"]["queue_status"]
    queue_degraded_reason = snapshot["operational_status"].get("queue_degraded_reason")
    if queue_status == "full":
        reasons.append("queue_full")
    elif queue_status != "ready":
        reasons.append(
            queue_degraded_reason
            if isinstance(queue_degraded_reason, str)
            and _SAFE_CODE_RE.fullmatch(queue_degraded_reason)
            else "queue_unavailable"
        )
    reasons.extend(redaction_reasons)
    return _unique_sorted(reasons)


def _resource_catalog_blocking_reason(cfg: object | None) -> str | None:
    catalog = _load_resource_catalog(cfg)
    if catalog is None:
        return "resource_catalog_missing"
    data = catalog.data
    for section in ("datasets", "models", "training_profiles"):
        entries = data.get(section)
        if not isinstance(entries, dict) or not entries:
            return "resource_catalog_invalid"
    profile = _sprint12_profile_entry(data)
    if profile is None:
        return "resource_catalog_invalid"
    if profile.get("profile_hash_sha256") != SPRINT12_SFT_PROFILE_HASH:
        return "resource_catalog_invalid"
    output_artifacts = profile.get("output_artifacts")
    if not isinstance(output_artifacts, Mapping):
        return "resource_catalog_invalid"
    if output_artifacts.get("allow_merge") is not True:
        return "resource_catalog_invalid"
    converter = output_artifacts.get("gguf_converter_path")
    if not isinstance(converter, str) or not converter.startswith("/"):
        return "resource_catalog_invalid"
    if not Path(converter).is_file() or not os.access(converter, os.X_OK):
        return "resource_catalog_invalid"
    return None


def _load_resource_catalog(cfg: object | None) -> ResourceCatalog | None:
    if cfg is None:
        return None
    raw_catalog = getattr(cfg, "sft_resource_catalog", None)
    if raw_catalog is None:
        data_dir = getattr(cfg, "data_dir", None)
        if not isinstance(data_dir, (str, Path)):
            return None
        catalog_path = Path(data_dir) / "sft_resources.json"
    elif isinstance(raw_catalog, (str, Path)):
        catalog_path = Path(raw_catalog)
    else:
        return None
    if not catalog_path.is_absolute():
        return None
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != "kiron_sft_resources_v1":
        return None
    return ResourceCatalog(path=catalog_path, data=data)


def _sprint12_profile_entry(catalog: Mapping[str, Any]) -> Mapping[str, Any] | None:
    profiles = catalog.get("training_profiles")
    if not isinstance(profiles, Mapping):
        return None
    entry = profiles.get(SPRINT12_SFT_PROFILE_REF)
    return entry if isinstance(entry, Mapping) else None


def _catalog_model_limits(cfg: object | None) -> dict[str, Any]:
    catalog = _load_resource_catalog(cfg)
    profile = None if catalog is None else _sprint12_profile_entry(catalog.data)
    if profile is None:
        return {
            "max_model_parameters_b": 1,
            "max_context_tokens": 1,
            "quantization_modes": [],
            "max_sequence_length": 1,
            "max_batch_size": 1,
            "gradient_accumulation_steps": None,
            "limit_source": "resource_catalog_missing",
            "measurement_status": "unknown",
        }
    training_args = profile.get("training_args")
    limits = profile.get("model_limits")
    if not isinstance(training_args, Mapping):
        training_args = {}
    if not isinstance(limits, Mapping):
        limits = {}
    quantization_modes = _quantization_modes_from_catalog(limits, training_args)
    max_context = _positive_int_from_catalog(
        limits,
        "max_context_tokens",
        default=_positive_int_from_catalog(
            training_args,
            "max_context_tokens",
            default=_positive_int_from_catalog(
                training_args,
                "max_sequence_length",
                default=_positive_int_from_catalog(training_args, "max_seq_length", default=1),
            ),
        ),
    )
    max_model = _positive_float_from_catalog(limits, "max_model_parameters_b", default=8.0)
    max_batch = _positive_int_from_catalog(
        limits,
        "max_batch_size",
        default=_positive_int_from_catalog(
            training_args,
            "max_batch_size",
            default=_positive_int_from_catalog(
                training_args,
                "per_device_train_batch_size",
                default=1,
            ),
        ),
    )
    grad_accum = _positive_int_from_catalog(
        training_args,
        "gradient_accumulation_steps",
        default=1,
    )
    complete = (
        max_model > 1
        and max_context > 1
        and max_batch > 0
        and bool(quantization_modes)
        and profile.get("profile_hash_sha256") == SPRINT12_SFT_PROFILE_HASH
    )
    return {
        "max_model_parameters_b": max_model,
        "max_context_tokens": max_context,
        "quantization_modes": quantization_modes,
        "max_sequence_length": max_context,
        "max_batch_size": max_batch,
        "gradient_accumulation_steps": grad_accum,
        "limit_source": "resource_catalog",
        "measurement_status": "complete" if complete else "unknown",
    }


def _quantization_modes_from_catalog(
    limits: Mapping[str, Any],
    training_args: Mapping[str, Any],
) -> list[str]:
    raw = limits.get("quantization_modes")
    if isinstance(raw, list) and all(isinstance(item, str) and _SAFE_CODE_RE.fullmatch(item) for item in raw):
        return sorted(set(raw))
    quantization_mode = training_args.get("quantization_mode")
    if isinstance(quantization_mode, str) and _SAFE_CODE_RE.fullmatch(quantization_mode):
        return [quantization_mode]
    return []


def _positive_int_from_catalog(
    source: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    raw = source.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return default
    return raw


def _positive_float_from_catalog(
    source: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    raw = source.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return default
    return float(raw)


def _queue_status_defaults(queue_status: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = {} if queue_status is None else dict(queue_status)
    status = raw.get("queue_status")
    if status not in {"ready", "full", "unavailable"}:
        status = "unavailable"
    reason = raw.get("queue_degraded_reason")
    if reason is not None and not (
        isinstance(reason, str) and _SAFE_CODE_RE.fullmatch(reason)
    ):
        reason = "queue_unavailable"
    depth = raw.get("queue_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        depth = 0
    return {
        "queue_enabled": raw.get("queue_enabled") is True and status != "unavailable",
        "lease_enabled": raw.get("lease_enabled") is True and status != "unavailable",
        "resume_persistence_enabled": raw.get("resume_persistence_enabled") is True
        and status != "unavailable",
        "queue_depth": depth,
        "queue_status": status,
        "queue_degraded_reason": reason,
    }


def _call_probe(func: Callable[[], Any], fallback: Any) -> Any:
    try:
        return func()
    except Exception:
        return fallback


def _unknown_hardware() -> HardwareProbeResult:
    return HardwareProbeResult(
        accelerators=(),
        measurement_status="unknown",
        measurement_warnings=("hardware_probe_failed",),
    )


def _unknown_trainer() -> TrainerProbeResult:
    return TrainerProbeResult(
        python_runtime=platform.python_version(),
        packages=(),
        measurement_status="unknown",
        measurement_warnings=("trainer_probe_failed",),
    )


def _accelerator_entry(
    raw: Mapping[str, Any],
    *,
    dispatch_open: bool = False,
) -> dict[str, Any]:
    kind = raw.get("kind")
    if kind not in _ACCELERATOR_KINDS:
        kind = "unknown"
    status = raw.get("capability_status")
    if status not in _CAPABILITY_STATUSES:
        status = "unknown"
    count = _nonnegative_int_or_zero(raw.get("count"))
    total = _nonnegative_int_or_none(raw.get("vram_total_bytes"))
    available = _nonnegative_int_or_none(raw.get("vram_available_bytes"))
    budget = _nonnegative_int_or_none(raw.get("vram_budget_bytes"))
    if dispatch_open and kind == "cuda" and count > 0:
        if available is None:
            budget = 0 if budget is None else budget
        elif available <= 0:
            budget = 0
            status = "unavailable"
        else:
            if budget in (None, 0) or budget > available:
                budget = available
            status = "available" if budget > 0 else "unavailable"
    result = {
        "kind": kind,
        "count": count,
        "vram_gib": _bytes_to_gib(total),
        "vram_total_bytes": total,
        "vram_available_bytes": available,
        "vram_budget_bytes": budget,
        "capability_status": status,
    }
    model_label = raw.get("model_label")
    if isinstance(model_label, str) and model_label.strip():
        result["model_label"] = model_label.strip()[:120]
    return result


def execution_resource_blocked_reason(probe: HardwareProbeResult) -> str | None:
    """Return the execution resource blocker for a fresh hardware probe."""

    hardware = _hardware_section(probe, dispatch_open=True)
    return _resource_blocked_reason_from_hardware_section(hardware)


def _resource_blocked_reason_from_hardware_section(
    hardware: Mapping[str, Any],
) -> str | None:
    if hardware.get("measurement_status") != "complete":
        return "resource_unknown"
    accelerators = hardware.get("accelerators")
    if not isinstance(accelerators, list) or not _has_schedulable_cuda(accelerators):
        return "resource_unknown"
    return None


def _has_schedulable_cuda(accelerators: list[dict[str, Any]]) -> bool:
    for accelerator in accelerators:
        if (
            accelerator.get("kind") == "cuda"
            and accelerator.get("count", 0) > 0
            and accelerator.get("capability_status") == "available"
            and (accelerator.get("vram_available_bytes") or 0) > 0
            and (accelerator.get("vram_budget_bytes") or 0) > 0
        ):
            return True
    return False


def _trainer_stack_ready(trainer_stack: Mapping[str, Any]) -> bool:
    packages = trainer_stack.get("packages")
    if not isinstance(packages, list):
        return False
    available = {
        package.get("name")
        for package in packages
        if isinstance(package, dict) and package.get("available") is True
    }
    return set(_TRAINER_PACKAGES).issubset(available)


def _package_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    name = raw.get("name")
    if not isinstance(name, str) or not _SAFE_CODE_RE.fullmatch(name.replace("-", "_")):
        name = "unknown"
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        version = "unknown"
    return {
        "name": name,
        "version": version[:80],
        "available": raw.get("available") is True,
    }


def _host_memory_bytes(warnings: list[str]) -> tuple[int | None, int | None]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        warnings.append("host_ram_unknown")
        return None, None
    if (
        not isinstance(pages, int)
        or not isinstance(available_pages, int)
        or not isinstance(page_size, int)
        or pages < 0
        or available_pages < 0
        or page_size <= 0
    ):
        warnings.append("host_ram_unknown")
        return None, None
    return pages * page_size, available_pages * page_size


def _runtime_free_bytes(cfg: object | None, warnings: list[str]) -> int | None:
    if cfg is None:
        warnings.append("artifact_staging_unavailable")
        return None
    try:
        root = artifact_staging.validate_staging_root(cfg)
        return int(root.free_bytes)
    except artifact_staging.ArtifactStagingError as exc:
        warnings.append(exc.reason_code)
        return None
    except (OSError, ValueError, TypeError):
        warnings.append("artifact_staging_unavailable")
        return None


def _cuda_accelerators() -> tuple[list[dict[str, Any]], str, list[str]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=0.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return [], "unknown", ["cuda_probe_unavailable"]
    if proc.returncode != 0:
        return [], "unknown", ["cuda_probe_unavailable"]
    accelerators: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            return [], "unknown", ["cuda_probe_shape_unknown"]
        try:
            total_mib = int(parts[1])
            free_mib = int(parts[2])
        except ValueError:
            return [], "unknown", ["cuda_probe_shape_unknown"]
        if total_mib <= 0 or free_mib < 0:
            return [], "unknown", ["cuda_probe_shape_unknown"]
        accelerators.append(
            {
                "kind": "cuda",
                "count": 1,
                "model_label": parts[0],
                "vram_total_bytes": total_mib * 1024 * 1024,
                "vram_available_bytes": free_mib * 1024 * 1024,
                "vram_budget_bytes": 0,
                "capability_status": "unavailable",
            }
        )
    if not accelerators:
        return [], "unknown", ["cuda_probe_empty"]
    return accelerators, "complete", []


def _measurement_status(value: str) -> str:
    return value if value in _MEASUREMENT_STATUSES else "unknown"


def _combine_measurement_status(statuses: list[str]) -> str:
    valid = [_measurement_status(status) for status in statuses]
    if all(status == "complete" for status in valid):
        return "complete"
    if any(status == "complete" for status in valid):
        return "partial"
    return "unknown"


def _combine_source_status(statuses: list[str]) -> str:
    valid = [status if status in _SOURCE_STATUSES else "unknown" for status in statuses]
    if all(status == "complete" for status in valid):
        return "complete"
    if any(status == "complete" for status in valid):
        return "partial"
    return "unknown"


def _safe_codes(values: tuple[str, ...] | list[str]) -> list[str]:
    if not values:
        return []
    result = []
    for value in values:
        result.append(value if isinstance(value, str) and _SAFE_CODE_RE.fullmatch(value) else "redacted_diagnostic")
    return _unique_sorted(result)


def _unique_sorted(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values))


def _safe_python_runtime(value: str) -> str:
    if not isinstance(value, str):
        return "unknown"
    parts = value.split(".")
    if len(parts) < 3:
        return "unknown"
    cleaned = ".".join(parts[:3])
    if not re.fullmatch(r"[0-9]+[.][0-9]+[.][0-9]+", cleaned):
        return "unknown"
    return cleaned


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_int_or_zero(value: Any) -> int:
    result = _nonnegative_int_or_none(value)
    return 0 if result is None else result


def _bytes_to_gib(value: int | None) -> float:
    if value is None or value <= 0:
        return 0.0
    return round(value / (1024**3), 3)


def _coerce_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _validate_ttl_seconds(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != DEFAULT_TTL_SECONDS:
        return DEFAULT_TTL_SECONDS
    return value


def _scrub_string(value: str) -> str:
    text = _SIGNED_URL_RE.sub("<redacted-url>", value)
    text = _DSN_RE.sub("<redacted-dsn>", text)
    text = _INTERNAL_URL_RE.sub("<redacted-url>", text)
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _SENSITIVE_QUERY_PARAM_RE.sub(r"\1<redacted>", text)
    text = _SENSITIVE_KEY_VALUE_RE.sub(r"\1<redacted>", text)
    text = _LOCAL_PATH_RE.sub("<redacted-path>", text)
    text = _ABSOLUTE_PATH_RE.sub("<redacted-path>", text)
    text = _SECRET_WORD_RE.sub("<redacted>", text)
    return " ".join(text.split())[:256]
