from __future__ import annotations

from datetime import datetime, timezone

import capabilities
import gpu_policy


class FreshMeasurementProbes:
    def __init__(self, conflict, *, vram_available_bytes=10 * 1024 * 1024 * 1024):
        self.conflict = conflict
        self.vram_available_bytes = vram_available_bytes
        self.conflict_calls = 0
        self.hardware_calls = 0

    def measure_hardware(self):
        self.hardware_calls += 1
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "cuda",
                    "count": 1,
                    "vram_total_bytes": 12 * 1024 * 1024 * 1024,
                    "vram_available_bytes": self.vram_available_bytes,
                    "vram_budget_bytes": 0,
                    "capability_status": "unavailable",
                },
            ),
            measurement_status="complete",
        )

    def measure_trainer_stack(self):
        raise AssertionError("execution policy must not use capability snapshots")

    def probe_inference_conflict(self):
        self.conflict_calls += 1
        if isinstance(self.conflict, Exception):
            raise self.conflict
        return self.conflict


def _decision(
    status,
    source_status="complete",
    maintenance="inactive",
    warnings=(),
    *,
    vram_available_bytes=10 * 1024 * 1024 * 1024,
):
    probes = FreshMeasurementProbes(
        capabilities.ConflictProbeResult(
            status=status,
            source_status=source_status,
            maintenance=maintenance,
            measurement_warnings=tuple(warnings),
        ),
        vram_available_bytes=vram_available_bytes,
    )
    now_calls = []
    decision = gpu_policy.decide_execution_start(
        probes=probes,
        now_fn=lambda: now_calls.append(1)
        or datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
    )
    assert probes.conflict_calls == 1
    assert probes.hardware_calls == 1
    assert now_calls == [1]
    return decision


def test_execution_policy_allows_clean_current_measurement():
    decision = _decision("none")

    assert decision.allowed is True
    assert decision.reason_code == "allowed"
    assert decision.conflict_status == "none"
    assert decision.source_status == "complete"
    assert decision.maintenance == "inactive"


def test_execution_policy_allows_ollama_active_with_free_resources():
    decision = _decision("ollama_active")

    assert decision.allowed is True
    assert decision.reason_code == "allowed"
    assert decision.conflict_status == "ollama_active"


def test_execution_policy_blocks_ollama_active_without_free_resources():
    decision = _decision("ollama_active", vram_available_bytes=0)

    assert decision.allowed is False
    assert decision.reason_code == "resource_unknown"


def test_execution_policy_blocks_policy_markers_and_maintenance():
    assert _decision("blocked_by_policy").reason_code == "gpu_policy_blocked"
    assert _decision("gpu_marker_active").reason_code == "gpu_policy_blocked"
    assert _decision("maintenance", maintenance="active").reason_code == "maintenance"


def test_execution_policy_blocks_when_resources_are_not_schedulable():
    decision = _decision("none", vram_available_bytes=0)

    assert decision.allowed is False
    assert decision.reason_code == "resource_unknown"


def test_execution_policy_fails_closed_for_unknown_and_unreadable_sources():
    unknown = _decision("unknown", source_status="complete")
    unreadable = _decision(
        "unknown",
        source_status="unknown",
        maintenance="unknown",
        warnings=("ollama_ps_unreachable",),
    )
    exception = gpu_policy.decide_execution_start(
        probes=FreshMeasurementProbes(RuntimeError("probe failed"))
    )

    assert unknown.allowed is False
    assert unknown.reason_code == "inference_conflict_unknown"
    assert unreadable.allowed is False
    assert unreadable.reason_code == "source_unreadable"
    assert exception.allowed is False
    assert exception.reason_code == "source_unreadable"
