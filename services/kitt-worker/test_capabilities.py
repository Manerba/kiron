from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

import pytest

import capabilities
import config


FIXED_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _join(*parts: str) -> str:
    return "".join(parts)


class KnownProbes:
    def __init__(self, conflict: capabilities.ConflictProbeResult | None = None) -> None:
        self._conflict = conflict or capabilities.ConflictProbeResult(
            status="none",
            source_status="complete",
            maintenance="inactive",
        )

    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "cuda",
                    "count": 1,
                    "vram_total_bytes": 12 * 1024 * 1024 * 1024,
                    "vram_available_bytes": 10 * 1024 * 1024 * 1024,
                    "vram_budget_bytes": 0,
                    "capability_status": "unavailable",
                },
            ),
            host_ram_total_bytes=64 * 1024 * 1024 * 1024,
            host_ram_available_bytes=48 * 1024 * 1024 * 1024,
            staging_free_bytes=100 * 1024 * 1024 * 1024,
            measurement_status="complete",
        )

    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(
                *(
                    {
                        "name": name,
                        "version": "2.7.0",
                        "available": True,
                    }
                    for name in (
                        "torch",
                        "transformers",
                        "datasets",
                        "peft",
                        "trl",
                        "accelerate",
                        "bitsandbytes",
                    )
                ),
            ),
            measurement_status="complete",
        )

    def probe_inference_conflict(self) -> capabilities.ConflictProbeResult:
        return self._conflict


class UnknownProbes(KnownProbes):
    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(),
            measurement_status="unknown",
            measurement_warnings=("resource_probe_unknown",),
        )

    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(),
            measurement_status="unknown",
            measurement_warnings=("trainer_stack_unknown",),
        )


class ZeroFreeCudaProbes(KnownProbes):
    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "cuda",
                    "count": 1,
                    "vram_total_bytes": 12 * 1024 * 1024 * 1024,
                    "vram_available_bytes": 0,
                    "vram_budget_bytes": 12 * 1024 * 1024 * 1024,
                    "capability_status": "available",
                },
            ),
            host_ram_total_bytes=64 * 1024 * 1024 * 1024,
            host_ram_available_bytes=48 * 1024 * 1024 * 1024,
            staging_free_bytes=100 * 1024 * 1024 * 1024,
            measurement_status="complete",
        )


class MissingTrainerPackageProbes(KnownProbes):
    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(
                {
                    "name": "torch",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "transformers",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "datasets",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "peft",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "trl",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "accelerate",
                    "version": "unknown",
                    "available": False,
                },
                {
                    "name": "bitsandbytes",
                    "version": "unknown",
                    "available": False,
                },
            ),
            measurement_status="complete",
        )


READY_QUEUE = {
    "queue_enabled": True,
    "lease_enabled": True,
    "resume_persistence_enabled": True,
    "queue_depth": 0,
    "queue_status": "ready",
    "queue_degraded_reason": None,
}


def _write_sft_trainer(tmp_path: Path) -> Path:
    return Path(__file__).resolve().with_name("sft_trainer.py")


def _write_resource_catalog(tmp_path: Path) -> Path:
    path = tmp_path / "sft_resources.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "kiron_sft_resources_v1",
                "datasets": {"dataset-1": {"format": "jsonl"}},
                "models": {"Qwen/Qwen2.5-7B-Instruct": {"revision": "test"}},
                "training_profiles": {
                    capabilities.SPRINT12_SFT_PROFILE_REF: {
                        "profile_hash_sha256": capabilities.SPRINT12_SFT_PROFILE_HASH,
                        "model_limits": {
                            "max_model_parameters_b": 8,
                            "max_context_tokens": 4096,
                            "quantization_modes": ["qlora_4bit"],
                            "max_batch_size": 1,
                        },
                        "training_args": {
                            "max_seq_length": 4096,
                            "per_device_train_batch_size": 1,
                            "gradient_accumulation_steps": 1,
                            "quantization_mode": "qlora_4bit",
                        },
                        "output_artifacts": {
                            "allow_merge": True,
                            "gguf_converter_path": sys.executable,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _open_cfg(tmp_path: Path):
    _write_resource_catalog(tmp_path)
    return SimpleNamespace(
        data_dir=tmp_path,
        dispatch_gate="open",
        executor_enabled=True,
        executor_runner="sft_subprocess",
        sft_command=f"{sys.executable} {_write_sft_trainer(tmp_path)}",
        executor_config_error_code=None,
        artifact_config_error_code=None,
    )


def _snapshot(probes=None, now=FIXED_NOW, active_jobs=0, queue_status=None, cfg=None):
    return capabilities.build_capability_snapshot(
        cfg=cfg,
        probes=KnownProbes() if probes is None else probes,
        now_fn=lambda: now,
        active_jobs=active_jobs,
        queue_status=READY_QUEUE if queue_status is None else queue_status,
    )


def test_snapshot_has_deterministic_ttl_sections_and_hash():
    snapshot = _snapshot(active_jobs=2)
    later = _snapshot(
        now=datetime(2026, 6, 15, 12, 5, 0, tzinfo=timezone.utc),
        active_jobs=2,
    )

    assert snapshot["schema_version"] == "kitt_worker_capabilities_v2"
    assert snapshot["snapshot_version"] == 1
    assert snapshot["snapshot_time"] == "2026-06-15T12:00:00.000Z"
    assert snapshot["valid_until"] == "2026-06-15T12:01:00.000Z"
    assert snapshot["ttl_seconds"] == 60
    assert snapshot["capability_mode"] == "policy_snapshot"
    assert len(snapshot["capability_hash_sha256"]) == 64
    assert snapshot["capability_hash_sha256"] == later["capability_hash_sha256"]
    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["scheduling_gate"]["reason_code"] == "tareas_dispatch_gate_closed"
    for section in (
        "hardware",
        "operations",
        "model_limits",
        "trainer_stack",
        "parallelism",
        "operational_status",
    ):
        assert section in snapshot
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert snapshot["parallelism"]["queue_enabled"] is True
    assert snapshot["parallelism"]["lease_enabled"] is True
    assert snapshot["parallelism"]["resume_persistence_enabled"] is True
    assert snapshot["operational_status"]["queue_status"] == "ready"
    assert snapshot["operational_status"]["queue_depth"] == 0
    assert snapshot["operational_status"]["queue_degraded_reason"] is None
    assert snapshot["operational_status"]["active_jobs"] == 2
    assert "fake_runner" not in json.dumps(snapshot, sort_keys=True)
    assert snapshot["operations"]["training"] == []
    assert snapshot["operations"]["features"] == []
    assert snapshot["operations"]["artifact_staging_supported"] is True
    assert snapshot["operations"]["publish_supported"] is False
    assert snapshot["operations"]["publish"] == {
        "supported": True,
        "ready": False,
        "execution_enabled": False,
        "job_type": "ollama_publish",
        "runner_kind": "ollama_publish",
        "contract_version": "kitt_publish_job_spec_v1",
        "requires_signed_spec": True,
        "requires_operator_approval": True,
        "approval_plane": "kitt",
        "side_effect": "ollama_create_local_tag",
        "blocked_reason": "execution_test_gate_closed",
        "rollback": "operator_tag_remove_only",
    }
    for phase in snapshot["operations"]["training_phases"]:
        assert phase["execution_enabled"] is False
        assert phase["blocked_reason"] == "execution_test_gate_closed"


def test_operator_open_gate_reports_schedulable_sft_executor_capacity(tmp_path):
    snapshot = _snapshot(cfg=_open_cfg(tmp_path))

    assert snapshot["valid_for_scheduling"] is True
    assert snapshot["execution_enabled"] is True
    assert snapshot["scheduling_gate"]["status"] == "open"
    assert snapshot["scheduling_gate"]["reason_code"] == "operator_dispatch_open"
    assert snapshot["scheduling_gate"]["policy_decision"] == "block_training"
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 1
    assert snapshot["parallelism"]["gpu_slots"] == 1
    assert snapshot["parallelism"]["inference_conflict_policy"] == "block_training"
    assert snapshot["hardware"]["accelerators"][0]["capability_status"] == "available"
    assert snapshot["hardware"]["accelerators"][0]["vram_gib"] == 12.0
    assert snapshot["hardware"]["accelerators"][0]["vram_budget_bytes"] == 10 * 1024 * 1024 * 1024
    assert snapshot["model_limits"]["max_model_parameters_b"] == 8.0
    assert snapshot["model_limits"]["max_context_tokens"] == 4096
    assert snapshot["model_limits"]["quantization_modes"] == ["qlora_4bit"]
    assert snapshot["operations"]["training"] == ["sft"]
    assert snapshot["operations"]["publish"]["supported"] is True
    assert snapshot["operations"]["publish"]["ready"] is False
    assert snapshot["operations"]["publish"]["execution_enabled"] is False
    assert (
        snapshot["operations"]["publish"]["blocked_reason"]
        == "publish_keyring_unavailable"
    )
    assert snapshot["operations"]["features"] == ["sft", "qlora"]
    assert snapshot["trainer_stack"]["features"] == ["sft", "qlora"]
    assert snapshot["operational_status"]["health"] == "healthy"
    assert snapshot["operational_status"]["blocking_reasons"] == []
    assert (
        snapshot["operational_status"]["inference_conflict"]["policy_decision"]
        == "block_training"
    )


def test_operator_open_gate_reports_publish_ready_only_with_keyring(tmp_path):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(cfg=cfg)

    assert snapshot["valid_for_scheduling"] is True
    assert snapshot["execution_enabled"] is True
    assert snapshot["operations"]["publish"]["ready"] is True
    assert snapshot["operations"]["publish"]["execution_enabled"] is True
    assert snapshot["operations"]["publish"]["blocked_reason"] is None
    phases = {phase["phase"]: phase for phase in snapshot["operations"]["training_phases"]}
    assert phases["sft"]["execution_enabled"] is True
    assert phases["sft"]["blocked_reason"] is None
    assert phases["dpo"]["execution_enabled"] is False
    assert phases["dpo"]["blocked_reason"] == "runner_phase_unsupported"
    assert phases["cp"]["execution_enabled"] is False
    assert phases["cp"]["blocked_reason"] == "runner_phase_unsupported"
    features = {
        feature["name"]: feature for feature in snapshot["trainer_stack"]["feature_status"]
    }
    assert features["sft"]["available"] is True
    assert features["sft"]["blocked_reason"] is None
    assert features["qlora"]["available"] is True
    assert features["qlora"]["blocked_reason"] is None
    assert features["dpo"]["available"] is False
    assert features["dpo"]["blocked_reason"] == "runner_feature_unsupported"
    assert features["cp"]["available"] is False
    assert features["cp"]["blocked_reason"] == "runner_feature_unsupported"
    assert features["checkpoint_resume"]["available"] is False
    assert (
        features["checkpoint_resume"]["blocked_reason"]
        == "runner_feature_unsupported"
    )


def test_publish_ready_rechecks_current_key_time_window(tmp_path):
    keyring_path = tmp_path / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": "testkey01",
                    "public_key_hex": "11" * 32,
                    "active": True,
                    "not_after": "2026-06-15T12:30:00Z",
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keys_file = keyring_path
    cfg.publish_verify_keyring_config_error_code = None

    before = _snapshot(cfg=cfg, now=FIXED_NOW)
    after = _snapshot(
        cfg=cfg,
        now=datetime(2026, 6, 15, 12, 30, 0, tzinfo=timezone.utc),
    )

    assert before["operations"]["publish"]["ready"] is True
    assert before["operations"]["publish"]["blocked_reason"] is None
    assert after["operations"]["publish"]["ready"] is False
    assert (
        after["operations"]["publish"]["blocked_reason"]
        == "publish_spec_signature_key_expired"
    )


def test_publish_ready_does_not_depend_on_sft_command(tmp_path):
    cfg = _open_cfg(tmp_path)
    cfg.sft_command = ""
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(cfg=cfg)

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert "execution_test_gate_closed" in snapshot["operational_status"][
        "blocking_reasons"
    ]
    assert snapshot["operations"]["training"] == []
    assert snapshot["operations"]["publish"]["ready"] is True
    assert snapshot["operations"]["publish"]["execution_enabled"] is True
    assert snapshot["operations"]["publish"]["blocked_reason"] is None


@pytest.mark.parametrize(
    ("probes", "reason"),
    [
        (ZeroFreeCudaProbes(), "resource_unknown"),
        (MissingTrainerPackageProbes(), "trainer_dependency_missing"),
    ],
)
def test_publish_ready_ignores_sft_runtime_blockers(tmp_path, probes, reason):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(cfg=cfg, probes=probes)

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert reason in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operations"]["training"] == []
    assert snapshot["operations"]["publish"]["ready"] is True
    assert snapshot["operations"]["publish"]["execution_enabled"] is True
    assert snapshot["operations"]["publish"]["blocked_reason"] is None


@pytest.mark.parametrize(
    ("queue_status", "queue_reason", "expected_reason"),
    [
        ("full", None, "queue_full"),
        ("unavailable", "sqlite_unavailable", "sqlite_unavailable"),
    ],
)
def test_publish_ready_blocks_queue_runtime_blockers(
    tmp_path,
    queue_status,
    queue_reason,
    expected_reason,
):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(
        cfg=cfg,
        queue_status={
            **READY_QUEUE,
            "queue_status": queue_status,
            "queue_degraded_reason": queue_reason,
        },
    )

    assert expected_reason in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operations"]["publish"]["ready"] is False
    assert snapshot["operations"]["publish"]["execution_enabled"] is False
    assert snapshot["operations"]["publish"]["blocked_reason"] == expected_reason


@pytest.mark.parametrize(
    ("status", "maintenance", "reason"),
    [
        ("maintenance", "active", "maintenance"),
        ("blocked_by_policy", "inactive", "gpu_policy_blocked"),
        ("unknown", "unknown", "inference_conflict_unknown"),
    ],
)
def test_publish_ready_blocks_maintenance_and_policy_conflicts(
    tmp_path,
    status,
    maintenance,
    reason,
):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(
        cfg=cfg,
        probes=KnownProbes(
            capabilities.ConflictProbeResult(
                status=status,
                source_status="complete" if status != "unknown" else "unknown",
                maintenance=maintenance,
            )
        ),
    )

    assert reason in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operations"]["publish"]["ready"] is False
    assert snapshot["operations"]["publish"]["execution_enabled"] is False
    assert snapshot["operations"]["publish"]["blocked_reason"] == reason


def test_publish_ready_allows_ollama_active_with_schedulable_resources(tmp_path):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(
        cfg=cfg,
        probes=KnownProbes(
            capabilities.ConflictProbeResult(
                status="ollama_active",
                source_status="complete",
                maintenance="inactive",
            )
        ),
    )

    assert snapshot["valid_for_scheduling"] is True
    assert snapshot["execution_enabled"] is True
    assert snapshot["operational_status"]["inference_conflict"]["status"] == "ollama_active"
    assert snapshot["operational_status"]["blocking_reasons"] == []
    assert snapshot["operations"]["publish"]["ready"] is True
    assert snapshot["operations"]["publish"]["execution_enabled"] is True
    assert snapshot["operations"]["publish"]["blocked_reason"] is None


def test_publish_ready_blocks_ollama_active_without_schedulable_resources(tmp_path):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(
        cfg=cfg,
        probes=ZeroFreeCudaProbes(
            capabilities.ConflictProbeResult(
                status="ollama_active",
                source_status="complete",
                maintenance="inactive",
            )
        ),
    )

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["operational_status"]["inference_conflict"]["status"] == "ollama_active"
    assert "resource_unknown" in snapshot["operational_status"]["blocking_reasons"]
    assert "ollama_active" not in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operations"]["publish"]["ready"] is False
    assert snapshot["operations"]["publish"]["execution_enabled"] is False
    assert snapshot["operations"]["publish"]["blocked_reason"] == "resource_unknown"


def test_publish_ready_blocks_when_worker_job_is_active(tmp_path):
    cfg = _open_cfg(tmp_path)
    cfg.publish_verify_keyring_config_error_code = None
    snapshot = _snapshot(cfg=cfg, active_jobs=1)

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert "worker_active_job" in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operations"]["publish"]["ready"] is False
    assert snapshot["operations"]["publish"]["execution_enabled"] is False
    assert snapshot["operations"]["publish"]["blocked_reason"] == "worker_active_job"


def test_operator_open_gate_without_resource_catalog_fails_closed(tmp_path):
    cfg = _open_cfg(tmp_path)
    (tmp_path / "sft_resources.json").unlink()

    snapshot = _snapshot(cfg=cfg)
    blockers = set(snapshot["operational_status"]["blocking_reasons"])

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["scheduling_gate"]["status"] == "closed"
    assert snapshot["scheduling_gate"]["reason_code"] == "resource_catalog_missing"
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert "resource_catalog_missing" in blockers


def test_operator_open_gate_with_zero_free_cuda_vram_fails_closed(tmp_path):
    snapshot = _snapshot(cfg=_open_cfg(tmp_path), probes=ZeroFreeCudaProbes())
    blockers = set(snapshot["operational_status"]["blocking_reasons"])

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["scheduling_gate"]["status"] == "closed"
    assert snapshot["scheduling_gate"]["reason_code"] == "resource_unknown"
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert snapshot["hardware"]["accelerators"][0]["capability_status"] == "unavailable"
    assert snapshot["hardware"]["accelerators"][0]["vram_available_bytes"] == 0
    assert snapshot["hardware"]["accelerators"][0]["vram_budget_bytes"] == 0
    assert "resource_unknown" in blockers
    phases = {phase["phase"]: phase for phase in snapshot["operations"]["training_phases"]}
    assert phases["sft"]["execution_enabled"] is False
    assert phases["sft"]["blocked_reason"] == "resource_unknown"
    features = {
        feature["name"]: feature for feature in snapshot["trainer_stack"]["feature_status"]
    }
    assert features["sft"]["available"] is False
    assert features["sft"]["blocked_reason"] == "resource_unknown"


def test_operator_open_gate_with_missing_trainer_packages_fails_closed(tmp_path):
    snapshot = _snapshot(cfg=_open_cfg(tmp_path), probes=MissingTrainerPackageProbes())
    blockers = set(snapshot["operational_status"]["blocking_reasons"])

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["scheduling_gate"]["status"] == "closed"
    assert snapshot["scheduling_gate"]["reason_code"] == "trainer_dependency_missing"
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert "trainer_dependency_missing" in blockers
    features = {
        feature["name"]: feature for feature in snapshot["trainer_stack"]["feature_status"]
    }
    assert features["sft"]["available"] is False
    assert features["sft"]["blocked_reason"] == "trainer_dependency_missing"


def test_unknown_measurements_fail_closed_without_positive_capacity():
    snapshot = _snapshot(
        probes=UnknownProbes(
            capabilities.ConflictProbeResult(
                status="unknown",
                source_status="unknown",
                maintenance="unknown",
            )
        )
    )
    blockers = set(snapshot["operational_status"]["blocking_reasons"])

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["hardware"]["measurement_status"] == "unknown"
    assert snapshot["hardware"]["accelerators"] == []
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert snapshot["parallelism"]["queue_enabled"] is True
    assert "resource_unknown" in blockers
    assert "trainer_stack_unknown" in blockers
    assert "inference_conflict_unknown" in blockers


def test_queue_unavailable_degrades_capabilities_without_positive_capacity():
    snapshot = _snapshot(
        queue_status={
            "queue_enabled": False,
            "lease_enabled": False,
            "resume_persistence_enabled": False,
            "queue_depth": 0,
            "queue_status": "unavailable",
            "queue_degraded_reason": "db_corrupt",
        }
    )

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert snapshot["parallelism"]["queue_enabled"] is False
    assert snapshot["parallelism"]["lease_enabled"] is False
    assert snapshot["parallelism"]["resume_persistence_enabled"] is False
    assert snapshot["operational_status"]["queue_status"] == "unavailable"
    assert snapshot["operational_status"]["queue_degraded_reason"] == "db_corrupt"
    assert "db_corrupt" in snapshot["operational_status"]["blocking_reasons"]


def test_queue_full_reports_degraded_reason_without_dispatch_capacity():
    snapshot = _snapshot(
        queue_status={
            "queue_enabled": True,
            "lease_enabled": True,
            "resume_persistence_enabled": True,
            "queue_depth": 1000,
            "queue_status": "full",
            "queue_degraded_reason": "queue_full",
        }
    )

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert snapshot["parallelism"]["max_concurrent_jobs"] == 0
    assert snapshot["parallelism"]["gpu_slots"] == 0
    assert snapshot["parallelism"]["queue_enabled"] is True
    assert snapshot["operational_status"]["queue_status"] == "full"
    assert snapshot["operational_status"]["queue_degraded_reason"] == "queue_full"
    assert "queue_full" in snapshot["operational_status"]["blocking_reasons"]


def test_default_hardware_probe_uses_validated_staging_root_without_data_fallback(tmp_path):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
        },
        data_root=data_dir,
        run_root=run_dir,
    )

    result = capabilities.DefaultCapabilityProbes(cfg).measure_hardware()

    assert result.staging_free_bytes is None
    assert "artifact_staging_unavailable" in result.measurement_warnings


def test_valid_staging_root_reports_free_bytes_but_foundation_only_stays_unsupported(tmp_path):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    staging_dir = data_dir / "staging"
    data_dir.mkdir()
    run_dir.mkdir()
    staging_dir.mkdir()
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    staging_dir.chmod(0o750)
    cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging_dir),
            "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "0",
        },
        data_root=data_dir,
        run_root=run_dir,
    )
    probe = capabilities.DefaultCapabilityProbes(cfg).measure_hardware()
    snapshot = capabilities.build_capability_snapshot(
        cfg=cfg,
        probes=KnownProbes(),
        queue_status=READY_QUEUE,
        now_fn=lambda: FIXED_NOW,
    )

    assert os.path.exists(staging_dir)
    assert isinstance(probe.staging_free_bytes, int)
    assert snapshot["operations"]["artifact_staging_supported"] is True
    assert "tareas_dispatch_gate_closed" in snapshot["operational_status"]["blocking_reasons"]


def test_redaction_removes_nested_paths_and_secret_values():
    bad_path = _join("/", "opt", "/", "kiron", "/", "private")
    bad_credential = _join("Bea", "rer", " ", "secret-value")

    class DirtyProbes(KnownProbes):
        def measure_hardware(self) -> capabilities.HardwareProbeResult:
            result = super().measure_hardware()
            return capabilities.HardwareProbeResult(
                accelerators=result.accelerators,
                host_ram_total_bytes=result.host_ram_total_bytes,
                host_ram_available_bytes=result.host_ram_available_bytes,
                staging_free_bytes=result.staging_free_bytes,
                measurement_status=result.measurement_status,
                measurement_warnings=(bad_path,),
            )

        def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
            return capabilities.TrainerProbeResult(
                python_runtime="3.12.0",
                packages=(
                    {
                        "name": "torch",
                        "version": f"2.7.0 {bad_path} {bad_credential}",
                        "available": True,
                    },
                ),
                measurement_status="complete",
            )

    snapshot = _snapshot(probes=DirtyProbes())
    text = json.dumps(snapshot, sort_keys=True)

    assert bad_path not in text
    assert bad_credential not in text
    assert "secret-value" not in text
    assert "redacted_diagnostic" in snapshot["hardware"]["measurement_warnings"]
    assert "capability_value_redacted" in snapshot["operational_status"]["blocking_reasons"]


def test_redaction_removes_generic_absolute_paths_and_key_ids():
    tmp_path_value = "/" + "tmp" + "/kiron-worker/probe.txt"
    mount_path_value = "/" + "mnt" + "/gpu-cache/item.bin"
    kid_value = "kid" + "=" + "kid-value-123"
    key_id_value = "key_" + "id" + "=" + "keyid-value-456"

    class DirtyProbes(KnownProbes):
        def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
            return capabilities.TrainerProbeResult(
                python_runtime="3.12.0",
                packages=(
                    {
                        "name": "torch",
                        "version": (
                            f"2.7.0 {tmp_path_value} {mount_path_value} "
                            f"{kid_value} {key_id_value}"
                        ),
                        "available": True,
                    },
                ),
                measurement_status="complete",
            )

    snapshot = _snapshot(probes=DirtyProbes())
    text = json.dumps(snapshot, sort_keys=True)

    assert tmp_path_value not in text
    assert mount_path_value not in text
    assert "kid-value-123" not in text
    assert "keyid-value-456" not in text
    assert "capability_value_redacted" in snapshot["operational_status"]["blocking_reasons"]


def test_redaction_removes_internal_urls_with_ports_and_sensitive_query():
    local_url = _join("http", "://", "127.0.0.1", ":11435", "/", "api", "/", "ps")
    localhost_url = _join("http", "://", "localhost", ":11441", "/", "internal", "/", "health")
    lan_url = _join("https", "://", "localhost", ":8505", "/", "path", "?to", "ken=raw-token")

    class DirtyProbes(KnownProbes):
        def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
            return capabilities.TrainerProbeResult(
                python_runtime="3.12.0",
                packages=(
                    {
                        "name": "torch",
                        "version": f"2.7.0 {local_url} {localhost_url} {lan_url}",
                        "available": True,
                    },
                ),
                measurement_status="complete",
            )

    snapshot = _snapshot(probes=DirtyProbes())
    text = json.dumps(snapshot, sort_keys=True)

    for forbidden in (
        "127.0.0.1",
        "localhost",
        "localhost",
        "11435",
        "11441",
        "8505",
        "raw-token",
    ):
        assert forbidden not in text
    assert "capability_value_redacted" in snapshot["operational_status"]["blocking_reasons"]


def test_redaction_removes_dsns_and_signed_urls():
    dsn = _join("post", "gresql", "://", "user:pass", "@db.internal", "/", "kitt")
    dsn_without_userinfo = _join("my", "sql", "://", "db.internal", "/", "kitt")
    file_url = _join("file", "://", "/", "tmp", "/", "kiron-worker", "/", "secret.txt")
    signed_url = _join(
        "https",
        "://",
        "example.invalid",
        "/object?sig",
        "nature=raw-signature&expires=1",
    )

    class DirtyProbes(KnownProbes):
        def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
            return capabilities.TrainerProbeResult(
                python_runtime="3.12.0",
                packages=(
                    {
                        "name": "torch",
                        "version": f"2.7.0 {dsn} {dsn_without_userinfo} {file_url} {signed_url}",
                        "available": True,
                    },
                ),
                measurement_status="complete",
            )

    snapshot = _snapshot(probes=DirtyProbes())
    text = json.dumps(snapshot, sort_keys=True)

    for forbidden in (
        _join("post", "gresql", "://"),
        _join("my", "sql", "://"),
        _join("file", "://"),
        "user:pass",
        "db.internal",
        "example.invalid",
        "raw-signature",
        "secret.txt",
    ):
        assert forbidden not in text
    assert "capability_value_redacted" in snapshot["operational_status"]["blocking_reasons"]


@pytest.mark.parametrize(
    ("status", "maintenance", "reason"),
    [
        ("maintenance", "active", "maintenance"),
        ("blocked_by_policy", "inactive", "gpu_policy_blocked"),
        ("gpu_marker_active", "inactive", "gpu_policy_blocked"),
        ("unknown", "unknown", "inference_conflict_unknown"),
    ],
)
def test_conflict_statuses_block_training(tmp_path, status, maintenance, reason):
    snapshot = _snapshot(
        cfg=_open_cfg(tmp_path),
        probes=KnownProbes(
            capabilities.ConflictProbeResult(
                status=status,
                source_status="complete" if status != "unknown" else "unknown",
                maintenance=maintenance,
            )
        )
    )
    conflict = snapshot["operational_status"]["inference_conflict"]

    assert conflict["status"] == status
    assert conflict["policy_decision"] == "block_training"
    assert reason in snapshot["operational_status"]["blocking_reasons"]


def test_ollama_active_does_not_block_training_with_schedulable_resources(tmp_path):
    snapshot = _snapshot(
        cfg=_open_cfg(tmp_path),
        probes=KnownProbes(
            capabilities.ConflictProbeResult(
                status="ollama_active",
                source_status="complete",
                maintenance="inactive",
            )
        ),
    )
    conflict = snapshot["operational_status"]["inference_conflict"]

    assert snapshot["valid_for_scheduling"] is True
    assert snapshot["execution_enabled"] is True
    assert conflict["status"] == "ollama_active"
    assert conflict["policy_decision"] == "block_training"
    assert "ollama_active" not in snapshot["operational_status"]["blocking_reasons"]
    assert snapshot["operational_status"]["blocking_reasons"] == []


def test_ollama_active_blocks_training_without_schedulable_resources(tmp_path):
    snapshot = _snapshot(
        cfg=_open_cfg(tmp_path),
        probes=ZeroFreeCudaProbes(
            capabilities.ConflictProbeResult(
                status="ollama_active",
                source_status="complete",
                maintenance="inactive",
            )
        ),
    )
    conflict = snapshot["operational_status"]["inference_conflict"]

    assert snapshot["valid_for_scheduling"] is False
    assert snapshot["execution_enabled"] is False
    assert conflict["status"] == "ollama_active"
    assert "resource_unknown" in snapshot["operational_status"]["blocking_reasons"]
    assert "ollama_active" not in snapshot["operational_status"]["blocking_reasons"]


class FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self, _limit: int) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _probe(tmp_path, *, maintenance_active=False, marker_payload=None, ollama_payload=None):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    marker = marker_dir / "gpu.json"
    if marker_payload is not None:
        marker.write_text(json.dumps(marker_payload), encoding="utf-8")
    maintenance = tmp_path / "maintenance.json"
    maintenance.write_text(json.dumps({"active": maintenance_active}), encoding="utf-8")

    def fake_urlopen(_url, timeout):
        assert timeout == 0.25
        if ollama_payload is None:
            raise OSError("unreachable")
        return FakeResponse(ollama_payload)

    return capabilities.InferenceConflictProbe(
        marker_paths={"gpu_service_loading": marker},
        maintenance_path=maintenance,
        urlopen=fake_urlopen,
        wall_time_fn=lambda: 100.0,
    )


def test_inference_probe_reports_maintenance_before_other_sources(tmp_path):
    result = _probe(
        tmp_path,
        maintenance_active=True,
        ollama_payload={"models": [{"size_vram": 0}]},
    ).probe()

    assert result.status == "maintenance"
    assert result.maintenance == "active"


def test_inference_probe_reports_active_gpu_marker(tmp_path):
    result = _probe(
        tmp_path,
        marker_payload={"created_wall": 95.0, "ttl_s": 10.0},
        ollama_payload={"models": [{"size_vram": 0}]},
    ).probe()

    assert result.status == "gpu_marker_active"
    assert result.source_status == "complete"


def test_inference_probe_reports_ollama_active_for_positive_vram(tmp_path):
    result = _probe(
        tmp_path,
        ollama_payload={"models": [{"size_vram": 1}]},
    ).probe()

    assert result.status == "ollama_active"
    assert result.source_status == "complete"


@pytest.mark.parametrize(
    "ollama_payload",
    [
        {"models": [{"size_vram": True}]},
        {"models": [{"missing": 0}]},
        {"models": ["bad"]},
        {"not_models": []},
    ],
)
def test_inference_probe_unknown_shapes_fail_closed(tmp_path, ollama_payload):
    result = _probe(tmp_path, ollama_payload=ollama_payload).probe()

    assert result.status == "unknown"
    assert "ollama_ps_shape_unknown" in result.measurement_warnings


def test_inference_probe_unreachable_source_fails_closed(tmp_path):
    result = _probe(tmp_path, ollama_payload=None).probe()

    assert result.status == "unknown"
    assert "ollama_ps_unreachable" in result.measurement_warnings


def test_inference_probe_unknown_marker_shape_fails_closed(tmp_path):
    result = _probe(
        tmp_path,
        marker_payload={"created_wall": "bad", "ttl_s": 10.0},
        ollama_payload={"models": [{"size_vram": 0}]},
    ).probe()

    assert result.status == "unknown"
    assert "gpu_marker_shape_unknown" in result.measurement_warnings
