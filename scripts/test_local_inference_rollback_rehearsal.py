"""Temporary restore effects and real offline provider/API integration."""
import asyncio
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from unittest import mock

import pytest


SPEC = importlib.util.spec_from_file_location("rollback_rehearsal", Path(__file__).with_name("rehearse-local-inference-rollback.py"))
rehearsal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rehearsal)


@pytest.mark.skipif(os.geteuid() != 0, reason="real root-owned compat decoder requires root; all writes are temporary")
def test_restore_reestablishes_real_registry_compat_provider_and_api_contract():
    result = rehearsal.rehearse()
    assert result["status"] == "passed"
    assert result["lock_inode_preserved"] and result["backup_and_restored_hashes_equal"]
    assert result["baseline_contract"] == result["restored_contract"]
    assert result["start_audit"] == ["simulated_start_after_complete_validation"]
    assert {case["case"] for case in result["negative_cases"]} == {
        "mixed_release", "interrupted_restore", "wrong_backup_hash",
        "hash_consistent_wrong_registry_schema", "hash_consistent_wrong_image_digest"}
    assert all(case["rejected"] and not case["start_triggered"] for case in result["negative_cases"])
    assert result["api"]["models_status"] == result["api"]["chat_status"] == 200
    assert result["api"]["admission_empty"] and result["api"]["restored_imports"]
    assert result["api"]["backend_paths"].count("/api/chat") == 1
    assert "/api/generate" not in result["api"]["backend_paths"]
    payload, = result["api"]["chat_payloads"]
    assert payload["options"] == {"num_predict": 8, "num_ctx": 1024, "num_gpu": 0}
    assert payload["truncate"] is False and payload["shift"] is False and payload["think"] is False
    assert "keep_alive" not in payload


def test_interruption_is_partial_and_cannot_pass_complete_verification(tmp_path):
    backup, active = tmp_path / "backup", tmp_path / "active"
    backup.mkdir()
    rehearsal.put(backup / "first", b"old first")
    rehearsal.put(backup / "second", b"old second")
    shutil.copytree(backup, active)
    manifest = rehearsal.inventory(backup)
    rehearsal.put(active / "first", b"mixed first")
    rehearsal.put(active / "second", b"mixed second")
    with pytest.raises(InterruptedError):
        rehearsal.restore(backup, active, manifest, interrupt_after=1)
    assert (active / "first").read_bytes() == b"old first"
    assert (active / "second").read_bytes() == b"mixed second"
    with pytest.raises(ValueError):
        rehearsal.verify(active, manifest)
    rehearsal.restore(backup, active, manifest)
    rehearsal.verify(active, manifest)
    assert not list(active.glob(".*.restore"))


@pytest.mark.parametrize("change", ["bytes", "mode", "symlink", "extra"])
def test_invalid_backup_is_rejected_before_active_mutation(tmp_path, change):
    backup, active = tmp_path / "backup", tmp_path / "active"
    backup.mkdir()
    rehearsal.put(backup / "file", b"baseline")
    shutil.copytree(backup, active)
    manifest = rehearsal.inventory(backup)
    if change == "bytes":
        (backup / "file").write_bytes(b"damaged")
    elif change == "mode":
        (backup / "file").chmod(0o666)
    elif change == "symlink":
        (backup / "file").unlink()
        (backup / "file").symlink_to(active / "file")
    else:
        rehearsal.put(backup / "extra", b"unexpected")
    before = rehearsal.inventory(active)
    with pytest.raises(ValueError):
        rehearsal.restore(backup, active, manifest)
    assert rehearsal.inventory(active) == before


def test_child_refuses_non_rehearsal_paths_before_import_or_io(tmp_path):
    with pytest.raises(ValueError, match="private temporary"):
        asyncio.run(rehearsal.probe(tmp_path, True))


def test_api_failure_after_file_and_decode_checks_never_emits_start(tmp_path):
    starts, baseline = [], {"registry_id": "validated fixture"}
    with mock.patch.object(rehearsal, "gate", return_value=baseline), \
            mock.patch.object(rehearsal, "child", side_effect=ValueError("API smoke failed")):
        with pytest.raises(ValueError, match="API smoke failed"):
            rehearsal.validated_start(tmp_path, {}, starts, baseline)
    assert starts == []


@pytest.mark.skipif(os.geteuid() != 0, reason="root-owned temporary source contract")
@pytest.mark.parametrize("fault", ["private_mode", "foreign_owner", "file_mode", "directory_mode", "symlink", "hardlink", "active_file"])
def test_child_rejects_unsafe_private_tree_before_source_import(fault):
    with tempfile.TemporaryDirectory(prefix=rehearsal.PREFIX, dir=rehearsal.TEMP_ROOT) as temporary:
        root = Path(temporary)
        active = root / "active"
        active.mkdir(mode=0o750)
        path = active / "not_importable.py"
        rehearsal.put(path, b"raise AssertionError('must not import')\n")
        if fault == "private_mode":
            root.chmod(0o750)
        elif fault == "foreign_owner":
            os.chown(path, 65534, os.getgid())
        elif fault == "file_mode":
            path.chmod(0o660)
        elif fault == "directory_mode":
            active.chmod(0o770)
        elif fault == "symlink":
            path.unlink()
            path.symlink_to("missing")
        elif fault == "hardlink":
            os.link(path, active / "second")
        else:
            path.unlink()
            active.rmdir()
            rehearsal.put(active, b"not a directory")
        with pytest.raises(ValueError):
            asyncio.run(rehearsal.probe(active, True))
