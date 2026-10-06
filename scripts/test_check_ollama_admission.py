"""Deploy rejects incompatible state without rewriting or deleting tickets."""
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest

path = Path(__file__).with_name("check-ollama-admission.py")
spec = importlib.util.spec_from_file_location("check_ollama_admission", path)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity


@pytest.fixture
def store(tmp_path, monkeypatch):
    tmp_path.chmod(0o2770)
    store = AdmissionStore(tmp_path, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
    monkeypatch.setattr(preflight, "AdmissionStore", lambda: store)
    return store


def test_empty_install_does_not_create_state(store):
    assert preflight.main() == 0
    assert not list(store.root.iterdir())


@pytest.mark.parametrize("schema", [2, 3])
def test_contract_is_checked_without_changing_active_reservations(store, schema, capsys):
    store.reserve(operation_id="work", owner="worker", generation="g", deployment_id="job", kind="request",
                  gpu_bytes=0, host_bytes=0, measure=lambda: MemorySnapshot(1, 1, time.monotonic()))
    path = store.root / "admission.json"
    value = json.loads(path.read_text())
    value["schema_version"] = schema
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    assert preflight.main() == (0 if schema == 3 else 1)
    assert path.read_bytes() == before
    if schema == 2:
        assert "keine laufenden Reservierungen löschen" in capsys.readouterr().err


def test_dangling_admission_symlink_is_rejected(store):
    (store.root / "admission.json").symlink_to(store.root / "missing")
    assert preflight.main() == 1
