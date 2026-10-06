"""Offline admission storage for proxy tests; never touch the runtime/GPU."""
import os
import sys
from pathlib import Path
import time

import pytest

from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity


@pytest.fixture(autouse=True)
def native_admission_runtime(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    import native_admission
    from kiron_common.gpu_admission.ollama_backend import OllamaBackend, BackendState
    async def inspect(self, target="kiron-ollama"):
        return BackendState("a" * 64, "2026-09-26T00:00:00Z", True, "running", 123)
    monkeypatch.setattr(OllamaBackend, "inspect", inspect)
    root = tmp_path / "native-admission"
    root.mkdir(mode=0o2770)
    root.chmod(0o2770)
    store = AdmissionStore(root, security=RuntimeSecurity(os.geteuid(), os.getegid(), frozenset({os.geteuid()})),
                           boot_id="test-native-admission")
    monkeypatch.setattr(native_admission, "make_store", lambda: store)
    if "app" in sys.modules:
        monkeypatch.setattr(sys.modules["app"], "ollama_admission_store", lambda: store)
    monkeypatch.setattr(native_admission, "measure_memory",
                        lambda: MemorySnapshot(12 * 1024**3, 32 * 1024**3, time.monotonic()))
    return store
