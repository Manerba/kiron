from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import main


ROOT = Path(__file__).resolve().parents[2]


def test_invalid_startup_config_exits_without_traceback_or_local_paths():
    env = os.environ.copy()
    env["KITT_WORKER_PORT"] = "not-a-port"

    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=ROOT / "services" / "kitt-worker",
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "config_invalid" in result.stderr
    for forbidden in (
        "Traceback",
        'File "',
        "/opt/kiron",
        "not-a-port",
        "RuntimeConfigError",
    ):
        assert forbidden not in result.stderr


def test_runtime_import_failure_uses_safe_startup_event(caplog):
    def broken_runtime_loader():
        raise ModuleNotFoundError("/opt/kiron/services/kitt-worker/private.py")

    rc = main.run(runtime_loader=broken_runtime_loader)

    assert rc == main.STARTUP_EXIT_RUNTIME_ERROR
    assert "startup_failed" in caplog.text
    for forbidden in (
        "Traceback",
        'File "',
        "/opt/kiron",
        "ModuleNotFoundError",
        "private.py",
    ):
        assert forbidden not in caplog.text
