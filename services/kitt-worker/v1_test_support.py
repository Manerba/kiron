from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import secrets
import string
from typing import Iterator

from starlette.testclient import TestClient

import auth
import capabilities
import main


ALPHABET = string.ascii_letters + string.digits + "_-"
FIXED_CAPABILITY_TIME = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


class StaticCapabilityProbes:
    def measure_hardware(self) -> capabilities.HardwareProbeResult:
        return capabilities.HardwareProbeResult(
            accelerators=(
                {
                    "kind": "unknown",
                    "count": 0,
                    "vram_total_bytes": None,
                    "vram_available_bytes": None,
                    "vram_budget_bytes": None,
                    "capability_status": "unknown",
                },
            ),
            measurement_status="unknown",
            measurement_warnings=("test_resource_unknown",),
        )

    def measure_trainer_stack(self) -> capabilities.TrainerProbeResult:
        return capabilities.TrainerProbeResult(
            python_runtime="3.12.0",
            packages=(),
            measurement_status="unknown",
            measurement_warnings=("test_trainer_unknown",),
        )

    def probe_inference_conflict(self) -> capabilities.ConflictProbeResult:
        return capabilities.ConflictProbeResult(
            status="unknown",
            source_status="unknown",
            maintenance="unknown",
            measurement_warnings=("test_inference_unknown",),
        )


def random_text(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def credential_store(scopes: list[str]) -> tuple[auth.CredentialStore, str]:
    kid = random_text(12)
    secret = random_text(48)
    credential = f"{kid}.{secret}"
    store = auth.parse_credentials_payload(
        {
            "version": 1,
            "current": {
                "kid": kid,
                "token_hash_sha256": auth.credential_hash(credential),
                "scopes": scopes,
            },
            "revoked_kids": [],
        }
    )
    return store, credential


def authorization_header(credential: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {credential}"}


def runtime_env(tmp_path: Path, extra_env: dict[str, str] | None = None):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    env = {
        "KITT_WORKER_DATA_DIR": str(data_dir),
        "KITT_WORKER_RUN_DIR": str(run_dir),
    }
    if extra_env:
        env.update(extra_env)
    return env, data_dir, run_dir


@contextmanager
def client_context(
    tmp_path: Path,
    *,
    scopes: list[str] | None = None,
    env: dict[str, str] | None = None,
    install_store: bool = True,
    raise_server_exceptions: bool = True,
    capability_probes=None,
    capability_clock=None,
) -> Iterator[tuple[TestClient, object, dict[str, str], Path, Path]]:
    runtime, data_dir, run_dir = runtime_env(tmp_path, env)
    app = main.create_app(
        validate_runtime=True,
        validate_auth=False,
        load_config_kwargs={
            "env": runtime,
            "data_root": data_dir,
            "run_root": run_dir,
            "expected_uid": os.getuid(),
            "expected_gid": os.getgid(),
        },
    )
    store = None
    credential = ""
    if scopes is not None:
        store, credential = credential_store(scopes)
    with TestClient(app, raise_server_exceptions=raise_server_exceptions) as client:
        app.state.capability_probes = (
            StaticCapabilityProbes() if capability_probes is None else capability_probes
        )
        app.state.capability_clock = (
            (lambda: FIXED_CAPABILITY_TIME)
            if capability_clock is None
            else capability_clock
        )
        if install_store and store is not None:
            app.state.credential_store = store
        yield client, app, authorization_header(credential) if credential else {}, data_dir, run_dir
