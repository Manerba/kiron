from __future__ import annotations

import os
from pathlib import Path
from urllib import error

import config
import healthcheck


class _Response:
    def __init__(self, status: int):
        self.status = status
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _cfg() -> config.WorkerConfig:
    return config.WorkerConfig(
        bind_host="localhost",
        port=11441,
        health_path="/internal/health",
        data_dir=Path("/usr/lib/kiron/data/kitt-worker"),
        run_dir=Path("/run/kiron/kitt-worker"),
        log_level="INFO",
        auth_mode="required",
        credentials_file=Path("/etc/kiron/kitt-worker/credentials.json"),
        transport_mode="internal_http",
        dispatch_gate="closed",
        worker_id="kiron-kitt-worker",
        worker_contract_version="adr-0008.v2",
        enable_v1=True,
        queue_max_jobs=config.DEFAULT_QUEUE_MAX_JOBS,
        lease_ttl_seconds=config.DEFAULT_LEASE_TTL_SECONDS,
        resume_limit=config.DEFAULT_RESUME_LIMIT,
        sqlite_busy_timeout_ms=config.DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
        queue_config_error_code=None,
        executor_enabled=False,
        executor_runner="sft_subprocess",
        sft_command="",
        executor_poll_interval_seconds=1.0,
        executor_renew_interval_seconds=60,
        executor_job_timeout_seconds=300,
        executor_shutdown_grace_seconds=10,
        executor_config_error_code=None,
        artifact_staging_dir=Path("/usr/lib/kiron/data/kitt-worker/staging"),
        artifact_max_bytes=config.DEFAULT_ARTIFACT_MAX_BYTES,
        artifact_job_quota_bytes=config.DEFAULT_ARTIFACT_JOB_QUOTA_BYTES,
        artifact_total_quota_bytes=config.DEFAULT_ARTIFACT_TOTAL_QUOTA_BYTES,
        artifact_min_free_bytes=config.DEFAULT_ARTIFACT_MIN_FREE_BYTES,
        artifact_cleanup_after_seconds=config.DEFAULT_ARTIFACT_CLEANUP_AFTER_SECONDS,
        artifact_config_error_code=None,
        monitoring_enabled=config.DEFAULT_MONITORING_ENABLED,
        alerts_enabled=config.DEFAULT_ALERTS_ENABLED,
        monitoring_interval_seconds=config.DEFAULT_MONITORING_INTERVAL_SECONDS,
        heartbeat_stale_seconds=config.DEFAULT_HEARTBEAT_STALE_SECONDS,
        alert_repeat_seconds=config.DEFAULT_ALERT_REPEAT_SECONDS,
        monitoring_config_error_code=None,
        publish_verify_keys_file=None,
        publish_verify_keyring_config_error_code="publish_keyring_unavailable",
    )


def _runtime_kwargs(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    return {
        "env": {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
        },
        "data_root": data_dir,
        "run_root": run_dir,
        "expected_uid": os.getuid(),
        "expected_gid": os.getgid(),
    }


def _clock():
    now = {"value": 0.0}

    def monotonic() -> float:
        return now["value"]

    def sleep(seconds: float) -> None:
        now["value"] += seconds

    return monotonic, sleep


def test_wait_for_health_retries_until_http_200():
    attempts = []
    outcomes = [
        OSError("connection refused"),
        _Response(503),
        _Response(200),
    ]
    monotonic, sleep = _clock()

    def request_fn(url: str, timeout: float):
        attempts.append((url, timeout))
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    assert healthcheck.wait_for_health(
        _cfg(),
        timeout=1.0,
        interval=0.1,
        request_fn=request_fn,
        sleep_fn=sleep,
        now_fn=monotonic,
    )
    assert len(attempts) == 3
    assert attempts[0][0] == "http://localhost:11441/internal/health"


def test_wait_for_health_times_out_without_http_200():
    monotonic, sleep = _clock()

    assert not healthcheck.wait_for_health(
        _cfg(),
        timeout=0.2,
        interval=0.1,
        request_fn=lambda _url, _timeout: _Response(503),
        sleep_fn=sleep,
        now_fn=monotonic,
    )


def test_probe_once_rejects_http_error():
    def request_fn(_url: str, _timeout: float):
        raise error.HTTPError(
            url="http://localhost:11441/internal/health",
            code=500,
            msg="boom",
            hdrs=None,
            fp=None,
        )

    assert not healthcheck.probe_once(
        "http://localhost:11441/internal/health",
        timeout=0.1,
        request_fn=request_fn,
    )


def test_run_exit_code_0_for_healthy(tmp_path):
    monotonic, sleep = _clock()

    rc = healthcheck.run(
        ["--timeout", "0.2", "--interval", "0.1"],
        request_fn=lambda _url, _timeout: _Response(200),
        sleep_fn=sleep,
        now_fn=monotonic,
        load_config_kwargs=_runtime_kwargs(tmp_path),
    )

    assert rc == healthcheck.EXIT_HEALTHY


def test_run_exit_code_1_for_timeout(tmp_path):
    monotonic, sleep = _clock()

    rc = healthcheck.run(
        ["--timeout", "0.2", "--interval", "0.1"],
        request_fn=lambda _url, _timeout: _Response(503),
        sleep_fn=sleep,
        now_fn=monotonic,
        load_config_kwargs=_runtime_kwargs(tmp_path),
    )

    assert rc == healthcheck.EXIT_UNHEALTHY


def test_run_exit_code_2_for_invalid_runtime_config():
    rc = healthcheck.run(
        ["--timeout", "0.2", "--interval", "0.1"],
        request_fn=lambda _url, _timeout: _Response(200),
        load_config_kwargs={"env": {"KITT_WORKER_PORT": "not-an-int"}},
    )

    assert rc == healthcheck.EXIT_INVALID_CONFIG


def test_run_exit_code_3_for_unsafe_target():
    rc = healthcheck.run(
        ["--timeout", "0.2", "--interval", "0.1"],
        request_fn=lambda _url, _timeout: _Response(200),
        load_config_kwargs={"env": {"KITT_WORKER_BIND_HOST": "0.0.0.0"}},
    )

    assert rc == healthcheck.EXIT_UNSAFE_TARGET


def test_run_exit_code_3_for_v1_health_target():
    rc = healthcheck.run(
        ["--timeout", "0.2", "--interval", "0.1"],
        request_fn=lambda _url, _timeout: _Response(200),
        load_config_kwargs={"env": {"KITT_WORKER_HEALTH_PATH": "/v1/health"}},
    )

    assert rc == healthcheck.EXIT_UNSAFE_TARGET
