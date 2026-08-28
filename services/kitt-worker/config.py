"""Configuration and runtime guardrails for the kitt-worker skeleton."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import pwd
import grp
import re
import stat
from typing import Mapping

import auth
import publish_contract
import sft_command


SERVICE_USER = "kitt-worker"
SERVICE_GROUP = "kitt-worker"

PRODUCTION_BIND_HOST = "localhost"
LOCAL_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_HOST = PRODUCTION_BIND_HOST
ALLOWED_BIND_HOSTS = frozenset({PRODUCTION_BIND_HOST, LOCAL_BIND_HOST})
DEFAULT_PORT = 11441
DEFAULT_HEALTH_PATH = "/internal/health"
DEFAULT_DATA_DIR = Path("/usr/lib/kiron/data/kitt-worker")
DEFAULT_RUN_DIR = Path("/run/kiron/kitt-worker")
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_AUTH_MODE = "required"
DEFAULT_CREDENTIALS_FILE = auth.DEFAULT_CREDENTIAL_FILE
DEFAULT_TRANSPORT_MODE = "internal_http"
DEFAULT_DISPATCH_GATE = "closed"
DEFAULT_WORKER_ID = "kiron-kitt-worker"
DEFAULT_WORKER_CONTRACT_VERSION = "adr-0008.v2"
DEFAULT_ENABLE_V1 = True
DEFAULT_QUEUE_MAX_JOBS = 1000
DEFAULT_LEASE_TTL_SECONDS = 300
DEFAULT_RESUME_LIMIT = 3
DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 1000
DEFAULT_EXECUTOR_ENABLED = False
DEFAULT_EXECUTOR_RUNNER = "sft_subprocess"
DEFAULT_SFT_COMMAND = ""
DEFAULT_EXECUTOR_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_EXECUTOR_RENEW_INTERVAL_SECONDS = 60
DEFAULT_EXECUTOR_JOB_TIMEOUT_SECONDS = 300
DEFAULT_EXECUTOR_SHUTDOWN_GRACE_SECONDS = 10
DEFAULT_ARTIFACT_STAGING_DIR = DEFAULT_DATA_DIR / "staging"
_GIB = 1024 * 1024 * 1024
DEFAULT_ARTIFACT_MAX_BYTES = 20 * _GIB
DEFAULT_ARTIFACT_JOB_QUOTA_BYTES = 40 * _GIB
DEFAULT_ARTIFACT_TOTAL_QUOTA_BYTES = 48 * _GIB
DEFAULT_ARTIFACT_MIN_FREE_BYTES = 1024 * 1024 * 1024
DEFAULT_ARTIFACT_CLEANUP_AFTER_SECONDS = 86400
DEFAULT_MONITORING_ENABLED = True
DEFAULT_ALERTS_ENABLED = True
DEFAULT_MONITORING_INTERVAL_SECONDS = 30.0
DEFAULT_HEARTBEAT_STALE_SECONDS = 300
DEFAULT_ALERT_REPEAT_SECONDS = 3600
DEFAULT_PUBLISH_VERIFY_KEYS_FILE: Path | None = None

ALLOWED_DATA_ROOT = DEFAULT_DATA_DIR
ALLOWED_RUN_ROOT = DEFAULT_RUN_DIR
ALLOWED_CREDENTIAL_ROOT = auth.DEFAULT_CREDENTIAL_DIR
REQUIRED_RUNTIME_MODE = 0o750

_LOG_LEVELS = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
}
_AUTH_MODES = {"required", "disabled"}
_DISPATCH_GATES = {"closed", "open"}
_WORKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_SECRET_WORDS = ("token", "secret", "credential", "password", "key")


class ConfigError(RuntimeError):
    """Base class for fail-closed configuration errors."""


class RuntimeConfigError(ConfigError):
    """Invalid or unusable local runtime configuration."""


class UnsafeTargetError(ConfigError):
    """The configured healthcheck target is outside the LAN HTTP contract."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    bind_host: str
    port: int
    health_path: str
    data_dir: Path
    run_dir: Path
    log_level: str
    auth_mode: str
    credentials_file: Path
    transport_mode: str
    dispatch_gate: str
    worker_id: str
    worker_contract_version: str
    enable_v1: bool
    queue_max_jobs: int
    lease_ttl_seconds: int
    resume_limit: int
    sqlite_busy_timeout_ms: int
    queue_config_error_code: str | None
    executor_enabled: bool
    executor_runner: str
    sft_command: str
    executor_poll_interval_seconds: float
    executor_renew_interval_seconds: int
    executor_job_timeout_seconds: int
    executor_shutdown_grace_seconds: int
    executor_config_error_code: str | None
    artifact_staging_dir: Path
    artifact_max_bytes: int
    artifact_job_quota_bytes: int
    artifact_total_quota_bytes: int
    artifact_min_free_bytes: int
    artifact_cleanup_after_seconds: int
    artifact_config_error_code: str | None
    monitoring_enabled: bool
    alerts_enabled: bool
    monitoring_interval_seconds: float
    heartbeat_stale_seconds: int
    alert_repeat_seconds: int
    monitoring_config_error_code: str | None
    publish_verify_keys_file: Path | None
    publish_verify_keyring_config_error_code: str | None


def load_config(
    env: Mapping[str, str] | None = None,
    *,
    validate_runtime: bool = False,
    validate_auth: bool = False,
    data_root: Path = ALLOWED_DATA_ROOT,
    run_root: Path = ALLOWED_RUN_ROOT,
    credential_root: Path = ALLOWED_CREDENTIAL_ROOT,
    credential_dir: Path = auth.DEFAULT_CREDENTIAL_DIR,
    expected_user: str = SERVICE_USER,
    expected_group: str = SERVICE_GROUP,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_credential_root_uid: int = 0,
    expected_credential_root_gid: int = 0,
    expected_credential_dir_uid: int = 0,
    expected_credential_dir_gid: int | None = None,
    expected_credential_file_uid: int = 0,
    expected_credential_file_gid: int | None = None,
) -> WorkerConfig:
    """Load env config and fail closed on unsafe values.

    Missing non-secret env values intentionally fall back to Sprint 11 defaults.
    Runtime directory existence, owner and mode are checked only when requested,
    so import smokes remain side-effect free.
    """

    env = os.environ if env is None else env
    _reject_removed_tls_env(env)
    queue_max_jobs, queue_max_jobs_error = _read_queue_int(
        env,
        "KITT_WORKER_QUEUE_MAX_JOBS",
        DEFAULT_QUEUE_MAX_JOBS,
        minimum=1,
        maximum=100000,
    )
    lease_ttl_seconds, lease_ttl_error = _read_queue_int(
        env,
        "KITT_WORKER_LEASE_TTL_SECONDS",
        DEFAULT_LEASE_TTL_SECONDS,
        minimum=30,
        maximum=86400,
    )
    resume_limit, resume_limit_error = _read_queue_int(
        env,
        "KITT_WORKER_RESUME_LIMIT",
        DEFAULT_RESUME_LIMIT,
        minimum=0,
        maximum=10,
    )
    sqlite_busy_timeout_ms, busy_timeout_error = _read_queue_int(
        env,
        "KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS",
        DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
        minimum=100,
        maximum=5000,
    )
    queue_config_error_code = next(
        (
            error
            for error in (
                queue_max_jobs_error,
                lease_ttl_error,
                resume_limit_error,
                busy_timeout_error,
            )
            if error is not None
        ),
        None,
    )
    executor_enabled, executor_enabled_error = _read_executor_bool(env)
    executor_runner, executor_runner_error = _read_executor_runner(env)
    sft_command, sft_command_error = _read_sft_command(
        env,
        executor_enabled=executor_enabled,
        executor_runner=executor_runner,
    )
    executor_poll_interval_seconds, executor_poll_error = _read_executor_float(
        env,
        "KITT_WORKER_EXECUTOR_POLL_INTERVAL_SECONDS",
        DEFAULT_EXECUTOR_POLL_INTERVAL_SECONDS,
        minimum=0.1,
        maximum=60.0,
    )
    executor_renew_interval_seconds, executor_renew_error = _read_executor_int(
        env,
        "KITT_WORKER_EXECUTOR_RENEW_INTERVAL_SECONDS",
        DEFAULT_EXECUTOR_RENEW_INTERVAL_SECONDS,
        minimum=1,
        maximum=86400,
    )
    executor_job_timeout_seconds, executor_timeout_error = _read_executor_int(
        env,
        "KITT_WORKER_EXECUTOR_JOB_TIMEOUT_SECONDS",
        DEFAULT_EXECUTOR_JOB_TIMEOUT_SECONDS,
        minimum=1,
        maximum=86400,
    )
    executor_shutdown_grace_seconds, executor_grace_error = _read_executor_int(
        env,
        "KITT_WORKER_EXECUTOR_SHUTDOWN_GRACE_SECONDS",
        DEFAULT_EXECUTOR_SHUTDOWN_GRACE_SECONDS,
        minimum=1,
        maximum=300,
    )
    if (
        executor_enabled
        and executor_renew_error is None
        and executor_renew_interval_seconds > max(0, lease_ttl_seconds // 2)
    ):
        executor_renew_error = "config_invalid"
    executor_config_error_code = next(
        (
            error
            for error in (
                executor_enabled_error,
                executor_runner_error,
                sft_command_error,
                executor_poll_error,
                executor_renew_error,
                executor_timeout_error,
                executor_grace_error,
            )
            if error is not None
        ),
        None,
    )

    data_dir = _read_runtime_path(
        env,
        "KITT_WORKER_DATA_DIR",
        DEFAULT_DATA_DIR,
        Path(data_root),
    )
    run_dir = _read_runtime_path(
        env,
        "KITT_WORKER_RUN_DIR",
        DEFAULT_RUN_DIR,
        Path(run_root),
    )
    artifact_max_bytes, artifact_max_error = _read_artifact_int(
        env,
        "KITT_WORKER_ARTIFACT_MAX_BYTES",
        DEFAULT_ARTIFACT_MAX_BYTES,
        minimum=1,
        maximum=9_223_372_036_854_775_807,
    )
    artifact_job_quota_bytes, artifact_job_quota_error = _read_artifact_int(
        env,
        "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES",
        DEFAULT_ARTIFACT_JOB_QUOTA_BYTES,
        minimum=1,
        maximum=9_223_372_036_854_775_807,
    )
    artifact_total_quota_bytes, artifact_total_quota_error = _read_artifact_int(
        env,
        "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES",
        DEFAULT_ARTIFACT_TOTAL_QUOTA_BYTES,
        minimum=1,
        maximum=9_223_372_036_854_775_807,
    )
    artifact_min_free_bytes, artifact_min_free_error = _read_artifact_int(
        env,
        "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES",
        DEFAULT_ARTIFACT_MIN_FREE_BYTES,
        minimum=0,
        maximum=9_223_372_036_854_775_807,
    )
    artifact_cleanup_after_seconds, artifact_cleanup_error = _read_artifact_int(
        env,
        "KITT_WORKER_ARTIFACT_CLEANUP_AFTER_SECONDS",
        DEFAULT_ARTIFACT_CLEANUP_AFTER_SECONDS,
        minimum=0,
        maximum=31_536_000,
    )
    try:
        artifact_staging_dir = _read_artifact_staging_path(env, data_dir)
        artifact_path_error = None
    except RuntimeConfigError:
        artifact_staging_dir = data_dir / "staging"
        artifact_path_error = "artifact_staging_unavailable"
    publish_verify_keys_file, publish_verify_keys_error = _read_publish_verify_keys_file(
        env,
        data_dir=data_dir,
        credential_root=Path(credential_root),
    )
    artifact_order_error = None
    if (
        artifact_max_error is None
        and artifact_job_quota_error is None
        and artifact_total_quota_error is None
        and (
            artifact_max_bytes > artifact_job_quota_bytes
            or artifact_job_quota_bytes > artifact_total_quota_bytes
        )
    ):
        artifact_order_error = "artifact_quota_invalid"
    artifact_config_error_code = next(
        (
            error
            for error in (
                artifact_path_error,
                artifact_max_error,
                artifact_job_quota_error,
                artifact_total_quota_error,
                artifact_min_free_error,
                artifact_cleanup_error,
                artifact_order_error,
            )
            if error is not None
        ),
        None,
    )
    monitoring_enabled, monitoring_enabled_error = _read_monitoring_bool(
        env,
        "KITT_WORKER_MONITORING_ENABLED",
        DEFAULT_MONITORING_ENABLED,
    )
    alerts_enabled, alerts_enabled_error = _read_monitoring_bool(
        env,
        "KITT_WORKER_ALERTS_ENABLED",
        DEFAULT_ALERTS_ENABLED,
    )
    monitoring_interval_seconds, monitoring_interval_error = _read_monitoring_float(
        env,
        "KITT_WORKER_MONITORING_INTERVAL_SECONDS",
        DEFAULT_MONITORING_INTERVAL_SECONDS,
        minimum=1.0,
        maximum=300.0,
    )
    heartbeat_stale_seconds, heartbeat_stale_error = _read_monitoring_int(
        env,
        "KITT_WORKER_HEARTBEAT_STALE_SECONDS",
        DEFAULT_HEARTBEAT_STALE_SECONDS,
        minimum=0,
        maximum=86400,
    )
    alert_repeat_seconds, alert_repeat_error = _read_monitoring_int(
        env,
        "KITT_WORKER_ALERT_REPEAT_SECONDS",
        DEFAULT_ALERT_REPEAT_SECONDS,
        minimum=60,
        maximum=86400,
    )
    monitoring_ttl_error = None
    if monitoring_interval_error is None and (
        monitoring_interval_seconds >= 60.0
    ):
        monitoring_ttl_error = "monitoring_config_invalid"
    monitoring_config_error_code = next(
        (
            error
            for error in (
                monitoring_enabled_error,
                alerts_enabled_error,
                monitoring_interval_error,
                heartbeat_stale_error,
                alert_repeat_error,
                monitoring_ttl_error,
            )
            if error is not None
        ),
        None,
    )
    if monitoring_config_error_code is not None or not monitoring_enabled:
        monitoring_enabled = False
        alerts_enabled = False

    cfg = WorkerConfig(
        bind_host=_read_string(env, "KITT_WORKER_BIND_HOST", DEFAULT_BIND_HOST),
        port=_read_port(env),
        health_path=_read_string(env, "KITT_WORKER_HEALTH_PATH", DEFAULT_HEALTH_PATH),
        data_dir=data_dir,
        run_dir=run_dir,
        log_level=_read_log_level(env),
        auth_mode=_read_auth_mode(env),
        credentials_file=_read_credentials_file(env, Path(credential_root)),
        transport_mode=DEFAULT_TRANSPORT_MODE,
        dispatch_gate=_read_dispatch_gate(env),
        worker_id=_read_worker_id(env),
        worker_contract_version=_read_contract_version(env),
        enable_v1=_read_enable_v1(env),
        queue_max_jobs=queue_max_jobs,
        lease_ttl_seconds=lease_ttl_seconds,
        resume_limit=resume_limit,
        sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
        queue_config_error_code=queue_config_error_code,
        executor_enabled=executor_enabled,
        executor_runner=executor_runner,
        sft_command=sft_command,
        executor_poll_interval_seconds=executor_poll_interval_seconds,
        executor_renew_interval_seconds=executor_renew_interval_seconds,
        executor_job_timeout_seconds=executor_job_timeout_seconds,
        executor_shutdown_grace_seconds=executor_shutdown_grace_seconds,
        executor_config_error_code=executor_config_error_code,
        artifact_staging_dir=artifact_staging_dir,
        artifact_max_bytes=artifact_max_bytes,
        artifact_job_quota_bytes=artifact_job_quota_bytes,
        artifact_total_quota_bytes=artifact_total_quota_bytes,
        artifact_min_free_bytes=artifact_min_free_bytes,
        artifact_cleanup_after_seconds=artifact_cleanup_after_seconds,
        artifact_config_error_code=artifact_config_error_code,
        monitoring_enabled=monitoring_enabled,
        alerts_enabled=alerts_enabled,
        monitoring_interval_seconds=monitoring_interval_seconds,
        heartbeat_stale_seconds=heartbeat_stale_seconds,
        alert_repeat_seconds=alert_repeat_seconds,
        monitoring_config_error_code=monitoring_config_error_code,
        publish_verify_keys_file=publish_verify_keys_file,
        publish_verify_keyring_config_error_code=publish_verify_keys_error,
    )
    validate_target(cfg)
    if validate_runtime:
        validate_runtime_paths(
            cfg,
            expected_user=expected_user,
            expected_group=expected_group,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
    if validate_auth and cfg.enable_v1 and cfg.auth_mode == "required":
        load_auth_store(
            cfg,
            credential_root=Path(credential_root).parent,
            credential_dir=Path(credential_dir),
            expected_group=expected_group,
            expected_root_uid=expected_credential_root_uid,
            expected_root_gid=expected_credential_root_gid,
            expected_dir_uid=expected_credential_dir_uid,
            expected_dir_gid=expected_credential_dir_gid,
            expected_file_uid=expected_credential_file_uid,
            expected_file_gid=expected_credential_file_gid,
        )
    return cfg


def validate_target(cfg: WorkerConfig) -> None:
    """Require the approved LAN/loopback internal healthcheck target."""

    if cfg.bind_host not in ALLOWED_BIND_HOSTS:
        raise UnsafeTargetError(
            "KITT_WORKER_BIND_HOST must be localhost or 127.0.0.1"
        )
    if cfg.port != DEFAULT_PORT:
        raise UnsafeTargetError(f"KITT_WORKER_PORT must be {DEFAULT_PORT}")
    if cfg.health_path != DEFAULT_HEALTH_PATH:
        raise UnsafeTargetError(
            f"KITT_WORKER_HEALTH_PATH must be {DEFAULT_HEALTH_PATH}"
        )
    if cfg.health_path.startswith("/v1/") or cfg.health_path == "/v1":
        raise UnsafeTargetError(
            "/v1 health targets are not valid for internal healthchecks"
        )


def validate_runtime_paths(
    cfg: WorkerConfig,
    *,
    expected_user: str = SERVICE_USER,
    expected_group: str = SERVICE_GROUP,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
) -> None:
    uid = _expected_uid(expected_user, expected_uid)
    gid = _expected_gid(expected_group, expected_gid)
    for path, label in ((cfg.data_dir, "data"), (cfg.run_dir, "run")):
        _validate_runtime_dir(path, label=label, uid=uid, gid=gid)


def load_auth_store(
    cfg: WorkerConfig,
    *,
    credential_root: Path = auth.DEFAULT_CREDENTIAL_ROOT,
    credential_dir: Path = auth.DEFAULT_CREDENTIAL_DIR,
    expected_group: str = SERVICE_GROUP,
    expected_root_uid: int = 0,
    expected_root_gid: int = 0,
    expected_dir_uid: int = 0,
    expected_dir_gid: int | None = None,
    expected_file_uid: int = 0,
    expected_file_gid: int | None = None,
    now_fn=None,
):
    if not cfg.enable_v1:
        return None
    if cfg.auth_mode == "disabled":
        return None
    if cfg.auth_mode != "required":
        raise RuntimeConfigError("KITT_WORKER_AUTH_MODE is invalid")
    try:
        return auth.load_credentials_file(
            cfg.credentials_file,
            expected_group=expected_group,
            expected_root_uid=expected_root_uid,
            expected_root_gid=expected_root_gid,
            expected_dir_uid=expected_dir_uid,
            expected_dir_gid=expected_dir_gid,
            expected_file_uid=expected_file_uid,
            expected_file_gid=expected_file_gid,
            credential_root=credential_root,
            credential_dir=credential_dir,
            now_fn=now_fn,
        )
    except auth.AuthConfigError as exc:
        raise RuntimeConfigError("KITT_WORKER_CREDENTIALS_FILE is invalid") from exc


def health_url(cfg: WorkerConfig) -> str:
    validate_target(cfg)
    return f"http://{cfg.bind_host}:{cfg.port}{cfg.health_path}"


def configure_logging(cfg: WorkerConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import monitoring

    monitoring.install_logging_redaction(getattr(logging, cfg.log_level))


def _read_string(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _read_port(env: Mapping[str, str]) -> int:
    raw = env.get("KITT_WORKER_PORT", str(DEFAULT_PORT))
    try:
        port = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError("KITT_WORKER_PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise RuntimeConfigError("KITT_WORKER_PORT must be in 1..65535")
    return port


def _read_log_level(env: Mapping[str, str]) -> str:
    level = env.get("KITT_WORKER_LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
    if level not in _LOG_LEVELS:
        raise RuntimeConfigError("KITT_WORKER_LOG_LEVEL is invalid")
    return level


def _read_auth_mode(env: Mapping[str, str]) -> str:
    mode = env.get("KITT_WORKER_AUTH_MODE", DEFAULT_AUTH_MODE)
    if not isinstance(mode, str):
        raise RuntimeConfigError("KITT_WORKER_AUTH_MODE is invalid")
    if mode not in _AUTH_MODES:
        raise RuntimeConfigError("KITT_WORKER_AUTH_MODE is invalid")
    return mode


def _read_dispatch_gate(env: Mapping[str, str]) -> str:
    raw = env.get("KITT_WORKER_DISPATCH_GATE", DEFAULT_DISPATCH_GATE)
    if not isinstance(raw, str):
        raise RuntimeConfigError("KITT_WORKER_DISPATCH_GATE is invalid")
    value = raw.strip().lower()
    if value not in _DISPATCH_GATES:
        raise RuntimeConfigError("KITT_WORKER_DISPATCH_GATE is invalid")
    return value


def _reject_removed_tls_env(env: Mapping[str, str]) -> None:
    if "KITT_WORKER_TLS_TERMINATION" in env:
        raise RuntimeConfigError(
            "KITT_WORKER_TLS_TERMINATION is removed; Sprint 11 uses internal HTTP"
        )


def _read_worker_id(env: Mapping[str, str]) -> str:
    value = _read_string(env, "KITT_WORKER_ID", DEFAULT_WORKER_ID)
    lowered = value.lower()
    if value != lowered or not _WORKER_ID_RE.fullmatch(value):
        raise RuntimeConfigError("KITT_WORKER_ID is invalid")
    if any(word in lowered for word in _SECRET_WORDS):
        raise RuntimeConfigError("KITT_WORKER_ID must not look secret-like")
    return value


def _read_contract_version(env: Mapping[str, str]) -> str:
    value = _read_string(
        env,
        "KITT_WORKER_CONTRACT_VERSION",
        DEFAULT_WORKER_CONTRACT_VERSION,
    )
    if value != DEFAULT_WORKER_CONTRACT_VERSION:
        raise RuntimeConfigError("KITT_WORKER_CONTRACT_VERSION is invalid")
    return value


def _read_publish_verify_keys_file(
    env: Mapping[str, str],
    *,
    data_dir: Path,
    credential_root: Path,
) -> tuple[Path | None, str | None]:
    raw = env.get("KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE", "")
    if not isinstance(raw, str):
        return None, "config_invalid"
    value = raw.strip()
    if not value:
        return None, "publish_keyring_unavailable"
    path = Path(value)
    if not path.is_absolute():
        return None, "config_invalid"
    resolved = path.resolve(strict=False)
    data_root = data_dir.resolve(strict=False)
    credential_root = credential_root.resolve(strict=False)
    if not (
        resolved.is_relative_to(data_root)
        or resolved.is_relative_to(credential_root)
    ):
        return None, "config_invalid"
    try:
        keyring = _load_publish_verify_keyring_from_path(resolved)
        _require_current_publish_verify_key_available(keyring)
    except publish_contract.PublishSpecError as exc:
        return resolved, exc.reason_code
    except RuntimeConfigError:
        return resolved, "config_invalid"
    return resolved, None


def load_publish_verify_keyring(
    cfg: WorkerConfig,
) -> publish_contract.PublishVerifyKeyring | None:
    if (
        cfg.publish_verify_keys_file is None
        or cfg.publish_verify_keyring_config_error_code is not None
    ):
        return None
    return _load_publish_verify_keyring_from_path(cfg.publish_verify_keys_file)


def publish_verify_keyring_availability_error(
    cfg: object,
    *,
    now: datetime | None = None,
) -> str | None:
    configured_error = getattr(
        cfg,
        "publish_verify_keyring_config_error_code",
        "publish_keyring_unavailable",
    )
    if isinstance(configured_error, str):
        return configured_error
    path = getattr(cfg, "publish_verify_keys_file", None)
    if path is None:
        if hasattr(cfg, "publish_verify_keys_file"):
            return "publish_keyring_unavailable"
        return None
    try:
        keyring = _load_publish_verify_keyring_from_path(Path(path))
        keyring.key_for_id(keyring.current.key_id, now=now)
    except publish_contract.PublishSpecError as exc:
        return exc.reason_code
    except RuntimeConfigError:
        return "config_invalid"
    return None


def _load_publish_verify_keyring_from_path(
    path: Path,
) -> publish_contract.PublishVerifyKeyring:
    try:
        file_stat = path.lstat()
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise OSError("publish verify keyring path is not a regular file")
        mode = stat.S_IMODE(file_stat.st_mode)
        if mode & 0o027 or mode not in {0o600, 0o640}:
            raise OSError("publish verify keyring mode is too broad")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError("publish verify keyring is invalid") from exc
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != "kitt_publish_verify_keys_v1"
    ):
        raise RuntimeConfigError("publish verify keyring is invalid")
    current = _publish_verify_key_from_payload(data.get("current"))
    previous_payload = data.get("previous")
    previous = (
        None
        if previous_payload is None
        else _publish_verify_key_from_payload(previous_payload)
    )
    return publish_contract.PublishVerifyKeyring(
        current=current,
        previous=previous,
    )


def _require_current_publish_verify_key_available(
    keyring: publish_contract.PublishVerifyKeyring,
) -> None:
    keyring.key_for_id(keyring.current.key_id)


def _publish_verify_key_from_payload(
    value: object,
) -> publish_contract.PublishVerifyKey:
    if not isinstance(value, dict):
        raise RuntimeConfigError("publish verify keyring is invalid")
    key_id = value.get("key_id")
    encoded_key = value.get("public_key_hex")
    if (
        not isinstance(key_id, str)
        or publish_contract.SAFE_KEY_ID_RE.fullmatch(key_id) is None
        or not isinstance(encoded_key, str)
    ):
        raise RuntimeConfigError("publish verify keyring is invalid")
    try:
        key_bytes = bytes.fromhex(encoded_key)
    except ValueError as exc:
        raise RuntimeConfigError("publish verify keyring is invalid") from exc
    if len(key_bytes) != 32:
        raise RuntimeConfigError("publish verify keyring is invalid")
    active = value.get("active", True)
    if not isinstance(active, bool):
        raise RuntimeConfigError("publish verify keyring is invalid")
    not_before = _publish_verify_key_time(value.get("not_before"))
    not_after = _publish_verify_key_time(value.get("not_after"))
    if not_before is not None and not_after is not None and not_after <= not_before:
        raise RuntimeConfigError("publish verify keyring is invalid")
    return publish_contract.PublishVerifyKey(
        key_id=key_id,
        public_key=key_bytes,
        active=active,
        not_before=not_before,
        not_after=not_after,
    )


def _publish_verify_key_time(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeConfigError("publish verify keyring is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeConfigError("publish verify keyring is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeConfigError("publish verify keyring is invalid")
    return parsed.astimezone(timezone.utc)


def _read_enable_v1(env: Mapping[str, str]) -> bool:
    raw = env.get("KITT_WORKER_ENABLE_V1", "true" if DEFAULT_ENABLE_V1 else "false")
    if not isinstance(raw, str):
        raise RuntimeConfigError("KITT_WORKER_ENABLE_V1 is invalid")
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeConfigError("KITT_WORKER_ENABLE_V1 is invalid")


def _read_queue_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, str | None]:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, "config_invalid"
    if isinstance(raw, bool) or value < minimum or value > maximum:
        return default, "config_invalid"
    return value, None


def _read_executor_bool(env: Mapping[str, str]) -> tuple[bool, str | None]:
    raw = env.get(
        "KITT_WORKER_EXECUTOR_ENABLED",
        "true" if DEFAULT_EXECUTOR_ENABLED else "false",
    )
    if not isinstance(raw, str):
        return DEFAULT_EXECUTOR_ENABLED, "config_invalid"
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True, None
    if value in {"0", "false", "no", "off"}:
        return False, None
    return DEFAULT_EXECUTOR_ENABLED, "config_invalid"


def _read_executor_runner(env: Mapping[str, str]) -> tuple[str, str | None]:
    raw = env.get("KITT_WORKER_EXECUTOR_RUNNER", DEFAULT_EXECUTOR_RUNNER)
    if not isinstance(raw, str):
        return DEFAULT_EXECUTOR_RUNNER, "config_invalid"
    value = raw.strip()
    if value == "sft_subprocess":
        return value, None
    return DEFAULT_EXECUTOR_RUNNER, "config_invalid"


def _read_sft_command(
    env: Mapping[str, str],
    *,
    executor_enabled: bool,
    executor_runner: str,
) -> tuple[str, str | None]:
    raw = env.get("KITT_WORKER_SFT_COMMAND", DEFAULT_SFT_COMMAND)
    if not isinstance(raw, str):
        return DEFAULT_SFT_COMMAND, "config_invalid"
    value = raw.strip()
    if not value:
        if executor_enabled and executor_runner == "sft_subprocess":
            return DEFAULT_SFT_COMMAND, "config_invalid"
        return DEFAULT_SFT_COMMAND, None
    lowered = value.lower()
    if any(word in lowered for word in _SECRET_WORDS):
        return DEFAULT_SFT_COMMAND, "config_invalid"
    if not sft_command.is_trusted_sft_command(value):
        return DEFAULT_SFT_COMMAND, "config_invalid"
    return value, None


def _read_executor_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, str | None]:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, "config_invalid"
    if isinstance(raw, bool) or value < minimum or value > maximum:
        return default, "config_invalid"
    return value, None


def _read_executor_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, str | None]:
    raw = env.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, "config_invalid"
    if (
        isinstance(raw, bool)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        return default, "config_invalid"
    return value, None


def _read_monitoring_bool(
    env: Mapping[str, str],
    name: str,
    default: bool,
) -> tuple[bool, str | None]:
    raw = env.get(name, "true" if default else "false")
    if not isinstance(raw, str):
        return False, "monitoring_config_invalid"
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True, None
    if value in {"0", "false", "no", "off"}:
        return False, None
    return False, "monitoring_config_invalid"


def _read_monitoring_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, str | None]:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, "monitoring_config_invalid"
    if isinstance(raw, bool) or value < minimum or value > maximum:
        return default, "monitoring_config_invalid"
    return value, None


def _read_monitoring_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> tuple[float, str | None]:
    raw = env.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, "monitoring_config_invalid"
    if (
        isinstance(raw, bool)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        return default, "monitoring_config_invalid"
    return value, None


def _read_artifact_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, str | None]:
    raw = env.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, "artifact_quota_invalid"
    if isinstance(raw, bool) or value < minimum or value > maximum:
        return default, "artifact_quota_invalid"
    return value, None


def _read_credentials_file(
    env: Mapping[str, str],
    allowed_root: Path,
) -> Path:
    raw = env.get("KITT_WORKER_CREDENTIALS_FILE", str(DEFAULT_CREDENTIALS_FILE))
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeConfigError("KITT_WORKER_CREDENTIALS_FILE must be absolute")
    resolved = path.resolve(strict=False)
    root = allowed_root.resolve(strict=False)
    if resolved != root and not resolved.is_relative_to(root):
        raise RuntimeConfigError("KITT_WORKER_CREDENTIALS_FILE is outside the allowed root")
    for prohibited in (
        Path("/opt/kiron"),
        Path("/usr/lib/kiron/data"),
        Path("/run/kiron"),
        Path("/usr/lib/kiron/services"),
    ):
        prohibited_resolved = prohibited.resolve(strict=False)
        if resolved == prohibited_resolved or resolved.is_relative_to(prohibited_resolved):
            raise RuntimeConfigError("KITT_WORKER_CREDENTIALS_FILE is not allowed")
    return resolved


def _read_artifact_staging_path(env: Mapping[str, str], data_dir: Path) -> Path:
    raw = env.get("KITT_WORKER_ARTIFACT_STAGING_DIR", str(Path(data_dir) / "staging"))
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeConfigError("KITT_WORKER_ARTIFACT_STAGING_DIR must be absolute")
    try:
        st = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RuntimeConfigError("KITT_WORKER_ARTIFACT_STAGING_DIR is not usable") from exc
    else:
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeConfigError("KITT_WORKER_ARTIFACT_STAGING_DIR must not be a symlink")
    resolved = path.resolve(strict=False)
    root = Path(data_dir).resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeConfigError(
            "KITT_WORKER_ARTIFACT_STAGING_DIR is outside the worker data root"
        )
    for prohibited in (
        Path("/opt/kiron"),
        Path("/usr/lib/kiron/services"),
        Path("/run/kiron"),
        Path("/etc/kiron"),
    ):
        prohibited_resolved = prohibited.resolve(strict=False)
        if resolved == prohibited_resolved or resolved.is_relative_to(prohibited_resolved):
            raise RuntimeConfigError("KITT_WORKER_ARTIFACT_STAGING_DIR is not allowed")
    return resolved


def _read_runtime_path(
    env: Mapping[str, str],
    name: str,
    default: Path,
    allowed_root: Path,
) -> Path:
    raw = env.get(name, str(default))
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeConfigError(f"{name} must be absolute")
    resolved = path.resolve(strict=False)
    root = allowed_root.resolve(strict=False)
    if resolved != root and not resolved.is_relative_to(root):
        raise RuntimeConfigError(f"{name} is outside the allowed runtime root")
    return resolved


def _expected_uid(expected_user: str, expected_uid: int | None) -> int:
    if expected_uid is not None:
        return expected_uid
    try:
        return pwd.getpwnam(expected_user).pw_uid
    except KeyError as exc:
        raise RuntimeConfigError(f"system user {expected_user!r} is missing") from exc


def _expected_gid(expected_group: str, expected_gid: int | None) -> int:
    if expected_gid is not None:
        return expected_gid
    try:
        return grp.getgrnam(expected_group).gr_gid
    except KeyError as exc:
        raise RuntimeConfigError(f"system group {expected_group!r} is missing") from exc


def _validate_runtime_dir(path: Path, *, label: str, uid: int, gid: int) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeConfigError(f"{label} runtime directory is missing") from exc
    if stat.S_ISLNK(st.st_mode):
        raise RuntimeConfigError(f"{label} runtime directory must not be a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise RuntimeConfigError(f"{label} runtime path must be a directory")
    mode = stat.S_IMODE(st.st_mode)
    if mode != REQUIRED_RUNTIME_MODE:
        raise RuntimeConfigError(
            f"{label} runtime directory mode must be 0750"
        )
    if st.st_uid != uid or st.st_gid != gid:
        raise RuntimeConfigError(
            f"{label} runtime directory owner must be {SERVICE_USER}:{SERVICE_GROUP}"
        )
    if os.geteuid() == uid and not os.access(path, os.W_OK | os.X_OK, effective_ids=True):
        raise RuntimeConfigError(f"{label} runtime directory is not writable")
