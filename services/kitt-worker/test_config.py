from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import string
import sys

import pytest
from starlette.testclient import TestClient

import auth
import config
import main
import publish_contract


ALPHABET = string.ascii_letters + string.digits + "_-"


def _random_text(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def _runtime_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    data_dir.mkdir()
    run_dir.mkdir()
    data_dir.chmod(0o750)
    run_dir.chmod(0o750)
    return (
        {
            "KITT_WORKER_DATA_DIR": str(data_dir),
            "KITT_WORKER_RUN_DIR": str(run_dir),
        },
        data_dir,
        run_dir,
    )


def _write_sft_trainer(tmp_path: Path) -> Path:
    return Path(__file__).resolve().with_name("sft_trainer.py")


def test_defaults_keep_internal_worker_contract_with_auth_required():
    cfg = config.load_config()

    assert cfg.bind_host == "localhost"
    assert cfg.port == 11441
    assert cfg.health_path == "/internal/health"
    assert cfg.data_dir == Path("/usr/lib/kiron/data/kitt-worker")
    assert cfg.run_dir == Path("/run/kiron/kitt-worker")
    assert cfg.log_level == "INFO"
    assert cfg.auth_mode == "required"
    assert cfg.credentials_file == Path("/etc/kiron/kitt-worker/credentials.json")
    assert cfg.transport_mode == "internal_http"
    assert cfg.dispatch_gate == "closed"
    assert cfg.worker_id == "kiron-kitt-worker"
    assert cfg.worker_contract_version == "adr-0008.v2"
    assert cfg.enable_v1 is True
    assert cfg.queue_max_jobs == 1000
    assert cfg.lease_ttl_seconds == 300
    assert cfg.resume_limit == 3
    assert cfg.sqlite_busy_timeout_ms == 1000
    assert cfg.queue_config_error_code is None
    assert cfg.executor_enabled is False
    assert cfg.executor_runner == "sft_subprocess"
    assert cfg.sft_command == ""
    assert cfg.executor_poll_interval_seconds == 1
    assert cfg.executor_renew_interval_seconds == 60
    assert cfg.executor_job_timeout_seconds == 300
    assert cfg.executor_shutdown_grace_seconds == 10
    assert cfg.executor_config_error_code is None
    assert cfg.artifact_staging_dir == Path("/usr/lib/kiron/data/kitt-worker/staging")
    assert cfg.artifact_max_bytes == 20 * 1024 * 1024 * 1024
    assert cfg.artifact_job_quota_bytes == 40 * 1024 * 1024 * 1024
    assert cfg.artifact_total_quota_bytes == 48 * 1024 * 1024 * 1024
    assert cfg.artifact_min_free_bytes == 1024 * 1024 * 1024
    assert cfg.artifact_cleanup_after_seconds == 86400
    assert cfg.artifact_config_error_code is None
    assert cfg.monitoring_enabled is True
    assert cfg.alerts_enabled is True
    assert cfg.monitoring_interval_seconds == 30.0
    assert cfg.heartbeat_stale_seconds == 300
    assert cfg.alert_repeat_seconds == 3600
    assert cfg.monitoring_config_error_code is None
    assert cfg.publish_verify_keys_file is None
    assert cfg.publish_verify_keyring_config_error_code == "publish_keyring_unavailable"
    assert config.health_url(cfg) == "http://localhost:11441/internal/health"


def test_localhost_bind_host_remains_allowed_for_local_smokes():
    cfg = config.load_config({"KITT_WORKER_BIND_HOST": "127.0.0.1"})

    assert cfg.bind_host == "127.0.0.1"
    assert config.health_url(cfg) == "http://127.0.0.1:11441/internal/health"


def test_invalid_monitoring_config_disables_monitoring_without_runtime_failure():
    cfg = config.load_config({"KITT_WORKER_MONITORING_INTERVAL_SECONDS": "60"})

    assert cfg.monitoring_enabled is False
    assert cfg.alerts_enabled is False
    assert cfg.monitoring_config_error_code == "monitoring_config_invalid"


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_non_finite_monitoring_float_disables_monitoring(raw):
    cfg = config.load_config({"KITT_WORKER_MONITORING_INTERVAL_SECONDS": raw})

    assert cfg.monitoring_enabled is False
    assert cfg.alerts_enabled is False
    assert cfg.monitoring_config_error_code == "monitoring_config_invalid"


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf"])
def test_non_finite_executor_float_fails_closed(raw):
    cfg = config.load_config({"KITT_WORKER_EXECUTOR_POLL_INTERVAL_SECONDS": raw})

    assert (
        cfg.executor_poll_interval_seconds
        == config.DEFAULT_EXECUTOR_POLL_INTERVAL_SECONDS
    )
    assert cfg.executor_config_error_code == "config_invalid"


def test_huge_monitoring_integer_disables_monitoring_without_crash():
    cfg = config.load_config({"KITT_WORKER_ALERT_REPEAT_SECONDS": "9" * 4000})

    assert cfg.monitoring_enabled is False
    assert cfg.alerts_enabled is False
    assert cfg.monitoring_config_error_code == "monitoring_config_invalid"


def test_huge_queue_integer_degrades_queue_without_crash():
    cfg = config.load_config({"KITT_WORKER_QUEUE_MAX_JOBS": "9" * 4000})

    assert cfg.queue_config_error_code == "config_invalid"


def test_huge_executor_integer_degrades_executor_without_crash():
    cfg = config.load_config(
        {"KITT_WORKER_EXECUTOR_RENEW_INTERVAL_SECONDS": "9" * 4000}
    )

    assert cfg.executor_config_error_code == "config_invalid"


def test_publish_verify_keyring_file_controls_publish_readiness(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    keyring_path = data_dir / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": "testkey01",
                    "public_key_hex": "11" * 32,
                    "active": True,
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)

    cfg = config.load_config(env, data_root=data_dir, run_root=run_dir)
    keyring = config.load_publish_verify_keyring(cfg)

    assert cfg.publish_verify_keys_file == keyring_path
    assert cfg.publish_verify_keyring_config_error_code is None
    assert keyring is not None
    assert keyring.current.key_id == "testkey01"


def test_publish_verify_keyring_file_preserves_key_time_windows(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    keyring_path = data_dir / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": "testkey01",
                    "public_key_hex": "11" * 32,
                    "active": True,
                    "not_before": "2026-06-14T11:00:00Z",
                    "not_after": "2030-06-14T13:00:00Z",
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)

    cfg = config.load_config(env, data_root=data_dir, run_root=run_dir)
    keyring = config.load_publish_verify_keyring(cfg)

    assert cfg.publish_verify_keyring_config_error_code is None
    assert keyring is not None
    assert keyring.current.not_before == datetime(
        2026,
        6,
        14,
        11,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert keyring.current.not_after == datetime(
        2030,
        6,
        14,
        13,
        0,
        0,
        tzinfo=timezone.utc,
    )
    with pytest.raises(publish_contract.PublishSpecError) as exc:
        keyring.key_for_id(
            "testkey01",
            now=datetime(2030, 6, 14, 13, 0, 0, tzinfo=timezone.utc),
        )
    assert exc.value.reason_code == "publish_spec_signature_key_expired"


def test_expired_publish_verify_keyring_blocks_publish_readiness(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    keyring_path = data_dir / "publish-verify.json"
    keyring_path.write_text(
        json.dumps(
            {
                "schema_version": "kitt_publish_verify_keys_v1",
                "current": {
                    "key_id": "testkey01",
                    "public_key_hex": "11" * 32,
                    "active": True,
                    "not_after": "2000-01-01T00:00:00Z",
                },
                "previous": None,
            }
        ),
        encoding="utf-8",
    )
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)

    cfg = config.load_config(env, data_root=data_dir, run_root=run_dir)

    assert cfg.publish_verify_keys_file == keyring_path
    assert (
        cfg.publish_verify_keyring_config_error_code
        == "publish_spec_signature_key_expired"
    )
    assert config.load_publish_verify_keyring(cfg) is None


def test_publish_verify_keyring_invalid_file_degrades_without_crash(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    keyring_path = data_dir / "publish-verify.json"
    keyring_path.write_text("{}", encoding="utf-8")
    keyring_path.chmod(0o600)
    env["KITT_WORKER_PUBLISH_VERIFY_KEYS_FILE"] = str(keyring_path)

    cfg = config.load_config(env, data_root=data_dir, run_root=run_dir)

    assert cfg.publish_verify_keys_file == keyring_path
    assert cfg.publish_verify_keyring_config_error_code == "config_invalid"
    assert config.load_publish_verify_keyring(cfg) is None


@pytest.mark.parametrize(
    ("env", "exc_type"),
    [
        ({"KITT_WORKER_BIND_HOST": "0.0.0.0"}, config.UnsafeTargetError),
        ({"KITT_WORKER_BIND_HOST": "localhost"}, config.UnsafeTargetError),
        ({"KITT_WORKER_BIND_HOST": "192.0.2.17"}, config.UnsafeTargetError),
        ({"KITT_WORKER_PORT": "11442"}, config.UnsafeTargetError),
        ({"KITT_WORKER_PORT": "not-a-port"}, config.RuntimeConfigError),
        ({"KITT_WORKER_HEALTH_PATH": "/v1/health"}, config.UnsafeTargetError),
        ({"KITT_WORKER_HEALTH_PATH": "/internal/ready"}, config.UnsafeTargetError),
        ({"KITT_WORKER_LOG_LEVEL": "verbose"}, config.RuntimeConfigError),
        ({"KITT_WORKER_AUTH_MODE": "optional"}, config.RuntimeConfigError),
        ({"KITT_WORKER_AUTH_MODE": " REQUIRED "}, config.RuntimeConfigError),
        ({"KITT_WORKER_AUTH_MODE": " disabled "}, config.RuntimeConfigError),
        ({"KITT_WORKER_TLS_TERMINATION": "reverse_proxy"}, config.RuntimeConfigError),
        ({"KITT_WORKER_CREDENTIALS_FILE": "credentials.json"}, config.RuntimeConfigError),
        ({"KITT_WORKER_CREDENTIALS_FILE": "/opt/kiron/credentials.json"}, config.RuntimeConfigError),
        ({"KITT_WORKER_ID": ""}, config.RuntimeConfigError),
        ({"KITT_WORKER_ID": "KIRON-KITT-WORKER"}, config.RuntimeConfigError),
        ({"KITT_WORKER_ID": "kiron/kitt-worker"}, config.RuntimeConfigError),
        ({"KITT_WORKER_ID": "token-worker"}, config.RuntimeConfigError),
        ({"KITT_WORKER_CONTRACT_VERSION": "adr-0008.v1"}, config.RuntimeConfigError),
        ({"KITT_WORKER_ENABLE_V1": "maybe"}, config.RuntimeConfigError),
    ],
)
def test_unsafe_or_invalid_config_fails_closed(env, exc_type):
    with pytest.raises(exc_type):
        config.load_config(env)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_enable_v1_flag_parses_boolean_values(raw, expected):
    cfg = config.load_config({"KITT_WORKER_ENABLE_V1": raw})

    assert cfg.enable_v1 is expected


def test_queue_config_parses_valid_non_secret_bounds():
    cfg = config.load_config(
        {
            "KITT_WORKER_QUEUE_MAX_JOBS": "7",
            "KITT_WORKER_LEASE_TTL_SECONDS": "30",
            "KITT_WORKER_RESUME_LIMIT": "0",
            "KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS": "5000",
        }
    )

    assert cfg.queue_max_jobs == 7
    assert cfg.lease_ttl_seconds == 30
    assert cfg.resume_limit == 0
    assert cfg.sqlite_busy_timeout_ms == 5000
    assert cfg.queue_config_error_code is None


def test_executor_config_defaults_closed_and_valid_enabled_gate(tmp_path):
    command = f"{sys.executable} {_write_sft_trainer(tmp_path)}"
    cfg = config.load_config(
        {
            "KITT_WORKER_DISPATCH_GATE": "open",
            "KITT_WORKER_EXECUTOR_ENABLED": "true",
            "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
            "KITT_WORKER_SFT_COMMAND": command,
            "KITT_WORKER_EXECUTOR_POLL_INTERVAL_SECONDS": "0.1",
            "KITT_WORKER_EXECUTOR_RENEW_INTERVAL_SECONDS": "10",
            "KITT_WORKER_EXECUTOR_JOB_TIMEOUT_SECONDS": "1",
            "KITT_WORKER_EXECUTOR_SHUTDOWN_GRACE_SECONDS": "1",
        }
    )

    assert cfg.dispatch_gate == "open"
    assert cfg.executor_enabled is True
    assert cfg.executor_runner == "sft_subprocess"
    assert cfg.sft_command == command
    assert cfg.executor_poll_interval_seconds == 0.1
    assert cfg.executor_renew_interval_seconds == 10
    assert cfg.executor_job_timeout_seconds == 1
    assert cfg.executor_shutdown_grace_seconds == 1
    assert cfg.executor_config_error_code is None


def test_sft_named_dummy_script_is_not_a_valid_executor_command(tmp_path):
    dummy = tmp_path / "kiron_sft_trainer.py"
    dummy.write_text(
        "\n".join(
            [
                "# Static marker strings must not make a script trusted.",
                'KIRON_SFT_TRAINER_ENTRYPOINT = "adr-0008.sft.v2"',
                "KITT_JOB_SPEC_PATH = 'KITT_JOB_SPEC_PATH'",
                "KITT_OUTPUT_DIR = 'KITT_OUTPUT_DIR'",
                "FILES = ('adapter_config.json', 'adapter_model.safetensors')",
                "PACKAGES = ('transformers', 'peft', 'trl')",
            ]
        ),
        encoding="utf-8",
    )

    cfg = config.load_config(
        {
            "KITT_WORKER_EXECUTOR_ENABLED": "true",
            "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
            "KITT_WORKER_SFT_COMMAND": f"{sys.executable} {dummy}",
        }
    )

    assert cfg.executor_config_error_code == "config_invalid"


def test_fake_python_interpreter_is_not_a_valid_executor_command(tmp_path):
    fake_python = tmp_path / "python-fake"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    cfg = config.load_config(
        {
            "KITT_WORKER_EXECUTOR_ENABLED": "true",
            "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
            "KITT_WORKER_SFT_COMMAND": f"{fake_python} {_write_sft_trainer(tmp_path)}",
        }
    )

    assert cfg.executor_config_error_code == "config_invalid"


@pytest.mark.parametrize(
    "env",
    [
        {"KITT_WORKER_EXECUTOR_ENABLED": "maybe"},
        {"KITT_WORKER_EXECUTOR_RUNNER": "fake"},
        {"KITT_WORKER_EXECUTOR_RUNNER": "real"},
        {
            "KITT_WORKER_EXECUTOR_ENABLED": "true",
            "KITT_WORKER_EXECUTOR_RUNNER": "sft_subprocess",
        },
        {"KITT_WORKER_SFT_COMMAND": sys.executable},
        {"KITT_WORKER_SFT_COMMAND": f"{sys.executable} -m kiron_sft_trainer"},
        {"KITT_WORKER_SFT_COMMAND": f"{sys.executable} relative_sft_trainer.py"},
        {"KITT_WORKER_SFT_COMMAND": "/bin/true"},
        {"KITT_WORKER_SFT_COMMAND": "/tmp/tokenized-runner"},
        {"KITT_WORKER_EXECUTOR_POLL_INTERVAL_SECONDS": "0.01"},
        {"KITT_WORKER_EXECUTOR_POLL_INTERVAL_SECONDS": "61"},
        {"KITT_WORKER_EXECUTOR_RENEW_INTERVAL_SECONDS": "0"},
        {"KITT_WORKER_EXECUTOR_JOB_TIMEOUT_SECONDS": "0"},
        {"KITT_WORKER_EXECUTOR_SHUTDOWN_GRACE_SECONDS": "301"},
        {
            "KITT_WORKER_EXECUTOR_ENABLED": "true",
            "KITT_WORKER_EXECUTOR_RENEW_INTERVAL_SECONDS": "200",
        },
    ],
)
def test_invalid_executor_config_degrades_executor_without_breaking_base_config(env):
    cfg = config.load_config(env)

    assert cfg.executor_config_error_code == "config_invalid"


def test_invalid_dispatch_gate_fails_closed():
    with pytest.raises(config.RuntimeConfigError):
        config.load_config({"KITT_WORKER_DISPATCH_GATE": "invalid"})


@pytest.mark.parametrize(
    "env",
    [
        {"KITT_WORKER_QUEUE_MAX_JOBS": "0"},
        {"KITT_WORKER_QUEUE_MAX_JOBS": "100001"},
        {"KITT_WORKER_LEASE_TTL_SECONDS": "29"},
        {"KITT_WORKER_LEASE_TTL_SECONDS": "86401"},
        {"KITT_WORKER_RESUME_LIMIT": "-1"},
        {"KITT_WORKER_RESUME_LIMIT": "11"},
        {"KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS": "99"},
        {"KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS": "5001"},
        {"KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS": "not-an-int"},
    ],
)
def test_invalid_queue_config_degrades_queue_without_breaking_base_config(env):
    cfg = config.load_config(env)

    assert cfg.queue_config_error_code == "config_invalid"


def test_artifact_config_parses_non_secret_bounds_under_data_root(tmp_path):
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    staging = data_root / "staging"
    data_root.mkdir()
    run_root.mkdir()

    cfg = config.load_config(
        {
            "KITT_WORKER_DATA_DIR": str(data_root),
            "KITT_WORKER_RUN_DIR": str(run_root),
            "KITT_WORKER_ARTIFACT_STAGING_DIR": str(staging),
            "KITT_WORKER_ARTIFACT_MAX_BYTES": "10",
            "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "20",
            "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "30",
            "KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "0",
            "KITT_WORKER_ARTIFACT_CLEANUP_AFTER_SECONDS": "60",
        },
        data_root=data_root,
        run_root=run_root,
    )

    assert cfg.artifact_staging_dir == staging.resolve(strict=False)
    assert cfg.artifact_max_bytes == 10
    assert cfg.artifact_job_quota_bytes == 20
    assert cfg.artifact_total_quota_bytes == 30
    assert cfg.artifact_min_free_bytes == 0
    assert cfg.artifact_cleanup_after_seconds == 60
    assert cfg.artifact_config_error_code is None


@pytest.mark.parametrize(
    "env",
    [
        {"KITT_WORKER_ARTIFACT_STAGING_DIR": "/opt/kiron/staging"},
        {"KITT_WORKER_ARTIFACT_STAGING_DIR": "relative/staging"},
        {"KITT_WORKER_ARTIFACT_MAX_BYTES": "0"},
        {"KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "0"},
        {"KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "0"},
        {"KITT_WORKER_ARTIFACT_MIN_FREE_BYTES": "-1"},
        {"KITT_WORKER_ARTIFACT_CLEANUP_AFTER_SECONDS": "-1"},
        {
            "KITT_WORKER_ARTIFACT_MAX_BYTES": "30",
            "KITT_WORKER_ARTIFACT_JOB_QUOTA_BYTES": "20",
            "KITT_WORKER_ARTIFACT_TOTAL_QUOTA_BYTES": "40",
        },
    ],
)
def test_invalid_artifact_config_degrades_staging_without_breaking_base_config(env):
    cfg = config.load_config(env)

    assert cfg.artifact_config_error_code in {
        "artifact_staging_unavailable",
        "artifact_quota_invalid",
    }


def _credential_tree(tmp_path: Path):
    root = tmp_path / "etc" / "kiron"
    credential_dir = root / "kitt-worker"
    credential_dir.mkdir(parents=True)
    root.chmod(0o755)
    credential_dir.chmod(0o750)
    credentials_file = credential_dir / "credentials.json"
    kid = _random_text(12)
    secret = _random_text(48)
    credential = f"{kid}.{secret}"
    credentials_file.write_text(
        json.dumps(
            {
                "version": 1,
                "current": {
                    "kid": kid,
                    "token_hash_sha256": auth.credential_hash(credential),
                    "scopes": ["read"],
                },
                "revoked_kids": [],
            }
        ),
        encoding="utf-8",
    )
    credentials_file.chmod(0o640)
    return root, credential_dir, credentials_file


def test_auth_required_fails_closed_without_credential_file(tmp_path):
    root = tmp_path / "etc" / "kiron"
    credential_dir = root / "kitt-worker"
    credential_dir.mkdir(parents=True)
    root.chmod(0o755)
    credential_dir.chmod(0o750)
    credentials_file = credential_dir / "credentials.json"

    with pytest.raises(config.RuntimeConfigError):
        config.load_config(
            {
                "KITT_WORKER_CREDENTIALS_FILE": str(credentials_file),
            },
            validate_auth=True,
            credential_root=credential_dir,
            credential_dir=credential_dir,
            expected_credential_root_uid=os.getuid(),
            expected_credential_root_gid=os.getgid(),
            expected_credential_dir_uid=os.getuid(),
            expected_credential_dir_gid=os.getgid(),
            expected_credential_file_uid=os.getuid(),
            expected_credential_file_gid=os.getgid(),
        )


def test_auth_disabled_is_rollback_mode_without_credential_file(tmp_path):
    root = tmp_path / "etc" / "kiron"
    credential_dir = root / "kitt-worker"
    credential_dir.mkdir(parents=True)
    credentials_file = credential_dir / "credentials.json"

    cfg = config.load_config(
        {
            "KITT_WORKER_AUTH_MODE": "disabled",
            "KITT_WORKER_CREDENTIALS_FILE": str(credentials_file),
        },
        validate_auth=True,
        credential_root=credential_dir,
        credential_dir=credential_dir,
    )

    assert cfg.auth_mode == "disabled"


def test_v1_disabled_skips_credential_store_requirement(tmp_path):
    credential_dir = tmp_path / "etc" / "kiron" / "kitt-worker"
    credential_dir.mkdir(parents=True)
    credentials_file = credential_dir / "credentials.json"

    cfg = config.load_config(
        {
            "KITT_WORKER_ENABLE_V1": "false",
            "KITT_WORKER_CREDENTIALS_FILE": str(credentials_file),
        },
        validate_auth=True,
        credential_root=credential_dir,
        credential_dir=credential_dir,
    )

    assert cfg.enable_v1 is False


def test_auth_required_accepts_valid_credential_file(tmp_path):
    root, credential_dir, credentials_file = _credential_tree(tmp_path)

    cfg = config.load_config(
        {
            "KITT_WORKER_CREDENTIALS_FILE": str(credentials_file),
        },
        validate_auth=True,
        credential_root=credential_dir,
        credential_dir=credential_dir,
        expected_credential_root_uid=os.getuid(),
        expected_credential_root_gid=os.getgid(),
        expected_credential_dir_uid=os.getuid(),
        expected_credential_dir_gid=os.getgid(),
        expected_credential_file_uid=os.getuid(),
        expected_credential_file_gid=os.getgid(),
    )

    assert cfg.credentials_file == credentials_file
    assert root


def test_runtime_paths_must_exist_with_required_owner_and_mode(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)

    cfg = config.load_config(
        env,
        validate_runtime=True,
        data_root=data_dir,
        run_root=run_dir,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert cfg.data_dir == data_dir.resolve()
    assert cfg.run_dir == run_dir.resolve()


def test_runtime_path_wrong_mode_fails(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    data_dir.chmod(0o755)

    with pytest.raises(config.RuntimeConfigError):
        config.load_config(
            env,
            validate_runtime=True,
            data_root=data_dir,
            run_root=run_dir,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_runtime_path_outside_allowed_root_fails(tmp_path):
    data_root = tmp_path / "data"
    run_root = tmp_path / "run"
    outside = tmp_path / "outside"
    data_root.mkdir()
    run_root.mkdir()
    outside.mkdir()

    with pytest.raises(config.RuntimeConfigError):
        config.load_config(
            {
                "KITT_WORKER_DATA_DIR": str(outside),
                "KITT_WORKER_RUN_DIR": str(run_root),
            },
            data_root=data_root,
            run_root=run_root,
        )


def test_internal_health_stays_plain_and_auth_disabled_v1_fails_closed(tmp_path):
    env, data_dir, run_dir = _runtime_env(tmp_path)
    env["KITT_WORKER_AUTH_MODE"] = "disabled"
    app = main.create_app(
        validate_runtime=True,
        load_config_kwargs={
            "env": env,
            "data_root": data_dir,
            "run_root": run_dir,
            "expected_uid": os.getuid(),
            "expected_gid": os.getgid(),
        },
    )

    with TestClient(app) as client:
        response = client.get("/internal/health")
        disabled = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "service": "kitt-worker",
        "mode": "skeleton",
    }
    body = json.dumps(payload).lower()
    assert str(tmp_path).lower() not in body
    assert "/usr/lib/kiron" not in body
    assert "/run/kiron" not in body
    for forbidden in ("secret", "token", "job", "queue", "artifact"):
        assert forbidden not in body
    assert disabled.status_code == 503
    disabled_payload = disabled.json()
    assert disabled_payload["ok"] is False
    assert disabled_payload["error_code"] == "v1_disabled"
    assert disabled_payload["worker_contract_version"] == "adr-0008.v2"
    assert disabled_payload["request_id"] == disabled.headers["X-Request-ID"]
    assert {path.name for path in data_dir.iterdir()}.issubset(
        {"queue.sqlite3", "queue.sqlite3-wal", "queue.sqlite3-shm"}
    )
    assert list(run_dir.iterdir()) == []
