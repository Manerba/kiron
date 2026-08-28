from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_kitt_worker_unit_uses_unprivileged_runtime_and_internal_healthcheck():
    unit = _read("systemd/kitt-worker.service")

    assert "User=kitt-worker" in unit
    assert "Group=kitt-worker" in unit
    assert "Wants=network-online.target" in unit
    assert "After=network-online.target" in unit
    assert "After=network.target" not in unit
    assert "Restart=on-failure" in unit
    assert "WatchdogSec" not in unit
    assert "WorkingDirectory=/usr/lib/kiron/services/kitt-worker" in unit
    assert "ExecStart=/usr/lib/kiron/services/kitt-worker/venv/bin/python main.py" in unit
    assert (
        "ExecStartPost=/usr/lib/kiron/services/kitt-worker/venv/bin/python "
        "/usr/lib/kiron/services/kitt-worker/healthcheck.py --timeout 20 --interval 0.25"
    ) in unit
    assert "Environment=KITT_WORKER_BIND_HOST=localhost" in unit
    assert "Environment=KITT_WORKER_PORT=11441" in unit
    assert "Environment=KITT_WORKER_HEALTH_PATH=/internal/health" in unit
    assert "Environment=KITT_WORKER_DATA_DIR=/usr/lib/kiron/data/kitt-worker" in unit
    assert "Environment=KITT_WORKER_RUN_DIR=/run/kiron/kitt-worker" in unit
    assert "Environment=KITT_WORKER_ID=kiron-kitt-worker" in unit
    assert "Environment=KITT_WORKER_CONTRACT_VERSION=adr-0008.v2" in unit
    assert "Environment=KITT_WORKER_ENABLE_V1=true" in unit
    assert "Environment=KITT_WORKER_AUTH_MODE=required" in unit
    assert (
        "Environment=KITT_WORKER_CREDENTIALS_FILE=/etc/kiron/kitt-worker/credentials.json"
        in unit
    )
    assert "KITT_WORKER_TLS_TERMINATION" not in unit
    assert "reverse_proxy" not in unit
    assert "Environment=KITT_WORKER_QUEUE_MAX_JOBS=1000" in unit
    assert "Environment=KITT_WORKER_LEASE_TTL_SECONDS=300" in unit
    assert "Environment=KITT_WORKER_RESUME_LIMIT=3" in unit
    assert "Environment=KITT_WORKER_SQLITE_BUSY_TIMEOUT_MS=1000" in unit
    assert "Environment=KITT_WORKER_DISPATCH_GATE=closed" in unit
    assert "Environment=KITT_WORKER_EXECUTOR_ENABLED=false" in unit
    assert "Environment=KITT_WORKER_EXECUTOR_RUNNER=sft_subprocess" in unit
    assert (
        'Environment="KITT_WORKER_SFT_COMMAND='
        "/usr/lib/kiron/services/kitt-worker/venv/bin/python "
        '/usr/lib/kiron/services/kitt-worker/sft_trainer.py"'
        in unit
    )
    assert "Environment=KITT_WORKER_MONITORING_ENABLED=true" in unit
    assert "Environment=KITT_WORKER_ALERTS_ENABLED=true" in unit
    assert "Environment=KITT_WORKER_MONITORING_INTERVAL_SECONDS=30" in unit
    assert "Environment=KITT_WORKER_HEARTBEAT_STALE_SECONDS=300" in unit
    assert "Environment=KITT_WORKER_ALERT_REPEAT_SECONDS=3600" in unit
    assert "PrivateDevices=false" in unit
    assert "PrivateDevices=true" not in unit
    assert "ProtectSystem=strict" in unit
    assert "ReadWritePaths=/usr/lib/kiron/data/kitt-worker /run/kiron/kitt-worker" in unit
    forbidden = (
        "TOKEN=",
        "SECRET=",
        "PASSWORD=",
        "PRIVATE_KEY",
        "TLS_KEY",
        "TLS_CERT",
        "SSL_KEY",
        "SSL_CERT",
        "ssl_keyfile",
        "ssl_certfile",
    )
    for value in forbidden:
        assert value not in unit


def test_kitt_worker_code_does_not_configure_worker_tls():
    main_py = _read("services/kitt-worker/main.py")
    config_py = _read("services/kitt-worker/config.py")

    for text in (main_py, config_py):
        assert "ssl_keyfile" not in text
        assert "ssl_certfile" not in text
        assert "ssl_ca_certs" not in text
        assert "KITT_WORKER_TLS_KEY" not in text
        assert "KITT_WORKER_TLS_CERT" not in text
        assert "tls_termination" not in text
        assert "reverse_proxy" not in text
    assert "KITT_WORKER_TLS_TERMINATION" not in main_py
    assert "Sprint 11 uses internal HTTP" in config_py
    assert "transport_mode" in config_py
    assert "internal_http" in config_py


def test_install_systemd_installs_and_drift_checks_kitt_worker():
    script = _read("scripts/install-systemd.sh")

    assert "for unit in /opt/kiron/systemd/*.service" in script
    assert "/etc/systemd/system/kitt-worker.service" in script
    assert 'systemctl enable --now kitt-worker   # erst nach Operatorfreigabe' in script
