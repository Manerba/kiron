from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


ALLOW_DISPATCH_TRUTHY_RE = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])
    ["']?(?:[A-Za-z0-9_]*_)?allow_dispatch["']?
    \s*(?:=|:)\s*
    ["']?(?:true|1|yes|on)["']?
    (?![A-Za-z0-9_])
    """
)


def test_kitt_worker_keeps_kitt_allow_dispatch_external_and_operator_gated():
    service_files = [
        "services/kitt-worker/main.py",
        "services/kitt-worker/config.py",
        "services/kitt-worker/auth.py",
        "services/kitt-worker/contract.py",
        "services/kitt-worker/healthcheck.py",
        "services/kitt-worker/job_stubs.py",
        "services/kitt-worker/queue_store.py",
        "services/kitt-worker/artifact_staging.py",
        "services/kitt-worker/capabilities.py",
        "services/kitt-worker/gpu_policy.py",
        "services/kitt-worker/runners.py",
        "services/kitt-worker/executor.py",
        "services/kitt-worker/monitoring.py",
        "services/kitt-worker/v1.py",
        "systemd/kitt-worker.service",
    ]
    integration_files = [
        "scripts/deploy-local.sh",
        "scripts/setup-venvs.sh",
        "scripts/install-systemd.sh",
        "scripts/install-system-configs.sh",
    ]
    service_text = "\n".join(_read(path) for path in service_files)
    integration_text = "\n".join(_read(path) for path in integration_files)

    assert not ALLOW_DISPATCH_TRUTHY_RE.search(service_text)
    assert not ALLOW_DISPATCH_TRUTHY_RE.search(integration_text)
    assert "app.include_router(v1.router)" in service_text
    assert "worker_contract_version" in service_text
    assert "valid_for_scheduling" in service_text
    assert "KITT_WORKER_DISPATCH_GATE" in service_text
    assert "dispatch_gate_open" in service_text
    assert "OPEN_SCHEDULING_GATE_REASON" in service_text
    assert "OPEN_POLICY_DECISION" in service_text
    for forbidden in (
        "queue.db",
        "queue.sqlite3-journal",
        "run_training",
        "execute_job",
        "download_url",
        "upload_url",
    ):
        assert forbidden not in service_text.lower()
    assert not re.search(r"queue[.]sqlite(?!3)", service_text.lower())
    assert '"queue.sqlite3"' in service_text
    assert '"queue_enabled": queue_status["queue_enabled"] is True' in service_text

    assert 'for queue_file in "$DST/data/kitt-worker"/queue*; do' in integration_text
    assert "queue.sqlite3|queue.sqlite3-wal|queue.sqlite3-shm" in integration_text
    assert "staging" in integration_text
    assert "work" in integration_text


def test_install_hint_keeps_kitt_worker_operator_gated():
    script = _read("scripts/install-systemd.sh")

    assert "systemctl enable --now kiron-proxy kiron-docling kiron-embeddings kiron-deberta" in script
    assert "systemctl enable --now kitt-worker   # erst nach Operatorfreigabe" in script
