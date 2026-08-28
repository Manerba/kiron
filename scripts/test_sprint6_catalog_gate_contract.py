from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_restart_catalog_gate_precedes_deploy_complete_marker() -> None:
    script = (ROOT / "scripts/deploy-local.sh").read_text(encoding="utf-8")

    assert "check_catalog_consistency_after_restart" in script
    assert '"$proxy_python" "$checker" --wait-seconds 30' in script
    restart = script.index('systemctl restart "$svc"')
    gate = script.rindex("check_catalog_consistency_after_restart")
    marker = script.index('cp "$SRC/version.txt" "$DST/version.txt"')
    assert restart < gate < marker


def test_restart_gate_uses_the_injectable_fail_closed_checker() -> None:
    checker = (ROOT / "services/kiron-proxy/catalog_health.py").read_text(
        encoding="utf-8"
    )

    assert "check_catalog_digests(" in checker
    assert "def evaluate_catalog_health(" in checker
    assert "def check_live_catalog_consistency(" in checker
    assert "fetcher:" in checker
    assert 'return 0 if report["consistent"] else 1' in checker
