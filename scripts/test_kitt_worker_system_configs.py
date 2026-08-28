from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_tmpfiles_declares_worker_runtime_paths_without_queue_or_artifact_files():
    tmpfiles = _read("system/tmpfiles.d/kiron-runtime.conf")

    assert "d /run/kiron 0755 root root -" in tmpfiles
    assert "d /run/kiron/kitt-worker 0750 kitt-worker kitt-worker -" in tmpfiles
    assert "d /usr/lib/kiron/data/kitt-worker 0750 kitt-worker kitt-worker -" in tmpfiles
    assert "d /usr/lib/kiron/data/kitt-worker/staging 0750 kitt-worker kitt-worker -" in tmpfiles
    assert "d /usr/lib/kiron/data/kitt-worker/work 0750 kitt-worker kitt-worker -" in tmpfiles
    assert "queue" not in tmpfiles.lower()
    assert "artifact" not in tmpfiles.lower()


def test_install_system_configs_bootstraps_user_before_tmpfiles():
    script = _read("scripts/install-system-configs.sh")

    identity_call = script.index("ensure_kitt_worker_identity\nverify_kitt_worker_identity")
    tmpfiles_install = script.index("systemd-tmpfiles --create /etc/tmpfiles.d/kiron-runtime.conf")
    assert identity_call < tmpfiles_install
    assert 'tmpfiles_src="/opt/kiron/system/tmpfiles.d/kiron-runtime.conf"' in script
    assert '[ -f "$tmpfiles_src" ] || fail "Pflicht-tmpfiles fehlt: $tmpfiles_src"' in script
    assert "if [ -f /opt/kiron/system/tmpfiles.d/kiron-runtime.conf ]; then" not in script
    assert "cp \"$tmpfiles_src\" /etc/tmpfiles.d/kiron-runtime.conf" in script
    assert "groupadd --system kitt-worker" in script
    assert "useradd \\" in script
    assert "--system" in script
    assert "--gid kitt-worker" in script
    assert "--home-dir /nonexistent" in script
    assert "--no-create-home" in script
    assert "--shell /usr/sbin/nologin" in script
    assert "id -nG kitt-worker" in script
    assert "passwd -S kitt-worker" in script
    assert "verify_kitt_worker_runtime_dir /usr/lib/kiron/data/kitt-worker" in script
    assert "verify_kitt_worker_runtime_dir /run/kiron/kitt-worker" in script
    assert "ensure_kitt_worker_secret_dir" in script
    assert "verify_not_symlink /etc/kiron" in script
    assert "verify_not_symlink /etc/kiron/kitt-worker" in script
    assert "verify_not_symlink /etc/kiron/kitt-worker/credentials.json" in script
    assert "mkdir -p /etc/kiron" in script
    assert "chown root:root /etc/kiron" in script
    assert "chmod 0755 /etc/kiron" in script
    assert "mkdir -p /etc/kiron/kitt-worker" in script
    assert "chown root:kitt-worker /etc/kiron/kitt-worker" in script
    assert "chmod 0750 /etc/kiron/kitt-worker" in script
    assert "root:kitt-worker 640" in script
    assert "kitt-worker:kitt-worker 750" in script
    assert "touch /etc/kiron/kitt-worker/credentials.json" not in script
    assert "cat > /etc/kiron/kitt-worker/credentials.json" not in script
    assert "> /etc/kiron/kitt-worker/credentials.json" not in script
