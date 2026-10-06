"""Offline install contracts; execute only extracted code against temp fixtures."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text()


def shell_function(text, name):
    return text.split(name + "() {", 1)[1].split("\n}\n", 1)[0]


def test_service_groups_match_units_and_both_preflights():
    expected = {
        "kiron-embeddings": {"kiron-models", "kiron-config", "kiron-common", "kiron-runtime", "video", "render"},
        "kiron-prism": {"kiron-common", "kiron-config", "kiron-runtime", "kiron-prism-control", "video", "render"},
        "kiron-proxy": {"docker", "kiron-runtime", "kiron-common", "kiron-config", "kiron-prism-control"},
        "kitt-worker": {"kiron-runtime"},
    }
    for service, groups in expected.items():
        unit = read(f"systemd/{service}.service")
        assert set(re.search(r"^SupplementaryGroups=(.+)$", unit, re.M)[1].split()) == groups
        assert f"User={service}\n" in unit and f"Group={service}\n" in unit
        for script in ("scripts/deploy-local.sh", "scripts/setup-venvs.sh"):
            body = shell_function(read(script), "required_identity_groups")
            result = subprocess.run(["bash", "-c", "role() {" + body + "\n}; role " + service],
                                    capture_output=True, text=True, timeout=5, check=True)
            assert set(result.stdout.split()) == groups
    installer = read("scripts/install-system-configs.sh")
    assert "usermod -G kiron-runtime kitt-worker" in installer
    assert '[ "$groups" = "kiron-runtime kitt-worker" ]' in installer
    assert " /run/kiron/vram\n" in read("systemd/kitt-worker.service")
    assert "ReadWritePaths=/var/cache/kiron/huggingface /run/kiron/vram\n" in read("systemd/kiron-embeddings.service")


def test_runtime_paths_and_service_lifecycle_are_installed_without_test_venvs():
    paths = set(read("system/tmpfiles.d/kiron-runtime.conf").splitlines())
    assert "d /run/kiron/prism 2750 kiron-prism kiron-prism-control -" in paths
    assert "d /usr/lib/kiron/data/gguf-models 2750 root kiron-common -" in paths
    assert "z /usr/lib/kiron/data/prism-runtime-policy.json 0640 root kiron-config -" in paths
    for script, variable in (("scripts/deploy-local.sh", "KIRON_SERVICES"),
                             ("scripts/setup-venvs.sh", "SERVICES")):
        source = read(script)
        services = re.search(r"^" + variable + r"=\(([^)]+)\)", source, re.M)[1].split()
        assert "kiron-prism" in services
        assert "test-venvs" not in source and "test-runtimes" not in source
    setup = read("scripts/setup-venvs.sh")
    assert '[ "$svc" = "kiron-prism" ]; then' in setup
    assert '"$venv_new/bin/pip" install "$COMMON_SRC"' in setup
    assert "import main, controller, composition" in setup
    deploy = read("scripts/deploy-local.sh")
    call = deploy.index('    check_kitt_common_snapshot "$SRC/services/kiron-common/kiron_common"')
    assert call < deploy.index('# Verzeichnisstruktur sicherstellen')
    assert deploy.index('    check_prism_restart_prereqs\n') < deploy.index('# Verzeichnisstruktur sicherstellen')
    body = shell_function(deploy, "check_prism_restart_prereqs")
    assert "Policy.load" in body and "RuntimeModelRegistry(readonly=True).list()" in body
    assert "runuser -u kiron-prism" in body
    assert "systemctl" not in body


def test_deploy_shares_only_prism_revision_sources(tmp_path):
    """Execute the permission dispatcher without touching host ownership."""
    script = read("scripts/deploy-local.sh")
    body = shell_function(script, "apply_code_permissions")
    commands = '''
service_group() { echo "$1"; }
apply_readonly_tree_permissions() { printf 'tree %s %s\n' "$1" "$2"; }
chgrp() { printf 'share %s %s\n' "$1" "$2"; }
'''
    result = subprocess.run(
        ["bash", "-eu", "-c", commands + '\nDST="$1"\n'
         'KIRON_SERVICES=(kiron-proxy kiron-prism)\napply_code_permissions() {'
         + body + '\n}\napply_code_permissions', "fixture", str(tmp_path)],
        check=True, text=True, capture_output=True, timeout=5,
    )
    prism = tmp_path / "services/kiron-prism"
    assert result.stdout.splitlines() == [
        f"tree {tmp_path}/services/kiron-common kiron-common",
        f"tree {tmp_path}/services/kiron-proxy kiron-proxy",
        f"tree {prism} kiron-prism",
        *[f"share kiron-prism-control {path}" for path in
          (prism, *(prism / name for name in
                    ("main.py", "composition.py", "controller.py", "process.py", "admission.py")))],
    ]


def registry_initializer(directory):
    body = shell_function(read("scripts/install-system-configs.sh"), "initialize_local_model_registry")
    code = body.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    # Replace only the installation target/account lookup; run the real code.
    code = code.replace('pwd.getpwnam("kiron-proxy").pw_uid', 'os.getuid()')
    code = code.replace('grp.getgrnam("kiron-common").gr_gid', 'os.getgid()')
    code = code.replace('"/usr/lib/kiron/data/shared"', repr(str(directory)))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)


def test_registry_bootstrap_is_valid_idempotent_and_preserves_existing_bytes(tmp_path):
    assert registry_initializer(tmp_path).returncode == 0
    registry = tmp_path / "local-model-registry.json"
    lock = registry.with_suffix(".json.lock")
    assert json.loads(registry.read_bytes()) == {"version": 2, "entries": []}
    for path in (registry, lock):
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
        assert path.stat().st_uid == os.getuid() and path.stat().st_gid == os.getgid()
    original_inode = lock.stat().st_ino
    registry.write_bytes(b"invalid existing evidence must not be repaired")
    assert registry_initializer(tmp_path).returncode == 0
    assert registry.read_bytes() == b"invalid existing evidence must not be repaired"
    assert lock.stat().st_ino == original_inode


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "busy_lock"])
def test_registry_bootstrap_rejects_hostile_or_busy_storage(tmp_path, kind):
    registry = tmp_path / "local-model-registry.json"
    outside = tmp_path / "untouched"
    outside.write_bytes(b"sentinel")
    outside.chmod(0o640)
    lock_stream = None
    if kind == "symlink":
        registry.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, registry)
    elif kind == "fifo":
        os.mkfifo(registry)
    else:
        assert registry_initializer(tmp_path).returncode == 0
        lock_stream = registry.with_suffix(".json.lock").open("rb")
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
    try:
        assert registry_initializer(tmp_path).returncode != 0
    finally:
        if lock_stream is not None:
            lock_stream.close()
    assert outside.read_bytes() == b"sentinel"


def test_privileged_helper_only_accepts_the_exact_prism_verbs(tmp_path):
    called = tmp_path / "called"
    fake_systemctl = tmp_path / "systemctl"
    fake_systemctl.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > ' + str(called) + '\n')
    fake_systemctl.chmod(0o700)
    helper = tmp_path / "helper"
    helper.write_text(read("system/sbin/kiron-service-control").replace("/usr/bin/systemctl", str(fake_systemctl)))
    for verb in ("start", "stop", "restart"):
        result = subprocess.run(["bash", str(helper), verb, "kiron-prism.service"], timeout=5)
        assert result.returncode == 0 and called.read_text().splitlines() == [verb, "kiron-prism.service"]
        called.unlink()
    for args in (("enable", "kiron-prism.service"), ("restart", "other.service"),
                 ("restart", "kiron-prism.service", "extra"), ("restart", "kiron-prism@other.service")):
        result = subprocess.run(["bash", str(helper), *args], capture_output=True, timeout=5)
        assert result.returncode == 64 and not called.exists()


def test_prism_restart_preflight_stops_on_failed_policy_validation(tmp_path):
    binary = tmp_path / "services/kiron-prism/venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.write_text("unused fixture")
    binary.chmod(0o700)
    body = shell_function(read("scripts/deploy-local.sh"), "check_prism_restart_prereqs")
    script = 'DST="$1"\nrunuser() { return 73; }\ncheck() {' + body + '\n}\ncheck'
    result = subprocess.run(["bash", "-c", script, "fixture", str(tmp_path)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1 and "nicht startbereit" in result.stderr


@pytest.mark.parametrize("fault", [None, "digest", "status", "version", "safety"])
def test_compat_report_copy_reuses_gate_and_preserves_exact_bytes(tmp_path, fault):
    body = shell_function(read("scripts/deploy-local.sh"), "install_compat_report")
    code = body.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    code = code.replace('grp.getgrnam("kiron-common").gr_gid', 'os.getgid()')
    code = code.replace('directory_info.st_uid != 0', 'directory_info.st_uid != os.getuid()')
    code = code.replace('os.chown(directory, 0, gid)', 'os.chown(directory, os.getuid(), gid)')
    code = code.replace('os.fchown(stream.fileno(), 0, gid)', 'os.fchown(stream.fileno(), os.getuid(), gid)')
    report = {"image": "test:1", "image_digest": "test@sha256:123", "report_status": "passed",
              "upgrade_allowed": True, "validator_version": "v1", "capabilities": {
                  "num_gpu_zero_chat_generate": {"data": {"num_gpu_zero_effective": True}},
                  "version_endpoint": {"ok": True, "data": {"version": "1"}}}}
    if fault == "digest":
        report["image_digest"] = "different"
    elif fault == "status":
        report["report_status"] = "failed"
    elif fault == "version":
        report["validator_version"] = "unreviewed"
    elif fault == "safety":
        report["capabilities"]["num_gpu_zero_chat_generate"]["data"]["num_gpu_zero_effective"] = False
    original = (json.dumps(report, indent=3) + "\n\n").encode()
    source = tmp_path / "source.json"
    source.write_bytes(original)
    destination = tmp_path / "runtime-reports"
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts/check-ollama-compat.py"),
                             str(source), "test:1", "test@sha256:123", "true", str(destination)],
                            capture_output=True, text=True, timeout=5)
    if fault:
        assert result.returncode != 0 and not destination.exists()
    else:
        assert result.returncode == 0, result.stderr
        installed = Path(result.stdout.strip())
        assert installed == destination / ("sha256-" + hashlib.sha256(original).hexdigest() + ".json")
        assert installed.read_bytes() == original
        assert stat.S_IMODE(installed.stat().st_mode) == 0o640
        assert stat.S_IMODE(destination.stat().st_mode) == 0o750
