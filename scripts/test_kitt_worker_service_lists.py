from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _extract_shell_function(text: str, name: str) -> str:
    start = text.index(f"{name}() {{")
    depth = 0
    lines = []
    for line in text[start:].splitlines():
        lines.append(line)
        depth += line.count("{") - line.count("}")
        if lines and depth == 0:
            return "\n".join(lines)
    raise AssertionError(f"shell function not closed: {name}")


def test_deploy_local_knows_kitt_worker_everywhere():
    text = _read("scripts/deploy-local.sh")

    assert "KIRON_SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kitt-worker)" in text
    assert '"$SRC/services/$svc/requirements.txt"' in text
    assert "check_kitt_worker_restart_prereqs" in text
    assert "check_kitt_worker_auth_restart_preflight" in text
    assert "KITT_WORKER_ENABLE_V1" in text
    assert "check_kitt_worker_credentials_store_loadable" in text
    assert "check_kitt_worker_no_dispatch_activation" in text
    assert 'scan_paths=("$@")' in text
    assert '"/etc/systemd/system/kitt-worker.service.d"' in text
    assert 'suffixes = {".py", ".json", ".yaml", ".yml", ".service", ".sh", ".conf"}' in text
    assert 'return path.name == ".env" or path.suffix in suffixes' in text
    assert 'check_kitt_worker_no_dispatch_activation "$SRC/services/kitt-worker" "$SRC/scripts" "$SRC/systemd"' in text
    assert 'check_kitt_worker_no_dispatch_activation "$DST/services/kitt-worker"' in text
    assert '"/etc/systemd/system/kitt-worker.service" \\' in text
    assert "kitt_worker_effective_env_value KITT_WORKER_AUTH_MODE" in text
    assert "kitt_worker_effective_env_value KITT_WORKER_CREDENTIALS_FILE" in text
    assert "kitt_worker_effective_env_value KITT_WORKER_ENABLE_V1" in text
    assert "validate_kitt_worker_credentials_path" in text
    assert "path.parent != credential_dir" in text
    assert "resolved.is_relative_to(root)" in text
    assert "auth_mode=\"required\"" in text
    assert "enable_v1=\"true\"" in text
    assert "kitt-worker /v1 disabled: Rollback-Modus" in text
    assert "Auth disabled: /v1 Rollback-Modus" in text
    assert "CredentialStore ist im Auth-required-Normalbetrieb nicht ladbar" in text
    assert "KITT_WORKER_CREDENTIALS_FILE fehlt in der effektiven systemd-Umgebung" in text
    assert "KITT_WORKER_CREDENTIALS_FILE muss direkt unter /etc/kiron/kitt-worker liegen" in text
    assert "check_not_symlink /etc/kiron" in text
    assert "check_not_symlink /etc/kiron/kitt-worker" in text
    assert "check_not_symlink \"$credentials_file\"" in text
    assert "root:root 755" in text
    assert "root:kitt-worker 750" in text
    assert "root:kitt-worker 640" in text
    assert "check_kitt_worker_no_external_routes" not in text
    assert 'mkdir -p "$DST/services/$svc"' in text
    assert 'sync_service "$svc" --exclude=' in text
    assert 'systemctl restart "$svc"' in text
    assert "getent group kitt-worker" in text
    assert "id -u kitt-worker" in text
    assert 'stat -c \'%U:%G %a\'' in text
    daemon_reload = text.index("systemctl daemon-reload")
    auth_preflight = text.index('    check_kitt_worker_auth_restart_preflight', daemon_reload)
    restart = text.index('systemctl restart "$svc"')
    source_dispatch_preflight = text.index(
        'check_kitt_worker_no_dispatch_activation "$SRC/services/kitt-worker" "$SRC/scripts" "$SRC/systemd"'
    )
    docker_pull = text.index('docker pull "$NEW_OLLAMA_IMAGE"')
    mkdirs = text.index('mkdir -p "$DST/services/kiron-common"')
    service_target_preflight = text.index(
        'check_kitt_worker_no_dispatch_activation "$DST/services/kitt-worker"'
    )
    systemd_copy = text.index('cp "$SRC/systemd/"*.service')
    systemd_target_preflight = text.index('check_kitt_worker_no_dispatch_activation \\\n')
    assert source_dispatch_preflight < docker_pull
    assert source_dispatch_preflight < mkdirs
    assert service_target_preflight < systemd_copy
    assert systemd_copy < systemd_target_preflight < daemon_reload
    assert daemon_reload < auth_preflight < restart


def test_deploy_credential_path_validator_rejects_nested_and_dotdot_paths():
    text = _read("scripts/deploy-local.sh")
    function = _extract_shell_function(text, "validate_kitt_worker_credentials_path")

    def run(path: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", f"{function}\nvalidate_kitt_worker_credentials_path \"$1\"", "bash", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    assert run("/etc/kiron/kitt-worker/credentials.json").returncode == 0
    assert run("/etc/kiron/kitt-worker/alternate.json").returncode == 0
    assert run("/etc/kiron/kitt-worker/nested/credentials.json").returncode != 0
    assert run("/etc/kiron/kitt-worker/../kitt-worker/credentials.json").returncode != 0
    assert run("/etc/kiron/credentials.json").returncode != 0
    assert run("credentials.json").returncode != 0


def test_setup_venvs_runs_kitt_worker_smoke_as_service_user():
    text = _read("scripts/setup-venvs.sh")
    required_files = "for required in main.py config.py auth.py contract.py job_stubs.py queue_store.py artifact_staging.py capabilities.py gpu_policy.py runners.py sft_command.py sft_trainer.py executor.py monitoring.py v1.py healthcheck.py; do"
    import_smoke_command = (
        "(cd \"$DST/services/kitt-worker\" && runuser -u kitt-worker -- "
        "\"$DST/services/kitt-worker/venv.new/bin/python\" -c 'import auth, config, contract, job_stubs, queue_store, artifact_staging, capabilities, gpu_policy, runners, sft_command, sft_trainer, executor, monitoring, main, v1, healthcheck')"
    )

    assert "SERVICES=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta kitt-worker)" in text
    assert "check_kitt_worker_prereqs" in text
    assert 'if [ -x "$venv_new/bin/python" ] && [ -f "$DST/services/kitt-worker/main.py" ]; then' not in text
    assert 'if [ ! -x "$venv_new/bin/python" ]; then' in text
    assert required_files in text
    assert "Service-Datei fehlt vor Pre-Swap-Smoke" in text
    assert "runuser -u kitt-worker -- test -x" in text
    assert import_smoke_command in text
    assert 'runuser -u kitt-worker -- test ! -w "$DST/services/kitt-worker"' in text
    assert 'runuser -u kitt-worker -- test ! -w "$DST/services/kitt-worker/venv.new"' in text
    assert 'runuser -u kitt-worker -- test -w "$DST/data/kitt-worker"' in text
    assert 'runuser -u kitt-worker -- test -w "/run/kiron/kitt-worker"' in text
    assert 'for required_dir in \\' in text
    assert 'for queue_file in "$DST/data/kitt-worker"/queue*; do' in text
    assert "queue.sqlite3|queue.sqlite3-wal|queue.sqlite3-shm" in text
    assert "$DST/data/kitt-worker/queue.db" not in text
    assert "$DST/data/kitt-worker/queue.sqlite3-journal" not in text
    assert "$DST/data/kitt-worker/artifacts" not in text
    assert "$DST/data/kitt-worker/staging" in text
    assert "$DST/data/kitt-worker/work" in text
    missing_python_check = text.index('if [ ! -x "$venv_new/bin/python" ]; then')
    required_files_check = text.index(required_files)
    import_smoke = text.index(import_smoke_command)
    swap = text.index('echo "Swappe venvs in Produktionspfad..."')
    assert missing_python_check < swap
    assert required_files_check < swap
    assert import_smoke < swap


def test_deploy_dispatch_gate_rejects_truthy_variants(tmp_path):
    text = _read("scripts/deploy-local.sh")
    function = _extract_shell_function(text, "check_kitt_worker_no_dispatch_activation")
    dst = tmp_path / "dst"
    service_dir = dst / "services" / "kitt-worker"
    service_dir.mkdir(parents=True)
    drop_in_dir = service_dir / "kitt-worker.service.d"
    drop_in_dir.mkdir()
    candidates = [
        service_dir / "candidate.py",
        service_dir / ".env",
        drop_in_dir / "override.conf",
    ]
    key = "allow_" + "dispatch"
    env_key = "KITT_WORKER_" + key.upper()
    truth = "tr" + "ue"
    variants = [
        f"{key}={truth}",
        f"{key} = {truth}",
        f'"{key}": {truth}',
        f"{key}: {truth}",
        f"{env_key}={truth}",
        f"{key}=1",
        f"{key}=yes",
        f"{key}=on",
    ]

    def run_gate() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                f'DST="$1"\n{function}\ncheck_kitt_worker_no_dispatch_activation',
                "bash",
                str(dst),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    for candidate in candidates:
        for variant in variants:
            for path in candidates:
                path.write_text("", encoding="utf-8")
            candidate.write_text(variant, encoding="utf-8")
            assert run_gate().returncode != 0, f"{candidate.name}: {variant}"

    for candidate in candidates:
        candidate.write_text(f"{key}=false\n{key}: no\n{key}=off\n", encoding="utf-8")
    assert run_gate().returncode == 0
