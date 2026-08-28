from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_tmpfiles_models_issue_859_runtime_data_and_cache_paths():
    tmpfiles = _read("system/tmpfiles.d/kiron-runtime.conf")

    expected_lines = {
        "d /run/kiron 0755 root root -",
        "d /run/kiron/vram 2770 root kiron-runtime -",
        "f /run/xtables.lock 0600 root root -",
        "d /usr/lib/kiron 0755 root root -",
        "d /usr/lib/kiron/data 0755 root root -",
        "d /usr/lib/kiron/data/kiron-proxy 0750 kiron-proxy kiron-proxy -",
        "d /usr/lib/kiron/data/shared 2750 kiron-proxy kiron-config -",
        "d /usr/lib/kiron/data/local-models 2750 root kiron-common -",
        "d /var/cache/kiron 0755 root root -",
        "d /var/cache/kiron/huggingface 2770 root kiron-models -",
        "d /var/cache/kiron/huggingface/hub 2770 root kiron-models -",
        "d /var/cache/kiron/huggingface/modules 2770 root kiron-models -",
        "d /var/cache/kiron/huggingface/sentence-transformers 2770 root kiron-models -",
        "d /var/cache/kiron/huggingface/xet 2770 root kiron-models -",
        "d /usr/lib/kiron/data/kitt-worker 0750 kitt-worker kitt-worker -",
        "d /usr/lib/kiron/data/kitt-worker/staging 0750 kitt-worker kitt-worker -",
        "d /usr/lib/kiron/data/kitt-worker/work 0750 kitt-worker kitt-worker -",
    }
    assert expected_lines <= set(tmpfiles.splitlines())


def test_install_system_configs_creates_identities_helpers_sudoers_and_tmpfiles():
    script = _read("scripts/install-system-configs.sh")

    for user in ("kiron-proxy", "kiron-docling", "kiron-embeddings", "kiron-deberta"):
        assert f"ensure_kiron_service_identity {user}" in script
        assert f"verify_service_identity {user}" in script
        assert f'id -u "$svc"' in _read("scripts/deploy-local.sh")

    assert "ensure_kiron_service_identity kiron-proxy docker kiron-runtime kiron-common" in script
    assert "ensure_kiron_service_identity kiron-docling docker kiron-runtime kiron-common" in script
    assert "ensure_kiron_service_identity kiron-embeddings kiron-models kiron-config kiron-common video render" in script
    assert "ensure_kiron_service_identity kiron-deberta kiron-models kiron-common video render" in script
    assert "verify_service_identity kiron-embeddings kiron-models kiron-config kiron-common video render" in script
    assert "verify_service_identity kiron-deberta kiron-models kiron-common video render" in script
    assert 'usermod -G "$csv" "$user"' in script
    assert 'groups="$(id -nG "$user" | tr ' in script
    assert "ensure_kitt_worker_identity" in script
    assert '[ "$groups" = "kitt-worker" ]' in script
    assert "ensure_kiron_dashboard_env_contract" in script
    assert "verify_not_symlink /etc/kiron/dashboard.env" in script
    assert "[ -f /etc/kiron/dashboard.env ]" in script
    assert 'stat -c \'%U:%G %a\' /etc/kiron/dashboard.env' in script
    assert "/etc/kiron/dashboard.env muss root:root 0600 sein" in script

    assert "install_wrapper_sources" in script
    assert "install_sudoers_sources" in script
    assert 'install -o root -g root -m 0755 "$src_dir/$helper" "$target"' in script
    assert 'install -o root -g root -m 0440 "$src" "$target"' in script
    assert "visudo -cf \"$src\"" in script
    assert "visudo -c" in script
    assert "verify_path_stat /run/kiron/vram \"root:kiron-runtime 2770\"" in script
    assert "verify_path_stat /run/xtables.lock \"root:root 600\"" in script
    assert "verify_path_stat /usr/lib/kiron/data/local-models \"root:kiron-common 2750\"" in script
    assert "verify_path_stat /var/cache/kiron/huggingface/sentence-transformers \"root:kiron-models 2770\"" in script

    for unit in (
        "kiron-proxy.service",
        "kiron-docling.service",
        "kiron-embeddings.service",
        "kiron-deberta.service",
        "kitt-worker.service",
    ):
        assert unit in script
    assert "validate_service_unit_sources" in script
    assert "install_service_unit_sources" in script
    assert "install -o root -g root -m 0644" in script
    assert '"/opt/kiron/systemd/$unit" "/etc/systemd/system/$unit"' in script
    assert 'cmp -s "/opt/kiron/systemd/$unit" "/etc/systemd/system/$unit"' in script
    assert "systemctl daemon-reload" in script
    assert script.index("validate_service_unit_sources") < script.index(
        "ensure_kiron_identities\n"
    )


def test_sudoers_sources_only_allow_wrappers_not_raw_privileged_commands():
    service_sudoers = _read("system/sudoers.d/kiron-proxy-service-control")
    firewall_sudoers = _read("system/sudoers.d/kiron-proxy-firewall")
    combined = service_sudoers + "\n" + firewall_sudoers

    assert "/usr/local/sbin/kiron-service-control *" in service_sudoers
    assert "/usr/local/sbin/kiron-maintenance-firewall *" in firewall_sudoers
    assert "kiron-proxy ALL=(root) NOPASSWD:" in combined
    assert "systemctl" not in combined
    assert "iptables" not in combined


def test_setup_venvs_sets_readonly_owner_model_and_service_user_smokes():
    script = _read("scripts/setup-venvs.sh")

    assert "apply_readonly_tree_permissions" in script
    assert 'chown -hR root:"$group" "$path"' in script
    assert 'find "$path" -type d -exec chmod 0750 {} +' in script
    assert 'find "$path" -type f -perm /111 -exec chmod 0750 {} +' in script
    assert 'find "$path" -type f ! -perm /111 -exec chmod 0640 {} +' in script
    assert 'apply_readonly_tree_permissions "$COMMON_SRC" kiron-common' in script
    for svc in ("kiron-proxy", "kiron-docling", "kiron-embeddings", "kiron-deberta"):
        assert f"smoke_service_venv_new \"$svc\"" in script
        assert f"{svc}) echo" in script
    assert 'runuser -u "$user" -- test -x "$venv_new/bin/python"' in script
    assert 'runuser -u "$user" -- test ! -w "$DST/services/$svc"' in script
    assert 'runuser -u "$user" -- test ! -w "$venv_new"' in script
    assert 'runuser -u "$user" -- env \\' in script
    assert "KIRON_RUNTIME_DIR=/run/kiron/vram" in script
    assert "HF_HOME=\"$HF_HOME\"" in script
    assert 'kiron-proxy)' in script
    assert 'kiron-embeddings)' in script
    assert 'kiron-deberta)' in script
    assert '"$venv_new/bin/pip" install -e "$COMMON_SRC"' in script
    assert 'import kiron_common' in script
    assert 'from kiron_common.catalog_consistency import check_catalog_digests' in script
    assert 'import app, catalog_health, metrics, main' in script
    assert 'assert MODEL_STATE_VIEW.catalog is MODEL_CATALOG' in script
    assert 'health = json.loads(main.health().body)' in script
    assert 'all(type(row["installed"]) is bool for row in health["model_states"])' in script
    assert 'resources.files("kiron_common.model_catalog.manifests")' in script
    assert 'assert MODEL_CATALOG.groups' in script
    assert 'routing_catalog.PROXY_ROUTING_VIEW.catalog is MODEL_CATALOG' in script
    assert 'check_issue_859_prereqs' in script
    assert 'runuser -u kitt-worker -- test ! -w "$DST/services/kitt-worker/venv.new"' in script

    pre_swap_permissions = script.index("apply_all_service_tree_permissions")
    first_smoke = script.index('smoke_service_venv_new "$svc"')
    swap = script.index('echo "Swappe venvs in Produktionspfad..."')
    post_swap_permissions = script.rindex("apply_all_service_tree_permissions")
    assert pre_swap_permissions < first_smoke < swap < post_swap_permissions


def test_deploy_local_applies_code_common_and_data_owner_model_without_restart_change():
    script = _read("scripts/deploy-local.sh")

    assert "check_issue_859_deploy_prereqs" in script
    assert "check_kiron_dashboard_env_contract" in script
    assert "check_not_symlink /etc/kiron" in script
    assert 'dashboard_env="/etc/kiron/dashboard.env"' in script
    assert 'check_not_symlink "$dashboard_env"' in script
    assert 'stat -c \'%U:%G %a\' /etc/kiron' in script
    assert 'stat -c \'%U:%G %a\' "$dashboard_env"' in script
    assert "/etc/kiron/dashboard.env muss root:root 0600 sein" in script
    assert "apply_code_permissions" in script
    assert "apply_data_permissions" in script
    assert "COMMON_CONSUMERS=(kiron-proxy kiron-docling kiron-embeddings kiron-deberta)" in script
    assert 'for svc in "${COMMON_CONSUMERS[@]}"; do' in script
    assert 'kiron_common fehlt im $svc venv' in script
    assert 'kiron_common.catalog_consistency, kiron_common.model_state' in script
    assert 'resources.files("kiron_common.model_catalog.manifests")' in script
    assert 'assert MODEL_CATALOG.groups' in script
    assert 'apply_readonly_tree_permissions "$DST/services/kiron-common" kiron-common' in script
    assert 'apply_readonly_tree_permissions "$DST/services/$svc" "$group"' in script
    assert 'chown kiron-proxy:kiron-proxy "$PROXY_DATA_DIR"' in script
    assert 'chmod 0750 "$PROXY_DATA_DIR"' in script
    assert 'chown kiron-proxy:kiron-config "$SHARED_DATA_DIR"' in script
    assert 'chmod 2750 "$SHARED_DATA_DIR"' in script
    assert 'chown root:kiron-proxy "$DST/data/db_config.json"' in script
    assert 'chmod 0640 "$DST/data/db_config.json"' in script
    assert 'chown root:root "$DST/data/ollama_compat_runtime.json"' in script
    assert 'chmod 0644 "$DST/data/ollama_compat_runtime.json"' in script
    assert 'chown kitt-worker:kitt-worker "$DST/data/kitt-worker"' in script

    source_gate = script.index('check_issue_859_deploy_prereqs')
    source_gate_call = script.rindex('check_issue_859_deploy_prereqs')
    docker_pull = script.index('docker pull "$NEW_OLLAMA_IMAGE"')
    rsync_common = script.index('"$SRC/services/kiron-common/" "$DST/services/kiron-common/"')
    code_permissions = script.index("apply_code_permissions", rsync_common)
    db_config = script.index('db_config.json: existiert bereits, uebersprungen')
    data_permissions = script.index("apply_data_permissions", db_config)
    unit_copy = script.index('cp "$SRC/systemd/"*.service /etc/systemd/system/')
    restart_guard = script.index('if [ "$1" = "--restart" ]; then', data_permissions)
    restart = script.index('systemctl restart "$svc"', restart_guard)
    assert source_gate < docker_pull
    assert source_gate_call < unit_copy
    assert rsync_common < code_permissions
    assert db_config < data_permissions < restart_guard < restart


def test_migration_doc_covers_required_moves_gates_and_rollback():
    doc = _read("docs/TODO_BUGFIX_859_SPRINT5_MIGRATION_ROLLBACK.md")

    for required in (
        "/root/.cache/huggingface",
        "/var/cache/kiron/huggingface",
        "metrics.db",
        "metrics.db-wal",
        "metrics.db-shm",
        "maintenance_mode.json",
        "selftest_results.json",
        "registry_cache.json",
        "benchmarks_cache.json",
        "/usr/lib/kiron/data/kiron-proxy",
        "/usr/lib/kiron/data/shared/runtime_config.json",
        "visudo -c",
        "sudo -l -U kiron-proxy",
        "Rollback",
    ):
        assert required in doc
    assert "erst mit Betreiberfreigabe produktiv ausgefuehrt" in doc
