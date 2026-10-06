from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "/usr/lib/kiron/data/shared/local-model-registry.json"
LOCK = REGISTRY + ".lock"
MODEL_ROOT = "/usr/lib/kiron/data/local-models"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_tmpfiles_declares_shared_directory_and_optional_registry_policy() -> None:
    lines = set(_read("system/tmpfiles.d/kiron-runtime.conf").splitlines())

    assert "d /usr/lib/kiron/data/shared 2750 kiron-proxy kiron-common -" in lines
    assert f"z {REGISTRY} 0640 kiron-proxy kiron-common -" in lines
    assert f"z {LOCK} 0640 kiron-proxy kiron-common -" in lines
    assert f"d {MODEL_ROOT} 2750 root kiron-common -" in lines
    assert not any(line.startswith(f"f {REGISTRY} ") for line in lines)


def test_deploy_preserves_and_normalizes_registry_and_lock() -> None:
    script = _read("scripts/deploy-local.sh")

    assert 'MODEL_REGISTRY_FILE="$SHARED_DATA_DIR/local-model-registry.json"' in script
    assert 'MODEL_REGISTRY_LOCK_FILE="$MODEL_REGISTRY_FILE.lock"' in script
    assert "check_local_model_registry_paths" in script
    assert "apply_local_model_registry_permissions" in script
    assert 'for file in "$MODEL_REGISTRY_FILE" "$MODEL_REGISTRY_LOCK_FILE"; do' in script
    assert 'chown kiron-proxy:kiron-common "$file"' in script
    assert 'chmod 0640 "$file"' in script
    assert 'stat -c \'%h\' "$file"' in script
    assert script.rindex("check_local_model_registry_paths") < script.index(
        'if [ "$OLLAMA_IMAGE_CHANGED" = "1" ]; then'
    )
    assert script.rindex("apply_local_model_registry_permissions") < script.index(
        "# systemd-Units installieren"
    )


def test_system_config_install_verifies_optional_registry_file_contract() -> None:
    script = _read("scripts/install-system-configs.sh")

    assert "verify_optional_regular_file" in script
    assert f"verify_optional_regular_file {REGISTRY}" in script
    assert f"verify_optional_regular_file {LOCK}" in script
    assert f'verify_optional_path_stat {REGISTRY} "kiron-proxy:kiron-common 640"' in script
    assert f'verify_optional_path_stat {LOCK} "kiron-proxy:kiron-common 640"' in script
    assert f'verify_path_stat {MODEL_ROOT} "root:kiron-common 2750"' in script
    first_host_mutation = script.index("ensure_kiron_identities\n")
    assert script.index(f"verify_optional_regular_file {REGISTRY}") < first_host_mutation
    assert script.index(f"verify_optional_regular_file {LOCK}") < first_host_mutation


def test_setup_installs_a_relocatable_registry_cli_entrypoint(
    tmp_path: Path,
) -> None:
    setup = _read("scripts/setup-venvs.sh")
    entrypoint = _read("scripts/kiron-model-registry-entrypoint.sh")

    assert (
        'REGISTRY_CLI_ENTRYPOINT_SOURCE="$SRC/scripts/'
        'kiron-model-registry-entrypoint.sh"'
    ) in setup
    assert (
        'install -m 0750 -o root -g root '
        '"$REGISTRY_CLI_ENTRYPOINT_SOURCE" '
        '"$venv_new/bin/kiron-model-registry"'
    ) in setup
    assert 'exec "$entrypoint_dir/python" -P -m ' in entrypoint
    assert "kiron_common.local_model_registry.cli" in entrypoint

    bin_dir = tmp_path / "venv.new" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n', encoding="utf-8")
    python.chmod(0o750)
    command = bin_dir / "kiron-model-registry"
    command.write_text(entrypoint, encoding="utf-8")
    command.chmod(0o750)

    completed = subprocess.run(
        [str(command), "list"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "-P",
        "-m",
        "kiron_common.local_model_registry.cli",
        "list",
    ]


def test_dashboard_unit_and_registry_implementation_share_atomic_contract() -> None:
    unit = _read("systemd/kiron-proxy.service")
    registry = _read(
        "services/kiron-common/kiron_common/local_model_registry/registry.py"
    )

    assert "/usr/lib/kiron/data/shared" in unit
    assert "ReadWritePaths=" in unit
    assert "_REGISTRY_MODE = 0o640" in registry
    assert "_LOCK_MODE = 0o640" in registry
    assert "tempfile.mkstemp(" in registry
    assert "os.fsync(descriptor)" in registry
    assert "os.replace(temporary_path, self._path)" in registry
    assert "os.fsync(directory)" in registry


def test_operator_and_api_docs_cover_the_complete_local_registration_contract() -> None:
    operations = _read("docs/local-model-registration-operations.md")
    api = _read("docs/kb-kiron-apis.md")
    project_rules = _read("CLAUDE.md")

    for required in (
        "statischer Release-Catalog",
        "dynamische lokale Runtime-Registry",
        "Modell registrieren",
        "kiron-model-registry register",
        "kiron-model-registry list",
        REGISTRY,
        LOCK,
        "kiron-proxy:kiron-common",
        "0640",
        "sentence_transformers",
        "transformers_last_token",
        "colbert_xmod",
        "cross_encoder",
        "mankei_last_token",
        MODEL_ROOT,
        "GET /api/models/registration-candidates",
        "keine Downloads",
        "kein Hugging Face Hub",
        "registry_corrupt",
        "Backup",
        "Wiederherstellung",
        "sha256sum -c",
        "Abbruchkriterien",
        "Rollback",
    ):
        assert required in operations

    assert "POST /api/models/register" in api
    assert "FastAPI-422" in api
    assert "syntaktisch ungueltiger JSON-Body" in api
    for code in (
        "invalid_provider",
        "invalid_reference",
        "invalid_loader",
        "model_not_found",
        "duplicate_model",
        "loader_metadata_invalid",
        "local_validation_failed",
        "registry_corrupt",
        "registry_access_failed",
    ):
        assert code in api

    assert "docs/local-model-registration-operations.md" in project_rules
