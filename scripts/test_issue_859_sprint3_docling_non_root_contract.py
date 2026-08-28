from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = "/run/kiron/vram"
DOCKER_SOCKET_PATHS = {"-/run/docker.sock", "-/var/run/docker.sock"}
DASHBOARD_ENV_FILE = "-/etc/kiron/dashboard.env"


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _unit_settings(relpath: str) -> dict[str, list[str]]:
    settings: dict[str, list[str]] = {}
    for raw_line in _read(relpath).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        key, sep, value = line.partition("=")
        if sep:
            settings.setdefault(key, []).append(value)
    return settings


def _env(settings: dict[str, list[str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in settings.get("Environment", []):
        key, sep, value = item.partition("=")
        assert sep, f"invalid Environment= entry: {item}"
        values[key] = value
    return values


def test_docling_unit_matches_sprint3_non_root_contract():
    settings = _unit_settings("systemd/kiron-docling.service")
    env = _env(settings)

    assert "docker.service" in " ".join(settings.get("After", []))
    assert settings["Requires"] == ["docker.service"]

    assert settings["User"] == ["kiron-docling"]
    assert settings["Group"] == ["kiron-docling"]
    assert set(settings["SupplementaryGroups"][0].split()) == {
        "docker",
        "kiron-runtime",
        "kiron-common",
    }
    assert "kiron-models" not in settings["SupplementaryGroups"][0].split()

    assert env["KIRON_RUNTIME_DIR"] == RUNTIME_DIR
    read_write_paths = settings["ReadWritePaths"][0].split()
    assert read_write_paths[0] == RUNTIME_DIR
    assert set(read_write_paths[1:]) == DOCKER_SOCKET_PATHS

    assert settings["UMask"] == ["0027"]
    assert settings["PrivateTmp"] == ["true"]
    assert settings["ProtectHome"] == ["true"]
    assert settings["ProtectSystem"] == ["strict"]
    assert settings["RestrictSUIDSGID"] == ["true"]
    assert settings["LockPersonality"] == ["true"]
    assert settings["NoNewPrivileges"] == ["true"]
    assert settings["PrivateDevices"] == ["false"]
    assert settings["CapabilityBoundingSet"] == [""]
    assert settings["AmbientCapabilities"] == [""]

    unit_text = _read("systemd/kiron-docling.service")
    assert "User=root" not in unit_text
    assert "sudo" not in unit_text


def test_proxy_and_docling_share_dashboard_credentials_envfile():
    proxy_settings = _unit_settings("systemd/kiron-proxy.service")
    docling_settings = _unit_settings("systemd/kiron-docling.service")

    assert proxy_settings["EnvironmentFile"] == [DASHBOARD_ENV_FILE]
    assert docling_settings["EnvironmentFile"] == [DASHBOARD_ENV_FILE]
    assert (
        proxy_settings["EnvironmentFile"]
        == docling_settings["EnvironmentFile"]
    )


def test_docling_proxy_uses_contract_runtime_dir_default():
    text = _read("services/kiron-docling/proxy.py")

    assert 'os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron/vram")' in text
    assert 'os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron")' not in text


def test_docling_gpu_drain_uses_dashboard_basic_auth():
    text = _read("services/kiron-docling/proxy.py")

    assert 'os.environ.get("KIRON_DASHBOARD_USER", "admin")' in text
    assert 'os.environ.get("KIRON_DASHBOARD_PASSWORD", "admin")' in text
    assert "auth=(KIRON_DASHBOARD_USER, KIRON_DASHBOARD_PASSWORD)" in text


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        if prefix is None:
            return node.attr
        return f"{prefix}.{node.attr}"
    return None


def _string_list(node: ast.AST) -> list[str | None] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str | None] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
        else:
            values.append(None)
    return values


def _subprocess_run_commands(relpath: str) -> list[list[str | None]]:
    tree = ast.parse(_read(relpath), filename=relpath)
    commands: list[list[str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "subprocess.run":
            continue
        if not node.args:
            continue
        command = _string_list(node.args[0])
        if command is not None:
            commands.append(command)
    return commands


def test_docling_docker_calls_remain_plain_cli_without_sudo():
    commands = _subprocess_run_commands("services/kiron-docling/proxy.py")

    assert commands, "Docling proxy must keep explicit subprocess commands"
    assert all(command[0] != "sudo" for command in commands)

    docker_commands = [
        command for command in commands
        if command and command[0] == "docker" and len(command) > 1
    ]
    docker_verbs = {command[1] for command in docker_commands}
    assert {"inspect", "update", "start", "stop"} <= docker_verbs

    text = _read("services/kiron-docling/proxy.py")
    assert "sudo" not in text
    assert "DOCKER_HOST" not in text
    assert "/run/docker.sock" not in text
    assert "/var/run/docker.sock" not in text
