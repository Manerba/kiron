from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = "/run/kiron/vram"
PROXY_DATA_DIR = "/usr/lib/kiron/data/kiron-proxy"
SHARED_DATA_DIR = "/usr/lib/kiron/data/shared"
SERVICE_HELPER = "/usr/local/sbin/kiron-service-control"
FIREWALL_HELPER = "/usr/local/sbin/kiron-maintenance-firewall"


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


def test_proxy_unit_matches_sprint4_non_root_contract():
    settings = _unit_settings("systemd/kiron-proxy.service")
    env = _env(settings)

    assert settings["User"] == ["kiron-proxy"]
    assert settings["Group"] == ["kiron-proxy"]
    assert set(settings["SupplementaryGroups"][0].split()) == {
        "docker",
        "kiron-runtime",
        "kiron-common",
        "kiron-config",
        "kiron-prism-control",
    }
    assert env["KIRON_RUNTIME_DIR"] == RUNTIME_DIR

    read_write_paths = settings["ReadWritePaths"][0].split()
    assert read_write_paths == [
        PROXY_DATA_DIR,
        SHARED_DATA_DIR,
        RUNTIME_DIR,
        "/run/xtables.lock",
        "-/run/docker.sock",
        "-/var/run/docker.sock",
    ]

    assert settings["UMask"] == ["0027"]
    assert settings["PrivateTmp"] == ["true"]
    assert settings["ProtectHome"] == ["true"]
    assert settings["ProtectSystem"] == ["strict"]
    assert settings["RestrictSUIDSGID"] == ["true"]
    assert settings["LockPersonality"] == ["true"]
    assert settings["NoNewPrivileges"] == ["false"]
    assert settings["PrivateDevices"] == ["false"]
    assert settings["CapabilityBoundingSet"] == [
        "CAP_SETUID CAP_SETGID CAP_AUDIT_WRITE CAP_NET_ADMIN"
    ]
    assert settings["AmbientCapabilities"] == [""]

    unit_text = _read("systemd/kiron-proxy.service")
    assert "User=root" not in unit_text
    assert "sudo" not in unit_text


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


def test_proxy_has_no_direct_systemctl_or_iptables_subprocess_commands():
    tree = ast.parse(_read("services/kiron-proxy/app.py"), filename="app.py")
    offenders: list[list[str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in {"subprocess.run", "subprocess.Popen"}:
            continue
        if not node.args:
            continue
        command = _string_list(node.args[0])
        if command and command[0] in {"systemctl", "iptables"}:
            offenders.append(command)

    assert offenders == []
    text = _read("services/kiron-proxy/app.py")
    assert f'SERVICE_CONTROL_HELPER = "{SERVICE_HELPER}"' in text
    assert f'FIREWALL_HELPER = "{FIREWALL_HELPER}"' in text
    assert '["docker",' in text


def test_proxy_mutable_paths_and_runtime_config_are_final_contract_paths():
    app_text = _read("services/kiron-proxy/app.py")
    main_text = _read("services/kiron-proxy/main.py")
    selftest_text = _read("services/kiron-proxy/selftest.py")
    embeddings_text = _read("services/kiron-embeddings/main.py")
    vram_text = _read("services/kiron-proxy/vram_lease.py")

    assert 'PROXY_DATA_DIR = DATA_ROOT / "kiron-proxy"' in app_text
    assert 'SHARED_DATA_DIR = DATA_ROOT / "shared"' in app_text
    assert 'RUNTIME_CONFIG_FILE = SHARED_DATA_DIR / "runtime_config.json"' in app_text
    assert 'BENCHMARKS_CACHE_FILE = PROXY_DATA_DIR / "benchmarks_cache.json"' in app_text
    assert "REGISTRY_CACHE_FILE" not in app_text
    assert 'MAINTENANCE_STATE_FILE = PROXY_DATA_DIR / "maintenance_mode.json"' in app_text
    assert 'data_dir = data_root / "kiron-proxy"' in main_text
    assert 'db_config_path = data_root / "db_config.json"' in main_text
    assert '"data" / "kiron-proxy" / "selftest_results.json"' in selftest_text
    assert '"data" / "shared" / "runtime_config.json"' in embeddings_text
    assert '"data" / "runtime_config.json"' not in embeddings_text
    assert 'os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron/vram")' in vram_text
    assert 'os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron")' not in vram_text


def _run_wrapper(relpath: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(ROOT / relpath), *args],
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_service_control_wrapper_is_fail_closed_before_systemctl_exec():
    allowed = {
        ("restart", "kiron-proxy.service"),
        ("start", "kiron-prism.service"),
        ("stop", "kiron-prism.service"),
        ("restart", "kiron-prism.service"),
        ("start", "kiron-embeddings.service"),
        ("stop", "kiron-embeddings.service"),
        ("restart", "kiron-embeddings.service"),
        ("start", "kiron-deberta.service"),
        ("stop", "kiron-deberta.service"),
        ("restart", "kiron-deberta.service"),
    }
    text = _read("system/sbin/kiron-service-control")

    for verb, unit in allowed:
        assert f"{verb} {unit}" in text
    assert "kiron-docling.service" not in text
    assert "eval" not in text
    assert "exec /usr/bin/systemctl" in text

    invalid = _run_wrapper(
        "system/sbin/kiron-service-control",
        "restart",
        "kiron-docling.service",
    )
    assert invalid.returncode != 0
    assert "nicht erlaubt" in invalid.stderr


def test_firewall_wrapper_is_fail_closed_and_whitelisted():
    text = _read("system/sbin/kiron-maintenance-firewall")

    for action in ("check", "insert", "delete"):
        assert action in text
    for chain in ("INPUT", "DOCKER-USER"):
        assert chain in text
    for port in ("5001", "11434", "11435", "11440"):
        assert port in text
    assert "XTABLES_LOCKFILE" not in text
    assert "eval" not in text
    assert "--comment \"$COMMENT\"" in text

    bad_action = _run_wrapper(
        "system/sbin/kiron-maintenance-firewall",
        "flush",
        "INPUT",
        "11434",
    )
    bad_chain = _run_wrapper(
        "system/sbin/kiron-maintenance-firewall",
        "check",
        "OUTPUT",
        "11434",
    )
    bad_port = _run_wrapper(
        "system/sbin/kiron-maintenance-firewall",
        "check",
        "INPUT",
        "22",
    )
    assert bad_action.returncode != 0
    assert bad_chain.returncode != 0
    assert bad_port.returncode != 0
