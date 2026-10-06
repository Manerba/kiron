from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_HOME = "/var/cache/kiron/huggingface"
HF_HUB_CACHE = f"{HF_HOME}/hub"
ST_HOME = f"{HF_HOME}/sentence-transformers"


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


def _assert_common_model_unit_contract(
    relpath: str,
    *,
    user: str,
    groups: set[str],
    writable_paths: str,
) -> None:
    settings = _unit_settings(relpath)
    env = _env(settings)

    assert settings["User"] == [user]
    assert settings["Group"] == [user]
    assert set(settings["SupplementaryGroups"][0].split()) == groups
    assert "docker" not in settings["SupplementaryGroups"][0].split()

    assert env["HF_HOME"] == HF_HOME
    assert env["HF_HUB_CACHE"] == HF_HUB_CACHE
    assert env["SENTENCE_TRANSFORMERS_HOME"] == ST_HOME
    assert env["XDG_CACHE_HOME"] == "/var/cache/kiron"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "/root/.cache" not in "\n".join(settings.get("Environment", []))

    assert settings["UMask"] == ["0027"]
    assert settings["PrivateTmp"] == ["true"]
    assert settings["ProtectHome"] == ["true"]
    assert settings["ProtectSystem"] == ["strict"]
    assert settings["ReadWritePaths"] == [writable_paths]
    assert settings["RestrictSUIDSGID"] == ["true"]
    assert settings["LockPersonality"] == ["true"]
    assert settings["NoNewPrivileges"] == ["true"]
    assert settings["PrivateDevices"] == ["false"]
    assert settings["CapabilityBoundingSet"] == [""]
    assert settings["AmbientCapabilities"] == [""]

    unit_text = _read(relpath)
    assert "sudo" not in unit_text
    assert "docker" not in unit_text
    assert "User=root" not in unit_text


def test_embeddings_unit_matches_sprint2_non_root_contract():
    _assert_common_model_unit_contract(
        "systemd/kiron-embeddings.service",
        user="kiron-embeddings",
        groups={"kiron-models", "kiron-config", "kiron-common", "kiron-runtime", "video", "render"},
        writable_paths=HF_HOME + " /run/kiron/vram",
    )


def test_deberta_unit_matches_sprint2_non_root_contract():
    _assert_common_model_unit_contract(
        "systemd/kiron-deberta.service",
        user="kiron-deberta",
        groups={"kiron-models", "kiron-common", "video", "render"},
        writable_paths=HF_HOME,
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        if prefix is None:
            return node.attr
        return f"{prefix}.{node.attr}"
    return None


def _has_true_keyword(call: ast.Call, keyword_name: str) -> bool:
    for keyword in call.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def _is_hf_hub_cache_or_none(node: ast.AST) -> bool:
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return False
    if len(node.values) != 2:
        return False
    env_call, fallback = node.values
    if not isinstance(fallback, ast.Constant) or fallback.value is not None:
        return False
    if not isinstance(env_call, ast.Call):
        return False
    if _call_name(env_call.func) != "os.environ.get":
        return False
    if len(env_call.args) != 1:
        return False
    arg = env_call.args[0]
    return isinstance(arg, ast.Constant) and arg.value == "HF_HUB_CACHE"


def _has_hf_hub_cache_folder(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "cache_folder":
            return _is_hf_hub_cache_or_none(keyword.value)
    return False


def _calls_with_name(relpath: str, names: set[str]) -> list[ast.Call]:
    tree = ast.parse(_read(relpath), filename=relpath)
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) in names:
            calls.append(node)
    return calls


def test_deberta_cross_encoder_load_is_local_only():
    calls = _calls_with_name(
        "services/kiron-deberta/loaders.py",
        {
            "CrossEncoder",
            "AutoTokenizer.from_pretrained",
            "AutoModel.from_pretrained",
            "hf_hub_download",
        },
    )

    assert calls, "DeBERTa must load models through known local-only APIs"
    for call in calls:
        assert _has_true_keyword(call, "local_files_only")

    cross_encoder_calls = _calls_with_name(
        "services/kiron-deberta/loaders.py", {"CrossEncoder"}
    )
    assert cross_encoder_calls, "DeBERTa must construct a CrossEncoder"
    assert all(
        _has_hf_hub_cache_folder(call) for call in cross_encoder_calls
    )


def test_embeddings_model_loads_remain_local_only():
    calls = _calls_with_name(
        "services/kiron-embeddings/loaders.py",
        {
            "SentenceTransformer",
            "AutoTokenizer.from_pretrained",
            "AutoModel.from_pretrained",
            "hf_hub_download",
        },
    )

    assert calls, "Embeddings must load HF models through known local-only APIs"
    for call in calls:
        assert _has_true_keyword(call, "local_files_only")

    sentence_transformer_calls = _calls_with_name(
        "services/kiron-embeddings/loaders.py",
        {"SentenceTransformer"},
    )
    assert sentence_transformer_calls, "Embeddings must construct SentenceTransformer"
    assert all(_has_hf_hub_cache_folder(call) for call in sentence_transformer_calls)
