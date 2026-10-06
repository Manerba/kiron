"""Versioned native Dense transport identity, shared by service and adapter."""
import hashlib
import json
from pathlib import Path

PROTOCOL_VERSION = 1
TOKEN_COUNTING = "forward_attention_mask_v1"


def request_fingerprint(payload):
    """Bind a native rejection to the complete original request, including input."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def service_revision(service_root, *, versions):
    root = Path(service_root)
    common = Path(__file__).resolve().parent.parent
    paths = [("embedding/" + name, root / name) for name in (
        "main.py", "model_worker.py", "loaders.py", "catalog_view.py", "token_usage.py", "native_runtime.py", "native_api.py", "residency.py")]
    paths.extend(("common/" + name, common / name) for name in (
        "embedding_contract.py", "embedding_registry.py", "model_state.py", "prism_runtime_policy.py"))
    for directory in ("local_inference", "model_catalog", "gpu_admission"):
        paths.extend((f"common/{directory}/{path.name}", path) for path in (common / directory).glob("*.py"))
    digest = hashlib.sha256()
    for label, path in sorted(paths):
        payload = path.read_bytes()
        digest.update(label.encode() + b"\0" + len(payload).to_bytes(8, "big") + payload)
    for name, version in sorted(versions.items()):
        digest.update(name.encode() + b"=" + version.encode() + b"\0")
    return "sha256:" + digest.hexdigest()
