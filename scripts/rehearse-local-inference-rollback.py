#!/usr/bin/env python3
"""Temporary-file rollback rehearsal; no service, container or model is started.

The only start trigger is an in-memory audit entry after every file and contract
passes. The restored Python code then runs against an in-memory HTTP backend.
This is deliberately not a production deployment or rollback utility.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
REGISTRY = "data/shared/local-model-registry.json"
LOCK = REGISTRY + ".lock"
HANDOFF = "data/ollama_compat_runtime.json"
REPORT = "data/ollama_compat_reports/fixture.json"
PREFIX = "kiron-rollback-rehearsal-"
TEMP_ROOT = Path("/tmp")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def environment():
    return {"python": sys.version, "executable": sys.executable,
            "packages": {item.metadata["Name"]: item.version for item in importlib.metadata.distributions()}}


def record(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("not a single regular file")
    return {"sha256": digest(path.read_bytes()), "mode": stat.S_IMODE(info.st_mode),
            "uid": info.st_uid, "gid": info.st_gid}


def put(path, raw, mode=0o640):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    path.write_bytes(raw)
    path.chmod(mode)


def inventory(root):
    return {str(path.relative_to(root)): record(path) for path in sorted(root.rglob("*"))
            if not path.is_dir() and str(path.relative_to(root)) != LOCK}


def verify(root, manifest):
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in rehearsal tree")
        if path.is_dir() and (path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022):
            raise ValueError("unsafe rehearsal directory")
    if inventory(root) != manifest:
        raise ValueError("file hash, metadata or inventory mismatch")


def restore(backup, active, manifest, *, interrupt_after=None):
    """Operator sequence: verify backup, stage/fsync/replace each file, fsync dir.

    There is no cross-file atomicity. The caller keeps its simulated service
    stopped until the complete post-restore gate succeeds. Lock is never copied.
    """
    verify(backup, manifest)
    for count, name in enumerate(sorted(manifest), 1):
        destination = active / name
        metadata = manifest[name]
        temporary = destination.with_name("." + destination.name + ".restore")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, metadata["mode"])
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((backup / name).read_bytes())
                os.fchmod(stream.fileno(), metadata["mode"])
                os.fchown(stream.fileno(), metadata["uid"], metadata["gid"])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        if count == interrupt_after:
            raise InterruptedError("injected restore interruption")


def seed(active):
    sys.path.insert(0, str(REPO / "services/kiron-common"))
    from kiron_common.local_model_registry import RegistryEntry
    from kiron_common.local_model_registry.codec import encode_registry
    from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
    from kiron_common.ollama_compat import OllamaCapabilities

    sources = sorted((REPO / "services/kiron-common/kiron_common").rglob("*.py"))
    sources += sorted((REPO / "services/kiron-common/kiron_common").rglob("*.json"))
    sources += sorted(path for path in (REPO / "services/kiron-proxy").glob("*.py")
                      if not path.name.startswith("test_") and path.name != "conftest.py")
    for source in sources:
        if source.is_symlink():
            raise ValueError("source symlink")
        put(active / source.relative_to(REPO), source.read_bytes())
    entry = RegistryEntry.create(runtime_provider=BackendType.OLLAMA, artifact_origin=ArtifactType.OLLAMA,
        artifact_format=ArtifactFormat.OLLAMA_MANIFEST, reference="rollback-fixture:latest",
        display_name="Isolated rollback fixture", loader=LoaderType.OLLAMA, sha256="a" * 64)
    entry = replace(entry, registered_at=datetime(2026, 9, 22, tzinfo=timezone.utc))
    put(active / REGISTRY, encode_registry((entry,)))
    put(active / LOCK, b"")
    checks = {field.name: {"ok": True, "severity": "info", "data": None}
              for field in fields(OllamaCapabilities)}
    checks["version_endpoint"]["data"] = {"version": "0.18.0"}
    checks["num_gpu_zero_chat_generate"]["data"] = {"num_gpu_zero_effective": True}
    image = "ollama@sha256:" + "b" * 64
    report = encoded({"image_digest": image, "report_status": "passed", "upgrade_allowed": True,
                      "failures": [], "capabilities": checks})
    put(active / REPORT, report)
    put(active / HANDOFF, encoded({"image_digest": image, "report_path": str(active / REPORT),
        "report_sha256": digest(report), "num_gpu_zero_effective": True}))
    # Byte restoration of config/bundle is covered; these inert fixtures are
    # explicitly not a real Prism policy or executable bundle validation.
    put(active / "config/policy-fixture.json", encoded({"fixture_only": True, "revision": "baseline"}))
    put(active / "bundle/inert-fixture", b"never executed; baseline\n", 0o550)
    put(active / "environment.json", encoded(environment()))
    return entry.id


def child(active, api=False):
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
        "--child", str(active), *( ("--api",) if api else ())], env=env,
        capture_output=True, text=True, timeout=25, check=False)
    if result.returncode:
        raise ValueError("restored contract rejected: " + result.stderr.strip()[-600:])
    return json.loads(result.stdout)


def gate(active, manifest):
    verify(active, manifest)
    return child(active)


def validated_start(active, manifest, starts, baseline):
    value = gate(active, manifest)
    if value != baseline:
        raise ValueError("restored contract differs from baseline")
    api = child(active, api=True)
    verify(active, manifest)
    starts.append("simulated_start_after_complete_validation")
    return value, api


def validate_private_source(active):
    if active.name != "active" or active.parent.parent != TEMP_ROOT or not active.parent.name.startswith(PREFIX):
        raise ValueError("only the private temporary rehearsal tree is allowed")
    private = active.parent.lstat()
    if (not stat.S_ISDIR(private.st_mode) or private.st_uid != 0
            or stat.S_IMODE(private.st_mode) != 0o700):
        raise ValueError("rehearsal root must be a private root-owned directory")
    if not stat.S_ISDIR(active.lstat().st_mode):
        raise ValueError("active source must be a real directory")
    for path in (active, *active.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise ValueError("unsafe source directory")
        else:
            metadata = record(path)  # Reject links, special files and hardlinks.
            if metadata["uid"] != 0 or metadata["mode"] & 0o022:
                raise ValueError("unsafe source file")


async def probe(active, api):
    # Enforce the root-owned contract on temporary files without relaxing its
    # production decoder. No alternate targets or live endpoints are accepted.
    validate_private_source(active)
    sys.path[:0] = [str(active / "services/kiron-common"), str(active / "services/kiron-proxy")]
    import socket
    def forbidden_connect(*args, **kwargs):
        raise AssertionError("network is forbidden in rollback rehearsal")
    socket.socket.connect = forbidden_connect
    import kiron_common
    from kiron_common.local_model_registry import RuntimeModelRegistry
    from kiron_common.local_model_registry.codec import decode_registry
    from kiron_common.model_catalog import BackendType, ModelCatalog
    from runtime_composition import load_ollama_compatibility
    import runtime_composition
    if not Path(kiron_common.__file__).is_relative_to(active) or not Path(runtime_composition.__file__).is_relative_to(active):
        raise AssertionError("imports did not use restored code")
    if json.loads((active / "environment.json").read_text()) != environment():
        raise ValueError("test environment changed")
    entries = RuntimeModelRegistry(active / REGISTRY, readonly=True).list()
    if len(entries) != 1 or entries != decode_registry((active / REGISTRY).read_bytes()):
        raise ValueError("registry decode/list differs")
    compatibility, version, image = load_ollama_compatibility(active / HANDOFF, data_root=active / "data")
    result = {"registry_id": entries[0].id, "created": entries[0].registered_at.isoformat(),
              "version": version, "image_digest": image, "restored_imports": True}
    if not api:
        return result

    import httpx
    from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
    from kiron_common.local_inference import (Capability, CapabilityEvidence, CapabilityName, CapabilitySet,
        CapabilityStatus, ParameterConstraint, RuntimeImplementation, RuntimeTimeouts, build_resolver_snapshot)
    from ollama_provider import OllamaProvider
    from openai_api import create_openai_api_app
    from runtime_service import RuntimeService
    snapshot = build_resolver_snapshot(ModelCatalog(), entries)
    model = snapshot.resolve(entries[0].id)
    deployment = model.deployment
    async def snapshot_value():
        return snapshot
    resolver = SimpleNamespace(snapshot=snapshot_value)
    implementation = RuntimeImplementation(image, None, "rollback-fixture-parser")
    evidence = CapabilityEvidence(image, deployment.artifact_identity.fingerprint, entries[0].sha256, None,
        None, implementation.parser_revision, deployment.configuration_fingerprint,
        "offline-rollback-fixture-only", datetime.now(timezone.utc))
    capability = Capability(CapabilityStatus.SUPPORTED, {
        "context_tokens": ParameterConstraint(allowed_values=(1024,)),
        "device": ParameterConstraint(allowed_values=("cpu",)),
        "roles": ParameterConstraint(allowed_values=("user", "assistant", "system")),
        "max_output_tokens": ParameterConstraint(minimum=1, maximum=8),
        "default_max_output_tokens": ParameterConstraint(allowed_values=(8,)),
        "token_budget": ParameterConstraint(allowed_values=("max_tokens", "max_completion_tokens"))}, (evidence,))
    requests, chat_payloads = [], []
    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path in {"/api/ps", "/api/tags"}:
            return httpx.Response(200, json={"models": [{"name": entries[0].reference,
                "model": entries[0].reference, "digest": entries[0].sha256, "size": 1024,
                "size_vram": 0, "context_length": 1024}]})
        if request.url.path == "/api/chat":
            assert any(ticket.kind == "request" for ticket in admission.snapshot())
            assert requests[-2] == "/api/ps"
            payload = json.loads(request.content)
            assert payload["options"] == {"num_predict": 8, "num_ctx": 1024, "num_gpu": 0}
            assert payload["truncate"] is False and payload["shift"] is False and payload["think"] is False
            assert "keep_alive" not in payload
            chat_payloads.append(payload)
            return httpx.Response(200, json={"model": entries[0].reference,
                "message": {"role": "assistant", "content": "OK"}, "done": True, "done_reason": "stop",
                "prompt_eval_count": 3, "eval_count": 1})
        raise AssertionError("unexpected endpoint: " + request.url.path)
    area = active.parent / "admission"
    area.mkdir(mode=0o2770)
    area.chmod(0o2770)
    admission = AdmissionStore(area, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
    provider = OllamaProvider(client=httpx.AsyncClient(base_url="http://127.0.0.1:1", trust_env=False,
        transport=httpx.MockTransport(respond)), resolver=resolver, implementation=implementation,
        capabilities={deployment.id: CapabilitySet({CapabilityName.CHAT: capability})},
        compatibility=compatibility, expected_version=version)
    service = RuntimeService(resolver=resolver, providers={BackendType.OLLAMA: provider}, admission=admission,
        measure=lambda: MemorySnapshot(1024, 1024, time.monotonic()), timeouts=RuntimeTimeouts(1, 1, 1, 1, 5, .1, .1))
    class Records:
        async def add_request(self, record):
            pass
        async def update_request(self, request_id, **values):
            pass
    app = create_openai_api_app(Records(), SimpleNamespace(validate_key=lambda key: {"fixture": True} if key == "rehearsal" else None))
    app.state.local_inference = service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture",
                headers={"Authorization": "Bearer rehearsal"}, trust_env=False) as client:
            models = await client.get("/v1/models")
            if models.status_code != 200 or [row["id"] for row in models.json()["data"]] != [model.api_model_id]:
                raise AssertionError("Models failed after restore: " + models.text)
            chat = await client.post("/v1/chat/completions", json={"model": model.api_model_id,
                "messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8})
            if chat.status_code != 200 or chat.json()["choices"][0]["message"]["content"] != "OK":
                raise AssertionError("Chat failed after restore: " + chat.text)
            if chat.json()["usage"] != {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}:
                raise AssertionError("usage changed")
        if admission.snapshot():
            raise AssertionError("request ticket leaked")
        result.update(models_status=200, chat_status=200, backend_paths=requests,
                      chat_payloads=chat_payloads, admission_empty=True)
    finally:
        await service.aclose()
    return result


def rehearse():
    if os.geteuid() != 0:
        raise ValueError("root is required only for real root-owned temporary configuration checks")
    with tempfile.TemporaryDirectory(prefix=PREFIX, dir=TEMP_ROOT) as temporary:
        root = Path(temporary)
        active, backup = root / "active", root / "backup"
        active.mkdir(mode=0o750)
        model_id = seed(active)
        manifest = inventory(active)
        shutil.copytree(active, backup, ignore=shutil.ignore_patterns("*.lock"))
        verify(backup, manifest)
        lock_identity = ((active / LOCK).stat().st_dev, (active / LOCK).stat().st_ino)
        baseline = gate(active, manifest)
        starts, cases = [], []
        def rejected(name, expected_manifest):
            before = len(starts)
            try:
                validated_start(active, expected_manifest, starts, baseline)
            except ValueError as exc:
                if len(starts) != before:
                    raise AssertionError("failure triggered start")
                cases.append({"case": name, "rejected": True, "start_triggered": False,
                              "reason": str(exc).splitlines()[-1]})
            else:
                raise AssertionError("invalid state accepted: " + name)
        put(active / "config/policy-fixture.json", encoded({"fixture_only": True, "revision": "mixed"}))
        put(active / "services/kiron-proxy/ollama_provider.py", b"raise RuntimeError('mixed code')\n")
        rejected("mixed_release", manifest)
        try:
            restore(backup, active, manifest, interrupt_after=1)
        except InterruptedError:
            pass
        else:
            raise AssertionError("interruption did not happen")
        rejected("interrupted_restore", manifest)
        before = inventory(active)
        wrong = {name: dict(value) for name, value in manifest.items()}
        wrong[REGISTRY]["sha256"] = "0" * 64
        try:
            restore(backup, active, wrong)
        except ValueError:
            if inventory(active) != before or starts:
                raise AssertionError("bad backup changed active files")
            cases.append({"case": "wrong_backup_hash", "rejected": True, "start_triggered": False})
        else:
            raise AssertionError("bad backup accepted")
        restore(backup, active, manifest)
        raw = json.loads((active / REGISTRY).read_text())
        raw["version"] = 1
        put(active / REGISTRY, encoded(raw))
        rejected("hash_consistent_wrong_registry_schema", inventory(active))
        restore(backup, active, manifest)
        raw = json.loads((active / HANDOFF).read_text())
        raw["image_digest"] = "ollama@sha256:" + "c" * 64
        put(active / HANDOFF, encoded(raw))
        rejected("hash_consistent_wrong_image_digest", inventory(active))
        restore(backup, active, manifest)
        if starts or lock_identity != ((active / LOCK).stat().st_dev, (active / LOCK).stat().st_ino):
            raise AssertionError("baseline identity or lock changed")
        after, api = validated_start(active, manifest, starts, baseline)
        return {"status": "passed", "scope": "temporary file/contract rollback; simulated start; real API with mock native HTTP",
            "rehearsal_sha256": digest(Path(__file__).read_bytes()),
            "files": len(manifest), "manifest_sha256": digest(encoded(manifest)), "manifest": manifest,
            "model_id": model_id, "negative_cases": cases, "start_audit": starts,
            "lock_inode_preserved": True, "backup_and_restored_hashes_equal": True,
            "baseline_contract": baseline, "restored_contract": after, "api": api,
            "environment_reused_not_restored": environment(), "temporary_tree_removed_on_return": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--api", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    value = asyncio.run(probe(args.child, args.api)) if args.child else rehearse()
    print(json.dumps(value, sort_keys=True, indent=2))
