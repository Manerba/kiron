"""Local artifact/profile integrity, schema closure and shared registration paths."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import struct

import pytest

from kiron_common.local_model_registry import (
    GGUFLocalValidator, GGUFRegistrationPolicy, InvalidReferenceError, LoaderMetadataError,
    RegistryCorruptionError, RuntimeModelRegistry,
)
from kiron_common.local_model_registry import gguf
from kiron_common.local_model_registry.cli import run_cli
from kiron_common.local_model_registry.codec import decode_registry, encode_registry
from kiron_common.local_model_registry.composition import build_model_registration_service
from kiron_common.prism_runtime_policy import Policy, PolicyError
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, CatalogValidationError, ModelCatalog, parse_manifest
from kiron_common.model_state import BackendRuntimeSnapshot, LocalModelInventory, RuntimeInventory, RuntimeState, build_model_state_view


PROFILE = "test-cuda40-c1024-v1"


def gguf_bytes(*, architecture="qwen35", dimension=5120):
    def text(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    fields = {"general.architecture": architecture, "general.name": "Local test model",
              "clip.vision.projection_dim" if architecture == "clip" else f"{architecture}.embedding_length": dimension}
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields))
    for key, value in fields.items():
        data += text(key)
        data += struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += text(value) if isinstance(value, str) else struct.pack("<I", value)
    return data + b"bounded-test-tensor-payload"


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "gguf-models"
    directory = root / "unit"
    directory.mkdir(parents=True)
    model, projector = directory / "model.gguf", directory / "projector.gguf"
    model.write_bytes(gguf_bytes())
    projector.write_bytes(gguf_bytes(architecture="clip"))
    policy = GGUFRegistrationPolicy(hashlib.sha256(model.read_bytes()).hexdigest(), "qwen35",
                                    hashlib.sha256(projector.read_bytes()).hexdigest())
    validator = GGUFLocalValidator(model_root=root, profiles={PROFILE: policy}, owner_uid=os.getuid())
    return root, model, projector, policy, validator


def test_file_hash_pairing_and_restart_persistence(artifacts, tmp_path):
    _, model, projector, policy, validator = artifacts
    entry = validator.validate(str(model), runtime_profile=PROFILE, projector_reference=str(projector)).to_entry()
    assert (entry.runtime_provider, entry.artifact_origin, entry.artifact_format) == (
        BackendType.PRISM, ArtifactType.LOCAL, ArtifactFormat.GGUF)
    assert entry.sha256 == policy.model_sha256
    assert entry.size_bytes == model.stat().st_size
    assert entry.projector.sha256 == policy.projector_sha256
    assert entry.capability_fingerprint is None
    registry = RuntimeModelRegistry(tmp_path / "registry.json")
    registry.add(entry)
    assert RuntimeModelRegistry(registry.path).get(entry.id) == entry
    assert json.loads(registry.path.read_text())["version"] == 2
    text_only = validator.validate(str(model), runtime_profile=PROFILE).to_entry()
    assert text_only.id == entry.id
    assert text_only.configuration_fingerprint != entry.configuration_fingerprint


def test_shared_composition_and_cli_register_exact_same_gguf(artifacts, tmp_path):
    root, model, _, policy, _ = artifacts
    class NoOllama:
        def list_models(self):
            raise RuntimeError("offline")
        def show_model(self, _name):
            raise AssertionError("GGUF must not contact Ollama")
    service = build_model_registration_service(registry=RuntimeModelRegistry(tmp_path / "registry.json"),
              ollama=NoOllama(), gguf_model_root=root, gguf_profiles={PROFILE: policy},
              huggingface_model_root=tmp_path / "missing-hf")
    discovered = service.list_candidates()
    assert discovered.errors == {BackendType.OLLAMA: "local_validation_failed"}
    assert {item.reference for item in discovered.candidates} == {str(model), str(model.with_name("projector.gguf"))}
    code, payload = run_cli(["register", "--runtime-provider", "prism", "--reference", str(model),
                           "--runtime-profile", PROFILE], service=service, effective_uid=0)
    assert code == 0
    assert payload["model"] == service.list_models()[0].to_dict()


def test_composition_uses_controller_policy_pins_and_root(artifacts, tmp_path, monkeypatch):
    root, model, _, profile, _ = artifacts
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("owned by the policy parser")
    class RuntimePolicy:
        artifact_roots = (root,)
        def registration_profiles(self):
            return {PROFILE: profile}
    seen = []
    def load(path):
        seen.append(path)
        return RuntimePolicy()
    monkeypatch.setattr(Policy, "load", load)
    service = build_model_registration_service(registry=RuntimeModelRegistry(tmp_path / "registry.json"),
                                               prism_policy_path=policy_path)
    code, payload = run_cli(["register", "--runtime-provider", "prism", "--reference", str(model),
                           "--runtime-profile", PROFILE], service=service, effective_uid=0)
    assert code == 0 and payload["model"]["sha256"] == profile.model_sha256
    assert seen == [policy_path]
    RuntimePolicy.artifact_roots = (root, root / "second")
    with pytest.raises(PolicyError, match="exactly one"):
        build_model_registration_service(prism_policy_path=policy_path)


def test_optional_missing_policy_is_unconfigured_but_existing_broken_policy_fails(artifacts, tmp_path, monkeypatch):
    _, model, _, _, _ = artifacts
    path = tmp_path / "missing-policy.json"
    def broken(_path):
        raise FileNotFoundError("referenced runtime bundle missing")
    monkeypatch.setattr(Policy, "load", broken)
    service = build_model_registration_service(registry=RuntimeModelRegistry(tmp_path / "registry.json"),
                                               prism_policy_path=path)
    code, _ = run_cli(["register", "--runtime-provider", "prism", "--reference", str(model),
                      "--runtime-profile", PROFILE], service=service, effective_uid=0)
    assert code != 0
    path.write_text("existing policy")
    with pytest.raises(FileNotFoundError, match="runtime bundle"):
        build_model_registration_service(prism_policy_path=path)


@pytest.mark.parametrize("fault", ["hash", "profile", "architecture", "projector_hash", "projector_dimension", "missing_profile"])
def test_unapproved_artifact_profile_and_projector_never_register(artifacts, fault):
    root, model, projector, policy, validator = artifacts
    arguments = {"runtime_profile": PROFILE, "projector_reference": str(projector)}
    if fault == "hash":
        model.write_bytes(model.read_bytes() + b"changed")
    elif fault == "profile":
        arguments["runtime_profile"] = "unapproved"
    elif fault == "architecture":
        validator = GGUFLocalValidator(model_root=root, profiles={PROFILE: replace(policy, architecture="other")})
    elif fault == "projector_hash":
        arguments["expected_projector_sha256"] = "0" * 64
    elif fault == "projector_dimension":
        projector.write_bytes(gguf_bytes(architecture="clip", dimension=123))
        validator = GGUFLocalValidator(model_root=root, profiles={PROFILE: replace(policy, projector_sha256=hashlib.sha256(projector.read_bytes()).hexdigest())})
    else:
        validator = GGUFLocalValidator(model_root=root)
    with pytest.raises(LoaderMetadataError):
        validator.validate(str(model), **arguments)


@pytest.mark.parametrize("fault", ["symlink", "parent_symlink", "fifo", "hardlink", "writable", "owner", "escape"])
def test_hostile_paths_are_rejected_without_reading_special_files(artifacts, tmp_path, fault):
    root, model, _, policy, validator = artifacts
    reference = str(model)
    if fault == "symlink":
        target = model.with_name("real.gguf")
        model.rename(target)
        model.symlink_to(target)
    elif fault == "parent_symlink":
        actual = root / "actual"
        model.parent.rename(actual)
        (root / "unit").symlink_to(actual, target_is_directory=True)
    elif fault == "fifo":
        model.unlink()
        os.mkfifo(model)
    elif fault == "hardlink":
        os.link(model, model.with_name("alias.gguf"))
    elif fault == "writable":
        model.chmod(0o666)
    elif fault == "owner":
        validator = GGUFLocalValidator(model_root=root, profiles={PROFILE: policy}, owner_uid=os.getuid() + 1)
    else:
        reference = str(root / "unit" / ".." / "unit" / "model.gguf")
    with pytest.raises((InvalidReferenceError, LoaderMetadataError)):
        validator.validate(reference, runtime_profile=PROFILE)


def test_file_replacement_during_probe_is_rejected(artifacts, monkeypatch):
    _, model, _, _, validator = artifacts
    original = gguf.read_gguf_metadata
    def replace_during_read(stream):
        result = original(stream)
        old = model.with_name("removed.gguf")
        model.rename(old)
        model.write_bytes(old.read_bytes())
        return result
    monkeypatch.setattr(gguf, "read_gguf_metadata", replace_during_read)
    with pytest.raises(LoaderMetadataError):
        validator.validate(str(model), runtime_profile=PROFILE)


@pytest.mark.parametrize("payload", [b"GGUF", b"NOPE" + bytes(20), b"GGUF" + struct.pack("<IQQ", 3, 1, 999999)])
def test_metadata_is_bounded_and_truncation_rejected(payload):
    with pytest.raises(ValueError):
        gguf.read_gguf_metadata(io.BytesIO(payload))


def test_registry_rejects_old_schema_unknown_fields_and_tampered_fingerprint(artifacts):
    _, model, _, _, validator = artifacts
    entry = validator.validate(str(model), runtime_profile=PROFILE).to_entry()
    document = json.loads(encode_registry([entry]))
    cases = [{**document, "version": 1}, {**document, "entries": [{**entry.to_dict(), "unexpected": True}]},
             {**document, "entries": [{**entry.to_dict(), "sha256": "0" * 64}]}]
    for case in cases:
        with pytest.raises(RegistryCorruptionError):
            decode_registry(json.dumps(case).encode())


def manifest(entry):
    return {"schema_version": 2, "canonical_model_id": "unit-chat", "aliases": [], "metadata": {},
        "request_defaults": [], "deployments": [{"id": "unit.prism", "backend": {"type": "prism", "parameters": {"model_name": "unit-chat"}},
        "artifact": {"type": "local", "format": "gguf", "repository": None, "revision": None,
            "manifest_digest": None, "trust_remote_code": False, "weights": [{"path": entry.reference, "sha256": entry.sha256, "size_bytes": entry.size_bytes}],
            "projector": None, "auxiliary": [], "metadata": {}},
        "loader": {"type": "prism_gguf", "parameters": {}}, "runtime_profile": PROFILE,
        "routes": [{"task": "chat", "endpoint": "/v1/chat/completions"}], "metadata": {}}],
        "profiles": [{"id": "unit.chat", "deployment_id": "unit.prism", "task": "chat", "endpoint": "/v1/chat/completions",
                      "default_for_endpoint": True, "metadata": {}}]}


def test_catalog_gguf_and_exact_inventory_runtime_projection(artifacts):
    _, model, _, _, validator = artifacts
    entry = validator.validate(str(model), runtime_profile=PROFILE).to_entry()
    catalog = ModelCatalog.from_manifests([manifest(entry)])
    view = build_model_state_view(catalog)
    file = catalog.groups[0].deployments[0].artifact.weights[0]
    wrong = LocalModelInventory(gguf_files=frozenset([replace(file, sha256="0" * 64)]))
    assert view.states(wrong, RuntimeInventory())[0].installed is False
    for field, state in (("unloading_names", RuntimeState.UNLOADING), ("failed_names", RuntimeState.FAILED)):
        runtime = RuntimeInventory({BackendType.PRISM: BackendRuntimeSnapshot(known=True, **{field: frozenset(["unit-chat"])})})
        actual = view.states(LocalModelInventory(gguf_files=frozenset([file])), runtime)[0]
        assert actual.installed and actual.runtime_state is state and not actual.loaded
    broken = manifest(entry)
    broken["deployments"][0]["artifact"]["format"] = "hf_weights"
    with pytest.raises(CatalogValidationError):
        parse_manifest(broken)
