"""Unified dashboard actions: real HTTP boundary, fake provider use cases."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import app as dashboard
import dashboard_runtime
from kiron_common.local_inference import CapabilityName, CapabilitySet, ParameterConstraint
from model_control import ACTION_LABELS, merge_inventory
from test_dashboard_runtime import runtime  # Real RuntimeService/admission fixture, offline.

AUTH = ("admin", "admin")
HEADERS = {"X-Kiron-Action": "models"}


@pytest.fixture
def controls(monkeypatch):
    native = {"name": "org/model:Q4", "backend": "ollama", "installed": True,
              "load_state": "unloaded", "loaded": False, "deployment_ids": [], "catalog_managed": False}
    native_read = AsyncMock(return_value={"models": [native]})
    runtime_read = AsyncMock(return_value={"models": []})
    runtime_action = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(dashboard, "get_local_models", native_read)
    monkeypatch.setattr(dashboard_runtime, "runtime_inventory", runtime_read)
    monkeypatch.setattr(dashboard_runtime, "runtime_action", runtime_action)
    monkeypatch.setattr(dashboard.vram_lease, "num_gpu_zero_effective", lambda: True)
    handlers = {}
    for name in ("load_model", "unload_model", "delete_model", "load_embedding_model", "unload_embedding_model",
                 "warmup_colbert_model", "load_deberta_model", "unload_deberta_model"):
        handlers[name] = AsyncMock(return_value={"status": "ok"})
        monkeypatch.setattr(dashboard, name, handlers[name])
    return SimpleNamespace(client=TestClient(dashboard.app), native=native, native_read=native_read,
                           runtime_read=runtime_read, runtime_action=runtime_action, handlers=handlers)


def read(client):
    response = client.get("/api/models/control", auth=AUTH)
    assert response.status_code == 200, response.text
    return response.json()["models"][0]


def command(client, row, action, **extra):
    return client.post("/api/models/control/action", auth=AUTH, headers=HEADERS,
                       json={"model": row["control_id"], "revision": row["control_revision"], "action": action, **extra})


def by_action(row, action):
    return next(item for item in row["controls"] if item["id"] == action)


def test_unknown_ollama_work_blocks_model_actions_but_keeps_diagnostics(controls, native_admission_runtime):
    from test_ollama_recovery import reserve
    reserve(native_admission_runtime)
    row = read(controls.client)
    assert by_action(row, "health")["enabled"]
    assert not by_action(row, "load")["enabled"]
    assert "qwen3:8b" in by_action(row, "load")["reason"]
    assert command(controls.client, row, "load").status_code == 409
    controls.handlers["load_model"].assert_not_awaited()
    payload = controls.client.get("/api/models/control", auth=AUTH).json()
    assert payload["ollama_recovery"]["recovery_available"]


@pytest.mark.parametrize("action,handler,payload", [
    ("load", "load_model", {"name": "org/model:Q4", "gpu": True}),
    ("load_cpu", "load_model", {"name": "org/model:Q4", "gpu": False}),
    ("delete", "delete_model", {"name": "org/model:Q4"}),
])
def test_exact_native_target_and_cpu_mode_reach_guarded_use_case(controls, action, handler, payload):
    row = read(controls.client)
    assert [item["id"] for item in row["controls"]] == list(ACTION_LABELS)
    assert command(controls.client, row, action).status_code == 200
    controls.handlers[handler].assert_awaited_once_with(payload)


def test_loaded_delete_is_disabled_without_force_escape(controls):
    controls.native.update(load_state="loaded", loaded=True)
    row = read(controls.client)
    assert not by_action(row, "delete")["enabled"]
    assert "entladen" in by_action(row, "delete")["reason"]
    assert command(controls.client, row, "delete").status_code == 409
    assert command(controls.client, row, "delete", force=True).status_code == 400
    controls.handlers["delete_model"].assert_not_awaited()
    assert command(controls.client, row, "unload").status_code == 200
    controls.handlers["unload_model"].assert_awaited_once_with({"name": "org/model:Q4"})


@pytest.mark.parametrize("change", [{"load_state": "loaded", "loaded": True}, {"installed": False}])
def test_stale_inventory_never_dispatches(controls, change):
    row = read(controls.client)
    controls.native.update(change)
    response = command(controls.client, row, "load")
    assert response.status_code == 409 and response.json()["error"]["code"] == "conflict"
    controls.handlers["load_model"].assert_not_awaited()


def test_unknown_state_blocks_mutations_but_keeps_diagnosis(controls):
    controls.native["load_state"] = "unknown"
    row = read(controls.client)
    assert all(item["enabled"] == (item["id"] == "health") for item in row["controls"])
    assert command(controls.client, row, "load").status_code == 409
    assert command(controls.client, row, "health").status_code == 200


def test_cpu_requires_compat_evidence(controls, monkeypatch):
    monkeypatch.setattr(dashboard.vram_lease, "num_gpu_zero_effective", lambda: False)
    row = read(controls.client)
    assert by_action(row, "load")["enabled"]
    assert not by_action(row, "load_cpu")["enabled"]
    assert command(controls.client, row, "load_cpu").status_code == 409
    controls.handlers["load_model"].assert_not_awaited()


@pytest.mark.parametrize("kind,action,handler", [("dense", "load", "load_embedding_model"),
                                                ("colbert", "warmup", "warmup_colbert_model")])
def test_catalog_embedding_controls_reuse_guarded_native_operations(controls, kind, action, handler):
    controls.native.update(backend="kiron_embeddings", catalog_managed=True, embedding_kind=kind)
    row = read(controls.client)
    assert not by_action(row, "load_cpu")["supported"]
    assert not by_action(row, "delete")["supported"]
    assert command(controls.client, row, action).status_code == 200
    controls.handlers[handler].assert_awaited_once_with({"model": "org/model:Q4"})


@pytest.mark.parametrize("action,state,handler", [
    ("load", "unloaded", "load_deberta_model"), ("unload", "loaded", "unload_deberta_model"),
])
def test_catalog_deberta_controls_use_native_state_and_guarded_operations(controls, action, state, handler):
    controls.native.update(name="bge-reranker-v2-m3", backend="kiron_deberta", catalog_managed=True,
                           load_state=state, deployment_ids=["bound"])
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "kiron_deberta", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": None, "load_state": "unknown", "actions": ["health"],
        "lifecycle_operations": [], "diagnostic": "Gemeinsamer Provider nicht verfuegbar.",
    }]}
    row = read(controls.client)
    assert row["operation_state"] == state
    assert row["load_state"] == "unknown" and row["installed"] is None
    assert by_action(row, action)["supported"] and by_action(row, action)["enabled"]
    assert not by_action(row, "load_cpu")["supported"] and not by_action(row, "delete")["supported"]
    assert command(controls.client, row, action).status_code == 200
    controls.handlers[handler].assert_awaited_once_with({"model": "bge-reranker-v2-m3"})
    controls.runtime_action.assert_not_awaited()


@pytest.mark.parametrize("state", ["unloaded", "unknown", "loading"])
def test_deberta_unload_requires_known_native_residency(controls, state):
    controls.native.update(backend="kiron_deberta", catalog_managed=True, load_state=state)
    row = read(controls.client)
    assert by_action(row, "unload")["supported"] and not by_action(row, "unload")["enabled"]
    assert command(controls.client, row, "unload").status_code == 409
    controls.handlers["unload_deberta_model"].assert_not_awaited()


def test_deberta_unload_revalidates_a_model_switch_before_calling_service(controls):
    controls.native.update(backend="kiron_deberta", catalog_managed=True, load_state="loaded")
    row = read(controls.client)
    controls.native["load_state"] = "unloaded"
    assert command(controls.client, row, "unload").status_code == 409
    controls.handlers["unload_deberta_model"].assert_not_awaited()


def test_unmanaged_deberta_entry_does_not_gain_native_operations(controls):
    controls.native.update(backend="kiron_deberta", catalog_managed=False, load_state="loaded")
    row = read(controls.client)
    assert not by_action(row, "load")["supported"] and not by_action(row, "unload")["supported"]


def test_runtime_profile_and_provider_permissions_are_not_bypassed(controls):
    controls.native_read.return_value = {"models": []}
    observed = {"name": "HF model", "backend": "kiron_deberta", "runtime": True,
                "deployment_ids": ["hf-one"], "registry_ids": [], "installed": True,
                "load_state": "unloaded", "lifecycle_operations": ["load", "unload"],
                "actions": ["health"], "resource_profile": None, "snapshot_revision": "snapshot"}
    controls.runtime_read.return_value = {"models": [observed]}
    row = read(controls.client)
    assert not by_action(row, "load")["enabled"]
    assert "Ressourcenprofil" in by_action(row, "load")["reason"]
    assert command(controls.client, row, "load").status_code == 409
    controls.runtime_action.assert_not_awaited()


def test_runtime_controls_use_real_admission_and_confirmed_state(runtime, monkeypatch):
    monkeypatch.setattr(dashboard, "get_local_models", AsyncMock(return_value={"models": []}))
    row = read(runtime.client)
    assert by_action(row, "load")["enabled"]
    assert command(runtime.client, row, "load").status_code == 200
    assert runtime.provider.calls == ["load"]
    assert [ticket.phase for ticket in runtime.admission.snapshot()] == ["resident"]
    row = read(runtime.client)
    assert not by_action(row, "load")["enabled"] and by_action(row, "unload")["enabled"]
    assert command(runtime.client, row, "unload").status_code == 200
    assert runtime.admission.snapshot() == ()


def test_partial_inventory_is_visible_and_blocks_mutations(controls):
    controls.runtime_read.return_value = JSONResponse({"error": "private failure"}, status_code=503)
    response = controls.client.get("/api/models/control", auth=AUTH)
    assert response.status_code == 200 and response.json()["warnings"]
    assert "private failure" not in response.text
    row = response.json()["models"][0]
    assert not by_action(row, "load")["enabled"]
    assert command(controls.client, row, "load").status_code == 409


def test_same_names_across_backends_do_not_merge():
    native = {"name": "same", "backend": "ollama", "deployment_ids": ["one"]}
    observed = {"name": "same", "backend": "prism", "deployment_ids": ["one"], "registry_ids": []}
    assert len(merge_inventory([native], [observed])) == 2
    observed["backend"] = "ollama"
    observed["load_state"] = "unknown"
    merged = merge_inventory([native], [observed])
    assert len(merged) == 1 and merged[0][0]["load_state"] == "unknown"
    with pytest.raises(ValueError):
        merge_inventory([native, deepcopy(native)], [observed])


@pytest.mark.parametrize("reverse", [False, True])
def test_dense_and_late_profiles_share_one_native_control_without_sharing_proofs(controls, reverse):
    controls.native.update(name="nomic-embed-text", backend="kiron_embeddings",
                           catalog_managed=True, load_state="loaded", loaded=True,
                           deployment_ids=["dense", "late"], profile_ids=["dense-profile", "late-profile"])
    observed = [
        {"name": "nomic-embed-text", "backend": "kiron_embeddings", "runtime": True,
         "deployment_ids": ["dense"], "profile_ids": ["dense-profile"], "api_model_ids": ["dense.query"],
         "installed": True, "load_state": "loaded", "capabilities": {"embeddings": {"status": "supported"}}},
        {"name": "nomic-embed-text", "backend": "kiron_embeddings", "runtime": True,
         "deployment_ids": ["late"], "profile_ids": ["late-profile"], "api_model_ids": ["late.query"],
         "installed": False, "load_state": "unloaded", "error_code": "artifact_missing_or_changed",
         "capabilities": {"embeddings": {"status": "unverified"}}},
    ]
    controls.runtime_read.return_value = {"models": list(reversed(observed)) if reverse else observed}
    original_native, original_observed = deepcopy(controls.native), deepcopy(observed)
    payload = controls.client.get("/api/models/control", auth=AUTH).json()
    assert len(payload["models"]) == 1
    row = payload["models"][0]
    assert row["deployment_ids"] == ["dense", "late"]
    assert row["profile_ids"] == ["dense-profile", "late-profile"]
    assert row["api_model_ids"] == ["dense.query", "late.query"]
    assert row["operation_state"] == "loaded"
    assert row["verification"]["status"] == "unverified"
    assert [item["verification"]["status"] for item in row["runtime_deployments"]] == ["verified", "unverified"]
    assert row["runtime_deployments"][1]["error_code"] == "artifact_missing_or_changed"
    assert controls.native == original_native and observed == original_observed
    assert command(controls.client, row, "unload").status_code == 200
    controls.handlers["unload_embedding_model"].assert_awaited_once_with({"model": "nomic-embed-text"})
    controls.runtime_action.assert_not_awaited()


def test_single_runtime_profile_keeps_the_complete_native_profile_identity():
    native = {"name": "model", "backend": "kiron_embeddings", "catalog_managed": True,
              "deployment_ids": ["dense", "late"], "profile_ids": ["dense-profile", "late-profile"]}
    observed = {"name": "model", "backend": "kiron_embeddings",
                "deployment_ids": ["dense"], "profile_ids": ["dense-profile"]}
    row, original = merge_inventory([native], [observed])[0]
    assert row["deployment_ids"] == native["deployment_ids"]
    assert row["profile_ids"] == native["profile_ids"]
    assert original is native


@pytest.mark.parametrize("backend,shared_state", [("ollama", "unknown"), ("kiron_embeddings", "unloaded")])
def test_native_runner_state_is_authoritative_for_native_actions(controls, backend, shared_state):
    controls.native.update(backend=backend, catalog_managed=True, deployment_ids=["bound"], load_state="loaded", loaded=True)
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": backend, "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": True, "load_state": shared_state, "actions": ["health"],
        "lifecycle_operations": [], "snapshot_revision": "revision",
    }]}
    row = read(controls.client)
    assert row["operation_state"] == "loaded"
    assert row["load_state"] == shared_state  # Inference validation remains separate.
    assert not by_action(row, "load")["enabled"]
    if backend == "ollama":
        assert by_action(row, "unload")["enabled"]
        assert command(controls.client, row, "unload").status_code == 200
        controls.handlers["unload_model"].assert_awaited_once_with({"name": "org/model:Q4"})
    else:
        assert by_action(row, "unload")["enabled"]
        assert command(controls.client, row, "unload").status_code == 200
        controls.handlers["unload_embedding_model"].assert_awaited_once_with({"model": "org/model:Q4"})


def test_native_embedding_unload_survives_missing_shared_artifact_proof(controls):
    controls.native.update(backend="kiron_embeddings", catalog_managed=True, load_state="loaded",
                           installed=False, deployment_ids=["bound"])
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "kiron_embeddings", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": False, "load_state": "unknown", "actions": ["health"],
        "lifecycle_operations": [], "diagnostic": "Artefakt nicht bestaetigt.",
    }]}
    row = read(controls.client)
    assert not by_action(row, "load")["enabled"]
    assert by_action(row, "unload")["enabled"]
    assert command(controls.client, row, "unload").status_code == 200
    controls.handlers["unload_embedding_model"].assert_awaited_once()


@pytest.mark.parametrize("state", ["unloaded", "unknown", "loading"])
def test_embedding_unload_requires_known_native_residency(controls, state):
    controls.native.update(backend="kiron_embeddings", catalog_managed=True, load_state=state)
    row = read(controls.client)
    assert by_action(row, "unload")["supported"]
    assert not by_action(row, "unload")["enabled"]
    assert command(controls.client, row, "unload").status_code == 409
    controls.handlers["unload_embedding_model"].assert_not_awaited()


@pytest.mark.parametrize("installed", [True, False, None])
def test_embedding_can_reload_from_native_local_files_without_resident_proof(controls, installed):
    controls.native.update(backend="kiron_embeddings", catalog_managed=True, load_state="unloaded",
                           installed=installed, deployment_ids=["bound"])
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "kiron_embeddings", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": False, "load_state": "unloaded", "actions": ["health"],
        "lifecycle_operations": [], "diagnostic": "Kein residenter Artefaktnachweis.",
    }]}
    row = read(controls.client)
    assert by_action(row, "load")["enabled"] == (installed is True)
    assert row["installed"] is False  # Shared inference proof is not fabricated.
    if installed is True:
        assert command(controls.client, row, "load").status_code == 200
        controls.handlers["load_embedding_model"].assert_awaited_once_with({"model": "org/model:Q4"})
    else:
        assert command(controls.client, row, "load").status_code == 409
        controls.handlers["load_embedding_model"].assert_not_awaited()


@pytest.mark.parametrize("device,context,state,expected", [
    ("gpu", 4096, "loaded", "deviating"),
    ("cpu", 1024, "loaded", "verified"),
    (None, 1024, "loaded", "unverified"),
    (None, None, "unknown", "unverified"),
    (None, None, "unloaded", "unverified"),
])
def test_residency_configuration_and_evidence_are_independent(controls, device, context, state, expected):
    controls.native.update(load_state=state, deployment_ids=["bound"], runtime_device=device,
                           runtime_context_length=context, vram_gb=6 if device == "gpu" else 0)
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "ollama", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": True, "load_state": "loaded" if expected == "verified" else "unknown", "actions": ["health"],
        "capabilities": {"chat": {"status": "supported", "configuration_constraints": {
            "device": {"allowed_values": ["cpu"]}, "context_tokens": {"allowed_values": [1024]}}}},
    }]}
    row = read(controls.client)
    assert row["operation_state"] == state
    assert row["verification"]["status"] == expected
    assert row["configuration"]["device"] == (device if state == "loaded" else None)
    assert by_action(row, "unload")["enabled"] == (state == "loaded")
    assert row["load_state"] == ("loaded" if expected == "verified" else "unknown")


def test_missing_evidence_does_not_block_native_load(controls):
    row = read(controls.client)
    assert row["verification"]["status"] == "unverified"
    assert command(controls.client, row, "load").status_code == 200
    controls.handlers["load_model"].assert_awaited_once()


def test_service_memory_survives_runtime_merge_without_becoming_model_configuration(controls):
    memory = {"vram_gb": 1.93, "ram_gb": 1.72, "loaded_models": ["reranker"], "process_count": 1}
    controls.native.update(name="reranker", backend="kiron_deberta", catalog_managed=True,
                           load_state="loaded", deployment_ids=["bound"], vram_gb=None, ram_gb=None,
                           service_memory=memory)
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "kiron_deberta", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "load_state": "unknown", "actions": ["health"],
    }]}
    row = read(controls.client)
    assert row["operation_state"] == "loaded"
    assert row["service_memory"] == memory
    assert row["configuration"]["vram_gb"] is None and row["configuration"]["ram_gb"] is None


def test_current_configuration_never_overrides_identity_validation(controls):
    controls.native.update(load_state="loaded", deployment_ids=["bound"], runtime_device="cpu", runtime_context_length=1024)
    controls.runtime_read.return_value = {"models": [{
        "name": "alias", "backend": "ollama", "deployment_ids": ["bound"], "registry_ids": [],
        "runtime": True, "installed": True, "load_state": "unknown", "error_code": "conflict",
        "capabilities": {"chat": {"status": "supported", "configuration_constraints": {
            "device": {"allowed_values": ["cpu"]}, "context_tokens": {"allowed_values": [1024]}}}},
    }]}
    row = read(controls.client)
    assert row["operation_state"] == "loaded"
    assert row["verification"]["status"] == "unverified"
    assert row["error_code"] == "conflict"


def test_only_configuration_constraints_are_exposed_from_bound_evidence(runtime):
    capability = runtime.provider.caps.by_name[CapabilityName.CHAT]
    runtime.provider.caps = CapabilitySet({CapabilityName.CHAT: replace(capability, constraints={
        "device": ParameterConstraint(allowed_values=("cpu",)),
        "context_tokens": ParameterConstraint(allowed_values=(1024,)),
        "private_parameter": ParameterConstraint(allowed_values=("/private/path",)),
    })})
    response = runtime.client.get("/api/models/runtime", auth=AUTH)
    assert response.status_code == 200
    constraints = response.json()["models"][0]["capabilities"]["chat"]["configuration_constraints"]
    assert set(constraints) == {"device", "context_tokens"}
    assert constraints["context_tokens"]["allowed_values"] == [1024]
    assert "/private/path" not in response.text


def test_csrf_auth_and_unknown_ids_are_rejected(controls):
    row = read(controls.client)
    body = {"model": row["control_id"], "revision": row["control_revision"], "action": "load"}
    assert controls.client.post("/api/models/control/action", json=body, headers=HEADERS).status_code == 401
    assert controls.client.post("/api/models/control/action", auth=AUTH, json=body).status_code == 403
    assert controls.client.post("/api/models/control/action", auth=AUTH, json=body,
                               headers={**HEADERS, "Origin": "https://foreign.invalid"}).status_code == 403
    assert command(controls.client, {**row, "control_id": "missing"}, "load").status_code == 404
    assert command(controls.client, row, "load", provider="prism").status_code == 400
    controls.handlers["load_model"].assert_not_awaited()


def test_underlying_gpu_denial_is_preserved(controls):
    controls.handlers["load_model"].return_value = JSONResponse({"error": "GPU-Operation blockiert"}, status_code=423)
    response = command(controls.client, read(controls.client), "load")
    assert response.status_code == 423
    assert response.json()["error"]["message"] == "GPU-Operation blockiert"


def test_frontend_renders_one_action_set_and_enforces_disabled_and_confirmation():
    program = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = vm.createContext({console, confirm: () => false});
context.window = context;
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const escapeHtml = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const controls = ['load','load_cpu','unload','delete','health','warmup'].map(id => ({
    id, label: id, enabled: id !== 'load_cpu', reason: id === 'load_cpu' ? 'Nicht verfügbar <Grund>' : null,
    confirmation: id === 'delete' ? 'Dateien löschen?' : null,
}));
const model = {name:'Model <one>', control_id:'opaque', control_revision:'revision', controls,
    operation_state:'loaded', load_state:'unknown', configuration:{device:'gpu',context_tokens:4096},
    verification:{status:'deviating'}};
assert.equal(context.statusLabel(model), 'Geladen');
const statusHtml = context.renderModelStatus({escapeHtml}, model);
assert.ok(!statusHtml.includes('GPU · Kontext 4096'));
assert.ok(!statusHtml.includes('Nachweise abweichend'));
assert.ok(!statusHtml.includes('Status unbekannt'));
const configurationHtml = context.renderModelConfiguration({escapeHtml}, model);
assert.ok(configurationHtml.includes('models-signal-green'));
assert.ok(configurationHtml.includes('GPU · Kontext 4096'));
const evidenceHtml = context.renderModelVerification({escapeHtml}, model);
assert.ok(evidenceHtml.includes('models-signal-red'));
assert.ok(evidenceHtml.includes('Nachweise abweichend'));
assert.ok(evidenceHtml.includes('tabindex="0"'));
assert.ok(evidenceHtml.includes('data-model-tooltip='));
assert.ok(context.renderModelConfiguration({escapeHtml}, {...model, configuration:{}}).includes('models-signal-grey'));
assert.ok(context.renderModelConfiguration({escapeHtml}, {...model, configuration:{device:'cpu'}}).includes('models-signal-yellow'));
assert.ok(context.renderModelVerification({escapeHtml}, {...model, verification:{status:'verified'}}).includes('models-signal-green'));
assert.ok(context.renderModelVerification({escapeHtml}, {...model, verification:{status:'unverified'}}).includes('models-signal-yellow'));
const nomic = {...model, verification:{status:'unverified'}, embedding_profiles:[{
    profile_id:'kiron-nomic-late-v1', endpoint:'/api/embed_late',
    verification:{status:'verified', blocking_reasons:[]},
    index_compatibility_id:'index-proof', query_compatibility_id:'query-proof',
}]};
const modelsTable = context.createModelsTable();
modelsTable.escapeHtml = escapeHtml;
const profileColumn = modelsTable.config.columns.find(column => column.label === 'Profil');
const runtimeColumn = modelsTable.config.columns.find(column => column.label === 'Laufzeit');
assert.ok(profileColumn && runtimeColumn);
assert.ok(!modelsTable.config.columns.some(column => column.label === 'Nachweise'));
const profileHtml = modelsTable.renderCell(nomic[profileColumn.field], profileColumn, {_model: nomic});
const runtimeHtml = modelsTable.renderCell(nomic[runtimeColumn.field], runtimeColumn, {_model: nomic});
assert.ok(profileHtml.includes('models-signal-green') && !profileHtml.includes('models-signal-yellow'));
assert.ok(runtimeHtml.includes('models-signal-yellow') && !runtimeHtml.includes('models-signal-green'));
assert.ok(profileHtml.includes('kiron-nomic-late-v1 (/api/embed_late): verifiziert'));
assert.ok(context.renderModelProfileVerification({escapeHtml}, model).includes('models-signal-grey'));
const profileDetail = context.renderEmbeddingProfileDetail({escapeHtml}, nomic);
assert.ok(profileDetail.includes('index-proof') && profileDetail.includes('query-proof'));
assert.equal(context.statusLabel({...model,operation_state:'unknown',load_state:'loaded'}), 'Status unbekannt');
const row = {id:'row',_model:model};
const html = context.renderModelActions({escapeHtml},row);
assert.equal((html.match(/class="models-actions"/g)||[]).length,1);
assert.equal((html.match(/<button /g)||[]).length,6);
assert.equal((html.match(/<svg /g)||[]).length,6);
assert.ok(html.includes('aria-disabled="true"'));
assert.ok(html.includes('Nicht verfügbar &lt;Grund>'));
assert.ok(!html.includes('Model <one>'));
assert.ok(!html.includes(' disabled')); // Disabled actions remain keyboard-focusable for their explanation.
const attrs={}; const button={disabled:false,setAttribute:(k,v)=>attrs[k]=v};
let requests=[], refreshes=0;
context.loadLocalModels = async () => { refreshes++; };
context.showNotification = () => {};
context.fetch = async (url,options) => {requests.push({url,options});return {ok:true,json:async()=>({status:'ok'})};};
(async()=>{
    await context.performModelAction(model, controls[3], button);
    assert.equal(requests.length,0); // Cancelled delete never reaches the server.
    context.confirm = () => true;
    await context.performModelAction(model, controls[3], button);
    assert.equal(requests[0].url,'/api/models/control/action');
    assert.equal(requests[0].options.headers['X-Kiron-Action'],'models');
    assert.deepEqual(JSON.parse(requests[0].options.body), {model:'opaque',revision:'revision',action:'delete'});
    assert.equal(refreshes,1);
    assert.equal(attrs['aria-busy'],'true');
    context.row=row;
    vm.runInContext('_modelsTable = {getAllData: () => [row]}',context);
    const disabled={dataset:{modelRow:'row',modelAction:'load_cpu'}, disabled:false};
    context.handleModelsTableAction({target:{closest:()=>disabled},preventDefault(){},stopPropagation(){}});
    assert.equal(requests.length,1); // aria-disabled action cannot be activated.
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run(["node", "-e", program, str(Path(__file__).parent / "static/js/tab_models.js"),
                    str(Path(__file__).parent / "static/js/expandable_table.js")],
                   check=True, timeout=5)
