"""Shared runtime, opaque registration and browser boundary; no live services."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

import app as dashboard
import dashboard_runtime as view
from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet,
    CapabilityStatus, DeploymentObservation, DiscoveredModel, DiscoverySnapshot,
    ErrorCode, LifecycleResult, LocalInferenceError, ProviderHealth, ProviderObservation,
    ResolvedDeployment, ResolvedModel, ResolverSnapshot, ResourceProfile, RuntimeFailure,
    RuntimeGeneration, RuntimeImplementation, RuntimeTimeouts, ModelSource,
    ModelLifecycleOperation,
)
from kiron_common.local_model_registry import GGUFRegistrationPolicy, RuntimeModelRegistry
from kiron_common.local_model_registry.composition import build_model_registration_service
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.model_state import RuntimeState
from runtime_service import RuntimeService

AUTH = ("admin", "admin")
HEADERS = {"X-Kiron-Action": "models"}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    generation = RuntimeGeneration("boot", "child")
    implementation = RuntimeImplementation("test-runtime", None, "test-parser")
    profile = ResourceProfile("measured-test", 1024, 128, 128, 1, 4, 40, False, 40, 20)
    deployment = ResolvedDeployment("local.test", BackendType.PRISM, "/private/model.gguf",
        ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100),
        LoaderType.PRISM_GGUF, profile, "b" * 64, ModelSource.LOCAL_REGISTRY, ("local.test",))
    model = ResolvedModel("local.test", deployment, None, None, "c" * 64)
    snapshot = ResolverSnapshot("c" * 64, {deployment.id: deployment}, {model.public_model_id: model})
    evidence = CapabilityEvidence("test-runtime", deployment.artifact_identity.fingerprint,
        "a" * 64, None, None, "test-parser", "b" * 64, "/private/report-with-secret.json", datetime.now(timezone.utc))
    class Provider:
        model_lifecycle_operations = frozenset(ModelLifecycleOperation)
        loaded = False
        changed = False
        unhealthy = False
        calls = []
        def __init__(self):
            self.implementation = implementation
            self.calls = []
            self.caps = CapabilitySet({CapabilityName.CHAT: Capability(CapabilityStatus.SUPPORTED, {}, (evidence,))})
        async def health(self, context):
            if self.unhealthy:
                raise RuntimeError("/private/credentials")
            models = {deployment.id: DeploymentObservation(deployment.id, RuntimeState.LOADED,
                       generation, deployment.configuration_fingerprint)} if self.loaded else {}
            return ProviderObservation(BackendType.PRISM, generation, datetime.now(timezone.utc), ProviderHealth.AVAILABLE, models)
        async def discover(self, context):
            artifact = replace(deployment.artifact_identity, sha256="d" * 64) if self.changed else deployment.artifact_identity
            return DiscoverySnapshot(BackendType.PRISM, "native", datetime.now(timezone.utc),
                (DiscoveredModel(deployment.reference, artifact, True),))
        async def capabilities(self, deployment):
            return self.caps
        async def load(self, deployment, *, context, **kwargs):
            self.calls.append("load")
            self.loaded = True
            return LifecycleResult(context.request_id, await self.health(context), True)
        async def unload(self, deployment, *, context, **kwargs):
            self.calls.append("unload")
            self.loaded = False
            return LifecycleResult(context.request_id, await self.health(context), True)
    provider = Provider()
    root = tmp_path / "admission"
    root.mkdir(mode=0o2770)
    root.chmod(0o2770)
    admission = AdmissionStore(root, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
    resolver = SimpleNamespace(snapshot=AsyncMock(return_value=snapshot))
    service = RuntimeService(resolver=resolver, providers={BackendType.PRISM: provider}, admission=admission,
        measure=lambda: MemorySnapshot(100, 100, time.monotonic()), timeouts=RuntimeTimeouts(1, 1, 1, 1, 1, 1, 1))
    monkeypatch.setattr(dashboard.app.state, "local_inference", service, raising=False)
    return SimpleNamespace(service=service, provider=provider, snapshot=snapshot, admission=admission,
                           client=TestClient(dashboard.app), model=model)


def action(runtime, action="load", **overrides):
    return runtime.client.post("/api/models/runtime/action", auth=AUTH, headers=HEADERS,
        json={"model": runtime.model.public_model_id, "snapshot_revision": runtime.snapshot.revision,
              "action": action, **overrides})


def test_shared_lifecycle_reserves_and_releases_with_confirmed_provider_state(runtime):
    before = runtime.client.get("/api/models/runtime", auth=AUTH)
    assert before.status_code == 200
    row = before.json()["models"][0]
    assert row["load_state"] == "unloaded" and row["actions"] == ["health", "load"]
    assert row["locally_registered"] and not row["catalog_managed"]
    assert row["capabilities"]["chat"]["status"] == "supported"
    assert "/private" not in before.text and "secret" not in before.text
    loaded = action(runtime)
    assert loaded.status_code == 200
    assert loaded.json()["models"][0]["load_state"] == "loaded"
    assert [ticket.phase for ticket in runtime.admission.snapshot()] == ["resident"]
    unloaded = action(runtime, "unload")
    assert unloaded.status_code == 200
    assert unloaded.json()["models"][0]["load_state"] == "unloaded"
    assert runtime.admission.snapshot() == ()
    assert runtime.provider.calls == ["load", "unload"]


def test_stale_snapshot_unknown_fields_and_same_origin_before_any_mutation(runtime):
    assert action(runtime, snapshot_revision="d" * 64).status_code == 409
    assert action(runtime, path="/any/path").status_code == 400
    assert action(runtime, action="restart").status_code == 400
    for headers in ({}, {**HEADERS, "Origin": "https://foreign.example"},
                    {**HEADERS, "Sec-Fetch-Site": "cross-site"}):
        response = runtime.client.post("/api/models/runtime/action", auth=AUTH, headers=headers,
            json={"model": runtime.model.public_model_id, "snapshot_revision": runtime.snapshot.revision, "action": "load"})
        assert response.status_code == 403
    assert runtime.provider.calls == [] and runtime.admission.snapshot() == ()
    assert runtime.client.get("/api/models/runtime").status_code == 401


def test_runtime_error_payload_never_exposes_provider_text(runtime):
    runtime.service.load = AsyncMock(side_effect=LocalInferenceError(RuntimeFailure(ErrorCode.OVERLOADED, "/private/secret")))
    response = action(runtime)
    assert response.status_code == 429 and response.json()["error"]["code"] == "overloaded"
    assert "/private" not in response.text and "secret" not in response.text


def test_evidence_is_invalidated_by_artifact_or_implementation_change(runtime):
    runtime.provider.changed = True
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["installed"] is False and row["actions"] == ["health"]
    assert row["error_code"] == "artifact_missing_or_changed"
    assert all(cap["status"] == "unverified" and not cap["evidence"] for cap in row["capabilities"].values())
    runtime.provider.changed = False
    runtime.provider.implementation = RuntimeImplementation("new-runtime", None, "test-parser")
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["capabilities"]["chat"]["status"] == "unverified"


def test_provider_failure_does_not_hide_other_provider(runtime):
    runtime.service.providers[BackendType.OLLAMA] = SimpleNamespace(
        health=AsyncMock(side_effect=RuntimeError("unreachable")), discover=AsyncMock(side_effect=RuntimeError("unreachable")))
    response = runtime.client.get("/api/models/runtime", auth=AUTH)
    assert response.status_code == 200 and response.json()["models"][0]["installed"] is True


def test_unconfirmed_artifact_has_diagnostic_no_load_and_clears_on_verified_read(runtime):
    discover = runtime.provider.discover
    runtime.provider.discover = AsyncMock(side_effect=TimeoutError("/private/verifier"))
    response = runtime.client.get("/api/models/runtime", auth=AUTH)
    row = response.json()["models"][0]
    assert row["installed"] is None and row["actions"] == ["health"]
    assert row["load_state"] == "unloaded" and row["provider_health"] == "available"
    assert row["error_code"] == "artifact_verification_unconfirmed" and "noch nicht bestätigt" in row["diagnostic"]
    assert all(cap["status"] == "unverified" for cap in row["capabilities"].values())
    assert "/private" not in response.text and not runtime.admission.snapshot()
    runtime.provider.discover = discover
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["installed"] is True and row["actions"] == ["health", "load"]
    assert row["error_code"] is None and row["diagnostic"] is None


@pytest.mark.parametrize("loaded", [False, True])
def test_declared_lifecycle_controls_dashboard_actions_without_provider_branch(runtime, loaded):
    runtime.provider.model_lifecycle_operations = frozenset()
    runtime.provider.loaded = loaded
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["installed"] is True and row["loaded"] is loaded
    assert row["actions"] == ["health"]
    runtime.provider.model_lifecycle_operations = frozenset({ModelLifecycleOperation.LOAD})
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["actions"] == (["health"] if loaded else ["health", "load"])
    runtime.provider.model_lifecycle_operations = frozenset({ModelLifecycleOperation.UNLOAD})
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["actions"] == (["health", "unload"] if loaded else ["health"])
    assert runtime.provider.calls == [] and not runtime.admission.snapshot()


def test_registry_change_after_restart_has_no_stale_provider_badge(runtime):
    first = runtime.client.get("/api/models/runtime", auth=AUTH).json()
    deployment = replace(runtime.model.deployment, provider=BackendType.OLLAMA, loader=LoaderType.OLLAMA)
    model = replace(runtime.model, deployment=deployment, snapshot_revision="d" * 64)
    runtime.service.resolver.snapshot.return_value = ResolverSnapshot("d" * 64, {deployment.id: deployment}, {model.public_model_id: model})
    second = runtime.client.get("/api/models/runtime", auth=AUTH).json()
    assert first["models"][0]["runtime_provider"] == "prism"
    assert second["models"][0]["runtime_provider"] == "ollama"
    assert second["models"][0]["provider_health"] == "unavailable"
    assert all(cap["status"] == "unverified" for cap in second["models"][0]["capabilities"].values())


def gguf(architecture):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    fields = {"general.architecture": architecture,
              "clip.vision.projection_dim" if architecture == "clip" else "qwen35.embedding_length": 5120}
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields))
    for key, value in fields.items():
        data += string(key) + struct.pack("<I", 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack("<I", value)
    return data + b"test tensor bytes"


def test_opaque_model_projector_registration_and_restart(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    model, projector = root / "model.gguf", root / "projector.gguf"
    model.write_bytes(gguf("qwen35")); projector.write_bytes(gguf("clip"))
    profiles = {"approved": GGUFRegistrationPolicy(hashlib.sha256(model.read_bytes()).hexdigest(), "qwen35",
                                                  hashlib.sha256(projector.read_bytes()).hexdigest())}
    service = build_model_registration_service(registry=RuntimeModelRegistry(tmp_path / "registry.json"),
        ollama=SimpleNamespace(list_models=lambda: {"models": []}, show_model=lambda _: {}), huggingface_model_root=tmp_path / "absent",
        gguf_model_root=root, gguf_profiles=profiles)
    monkeypatch.setattr(dashboard, "_model_registration_service", service)
    client = TestClient(dashboard.app)
    response = client.get("/api/models/registration-candidates", auth=AUTH)
    assert response.status_code == 200 and str(root) not in response.text
    payload = response.json()
    assert payload["runtime_profiles"] == [{"id": "approved", "projector_supported": True}]
    candidates = {row["display_name"]: row["candidate_id"] for row in payload["candidates"]}
    body = {"candidate_id": candidates["model.gguf"], "projector_id": candidates["projector.gguf"], "runtime_profile": "approved"}
    assert client.post("/api/models/register", auth=AUTH, json=body).status_code == 403
    assert client.post("/api/models/register", auth=AUTH, headers=HEADERS,
                       json={"runtime_provider": "prism", "reference": str(model)}).status_code == 400
    # Wrong fixed profile / mismatched projector is rejected by the same use case.
    rejected = client.post("/api/models/register", auth=AUTH, headers=HEADERS, json={**body, "runtime_profile": "other"})
    assert rejected.status_code == 422
    result = client.post("/api/models/register", auth=AUTH, headers=HEADERS, json=body)
    assert result.status_code == 201 and str(root) not in result.text
    restarted = RuntimeModelRegistry(tmp_path / "registry.json")
    entry = restarted.list()[0]
    assert entry.reference == str(model) and entry.projector.reference == str(projector)
    assert view.public_registration(entry) == result.json()["model"]
    assert client.post("/api/models/register", auth=AUTH, headers=HEADERS, json=body).status_code == 404


def test_javascript_displays_partial_control_inventory_and_server_warning():
    script = Path(__file__).parent / "static/js/tab_models.js"
    program = '''const fs = require('fs'), vm = require('vm');
const status = {textContent:''}; const sandbox = {console, document:{getElementById:id=>id==='modelsRuntimeStatus'?status:null}};
vm.createContext(sandbox); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
sandbox.fetch = async url => ({ok:true,json:async()=>({warnings:['Natives Inventar offline'],
 models:[{name:'Bonsai',runtime:true,registry_ids:[],deployment_ids:['one'],runtime_provider:'prism'}]})});
(async()=>{await vm.runInContext('fetchLocalModels()',sandbox);
const rows=vm.runInContext('_modelsData',sandbox);
if(rows.length!==1 || rows[0].runtime_provider!=='prism' || !status.textContent.includes('offline')) process.exit(1);
sandbox.fetch = async url => ({ok:true,json:async()=>({models:[]})});
await vm.runInContext('fetchLocalModels()',sandbox);
if(vm.runInContext('_modelsData.length',sandbox)!==0)process.exit(2);
})().catch(()=>process.exit(3));'''
    subprocess.run(["node", "-e", program, str(script)], check=True, timeout=5)


def test_javascript_verification_poll_is_bounded_visible_and_updates_only_observed_state():
    script = Path(__file__).parent / "static/js/tab_models.js"
    program = '''const fs = require('fs'), vm = require('vm');
const status={textContent:''}, timers=new Map(); let table={}, timerId=0, now=0, renders=0, runtimeCalls=0;
const sandbox={console, currentTab:'models', Date:{now:()=>now},
 document:{getElementById:id=>id==='modelsTableContainer'?table:id==='modelsRuntimeStatus'?status:null},
 setTimeout:(fn,ms)=>{timers.set(++timerId,{fn,ms});return timerId;},clearTimeout:id=>timers.delete(id)};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
sandbox.renderModelsTable=()=>{renders++;};
const pending={name:'Bonsai',runtime:true,registry_ids:[],deployment_ids:['one'],installed:null,
 runtime_provider:'prism',operation_state:'unloaded',error_code:'artifact_verification_unconfirmed',actions:['health']};
let verified=false, release=null;
sandbox.fetch=async url=>{if(url.endsWith('/control')){runtimeCalls++;if(release)await release;}
 return {ok:true,json:async()=>({models:[verified?{...pending,installed:true,error_code:null,actions:['health','load']}:pending]})};};
const call=code=>vm.runInContext(code,sandbox);
const fire=async()=>{const entry=timers.entries().next().value;if(!entry)throw Error('missing timer');
 timers.delete(entry[0]);now+=entry[1].ms;entry[1].fn();const work=call('_modelsRefreshPending');if(work)await work;await Promise.resolve();};
(async()=>{
 await call('loadLocalModels()');
 if(runtimeCalls!==1||timers.size!==1||!status.textContent.includes('automatische Nachprüfung'))throw Error('no automatic recheck');
 if(call('statusLabel(_modelsData[0])')!=='Nicht geladen'||call('_modelsData[0].installed')!==null)throw Error('invented installation');
 verified=true;await fire();
 if(runtimeCalls!==2||timers.size||call('_modelsData[0].installed')!==true)throw Error('verified read not applied');
 verified=false;await call('loadLocalModels()');
 sandbox.currentTab='dashboard';const before=runtimeCalls;await fire();
 if(runtimeCalls!==before||timers.size)throw Error('hidden polling');
 sandbox.currentTab='models';await call('loadLocalModels()');now+=90000;await fire();
 if(timers.size||!status.textContent.includes('weiterhin ungeklärt'))throw Error('unbounded polling');
 const expired=runtimeCalls;await Promise.resolve();if(runtimeCalls!==expired)throw Error('late fetch');
 let unblock;release=new Promise(resolve=>unblock=resolve);const first=call('loadLocalModels()');
 const oldRenders=renders;table={};const second=call('loadLocalModels()');
 const inflight=runtimeCalls;unblock();await Promise.all([first,second]);
 if(runtimeCalls!==inflight)throw Error('overlapping refresh');
 if(renders!==oldRenders+1)throw Error('reentered tab did not render shared observation');
 if(renders<4)throw Error('missing render');
})().catch(error=>{console.error(error);process.exit(1);});'''
    subprocess.run(["node", "-e", program, str(script)], check=True, timeout=5)


def test_memory_shortage_is_typed_and_never_calls_provider_load(runtime):
    runtime.service.measure = lambda: MemorySnapshot(0, 100, time.monotonic())
    response = action(runtime)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "overloaded"
    assert runtime.provider.calls == [] and runtime.admission.snapshot() == ()


def test_action_deadline_is_bounded_and_never_reports_loaded(runtime, monkeypatch):
    monkeypatch.setattr(view, "ACTION_TIMEOUT", .03)
    async def hanging_load(*args):
        await asyncio.Event().wait()
    runtime.service.load = hanging_load
    started = time.monotonic()
    response = action(runtime)
    assert time.monotonic() - started < 1
    assert response.status_code == 504 and response.json()["error"]["code"] == "timeout"
    assert runtime.provider.calls == []


def test_healthy_row_does_not_inherit_missing_resource_or_stale_configuration(runtime):
    runtime.provider.loaded = True
    original = runtime.provider.health
    async def stale(context):
        observation = await original(context)
        current = observation.models[runtime.model.deployment.id]
        return replace(observation, models={current.deployment_id: replace(current, configuration_fingerprint="d" * 64)})
    runtime.provider.health = stale
    row = runtime.client.get("/api/models/runtime", auth=AUTH).json()["models"][0]
    assert row["load_state"] == "unknown" and not row["loaded"]
    assert row["error_code"] == "conflict" and "unload" in row["actions"]


def test_runtime_action_cannot_create_arbitrary_command_or_parameter(runtime):
    for extra in ({"command": "stop"}, {"parameters": {"gpu_layers": 65}}, {"reference": "/tmp/model"}):
        assert action(runtime, **extra).status_code == 400
    assert runtime.provider.calls == []


def test_cancel_resistant_disconnect_watcher_cannot_hold_completed_response(monkeypatch):
    monkeypatch.setattr(view, "CLEANUP_TIMEOUT", .01)
    async def exercise():
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class Request:
            app = SimpleNamespace(state=SimpleNamespace(local_inference=object()))
            async def is_disconnected(self):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                return False
        async def done(service, context):
            await entered.wait()
            return {"status": "ok"}
        pending = asyncio.create_task(view._bounded(Request(), done, timeout=.02))
        try:
            result = await asyncio.wait_for(asyncio.shield(pending), .15)
            assert result == {"status": "ok"} and cancelled.is_set()
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.sleep(0)
    asyncio.run(exercise())
