"""Offline test of the exact Dashboard harness module with shared admission."""
import asyncio
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from kiron_common.local_inference import ArtifactFileReference, ResolverSnapshot
from test_dashboard_runtime import runtime  # real RuntimeService / isolated Admission fixture

spec = importlib.util.spec_from_file_location("dashboard_probe", Path(__file__).with_name("probe-dashboard-runtime.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_dashboard_probe_requires_projector_and_closed_kind(runtime):
    with pytest.raises(ValueError):
        asyncio.run(probe.probe(runtime.service, runtime.model, runtime.admission, "wrong"))
    with pytest.raises(ValueError):
        asyncio.run(probe.probe(runtime.service, runtime.model, runtime.admission, "dashboard-runtime"))


def projected_model(runtime):
    # Extend the fixture's actual immutable identity and matching observations.
    deployment = replace(runtime.model.deployment, artifact_identity=replace(runtime.model.deployment.artifact_identity,
        projector=ArtifactFileReference("/private/projector.gguf", "f" * 64, 20)))
    model = replace(runtime.model, deployment=deployment)
    runtime.service.resolver.snapshot.return_value = ResolverSnapshot(runtime.snapshot.revision, {deployment.id: deployment}, {model.public_model_id: model})
    discover = runtime.provider.discover
    async def projected(context):
        value = await discover(context)
        return replace(value, models=(replace(value.models[0], artifact_identity=deployment.artifact_identity),))
    runtime.provider.discover = projected
    return model


def test_dashboard_probe_full_route_contract_and_auth_restore(runtime, tmp_path):
    import app
    model = projected_model(runtime)
    original = app.DASHBOARD_AUTH_USER, app.DASHBOARD_AUTH_PASSWORD
    result = asyncio.run(probe.probe(runtime.service, model, runtime.admission, "dashboard-runtime", report=tmp_path))
    assert result["status"] == "passed"
    assert result["cases"] == ["unauthorized", "initial", "csrf-rejected", "stale-snapshot", "load", "health", "unload"]
    assert len(list(tmp_path.glob("dashboard-*.json"))) == 7
    assert runtime.provider.calls == ["load", "unload"] and not runtime.admission.snapshot()
    assert (app.DASHBOARD_AUTH_USER, app.DASHBOARD_AUTH_PASSWORD) == original
    assert all("/private/" not in path.read_text() for path in tmp_path.glob("*.json"))


def test_dashboard_probe_rechecks_unconfirmed_verification_without_early_load(runtime, tmp_path, monkeypatch):
    model = projected_model(runtime)
    discover = runtime.provider.discover
    calls = 0
    async def delayed(context):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert runtime.provider.calls == [] and not runtime.admission.snapshot()
            raise TimeoutError("initial artifact verification still running")
        return await discover(context)
    runtime.provider.discover = delayed
    monkeypatch.setattr(probe, "VERIFICATION_POLL_SECONDS", .001)
    result = asyncio.run(probe.probe(runtime.service, model, runtime.admission, "dashboard-runtime", report=tmp_path))
    assert result["status"] == "passed" and "initial-recheck-001" in result["cases"]
    initial = probe.model_row(result["results"]["initial"]["body"], model, allow_unverified=True)
    assert initial["installed"] is None and initial["actions"] == ["health"]
    assert initial["error_code"] == "artifact_verification_unconfirmed" and initial["diagnostic"]
    verified = probe.model_row(result["results"]["initial-recheck-001"]["body"], model)
    assert verified["installed"] is True and runtime.provider.calls == ["load", "unload"]


def test_dashboard_probe_verification_timeout_never_loads_or_claims_installed(runtime, monkeypatch):
    model = projected_model(runtime)
    async def unavailable(context):
        raise TimeoutError("unconfirmed")
    runtime.provider.discover = unavailable
    monkeypatch.setattr(probe, "VERIFICATION_SECONDS", .03)
    monkeypatch.setattr(probe, "VERIFICATION_POLL_SECONDS", .001)
    with pytest.raises(TimeoutError):
        asyncio.run(probe.probe(runtime.service, model, runtime.admission, "dashboard-runtime"))
    assert runtime.provider.calls == [] and not runtime.admission.snapshot()


def test_dashboard_probe_rejects_changed_artifact_without_polling(runtime):
    model = projected_model(runtime)
    discover = runtime.provider.discover
    async def changed(context):
        value = await discover(context)
        return replace(value, models=())
    runtime.provider.discover = changed
    from unittest.mock import AsyncMock
    import httpx
    from dashboard_runtime import inventory
    from kiron_common.local_inference import RequestContext
    import time
    async def exercise():
        payload = await inventory(runtime.service, RequestContext("test", time.monotonic() + 1, asyncio.Event()))
        client = type("Client", (), {"get": AsyncMock(return_value=httpx.Response(200, json=payload))})()
        with pytest.raises(ValueError):
            await probe.verified_inventory(client, model, runtime.admission, lambda name, response: response.json())
        client.get.assert_awaited_once()
    asyncio.run(exercise())
    assert runtime.provider.calls == [] and not runtime.admission.snapshot()
