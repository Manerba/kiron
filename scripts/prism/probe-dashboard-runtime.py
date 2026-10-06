"""Actual dashboard ASGI routes against the harness's isolated runtime service.

No process, GPU, registry, unit or DB composition is created here. The enclosing
controller harness owns its cold model, deadline, admission and final teardown.
This probe loads and unloads once and performs no model generation.
"""
import asyncio
import json
from pathlib import Path
import secrets

import httpx

KINDS = frozenset(("dashboard-runtime",))
VERIFICATION_SECONDS = 90.0
VERIFICATION_POLL_SECONDS = 1.0


def model_row(payload, model, *, allow_unverified=False):
    if type(payload) is not dict or payload.get("status") != "ok" or type(payload.get("models")) is not list:
        raise ValueError("dashboard inventory shape failed")
    rows = [row for row in payload["models"] if model.deployment.id in row.get("deployment_ids", [])]
    if len(rows) != 1 or model.api_model_id not in rows[0].get("api_model_ids", []):
        raise ValueError("isolated model missing or duplicated in dashboard")
    row = rows[0]
    if (row.get("runtime_provider") != model.deployment.provider.value
            or row.get("snapshot_revision") != model.snapshot_revision
            or (row.get("installed") is not True and not (allow_unverified and row.get("installed") is None))
            or row.get("projector") is not True
            or row.get("resource_profile", {}).get("id") != model.deployment.resource_profile.id):
        raise ValueError("dashboard identity/profile differs from the isolated model")
    if row.get("locally_registered") is not True or row.get("catalog_managed") is not False:
        raise ValueError("dashboard registration origin differs")
    from kiron_common.local_inference import CapabilityName
    caps = row.get("capabilities")
    if type(caps) is not dict or set(caps) != {name.value for name in CapabilityName}:
        raise ValueError("dashboard capabilities are incomplete")
    for cap in caps.values():
        if (type(cap) is not dict or cap.get("status") not in {"supported", "unsupported", "unverified"}
                or type(cap.get("evidence")) is not list
                or (cap["status"] == "supported" and not cap["evidence"])):
            raise ValueError("dashboard claims capability without evidence")
    encoded = json.dumps(payload)
    paths = [model.deployment.reference, model.deployment.artifact_identity.projector.reference]
    if any(path in encoded for path in paths):
        raise ValueError("dashboard response exposes private artifact path")
    return row


async def verified_inventory(client, model, store, save):
    """Retry bounded observations; never turn an unverified artifact into installed."""
    async with asyncio.timeout(VERIFICATION_SECONDS):
        attempt = 0
        while True:
            response = await client.get("/api/models/runtime")
            payload = save("initial" if not attempt else f"initial-recheck-{attempt:03d}", response)
            row = model_row(payload, model, allow_unverified=True)
            if (response.status_code != 200 or row["load_state"] != "unloaded"
                    or row["provider_health"] not in {"available", "startable"} or store.snapshot()):
                raise ValueError("dashboard needs the isolated cold model without admission")
            if row["installed"] is True:
                if "load" not in row["actions"]:
                    raise ValueError("verified cold model has no dashboard load action")
                return row
            if (row["actions"] != ["health"] or row["error_code"] != "artifact_verification_unconfirmed"
                    or not row["diagnostic"] or any(cap["status"] != "unverified" for cap in row["capabilities"].values())):
                raise ValueError("dashboard unverified artifact is not safely represented")
            attempt += 1
            await asyncio.sleep(VERIFICATION_POLL_SECONDS)


async def probe(service, model, store, kind, *, report=None):
    if kind not in KINDS or model.deployment.artifact_identity.projector is None:
        raise ValueError("isolated dashboard model/projector fixture required")
    import app as dashboard
    username, password = "isolated-" + secrets.token_hex(8), secrets.token_hex(32)
    prior_auth = dashboard.DASHBOARD_AUTH_USER, dashboard.DASHBOARD_AUTH_PASSWORD
    prior_runtime = getattr(dashboard.app.state, "local_inference", None)
    dashboard.DASHBOARD_AUTH_USER, dashboard.DASHBOARD_AUTH_PASSWORD = username, password
    dashboard.app.state.local_inference = service
    results = {}

    def save(name, response):
        payload = response.json()
        encoded = json.dumps(payload)
        if username in encoded or password in encoded:
            raise ValueError("dashboard response exposes fixture credentials")
        value = {"status": response.status_code, "body": payload}
        results[name] = value
        if report is not None:
            with (Path(report) / ("dashboard-" + name + ".json")).open("x") as stream:
                json.dump(value, stream, indent=2)
        return payload

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(dashboard.app),
                base_url="http://isolated-dashboard", timeout=None) as client:
            unauthorized = await client.get("/api/models/runtime")
            save("unauthorized", unauthorized)
            if unauthorized.status_code != 401:
                raise ValueError("dashboard authentication was bypassed")
            client.auth = (username, password)
            baseline = store.snapshot()
            if baseline:
                raise ValueError("dashboard probe must start with empty isolated admission")
            await verified_inventory(client, model, store, save)
            body = {"model": model.api_model_id, "snapshot_revision": model.snapshot_revision, "action": "load"}
            csrf = await client.post("/api/models/runtime/action", json=body)
            stale = await client.post("/api/models/runtime/action", headers={"X-Kiron-Action": "models"},
                                      json={**body, "snapshot_revision": "0" * 64})
            save("csrf-rejected", csrf); save("stale-snapshot", stale)
            if csrf.status_code != 403 or stale.status_code != 409 or store.snapshot() != baseline:
                raise ValueError("negative dashboard action mutated isolated admission")
            for action, state in (("load", "loaded"), ("health", "loaded"), ("unload", "unloaded")):
                response = await client.post("/api/models/runtime/action", headers={"X-Kiron-Action": "models"},
                                            json={**body, "action": action})
                row = model_row(save(action, response), model)
                if response.status_code != 200 or row["load_state"] != state:
                    raise ValueError("dashboard did not confirm " + action)
                tickets = store.snapshot()
                if state == "loaded" and not any(ticket.phase == "resident" and ticket.deployment_id == model.deployment.id for ticket in tickets):
                    raise ValueError("dashboard loaded state lacks resident admission")
                if state == "unloaded" and tickets:
                    raise ValueError("dashboard unload left isolated admission tickets")
            return {"status": "passed", "cases": list(results), "results": results,
                    "scope": "Actual authenticated dashboard ASGI routes; one native load/health/unload, no generation, no browser or production changes."}
    finally:
        dashboard.DASHBOARD_AUTH_USER, dashboard.DASHBOARD_AUTH_PASSWORD = prior_auth
        if prior_runtime is None:
            del dashboard.app.state.local_inference
        else:
            dashboard.app.state.local_inference = prior_runtime
