"""Dashboard model controls: one inventory and command contract for all providers.

Adapters reuse the existing guarded use cases. A displayed action is advisory:
every command resolves its opaque ID and checks a fresh server-side inventory.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

from starlette.responses import JSONResponse

import dashboard_runtime


ACTION_LABELS = {
    "load": "Modell laden",
    "load_cpu": "Auf CPU laden",
    "unload": "Modell entladen",
    "delete": "Lokale Modelldateien löschen",
    "health": "Diagnose aktualisieren",
    "warmup": "ColBERT-Warmup ausführen",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _error(code, message, status=409):
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _payload(result):
    if isinstance(result, JSONResponse):
        if result.status_code >= 400:
            return None
        result = json.loads(result.body)
    return result if isinstance(result, dict) and isinstance(result.get("models"), list) else None


def merge_inventory(native, runtime):
    """Match by deployment/registry identity, never by a display name."""
    observations = [[] for _ in native]
    unmatched = []
    for observed in runtime:
        matches = [index for index, row in enumerate(native)
                   if row.get("backend") == observed.get("backend") and (
                       (row.get("registry_id") and row["registry_id"] in observed.get("registry_ids", []))
                       or set(row.get("deployment_ids", [])) & set(observed.get("deployment_ids", [])))]
        if len(matches) > 1:
            raise ValueError("Ambiguous model inventory identity")
        if matches:
            observations[matches[0]].append(dict(observed))
        else:
            unmatched.append((dict(observed), None))
    rows = []
    for original, observed in zip(native, observations):
        row = dict(original)
        if len(observed) == 1:
            row.update(observed[0], name=original["name"])
            # One runner can expose several independently verified profiles.
            # Keep its complete identity even if only one profile was observed.
            for key in ("deployment_ids", "profile_ids"):
                row[key] = sorted(set(original.get(key, [])) | set(row.get(key, [])))
        elif observed:
            if not original.get("catalog_managed"):
                raise ValueError("Ambiguous model inventory identity")
            revisions = {item.get("snapshot_revision") for item in observed}
            if len(revisions) != 1:
                raise ValueError("Inconsistent runtime inventory snapshot")
            observed.sort(key=lambda item: tuple(item.get("deployment_ids", [])))
            row.update(runtime=True, runtime_provider=original["backend"],
                       snapshot_revision=next(iter(revisions)),
                       runtime_deployments=observed,
                       api_model_ids=sorted({name for item in observed for name in item.get("api_model_ids", [])}))
            # Profile observations retain their own capability and identity
            # checks; no profile can replace another profile's proof.
        rows.append((row, original))
    return rows + unmatched


def project_observation(row, native):
    """Separate native residency, observed configuration and evidence matching.

    Shared runtime state remains the inference validator's observation. It is
    not a substitute for the native service's independently observed residency.
    No evidence or lifecycle permission is manufactured by this projection.
    """
    source = native if native and (native.get("backend") == "ollama" or native.get("catalog_managed")) else row
    state = source.get("load_state", "unknown")
    row["operation_state"] = state
    configuration = {"device": None, "context_tokens": None, "vram_gb": None, "ram_gb": None}
    if state == "loaded":
        configuration.update(vram_gb=source.get("vram_gb"), ram_gb=source.get("ram_gb"))
        if source.get("backend") == "ollama":
            configuration["context_tokens"] = source.get("runtime_context_length")
            configuration["device"] = source.get("runtime_device")
    row["configuration"] = configuration
    checks = []
    for observed in row.get("runtime_deployments", []):
        project_observation(observed, native)
        checks.extend({**check, "deployment_ids": observed["deployment_ids"]}
                      for check in observed["verification"]["checks"])
    for capability, info in row.get("capabilities", {}).items():
        if info.get("status") != "supported":
            continue
        expected = info.get("configuration_constraints", {})
        status = "verified"
        if state != "loaded":
            status = "unverified"
        elif expected:
            for key, constraint in expected.items():
                observed = configuration.get(key)
                if observed is None:
                    if status != "deviating":
                        status = "unverified"
                    continue
                allowed = constraint.get("allowed_values")
                minimum, maximum = constraint.get("minimum"), constraint.get("maximum")
                if ((allowed is not None and observed not in allowed)
                        or (minimum is not None and observed < minimum)
                        or (maximum is not None and observed > maximum)):
                    status = "deviating"
        elif row.get("load_state") != "loaded" or row.get("error_code"):
            status = "unverified"
        if status == "verified" and (row.get("installed") is not True
                or row.get("load_state") != "loaded" or row.get("error_code")):
            status = "unverified"
        checks.append({"capability": capability, "status": status, "expected": expected})
    statuses = [check["status"] for check in checks]
    statuses.extend(observed["verification"]["status"] for observed in row.get("runtime_deployments", []))
    status = ("deviating" if "deviating" in statuses else
              "verified" if statuses and all(value == "verified" for value in statuses) else "unverified")
    row["verification"] = {"status": status, "checks": checks}


def project_controls(row, native, *, complete, cpu_verified):
    """Return public capabilities plus private command routes."""
    routes = {"health": "health"}
    backend = row.get("backend")
    if native and backend == "ollama":
        routes.update(load="ollama_load", load_cpu="ollama_cpu", unload="ollama_unload", delete="ollama_delete")
    elif native and backend == "kiron_embeddings" and native.get("catalog_managed"):
        routes.update(load="embedding_load", unload="embedding_unload")
        if native.get("embedding_kind") == "colbert":
            routes["warmup"] = "embedding_warmup"
    elif native and backend == "kiron_deberta" and native.get("catalog_managed"):
        routes.update(load="deberta_load", unload="deberta_unload")
    # Runtime-only deployments keep their measured profiles and provider gates.
    for action in ("load", "unload"):
        if action in row.get("lifecycle_operations", []) and action not in routes:
            routes[action] = "runtime"

    controls = []
    for action, label in ACTION_LABELS.items():
        route = routes.get(action)
        # Native name-based operations observe the native runner. A shared
        # profile mismatch must not turn a known native runner into "unloaded".
        state = (native if route and route.startswith(("ollama_", "embedding_", "deberta_")) else row).get("load_state", "unknown")
        # Embedding inference discovery proves resident artifacts only. Native
        # catalog discovery can still confirm local files after an unload.
        installed = (native if route and route.startswith(("embedding_", "deberta_")) else row).get("installed")
        supported = action in routes
        reason = None
        if not supported:
            reason = {
                "load_cpu": "Für diesen Modelldienst ist keine separate CPU-Ladeaktion verfügbar.",
                "delete": "Dateilöschung ist nur für lokale Ollama-Modelle verfügbar.",
                "warmup": "Ein separater Warmup ist nur für ColBERT verfügbar.",
            }.get(action, row.get("diagnostic") or (
                "Bereits geladen." if action == "load" and state == "loaded" else
                "Bereits entladen." if action == "unload" and state == "unloaded" else
                "Für diesen Zustand ist keine freigegebene Laufzeitaktion verfügbar."))
        elif action != "health":
            if row.get("ollama_blocked"):
                reason = row["ollama_blocked"]
            elif not complete:
                reason = "Modellinventar unvollständig. Bitte Diagnose aktualisieren."
            elif route == "runtime":
                if action not in row.get("actions", []):
                    reason = row.get("diagnostic") or (
                        "Ein freigegebenes Ressourcenprofil zum Laden fehlt." if action == "load" and not row.get("resource_profile")
                        else "Der Modelldienst gibt diese Aktion derzeit nicht frei.")
            elif action != "unload" and installed is not True:
                reason = "Modell fehlt oder seine lokale Identität ist noch nicht bestätigt."
            elif state not in {"loaded", "unloaded"}:
                reason = "Modellzustand ist unbekannt oder eine Aktion läuft. Bitte Diagnose aktualisieren."
            elif action in {"load", "load_cpu"} and state != "unloaded":
                reason = "Bereits geladen. Vor einem Wechsel bitte entladen."
            elif action == "unload" and state != "loaded":
                reason = "Bereits entladen."
            elif action == "delete" and state != "unloaded":
                reason = "Vor dem Löschen bitte entladen."
            elif action == "load_cpu" and not cpu_verified:
                reason = "CPU-Laden ist für die laufende Ollama-Version nicht verifiziert."
        controls.append({"id": action, "label": label, "supported": supported,
                         "enabled": reason is None, "reason": reason,
                         "confirmation": ("Lokale Modelldateien wirklich löschen?" if action == "delete" else None)})

    row["control_id"] = _digest([backend, row.get("deployment_ids"), row.get("registry_id"), row["name"]])
    row["controls"] = controls
    row["control_revision"] = _digest({key: row.get(key) for key in (
        "control_id", "snapshot_revision", "catalog_digest", "installed", "load_state", "controls",
        "error_code", "resource_profile", "canonical_model_id")})
    return routes


class ModelControl:
    def __init__(self, *, read_native, read_runtime, runtime_action, handlers, cpu_verified,
                 read_ollama_state=None):
        self.read_native = read_native
        self.read_runtime = read_runtime
        self.runtime_action = runtime_action
        self.handlers = handlers
        self.cpu_verified = cpu_verified
        self.read_ollama_state = read_ollama_state

    async def inventory(self, request):
        results = await asyncio.gather(self.read_native(), self.read_runtime(request), return_exceptions=True)
        payloads = [_payload(result) for result in results]
        warnings = [f"{name} ist derzeit nicht verfügbar."
                    for name, result in zip(("Natives Inventar", "Gemeinsame Laufzeit"), payloads) if result is None]
        if all(result is None for result in payloads):
            return _error("provider_unavailable", "Modellinventare sind derzeit nicht erreichbar.", 503), {}
        try:
            merged = merge_inventory(*(result["models"] if result else [] for result in payloads))
        except ValueError:
            return _error("conflict", "Modellidentität ist nicht eindeutig. Bitte Inventare prüfen."), {}
        commands = {}
        ollama_state = await asyncio.to_thread(self.read_ollama_state) if self.read_ollama_state else None
        for row, native in merged:
            if row.get("backend") == "ollama" and ollama_state and ollama_state["blocked"]:
                row["ollama_blocked"] = ollama_state["message"]
            project_observation(row, native)
            routes = project_controls(row, native, complete=not warnings, cpu_verified=self.cpu_verified())
            if row["control_id"] in commands:
                return _error("conflict", "Modellidentität ist nicht eindeutig."), {}
            commands[row["control_id"]] = (row, native, routes)
        return {"status": "ok", "models": [row for row, _ in merged], "warnings": warnings,
                "ollama_recovery": ollama_state}, commands

    async def execute(self, request, body):
        if not dashboard_runtime.mutation_allowed(request):
            return _error("csrf_rejected", "Aktion nur vom eigenen Dashboard erlaubt.", 403)
        if (type(body) is not dict or set(body) != {"model", "revision", "action"}
                or any(type(value) is not str for value in body.values())
                or body["action"] not in ACTION_LABELS):
            return dashboard_runtime.error_response("invalid_request")
        payload, commands = await self.inventory(request)
        if isinstance(payload, JSONResponse):
            return payload
        command = commands.get(body["model"])
        if command is None:
            return dashboard_runtime.error_response("model_not_found")
        row, native, routes = command
        if body["action"] != "health" and row["control_revision"] != body["revision"]:
            return _error("conflict", "Modellzustand oder Freigabe hat sich geändert. Bitte erneut prüfen.")
        action = body["action"]
        control = next(item for item in row["controls"] if item["id"] == action)
        if not control["enabled"]:
            return _error("action_unavailable", control["reason"])
        route = routes[action]
        if route == "health":
            return {"status": "ok"}
        if route == "runtime":
            result = await self.runtime_action(request, {
                "model": row["canonical_model_id"], "snapshot_revision": row["snapshot_revision"], "action": action})
        else:
            # No client-supplied provider/name/force reaches these use cases.
            name = native["name"]
            arguments = ({"name": name} if route.startswith("ollama_") else {"model": name})
            if route in {"ollama_load", "ollama_cpu"}:
                arguments["gpu"] = route == "ollama_load"
            result = await self.handlers[route](arguments)
        if isinstance(result, JSONResponse) and result.status_code >= 400:
            data = json.loads(result.body)
            error = data.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            return _error(error.get("code", "provider_error") if isinstance(error, dict) else data.get("code", "provider_error"),
                          message or "Der Modelldienst hat die Aktion nicht bestätigt.", result.status_code)
        return {"status": "ok"}
