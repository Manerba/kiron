"""Safe dashboard projection of the shared resolver and runtime use cases."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import time
import uuid

from starlette.responses import JSONResponse

from kiron_common.local_inference import (
    CapabilitySet, CapabilityStatus, ErrorCode, LocalInferenceError, ProviderHealth,
    RequestContext, RuntimeFailure, ProviderObservation, ModelLifecycleOperation,
)
from kiron_common.model_state import RuntimeState
from provider_transport import with_context

READ_TIMEOUT = 15.0
ACTION_TIMEOUT = 180.0
CLEANUP_TIMEOUT = .1


def _observe_task(task):
    if not task.cancelled():
        task.exception()

_MESSAGES = {
    "invalid_request": "Die Anfrage ist ungültig.",
    "model_not_found": "Das Modell ist nicht mehr registriert. Bitte aktualisieren.",
    "invalid_configuration": "Registry, Artefakte oder Laufzeitprofil sind inkonsistent.",
    "provider_unavailable": "Der Modelldienst ist nicht erreichbar oder nicht startbereit.",
    "provider_error": "Der Modelldienst hat keinen gültigen Zustand bestätigt.",
    "unsupported_capability": "Diese Aktion ist für das Modell nicht belegt oder freigegeben.",
    "conflict": "Modellzustand, Registry oder Ressourcenbelegung haben sich geändert.",
    "overloaded": "Für diese Aktion sind derzeit nicht genügend Ressourcen verfügbar.",
    "timeout": "Der Modelldienst hat die Aktion nicht rechtzeitig bestätigt.",
    "cancelled": "Die Anfrage wurde abgebrochen; der Laufzeitzustand muss erneut geprüft werden.",
}
_STATUS = {"invalid_request": 400, "model_not_found": 404, "conflict": 409,
           "overloaded": 429, "unsupported_capability": 400, "timeout": 504, "cancelled": 409}


def error_response(code):
    code = code if code in _MESSAGES else "provider_error"
    return JSONResponse({"status": "error", "error": {"code": code, "message": _MESSAGES[code]}},
                        status_code=_STATUS.get(code, 503))


def mutation_allowed(request):
    """A browser cross-origin form cannot supply this non-simple header."""
    return (request.headers.get("content-type", "").split(";")[0].strip().lower() == "application/json"
            and request.headers.get("x-kiron-action") == "models"
            and request.headers.get("sec-fetch-site", "none") in {"same-origin", "none"}
            and request.headers.get("origin", str(request.base_url).rstrip("/"))
            == str(request.base_url).rstrip("/"))


def candidate_id(candidate):
    return hashlib.sha256((candidate.runtime_provider.value + "\0" + candidate.reference).encode()).hexdigest()


def public_candidate(candidate):
    return {"candidate_id": candidate_id(candidate), "runtime_provider": candidate.runtime_provider.value,
            "artifact_origin": candidate.artifact_origin.value, "artifact_format": candidate.artifact_format.value,
            "display_name": candidate.display_name}


def public_registration(entry):
    return {"id": entry.id, "runtime_provider": entry.runtime_provider.value,
            "artifact_origin": entry.artifact_origin.value, "artifact_format": entry.artifact_format.value,
            "display_name": entry.display_name, "loader": entry.loader.value,
            "runtime_profile": entry.runtime_profile, "projector": entry.projector is not None,
            "registered_at": entry.registered_at.isoformat().replace("+00:00", "Z")}


async def _bounded(request, fn, *, timeout):
    context = RequestContext(uuid.uuid4().hex, time.monotonic() + timeout, asyncio.Event())
    async def watch():
        while not context.cancellation.is_set():
            if await request.is_disconnected():
                context.cancellation.set()
                return
            await asyncio.sleep(.05)
    watcher = asyncio.create_task(watch())
    try:
        service = getattr(request.app.state, "local_inference", None)
        if service is None:
            return error_response("provider_unavailable")
        return await with_context(fn(service, context), context)
    except LocalInferenceError as exc:
        return error_response(exc.failure.code.value)
    except Exception:
        return error_response("provider_error")
    finally:
        context.cancellation.set()
        watcher.cancel()
        await asyncio.wait({watcher}, timeout=CLEANUP_TIMEOUT)
        watcher.add_done_callback(_observe_task)


def _capabilities(capabilities, deployment, provider, installed):
    effective = capabilities.for_deployment(deployment, provider.implementation)
    result = {}
    for name, cap in effective.by_name.items():
        status = cap.status if installed else CapabilityStatus.UNVERIFIED
        # Evidence references may contain private report paths. Publish a stable
        # digest and date, never the original string or provider transport details.
        evidence = [{"id": hashlib.sha256(item.test_reference.encode()).hexdigest(),
                     "observed_at": item.observed_at.isoformat()}
                    for item in cap.evidence] if status is CapabilityStatus.SUPPORTED else []
        configuration_constraints = {
            key: {"allowed_values": list(value.allowed_values) if value.allowed_values is not None else None,
                  "minimum": value.minimum, "maximum": value.maximum}
            for key, value in cap.constraints.items() if key in {"device", "context_tokens"}
        } if status is CapabilityStatus.SUPPORTED else {}
        result[name.value] = {"status": status.value, "evidence": evidence,
                              "configuration_constraints": configuration_constraints}
    return result


async def inventory(service, context):
    snapshot = await with_context(service.resolver.snapshot(), context)
    async def inspect(key, provider):
        async def read(method):
            child = replace(context, deadline_monotonic=min(context.deadline_monotonic, time.monotonic() + 3))
            try:
                return await with_context(method(child), child)
            except Exception:
                return None
        observation, discovery = await asyncio.gather(read(provider.health), read(provider.discover))
        if observation is None:
            observation = ProviderObservation(key, None, datetime.now(timezone.utc), ProviderHealth.UNAVAILABLE)
        return key, (observation, discovery)
    inspected = dict(await asyncio.gather(*(inspect(key, provider) for key, provider in service.providers.items())))
    observations = {key: values[0] for key, values in inspected.items()}
    discoveries = {key: values[1] for key, values in inspected.items()}
    grouped = {}
    for model in snapshot.models.values():
        grouped.setdefault(model.deployment.id, []).append(model)
    rows = []
    for deployment_id, models in grouped.items():
        deployment = models[0].deployment
        model_ids = sorted({model.api_model_id for model in models})
        model = next(item for item in models if item.public_model_id == model_ids[0])
        provider = service.providers.get(deployment.provider)
        observation = observations.get(deployment.provider)
        discovery = discoveries.get(deployment.provider)
        installed = None
        identity = deployment.artifact_identity.content_key
        if identity is not None and discovery is not None and discovery.error is None:
            installed = any(
                item.installed and item.reference == deployment.reference and item.artifact_identity.content_key == identity
                for item in discovery.models)
        health = observation.health if observation else ProviderHealth.UNAVAILABLE
        resident = observation.models.get(deployment.id) if observation else None
        state = resident.state if resident else (RuntimeState.UNLOADED if health in
                {ProviderHealth.AVAILABLE, ProviderHealth.STARTABLE} else RuntimeState.UNKNOWN)
        if resident and resident.configuration_fingerprint not in {None, deployment.configuration_fingerprint}:
            state = RuntimeState.UNKNOWN
        capabilities = CapabilitySet()
        if provider:
            try:
                capabilities = await with_context(provider.capabilities(deployment), replace(context,
                    deadline_monotonic=min(context.deadline_monotonic, time.monotonic() + 1)))
            except Exception:
                pass
        capability_rows = (_capabilities(capabilities, deployment, provider, installed is True)
                           if provider else {name.value: {"status": "unverified", "evidence": []}
                                             for name in capabilities.by_name})
        profile = deployment.resource_profile
        actions = ["health"]
        if (provider and ModelLifecycleOperation.LOAD in provider.model_lifecycle_operations
                and installed is True and profile and health in {ProviderHealth.AVAILABLE, ProviderHealth.STARTABLE}
                and state is RuntimeState.UNLOADED):
            actions.append("load")
        if (provider and ModelLifecycleOperation.UNLOAD in provider.model_lifecycle_operations
                and resident is not None and health is ProviderHealth.AVAILABLE and state is not RuntimeState.UNLOADED):
            actions.append("unload")
        error = resident.error if resident and resident.error else (observation.error if observation else None)
        code = error.code.value if error else ("artifact_missing_or_changed" if installed is False else None)
        if resident and resident.configuration_fingerprint not in {None, deployment.configuration_fingerprint}:
            code = "conflict"
        diagnostic = (_MESSAGES.get(code, "Der Zustand muss erneut geprüft werden.") if code else None)
        if code == "artifact_missing_or_changed":
            diagnostic = "Registriertes Artefakt oder Projektor fehlt oder seine Identität stimmt nicht mehr."
        if health is ProviderHealth.UNAVAILABLE and code is None:
            code, diagnostic = "provider_unavailable", _MESSAGES["provider_unavailable"]
        if installed is None and code is None:
            code = "artifact_verification_unconfirmed"
            diagnostic = "Die Artefaktprüfung ist noch nicht bestätigt. Der Zustand wird erneut geprüft; Laden bleibt bis zum Nachweis gesperrt."
        rows.append({"name": deployment.reference.rsplit("/", 1)[-1] if deployment.reference.startswith("/") else deployment.reference,
            "canonical_model_id": model.api_model_id,
            "runtime": True, "runtime_provider": deployment.provider.value, "backend": deployment.provider.value,
            "deployment_ids": [deployment_id], "registry_ids": list(deployment.registry_ids),
            "profile_ids": sorted({m.profile_id for m in models if m.profile_id}), "api_model_ids": model_ids,
            "registry_id": deployment.registry_ids[0] if deployment.registry_ids else None,
            "catalog_managed": deployment.source.value == "catalog", "registered": bool(deployment.registry_ids), "locally_registered": bool(deployment.registry_ids),
            "local_only": deployment.source.value == "local_registry", "configured": True,
            "source": deployment.artifact_identity.origin.value, "format": deployment.artifact_identity.format.value,
            "loader": deployment.loader.value, "installed": installed, "loaded": state is RuntimeState.LOADED,
            "load_state": state.value, "provider_health": health.value,
            "observed_at": observation.observed_at.isoformat() if observation else None,
            "capabilities": capability_rows, "resource_profile": asdict(profile) if profile else None,
            "projector": deployment.artifact_identity.projector is not None,
            "lifecycle_operations": sorted(operation.value for operation in provider.model_lifecycle_operations) if provider else [],
            "actions": actions, "error_code": code, "diagnostic": diagnostic, "snapshot_revision": snapshot.revision})
    return {"status": "ok", "snapshot_revision": snapshot.revision, "models": rows}


async def runtime_inventory(request):
    return await _bounded(request, inventory, timeout=READ_TIMEOUT)


async def runtime_action(request, body):
    if not mutation_allowed(request):
        return JSONResponse({"error": {"code": "csrf_rejected", "message": "Aktion nur vom eigenen Dashboard erlaubt."}}, status_code=403)
    if (type(body) is not dict or set(body) != {"model", "snapshot_revision", "action"}
            or any(type(body[key]) is not str for key in body)
            or body["action"] not in {"load", "unload", "health"}):
        return error_response("invalid_request")
    async def perform(service, context):
        snapshot = await with_context(service.resolver.snapshot(), context)
        if snapshot.revision != body["snapshot_revision"]:
            raise LocalInferenceError(RuntimeFailure(ErrorCode.CONFLICT, "Stale resolver snapshot"))
        model = snapshot.resolve(body["model"])
        if body["action"] != "health":
            await with_context(getattr(service, body["action"])(model, context), context)
        # Re-read observations after the mutation. Never synthesize success state.
        return await inventory(service, context)
    return await _bounded(request, perform, timeout=ACTION_TIMEOUT)
