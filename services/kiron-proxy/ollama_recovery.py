"""Operator recovery with a container-bound stop proof and shared admission."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
from uuid import uuid4

import httpx

from kiron_common.gpu_admission import AdmissionError
from kiron_common.gpu_admission.ollama_backend import BackendLock, OllamaBackend


logger = logging.getLogger(__name__)
_operations = set()


def ollama_tickets(store):
    return tuple(t for t in store.snapshot() if t.lifecycle_domain == "ollama" or t.backend_instance is not None)


def revision(tickets):
    return hashlib.sha256(json.dumps([asdict(t) for t in sorted(tickets, key=lambda t: t.operation_id)],
                                     sort_keys=True).encode()).hexdigest()


def status(store):
    try:
        tickets = ollama_tickets(store)
    except AdmissionError as exc:
        return {"blocked": True, "recovery_available": False, "revision": None,
                "message": "Ollama-Zustand nicht lesbar: " + str(exc), "operations": []}
    unknown = [t for t in tickets if t.phase == "unknown"]
    operations = []
    for ticket in tickets:
        seen = time.time() - (store.clock() - ticket.heartbeat_monotonic)
        operations.append({"operation_id": ticket.operation_id, "model": ticket.deployment_id.removeprefix("native:ollama:"),
            "state": ticket.phase, "kind": ticket.kind, "owner": ticket.owner,
            "last_seen_at": datetime.fromtimestamp(seen, timezone.utc).isoformat()})
    names = ", ".join(dict.fromkeys(t.deployment_id.removeprefix("native:ollama:") for t in unknown))
    return {"blocked": bool(unknown), "revision": revision(tickets), "operations": operations,
            "recovery_available": bool(unknown) and all(t.backend_instance and t.phase in {"unknown", "resident"}
                                                       for t in tickets),
            "message": ("Ollama ist durch eine unbestätigte Operation blockiert: " + names
                        + ". Eine Wiederherstellung beendet alle geladenen Ollama-Modelle.") if unknown else ""}


def conflict_message(store, exc):
    state = status(store)
    return state["message"] or str(exc)


async def wait_healthy():
    deadline = time.monotonic() + 15
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get("/api/version")
                if response.status_code == 200 and isinstance(response.json().get("version"), str):
                    return
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
            await asyncio.sleep(.2)
    raise AdmissionError("ollama_start_unconfirmed", "Ollama antwortet nach dem Neustart nicht")


async def _recover(store, expected_revision, backend, health):
    lock = BackendLock(store, exclusive=True).acquire()
    fence = None
    try:
        tickets = ollama_tickets(store)
        if revision(tickets) != expected_revision:
            raise AdmissionError("operation_conflict", "Ollama-Zustand hat sich geändert. Bitte Diagnose aktualisieren.")
        state = await backend.inspect()
        fence = store.begin_backend_recovery(uuid4().hex, backend_instance=state.instance, expected=tickets)
        logger.warning("Ollama recovery started instance=%s operations=%s", state.instance,
                       [t.operation_id for t in tickets])
        await backend.stop_and_verify(state)
        store.confirm_backend_terminated(fence, tickets)
        current = await backend.start_and_verify(state)
        await health()
        # A name change/replacement during health must not inherit this proof.
        observed = await backend.inspect()
        if (observed.instance != current.instance or not observed.running
                or observed.status != "running" or observed.pid <= 0):
            raise AdmissionError("ollama_identity_unknown", "Ollama-Instanz hat sich während der Wiederherstellung geändert")
        store.release(fence.operation_id, owner=fence.owner, generation=fence.generation, confirmed_terminated=True)
        fence = None
        logger.warning("Ollama recovery completed instance=%s", current.instance)
        return {"status": "recovered", "cleared_operations": len(tickets)}
    finally:
        try:
            if fence is not None:
                store.release(fence.operation_id, owner=fence.owner, generation=fence.generation, confirmed_terminated=False)
        finally:
            lock.close()


async def recover(store, expected_revision, *, backend=None, health=None):
    # A browser disconnect must not interrupt stop/proof/start halfway through.
    operation = asyncio.create_task(_recover(store, expected_revision, backend or OllamaBackend(), health or wait_healthy))
    _operations.add(operation)
    def finished(task):
        _operations.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("Ollama recovery failed: %s", error)
    operation.add_done_callback(finished)
    return await asyncio.shield(operation)
