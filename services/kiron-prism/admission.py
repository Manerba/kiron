"""Controller callbacks against the proxy-owned shared GPU reservation."""
from __future__ import annotations

import asyncio
import json
import time

from kiron_common.gpu_admission import AdmissionError, AdmissionStore


OWNER = "kiron-proxy"


def generation_key(generation):
    return json.dumps([generation.boot_id, generation.process_id], separators=(",", ":"))


class ControllerAdmission:
    def __init__(self, store: AdmissionStore):
        self.store = store
        # Binds only operations this controller validated before its own spawn.
        self._loads = {}

    async def _tickets(self):
        return await asyncio.to_thread(self.store.snapshot)

    async def validate(self, operation_id, deployment, generation, action):
        tickets = await self._tickets()
        key = generation_key(generation)
        ticket = next((t for t in tickets if t.operation_id == operation_id), None)
        if (ticket is None or ticket.owner != OWNER or ticket.generation != key
                or ticket.deployment_id != deployment.id or ticket.kind != action
                or ticket.phase not in {"reserved", "active"}):
            raise AdmissionError("operation_conflict", "matching live admission ticket missing")
        if action == "load":
            profile = deployment.resource_profile
            if (ticket.resident_slot != "prism" or profile is None or ticket.gpu_bytes < profile.gpu_memory_bytes
                    or ticket.host_bytes < profile.host_memory_bytes):
                raise AdmissionError("operation_conflict", "reservation does not cover the resource profile")
            self._loads[operation_id] = (deployment.id, key, None)
        elif action != "unload":
            raise AdmissionError("operation_conflict", "unsupported lifecycle action")

    async def loaded(self, operation_id, deployment, generation):
        prior = self._loads.get(operation_id)
        if prior is None or prior[0] != deployment.id:
            raise AdmissionError("operation_conflict", "load was not admitted by this controller")
        key = generation_key(generation)
        await asyncio.to_thread(self.store.transition, operation_id, owner=OWNER,
                                expected_generation=prior[1], phase="resident", generation=key)
        self._loads[operation_id] = (deployment.id, key, key)

    async def rejected(self, operation_id, deployment_id, generation):
        """Called only by an executed load task that has not attempted a spawn."""
        key = generation_key(generation)
        prior = self._loads.get(operation_id)
        if prior is not None and prior != (deployment_id, key, None):
            raise AdmissionError("operation_conflict", "operation may own a live process")
        await asyncio.to_thread(self.store.reject_load, operation_id, owner=OWNER,
                                generation=key, deployment_id=deployment_id)
        self._loads.pop(operation_id, None)

    async def heartbeat(self, operation_id, deployment, generation):
        prior = self._loads.get(operation_id)
        key = generation_key(generation)
        if prior is None or prior != (deployment.id, key, key):
            raise AdmissionError("operation_conflict", "resident generation is not owned")
        await asyncio.to_thread(self.store.heartbeat, operation_id, owner=OWNER, generation=key)

    async def drain(self, operation_id, deployment, generation, deadline_monotonic):
        key = generation_key(generation)
        while True:
            requests = [t for t in await self._tickets() if t.deployment_id == deployment.id
                        and t.generation == key and t.kind == "request"]
            if not requests:
                return
            # Unknown requests are still work until the backend proves otherwise.
            if time.monotonic() >= deadline_monotonic:
                raise AdmissionError("resource_conflict", "inference drain deadline exceeded")
            await asyncio.sleep(min(0.05, max(0, deadline_monotonic - time.monotonic())))

    async def terminated(self, operation_id, deployment, generation):
        key = generation_key(generation)
        # A failed pre-readiness load still owns the original reservation; there
        # was no resident transition to the new process generation in that case.
        owned = [(op, item) for op, item in self._loads.items() if item[0] == deployment.id]
        if len(owned) != 1:
            raise AdmissionError("operation_conflict", "terminated child has no unique admitted load")
        load_op, (_, reservation_generation, loaded_generation) = owned[0]
        if loaded_generation is not None and loaded_generation != key:
            raise AdmissionError("operation_conflict", "terminated generation does not own residency")
        if loaded_generation is None:
            if (operation_id != load_op or generation.process_id is None
                    or generation.boot_id != json.loads(reservation_generation)[0]):
                raise AdmissionError("operation_conflict", "failed spawn belongs to a different operation")
        await asyncio.to_thread(self.store.confirm_deployment_terminated, owner=OWNER,
                                generation=reservation_generation, deployment_id=deployment.id)
        del self._loads[load_op]
