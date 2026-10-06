"""Fixed production composition; constructing it performs no model operations."""
from __future__ import annotations

import asyncio

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.gpu_admission import AdmissionStore
from kiron_common.local_inference import build_resolver_snapshot
from kiron_common.local_model_registry import RuntimeModelRegistry

from admission import ControllerAdmission
from controller import Controller


class RegistryResolver:
    def __init__(self, registry, profiles, catalog=MODEL_CATALOG):
        self.registry, self.profiles, self.catalog = registry, dict(profiles), catalog

    async def snapshot(self):
        entries = await asyncio.to_thread(self.registry.list)
        return build_resolver_snapshot(self.catalog, entries, resource_profiles=self.profiles)


def build_controller(policy):
    resolver = RegistryResolver(RuntimeModelRegistry(readonly=True), policy.resource_profiles())
    return Controller(policy, resolver, ControllerAdmission(AdmissionStore()))
