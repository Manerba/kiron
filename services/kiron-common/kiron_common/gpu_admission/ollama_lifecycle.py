"""Conservative cross-process Ollama mutation fence; no native I/O or identity adoption."""
from __future__ import annotations

import asyncio
import math
import time

from .store import AdmissionError, AdmissionStore
from .ollama_backend import OllamaBackendSession

OLLAMA_DOMAIN = "ollama"


def check_deadline(deadline_monotonic, cancellation=None):
    if cancellation is not None and cancellation.is_set():
        raise asyncio.CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise AdmissionError("operation_conflict", "Ollama lifecycle deadline exceeded")


async def drain_ollama_requests(store, *, deadline_monotonic, cancellation=None):
    """Observe all owners/epochs without treating any of them as our own.

    Unknown work cannot be cleared by /api/ps or another successful unload.
    Only the request owner can supply its own end proof and remove its ticket.
    """
    while True:
        check_deadline(deadline_monotonic, cancellation)
        requests = [t for t in await asyncio.to_thread(store.snapshot)
                    if t.lifecycle_domain == OLLAMA_DOMAIN and t.kind == "request"]
        if not requests:
            return
        if any(t.phase == "unknown" for t in requests):
            raise AdmissionError("operation_conflict", "unconfirmed Ollama requests prevent mutation")
        await asyncio.sleep(min(.02, max(0, deadline_monotonic - time.monotonic())))


class OllamaLifecycleOperation:
    """Own one fence around a direct native unload/reconfiguration.

    Canonical managed residents require RuntimeService's exact identity path.
    Independent Docling GPU tickets/overlays are not Ollama work and need no
    exception here: this fence grants no GPU budget and releases only itself.
    """
    def __init__(self, *, store: AdmissionStore, owner: str, generation: str,
                 operation_id: str, deployment_id: str, deadline_monotonic: float,
                 cancellation=None):
        if (type(deadline_monotonic) not in (int, float) or not math.isfinite(deadline_monotonic)
                or not 0 < deadline_monotonic - time.monotonic() <= 3600):
            raise ValueError("finite lifecycle deadline within one hour required")
        self.store, self.owner, self.generation = store, owner, generation
        self.operation_id, self.deployment_id = operation_id, deployment_id
        self.deadline, self.cancellation = deadline_monotonic, cancellation
        self.ticket = None
        self.started = self.confirmed = self.entered = False
        self._cleanup = None
        self.backend_session = OllamaBackendSession(store)

    async def __aenter__(self):
        if self.entered:
            raise RuntimeError("lifecycle operation cannot be reused")
        self.entered = True
        check_deadline(self.deadline, self.cancellation)
        await self.backend_session.__aenter__()
        pending = asyncio.create_task(asyncio.to_thread(self.store.begin_unload, self.operation_id,
            owner=self.owner, generation=self.generation, deployment_id=self.deployment_id,
            require_resident=False, allow_existing=False, lifecycle_domain=OLLAMA_DOMAIN,
            ttl_seconds=max(1, self.deadline - time.monotonic() + 5),
            backend_instance=self.backend_session.instance))
        try:
            self.ticket = await asyncio.shield(pending)
            await drain_ollama_requests(self.store, deadline_monotonic=self.deadline,
                                        cancellation=self.cancellation)
            return self
        except BaseException:
            async def undo_unstarted():
                try:
                    self.ticket = await pending
                except Exception:
                    self.backend_session.close()
                    return  # A failed/duplicate reservation grants no release authority.
                await self.close()
            cleanup = asyncio.create_task(undo_unstarted())
            cleanup.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            await asyncio.shield(cleanup)
            raise

    def mark_started(self):
        if self.ticket is None or self.started or self._cleanup is not None:
            raise RuntimeError("lifecycle fence is not ready")
        check_deadline(self.deadline, self.cancellation)
        self.started = True

    def confirm_end(self):
        if not self.started or self._cleanup is not None:
            raise RuntimeError("native mutation never started")
        self.confirmed = True

    async def close(self):
        if self._cleanup is None:
            async def finish():
                try:
                    if self.ticket is not None:
                        await asyncio.to_thread(self.store.release, self.operation_id, owner=self.owner,
                            generation=self.generation, confirmed_terminated=not self.started or self.confirmed)
                finally:
                    self.backend_session.close()
            self._cleanup = asyncio.create_task(finish())
            self._cleanup.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        await asyncio.shield(self._cleanup)

    async def __aexit__(self, *exc):
        await self.close()
