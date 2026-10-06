"""Shared admission for existing native routes, independent of model registration.

Zero request bytes claim no measured model-load budget. Unknown native models
therefore conflict with the managed Prism slot; no implicit eviction is allowed.
Verified CPU requests skip GPU budgeting, while retaining lifecycle coordination.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import httpx
from starlette.responses import StreamingResponse

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot
from kiron_common.gpu_admission.ollama_lifecycle import OLLAMA_DOMAIN, OllamaLifecycleOperation
from kiron_common.gpu_admission.ollama_backend import OllamaBackendSession
from kiron_common.gpu_admission.native_contract import (
    COMPLETION_HEADER, OPERATION_HEADER, OVERLAY_HEADER, OPERATION_PATH,
    valid_operation_id,
)
from kiron_common.model_catalog import ModelEndpoint
from routing_catalog import PROXY_ROUTING_VIEW
import vram_lease


logger = logging.getLogger(__name__)
INFERENCE_PATHS = frozenset({
    "/api/chat", "/api/generate", "/api/embed", "/api/embeddings",
    "/api/embed_late", "/api/embed_colbert", "/api/rerank", "/api/score",
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses",
})
_EPOCH = "native-proxy:" + uuid4().hex  # Proxy operation epoch, never a native process ID.
# Match the native proxy's existing ColBERT/standard response bound. A smaller
# proof buffer would strand legitimate completed large embedding requests.
_MAX_CAPTURE = 128 * 1024 * 1024
_MAX_LINE = 10 * 1024 * 1024  # Existing native/OpenAI stream buffer bound.
OLLAMA_DRAIN_SECONDS = 30.0


def make_store() -> AdmissionStore:
    return AdmissionStore(vram_lease.RUNTIME_MARKER_DIR, security=vram_lease._admission_security())


def measure_memory() -> MemorySnapshot:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=0.5, check=True)
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            raise ValueError("one GPU required")
        free = int(rows[0]) * 1024**2
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        host = int(meminfo["MemAvailable"].split()[0]) * 1024
        if free < 0 or host <= 0:
            raise ValueError("invalid availability")
        return MemorySnapshot(free, host, time.monotonic())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise AdmissionError("resource_unknown", "native GPU availability is unknown") from exc


def verified_cpu(body: bytes, endpoint: str, backend: str = "ollama") -> bool:
    if backend != "ollama" or endpoint not in {"/api/chat", "/api/generate"}:
        return False
    try:
        data = json.loads(body)
        options = data.get("options") if isinstance(data, dict) else None
        return (isinstance(options, dict) and type(options.get("num_gpu")) is int
                and options["num_gpu"] == 0 and vram_lease.num_gpu_zero_effective())
    except (ValueError, UnicodeError):
        return False


class NativeRequestOperation:
    def __init__(self, *, backend: str, endpoint: str, model: str, body: bytes,
                 gpu_operation=None, store=None, measure=None, method="POST",
                 routing_view=PROXY_ROUTING_VIEW):
        if backend not in {"ollama", "kiron_embeddings", "kiron_deberta"}:
            raise ValueError("unknown native backend")
        self.backend, self.endpoint = backend, endpoint.split("?", 1)[0].rstrip("/")
        self.memory_budget = None
        if backend == "kiron_deberta" and self.endpoint in {"/api/rerank", "/api/score"}:
            route = routing_view.resolve(model, ModelEndpoint(self.endpoint))
            self.memory_budget = route.gpu_memory if route is not None else None
        self.gpu_operation = gpu_operation
        self.backend_mutation = backend == "ollama" and method not in {"GET", "HEAD", "OPTIONS"}
        self.enabled = method == "POST" and self.endpoint in INFERENCE_PATHS
        self.cpu_only = verified_cpu(body, self.endpoint, backend)
        self.unload = (self.enabled and backend == "ollama"
                       and vram_lease.is_native_unload(body, self.endpoint))
        self.lifecycle = None
        self.backend_session = None
        self.operation_id = uuid4().hex
        # These identify the native operation, not a catalog artifact or digest.
        self.deployment_id = f"native:{backend}:{model}"[:512]
        self.generation = f"{_EPOCH}:{backend}"
        self.store, self.measure = store, measure
        self.ticket = self.response = self._heartbeat = self._cleanup = None
        self.started = self.confirmed_end = False
        self._sent = self._bad = False
        self._terminal = self._eof = False
        self._capture = bytearray()
        self._reader = None
        self._queue = asyncio.Queue(maxsize=4)
        self._detached = asyncio.Event()
        self._reader_error = None
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            value = {}
        self._ndjson = self.endpoint in {"/api/chat", "/api/generate"} and (
            not isinstance(value, dict) or value.get("stream", True) is not False)
        self._sse = self.endpoint in {"/v1/chat/completions", "/v1/completions", "/v1/responses"} and (
            isinstance(value, dict) and value.get("stream") is True)

    async def send(self, client, request):
        if self._sent:
            raise RuntimeError("native operation can be sent only once")
        self._sent = True
        if self.backend_mutation and not self.unload:
            self.store = self.store or make_store()
            self.backend_session = OllamaBackendSession(self.store)
            await self.backend_session.__aenter__()
        if self.enabled:
            self.store = self.store or make_store()
            if self.unload:
                self.lifecycle = OllamaLifecycleOperation(store=self.store, owner="kiron-proxy-native",
                    generation=self.generation, operation_id=self.operation_id, deployment_id=self.deployment_id,
                    deadline_monotonic=time.monotonic() + 5)
                await self.lifecycle.__aenter__()
                self.ticket = self.lifecycle.ticket
            else:
                overlays = ({"gpu-service-loading.json": self.gpu_operation.token}
                            if self.gpu_operation is not None and self.gpu_operation.token else None)
                budget = self.memory_budget
                loading = self.gpu_operation is None or self.gpu_operation.token is not None
                self.ticket = self.store.reserve(operation_id=self.operation_id, owner="kiron-proxy-native",
                    generation=self.generation, deployment_id=self.deployment_id, kind="request",
                    gpu_bytes=budget.additional_bytes(loading=loading) if budget else 0,
                    headroom_bytes=budget.headroom_bytes if budget else 0,
                    host_bytes=0, measure=self.measure or measure_memory, allow_existing=False,
                    lifecycle_domain=OLLAMA_DOMAIN if self.backend == "ollama" else None,
                    gpu_guard=not self.cpu_only, ttl_seconds=300, owned_overlays=overlays,
                    backend_instance=self.backend_session.instance if self.backend_session else None,
                    overlay_token=self.gpu_operation.token if self.gpu_operation else None,
                    conflicting_resident_slots=() if self.cpu_only else ("prism",))
            self._heartbeat = asyncio.create_task(self._beat())
        if self.enabled and self.backend == "kiron_deberta":
            # Replace client-supplied identities with this admission's nonce.
            request.headers[OPERATION_HEADER] = self.operation_id
            request.headers.pop(OVERLAY_HEADER, None)
            if self.gpu_operation is not None and self.gpu_operation.token:
                request.headers[OVERLAY_HEADER] = self.gpu_operation.token
        if self.lifecycle is not None:
            self.lifecycle.mark_started()
        self.started = True
        try:
            self.response = await client.send(request, stream=True)
            return self.response
        except (httpx.ConnectError, httpx.PoolTimeout):
            self.started = False  # Connection/pool failure proves no backend request started.
            if self.lifecycle is not None:
                self.lifecycle.started = False
            raise

    async def _beat(self):
        try:
            while True:
                await asyncio.sleep(60)
                self.store.heartbeat(self.operation_id, owner="kiron-proxy-native", generation=self.generation)
        except AdmissionError:
            logger.error("native request admission heartbeat became unknown")

    def _line(self, line):
        if not line.strip():
            return
        if self._terminal and (not self._sse or line.startswith(b"data:")):
            self._bad = True  # A second data frame after the terminal violates the stream.
            self.confirmed_end = False
            return
        if self._sse:
            # Native OpenAI pass-through: only a complete backend SSE terminal
            # event proves completion; client-facing synthesized DONE never does.
            if line.startswith(b"data:"):
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    self._terminal = not self._bad
                    return
                try:
                    value = json.loads(payload)
                    if isinstance(value, dict) and value.get("type") in {"response.completed", "response.incomplete"}:
                        self._terminal = not self._bad
                except (ValueError, UnicodeError):
                    self._bad = True
            return
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("error"):
                self._bad = True
            elif value.get("done") is True:
                self._terminal = not self._bad
        except (ValueError, UnicodeError):
            self._bad = True

    async def _read_chunks(self, response):
        if response is not self.response:
            raise RuntimeError("response does not belong to native operation")
        if not 200 <= response.status_code < 300:
            self._ndjson = self._sse = False
        async for chunk in response.aiter_bytes():
            if self.enabled and not self._bad:
                self._capture.extend(chunk)
                if self._ndjson or self._sse:
                    while b"\n" in self._capture:
                        line, _, tail = self._capture.partition(b"\n")
                        self._capture = bytearray(tail)
                        if len(line) > _MAX_LINE:
                            self._bad = True
                            break
                        self._line(line.rstrip(b"\r"))
                    if len(self._capture) > _MAX_LINE:
                        self._bad = True
                elif len(self._capture) > _MAX_CAPTURE:
                    self._bad = True
                if self._bad:
                    self.confirmed_end = False
                    self._capture.clear()
            yield chunk
        self._eof = True
        if self.enabled and not self._bad:
            if self._ndjson:
                if self._capture.strip():
                    self._line(bytes(self._capture))
            elif not self._sse:
                self._complete_json(response.status_code)
            if self._ndjson or self._sse:
                self.confirmed_end = self._terminal and not self._bad
        if self._bad:
            self.confirmed_end = False
        self._capture.clear()

    async def _pump(self):
        """One owner keeps reading the backend when the downstream disappears."""
        try:
            async for chunk in self._read_chunks(self.response):
                if self._detached.is_set():
                    continue
                put = asyncio.create_task(self._queue.put(chunk))
                detached = asyncio.create_task(self._detached.wait())
                try:
                    await asyncio.wait({put, detached}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in (put, detached):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(put, detached, return_exceptions=True)
        except BaseException as exc:
            self._reader_error = exc
            if isinstance(exc, asyncio.CancelledError):
                raise

    def _start_reader(self):
        if self._reader is None:
            self._reader = asyncio.create_task(self._pump())

    async def chunks(self, response):
        if response is not self.response:
            raise RuntimeError("response does not belong to native operation")
        if self.backend != "ollama" or not self.enabled or self.unload:
            async for chunk in self._read_chunks(response):
                yield chunk
            return
        self._start_reader()
        try:
            while not self._reader.done() or not self._queue.empty():
                item = asyncio.create_task(self._queue.get())
                try:
                    await asyncio.wait({item, self._reader}, return_when=asyncio.FIRST_COMPLETED)
                    if item.done():
                        yield item.result()
                    elif self._reader.done():
                        break
                finally:
                    if not item.done():
                        item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
            if self._reader_error is not None:
                raise self._reader_error
        finally:
            self._detached.set()

    def _complete_json(self, status):
        try:
            value = json.loads(self._capture)
        except (ValueError, UnicodeError):
            return
        if not isinstance(value, dict):
            return
        if self.backend == "kiron_deberta" and self.endpoint in {"/api/rerank", "/api/score"}:
            # Issued only after the service's worker and GPU work have ended.
            # _finish still requires EOF and a successfully closed transport.
            self.confirmed_end = self.response.headers.get(COMPLETION_HEADER) == self.operation_id
        elif 400 <= status < 500 and value.get("error"):
            self.confirmed_end = True  # A complete structured native rejection.
        elif 200 <= status < 300 and not value.get("error"):
            if self.endpoint in {"/api/chat", "/api/generate"}:
                self.confirmed_end = value.get("done") is True
            elif self.endpoint in {"/v1/chat/completions", "/v1/completions"}:
                choices = value.get("choices")
                self.confirmed_end = isinstance(choices, list) and bool(choices) and all(
                    isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)
            elif self.endpoint == "/v1/responses":
                self.confirmed_end = value.get("status") in {"completed", "incomplete", "failed"}
            else:
                # These service routes return only after synchronous inference.
                self.confirmed_end = True

    async def _finish(self):
        if self.response is not None and self.backend == "ollama" and self.enabled and not self.unload:
            self._detached.set()
            self._start_reader()
            done, _ = await asyncio.wait({self._reader}, timeout=OLLAMA_DRAIN_SECONDS)
            if not done:
                self._reader.cancel()
                await asyncio.wait({self._reader}, timeout=1)
            if self._reader.done() and not self._reader.cancelled():
                self._reader.exception()
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat
        close_ok = True
        if self.response is not None:
            close = asyncio.create_task(self.response.aclose())
            done, _ = await asyncio.wait({close}, timeout=1)
            if not done:
                close_ok = False
                close.cancel()
            elif close.cancelled() or close.exception() is not None:
                close_ok = False
            close.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        # A terminal frame alone never proves EOF, including consumer-close
        # immediately after that frame. No other owner's work is released.
        pending_tail = (self._ndjson or self._sse) and bool(self._capture.strip())
        confirmed = not self.started or (self.confirmed_end and self._eof and not pending_tail and close_ok
                                        and (self._reader is None or self._reader.done()))
        try:
            if self.lifecycle is not None:
                if self.started and confirmed:
                    self.lifecycle.confirm_end()
                await self.lifecycle.close()
            elif self.ticket is not None:
                overlays = None
                if confirmed and self.gpu_operation is not None and self.gpu_operation.token:
                    overlays = {"gpu-service-loading.json": self.gpu_operation.token}
                self.store.release(self.operation_id, owner="kiron-proxy-native",
                                   generation=self.generation, confirmed_terminated=confirmed,
                                   owned_overlays=overlays)
        finally:
            try:
                if self.gpu_operation is not None:
                    self.gpu_operation.clear_marker = confirmed
                    await vram_lease.finish_gpu_service_operation(self.gpu_operation)
            finally:
                if self.backend_session is not None:
                    self.backend_session.close()

    async def close(self):
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._finish())
        await asyncio.shield(self._cleanup)


async def reconcile_native_operations(client, store=None):
    """Release only individually attested, terminated native DeBERTa requests.

    Neither healthy/no_model nor elapsed TTL is evidence. The ledger must still
    know this exact admission nonce; a restarted backend returns unknown.
    """
    store = store or make_store()
    for ticket in store.snapshot():
        inference = (ticket.owner == "kiron-proxy-native"
                     and ticket.deployment_id.startswith("native:kiron_deberta:"))
        model_load = (ticket.owner == "kiron-proxy-lifecycle"
                      and ticket.deployment_id == "native-lifecycle:DeBERTa-Service")
        if (not (inference or model_load) or ticket.kind != "request" or ticket.phase != "unknown"
                or not valid_operation_id(ticket.operation_id)):
            continue
        try:
            response = await client.get(OPERATION_PATH + ticket.operation_id, timeout=1.0)
            if response.status_code != 200 or len(response.content) > 4096:
                continue
            proof = response.json()
            if (not isinstance(proof, dict) or set(proof) != {
                    "schema_version", "backend_instance", "operation_id", "overlay_token", "state"}
                    or type(proof.get("schema_version")) is not int
                    or proof["schema_version"] != 1 or proof.get("state") != "terminated"
                    or proof.get("operation_id") != ticket.operation_id
                    or not valid_operation_id(proof.get("backend_instance"))):
                continue
            token = proof.get("overlay_token")
            if token is not None and not valid_operation_id(token):
                continue
            store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                          confirmed_terminated=True,
                          owned_overlays={"gpu-service-loading.json": token} if token else None)
            logger.info("Native GPU operation reconciled: %s", ticket.operation_id)
        except (httpx.HTTPError, ValueError, AdmissionError):
            logger.warning("Native GPU operation reconciliation remains unconfirmed: %s", ticket.operation_id)


async def native_operation_reconciler():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11437", follow_redirects=False) as client:
        while True:
            try:
                await reconcile_native_operations(client)
            except Exception:
                logger.exception("Native GPU operation reconciliation failed")
            await asyncio.sleep(5)


class NativeStreamingResponse(StreamingResponse):
    """Own cleanup even when ASGI headers fail or the body is never entered."""
    def __init__(self, content, *, operation, finalize=None, **kwargs):
        super().__init__(content, **kwargs)
        self.operation = operation
        self.finalize = finalize
        self._cleanup = None

    async def _finish(self):
        try:
            # A failed client send can leave the generator suspended at yield.
            await self.body_iterator.aclose()
        finally:
            try:
                if self.finalize is not None:
                    await self.finalize()
            finally:
                await self.operation.close()

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self._cleanup is None:
                self._cleanup = asyncio.create_task(self._finish())
                self._cleanup.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            await asyncio.shield(self._cleanup)
