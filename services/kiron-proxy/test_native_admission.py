"""Native-route GPU admission over temporary state and fake HTTP responses."""
import asyncio
import json
import grp
import os
import pwd
import time

import httpx
import pytest
from starlette.requests import ClientDisconnect

from kiron_common.gpu_admission import AdmissionError, MemorySnapshot
from native_admission import INFERENCE_PATHS, NativeRequestOperation, NativeStreamingResponse
import native_admission


class Response:
    def __init__(self, chunks=(), *, status=200, failure=None):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.blocks, self.failure = chunks, failure
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self.blocks:
            yield chunk
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


class Client:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.sent = response, error, []

    async def send(self, request, *, stream):
        self.sent.append(request)
        if self.error:
            raise self.error
        return self.response


def operation(**kwargs):
    defaults = dict(backend="ollama", endpoint="/api/chat", model="unregistered-local-tag:latest",
                    body=b'{"model":"unregistered-local-tag:latest","stream":true}')
    return NativeRequestOperation(**{**defaults, **kwargs})


def reserve(store, **kwargs):
    defaults = dict(operation_id="other", owner="other", generation="boot:child", deployment_id="model",
                    kind="load", gpu_bytes=0, host_bytes=0,
                    measure=lambda: MemorySnapshot(100, 100, time.monotonic()))
    return store.reserve(**{**defaults, **kwargs})


@pytest.mark.parametrize("endpoint", sorted(INFERENCE_PATHS))
def test_all_native_inference_routes_reserve_before_backend_io(endpoint, native_admission_runtime):
    async def run():
        op = operation(endpoint=endpoint)
        client = Client(Response())
        await op.send(client, httpx.Request("POST", "http://backend"))
        ticket, = native_admission_runtime.snapshot()
        assert ticket.kind == "request" and ticket.deployment_id.startswith("native:ollama:")
        assert ticket.gpu_bytes == 0 and ticket.host_bytes == 0
        assert "unregistered-local-tag" in ticket.deployment_id
        await op.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["reserved", "resident", "draining"])
def test_prism_slot_blocks_native_before_send(phase, native_admission_runtime):
    async def run():
        store = native_admission_runtime
        reserve(store, resident_slot="prism")
        if phase != "reserved":
            store.transition("other", owner="other", expected_generation="boot:child", phase="resident")
        if phase == "draining":
            store.begin_unload("unload", owner="other", generation="boot:child", deployment_id="model")
        op, client = operation(), Client(Response())
        with pytest.raises(AdmissionError, match="managed runtime"):
            await op.send(client, httpx.Request("POST", "http://backend"))
        await op.close()
        assert not client.sent
        assert all(ticket.owner == "other" for ticket in store.snapshot())
    asyncio.run(run())


def test_native_request_prevents_prism_load_and_exclusive_training(native_admission_runtime):
    async def run():
        op = operation()
        await op.send(Client(Response()), httpx.Request("POST", "http://backend"))
        with pytest.raises(AdmissionError):
            reserve(native_admission_runtime, resident_slot="prism")
        with pytest.raises(AdmissionError):
            reserve(native_admission_runtime, kind="training", exclusive=True)
        await op.close()
    asyncio.run(run())


def test_verified_cpu_skips_gpu_budget_but_retains_lifecycle_ticket(native_admission_runtime, monkeypatch):
    async def run():
        reserve(native_admission_runtime, resident_slot="prism")
        monkeypatch.setattr(native_admission.vram_lease, "num_gpu_zero_effective", lambda: True)
        for endpoint in ("/api/chat", "/api/generate"):
            op = operation(endpoint=endpoint, body=b'{"options":{"num_gpu":0}}')
            response = await op.send(Client(Response([b'{"done":true}\n'])), httpx.Request("POST", "http://backend"))
            ticket = next(t for t in native_admission_runtime.snapshot() if t.operation_id == op.operation_id)
            assert ticket.lifecycle_domain == "ollama" and ticket.gpu_guard is False
            assert [part async for part in op.chunks(response)]
            await op.close()
        for body, endpoint, backend in ((b'{"options":{"num_gpu":false}}', "/api/chat", "ollama"),
                                       (b'{"options":{"num_gpu":0}}', "/api/embed", "ollama"),
                                       (b'{"options":{"num_gpu":0}}', "/api/rerank", "kiron_deberta")):
            op = operation(body=body, endpoint=endpoint, backend=backend)
            with pytest.raises(AdmissionError):
                await op.send(Client(Response()), httpx.Request("POST", "http://backend"))
            await op.close()
        monkeypatch.setattr(native_admission.vram_lease, "num_gpu_zero_effective", lambda: False)
        op = operation(body=b'{"options":{"num_gpu":0}}')
        with pytest.raises(AdmissionError):
            await op.send(Client(Response()), httpx.Request("POST", "http://backend"))
        await op.close()
        assert len(native_admission_runtime.snapshot()) == 1
    asyncio.run(run())


@pytest.mark.parametrize("endpoint,method", [("/api/tags", "GET"), ("/api/ps", "GET"), ("/api/chat", "OPTIONS")])
def test_metadata_and_non_post_methods_need_no_gpu_ticket(endpoint, method, native_admission_runtime):
    async def run():
        reserve(native_admission_runtime, resident_slot="prism")
        op = operation(endpoint=endpoint, method=method)
        await op.send(Client(Response()), httpx.Request("POST", "http://backend"))
        await op.close()
        assert len(native_admission_runtime.snapshot()) == 1
    asyncio.run(run())


def test_fragmented_native_done_requires_consumed_eof(native_admission_runtime):
    async def run():
        op = operation()
        response = await op.send(Client(Response([b'{"done":', b'true}\n'])), httpx.Request("POST", "http://backend"))
        chunks = op.chunks(response)
        assert await anext(chunks) == b'{"done":'
        assert not op.confirmed_end
        assert await anext(chunks) == b'true}\n'
        assert not op.confirmed_end
        with pytest.raises(StopAsyncIteration):
            await anext(chunks)
        assert op.confirmed_end
        await op.close()
        await chunks.aclose()
        assert response.closed and native_admission_runtime.snapshot() == ()
    asyncio.run(run())


def test_early_consumer_stop_does_not_ignore_already_buffered_invalid_tail(native_admission_runtime):
    async def run():
        op = operation()
        response = await op.send(Client(Response([b'{"done":true}\nbroken'])), httpx.Request("POST", "http://backend"))
        chunks = op.chunks(response)
        assert await anext(chunks) == b'{"done":true}\nbroken'
        await op.close()  # Mirrors a translator stopping upon its native done.
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
        await chunks.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("stream,over_limit", [(False, False), (False, True), (True, False), (True, True)])
def test_proof_capture_boundary_keeps_wire_and_releases_only_within_bound(
        stream, over_limit, native_admission_runtime, monkeypatch):
    async def run():
        body = b'{"done":true}' if stream else b'{"embeddings":[[1,2,3]]}'
        monkeypatch.setattr(native_admission, "_MAX_LINE" if stream else "_MAX_CAPTURE", len(body) - int(over_limit))
        if stream:
            body += b'\n'
        op = operation(endpoint="/api/generate" if stream else "/api/embed")
        response = await op.send(Client(Response([body[:5], body[5:]])), httpx.Request("POST", "http://backend"))
        assert b''.join([part async for part in op.chunks(response)]) == body
        await op.close()
        if over_limit:
            assert native_admission_runtime.snapshot()[0].phase == "unknown"
        else:
            assert native_admission_runtime.snapshot() == ()
    asyncio.run(run())


@pytest.mark.parametrize("body", [b'{"done":false}\n', b'{"done":true,"error":"failure"}\n',
                                 b'broken\n{"done":true}\n', b'{"done":true}\nbroken',
                                 b'{"done":true}\n{"done":false}\n',
                                 b'{"done":true}\n{"done":true}\n'])
def test_incomplete_or_corrupt_stream_keeps_unknown_ticket(body, native_admission_runtime):
    async def run():
        op = operation()
        response = await op.send(Client(Response([body])), httpx.Request("POST", "http://backend"))
        assert b"".join([part async for part in op.chunks(response)]) == body
        await op.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())


@pytest.mark.parametrize("endpoint,backend,body", [
    ("/api/embed", "ollama", b'{"embeddings":[[1,2]]}'),
    ("/api/embed_colbert", "kiron_embeddings", b'{"embeddings":[[[1,2]]]}'),
    ("/api/score", "kiron_deberta", b'{"scores":[0.2]}'),
    ("/v1/embeddings", "ollama", b'{"object":"list","data":[]}'),
])
def test_synchronous_embedding_scoring_completion_releases(endpoint, backend, body, native_admission_runtime):
    async def run():
        op = operation(endpoint=endpoint, backend=backend)
        reply = Response([body])
        if backend == "kiron_deberta":
            reply.headers[native_admission.COMPLETION_HEADER] = op.operation_id
        response = await op.send(Client(reply), httpx.Request("POST", "http://backend"))
        assert [part async for part in op.chunks(response)] == [body]
        await op.close()
        assert native_admission_runtime.snapshot() == ()
    asyncio.run(run())


def test_header_failure_drains_backend_even_before_generator_entry(native_admission_runtime):
    async def run():
        op = operation()
        response = await op.send(Client(Response([b'{"done":true}\n'])), httpx.Request("POST", "http://backend"))
        entered = False
        async def body():
            nonlocal entered
            entered = True
            async for part in op.chunks(response):
                yield part
        async def send(_):
            raise OSError("ASGI header failure")
        async def receive():
            await asyncio.sleep(10)
        stream = NativeStreamingResponse(body(), operation=op)
        with pytest.raises((OSError, ClientDisconnect, ExceptionGroup)):
            await stream({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        assert not entered and response.closed
        assert op.confirmed_end and op._eof
        assert native_admission_runtime.snapshot() == ()
        await op.close()
    asyncio.run(run())


def test_cancellation_during_send_cannot_release_uncertain_backend(native_admission_runtime):
    async def run():
        started = asyncio.Event()
        class SlowClient:
            async def send(self, *args, **kwargs):
                started.set()
                await asyncio.sleep(10)
        op = operation()
        task = asyncio.create_task(op.send(SlowClient(), httpx.Request("POST", "http://backend")))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await op.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())


def test_connect_error_and_pre_send_failure_leave_no_reservation(native_admission_runtime):
    async def run():
        op = operation()
        with pytest.raises(httpx.ConnectError):
            await op.send(Client(error=httpx.ConnectError("not connected")), httpx.Request("POST", "http://backend"))
        await op.close()
        assert native_admission_runtime.snapshot() == ()
        never_sent = operation()
        await never_sent.close()
        assert native_admission_runtime.snapshot() == ()
    asyncio.run(run())


def test_rerank_capacity_response_reports_measurement_and_persists_decision(
        marker_runtime, native_admission_runtime, monkeypatch):
    import proxy

    async def run():
        updates = []
        class Records:
            async def add_request(self, record):
                pass
            async def update_request(self, *args, **kwargs):
                updates.append(kwargs)

        class Backend:
            def __init__(self, **kwargs):
                pass
            async def get(self, path):
                return httpx.Response(503, json={"status": "no_model", "current_model": None})
            def build_request(self, **kwargs):
                return httpx.Request(**kwargs)
            async def send(self, request, **kwargs):
                raise AssertionError("rejected inference must not reach the backend")
            async def aclose(self):
                pass

        monkeypatch.setattr(native_admission, "measure_memory",
            lambda: MemorySnapshot(616 * 1024**2, 32 * 1024**3, time.monotonic()))
        real_client = httpx.AsyncClient
        monkeypatch.setattr(proxy.httpx, "AsyncClient", Backend)
        app = proxy.create_proxy_app(Records())
        monkeypatch.setattr(proxy.httpx, "AsyncClient", real_client)
        async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post('/api/rerank', json={
                'model': 'bge-reranker-v2-m3', 'query': 'test', 'documents': ['text'] * 80})
        assert response.status_code == 503
        body = response.json()
        assert (body['code'], body['stage'], body['inference_started']) == (
            'resource_exhausted', 'admission', False)
        assert body['vram_lease'] == 'not_acquired'
        assert '616 MiB frei' in response.text[:300]
        assert '3584 MiB erforderlich' in response.text[:300]
        assert body['memory']['gpu_required_bytes'] == 3584 * 1024**2
        assert body['memory']['gpu_pending_bytes'] == 0
        error = next(update for update in updates if update.get('status_code') == 503)
        details = json.loads(error['error_message'].split('resource_exhausted ', 1)[1])
        assert details['reservations'] == []
        assert details['gpu_free_bytes'] == body['memory']['gpu_free_bytes']
        assert json.loads(error['response_body']) == body
        assert native_admission_runtime.snapshot() == ()
        assert not marker_runtime.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        assert not marker_runtime.gpu_service_ops_lock.locked()
    asyncio.run(run())


@pytest.fixture
def marker_runtime(native_admission_runtime, monkeypatch):
    vl = native_admission.vram_lease
    root = native_admission_runtime.root
    for name, value in {
        "RUNTIME_MARKER_DIR": root,
        "STARTUP_MARKER_PATH": root / "docling-vram-startup.json",
        "SHUTDOWN_MARKER_PATH": root / "docling-vram-shutdown.json",
        "GPU_SERVICE_LOADING_MARKER_PATH": root / "gpu-service-loading.json",
        "RUNTIME_MARKER_GROUP": grp.getgrgid(os.getgid()).gr_name,
        "RUNTIME_MARKER_FILE_OWNER_NAMES": frozenset({pwd.getpwuid(os.getuid()).pw_name}),
        "_runtime_marker_dir_owner_uid": lambda: os.getuid(),
        "gpu_service_ops_lock": asyncio.Lock(),
    }.items():
        monkeypatch.setattr(vl, name, value)
    async def inactive():
        return False
    monkeypatch.setattr(vl, "snapshot", inactive)
    return vl


@pytest.mark.parametrize("confirmed", [True, False])
def test_lifecycle_ticket_uses_owned_marker_and_requires_endpoint_proof(
        confirmed, marker_runtime, native_admission_runtime):
    async def run():
        vl, store = marker_runtime, native_admission_runtime
        async with vl.gpu_service_operation(service_name="Embedding-Service") as op:
            assert op.allowed
            ticket, = store.snapshot()
            assert ticket.kind == "request" and ticket.owner == "kiron-proxy-lifecycle"
            assert ticket.gpu_bytes == 0 and ticket.host_bytes == 0
            assert vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
            with pytest.raises(AdmissionError):
                reserve(store, resident_slot="prism")
            op.clear_marker = confirmed
        assert not vl.gpu_service_ops_lock.locked()
        if confirmed:
            assert store.snapshot() == () and not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        else:
            assert store.snapshot()[0].phase == "unknown" and vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())


def test_lifecycle_prism_conflict_clears_only_own_marker_before_backend(
        marker_runtime, native_admission_runtime):
    async def run():
        reserve(native_admission_runtime, resident_slot="prism")
        async with marker_runtime.gpu_service_operation(force=True) as op:
            assert not op.allowed and op.decision.status_code == 409
        assert not marker_runtime.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        assert not marker_runtime.gpu_service_ops_lock.locked()
        assert len(native_admission_runtime.snapshot()) == 1
    asyncio.run(run())


def test_resource_release_ignores_admission_and_foreign_overlay_without_clearing_it(
        marker_runtime, native_admission_runtime):
    async def run():
        reserve(native_admission_runtime, resident_slot="prism")
        marker_runtime.write_overlay_marker("startup", ttl_s=60)
        await marker_runtime.gpu_service_ops_lock.acquire()
        try:
            op = await asyncio.wait_for(marker_runtime.begin_gpu_service_operation(releases_resources=True), 0.2)
            assert op.allowed and op.token is None and op.admission_id is None and not op.lock_acquired
            op.clear_marker = True
            await marker_runtime.finish_gpu_service_operation(op)
            assert marker_runtime.gpu_service_ops_lock.locked()
        finally:
            marker_runtime.gpu_service_ops_lock.release()
        assert marker_runtime.STARTUP_MARKER_PATH.exists()
        assert len(native_admission_runtime.snapshot()) == 1
    asyncio.run(run())


def test_lifecycle_cancel_during_gate_releases_local_lock(marker_runtime, monkeypatch):
    async def run():
        entered = asyncio.Event()
        async def gate(**kwargs):
            entered.set()
            await asyncio.sleep(10)
        monkeypatch.setattr(marker_runtime, "gpu_gate_decision", gate)
        task = asyncio.create_task(marker_runtime.begin_gpu_service_operation())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not marker_runtime.gpu_service_ops_lock.locked()
    asyncio.run(run())


def test_lazy_marker_is_retained_on_incomplete_native_response(marker_runtime, native_admission_runtime):
    async def run():
        token = marker_runtime.write_overlay_marker("gpu_service_loading", ttl_s=60)
        gpu_op = marker_runtime.GPUServiceOperation(True, marker_runtime.GPUGateDecision(True, "test"),
                                                  token=token)
        op = operation(gpu_operation=gpu_op)
        response = await op.send(Client(Response([b'{"done":false}\n'])), httpx.Request("POST", "http://backend"))
        assert [part async for part in op.chunks(response)]
        gpu_op.clear_marker = True  # An optimistic old caller cannot release it.
        await op.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
        assert marker_runtime.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["inference", "unload", "get", "options"])
@pytest.mark.parametrize("stream,prism", [(False, False), (True, False), (False, True), (True, True)])
def test_native_proxy_route_has_admission_before_unregistered_backend_send(
        mode, stream, prism, marker_runtime, native_admission_runtime, monkeypatch):
    import proxy
    async def run():
        store = native_admission_runtime
        if prism:
            reserve(store, resident_slot="prism")
        if mode != "inference":
            marker_runtime.write_overlay_marker("startup", ttl_s=60)
        class Records:
            async def add_request(self, record):
                pass
            async def update_request(self, *args, **kwargs):
                pass
        clients = []
        class Backend:
            def __init__(self, **kwargs):
                clients.append(self)
                self.sent = []
            async def get(self, path):
                return httpx.Response(200, json={"models":[
                    {"name":"unregistered-local-tag:latest", "size_vram":123}]})
            def build_request(self, **kwargs):
                return kwargs
            async def send(self, request, *, stream):
                if mode == "inference":
                    ticket, = store.snapshot()
                    assert ticket.kind == "request" and ticket.owner == "kiron-proxy-native"
                elif mode == "unload":
                    ticket = next(t for t in store.snapshot() if t.kind == "unload")
                    assert ticket.lifecycle_domain == "ollama"
                    assert len(store.snapshot()) == int(prism) + 1
                else:
                    assert len(store.snapshot()) == int(prism)
                self.sent.append(request)
                return Response([b'{"model":"unregistered-local-tag:latest","done":true,"response":"ok"}\n'])
            async def aclose(self):
                pass
        real_client = httpx.AsyncClient
        monkeypatch.setattr(proxy.httpx, "AsyncClient", Backend)
        app = proxy.create_proxy_app(Records())
        monkeypatch.setattr(proxy.httpx, "AsyncClient", real_client)
        async with real_client(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"model":"unregistered-local-tag:latest", "prompt":"hi", "stream":stream}
            if mode == "unload":
                body = {"model":"unregistered-local-tag:latest", "keep_alive":0, "stream":False}
            result = await client.request(mode.upper() if mode in {"get", "options"} else "POST",
                                          "/api/generate", json=body)
        blocked = prism and mode == "inference"
        assert result.status_code == (409 if blocked else 200)
        assert bool(clients[0].sent) is not blocked
        assert len(store.snapshot()) == (1 if prism else 0)
    asyncio.run(run())


@pytest.mark.parametrize("change", [{"prompt":"inference"}, {"keep_alive":False}, {"stream":True},
                                   {"options":{}}, {"images":[]}, {"think":True}, {"suffix":"x"}])
def test_keep_alive_zero_inference_and_unknown_fields_still_require_admission(
        change, native_admission_runtime):
    async def run():
        reserve(native_admission_runtime, resident_slot="prism")
        body = {"model":"unregistered-local-tag:latest", "keep_alive":0, "stream":False, **change}
        op = operation(endpoint="/api/generate", body=json.dumps(body).encode())
        with pytest.raises(AdmissionError):
            await op.send(Client(Response()), httpx.Request("POST", "http://backend"))
        await op.close()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["ndjson", "sse"])
@pytest.mark.parametrize("end", ["consumer-close", "late-error", "eof"])
def test_native_cleanup_requires_terminal_and_clean_eof(kind, end, native_admission_runtime):
    async def run():
        data = b'{"done":true}\n' if kind == "ndjson" else b'data: [DONE]\n\n'
        op = operation(endpoint="/api/chat" if kind == "ndjson" else "/v1/chat/completions")
        failure = httpx.ReadError("late read failure") if end == "late-error" else None
        response = await op.send(Client(Response([data], failure=failure)), httpx.Request("POST", "http://backend"))
        chunks = op.chunks(response)
        assert await anext(chunks) == data
        assert native_admission_runtime.snapshot()  # Consumer still owns its reservation.
        if end == "eof":
            with pytest.raises(StopAsyncIteration):
                await anext(chunks)
        elif end == "late-error":
            with pytest.raises(httpx.ReadError):
                await anext(chunks)
        await op.close()
        await chunks.aclose()
        if end != "late-error":
            assert native_admission_runtime.snapshot() == ()
        else:
            assert native_admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())


def test_eof_with_failed_response_close_keeps_unknown(native_admission_runtime):
    async def run():
        class BadClose(Response):
            async def aclose(self):
                raise httpx.ReadError("close failed")
        operation_ = operation()
        response = await operation_.send(Client(BadClose([b'{"done":true}\n'])), httpx.Request("POST", "http://backend"))
        assert [part async for part in operation_.chunks(response)]
        await operation_.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
    asyncio.run(run())
