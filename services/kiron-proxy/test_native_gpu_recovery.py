"""Regression of external VRAM pressure, native OOM and subsequent chat."""
import asyncio
import json
import time

import httpx
import pytest

import native_admission as native
import proxy
import vram_lease as vl
from kiron_common.gpu_admission import AdmissionError, MemorySnapshot
from test_native_admission import Client, Response, marker_runtime, operation


def rerank_operation(*, marker=True, measure=None):
    token = vl.write_overlay_marker("gpu_service_loading") if marker else None
    gpu = vl.GPUServiceOperation(True, vl.GPUGateDecision(True, "test"), token=token)
    return operation(backend="kiron_deberta", endpoint="/api/rerank", model="bge-reranker-v2-m3",
                     body=b'{"model":"bge-reranker-v2-m3"}', gpu_operation=gpu, measure=measure)


def request():
    return httpx.Request("POST", "http://backend/api/rerank")


def test_external_memory_pressure_rejects_before_backend_and_leaves_no_gate(marker_runtime, native_admission_runtime):
    async def run():
        op = rerank_operation(measure=lambda: MemorySnapshot(211 * 1024**2, 32 * 1024**3, time.monotonic()))
        client = Client(Response())
        with pytest.raises(AdmissionError) as caught:
            await op.send(client, request())
        assert caught.value.code == "resource_exhausted"
        await op.close()
        assert not client.sent and not native_admission_runtime.snapshot()
        assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())


def test_warm_budget_counts_only_additional_work_not_resident_weights(marker_runtime, native_admission_runtime):
    async def run():
        op = rerank_operation(marker=False)
        budget = op.memory_budget
        op.measure = lambda: MemorySnapshot(budget.request_bytes + budget.headroom_bytes, 32 * 1024**3, time.monotonic())
        reply = Response([b'{"results":[]}'])
        reply.headers[native.COMPLETION_HEADER] = op.operation_id
        response = await op.send(Client(reply), request())
        assert native_admission_runtime.snapshot()[0].gpu_bytes == budget.request_bytes
        assert [part async for part in op.chunks(response)]
        await op.close()
        assert not native_admission_runtime.snapshot()
    asyncio.run(run())


@pytest.mark.parametrize("proof", ["correct", "missing", "wrong"])
def test_oom_requires_matching_completion_then_next_chat_succeeds(proof, marker_runtime, native_admission_runtime):
    async def run():
        op = rerank_operation()
        reply = Response([b'{"error":"CUDA out of memory"}'], status=503)
        if proof != "missing":
            reply.headers[native.COMPLETION_HEADER] = op.operation_id if proof == "correct" else "f" * 32
        req = request()
        req.headers[native.OPERATION_HEADER] = "client-controlled"
        response = await op.send(Client(reply), req)
        assert req.headers[native.OPERATION_HEADER] == op.operation_id
        assert req.headers[native.OVERLAY_HEADER] == op.gpu_operation.token
        assert [part async for part in op.chunks(response)]
        await op.close()
        await op.close()
        if proof != "correct":
            assert native_admission_runtime.snapshot()[0].phase == "unknown"
            assert vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
            return
        assert not native_admission_runtime.snapshot()
        assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        chat = operation()
        response = await chat.send(Client(Response([b'{"done":true}\n'])), request())
        assert [part async for part in chat.chunks(response)]
        await chat.close()
        assert not native_admission_runtime.snapshot()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["missing_eof", "close_failure"])
def test_receipt_without_complete_transport_remains_unknown(failure, marker_runtime, native_admission_runtime):
    async def run():
        op = rerank_operation()
        reply = Response([b'{"error":"OOM"}'], status=503,
                         failure=httpx.ReadError("disconnected") if failure == "missing_eof" else None)
        reply.headers[native.COMPLETION_HEADER] = op.operation_id
        if failure == "close_failure":
            async def broken_close():
                raise httpx.ReadError("close failed")
            reply.aclose = broken_close
        response = await op.send(Client(reply), request())
        try:
            assert [part async for part in op.chunks(response)]
        except httpx.ReadError:
            assert failure == "missing_eof"
        await op.close()
        assert native_admission_runtime.snapshot()[0].phase == "unknown"
        assert vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())


@pytest.mark.parametrize("state", ["terminated", "active", "unknown", "missing", "wrong_id", "wrong_token"])
def test_reconcile_requires_exact_operation_and_preserves_foreign_markers(state, marker_runtime, native_admission_runtime):
    async def run():
        op = rerank_operation()
        await op.send(Client(Response(status=503)), request())
        await op.close()
        # Expiry alone must not release either guard.
        path = vl.GPU_SERVICE_LOADING_MARKER_PATH
        marker = json.loads(path.read_text())
        marker["deadline_monotonic"] = time.monotonic() - 1
        path.write_text(json.dumps(marker))

        def reply(req):
            assert req.url.path.endswith(op.operation_id)
            return httpx.Response(404 if state == "missing" else 200, json={
                "schema_version": 1, "backend_instance": "a" * 32,
                "operation_id": "b" * 32 if state == "wrong_id" else op.operation_id,
                "overlay_token": "c" * 32 if state == "wrong_token" else marker["token"],
                "state": state if state in {"active", "unknown"} else "terminated",
            })
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply), base_url="http://backend") as client:
            await native.reconcile_native_operations(client, native_admission_runtime)
            await native.reconcile_native_operations(client, native_admission_runtime)
        if state == "terminated":
            assert not native_admission_runtime.snapshot() and not path.exists()
        else:
            assert native_admission_runtime.snapshot()[0].phase == "unknown" and path.exists()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["preflight", "backend_oom"])
def test_real_proxy_error_path_does_not_block_following_chat(failure, marker_runtime, native_admission_runtime, monkeypatch):
    class Records:
        async def add_request(self, record):
            pass
        async def update_request(self, request_id, **fields):
            pass

    original_client = httpx.AsyncClient
    sent = []
    def backend(req):
        if req.url.path == "/health":
            return httpx.Response(503, json={"status": "no_model", "current_model": None})
        if req.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size_vram": 100}]})
        sent.append(req.url.path)
        if req.url.path == "/api/rerank":
            return httpx.Response(503, json={"error": "CUDA out of memory"},
                headers={native.COMPLETION_HEADER: req.headers[native.OPERATION_HEADER]})
        if req.url.path == "/api/chat":
            return httpx.Response(200, json={"done": True, "message": {"role": "assistant", "content": "OK"}})
        raise AssertionError(req.url)

    monkeypatch.setattr(proxy.httpx, "AsyncClient", lambda *args, **kwargs:
        original_client(*args, **kwargs, transport=httpx.MockTransport(backend)))

    async def run():
        app = proxy.create_proxy_app(Records())
        monkeypatch.setattr(native, "measure_memory", lambda:
            MemorySnapshot((211 * 1024**2) if failure == "preflight" else 12 * 1024**3,
                           32 * 1024**3, time.monotonic()))
        async with original_client(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
            error = await client.post("/api/rerank", json={"model": "bge-reranker-v2-m3", "query": "q", "documents": ["d"]})
            assert error.status_code == 503, error.text
            assert sent == ([] if failure == "preflight" else ["/api/rerank"])
            assert not native_admission_runtime.snapshot()
            assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
            monkeypatch.setattr(native, "measure_memory", lambda: MemorySnapshot(5 * 1024**3, 32 * 1024**3, time.monotonic()))
            chat = await client.post("/api/chat", json={"model": "qwen3:8b", "stream": False,
                                                       "messages": [{"role": "user", "content": "hello"}]})
            assert chat.status_code == 200, chat.text
            assert chat.json()["message"]["content"] == "OK"
            assert not native_admission_runtime.snapshot()
    asyncio.run(run())


@pytest.mark.parametrize("proof", [True, False])
def test_dashboard_load_oom_uses_lifecycle_receipt(proof, marker_runtime, native_admission_runtime, monkeypatch):
    import app as dashboard
    original_client = httpx.AsyncClient
    def backend(req):
        assert req.url.path == "/api/load"
        assert native.valid_operation_id(req.headers[native.OPERATION_HEADER])
        assert native.valid_operation_id(req.headers[native.OVERLAY_HEADER])
        return httpx.Response(503, json={"error": "not enough GPU memory", "code": "resource_exhausted"},
            headers={native.COMPLETION_HEADER: req.headers[native.OPERATION_HEADER]} if proof else {})
    monkeypatch.setattr(dashboard.httpx, "AsyncClient", lambda *args, **kwargs:
        original_client(*args, **kwargs, transport=httpx.MockTransport(backend)))
    async def run():
        response = await dashboard.load_deberta_model({"model": "bge-reranker-v2-m3"})
        assert response.status_code == 503
        if proof:
            assert not native_admission_runtime.snapshot()
            assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        else:
            ticket, = native_admission_runtime.snapshot()
            assert ticket.phase == "unknown" and ticket.owner == "kiron-proxy-lifecycle"
            marker = json.loads(vl.GPU_SERVICE_LOADING_MARKER_PATH.read_text())
            async with original_client(base_url="http://backend", transport=httpx.MockTransport(lambda req:
                httpx.Response(200, json={"schema_version": 1, "backend_instance": "f" * 32,
                    "state": "terminated", "operation_id": ticket.operation_id,
                    "overlay_token": marker["token"]}))) as client:
                await native.reconcile_native_operations(client, native_admission_runtime)
            assert not native_admission_runtime.snapshot()
            assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())


def test_dashboard_memory_refusal_never_sends_load(marker_runtime, native_admission_runtime, monkeypatch):
    import app as dashboard
    monkeypatch.setattr(native, "measure_memory", lambda: MemorySnapshot(211 * 1024**2, 32 * 1024**3, time.monotonic()))
    monkeypatch.setattr(dashboard.httpx, "AsyncClient", lambda **kwargs: pytest.fail("must reject before HTTP"))
    async def run():
        response = await dashboard.load_deberta_model({"model": "bge-reranker-v2-m3"})
        assert response.status_code == 503
        assert json.loads(response.body)["reason"] == "resource_exhausted"
        assert not native_admission_runtime.snapshot()
        assert not vl.GPU_SERVICE_LOADING_MARKER_PATH.exists()
    asyncio.run(run())
