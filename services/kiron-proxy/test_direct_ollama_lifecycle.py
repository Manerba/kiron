"""Real cross-process store, real direct handlers, fake native HTTP; no models."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from kiron_common.gpu_admission import AdmissionError, MemorySnapshot

PATHS = ("promotion", "load", "unload", "delete", "docling")
MODEL = "fixture:latest"


@pytest.fixture
def direct(monkeypatch, tmp_path, native_admission_runtime):
    import app
    import main
    import native_admission
    path = Path(__file__).parents[1] / "kiron-docling" / "proxy.py"
    spec = importlib.util.spec_from_file_location("_direct_lifecycle_docling", path)
    docling = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, docling)
    monkeypatch.setenv("KIRON_RUNTIME_DIR", str(tmp_path / "markers"))
    spec.loader.exec_module(docling)
    store = native_admission_runtime
    for module, name in ((app, "ollama_admission_store"), (main, "ollama_admission_store"),
                         (docling, "_admission_store")):
        monkeypatch.setattr(module, name, lambda: store)

    @asynccontextmanager
    async def marker(**kwargs):
        yield SimpleNamespace(allowed=True, clear_marker=False)

    async def no_overlay():
        return False

    monkeypatch.setattr(app.vram_lease, "gpu_service_operation", marker)
    monkeypatch.setattr(app.vram_lease, "effective_snapshot", no_overlay)
    monkeypatch.setattr(native_admission.vram_lease, "num_gpu_zero_effective", lambda: True)
    original_client = httpx.AsyncClient

    def transport(handler):
        monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs:
                            original_client(*args, **kwargs, transport=httpx.MockTransport(handler),
                                            trust_env=False))

    async def invoke(kind):
        if kind == "promotion":
            return await main._promote_cpu_resident_models()
        if kind == "docling":
            return await docling._free_vram_for_docling()
        return await getattr(app, kind + "_model")({"name": MODEL, "gpu": False, "force": True})

    return SimpleNamespace(store=store, app=app, docling=docling, native=native_admission,
                           transport=transport, invoke=invoke)


def reserve(store, *, operation="request", canonical=False, domain="ollama", kind="request"):
    return store.reserve(operation_id=operation, owner="original-owner", generation="original-generation",
                         deployment_id="canonical-model" if canonical else "native:" + domain + ":" + MODEL,
                         kind=kind, gpu_bytes=0, host_bytes=0,
                         measure=lambda: MemorySnapshot(10**9, 10**9, time.monotonic()),
                         lifecycle_domain=domain, lifecycle_model=MODEL if canonical else None,
                         gpu_guard=canonical or kind != "request")


def release(store, operation="request", *, confirmed=True):
    store.release(operation, owner="original-owner", generation="original-generation",
                  confirmed_terminated=confirmed)


def state(kind, *, before=False):
    loaded = before and kind in ("promotion", "docling") or kind == "load" and not before
    return {"models": [{"name": MODEL, "size_vram": 8 * 1024**3 if kind == "docling" else 0}]
            if loaded else []}


async def invoke_denied(direct, kind):
    if kind == "docling":
        with pytest.raises(direct.docling.VramGateError):
            await direct.invoke(kind)
    else:
        result = await direct.invoke(kind)
        assert result == 0 if kind == "promotion" else result.status_code == 409


@pytest.mark.parametrize("kind", PATHS)
@pytest.mark.parametrize("canonical", (False, True))
def test_active_request_drains_then_fence_holds_until_native_proof(direct, monkeypatch, kind, canonical):
    async def run():
        store = direct.store
        reserve(store, canonical=canonical)
        foreign = reserve(store, operation="foreign", domain="unrelated")
        fenced, verifying, permit_proof = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_begin = store.begin_unload
        loop = asyncio.get_running_loop()

        def begin(*args, **kwargs):
            result = original_begin(*args, **kwargs)
            loop.call_soon_threadsafe(fenced.set)
            return result

        monkeypatch.setattr(store, "begin_unload", begin)
        mutations = []

        async def backend(request):
            if request.method in ("POST", "DELETE"):
                mutations.append(request)
                assert not any(t.operation_id == "request" for t in store.snapshot())
                assert any(t.kind == "unload" for t in store.snapshot())
                if kind == "delete":  # DELETE's synchronous HTTP result is its end proof.
                    verifying.set()
                    await permit_proof.wait()
                return httpx.Response(200, json={"done": True})
            if mutations:
                assert any(t.kind == "unload" for t in store.snapshot())
                verifying.set()
                await permit_proof.wait()
            return httpx.Response(200, json=state(kind, before=not mutations))

        direct.transport(backend)
        task = asyncio.create_task(direct.invoke(kind))
        try:
            await asyncio.wait_for(fenced.wait(), 2)
            assert not mutations
            sent = []

            class Client:
                async def send(self, *args, **kwargs):
                    sent.append(args)

            op = direct.native.NativeRequestOperation(backend="ollama", endpoint="/api/chat", model=MODEL,
                body=b'{"options":{"num_gpu":0}}', store=store)
            with pytest.raises(AdmissionError):
                await op.send(Client(), "unused")
            await op.close()
            assert not sent  # Verified CPU must still honor the lifecycle fence.
            release(store)
            await asyncio.wait_for(verifying.wait(), 2)
            assert any(t.kind == "unload" and t.phase != "unknown" for t in store.snapshot())
            permit_proof.set()
            await asyncio.wait_for(task, 2)
            assert len(mutations) == 1
            assert store.snapshot() == (foreign,)
        finally:
            permit_proof.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize("kind", PATHS)
@pytest.mark.parametrize("blocker", ("native_unknown", "canonical_unknown", "canonical_resident"))
def test_unknown_or_managed_resident_prevents_direct_mutation_even_force(direct, kind, blocker):
    async def run():
        store = direct.store
        resident = blocker == "canonical_resident"
        reserve(store, canonical=blocker != "native_unknown", kind="load" if resident else "request")
        if resident:
            store.transition("request", owner="original-owner", expected_generation="original-generation",
                             phase="resident")
        else:
            release(store, confirmed=False)
        before = store.snapshot()
        mutations = []

        async def backend(request):
            if request.method != "GET":
                mutations.append(request)
            return httpx.Response(200, json=state(kind, before=True))

        direct.transport(backend)
        await asyncio.wait_for(invoke_denied(direct, kind), 2)
        assert not mutations
        assert store.snapshot() == before
    asyncio.run(run())


@pytest.mark.parametrize("kind", PATHS)
@pytest.mark.parametrize("failure", ("http500", "read_error", "cancel"))
def test_failure_after_native_start_keeps_unknown_and_foreign_tickets(direct, kind, failure):
    async def run():
        store = direct.store
        foreign = reserve(store, operation="foreign", domain="unrelated")

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json=state(kind, before=True))
            assert any(t.kind == "unload" for t in store.snapshot())
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "read_error":
                raise httpx.ReadError("injected incomplete native response", request=request)
            return httpx.Response(500, json={"error": "injected"})

        direct.transport(backend)
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await direct.invoke(kind)
        elif kind == "docling":
            with pytest.raises(direct.docling.VramGateError):
                await direct.invoke(kind)
        else:
            result = await direct.invoke(kind)
            assert result == 0 if kind == "promotion" else result.status_code >= 400
        tickets = store.snapshot()
        assert foreign in tickets
        unknown = [t for t in tickets if t.operation_id != "foreign"]
        assert len(unknown) == 1 and unknown[0].kind == "unload" and unknown[0].phase == "unknown"
    asyncio.run(run())


@pytest.mark.parametrize("kind", PATHS)
def test_cancel_during_drain_removes_only_unstarted_own_fence(direct, monkeypatch, kind):
    async def run():
        store = direct.store
        request = reserve(store)
        original_begin = store.begin_unload
        loop, fenced = asyncio.get_running_loop(), asyncio.Event()

        def begin(*args, **kwargs):
            result = original_begin(*args, **kwargs)
            loop.call_soon_threadsafe(fenced.set)
            return result

        monkeypatch.setattr(store, "begin_unload", begin)
        async def backend(req):
            assert req.method == "GET"
            return httpx.Response(200, json=state(kind, before=True))
        direct.transport(backend)
        task = asyncio.create_task(direct.invoke(kind))
        await asyncio.wait_for(fenced.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.snapshot() == (request,)
    asyncio.run(run())


@pytest.mark.parametrize("kind", ("promotion", "load", "unload", "docling"))
def test_unknown_post_mutation_ps_cannot_release_fence(direct, kind):
    async def run():
        mutated = False
        async def backend(request):
            nonlocal mutated
            if request.method == "POST":
                mutated = True
                return httpx.Response(200, json={"done": True})
            return httpx.Response(200, json={"models": None} if mutated else state(kind, before=True))
        direct.transport(backend)
        if kind == "docling":
            with pytest.raises(direct.docling.VramGateError):
                await direct.invoke(kind)
        else:
            result = await direct.invoke(kind)
            assert result == 0 if kind == "promotion" else result.status_code == 502
        ticket, = direct.store.snapshot()
        assert mutated and ticket.kind == "unload" and ticket.phase == "unknown"
    asyncio.run(run())


def test_dashboard_load_requires_done_even_when_ps_reports_residency(direct):
    async def run():
        async def backend(request):
            return httpx.Response(200, json={"done": False} if request.method == "POST" else state("load"))
        direct.transport(backend)
        result = await direct.invoke("load")
        assert result.status_code == 502
        ticket, = direct.store.snapshot()
        assert ticket.kind == "unload" and ticket.phase == "unknown"
    asyncio.run(run())


def test_docling_unknown_initial_name_cannot_claim_vram_is_free(direct):
    async def run():
        async def backend(request):
            assert request.method == "GET"
            return httpx.Response(200, json={"models": [{"size_vram": 8 * 1024**3}]})
        direct.transport(backend)
        with pytest.raises(direct.docling.VramGateError):
            await direct.invoke("docling")
        assert not direct.store.snapshot()
    asyncio.run(run())


@pytest.mark.parametrize("bad_row", ({}, {"name": None}, {"name": ""}))
def test_docling_unknown_verification_rows_cannot_prove_end(direct, bad_row):
    async def run():
        mutations = []
        async def backend(request):
            if request.method == "POST":
                mutations.append(request)
                return httpx.Response(200, json={"done": True})
            return httpx.Response(200, json={"models": [bad_row]} if mutations else state("docling", before=True))
        direct.transport(backend)
        with pytest.raises(direct.docling.VramGateError):
            await direct.invoke("docling")
        ticket, = direct.store.snapshot()
        assert ticket.kind == "unload" and ticket.phase == "unknown"
    asyncio.run(run())


def test_docling_all_unloaded_targets_must_be_absent_in_same_snapshot(direct, monkeypatch):
    async def run():
        rows = [{"name": name, "size_vram": 8 * 1024**3} for name in (MODEL, "other:latest")]
        mutations, observations = [], []
        async def backend(request):
            if request.method == "POST":
                mutations.append(request)
                return httpx.Response(200, json={"done": True})
            if not mutations:
                return httpx.Response(200, json={"models": rows})
            observed = [rows[1]] if not observations else [rows[0]] if len(observations) == 1 else []
            observations.append(observed)
            return httpx.Response(200, json={"models": observed})
        async def yield_only(delay):
            await asyncio.sleep(0)
        monkeypatch.setattr(direct.docling, "asyncio", SimpleNamespace(sleep=yield_only))
        direct.transport(backend)
        await direct.invoke("docling")
        assert len(observations) == 3
        assert not direct.store.snapshot()
    asyncio.run(run())
