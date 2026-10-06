"""Container recovery, real admission files and locks, no host mutations."""
import asyncio
import json
from dataclasses import replace
from pathlib import Path
import time
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.testclient import TestClient

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot
from kiron_common.gpu_admission.ollama_backend import BackendLock, BackendState, OllamaBackend, OllamaBackendSession
import native_admission
from native_admission import NativeRequestOperation
import ollama_recovery
from test_native_admission import Client, Response, operation


INSTANCE = "a" * 64 + "@2026-09-26T00:00:00Z"


def reserve(store, *, unknown=True, **changes):
    args = dict(operation_id="old", owner="kiron-proxy-native", generation="proxy-before-restart",
        deployment_id="native:ollama:qwen3:8b", kind="request", gpu_bytes=0, host_bytes=0,
        measure=lambda: MemorySnapshot(10**9, 10**9, time.monotonic()),
        lifecycle_domain="ollama", backend_instance=INSTANCE)
    ticket = store.reserve(**(args | changes))
    if unknown:
        store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation, confirmed_terminated=False)
    return next(t for t in store.snapshot() if t.operation_id == ticket.operation_id)


class Backend:
    def __init__(self, fail=None):
        self.state = BackendState("a" * 64, "2026-09-26T00:00:00Z", True, "running", 123)
        self.calls, self.fail = [], fail

    async def inspect(self):
        return self.state

    async def stop_and_verify(self, state):
        self.calls.append("stop")
        if self.fail == "stop":
            raise AdmissionError("ollama_stop_unconfirmed", "fixture process still running")
        self.state = replace(state, running=False, status="exited", pid=0)

    async def start_and_verify(self, state):
        self.calls.append("start")
        if self.fail == "start":
            raise AdmissionError("ollama_start_unconfirmed", "fixture start failed")
        self.state = replace(state, started_at="2026-09-26T01:00:00Z", running=True, status="running", pid=124)
        return self.state


def test_recovery_after_proxy_restart_preserves_other_service(native_admission_runtime):
    async def run():
        store = native_admission_runtime
        foreign = reserve(store, unknown=False, operation_id="deberta", deployment_id="native:kiron_deberta:model",
                          lifecycle_domain=None, backend_instance=None)
        old = reserve(store)
        restarted = AdmissionStore(store.root, security=store.security, boot_id=store._boot())
        backend = Backend()
        async def health():
            assert len(store.snapshot()) == 2  # Foreign service plus recovery fence.
            with pytest.raises(AdmissionError):
                async with OllamaBackendSession(store):
                    pytest.fail("new backend work bypassed recovery")
        result = await ollama_recovery.recover(restarted, ollama_recovery.status(restarted)["revision"],
                                              backend=backend, health=health)
        assert result == {"status": "recovered", "cleared_operations": 1}
        assert backend.calls == ["stop", "start"]
        assert store.snapshot() == (foreign,)
        assert old.backend_instance == INSTANCE
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["active", "foreign", "unbound", "stale_revision", "held_lock"])
def test_recovery_cannot_bypass_active_foreign_or_changed_work(reason, native_admission_runtime):
    async def run():
        store, backend = native_admission_runtime, Backend()
        reserve(store, unknown=reason != "active", backend_instance=("b" + INSTANCE[1:] if reason == "foreign"
                else None if reason == "unbound" else INSTANCE))
        before = store.snapshot()
        lock = BackendLock(store).acquire() if reason == "held_lock" else None
        try:
            rev = "0" * 64 if reason == "stale_revision" else ollama_recovery.status(store)["revision"]
            with pytest.raises(AdmissionError):
                await ollama_recovery.recover(store, rev, backend=backend, health=AsyncMock())
            assert store.snapshot() == before
            assert backend.calls == []
        finally:
            if lock:
                lock.close()
    asyncio.run(run())


@pytest.mark.parametrize("step", ["stop", "start", "health"])
def test_failed_recovery_keeps_fence_and_can_be_retried(step, native_admission_runtime):
    async def run():
        store, backend = native_admission_runtime, Backend(step)
        old = reserve(store)
        health = AsyncMock(side_effect=AdmissionError("health", "fixture") if step == "health" else None)
        with pytest.raises(AdmissionError):
            await ollama_recovery.recover(store, ollama_recovery.status(store)["revision"], backend=backend, health=health)
        assert all(t.phase == "unknown" for t in store.snapshot())
        assert (old in store.snapshot()) == (step == "stop")
        assert any(t.owner == "kiron-ollama-recovery" for t in store.snapshot())
        backend.fail = None
        await ollama_recovery.recover(store, ollama_recovery.status(store)["revision"], backend=backend, health=AsyncMock())
        assert store.snapshot() == ()
    asyncio.run(run())


def test_browser_disconnect_does_not_cancel_recovery(native_admission_runtime):
    async def run():
        store, backend = native_admission_runtime, Backend()
        reserve(store)
        entered, proceed = asyncio.Event(), asyncio.Event()
        async def health():
            entered.set()
            await proceed.wait()
        task = asyncio.create_task(ollama_recovery.recover(store, ollama_recovery.status(store)["revision"],
                                                         backend=backend, health=health))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proceed.set()
        await asyncio.gather(*tuple(ollama_recovery._operations))
        assert store.snapshot() == ()
    asyncio.run(run())


@pytest.mark.parametrize("matching", [True, False])
def test_recovery_clears_only_owned_overlay_tokens(matching, native_admission_runtime):
    async def run():
        store, backend = native_admission_runtime, Backend()
        old = reserve(store, overlay_token="c" * 32)
        path = store.root / "gpu-service-loading.json"
        path.write_text(json.dumps({"kind": "gpu_service_loading", "token": ("c" if matching else "d") * 32}))
        path.chmod(0o660)
        before = path.read_bytes()
        recovering = ollama_recovery.recover(store, ollama_recovery.status(store)["revision"],
                                             backend=backend, health=AsyncMock())
        if matching:
            await recovering
            assert not path.exists() and store.snapshot() == ()
            assert backend.calls == ["stop", "start"]
        else:
            with pytest.raises(AdmissionError, match="another operation"):
                await recovering
            assert path.read_bytes() == before and old in store.snapshot()
            assert backend.calls == ["stop"]
    asyncio.run(run())


@pytest.mark.parametrize("method,endpoint", [("DELETE", "/api/delete"), ("POST", "/api/pull")])
def test_recovery_also_fences_native_administrative_requests(method, endpoint, native_admission_runtime):
    async def run():
        store = native_admission_runtime
        lock = BackendLock(store, exclusive=True).acquire()
        client = Client(Response())
        op = operation(endpoint=endpoint, method=method)
        try:
            with pytest.raises(AdmissionError):
                await op.send(client, httpx.Request(method, "http://backend" + endpoint))
            await op.close()
            assert not client.sent
        finally:
            lock.close()
    asyncio.run(run())


@pytest.mark.parametrize("ending", ["complete", "timeout", "late_error"])
def test_disconnected_stream_drain_requires_backend_eof(ending, native_admission_runtime, monkeypatch):
    async def run():
        store = native_admission_runtime
        ready, finish = asyncio.Event(), asyncio.Event()
        class SlowResponse(Response):
            async def aiter_bytes(self):
                yield b'{"done":true}\n'
                ready.set()
                await finish.wait()
                if ending == "late_error":
                    raise httpx.ReadError("fixture")
        monkeypatch.setattr(native_admission, "OLLAMA_DRAIN_SECONDS", .02)
        op = operation()
        response = await op.send(Client(SlowResponse()), httpx.Request("POST", "http://backend"))
        chunks = op.chunks(response)
        assert await anext(chunks) == b'{"done":true}\n'
        await ready.wait()
        assert not op.confirmed_end and not op._eof
        await chunks.aclose()
        closing = asyncio.create_task(op.close())
        await asyncio.sleep(0)
        with pytest.raises(AdmissionError):
            BackendLock(store, exclusive=True).acquire()
        if ending != "timeout":
            finish.set()
        await closing
        assert response.closed
        if ending == "complete":
            assert store.snapshot() == ()
        else:
            assert store.snapshot()[0].phase == "unknown"
        lock = BackendLock(store, exclusive=True).acquire()
        lock.close()
    asyncio.run(run())


@pytest.mark.parametrize("populated", ["0", "1"])
def test_stop_command_requires_empty_container_cgroup(populated, tmp_path):
    async def run():
        state = BackendState("a" * 64, "2026-09-26T00:00:00Z", True, "running", 123)
        group = f"/system.slice/docker-{state.container_id}.scope"
        proc = tmp_path / "proc/123"
        proc.mkdir(parents=True)
        (proc / "cgroup").write_text("0::" + group + "\n")
        cgroup = tmp_path / "cgroup" / group.lstrip("/")
        cgroup.mkdir(parents=True)
        (cgroup / "cgroup.events").write_text("populated " + populated + "\n")
        class Control(OllamaBackend):
            async def inspect(self, target=None):
                return replace(state, running=False, status="exited", pid=0)
        command = AsyncMock()
        control = Control(run=command, proc_root=tmp_path / "proc", cgroup_root=tmp_path / "cgroup")
        if populated == "1":
            with pytest.raises(AdmissionError, match="laufen noch"):
                await control.stop_and_verify(state)
        else:
            await control.stop_and_verify(state)
        command.assert_awaited_once_with(["stop", "--time", "10", state.container_id], timeout=20)
    asyncio.run(run())


def test_dashboard_exposes_blocker_and_checks_recovery_origin(native_admission_runtime, monkeypatch):
    import app
    reserve(native_admission_runtime)
    client = TestClient(app.app)
    state = client.get("/api/ollama/recovery", auth=("admin", "admin")).json()
    assert state["blocked"] and state["recovery_available"]
    assert state["operations"][0]["model"] == "qwen3:8b"
    recover = AsyncMock(return_value={"status": "recovered"})
    monkeypatch.setattr(ollama_recovery, "recover", recover)
    assert client.post("/api/ollama/recovery", json={"revision": state["revision"]}, auth=("admin", "admin")).status_code == 403
    assert client.post("/api/ollama/recovery", json={"revision": state["revision"]}, auth=("admin", "admin"),
                       headers={"X-Kiron-Action": "models"}).status_code == 200
    recover.assert_awaited_once_with(native_admission_runtime, state["revision"])
