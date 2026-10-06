"""Lifecycle fences use actual locked state; all backend work is simulated."""
import asyncio
import json
import os
import threading
import time
from unittest import mock

import pytest

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.gpu_admission.ollama_lifecycle import OllamaLifecycleOperation


@pytest.fixture
def store(tmp_path, monkeypatch):
    from kiron_common.gpu_admission.ollama_backend import OllamaBackend, BackendState
    async def inspect(self, target="kiron-ollama"):
        return BackendState("a" * 64, "2026-09-26T00:00:00Z", True, "running", 123)
    monkeypatch.setattr(OllamaBackend, "inspect", inspect)
    tmp_path.chmod(0o2770)
    return AdmissionStore(tmp_path, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))


def reserve(store, **changes):
    args = dict(operation_id="request", owner="native-owner", generation="native-epoch",
                deployment_id="native:alias", kind="request", gpu_bytes=0, host_bytes=0,
                measure=lambda: MemorySnapshot(1000, 1000, time.monotonic()),
                lifecycle_domain="ollama", gpu_guard=False)
    return store.reserve(**(args | changes))


def mutation(store, **changes):
    args = dict(store=store, owner="operator", generation="operator-epoch", operation_id="unload",
                deployment_id="direct:target", deadline_monotonic=time.monotonic() + .2)
    return OllamaLifecycleOperation(**(args | changes))


@pytest.mark.parametrize("unknown", [False, True])
def test_active_and_unknown_cpu_work_blocks_direct_mutation_without_releasing_it(store, unknown):
    async def run():
        ticket = reserve(store)
        if unknown:
            store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                          confirmed_terminated=False)
        before = store.snapshot()
        with pytest.raises(AdmissionError):
            async with mutation(store, deadline_monotonic=time.monotonic() + .03):
                pytest.fail("mutation reached backend")
        assert store.snapshot() == before
    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["unstarted", "confirmed", "uncertain"])
def test_only_own_fence_is_released_and_gpu_ticket_overlay_never_adopted(store, outcome):
    async def run():
        foreign = reserve(store, operation_id="docling", kind="docling", gpu_guard=True,
                          lifecycle_domain=None, deployment_id="docling", owner="docling")
        overlay = store.root / "docling-vram-startup.json"
        overlay.write_text("foreign marker remains byte exact")
        async with mutation(store) as operation:
            with pytest.raises(AdmissionError):
                reserve(store, operation_id="new", deployment_id="different-alias")
            if outcome != "unstarted":
                operation.mark_started()
            if outcome == "confirmed":
                operation.confirm_end()
        tickets = store.snapshot()
        assert foreign in tickets and overlay.read_text() == "foreign marker remains byte exact"
        if outcome == "uncertain":
            assert len(tickets) == 2 and tickets[-1].phase == "unknown"
        else:
            assert tickets == (foreign,)
    asyncio.run(run())


def test_drain_observes_foreign_owner_end_without_adopting_its_epoch(store):
    async def run():
        ticket = reserve(store)
        pending = asyncio.create_task(mutation(store).__aenter__())
        for _ in range(100):
            if any(t.kind == "unload" for t in store.snapshot()):
                break
            await asyncio.sleep(.001)
        else:
            pytest.fail("missing fence")
        store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                      confirmed_terminated=True)
        operation = await pending
        assert operation.ticket.owner == "operator" and operation.ticket.generation == "operator-epoch"
        await operation.close()
        assert store.snapshot() == ()
    asyncio.run(run())


def test_cancelled_reservation_thread_releases_only_new_fence(store):
    async def run():
        entered, proceed = threading.Event(), threading.Event()
        original = store.begin_unload
        def delayed(*args, **kwargs):
            entered.set()
            assert proceed.wait(1)
            return original(*args, **kwargs)
        with mock.patch.object(store, "begin_unload", side_effect=delayed):
            task = asyncio.create_task(mutation(store).__aenter__())
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert store.snapshot() == ()
    asyncio.run(run())


def test_duplicate_operation_id_never_acquires_release_authority(store):
    async def run():
        original = store.begin_unload("unload", owner="operator", generation="operator-epoch",
                    deployment_id="direct:target", require_resident=False, lifecycle_domain="ollama")
        with pytest.raises(AdmissionError):
            async with mutation(store):
                pytest.fail("duplicate operation entered")
        assert store.snapshot() == (original,)
    asyncio.run(run())


def test_cpu_request_is_not_gpu_exclusivity_but_domain_unknown_remains_blocking(store):
    cpu = reserve(store)
    gpu = reserve(store, operation_id="training", kind="training", lifecycle_domain=None,
                  gpu_guard=True, exclusive=True)
    assert store.snapshot() == (cpu, gpu)
    with pytest.raises(AdmissionError):
        reserve(store, operation_id="canonical", generation="different", lifecycle_model="digest")
    store.release(cpu.operation_id, owner=cpu.owner, generation=cpu.generation, confirmed_terminated=False)
    with pytest.raises(AdmissionError):
        reserve(store, operation_id="another-cpu")


def test_load_fence_and_aliased_residents_cannot_be_bypassed(store):
    resident = reserve(store, kind="load", gpu_guard=True, lifecycle_model="digest")
    with pytest.raises(AdmissionError):
        reserve(store, operation_id="native")
    with pytest.raises(AdmissionError):
        store.begin_unload("unload", owner="operator", generation="operator-epoch",
            deployment_id="another-profile", require_resident=False,
            lifecycle_domain="ollama", lifecycle_model="digest")
    assert store.snapshot() == (resident,)
    resident = store.transition(resident.operation_id, owner=resident.owner,
                               expected_generation=resident.generation, phase="resident")
    with pytest.raises(AdmissionError):
        reserve(store, operation_id="native")
    with pytest.raises(AdmissionError):
        store.begin_unload("unload", owner="operator", generation="operator-epoch",
            deployment_id="another-profile", require_resident=False,
            lifecycle_domain="ollama", lifecycle_model="digest")
    assert store.snapshot() == (resident,)


def test_old_admission_schema_has_no_implicit_ticket_migration(store):
    reserve(store)
    path = store.root / "admission.json"
    value = json.loads(path.read_bytes())
    assert value["schema_version"] == 3
    value["schema_version"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(AdmissionError, match="state invalid"):
        store.snapshot()


def test_closed_fence_cannot_authorize_later_mutation(store):
    async def run():
        async with mutation(store) as operation:
            pass
        with pytest.raises(RuntimeError):
            operation.mark_started()
        assert store.snapshot() == ()
    asyncio.run(run())
