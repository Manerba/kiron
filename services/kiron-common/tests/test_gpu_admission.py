from dataclasses import replace
import multiprocessing
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity


def _security():
    return RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()}))


def _competing_load(root, start, results, operation):
    store = AdmissionStore(Path(root), security=_security(), boot_id="test")
    start.wait(5)
    try:
        store.reserve(operation_id=operation, owner="proxy", generation="boot:0", deployment_id=operation,
                      kind="load", gpu_bytes=70, host_bytes=0,
                      measure=lambda: MemorySnapshot(100, 100, time.monotonic()))
        results.put("allowed")
    except AdmissionError as exc:
        results.put(exc.code)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o2770)
        self.now = 100.0
        self.store = AdmissionStore(self.root, security=_security(), clock=lambda: self.now, boot_id="test")

    def reserve(self, **changes):
        args = dict(operation_id="op", owner="proxy", generation="boot:0", deployment_id="bonsai",
                    kind="load", gpu_bytes=40, host_bytes=20, ttl_seconds=10,
                    measure=lambda: MemorySnapshot(100, 100, self.now))
        args.update(changes)
        return self.store.reserve(**args)

    def test_two_processes_cannot_reserve_same_free_memory(self):
        ctx = multiprocessing.get_context("spawn")
        start, results = ctx.Event(), ctx.Queue()
        processes = [ctx.Process(target=_competing_load, args=(str(self.root), start, results, str(i)))
                     for i in range(2)]
        for process in processes:
            process.start()
        start.set()
        outcomes = sorted(results.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(outcomes, ["allowed", "resource_exhausted"])

    def test_resident_memory_is_not_double_counted(self):
        self.reserve()
        self.store.transition("op", owner="proxy", expected_generation="boot:0", phase="resident",
                              generation="boot:child")
        # 60 free is the actual post-load measurement, not total device memory.
        self.reserve(operation_id="other", gpu_bytes=55, host_bytes=0,
                     measure=lambda: MemorySnapshot(60, 80, self.now))

    def test_capacity_diagnostics_use_locked_measurement_and_only_pending_budget(self):
        self.reserve()
        self.store.transition("op", owner="proxy", expected_generation="boot:0", phase="resident")
        self.reserve(operation_id="pending", gpu_bytes=50, host_bytes=0)
        before = self.store.snapshot()
        with self.assertRaises(AdmissionError) as caught:
            self.reserve(operation_id="rejected", gpu_bytes=15, host_bytes=0, headroom_bytes=10,
                         measure=lambda: MemorySnapshot(60, 80, self.now))
        error = caught.exception
        self.assertEqual(error.code, "resource_exhausted")
        self.assertEqual(error.details["gpu_free_bytes"], 60)
        self.assertEqual(error.details["gpu_requested_bytes"], 15)
        self.assertEqual(error.details["headroom_bytes"], 10)
        self.assertEqual(error.details["gpu_pending_bytes"], 50)
        self.assertEqual({r["operation_id"] for r in error.details["reservations"]}, {"op", "pending"})
        self.assertEqual(self.store.snapshot(), before)

    def test_expiry_never_releases_resources(self):
        self.reserve()
        self.now += 11
        self.assertEqual(self.store.snapshot()[0].phase, "unknown")
        with self.assertRaisesRegex(AdmissionError, "unconfirmed"):
            self.reserve(operation_id="other")
        with self.assertRaisesRegex(AdmissionError, "cannot reconcile"):
            self.store.heartbeat("op", owner="proxy", generation="boot:0")
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.reserve(operation_id="other")

    def test_wrong_generation_cannot_clear_current_residency(self):
        self.reserve()
        self.store.transition("op", owner="proxy", expected_generation="boot:0", phase="resident",
                              generation="boot:child")
        with self.assertRaises(AdmissionError):
            self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.assertEqual(self.store.snapshot()[0].phase, "resident")

    def test_unknown_backend_end_remains_blocking_and_cleanup_is_idempotent(self):
        self.reserve()
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=False)
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="other")
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.assertEqual(self.store.snapshot(), ())

    def test_confirmed_overlay_cleanup_is_atomic_and_crash_stays_blocked(self):
        self.reserve()
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=False)
        marker = self.root / "gpu-service-loading.json"
        token = "a" * 32
        marker.write_text(json.dumps({"kind": "gpu_service_loading", "token": token}))
        marker.chmod(0o660)
        with mock.patch.object(self.store, "_write", side_effect=OSError("interrupted write")):
            with self.assertRaises(AdmissionError):
                self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True,
                                   owned_overlays={marker.name: token})
        self.assertFalse(marker.exists())
        self.assertEqual(self.store.snapshot()[0].phase, "unknown")
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="new")
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True,
                           owned_overlays={marker.name: token})
        self.assertEqual(self.store.snapshot(), ())

    def test_cleanup_rejects_symlink_marker_and_keeps_ticket(self):
        self.reserve()
        target = self.root / "other.json"
        target.write_text(json.dumps({"kind": "gpu_service_loading", "token": "a" * 32}))
        target.chmod(0o660)
        marker = self.root / "gpu-service-loading.json"
        marker.symlink_to(target)
        with self.assertRaises(AdmissionError):
            self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True,
                               owned_overlays={marker.name: "a" * 32})
        self.assertTrue(target.exists() and marker.is_symlink())
        self.assertEqual(len(self.store.snapshot()), 1)

    def test_external_unload_fences_requests_without_inventing_residency(self):
        with self.assertRaisesRegex(AdmissionError, "resident reservation missing"):
            self.store.begin_unload("unload", owner="proxy", generation="boot:0", deployment_id="bonsai")
        self.assertEqual(self.store.snapshot(), ())
        ticket = self.store.begin_unload("unload", owner="proxy", generation="boot:0",
                                        deployment_id="bonsai", require_resident=False)
        self.assertEqual((ticket.kind, ticket.phase, ticket.gpu_bytes, ticket.host_bytes), ("unload", "active", 0, 0))
        self.assertEqual(self.store.snapshot(), (ticket,))
        with self.assertRaisesRegex(AdmissionError, "draining"):
            self.reserve(operation_id="request", kind="request", gpu_bytes=0, host_bytes=0)
        self.assertEqual(self.store.begin_unload("unload", owner="proxy", generation="boot:0",
            deployment_id="bonsai", require_resident=False), ticket)
        self.store.confirm_deployment_terminated(owner="proxy", generation="boot:0", deployment_id="bonsai")
        self.assertEqual(self.store.snapshot(), ())

    def test_unload_rejects_foreign_target_work_without_modifying_it(self):
        for changes in ({"owner": "foreign"}, {"generation": "old:0"}):
            with self.subTest(changes=changes):
                ticket = self.reserve(**changes)
                before = self.store.snapshot()
                with self.assertRaisesRegex(AdmissionError, "foreign or previous-generation"):
                    self.store.begin_unload("unload", owner="proxy", generation="boot:0",
                                            deployment_id="bonsai", require_resident=False)
                self.assertEqual(self.store.snapshot(), before)
                self.store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                                   confirmed_terminated=True)

    def test_target_termination_preserves_other_deployments_owners_and_generations(self):
        tickets = [self.reserve(operation_id="own")]
        for operation, values in (("other-model", {"deployment_id": "other"}),
                                  ("foreign", {"owner": "foreign"}), ("old", {"generation": "old:0"})):
            tickets.append(self.reserve(operation_id=operation, gpu_bytes=0, host_bytes=0, **values))
        self.store.confirm_deployment_terminated(owner="proxy", generation="boot:0", deployment_id="bonsai")
        self.assertEqual(set(self.store.snapshot()), set(tickets[1:]))

    def test_uncertain_external_unload_keeps_blocking_admission(self):
        self.store.begin_unload("unload", owner="proxy", generation="boot:0", deployment_id="bonsai",
                                require_resident=False)
        self.store.release("unload", owner="proxy", generation="boot:0", confirmed_terminated=False)
        with self.assertRaisesRegex(AdmissionError, "unconfirmed"):
            self.reserve(operation_id="request", kind="request", gpu_bytes=0, host_bytes=0)
        self.assertEqual(self.store.snapshot()[0].phase, "unknown")

    def test_idempotent_operation_requires_matching_parameters(self):
        first = self.reserve()
        self.now += 1
        self.assertEqual(self.reserve(), first)
        with self.assertRaises(AdmissionError):
            self.reserve(gpu_bytes=41)

    def test_training_blocks_inference_and_inference_blocks_exclusive_training(self):
        self.reserve(kind="training", exclusive=True)
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="request", kind="request", gpu_bytes=0, host_bytes=0)
        self.store.release("op", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.reserve(kind="request", gpu_bytes=0, host_bytes=0)
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="training", kind="training", exclusive=True)

    def test_request_slot_is_reserved_atomically(self):
        self.reserve(kind="request", slot_limit=1)
        with self.assertRaisesRegex(AdmissionError, "slots"):
            self.reserve(operation_id="other", kind="request", slot_limit=1)

    def test_stale_or_invalid_memory_fails_closed(self):
        for value in (MemorySnapshot(100, 100, 97), MemorySnapshot(100, 100, 101),
                      MemorySnapshot(True, 100, 100), MemorySnapshot(100, -1, 100),
                      MemorySnapshot(100, 100, float("nan"))):
            with self.subTest(value=value), self.assertRaises(AdmissionError):
                self.reserve(measure=lambda: value)

    def test_headroom_and_host_reservations_count(self):
        with self.assertRaises(AdmissionError):
            self.reserve(gpu_bytes=90, headroom_bytes=11)
        self.reserve(host_bytes=90)
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="other", gpu_bytes=0, host_bytes=11)

    def test_any_overlay_requires_confirmed_cleanup(self):
        marker = self.root / "docling-vram-startup.json"
        marker.write_text('{"ttl_s": 0}')
        with self.assertRaisesRegex(AdmissionError, "overlay"):
            self.reserve()

    def test_corrupt_state_never_becomes_empty(self):
        self.reserve()
        (self.root / "admission.json").write_text("{")
        with self.assertRaises(AdmissionError):
            self.store.snapshot()

    def test_symlink_hardlink_and_unsafe_permissions_fail_closed(self):
        outside = self.root / "target"
        outside.write_text("sentinel")
        for mode in ("symlink", "hardlink", "unsafe"):
            lock = self.root / ".admission.lock"
            with self.subTest(mode=mode):
                if mode == "symlink":
                    lock.symlink_to(outside)
                elif mode == "hardlink":
                    os.link(outside, lock)
                else:
                    lock.write_text("")
                    lock.chmod(0o666)
                with self.assertRaises(AdmissionError):
                    self.reserve()
                lock.unlink()
        self.assertEqual(outside.read_text(), "sentinel")

    def test_changed_host_boot_requires_reconciliation(self):
        self.reserve()
        self.store._boot_id = "next-boot"
        with self.assertRaises(AdmissionError):
            self.store.snapshot()

    def test_only_live_owned_marker_token_can_authorize_startup_reservation(self):
        marker = self.root / "docling-vram-startup.json"
        token = "a" * 32
        base = dict(token=token, kind="startup", pid=os.getpid(), deadline_monotonic=110, ttl_s=10)
        for changed in ({"pid": os.getpid() + 1}, {"deadline_monotonic": 99}, {"token": "b" * 32},
                        {"kind": "shutdown"}, {"ttl_s": float("nan")}):
            marker.write_text(json.dumps({**base, **changed}))
            marker.chmod(0o660)
            with self.subTest(changed=changed), self.assertRaises(AdmissionError):
                self.reserve(kind="docling", owned_overlays={marker.name: token})
        marker.write_text(json.dumps(base))
        self.reserve(kind="docling", owned_overlays={marker.name: token})

    def test_warm_docling_residency_allows_fit_but_requires_exclusive_reactivation(self):
        self.reserve(kind="docling", exclusive=True)
        self.store.transition("op", owner="proxy", expected_generation="boot:0", phase="resident")
        self.reserve(operation_id="other", gpu_bytes=1, host_bytes=0)
        with self.assertRaises(AdmissionError):
            self.store.activate_docling("op", owner="proxy", generation="boot:0")
        self.store.release("other", owner="proxy", generation="boot:0", confirmed_terminated=True)
        self.assertEqual(self.store.activate_docling("op", owner="proxy", generation="boot:0").phase, "active")

    def embedding_parent(self, **changes):
        parent = self.reserve(operation_id="a" * 32, owner="kiron-proxy-lifecycle",
            deployment_id="native-lifecycle:Embedding-Service", kind="request", gpu_bytes=0,
            host_bytes=0, overlay_token="b" * 32, **changes)
        marker = self.root / "gpu-service-loading.json"
        marker.write_text(json.dumps(dict(token=parent.overlay_token, kind="gpu_service_loading",
            pid=os.getpid() + 1, deadline_monotonic=self.now + 10, ttl_s=10)))
        marker.chmod(0o660)
        return parent, marker

    def embedding_load(self, parent, **changes):
        return self.reserve(owner="kiron-embeddings",
            embedding_load_parent=(parent.operation_id, parent.overlay_token), **changes)

    def test_embedding_handoff_accepts_live_parent_across_processes_and_preserves_gate(self):
        parent, marker = self.embedding_parent()
        # The worker runs under a different service UID and PID from the proxy.
        with mock.patch("kiron_common.gpu_admission.store.os.geteuid", return_value=os.getuid() + 1):
            child = self.embedding_load(parent)
        self.assertEqual(set(self.store.snapshot()), {parent, child})
        self.assertTrue(marker.exists())
        with self.assertRaisesRegex(AdmissionError, "overlay"):
            self.reserve(operation_id="unrelated", gpu_bytes=0, host_bytes=0)

    def test_embedding_handoff_requires_exact_live_parent(self):
        parent, marker = self.embedding_parent()
        for invalid in (replace(parent, operation_id="c" * 32), replace(parent, overlay_token="d" * 32)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(AdmissionError, "parent"):
                self.embedding_load(invalid)
            self.assertEqual(self.store.snapshot(), (parent,))
        self.now += 11
        with self.assertRaisesRegex(AdmissionError, "parent"):
            self.embedding_load(parent)
        self.assertTrue(marker.exists())

    def test_embedding_handoff_rejects_unknown_or_wrong_service_parent(self):
        parent, marker = self.embedding_parent()
        self.store.release(parent.operation_id, owner=parent.owner, generation=parent.generation,
                           confirmed_terminated=False)
        with self.assertRaisesRegex(AdmissionError, "parent"):
            self.embedding_load(parent)
        self.store.release(parent.operation_id, owner=parent.owner, generation=parent.generation,
                           confirmed_terminated=True, owned_overlays={marker.name: parent.overlay_token})
        foreign = self.reserve(operation_id=parent.operation_id, owner=parent.owner,
            deployment_id="native-lifecycle:DeBERTa-Service", kind="request", overlay_token=parent.overlay_token)
        with self.assertRaisesRegex(AdmissionError, "parent"):
            self.embedding_load(foreign)

    def test_embedding_handoff_requires_matching_live_marker_and_respects_other_overlays(self):
        parent, marker = self.embedding_parent()
        original = json.loads(marker.read_text())
        for change in ({"token": "c" * 32}, {"deadline_monotonic": self.now - 1},
                       {"kind": "startup"}, {"pid": True}):
            marker.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(AdmissionError):
                self.embedding_load(parent)
        marker.unlink()
        with self.assertRaisesRegex(AdmissionError, "marker is missing"):
            self.embedding_load(parent)
        marker.write_text(json.dumps(original))
        marker.chmod(0o660)
        (self.root / "docling-vram-startup.json").write_text("{}")
        with self.assertRaisesRegex(AdmissionError, "overlay"):
            self.embedding_load(parent)
        self.assertEqual(self.store.snapshot(), (parent,))

    def test_embedding_handoff_keeps_memory_limits_and_unknown_tickets(self):
        parent, marker = self.embedding_parent()
        with self.assertRaisesRegex(AdmissionError, "insufficient"):
            self.embedding_load(parent, gpu_bytes=101)
        marker.unlink()
        foreign = self.reserve(operation_id="foreign", gpu_bytes=0, host_bytes=0)
        self.store.release(foreign.operation_id, owner=foreign.owner, generation=foreign.generation,
                           confirmed_terminated=False)
        marker.write_text(json.dumps(dict(token=parent.overlay_token, kind="gpu_service_loading",
            pid=os.getpid() + 1, deadline_monotonic=self.now + 10, ttl_s=10)))
        marker.chmod(0o660)
        with self.assertRaisesRegex(AdmissionError, "unconfirmed"):
            self.embedding_load(parent)

    def test_embedding_handoff_is_only_for_guarded_embedding_loads(self):
        parent, _ = self.embedding_parent()
        for changes in ({"owner": "foreign"}, {"kind": "request"}, {"gpu_guard": False},
                        {"embedding_load_parent": (parent.operation_id, "invalid")}):
            args = dict(owner="kiron-embeddings", embedding_load_parent=(parent.operation_id, parent.overlay_token))
            with self.subTest(changes=changes), self.assertRaises((ValueError, AdmissionError)):
                self.reserve(**{**args, **changes})
        with self.assertRaises(AdmissionError):
            self.reserve(operation_id="other", gpu_bytes=1, host_bytes=0)

    def test_controller_restart_cannot_replace_a_still_reserved_runtime_slot(self):
        self.reserve(resident_slot="prism")
        self.store.transition("op", owner="proxy", expected_generation="boot:0", phase="resident")
        with self.assertRaisesRegex(AdmissionError, "slot"):
            self.reserve(operation_id="replacement", generation="new-boot:0", resident_slot="prism")
        with self.assertRaisesRegex(AdmissionError, "generation"):
            self.reserve(operation_id="request", kind="request", generation="new-boot:0", resident_slot="prism", gpu_bytes=0, host_bytes=0)


if __name__ == "__main__":
    unittest.main()
