"""No models/CUDA: real shared-store transactions and serial-worker boundaries."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import build_resolver_snapshot
from kiron_common.prism_runtime_policy import immutable_path
from model_worker import SerialModelWorker, StaleGenerationError
from native_runtime import generation
from residency import OWNER, Profile, ResourceSample, ResidencyOwner, load_profiles


class ResidencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o2770)
        self.now = 100.0
        self.store = AdmissionStore(self.root, security=RuntimeSecurity(os.geteuid(), os.getegid(), frozenset({os.geteuid()})),
                                    clock=lambda: self.now)
        self.model = build_resolver_snapshot(MODEL_CATALOG, ()).resolve('kiron-mankei-dense-v1.query')
        dep = self.model.deployment
        self.name = dep.reference
        self.profile = Profile(dep.id, dep.reference, dep.artifact_identity.fingerprint,
                               dep.configuration_fingerprint, 'cuda', 3 << 30, 4 << 30, 256 << 20, 180)
        self.rss, self.gpu = 512 << 20, 0
        self.snapshot = {'model_epoch': 0, 'loaded_models': [], 'verified_artifacts': {}, 'device': 'cuda'}
        self.instances = {}
        self.owner = ResidencyOwner(profiles={(self.name, 'cuda'): self.profile}, store=self.store,
            generation=self.key, measure=self.measure, clock=lambda: self.now)

    @staticmethod
    def key(snapshot):
        value = generation(snapshot)
        return json.dumps([value['boot_id'], value['process_id']], separators=(',', ':'))

    def measure(self):
        return ResourceSample(MemorySnapshot(10 << 30, 32 << 30, self.now), self.rss, self.gpu)

    def reserve(self, **changes):
        args = dict(operation_id='foreign', owner='kiron-proxy', generation='foreign-generation',
                    deployment_id=self.profile.deployment_id, kind='request', gpu_bytes=0, host_bytes=0,
                    measure=lambda: self.measure().available)
        args.update(changes)
        return self.store.reserve(**args)

    def load(self):
        self.owner.before_load(self.name, 'cuda', self.snapshot, None)
        self.snapshot.update(model_epoch=1, loaded_models=[self.name],
                             verified_artifacts={self.name: self.profile.artifact_fingerprint})
        self.instances[self.name] = 1
        self.rss += 2 << 30
        self.gpu += 1 << 30
        self.owner.complete(self.snapshot, self.instances)
        return self.owner.residents[self.name]

    def test_reserve_precedes_load_and_measured_resident_uses_actual_generation(self):
        self.owner.before_load(self.name, 'cuda', self.snapshot, None)
        before = self.store.snapshot()[0]
        self.assertEqual((before.owner, before.phase), (OWNER, 'reserved'))
        self.assertEqual((before.gpu_bytes, before.host_bytes), (3 << 30, 4 << 30))
        self.snapshot.update(model_epoch=1, loaded_models=[self.name],
                             verified_artifacts={self.name: self.profile.artifact_fingerprint})
        self.instances[self.name] = 1
        self.owner.complete(self.snapshot, self.instances)
        after = self.store.snapshot()[0]
        self.assertEqual((after.phase, after.generation), ('resident', self.key(self.snapshot)))
        self.assertEqual(self.owner.observations[self.name]['rss_bytes'], self.rss)

    def test_active_native_or_canonical_request_does_not_deadlock_its_queued_load(self):
        foreign = self.reserve()
        self.load()
        self.assertIn(foreign, self.store.snapshot())
        self.assertEqual(len(self.store.snapshot()), 2)

    def test_dashboard_parent_allows_owned_reservation_behind_proxy_marker(self):
        parent = self.reserve(operation_id="a" * 32, owner="kiron-proxy-lifecycle",
            deployment_id="native-lifecycle:Embedding-Service", overlay_token="b" * 32)
        marker = self.root / "gpu-service-loading.json"
        marker.write_text(json.dumps(dict(kind="gpu_service_loading", token=parent.overlay_token,
            pid=os.getpid() + 1, ttl_s=60, deadline_monotonic=self.now + 60)))
        marker.chmod(0o660)
        self.owner.before_load(self.name, "cuda", self.snapshot, None,
                               load_parent=(parent.operation_id, parent.overlay_token))
        self.assertIn(self.name, self.owner.pending)
        self.assertIn(parent, self.store.snapshot())
        self.assertEqual(len(self.store.snapshot()), 2)

    def test_other_cache_mutation_rotates_only_owned_resident(self):
        ticket = self.load()
        foreign = self.reserve()
        self.snapshot['model_epoch'] += 1
        self.snapshot['loaded_models'].append('unprofiled-native-cache')
        self.owner.complete(self.snapshot, self.instances)
        current = self.owner.residents[self.name]
        self.assertEqual(current.operation_id, ticket.operation_id)
        self.assertNotEqual(current.generation, ticket.generation)
        self.assertIn(foreign, self.store.snapshot())

    def test_foreign_draining_unknown_or_stale_resident_rejects_before_mutation(self):
        for phase in ('resident', 'unknown', 'draining'):
            with self.subTest(phase=phase):
                ticket = self.reserve(operation_id='foreign-' + phase, kind='load')
                self.store.transition(ticket.operation_id, owner=ticket.owner, expected_generation=ticket.generation,
                                      phase='resident' if phase == 'draining' else phase)
                if phase == 'draining':
                    self.store.begin_unload('drain', owner=ticket.owner, generation=ticket.generation,
                                            deployment_id=ticket.deployment_id)
                with self.assertRaises(AdmissionError):
                    self.owner.before_load(self.name, 'cuda', self.snapshot, None)
                self.assertEqual(self.owner.pending, {})
                self.store.confirm_deployment_terminated(owner=ticket.owner, generation=ticket.generation,
                                                         deployment_id=ticket.deployment_id)

    def test_own_reload_keeps_old_reservation_until_serial_replacement(self):
        old = self.load()
        self.owner.before_load(self.name, 'cuda', self.snapshot, self.instances[self.name])
        self.assertEqual(len(self.store.snapshot()), 2)
        self.assertIn(old, self.store.snapshot())
        self.snapshot['model_epoch'] += 1
        self.instances[self.name] = 2
        self.owner.complete(self.snapshot, self.instances)
        self.assertEqual(len(self.store.snapshot()), 1)
        self.assertNotEqual(self.owner.residents[self.name].operation_id, old.operation_id)

    def test_failed_reload_releases_only_attempt_and_keeps_same_old_model(self):
        old = self.load()
        self.owner.before_load(self.name, 'cuda', self.snapshot, 1)
        self.owner.complete(self.snapshot, self.instances)
        self.assertEqual(self.store.snapshot(), (old,))

    def test_failed_cold_load_releases_only_its_own_known_ended_attempt(self):
        foreign = self.reserve()
        self.owner.before_load(self.name, 'cuda', self.snapshot, None)
        self.owner.complete(self.snapshot, self.instances)
        self.assertEqual(self.store.snapshot(), (foreign,))

    def test_new_instance_without_exact_proof_remains_unknown_until_confirmed_drop(self):
        for delta in ({'verified_artifacts': {}},
                      {'verified_artifacts': {self.name: 'wrong'}}, {'device': 'cpu'}):
            with self.subTest(delta=delta):
                foreign = self.reserve()
                self.owner.before_load(self.name, 'cuda', self.snapshot, None)
                self.snapshot.update({'model_epoch': 1, 'loaded_models': [self.name],
                    'verified_artifacts': {self.name: self.profile.artifact_fingerprint}, **delta})
                self.instances[self.name] = 1
                with self.assertRaises(AdmissionError):
                    self.owner.complete(self.snapshot, self.instances)
                owned = [t for t in self.store.snapshot() if t.owner == OWNER]
                self.assertEqual(len(owned), 1)
                self.assertEqual(owned[0].phase, 'unknown')
                self.assertIn(foreign, self.store.snapshot())
                # An actual serial drop, not a health observation, proves end.
                self.snapshot.update(model_epoch=2, loaded_models=[], verified_artifacts={}, device='cuda')
                self.owner.complete(self.snapshot, {})
                self.assertEqual(self.store.snapshot(), (foreign,))
                self.store.release(foreign.operation_id, owner=foreign.owner,
                    generation=foreign.generation, confirmed_terminated=True)

    def test_eviction_oom_drop_and_stop_release_owned_residents_not_requests(self):
        self.load()
        foreign = self.reserve()
        self.snapshot.update(model_epoch=2, loaded_models=[], verified_artifacts={})
        self.owner.complete(self.snapshot, {})
        self.assertEqual(self.store.snapshot(), (foreign,))

    def test_targeted_unload_releases_only_target_and_rotates_remaining_resident(self):
        import main
        self.load()
        other = replace(self.profile, deployment_id="other-deployment", reference="other")
        self.owner.profiles[other.reference, "cuda"] = other
        self.owner.before_load(other.reference, "cuda", self.snapshot, None)
        self.snapshot["loaded_models"].append(other.reference)
        self.snapshot["verified_artifacts"][other.reference] = other.artifact_fingerprint
        self.snapshot["model_epoch"] += 1
        self.instances[other.reference] = 2
        self.owner.complete(self.snapshot, self.instances)
        foreign = self.reserve()
        previous_other = self.owner.residents[other.reference]
        manager = main.ModelManager()
        manager.set_worker_thread()
        manager.device = "cuda"
        manager._model_epoch = self.snapshot["model_epoch"]
        for name in self.snapshot["loaded_models"]:
            model = object()
            manager.models[name] = model
            manager._artifact_proofs[name] = (id(model), SimpleNamespace(fingerprint=self.profile.artifact_fingerprint))
        manager.current_model_name = self.name
        manager.model = manager.models[self.name]
        manager.residency = self.owner
        with mock.patch.object(main.torch.cuda, "synchronize"), mock.patch.object(main.torch.cuda, "empty_cache"):
            self.assertTrue(manager._unload_model_sync(self.name))
        manager.residency_boundary()
        self.assertEqual(list(manager.models), [other.reference])
        self.assertNotIn(self.name, self.owner.residents)
        current_other = self.owner.residents[other.reference]
        self.assertEqual(current_other.operation_id, previous_other.operation_id)
        self.assertNotEqual(current_other.generation, previous_other.generation)
        self.assertEqual(current_other.generation, self.key(manager.snapshot()))
        self.assertEqual(set(self.store.snapshot()), {foreign, current_other})

    def test_measured_device_switch_uses_new_profile_and_actual_generation(self):
        old = self.load()
        cpu = replace(self.profile, device='cpu', gpu_bytes=0)
        self.owner.profiles[self.name, 'cpu'] = cpu
        self.owner.before_load(self.name, 'cpu', self.snapshot, 1)
        self.snapshot.update(model_epoch=3, device='cpu')
        self.instances[self.name] = 2
        self.gpu = 0
        self.owner.complete(self.snapshot, self.instances)
        current = self.store.snapshot()[0]
        self.assertEqual(current.gpu_bytes, 0)
        self.assertEqual(current.generation, self.key(self.snapshot))
        self.assertNotEqual(current.operation_id, old.operation_id)

    def test_unmeasured_device_change_is_closed_and_does_not_adopt(self):
        self.load()
        with self.assertRaises(AdmissionError):
            self.owner.before_load(self.name, 'cpu', self.snapshot, 1)
        self.assertEqual(self.store.snapshot()[0].phase, 'unknown')
        self.assertEqual(self.owner.pending, {})

    def test_heartbeat_requires_same_healthy_identity_and_fresh_measurement(self):
        self.load()
        self.now += 31
        self.owner.heartbeat(self.snapshot)
        self.assertEqual(self.store.snapshot()[0].heartbeat_monotonic, self.now)
        self.now += 31
        self.snapshot['model_epoch'] += 1
        with self.assertRaises(AdmissionError):
            self.owner.heartbeat(self.snapshot)
        self.assertEqual(self.store.snapshot()[0].phase, 'unknown')

    def test_timeout_or_resource_overrun_never_confirms_loaded(self):
        for field in ('time', 'gpu', 'rss'):
            with self.subTest(field=field):
                self.owner.before_load(self.name, 'cuda', self.snapshot, None)
                self.snapshot.update(model_epoch=1, loaded_models=[self.name],
                                     verified_artifacts={self.name: self.profile.artifact_fingerprint})
                self.instances[self.name] = 1
                if field == 'time': self.now += 181
                if field == 'gpu': self.gpu += 4 << 30
                if field == 'rss': self.rss += 5 << 30
                with self.assertRaises(AdmissionError):
                    self.owner.complete(self.snapshot, self.instances)
                self.assertTrue(all(t.phase == 'unknown' for t in self.store.snapshot()))
                for ticket in self.store.snapshot():
                    self.store.release(ticket.operation_id, owner=ticket.owner, generation=ticket.generation,
                                       confirmed_terminated=True)
                self.owner.pending.clear()
                self.owner.residents.clear()
                self.owner.failed = False
                self.snapshot.update(model_epoch=0, loaded_models=[], verified_artifacts={})

    def test_measurement_failure_marks_only_owned_work_unknown(self):
        self.load()
        foreign = self.reserve()
        self.owner.measure = mock.Mock(side_effect=AdmissionError('resource_unknown', 'fixture'))
        with self.assertRaises(AdmissionError):
            self.owner.complete(self.snapshot, self.instances)
        self.assertIn(foreign, self.store.snapshot())
        self.assertEqual(self.store.snapshot()[0].phase, 'unknown')

    def test_expired_ticket_cannot_be_renewed_or_replaced(self):
        self.load()
        self.now += 301
        with self.assertRaises(AdmissionError):
            self.owner.heartbeat(self.snapshot)
        with self.assertRaises(AdmissionError):
            self.owner.before_load(self.name, 'cuda', self.snapshot, 1)
        self.assertEqual(len(self.store.snapshot()), 1)

    def test_unprofiled_native_model_gets_no_ticket(self):
        self.owner.before_load('unprofiled', 'cuda', self.snapshot, None)
        self.owner.complete(self.snapshot, {})
        self.assertEqual(self.store.snapshot(), ())


class ServiceIdentityTests(unittest.TestCase):
    def test_revision_binds_owner_and_its_common_policy_store_dependencies(self):
        from kiron_common.local_inference.embedding_native import service_revision
        root = Path(__file__).parent
        original = Path.read_bytes
        baseline = service_revision(root, versions={'fixture': '1'})
        for suffix in ('residency.py', 'gpu_admission/store.py',
                       'gpu_admission/__init__.py', 'prism_runtime_policy.py'):
            with self.subTest(source=suffix):
                def changed(path):
                    value = original(path)
                    return value + b'\n# source mutation\n' if str(path).endswith(suffix) else value
                with mock.patch.object(Path, 'read_bytes', changed):
                    self.assertNotEqual(service_revision(root, versions={'fixture': '1'}), baseline)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.path = self.root / 'policy.json'
        dep = build_resolver_snapshot(MODEL_CATALOG, ()).resolve('kiron-mankei-dense-v1.query').deployment
        self.row = dict(deployment_id=dep.id, artifact_fingerprint=dep.artifact_identity.fingerprint,
            configuration_fingerprint=dep.configuration_fingerprint, device='cuda', gpu_bytes=3 << 30,
            host_bytes=4 << 30, headroom_bytes=256 << 20, load_timeout=180)
        self.patch = mock.patch('residency.immutable_path', side_effect=lambda path: immutable_path(path, anchor=self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, profiles):
        self.path.write_text(json.dumps({'version': 1, 'profiles': profiles}))
        self.path.chmod(0o640)

    def test_exact_catalog_bound_profile_and_absence(self):
        self.assertEqual(load_profiles(self.path, MODEL_CATALOG), {})
        self.write([self.row])
        self.assertEqual(len(load_profiles(self.path, MODEL_CATALOG)), 1)

    def test_schema_identity_bounds_and_device_reject(self):
        for delta in ({'host_bytes': True}, {'gpu_bytes': 0}, {'load_timeout': 181},
                      {'device': 'automatic'}, {'artifact_fingerprint': '0' * 64},
                      {'configuration_fingerprint': '0' * 64}, {'new': 0}):
            with self.subTest(delta=delta):
                self.write([{**self.row, **delta}])
                with self.assertRaises((ValueError, TypeError)):
                    load_profiles(self.path, MODEL_CATALOG)

    def test_policy_rejects_writable_duplicate_and_symlink(self):
        self.write([self.row])
        self.path.chmod(0o660)
        with self.assertRaises(ValueError): load_profiles(self.path, MODEL_CATALOG)
        self.write([self.row, self.row])
        with self.assertRaises(ValueError): load_profiles(self.path, MODEL_CATALOG)
        self.path.unlink()
        self.path.symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError): load_profiles(self.path, MODEL_CATALOG)


class WorkerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_real_manager_rejects_before_model_or_lru_access(self):
        import main
        manager = main.ModelManager()
        manager.set_worker_thread()
        model = object()
        manager.models['fixture'] = model
        manager._model_epoch = 3
        manager._artifact_proofs['fixture'] = (id(model), SimpleNamespace(fingerprint='measured'))
        with mock.patch.object(manager, '_cached_model_unlocked') as touch:
            with self.assertRaises(StaleGenerationError):
                manager._encode_sync('fixture', ['x'], None, expected_artifact='measured', expected_epoch=2)
            with self.assertRaises(StaleGenerationError):
                manager._encode_sync('fixture', ['x'], None, expected_artifact='foreign', expected_epoch=3)
            touch.assert_not_called()
        self.assertEqual(list(manager.models), ['fixture'])

    async def test_actual_worker_finishes_residency_before_future_and_confirms_stopdrop(self):
        calls = []
        manager = SimpleNamespace(set_worker_thread=lambda: None, clear_loading_for_crash=lambda: None,
            _ensure_model_sync=lambda name, *, load_parent=None: calls.append('load'),
            _drop_model_sync=lambda: calls.append('drop'),
            residency_boundary=lambda: calls.append('reconcile'), residency_idle=lambda: None)
        worker = SerialModelWorker(manager)
        worker.start()
        self.addAsyncCleanup(worker.stop)
        await worker.load('fixture')
        self.assertEqual(calls, ['load', 'reconcile'])
        await worker.stop()
        self.assertEqual(calls, ['load', 'reconcile', 'drop', 'reconcile'])

    async def test_failed_stopdrop_marks_unknown_and_never_confirms_end(self):
        calls = []
        def bad_drop():
            raise RuntimeError('fixture drop failure')
        manager = SimpleNamespace(set_worker_thread=lambda: None, clear_loading_for_crash=lambda: None,
            _drop_model_sync=bad_drop, residency_boundary=lambda: calls.append('reconcile'),
            residency_unknown=lambda: calls.append('unknown'), residency_idle=lambda: None)
        worker = SerialModelWorker(manager)
        worker.start()
        await worker.stop()
        self.assertEqual(calls, ['unknown'])
