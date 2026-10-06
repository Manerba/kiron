"""Offline controller harness preparation checks; never start a controller/model."""
from dataclasses import dataclass
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock
import httpx


SPEC = importlib.util.spec_from_file_location('controller_smoke', Path(__file__).with_name('smoke-controller.py'))
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
ORIGINAL_WORKSPACE = smoke.WORKSPACE


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.base.chmod(0o755)
        self.report = self.base / 'controller-test'
        patch = mock.patch.object(smoke, 'REPORT_ROOT', self.base)
        patch.start()
        self.addCleanup(patch.stop)

    def fixture(self):
        fixture = self.base / 'fixture'
        fixture.write_bytes(b'not a GGUF, descriptor fixture only')
        controlled = SimpleNamespace(verify_bundle=mock.Mock(),
            open_artifact=mock.Mock(side_effect=lambda *args: os.open(fixture, os.O_RDONLY)),
            verify_metadata=mock.Mock(), profiles={smoke.PROFILE_ID: object()})
        measured = {'gpu_free_bytes': 6 << 30, 'gpu_used_bytes': 5 << 30,
                    'host_available_bytes': 18 << 30, 'monotonic': 1.0}
        self.workspace = self.base / 'workspace'
        files = {'scripts/prism/smoke-controller.py': Path(smoke.__file__).read_text(),
                 'scripts/prism/package-runtime.py': 'pass\n', 'scripts/prism/probe-openai-api.py': 'pass\n',
                 'scripts/prism/probe-openai-features.py': 'pass\n',
                 'scripts/prism/probe-openai-responses.py': 'pass\n',
                 'scripts/prism/probe-controller-faults.py': 'pass\n',
                 'scripts/prism/probe-dashboard-runtime.py': 'pass\n',
                 'scripts/prism/probe-runtime-soak.py': 'pass\n',
                 'scripts/prism/probe-controller-cancel.py': 'pass\n',
                 'scripts/prism/fixtures/red.png': 'synthetic red fixture',
                 'scripts/prism/fixtures/blue.png': 'synthetic blue fixture',
                 'services/kiron-common/pyproject.toml': '[project]\nname="fixture"\n',
                 'services/kiron-common/kiron_common/__init__.py': 'ORIGIN="snapshot"\n',
                 'services/kiron-common/kiron_common/manifests/model.json': '{}\n',
                 'services/kiron-proxy/snapshot_marker.py': 'ORIGIN="snapshot"\n',
                 'services/kiron-proxy/static/js/tab_models.js': '// fixture\n',
                 'services/kiron-proxy/static/css/models.css': '/* fixture */\n',
                 'services/kiron-proxy/templates/index.html': '<html></html>\n',
                 'services/kiron-proxy/test_excluded.py': 'raise AssertionError("never snapshot tests")\n',
                 'services/kiron-proxy/requirements-contract.txt': 'fixture==1\n',
                 'services/kiron-prism/controller.py': 'pass\n',
                 'services/kiron-prism/requirements.txt': 'fixture==1\n',
                 'data/private.json': 'must not enter source snapshot'}
        for relative, content in files.items():
            path = self.workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for name, value in [('policy', mock.Mock(return_value=controlled)), ('imports', mock.Mock()),
                            ('WORKSPACE', self.workspace), ('free_port', mock.Mock()),
                            ('memory', mock.Mock(return_value=measured))]:
            patch = mock.patch.object(smoke, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        return controlled, measured

    def test_report_target_must_be_isolated_and_canonical(self):
        self.assertEqual(smoke.report_path(self.report), self.report)
        for path in (Path('/run/kiron/controller-prod'), self.base / '..', self.base / 'not-controller'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                smoke.report_path(path)
        link = self.base / 'controller-link'
        link.symlink_to(self.base)
        with self.assertRaises(ValueError):
            smoke.report_path(link)

    def test_prepare_creates_root_owned_readonly_registry_and_separate_state(self):
        controlled, _ = self.fixture()
        plan = smoke.prepare(self.report)
        controlled.verify_bundle.assert_called_once()
        controlled.verify_metadata.assert_called_once()
        for name in ('registry/models.json', 'registry/models.json.lock'):
            info = (self.report / name).stat()
            self.assertEqual((info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)), (0, 982, 0o640))
        self.assertEqual(stat.S_IMODE((self.report / 'admission').stat().st_mode), 0o2770)
        self.assertEqual((self.report / 'uds').stat().st_uid, pwd.getpwnam('nobody').pw_uid)
        self.assertEqual(plan['port'], 18089)
        self.assertEqual(plan['model_sha256'], smoke.MODEL_SHA)
        self.assertNotIn('/run/kiron', json.dumps(plan))
        self.assertEqual(smoke.verify_plan(self.report)['model_id'], smoke.entry().id)
        self.assertEqual(list((self.report / 'admission').iterdir()), [])
        self.assertEqual(list((self.report / 'results').iterdir()), [])

    def test_production_artifact_layout_binds_paths_without_production_state(self):
        controlled, _ = self.fixture()
        plan = smoke.prepare(self.report, layout='production')
        self.assertEqual(plan['projector_path'], str(smoke.MODEL.parent / 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'))
        self.assertEqual(plan['runtime_root'], '/usr/lib/kiron/runtimes/prism/9a9394a-sm86-tokens-v1')
        self.assertEqual(plan['port'], 18089)
        self.assertEqual(smoke.verify_plan(self.report)['artifact_layout'], 'production')
        controlled.open_artifact.assert_any_call(plan['projector_path'], smoke.PROJECTOR_SHA, smoke.PROJECTOR_BYTES)
        changed = dict(plan, projector_path='/tmp/foreign.gguf')
        (self.report / 'plan.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'fixture changed'):
            smoke.verify_plan(self.report)
        changed = dict(plan, artifact_layout='arbitrary')
        (self.report / 'plan.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'artifact layout'):
            smoke.verify_plan(self.report)

    def test_unknown_artifact_layout_rejected_before_preparation(self):
        with self.assertRaisesRegex(ValueError, 'artifact layout'):
            smoke.prepare(self.report, layout='/tmp/runtime')
        self.assertFalse(self.report.exists())

    def test_fixture_registry_is_readable_as_nobody_but_not_writable(self):
        self.fixture()
        smoke.prepare(self.report)
        code = '''
from pathlib import Path
import os,sys
from kiron_common.local_model_registry import RuntimeModelRegistry,RegistryFilePolicy
p=Path(sys.argv[1])
entries=RuntimeModelRegistry(p,readonly=True,file_policy=RegistryFilePolicy(0,982)).list()
assert len(entries)==1 and entries[0].sha256==sys.argv[2]
assert not os.access(p,os.W_OK)
'''
        result = subprocess.run([sys.executable, '-c', code, str(self.report / 'registry/models.json'), smoke.MODEL_SHA],
            capture_output=True, text=True, timeout=10, user='nobody', group=982, extra_groups=[],
            env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1',
                 'PYTHONPATH': str(ORIGINAL_WORKSPACE / 'services/kiron-common')})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_insufficient_memory_prevents_fixture_publication(self):
        _, measured = self.fixture()
        measured['gpu_free_bytes'] = 5 << 30  # less than 4608+512 MiB? exactly boundary passes
        measured['gpu_free_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            smoke.prepare(self.report)
        self.assertFalse(self.report.exists())

    def test_artifact_failure_closes_already_open_fd_and_creates_no_report(self):
        controlled, _ = self.fixture()
        controlled.open_artifact.side_effect = [os.open(__file__, os.O_RDONLY), ValueError('wrong projector')]
        with mock.patch.object(smoke.os, 'close', wraps=os.close) as close:
            with self.assertRaisesRegex(ValueError, 'wrong projector'):
                smoke.prepare(self.report)
            self.assertEqual(close.call_count, 1)
        self.assertFalse(self.report.exists())

    def test_workspace_drift_is_irrelevant_but_snapshot_registry_and_result_drift_fail_closed(self):
        self.fixture()
        smoke.prepare(self.report)
        with self.assertRaisesRegex(ValueError, 'never overwrite'):
            smoke.prepare(self.report)
        relative = 'services/kiron-proxy/snapshot_marker.py'
        (self.workspace / relative).write_text('ORIGIN="later development"\n')
        smoke.verify_plan(self.report)
        pinned = self.report / 'source' / relative
        original = pinned.read_bytes()
        pinned.write_bytes(b'changed snapshot\n')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            smoke.verify_plan(self.report)
        pinned.write_bytes(original)
        (self.report / 'results/old.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'never rerun'):
            smoke.verify_plan(self.report)
        (self.report / 'results/old.json').unlink()
        (self.report / 'registry/models.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            smoke.verify_plan(self.report)

    def test_resource_guard_rejects_high_gpu_or_low_ram(self):
        smoke.check_memory({'gpu_used_bytes': smoke.MAX_GPU, 'host_available_bytes': smoke.MIN_RAM})
        for values in ({'gpu_used_bytes': smoke.MAX_GPU + 1, 'host_available_bytes': smoke.MIN_RAM},
                       {'gpu_used_bytes': 0, 'host_available_bytes': smoke.MIN_RAM - 1}):
            with self.assertRaisesRegex(ValueError, 'resource abort'):
                smoke.check_memory(values)

    def test_immutable_common_values_serialize_without_deepcopy(self):
        @dataclass(frozen=True)
        class Value:
            models: object
        self.assertEqual(smoke.report_value(Value(MappingProxyType({'a': (1, 2)}))), {'models': {'a': [1, 2]}})

    def test_prepare_requires_root_before_any_artifact_access(self):
        controlled, _ = self.fixture()
        with mock.patch.object(smoke.os, 'geteuid', return_value=65534):
            with self.assertRaisesRegex(ValueError, 'requires root'):
                smoke.prepare(self.report)
        controlled.verify_bundle.assert_not_called()

    def test_public_api_probe_is_pinned_in_the_reviewable_plan(self):
        self.fixture()
        plan = smoke.prepare(self.report, 'public-api')
        self.assertEqual(plan['probe'], 'public-api')
        self.assertEqual(smoke.verify_plan(self.report)['probe'], 'public-api')
        changed = dict(plan, probe='arbitrary')
        (self.report / 'plan.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'changed'):
            smoke.verify_plan(self.report)

    def test_feature_probes_share_pinned_source_and_finite_workflow_budget(self):
        self.fixture()
        for kind in smoke.FEATURE_PROBES:
            report = self.base / ('controller-' + kind)
            with self.subTest(kind=kind):
                plan = smoke.prepare(report, kind)
                self.assertEqual(smoke.verify_plan(report)['probe'], kind)
                self.assertEqual(plan['max_seconds'], 600)
                self.assertEqual(plan['source_root'], str(report / 'source'))
                self.assertIn('scripts/prism/probe-openai-features.py', plan['source_sha256'])
                self.assertIn('scripts/prism/probe-openai-responses.py', plan['source_sha256'])
                self.assertIn('scripts/prism/probe-controller-faults.py', plan['source_sha256'])
                self.assertIn('services/kiron-common/kiron_common/manifests/model.json', plan['source_sha256'])
                self.assertIn('services/kiron-proxy/requirements-contract.txt', plan['source_sha256'])
                self.assertNotIn('data/private.json', plan['source_sha256'])
                self.assertNotIn('services/kiron-proxy/test_excluded.py', plan['source_sha256'])
                self.assertEqual(stat.S_IMODE((report / 'plan.json').stat().st_mode), 0o440)
                module = SimpleNamespace(capabilities=mock.Mock(return_value='feature candidate'))
                self.assertEqual(smoke.candidate_capabilities(kind, 'evidence', module), 'feature candidate')
                module.capabilities.assert_called_once_with(kind, 'evidence')

    def test_snapshot_rejects_writable_files_extra_code_symlinks_and_changed_plan_permissions(self):
        self.fixture()
        smoke.prepare(self.report)
        root = self.report / 'source'
        path = root / 'services/kiron-proxy/snapshot_marker.py'
        path.chmod(0o640)
        with self.assertRaisesRegex(ValueError, 'unsafe immutable source'):
            smoke.verify_plan(self.report)
        path.chmod(0o440)
        extra = root / 'services/kiron-proxy/unpinned.py'
        extra.write_text('pass\n')
        os.chown(extra, 0, 982)
        extra.chmod(0o440)
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            smoke.verify_plan(self.report)
        extra.unlink()
        extra.symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'unsafe immutable source'):
            smoke.verify_plan(self.report)
        extra.unlink()
        (self.report / 'plan.json').chmod(0o640)
        with self.assertRaisesRegex(ValueError, 'immutable smoke plan'):
            smoke.verify_plan(self.report)

    def test_dashboard_snapshot_includes_assets_without_granting_inference_capabilities(self):
        self.fixture()
        plan = smoke.prepare(self.report, 'dashboard-runtime')
        self.assertEqual(smoke.verify_plan(self.report)['probe'], 'dashboard-runtime')
        for relative in ('scripts/prism/probe-dashboard-runtime.py',
                         'services/kiron-proxy/static/js/tab_models.js',
                         'services/kiron-proxy/static/css/models.css',
                         'services/kiron-proxy/templates/index.html'):
            self.assertIn(relative, plan['source_sha256'])
        self.assertNotIn('dashboard-runtime', smoke.FEATURE_PROBES)
        from kiron_common.local_inference import CapabilityName, CapabilityStatus
        capabilities = smoke.candidate_capabilities('dashboard-runtime', None)
        for name in CapabilityName:
            self.assertNotEqual(capabilities.by_name[name].status, CapabilityStatus.SUPPORTED)

    def test_source_symlinks_are_not_copied(self):
        self.fixture()
        path = self.workspace / 'services/kiron-proxy/snapshot_marker.py'
        path.unlink()
        path.symlink_to(self.workspace / 'data/private.json')
        with self.assertRaisesRegex(ValueError, 'canonical regular'):
            smoke.prepare(self.report)
        self.assertFalse((self.report / 'plan.json').exists())

    def test_run_origin_requires_exact_snapshot_and_isolated_interpreter(self):
        self.fixture()
        smoke.prepare(self.report)
        with self.assertRaisesRegex(ValueError, 'run only the prepared'):
            smoke.require_snapshot_execution(self.report)
        script = self.report / 'source/scripts/prism/smoke-controller.py'
        code = '''
import importlib.util,pathlib,sys
script=pathlib.Path(sys.argv[1]); report=pathlib.Path(sys.argv[2])
spec=importlib.util.spec_from_file_location("isolated_harness",script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.require_snapshot_execution(report)
module.imports()
import snapshot_marker,kiron_common
assert snapshot_marker.ORIGIN == kiron_common.ORIGIN == "snapshot"
assert pathlib.Path(snapshot_marker.__file__).is_relative_to(report/"source")
assert pathlib.Path(kiron_common.__file__).is_relative_to(report/"source")
assert not pathlib.Path(sys.argv[3]).exists()
'''
        (self.workspace / 'services/kiron-proxy/snapshot_marker.py').write_text('raise AssertionError("workspace imported")\n')
        forbidden = self.base / 'never-created'
        argv = [sys.executable, '-I', '-B', '-c', code, str(script), str(self.report), str(forbidden)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, user='nobody', group=982,
            extra_groups=[], env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(self.workspace / 'services/kiron-proxy')})
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-B', *argv[3:]], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python -I -B', result.stderr)


class BearerTests(unittest.IsolatedAsyncioTestCase):
    async def test_kernel_records_no_new_privileges_in_fresh_process(self):
        script = Path(smoke.__file__).resolve()
        code = '''
import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location("smoke",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.no_new_privileges()
assert "NoNewPrivs:\\t1" in pathlib.Path("/proc/self/status").read_text()
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(script)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    async def test_feature_callback_receives_pinned_kind_and_isolated_results_directory(self):
        for kind in smoke.PROBES[1:]:
            module = SimpleNamespace(probe=mock.AsyncMock(return_value={'passed': kind}))
            report = Path('/isolated/controller-case')
            result = await smoke.public_probe(kind, module, 'service', 'model', 'store', report,
                                               controller='controller', provider='provider')
            self.assertEqual(result, {'passed': kind})
            if kind in smoke.CONTROLLER_PROBES:
                module.probe.assert_awaited_once_with('service', 'model', 'store', kind,
                    controller='controller', provider='provider', report=report / 'results')
            elif kind in smoke.FEATURE_PROBES or kind == 'dashboard-runtime':
                module.probe.assert_awaited_once_with('service', 'model', 'store', kind, report=report / 'results')
            else:
                module.probe.assert_awaited_once_with('service', 'model', 'store', report=report / 'results')

    async def test_cleanup_wait_is_bounded_for_cancellation_resistant_task(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        async def stubborn():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await finish.wait()

        task = asyncio.create_task(stubborn())
        await entered.wait()
        task.cancel()
        start = time.monotonic()
        pending = await smoke.settle_tasks((task,), 0.02)
        self.assertEqual(pending, {task})
        self.assertLess(time.monotonic() - start, 0.5)
        finish.set()
        await task

    async def test_wrong_and_foreign_generation_are_401_and_do_not_change_native_slot(self):
        smoke.imports()
        from provider_transport import bounded_request
        values = {'state': 'loaded', 'active_requests': 0, 'slot_task_id': None,
                  'generation': {'boot_id': 'boot', 'process_id': 'current'}, 'backend_model': 'kiron-prism-current'}
        seen = []

        def responder(request):
            if request.url.path == '/health':
                return httpx.Response(200, json=values)
            seen.append(request.headers['Authorization'])
            return httpx.Response(401, json={'error': 'unauthorized'})

        async with httpx.AsyncClient(base_url='http://127.0.0.1', trust_env=False,
                                    transport=httpx.MockTransport(responder)) as client:
            result = await smoke.verify_foreign_bearer(client, client)
        self.assertEqual(result['statuses'], [401, 401])
        self.assertEqual(len(set(seen)), 2)
        self.assertNotIn('Bearer kiron-prism-current', seen)

    async def test_unexpected_acceptance_or_slot_change_fails_the_probe(self):
        smoke.imports()
        for status, changed in ((200, False), (401, True)):
            with self.subTest(status=status, changed=changed):
                count = 0

                def responder(request):
                    nonlocal count
                    if request.url.path == '/health':
                        count += 1
                        return httpx.Response(200, json={'state': 'loaded', 'active_requests': 0,
                            'slot_task_id': 1 if changed and count > 1 else None,
                            'generation': {'boot_id': 'boot', 'process_id': 'current'}, 'backend_model': 'kiron-prism-current'})
                    return httpx.Response(status, json={})

                async with httpx.AsyncClient(base_url='http://127.0.0.1', trust_env=False,
                                            transport=httpx.MockTransport(responder)) as client:
                    with self.assertRaises(ValueError):
                        await smoke.verify_foreign_bearer(client, client)


if __name__ == '__main__':
    unittest.main()
