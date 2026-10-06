"""Offline launch, cleanup and candidate guards; never start Docker or a model."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner, probe = module('smoke-runtime'), module('probe-runtime')


class GuardTests(unittest.TestCase):
    def test_production_implementation_requires_the_exact_compat_repo_digest(self):
        digest = 'ollama/ollama@sha256:' + 'c' * 64
        plan = {'image': {'Id': 'sha256:' + 'd' * 64, 'RepoDigests': [digest]}}
        compat = {'image_digest': digest, 'report_status': 'passed'}
        value = probe.measured_implementation(plan, compat, 'parser')
        self.assertEqual(value.provider_revision, digest)
        self.assertIsNone(value.template_revision)
        for changed in ({**compat, 'image_digest': plan['image']['Id']},
                        {**compat, 'image_digest': digest + 'x'},
                        {**compat, 'report_status': 'failed'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                probe.measured_implementation(plan, changed, 'parser')

    def test_context_fixtures_keep_the_short_latest_message_and_older_history(self):
        single, history = probe.context_messages(False), probe.context_messages(True)
        self.assertEqual(len(single), 1)
        self.assertGreater(len(single[0]['content']), 10000)
        self.assertEqual(len(history), 33)
        self.assertEqual(history[-1], {'role': 'user', 'content': 'Reply only OK.'})
        self.assertEqual([row['role'] for row in history[:-1]], ['user', 'assistant'] * 16)
        self.assertLess(len(json.dumps(history).encode()), 32 * 1024)

    def test_context_evidence_rejects_truncation_and_missing_eof(self):
        with tempfile.TemporaryDirectory() as name:
            report = Path(name)
            messages = probe.context_messages(False)
            request = {'method': 'POST', 'path': '/api/chat', 'body': {
                'messages': messages, 'stream': False, 'truncate': False, 'shift': False, 'think': False,
                'options': {'num_ctx': 1024, 'num_gpu': 0, 'num_predict': 8}}}
            files = {'native-001.request.json': {'method': 'GET', 'path': '/api/ps', 'body': None},
                'native-001.response.raw': {'models': [{'name': 'qwen3:8b', 'digest': probe.MODEL_SHA,
                    'size_vram': 0, 'context_length': 1024, 'size': 1024}]},
                'native-002.request.json': request, 'native-002.status.json': {'status': 400},
                'native-002.response.raw': {'error': probe.CONTEXT_OVERFLOW}}
            for path, value in files.items():
                (report / path).write_text(json.dumps(value))
            for path in ('native-001.eof', 'native-002.eof'):
                (report / path).touch()
            self.assertTrue(probe.context_trace(report, 0, 2, messages, False)['eof'])
            for key in ('truncate', 'shift'):
                changed = deepcopy(request)
                changed['body'][key] = True
                (report / 'native-002.request.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    probe.context_trace(report, 0, 2, messages, False)
            (report / 'native-002.request.json').write_text(json.dumps(request))
            (report / 'native-002.eof').unlink()
            with self.assertRaisesRegex(ValueError, 'EOF'):
                probe.context_trace(report, 0, 2, messages, False)

    def test_text_contract_accepts_punctuation_but_not_invalid_identity_or_usage(self):
        value = SimpleNamespace(model='public-model', choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content='OK.', tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=21, completion_tokens=3, total_tokens=24))
        probe.check_text(value, 'public-model')
        variants = []
        for field, replacement in (('content', '  '), ('content', None), ('tool_calls', ['unexpected'])):
            changed = deepcopy(value)
            setattr(changed.choices[0].message, field, replacement)
            variants.append(changed)
        for field, replacement in (('prompt_tokens', True), ('completion_tokens', 9), ('total_tokens', 25)):
            changed = deepcopy(value)
            setattr(changed.usage, field, replacement)
            variants.append(changed)
        changed = deepcopy(value)
        changed.model = 'foreign'
        variants.append(changed)
        changed = deepcopy(value)
        changed.choices[0].finish_reason = 'length'
        variants.append(changed)
        for changed in variants:
            with self.subTest(value=changed), self.assertRaises(ValueError):
                probe.check_text(changed, 'public-model')

    def test_native_container_requires_only_repeated_primary_gid_and_reports_observation(self):
        def read(path, *args, **kwargs):
            values = {'status': 'Uid:\t65534 65534 65534 65534\nGid:\t982 982 982 982\n'
                       'Groups:\t' + groups + '\nNoNewPrivs:\t1\nCapEff:\t0000000000000000\n',
                      'mountinfo': '30 20 8:2 /volume /models ro,relatime master:1 - ext4 /dev/sda2 rw\n',
                      'cgroup': '0::/fixture\n', 'cpu.max': '200000 100000',
                      'memory.max': str(runner.RAM_LIMIT), 'memory.swap.max': '0',
                      'pids.max': '128',
                      'memory.current': '1024', 'memory.peak': '2048'}
            return values[path.name]
        with mock.patch.object(Path, 'read_text', new=read), mock.patch.object(Path, 'glob', return_value=[]), \
                mock.patch.object(Path, 'exists', return_value=False), mock.patch.object(runner, 'available_ram', return_value=12345):
            groups = '982'
            measured = runner.process_limits(123)
            self.assertEqual(measured['groups'], [982])
            self.assertEqual(measured['uid'], 65534)
            self.assertEqual(measured['gid'], 982)
            self.assertEqual(measured['no_new_privs'], 1)
            self.assertEqual(measured['cap_effective'], 0)
            self.assertEqual(measured['cgroup_limits']['pids.max'], '128')
            for groups in ('', '44', '982 44', '982 982'):
                with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, 'privileges differ'):
                    runner.process_limits(123)

    def test_cgroup_max_diagnostics_retain_all_raw_limits_and_never_relax_bounds(self):
        original = {'cpu.max': '200000 100000\n', 'memory.max': str(runner.RAM_LIMIT) + '\n',
                    'memory.swap.max': '0\n', 'pids.max': '128\n'}
        for field in original:
            limits = {**original, field: 'max 100000\n' if field == 'cpu.max' else 'max\n'}
            with self.subTest(field=field), mock.patch.object(Path, 'read_text',
                    new=lambda path: limits[path.name]), self.assertRaises(ValueError) as caught:
                runner.cgroup_limits(Path('/sys/fs/cgroup/fixture'), 123)
            observed = json.loads(str(caught.exception).split(': ', 1)[1])
            self.assertEqual(observed, {'pid': 123, 'cgroup': '/sys/fs/cgroup/fixture',
                'limits': limits, 'unbounded_fields': [field]})
        for field, raw in (('cpu.max', '300000 100000'), ('cpu.max', '200000 0'),
                           ('memory.max', str(runner.RAM_LIMIT + 1)), ('memory.swap.max', '1'),
                           ('pids.max', '129'), ('pids.max', 'broken' * 100)):
            limits = {**original, field: raw}
            with self.subTest(field=field, raw=raw[:20]), mock.patch.object(Path, 'read_text',
                    new=lambda path: limits[path.name]), self.assertRaises(ValueError) as caught:
                runner.cgroup_limits(Path('/sys/fs/cgroup/fixture'), 123)
            observed = json.loads(str(caught.exception).split(': ', 1)[1])
            self.assertLessEqual(len(observed['limits'][field]), 128)
        with mock.patch.object(Path, 'read_text', new=lambda path: original[path.name]):
            self.assertEqual(runner.cgroup_limits(Path('/sys/fs/cgroup/fixture'), 123), original)

    def test_cgroup_read_failure_still_captures_other_limit_files(self):
        def read(path):
            if path.name == 'memory.max':
                raise FileNotFoundError('not included in diagnostic')
            return {'cpu.max': '200000 100000', 'memory.swap.max': '0', 'pids.max': '128'}[path.name]
        with mock.patch.object(Path, 'read_text', new=read), self.assertRaises(ValueError) as caught:
            runner.cgroup_limits(Path('/sys/fs/cgroup/fixture'), 123)
        observed = json.loads(str(caught.exception).split(': ', 1)[1])
        self.assertEqual(set(observed['limits']), {'cpu.max', 'memory.max', 'memory.swap.max', 'pids.max'})
        self.assertEqual(observed['limits']['memory.max'], '<read failed: FileNotFoundError>')

    def test_privilege_rejection_retains_bounded_actual_values_without_relaxing_guard(self):
        status = 'Uid:\t65534\t65534\t65534\t65534\nGid:\t982\t982\t982\t982\n'
        status += 'Groups:\t' + ' '.join(str(group) for group in range(100)) + '\nNoNewPrivs:\t0\nCapEff:\t0000000000000001\n'
        with mock.patch.object(Path, 'read_text', return_value=status), self.assertRaises(ValueError) as caught:
            runner.process_limits(123)
        observed = json.loads(str(caught.exception).split(': ', 1)[1])
        self.assertEqual(observed, {'pid': 123, 'uids': [65534] * 4, 'gids': [982] * 4,
            'groups': list(range(16)), 'group_count': 100, 'no_new_privs': 0, 'cap_effective': '0x1'})

    def test_docker_failures_preserve_bounded_sanitized_stderr_without_command_or_stdout(self):
        for exception in (
                subprocess.CalledProcessError(1, ['docker', 'command-secret'], output='stdout-secret',
                    stderr='\x1b[31mError: invalid tmpfs option\x1b[0m\x00\nAuthorization: Bearer bearer-secret\n'
                           'password="space secret" api_key=key-secret https://' 'user:password@host/error\n' + 'x' * 5000),
                subprocess.TimeoutExpired(['docker', 'command-secret'], 3, output=b'stdout-secret',
                    stderr=b'partial diagnostic\xff token=token-secret')):
            with self.subTest(exception=type(exception).__name__), \
                    mock.patch.object(runner.subprocess, 'run', side_effect=exception), \
                    self.assertRaises(runner.DockerError) as caught:
                runner.docker('create', '--env', 'command-secret')
            value = caught.exception.diagnostic
            self.assertEqual(value['operation'], 'create')
            self.assertLessEqual(len(value['stderr']), 4096)
            self.assertNotIn('\x1b', value['stderr'])
            self.assertNotIn('\x00', value['stderr'])
            for secret in ('command-secret', 'stdout-secret', 'bearer-secret', 'space secret',
                           'key-secret', 'user:password', 'token-secret'):
                self.assertNotIn(secret, str(caught.exception))
            if isinstance(exception, subprocess.TimeoutExpired):
                self.assertEqual(value['timeout_seconds'], 3)
                self.assertIn('partial diagnostic', value['stderr'])
            else:
                self.assertEqual(value['returncode'], 1)
                self.assertIn('invalid tmpfs option', value['stderr'])

    def test_create_failure_is_saved_before_any_native_process_with_cleanup_evidence(self):
        plan = {'scope': 'fixture', 'run_id': 'b' * 32, 'model_sha256': {}, 'source_sha256': {}}
        error = subprocess.CalledProcessError(1, ['docker', 'create'], stderr='Error: fixture create rejected')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner, 'verify', return_value=plan), \
                mock.patch.object(runner, 'model_inventory', return_value={}), \
                mock.patch.object(runner, 'verify_source') as source, \
                mock.patch.object(runner, 'free_port') as port, \
                mock.patch.object(runner.signal, 'signal'), \
                mock.patch.object(runner.subprocess, 'run', side_effect=error), \
                mock.patch.object(runner.subprocess, 'Popen') as spawn:
            report = Path(directory)
            with self.assertRaises(RuntimeError):
                runner.run(report)
            result = json.loads((report / 'result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['docker_error'], {'operation': 'create', 'returncode': 1,
                                                    'stderr': 'Error: fixture create rejected'})
            self.assertTrue(result['port_closed'] and result['sdk_reaped'] and result['source_verified_after'])
            self.assertFalse((report / 'container.cid').exists())
            spawn.assert_not_called()
            source.assert_called_once_with(report, {})
            port.assert_called_once_with()

    def test_time_wait_is_not_a_running_server(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            client.settimeout(2)
            client.connect(('127.0.0.1', port))
            connection, _ = listener.accept()
            connection.close()
            self.assertEqual(client.recv(1), b'')
        # Real server-side TIME_WAIT reproduces the old cleanup false failure.
        with socket.socket() as strict, self.assertRaises(OSError):
            strict.bind(('127.0.0.1', port))
        with mock.patch.object(runner, 'PORT', port):
            runner.free_port()

    def test_foreign_listener_with_or_without_reuseaddr_is_never_contacted(self):
        for reuse in (0, 1):
            with self.subTest(reuse=reuse), socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuse)
                listener.bind(('127.0.0.1', 0))
                listener.listen()
                with mock.patch.object(runner, 'PORT', listener.getsockname()[1]), self.assertRaises(OSError):
                    runner.free_port()
                listener.settimeout(.02)
                with self.assertRaises(TimeoutError):
                    listener.accept()

    def test_snapshot_recheck_rejects_content_file_set_or_permission_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            source = report / 'source'
            relative = ('scripts/ollama/smoke-runtime.py', 'scripts/ollama/probe-runtime.py',
                        'scripts/prism/probe-openai-features.py', 'services/kiron-common/pyproject.toml',
                        'services/kiron-common/kiron_common/__init__.py')
            for name in relative:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n')
                os.chown(path, 0, runner.GID)
                path.chmod(0o440)
            for path in [source, *(p for p in source.rglob('*') if p.is_dir())]:
                os.chown(path, 0, runner.GID)
                path.chmod(0o550)
            expected = runner.inventory(source)
            runner.verify_source(report, expected)
            changed = source / relative[0]
            changed.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                runner.verify_source(report, expected)
            changed.write_text('fixture\n')
            changed.chmod(0o640)
            with self.assertRaisesRegex(ValueError, 'mutable source'):
                runner.verify_source(report, expected)
            changed.chmod(0o440)
            extra = source / 'unlisted.py'
            extra.write_text('fixture\n')
            os.chown(extra, 0, runner.GID)
            extra.chmod(0o440)
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                runner.verify_source(report, expected)

    def launch(self):
        cid, run_id = 'a' * 64, 'b' * 32
        return cid, run_id, {'Id': cid, 'Image': runner.IMAGE,
            'Config': {'Labels': {'kiron.isolated-ollama': run_id}, 'User': '65534:982',
                'Entrypoint': ['/bin/ollama'], 'Cmd': ['serve'], 'Env': [
                    'OLLAMA_HOST=127.0.0.1:18095', 'OLLAMA_MODELS=/models', 'OLLAMA_NUM_PARALLEL=1',
                    'OLLAMA_CONTEXT_LENGTH=1024', 'NVIDIA_VISIBLE_DEVICES=void', 'CUDA_VISIBLE_DEVICES=-1']},
            'HostConfig': {'Privileged': False, 'ReadonlyRootfs': True, 'Runtime': 'runc', 'NetworkMode': 'host',
                'Devices': [], 'DeviceRequests': [], 'GroupAdd': [], 'CapAdd': [], 'NanoCpus': 2_000_000_000,
                'Memory': runner.RAM_LIMIT, 'MemorySwap': runner.RAM_LIMIT,
                'SecurityOpt': ['no-new-privileges'], 'CapDrop': ['ALL'], 'PidsLimit': 128,
                'Tmpfs': {'/tmp': 'rw,noexec,nosuid,nodev,size=64m'}},
            'Mounts': [{'Type': 'bind', 'Source': str(runner.CACHE), 'Destination': '/models', 'RW': False,
                        'Propagation': 'rslave'}],
            'State': {'Running': False, 'Pid': 0}}

    def test_fixed_launch_has_no_pull_gpu_writable_cache_or_extra_groups(self):
        args = runner.container_args(Path('/test/report'), 'b' * 32)
        for value in ('--pull=never', '--runtime=runc', '--read-only', '--cpus=2', '--memory=9g',
                      '--memory-swap=9g', '--cap-drop=ALL', '--security-opt=no-new-privileges'):
            self.assertIn(value, args)
        self.assertEqual(args[-2:], [runner.IMAGE, 'serve'])
        self.assertIn('OLLAMA_HOST=127.0.0.1:18095', args)
        self.assertEqual(args[args.index('--mount') + 1],
            f'type=bind,src={runner.CACHE},dst=/models,readonly,bind-propagation=rslave')
        for forbidden in ('--gpus', '--privileged', '--group-add', '11435', '11434', '11442'):
            self.assertNotIn(forbidden, args)

    def test_full_container_identity_rejects_policy_drift(self):
        cid, run_id, original = self.launch()
        with mock.patch.object(runner, 'docker', return_value=json.dumps([original])):
            self.assertEqual(runner.inspect_owned(cid, run_id), original)
        variants = [('Image', 'other'), ('Id', 'c' * 64)]
        for key, value in variants:
            modified = deepcopy(original)
            modified[key] = value
            with self.subTest(key=key), mock.patch.object(runner, 'docker', return_value=json.dumps([modified])), self.assertRaises(ValueError):
                runner.inspect_owned(cid, run_id)
        for key, value in [('Privileged', True), ('ReadonlyRootfs', False), ('Runtime', 'nvidia'),
                ('NanoCpus', 4_000_000_000), ('Memory', 10 * 1024**3), ('MemorySwap', -1),
                ('Devices', [{'PathOnHost': '/dev/nvidia0'}]), ('DeviceRequests', [{'Count': -1}]),
                ('GroupAdd', ['44']), ('CapAdd', ['SYS_ADMIN']), ('SecurityOpt', []), ('PidsLimit', -1)]:
            modified = deepcopy(original)
            modified['HostConfig'][key] = value
            with self.subTest(key=key), mock.patch.object(runner, 'docker', return_value=json.dumps([modified])), self.assertRaises(ValueError):
                runner.inspect_owned(cid, run_id)
        for key, value in [('Source', '/production'), ('Destination', '/writable'), ('RW', True),
                           ('Propagation', 'rshared'), ('Propagation', 'rprivate'), ('Propagation', None)]:
            modified = deepcopy(original)
            modified['Mounts'][0][key] = value
            with self.subTest(key=key), mock.patch.object(runner, 'docker', return_value=json.dumps([modified])), self.assertRaises(ValueError):
                runner.inspect_owned(cid, run_id)

    def test_effective_model_mount_flags_include_propagated_submounts(self):
        root = '30 20 8:2 /volume /models ro,relatime master:1 - ext4 /dev/sda2 rw\n'
        nested = '31 30 8:3 / /models/blobs ro,relatime master:2 - ext4 /dev/sdb rw\n'
        runner.require_readonly_model_mounts(root)
        runner.require_readonly_model_mounts(root + nested)
        # The superblock may be writable; effective VFS mount flags must be ro.
        for value in ('', nested, root.replace('ro,relatime', 'rw,relatime'),
                      root + nested.replace('ro,relatime', 'rw,relatime'),
                      root + nested.replace('/models/blobs ro', r'/models/new\040mount rw')):
            with self.subTest(mountinfo=value), self.assertRaises(ValueError):
                runner.require_readonly_model_mounts(value)

    def test_preflight_rejects_cache_submounts_without_rejecting_neighbor_paths(self):
        template = '30 20 8:2 / {} ro,relatime master:1 - ext4 /dev/sda2 rw\n'
        runner.require_no_cache_submounts(template.format('/'))
        runner.require_no_cache_submounts(template.format(runner.CACHE))
        runner.require_no_cache_submounts(template.format(str(runner.CACHE) + '-neighbor/blobs'))
        for target in (str(runner.CACHE) + '/blobs', str(runner.CACHE) + r'/sub\040mount'):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'submounts'):
                runner.require_no_cache_submounts(template.format(target))

    def test_foreign_container_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, 'inspect_owned', side_effect=ValueError('foreign')), \
                mock.patch.object(runner, 'docker') as docker, self.assertRaises(ValueError):
            runner.stop_owned(Path(directory), 'a' * 64, 'b' * 32, {})
        docker.assert_not_called()

    def test_log_failure_does_not_prevent_owned_stop_and_removal(self):
        cid, run_id, value = self.launch()
        result = {'status': 'passed'}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, 'inspect_owned', return_value=value), \
                mock.patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('logs', 15)), \
                mock.patch.object(runner, 'docker') as docker:
            runner.stop_owned(Path(directory), cid, run_id, result)
        self.assertTrue(result['container_removed'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(docker.call_args_list, [mock.call('stop', '--time', '10', cid, timeout=20), mock.call('rm', cid)])

    def test_stop_timeout_rechecks_owned_identity_before_container_kill(self):
        cid, run_id, stopped = self.launch()
        running = deepcopy(stopped)
        running['State'] = {'Running': True, 'Pid': 123}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(runner, 'inspect_owned', side_effect=[running, running, stopped]) as inspect, \
                mock.patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout=b'', stderr=b'')), \
                mock.patch.object(runner, 'docker', side_effect=[runner.DockerError('stop', subprocess.TimeoutExpired('stop', 20)), '', '']) as docker:
            result = {}
            runner.stop_owned(Path(directory), cid, run_id, result)
        self.assertEqual(inspect.call_count, 3)
        self.assertEqual(docker.call_args_list[1], mock.call('kill', cid))
        self.assertTrue(result['container_removed'])

    def test_report_paths_and_mutable_symlink_inputs_are_rejected(self):
        for path in (Path('/tmp/cpu-one'), runner.ROOT / 'other', runner.ROOT / '../cpu-one'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.report_path(path)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'file'
            file.write_bytes(b'fixture')
            link = Path(directory) / 'link'
            link.symlink_to(file)
            with self.assertRaises(ValueError):
                runner.regular(link)
            file.chmod(0o666)
            with self.assertRaises(ValueError):
                runner.regular(file)

    def test_model_manifest_mismatch_prevents_any_blob_read(self):
        with mock.patch.object(runner, 'regular'), mock.patch.object(runner, 'sha', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'manifest'):
            runner.model_inventory()

    def test_cpu_residency_rejects_foreign_gpu_context_or_unbounded_size(self):
        valid = {'name': 'qwen3:8b', 'digest': probe.MODEL_SHA, 'size_vram': 0, 'context_length': 1024, 'size': 1024}
        self.assertEqual(probe.resident({'models': [valid]}), valid)
        for key, value in [('name', 'other'), ('digest', '0' * 64), ('size_vram', 1), ('size_vram', False),
                           ('context_length', 2048), ('context_length', True), ('size', 0), ('size', 10 * 1024**3)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.resident({'models': [{**valid, key: value}]})
        with self.assertRaises(ValueError):
            probe.resident({'models': [valid, valid]})

    def test_candidate_controls_and_call_correlation_remain_closed(self):
        from kiron_common.local_inference import CapabilityEvidence, CapabilityName as N
        evidence = CapabilityEvidence('native', 'a'*64, 'b'*64, None, 'template', 'parser', 'c'*64,
                                      'test candidate', datetime.now(timezone.utc))
        caps = probe.capabilities(evidence)
        self.assertEqual(caps.by_name[N.CHAT].constraints['context_tokens'].allowed_values, (1024,))
        self.assertEqual(caps.by_name[N.CHAT].constraints['device'].allowed_values, ('cpu',))
        self.assertEqual(caps.by_name[N.FUNCTION_TOOLS].constraints['tool_choice'].allowed_values, ('none', 'auto'))
        self.assertFalse(caps.by_name[N.FUNCTION_TOOLS].constraints['strict'].accepts(True))
        self.assertEqual(caps.by_name[N.VISION].status.value, 'unverified')
        self.assertEqual(caps.by_name[N.REASONING].status.value, 'unverified')
        def call(id, city):
            return SimpleNamespace(id=id, function=SimpleNamespace(name='lookup_value', arguments=json.dumps({'city': city})))
        completion = SimpleNamespace(choices=[SimpleNamespace(finish_reason='tool_calls',
            message=SimpleNamespace(tool_calls=[call('a', 'Berlin'), call('b', 'Paris')]))])
        self.assertEqual(len(probe.check_calls(completion, {'Berlin', 'Paris'})), 2)
        completion.choices[0].message.tool_calls[1].id = 'a'
        with self.assertRaises(ValueError):
            probe.check_calls(completion, {'Berlin', 'Paris'})


if __name__ == '__main__':
    unittest.main()
