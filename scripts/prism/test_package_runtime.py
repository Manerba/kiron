"""Offline package fixtures; no downloads, model execution or production writes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('prism_package', Path(__file__).with_name('package-runtime.py'))
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def test_fd_copy_hash_rejection_and_special_file_rejection(self):
        source, target = self.base / 'source', self.base / 'target'
        source.write_bytes(b'pinned content')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            package.fingerprint(source, '0' * 64, target)
        link = self.base / 'link'
        link.symlink_to(source)
        with self.assertRaises(OSError):
            package.fingerprint(link)
        pipe = self.base / 'pipe'
        os.mkfifo(pipe)
        with self.assertRaisesRegex(ValueError, 'input type'):
            package.fingerprint(pipe)

    def test_flat_link_closure_rejects_escape_missing_and_cycle(self):
        cases = [
            {'../outside': {'sha256': 'a'}},
            {'a': {'link': '/etc/passwd'}},
            {'a': {'link': '../other'}},
            {'a': {'link': 'absent'}},
            {'a': {'link': 'b'}, 'b': {'link': 'a'}},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                package.check_links(value)
        package.check_links({'a': {'link': 'b'}, 'b': {'link': 'c'}, 'c': {'sha256': 'a'}})

    def elf(self, name='llama-server'):
        path = self.base / name
        path.write_bytes(b'\x7fELFfixture')
        return path

    def test_missing_bundle_dependency_rejects_before_loader_even_if_host_has_it(self):
        self.elf()
        output = '(NEEDED) Shared library: [libcudart.so.12]\n'
        with mock.patch.object(package, 'command', return_value=output) as command:
            with self.assertRaisesRegex(ValueError, 'missing bundled dependency'):
                package.audit_dependencies(self.base)
        self.assertEqual(command.call_count, 1)
        self.assertEqual(command.call_args.args[0][0], '/usr/bin/readelf')

    def test_absolute_and_empty_runpath_rejected_origin_allowed(self):
        self.elf()
        for path in ('/usr/lib/kiron/test-venvs/fixture', '', '$ORIGIN:/tmp'):
            with self.subTest(path=path), mock.patch.object(package, 'command', return_value=f'(RUNPATH) [{path}]'):
                with self.assertRaisesRegex(ValueError, 'nonrelocatable'):
                    package.audit_dependencies(self.base, execute=False)
        with mock.patch.object(package, 'command', return_value='(RUNPATH) [$ORIGIN]'):
            self.assertIn('llama-server', package.audit_dependencies(self.base, execute=False)['dynamic'])

    def test_loader_resolution_into_testvenv_rejected(self):
        self.elf()
        outside = self.base / 'outside'
        outside.mkdir()
        target = outside / 'libcudart.so.12'
        target.write_bytes(b'fixture')
        self.elf('libcudart.so.12')

        def run(args, **kwargs):
            if args[0] == '/usr/bin/readelf':
                return '(NEEDED) [libcudart.so.12]'
            return f'libcudart.so.12 => {target} (0x1234)'

        # outside must be beyond the runtime root, as with a test-venv fallback.
        runtime = self.base / 'runtime'
        runtime.mkdir()
        for name in ('llama-server', 'libcudart.so.12'):
            (self.base / name).rename(runtime / name)
        with mock.patch.object(package, 'command', side_effect=run):
            with self.assertRaisesRegex(ValueError, 'escaped bundle'):
                package.audit_dependencies(runtime)

    def test_commands_have_finite_timeout_clean_environment_and_unprivileged_identity(self):
        with mock.patch.object(package.subprocess, 'run', return_value=mock.Mock(returncode=0, stdout='ok', stderr='')) as run:
            with mock.patch.dict(os.environ, {'LLAMA_ARG_TOOLS': 'shell', 'LD_PRELOAD': '/bad'}):
                package.command(['/bin/true'], library_path=self.base, unprivileged=True)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs['env'], {**package.ENV, 'LD_LIBRARY_PATH': str(self.base)})
        self.assertEqual(kwargs['extra_groups'], [])
        self.assertNotEqual(kwargs['user'], 0)
        self.assertEqual(kwargs['timeout'], 20)

    def fixture_package(self):
        build, prepared, wheel = (self.base / name for name in ('build', 'prepared', 'wheels'))
        (build / 'build/bin').mkdir(parents=True)
        prepared.mkdir()
        wheel.mkdir()
        output_root = self.base / 'bundles'
        output_root.mkdir()
        manifest = {'llama-server': {'sha256': digest(b'\x7fELFserver')},
                    'libfixture.so.1': {'sha256': digest(b'\x7fELFlibrary')},
                    'libfixture.so': {'link': 'libfixture.so.1'}}
        for name, data in (('llama-server', b'\x7fELFserver'), ('libfixture.so.1', b'\x7fELFlibrary')):
            (build / 'build/bin' / name).write_bytes(data)
        (build / 'build/bin/libfixture.so').symlink_to('libfixture.so.1')
        source_names = ['source/LICENSE', 'source/licenses/LICENSE-jsonhpp', 'source/vendor/cpp-httplib/LICENSE',
                        'source/vendor/hash/xxhash/LICENSE', 'source/vendor/hash/sha256/LICENSE',
                        'source/vendor/hash/rotate-bits/LICENSE.md', 'cuda/cudart-LICENSE',
                        'cuda/lib/libcudart.so.12.8.90']
        inventory = {}
        for name in source_names:
            target = prepared / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'fixture input ' + name.encode())
            inventory[name] = {'sha256': digest(target.read_bytes())}
        license_path = wheel / 'nvidia_cublas_cu12-12.8.4.1.dist-info/License.txt'
        license_path.parent.mkdir()
        license_path.write_bytes(b'NVIDIA fixture')
        tool = self.base / 'patchelf'
        tool.write_bytes(b'fixture tool')
        result = {'status': 'complete', 'lock_sha256': package.LOCK_SHA}

        def metadata(path, expected):
            if path.name == 'source-toolchain-lock.json':
                return {'artifacts': {'cudart': {}}, 'cublas_wheel': {}, 'source': {'patches': []}}
            return {'prepared.json': {'inventory': inventory}, 'binary-manifest.json': manifest,
                    'result.json': result, 'launch.json': {}}[path.name]

        patches = {
            'OUTPUT_ROOT': output_root, 'BUILD': build, 'PREPARED': prepared, 'WHEEL_ROOT': wheel,
            'TOOL_SHA': digest(tool.read_bytes()), 'CUBLAS_PINS': {},
            'CUBLAS_LICENSE_SHA': digest(b'NVIDIA fixture'),
        }
        for key, value in patches.items():
            patch = mock.patch.object(package, key, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(package, 'pinned_json', side_effect=metadata)
        patch.start()
        self.addCleanup(patch.stop)
        # Temp paths need not satisfy the production / ancestors ownership gate.
        original = package.immutable
        patch = mock.patch.object(package, 'immutable', side_effect=lambda path, **kw: original(path))
        patch.start()
        self.addCleanup(patch.stop)
        return output_root / 'candidate', tool, build

    def test_atomic_package_manifest_reuse_and_drift_detection(self):
        output, tool, build = self.fixture_package()
        verification = {'version': '9a9394a fixture'}
        with mock.patch.object(package, 'command', return_value='') as commands, \
                mock.patch.object(package, 'audit_dependencies', return_value=verification):
            package.package(output, tool)
            before = (output / 'runtime/llama-server').stat().st_mtime_ns
            package.package(output, tool)
            self.assertEqual((output / 'runtime/llama-server').stat().st_mtime_ns, before)
            self.assertEqual(commands.call_count, 2)  # only two regular copied Prism ELFs
            self.assertTrue(all(call.args[0][1] == '--remove-rpath' for call in commands.call_args_list))
            self.assertEqual((build / 'build/bin/llama-server').read_bytes(), b'\x7fELFserver')
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o555)
            manifest = json.loads((output / 'bundle-manifest.json').read_text())
            self.assertEqual(manifest['libfixture.so'], {'link': 'libfixture.so.1'})
            (output / 'runtime/llama-server').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                package.verify(output)

    def test_failed_dependency_verification_leaves_no_published_or_staged_bundle(self):
        output, tool, _ = self.fixture_package()
        with mock.patch.object(package, 'command', return_value=''), \
                mock.patch.object(package, 'audit_dependencies', side_effect=ValueError('missing library')):
            with self.assertRaisesRegex(ValueError, 'missing library'):
                package.package(output, tool)
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob('.bundle-stage-*')), [])

    def test_changed_input_rejected_before_publish_and_production_target_forbidden(self):
        output, tool, build = self.fixture_package()
        (build / 'build/bin/llama-server').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            package.package(output, tool)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'isolated bundle'):
            package.package(Path('/usr/lib/kiron/runtimes/prism/production'), tool)

    def test_low_disk_space_rejected_before_staging(self):
        output, tool, _ = self.fixture_package()
        with mock.patch.object(package.shutil, 'disk_usage', return_value=mock.Mock(free=1024)):
            with self.assertRaisesRegex(ValueError, '3 GiB'):
                package.package(output, tool)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
