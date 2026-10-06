#!/usr/bin/env python3
"""Package the verified sm86 source build into an isolated, relocatable test bundle.

No downloads, model starts, production installs, global linker changes or rebuilds.
Requires the pinned Ubuntu patchelf executable extracted into package-tools; its
106300-byte .deb SHA256 is recorded below. Only copied Prism ELFs are modified.
The NVIDIA $ORIGIN RUNPATH is retained; absolute/build RUNPATHs are forbidden.
Output is a candidate for a separately authorized production install, not one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import tempfile


BASE = Path('/usr/lib/kiron/test-runtimes/prism')
PROFILE = BASE / 'source-builds/cuda12.8-sm86-tokens-v1'
BUILD = PROFILE / 'build-svf6d3wf'
PREPARED = PROFILE / 'prepared'
WHEEL_ROOT = Path('/usr/lib/kiron/test-venvs/prism-runtime/lib/python3.12/site-packages')
OUTPUT_ROOT = BASE / 'bundles'
DEFAULT_OUTPUT = OUTPUT_ROOT / 'prism-9a9394a-sm86-tokens-v1'
TOOL = BASE / 'package-tools/patchelf-0.18.0-1.1build1/extracted/usr/bin/patchelf'
SOURCE_COMMIT = '9a9394a895b96003ca842a6041cb28ac49a108f7'
PINS = {
    'result.json': 'fd62e5c59c21ba8315014094d2d7995a9f79b88d0f840c9775104954676d0451',
    'binary-manifest.json': '88903b80892715fd27fb20e44d9aabc3c431c2177b820db50ec6e8aaad5114b9',
    'launch.json': 'b0f1d7bdd3129d6b41652714a540ee04226d841a8e44d137d53226e5faaebd94',
}
PREPARED_SHA = 'b94273ae234a132b8912f2daf1db71020e833d0f8676ebd7323185a3498aef36'
LOCK_SHA = '87c789d179443f473ba0d879472f0699d4922d4064a55c1cfd84fe5f05de6fb5'
TOOL_SHA = '35fc95654387035338a74bb8cf62fde3712ec83dd8ca30a768deb714d07f063a'
TOOL_DEB_SHA = '962a43e33cd56061522554898557a038ccbb8aa4e1e0f421b2d6f6adf1f80c60'
CUBLAS_PINS = {
    'lib/libcublas.so.12': '031ce6c2cbfbb9468f040527cab5c599069ce5609e73e28f87503881063eac21',
    'lib/libcublasLt.so.12': '10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c',
}
CUBLAS_LICENSE_SHA = 'ad6f5853fba0ca0d159d0f58d49ae49830c2f8c93f7a92648b9ce90adb4c6ccd'
SYSTEM_LIBS = frozenset(('libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6', 'libm.so.6',
    'libgomp.so.1', 'libpthread.so.0', 'librt.so.1', 'libdl.so.2', 'ld-linux-x86-64.so.2', 'libcuda.so.1'))
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'}


def fingerprint(path, expected=None, destination=None):
    """Copy/hash one held regular FD; source may be a non-root build product."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    target = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 2 * 1024**3:
            raise ValueError('invalid input type or size')
        digest = hashlib.sha256()
        if destination is not None:
            target = open(destination, 'xb')
        with os.fdopen(fd, 'rb', closefd=False) as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
                if target is not None:
                    target.write(block)
        after = os.fstat(fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('input changed while copying')
        value = digest.hexdigest()
        if expected is not None and value != expected:
            raise ValueError(f'input hash mismatch: {path}')
        return value
    finally:
        if target is not None:
            target.close()
        os.close(fd)


def pinned_json(path, expected):
    fingerprint(path, expected)
    # Metadata roots are root-owned; no service can replace these documents.
    immutable(path)
    return json.loads(path.read_text())


def immutable(path, *, ancestors=False):
    items = (path, *path.parents) if ancestors else (path,)
    for item in items:
        info = item.lstat()
        if (info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode)
                or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))):
            raise ValueError(f'path must be root-owned and immutable to services: {item}')


def checked_name(name):
    if not isinstance(name, str) or name in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('unsafe flat bundle name')
    return name


def check_links(manifest):
    for name, record in manifest.items():
        checked_name(name)
        current, seen = name, set()
        while 'link' in manifest[current]:
            if current in seen:
                raise ValueError('cyclic bundle link')
            seen.add(current)
            current = checked_name(manifest[current]['link'])
            if current not in manifest:
                raise ValueError('dangling bundle link')


def command(args, *, library_path=None, unprivileged=False):
    env = dict(ENV)
    if library_path is not None:
        env['LD_LIBRARY_PATH'] = str(library_path)
    options = {}
    if unprivileged:
        nobody = pwd.getpwnam('nobody')
        options = dict(user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[])
    result = subprocess.run(list(map(str, args)), stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=20, env=env, cwd='/', **options)
    if result.returncode:
        raise ValueError(f'command failed ({result.returncode}): {args[0]}: {result.stderr[-2000:]}')
    if len(result.stdout) + len(result.stderr) > 1024 * 1024:
        raise ValueError('command output limit exceeded')
    return result.stdout + result.stderr


def audit_dependencies(root, *, execute=True):
    """Require own/CUDA DT_NEEDED closure before invoking the loader at all."""
    dynamic, elf_files = {}, []
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open('rb') as source:
            if source.read(4) != b'\x7fELF':
                continue
        elf_files.append(path)
        output = command(['/usr/bin/readelf', '-d', path])
        for value in re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', output):
            if value != '$ORIGIN':
                raise ValueError(f'nonrelocatable runtime search path: {path.name}')
        dependencies = re.findall(r'\(NEEDED\).*?\[(.*?)\]', output)
        for name in dependencies:
            if name not in SYSTEM_LIBS:
                checked_name(name)
                dependency = root / name
                if not dependency.is_file() or not dependency.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f'missing bundled dependency: {name}')
        dynamic[path.name] = {'needed': dependencies, 'readelf': output}
    if not elf_files or root / 'llama-server' not in elf_files:
        raise ValueError('bundle lacks server ELF')
    if not execute:
        return {'dynamic': dynamic}
    resolved = {}
    for path in elf_files:
        output = command(['/usr/bin/ldd', path], library_path=root, unprivileged=True)
        if 'not found' in output:
            raise ValueError('loader reports missing dependency')
        for name, location in re.findall(r'^\s*(\S+)\s+=>\s+(\S+)\s+\(', output, re.M):
            target = Path(location).resolve(strict=True)
            if name in SYSTEM_LIBS:
                if not any(target.is_relative_to(base) for base in (Path('/usr/lib'), Path('/lib'))):
                    raise ValueError(f'non-system host dependency: {name}')
            elif not target.is_relative_to(root.resolve()):
                raise ValueError(f'dependency escaped bundle: {name}')
            resolved[name] = str(target)
        dynamic[path.name]['ldd'] = output
    version = command([root / 'llama-server', '--version'], library_path=root, unprivileged=True)
    if '9a9394a' not in version:
        raise ValueError('server version differs from source pin')
    return {'dynamic': dynamic, 'resolved': resolved, 'version': version,
            'execution_user': 'nobody', 'environment': {**ENV, 'LD_LIBRARY_PATH': str(root)},
            'scope': 'Loader/version check only; no model or CUDA computation executed.'}


def runtime_manifest(root):
    result = {}
    for path in sorted(root.rglob('*')):
        key = str(path.relative_to(root))
        if path.is_symlink():
            if path.lstat().st_uid != 0 or not path.resolve().is_relative_to(root):
                raise ValueError('unsafe runtime symlink')
            result[key] = {'link': os.readlink(path)}
        elif path.is_dir():
            immutable(path)
        else:
            immutable(path)
            result[key] = {'sha256': fingerprint(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    path.chmod(0o444)


def verify(output):
    immutable(output, ancestors=True)
    for name in ('bundle-manifest.json', 'provenance.json', 'verification.json'):
        immutable(output / name)
    manifest = json.loads((output / 'bundle-manifest.json').read_text())
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance.get('source_commit') != SOURCE_COMMIT or provenance.get('build_report_hashes') != PINS:
        raise ValueError('unexpected bundle provenance')
    if fingerprint(output / 'bundle-manifest.json') != provenance.get('bundle_manifest_sha256'):
        raise ValueError('manifest provenance mismatch')
    root = output / 'runtime'
    immutable(root)
    if runtime_manifest(root) != manifest:
        raise ValueError('bundle inventory mismatch')
    return audit_dependencies(root)


def package(output=DEFAULT_OUTPUT, tool=TOOL):
    if os.geteuid() != 0:
        raise ValueError('packaging requires root ownership; execution probes use nobody')
    output = Path(output)
    if not output.is_absolute() or output.parent != OUTPUT_ROOT or not re.fullmatch(r'[A-Za-z0-9_.-]+', output.name):
        raise ValueError('output must be one isolated bundle directly under the test bundles root')
    OUTPUT_ROOT.mkdir(mode=0o755, exist_ok=True)
    immutable(OUTPUT_ROOT, ancestors=True)
    if output.exists() or output.is_symlink():
        return verify(output)
    if shutil.disk_usage(OUTPUT_ROOT).free < 3 * 1024**3:
        raise ValueError('at least 3 GiB free required for bounded package staging')
    immutable(tool, ancestors=True)
    fingerprint(tool, TOOL_SHA)
    lock_path = Path(__file__).with_name('source-toolchain-lock.json')
    lock = pinned_json(lock_path, LOCK_SHA)
    prepared = pinned_json(PREPARED / 'prepared.json', PREPARED_SHA)['inventory']
    reports = {name: pinned_json(BUILD / name, digest) for name, digest in PINS.items()}
    if reports['result.json']['status'] != 'complete' or reports['result.json']['lock_sha256'] != LOCK_SHA:
        raise ValueError('source build did not pass')
    manifest = reports['binary-manifest.json']
    check_links(manifest)
    stage = Path(tempfile.mkdtemp(prefix='.bundle-stage-', dir=OUTPUT_ROOT))
    # The real nobody loader probe needs traversal; parent remains root-only writable.
    stage.chmod(0o755)
    root = stage / 'runtime'
    root.mkdir(mode=0o755)
    inputs = {}
    try:
        for name, record in manifest.items():
            source, target = BUILD / 'build/bin' / name, root / name
            if 'link' in record:
                if not source.is_symlink() or os.readlink(source) != record['link']:
                    raise ValueError('source library link changed')
                target.symlink_to(record['link'])
            else:
                inputs[name] = fingerprint(source, record['sha256'], target)
                command([tool, '--remove-rpath', target])
                target.chmod(0o555)
        for relative, digest in CUBLAS_PINS.items():
            source = WHEEL_ROOT / 'nvidia/cublas' / relative
            inputs[Path(relative).name] = fingerprint(source, digest, root / Path(relative).name)
        cudart = 'cuda/lib/libcudart.so.12.8.90'
        inputs['libcudart.so.12.8.90'] = fingerprint(PREPARED / cudart, prepared[cudart]['sha256'],
                                                   root / 'libcudart.so.12.8.90')
        (root / 'libcudart.so.12').symlink_to('libcudart.so.12.8.90')
        licenses = root / 'licenses'
        licenses.mkdir(mode=0o755)
        source_licenses = {'source/LICENSE': 'Prism-MIT.txt', 'source/licenses/LICENSE-jsonhpp': 'jsonhpp.txt',
            'source/vendor/cpp-httplib/LICENSE': 'cpp-httplib.txt',
            'source/vendor/hash/xxhash/LICENSE': 'xxhash.txt', 'source/vendor/hash/sha256/LICENSE': 'sha256.txt',
            'source/vendor/hash/rotate-bits/LICENSE.md': 'rotate-bits.txt', 'cuda/cudart-LICENSE': 'NVIDIA-CUDART.txt'}
        for source, target in source_licenses.items():
            fingerprint(PREPARED / source, prepared[source]['sha256'], licenses / target)
        fingerprint(WHEEL_ROOT / 'nvidia_cublas_cu12-12.8.4.1.dist-info/License.txt', CUBLAS_LICENSE_SHA,
                    licenses / 'NVIDIA-cuBLAS.txt')
        for path in root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o555 if path.is_dir() or path.name in manifest else 0o444)
        root.chmod(0o555)
        write_json(stage / 'bundle-manifest.json', runtime_manifest(root))
        provenance = {'source_commit': SOURCE_COMMIT, 'build_report_hashes': PINS,
            'source_lock_sha256': LOCK_SHA, 'prepared_sha256': PREPARED_SHA,
            'source_patches': lock['source']['patches'],
            'source_build': str(BUILD), 'build_result': reports['result.json'],
            'input_file_sha256': inputs, 'cuda_versions': {'cudart': lock['artifacts']['cudart'],
                                                         'cublas': lock['cublas_wheel']},
            'patchelf': {'version': '0.18.0-1.1build1', 'sha256': TOOL_SHA, 'deb_sha256': TOOL_DEB_SHA,
                        'operation': '--remove-rpath on copied Prism binaries only'},
            'package_recipe_sha256': fingerprint(Path(__file__)),
            'bundle_manifest_sha256': fingerprint(stage / 'bundle-manifest.json'),
            'host_requirements': 'Linux x86_64 glibc/libstdc++/OpenMP and NVIDIA driver remain host dependencies.'}
        write_json(stage / 'provenance.json', provenance)
        write_json(stage / 'verification.json', audit_dependencies(root))
        # Verify after moving to the final name too: absolute staging paths cannot pass.
        stage.rename(output)
        try:
            verification = verify(output)
            write_json(output / 'verification.json', verification)
            output.chmod(0o555)
            return verification
        except BaseException:
            shutil.rmtree(output)
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--patchelf', type=Path, default=TOOL)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verification = verify(args.output) if args.verify else package(args.output, args.patchelf)
    print(json.dumps({'output': str(args.output), 'manifest_sha256': fingerprint(args.output / 'bundle-manifest.json'),
                      'binary_sha256': fingerprint(args.output / 'runtime/llama-server'),
                      'version': verification['version'].strip()}, indent=2))


if __name__ == '__main__':
    main()
