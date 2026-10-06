#!/usr/bin/env python3
"""Explicit isolated CPU gate. No production service, registry or marker access.

Root creates a private mount namespace/cgroup; both model and SDK run as nobody
without supplementary groups. No weight copy or download. Only --run loads a
model. The native CPU gate does not grant production capabilities/cold loads.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # Root cleanup imports the immutable source snapshot.

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path('/usr/lib/kiron/test-runtimes/prism/reports')
HF = Path('/var/cache/kiron/huggingface/hub/models--keyvan-ai--Mankei-326M-Embedder')
REVISION = '86f562cf94d5510175b546e6e9156f99bbd790b5'
WEIGHT = 'bbfe289c78cad5ef7d64b0dd5cc547380a274a82823207e7ee4fcf44f49264d8'
KITT = '/usr/lib/kiron/test-venvs/kitt-worker/bin/python'
SDK = '/usr/lib/kiron/test-venvs/local-inference/bin/python'
PORT, LIMIT, MIN_RAM, SECONDS = 18096, 4 * 1024**3, 8 * 1024**3, 600


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
    path.chmod(0o444)


def available_ram():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) * 1024


def require_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', PORT))


def report_path(value):
    path = Path(value)
    if (path.parent != REPORTS or path != path.resolve() or not path.name.startswith('embedding-cpu-')
            or not all(c.isalnum() or c in '-_' for c in path.name)):
        raise ValueError('report must be a new embedding-cpu-NAME directory in the isolated report root')
    return path


def inventory_sources():
    paths = []
    for service in ('kiron-proxy', 'kiron-embeddings', 'kiron-common'):
        base = ROOT / 'services' / service
        iterator = (base / 'kiron_common').rglob('*') if service == 'kiron-common' else base.glob('*.py')
        paths.extend(p for p in iterator if p.is_file() and p.suffix in {'.py', '.json'}
                     and not p.name.startswith(('test_', 'conftest')) and '__pycache__' not in p.parts)
    paths.extend((Path(__file__).resolve(), Path(__file__).with_name('cpu_probe.py').resolve()))
    return sorted(paths)


def prepare(path):
    if os.geteuid() != 0:
        raise PermissionError('prepare requires root-owned source and mountpoint preparation')
    if path.exists():
        raise ValueError('report directory must be new')
    require_port()
    ram = available_ram()
    if ram < MIN_RAM:
        raise RuntimeError('less than 8 GiB host available')
    snapshot = HF / 'snapshots' / REVISION
    weight = snapshot / 'model.safetensors'
    if sha(weight) != WEIGHT or weight.stat().st_size != 652884384:
        raise ValueError('Mankei local weight pin mismatch')
    artifacts = {}
    for file in sorted(snapshot.rglob('*')):
        if file.is_file():
            if not file.resolve().is_relative_to(HF):
                raise ValueError('HF file leaves the selected repository')
            artifacts[str(file.relative_to(snapshot))] = {'sha256': sha(file), 'bytes': file.stat().st_size}
    if set(artifacts) != {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'README.md'}:
        raise ValueError('unreviewed Mankei snapshot file inventory')
    path.mkdir(mode=0o755)
    source = path / 'source'
    inventory = {}
    for original in inventory_sources():
        relative = original.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = original.read_bytes()
        target.write_bytes(raw)
        target.chmod(0o444)
        inventory[str(relative)] = hashlib.sha256(raw).hexdigest()
    for directory in reversed(sorted(p for p in source.rglob('*') if p.is_dir())):
        directory.chmod(0o555)
    source.chmod(0o555)
    uid = pwd.getpwnam('nobody').pw_uid
    gid = pwd.getpwnam('nobody').pw_gid
    for name in ('work', 'admission'):
        target = path / name
        target.mkdir(mode=0o2770 if name == 'admission' else 0o700)
        os.chown(target, uid, gid)
        target.chmod(0o2770 if name == 'admission' else 0o700)
    (path / 'cache' / HF.name).mkdir(parents=True, mode=0o755)
    manifest = {'version': 1, 'scope': 'isolated CPU resident native → provider → SDK gate; no production grant',
        'uid': uid, 'gid': gid, 'port': PORT, 'max_bytes': LIMIT, 'max_seconds': SECONDS,
        'host_available_before': ram, 'sources': inventory, 'hf_source': str(HF),
        'hf_revision': REVISION, 'hf_artifacts': artifacts}
    write(path / 'prepared.json', manifest)
    return manifest


def verify(path):
    def secure(item, *, readonly=False):
        info = item.lstat()
        if (info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode)
                or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))
                or (readonly and info.st_mode & 0o222)
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)):
            raise ValueError('unsafe frozen source/manifest ownership, type or mode')
    for parent in (path, *path.parents):
        secure(parent)
    secure(path / 'prepared.json', readonly=True)
    manifest = json.loads((path / 'prepared.json').read_text())
    expected = {'version': 1, 'uid': pwd.getpwnam('nobody').pw_uid, 'gid': pwd.getpwnam('nobody').pw_gid,
        'port': PORT, 'max_bytes': LIMIT, 'max_seconds': SECONDS, 'hf_source': str(HF), 'hf_revision': REVISION}
    if any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in expected.items()):
        raise ValueError('frozen launch parameters changed')
    source = path / 'source'
    secure(source, readonly=True)
    actual = set()
    for item in source.rglob('*'):
        secure(item, readonly=True)
        if item.is_file():
            actual.add(str(item.relative_to(source)))
    if type(manifest['sources']) is not dict or set(manifest['sources']) != actual:
        raise ValueError('frozen source inventory changed')
    for name, digest in manifest['sources'].items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or str(relative) != name
                or not (name.startswith(('services/kiron-common/', 'services/kiron-proxy/', 'services/kiron-embeddings/'))
                    or name in {'scripts/embeddings/smoke-cpu.py', 'scripts/embeddings/cpu_probe.py'})):
            raise ValueError('invalid frozen source path')
        if sha(path / 'source' / name) != digest:
            raise ValueError('frozen source changed')
    verify_artifacts(manifest)
    return manifest


def verify_artifacts(manifest):
    if (type(manifest['hf_artifacts']) is not dict or set(manifest['hf_artifacts']) !=
            {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'README.md'}):
        raise ValueError('unreviewed artifact inventory')
    if manifest['hf_artifacts']['model.safetensors'] != {'sha256': WEIGHT, 'bytes': 652884384}:
        raise ValueError('weight pin changed')
    for name, item in manifest['hf_artifacts'].items():
        source = HF / 'snapshots' / REVISION / name
        if sha(source) != item['sha256'] or source.stat().st_size != item['bytes']:
            raise ValueError('local artifact changed')


def clean_environment(path):
    return {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'TMPDIR': str(path / 'work'), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_CACHE': str(path / 'cache'), 'HF_HOME': str(path / 'work' / 'hf'),
        'HF_MODULES_CACHE': str(path / 'work' / 'hf-modules'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'TOKENIZERS_PARALLELISM': 'false'}


def namespace(path):
    # Called only by this run's unshare child, still root until read-only bind.
    manifest = json.loads((path / 'prepared.json').read_text())
    target = path / 'cache' / HF.name
    subprocess.run(['/usr/bin/mount', '--bind', str(HF), str(target)], check=True, timeout=10)
    subprocess.run(['/usr/bin/mount', '-o', 'remount,bind,ro,nosuid,nodev,noexec', str(target)], check=True, timeout=10)
    execute_unprivileged(path, manifest, native=True)


def no_new_privileges():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'no_new_privs failed')


def execute_unprivileged(path, manifest, *, native):
    no_new_privileges()
    os.setgroups([])
    os.setgid(manifest['gid'])
    os.setuid(manifest['uid'])
    script = path / 'source/scripts/embeddings/cpu_probe.py'
    interpreter = KITT if native else SDK
    os.execve(interpreter, [interpreter, '-I', '-B', str(script), '--native' if native else '--sdk', str(path)],
              clean_environment(path))


def create_cgroup(path):
    group = Path('/sys/fs/cgroup') / ('kiron-test-' + path.name)
    group.mkdir()  # Existing group is never reused or signalled.
    try:
        (group / 'memory.max').write_text(str(LIMIT))
        (group / 'memory.swap.max').write_text('0')
        (group / 'cpu.max').write_text('200000 100000')
        (group / 'pids.max').write_text('128')
    except BaseException:
        group.rmdir()
        raise
    return group


def run(path):
    if os.geteuid() != 0:
        raise PermissionError('run requires root solely for isolated namespace/cgroup and UID drop')
    manifest = verify(path)
    if available_ram() < MIN_RAM:
        raise RuntimeError('less than 8 GiB host available before start')
    require_port()
    group = create_cgroup(path)
    processes, streams = [], []
    result = {'status': 'failed', 'scope': manifest['scope'], 'processes': []}
    deadline = time.monotonic() + SECONDS
    def join_group():
        (group / 'cgroup.procs').write_text(str(os.getpid()))
    def spawn(args, *, logfile):
        output = (path / logfile).open('x')
        streams.append(output)
        process = subprocess.Popen(args, cwd=path / 'work', env=clean_environment(path),
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=join_group)
        processes.append(process)
        result['processes'].append({'pid': process.pid, 'argv': args})
        print(json.dumps({'started_pid': process.pid, 'report': str(path), 'port': PORT}), flush=True)
        return process
    def guarded():
        rss = 0
        for process in processes:
            if process.poll() is None:
                lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                rss += sum(int(line.split()[1]) * 1024 for line in lines if line.startswith('VmRSS:'))
        result['rss_peak_bytes'] = max(result.get('rss_peak_bytes', 0), rss)
        return time.monotonic() < deadline and available_ram() >= MIN_RAM and rss <= LIMIT
    try:
        frozen = path / 'source/scripts/embeddings/smoke-cpu.py'
        native = spawn(['/usr/bin/unshare', '--mount', '--propagation', 'private', '--',
                        '/usr/bin/python3', '-I', '-B', str(frozen), '--namespace', str(path)], logfile='native.log')
        ready = path / 'work/native-ready.json'
        while not ready.exists():
            if native.poll() is not None or not guarded():
                raise RuntimeError('native failed/readiness timeout/host memory guard')
            time.sleep(.2)
        probe = spawn(['/usr/bin/python3', '-I', '-B', str(frozen), '--sdk-child', str(path)], logfile='sdk.log')
        while probe.poll() is None:
            if native.poll() is not None or not guarded():
                raise RuntimeError('probe runtime/host memory guard')
            time.sleep(.2)
        if probe.returncode != 0:
            raise RuntimeError('SDK/native gate failed; see isolated logs')
        result['probe'] = json.loads((path / 'work/sdk-result.json').read_text())
        result['status'] = 'passed'
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        # Signal only children retained by Popen (unreaped PID cannot be reused).
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        remaining = (group / 'cgroup.procs').read_text().strip()
        result['cgroup_empty'] = not remaining
        result['memory_peak_bytes'] = int((group / 'memory.peak').read_text())
        result['memory_events'] = (group / 'memory.events').read_text()
        stopped = path / 'work/native-stopped.json'
        if result['status'] == 'passed' and (not stopped.is_file()
                or json.loads(stopped.read_text()).get('worker_alive') is not False):
            result.update(status='failed', error='native graceful worker shutdown was not observed')
        if remaining:
            # This newly created cgroup contains only our descendants.
            (group / 'cgroup.kill').write_text('1')
            result['status'] = 'failed'
            result['error'] = 'unexpected surviving child processes'
        else:
            group.rmdir()
            # Both owned subprocesses are reaped and their dedicated cgroup is
            # empty. Only now may this root controller clear its private store.
            sys.path.insert(0, str(path / 'source/services/kiron-common'))
            from kiron_common.gpu_admission import AdmissionStore, RuntimeSecurity
            store = AdmissionStore(path / 'admission', security=RuntimeSecurity(
                manifest['uid'], manifest['gid'], frozenset({0, manifest['uid']})))
            for ticket in store.snapshot():
                store.release(ticket.operation_id, owner=ticket.owner,
                              generation=ticket.generation, confirmed_terminated=True)
            result['admission_empty_after_process_end'] = not store.snapshot()
        for output in streams:
            output.close()
        try:
            require_port()
            verify(path)
            result['artifact_and_source_recheck'] = 'passed'
        except Exception as exc:
            result.update(status='failed', cleanup_error=str(exc))
        write(path / 'result.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--namespace', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--sdk-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('report')
    args = parser.parse_args()
    path = report_path(args.report)
    if args.prepare:
        print(json.dumps(prepare(path), indent=2))
    elif args.namespace:
        namespace(path)
    elif args.sdk_child:
        execute_unprivileged(path, json.loads((path / 'prepared.json').read_text()), native=False)
    else:
        sys.exit(run(path))
