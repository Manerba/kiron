#!/usr/bin/env python3
"""Prepare an immutable fixture or explicitly run one isolated CPU Ollama test.

Never pull an image, install packages, contact production Ollama or modify its
cache. Run only the prepared source snapshot after an independent review.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = Path('/usr/lib/kiron/test-runtimes/ollama/reports')
IMAGE = 'sha256:684edc911db13b64ad072af7f83c3ca2e6545e4933961d67ed11f2893f897525'
REPO_DIGEST = 'ollama/ollama@sha256:e23e6499890325d31f3454cdc83f7a613a2423c8a9e167bbafea0b0e3c27d7c6'
CACHE = Path('/var/lib/docker/volumes/botmin_ollama/_data/models')
MANIFEST = 'manifests/registry.ollama.ai/library/qwen3/8b'
MODEL_SHA = '500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41'
COMPAT = 'data/ollama_compat_reports/20260820-043532-ollama-0.18.0-e23e64998903.json'
COMPAT_SHA = '5681a5c1159fbe527a323a09c79935d0cd1e436caeb16c468dd563aa55792e44'
PYTHON = '/usr/lib/kiron/test-venvs/local-inference/bin/python'
PORT, UID, GID = 18095, 65534, 982
RAM_LIMIT, MIN_RAM = 9 * 1024**3, 2 * 1024**3
MAX_SECONDS = 900  # supervised child + setup; cleanup adds at most bounded CLI waits
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, mode=0o440):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.chown(path, 0, GID)
    path.chmod(mode)


def diagnostic_stderr(value):
    """Keep bounded CLI diagnostics, excluding terminal controls and credentials."""
    value = value.decode('utf-8', 'replace') if isinstance(value, bytes) else (value or '')
    value = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)
    value = ''.join(character for character in value if character.isprintable() or character in '\n\t')
    value = re.sub(r'(?i)(https?://)[^\s/@]+:[^\s/@]+@', r'\1[redacted]@', value)
    value = re.sub(r'(?i)\b(Bearer|Basic)\s+[^\s,;]+', r'\1 [redacted]', value)
    value = re.sub(r'''(?i)(\b(?:password|passwd|token|secret|api[_-]?key|authorization)\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''',
                   r'\1[redacted]', value)
    return value[:4096].strip()


class DockerError(RuntimeError):
    def __init__(self, operation, exc):
        self.diagnostic = {'operation': operation, 'stderr': diagnostic_stderr(exc.stderr)}
        if isinstance(exc, subprocess.TimeoutExpired):
            self.diagnostic['timeout_seconds'] = exc.timeout
        else:
            self.diagnostic['returncode'] = exc.returncode
        # Never include inherited command arguments, stdout or exception repr.
        super().__init__('Docker failure: ' + json.dumps(self.diagnostic, ensure_ascii=True))


def docker(*args, timeout=15):
    try:
        return subprocess.run(['/usr/bin/docker', *args], capture_output=True, text=True,
            encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL,
            env=ENV, timeout=timeout, check=True).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DockerError(args[0], exc) from None


def image_identity():
    value = json.loads(docker('image', 'inspect', IMAGE))[0]
    if (value['Id'] != IMAGE or REPO_DIGEST not in value.get('RepoDigests', [])
            or value['Architecture'] != 'amd64' or value['Os'] != 'linux'
            or value['Config']['Entrypoint'] != ['/bin/ollama']
            or value['Config'].get('Volumes') or value['Config'].get('Healthcheck')):
        raise ValueError('unexpected local image identity')
    return {key: value[key] for key in ('Id', 'RepoDigests', 'Architecture', 'Os', 'RootFS')}


def regular(path, *, readable=False):
    info = path.lstat()
    if (path.resolve() != path or not stat.S_ISREG(info.st_mode) or info.st_uid != 0
            or info.st_mode & 0o022 or (readable and not info.st_mode & 0o004)):
        raise ValueError('unsafe pinned input')
    for parent in path.parents if readable else ():
        info_parent = parent.lstat()
        if info_parent.st_uid != 0 or info_parent.st_mode & 0o022 or not stat.S_ISDIR(info_parent.st_mode):
            raise ValueError('mutable pinned input parent')
        if parent.is_relative_to(CACHE) and not info_parent.st_mode & 0o001:
            raise ValueError('model cache directory is not readable by isolated UID')
    return info


def require_no_cache_submounts(mountinfo):
    # Docker requires rslave under its own data root, while its CLI forbids
    # force-recursive-readonly with rslave. This fixture has no submounts.
    for line in mountinfo.splitlines():
        target = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), line.split()[4])
        if target.startswith(str(CACHE) + '/'):
            raise ValueError('model cache submounts are unsupported')


def model_inventory():
    require_no_cache_submounts(Path('/proc/self/mountinfo').read_text())
    regular(CACHE / MANIFEST, readable=True)
    if sha(CACHE / MANIFEST) != MODEL_SHA:
        raise ValueError('model manifest digest changed')
    manifest = json.loads((CACHE / MANIFEST).read_text())
    result = {MANIFEST: MODEL_SHA}
    for item in [manifest['config'], *manifest['layers']]:
        if not re.fullmatch('sha256:[0-9a-f]{64}', item['digest']):
            raise ValueError('invalid pinned model blob')
        relative = 'blobs/' + item['digest'].replace(':', '-')
        path = CACHE / relative
        if regular(path, readable=True).st_size != item['size'] or sha(path) != item['digest'][7:]:
            raise ValueError('model blob size/hash differs from manifest')
        result[relative] = item['digest'][7:]
    return result


def inventory(root):
    paths = {root / 'scripts/ollama' / name for name in ('smoke-runtime.py', 'probe-runtime.py')}
    paths.add(root / 'scripts/prism/probe-openai-features.py')  # bounded native trace, no auth headers
    common = root / 'services/kiron-common'
    paths.add(common / 'pyproject.toml')
    paths.update(p for p in (common / 'kiron_common').rglob('*') if p.suffix in {'.py', '.json'})
    for service in ('kiron-proxy', 'kiron-prism'):
        directory = root / 'services' / service
        paths.update(p for p in directory.glob('*.py') if not p.name.startswith('test_') and p.name != 'conftest.py')
        paths.update(directory.glob('requirements*.txt'))
    result = {}
    for path in sorted(paths):
        if regular(path).st_size > 4 * 1024**2:
            raise ValueError('oversized source file')
        result[str(path.relative_to(root))] = sha(path)
    return result


def report_path(path):
    if path.parent != ROOT or path != path.resolve() or not re.fullmatch('cpu-[A-Za-z0-9_-]+', path.name):
        raise ValueError('report must be canonical reports/cpu-NAME')
    return path


def free_port():
    with socket.socket() as sock:
        # Like the native Go listener, allow ended TIME_WAIT connections. Keep
        # SO_REUSEPORT disabled so an actual foreign listener still blocks bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', PORT))


def available_ram():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def prepare(report):
    if os.geteuid() != 0 or report.exists():
        raise ValueError('root must prepare a new report')
    report_path(report)
    free_port()
    image, models, source = image_identity(), model_inventory(), inventory(WORKSPACE)
    if sha(WORKSPACE / COMPAT) != COMPAT_SHA or available_ram() < RAM_LIMIT + MIN_RAM:
        raise ValueError('compatibility pin or host RAM preflight failed')
    for name in ('kiron-common', 'kiron-proxy'):
        sys.path.insert(0, str(WORKSPACE / 'services' / name))
    from kiron_common.local_model_registry import RegistryEntry, RegistryFilePolicy, RuntimeModelRegistry
    from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
    report.mkdir(parents=True, mode=0o750)
    os.chown(report, 0, GID)
    for name, owner, mode in (('results', UID, 0o750), ('registry', 0, 0o750), ('admission', 0, 0o2770)):
        path = report / name
        path.mkdir()
        os.chown(path, owner, GID)
        path.chmod(mode)
    for relative, digest in source.items():
        target = report / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (WORKSPACE / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source changed during preparation')
        target.write_bytes(data)
        os.chown(target, 0, GID)
        target.chmod(0o440)
    for path in [report / 'source', *(p for p in (report / 'source').rglob('*') if p.is_dir())]:
        os.chown(path, 0, GID)
        path.chmod(0o550)
    if inventory(report / 'source') != source or inventory(WORKSPACE) != source:
        raise ValueError('source snapshot mismatch')
    entry = RegistryEntry.create(runtime_provider=BackendType.OLLAMA, artifact_origin=ArtifactType.OLLAMA,
        artifact_format=ArtifactFormat.OLLAMA_MANIFEST, reference='qwen3:8b', display_name='Isolated Ollama CPU probe',
        loader=LoaderType.OLLAMA, sha256=MODEL_SHA)
    RuntimeModelRegistry(report / 'registry/models.json', file_policy=RegistryFilePolicy(0, GID)).add(entry)
    compat = json.loads((WORKSPACE / COMPAT).read_text())
    write(report / 'compat.json', compat)
    plan = {'scope': 'isolated CPU container and SDK/RuntimeService adapter measurement; no production mutation or coldload profile',
        'image': image, 'model_sha256': models, 'source_sha256': source, 'registry_sha256': sha(report / 'registry/models.json'),
        'compat_source_sha256': COMPAT_SHA, 'compat_sha256': sha(report / 'compat.json'),
        'model_id': entry.id, 'run_id': uuid.uuid4().hex, 'port': PORT, 'uid': UID, 'gid': GID,
        'max_seconds': MAX_SECONDS, 'prepared_at': time.time()}
    write(report / 'plan.json', plan)
    return plan


def verify_source(report, expected):
    actual = set()
    for path in [report / 'source', *(report / 'source').rglob('*')]:
        info = path.lstat()
        isdir = stat.S_ISDIR(info.st_mode)
        if (info.st_uid != 0 or info.st_gid != GID or path.resolve() != path
                or stat.S_IMODE(info.st_mode) != (0o550 if isdir else 0o440)):
            raise ValueError('mutable source snapshot')
        if not isdir:
            actual.add(str(path.relative_to(report / 'source')))
    if actual != set(expected) or inventory(report / 'source') != expected:
        raise ValueError('source snapshot changed')


def verify(report):
    report_path(report)
    if Path(__file__).resolve() != report / 'source/scripts/ollama/smoke-runtime.py' or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('run immutable snapshot using TEST Python -I -B')
    plan = json.loads((report / 'plan.json').read_text())
    for path in (report, report / 'plan.json', report / 'compat.json', report / 'registry',
                 report / 'registry/models.json', report / 'registry/models.json.lock'):
        info = path.lstat()
        if info.st_uid != 0 or info.st_gid != GID or info.st_mode & 0o022 or path.resolve() != path:
            raise ValueError('mutable fixture')
    verify_source(report, plan['source_sha256'])
    if (sha(report / 'registry/models.json') != plan['registry_sha256']
            or sha(report / 'compat.json') != plan['compat_sha256']
            or plan['compat_source_sha256'] != COMPAT_SHA or plan['image'] != image_identity()
            or plan['model_sha256'] != model_inventory() or plan['port'] != PORT
            or plan['uid'] != UID or plan['gid'] != GID or plan['max_seconds'] != MAX_SECONDS
            or not re.fullmatch('[0-9a-f]{32}', plan['run_id']) or any((report / 'results').iterdir())):
        raise ValueError('changed or reused fixture')
    free_port()
    if available_ram() < RAM_LIMIT + MIN_RAM:
        raise ValueError('insufficient host RAM')
    return plan


def container_args(report, run_id):
    environment = {'HOME': '/tmp', 'OLLAMA_HOST': f'127.0.0.1:{PORT}', 'OLLAMA_MODELS': '/models',
        'OLLAMA_CONTEXT_LENGTH': '1024', 'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_MAX_LOADED_MODELS': '1',
        'OLLAMA_LOAD_TIMEOUT': '180s', 'OLLAMA_KEEP_ALIVE': '10m', 'OLLAMA_NO_CLOUD': '1',
        'CUDA_VISIBLE_DEVICES': '-1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'ROCR_VISIBLE_DEVICES': '-1',
        'OMP_NUM_THREADS': '2', 'LD_LIBRARY_PATH': ''}
    result = ['create', '--pull=never', '--runtime=runc', '--network=host', '--read-only',
        '--user', f'{UID}:{GID}', '--cap-drop=ALL', '--security-opt=no-new-privileges',
        '--cpus=2', '--memory=9g', '--memory-swap=9g', '--pids-limit=128', '--ulimit', 'core=0',
        '--restart=no', '--label', 'kiron.isolated-ollama=' + run_id,
        '--cidfile', str(report / 'container.cid'), '--name', 'kiron-ollama-cpu-' + run_id,
        '--mount', f'type=bind,src={CACHE},dst=/models,readonly,bind-propagation=rslave',
        '--tmpfs', f'/tmp:rw,noexec,nosuid,nodev,size=64m,uid={UID},gid={GID},mode=1777',
        '--entrypoint', '/bin/ollama']
    for key, value in environment.items():
        result.extend(['--env', key + '=' + value])
    return [*result, IMAGE, 'serve']


def inspect_owned(cid, run_id):
    if not re.fullmatch('[0-9a-f]{64}', cid):
        raise ValueError('invalid owned container ID')
    value = json.loads(docker('inspect', cid))[0]
    config, host = value['Config'], value['HostConfig']
    mounts = [item for item in value['Mounts'] if item['Type'] == 'bind']
    if (value['Id'] != cid or value['Image'] != IMAGE or config.get('Labels', {}).get('kiron.isolated-ollama') != run_id
            or config['User'] != f'{UID}:{GID}' or host['Privileged'] or not host['ReadonlyRootfs']
            or host['Runtime'] != 'runc' or host['NetworkMode'] != 'host' or host.get('Devices')
            or host.get('DeviceRequests') or host.get('GroupAdd') or host.get('CapAdd')
            or host['NanoCpus'] != 2_000_000_000 or host['Memory'] != RAM_LIMIT or host['MemorySwap'] != RAM_LIMIT
            or 'no-new-privileges' not in host['SecurityOpt'] or host['CapDrop'] != ['ALL']
            or config['Entrypoint'] != ['/bin/ollama'] or config['Cmd'] != ['serve']
            or host['PidsLimit'] != 128 or set(host.get('Tmpfs') or {}) != {'/tmp'}
            or any(item['Type'] not in {'bind', 'tmpfs'} for item in value['Mounts'])
            or len(mounts) != 1 or mounts[0]['Source'] != str(CACHE)
            or mounts[0]['Destination'] != '/models' or mounts[0]['RW']
            or mounts[0].get('Propagation') != 'rslave'):
        raise ValueError('container identity/isolation differs from pinned launch')
    env = dict(item.split('=', 1) for item in config['Env'])
    for key, expected in {'OLLAMA_HOST': f'127.0.0.1:{PORT}', 'OLLAMA_MODELS': '/models',
                          'OLLAMA_NUM_PARALLEL': '1', 'OLLAMA_CONTEXT_LENGTH': '1024',
                          'NVIDIA_VISIBLE_DEVICES': 'void', 'CUDA_VISIBLE_DEVICES': '-1'}.items():
        if env.get(key) != expected:
            raise ValueError('container environment differs from pinned launch')
    return value


def require_readonly_model_mounts(mountinfo):
    """Check effective per-mount flags, including any propagated submounts."""
    observed = []
    for line in mountinfo.splitlines():
        fields = line.split()
        target = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4])
        if target == '/models' or target.startswith('/models/'):
            if 'ro' not in fields[5].split(',') or 'rw' in fields[5].split(','):
                raise ValueError('model cache mount is writable')
            observed.append(target)
    if '/models' not in observed:
        raise ValueError('model cache mount missing')


def cgroup_limits(path, pid):
    """Retain every observed limit before parsing, including an explicit max."""
    limits, oversized = {}, []
    for name in ('cpu.max', 'memory.max', 'memory.swap.max', 'pids.max'):
        try:
            raw = (path / name).read_text()
            limits[name] = raw[:128]
            if len(raw) > 128:
                oversized.append(name)
        except OSError as error:
            limits[name] = '<read failed: ' + type(error).__name__ + '>'
    unbounded = [name for name, raw in limits.items() if 'max' in raw.split()]
    valid = (not oversized and re.fullmatch(r'[0-9]+ [0-9]+\s*', limits['cpu.max']) is not None
             and all(re.fullmatch(r'[0-9]+\s*', limits[name]) is not None
                     for name in ('memory.max', 'memory.swap.max', 'pids.max')))
    if valid:
        quota, period = map(int, limits['cpu.max'].split())
        valid = (period > 0 and 0 < quota <= 2 * period
                 and int(limits['memory.max']) == RAM_LIMIT
                 and int(limits['memory.swap.max']) == 0 and int(limits['pids.max']) == 128)
    if not valid:
        observed = {'pid': pid, 'cgroup': str(path), 'limits': limits,
                    'unbounded_fields': unbounded}
        if oversized:
            observed['truncated_fields'] = oversized
        raise ValueError('container cgroup bounds not effective: ' + json.dumps(observed, sort_keys=True))
    return limits


def process_limits(pid):
    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    uids, gids = list(map(int, status['Uid'].split())), list(map(int, status['Gid'].split()))
    groups = list(map(int, status['Groups'].split()))
    nnp, capabilities = int(status['NoNewPrivs']), int(status['CapEff'], 16)
    # The pinned Docker/runc launch repeats its primary GID as its sole group.
    # The SDK child still uses extra_groups=[]; these are separate launch paths.
    if uids != [UID] * 4 or gids != [GID] * 4 or groups != [GID] or nnp != 1 or capabilities != 0:
        observed = {'pid': pid, 'uids': uids[:4], 'gids': gids[:4], 'groups': groups[:16],
                    'group_count': len(groups), 'no_new_privs': nnp, 'cap_effective': hex(capabilities)}
        raise ValueError('native process privileges differ: ' + json.dumps(observed, sort_keys=True))
    device_root = Path(f'/proc/{pid}/root/dev')
    if list(device_root.glob('nvidia*')) or (device_root / 'dri').exists() or (device_root / 'kfd').exists():
        raise ValueError('GPU device visible in CPU container')
    require_readonly_model_mounts(Path(f'/proc/{pid}/mountinfo').read_text())
    lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    if len(lines) != 1 or not lines[0].startswith('0::/'):
        raise ValueError('cgroup v2 required')
    path = Path('/sys/fs/cgroup') / lines[0][4:]
    if not path.resolve().is_relative_to('/sys/fs/cgroup'):
        raise ValueError('invalid cgroup path')
    limits = cgroup_limits(path, pid)
    return {'pid': pid, 'cgroup': str(path), 'memory_current': int((path / 'memory.current').read_text()),
            'memory_peak': int((path / 'memory.peak').read_text()), 'host_available': available_ram(),
            'uid': uids[0], 'gid': gids[0], 'groups': groups, 'no_new_privs': nnp,
            'cap_effective': capabilities, 'gpu_devices': [],
            'model_mounts_readonly': True, 'cgroup_limits': limits}


def no_new_privileges():
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
        raise OSError('no_new_privs failed')


def stop_owned(report, cid, run_id, result):
    inspect_owned(cid, run_id)
    try:
        logs = subprocess.run(['/usr/bin/docker', 'logs', '--tail', '2000', cid],
            capture_output=True, timeout=15, env=ENV, check=True)
        (report / 'native.log').write_bytes(logs.stdout + logs.stderr)
    except Exception as exc:
        error = DockerError('logs', exc) if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)) else exc
        result['status'], result['log_error'] = 'failed', str(error)
    try:
        docker('stop', '--time', '10', cid, timeout=20)
    except DockerError:
        # Recheck the full owned ID/config before escalating within this container.
        if inspect_owned(cid, run_id)['State']['Running']:
            docker('kill', cid)
    stopped = inspect_owned(cid, run_id)
    if stopped['State']['Running'] or stopped['State']['Pid'] != 0:
        raise ValueError('owned container termination unconfirmed')
    result['container_state'] = stopped['State']
    docker('rm', cid)
    result['container_removed'] = True


def run(report):
    if os.geteuid() != 0:
        raise ValueError('root supervises Docker; SDK/native processes drop privileges')
    plan = verify(report)
    started, child, cid = time.monotonic(), None, None
    result = {'status': 'failed', 'scope': plan['scope']}
    def interrupted(*_):
        raise InterruptedError('operator interrupted isolated test')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        docker(*container_args(report, plan['run_id']), timeout=30)
        cid = (report / 'container.cid').read_text().strip()
        created = inspect_owned(cid, plan['run_id'])
        write(report / 'launch.json', {'argv': container_args(report, plan['run_id']), 'container_id': cid,
              'config': created['Config'], 'host_config': created['HostConfig'], 'mounts': created['Mounts']})
        docker('start', cid, timeout=30)
        state = inspect_owned(cid, plan['run_id'])['State']
        if not state['Running']:
            raise ValueError('isolated container did not start')
        process_limits(state['Pid'])
        with (report / 'sdk-console.log').open('x') as log, (report / 'metrics.jsonl').open('x') as metrics:
            child = subprocess.Popen([PYTHON, '-I', '-B', str(report / 'source/scripts/ollama/probe-runtime.py'),
                '--report-dir', str(report)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=ENV, user=UID, group=GID, extra_groups=[], preexec_fn=no_new_privileges, start_new_session=True)
            while child.poll() is None:
                if time.monotonic() - started > MAX_SECONDS or (report / 'STOP').exists():
                    raise TimeoutError('isolated test stopped or expired')
                state = inspect_owned(cid, plan['run_id'])['State']
                if not state['Running']:
                    raise ValueError('isolated container exited')
                measured = process_limits(state['Pid'])
                metrics.write(json.dumps({'monotonic': time.monotonic(), **measured}) + '\n')
                metrics.flush()
                if measured['host_available'] < MIN_RAM:
                    raise ValueError('host RAM abort threshold reached')
                time.sleep(2)
            if child.returncode != 0 or json.loads((report / 'results/result.json').read_text())['status'] != 'passed':
                raise ValueError('SDK/RuntimeService probe failed')
        result['status'] = 'passed'
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        if isinstance(exc, DockerError):
            result['docker_error'] = exc.diagnostic
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            result['sdk_reaped'] = child is None or child.returncode is not None
        except BaseException as exc:
            # A client cleanup failure must never skip native-container cleanup.
            result['status'], result['sdk_cleanup_error'] = 'failed', type(exc).__name__
        cidpath = report / 'container.cid'
        if cid is None and cidpath.exists():
            cid = cidpath.read_text().strip()
        try:
            if cid:
                stop_owned(report, cid, plan['run_id'], result)
            free_port()
            result['port_closed'] = True
            if model_inventory() != plan['model_sha256']:
                raise ValueError('readonly model inputs changed during probe')
        except BaseException as exc:
            result['status'], result['cleanup_error'] = 'failed', type(exc).__name__ + ': ' + str(exc)
        try:
            verify_source(report, plan['source_sha256'])
            result['source_verified_after'] = True
        except BaseException as exc:
            result['status'], result['source_verified_after'] = 'failed', False
            result['source_error'] = type(exc).__name__ + ': ' + str(exc)
        result['seconds'] = time.monotonic() - started
        write(report / 'result.json', result)
    if result['status'] != 'passed':
        raise RuntimeError(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.report_dir) if args.action == 'prepare' else run(args.report_dir), indent=2))
