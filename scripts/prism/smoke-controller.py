#!/usr/bin/env python3
"""Prepare/run one isolated controller smoke; never install units or touch production state.

`prepare --report-dir .../reports/controller-NAME` verifies pins and creates only
an isolated registry/admission/socket/report fixture. `run` explicitly performs
one bounded feature workflow using the real controller, UDS, PrismProvider and RuntimeService.
Run requires root solely for dropping to nobody:kiron-common before any controller
or server starts. The process needs httpx, starlette and uvicorn in its TEST venv.
Resource thresholds are observation/abort guards, not hard cgroup limits.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from datetime import datetime, timezone
import grp
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import stat
import subprocess
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
TEST_ROOT = Path('/usr/lib/kiron/test-runtimes/prism')
REPORT_ROOT = TEST_ROOT / 'reports'
BUNDLE = TEST_ROOT / 'bundles/prism-9a9394a-sm86-tokens-v1'
MODEL = Path('/usr/lib/kiron/data/gguf-models/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf')
PROJECTOR = TEST_ROOT / 'artifacts/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'
RUNTIME = BUNDLE / 'runtime'
MODEL_SHA = '3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1'
PROJECTOR_SHA = '6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903'
MODEL_BYTES, PROJECTOR_BYTES = 7206168928, 629246976
BUNDLE_MANIFEST_SHA = '3c6d8369a57cf1809d489d1e48771f1fe2fe556b30ee1844fbbc3be93ef4ee00'
PROFILE_ID = 'prism-bonsai27b-cuda40-c1024-v1'
PORT = 18089
MAX_SECONDS = 600
MIN_RAM, MAX_GPU = 2 * 1024**3, 11 * 1024**3
PROBES = ('lifecycle', 'public-api', 'public-tools', 'public-vision', 'public-structured', 'public-reasoning',
          'public-tools-roundtrip', 'public-responses', 'public-responses-budget', 'native-crash',
          'dashboard-runtime', 'runtime-soak', 'native-cancel')
FEATURE_PROBES = frozenset(kind for kind in PROBES[2:] if kind != 'dashboard-runtime')
CONTROLLER_PROBES = frozenset(('native-crash', 'runtime-soak', 'native-cancel'))
SELF_UNLOADING_PROBES = CONTROLLER_PROBES | {'dashboard-runtime'}
MAX_SOURCE_FILE = 4 * 1024**2


def imports():
    for name in ('kiron-proxy', 'kiron-prism', 'kiron-common'):
        value = str(WORKSPACE / 'services' / name)
        if value not in sys.path:
            sys.path.insert(0, value)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, mode=0o640, exclusive=True):
    with path.open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=lambda item:
                  item.isoformat() if isinstance(item, datetime) else item.value)
        stream.write('\n')
    path.chmod(mode)


def report_value(value):
    # Common immutable mappings are MappingProxyType; dataclasses.asdict deepcopies
    # them and fails. Serialize their values without mutating/copying live objects.
    if is_dataclass(value):
        return {field.name: report_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: report_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [report_value(item) for item in value]
    return value


def report_path(path):
    if (not path.is_absolute() or path.parent != REPORT_ROOT or path.name in {'.', '..'}
            or not re.fullmatch(r'controller-[A-Za-z0-9_-]+', path.name)
            or path != path.resolve() or len(str(path / 'uds/control.sock').encode()) >= 108):
        raise ValueError('report must be a new canonical controller-NAME under the isolated reports root')
    return path


def code_inventory(root=None):
    """Only source/package metadata and fixed synthetic fixtures; never runtime data."""
    root = WORKSPACE if root is None else root
    paths = {root / 'scripts/prism' / name for name in
             ('smoke-controller.py', 'package-runtime.py', 'probe-openai-api.py', 'probe-openai-features.py',
              'probe-openai-responses.py', 'probe-controller-faults.py', 'probe-dashboard-runtime.py',
              'probe-runtime-soak.py', 'probe-controller-cancel.py')}
    paths.update(root / 'scripts/prism/fixtures' / name for name in ('red.png', 'blue.png'))
    paths.update(root / 'services/kiron-proxy' / name for name in
                 ('static/js/tab_models.js', 'static/css/models.css', 'templates/index.html'))
    common = root / 'services/kiron-common'
    paths.add(common / 'pyproject.toml')
    paths.update(path for path in (common / 'kiron_common').rglob('*') if path.suffix in {'.py', '.json'})
    for service in ('kiron-proxy', 'kiron-prism'):
        directory = root / 'services' / service
        paths.update(path for path in directory.rglob('*.py')
                     if not any(part in {'tests', 'contracts', '__pycache__'} for part in path.relative_to(directory).parts)
                     and not path.name.startswith('test_') and path.name != 'conftest.py')
    for service in ('kiron-common', 'kiron-proxy', 'kiron-prism'):
        paths.update((root / 'services' / service).glob('requirements*.txt'))
    result = {}
    for path in sorted(paths):
        info = path.lstat()
        if path.resolve() != path or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SOURCE_FILE:
            raise ValueError('source inventory requires canonical regular files')
        result[str(path.relative_to(root))] = sha(path)
    return result


def snapshot_source(report, gid):
    expected, destination = code_inventory(), report / 'source'
    destination.mkdir(mode=0o750)
    for relative, digest in expected.items():
        source, target = WORKSPACE / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('source snapshot requires regular files')
            data = stream.read(MAX_SOURCE_FILE + 1)
        if len(data) > MAX_SOURCE_FILE:
            raise ValueError('source file exceeds snapshot bound')
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('source changed while preparing snapshot')
        with target.open('xb') as stream:
            stream.write(data)
        os.chown(target, 0, gid)
        target.chmod(0o440)
    if code_inventory() != expected or code_inventory(destination) != expected:
        raise ValueError('source changed while preparing snapshot')
    for directory in (destination, *(path for path in destination.rglob('*') if path.is_dir())):
        os.chown(directory, 0, gid)
        directory.chmod(0o550)
    return expected


def verify_snapshot(report, expected):
    root = report / 'source'
    actual = set()
    for path in (root, *root.rglob('*')):
        info = path.lstat()
        directory = stat.S_ISDIR(info.st_mode)
        if (info.st_uid != 0 or info.st_gid != 982 or stat.S_IMODE(info.st_mode) != (0o550 if directory else 0o440)
                or not (directory or stat.S_ISREG(info.st_mode)) or path.resolve() != path):
            raise ValueError('unsafe immutable source snapshot')
        if not directory:
            actual.add(str(path.relative_to(root)))
    if actual != set(expected) or code_inventory(root) != expected:
        raise ValueError('source snapshot changed')


def require_snapshot_execution(report):
    expected = report / 'source/scripts/prism/smoke-controller.py'
    if Path(__file__).resolve() != expected or WORKSPACE != report / 'source':
        raise ValueError('run only the prepared report/source/scripts/prism/smoke-controller.py')
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('run the snapshot with TEST-venv Python -I -B')


def artifact_layout(name):
    """Two fixed read-only artifact layouts; all execution stays isolated."""
    if name == 'test':
        return BUNDLE / 'runtime', TEST_ROOT / 'artifacts/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'
    if name == 'production':
        return (Path('/usr/lib/kiron/runtimes/prism/9a9394a-sm86-tokens-v1'),
                MODEL.parent / 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf')
    raise ValueError('unknown artifact layout')


def policy():
    imports()
    from kiron_common.prism_runtime_policy import Policy, Profile
    manifest_file = BUNDLE / 'bundle-manifest.json'
    if sha(manifest_file) != BUNDLE_MANIFEST_SHA:
        raise ValueError('unexpected bundle manifest')
    profile = Profile(PROFILE_ID, 40, 4, MODEL_SHA, 'qwen35', 4608 << 20, 8192 << 20, 512 << 20,
                      projector_sha256=PROJECTOR_SHA)
    # Deliberate test-only dependency injection, never a permissive Policy.load.
    root = RUNTIME
    return Policy(root, root / 'llama-server', json.loads(manifest_file.read_text()), (root,),
                  (MODEL.parent, PROJECTOR.parent), {PROFILE_ID: profile}, port=PORT)


def entry():
    imports()
    from kiron_common.local_model_registry import RegistryEntry
    from kiron_common.local_model_registry.models import LocalArtifactFile
    from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
    return RegistryEntry.create(runtime_provider=BackendType.PRISM, artifact_origin=ArtifactType.LOCAL,
        artifact_format=ArtifactFormat.GGUF, reference=str(MODEL), display_name='Isolated controller smoke',
        loader=LoaderType.PRISM_GGUF, sha256=MODEL_SHA, size_bytes=MODEL_BYTES,
        projector=LocalArtifactFile(str(PROJECTOR), PROJECTOR_SHA, PROJECTOR_BYTES), runtime_profile=PROFILE_ID)


def memory():
    result = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=memory.free,memory.used',
        '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=2, check=True, env={'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise ValueError('exactly one GPU required')
    free, used = (int(value.strip()) * 1024**2 for value in rows[0].split(','))
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    ram = int(values['MemAvailable'].split()[0]) * 1024
    if free < 0 or used < 0 or ram <= 0:
        raise ValueError('invalid GPU/RAM measurement')
    return {'gpu_free_bytes': free, 'gpu_used_bytes': used, 'host_available_bytes': ram,
            'monotonic': time.monotonic()}


def check_memory(values):
    if values['gpu_used_bytes'] > MAX_GPU or values['host_available_bytes'] < MIN_RAM:
        raise ValueError('isolated smoke resource abort threshold reached')


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', PORT))


def prepare(report, probe='lifecycle', layout='test'):
    global RUNTIME, PROJECTOR
    RUNTIME, PROJECTOR = artifact_layout(layout)
    if os.geteuid() != 0:
        raise ValueError('prepare requires root to create the immutable test fixture')
    report_path(report)
    if probe not in PROBES:
        raise ValueError('unknown isolated probe')
    if report.exists():
        raise ValueError('never overwrite a smoke report')
    imports()
    from kiron_common.local_model_registry import RuntimeModelRegistry, RegistryFilePolicy
    controlled = policy()
    controlled.verify_bundle()
    descriptors = []
    try:
        descriptors.append(controlled.open_artifact(str(MODEL), MODEL_SHA, MODEL_BYTES))
        descriptors.append(controlled.open_artifact(str(PROJECTOR), PROJECTOR_SHA, PROJECTOR_BYTES))
        controlled.verify_metadata(controlled.profiles[PROFILE_ID], *descriptors)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    free_port()
    measured = memory()
    check_memory(measured)
    if (measured['gpu_free_bytes'] < (4608 + 512) << 20
            or measured['host_available_bytes'] < 8192 << 20):
        raise ValueError('measured profile plus headroom does not fit')
    nobody, gid = pwd.getpwnam('nobody').pw_uid, grp.getgrnam('kiron-common').gr_gid
    if gid != 982:
        raise ValueError('this host fixture pins kiron-common gid 982')
    report.mkdir(mode=0o750)
    os.chown(report, 0, gid)
    report.chmod(0o750)
    for name, owner, mode in (('registry', 0, 0o750), ('admission', 0, 0o2770),
                              ('uds', nobody, 0o750), ('results', nobody, 0o750)):
        path = report / name
        path.mkdir()
        os.chown(path, owner, gid)
        path.chmod(mode)
    inventory = snapshot_source(report, gid)
    registration = entry()
    registry = RuntimeModelRegistry(report / 'registry/models.json', file_policy=RegistryFilePolicy(0, gid))
    registry.add(registration)
    plan = {'scope': 'isolated source-snapshot controller smoke; no production registry, admission, unit or model copy',
            'model_id': registration.id, 'probe': probe, 'port': PORT, 'profile': PROFILE_ID, 'uid': nobody, 'gid': gid,
            'bundle': str(BUNDLE), 'bundle_manifest_sha256': BUNDLE_MANIFEST_SHA,
            'artifact_layout': layout, 'runtime_root': str(RUNTIME), 'projector_path': str(PROJECTOR),
            'source_sha256': inventory, 'source_root': str(report / 'source'),
            'registry_sha256': sha(report / 'registry/models.json'),
            'model_sha256': MODEL_SHA, 'projector_sha256': PROJECTOR_SHA, 'baseline_memory': measured,
            'max_seconds': MAX_SECONDS, 'prepared_at': datetime.now(timezone.utc).isoformat(),
            'capability_scope': 'Harness-only ' + probe + ' candidate authorization to measure the pinned cases; no capability publication.'}
    write_json(report / 'plan.json', plan, mode=0o440)
    os.chown(report / 'plan.json', 0, gid)
    return plan


def verify_plan(report):
    global RUNTIME, PROJECTOR
    report_path(report)
    plan_path = report / 'plan.json'
    for path in (report, plan_path, report / 'registry', report / 'registry/models.json',
                 report / 'registry/models.json.lock'):
        info = path.lstat()
        if (info.st_uid != 0 or info.st_gid != 982 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode)):
            raise ValueError('unsafe immutable smoke fixture')
    plan = json.loads(plan_path.read_text())
    if stat.S_IMODE(plan_path.stat().st_mode) != 0o440:
        raise ValueError('unsafe immutable smoke plan')
    verify_snapshot(report, plan['source_sha256'])
    RUNTIME, PROJECTOR = artifact_layout(plan['artifact_layout'])
    if (plan['probe'] not in PROBES or plan['source_root'] != str(report / 'source')
            or plan['runtime_root'] != str(RUNTIME) or plan['projector_path'] != str(PROJECTOR)
            or plan['registry_sha256'] != sha(report / 'registry/models.json')
            or plan['model_id'] != entry().id or plan['bundle_manifest_sha256'] != BUNDLE_MANIFEST_SHA
            or plan['bundle'] != str(BUNDLE) or plan['profile'] != PROFILE_ID or plan['max_seconds'] != MAX_SECONDS
            or plan['model_sha256'] != MODEL_SHA or plan['projector_sha256'] != PROJECTOR_SHA
            or plan['port'] != PORT or plan['uid'] != pwd.getpwnam('nobody').pw_uid or plan['gid'] != 982):
        raise ValueError('prepared fixture changed; prepare a new report')
    if any((report / 'results').iterdir()):
        raise ValueError('never rerun an existing smoke report')
    free_port()
    return plan


def context(name, seconds):
    from kiron_common.local_inference import RequestContext
    return RequestContext(name, time.monotonic() + seconds, asyncio.Event())


def probe_module(kind):
    if kind not in PROBES[1:]:
        raise ValueError('unknown public probe')
    name = ('probe-controller-cancel.py' if kind == 'native-cancel' else
            'probe-runtime-soak.py' if kind == 'runtime-soak' else
            'probe-dashboard-runtime.py' if kind == 'dashboard-runtime' else
            'probe-controller-faults.py' if kind == 'native-crash' else
            'probe-openai-responses.py' if kind in ('public-responses', 'public-responses-budget') else
            'probe-openai-features.py' if kind in FEATURE_PROBES else 'probe-openai-api.py')
    spec = importlib.util.spec_from_file_location('isolated_' + name.replace('-', '_').removesuffix('.py'),
                                                 Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_capabilities(kind, evidence, module=None):
    from kiron_common.local_inference import Capability, CapabilityName, CapabilitySet, CapabilityStatus, ParameterConstraint
    if kind not in PROBES:
        raise ValueError('unknown isolated probe')
    if kind == 'dashboard-runtime':
        return CapabilitySet({})  # Lifecycle evidence grants no inference feature.
    if kind in FEATURE_PROBES:
        return module.capabilities(kind, evidence)
    capability = Capability(CapabilityStatus.SUPPORTED, {
        'roles': ParameterConstraint(allowed_values=('user',)),
        'max_output_tokens': ParameterConstraint(minimum=1, maximum=16),
        'default_max_output_tokens': ParameterConstraint(allowed_values=(16,)),
        'token_budget': ParameterConstraint(allowed_values=('max_tokens', 'max_completion_tokens')),
    }, (evidence,))
    return CapabilitySet({CapabilityName.CHAT: capability,
        **({CapabilityName.STREAMING: capability} if kind == 'public-api' else {})})


async def public_probe(kind, module, service, model, store, report, *, controller=None, provider=None):
    if kind in CONTROLLER_PROBES:
        return await module.probe(service, model, store, kind, controller=controller, provider=provider,
                                  report=report / 'results')
    if kind in FEATURE_PROBES or kind == 'dashboard-runtime':
        return await module.probe(service, model, store, kind, report=report / 'results')
    if kind == 'public-api':
        return await module.probe(service, model, store, report=report / 'results')
    raise ValueError('unknown public probe')


async def verify_foreign_bearer(control, inference):
    """Reject foreign tokens while the loaded slot is verified idle."""
    from provider_transport import bounded_request

    async def health():
        status, body = await bounded_request(control, 'GET', '/health', context('auth-health', 10), limit=65536)
        value = json.loads(body)
        if status != 200 or value['state'] != 'loaded' or value['active_requests'] != 0:
            raise ValueError('auth negative probe needs verified idle loaded slot')
        return value

    before = await health()
    statuses = []
    foreign = 'kiron-prism-' + ('0' * 32 if before['backend_model'] != 'kiron-prism-' + '0' * 32 else '1' * 32)
    for key in ('invalid-controller-smoke-token', foreign):
        status, _ = await bounded_request(inference, 'POST', '/v1/chat/completions', context('negative-auth', 10),
            limit=65536, headers={'Authorization': 'Bearer ' + key},
            json={'model': before['backend_model'], 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                  'max_tokens': 1, 'stream': False})
        if status != 401:
            raise ValueError('native backend did not reject foreign generation bearer')
        after = await health()
        if (after['generation'] != before['generation'] or after['slot_task_id'] != before['slot_task_id']):
            raise ValueError('native slot changed during rejected request')
        statuses.append(status)
    return {'statuses': statuses, 'slot_task_id': before['slot_task_id'], 'active_requests': 0,
            'scope': 'Wrong and foreign-generation token rejected; no second model load or historical-token replay claimed.'}


def consume_task(task):
    if not task.cancelled():
        task.exception()


async def settle_tasks(tasks, timeout):
    """Bound our wait even when an injected/native awaitable resists cancellation.

    Remaining work is a failed cleanup, never backend-termination evidence.
    asyncio.run may still wait for stubborn tasks/executor work when closing;
    the operator's external process supervision is the final time boundary.
    """
    tasks = {task for task in tasks if task is not None}
    if not tasks:
        return set()
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in done:
        consume_task(task)
    for task in pending:
        task.add_done_callback(consume_task)
    return pending


async def execute(report, plan):
    require_snapshot_execution(report)
    import httpx
    import uvicorn
    from main import ControlApp, control_socket
    from controller import Controller
    from composition import RegistryResolver
    from admission import ControllerAdmission
    from prism_provider import PrismProvider
    from runtime_service import RuntimeService
    from runtime_composition import ArtifactVerifier, adapter_revision
    from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
    from kiron_common.local_model_registry import RuntimeModelRegistry, RegistryFilePolicy
    from kiron_common.local_inference import (CapabilityEvidence, GenerationOptions, InferenceRequest, Message, MessageRole,
        RuntimeImplementation, RuntimeTimeouts, TextPart)
    from kiron_common.model_catalog import BackendType
    if os.geteuid() != plan['uid'] or os.getegid() != plan['gid'] or os.getgroups():
        raise ValueError('controller must run as nobody:kiron-common without supplemental groups')
    controlled = policy()
    registry = RuntimeModelRegistry(report / 'registry/models.json', readonly=True,
                                   file_policy=RegistryFilePolicy(0, plan['gid']))
    resolver = RegistryResolver(registry, controlled.resource_profiles())
    store = AdmissionStore(report / 'admission', security=RuntimeSecurity(0, plan['gid'], frozenset({0, plan['uid']})))
    controller = Controller(controlled, resolver, ControllerAdmission(store))

    class Server(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            yield  # SIGTERM cancels workflow, then finally drains the real controller.

    result = {'status': 'failed', 'scope': plan['scope'], 'started_at': datetime.now(timezone.utc).isoformat()}
    workflow_task = monitor_task = server_task = service = None
    with control_socket(report / 'uds/control.sock', control_gid=plan['gid']) as sock:
        server = Server(uvicorn.Config(ControlApp(controller), access_log=False, workers=1,
                                       proxy_headers=False, timeout_graceful_shutdown=5))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if server_task.done():
                        await server_task
                        raise RuntimeError('controller server exited before readiness')
                    await asyncio.sleep(0.05)
            model = (await resolver.snapshot()).resolve(plan['model_id'])
            implementation = RuntimeImplementation(controlled.runtime_revision, None, adapter_revision('prism_provider.py'))
            evidence = CapabilityEvidence(implementation.provider_revision, model.deployment.artifact_identity.fingerprint,
                MODEL_SHA, PROJECTOR_SHA, implementation.template_revision, implementation.parser_revision,
                model.deployment.configuration_fingerprint, plan['capability_scope'], datetime.now(timezone.utc))
            module = probe_module(plan['probe']) if plan['probe'] != 'lifecycle' else None
            capabilities = candidate_capabilities(plan['probe'], evidence, module)
            trace_module = (module if plan['probe'] in FEATURE_PROBES else
                            probe_module('public-tools') if plan['probe'] == 'public-api' else None)
            # Context deadlines own the complete operation; httpx's implicit 5s
            # read timeout would abort verified multi-GB hashing during load.
            provider = PrismProvider(control=httpx.AsyncClient(base_url='http://prism-control', trust_env=False, timeout=None,
                transport=httpx.AsyncHTTPTransport(uds=str(report / 'uds/control.sock'))),
                inference=httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', trust_env=False, timeout=None,
                    **({'transport': trace_module.NativeTrace(report / 'results')} if trace_module is not None else {})), resolver=resolver,
                implementation=implementation, verify_artifact=ArtifactVerifier(controlled),
                capabilities={model.deployment.id: capabilities})

            def measure():
                value = memory()
                check_memory(value)
                return MemorySnapshot(value['gpu_free_bytes'], value['host_available_bytes'], value['monotonic'])

            service = RuntimeService(resolver=resolver, providers={BackendType.PRISM: provider}, admission=store,
                measure=measure, timeouts=RuntimeTimeouts(180, 10, 120, 60, 240, 30, 15))

            async def workflow():
                result['before'] = report_value(await provider.health(context('initial-health', 10)))
                if plan['probe'] != 'lifecycle':
                    key = ('native_cancel' if plan['probe'] == 'native-cancel' else
                           'native_soak' if plan['probe'] == 'runtime-soak' else
                           'dashboard_runtime' if plan['probe'] == 'dashboard-runtime' else
                           'native_fault' if plan['probe'] == 'native-crash' else
                           'public_features' if plan['probe'] in FEATURE_PROBES else 'public_api')
                    result[key] = await public_probe(plan['probe'], module, service, model, store, report,
                                                     controller=controller, provider=provider)
                else:
                    result['loaded'] = report_value(await service.load(model, context('load-smoke', 200)))
                    request = InferenceRequest(model, (Message(MessageRole.USER, (TextPart('Reply with exactly OK.'),)),),
                                               GenerationOptions(16), context('text-smoke', 240))
                    result['text'] = report_value(await service.chat(request))
                    if not any(part.get('text', '').strip() for part in result['text']['content']):
                        raise ValueError('text probe returned no text')
                if plan['probe'] not in SELF_UNLOADING_PROBES:
                    result['loaded_health'] = report_value(await provider.health(context('loaded-health', 10)))
                    result['foreign_bearer'] = await verify_foreign_bearer(provider.control, provider.inference)
                    result['unloaded'] = report_value(await service.unload(model, context('unload-smoke', 60)))
                result['after'] = report_value(await provider.health(context('final-health', 10)))
                if store.snapshot():
                    raise ValueError('admission tickets remain after confirmed unload')
                result['status'] = 'passed'

            async def monitor():
                with (report / 'results/metrics.jsonl').open('x') as stream:
                    while True:
                        value = await asyncio.to_thread(memory)
                        child = controller.child
                        if child is not None:
                            value['native_pid'] = child.process.pid
                            try:
                                value['native_start_ticks'] = Path(f'/proc/{child.process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                            except FileNotFoundError:
                                pass
                        stream.write(json.dumps(value) + '\n')
                        stream.flush()
                        check_memory(value)
                        if (report / 'STOP').exists():
                            raise RuntimeError('operator STOP requested')
                        await asyncio.sleep(2)

            workflow_task, monitor_task = asyncio.create_task(workflow()), asyncio.create_task(monitor())
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signum, workflow_task.cancel)
            done, _ = await asyncio.wait((workflow_task, monitor_task), timeout=MAX_SECONDS,
                                         return_when=asyncio.FIRST_COMPLETED)
            if monitor_task in done:
                await monitor_task
            if workflow_task not in done:
                raise TimeoutError('controller smoke lifetime expired')
            await workflow_task
        except BaseException as error:
            result['status'], result['error'] = 'failed', type(error).__name__ + ': ' + str(error)
            raise
        finally:
            server.should_exit = True
            for task in (workflow_task, monitor_task):
                if task is not None and not task.done():
                    task.cancel()
            pending = await settle_tasks((workflow_task, monitor_task), 5)
            if pending:
                result['status'], result['cancel_cleanup_pending'] = 'failed', len(pending)
            try:
                if await settle_tasks((server_task,), 65):
                    raise TimeoutError('controller server cleanup deadline expired')
                server_task.result()
                result['controller_cleanup_confirmed'] = controller.child is None and controller.deployment is None
            except BaseException as error:
                result['status'], result['cleanup_error'] = 'failed', type(error).__name__
                result['controller_cleanup_confirmed'] = False
            if service is not None:
                service_close = asyncio.create_task(service.aclose())
                try:
                    if await settle_tasks((service_close,), 65):
                        service_close.cancel()
                        raise TimeoutError('runtime service cleanup deadline expired')
                    service_close.result()
                except BaseException as error:
                    result['status'], result['service_cleanup_error'] = 'failed', type(error).__name__
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            write_json(report / 'results/result.json', result)
    return result


def no_new_privileges():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'no_new_privs failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--probe', choices=PROBES, help='prepare only; run uses the pinned plan')
    parser.add_argument('--artifact-layout', choices=('test', 'production'),
                        help='prepare only; read pinned artifacts at test or production paths')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.report_dir, args.probe or 'lifecycle', args.artifact_layout or 'test'), indent=2))
        return
    if args.probe is not None or args.artifact_layout is not None:
        parser.error('run uses the probe and artifact layout in the prepared plan')
    if os.geteuid() != 0:
        parser.error('run starts as root only to verify the fixture and drop privileges immediately')
    require_snapshot_execution(args.report_dir)
    plan = verify_plan(args.report_dir)
    imports()
    for module in ('httpx', 'starlette', 'uvicorn', *(('openai', 'pymysql') if plan['probe'] != 'lifecycle' else ()),
                   *(('PIL',) if plan['probe'] == 'public-vision' else ()),
                   *(('fastapi', 'psutil') if plan['probe'] == 'dashboard-runtime' else ())):
        if importlib.util.find_spec(module) is None:
            parser.error(f'{module} missing from the isolated TEST venv')
    os.environ.clear()
    os.environ.update(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    no_new_privileges()
    os.setgroups([])
    os.setgid(plan['gid'])
    os.setuid(plan['uid'])
    result = asyncio.run(execute(args.report_dir, plan))
    if result['status'] != 'passed' or not result['controller_cleanup_confirmed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
