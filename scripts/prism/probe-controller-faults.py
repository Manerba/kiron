"""One isolated native crash/recovery probe; never signals a caller-supplied PID.

The surrounding snapshot harness owns the controller, GPU watchdog and final
cleanup. This module never clears shared admission state or controls a unit.
"""
import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import signal
import time

KINDS = frozenset(('native-crash',))
MAX_SECONDS = 600
CRASH_TOKENS = 64
RECOVERY_TOKENS = 8


def _features():
    spec = importlib.util.spec_from_file_location('fault_probe_features', Path(__file__).with_name('probe-openai-features.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NativeTrace = _features().NativeTrace


def capabilities(kind, evidence):
    from kiron_common.local_inference import Capability, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    if kind not in KINDS:
        raise ValueError('unknown isolated fault probe')
    cap = Capability(CapabilityStatus.SUPPORTED, {
        'roles':P(allowed_values=('user',)), 'max_output_tokens':P(minimum=1,maximum=64),
        'default_max_output_tokens':P(allowed_values=(8,)),
        'token_budget':P(allowed_values=('max_completion_tokens',)),
    }, (evidence,))
    return CapabilitySet({N.CHAT:cap,N.STREAMING:cap})


def process_identity(pid):
    """Read only public process metadata; callers also hold the Popen owner."""
    if type(pid) is not int or pid <= 1:
        raise ValueError('invalid native PID')
    proc = Path('/proc') / str(pid)
    fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
    status = dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
    info = (proc/'exe').stat()
    return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),
        'session':int(fields[3]),'start_ticks':int(fields[19]),
        'uids':[int(v) for v in status['Uid'].split()], 'gids':[int(v) for v in status['Gid'].split()],
        'groups':[int(v) for v in status['Groups'].split()], 'no_new_privs':int(status['NoNewPrivs']),
        'exe_device':info.st_dev,'exe_inode':info.st_ino}


def guard_identity(identity, binary):
    expected = Path(binary).stat()
    pid = identity['pid']
    if (os.geteuid() == 0 or os.getgroups() or identity['state'] in ('Z','X')
            or identity['groups'] or identity['no_new_privs'] != 1
            or identity['uids'] != [os.geteuid()]*4 or identity['gids'] != [os.getegid()]*4
            or identity['ppid'] != os.getpid() or identity['pgid'] != pid or identity['session'] != pid
            or (identity['exe_device'],identity['exe_inode']) != (expected.st_dev,expected.st_ino)):
        raise ValueError('native identity is not the isolated owned child')


def same_process(left, right):
    # Scheduler state changes during normal inference; start ticks and every
    # ownership field must remain identical. Zombie/dead state is checked below.
    return {k:v for k,v in left.items() if k!='state'} == {k:v for k,v in right.items() if k!='state'}


def signal_owned_child(controller, generation, child, identity):
    """Validate twice around pidfd acquisition and signal only that descriptor."""
    if (controller.child is not child or controller.generation != generation
            or child.process.returncode is not None or child.process.pid != identity['pid']):
        raise ValueError('native ownership changed before crash injection')
    current = process_identity(child.process.pid)
    if not same_process(current,identity):
        raise ValueError('native process identity changed')
    guard_identity(current,controller.policy.binary)
    descriptor = os.pidfd_open(child.process.pid,0)
    try:
        current = process_identity(child.process.pid)
        if (controller.child is not child or controller.generation != generation
                or not same_process(current,identity)):
            raise ValueError('native ownership changed after pidfd acquisition')
        guard_identity(current,controller.policy.binary)
        signal.pidfd_send_signal(descriptor,signal.SIGKILL,None,0)
    finally:
        os.close(descriptor)


def active_ticket(tickets, operation_id, deployment_id, generation):
    key = json.dumps([generation.boot_id,generation.process_id],separators=(',',':'))
    return any(t.operation_id==operation_id and t.owner=='kiron-proxy' and t.kind=='request'
               and t.phase=='active' and t.deployment_id==deployment_id and t.generation==key for t in tickets)


def post_failure_state(tickets, operation_id, deployment_id, generation, *, cleanup_confirmed):
    key = json.dumps([generation.boot_id,generation.process_id],separators=(',',':'))
    pending = [t for t in tickets if t.operation_id==operation_id]
    if (len(pending)==1 and pending[0].phase=='unknown' and pending[0].owner=='kiron-proxy'
            and pending[0].kind=='request' and pending[0].deployment_id==deployment_id and pending[0].generation==key):
        return 'unknown'
    if not tickets and cleanup_confirmed:
        return 'terminated'  # The ordinary controller monitor may already reap.
    raise ValueError('unconfirmed native work was released before controller reconciliation')


async def probe(service, model, store, kind, *, controller, provider, report=None):
    from kiron_common.local_inference import EventKind, FinishReason, GenerationOptions, InferenceRequest, Message, MessageRole, RequestContext, TextPart
    from provider_transport import bounded_request, decode_provider_json
    if kind not in KINDS or os.geteuid()==0 or os.getgroups():
        raise ValueError('isolated non-root fault harness required')
    deadline = time.monotonic()+MAX_SECONDS
    steps = []
    operation = consumer = None

    def context(name, seconds):
        return RequestContext('fault-'+name,min(deadline,time.monotonic()+seconds),asyncio.Event())

    def record(name, **values):
        entry = {'step':name,'monotonic':time.monotonic(),**values}
        steps.append(entry)
        if report is not None:
            with (Path(report)/('fault-'+name+'.json')).open('x') as output:
                json.dump(entry,output,indent=2)

    async def health(name):
        status, body = await bounded_request(provider.control,'GET','/health',context(name,15),limit=65536)
        if status!=200:
            raise ValueError('controller health failed during fault probe')
        return decode_provider_json(body)

    async def bounded(task, seconds):
        done,_ = await asyncio.wait({task},timeout=max(0,min(seconds,deadline-time.monotonic())))
        if not done:
            raise TimeoutError('fault probe step timed out')
        return task.result()

    try:
        loaded = await service.load(model,context('initial-load',200))
        generation = loaded.observation.generation
        child = controller.child
        if child is None or generation!=controller.generation or generation.process_id is None:
            raise ValueError('loaded controller identity mismatch')
        identity = process_identity(child.process.pid)
        guard_identity(identity,controller.policy.binary)
        record('loaded',generation=asdict(generation),identity=identity)
        request = InferenceRequest(model,(Message(MessageRole.USER,(TextPart(
            'Write the integers from 1 to 100, one number per line. Continue until all numbers are written.'),)),),
            GenerationOptions(CRASH_TOKENS),context('active-stream',180))
        operation = await service.prepare(request,streaming=True)
        events = []

        async def consume():
            try:
                async for event in operation.events():
                    events.append({'kind':event.kind.value,'request_id':event.request_id,
                                   'text_bytes':len(event.text.encode()) if event.text is not None else 0})
            finally:
                await operation.close()

        consumer = asyncio.create_task(consume())
        active_deadline = min(deadline,time.monotonic()+90)
        while True:
            observed = await health('active-health')
            tickets = store.snapshot()
            if consumer.done():
                await consumer
                raise ValueError('native request ended before active crash injection')
            if (observed.get('generation')==asdict(generation) and observed.get('state')=='loaded'
                    and observed.get('active_requests')==1 and type(observed.get('slot_task_id')) is int
                    and active_ticket(tickets,request.context.request_id,model.deployment.id,generation)):
                break
            if time.monotonic()>=active_deadline:
                raise TimeoutError('active native slot was not proven')
            await asyncio.sleep(.05)
        record('before-crash',health=observed,tickets=[asdict(t) for t in tickets],identity=identity)
        signal_owned_child(controller,generation,child,identity)
        record('signal',generation=asdict(generation),identity=identity,signal='SIGKILL',target='owned-pidfd')
        await bounded(consumer,45)
        if not events or events[-1]['kind']!='failed' or any(v['kind']=='completed' for v in events):
            raise ValueError('crashed stream did not have exactly a failed completion')
        tickets = store.snapshot()
        state = post_failure_state(tickets,request.context.request_id,model.deployment.id,generation,
            cleanup_confirmed=controller.child is None and child.process.returncode is not None)
        record('stream-failed',events=events,tickets=[asdict(t) for t in tickets],work_state=state)
        cleanup_deadline = min(deadline,time.monotonic()+45)
        while True:
            observed = await health('reconcile')
            if (controller.child is None and child.process.returncode is not None and observed.get('state')=='unloaded'
                    and not store.snapshot() and not Path('/proc',str(identity['pid'])).exists()):
                break
            if time.monotonic()>=cleanup_deadline:
                raise ValueError('controller did not confirm crashed process cleanup')
            await asyncio.sleep(.05)
        record('reconciled',health=observed,returncode=child.process.returncode,tickets=[])
        loaded = await service.load(model,context('recovery-load',200))
        current = loaded.observation.generation
        if current.boot_id!=generation.boot_id or current.process_id in (None,generation.process_id):
            raise ValueError('recovery did not create a new native generation')
        # Old generation token must be rejected before any native request work.
        status,_ = await bounded_request(provider.inference,'POST','/v1/chat/completions',context('old-token',15),limit=65536,
            json={'model':'old-generation','messages':[{'role':'user','content':'Do not run'}],'max_tokens':1},
            headers={'Authorization':'Bearer kiron-prism-'+generation.process_id})
        if status!=401:
            raise ValueError('old process generation was not rejected')
        recovery = InferenceRequest(model,(Message(MessageRole.USER,(TextPart('Reply only OK.'),)),),
                                    GenerationOptions(RECOVERY_TOKENS),context('recovery-text',90))
        answer = await service.chat(recovery)
        if (answer.finish_reason is not FinishReason.STOP or ''.join(p.text for p in answer.content).strip()!='OK'
                or answer.usage is None or not 1<=answer.usage.output_tokens<=RECOVERY_TOKENS):
            raise ValueError('recovery text probe failed')
        record('recovered',generation=asdict(current),old_token_status=status,text='OK',usage=asdict(answer.usage))
        await service.unload(model,context('unload',60))
        observed = await health('final-health')
        if controller.child is not None or observed.get('state')!='unloaded' or store.snapshot():
            raise ValueError('fault recovery left native work or admission behind')
        record('unloaded',health=observed,tickets=[])
        return {'kind':kind,'steps':steps,'native_generations':2,'recovery_max_output_tokens':RECOVERY_TOKENS}
    finally:
        if consumer is not None and not consumer.done():
            consumer.cancel()
            done,_ = await asyncio.wait({consumer},timeout=5)
            if not done:
                consumer.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                record('cleanup-pending',consumer=True)
        if operation is not None and (consumer is None or consumer.done()):
            close = asyncio.create_task(operation.close())
            done,_ = await asyncio.wait({close},timeout=5)
            if not done:
                close.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
                record('operation-cleanup-pending',operation=True)
            else:
                close.result()
