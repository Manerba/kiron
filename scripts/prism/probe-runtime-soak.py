"""Bounded isolated native soak; the enclosing harness owns all process cleanup.

Six short requests plus idle health observations prove only this pinned test
generation. The competing admission is deliberately impossible, starts no
process, and uses the same private store. This is not a production load test.
"""
import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import time

KINDS = frozenset(('runtime-soak',))
ROUNDS, TOKENS, TOTAL_TOKENS = 6, 8, 64
MIN_SECONDS, MAX_SECONDS, IDLE_SECONDS = 120, 300, 5


def _features():
    spec = importlib.util.spec_from_file_location('soak_features', Path(__file__).with_name('probe-openai-features.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NativeTrace = _features().NativeTrace


def capabilities(kind, evidence):
    from kiron_common.local_inference import Capability, CapabilityName as N, CapabilitySet, CapabilityStatus, ParameterConstraint as P
    if kind not in KINDS:
        raise ValueError('unknown isolated soak probe')
    cap = Capability(CapabilityStatus.SUPPORTED, {
        'roles': P(allowed_values=('user',)), 'max_output_tokens': P(minimum=1, maximum=TOKENS),
        'default_max_output_tokens': P(allowed_values=(TOKENS,)),
        'token_budget': P(allowed_values=('max_completion_tokens',)),
    }, (evidence,))
    return CapabilitySet({N.CHAT: cap, N.STREAMING: cap})


def resident_tickets(store, model, generation):
    from runtime_service import generation_key
    tickets = store.snapshot()
    if (len(tickets) != 1 or any(t.kind != 'load' or t.phase != 'resident' or t.owner != 'kiron-proxy'
            or t.deployment_id != model.deployment.id or t.generation != generation_key(generation) for t in tickets)):
        raise ValueError('soak must retain exactly its resident ticket between rounds')
    return tickets


def _observe(task):
    if not task.cancelled():
        task.exception()


async def probe(service, model, store, kind, *, controller, provider, report=None):
    from kiron_common.gpu_admission import AdmissionError
    from kiron_common.local_inference import (EventKind, FinishReason, GenerationOptions, InferenceRequest,
        Message, MessageRole, RequestContext, TextPart)
    from kiron_common.local_inference.events import validate_event_sequence
    from provider_transport import bounded_request, decode_provider_json
    from runtime_service import generation_key
    if kind not in KINDS or os.geteuid() == 0 or os.getgroups():
        raise ValueError('isolated non-root soak harness required')
    if store is not service.admission or store.snapshot():
        raise ValueError('soak requires the empty injected admission store')
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    cancellation = asyncio.Event()
    tasks, records = set(), []

    def context(name, seconds):
        if cancellation.is_set():
            raise asyncio.CancelledError
        return RequestContext('soak-' + name, min(deadline, time.monotonic() + seconds), cancellation)

    def record(name, **values):
        if cancellation.is_set():
            raise asyncio.CancelledError
        value = {'step': name, 'elapsed_seconds': time.monotonic() - started, **values}
        records.append(value)
        if report is not None:
            with (Path(report) / ('soak-' + name + '.json')).open('x') as output:
                json.dump(value, output, indent=2)

    async def health(name, generation, child, *, idle):
        status, body = await bounded_request(provider.control, 'GET', '/health', context(name, 10), limit=65536)
        value = decode_provider_json(body)
        if (status != 200 or type(value) is not dict or value.get('generation') != asdict(generation)
                or value.get('state') != 'loaded' or controller.generation != generation
                or controller.child is not child or not child.alive()
                or (idle and value.get('active_requests') != 0)):
            raise ValueError('soak native health or generation changed')
        return value

    async def competition(consumer, request, generation, child):
        physical_host = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        if physical_host <= 0:
            raise ValueError('physical host memory unavailable')
        while not consumer.done():
            observed = await health('competition-health', generation, child, idle=False)
            active = [t for t in store.snapshot() if t.operation_id == request.context.request_id
                      and t.kind == 'request' and t.phase == 'active' and t.owner == 'kiron-proxy'
                      and t.deployment_id == model.deployment.id and t.generation == generation_key(generation)]
            if (observed.get('active_requests') == 1 and type(observed.get('slot_task_id')) is int
                    and observed['slot_task_id'] >= 0 and len(active) == 1 and not consumer.done()):
                break
            await asyncio.sleep(.05)
        else:
            raise ValueError('native request ended before concurrent admission was observed')
        operation_id = 'soak-impossible-load'
        measurement = {}

        def measure():
            memory = service.measure()
            if memory.host_available_bytes > physical_host:
                raise ValueError('host measurement exceeds physical memory')
            measurement.update(asdict(memory))
            return memory

        try:
            await asyncio.to_thread(store.reserve, operation_id=operation_id, owner='isolated-soak',
                generation=generation_key(generation), deployment_id='isolated-impossible', kind='load',
                gpu_bytes=0, host_bytes=physical_host + 1, measure=measure, allow_existing=False)
        except AdmissionError as exc:
            if exc.code != 'resource_exhausted' or not measurement:
                raise ValueError('concurrent admission was not rejected by the measured resource budget') from exc
        else:
            raise ValueError('impossible competing admission was accepted')
        if any(t.operation_id == operation_id for t in store.snapshot()):
            raise ValueError('rejected competing admission left a ticket')
        record('competition', code='resource_exhausted', native_health=observed,
               active_request_id=request.context.request_id, physical_host_bytes=physical_host,
               requested_host_bytes=physical_host + 1, measurement=measurement,
               scope='private-store admission only; no competing native process')

    async def workflow():
        loaded = await service.load(model, context('load', 180))
        generation, child = loaded.observation.generation, controller.child
        if child is None or generation != controller.generation or generation.process_id is None:
            raise ValueError('soak initial native generation missing')
        soak_started = time.monotonic()
        await health('loaded-health', generation, child, idle=True)
        resident_tickets(store, model, generation)
        record('loaded', generation=asdict(generation))
        total = 0
        for index in range(ROUNDS):
            request = InferenceRequest(model, (Message(MessageRole.USER, (TextPart('Reply only OK.'),)),),
                                       GenerationOptions(TOKENS), context('round-' + str(index + 1), 60))
            streaming = index % 2 == 1
            if streaming:
                operation = await service.prepare(request, streaming=True)
                events = []

                async def consume():
                    try:
                        async for event in operation.events():
                            events.append(event)
                            if len(events) > 128 or sum(len(e.text or '') for e in events) > 1024:
                                raise ValueError('soak stream exceeded fixture bounds')
                    finally:
                        await operation.close()

                consumer = asyncio.create_task(consume())
                tasks.add(consumer)
                if index == 1:
                    contender = asyncio.create_task(competition(consumer, request, generation, child))
                    tasks.add(contender)
                    await asyncio.gather(consumer, contender)
                else:
                    await consumer
                validate_event_sequence(events)
                usages = [event.usage for event in events if event.kind is EventKind.USAGE]
                if (len(usages) != 1 or any(e.kind not in {EventKind.STARTED, EventKind.TEXT_DELTA,
                        EventKind.USAGE, EventKind.COMPLETED} for e in events)):
                    raise ValueError('soak stream lacks exact plain-text usage')
                usage, finish = usages[0], events[-1].finish_reason
                text = ''.join(event.text or '' for event in events)
            else:
                answer = await service.chat(request)
                usage, finish = answer.usage, answer.finish_reason
                text = ''.join(part.text for part in answer.content)
                if answer.request_id != request.context.request_id or answer.tool_calls or answer.reasoning:
                    raise ValueError('soak response identity or content differs')
            if (finish is not FinishReason.STOP or text.strip() != 'OK' or usage is None
                    or usage.input_tokens < 1 or not 1 <= usage.output_tokens <= TOKENS):
                raise ValueError('soak output or exact usage failed')
            total += usage.output_tokens
            if total > TOTAL_TOKENS:
                raise ValueError('soak total output budget exceeded')
            resident_tickets(store, model, generation)
            observed = await health('round-health', generation, child, idle=True)
            record('round-' + str(index + 1), streaming=streaming, text=text, usage=asdict(usage), health=observed)
        idle_count = 0
        while time.monotonic() - soak_started < MIN_SECONDS:
            await asyncio.sleep(min(IDLE_SECONDS, MIN_SECONDS - (time.monotonic() - soak_started)))
            observed = await health('idle-health', generation, child, idle=True)
            resident_tickets(store, model, generation)
            idle_count += 1
            record('idle-' + str(idle_count), health=observed)
        duration = time.monotonic() - soak_started
        await service.unload(model, context('unload', 60))
        if controller.child is not None or store.snapshot():
            raise ValueError('soak unload left native child or admission')
        record('unloaded', tickets=[])
        return {'status': 'passed', 'kind': kind, 'rounds': ROUNDS, 'max_tokens_per_request': TOKENS,
                'requested_output_tokens': ROUNDS * TOKENS, 'actual_output_tokens': total,
                'soak_seconds': duration, 'idle_health_checks': idle_count, 'generation': asdict(generation),
                'steps': records, 'scope': 'bounded isolated native load/soak and private admission contention; no production load test'}

    task = asyncio.create_task(workflow())
    tasks.add(task)
    try:
        done, _ = await asyncio.wait({task}, timeout=MAX_SECONDS)
        if not done:
            raise TimeoutError('isolated soak exceeded its total deadline')
        return task.result()
    finally:
        cancellation.set()
        for pending in tasks:
            if not pending.done():
                pending.cancel()
        await asyncio.wait(tasks, timeout=5)
        for pending in tasks:
            pending.add_done_callback(_observe)
