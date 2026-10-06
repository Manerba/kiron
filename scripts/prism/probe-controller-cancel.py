"""Native request cancellation plus explicit controller shutdown recovery.

No signal/force-unload/unit operation is introduced. The enclosing isolated
harness owns process identity, source pins, the 600s watchdog and final cleanup.
"""
import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import time

KINDS = frozenset(("native-cancel",))
MAX_SECONDS = 600
CANCEL_TOKENS = 64


def _faults():
    spec = importlib.util.spec_from_file_location("cancel_probe_fault_helpers", Path(__file__).with_name("probe-controller-faults.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


faults = _faults()
NativeTrace = faults.NativeTrace


def capabilities(kind, evidence):
    if kind not in KINDS:
        raise ValueError("unknown native cancellation probe")
    return faults.capabilities("native-crash", evidence)


def _observe(task):
    if not task.cancelled():
        task.exception()


def reaped(controller, child, identity):
    # Controller clears child only after its normal process-group cleanup proof.
    return (controller.child is None and child.process.returncode is not None
            and not Path("/proc", str(identity["pid"])).exists())


async def probe(service, model, store, kind, *, controller, provider, report=None):
    from kiron_common.local_inference import GenerationOptions, InferenceRequest, Message, MessageRole, RequestContext, TextPart
    from provider_transport import bounded_request, decode_provider_json
    if kind not in KINDS or os.geteuid() == 0 or os.getgroups():
        raise ValueError("isolated non-root cancellation harness required")
    deadline = time.monotonic() + MAX_SECONDS
    steps, events = [], []
    operation = consumer = shutdown = None

    def context(name, seconds):
        return RequestContext("cancel-" + name, min(deadline, time.monotonic() + seconds), asyncio.Event())

    def record(name, **values):
        entry = {"step": name, "monotonic": time.monotonic(), **values}
        steps.append(entry)
        if report is not None:
            with (Path(report) / ("cancel-" + name + ".json")).open("x") as stream:
                json.dump(entry, stream, indent=2)

    async def health(name):
        status, body = await bounded_request(provider.control, "GET", "/health", context(name, 15), limit=65536)
        if status != 200:
            raise ValueError("controller health failed during cancellation probe")
        return decode_provider_json(body)

    async def bounded(task, seconds):
        done, _ = await asyncio.wait({task}, timeout=max(0, min(seconds, deadline - time.monotonic())))
        if not done:
            raise TimeoutError("cancellation probe step timed out")
        return task.result()

    try:
        if store.snapshot():
            raise ValueError("cancellation probe requires empty isolated admission")
        loaded = await service.load(model, context("initial-load", 200))
        generation, child = loaded.observation.generation, controller.child
        if child is None or generation != controller.generation or generation.process_id is None:
            raise ValueError("loaded controller identity mismatch")
        identity = faults.process_identity(child.process.pid)
        faults.guard_identity(identity, controller.policy.binary)
        record("loaded", generation=asdict(generation), identity=identity)
        request = InferenceRequest(model, (Message(MessageRole.USER, (TextPart(
            "Write the integers from 1 to 100, one number per line. Continue until all numbers are written."),)),),
            GenerationOptions(CANCEL_TOKENS), context("active-stream", 180))
        operation = await service.prepare(request, streaming=True)

        async def consume():
            try:
                async for event in operation.events():
                    events.append({"kind": event.kind.value, "request_id": event.request_id,
                                   "text_bytes": len(event.text.encode()) if event.text is not None else 0})
            finally:
                await operation.close()

        consumer = asyncio.create_task(consume())
        active_deadline = min(deadline, time.monotonic() + 90)
        while True:
            observed, tickets = await health("active-health"), store.snapshot()
            if consumer.done():
                await consumer
                raise ValueError("request ended before a native cancellation could be injected")
            if (observed.get("generation") == asdict(generation) and observed.get("state") == "loaded"
                    and observed.get("active_requests") == 1 and type(observed.get("slot_task_id")) is int
                    and faults.active_ticket(tickets, request.context.request_id, model.deployment.id, generation)):
                break
            if time.monotonic() >= active_deadline:
                raise TimeoutError("active native request was not proven")
            await asyncio.sleep(.05)
        current = faults.process_identity(child.process.pid)
        if (controller.child is not child or controller.generation != generation
                or not faults.same_process(identity, current)):
            raise ValueError("native ownership changed before cancellation")
        faults.guard_identity(current, controller.policy.binary)
        record("before-cancel", health=observed, tickets=[asdict(ticket) for ticket in tickets], identity=current)
        request.context.cancellation.set()
        consumer.cancel()
        try:
            await bounded(consumer, 15)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise  # The outer watchdog is not the cancellation under test.
        if any(event["kind"] == "completed" for event in events):
            raise ValueError("cancelled request falsely completed successfully")
        tickets = store.snapshot()
        state = faults.post_failure_state(tickets, request.context.request_id, model.deployment.id, generation,
                                          cleanup_confirmed=reaped(controller, child, identity))
        record("cancelled", events=events, tickets=[asdict(ticket) for ticket in tickets], work_state=state)
        # This is the existing shutdown path: bounded drain, owned process-tree
        # termination, then exact-generation admission cleanup. No force unload.
        shutdown = asyncio.create_task(controller.close())
        await bounded(shutdown, 90)
        observed = await health("after-shutdown")
        if not reaped(controller, child, identity) or observed.get("state") != "unloaded" or store.snapshot():
            raise ValueError("explicit controller shutdown did not prove cancellation recovery")
        record("shutdown-recovered", health=observed, returncode=child.process.returncode, tickets=[])
        # ControlApp lifespan performs the same second close on normal exit.
        shutdown = asyncio.create_task(controller.close())
        await bounded(shutdown, 15)
        if controller.child is not None or store.snapshot():
            raise ValueError("second controller close changed the recovered state")
        return {"kind": kind, "steps": steps, "max_output_tokens": CANCEL_TOKENS,
                "recovery": "existing Controller.close; controller remains closed", "native_generations": 1}
    finally:
        if shutdown is not None and not shutdown.done():
            shutdown.add_done_callback(_observe)
        if consumer is not None and not consumer.done():
            consumer.cancel()
            done, _ = await asyncio.wait({consumer}, timeout=5)
            consumer.add_done_callback(_observe)
            if not done:
                raise TimeoutError("cancelled consumer cleanup remains pending")
        if operation is not None and (consumer is None or consumer.done()):
            cleanup = asyncio.create_task(operation.close())
            done, _ = await asyncio.wait({cleanup}, timeout=5)
            cleanup.add_done_callback(_observe)
            if not done:
                record("operation-cleanup-pending", operation=True)
                raise TimeoutError("operation cleanup remains pending")
            cleanup.result()
