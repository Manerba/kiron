"""Real isolated AdmissionStore/RuntimeService, fake provider; no native starts."""
import asyncio
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import httpx
import pytest

from kiron_common.gpu_admission import AdmissionError
from kiron_common.local_inference import (EventKind, FinishReason, InferenceEvent, InferenceResult,
                                         RuntimeGeneration, TextPart, TokenUsage)
from kiron_common.model_catalog import BackendType
from test_dashboard_runtime import runtime

spec = importlib.util.spec_from_file_location('soak_probe', Path(__file__).with_name('probe-runtime-soak.py'))
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


@pytest.fixture
def fixture(runtime, monkeypatch):
    r = runtime
    r.generation = RuntimeGeneration('boot', 'child')
    r.child = SimpleNamespace(alive=lambda: True)
    r.controller = SimpleNamespace(child=None, generation=r.generation)
    r.count, r.finished, r.usage_count, r.stream_usage = 0, FinishReason.STOP, 1, True
    r.prove_active, r.bad_health, r.reject_code = True, False, None
    r.stream_active = False
    r.provider.provider = BackendType.PRISM
    r.provider.validate_request = lambda *args: None
    evidence = next(iter(r.provider.caps.by_name.values())).evidence[0]
    r.provider.caps = soak.capabilities('runtime-soak', evidence)
    original_load, original_unload, original_reserve = r.provider.load, r.provider.unload, r.admission.reserve

    async def load(*args, **kwargs):
        result = await original_load(*args, **kwargs)
        r.controller.child = r.child
        return result

    async def unload(*args, **kwargs):
        result = await original_unload(*args, **kwargs)
        r.controller.child = None
        return result

    async def chat(request):
        r.count += 1
        assert request.options.max_output_tokens == 8
        assert request.messages[0].content == (TextPart('Reply only OK.'),)
        return InferenceResult(request.context.request_id, (TextPart('OK'),), (), (),
                               TokenUsage(2, r.usage_count), r.finished)

    async def stream(request):
        r.count += 1
        r.stream_active = True
        try:
            yield InferenceEvent(EventKind.STARTED, request.context.request_id)
            if r.count == 2 and r.prove_active:
                r.loop = asyncio.get_running_loop()
                r.raced = asyncio.Event()
                await r.raced.wait()
            yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id, text='OK', output_item_index=0, part_index=0)
            if r.stream_usage:
                yield InferenceEvent(EventKind.USAGE, request.context.request_id, usage=TokenUsage(2, r.usage_count))
            yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=r.finished)
        finally:
            r.stream_active = False

    def reserve(**kwargs):
        if kwargs['operation_id'] == 'soak-impossible-load':
            if r.reject_code:
                r.loop.call_soon_threadsafe(r.raced.set)
                raise AdmissionError(r.reject_code, 'fixture rejection')
            try:
                return original_reserve(**kwargs)
            finally:
                r.loop.call_soon_threadsafe(r.raced.set)
        return original_reserve(**kwargs)

    async def control(request):
        assert request.url.path == '/health' and request.method == 'GET'
        return httpx.Response(200, json={'state': 'loaded' if r.provider.loaded else 'unloaded',
            'generation': asdict(replace(r.generation, process_id='foreign') if r.bad_health else r.generation),
            'active_requests': int(r.stream_active and r.prove_active),
            'slot_task_id': 7 if r.stream_active and r.prove_active else None})

    async def wait_end(*args, **kwargs):
        return False

    r.provider.load, r.provider.unload = load, unload
    r.provider.chat, r.provider.stream, r.provider.wait_request_end = chat, stream, wait_end
    r.provider.control = httpx.AsyncClient(transport=httpx.MockTransport(control), base_url='http://isolated')
    monkeypatch.setattr(r.admission, 'reserve', reserve)
    monkeypatch.setattr(soak.os, 'geteuid', lambda: 65534)
    monkeypatch.setattr(soak.os, 'getgroups', lambda: [])
    monkeypatch.setattr(soak.os, 'sysconf', lambda key: 100)
    monkeypatch.setattr(soak, 'MIN_SECONDS', .05)
    monkeypatch.setattr(soak, 'IDLE_SECONDS', .01)
    monkeypatch.setattr(soak, 'MAX_SECONDS', 2)
    return r


def run(r, **kwargs):
    async def execute():
        try:
            return await soak.probe(r.service, r.model, r.admission, 'runtime-soak',
                                    controller=r.controller, provider=r.provider, **kwargs)
        finally:
            await r.provider.control.aclose()
    return asyncio.run(execute())


def test_six_real_runtime_rounds_and_resource_rejection_preserve_residency(fixture, tmp_path):
    result = run(fixture, report=tmp_path)
    assert result['status'] == 'passed' and result['rounds'] == fixture.count == 6
    assert result['actual_output_tokens'] == 6 and result['requested_output_tokens'] == 48
    assert result['soak_seconds'] >= .05
    assert [r['streaming'] for r in result['steps'] if r['step'].startswith('round-')] == [False, True] * 3
    competition = next(r for r in result['steps'] if r['step'] == 'competition')
    assert competition['code'] == 'resource_exhausted'
    assert competition['native_health']['active_requests'] == 1
    assert competition['requested_host_bytes'] == competition['physical_host_bytes'] + 1
    assert not fixture.admission.snapshot() and fixture.controller.child is None
    assert len(list(tmp_path.glob('soak-*.json'))) == len(result['steps'])


@pytest.mark.parametrize('field,value', [('usage_count', 9), ('usage_count', 0),
    ('finished', FinishReason.LENGTH), ('stream_usage', False), ('bad_health', True), ('prove_active', False),
    ('reject_code', 'resource_conflict')])
def test_bad_results_or_unproven_competition_never_pass(fixture, field, value):
    setattr(fixture, field, value)
    with pytest.raises(Exception):
        run(fixture)
    assert fixture.count < 6
    assert not any(t.operation_id == 'soak-impossible-load' for t in fixture.admission.snapshot())


def test_deadline_stops_without_forcing_native_or_clearing_resident(fixture, monkeypatch):
    monkeypatch.setattr(soak, 'MAX_SECONDS', .02)
    monkeypatch.setattr(soak, 'MIN_SECONDS', 120)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run(fixture)
    assert time.monotonic() - started < 1
    assert fixture.controller.child is fixture.child
    assert any(t.phase == 'resident' for t in fixture.admission.snapshot())


def test_closed_kind_and_root_rejected_before_load(fixture, monkeypatch):
    monkeypatch.setattr(soak.os, 'geteuid', lambda: 0)
    with pytest.raises(ValueError):
        run(fixture)
    assert fixture.provider.calls == [] and not fixture.admission.snapshot()
    with pytest.raises(ValueError):
        soak.capabilities('unknown', None)
