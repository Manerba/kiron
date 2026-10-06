"""Actual SDK/API/Prism/Runtime path; only native HTTP and installation are mocked."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import CapabilityName, CapabilitySet, ParameterConstraint, ResolverSnapshot, RuntimeTimeouts
from kiron_common.model_catalog import BackendType
from runtime_service import RuntimeService
import test_prism_provider as fixture

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts/prism'


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class PublicApiProbeSdkTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixture.PrismProviderTests.setUp

    async def test_real_probe_maps_two_native_errors_releases_tickets_and_recovers(self):
        probe, features = module('probe-openai-api'), module('probe-openai-features')
        native_bodies = []
        error = {'error': {'code': 400, 'type': 'exceed_context_size_error', 'message': 'private native message',
                           'n_prompt_tokens': 4097, 'n_ctx': 1024}}
        async def native(request):
            body = json.loads(request.content)
            native_bodies.append(body)
            if body['messages'][0]['content'] == probe.CONTEXT_PROMPT:
                self.assertEqual(body['max_tokens'], 1)
                return httpx.Response(400, json=error)
            fixture_name = 'text_stream' if body['stream'] else 'text'
            return httpx.Response(200, content=(fixture.FIXTURES / (fixture_name + '.response.raw')).read_bytes())
        provider = await fixture.PrismProviderTests.make(self, inference_handler=native)
        loaded = dict(self.observation)
        unloaded = {**loaded, 'state': 'unloaded', 'deployment_id': None,
                    'generation': {'boot_id': 'boot', 'process_id': None}}
        current = unloaded
        async def control(request):
            nonlocal current
            if request.url.path == '/load':
                current = loaded
            return httpx.Response(200, json=current)
        await provider.control.aclose()
        provider.control = httpx.AsyncClient(base_url='http://control.invalid', transport=httpx.MockTransport(control))
        model = replace(self.model, created=1234567890)
        snapshot = ResolverSnapshot(model.snapshot_revision, {self.deployment.id: self.deployment}, {model.public_model_id: model})
        resolver = SimpleNamespace(snapshot=mock.AsyncMock(return_value=snapshot))
        provider.resolver = resolver
        provider.verify_artifact = mock.AsyncMock(return_value=True)
        capabilities = provider._capabilities[self.deployment.id]
        provider._capabilities[self.deployment.id] = CapabilitySet({name: replace(capability,
            constraints={**capability.constraints, 'token_budget': ParameterConstraint(allowed_values=('max_tokens', 'max_completion_tokens'))})
            if name in (CapabilityName.CHAT, CapabilityName.STREAMING) else capability
            for name, capability in capabilities.by_name.items()})
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); root.chmod(0o2770)
        results = root / 'results'; results.mkdir()
        admission = root / 'admission'; admission.mkdir(mode=0o2770); admission.chmod(0o2770)
        store = AdmissionStore(admission, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        trace = features.NativeTrace(results)
        await trace.transport.aclose()
        trace.transport = httpx.MockTransport(native)
        await provider.inference.aclose()
        provider.inference = httpx.AsyncClient(base_url='http://native.invalid', transport=trace, timeout=None)
        service = RuntimeService(resolver=resolver, providers={BackendType.PRISM: provider}, admission=store,
            measure=lambda: MemorySnapshot(1000, 1000, time.monotonic()), timeouts=RuntimeTimeouts(2, 2, 2, 2, 5, .1, .1))
        self.addAsyncCleanup(service.aclose)
        result = await probe.probe(service, model, store, report=results)
        self.assertEqual([v['http_status'] for v in result['context_rejections']], [400, 200])
        self.assertEqual([v['request_tickets_after_eof'] for v in result['context_rejections']], [[], []])
        self.assertEqual([v['status'] for v in result['request_records']], [200, 200, 400, 400, 200, 200, 200])
        self.assertEqual([body['stream'] for body in native_bodies], [False, True, False, True])
        self.assertEqual(result['text']['choices'][0]['message']['content'], 'OK')
        self.assertEqual(result['stream']['choices'][0]['message']['content'], 'OK')
        self.assertEqual([(ticket.kind, ticket.phase) for ticket in store.snapshot()], [('load', 'resident')])
        for index in (1, 2):
            self.assertEqual(json.loads((results / f'native-{index:03d}.response.raw').read_bytes()), error)
            self.assertEqual(json.loads((results / f'native-{index:03d}.status.json').read_bytes()), {'status': 400})
            self.assertTrue((results / f'native-{index:03d}.eof').is_file())
        self.assertNotIn(b'private native', (results / 'context-json.public.raw').read_bytes())
        self.assertNotIn(b'private native', (results / 'context-stream.public.raw').read_bytes())


if __name__ == '__main__':
    unittest.main()
