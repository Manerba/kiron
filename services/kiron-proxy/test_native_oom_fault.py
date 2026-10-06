"""Inject an OOM HTTP failure through the actual adapter/runtime/public API.

No GPU memory is exhausted. The native HTTP response and installed artifact
proof are fixtures; API serialization, adapter, admission and cleanup are real.
"""
from dataclasses import replace
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
from kiron_common.local_inference import ResolverSnapshot, RuntimeTimeouts
from kiron_common.model_catalog import BackendType
from openai_api import create_openai_api_app
from runtime_service import RuntimeService, generation_key
import test_prism_provider as prism_fixture
from test_openai_runtime_api import Keys, Records


class NativeOOMFaultTests(unittest.IsolatedAsyncioTestCase):
    setUp = prism_fixture.PrismProviderTests.setUp

    async def exercise(self, streaming):
        async def out_of_memory(request):
            self.assertEqual(request.url.path, '/v1/chat/completions')
            return httpx.Response(500, json={'error': {'message':
                'CUDA out of memory /private/weights.gguf token=secret'}})

        provider = await prism_fixture.PrismProviderTests.make(self, inference_handler=out_of_memory)
        model = replace(self.model, created=1234567890)
        snapshot = ResolverSnapshot(model.snapshot_revision, {self.deployment.id: self.deployment},
                                    {model.public_model_id: model})
        resolver = SimpleNamespace(snapshot=mock.AsyncMock(return_value=snapshot))
        provider.resolver = resolver
        provider.verify_artifact = mock.AsyncMock(return_value=True)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        root.chmod(0o2770)
        store = AdmissionStore(root, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        measure = lambda: MemorySnapshot(1000, 1000, time.monotonic())
        generation = generation_key(self.request.execution_generation)
        store.reserve(operation_id='resident', owner='kiron-proxy', generation=generation,
            deployment_id=self.deployment.id, kind='load', gpu_bytes=40, host_bytes=20,
            resident_slot='prism', measure=measure)
        store.transition('resident', owner='kiron-proxy', expected_generation=generation, phase='resident')
        service = RuntimeService(resolver=resolver, providers={BackendType.PRISM: provider}, admission=store,
                                 measure=measure, timeouts=RuntimeTimeouts(2, 2, 2, 2, 2, 2, 2))
        self.addAsyncCleanup(service.aclose)
        app = create_openai_api_app(Records(), Keys())
        app.state.local_inference = service
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://fixture',
                headers={'Authorization': 'Bearer test-key'}) as client:
            response = await client.post('/v1/chat/completions', json={
                'model': model.public_model_id, 'messages': [{'role': 'user', 'content': 'OK'}], 'stream': streaming})
            if streaming:
                self.assertEqual(response.status_code, 200)
                frames = [json.loads(line[6:]) for line in response.text.splitlines()
                          if line.startswith('data: ') and line != 'data: [DONE]']
                errors = [frame['error'] for frame in frames if 'error' in frame]
                self.assertEqual(len(errors), 1)
                self.assertFalse(any(choice.get('finish_reason') for frame in frames for choice in frame.get('choices', [])))
                self.assertEqual(response.text.count('data: [DONE]'), 1)
                error = errors[0]
            else:
                self.assertEqual(response.status_code, 503)
                error = response.json()['error']
            self.assertEqual(error['code'], 'provider_unavailable')
            self.assertNotIn('/private/', response.text)
            self.assertNotIn('token=secret', response.text)
        # An OOM error body and healthy model do not prove the backend stopped.
        tickets = store.snapshot()
        self.assertEqual([(ticket.kind, ticket.phase) for ticket in tickets],
                         [('load', 'resident'), ('request', 'unknown')])
        await service.aclose()
        self.assertEqual(store.snapshot(), tickets)

    async def test_nonstream_oom_is_sanitized_and_keeps_unconfirmed_work_reserved(self):
        await self.exercise(False)

    async def test_stream_oom_has_one_error_no_success_and_keeps_unconfirmed_work_reserved(self):
        await self.exercise(True)
