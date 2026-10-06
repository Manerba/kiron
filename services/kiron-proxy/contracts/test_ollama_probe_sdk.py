"""The exact native Ollama probe through SDK/API/Runtime/adapter, with mock HTTP only."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'services/kiron-proxy'))
sys.path.insert(0, str(ROOT / 'services/kiron-common'))
spec = importlib.util.spec_from_file_location('ollama_probe_sdk', ROOT / 'scripts/ollama/probe-runtime.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import (ArtifactIdentity, CapabilityEvidence, ResolvedDeployment,
    ResolvedModel, ResolverSnapshot, RuntimeTimeouts)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.ollama_compat import OllamaCapabilities
from ollama_provider import OllamaProvider
from runtime_service import RuntimeService


class OllamaProbeSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_probe_uses_real_adapter_and_preserves_distinct_tool_results(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        admission = directory / 'admission'
        admission.mkdir(mode=0o2770)
        admission.chmod(0o2770)
        results = directory / 'results'
        results.mkdir()
        store = AdmissionStore(admission, security=RuntimeSecurity(os.getuid(), os.getgid(), frozenset({os.getuid()})))
        artifact = ArtifactIdentity(ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST,
                                    sha256=probe.MODEL_SHA, manifest_digest=probe.MODEL_SHA)
        deployment = ResolvedDeployment('qwen', BackendType.OLLAMA, 'qwen3:8b', artifact,
                                         LoaderType.OLLAMA, None, 'a' * 64)
        model = ResolvedModel('qwen-public', deployment, None, None, 'b' * 64, created=1)
        snapshot = ResolverSnapshot('b' * 64, {deployment.id: deployment}, {model.public_model_id: model})
        class Resolver:
            async def snapshot(self):
                return snapshot
        implementation = probe.measured_implementation({'image': {'RepoDigests': ['ollama/ollama@sha256:' + 'c' * 64]}},
            {'image_digest': 'ollama/ollama@sha256:' + 'c' * 64, 'report_status': 'passed'}, 'mock-parser')
        evidence = CapabilityEvidence(implementation.provider_revision, artifact.fingerprint, probe.MODEL_SHA, None,
            None, 'mock-parser', 'a' * 64, 'offline only', datetime.now(timezone.utc))
        trace_spec = importlib.util.spec_from_file_location('probe_native_trace', ROOT / 'scripts/prism/probe-openai-features.py')
        trace_module = importlib.util.module_from_spec(trace_spec)
        trace_spec.loader.exec_module(trace_module)
        trace, requests = trace_module.NativeTrace(results), []
        await trace.transport.aclose()
        def respond(request):
            if request.url.path == '/api/version':
                return httpx.Response(200, json={'version': '0.18.0'})
            if request.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': 'qwen3:8b', 'model': 'qwen3:8b',
                    'digest': probe.MODEL_SHA, 'size': 100, 'size_vram': 0, 'context_length': 1024}]})
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': 'qwen3:8b', 'model': 'qwen3:8b',
                    'digest': probe.MODEL_SHA, 'size': 100}]})
            self.assertEqual(request.url.path, '/api/chat')
            body = json.loads(request.content)
            requests.append(body)
            self.assertIs(body['think'], False)
            self.assertEqual(body['options']['num_gpu'], 0)
            self.assertEqual(body['options']['num_ctx'], 1024)
            self.assertIs(body['truncate'], False)
            self.assertIs(body['shift'], False)
            self.assertNotIn('keep_alive', body)
            if any(body['messages'] == probe.context_messages(history) for history in (False, True)):
                # Both upstream error forms remain native fixtures, never a
                # token-count heuristic masquerading as a real model measure.
                if body['stream']:
                    return httpx.Response(200, content=json.dumps({
                        'error': probe.CONTEXT_OVERFLOW, 'status': 400}).encode() + b'\n')
                return httpx.Response(400, json={'error': probe.CONTEXT_OVERFLOW})
            message = {'role': 'assistant', 'content': 'OK.'}
            if any(item['role'] == 'tool' for item in body['messages']):
                results = [item for item in body['messages'] if item['role'] == 'tool']
                self.assertEqual([json.loads(item['content']) for item in results], [{'value': 'ALPHA'}, {'value': 'BETA'}])
                self.assertEqual([item['tool_name'] for item in results], ['lookup_value', 'lookup_value'])
                # The prompt specifies only output shape; result values must be
                # recovered from correlated tool results, never copied from it.
                for item in body['messages']:
                    if item['role'] != 'tool':
                        self.assertNotIn('ALPHA', json.dumps(item))
                        self.assertNotIn('BETA', json.dumps(item))
                self.assertEqual(body['messages'][-1]['content'],
                    'Use the returned values. Reply only Berlin=<returned value>;Paris=<returned value>, with no spaces.')
                message['content'] = 'Berlin=ALPHA;Paris=BETA'
            elif body.get('tools'):
                cities = ['Berlin', 'Paris'] if 'Paris' in body['messages'][-1]['content'] else ['Berlin']
                message['content'] = ''
                message['tool_calls'] = [{'function': {'name': 'lookup_value', 'arguments': {'city': city}}} for city in cities]
            elif 'format' in body:
                message['content'] = '{"status":"ok"}'
            done = {'model': 'qwen3:8b', 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 10, 'eval_count': 3}
            if body['stream']:
                frames = [{'model': 'qwen3:8b', 'done': False, 'message': message},
                          {**done, 'message': {'role': 'assistant', 'content': ''}}]
                return httpx.Response(200, content=b''.join(json.dumps(frame).encode() + b'\n' for frame in frames))
            return httpx.Response(200, json={**done, 'message': message})
        trace.transport = httpx.MockTransport(respond)
        provider = OllamaProvider(client=httpx.AsyncClient(base_url='http://127.0.0.1:18095', trust_env=False,
            transport=trace), resolver=Resolver(), implementation=implementation,
            capabilities={deployment.id: probe.capabilities(evidence)}, compatibility=OllamaCapabilities(),
            expected_version='0.18.0')
        service = RuntimeService(resolver=Resolver(), providers={BackendType.OLLAMA: provider}, admission=store,
            measure=lambda: MemorySnapshot(0, 10000, time.monotonic()), timeouts=RuntimeTimeouts(1, 1, 1, 1, 10, 1, 1))
        self.addAsyncCleanup(service.aclose)
        measurements = []
        async def verify_resident():
            measurements.append(True)
        values = await probe.cases(service, model, store, trace, results, verify_resident)
        self.assertEqual(len(measurements), 12)
        self.assertEqual(len(requests), 12)
        for name in ('text-False', 'text-True'):
            self.assertEqual(values[name]['choices'][0]['message']['content'], 'OK.')
        self.assertEqual(values['tools-replay']['choices'][0]['message']['content'], 'Berlin=ALPHA;Paris=BETA')
        for form in ('single', 'history'):
            for mode in ('JSON', 'SSE'):
                failure = values['context-' + form + '-' + mode]
                self.assertEqual(failure['code'], 'context_length_exceeded')
                self.assertEqual(failure['http_status'], 400 if mode == 'JSON' else None)
                self.assertTrue(failure['native']['eof'])
                self.assertEqual(failure['request_tickets'], 0)
        self.assertEqual(values['context-recovery']['choices'][0]['message']['content'], 'OK.')
        self.assertEqual(len([key for key in values if key.startswith('unsupported-')]), 5)
        self.assertEqual(store.snapshot(), ())


if __name__ == '__main__':
    unittest.main()
