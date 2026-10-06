"""Offline checks of the fixed review recipe; no host archive, Docker or model I/O."""
import asyncio
import copy
from datetime import datetime, timezone
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

from kiron_common.local_inference import (CapabilityName as N, CapabilityStatus, RequestContext,
    RuntimeImplementation, RuntimeTimeouts, build_resolver_snapshot)
from kiron_common.local_model_registry import RegistryEntry
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType, ModelCatalog
from runtime_capabilities import decode_provider_evidence

spec = importlib.util.spec_from_file_location('ollama_export_review', Path(__file__).with_name('export-review.py'))
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def fixture():
    entry = RegistryEntry.create(runtime_provider=BackendType.OLLAMA, artifact_origin=ArtifactType.OLLAMA,
        artifact_format=ArtifactFormat.OLLAMA_MANIFEST, reference='qwen3:8b', display_name='Fixture',
        loader=LoaderType.OLLAMA, sha256='a' * 64)
    model = build_resolver_snapshot(ModelCatalog(), (entry,)).resolve(entry.id)
    impl = RuntimeImplementation('ollama/ollama@sha256:' + 'b' * 64, None, 'sha256:' + 'd' * 64)
    encoded = recipe.candidate(model, impl, datetime(2026, 9, 22, tzinfo=timezone.utc))
    return model, impl, encoded


class RecipeTests(unittest.TestCase):
    def setUp(self):
        # Publication deliberately refuses writable ancestors such as /tmp.
        self.temporary = tempfile.TemporaryDirectory(prefix='.ollama-review-test-', dir=Path.home())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_archive_pin_catches_bytes_owner_mode_and_extra_directory(self):
        source = self.root / 'report'; source.mkdir()
        file = source / 'result.json'; file.write_text('{"status":"passed"}')
        initial = recipe.digest(recipe.canonical(recipe.inventory(source)))
        for change in ('bytes', 'mode', 'directory'):
            with self.subTest(change=change):
                file.write_text('{"status":"passed"}'); file.chmod(0o644)
                if change == 'bytes': file.write_text('{"status":"failed"}')
                elif change == 'mode': file.chmod(0o640)
                else: (source / 'unreviewed').mkdir()
                self.assertNotEqual(recipe.digest(recipe.canonical(recipe.inventory(source))), initial)

    def test_fixed_pin_rejects_archive_before_loading_any_current_helper(self):
        source = self.root / 'report'; source.mkdir()
        (source / 'result.json').write_text('{"status":"passed"}')
        with mock.patch.object(recipe, 'REPORT', source), mock.patch.object(recipe, 'current_harness') as helper:
            with self.assertRaisesRegex(ValueError, 'fixed independent review pin'):
                recipe.assemble()
            helper.assert_not_called()

    def test_inventory_bounds_empty_directories_depth_bytes_and_links(self):
        source = self.root / 'report'; source.mkdir()
        file = source / 'a'; file.write_bytes(b'a')
        (source / 'empty').mkdir()
        with mock.patch.object(recipe, 'MAX_ENTRIES', 1):
            with self.assertRaisesRegex(ValueError, 'entry limit'): recipe.inventory(source)
        with mock.patch.object(recipe, 'MAX_DEPTH', 0):
            with self.assertRaisesRegex(ValueError, 'depth'): recipe.inventory(source)
        with mock.patch.object(recipe, 'MAX_TOTAL', 0):
            with self.assertRaisesRegex(ValueError, 'byte limit'): recipe.inventory(source)
        link = source / 'link'; link.symlink_to(file)
        with self.assertRaises(ValueError): recipe.inventory(source)
        link.unlink(); os.link(file, link)
        with self.assertRaises(ValueError): recipe.inventory(source)

    def test_decoder_rejects_duplicate_nonfinite_and_oversized_file(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): recipe.decode(raw)
        file = self.root / 'large'; file.write_bytes(b'ab')
        with mock.patch.object(recipe, 'MAX_FILE', 1):
            with self.assertRaises(ValueError): recipe.read(file)

    def test_exact_evidence_and_closed_profile(self):
        model, impl, encoded = fixture()
        caps = decode_provider_evidence(encoded, BackendType.OLLAMA)[model.deployment.id]
        self.assertIsNone(model.deployment.resource_profile)
        for name in (N.CHAT, N.STREAMING, N.FUNCTION_TOOLS, N.PARALLEL_TOOLS, N.STRUCTURED_OUTPUT):
            self.assertTrue(caps.supports(name, model.deployment, impl))
        for name in (N.VISION, N.REASONING, N.EMBEDDINGS):
            self.assertEqual(caps.by_name[name].status, CapabilityStatus.UNVERIFIED)
        chat = caps.by_name[N.CHAT].constraints
        self.assertEqual(chat['context_tokens'].allowed_values, (1024,))
        self.assertEqual(chat['device'].allowed_values, ('cpu',))
        self.assertEqual(chat['default_max_output_tokens'].allowed_values, (8,))
        self.assertNotIn('usage_fields', chat)
        for field, value in (('roles', 'system'), ('roles', 'developer'), ('token_budget', 'max_tokens'),
                             ('temperature', 1), ('max_output_tokens', 64)):
            with self.subTest(field=field, value=value): self.assertFalse(chat[field].accepts(value))
        for field, value in (('strict', True), ('tool_choice', 'required'), ('tool_choice', 'named'), ('max_tools', 2)):
            with self.subTest(field=field, value=value): self.assertFalse(caps.by_name[N.FUNCTION_TOOLS].constraints[field].accepts(value))
        other = RuntimeImplementation('different-RepoDigest', None, impl.parser_revision)
        self.assertFalse(caps.supports(N.CHAT, model.deployment, other))

    def test_context_evidence_requires_original_input_controls_and_error_only(self):
        report = self.root / 'context'
        (report / 'results').mkdir(parents=True)
        for name in recipe.CONTEXT_CASES:
            streaming = name.endswith('-SSE')
            words = 'alpha beta gamma delta '
            messages = ([{'role': role, 'content': words * 32} for _ in range(16)
                         for role in ('user', 'assistant')] + [{'role': 'user', 'content': 'Reply only OK.'}]
                        if '-history-' in name else [{'role': 'user', 'content': words * 512}])
            body = {'model': 'qwen3:8b', 'messages': messages, 'stream': streaming, 'think': False,
                    'truncate': False, 'shift': False, 'tools': [],
                    'options': {'num_predict': 8, 'temperature': 0, 'num_gpu': 0, 'num_ctx': 1024}}
            error = {'error': recipe.CONTEXT_OVERFLOW, **({'status': 400} if streaming else {})}
            status = {'status': 200 if streaming else 400}
            raw = recipe.canonical(error) + (b'\n' if streaming else b'')
            public = {'code': 'context_length_exceeded', 'request_tickets': 0,
                      'native': {'prefix': 'native-001', 'status': status['status'], 'eof': True},
                      'sdk_error': 'APIError' if streaming else 'BadRequestError',
                      'http_status': None if streaming else 400, 'chunks': []}
            file = report / ('results/sdk-' + name + '.json')
            file.write_bytes(recipe.canonical(public))
            result = {'cases': {name: public}}
            call = (body, [error], status, 'native-001', raw)
            recipe.validate_context_case(report, result, name, call)
            mutations = [
                ('truncate', lambda b, f, p: b.update(truncate=True)),
                ('shift', lambda b, f, p: b.update(shift=True)),
                ('messages', lambda b, f, p: b.update(messages=[{'role': 'user', 'content': 'Reply only OK.'}])),
                ('device', lambda b, f, p: b['options'].update(num_gpu=20)),
                ('context', lambda b, f, p: b['options'].update(num_ctx=4096)),
                ('keepalive', lambda b, f, p: b.update(keep_alive='10m')),
                ('extra-frame', lambda b, f, p: f.append({'done': True})),
                ('tickets', lambda b, f, p: p.update(request_tickets=1)),
                ('EOF', lambda b, f, p: p['native'].update(eof=False)),
                ('code', lambda b, f, p: p.update(code='provider_unavailable')),
                ('chunks', lambda b, f, p: p['chunks'].append({'usage': None, 'choices': [
                    {'finish_reason': None, 'delta': {'content': 'silently truncated'}}]})),
            ]
            for label, mutate in mutations:
                with self.subTest(name=name, mutation=label):
                    changed_body, frames, changed_public = copy.deepcopy((body, [error], public))
                    mutate(changed_body, frames, changed_public)
                    file.write_bytes(recipe.canonical(changed_public))
                    with self.assertRaises(ValueError):
                        recipe.validate_context_case(report, {'cases': {name: changed_public}}, name,
                            (changed_body, frames, status, 'native-001', raw))
            file.write_bytes(recipe.canonical(public))
            if streaming:
                with self.assertRaisesRegex(ValueError, 'frame incomplete'):
                    recipe.validate_context_case(report, result, name, (*call[:-1], raw.rstrip(b'\n')))

    def test_new_output_atomic_no_replace_and_changed_candidate_rejected(self):
        if os.geteuid() != 0:
            self.skipTest('root-owned publication contract')
        root = self.root / 'candidates'
        encoded, provenance = {'fixture': True}, {'candidate_sha256': 'e' * 64}
        with mock.patch.object(recipe, 'OUTPUT_ROOT', root), mock.patch.object(recipe, 'assemble', return_value=(encoded, provenance)):
            target = root / 'review'
            recipe.publish(target, encoded, provenance)
            self.assertEqual(recipe.verify(target), 'e' * 64)
            with self.assertRaises(OSError): recipe.publish(target, {'changed': True}, provenance)
            self.assertEqual(list(root.iterdir()), [target])
            (target / 'ollama.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'content differs'): recipe.verify(target)
            with self.assertRaises(ValueError): recipe.output_path(Path('/usr/lib/kiron/data/local-inference-capabilities/new'))


class RuntimeContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_resident_profile_closes_public_discovery_on_cpu_or_context_drift(self):
        from dataclasses import fields
        from kiron_common.local_inference import LocalInferenceError, ResolverSnapshot
        from kiron_common.ollama_compat import CompatResult, OllamaCapabilities
        from ollama_provider import OllamaProvider
        from runtime_service import RuntimeService
        model, impl, encoded = fixture()
        snapshot = ResolverSnapshot(model.snapshot_revision, {model.deployment.id: model.deployment},
                                    {model.public_model_id: model})
        resolver = SimpleNamespace(snapshot=mock.AsyncMock(return_value=snapshot))
        caps = decode_provider_evidence(encoded, BackendType.OLLAMA)
        observed = {"context_length": 1024, "size_vram": 0}
        calls = []
        def native(request):
            calls.append((request.method, request.url.path))
            self.assertEqual(request.method, 'GET')
            entry = {'name': model.deployment.reference, 'model': model.deployment.reference,
                     'digest': model.deployment.artifact_identity.sha256, 'size': 4096}
            if request.url.path == '/api/version':
                return httpx.Response(200, json={'version': '0.18.0'})
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [entry]})
            self.assertEqual(request.url.path, '/api/ps')
            return httpx.Response(200, json={'models': [entry | observed]})
        provider = OllamaProvider(client=httpx.AsyncClient(base_url='http://127.0.0.1:11435',
            trust_env=False, transport=httpx.MockTransport(native)), resolver=resolver,
            implementation=impl, capabilities=caps, expected_version='0.18.0',
            compatibility=OllamaCapabilities(**{f.name: CompatResult(True) for f in fields(OllamaCapabilities)}))
        service = RuntimeService(resolver=resolver, providers={BackendType.OLLAMA: provider},
            admission=mock.Mock(), measure=mock.Mock(), timeouts=RuntimeTimeouts(1,1,1,1,1,1,1))
        self.addAsyncCleanup(service.aclose)
        context = RequestContext('resident-discovery', time.monotonic()+10, asyncio.Event())
        self.assertEqual((await service.public_models(context))[0]['id'], model.api_model_id)
        for delta in ({'context_length': 4096, 'size_vram': 0},
                      {'context_length': 1024, 'size_vram': 2048}):
            observed = delta
            with self.subTest(observed=observed), self.assertRaises(LocalInferenceError):
                await service.public_models(context)
        self.assertTrue(calls)
        self.assertTrue(all(method == 'GET' for method, _ in calls))
        service.admission.reserve.assert_not_called()

    async def test_default_and_all_measured_parameter_families_validate_without_io(self):
        from openai_wire import parse_chat
        from runtime_service import RuntimeService
        model, impl, encoded = fixture()
        caps = decode_provider_evidence(encoded, BackendType.OLLAMA)[model.deployment.id]
        provider = SimpleNamespace(implementation=impl, capabilities=mock.AsyncMock(return_value=caps),
            validate_request=mock.Mock(), aclose=mock.AsyncMock(), load=mock.AsyncMock())
        service = RuntimeService(resolver=mock.Mock(), providers={BackendType.OLLAMA: provider},
            admission=mock.Mock(), measure=mock.Mock(), timeouts=RuntimeTimeouts(1,1,1,1,1,1,1))
        self.addAsyncCleanup(service.aclose)
        context = RequestContext('review', time.monotonic() + 10, asyncio.Event())
        tool = {'type': 'function', 'function': {'name': 'lookup_value', 'strict': False,
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
        base = {'model': model.api_model_id, 'messages': [{'role': 'user', 'content': 'fixture'}], 'temperature': 0}
        default = await service.validate_chat(parse_chat(base), model, context)
        self.assertEqual(default.options.max_output_tokens, 8)
        for delta in (
            {'max_completion_tokens': 8, 'stream': True},
            {'max_completion_tokens': 96, 'tools': [tool], 'tool_choice': 'auto', 'parallel_tool_calls': True},
            {'max_completion_tokens': 24, 'tools': [tool], 'tool_choice': 'none'},
            {'max_completion_tokens': 24, 'response_format': {'type': 'json_object'}},
            {'max_completion_tokens': 24, 'response_format': {'type': 'json_schema', 'json_schema': {'name': 'status', 'strict': True, 'schema': recipe.SCHEMA}}},
        ):
            with self.subTest(fields=tuple(delta)):
                await service.validate_chat(parse_chat(base | delta), model, context)
        provider.load.assert_not_called()
        service.admission.reserve.assert_not_called()
