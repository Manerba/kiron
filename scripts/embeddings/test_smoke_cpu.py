"""Offline harness boundary checks; never mount, spawn, or load a model."""
import importlib.util
import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import json
import pwd
import tempfile
import unittest
from unittest import mock


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


smoke = module('embedding_cpu_smoke', 'smoke-cpu.py')
probe = module('embedding_cpu_probe', 'cpu_probe.py')


class CpuHarnessTests(unittest.TestCase):
    def test_report_is_confined_to_new_isolated_namespace(self):
        valid = smoke.REPORTS / 'embedding-cpu-fixture'
        self.assertEqual(smoke.report_path(str(valid)), valid)
        for path in ('/usr/lib/kiron/data/embedding-cpu-fixture', '/tmp/embedding-cpu-fixture',
                     str(smoke.REPORTS / 'controller-fixture'), str(smoke.REPORTS / 'embedding-cpu-a/b')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                smoke.report_path(path)

    def test_environment_is_offline_cpu_only_and_not_inherited(self):
        with mock.patch.dict(os.environ, {'HTTPS_PROXY': 'http://foreign', 'CUDA_VISIBLE_DEVICES': '0'}):
            env = smoke.clean_environment(smoke.REPORTS / 'embedding-cpu-fixture')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertNotIn('HOME', env)
        self.assertEqual(env['OMP_NUM_THREADS'], '2')

    def test_source_inventory_copies_no_weights_or_runtime_data(self):
        paths = smoke.inventory_sources()
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(path.is_relative_to(smoke.ROOT))
            self.assertIn(path.suffix, {'.py', '.json'})
            self.assertNotIn('data', path.relative_to(smoke.ROOT).parts)
            self.assertNotIn('__pycache__', path.parts)

    def test_uid_drop_clears_groups_before_interpreter_exec(self):
        order = []
        with mock.patch.object(smoke, 'no_new_privileges', side_effect=lambda: order.append(('nnp', True))), \
             mock.patch.object(smoke.os, 'setgroups', side_effect=lambda value: order.append(('groups', value))), \
             mock.patch.object(smoke.os, 'setgid', side_effect=lambda value: order.append(('gid', value))), \
             mock.patch.object(smoke.os, 'setuid', side_effect=lambda value: order.append(('uid', value))), \
             mock.patch.object(smoke.os, 'execve', side_effect=lambda *args: order.append(('exec', args[0]))):
            smoke.execute_unprivileged(smoke.REPORTS / 'embedding-cpu-fixture', {'uid': 65534, 'gid': 65534}, native=False)
        self.assertEqual(order, [('nnp', True), ('groups', []), ('gid', 65534), ('uid', 65534), ('exec', smoke.SDK)])

    def test_native_vector_comparison_uses_float32_bytes(self):
        self.assertEqual(len(probe.f32([.25, -.5, 1.])), 12)
        self.assertNotEqual(probe.f32([.25]), probe.f32([.250001]))

    @unittest.skipUnless(os.geteuid() == 0, 'permission fixture requires root-owned isolated paths')
    def test_frozen_snapshot_rejects_extra_files_writable_manifest_symlinks_and_launch_drift(self):
        with tempfile.TemporaryDirectory(prefix='embedding-cpu-offline-', dir=smoke.REPORTS) as temporary:
            root = Path(temporary)
            source = root / 'source'
            leaf = source / 'scripts/embeddings/cpu_probe.py'
            leaf.parent.mkdir(parents=True)
            leaf.write_text('# fixture\n')
            leaf.chmod(0o444)
            for path in (source, source / 'scripts', leaf.parent):
                path.chmod(0o555)
            identity = pwd.getpwnam('nobody')
            manifest = {'version': 1, 'uid': identity.pw_uid, 'gid': identity.pw_gid,
                'port': smoke.PORT, 'max_bytes': smoke.LIMIT, 'max_seconds': smoke.SECONDS,
                'hf_source': str(smoke.HF), 'hf_revision': smoke.REVISION,
                'sources': {'scripts/embeddings/cpu_probe.py': smoke.sha(leaf)}}
            path = root / 'prepared.json'
            smoke.write(path, manifest)
            with mock.patch.object(smoke, 'verify_artifacts'):
                smoke.verify(root)
                path.chmod(0o644)
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    smoke.verify(root)
                path.chmod(0o444)
                extra = leaf.with_name('unlisted.py')
                extra.write_text('# unexpected\n')
                extra.chmod(0o444)
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    smoke.verify(root)
                extra.unlink()
                leaf.unlink()
                leaf.symlink_to(path)
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    smoke.verify(root)
                leaf.unlink()
                leaf.write_text('# fixture\n')
                leaf.chmod(0o444)
                path.unlink()
                smoke.write(path, {**manifest, 'port': 11436})
                with self.assertRaisesRegex(ValueError, 'launch'):
                    smoke.verify(root)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        from kiron_common.embedding_registry import MODEL_CATALOG
        from kiron_common.local_inference import RuntimeImplementation, build_resolver_snapshot
        self.model = build_resolver_snapshot(MODEL_CATALOG, ()).resolve(probe.PROFILE + '.query')
        self.implementation = RuntimeImplementation('offline-native-fixture', None, 'offline-adapter-fixture')
        self.checks = [{'role': role, 'encoding': encoding, 'input_characters': len(text),
            'batch_size': 1, 'dimensions': 960, 'tokens': 3, 'vector_float32_sha256': 'a' * 64}
            for role in probe.ROLES for encoding, text in
            (('float', probe.TEXT), ('float', probe.SHORT_TEXT), ('base64', probe.TEXT))]
        self.checks.extend({'role': role, 'encoding': 'float',
            'input_characters': [len(text) for text in probe.batch_texts(role)],
            'batch_size': 2, 'dimensions': 960, 'tokens': 14,
            'vector_float32_sha256': ['a' * 64, 'b' * 64],
            'per_input_tokens': [10, 4] if role == 'search_query' else [4, 10],
            'padding_tokens_excluded': 6, 'single_vector_max_abs_delta': [0, 0]}
            for role in probe.ROLES)

    def fixture(self, root):
        (root / 'work').mkdir()
        probe.write(root / 'prepared.json', {'sources': {'fixture.py': 'b' * 64},
            'hf_revision': 'fixture-only', 'hf_artifacts': {'fixture': {'sha256': 'c' * 64, 'bytes': 1}}})
        probe.write(root / 'work/native-ready.json', {'scope': 'offline fixture'})
        (root / 'work/forward-inputs.ndjson').write_text('{}\n')

    def test_candidate_roundtrips_exact_identity_and_excludes_unmeasured_features(self):
        from kiron_common.local_inference import CapabilityName
        from kiron_common.model_catalog import BackendType
        from runtime_capabilities import decode_provider_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            digest = probe.export_candidate(root, self.model.deployment, self.implementation, self.checks)
            path = root / 'work/capability-candidate.json'
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            decoded = decode_provider_evidence(json.loads(path.read_text()), BackendType.KIRON_EMBEDDINGS)
            caps = decoded[self.model.deployment.id]
            self.assertEqual({name for name, value in caps.by_name.items() if value.status.value == 'supported'},
                             {CapabilityName.EMBEDDINGS})
            self.assertTrue(caps.supports(CapabilityName.EMBEDDINGS, self.model.deployment, self.implementation))
            self.assertFalse(caps.supports(CapabilityName.EMBEDDINGS, self.model.deployment,
                replace(self.implementation, parser_revision='different')))
            cap = caps.by_name[CapabilityName.EMBEDDINGS]
            self.assertIsNone(cap.evidence[0].artifact_sha256)  # HF files fingerprint, no invented combined digest.
            self.assertIsNotNone(cap.evidence[0].observed_at.tzinfo)
            expected = {'profiles': (probe.PROFILE,), 'roles': probe.ROLES, 'devices': ('cpu',),
                'encoding_formats': ('float', 'base64'), 'dimensions': (960,)}
            for name, values in expected.items():
                self.assertEqual(cap.constraints[name].allowed_values, values)
            self.assertTrue(cap.constraints['max_batch_size'].accepts(2))
            self.assertFalse(cap.constraints['max_batch_size'].accepts(3))
            self.assertFalse(cap.constraints['max_input_characters'].accepts(0))
            self.assertFalse(cap.constraints['max_input_characters'].accepts(len(probe.TEXT) + 1))
            self.assertNotIn('native_dimensions', cap.constraints)
            provenance = json.loads((root / 'work/capability-provenance.json').read_text())
            self.assertEqual(provenance['capability_sha256'], digest)
            self.assertFalse(provenance['production_capability_granted'])
            self.assertEqual(provenance['measurement_scope']['sdk_checks'], 8)
            self.assertIn('base64 short and batch2', provenance['measurement_scope']['derived'])
            with self.assertRaises(FileExistsError):
                probe.export_candidate(root, self.model.deployment, self.implementation, self.checks)

    def test_partial_encoding_role_or_input_matrix_cannot_create_candidate(self):
        cases = [self.checks[:-1], [*self.checks[:-1], self.checks[0]],
            [dict(row, dimensions=3) for row in self.checks],
            [dict(row, batch_size=3) for row in self.checks],
            [dict(row, tokens=0) for row in self.checks], self.checks[:-2],
            [dict(row, padding_tokens_excluded=0) for row in self.checks],
            [dict(row, per_input_tokens=[14, 14]) for row in self.checks],
            [dict(row, single_vector_max_abs_delta=[float('nan'), 0]) for row in self.checks]]
        for checks in cases:
            with self.subTest(checks=checks), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.fixture(root)
                with self.assertRaises(ValueError):
                    probe.export_candidate(root, self.model.deployment, self.implementation, checks)
                self.assertFalse((root / 'work/capability-candidate.json').exists())


class BatchEvidenceTests(unittest.TestCase):
    def fixture(self, role):
        singles = {}
        for label, count, value in (('long', 10, .25), ('short', 4, -.5)):
            singles[label] = {role: {'prompt_eval_count': count, 'embeddings': [[value] * 960],
                'forward_batches': [{'input_ids': [list(range(1, count + 1))],
                                     'attention_mask': [[1] * count]}]}}
        labels = ('long', 'short') if role == 'search_query' else ('short', 'long')
        records = [{'input_ids': [], 'attention_mask': []}]
        vectors = []
        for label in labels:
            one = singles[label][role]
            count = one['prompt_eval_count']
            vectors.append(one['embeddings'][0])
            records[0]['input_ids'].append(list(range(1, count + 1)) + [0] * (10 - count))
            records[0]['attention_mask'].append([1] * count + [0] * (10 - count))
        reference = {'texts': probe.batch_texts(role), 'embeddings': vectors,
            'prompt_eval_count': 14, 'forward_batches': records}
        return {'role': role, 'singles': singles, 'reference': reference,
            'vectors': vectors, 'indexes': [0, 1], 'tokens': 14, 'records': records}

    def test_both_orders_preserve_vectors_and_count_masks_without_padding(self):
        for role in probe.ROLES:
            with self.subTest(role=role):
                result = probe.batch_check(**self.fixture(role))
                self.assertEqual(result['per_input_tokens'], [10, 4] if role == 'search_query' else [4, 10])
                self.assertEqual(result['tokens'], 14)
                self.assertEqual(result['padding_tokens_excluded'], 6)
                self.assertEqual(result['single_vector_max_abs_delta'], [0, 0])

    def test_reordered_vectors_indexes_padding_usage_and_wrong_forward_ids_fail(self):
        base = self.fixture('search_document')
        cases = [dict(base, vectors=list(reversed(base['vectors']))),
            dict(base, indexes=[1, 0]), dict(base, tokens=20), dict(base, vectors=base['vectors'][:1])]
        bad_ids = copy.deepcopy(base)
        bad_ids['records'][0]['input_ids'][0][0] = 999
        cases.append(bad_ids)
        bad_mask = copy.deepcopy(base)
        bad_mask['records'][0]['attention_mask'][0] = [1] * 10
        cases.append(bad_mask)
        bad_single = copy.deepcopy(base)
        bad_single['singles']['short']['search_document']['embeddings'] = [[.125] * 960]
        cases.append(bad_single)
        for case in cases:
            with self.subTest(change=[name for name in case if case[name] != base[name]]), self.assertRaises(AssertionError):
                probe.batch_check(**case)
