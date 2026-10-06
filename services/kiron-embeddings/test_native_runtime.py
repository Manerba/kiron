"""Local identity fixtures only; no Hugging Face network or real weight reads."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.model_catalog import ArtifactFile
from native_runtime import generation, has_closed_loader_provenance, state_payload, verify_local_artifact
import native_runtime


class NativeArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = MODEL_CATALOG.require("bge-m3:latest").deployments[0].artifact
        self.artifact = replace(original, repository="fixture/tiny", revision="a" * 40,
            weights=(ArtifactFile("model.safetensors", hashlib.sha256(b"fixture").hexdigest(), 7),), auxiliary=())
        repository = self.root / "models--fixture--tiny"
        self.snapshot = repository / "snapshots" / self.artifact.revision
        self.snapshot.mkdir(parents=True)
        self.blob = repository / "blobs" / "tiny"
        self.blob.parent.mkdir()
        self.blob.write_bytes(b"fixture")
        self.path = self.snapshot / "model.safetensors"
        self.path.symlink_to(self.blob)

    def test_pinned_hf_blob_and_post_construction_identity(self):
        proof = verify_local_artifact(self.artifact, cache=self.root)
        proof.recheck()
        self.blob.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            proof.recheck()
        with self.assertRaisesRegex(ValueError, "digest"):
            verify_local_artifact(self.artifact, cache=self.root)

    def test_added_loader_file_invalidates_existing_proof(self):
        proof = verify_local_artifact(self.artifact, cache=self.root)
        (self.snapshot / 'special_tokens_map.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            proof.recheck()
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            verify_local_artifact(self.artifact, cache=self.root)

    def test_equal_metadata_never_substitutes_for_rehashing_bytes(self):
        # Deterministic coverage of the filesystem's observed timestamp alias:
        # the independent 100-write repro separately uses wholly real stat data.
        with mock.patch.object(native_runtime, "_signature", return_value=(1, 2, 7, 3, 4)):
            proof = verify_local_artifact(self.artifact, cache=self.root)
            self.blob.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "catalog digest"):
                proof.recheck()

    def test_same_bytes_at_replaced_inode_are_rejected_during_hash(self):
        proof = verify_local_artifact(self.artifact, cache=self.root)
        original_fstat = os.fstat
        checks = 0
        def replace_after_read(descriptor):
            nonlocal checks
            checks += 1
            result = original_fstat(descriptor)
            if checks == 2:
                replacement = self.blob.with_name("replacement")
                replacement.write_bytes(b"fixture")
                os.replace(replacement, self.blob)
            return result
        with mock.patch.object(native_runtime.os, "fstat", side_effect=replace_after_read):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                proof.recheck()

    def test_retargeted_hf_link_during_hash_is_rejected(self):
        proof = verify_local_artifact(self.artifact, cache=self.root)
        other = self.blob.with_name("other")
        other.write_bytes(b"fixture")
        original_fstat = os.fstat
        checks = 0
        def replace_link_after_read(descriptor):
            nonlocal checks
            checks += 1
            result = original_fstat(descriptor)
            if checks == 2:
                self.path.unlink()
                self.path.symlink_to(other)
            return result
        with mock.patch.object(native_runtime.os, "fstat", side_effect=replace_link_after_read):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                proof.recheck()

    def test_growth_during_read_is_bounded_by_verified_byte_count(self):
        proof = verify_local_artifact(self.artifact, cache=self.root)
        original_fstat = os.fstat
        checks = 0
        def append_after_stat(descriptor):
            nonlocal checks
            checks += 1
            result = original_fstat(descriptor)
            if checks == 1:
                with self.blob.open("ab") as stream:
                    stream.write(b"extra")
            return result
        with mock.patch.object(native_runtime.os, "fstat", side_effect=append_after_stat):
            with self.assertRaisesRegex(ValueError, "exceeds its pinned size"):
                proof.recheck()

    def test_weight_or_tokenizer_mutation_during_construction_never_becomes_resident(self):
        import main
        config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model("mankei-326m-embedder")
        auxiliary = []
        for name in sorted(native_runtime.LOADER_AUXILIARY):
            (self.snapshot / name).write_bytes(b"{}")
            auxiliary.append(ArtifactFile(name, hashlib.sha256(b"{}").hexdigest(), 2))
        artifact = replace(self.artifact, auxiliary=tuple(auxiliary))
        config = replace(config, artifact=artifact)
        self.assertTrue(has_closed_loader_provenance(config))
        for changed in (self.blob, self.snapshot / "tokenizer.json"):
            with self.subTest(changed=changed.name):
                self.blob.write_bytes(b"fixture")
                (self.snapshot / "tokenizer.json").write_bytes(b"{}")
                manager = main.ModelManager()
                manager.set_worker_thread()
                manager.device = "cpu"
                def construct(_):
                    changed.write_bytes(b"changed" if changed == self.blob else b"[]")
                    return object()
                with mock.patch.object(main, "EMBEDDING_SERVICE_VIEW", SimpleNamespace(require_runtime_model=lambda _: config)), \
                        mock.patch.object(main, "verify_local_artifact", side_effect=lambda value: verify_local_artifact(value, cache=self.root)), \
                        mock.patch.object(manager, "_detect_device", return_value="cpu"), \
                        mock.patch.object(manager, "_construct_model_cpu_sync", side_effect=construct) as constructor:
                    with self.assertRaises(ValueError):
                        manager._load_model_sync("mankei-326m-embedder")
                constructor.assert_called_once_with(config)
                snapshot = manager.snapshot()
                self.assertEqual(snapshot["loaded_models"], [])
                self.assertEqual(snapshot["verified_artifacts"], {})
                self.assertIsNone(manager.loading_model)

    def test_only_closed_local_loader_gets_a_native_api_provenance_claim(self):
        import main
        config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model('mankei-326m-embedder')
        self.assertTrue(has_closed_loader_provenance(config))
        self.assertFalse(has_closed_loader_provenance(replace(config,
            artifact=replace(config.artifact, auxiliary=()))))
        for name in ('nomic-embed-text', 'bge-m3'):
            self.assertFalse(has_closed_loader_provenance(main.EMBEDDING_SERVICE_VIEW.require_runtime_model(name)))

    def test_auxiliary_bytes_are_part_of_identity_and_must_match_before_load(self):
        path = self.snapshot / 'config.json'
        path.write_bytes(b'{}')
        auxiliary = ArtifactFile('config.json', hashlib.sha256(b'{}').hexdigest(), 2)
        artifact = replace(self.artifact, auxiliary=(auxiliary,))
        proof = verify_local_artifact(artifact, cache=self.root)
        path.write_bytes(b'[]')
        with self.assertRaises(ValueError):
            proof.recheck()
        with self.assertRaisesRegex(ValueError, 'digest'):
            verify_local_artifact(artifact, cache=self.root)

    def test_missing_and_external_files_never_get_identity_proof(self):
        self.path.unlink()
        with self.assertRaises(ValueError):
            verify_local_artifact(self.artifact, cache=self.root)
        outside = self.root / "outside"
        outside.write_bytes(b"fixture")
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_local_artifact(self.artifact, cache=self.root)

    def test_dense_loader_selects_declared_weights_without_automatic_format_fallback(self):
        from unittest import mock
        import main
        import loaders
        for name, expected in (("bge-m3", False), ("nomic-embed-text", True)):
            with self.subTest(name=name), mock.patch.object(loaders, "SentenceTransformer") as constructor:
                config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model(name)
                loaders.load_sentence_transformers_cpu(config)
                self.assertEqual(constructor.call_args.kwargs["model_kwargs"], {"use_safetensors": expected})
                ambiguous = replace(config, artifact=replace(config.artifact,
                    weights=(*config.artifact.weights, ArtifactFile("unselected.bin", "f" * 64, 1))))
                constructor.reset_mock()
                with self.assertRaises(ValueError):
                    loaders.load_sentence_transformers_cpu(ambiguous)
                constructor.assert_not_called()

    def test_residency_without_matching_hash_proof_is_not_advertised(self):
        snapshot = {"model_epoch": 1, "loaded_models": ["bge-m3"], "device": "cpu",
                    "worker_accepting": True, "worker_thread_alive": True}
        payload = state_payload(MODEL_CATALOG, snapshot, "fixture-only")
        row = next(item for item in payload["deployments"] if item["reference"] == "bge-m3")
        self.assertFalse(row["loaded"])
        snapshot["verified_artifacts"] = {"bge-m3": row["artifact_fingerprint"]}
        payload = state_payload(MODEL_CATALOG, snapshot, "fixture-only")
        self.assertTrue(next(item for item in payload["deployments"] if item["reference"] == "bge-m3")["loaded"])
        self.assertNotEqual(generation(snapshot), generation({**snapshot, "model_epoch": 2}))
