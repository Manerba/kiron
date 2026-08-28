"""Focused production tests for embedding capability discovery and strict roles."""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402

from kiron_common.embedding_contract import canonical_json  # noqa: E402
from kiron_common.embedding_registry import (  # noqa: E402
    EMBEDDING_REGISTRY,
    MODEL_CATALOG,
)


class _DeterministicModel:
    max_seq_length = 8192

    @staticmethod
    def tokenizer(texts, **_kwargs):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}

    @staticmethod
    def encode(texts, *, batch_size, convert_to_numpy):
        del batch_size, convert_to_numpy
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([float(value + 1) for value in digest[:8]])
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingDiscoveryContractTests(unittest.TestCase):
    def test_systemd_identity_can_read_the_shared_contract_source(self):
        unit_path = main.Path(__file__).resolve().parents[2] / "systemd/kiron-embeddings.service"
        unit = unit_path.read_text(encoding="utf-8")
        supplementary = next(
            line for line in unit.splitlines() if line.startswith("SupplementaryGroups=")
        )
        self.assertIn("kiron-common", supplementary.split("=", 1)[1].split())

    def test_tags_expose_complete_service_groups_with_late_and_colbert_profiles(self):
        payload = main.tags()
        rows = payload["models"]
        self.assertEqual(len(rows), 6)
        by_name = {row["name"]: row for row in rows}

        nomic_profiles = {
            profile["profile_id"]
            for profile in by_name["nomic-embed-text"]["kiron_capabilities"][
                "profiles"
            ]
        }
        self.assertIn("kiron-nomic-dense-v1", nomic_profiles)
        self.assertIn("kiron-nomic-late-v1", nomic_profiles)
        self.assertIn("ollama-nomic-dense-v1", nomic_profiles)

        colbert_profiles = by_name["colbert-xm"]["kiron_capabilities"]["profiles"]
        self.assertEqual(
            [profile["profile_id"] for profile in colbert_profiles],
            ["kiron-colbert-xm-multivector-v1"],
        )
        self.assertEqual(colbert_profiles[0]["endpoint"], "/api/embed_colbert")

    def test_tags_and_show_emit_identical_capability_bytes_for_every_service_group(self):
        for row in main.tags()["models"]:
            shown = main.show_model_endpoint(main.ShowRequest(model=row["name"]))
            self.assertEqual(
                canonical_json(row["kiron_capabilities"]),
                canonical_json(shown["kiron_capabilities"]),
                row["name"],
            )

    def test_tags_and_show_expand_normalized_manifest_profiles(self):
        metadata_fields = {
            "kind",
            "dimensions",
            "max_input_tokens",
            "output_normalized",
            "similarity",
            "input_type",
            "formatting_owner",
            "pipeline",
            "verification",
        }
        for row in main.tags()["models"]:
            shown = main.show_model_endpoint(main.ShowRequest(model=row["name"]))
            capabilities = row["kiron_capabilities"]
            group = MODEL_CATALOG.require(capabilities["canonical_model_id"])
            manifest = group.to_manifest_dict(schema_version=1)
            profiles = {item["id"]: item for item in manifest["profiles"]}
            deployments = {
                item["id"]: item for item in manifest["deployments"]
            }

            self.assertEqual(capabilities["aliases"], manifest["aliases"])
            self.assertEqual(
                canonical_json(capabilities),
                canonical_json(shown["kiron_capabilities"]),
            )
            for capability_profile in capabilities["profiles"]:
                profile = profiles[capability_profile["profile_id"]]
                deployment = deployments[profile["deployment_id"]]
                self.assertEqual(
                    {
                        field: capability_profile[field]
                        for field in metadata_fields
                    },
                    profile["metadata"],
                )
                self.assertEqual(
                    capability_profile["backend"],
                    {
                        "type": deployment["backend"]["type"],
                        "implementation": deployment["backend"]["parameters"][
                            "implementation"
                        ],
                        "implementation_revision": deployment["backend"][
                            "parameters"
                        ]["implementation_revision"],
                    },
                )
                self.assertEqual(
                    capability_profile["artifact"],
                    {
                        field: deployment["artifact"][field]
                        for field in (
                            "repository",
                            "revision",
                            "manifest_digest",
                            "weights",
                            "auxiliary",
                        )
                    },
                )

    def test_model_resolution_accepts_only_literal_registry_names(self):
        self.assertEqual(
            main.normalize_model_name("nomic-embed-text:latest"),
            "nomic-embed-text",
        )
        self.assertEqual(
            main.normalize_colbert_model_name("colbert-xm:latest"),
            "colbert-xm",
        )
        self.assertIsNone(
            main.normalize_model_name("vendor/nomic-embed-text:latest")
        )
        self.assertIsNone(main.normalize_model_name("NOMIC-EMBED-TEXT"))

    def test_all_role_sensitive_profiles_are_strict(self):
        for group in EMBEDDING_REGISTRY.groups:
            for profile in group.capabilities["profiles"]:
                input_type = profile["input_type"]
                if input_type["role_sensitive"] is True:
                    self.assertTrue(input_type["required"])
                    self.assertIsNone(input_type["default"])
                    self.assertEqual(input_type["missing_role_behavior"], "reject")
                else:
                    self.assertFalse(input_type["required"])

    def test_missing_strict_role_has_normative_structured_error(self):
        decision = main.resolve_input_type(
            "nomic-embed-text", "/api/embed", None
        )
        self.assertIsNotNone(decision)
        response = main.input_type_error_response(decision)
        import json

        payload = json.loads(bytes(response.body))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["code"], "missing_required_input_type")
        self.assertEqual(payload["error"]["field_path"], "/input_type")
        self.assertEqual(payload["error"]["model"], "nomic-embed-text:latest")
        self.assertEqual(payload["error"]["profile_id"], "kiron-nomic-dense-v1")
        self.assertEqual(
            payload["error"]["supported"],
            ["search_document", "search_query"],
        )


class DocumentDefaultVectorRegressionTests(unittest.TestCase):
    def _encode(self, model_name: str, input_type: str | None) -> list[list[float]]:
        manager = main.ModelManager(model_slots=1)
        manager.set_worker_thread()
        manager.device = "cpu"
        with mock.patch.object(
            manager,
            "_ensure_model_sync",
            return_value=(_DeterministicModel(), 0),
        ):
            return manager._encode_sync(
                model_name,
                ["Ein Dokument mit Umlauten: Grüße"],
                input_type,
            ).embeddings

    def test_internal_nomic_document_formatting_is_unchanged_by_strict_phase(self):
        missing = self._encode("nomic-embed-text", None)
        explicit = self._encode("nomic-embed-text", "search_document")
        query = self._encode("nomic-embed-text", "search_query")
        self.assertEqual(missing, explicit)
        self.assertNotEqual(explicit, query)

    def test_internal_mankei_document_formatting_is_unchanged_by_strict_phase(self):
        missing = self._encode("mankei-326m-embedder", None)
        explicit = self._encode("mankei-326m-embedder", "search_document")
        query = self._encode("mankei-326m-embedder", "search_query")
        self.assertEqual(missing, explicit)
        self.assertNotEqual(explicit, query)


if __name__ == "__main__":
    unittest.main()
