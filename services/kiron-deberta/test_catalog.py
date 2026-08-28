"""Catalog, endpoint, and loader-registry contracts for kiron-deberta."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_view  # noqa: E402
import loaders  # noqa: E402
import main  # noqa: E402
from kiron_common.model_catalog import (  # noqa: E402
    BackendType,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
)


EXPECTED_MODELS = (
    "nli-deberta-v3-base",
    "ms-marco-MiniLM-L-6-v2",
    "bge-reranker-v2-m3",
    "mdeberta-v3-xnli",
    "mankei-326m-reranker",
)


def _production_manifests() -> list[dict]:
    return [
        group.to_manifest_dict(schema_version=1)
        for group in main.SHARED_MODEL_CATALOG.groups
    ]


def _manifest(canonical_model_id: str) -> dict:
    return copy.deepcopy(
        next(
            group
            for group in _production_manifests()
            if group["canonical_model_id"] == canonical_model_id
        )
    )


def _temporary_standard_manifest() -> dict:
    return {
        "schema_version": 1,
        "canonical_model_id": "temporary-reranker",
        "aliases": ["example/temporary-reranker"],
        "deployments": [
            {
                "id": "kiron-deberta-temporary-v1.deployment",
                "backend": {
                    "type": "kiron_deberta",
                    "parameters": {
                        "implementation": "sentence-transformers-cross-encoder",
                        "implementation_revision": "5.6.0",
                        "model_name": "temporary-reranker",
                    },
                },
                "artifact": {
                    "type": "huggingface",
                    "repository": "example/temporary-reranker",
                    "revision": "f" * 40,
                    "manifest_digest": None,
                    "trust_remote_code": False,
                    "weights": [
                        {"path": "model.safetensors", "sha256": "e" * 64}
                    ],
                    "auxiliary": [],
                    "metadata": {},
                },
                "routes": [
                    {"task": "rerank", "endpoint": "/api/rerank"},
                    {"task": "nli", "endpoint": "/api/score"},
                ],
                "loader": {
                    "type": "cross_encoder",
                    "parameters": {
                        "max_length": 512,
                        "torch_dtype": "float16",
                    },
                },
                "metadata": {
                    "service": {
                        "display_order": 6,
                        "labels": None,
                        "rerank_label": None,
                        "size": 123456,
                    }
                },
            }
        ],
        "profiles": [
            {
                "id": "kiron-temporary-rerank-v1",
                "deployment_id": "kiron-deberta-temporary-v1.deployment",
                "task": "rerank",
                "endpoint": "/api/rerank",
                "default_for_endpoint": True,
                "metadata": {},
            },
            {
                "id": "kiron-temporary-score-v1",
                "deployment_id": "kiron-deberta-temporary-v1.deployment",
                "task": "nli",
                "endpoint": "/api/score",
                "default_for_endpoint": True,
                "metadata": {},
            },
        ],
        "request_defaults": [],
        "metadata": {},
    }


class CatalogViewTests(unittest.TestCase):
    def test_production_models_routes_defaults_and_order(self):
        view = main.DEBERTA_CATALOG_VIEW

        self.assertEqual(view.available_model_names(), EXPECTED_MODELS)
        self.assertEqual(
            view.available_model_names(ModelEndpoint.RERANK), EXPECTED_MODELS
        )
        self.assertEqual(
            view.available_model_names(ModelEndpoint.SCORE), EXPECTED_MODELS
        )
        self.assertEqual(
            view.request_default(ModelEndpoint.RERANK).model_name,
            "ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(
            view.request_default(ModelEndpoint.SCORE).model_name,
            "ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(
            main.RerankRequest.model_fields["model"].default,
            "ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(
            main.ScoreRequest.model_fields["model"].default,
            "ms-marco-MiniLM-L-6-v2",
        )

    def test_labels_rerank_semantics_sizes_and_loader_parameters(self):
        view = main.DEBERTA_CATALOG_VIEW
        nli = view.require_runtime_model("nli-deberta-v3-base")
        multilingual = view.require_runtime_model("mdeberta-v3-xnli")
        mankei = view.require_runtime_model("mankei-326m-reranker")

        self.assertEqual(
            nli.labels, ("contradiction", "entailment", "neutral")
        )
        self.assertEqual(nli.rerank_label, "entailment")
        self.assertEqual(
            multilingual.labels, ("entailment", "neutral", "contradiction")
        )
        self.assertEqual(multilingual.rerank_label, "entailment")
        self.assertEqual(nli.max_length, 512)
        self.assertEqual(nli.precision, "fp16")
        self.assertEqual(nli.size, 748_850_969)
        self.assertEqual(mankei.max_length, 192)
        self.assertEqual(mankei.precision, "bf16")
        self.assertEqual(mankei.size, 652_890_301)
        self.assertIsInstance(
            mankei.loader_parameters,
            catalog_view.MankeiLastTokenLoaderParameters,
        )
        self.assertEqual(
            mankei.loader_parameters.pair_template,
            "Frage: {query}\nPassage: {passage}",
        )

    def test_all_artifacts_are_pinned_to_verified_snapshots_and_hashes(self):
        expected = {
            "nli-deberta-v3-base": (
                "cross-encoder/nli-deberta-v3-base",
                "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
                "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
            ),
            "ms-marco-MiniLM-L-6-v2": (
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "c5ee24cb16019beea0893ab7796b1df96625c6b8",
                "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
            ),
            "bge-reranker-v2-m3": (
                "BAAI/bge-reranker-v2-m3",
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
            ),
            "mdeberta-v3-xnli": (
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
                "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541",
            ),
            "mankei-326m-reranker": (
                "keyvan-ai/Mankei-326M-Reranker",
                "e1219e343a9ebdcf2e381c058a480e34defbeb95",
                "34566de5240d16995f2830c00ea5981d5f3013f165b2c75a440100b7c49375d8",
            ),
        }
        for model_name, coordinates in expected.items():
            model = main.DEBERTA_CATALOG_VIEW.require_runtime_model(model_name)
            self.assertEqual(
                (
                    model.artifact.repository,
                    model.artifact.revision,
                    model.artifact.weights[0].sha256,
                ),
                coordinates,
            )
        mankei = main.DEBERTA_CATALOG_VIEW.require_runtime_model(
            "mankei-326m-reranker"
        )
        self.assertEqual(mankei.artifact.auxiliary[0].path, "head.pt")
        self.assertEqual(
            mankei.artifact.auxiliary[0].sha256,
            "ce0cf93891cbee260a7acfb9c5d66bf3d9c13c68e81fd9eb5cac99838b543edf",
        )

    def test_resolution_is_exact_and_aliases_are_explicit(self):
        self.assertEqual(
            main.normalize_model_name("cross-encoder/nli-deberta-v3-base"),
            "nli-deberta-v3-base",
        )
        self.assertEqual(
            main.normalize_model_name("keyvan-ai/Mankei-326M-Reranker"),
            "mankei-326m-reranker",
        )
        for candidate in (
            "vendor/nli-deberta-v3-base",
            "nli-deberta-v3-base:latest",
            "NLI-DEBERTA-V3-BASE",
            " nli-deberta-v3-base",
            "mankei-326m-reranker:latest",
            None,
            1,
        ):
            self.assertIsNone(main.normalize_model_name(candidate), candidate)

    def test_loader_registry_is_explicit_and_complete(self):
        self.assertEqual(
            set(loaders.LOADER_REGISTRY),
            {LoaderType.CROSS_ENCODER, LoaderType.MANKEI_LAST_TOKEN},
        )
        for model in main.DEBERTA_CATALOG_VIEW.models:
            self.assertIs(
                main.DEBERTA_CATALOG_VIEW.loader_for(model),
                loaders.LOADER_REGISTRY[model.loader_type],
            )

    def test_temporary_standard_model_is_manifest_only_and_loader_bound(self):
        catalog = ModelCatalog.from_manifests(
            [*_production_manifests(), _temporary_standard_manifest()]
        )
        view = catalog_view.build_deberta_service_view(
            catalog, loaders.LOADER_REGISTRY
        )
        model = view.resolve(
            "example/temporary-reranker", ModelEndpoint.RERANK
        )

        self.assertIsNotNone(model)
        self.assertEqual(model.model_name, "temporary-reranker")
        self.assertEqual(model.artifact.repository, "example/temporary-reranker")
        self.assertIs(view.loader_for(model), loaders.load_cross_encoder_cuda)
        self.assertNotIn(
            "temporary-reranker",
            Path(main.__file__).read_text(encoding="utf-8"),
        )

    def test_wrong_backend_unknown_loader_and_missing_fields_fail_fast(self):
        wrong_backend = _manifest("ms-marco-MiniLM-L-6-v2")
        wrong_backend["deployments"][0]["backend"]["type"] = "kiron_embeddings"
        with self.assertRaisesRegex(
            catalog_view.DebertaServiceCatalogError,
            "belongs to another backend",
        ):
            catalog_view.build_deberta_service_view(
                ModelCatalog.from_manifests([wrong_backend]),
                loaders.LOADER_REGISTRY,
            )

        wrong_loader = _manifest("ms-marco-MiniLM-L-6-v2")
        wrong_loader["deployments"][0]["loader"] = {
            "type": "sentence_transformers",
            "parameters": {"additional_role_template": None},
        }
        with self.assertRaisesRegex(
            catalog_view.DebertaServiceCatalogError,
            "is not allowed",
        ):
            catalog_view.build_deberta_service_view(
                ModelCatalog.from_manifests([wrong_loader]),
                loaders.LOADER_REGISTRY,
            )

        missing_field = _manifest("ms-marco-MiniLM-L-6-v2")
        del missing_field["deployments"][0]["metadata"]["service"]["size"]
        with self.assertRaisesRegex(
            catalog_view.DebertaServiceCatalogError,
            "missing required fields: size",
        ):
            catalog_view.build_deberta_service_view(
                ModelCatalog.from_manifests([missing_field]),
                loaders.LOADER_REGISTRY,
            )

    def test_unregistered_loader_fails_fast(self):
        with self.assertRaisesRegex(
            catalog_view.DebertaServiceCatalogError,
            "cross_encoder.*not registered",
        ):
            catalog_view.build_deberta_service_view(
                main.SHARED_MODEL_CATALOG,
                {
                    LoaderType.MANKEI_LAST_TOKEN:
                        loaders.load_mankei_last_token_cuda,
                },
            )

    def test_tags_and_existing_model_list_endpoint_are_catalog_derived(self):
        self.assertEqual(
            [item["name"] for item in main.tags()["models"]],
            list(EXPECTED_MODELS),
        )
        paths = {route.path for route in main.app.routes}
        self.assertIn("/api/tags", paths)
        self.assertNotIn("/api/models", paths)


class _Predictor:
    def predict(self, pairs, *, apply_softmax=False):
        if apply_softmax:
            values = [[0.2, 0.9, 0.1], [0.1, 0.3, 0.6]]
            return np.asarray(values[: len(pairs)], dtype=np.float32)
        return np.asarray([0.8, 0.2][: len(pairs)], dtype=np.float32)


class _EndpointManager:
    def __init__(self, model_name: str):
        self._lock = asyncio.Lock()
        self.model_name = model_name
        self.resets = 0

    async def get_model(self, _resolved):
        return (
            _Predictor(),
            main.DEBERTA_CATALOG_VIEW.require_runtime_model(self.model_name),
        )

    def force_reset_locked(self):
        self.resets += 1


class EndpointSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_nli_label_order_and_rerank_label_are_preserved(self):
        manager = _EndpointManager("nli-deberta-v3-base")
        with (
            mock.patch.object(main, "model_manager", manager),
            mock.patch.object(
                main.torch.amp,
                "autocast",
                side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
            ),
        ):
            scored = await main.score(
                main.ScoreRequest(
                    model="nli-deberta-v3-base",
                    pairs=[["a", "b"]],
                )
            )
            reranked = await main.rerank(
                main.RerankRequest(
                    model="nli-deberta-v3-base",
                    query="q",
                    documents=["first", "second"],
                )
            )

        self.assertEqual(
            list(scored["results"][0]),
            ["contradiction", "entailment", "neutral"],
        )
        self.assertEqual(
            [item["document"] for item in reranked["results"]],
            ["first", "second"],
        )

    def test_health_reports_catalog_precision_and_digest(self):
        manager = mock.Mock()
        manager.snapshot.return_value = {
            "loading_model": None,
            "current_model": "mankei-326m-reranker",
            "loaded_models": ["mankei-326m-reranker"],
        }
        with mock.patch.object(main, "model_manager", manager), mock.patch.object(
            main,
            "_local_model_inventory",
            return_value=main.LocalModelInventory(),
        ):
            response = main.health()
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["precision"], "bf16")
        self.assertEqual(payload["available_models"], list(EXPECTED_MODELS))
        self.assertEqual(
            payload["catalog_digest"],
            "sha256:c91229d7ea472b49d87f6344dbfb640fc760f43e8cace398421d5b364452e6f6",
        )
        self.assertEqual(payload["loaded_models"], ["mankei-326m-reranker"])


if __name__ == "__main__":
    unittest.main()
