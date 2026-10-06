"""Actual native ASGI routes with offline resident-worker fixtures."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx
from fastapi import FastAPI

import main
from kiron_common.local_inference import build_resolver_snapshot
from native_api import create_native_router
from native_runtime import generation
from model_worker import StaleGenerationError
from kiron_common.local_inference.embedding_native import request_fingerprint


class NativeApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.catalog = main.MODEL_CATALOG
        self.model = build_resolver_snapshot(self.catalog, ()).resolve("kiron-bge-m3-dense-v1")
        self.deployment = self.model.deployment
        self.snapshot = {"model_epoch": 7, "loaded_models": [self.deployment.reference],
            "verified_artifacts": {self.deployment.reference: self.deployment.artifact_identity.fingerprint},
            "device": "cpu", "worker_accepting": True, "worker_thread_alive": True,
            "current_job": None, "queue_depth": 0}
        self.result = SimpleNamespace(embeddings=[[.25] * 1024], prompt_eval_count=13,
            execution_epoch=7, execution_artifact=self.deployment.artifact_identity.fingerprint)
        self.worker = SimpleNamespace(snapshot=lambda: dict(self.snapshot),
                                      encode=mock.AsyncMock(side_effect=lambda *a, **k: self.result))
        app = FastAPI()
        app.include_router(create_native_router(self.catalog, main.EMBEDDING_SERVICE_VIEW, lambda: self.worker))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture")
        self.addAsyncCleanup(self.client.aclose)
        self.body = {"version": 1, "request_id": "native-fixture", "generation": generation(self.snapshot),
            "catalog_digest": self.catalog.catalog_digest, "deployment_id": self.deployment.id,
            "profile_id": self.model.profile_id, "input_type": None,
            "artifact_fingerprint": self.deployment.artifact_identity.fingerprint,
            "configuration_fingerprint": self.deployment.configuration_fingerprint,
            "inputs": ["  Original Grün  "], "dimensions": 1024}

    async def test_native_identity_and_roles_preserve_original_input(self):
        for role in (None, "search_query", "search_document"):
            with self.subTest(role=role):
                response = await self.client.post("/api/inference/embed", json={**self.body, "input_type": role})
                self.assertEqual(response.status_code, 200, response.text)
                value = response.json()
                self.assertTrue(value["done"])
                self.assertEqual(value["usage"], {"input_tokens": 13, "output_tokens": 0})
                self.assertEqual(value["token_counting"], "forward_attention_mask_v1")
                self.worker.encode.assert_awaited_with(self.deployment.reference, self.body["inputs"], role,
                    expected_artifact=self.body["artifact_fingerprint"], expected_epoch=7)

    async def test_identity_or_unproven_residency_reject_before_worker(self):
        for key, value in (("generation", {"boot_id": "old", "process_id": "7"}),
            ("artifact_fingerprint", "foreign"), ("profile_id", "kiron-nomic-dense-v1"),
            ("configuration_fingerprint", "foreign"), ("dimensions", 3), ("catalog_digest", "foreign")):
            with self.subTest(key=key):
                response = await self.client.post("/api/inference/embed", json={**self.body, key: value})
                self.assertEqual(response.status_code, 409)
                proof = response.json()
                self.assertIs(proof["rejected"], True)
                self.assertEqual(proof["request_sha256"], request_fingerprint({**self.body, key: value}))
                self.assertEqual(proof["request_id"], self.body["request_id"])
        self.snapshot["verified_artifacts"] = {}
        self.assertEqual((await self.client.post("/api/inference/embed", json=self.body)).status_code, 409)
        self.worker.encode.assert_not_awaited()

    async def test_queued_stale_generation_has_the_same_definitive_rejection(self):
        self.worker.encode.side_effect = StaleGenerationError("resident identity changed")
        response = await self.client.post("/api/inference/embed", json=self.body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"version": 1,
            "request_id": self.body["request_id"], "generation": self.body["generation"],
            "request_sha256": request_fingerprint(self.body), "rejected": True,
            "error": {"code": "identity_conflict"}})
        # An arbitrary failure, including a generic ValueError, cannot assert
        # the before-execution property of the specific worker exception.
        for error in (ValueError("identity changed"), RuntimeError("execution failed")):
            self.worker.encode.side_effect = error
            failed = await self.client.post("/api/inference/embed", json=self.body)
            self.assertEqual(failed.status_code, 503)
            self.assertNotIn("rejected", failed.json())

    async def test_unknown_end_generation_or_invalid_usage_never_returns_done(self):
        for count in (None, True, 0):
            with self.subTest(count=count):
                self.result.prompt_eval_count = count
                response = await self.client.post("/api/inference/embed", json=self.body)
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("done", response.json())
        self.result.prompt_eval_count = 13
        for field, invalid in (("execution_epoch", None), ("execution_epoch", True),
                ("execution_epoch", 8), ("execution_artifact", None), ("execution_artifact", "foreign")):
            with self.subTest(field=field, value=invalid):
                original = getattr(self.result, field)
                setattr(self.result, field, invalid)
                response = await self.client.post("/api/inference/embed", json=self.body)
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("done", response.json())
                setattr(self.result, field, original)

    async def test_later_cache_mutation_does_not_erase_own_completed_execution(self):
        async def reload(*args, **kwargs):
            self.snapshot["model_epoch"] += 1
            return self.result
        self.worker.encode.side_effect = reload
        response = await self.client.post("/api/inference/embed", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["done"])
        self.assertEqual(response.json()["generation"], self.body["generation"])

    def test_manager_checks_queued_generation_without_any_autoload(self):
        manager = main.ModelManager()
        manager.set_worker_thread()
        manager.device = "cpu"
        encoder = object()
        with manager.state_lock:
            manager._store_model_unlocked(self.deployment.reference, encoder)
            manager._artifact_proofs[self.deployment.reference] = (
                id(encoder), SimpleNamespace(fingerprint=self.body["artifact_fingerprint"]))
        with mock.patch.object(manager, "_ensure_model_sync") as ensure:
            with self.assertRaisesRegex(ValueError, "identity changed"):
                manager._encode_sync(self.deployment.reference, ["x"], None,
                    expected_artifact=self.body["artifact_fingerprint"], expected_epoch=-1)
            ensure.assert_not_called()
