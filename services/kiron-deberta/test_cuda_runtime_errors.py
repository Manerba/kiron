import asyncio
import json
import unittest
from unittest import mock

from pydantic import ValidationError

import main
import loaders
from kiron_common.model_catalog import ModelEndpoint


def _request_model(endpoint: ModelEndpoint = ModelEndpoint.RERANK) -> str:
    return main.DEBERTA_CATALOG_VIEW.request_default(endpoint).model_name


class _FakeModelManager:
    def __init__(self, *, load_error=None, predict_error=None):
        self._lock = asyncio.Lock()
        self.load_error = load_error
        self.predict_error = predict_error
        self.resets = 0

    async def get_model(self, resolved):
        if self.load_error is not None:
            raise self.load_error
        return (
            _FakeModel(self.predict_error),
            main.DEBERTA_CATALOG_VIEW.require_runtime_model(resolved),
        )

    def force_reset_locked(self):
        self.resets += 1


class _FakeModel:
    def __init__(self, predict_error):
        self.predict_error = predict_error

    def predict(self, pairs, apply_softmax=False):
        if self.predict_error is not None:
            raise self.predict_error
        return [1.0 for _ in pairs]


class CudaRuntimeErrorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_model_manager = main.model_manager

    def tearDown(self):
        main.model_manager = self.original_model_manager

    async def test_load_cuda_runtime_error_returns_503_and_resets(self):
        fake = _FakeModelManager(load_error=RuntimeError("CUDA out of memory"))
        main.model_manager = fake

        resp = await main.load_model_endpoint({"model": _request_model()})

        self.assertEqual(resp.status_code, 503)
        self.assertEqual(fake.resets, 1)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("GPU-Fehler/OOM", body["error"])

    async def test_inference_cuda_runtime_error_returns_503_and_resets(self):
        fake = _FakeModelManager(
            predict_error=RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED")
        )
        main.model_manager = fake

        resp = await main.rerank(
            main.RerankRequest(query="q", documents=["document"])
        )

        self.assertEqual(resp.status_code, 503)
        self.assertEqual(fake.resets, 1)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("GPU-Fehler/OOM", body["error"])

    async def test_generic_runtime_error_stays_500_without_reset(self):
        fake = _FakeModelManager(predict_error=RuntimeError("shape mismatch"))
        main.model_manager = fake

        resp = await main.rerank(
            main.RerankRequest(query="q", documents=["document"])
        )

        self.assertEqual(resp.status_code, 500)
        self.assertEqual(fake.resets, 0)


class RerankTopKTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_model_manager = main.model_manager

    def tearDown(self):
        main.model_manager = self.original_model_manager

    async def test_top_k_zero_returns_no_results_without_model_load(self):
        class NoLoadManager(_FakeModelManager):
            async def get_model(self, resolved):
                raise AssertionError("top_k=0 must not load the model")

        main.model_manager = NoLoadManager()

        resp = await main.rerank(
            main.RerankRequest(
                query="q",
                documents=["a", "b"],
                top_k=0,
            )
        )

        self.assertEqual(resp, {"model": _request_model(), "results": []})

    async def test_positive_top_k_limits_results(self):
        main.model_manager = _FakeModelManager()

        resp = await main.rerank(
            main.RerankRequest(
                query="q",
                documents=["a", "b", "c"],
                top_k=1,
            )
        )

        self.assertEqual(len(resp["results"]), 1)

    def test_negative_top_k_is_rejected(self):
        with self.assertRaises(ValidationError):
            main.RerankRequest(query="q", documents=["a"], top_k=-1)


class LocalOnlyModelLoadTests(unittest.TestCase):
    def test_cross_encoder_load_uses_local_files_only(self):
        manager = main.ModelManager()
        fake_model = object()

        with (
            mock.patch.object(main.torch.cuda, "is_available", return_value=True),
            mock.patch.object(main.torch.cuda, "empty_cache"),
            mock.patch.object(
                loaders,
                "CrossEncoder",
                return_value=fake_model,
            ) as cross_encoder,
        ):
            loaded, config = manager._load_model(_request_model())

        self.assertIs(loaded, fake_model)
        self.assertIs(
            config,
            main.DEBERTA_CATALOG_VIEW.require_runtime_model(_request_model()),
        )
        self.assertIs(manager.model, fake_model)
        _, kwargs = cross_encoder.call_args
        self.assertEqual(kwargs["device"], "cuda")
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["model_kwargs"], {"torch_dtype": main.torch.float16})
        self.assertEqual(kwargs["revision"], config.artifact.revision)

    def test_missing_local_model_is_hard_load_error(self):
        manager = main.ModelManager()
        old_model = object()
        manager.model = old_model
        manager.current_model_name = _request_model()
        manager.config = main.DEBERTA_CATALOG_VIEW.require_runtime_model(
            _request_model()
        )

        with (
            mock.patch.object(main.torch.cuda, "is_available", return_value=True),
            mock.patch.object(main.torch.cuda, "empty_cache"),
            mock.patch.object(
                loaders,
                "CrossEncoder",
                side_effect=OSError("cache miss"),
            ) as cross_encoder,
        ):
            with self.assertRaises(OSError):
                manager._load_model("nli-deberta-v3-base")

        self.assertIs(manager.model, old_model)
        self.assertEqual(manager.current_model_name, _request_model())
        _, kwargs = cross_encoder.call_args
        self.assertTrue(kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
