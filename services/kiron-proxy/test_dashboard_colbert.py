"""Dashboard-Vertraege fuer ColBERT-Anzeige und -Kontrollen."""

import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from kiron_common.local_model_registry import LocalModelProvider, RegistryEntry
from kiron_common.model_catalog import LoaderType


PROXY_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROXY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(PROXY_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(PROXY_DIR))
    return module


def _async_return(value):
    async def _coro(*args, **kwargs):
        return value

    return _coro


def _metrics_cache(embedding_health: dict) -> dict:
    return {
        "system": {
            "embedding": {"running": True, **embedding_health},
            "deberta": {"running": False, "status": "down"},
        }
    }


class _FakeResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class _FakeOllamaClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, path: str):
        if path == "/api/tags":
            return _FakeResponse(200, {"models": []})
        if path == "/api/ps":
            return _FakeResponse(200, {"models": []})
        raise AssertionError(f"unexpected GET {path}")


class _FakeOllamaModelDetailsClient(_FakeOllamaClient):
    show_calls: list[str] = []

    async def get(self, path: str):
        if path == "/api/tags":
            return _FakeResponse(
                200,
                {
                    "models": [
                        {
                            "name": "qwen3:8b",
                            "size": 6_005_854_336,
                            "details": {
                                "family": "qwen3",
                                "format": "gguf",
                                "parameter_size": "8.2B",
                                "quantization_level": "Q4_K_M",
                            },
                        }
                    ]
                },
            )
        if path == "/api/ps":
            return _FakeResponse(
                200,
                {
                    "models": [
                        {
                            "name": "qwen3:8b",
                            "size": 6_005_854_336,
                            "size_vram": 6_005_854_336,
                            "context_length": 4096,
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected GET {path}")

    async def post(self, path: str, json: dict):
        assert path == "/api/show"
        self.show_calls.append(json["model"])
        return _FakeResponse(
            200,
            {
                "model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.context_length": 40960,
                }
            },
        )


class _FakeWarmupClient:
    calls: list[tuple[str, dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, json: dict):
        self.calls.append((url, json))
        return _FakeResponse(
            200,
            {
                "model": "colbert-xm",
                "embeddings": [[[1.0, 0.0]]],
                "prompt_eval_count": 1,
            },
        )


class _EmptyRegistrationService:
    def list_models(self):
        return ()


class _QwenRegistrationService:
    def __init__(self):
        self._entry = RegistryEntry.create(
            provider=LocalModelProvider.OLLAMA,
            reference="qwen3:8b",
            display_name="qwen3:8b",
            loader=LoaderType.OLLAMA,
        )

    def list_models(self):
        return (self._entry,)


class _AllowedGpuOp:
    def __init__(self):
        self.allowed = True
        self.decision = None
        self.clear_marker = False


class _GpuOperationContext:
    def __init__(self):
        self.op = _AllowedGpuOp()

    async def __aenter__(self):
        return self.op

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DashboardColbertTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_module("kiron_proxy_app_colbert_test", "app.py")
        self._registration_patcher = mock.patch.object(
            self.app_mod,
            "get_model_registration_service",
            return_value=_EmptyRegistrationService(),
        )
        self._registration_patcher.start()

    def tearDown(self):
        self._registration_patcher.stop()

    async def test_local_models_includes_embedding_only_colbert_row(self):
        health = {
            "status": "ok",
            "current_model": "colbert-xm",
            "loaded_models": ["colbert-xm"],
            "loading_model": None,
            "available_models": ["nomic-embed-text"],
            "available_colbert_models": ["colbert-xm"],
        }
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", _FakeOllamaClient), \
             mock.patch.object(
                 self.app_mod,
                 "get_cached_payload",
                 return_value=_metrics_cache(health),
             ), \
             mock.patch.object(
                 self.app_mod,
                 "service_huggingface_inventory",
                 return_value=frozenset(),
             ):
            resp = await self.app_mod.get_local_models()

        rows = {item["name"]: item for item in resp["models"]}
        self.assertIn("colbert-xm", rows)
        colbert = rows["colbert-xm"]
        self.assertTrue(colbert["embedding_only"])
        self.assertEqual(colbert["model_type"], "colbert")
        self.assertEqual(colbert["embedding_kind"], "colbert")
        self.assertTrue(colbert["embedding_active"])
        self.assertEqual(
            resp["embedding_service"]["available_colbert_models"],
            ["colbert-xm"],
        )

    async def test_local_models_marks_all_loaded_embedding_slots_active(self):
        health = {
            "status": "ok",
            "current_model": "colbert-xm",
            "loaded_models": ["nomic-embed-text", "colbert-xm"],
            "model_slots": 2,
            "loading_model": None,
            "available_models": ["nomic-embed-text"],
            "available_colbert_models": ["colbert-xm"],
        }
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", _FakeOllamaClient), \
             mock.patch.object(
                 self.app_mod,
                 "get_cached_payload",
                 return_value=_metrics_cache(health),
             ), \
             mock.patch.object(
                 self.app_mod,
                 "service_huggingface_inventory",
                 return_value=frozenset(),
             ):
            resp = await self.app_mod.get_local_models()

        rows = {item["name"]: item for item in resp["models"]}
        self.assertTrue(rows["nomic-embed-text"]["embedding_active"])
        self.assertTrue(rows["colbert-xm"]["embedding_active"])
        self.assertEqual(
            resp["embedding_service"]["loaded_models"],
            ["colbert-xm", "nomic-embed-text"],
        )
        self.assertEqual(resp["embedding_service"]["model_slots"], 2)

    async def test_local_models_does_not_use_current_model_as_active_fallback(self):
        health = {
            "status": "ok",
            "current_model": "colbert-xm",
            "loaded_models": [],
            "loading_model": None,
            "available_models": ["nomic-embed-text"],
            "available_colbert_models": ["colbert-xm"],
        }
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", _FakeOllamaClient), \
             mock.patch.object(
                 self.app_mod,
                 "get_cached_payload",
                 return_value=_metrics_cache(health),
             ), \
             mock.patch.object(
                 self.app_mod,
                 "service_huggingface_inventory",
                 return_value=frozenset(),
             ):
            resp = await self.app_mod.get_local_models()

        rows = {item["name"]: item for item in resp["models"]}
        self.assertFalse(rows["colbert-xm"]["embedding_active"])
        self.assertEqual(resp["embedding_service"]["loaded_models"], [])

    async def test_local_models_separates_native_and_runtime_ollama_context(self):
        _FakeOllamaModelDetailsClient.show_calls = []
        registrations = _QwenRegistrationService()
        with mock.patch.object(
            self.app_mod.httpx,
            "AsyncClient",
            _FakeOllamaModelDetailsClient,
        ), mock.patch.object(
            self.app_mod,
            "get_model_registration_service",
            return_value=registrations,
        ), mock.patch.object(
            self.app_mod,
            "get_cached_payload",
            return_value=_metrics_cache({"status": "no_model"}),
        ), mock.patch.object(
            self.app_mod,
            "service_huggingface_inventory",
            return_value=frozenset(),
        ):
            resp = await self.app_mod.get_local_models()

        qwen = next(item for item in resp["models"] if item["name"] == "qwen3:8b")
        self.assertEqual(qwen["native_context_length"], 40960)
        self.assertEqual(qwen["runtime_context_length"], 4096)
        self.assertIsNone(qwen["catalog_context_length"])
        self.assertEqual(_FakeOllamaModelDetailsClient.show_calls, ["qwen3:8b"])

    async def test_colbert_warmup_posts_token_level_endpoint(self):
        _FakeWarmupClient.calls = []
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", _FakeWarmupClient), \
             mock.patch.object(
                 self.app_mod.vram_lease,
                 "gpu_service_operation",
                 return_value=_GpuOperationContext(),
             ):
            resp = await self.app_mod.warmup_colbert_model({"model": "colbert-xm"})

        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["model"], "colbert-xm")
        self.assertEqual(
            _FakeWarmupClient.calls,
            [
                (
                    "http://127.0.0.1:11436/api/embed_colbert",
                    {
                        "model": "colbert-xm",
                        "input": ["warmup"],
                        "language": "de",
                    },
                )
            ],
        )


class MetricsColbertTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.metrics_mod = _load_module("kiron_proxy_metrics_colbert_test", "metrics.py")

    async def test_embedding_status_reports_colbert_models(self):
        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url: str):
                return _FakeResponse(
                    200,
                    {
                        "status": "ok",
                        "current_model": "colbert-xm",
                        "loaded_models": ["nomic-embed-text", "colbert-xm"],
                        "model_slots": 2,
                        "loading_model": None,
                        "available_models": ["nomic-embed-text"],
                        "available_colbert_models": ["colbert-xm"],
                    },
                )

        with mock.patch.object(self.metrics_mod.httpx, "AsyncClient", _Client):
            resp = await self.metrics_mod.get_embedding_status()

        self.assertEqual(resp["model"], "colbert-xm")
        self.assertEqual(resp["available_colbert_models"], ["colbert-xm"])
        self.assertEqual(resp["loaded_models"], ["nomic-embed-text", "colbert-xm"])
        self.assertEqual(resp["model_slots"], 2)


class RuntimeConfigTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_module("kiron_proxy_app_config_test", "app.py")

    async def test_runtime_config_get_reports_desired_and_effective_slots(self):
        health = {
            "status": "ok",
            "current_model": "colbert-xm",
            "loaded_models": ["nomic-embed-text", "colbert-xm"],
            "model_slots": 2,
            "loading_model": None,
        }
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(
                 self.app_mod,
                 "RUNTIME_CONFIG_FILE",
                 Path(tmp) / "runtime_config.json",
             ), \
             mock.patch.object(
                 self.app_mod,
                 "_json_service_health",
                 new=_async_return((True, health, 200)),
             ):
            resp = await self.app_mod.get_runtime_config()

        self.assertEqual(resp["embedding"]["model_slots"], 2)
        self.assertEqual(resp["embedding"]["effective_model_slots"], 2)
        self.assertFalse(resp["embedding"]["pending_restart"])
        self.assertEqual(
            resp["embedding"]["loaded_models"],
            ["colbert-xm", "nomic-embed-text"],
        )

    async def test_update_embedding_config_writes_slots_and_restarts(self):
        async def _restart_ok():
            return True, {"status": "no_model", "model_slots": 3}, None

        health = {
            "status": "no_model",
            "current_model": None,
            "loaded_models": [],
            "model_slots": 3,
            "loading_model": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "runtime_config.json"
            with mock.patch.object(self.app_mod, "RUNTIME_CONFIG_FILE", config_path), \
                 mock.patch.object(
                     self.app_mod,
                     "_json_service_health",
                     new=_async_return((True, health, 503)),
                 ), \
                 mock.patch.object(
                     self.app_mod,
                     "_restart_embedding_service_with_wait",
                     new=_restart_ok,
                 ):
                resp = await self.app_mod.update_embedding_config(
                    {"model_slots": 3, "restart": True}
                )

            saved = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["embedding"]["model_slots"], 3)
        self.assertEqual(resp["embedding"]["model_slots"], 3)
        self.assertEqual(resp["embedding"]["effective_model_slots"], 3)
        self.assertTrue(resp["restart"]["requested"])
        self.assertTrue(resp["restart"]["ok"])

    async def test_update_embedding_config_rejects_bool_float_and_string_slots(self):
        invalid_values = [True, False, 2.5, "2"]
        for value in invalid_values:
            with self.subTest(value=value):
                resp = await self.app_mod.update_embedding_config(
                    {"model_slots": value, "restart": False}
                )

                self.assertEqual(resp.status_code, 400)

    def test_embedding_health_loaded_models_has_no_current_model_fallback(self):
        loaded = self.app_mod._loaded_models_from_embedding_health({
            "current_model": "colbert-xm",
        })

        self.assertEqual(loaded, set())


if __name__ == "__main__":
    unittest.main()
