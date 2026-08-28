"""Validation- und Endpoint-Tests fuer den Embedding-Service."""

import asyncio
import copy
import hashlib
import inspect
import json
import os
import sys
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402
import loaders  # noqa: E402
from catalog_view import (  # noqa: E402
    EmbeddingServiceCatalogError,
    SentenceTransformersLoaderParameters,
    build_embedding_service_view,
)
from kiron_common.model_catalog import (  # noqa: E402
    BackendType,
    CatalogValidationError,
    LoaderType,
    ModelCatalog,
    ModelEndpoint,
)
from model_worker import (  # noqa: E402
    ColbertEmbedResult,
    CudaRuntimeError,
    EncodeResult,
    LateEmbedResult,
    NonFiniteEmbeddingError,
    WorkerQueueFullError,
    WorkerStoppedError,
)


def _temporary_standard_manifest() -> dict:
    manifest = copy.deepcopy(
        main.MODEL_CATALOG.require("bge-m3:latest").to_manifest_dict(
            schema_version=1
        )
    )
    deployment = next(
        item
        for item in manifest["deployments"]
        if item["backend"]["type"] == BackendType.KIRON_EMBEDDINGS.value
    )
    profile = next(
        item
        for item in manifest["profiles"]
        if item["deployment_id"] == deployment["id"]
    )
    manifest["canonical_model_id"] = "test-standard-embed:latest"
    manifest["aliases"] = ["test-standard-embed"]
    deployment["id"] = "test-standard-embed.deployment"
    deployment["backend"]["parameters"]["model_name"] = "test-standard-embed"
    deployment["artifact"]["repository"] = "example/test-standard-embed"
    deployment["artifact"]["revision"] = "a" * 40
    deployment["artifact"]["weights"] = [
        {"path": "model.safetensors", "sha256": "b" * 64}
    ]
    profile["id"] = "test-standard-embed.profile"
    profile["deployment_id"] = deployment["id"]
    profile["metadata"]["pipeline"]["tokenizer"] = {
        "repository": deployment["artifact"]["repository"],
        "revision": deployment["artifact"]["revision"],
    }
    manifest["deployments"] = [deployment]
    manifest["profiles"] = [profile]
    manifest["request_defaults"] = []
    return manifest


class CatalogServiceViewTests(unittest.TestCase):
    def test_loader_registry_is_explicit_and_keyed_only_by_loader_type(self):
        self.assertEqual(
            set(loaders.LOADER_REGISTRY),
            {
                LoaderType.SENTENCE_TRANSFORMERS,
                LoaderType.TRANSFORMERS_LAST_TOKEN,
                LoaderType.COLBERT_XMOD,
            },
        )
        self.assertTrue(all(callable(item) for item in loaders.LOADER_REGISTRY.values()))

    def test_temporary_standard_manifest_is_discovered_and_loader_bound(self):
        catalog = ModelCatalog.from_manifests([_temporary_standard_manifest()])
        view = build_embedding_service_view(
            catalog,
            loaders.LOADER_REGISTRY,
        )

        model = view.resolve("test-standard-embed:latest", ModelEndpoint.EMBED)
        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model.model_name, "test-standard-embed")
        self.assertIs(
            view.resolve("test-standard-embed", ModelEndpoint.EMBED),
            model,
        )
        self.assertIsNone(view.resolve("TEST-STANDARD-EMBED"))
        self.assertIsNone(view.resolve("vendor/test-standard-embed:latest"))
        self.assertEqual(
            view.available_model_names(ModelEndpoint.EMBED),
            ("test-standard-embed",),
        )
        self.assertIsInstance(
            model.loader_parameters,
            SentenceTransformersLoaderParameters,
        )
        self.assertIs(
            view.loader_for(model),
            loaders.load_sentence_transformers_cpu,
        )
        constructed = object()
        with mock.patch.object(
            loaders,
            "SentenceTransformer",
            return_value=constructed,
        ) as constructor:
            loaded = view.loader_for(model)(model)
        self.assertIs(loaded, constructed)
        constructor.assert_called_once_with(
            "example/test-standard-embed",
            cache_folder=mock.ANY,
            device="cpu",
            trust_remote_code=False,
            revision="a" * 40,
            local_files_only=True,
        )

    def test_service_view_is_deeply_immutable(self):
        catalog = ModelCatalog.from_manifests([_temporary_standard_manifest()])
        view = build_embedding_service_view(
            catalog,
            {LoaderType.SENTENCE_TRANSFORMERS: mock.Mock()},
        )
        model = view.models[0]
        with self.assertRaises(FrozenInstanceError):
            model.model_name = "changed"
        with self.assertRaises(TypeError):
            view._names["changed"] = model

    def test_unknown_loader_type_is_rejected_by_catalog(self):
        manifest = _temporary_standard_manifest()
        manifest["deployments"][0]["loader"]["type"] = "unknown_loader"
        with self.assertRaisesRegex(
            CatalogValidationError,
            r"loader/type.*unknown_loader",
        ):
            ModelCatalog.from_manifests([manifest])

    def test_backend_foreign_loader_fails_fast(self):
        manifest = _temporary_standard_manifest()
        manifest["deployments"][0]["loader"]["type"] = (
            LoaderType.CROSS_ENCODER.value
        )
        catalog = ModelCatalog.from_manifests([manifest])
        with self.assertRaisesRegex(
            EmbeddingServiceCatalogError,
            r"cross_encoder.*not allowed.*kiron_embeddings",
        ):
            build_embedding_service_view(
                catalog,
                {LoaderType.CROSS_ENCODER: mock.Mock()},
            )

    def test_wrong_backend_type_is_explicitly_filtered_out(self):
        manifest = _temporary_standard_manifest()
        manifest["deployments"][0]["backend"]["type"] = (
            BackendType.KIRON_DEBERTA.value
        )
        catalog = ModelCatalog.from_manifests([manifest])
        view = build_embedding_service_view(
            catalog,
            {LoaderType.SENTENCE_TRANSFORMERS: mock.Mock()},
        )
        self.assertEqual(view.models, ())
        self.assertIsNone(view.resolve("test-standard-embed"))

    def test_configured_loader_must_exist_in_registry(self):
        catalog = ModelCatalog.from_manifests([_temporary_standard_manifest()])
        with self.assertRaisesRegex(
            EmbeddingServiceCatalogError,
            r"sentence_transformers.*not registered",
        ):
            build_embedding_service_view(catalog, {})

    def test_missing_loader_and_discovery_fields_fail_fast(self):
        missing_loader = _temporary_standard_manifest()
        del missing_loader["deployments"][0]["loader"]["parameters"][
            "additional_role_template"
        ]
        with self.assertRaisesRegex(
            EmbeddingServiceCatalogError,
            r"loader/parameters.*missing required fields",
        ):
            build_embedding_service_view(
                ModelCatalog.from_manifests([missing_loader]),
                {LoaderType.SENTENCE_TRANSFORMERS: mock.Mock()},
            )

        missing_discovery = _temporary_standard_manifest()
        del missing_discovery["deployments"][0]["metadata"]["discovery"][
            "size"
        ]
        with self.assertRaisesRegex(
            EmbeddingServiceCatalogError,
            r"metadata/discovery.*missing required fields",
        ):
            build_embedding_service_view(
                ModelCatalog.from_manifests([missing_discovery]),
                {LoaderType.SENTENCE_TRANSFORMERS: mock.Mock()},
            )

    def test_conflicting_deployments_for_one_service_identity_fail_fast(self):
        manifest = _temporary_standard_manifest()
        second_deployment = copy.deepcopy(manifest["deployments"][0])
        second_deployment["id"] = "test-standard-embed-late.deployment"
        second_deployment["routes"] = [
            {"task": "embedding", "endpoint": "/api/embed_late"}
        ]
        second_deployment["metadata"]["discovery"]["size"] += 1
        second_profile = copy.deepcopy(manifest["profiles"][0])
        second_profile["id"] = "test-standard-embed-late.profile"
        second_profile["deployment_id"] = second_deployment["id"]
        second_profile["endpoint"] = "/api/embed_late"
        second_profile["metadata"]["kind"] = "late_chunking"
        manifest["deployments"].append(second_deployment)
        manifest["profiles"].append(second_profile)

        with self.assertRaisesRegex(
            EmbeddingServiceCatalogError,
            r"conflicting deployment data.*metadata\.discovery",
        ):
            build_embedding_service_view(
                ModelCatalog.from_manifests([manifest]),
                {LoaderType.SENTENCE_TRANSFORMERS: mock.Mock()},
            )


class FakeWorker:
    """Fake fuer SerialModelWorker. Fuer Endpoint-Tests."""

    def __init__(
        self,
        snapshot: dict | None = None,
        encode_result: EncodeResult | None = None,
        encode_exc: BaseException | None = None,
        load_exc: BaseException | None = None,
        late_embed_result: LateEmbedResult | None = None,
        late_embed_exc: BaseException | None = None,
        late_embed_result_factory=None,
        colbert_embed_result: ColbertEmbedResult | None = None,
        colbert_embed_exc: BaseException | None = None,
    ):
        self._snapshot = snapshot or {
            "current_model": None,
            "loading_model": None,
            "device": "unknown",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
            "worker_started": True,
        }
        self.started = False
        self.stopped = False
        self.stop_timeout: float | None = None
        self.load_calls: list[str] = []
        self.encode_calls: list[tuple] = []
        self.late_embed_calls: list[tuple] = []
        self.colbert_embed_calls: list[tuple] = []
        self._encode_result = encode_result or EncodeResult(
            embeddings=[[1.0, 0.0]],
            prompt_eval_count=1,
            load_duration_ns=0,
        )
        self._encode_exc = encode_exc
        self._load_exc = load_exc
        self._late_embed_result = late_embed_result
        self._late_embed_exc = late_embed_exc
        self._late_embed_result_factory = late_embed_result_factory
        self._colbert_embed_result = colbert_embed_result or ColbertEmbedResult(
            embeddings=[[[1.0, 0.0]]],
            prompt_eval_count=1,
            load_duration_ns=0,
        )
        self._colbert_embed_exc = colbert_embed_exc

    def start(self) -> None:
        self.started = True

    async def stop(self, timeout: float = 30.0) -> None:
        self.stopped = True
        self.stop_timeout = timeout

    def snapshot(self) -> dict:
        return dict(self._snapshot)

    async def load(self, name: str) -> str:
        self.load_calls.append(name)
        if self._load_exc is not None:
            raise self._load_exc
        return name

    async def encode(self, name: str, texts: list, input_type) -> EncodeResult:
        self.encode_calls.append((name, list(texts), input_type))
        if self._encode_exc is not None:
            raise self._encode_exc
        return self._encode_result

    async def late_embed(
        self, name: str, document: str, chunks: list, input_type
    ) -> LateEmbedResult:
        self.late_embed_calls.append((name, document, list(chunks), input_type))
        if self._late_embed_exc is not None:
            raise self._late_embed_exc
        if self._late_embed_result_factory is not None:
            return self._late_embed_result_factory(chunks)
        if self._late_embed_result is not None:
            return self._late_embed_result
        return LateEmbedResult(
            embeddings=[[1.0, 0.0]] * len(chunks),
            prompt_eval_count=1,
            load_duration_ns=0,
            fallback_count=0,
        )

    async def colbert_embed(
        self, name: str, texts: list, input_type, language
    ) -> ColbertEmbedResult:
        self.colbert_embed_calls.append((name, list(texts), input_type, language))
        if self._colbert_embed_exc is not None:
            raise self._colbert_embed_exc
        return self._colbert_embed_result


def _swap_worker(new_worker):
    """Context-Helper: tauscht main.model_worker zeitweise aus."""
    old = main.model_worker
    main.model_worker = new_worker
    return old


# --- Validation -------------------------------------------------------------


class EmbeddingValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_role_is_rejected_for_every_role_sensitive_dense_profile(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            for model in (
                "nomic-embed-text",
                "mankei-326m-embedder",
                "mxbai-embed-large",
                "snowflake-arctic-embed",
            ):
                with self.subTest(model=model):
                    response = await main.embed(
                        main.EmbedRequest(model=model, input="contract witness")
                    )
                    self.assertEqual(response.status_code, 400)
                    payload = json.loads(bytes(response.body))
                    self.assertEqual(
                        payload["error"]["code"], "missing_required_input_type"
                    )
                    self.assertEqual(payload["error"]["field_path"], "/input_type")
                    self.assertTrue(payload["error"]["profile_id"])
                    self.assertEqual(
                        payload["error"]["supported"],
                        ["search_document", "search_query"],
                    )
        finally:
            main.model_worker = old
        self.assertEqual(fake.encode_calls, [])
        self.assertEqual(fake.load_calls, [])

    async def test_scalar_empty_string_is_rejected(self):
        req = main.EmbedRequest(
            model="mxbai-embed-large", input="", input_type="search_document"
        )
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)

    async def test_whitespace_list_item_is_rejected(self):
        req = main.EmbedRequest(
            model="mxbai-embed-large",
            input=["valid", " \t\n"],
            input_type="search_document",
        )
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)

    async def test_unknown_model_returns_400(self):
        req = main.EmbedRequest(model="frob-embed", input="hello")
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)

    async def test_empty_list_with_invalid_input_type_returns_400(self):
        # #905: input=[] mit ungueltigem input_type muss konsistent 400 liefern
        # (frueher 200, weil Empty-Short-Circuit vor input_type-Validierung lief).
        req = main.EmbedRequest(
            model="nomic-embed-text", input=[], input_type="ungueltig"
        )
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["error"]["code"], "unsupported_input_type")
        self.assertEqual(body["error"]["field_path"], "/input_type")

    async def test_empty_list_with_valid_input_type_returns_200(self):
        # #905: Gueltiger input_type + leere Liste -> 200 mit embeddings=[]
        # (Empty-Pfad-Verhalten bleibt erhalten).
        req = main.EmbedRequest(
            model="nomic-embed-text", input=[], input_type="search_query"
        )
        resp = await main.embed(req)
        self.assertEqual(resp["embeddings"], [])
        self.assertEqual(resp["prompt_eval_count"], 0)

    async def test_role_independent_empty_list_without_input_type_returns_200(self):
        req = main.EmbedRequest(model="bge-m3", input=[])
        resp = await main.embed(req)
        self.assertEqual(resp["embeddings"], [])
        self.assertEqual(resp["prompt_eval_count"], 0)

    async def test_role_sensitive_empty_list_without_input_type_returns_400(self):
        req = main.EmbedRequest(model="nomic-embed-text", input=[])
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["error"]["code"], "missing_required_input_type")


# --- Endpoint /api/embed ----------------------------------------------------


class EmbedEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_uses_worker_and_returns_response_shape(self):
        fake = FakeWorker(
            encode_result=EncodeResult(
                embeddings=[[0.0, 1.0]],
                prompt_eval_count=3,
                load_duration_ns=12345,
            )
        )
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="hello",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["model"], "mxbai-embed-large")
        self.assertEqual(resp["embeddings"], [[0.0, 1.0]])
        self.assertEqual(resp["prompt_eval_count"], 3)
        self.assertEqual(resp["load_duration"], 12345)
        self.assertGreaterEqual(resp["total_duration"], 0)
        self.assertEqual(
            fake.encode_calls,
            [("mxbai-embed-large", ["hello"], "search_document")],
        )

    async def test_explicit_canonical_and_additional_roles_reach_worker_unchanged(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            cases = (
                ("nomic-embed-text", "search_document"),
                ("nomic-embed-text", "search_query"),
                ("nomic-embed-text", "classification"),
                ("nomic-embed-text", "clustering"),
                ("mankei-326m-embedder", "search_document"),
                ("mankei-326m-embedder", "search_query"),
            )
            for model, role in cases:
                response = await main.embed(
                    main.EmbedRequest(
                        model=model,
                        input="contract witness",
                        input_type=role,
                    )
                )
                self.assertIsInstance(response, dict)
        finally:
            main.model_worker = old

        self.assertEqual(
            [(model, role) for model, _texts, role in fake.encode_calls],
            list(cases),
        )

    async def test_role_independent_profile_without_role_reaches_worker(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            response = await main.embed(
                main.EmbedRequest(model="bge-m3", input="contract witness")
            )
        finally:
            main.model_worker = old

        self.assertIsInstance(response, dict)
        self.assertEqual(fake.encode_calls, [("bge-m3", ["contract witness"], None)])

    async def test_non_finite_embeddings_return_503(self):
        fake = FakeWorker(encode_exc=NonFiniteEmbeddingError([0, 1]))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input=["bad nan", "bad inf"],
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["non_finite_indices"], [0, 1])

    async def test_zero_embeddings_pass_through(self):
        fake = FakeWorker(
            encode_result=EncodeResult(
                embeddings=[[0.0, 0.0]],
                prompt_eval_count=1,
                load_duration_ns=0,
            )
        )
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="zero",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["embeddings"], [[0.0, 0.0]])

    async def test_prompt_eval_count_passed_through(self):
        # prompt_eval_count wird im Worker erzeugt — das Endpoint mappt 1:1.
        fake = FakeWorker(
            encode_result=EncodeResult(
                embeddings=[[1.0, 0.0]],
                prompt_eval_count=42,
                load_duration_ns=0,
            )
        )
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="abcdef",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["prompt_eval_count"], 42)

    async def test_worker_queue_full_returns_503(self):
        fake = FakeWorker(encode_exc=WorkerQueueFullError("voll"))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="hello",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Queue", body["error"])

    async def test_worker_stopped_returns_503(self):
        fake = FakeWorker(encode_exc=WorkerStoppedError("stoppt"))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="hello",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)

    async def test_cuda_runtime_error_returns_503(self):
        # #858: CUDA-RuntimeError (illegal memory access etc.) -> 503, nicht 500
        original = RuntimeError("CUDA error: an illegal memory access was encountered")
        fake = FakeWorker(encode_exc=CudaRuntimeError(original))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large",
                input="hello",
                input_type="search_document",
            )
            resp = await main.embed(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("CUDA-Fehler", body["error"])
        self.assertIn("illegal memory access", body["detail"])


# --- Endpoint /api/embed_colbert -------------------------------------------


class ColbertEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_colbert_missing_role_is_rejected_before_worker(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            response = await main.embed_colbert(
                main.ColBERTRequest(model="colbert-xm", input="hello")
            )
        finally:
            main.model_worker = old

        self.assertEqual(response.status_code, 400)
        payload = json.loads(bytes(response.body))
        self.assertEqual(payload["error"]["code"], "missing_required_input_type")
        self.assertEqual(
            payload["error"]["profile_id"],
            "kiron-colbert-xm-multivector-v1",
        )
        self.assertEqual(fake.colbert_embed_calls, [])

    async def test_colbert_declared_aliases_are_canonicalized(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            for alias in ("document", "query"):
                response = await main.embed_colbert(
                    main.ColBERTRequest(
                        model="colbert-xm", input="hello", input_type=alias
                    )
                )
                self.assertIsInstance(response, dict)
        finally:
            main.model_worker = old

        self.assertEqual(
            [call[2] for call in fake.colbert_embed_calls],
            ["search_document", "search_query"],
        )

    async def test_colbert_uses_worker_and_returns_response_shape(self):
        fake = FakeWorker(
            colbert_embed_result=ColbertEmbedResult(
                embeddings=[[[0.1, 0.2], [0.3, 0.4]]],
                prompt_eval_count=2,
                load_duration_ns=123,
            )
        )
        old = _swap_worker(fake)
        try:
            req = main.ColBERTRequest(
                model="colbert-xm",
                input="hello",
                input_type="search_query",
                language="de",
            )
            resp = await main.embed_colbert(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["model"], "colbert-xm")
        self.assertEqual(resp["embeddings"], [[[0.1, 0.2], [0.3, 0.4]]])
        self.assertEqual(resp["prompt_eval_count"], 2)
        self.assertEqual(resp["load_duration"], 123)
        self.assertEqual(
            fake.colbert_embed_calls,
            [("colbert-xm", ["hello"], "search_query", "de")],
        )

    async def test_colbert_unknown_model_returns_400(self):
        req = main.ColBERTRequest(model="nomic-embed-text", input="hello")
        resp = await main.embed_colbert(req)
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["available_models"], ["colbert-xm"])

    async def test_colbert_empty_text_rejected(self):
        req = main.ColBERTRequest(
            model="colbert-xm", input=["ok", " "], input_type="search_document"
        )
        resp = await main.embed_colbert(req)
        self.assertEqual(resp.status_code, 400)

    async def test_colbert_invalid_input_type_rejected(self):
        req = main.ColBERTRequest(
            model="colbert-xm", input="hello", input_type="classification"
        )
        resp = await main.embed_colbert(req)
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["error"]["code"], "unsupported_input_type")
        self.assertEqual(body["error"]["field_path"], "/input_type")

    async def test_colbert_empty_list_returns_200_without_worker(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = main.ColBERTRequest(
                model="colbert-xm", input=[], input_type="search_query"
            )
            resp = await main.embed_colbert(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["model"], "colbert-xm")
        self.assertEqual(resp["embeddings"], [])
        self.assertEqual(resp["prompt_eval_count"], 0)
        self.assertEqual(fake.colbert_embed_calls, [])

    async def test_colbert_non_finite_returns_503(self):
        fake = FakeWorker(colbert_embed_exc=NonFiniteEmbeddingError([0]))
        old = _swap_worker(fake)
        try:
            req = main.ColBERTRequest(
                model="colbert-xm", input="bad", input_type="search_document"
            )
            resp = await main.embed_colbert(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["non_finite_indices"], [0])


# --- Endpoint /api/load -----------------------------------------------------


class LoadEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_uses_worker(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = main.LoadRequest(model="mxbai-embed-large")
            resp = await main.load_model_endpoint(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp, {"status": "ok", "model": "mxbai-embed-large"})
        self.assertEqual(fake.load_calls, ["mxbai-embed-large"])

    async def test_load_oserror_maps_to_500(self):
        fake = FakeWorker(load_exc=OSError("missing file"))
        old = _swap_worker(fake)
        try:
            resp = await main.load_model_endpoint(
                main.LoadRequest(model="mxbai-embed-large")
            )
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 500)

    async def test_load_queue_full_returns_503(self):
        fake = FakeWorker(load_exc=WorkerQueueFullError("voll"))
        old = _swap_worker(fake)
        try:
            resp = await main.load_model_endpoint(
                main.LoadRequest(model="mxbai-embed-large")
            )
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)

    async def test_load_unknown_model_returns_400(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            resp = await main.load_model_endpoint(main.LoadRequest(model="nope"))
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)

    async def test_load_cuda_runtime_error_returns_503(self):
        # #858: CUDA-RuntimeError im Load-Pfad -> 503, nicht 500
        original = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        fake = FakeWorker(load_exc=CudaRuntimeError(original))
        old = _swap_worker(fake)
        try:
            resp = await main.load_model_endpoint(
                main.LoadRequest(model="mxbai-embed-large")
            )
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("CUDA-Fehler", body["error"])


# --- Endpoint /health -------------------------------------------------------


class HealthEndpointTests(unittest.TestCase):
    def _health_with_snapshot(self, snapshot: dict):
        fake = FakeWorker(snapshot=snapshot)
        old = _swap_worker(fake)
        try:
            with mock.patch.object(
                main,
                "_local_model_inventory",
                return_value=main.LocalModelInventory(),
            ):
                return main.health()
        finally:
            main.model_worker = old

    def test_health_no_model(self):
        resp = self._health_with_snapshot({
            "current_model": None,
            "loading_model": None,
            "device": "unknown",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["status"], "no_model")
        self.assertEqual(
            body["catalog_digest"],
            "sha256:c91229d7ea472b49d87f6344dbfb640fc760f43e8cace398421d5b364452e6f6",
        )
        self.assertTrue(body["worker_thread_alive"])
        self.assertEqual(body["queue_depth"], 0)

    def test_health_loading(self):
        resp = self._health_with_snapshot({
            "current_model": None,
            "loading_model": "mxbai-embed-large",
            "device": "cuda",
            "queue_depth": 0,
            "current_job": "load",
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["status"], "loading")
        self.assertEqual(body["loading_model"], "mxbai-embed-large")

    def test_health_ok_during_encode(self):
        resp = self._health_with_snapshot({
            "current_model": "mxbai-embed-large",
            "loaded_models": ["nomic-embed-text", "mxbai-embed-large"],
            "model_slots": 2,
            "loading_model": None,
            "device": "cuda",
            "queue_depth": 0,
            "current_job": "encode",
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["worker_busy"])
        self.assertEqual(body["current_job"], "encode")
        self.assertEqual(
            body["loaded_models"],
            ["nomic-embed-text", "mxbai-embed-large"],
        )
        self.assertEqual(body["model_slots"], 2)

    def test_health_stopping_when_thread_dead(self):
        resp = self._health_with_snapshot({
            "current_model": "mxbai-embed-large",
            "loading_model": None,
            "device": "cuda",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": False,
        })
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["status"], "stopping")
        self.assertFalse(body["worker_thread_alive"])

    def test_health_reports_stored_device_without_probe(self):
        resp = self._health_with_snapshot({
            "current_model": "mxbai-embed-large",
            "loading_model": None,
            "device": "cuda",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        self.assertEqual(resp.status_code, 200)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["device"], "cuda")
        self.assertEqual(body["precision"], "fp16")


# --- Endpoint /api/tags -----------------------------------------------------


class TagsEndpointTests(unittest.TestCase):
    def test_all_models_are_revision_pinned(self):
        for config in main.EMBEDDING_SERVICE_VIEW.models:
            with self.subTest(model=config.model_name):
                revision = config.artifact.revision
                self.assertIsInstance(revision, str)
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_tags_uses_snapshot_no_worker_job(self):
        fake = FakeWorker(snapshot={
            "current_model": None,
            "loading_model": None,
            "device": "cuda",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        old = _swap_worker(fake)
        try:
            resp = main.tags()
        finally:
            main.model_worker = old

        # Keine Job-Submissions
        self.assertEqual(fake.load_calls, [])
        self.assertEqual(fake.encode_calls, [])
        self.assertIn("models", resp)
        # Service-Default F16; Mankei bleibt modellbedingt BF16.
        for m in resp["models"]:
            expected = "BF16" if m["name"] == "mankei-326m-embedder" else "F16"
            self.assertEqual(m["details"]["quantization_level"], expected)

    def test_tags_cpu_device_uses_F32(self):
        fake = FakeWorker(snapshot={
            "current_model": None,
            "loading_model": None,
            "device": "cpu",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        old = _swap_worker(fake)
        try:
            resp = main.tags()
        finally:
            main.model_worker = old

        for m in resp["models"]:
            expected = "BF16" if m["name"] == "mankei-326m-embedder" else "F32"
            self.assertEqual(m["details"]["quantization_level"], expected)

    def test_tags_digest_uses_pinned_revision(self):
        fake = FakeWorker(snapshot={
            "current_model": None,
            "loading_model": None,
            "device": "cpu",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        old = _swap_worker(fake)
        try:
            resp = main.tags()
        finally:
            main.model_worker = old

        by_name = {item["name"]: item for item in resp["models"]}
        for config in main.EMBEDDING_SERVICE_VIEW.models:
            expected = hashlib.sha256(
                f"{config.artifact.repository}@{config.artifact.revision}".encode()
            ).hexdigest()
            self.assertEqual(by_name[config.model_name]["digest"], expected)

    def test_tags_modified_at_is_deterministic_per_revision(self):
        # /api/tags darf modified_at nicht aus der Service-Startzeit ableiten —
        # sonst invalidiert jeder Restart die Caches in Ollama-Clients.
        fake = FakeWorker(snapshot={
            "current_model": None,
            "loading_model": None,
            "device": "cpu",
            "queue_depth": 0,
            "current_job": None,
            "worker_accepting": True,
            "worker_thread_alive": True,
        })
        old = _swap_worker(fake)
        try:
            first = main.tags()
            second = main.tags()
        finally:
            main.model_worker = old

        first_by_name = {m["name"]: m["modified_at"] for m in first["models"]}
        second_by_name = {m["name"]: m["modified_at"] for m in second["models"]}
        # Stabil ueber Aufrufe (Proxy fuer Service-Restarts).
        self.assertEqual(first_by_name, second_by_name)
        # ISO 8601-parsebar — kiron-proxy/openai_api.py nutzt fromisoformat.
        from datetime import datetime as _dt
        for value in first_by_name.values():
            _dt.fromisoformat(value)
        # Pro Revision ein eigener Wert; gleiche Revision -> gleicher Timestamp.
        for config in main.EMBEDDING_SERVICE_VIEW.models:
            self.assertEqual(
                first_by_name[config.model_name],
                main._model_modified_at(config),
            )


# --- Lifespan ---------------------------------------------------------------


class LifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_starts_worker_no_preload_stops_after_yield(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            cm = main.lifespan(None)
            await cm.__aenter__()
            self.assertTrue(fake.started, "worker.start() wurde nicht gerufen")
            self.assertEqual(fake.load_calls, [], "kein Preload erlaubt")
            self.assertFalse(fake.stopped)

            await asyncio.wait_for(cm.__aexit__(None, None, None), timeout=2.0)
            self.assertTrue(fake.stopped, "worker.stop() wurde nicht gerufen")
        finally:
            main.model_worker = old

    async def test_lifespan_swallows_stop_exceptions(self):
        class BadStopWorker(FakeWorker):
            async def stop(self, timeout=30.0):
                raise RuntimeError("stop failed")

        bad = BadStopWorker()
        old = _swap_worker(bad)
        try:
            cm = main.lifespan(None)
            await cm.__aenter__()
            # __aexit__ darf NICHT raisen
            await asyncio.wait_for(cm.__aexit__(None, None, None), timeout=2.0)
        finally:
            main.model_worker = old


# --- CUDA-RuntimeError-Detection --------------------------------------------


class CudaRuntimeErrorDetectionTests(unittest.TestCase):
    """#858: _is_cuda_runtime_error erkennt CUDA-Fehler-Signaturen,
    schluckt aber keine OOM oder Nicht-CUDA-RuntimeErrors."""

    def test_detects_illegal_memory_access(self):
        e = RuntimeError("CUDA error: an illegal memory access was encountered")
        self.assertTrue(main._is_cuda_runtime_error(e))

    def test_detects_cublas_alloc_failed(self):
        e = RuntimeError("cublas runtime error: CUBLAS_STATUS_ALLOC_FAILED")
        self.assertTrue(main._is_cuda_runtime_error(e))

    def test_detects_cudnn_status(self):
        e = RuntimeError("CUDNN_STATUS_EXECUTION_FAILED")
        self.assertTrue(main._is_cuda_runtime_error(e))

    def test_detects_device_side_assert(self):
        e = RuntimeError("CUDA kernel errors might be asynchronously reported. device-side assert triggered")
        self.assertTrue(main._is_cuda_runtime_error(e))

    def test_does_not_match_oom_message(self):
        # OOM message wird separat von torch.cuda.OutOfMemoryError gefangen,
        # _is_cuda_runtime_error darf nicht zusaetzlich matchen.
        e = RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        self.assertFalse(main._is_cuda_runtime_error(e))

    def test_does_not_match_unrelated_runtime_error(self):
        e = RuntimeError("model.encode failed: tensor shape mismatch")
        self.assertFalse(main._is_cuda_runtime_error(e))

    def test_does_not_match_value_error(self):
        e = ValueError("CUDA error: even with a CUDA-like message, ValueError is not RuntimeError")
        self.assertFalse(main._is_cuda_runtime_error(e))


# --- Endpoint /api/embed_late -----------------------------------------------


class EmbedLateEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Tests fuer /api/embed_late: D2-Filter, Validierung, Exception-Mapping."""

    def _make_request(self, **kwargs):
        defaults = {
            "model": "nomic-embed-text",
            "document": "hello world",
            "chunks": [{"text": "hello", "char_start": 0, "char_end": 5}],
            "input_type": "search_document",
        }
        defaults.update(kwargs)
        return main.LateBatchRequest(**defaults)

    async def test_late_unknown_model(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(model="frob-embed")
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("nicht unterstuetzt", body["error"])
        self.assertIn("nomic-embed-text", body["available_models"])
        self.assertEqual(fake.late_embed_calls, [])

    async def test_late_non_nomic_model_rejected(self):
        # D2: mxbai/snowflake/bge-m3 -> 400 mit available_models=["nomic-embed-text"]
        for non_nomic in ("mxbai-embed-large", "snowflake-arctic-embed", "bge-m3"):
            with self.subTest(model=non_nomic):
                fake = FakeWorker()
                old = _swap_worker(fake)
                try:
                    req = self._make_request(model=non_nomic)
                    resp = await main.embed_late(req)
                finally:
                    main.model_worker = old
                self.assertEqual(resp.status_code, 400)
                body = json.loads(bytes(resp.body).decode())
                self.assertEqual(body["available_models"], ["nomic-embed-text"])
                self.assertIn("nicht aktiv", body["error"])
                self.assertEqual(fake.late_embed_calls, [])

    async def test_late_empty_chunks(self):
        # D9: chunks=[] -> 200 ohne Worker-Touch.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(chunks=[])
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp, {"model": "nomic-embed-text", "embeddings": []})
        self.assertEqual(fake.late_embed_calls, [])

    async def test_late_document_too_long(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            big_doc = "x" * (main.MAX_DOC_LEN + 1)
            req = self._make_request(
                document=big_doc,
                chunks=[{"text": "x", "char_start": 0, "char_end": 1}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("zu lang", body["error"])

    async def test_late_too_many_chunks(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            doc = "x" * 1024
            chunks = [
                {"text": "x", "char_start": 0, "char_end": 1}
                for _ in range(main.MAX_LATE_CHUNKS + 1)
            ]
            req = self._make_request(document=doc, chunks=chunks)
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Zu viele Chunks", body["error"])

    async def test_late_document_surrogate_rejected(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            # Lone low-surrogate U+DC00 — _SURROGATE_RE matched.
            req = self._make_request(
                document="hello\udc00 world",
                chunks=[{"text": "hello", "char_start": 0, "char_end": 5}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Document", body["error"])
        self.assertIn("Surrogate", body["error"])

    async def test_late_chunk_text_surrogate_rejected(self):
        # Defensiver Check fuer Chunk-Surrogate. In der derzeitigen Reihenfolge
        # gating Document-Surrogate (Schritt 4) immer zuerst — der per-Chunk-
        # Check ist Defense-in-Depth fuer den Fall, dass die globale Pruefung
        # entfaellt. Test stellt sicher, dass eine Surrogate-Eingabe in jedem
        # Fall mit 400 + "Surrogate"-Message abgelehnt wird.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            doc = "hello\udc00ello"
            req = self._make_request(
                document=doc,
                chunks=[{"text": doc, "char_start": 0, "char_end": len(doc)}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Surrogate", body["error"])

    async def test_late_bounds_violation_negative(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(
                chunks=[{"text": "h", "char_start": -1, "char_end": 1}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Chunk 0", body["error"])
        self.assertIn("Bounds", body["error"])

    async def test_late_bounds_violation_overflow(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(
                document="hi",
                chunks=[{"text": "hi", "char_start": 0, "char_end": 99}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Chunk 0", body["error"])
        self.assertIn("Bounds", body["error"])

    async def test_late_bounds_violation_swapped(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(
                document="hello world",
                chunks=[{"text": "hello", "char_start": 5, "char_end": 0}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Bounds", body["error"])

    async def test_late_text_mismatch(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            # Erste Chunk OK, zweite Chunk falsch — Index 1 muss berichtet werden.
            req = self._make_request(
                document="hello world",
                chunks=[
                    {"text": "hello", "char_start": 0, "char_end": 5},
                    {"text": "bogus", "char_start": 6, "char_end": 11},
                ],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Chunk 1", body["error"])
        self.assertIn("mismatch", body["error"])

    async def test_late_text_mismatch_unicode(self):
        # Codepoint-Zaehlung: "Mueszig" mit Umlaut/sz-Ligatur hat 5 Codepoints.
        # char_end=5 muss "Müßig" exakt liefern, kein false positive durch
        # Byte-vs-Codepoint-Drift.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            doc = "Müßig 中文"
            req = self._make_request(
                document=doc,
                chunks=[{"text": "Müßig", "char_start": 0, "char_end": 5}],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        # Sollte 200 liefern (Codepoint-Zaehlung greift korrekt).
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp["model"], "nomic-embed-text")
        self.assertEqual(len(resp["embeddings"]), 1)

    async def test_late_invalid_input_type_for_nomic(self):
        # D4: nomic + ungueltiger input_type -> 400.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(input_type="ungueltig")
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["error"]["code"], "unsupported_input_type")
        self.assertEqual(body["error"]["field_path"], "/input_type")
        self.assertIn("search_document", body["valid_input_types"])

    async def test_late_missing_input_type_is_rejected_before_worker(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(input_type=None)
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 400)
        payload = json.loads(bytes(resp.body))
        self.assertEqual(payload["error"]["code"], "missing_required_input_type")
        self.assertEqual(payload["error"]["profile_id"], "kiron-nomic-late-v1")
        self.assertEqual(fake.late_embed_calls, [])

    async def test_late_input_type_search_document_applies(self):
        # F0.2 Live-Pfad: input_type="search_document" -> Worker erhaelt's exakt.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(input_type="search_document")
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp["model"], "nomic-embed-text")
        _, _, _, recorded_input_type = fake.late_embed_calls[0]
        self.assertEqual(recorded_input_type, "search_document")

    async def test_late_additional_declared_role_applies(self):
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request(input_type="classification")
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp["model"], "nomic-embed-text")
        self.assertEqual(fake.late_embed_calls[0][3], "classification")

    async def test_late_response_schema_invariant(self):
        # D13: Mismatch zwischen result.embeddings-Length und chunks-Length -> 500.
        # Server-Garantie als Last-Line-of-Defense.
        bad_result = LateEmbedResult(
            embeddings=[[1.0, 0.0]],  # 1 Embedding ...
            prompt_eval_count=1,
            load_duration_ns=0,
            fallback_count=0,
        )
        fake = FakeWorker(late_embed_result=bad_result)
        old = _swap_worker(fake)
        try:
            req = self._make_request(
                document="hello world",
                chunks=[
                    {"text": "hello", "char_start": 0, "char_end": 5},
                    {"text": "world", "char_start": 6, "char_end": 11},
                ],  # ... aber 2 Chunks angefragt
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 500)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("Length-Mismatch", body["error"])

    async def test_late_response_schema_invariant_match_passes(self):
        good_result = LateEmbedResult(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            prompt_eval_count=2,
            load_duration_ns=0,
            fallback_count=0,
        )
        fake = FakeWorker(late_embed_result=good_result)
        old = _swap_worker(fake)
        try:
            req = self._make_request(
                document="hello world",
                chunks=[
                    {"text": "hello", "char_start": 0, "char_end": 5},
                    {"text": "world", "char_start": 6, "char_end": 11},
                ],
            )
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp["model"], "nomic-embed-text")
        self.assertEqual(len(resp["embeddings"]), 2)

    async def test_late_nonfinite_returns_503(self):
        # D16: NonFiniteEmbeddingError -> 503 (konsistent mit /api/embed).
        fake = FakeWorker(late_embed_exc=NonFiniteEmbeddingError([0]))
        old = _swap_worker(fake)
        try:
            req = self._make_request()
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertEqual(body["non_finite_indices"], [0])

    async def test_late_worker_stopped_returns_503(self):
        fake = FakeWorker(late_embed_exc=WorkerStoppedError("stoppt"))
        old = _swap_worker(fake)
        try:
            req = self._make_request()
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)

    async def test_late_queue_full_returns_503(self):
        fake = FakeWorker(late_embed_exc=WorkerQueueFullError("voll"))
        old = _swap_worker(fake)
        try:
            req = self._make_request()
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)

    async def test_late_cuda_runtime_returns_503(self):
        original = RuntimeError("CUDA error: an illegal memory access was encountered")
        fake = FakeWorker(late_embed_exc=CudaRuntimeError(original))
        old = _swap_worker(fake)
        try:
            req = self._make_request()
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("CUDA-Fehler", body["error"])

    async def test_late_oserror_returns_500(self):
        fake = FakeWorker(late_embed_exc=OSError("missing file"))
        old = _swap_worker(fake)
        try:
            req = self._make_request()
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 500)


# --- Statische Absicherung --------------------------------------------------


class MainSourceStaticChecks(unittest.TestCase):
    def test_main_has_no_request_path_to_thread_load_or_encode(self):
        """In den Funktions-Bodies der HTTP-Endpoints:
        - kein asyncio.to_thread(
        - kein direkter Aufruf von _load_model_sync / _encode_sync /
          _drop_model_sync / _late_embed_sync / _colbert_embed_sync
          (Modelloperationen MUESSEN ueber model_worker laufen).
        """
        embed_src = inspect.getsource(main.embed)
        colbert_src = inspect.getsource(main.embed_colbert)
        load_src = inspect.getsource(main.load_model_endpoint)
        late_src = inspect.getsource(main.embed_late)

        for name, src in (
            ("embed", embed_src),
            ("embed_colbert", colbert_src),
            ("load_model_endpoint", load_src),
            ("embed_late", late_src),
        ):
            self.assertNotIn(
                "asyncio.to_thread(",
                src,
                f"{name} darf kein asyncio.to_thread im Requestpfad nutzen",
            )
            for forbidden in (
                "_load_model_sync",
                "_encode_sync",
                "_drop_model_sync",
                "_late_embed_sync",
                "_colbert_embed_sync",
            ):
                self.assertNotIn(
                    forbidden,
                    src,
                    f"{name} darf {forbidden} nicht direkt aufrufen",
                )


if __name__ == "__main__":
    unittest.main()
