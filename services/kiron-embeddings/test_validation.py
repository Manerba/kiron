"""Validation- und Endpoint-Tests fuer den Embedding-Service."""

import asyncio
import hashlib
import inspect
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402
from model_worker import (  # noqa: E402
    CudaRuntimeError,
    EncodeResult,
    LateEmbedResult,
    NonFiniteEmbeddingError,
    WorkerQueueFullError,
    WorkerStoppedError,
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


def _swap_worker(new_worker):
    """Context-Helper: tauscht main.model_worker zeitweise aus."""
    old = main.model_worker
    main.model_worker = new_worker
    return old


# --- Validation -------------------------------------------------------------


class EmbeddingValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_scalar_empty_string_is_rejected(self):
        req = main.EmbedRequest(model="mxbai-embed-large", input="")
        resp = await main.embed(req)
        self.assertEqual(resp.status_code, 400)

    async def test_whitespace_list_item_is_rejected(self):
        req = main.EmbedRequest(model="mxbai-embed-large", input=["valid", " \t\n"])
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
        self.assertIn("input_type", body["error"])

    async def test_empty_list_with_valid_input_type_returns_200(self):
        # #905: Gueltiger input_type + leere Liste -> 200 mit embeddings=[]
        # (Empty-Pfad-Verhalten bleibt erhalten).
        req = main.EmbedRequest(
            model="nomic-embed-text", input=[], input_type="search_query"
        )
        resp = await main.embed(req)
        self.assertEqual(resp["embeddings"], [])
        self.assertEqual(resp["prompt_eval_count"], 0)

    async def test_empty_list_without_input_type_returns_200(self):
        # Regression: input=[] ohne input_type bleibt 200 mit embeddings=[].
        req = main.EmbedRequest(model="mxbai-embed-large", input=[])
        resp = await main.embed(req)
        self.assertEqual(resp["embeddings"], [])
        self.assertEqual(resp["prompt_eval_count"], 0)


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
            req = main.EmbedRequest(model="mxbai-embed-large", input="hello")
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["model"], "mxbai-embed-large")
        self.assertEqual(resp["embeddings"], [[0.0, 1.0]])
        self.assertEqual(resp["prompt_eval_count"], 3)
        self.assertEqual(resp["load_duration"], 12345)
        self.assertGreaterEqual(resp["total_duration"], 0)
        self.assertEqual(fake.encode_calls, [("mxbai-embed-large", ["hello"], None)])

    async def test_non_finite_embeddings_return_503(self):
        fake = FakeWorker(encode_exc=NonFiniteEmbeddingError([0, 1]))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(
                model="mxbai-embed-large", input=["bad nan", "bad inf"]
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
            req = main.EmbedRequest(model="mxbai-embed-large", input="zero")
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
            req = main.EmbedRequest(model="mxbai-embed-large", input="abcdef")
            resp = await main.embed(req)
        finally:
            main.model_worker = old

        self.assertEqual(resp["prompt_eval_count"], 42)

    async def test_worker_queue_full_returns_503(self):
        fake = FakeWorker(encode_exc=WorkerQueueFullError("voll"))
        old = _swap_worker(fake)
        try:
            req = main.EmbedRequest(model="mxbai-embed-large", input="hello")
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
            req = main.EmbedRequest(model="mxbai-embed-large", input="hello")
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
            req = main.EmbedRequest(model="mxbai-embed-large", input="hello")
            resp = await main.embed(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp.status_code, 503)
        body = json.loads(bytes(resp.body).decode())
        self.assertIn("CUDA-Fehler", body["error"])
        self.assertIn("illegal memory access", body["detail"])


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
        for name, cfg in main.OLLAMA_TO_HF.items():
            with self.subTest(model=name):
                revision = cfg.get("revision")
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
        # quantization F16 wegen device=cuda
        for m in resp["models"]:
            self.assertEqual(m["details"]["quantization_level"], "F16")

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
            self.assertEqual(m["details"]["quantization_level"], "F32")

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
        for name, cfg in main.OLLAMA_TO_HF.items():
            expected = hashlib.sha256(
                f"{cfg['hf']}@{cfg['revision']}".encode()
            ).hexdigest()
            self.assertEqual(by_name[name]["digest"], expected)

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
        for name, cfg in main.OLLAMA_TO_HF.items():
            self.assertEqual(
                first_by_name[name],
                main._model_modified_at(cfg),
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
        self.assertIn("input_type", body["error"])
        self.assertIn("search_document", body["valid_input_types"])

    async def test_late_default_input_type_neutral_for_nomic(self):
        # F0.2 Default-Pfad: input_type=None -> Worker erhaelt None.
        fake = FakeWorker()
        old = _swap_worker(fake)
        try:
            req = self._make_request()  # input_type unset
            resp = await main.embed_late(req)
        finally:
            main.model_worker = old
        self.assertEqual(resp["model"], "nomic-embed-text")
        self.assertEqual(len(fake.late_embed_calls), 1)
        _, _, _, recorded_input_type = fake.late_embed_calls[0]
        self.assertIsNone(recorded_input_type)

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
        """In den Funktions-Bodies von embed, load_model_endpoint und embed_late:
        - kein asyncio.to_thread(
        - kein direkter Aufruf von _load_model_sync / _encode_sync /
          _drop_model_sync / _late_embed_sync (Modelloperationen MUESSEN ueber
          model_worker laufen).
        """
        embed_src = inspect.getsource(main.embed)
        load_src = inspect.getsource(main.load_model_endpoint)
        late_src = inspect.getsource(main.embed_late)

        for name, src in (
            ("embed", embed_src),
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
            ):
                self.assertNotIn(
                    forbidden,
                    src,
                    f"{name} darf {forbidden} nicht direkt aufrufen",
                )


if __name__ == "__main__":
    unittest.main()
