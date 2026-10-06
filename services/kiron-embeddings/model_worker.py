"""Serial Model Worker fuer kiron-embeddings.

Genau ein dedizierter Worker-Thread fuehrt alle Modelloperationen seriell
aus. Verhindert parallele Modellmutationen und Cancellation-/Thread-Races
(#757). Der Worker wird vom FastAPI-Lifespan in main.py erstellt; HTTP-
Endpoints senden Jobs ueber load()/encode() und warten auf das Future.
"""

import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MODEL_WORKER_QUEUE_MAX = 16
MODEL_WORKER_SHUTDOWN_TIMEOUT_S = 30.0


@dataclass(slots=True)
class EncodeResult:
    embeddings: list[list[float]]
    prompt_eval_count: int
    load_duration_ns: int
    execution_epoch: int | None = None
    execution_artifact: str | None = None


@dataclass(slots=True)
class LateEmbedResult:
    embeddings: list[list[float]]
    prompt_eval_count: int
    load_duration_ns: int
    fallback_count: int


@dataclass(slots=True)
class ColbertEmbedResult:
    embeddings: list[list[list[float]]]
    prompt_eval_count: int
    load_duration_ns: int


@dataclass(slots=True)
class ModelJob:
    kind: Literal["load", "unload", "encode", "drop", "late_embed", "colbert_embed"]
    payload: dict[str, object]
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    cancelled: threading.Event
    job_id: int


class WorkerQueueFullError(RuntimeError):
    pass


class WorkerStoppedError(RuntimeError):
    pass


class StaleGenerationError(ValueError):
    """The expected resident changed; no model access/execution was attempted."""


class NonFiniteEmbeddingError(RuntimeError):
    def __init__(self, indices: list[int]):
        super().__init__("non-finite embedding values")
        self.indices = indices


class CudaRuntimeError(RuntimeError):
    """CUDA-Fehler ausserhalb von OutOfMemoryError (z.B. illegal memory access,
    CUBLAS_STATUS_*, CUDNN_STATUS_*). Modell wurde entladen und Cache geleert,
    der CUDA-Kontext kann aber weiterhin beschaedigt sein — Retry mit Backoff."""

    def __init__(self, original_exc: BaseException):
        super().__init__(str(original_exc))
        self.original_type = type(original_exc).__name__


def _complete_future_with_result(
    future: asyncio.Future, job: ModelJob, value: Any
) -> None:
    # Ein gecancelter Awaiter darf kein nachtraegliches Result bekommen,
    # und die Future darf bei doppeltem Set keinen InvalidStateError werfen.
    if future.done() or future.cancelled() or job.cancelled.is_set():
        return
    try:
        future.set_result(value)
    except asyncio.InvalidStateError:
        pass


def _complete_future_with_exception(
    future: asyncio.Future, job: ModelJob, exc: BaseException
) -> None:
    if future.done() or future.cancelled() or job.cancelled.is_set():
        return
    try:
        future.set_exception(exc)
    except asyncio.InvalidStateError:
        pass


def derive_health_status(snapshot: dict[str, object]) -> str:
    """Status-Praezedenz fuer /health (erste Treffer-Regel gewinnt).

    stopping > loading > ok > no_model. Ausgelagert, damit Tests die Logik
    isoliert pruefen koennen.

    "loading" erfordert atomar gesetztes loading_model — current_job=="load"
    allein reicht nicht: bei Same-Model-Cache-Hit kehrt _ensure_model_sync
    ohne loading_model zurueck, und im Race-Fenster zwischen _set_current_job
    und _load_model_sync existiert noch kein konkretes Lade-Ziel. Sonst
    wuerde /health status=loading ohne loading_model-Feld liefern und den
    Vertrag fuer Proxy/UI brechen.
    """
    if not snapshot.get("worker_thread_alive", False) or not snapshot.get(
        "worker_accepting", False
    ):
        return "stopping"
    if snapshot.get("loading_model") is not None:
        return "loading"
    if snapshot.get("current_model") is not None:
        return "ok"
    return "no_model"


class SerialModelWorker:
    """Single-Thread-Worker, der alle Modelloperationen seriell ausfuehrt."""

    THREAD_NAME = "kiron-embeddings-model-worker"
    _POLL_TIMEOUT_S = 0.1

    def __init__(self, manager: Any, queue_max: int = MODEL_WORKER_QUEUE_MAX):
        self._manager = manager
        self._queue: queue.Queue[ModelJob] = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._next_job_id = 0

        self._started = False
        self._accepting = False
        self._thread_alive = False
        self._current_job: str | None = None
        self._last_error_type: str | None = None
        self._last_error_message: str | None = None

    # --- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("worker already started")
            self._thread = threading.Thread(
                target=self._run, name=self.THREAD_NAME, daemon=True
            )
            self._thread.start()
            # thread_alive synchron setzen, damit ein direkt folgendes /health
            # nicht status=stopping meldet, nur weil das Thread-Target sein
            # try-Block noch nicht erreicht hat.
            self._started = True
            self._accepting = True
            self._thread_alive = True

    async def stop(self, timeout: float = MODEL_WORKER_SHUTDOWN_TIMEOUT_S) -> None:
        with self._lock:
            self._accepting = False
        # stop_event ohne Lock setzen, damit Submit-Aufrufer nach Lock-Release
        # sofort WorkerStoppedError sehen.
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        await asyncio.to_thread(thread.join, timeout)
        if thread.is_alive():
            logger.warning(
                "kiron-embeddings model worker hat nach %.1fs nicht beendet",
                timeout,
            )

    # --- Submit --------------------------------------------------------------

    def _new_job_id(self) -> int:
        with self._lock:
            self._next_job_id += 1
            return self._next_job_id

    def _submit(
        self, kind: str, payload: dict[str, object]
    ) -> tuple[asyncio.Future, ModelJob]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = ModelJob(
            kind=kind,  # type: ignore[arg-type]
            payload=payload,
            future=future,
            loop=loop,
            cancelled=threading.Event(),
            job_id=self._new_job_id(),
        )
        # Worker-Lock ueber GESAMTEN Vorgang halten (Stop/Submit-Atomicity):
        # Zwischen accepting-Check und put_nowait kann kein stop() einschieben.
        with self._lock:
            if not self._started:
                raise RuntimeError("worker not started")
            if not self._accepting:
                raise WorkerStoppedError("Embedding-Service stoppt")
            try:
                self._queue.put_nowait(job)
            except queue.Full as e:
                raise WorkerQueueFullError("Model-Worker Queue voll") from e
        return future, job

    async def _await_with_cancel(self, future: asyncio.Future, job: ModelJob):
        try:
            return await future
        except asyncio.CancelledError:
            job.cancelled.set()
            raise

    async def load(self, model_name: str, *, load_parent: tuple[str, str] | None = None) -> str:
        future, job = self._submit("load", {"model_name": model_name, "load_parent": load_parent})
        return await self._await_with_cancel(future, job)

    async def unload(self, model_name: str) -> bool:
        future, job = self._submit("unload", {"model_name": model_name})
        return await self._await_with_cancel(future, job)

    async def encode(
        self, model_name: str, texts: list[str], input_type: str | None, *, expected_artifact: str | None = None,
        expected_epoch: int | None = None
    ) -> EncodeResult:
        future, job = self._submit(
            "encode",
            {"model_name": model_name, "texts": texts, "input_type": input_type,
             **({"expected_artifact": expected_artifact, "expected_epoch": expected_epoch}
                if expected_artifact is not None else {})},
        )
        return await self._await_with_cancel(future, job)

    async def late_embed(
        self,
        model_name: str,
        document: str,
        chunks: list[dict],
        input_type: str | None,
    ) -> LateEmbedResult:
        future, job = self._submit(
            "late_embed",
            {
                "model_name": model_name,
                "document": document,
                "chunks": chunks,
                "input_type": input_type,
            },
        )
        return await self._await_with_cancel(future, job)

    async def colbert_embed(
        self,
        model_name: str,
        texts: list[str],
        input_type: str | None,
        language: str | None,
    ) -> ColbertEmbedResult:
        future, job = self._submit(
            "colbert_embed",
            {
                "model_name": model_name,
                "texts": texts,
                "input_type": input_type,
                "language": language,
            },
        )
        return await self._await_with_cancel(future, job)

    async def _drop_for_test(self) -> None:
        """Test-Helper: queued einen Drop-Job. In Production wird drop nur
        inline waehrend stop() aufgerufen."""
        future, job = self._submit("drop", {})
        await self._await_with_cancel(future, job)

    # --- Snapshot ------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        # Reihenfolge: Worker-Lock nehmen, Worker-Felder kopieren, Worker-Lock
        # FREIGEBEN, danach manager.snapshot(). Damit nie Worker-Lock waehrend
        # manager.state_lock gehalten -> kein Lock-Nesting.
        with self._lock:
            worker_data: dict[str, object] = {
                "queue_depth": self._queue.qsize(),
                "current_job": self._current_job,
                "worker_accepting": self._accepting,
                "worker_thread_alive": self._thread_alive,
                "worker_started": self._started,
                "last_error_type": self._last_error_type,
                "last_error_message": self._last_error_message,
            }
        manager_data = self._manager.snapshot()
        merged = dict(worker_data)
        merged.update(manager_data)
        return merged

    # --- Worker-Loop ---------------------------------------------------------

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._last_error_type = type(exc).__name__
            self._last_error_message = str(exc)

    def _set_current_job(self, kind: str | None) -> None:
        with self._lock:
            self._current_job = kind

    def _run(self) -> None:
        fatal = False
        try:
            self._manager.set_worker_thread()
            while True:
                # Iterations-Anfang: stop_event vor queue.get pruefen, damit
                # leere Queue + stop_event nicht in queue.Empty -> continue
                # haengt.
                if self._stop_event.is_set():
                    self._stop_sequence(initial_job=None)
                    return
                try:
                    job = self._queue.get(timeout=self._POLL_TIMEOUT_S)
                except queue.Empty:
                    idle = getattr(self._manager, "residency_idle", None)
                    if idle is not None:
                        try:
                            idle()
                        except Exception as exc:
                            self._record_error(exc)
                            logger.exception("Embedding resident heartbeat failed closed")
                    continue
                # Race-Fenster: stop_event wurde gesetzt, waehrend wir in
                # queue.get warteten. Drainen mit dem gerade gepullten Job.
                if self._stop_event.is_set():
                    self._stop_sequence(initial_job=job)
                    return
                self._dispatch_job(job)
        except BaseException as exc:
            fatal = True
            logger.exception(
                "kiron-embeddings model worker fatal exit: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._record_error(exc)
        finally:
            with self._lock:
                self._thread_alive = False
                self._current_job = None
            try:
                self._manager.clear_loading_for_crash()
            except Exception:
                logger.exception("clear_loading_for_crash fehlgeschlagen")

        if fatal:
            unknown = getattr(self._manager, "residency_unknown", None)
            if unknown is not None:
                unknown()
            # Fail-Fast: systemd startet den Prozess neu. Ohne os._exit
            # bliebe der Service HTTP-erreichbar mit totem Worker.
            os._exit(1)

    def _stop_sequence(self, initial_job: ModelJob | None) -> None:
        pending: list[ModelJob] = []
        if initial_job is not None:
            pending.append(initial_job)
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except queue.Empty:
                break

        for job in pending:
            try:
                job.loop.call_soon_threadsafe(
                    _complete_future_with_exception,
                    job.future,
                    job,
                    WorkerStoppedError("Embedding-Service stoppt"),
                )
            except Exception:
                logger.exception(
                    "call_soon_threadsafe waehrend Stop-Drain fehlgeschlagen"
                )

        self._set_current_job("drop")
        try:
            self._manager._drop_model_sync()
        except Exception:
            logger.exception("Fehler im Drop-Pfad waehrend Stop")
            unknown = getattr(self._manager, "residency_unknown", None)
            if unknown is not None:
                unknown()
        else:
            boundary = getattr(self._manager, "residency_boundary", None)
            if boundary is not None:
                boundary()
        finally:
            self._set_current_job(None)

    def _dispatch_job(self, job: ModelJob) -> None:
        # Pre-Dispatch Cancellation-Check: wartender Job wird ohne Modell-Touch
        # verworfen. Future ist im cancelled-Pfad bereits selbst cancelled.
        if job.cancelled.is_set():
            return

        completion: tuple[object, object] | None = None
        self._set_current_job(job.kind)
        try:
            try:
                if job.kind == "load":
                    name = job.payload["model_name"]
                    self._manager._ensure_model_sync(name, load_parent=job.payload["load_parent"])
                    completion = (_complete_future_with_result, name)
                elif job.kind == "unload":
                    removed = self._manager._unload_model_sync(job.payload["model_name"])
                    completion = (_complete_future_with_result, removed)
                elif job.kind == "encode":
                    name = job.payload["model_name"]
                    texts = job.payload["texts"]
                    input_type = job.payload["input_type"]
                    options = ({"expected_artifact": job.payload["expected_artifact"],
                                "expected_epoch": job.payload["expected_epoch"]}
                               if "expected_artifact" in job.payload else {})
                    result = self._manager._encode_sync(name, texts, input_type, **options)
                    completion = (_complete_future_with_result, result)
                elif job.kind == "late_embed":
                    result = self._manager._late_embed_sync(
                        job.payload["model_name"],
                        job.payload["document"],
                        job.payload["chunks"],
                        job.payload["input_type"],
                    )
                    completion = (_complete_future_with_result, result)
                elif job.kind == "colbert_embed":
                    result = self._manager._colbert_embed_sync(
                        job.payload["model_name"],
                        job.payload["texts"],
                        job.payload["input_type"],
                        job.payload["language"],
                    )
                    completion = (_complete_future_with_result, result)
                elif job.kind == "drop":
                    self._manager._drop_model_sync()
                    completion = (_complete_future_with_result, None)
                else:
                    raise RuntimeError(f"unknown job kind: {job.kind!r}")
            except Exception as exc:
                # Per-Job-Exception: Future bekommt Exception, Worker laeuft
                # weiter. KeyboardInterrupt/SystemExit (BaseException) bubbeln
                # durch zur outer except in _run -> Fail-Fast.
                self._record_error(exc)
                completion = (_complete_future_with_exception, exc)
        finally:
            boundary = getattr(self._manager, "residency_boundary", None)
            if boundary is not None:
                try:
                    boundary()
                except Exception as exc:
                    self._record_error(exc)
                    completion = (_complete_future_with_exception, exc)
            # Der atomare Runtime-Snapshot muss bereits idle sein, bevor die
            # Future-Aufloesung den Awaiter wieder laufen lassen kann. Sonst
            # kann unmittelbar nach einem erfolgreichen await noch der alte
            # current_job sichtbar sein.
            self._set_current_job(None)

        callback, value = completion
        try:
            job.loop.call_soon_threadsafe(callback, job.future, job, value)
        except Exception:
            logger.exception("call_soon_threadsafe fuer Job-Abschluss fehlgeschlagen")
