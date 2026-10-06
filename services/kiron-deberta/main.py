"""Kiron DeBERTa Cross-Encoder Service mit Reranking- und NLI-API."""

import asyncio
import gc
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from kiron_common.embedding_registry import (
    MODEL_CATALOG as SHARED_MODEL_CATALOG,
    MODEL_STATE_VIEW,
)
from kiron_common.model_catalog import BackendType, ModelEndpoint
from kiron_common.model_state import (
    BackendRuntimeSnapshot,
    LocalModelInventory,
    RuntimeInventory,
    default_huggingface_hub_cache,
    scan_huggingface_inventory,
)

from catalog_view import (
    DebertaServiceModel,
    DebertaServiceView,
    build_deberta_service_view,
)
from loaders import LOADER_REGISTRY
from operation_tracking import OperationLedger, tracked_route_class
from kiron_common.gpu_admission.native_contract import OPERATION_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEBERTA_CATALOG_VIEW = build_deberta_service_view(
    SHARED_MODEL_CATALOG,
    LOADER_REGISTRY,
)


def _local_model_inventory() -> LocalModelInventory:
    return LocalModelInventory(
        huggingface_revisions=scan_huggingface_inventory(
            MODEL_STATE_VIEW,
            default_huggingface_hub_cache(),
        )
    )


def _model_states_from_snapshot(
    snapshot: dict[str, object],
    local_inventory: LocalModelInventory,
) -> list[dict[str, object]]:
    raw_current = snapshot.get("current_model")
    loaded = (
        frozenset((raw_current,))
        if isinstance(raw_current, str) and raw_current
        else frozenset()
    )
    raw_loading = snapshot.get("loading_model")
    loading = (
        frozenset((raw_loading,))
        if isinstance(raw_loading, str) and raw_loading
        else frozenset()
    )
    runtime = RuntimeInventory({
        BackendType.KIRON_DEBERTA: BackendRuntimeSnapshot(
            known=True,
            loaded_names=loaded,
            loading_names=loading,
        )
    })
    return [
        state.to_dict()
        for state in MODEL_STATE_VIEW.states(
            local_inventory,
            runtime,
            backends=(BackendType.KIRON_DEBERTA,),
        )
    ]
_CUDA_RUNTIME_ERROR_HINTS = (
    "out of memory",
    "cuda",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "device-side assert",
    "illegal memory access",
)


def normalize_model_name(
    name: object,
    endpoint: ModelEndpoint | None = None,
) -> str | None:
    """Resolve only exact canonical IDs and explicitly declared aliases."""

    model = DEBERTA_CATALOG_VIEW.resolve(name, endpoint)
    return model.model_name if model is not None else None


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _CUDA_RUNTIME_ERROR_HINTS)


def _gpu_error_response(error: str, exc: BaseException) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": str(exc)},
        status_code=503,
    )


class GPUCapacityError(Exception):
    def __init__(self, required_bytes: int, free_bytes: int):
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes
        super().__init__("Nicht genuegend freier GPU-Speicher fuer das Modell.")


def _capacity_error_response(exc: GPUCapacityError) -> JSONResponse:
    return JSONResponse({"error": str(exc), "code": "resource_exhausted",
                         "required_bytes": exc.required_bytes,
                         "free_bytes": exc.free_bytes}, status_code=503)


async def _shielded_to_thread(func, /, *args, on_cancel_error=None, **kwargs):
    """asyncio.to_thread-Variante, die bei Cancellation auf Thread-Ende wartet (#835).

    asyncio.to_thread/run_in_executor kann den OS-Thread nicht killen. Ohne diesen
    Schutz wuerde eine CancelledError im umgebenden ``async with model_manager._lock``
    den Lock freigeben, waehrend der Predict-Thread noch self.model benutzt — ein
    paralleler Request koennte den Lock holen und einen zweiten predict-Aufruf auf
    demselben Modell-Objekt starten. Hier wird die CancelledError erst nach
    Thread-Abschluss propagiert, sodass der Lock bis dahin gehalten bleibt.

    Wenn der Worker-Thread waehrend Cancellation mit einer Exception endet (z.B.
    CUDA-OOM/RuntimeError), wuerde diese ohne Callback verworfen — der Endpoint
    saehe nur CancelledError und ruefe force_reset_locked nicht auf, sodass der
    GPU-State potentiell korrupt im ModelManager bliebe (#855). Der optionale
    on_cancel_error-Callback erhaelt die Worker-Exception und wird unter dem
    Lock-Kontext des Aufrufers ausgefuehrt, bevor CancelledError propagiert.
    """
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # #1032: Wait-Schleife statt einmaligem shield. Wird der umgebende Task
        # waehrend des Recovery-Wait erneut gecancelt, raised der innere shield
        # erneut CancelledError — ohne Schleife wuerde diese vor task.done()
        # propagieren und der Caller-Lock waehrend laufendem Thread freigegeben.
        while not task.done():
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        worker_exc: BaseException | None = None
        try:
            worker_exc = task.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            worker_exc = None
        if worker_exc is not None and on_cancel_error is not None:
            try:
                on_cancel_error(worker_exc)
            except Exception:
                logger.exception("on_cancel_error callback raised")
        raise


# --- Request-Schemas ---

# #502: Per-Item-Limit gegen DoS-Vektor — Tokenizer trunktiert ohnehin bei ~512 Tokens (~2-4 KB),
# 50 KB ist grosszuegig genug fuer legitime Document-Snippets und blockt MB-grosse Pathologie-Inputs.
LongString = Annotated[str, StringConstraints(max_length=50_000)]


class RerankRequest(BaseModel):
    model: str = DEBERTA_CATALOG_VIEW.request_default(
        ModelEndpoint.RERANK
    ).model_name
    query: LongString
    documents: list[LongString] = Field(max_length=256)
    top_k: int | None = Field(default=None, ge=0, strict=True)


class ScoreRequest(BaseModel):
    model: str = DEBERTA_CATALOG_VIEW.request_default(
        ModelEndpoint.SCORE
    ).model_name
    # #902: Inner-Strukturen tolerant aufnehmen (kein list[list[Any]], kein StringConstraints),
    # damit Format-Fehler im Endpoint einheitlich als 400+error gemeldet werden statt FastAPI-422+detail.
    # Doku (kb-kiron-validation-contracts.md) spezifiziert 400 fuer falsches Format.
    pairs: list[Any] = Field(max_length=256)


class UnloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1)


# --- Model Manager ---

class ModelManager:
    def __init__(
        self,
        service_view: DebertaServiceView = DEBERTA_CATALOG_VIEW,
    ) -> None:
        self._service_view = service_view
        self.current_model_name: str | None = None
        self.model: Any | None = None
        self.config: DebertaServiceModel | None = None
        self.loading_model: str | None = None
        self._lock = asyncio.Lock()
        self._state_lock = threading.RLock()

    def snapshot(self) -> dict[str, object]:
        """Return current/loading state from one atomic read."""

        with self._state_lock:
            current = (
                self.current_model_name
                if self.model is not None and self.config is not None
                else None
            )
            return {
                "current_model": current,
                "loaded_models": [current] if current is not None else [],
                "loading_model": self.loading_model,
            }

    async def get_model(
        self, short_name: str
    ) -> tuple[Any, DebertaServiceModel]:
        """Gibt Modell+Config zurueck. Lock wird NICHT gehalten — Caller muss _lock verwenden."""
        with self._state_lock:
            if (
                self.current_model_name == short_name
                and self.model is not None
                and self.config is not None
            ):
                self._check_memory(self.config, loading=False)
                return self.model, self.config
        return await _shielded_to_thread(self._load_model, short_name)

    def _check_memory(self, config: DebertaServiceModel, *, loading: bool) -> None:
        # Fresh device-wide free memory includes allocations by unmanaged apps.
        # The old resident remains accounted for during transactional replacement.
        free, _total = torch.cuda.mem_get_info()
        budget = config.gpu_memory
        required = budget.additional_bytes(loading=loading) + budget.headroom_bytes
        if free < required:
            raise GPUCapacityError(required, free)

    def _load_model(
        self, short_name: str
    ) -> tuple[Any, DebertaServiceModel]:
        with self._state_lock:
            self.loading_model = short_name
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA nicht verfuegbar - DeBERTa Service benoetigt GPU")
            config = self._service_view.require_runtime_model(short_name)
            self._check_memory(config, loading=True)

            # Transactional: neues Modell laden, altes erst nach erfolgreichem Load freigeben (#438)
            hf_name = config.artifact.repository
            precision = config.precision.upper()
            logger.info(f"Lade {hf_name} auf GPU ({precision})...")
            try:
                loader = self._service_view.loader_for(config)
                new_model = loader(config)
            except (RuntimeError, OSError):
                # Neuer Load fehlgeschlagen - altes Modell bleibt aktiv, Service bleibt funktionsfaehig
                # Lokale Referenz vor empty_cache freigeben, damit VRAM tatsaechlich freigegeben werden kann
                try:
                    del new_model
                except NameError:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise

            # Commit des neuen Runtime-Snapshots atomar; teure Freigabe des alten
            # Objekts danach ausserhalb des State-Locks.
            with self._state_lock:
                old_model = self.model
                old_name = self.current_model_name
                self.model = new_model
                self.config = config
                self.current_model_name = short_name
            if old_model is not None and old_model is not new_model:
                logger.info(f"Entlade {old_name}...")
                del old_model
                torch.cuda.empty_cache()
            logger.info(f"Modell {short_name} ({hf_name}) bereit.")
            return new_model, config
        finally:
            with self._state_lock:
                self.loading_model = None

    async def unload(self, expected_model: str | None = None) -> bool:
        async with self._lock:
            # Check the target under the inference/load lock. A model switch
            # after dashboard discovery must never unload the replacement.
            with self._state_lock:
                if expected_model is not None and self.current_model_name != expected_model:
                    return False
            return await _shielded_to_thread(self._unload_locked)

    def _unload_locked(self) -> bool:
        """Modell entladen - Caller MUSS _lock halten."""
        with self._state_lock:
            if self.model is None:
                return False
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            old_model = self.model
            old_name = self.current_model_name
            self.model = None
            self.current_model_name = None
            self.config = None
            self.loading_model = None
        logger.info(f"Entlade {old_name}...")
        del old_model
        # Transformers can retain cyclic Python references. Collect them before
        # releasing the CUDA allocator cache, while the inference lock is held.
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Modell entladen, VRAM freigegeben.")
        return True

    def force_reset_locked(self):
        """Nach CUDA-OOM/Fehler State haerter zuruecksetzen, damit naechster Request neu laedt (#220).

        Caller MUSS _lock halten.
        """
        logger.warning("force_reset_locked: setze ModelManager-State zurueck nach CUDA-Fehler")
        with self._state_lock:
            self.model = None
            self.current_model_name = None
            self.config = None
            self.loading_model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app):
    yield
    try:
        await model_manager.unload()
    except Exception:
        logger.exception("Shutdown cleanup failed")


app = FastAPI(title="Kiron DeBERTa Cross-Encoder", lifespan=lifespan)
operation_ledger = OperationLedger()


async def _confirm_operation_end() -> bool:
    # All model work uses _shielded_to_thread, including cancellation. Acquire
    # the same lock before confirming that no GPU work remains in this process.
    async with model_manager._lock:
        if torch.cuda.is_initialized():
            try:
                await _shielded_to_thread(torch.cuda.synchronize)
            except Exception:
                logger.exception("Native operation end could not be confirmed")
                return False
        return True


app.router.route_class = tracked_route_class(operation_ledger, _confirm_operation_end)


@app.get(OPERATION_PATH + "{operation_id}")
async def operation_status(operation_id: str):
    snapshot = operation_ledger.snapshot(operation_id)
    if snapshot is None:
        return JSONResponse({"error": "operation unknown"}, status_code=404)
    return JSONResponse(snapshot, headers={"Cache-Control": "no-store"})


# --- Reranking-Endpoint ---

@app.post("/api/rerank")
async def rerank(req: RerankRequest):
    resolved = normalize_model_name(req.model, ModelEndpoint.RERANK)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird nicht unterstuetzt.",
             "available_models": list(
                 DEBERTA_CATALOG_VIEW.available_model_names(
                     ModelEndpoint.RERANK
                 )
             )},
            status_code=400,
        )

    # #336: Query-Validierung VOR dem documents=[] Short-Circuit, sonst akzeptiert der
    # Endpoint query="" solange documents leer ist - inkonsistent zum docs-Pfad der 400 wirft.
    if not req.query or not req.query.strip():
        return JSONResponse({"error": "query darf nicht leer sein."}, status_code=400)

    if not req.documents:
        return {"model": resolved, "results": []}

    empty_docs = [i for i, d in enumerate(req.documents) if not d or not d.strip()]
    if empty_docs:
        return JSONResponse(
            {"error": f"documents enthaelt leere Strings an Index {empty_docs}"},
            status_code=400,
        )

    if req.top_k == 0:
        return {"model": resolved, "results": []}

    async with model_manager._lock:
        try:
            model, config = await model_manager.get_model(resolved)
        except GPUCapacityError as exc:
            return _capacity_error_response(exc)
        except OSError:
            logger.exception(f"OSError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": f"Modell '{resolved}' nicht auf dem Server installiert.",
                 "hint": "Server-Administrator muss das Modell herunterladen."},
                status_code=500,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM beim Laden von Modell '{resolved}'")
            model_manager.force_reset_locked()
            return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.exception(f"CUDA RuntimeError beim Laden von Modell '{resolved}'")
                model_manager.force_reset_locked()
                return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
            logger.exception(f"RuntimeError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": "Interner Fehler beim Laden des Modells.", "detail": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler beim Laden von Modell '{resolved}': {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": f"Modell '{resolved}' konnte nicht geladen werden ({type(e).__name__}).",
                 "detail": str(e)},
                status_code=500,
            )

        start = time.monotonic()
        pairs = [(req.query, doc) for doc in req.documents]

        is_classification = config.labels is not None

        def _predict():
            with torch.amp.autocast(device_type="cuda"):
                return model.predict(pairs, apply_softmax=is_classification)

        def _on_cancel_cuda_reset(exc: BaseException) -> None:
            # #855: Worker-Exception aus Cancellation-Pfad — bei CUDA-Fehler State unter Lock zuruecksetzen.
            if isinstance(exc, torch.cuda.OutOfMemoryError) or (
                isinstance(exc, RuntimeError) and _is_cuda_runtime_error(exc)
            ):
                logger.error(
                    f"CUDA-Fehler im Worker nach Cancellation ({resolved}): {type(exc).__name__}: {exc}",
                    exc_info=exc,
                )
                model_manager.force_reset_locked()

        try:
            scores = await _shielded_to_thread(_predict, on_cancel_error=_on_cancel_cuda_reset)
        except OSError as e:
            logger.exception(f"OSError bei Inference ({resolved})")
            return JSONResponse(
                {"error": "Inference fehlgeschlagen (I/O-Fehler).", "detail": str(e)},
                status_code=500,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM bei Inference ({resolved})")
            # Modell-State nach CUDA-OOM ist potentiell korrupt - force reset (#220).
            # Lokale model/_predict-Refs vor empty_cache freigeben, sonst behaelt PyTorch das VRAM.
            del model, _predict
            model_manager.force_reset_locked()
            return _gpu_error_response("Inference fehlgeschlagen (GPU-Fehler/OOM).", e)
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.exception(f"CUDA RuntimeError bei Inference ({resolved})")
                del model, _predict
                model_manager.force_reset_locked()
                return _gpu_error_response("Inference fehlgeschlagen (GPU-Fehler/OOM).", e)
            logger.exception(f"RuntimeError bei Inference ({resolved})")
            return JSONResponse(
                {"error": "Interner Inference-Fehler.", "detail": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler bei Inference ({resolved}): {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": f"Inference fehlgeschlagen ({type(e).__name__}).",
                 "detail": str(e)},
                status_code=500,
            )
        scores = np.array(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if is_classification and scores.ndim == 1:
            scores = scores.reshape(1, -1)
        # NaN/Inf neutralisieren (FP16-Overflow) — sonst produziert json.dumps nicht-standard Literals.
        # posinf/neginf auf float32-Grenzen (nicht 0.0), sonst wuerde Overflow das Ranking bei Regression invertieren (#538).
        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=float(np.finfo(np.float32).max),
            neginf=float(np.finfo(np.float32).min),
        )

        # Reranking-Score bestimmen
        if is_classification:
            # #537: Defensiver Shape-Check — bei unerwartetem Skalar-Output von model.predict()
            # wuerde scores.reshape(1,-1) zu (1,1) werden und scores[:, rerank_idx>0] IndexError werfen.
            n_expected = len(config.labels)
            if scores.ndim != 2 or scores.shape[1] != n_expected:
                # #873: Strukturierte JSONResponse statt raise — sonst faellt FastAPI auf
                # generisches {"detail": "Internal Server Error"} zurueck.
                detail = (
                    f"Unerwartete Score-Shape {scores.shape} fuer classification-Modell "
                    f"(erwartet (n,{n_expected}))"
                )
                logger.error(f"{detail} ({resolved})")
                return JSONResponse(
                    {"error": "Interner Inference-Fehler.", "detail": detail},
                    status_code=500,
                )
            rerank_idx = config.labels.index(config.rerank_label)
            rerank_scores = scores[:, rerank_idx]
        else:
            # Single-Score Regression (z.B. ms-marco)
            rerank_scores = scores

        # Nach Score absteigend sortieren (stable fuer deterministische Tie-Breaks)
        ranked_indices = np.argsort(-rerank_scores, kind="stable")
        top_k = len(req.documents) if req.top_k is None else min(req.top_k, len(req.documents))
        ranked_indices = ranked_indices[:top_k]

        results = [
            {
                "index": int(idx),
                "score": round(float(rerank_scores[idx]), 6),
                "document": req.documents[idx],
            }
            for idx in ranked_indices
        ]

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(f"[{resolved}] Reranked {len(req.documents)} docs in {duration_ms:.1f}ms")

        return {"model": resolved, "results": results}


# --- NLI-Score-Endpoint ---

@app.post("/api/score")
async def score(req: ScoreRequest):
    resolved = normalize_model_name(req.model, ModelEndpoint.SCORE)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird nicht unterstuetzt.",
             "available_models": list(
                 DEBERTA_CATALOG_VIEW.available_model_names(
                     ModelEndpoint.SCORE
                 )
             )},
            status_code=400,
        )

    if not req.pairs:
        return {"model": resolved, "results": []}

    for i, p in enumerate(req.pairs):
        if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(x, str) for x in p):
            return JSONResponse(
                {"error": f"pairs[{i}] muss genau 2 Strings enthalten (text, text_pair)."},
                status_code=400,
            )
        if len(p[0]) > 50_000 or len(p[1]) > 50_000:
            return JSONResponse(
                {"error": f"pairs[{i}] enthaelt Strings >50000 Zeichen."},
                status_code=400,
            )
        if not p[0].strip() or not p[1].strip():
            return JSONResponse(
                {"error": f"pairs[{i}] enthaelt leere Strings."},
                status_code=400,
            )

    async with model_manager._lock:
        try:
            model, config = await model_manager.get_model(resolved)
        except GPUCapacityError as exc:
            return _capacity_error_response(exc)
        except OSError:
            logger.exception(f"OSError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": f"Modell '{resolved}' nicht auf dem Server installiert."},
                status_code=500,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM beim Laden von Modell '{resolved}'")
            model_manager.force_reset_locked()
            return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.exception(f"CUDA RuntimeError beim Laden von Modell '{resolved}'")
                model_manager.force_reset_locked()
                return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
            logger.exception(f"RuntimeError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": "Interner Fehler beim Laden des Modells.", "detail": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler beim Laden von Modell '{resolved}': {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": f"Modell '{resolved}' konnte nicht geladen werden ({type(e).__name__}).",
                 "detail": str(e)},
                status_code=500,
            )

        start = time.monotonic()
        pairs = [tuple(p) for p in req.pairs]
        is_classification = config.labels is not None

        def _predict():
            with torch.amp.autocast(device_type="cuda"):
                return model.predict(pairs, apply_softmax=is_classification)

        def _on_cancel_cuda_reset(exc: BaseException) -> None:
            # #855: Worker-Exception aus Cancellation-Pfad — bei CUDA-Fehler State unter Lock zuruecksetzen.
            if isinstance(exc, torch.cuda.OutOfMemoryError) or (
                isinstance(exc, RuntimeError) and _is_cuda_runtime_error(exc)
            ):
                logger.error(
                    f"CUDA-Fehler im Worker nach Cancellation ({resolved}): {type(exc).__name__}: {exc}",
                    exc_info=exc,
                )
                model_manager.force_reset_locked()

        try:
            scores = await _shielded_to_thread(_predict, on_cancel_error=_on_cancel_cuda_reset)
        except OSError as e:
            logger.exception(f"OSError bei Inference ({resolved})")
            return JSONResponse(
                {"error": "Inference fehlgeschlagen (I/O-Fehler).", "detail": str(e)},
                status_code=500,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM bei Inference ({resolved})")
            # Modell-State nach CUDA-OOM ist potentiell korrupt - force reset (#220).
            # Lokale model/_predict-Refs vor empty_cache freigeben, sonst behaelt PyTorch das VRAM.
            del model, _predict
            model_manager.force_reset_locked()
            return _gpu_error_response("Inference fehlgeschlagen (GPU-Fehler/OOM).", e)
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.exception(f"CUDA RuntimeError bei Inference ({resolved})")
                del model, _predict
                model_manager.force_reset_locked()
                return _gpu_error_response("Inference fehlgeschlagen (GPU-Fehler/OOM).", e)
            logger.exception(f"RuntimeError bei Inference ({resolved})")
            return JSONResponse(
                {"error": "Interner Inference-Fehler.", "detail": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler bei Inference ({resolved}): {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": f"Inference fehlgeschlagen ({type(e).__name__}).",
                 "detail": str(e)},
                status_code=500,
            )
        scores = np.array(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        # NaN/Inf neutralisieren (FP16-Overflow) — sonst produziert json.dumps nicht-standard Literals
        # posinf/neginf auf float32-Grenzen (nicht 0.0), sonst wuerde Overflow das Ranking bei Regression invertieren (#539).
        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=float(np.finfo(np.float32).max),
            neginf=float(np.finfo(np.float32).min),
        )

        # Label-Namen zuordnen
        labels = config.labels
        if labels is not None:
            # #1033: Strikter ndim-Check analog zu /api/rerank (#537), sonst faellt eine
            # unerwartete 3D-Shape (z.B. (1,1,3)) in den else-Zweig und liefert
            # Regression-Score-Dicts statt Label-Dicts.
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)
            n_expected = len(labels)
            if scores.ndim != 2 or scores.shape[1] != n_expected:
                # #873: Strukturierte JSONResponse statt raise — sonst faellt FastAPI auf
                # generisches {"detail": "Internal Server Error"} zurueck.
                detail = (
                    f"Unerwartete Score-Shape {scores.shape} fuer classification-Modell "
                    f"(erwartet (n,{n_expected}))"
                )
                logger.error(f"{detail} ({resolved})")
                return JSONResponse(
                    {"error": "Interner Inference-Fehler.", "detail": detail},
                    status_code=500,
                )
            results = [
                {label: round(float(scores[i, j]), 6) for j, label in enumerate(labels)}
                for i in range(len(pairs))
            ]
        else:
            scores_flat = scores.squeeze() if scores.ndim > 1 else scores
            if scores_flat.ndim == 0:
                scores_flat = scores_flat.reshape(1)
            results = [{"score": round(float(s), 6)} for s in scores_flat]

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(f"[{resolved}] Scored {len(pairs)} pairs in {duration_ms:.1f}ms")

        return {"model": resolved, "results": results}


# --- Management-Endpoints ---

@app.get("/health")
def health():
    snapshot = model_manager.snapshot()
    loading = snapshot["loading_model"]
    current = snapshot["current_model"]
    current_config = DEBERTA_CATALOG_VIEW.runtime_model(current)
    model_loaded = current is not None
    if loading is not None:
        status = "loading"
        status_code = 503
    else:
        status = "ok" if model_loaded else "no_model"
        status_code = 200 if model_loaded else 503
    content = {
        "status": status,
        "catalog_digest": DEBERTA_CATALOG_VIEW.catalog_digest,
        "current_model": current,
        "loaded_models": snapshot["loaded_models"],
        "precision": (
            current_config.precision if current_config is not None else "fp16"
        ),
        "available_models": list(
            DEBERTA_CATALOG_VIEW.available_model_names()
        ),
        "model_states": _model_states_from_snapshot(
            snapshot,
            _local_model_inventory(),
        ),
    }
    if loading is not None:
        content["loading_model"] = loading
    return JSONResponse(
        content,
        status_code=status_code,
    )


@app.post("/api/load")
async def load_model_endpoint(body: dict):
    """Modell explizit laden/wechseln."""
    name = body.get("model", "")
    if not isinstance(name, str):
        return JSONResponse(
            {"error": f"Modell '{name}' nicht unterstuetzt.",
             "available_models": list(
                 DEBERTA_CATALOG_VIEW.available_model_names()
             )},
            status_code=400,
        )
    resolved = normalize_model_name(name)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{name}' nicht unterstuetzt.",
             "available_models": list(
                 DEBERTA_CATALOG_VIEW.available_model_names()
             )},
            status_code=400,
        )
    async with model_manager._lock:
        try:
            await model_manager.get_model(resolved)
            return {"status": "ok", "model": resolved}
        except GPUCapacityError as exc:
            return _capacity_error_response(exc)
        except OSError:
            logger.exception(f"OSError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": f"Modell '{resolved}' nicht auf dem Server installiert."},
                status_code=500,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM beim Laden von Modell '{resolved}'")
            model_manager.force_reset_locked()
            return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.exception(f"CUDA RuntimeError beim Laden von Modell '{resolved}'")
                model_manager.force_reset_locked()
                return _gpu_error_response("Modell-Load fehlgeschlagen (GPU-Fehler/OOM).", e)
            logger.exception(f"RuntimeError beim Laden von Modell '{resolved}'")
            return JSONResponse(
                {"error": "Interner Fehler beim Laden des Modells.", "detail": str(e)},
                status_code=500,
            )
        except Exception as e:
            logger.exception(f"Unerwarteter Fehler beim Laden von Modell '{resolved}': {type(e).__name__}: {e}")
            return JSONResponse(
                {"error": f"Modell '{resolved}' konnte nicht geladen werden ({type(e).__name__}).",
                 "detail": str(e)},
                status_code=500,
            )


@app.post("/api/unload")
async def unload_model_endpoint(req: UnloadRequest):
    """Genau das angegebene Modell entladen; wiederholte Aufrufe sind harmlos."""
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse({"error": f"Modell '{req.model}' nicht unterstuetzt."}, status_code=400)
    try:
        unloaded = await model_manager.unload(expected_model=resolved)
        return {"status": "unloaded", "model": resolved, "already": not unloaded}
    except Exception:
        logger.exception("[%s] Fehler beim Entladen", resolved)
        return JSONResponse({"error": f"Modell '{resolved}' konnte nicht entladen werden."}, status_code=500)


@app.get("/api/tags")
def tags():
    """Listet alle unterstuetzten Modelle."""
    return {
        "models": [
            {"name": model.model_name, "size": model.size}
            for model in DEBERTA_CATALOG_VIEW.models
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11437)
