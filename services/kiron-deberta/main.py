"""Kiron DeBERTa Cross-Encoder Service mit Reranking- und NLI-API."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Modell-Mapping: Kurzname → HuggingFace-Name + Konfiguration
# labels=None → Single-Score Regression (Reranking)
# labels=[...] → Multi-Label Klassifikation (NLI)
MODELS = {
    "nli-deberta-v3-base": {
        "hf": "cross-encoder/nli-deberta-v3-base",
        "labels": ["contradiction", "entailment", "neutral"],
        "rerank_label": "entailment",
        "size": 748_850_969,
    },
    "ms-marco-MiniLM-L-6-v2": {
        "hf": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "labels": None,
        "rerank_label": None,
        "size": 91_816_134,
    },
}

DEFAULT_MODEL = "ms-marco-MiniLM-L-6-v2"
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


def normalize_model_name(name: str) -> str | None:
    """Modellname normalisieren und gegen Mapping pruefen."""
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    if ":" in name:
        name = name.split(":")[0]
    return name if name in MODELS else None


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _CUDA_RUNTIME_ERROR_HINTS)


def _gpu_error_response(error: str, exc: BaseException) -> JSONResponse:
    return JSONResponse(
        {"error": error, "detail": str(exc)},
        status_code=503,
    )


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
    model: str = DEFAULT_MODEL
    query: LongString
    documents: list[LongString] = Field(max_length=256)
    top_k: int | None = Field(default=None, ge=0, strict=True)


class ScoreRequest(BaseModel):
    model: str = DEFAULT_MODEL
    # #902: Inner-Strukturen tolerant aufnehmen (kein list[list[Any]], kein StringConstraints),
    # damit Format-Fehler im Endpoint einheitlich als 400+error gemeldet werden statt FastAPI-422+detail.
    # Doku (kb-kiron-validation-contracts.md) spezifiziert 400 fuer falsches Format.
    pairs: list[Any] = Field(max_length=256)


# --- Model Manager ---

class ModelManager:
    def __init__(self):
        self.current_model_name: str | None = None
        self.model: CrossEncoder | None = None
        self.config: dict | None = None
        self.loading_model: str | None = None
        self._lock = asyncio.Lock()

    async def get_model(self, short_name: str) -> tuple[CrossEncoder, dict]:
        """Gibt Modell+Config zurueck. Lock wird NICHT gehalten — Caller muss _lock verwenden."""
        if self.current_model_name == short_name and self.model is not None and self.config is not None:
            return self.model, self.config
        return await _shielded_to_thread(self._load_model, short_name)

    def _load_model(self, short_name: str) -> tuple[CrossEncoder, dict]:
        self.loading_model = short_name
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA nicht verfuegbar - DeBERTa Service benoetigt GPU")
            config = MODELS[short_name]

            # Transactional: neues Modell laden, altes erst nach erfolgreichem Load freigeben (#438)
            hf_name = config["hf"]
            logger.info(f"Lade {hf_name} auf GPU (FP16)...")
            try:
                # Direkt in FP16 laden, nicht erst FP32 -> .half() (sonst doppelter VRAM-Peak beim Load)
                new_model = CrossEncoder(
                    hf_name,
                    max_length=512,
                    device="cuda",
                    model_kwargs={"torch_dtype": torch.float16},
                )
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

            # Neuer Load erfolgreich - altes Modell entladen
            if self.model is not None:
                logger.info(f"Entlade {self.current_model_name}...")
                del self.model
                torch.cuda.empty_cache()

            self.model = new_model
            self.config = config
            self.current_model_name = short_name
            logger.info(f"Modell {short_name} ({hf_name}) bereit.")
            return new_model, config
        finally:
            self.loading_model = None

    async def unload(self) -> bool:
        async with self._lock:
            return self._unload_locked()

    def _unload_locked(self) -> bool:
        """Modell entladen - Caller MUSS _lock halten."""
        if self.model is None:
            return False
        logger.info(f"Entlade {self.current_model_name}...")
        del self.model
        self.model = None
        self.current_model_name = None
        self.config = None
        self.loading_model = None
        torch.cuda.empty_cache()
        logger.info("Modell entladen, VRAM freigegeben.")
        return True

    def force_reset_locked(self):
        """Nach CUDA-OOM/Fehler State haerter zuruecksetzen, damit naechster Request neu laedt (#220).

        Caller MUSS _lock halten.
        """
        logger.warning("force_reset_locked: setze ModelManager-State zurueck nach CUDA-Fehler")
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


# --- Reranking-Endpoint ---

@app.post("/api/rerank")
async def rerank(req: RerankRequest):
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird nicht unterstuetzt.",
             "available_models": list(MODELS.keys())},
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

        is_classification = config["labels"] is not None

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
            n_expected = len(config["labels"])
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
            rerank_idx = config["labels"].index(config["rerank_label"])
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
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird nicht unterstuetzt.",
             "available_models": list(MODELS.keys())},
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
        is_classification = config["labels"] is not None

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
        labels = config["labels"]
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
    # Snapshot, damit status und current_model nicht durch concurrent unload divergieren
    loading = model_manager.loading_model
    current = model_manager.current_model_name
    model_loaded = current is not None
    if loading is not None:
        status = "loading"
        status_code = 503
    else:
        status = "ok" if model_loaded else "no_model"
        status_code = 200 if model_loaded else 503
    content = {
        "status": status,
        "current_model": current,
        "precision": "fp16",
        "available_models": list(MODELS.keys()),
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
             "available_models": list(MODELS.keys())},
            status_code=400,
        )
    resolved = normalize_model_name(name)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{name}' nicht unterstuetzt.",
             "available_models": list(MODELS.keys())},
            status_code=400,
        )
    async with model_manager._lock:
        try:
            await model_manager.get_model(resolved)
            return {"status": "ok", "model": resolved}
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
async def unload_model_endpoint():
    """Modell entladen und VRAM freigeben."""
    unloaded = await model_manager.unload()
    if unloaded:
        return {"status": "unloaded"}
    return {"status": "no_model_loaded"}


@app.get("/api/tags")
def tags():
    """Listet alle unterstuetzten Modelle."""
    return {"models": [{"name": k, "size": cfg.get("size", 0)} for k, cfg in MODELS.items()]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11437)
