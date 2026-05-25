"""Sentence-Transformers Embedding Service mit Ollama-kompatibler API."""

import hashlib
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from model_worker import (
    MODEL_WORKER_SHUTDOWN_TIMEOUT_S,
    CudaRuntimeError,
    EncodeResult,
    LateEmbedResult,
    NonFiniteEmbeddingError,
    SerialModelWorker,
    WorkerQueueFullError,
    WorkerStoppedError,
    derive_health_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# size: FP16-Gewichte (~2 Bytes/Param), grobe Schaetzung fuer Ollama-Kompatibilitaet
OLLAMA_TO_HF = {
    # Revision gepinnt (#229) - trust_remote_code laedt beliebigen Python-Code; ohne Pin
    # wuerde bei jedem Service-Start der aktuelle main-Branch vom HF-Repo geladen und neuer
    # Code ausgefuehrt. Pin auf bekannt-funktionierenden Commit vom 2026-04-07.
    "nomic-embed-text":       {"hf": "nomic-ai/nomic-embed-text-v1.5", "trust_remote_code": True,
                                "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
                                "family": "nomic-bert", "parameter_size": "137M", "size": 274_000_000,
                                "prefix_scheme": "nomic"},
    "mxbai-embed-large":      {"hf": "mixedbread-ai/mxbai-embed-large-v1",
                                "revision": "b33106f585b9ce46904ad7443a3b52b7a63e231c",
                                "family": "bert", "parameter_size": "335M", "size": 670_000_000},
    "snowflake-arctic-embed":  {"hf": "Snowflake/snowflake-arctic-embed-l",
                                "revision": "d8fb21ca8d905d2832ee8b96c894d3298964346b",
                                "family": "bert", "parameter_size": "335M", "size": 670_000_000},
    "bge-m3":                  {"hf": "BAAI/bge-m3",
                                "revision": "5617a9f61b028005a4858fdac845db406aefb181",
                                "family": "xlm-roberta", "parameter_size": "568M", "size": 1_136_000_000},
}

# nomic-v1.5 wurde mit Task-Prefixes trainiert. Ohne Prefix landen Vektoren
# im neutralen Space statt im Query<->Document-optimierten Space (#171).
_NOMIC_VALID_INPUT_TYPES = {"search_document", "search_query", "classification", "clustering"}
_NOMIC_DEFAULT_INPUT_TYPE = "search_document"


def _apply_prefix(texts: list[str], config: dict, input_type: str | None) -> list[str]:
    """Wenn das Modell eine Prefix-Konvention hat, Text mit {input_type}: praefixieren.

    nomic-v1.5: Default search_document; valide Werte siehe _NOMIC_VALID_INPUT_TYPES.
    Andere Modelle: input_type wird ignoriert (Rueckgabe unveraendert).
    """
    scheme = config.get("prefix_scheme")
    if scheme != "nomic":
        return texts
    prefix = input_type if input_type else _NOMIC_DEFAULT_INPUT_TYPE
    return [f"{prefix}: {t}" for t in texts]

# Deterministischer ISO-Timestamp pro Modell aus hf+revision-Hash. Stabil ueber
# Service-Restarts, aendert sich nur bei Revision-Pin-Wechsel — sonst wuerde
# jeder Restart Ollama-Client-Caches invalidieren. Mapping in [2020-01-01,
# 2030-01-01) damit das Feld als plausibler Modellzeitstempel wirkt.
_MODIFIED_AT_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc)
_MODIFIED_AT_SPAN_S = 10 * 365 * 24 * 3600


def _model_modified_at(cfg: dict) -> str:
    digest = hashlib.sha256(f"{cfg['hf']}@{cfg['revision']}".encode()).digest()
    offset_s = int.from_bytes(digest[:4], "big") % _MODIFIED_AT_SPAN_S
    return (_MODIFIED_AT_EPOCH + timedelta(seconds=offset_s)).isoformat()

# C-implementiertes Regex statt Pure-Python any(ord()) — letzteres iteriert
# bei Maximalanfrage (512*32768 chars) ueber 16M Codepoints und blockiert
# den Event-Loop sekundenlang.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# CUDA-RuntimeError-Marker: PyTorch wirft fuer ALLOC_FAILED, illegal memory
# access, CUBLAS/CUDNN-Statusfehler etc. einen generischen RuntimeError, der
# nicht von torch.cuda.OutOfMemoryError abgedeckt ist. Erkennung anhand der
# Message-Praefixe, die torch in `c10/cuda/CUDAException.cpp` setzt.
_CUDA_RUNTIME_MARKERS = (
    "CUDA error:",
    "CUBLAS_STATUS_",
    "CUDNN_STATUS_",
    "illegal memory access",
    "device-side assert",
    "CUDA driver version",
)


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_RUNTIME_MARKERS)


def normalize_model_name(name: str) -> str | None:
    """Ollama-Modellname normalisieren und gegen Mapping pruefen."""
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    if ":" in name:
        name = name.split(":")[0]
    return name if name in OLLAMA_TO_HF else None


class EmbedRequest(BaseModel):
    model: str
    input: list[str] | str
    # Optional, nur von Modellen mit Task-Prefix-Schema (aktuell nomic-v1.5) ausgewertet.
    # Erlaubte Werte fuer nomic: search_document, search_query, classification, clustering.
    # Default bei nomic wenn nicht gesetzt: search_document.
    input_type: str | None = None


class LateChunkInput(BaseModel):
    text: str
    char_start: int
    char_end: int


class LateBatchRequest(BaseModel):
    model: str
    document: str
    chunks: list[LateChunkInput]
    input_type: str | None = None


class LoadRequest(BaseModel):
    model: str


# Schwelle analog MAX_TEXTS/MAX_TEXT_LEN bei /api/embed. MAX_DOC_LEN ist
# Body-Schutz (32k Char), nicht Token-Limit — Tokenizer truncated zusaetzlich
# auf model.max_seq_length, abgeschnittene Chunks landen im Fallback-Pfad.
MAX_LATE_CHUNKS = 512
MAX_DOC_LEN = 32768
LATE_FALLBACK_RATIO_WARN = 0.5


class ModelManager:
    """Synchroner State- und Modellhalter. Alle mutierenden Methoden laufen
    nur im Worker-Thread. Source-of-Truth fuer Modellzustand."""

    def __init__(self):
        self.state_lock = threading.RLock()
        self.owner_thread_id: int | None = None
        self.current_model_name: str | None = None
        self.model: SentenceTransformer | None = None
        self.loading_model: str | None = None
        self.device: str = "unknown"

    def set_worker_thread(self) -> None:
        with self.state_lock:
            self.owner_thread_id = threading.get_ident()

    def assert_worker_thread(self) -> None:
        if (
            self.owner_thread_id is None
            or self.owner_thread_id != threading.get_ident()
        ):
            raise RuntimeError(
                f"ModelManager mutating method called outside worker thread "
                f"(owner={self.owner_thread_id}, current={threading.get_ident()})"
            )

    def snapshot(self) -> dict[str, object]:
        with self.state_lock:
            return {
                "current_model": self.current_model_name,
                "loading_model": self.loading_model,
                "device": self.device,
            }

    def clear_loading_for_crash(self) -> None:
        with self.state_lock:
            self.loading_model = None

    def _detect_device(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"CUDA Health-Probe fehlgeschlagen: {type(e).__name__}: {e}")
            return "cpu"
        return "cuda"

    def _load_model_sync(self, ollama_name: str) -> SentenceTransformer:
        self.assert_worker_thread()
        with self.state_lock:
            self.loading_model = ollama_name
            device = self.device
        try:
            detected = self._detect_device()
            if detected != device:
                logger.warning(
                    f"Embedding Device-Wechsel erkannt: {device} -> {detected}"
                )
                with self.state_lock:
                    if self.model is not None:
                        old = self.model
                        self.model = None
                        self.current_model_name = None
                        del old
                        if device == "cuda":
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    self.device = detected
                device = detected

            config = OLLAMA_TO_HF[ollama_name]
            hf_name = config["hf"]
            logger.info(
                f"Lade {hf_name} auf {device.upper()}"
                f"{' (FP16)' if device == 'cuda' else ''}..."
            )

            # #288: explizit device="cpu", sonst defaultet sentence-transformers auf cuda
            # und laedt das neue Modell als FP32 in den VRAM neben dem alten — OOM-Risiko.
            # #881: local_files_only=True erzwingt Local-Only-Vertrag — bei Cache-Miss
            # wirft HF LocalEntryNotFoundError (OSError-Subklasse), der vom bestehenden
            # OSError-Handler als "nicht auf dem Server installiert" gemeldet wird.
            new_model = SentenceTransformer(
                hf_name,
                device="cpu",
                trust_remote_code=config.get("trust_remote_code", False),
                revision=config.get("revision"),
                local_files_only=True,
            )

            if device == "cuda":
                # CUDA: altes Modell ZUERST entladen, BEVOR neues transferiert wird.
                # Bei OOM auf .half().to("cuda") ist das alte zwar verloren, aber
                # CUDA-OOM bei doppelter VRAM-Belegung waere garantiert.
                with self.state_lock:
                    if self.model is not None:
                        logger.info(f"Entlade {self.current_model_name}...")
                        old = self.model
                        self.model = None
                        self.current_model_name = None
                        del old
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                try:
                    new_model = new_model.half().to("cuda")
                except Exception as transfer_exc:
                    # Partielle GPU-Tensoren freigeben, sonst bleiben sie im
                    # PyTorch-Caching-Allocator bis Process-Exit.
                    del new_model
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # #858: CUDA-RuntimeError (kein OOM) als CudaRuntimeError
                    # weiterreichen, damit Handler 503 statt 500 liefern.
                    if not isinstance(
                        transfer_exc, torch.cuda.OutOfMemoryError
                    ) and _is_cuda_runtime_error(transfer_exc):
                        raise CudaRuntimeError(transfer_exc) from transfer_exc
                    raise

                with self.state_lock:
                    self.model = new_model
                    self.current_model_name = ollama_name
            else:
                logger.warning(
                    "CUDA nicht verfuegbar — CPU-Modus (langsam, nur fuer Tests)"
                )
                # CPU: altes Modell erst nach erfolgreichem Konstruktor des neuen
                # entladen, damit ein Load-Fehler nicht zum modelllosen Service fuehrt.
                # model + name in DERSELBEN state_lock-Acquisition setzen.
                with self.state_lock:
                    if self.model is not None:
                        logger.info(f"Entlade {self.current_model_name}...")
                        old = self.model
                        self.model = None
                        del old
                    self.model = new_model
                    self.current_model_name = ollama_name

            logger.info(
                f"Modell {ollama_name} ({hf_name}) bereit auf {device.upper()}."
            )
            return new_model
        finally:
            with self.state_lock:
                self.loading_model = None

    def _ensure_model_sync(self, ollama_name: str) -> tuple[SentenceTransformer, int]:
        self.assert_worker_thread()
        with self.state_lock:
            cached = self.current_model_name == ollama_name and self.model is not None
            current_device = self.device
        if cached:
            # #880: Device-Probe vor Short-Circuit. Ohne diese Pruefung bleibt
            # ein CPU-Fallback nach CUDA-Recovery dauerhaft auf CPU und ein
            # stales CUDA-Modell wird vor dem naechsten Encode nicht
            # invalidiert. Bei Device-Wechsel _load_model_sync erzwingen, das
            # die Transition korrekt handhabt.
            if self._detect_device() == current_device:
                with self.state_lock:
                    return (self.model, 0)
        load_start = time.monotonic()
        model = self._load_model_sync(ollama_name)
        load_duration_ns = int((time.monotonic() - load_start) * 1_000_000_000)
        return (model, load_duration_ns)

    def _encode_sync(
        self,
        ollama_name: str,
        texts: list[str],
        input_type: str | None,
    ) -> EncodeResult:
        self.assert_worker_thread()
        model, load_duration_ns = self._ensure_model_sync(ollama_name)

        config = OLLAMA_TO_HF[ollama_name]
        encode_texts = _apply_prefix(texts, config, input_type)

        with self.state_lock:
            use_cuda = self.device == "cuda"

        # #474: max_len statt avg_len. Tokenizer padded auf Laenge des laengsten Texts
        # — Mixed-Input (1 langer + N kurze) braucht VRAM nach max, nicht nach avg.
        # Aus encode_texts (mit Prefix), nicht texts: nomic-Prefixe (~17 Zeichen)
        # koennen Texte knapp unter den Schwellen 512/2048/8192 darueber schieben.
        # #935: Tokenizer-Laenge statt Codepoints. len(t) unterschaetzt CJK/multilinguale
        # Eingaben (xlm-roberta, BGE-M3) — 1-2 Token pro Zeichen statt Englisch ~0.25 —
        # und teilte zu grosse Batches zu, die auf 12 GiB VRAM OOM-en. truncation auf
        # max_seq_length, weil das Modell ohnehin trunkiert; Char-Fallback bei Fehler.
        try:
            batch_tokenizer_kwargs: dict = {"padding": False, "truncation": True}
            batch_max_seq_length = getattr(model, "max_seq_length", None)
            if (
                isinstance(batch_max_seq_length, int)
                and not isinstance(batch_max_seq_length, bool)
                and batch_max_seq_length > 0
            ):
                batch_tokenizer_kwargs["max_length"] = batch_max_seq_length
            batch_tokenized = model.tokenizer(encode_texts, **batch_tokenizer_kwargs)
            max_len = max(len(ids) for ids in batch_tokenized["input_ids"])
        except Exception:
            max_len = max(len(t) for t in encode_texts)
        if use_cuda:
            if max_len < 512:
                batch_size = 64
            elif max_len < 2048:
                batch_size = 32
            elif max_len < 8192:
                batch_size = 16
            else:
                batch_size = 8
        else:
            batch_size = 8 if max_len > 2048 else 16

        try:
            if use_cuda:
                with torch.amp.autocast("cuda"):
                    embeddings = model.encode(
                        encode_texts, batch_size=batch_size, convert_to_numpy=True
                    )
            else:
                embeddings = model.encode(
                    encode_texts, batch_size=batch_size, convert_to_numpy=True
                )
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"[{ollama_name}] CUDA OOM waehrend Encode — entlade Modell und leere Cache"
            )
            try:
                self._drop_model_sync()
            except Exception:
                logger.exception("Drop-Model nach CUDA-OOM fehlgeschlagen")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise
        except RuntimeError as e:
            # #858: CUDA-RuntimeError ausserhalb von OutOfMemoryError
            # (illegal memory access, CUBLAS_STATUS_*, CUDNN_STATUS_*)
            # kann den CUDA-Kontext und Modell-State korrumpieren —
            # Modell entladen, Cache leeren, als CudaRuntimeError raisen.
            if _is_cuda_runtime_error(e):
                logger.error(
                    f"[{ollama_name}] CUDA-RuntimeError ({type(e).__name__}): {e} "
                    "— entlade Modell und leere Cache"
                )
                try:
                    self._drop_model_sync()
                except Exception:
                    logger.exception("Drop-Model nach CUDA-RuntimeError fehlgeschlagen")
                raise CudaRuntimeError(e) from e
            raise

        # #605: prompt_eval_count via Tokenizer (padding=False, sonst Pad-Tokens
        # mitgezaehlt). Char//4-Heuristik als Fallback (#687, OpenAI-Standard).
        try:
            tokenizer_kwargs = {"padding": False, "truncation": True}
            max_seq_length = getattr(model, "max_seq_length", None)
            if (
                isinstance(max_seq_length, int)
                and not isinstance(max_seq_length, bool)
                and max_seq_length > 0
            ):
                tokenizer_kwargs["max_length"] = max_seq_length
            tokenized = model.tokenizer(encode_texts, **tokenizer_kwargs)
            prompt_eval_count = sum(len(ids) for ids in tokenized["input_ids"])
        except Exception:
            prompt_eval_count = sum(len(t) for t in encode_texts) // 4

        # L2-Normalisierung in FP32 (Praezision gegen FP16-Rundungsfehler)
        embeddings = embeddings.astype(np.float32)
        finite_mask = np.isfinite(embeddings).all(axis=1)
        if not finite_mask.all():
            bad_indices = np.where(~finite_mask)[0].astype(int).tolist()
            logger.error(
                f"[{ollama_name}] {len(bad_indices)} text(s) produced non-finite "
                f"embedding values (indices={bad_indices[:10]})"
            )
            # #772: kein stilles nan_to_num, kein Norm — Fehler hochreichen.
            raise NonFiniteEmbeddingError(bad_indices)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        zero_count = int(np.sum(norms == 0))
        if zero_count > 0:
            logger.warning(
                f"[{ollama_name}] {zero_count} text(s) produced zero embedding "
                "— likely pathological input or tokenizer truncation"
            )
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        return EncodeResult(
            embeddings=embeddings.tolist(),
            prompt_eval_count=prompt_eval_count,
            load_duration_ns=load_duration_ns,
        )

    def _late_embed_sync(
        self,
        ollama_name: str,
        document: str,
        chunks: list[dict],
        input_type: str | None,
    ) -> LateEmbedResult:
        self.assert_worker_thread()
        model, load_duration_ns = self._ensure_model_sync(ollama_name)

        with self.state_lock:
            use_cuda = self.device == "cuda"

        config = OLLAMA_TO_HF[ollama_name]
        prefixed_doc = _apply_prefix([document], config, input_type)[0]
        prefix_char_len = len(prefixed_doc) - len(document)

        # Tokenisierung mit defensive max_length-Behandlung (analog _encode_sync,
        # main.py:329-336). offset_mapping fuer Token-zu-Char-Mapping pro Chunk.
        tok_kwargs: dict = {
            "return_tensors": "pt",
            "truncation": True,
            "return_offsets_mapping": True,
            "padding": False,
        }
        msl = getattr(model, "max_seq_length", None)
        if isinstance(msl, int) and not isinstance(msl, bool) and msl > 0:
            tok_kwargs["max_length"] = msl
        encoding = model.tokenizer(prefixed_doc, **tok_kwargs)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # Forward-Pass auf Token-Ebene. torch.no_grad() zwingend, sonst speichert
        # PyTorch Activations fuer Backward (8192-Token-Forward = 1-2 GiB extra
        # VRAM auf nomic). cuda.amp.autocast nur unter use_cuda — sonst FP32-CPU.
        try:
            transformer = model[0].auto_model
            input_ids = encoding["input_ids"].to(transformer.device)
            attention_mask = encoding["attention_mask"].to(transformer.device)
            with torch.no_grad():
                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        outputs = transformer(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                else:
                    outputs = transformer(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
            token_embeddings = outputs.last_hidden_state[0]
        except torch.cuda.OutOfMemoryError:
            logger.error(
                f"[{ollama_name}] CUDA OOM waehrend Late-Embed Forward — entlade Modell und leere Cache"
            )
            try:
                self._drop_model_sync()
            except Exception:
                logger.exception("Drop-Model nach CUDA-OOM fehlgeschlagen")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise
        except RuntimeError as e:
            if _is_cuda_runtime_error(e):
                logger.error(
                    f"[{ollama_name}] CUDA-RuntimeError ({type(e).__name__}): {e} "
                    "— entlade Modell und leere Cache"
                )
                try:
                    self._drop_model_sync()
                except Exception:
                    logger.exception("Drop-Model nach CUDA-RuntimeError fehlgeschlagen")
                raise CudaRuntimeError(e) from e
            raise

        # Maximaler von der Tokenisierung abgedeckter Char-Offset (in prefixed-
        # Koordinaten). default=0 ist Pflicht — leerer Generator (nur Special-
        # Tokens) wuerde max() mit ValueError abstuerzen.
        max_char_covered_prefixed = max(
            (
                tok_end
                for (tok_start, tok_end) in offset_mapping
                if not (tok_start == 0 and tok_end == 0)
            ),
            default=0,
        )

        results: list = [None] * len(chunks)
        fallback_texts: list[str] = []
        fallback_indices: list[int] = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_start = chunk["char_start"]
            chunk_end = chunk["char_end"]
            shifted_start = chunk_start + prefix_char_len
            shifted_end = chunk_end + prefix_char_len

            # Fallback wenn Chunk hinter dem Token-Window beginnt
            # (Vergleich in prefixed-Koordinaten — F7).
            if shifted_start >= max_char_covered_prefixed:
                fallback_texts.append(chunk_text)
                fallback_indices.append(i)
                continue

            token_indices: list[int] = []
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # Special-Tokens (CLS/SEP/PAD) ueberspringen.
                if tok_start == 0 and tok_end == 0:
                    continue
                # Token komplett im Prefix-Bereich ueberspringen.
                if tok_end <= prefix_char_len:
                    continue
                if tok_start >= shifted_end:
                    break
                if tok_end > shifted_start:
                    token_indices.append(tok_idx)

            if not token_indices:
                fallback_texts.append(chunk_text)
                fallback_indices.append(i)
                continue

            # Mean-Pool ueber Chunk-Tokens. .float() VOR mean: FP16-Sum ueber
            # lange Token-Sequenzen kann an Praezision verlieren; FP32-Cast
            # vor mean vermeidet Akkumulationsfehler.
            chunk_tokens = token_embeddings[token_indices]
            pooled = chunk_tokens.float().mean(dim=0)
            results[i] = pooled.cpu().numpy().astype(np.float32)

        # Fallback-Pfad: ueber model.encode mit applyem Prefix — gleicher
        # Vektorraum wie der Token-Embed-Pfad.
        if fallback_texts:
            fb_encode_texts = _apply_prefix(fallback_texts, config, input_type)
            try:
                if use_cuda:
                    with torch.amp.autocast("cuda"):
                        fb_vecs = model.encode(
                            fb_encode_texts,
                            batch_size=16,
                            convert_to_numpy=True,
                        )
                else:
                    fb_vecs = model.encode(
                        fb_encode_texts,
                        batch_size=16,
                        convert_to_numpy=True,
                    )
            except torch.cuda.OutOfMemoryError:
                logger.error(
                    f"[{ollama_name}] CUDA OOM waehrend Late-Embed Fallback — entlade Modell"
                )
                try:
                    self._drop_model_sync()
                except Exception:
                    logger.exception("Drop-Model nach CUDA-OOM fehlgeschlagen")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise
            except RuntimeError as e:
                if _is_cuda_runtime_error(e):
                    logger.error(
                        f"[{ollama_name}] CUDA-RuntimeError im Late-Embed Fallback "
                        f"({type(e).__name__}): {e}"
                    )
                    try:
                        self._drop_model_sync()
                    except Exception:
                        logger.exception(
                            "Drop-Model nach CUDA-RuntimeError fehlgeschlagen"
                        )
                    raise CudaRuntimeError(e) from e
                raise

            fb_vecs = fb_vecs.astype(np.float32)
            for fb_pos, target_idx in enumerate(fallback_indices):
                results[target_idx] = fb_vecs[fb_pos]

        # Ein np.array aus den finalen Vektoren bauen, dann L2-Norm in FP32.
        embeddings_arr = np.stack(results, axis=0).astype(np.float32)
        finite_mask = np.isfinite(embeddings_arr).all(axis=1)
        if not finite_mask.all():
            bad_indices = np.where(~finite_mask)[0].astype(int).tolist()
            logger.error(
                f"[{ollama_name}] {len(bad_indices)} chunk(s) produced non-finite "
                f"late-embed values (indices={bad_indices[:10]})"
            )
            raise NonFiniteEmbeddingError(bad_indices)

        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        zero_indices = np.where(norms.flatten() == 0)[0]
        if zero_indices.size > 0:
            logger.warning(
                f"[{ollama_name}] {zero_indices.size} late-embed chunk(s) produced "
                f"zero embedding (indices={zero_indices[:10].tolist()})"
            )
        norms = np.where(norms == 0, 1, norms)
        embeddings_arr = embeddings_arr / norms

        # prompt_eval_count: Forward-Pass-Tokens + Fallback-Tokens, beide
        # konsistent mit Prefix und Truncation tokenisiert (analog _encode_sync).
        try:
            doc_token_count = int(input_ids.shape[1])
        except Exception:
            doc_token_count = 0
        fallback_token_count = 0
        if fallback_texts:
            fb_kwargs: dict = {"padding": False, "truncation": True}
            if isinstance(msl, int) and not isinstance(msl, bool) and msl > 0:
                fb_kwargs["max_length"] = msl
            try:
                fb_tokenized = model.tokenizer(
                    _apply_prefix(fallback_texts, config, input_type),
                    **fb_kwargs,
                )
                fallback_token_count = sum(
                    len(ids) for ids in fb_tokenized["input_ids"]
                )
            except Exception:
                fallback_token_count = sum(len(t) for t in fallback_texts) // 4
        prompt_eval_count = doc_token_count + fallback_token_count

        fallback_count = len(fallback_texts)
        if fallback_count > 0:
            ratio = fallback_count / max(1, len(chunks))
            if ratio > LATE_FALLBACK_RATIO_WARN:
                logger.warning(
                    f"[{ollama_name}] late-embed fallback ratio {ratio:.2f} "
                    f"({fallback_count}/{len(chunks)}) — Document moeglicherweise "
                    "laenger als max_seq_length"
                )

        return LateEmbedResult(
            embeddings=embeddings_arr.tolist(),
            prompt_eval_count=prompt_eval_count,
            load_duration_ns=load_duration_ns,
            fallback_count=fallback_count,
        )

    def _drop_model_sync(self) -> None:
        self.assert_worker_thread()
        with self.state_lock:
            if self.model is None:
                self.current_model_name = None
                return
            logger.info(f"Entlade {self.current_model_name}...")
            old_model = self.model
            self.model = None
            self.current_model_name = None
            del old_model
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass


model_manager = ModelManager()
model_worker = SerialModelWorker(model_manager)


@asynccontextmanager
async def lifespan(app):
    model_worker.start()
    try:
        yield
    finally:
        try:
            await model_worker.stop(MODEL_WORKER_SHUTDOWN_TIMEOUT_S)
        except Exception:
            logger.exception("Worker-Stop fehlgeschlagen")

app = FastAPI(title="Embedding Service", lifespan=lifespan)

# #567: Pydantic parst den kompletten Body bevor MAX_TEXTS/MAX_TEXT_LEN
# in der Handler-Logik greifen. Body-Limit vor dem Parsing verhindert OOM
# durch pathologisch grosse Requests. Legit-Maximum: 512 * 32768 Zeichen.
# 4-Byte-UTF-8 worst case = 512 * 32768 * 4 = 64 MiB raw + JSON-Overhead
# (Quotes, Kommas, Struktur). 80 MiB deckt den dokumentierten Vertrag mit
# komfortabler Reserve fuer JSON-Encoding ab (#882).
MAX_BODY_BYTES = 80 * 1024 * 1024


class BodyLimitMiddleware:
    """Pure-ASGI Body-Limit: zaehlt empfangene Bytes inkrementell und bricht
    bei Ueberschreitung mit 413 ab. #622: Vorherige HTTP-Middleware konnte
    nur den Content-Length-Header pruefen — chunked-Encoding oder ein
    fehlender Header hebelte die Pruefung aus, sodass Pydantic den Body
    trotzdem vollstaendig in den Speicher las (OOM-Vektor)."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Early reject via Content-Length, falls Header gesetzt und gueltig
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_body_bytes:
                        await self._send_413(send, value.decode("ascii", "replace"))
                        return
                except ValueError:
                    pass
                break

        bytes_received = 0
        too_large = False

        async def limited_receive():
            nonlocal bytes_received, too_large
            if too_large:
                return {"type": "http.disconnect"}
            message = await receive()
            if message.get("type") == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > self.max_body_bytes:
                    too_large = True
                    return {"type": "http.disconnect"}
            return message

        response_started = False

        async def wrapped_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, wrapped_send)
        except Exception:
            # ClientDisconnect (von Starlette beim http.disconnect-Receive) faengt
            # hier ein — bei too_large schreiben wir die 413, sonst weiterreichen.
            if too_large and not response_started:
                await self._send_413(send, str(bytes_received))
                return
            raise

        if too_large and not response_started:
            await self._send_413(send, str(bytes_received))

    async def _send_413(self, send, size_str: str):
        body = (
            f'{{"error": "Request body zu gross ({size_str} bytes). '
            f'Maximum: {self.max_body_bytes} bytes."}}'
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


app.add_middleware(BodyLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)


@app.post("/api/embed")
async def embed(req: EmbedRequest):
    # #605: Ollama-Compat — total_duration in ns ab Request-Start
    request_start = time.monotonic()
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird vom Embedding-Service nicht unterstuetzt.",
             "available_models": list(OLLAMA_TO_HF.keys())},
            status_code=400,
        )

    texts = req.input if isinstance(req.input, list) else [req.input]

    # #905: input_type-Validierung vor Empty-Short-Circuit, damit input=[] mit
    # ungueltigem input_type konsistent 400 liefert (statt 200 mit embeddings=[]).
    config = OLLAMA_TO_HF[resolved]
    if config.get("prefix_scheme") == "nomic" and req.input_type is not None:
        if req.input_type not in _NOMIC_VALID_INPUT_TYPES:
            return JSONResponse(
                {"error": f"input_type '{req.input_type}' nicht gueltig fuer nomic-v1.5.",
                 "valid_input_types": sorted(_NOMIC_VALID_INPUT_TYPES)},
                status_code=400,
            )
    elif req.input_type is not None:
        logger.warning(f"[{resolved}] input_type='{req.input_type}' wird ignoriert — Modell unterstuetzt kein Prefix-Schema")

    if not texts:
        # #542: resolved statt req.model fuer Konsistenz mit /api/load
        # #605: Ollama-Compat-Felder auch im Empty-Pfad mitsenden
        return {
            "model": resolved,
            "embeddings": [],
            "total_duration": int((time.monotonic() - request_start) * 1_000_000_000),
            "load_duration": 0,
            "prompt_eval_count": 0,
        }

    MAX_TEXTS = 512
    MAX_TEXT_LEN = 32768  # 32KB pro Text
    if len(texts) > MAX_TEXTS:
        return JSONResponse(
            {"error": f"Zu viele Texte ({len(texts)}). Maximum: {MAX_TEXTS}."},
            status_code=400,
        )
    for i, t in enumerate(texts):
        if not t.strip():
            return JSONResponse(
                {"error": f"Text {i} darf nicht leer oder nur Whitespace sein."},
                status_code=400,
            )
        if len(t) > MAX_TEXT_LEN:
            return JSONResponse(
                {"error": f"Text {i} zu lang ({len(t)} Zeichen). Maximum: {MAX_TEXT_LEN}."},
                status_code=400,
            )
        if _SURROGATE_RE.search(t):
            return JSONResponse(
                {"error": f"Text {i} enthaelt ungueltige Unicode-Surrogate (U+D800-U+DFFF)."},
                status_code=400,
            )

    try:
        result = await model_worker.encode(resolved, texts, req.input_type)
    except WorkerStoppedError:
        return JSONResponse(
            {"error": "Embedding-Service stoppt"},
            status_code=503,
        )
    except WorkerQueueFullError:
        return JSONResponse(
            {"error": "Model-Worker Queue voll"},
            status_code=503,
        )
    except NonFiniteEmbeddingError as e:
        return JSONResponse(
            {
                "error": "Encoding produzierte non-finite Embedding-Werte (NaN/Inf).",
                "hint": "FP16/CUDA-Numerikfehler wahrscheinlich. Request mit Backoff wiederholen oder Eingabe reduzieren.",
                "non_finite_indices": e.indices,
            },
            status_code=503,
        )
    except FileNotFoundError as e:
        logger.error(f"[{resolved}] FileNotFoundError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Modell '{resolved}' ist konfiguriert, aber nicht auf dem Server installiert.",
             "hint": "Server-Administrator muss das Modell herunterladen."},
            status_code=500,
        )
    except PermissionError as e:
        logger.error(f"[{resolved}] PermissionError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Cache-Permission-Fehler beim Laden von '{resolved}': {e}",
             "hint": "Server-Administrator muss Lese-/Schreibrechte auf HF_HOME pruefen."},
            status_code=500,
        )
    except OSError as e:
        logger.error(f"[{resolved}] OSError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Filesystem-Fehler beim Laden von '{resolved}' ({type(e).__name__}): {e}",
             "hint": "Server-Administrator muss Disk-Space, FS-Integritaet und HF-Cache pruefen."},
            status_code=500,
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[{resolved}] CUDA OOM: {e}")
        return JSONResponse(
            {"error": f"CUDA out of memory bei '{resolved}'.",
             "hint": "Nicht genug VRAM verfuegbar. Retry mit Backoff oder Eingabe reduzieren."},
            status_code=503,
        )
    except CudaRuntimeError as e:
        logger.error(f"[{resolved}] CUDA-RuntimeError ({e.original_type}): {e}")
        return JSONResponse(
            {"error": f"CUDA-Fehler bei '{resolved}' ({e.original_type}).",
             "detail": str(e),
             "hint": "GPU-Fehler — Modell wurde entladen. Retry mit Backoff."},
            status_code=503,
        )
    except ImportError as e:
        logger.error(f"[{resolved}] ImportError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Fehlende Python-Abhaengigkeit fuer Modell '{resolved}': {e}",
             "hint": "Server-Administrator muss trust_remote_code Abhaengigkeiten installieren (z.B. einops fuer nomic)."},
            status_code=500,
        )
    except Exception as e:
        logger.exception(f"[{resolved}] Unerwarteter Fehler: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Embedding fehlgeschlagen ({type(e).__name__}).",
             "detail": str(e)},
            status_code=500,
        )

    duration_ms = result.load_duration_ns / 1_000_000 if result.load_duration_ns else 0
    logger.info(
        f"[{resolved}] Embedded {len(texts)} texts (load: {duration_ms:.1f}ms)"
    )

    # #542: resolved statt req.model fuer Konsistenz mit /api/load
    # #605: Ollama-Compat-Felder (total_duration/load_duration in ns)
    return {
        "model": resolved,
        "embeddings": result.embeddings,
        "total_duration": int((time.monotonic() - request_start) * 1_000_000_000),
        "load_duration": result.load_duration_ns,
        "prompt_eval_count": result.prompt_eval_count,
    }


@app.post("/api/embed_late")
async def embed_late(req: LateBatchRequest):
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' wird vom Embedding-Service nicht unterstuetzt.",
             "available_models": sorted(OLLAMA_TO_HF.keys())},
            status_code=400,
        )

    config = OLLAMA_TO_HF[resolved]
    # D2: Late Chunking semantisch nur fuer Mean-Pool-Modelle (nomic).
    # CLS-/dense-Pooling-Modelle wuerden Vektoren in fremdem Distributionsraum
    # liefern — Cosine-Similarity zu /api/embed-Output waere unzuverlaessig.
    if config.get("prefix_scheme") != "nomic":
        return JSONResponse(
            {"error": f"Late Chunking ist fuer Modell '{resolved}' nicht aktiv (nicht-Mean-Pooling).",
             "hint": "Re-Ingest nutzt Standard-Embedding via /api/embed.",
             "available_models": ["nomic-embed-text"]},
            status_code=400,
        )

    if len(req.document) > MAX_DOC_LEN:
        return JSONResponse(
            {"error": f"Document zu lang ({len(req.document)} Zeichen). Maximum: {MAX_DOC_LEN}."},
            status_code=400,
        )
    if _SURROGATE_RE.search(req.document):
        return JSONResponse(
            {"error": "Document enthaelt ungueltige Unicode-Surrogate (U+D800-U+DFFF)."},
            status_code=400,
        )
    if len(req.chunks) > MAX_LATE_CHUNKS:
        return JSONResponse(
            {"error": f"Zu viele Chunks ({len(req.chunks)}). Maximum: {MAX_LATE_CHUNKS}."},
            status_code=400,
        )

    if not req.chunks:
        # D9: leere Chunks-Liste -> 200 ohne Worker-Touch.
        return {"model": req.model, "embeddings": []}

    # input_type-Validierung analog /api/embed (main.py:579-587).
    if config.get("prefix_scheme") == "nomic" and req.input_type is not None:
        if req.input_type not in _NOMIC_VALID_INPUT_TYPES:
            return JSONResponse(
                {"error": f"input_type '{req.input_type}' nicht gueltig fuer nomic-v1.5.",
                 "valid_input_types": sorted(_NOMIC_VALID_INPUT_TYPES)},
                status_code=400,
            )
    elif req.input_type is not None:
        logger.warning(
            f"[{resolved}] input_type='{req.input_type}' wird ignoriert — "
            "Modell unterstuetzt kein Prefix-Schema"
        )

    # Per-Chunk-Validierung (D8) — fail-fast 400 mit Index.
    doc_len = len(req.document)
    for i, chunk in enumerate(req.chunks):
        if not (0 <= chunk.char_start <= chunk.char_end <= doc_len):
            return JSONResponse(
                {"error": f"Chunk {i}: ungueltige Bounds char_start={chunk.char_start}, "
                          f"char_end={chunk.char_end}, doc_len={doc_len}"},
                status_code=400,
            )
        actual = req.document[chunk.char_start:chunk.char_end]
        if actual != chunk.text:
            return JSONResponse(
                {"error": f"Chunk {i}: text mismatch — "
                          f"document[{chunk.char_start}:{chunk.char_end}] != chunk.text "
                          f"(got {len(actual)} chars, expected {len(chunk.text)})"},
                status_code=400,
            )
        if _SURROGATE_RE.search(chunk.text):
            return JSONResponse(
                {"error": f"Chunk {i}: text enthaelt ungueltige Unicode-Surrogate."},
                status_code=400,
            )

    chunk_payload = [c.model_dump() for c in req.chunks]
    try:
        result = await model_worker.late_embed(
            resolved, req.document, chunk_payload, req.input_type
        )
    except WorkerStoppedError:
        return JSONResponse(
            {"error": "Embedding-Service stoppt"},
            status_code=503,
        )
    except WorkerQueueFullError:
        return JSONResponse(
            {"error": "Model-Worker Queue voll"},
            status_code=503,
        )
    except NonFiniteEmbeddingError as e:
        # D16: 503, konsistent mit /api/embed.
        return JSONResponse(
            {
                "error": "Encoding produzierte non-finite Embedding-Werte (NaN/Inf).",
                "hint": "FP16/CUDA-Numerikfehler wahrscheinlich. Request mit Backoff wiederholen oder Eingabe reduzieren.",
                "non_finite_indices": e.indices,
            },
            status_code=503,
        )
    except FileNotFoundError as e:
        logger.error(f"[{resolved}] FileNotFoundError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Modell '{resolved}' nicht auf dem Server installiert."},
            status_code=500,
        )
    except PermissionError as e:
        logger.error(f"[{resolved}] PermissionError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Cache-Permission-Fehler beim Laden von '{resolved}': {e}"},
            status_code=500,
        )
    except OSError as e:
        logger.error(f"[{resolved}] OSError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Filesystem-Fehler beim Laden von '{resolved}' ({type(e).__name__}): {e}"},
            status_code=500,
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[{resolved}] CUDA OOM (late-embed): {e}")
        return JSONResponse(
            {"error": f"CUDA out of memory bei '{resolved}'.",
             "hint": "Nicht genug VRAM verfuegbar."},
            status_code=503,
        )
    except CudaRuntimeError as e:
        logger.error(f"[{resolved}] CUDA-RuntimeError (late-embed) ({e.original_type}): {e}")
        return JSONResponse(
            {"error": f"CUDA-Fehler bei '{resolved}' ({e.original_type}).",
             "detail": str(e),
             "hint": "GPU-Fehler — Modell wurde entladen. Retry mit Backoff."},
            status_code=503,
        )
    except ImportError as e:
        logger.error(f"[{resolved}] ImportError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Fehlende Python-Abhaengigkeit fuer Modell '{resolved}': {e}"},
            status_code=500,
        )
    except Exception as e:
        logger.exception(f"[{resolved}] Unerwarteter Fehler (late-embed): {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Late-Embed fehlgeschlagen ({type(e).__name__}).",
             "detail": str(e)},
            status_code=500,
        )

    # D13: Server-Garantie — Length-Equality. Sollte unmoeglich sein, dient
    # als Last-Line-of-Defense gegen Worker-Bugs (kein silent partial).
    if len(result.embeddings) != len(req.chunks):
        logger.error(
            f"[{resolved}] Late-Embed Length-Mismatch: "
            f"got {len(result.embeddings)}, expected {len(req.chunks)} "
            "(Server-Invariante D13 verletzt)"
        )
        return JSONResponse(
            {"error": "Late-Embed Length-Mismatch (Server-Invariante D13 verletzt)."},
            status_code=500,
        )

    return {"model": req.model, "embeddings": result.embeddings}


@app.get("/health")
def health():
    snapshot = model_worker.snapshot()
    status = derive_health_status(snapshot)
    device = snapshot.get("device", "unknown")
    content: dict[str, object] = {
        "status": status,
        "current_model": snapshot.get("current_model"),
        "device": device,
        "precision": "fp16" if device == "cuda" else "fp32" if device == "cpu" else None,
        "available_models": list(OLLAMA_TO_HF.keys()),
        "queue_depth": snapshot.get("queue_depth", 0),
        "worker_busy": snapshot.get("current_job") is not None,
        "current_job": snapshot.get("current_job"),
        "worker_accepting": snapshot.get("worker_accepting", False),
        "worker_thread_alive": snapshot.get("worker_thread_alive", False),
    }
    loading = snapshot.get("loading_model")
    if loading is not None:
        content["loading_model"] = loading
    return JSONResponse(
        content=content,
        status_code=200 if status == "ok" else 503,
    )


@app.post("/api/load")
async def load_model_endpoint(req: LoadRequest):
    """Modell explizit laden/wechseln ohne Embed-Request."""
    resolved = normalize_model_name(req.model)
    if resolved is None:
        return JSONResponse(
            {"error": f"Modell '{req.model}' nicht unterstuetzt.",
             "available_models": list(OLLAMA_TO_HF.keys())},
            status_code=400,
        )
    try:
        await model_worker.load(resolved)
        return {"status": "ok", "model": resolved}
    except WorkerStoppedError:
        return JSONResponse(
            {"error": "Embedding-Service stoppt"},
            status_code=503,
        )
    except WorkerQueueFullError:
        return JSONResponse(
            {"error": "Model-Worker Queue voll"},
            status_code=503,
        )
    except FileNotFoundError as e:
        logger.error(f"[{resolved}] FileNotFoundError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Modell '{resolved}' nicht auf dem Server installiert."},
            status_code=500,
        )
    except PermissionError as e:
        logger.error(f"[{resolved}] PermissionError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Cache-Permission-Fehler beim Laden von '{resolved}': {e}",
             "hint": "Server-Administrator muss Lese-/Schreibrechte auf HF_HOME pruefen."},
            status_code=500,
        )
    except OSError as e:
        logger.error(f"[{resolved}] OSError beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Filesystem-Fehler beim Laden von '{resolved}' ({type(e).__name__}): {e}",
             "hint": "Server-Administrator muss Disk-Space, FS-Integritaet und HF-Cache pruefen."},
            status_code=500,
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[{resolved}] CUDA OOM beim Laden: {e}")
        return JSONResponse(
            {"error": f"CUDA out of memory beim Laden von Modell '{resolved}'.",
             "hint": "Nicht genug VRAM verfuegbar."},
            status_code=503,
        )
    except CudaRuntimeError as e:
        logger.error(f"[{resolved}] CUDA-RuntimeError beim Laden ({e.original_type}): {e}")
        return JSONResponse(
            {"error": f"CUDA-Fehler beim Laden von '{resolved}' ({e.original_type}).",
             "detail": str(e),
             "hint": "GPU-Fehler — Cache geleert, Modell nicht geladen. Retry mit Backoff."},
            status_code=503,
        )
    except ImportError as e:
        logger.error(f"[{resolved}] ImportError beim Laden: {e}")
        return JSONResponse(
            {"error": f"Fehlende Python-Abhaengigkeit fuer Modell '{resolved}': {e}",
             "hint": "Server-Administrator muss trust_remote_code Abhaengigkeiten installieren (z.B. einops fuer nomic)."},
            status_code=500,
        )
    except Exception as e:
        logger.exception(f"[{resolved}] Unerwarteter Fehler beim Laden: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Modell '{resolved}' konnte nicht geladen werden ({type(e).__name__}).",
             "detail": str(e)},
            status_code=500,
        )


@app.get("/api/tags")
def tags():
    """Ollama-kompatibler Endpoint: Listet alle unterstuetzten Modelle."""
    snapshot = model_worker.snapshot()
    # Vor dem ersten Load ist device="unknown" — wir kennen die Praezision
    # noch nicht. Leeres Feld statt F32 luegen, damit Clients auf CUDA-Boxen
    # nicht faelschlich F32 annehmen, obwohl spaeter FP16 geladen wird.
    quant = {"cuda": "F16", "cpu": "F32"}.get(snapshot.get("device"), "")
    return {
        "models": [
            {
                "name": k,
                "model": k,
                "modified_at": _model_modified_at(cfg),
                "size": cfg.get("size", 0),
                # Synthetischer Digest aus hf+revision statt echtem Datei-Hash:
                # Multi-GB-Modelle pro /api/tags-Call zu hashen waere
                # unverhaeltnismaessig. Alle Modelle muessen gepinnt sein.
                "digest": hashlib.sha256(f"{cfg['hf']}@{cfg['revision']}".encode()).hexdigest(),
                "details": {
                    "format": "safetensors",
                    "family": cfg.get("family", "bert"),
                    "families": [cfg.get("family", "bert")],
                    "parameter_size": cfg.get("parameter_size", ""),
                    "quantization_level": quant,
                },
            }
            for k, cfg in OLLAMA_TO_HF.items()
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11436)
