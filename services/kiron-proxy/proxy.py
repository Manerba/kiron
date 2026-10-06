"""Reverse Proxy - Starlette ASGI Proxy fuer Ollama mit Request-Logging."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route, Mount

from request_store import RequestStore, RequestRecord
import vram_lease
from native_admission import NativeRequestOperation, NativeStreamingResponse
from ollama_embedding import EmbeddingRequestError, prepare_request as prepare_ollama_embedding
from kiron_common.gpu_admission import AdmissionError

from kiron_common.embedding_contract import EmbeddingContractError
from kiron_common.embedding_registry import (
    EMBEDDING_REGISTRY,
    attach_show_capabilities,
    build_embedding_registry,
    input_type_error_payload,
    merge_discovery_tags,
    resolve_profile_input_type,
)
from kiron_common.model_catalog import BackendType, ModelEndpoint
from kiron_common.ollama_compat import apply_think_false_bytes

from routing_catalog import (
    PROXY_ROUTING_VIEW,
    ProxyRoute,
    ProxyRoutingView,
)


# Ollama Backend-Adresse
OLLAMA_BACKEND = "http://127.0.0.1:11435"

# Sentence-Transformers Embedding Service
EMBED_BACKEND = "http://127.0.0.1:11436"

# DeBERTa Cross-Encoder Service (Reranking / NLI)
DEBERTA_BACKEND = "http://127.0.0.1:11437"

# Pfade die gestreamt werden (Ollama NDJSON Streaming)
STREAMING_PATHS = {"/api/chat", "/api/generate"}

# Headers die NICHT weitergeleitet werden sollen (Hop-by-Hop)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}


def get_client_ip(request: Request) -> str:
    """Client-IP aus direkter Verbindung extrahieren.

    #598: X-Forwarded-For wird NICHT mehr ausgewertet, weil kiron-proxy direkt
    auf 0.0.0.0:11434 bindet und keinen Trusted Reverse-Proxy davor hat. Ohne
    Trust-Grenze koennte jeder direkte Client seine geloggte IP beliebig auf
    eine gueltige IP setzen und damit Audit-Logs (top_clients, ip-filter)
    verfaelschen. Bei spaeterem Deployment hinter Reverse-Proxy (nginx/Caddy)
    muss hier eine Trusted-Proxy-Liste ergaenzt werden, bevor XFF wieder
    beruecksichtigt wird.
    """
    if request.client:
        return request.client.host
    return "unknown"


def filter_headers(headers, exclude_set: set) -> dict:
    """Headers filtern und Hop-by-Hop Headers entfernen.

    Beachtet RFC 7230: Header-Namen, die im Connection-Header gelistet sind,
    werden zusaetzlich zur statischen Hop-by-Hop-Liste entfernt.
    """
    dynamic_hop_by_hop: set = set()
    for key, value in headers.items():
        if key.lower() == "connection":
            for token in value.split(","):
                token = token.strip().lower()
                if token:
                    dynamic_hop_by_hop.add(token)
    effective_exclude = exclude_set | dynamic_hop_by_hop
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in effective_exclude
    }


def normalize_embed_model(
    name: object,
    endpoint: ModelEndpoint = ModelEndpoint.EMBED,
) -> str:
    """Resolve an exact managed input to its selected backend model name."""

    route = PROXY_ROUTING_VIEW.resolve(name, endpoint)
    return route.backend_model_name if route is not None else ""


def extract_model_from_body(body: bytes) -> str:
    """Modellname aus JSON-Body extrahieren."""
    if not body:
        return ""
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return ""
        model = data.get("model", "")
        return model if isinstance(model, str) else ""
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def _body_json(body: bytes) -> dict | None:
    try:
        data = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _replace_model_in_body(body: bytes, model_name: str) -> bytes:
    """Rewrite an explicitly resolved alias to its concrete backend name."""
    data = _body_json(body)
    if data is None:
        return body
    data["model"] = model_name
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _explicit_verified_cpu_offload(body: bytes) -> bool:
    data = _body_json(body)
    if data is None:
        return False
    options = data.get("options")
    if not isinstance(options, dict):
        return False
    value = options.get("num_gpu")
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == 0
        and vram_lease.num_gpu_zero_effective()
    )


def normalize_deberta_model(
    name: object,
    endpoint: ModelEndpoint = ModelEndpoint.RERANK,
) -> str | None:
    """Resolve only an exact DeBERTa canonical ID or declared alias."""

    route = PROXY_ROUTING_VIEW.resolve(
        name,
        endpoint,
        backend=BackendType.KIRON_DEBERTA,
    )
    return route.backend_model_name if route is not None else None


def merge_model_tags(ollama_payload: object, embedding_payload: object) -> dict:
    """Merge vector rows by explicit canonical group, fail-closed."""
    return merge_discovery_tags(ollama_payload, embedding_payload)


def inject_think_false(body: bytes, path: str) -> bytes:
    """Setzt 'think': false im Request-Body, wenn nicht explizit 'think': true gesetzt ist."""
    return apply_think_false_bytes(body, path)


# Maximaler num_ctx fuer vollstaendiges GPU-Offload (RTX 3060 12 GiB).
# Empirisch getestet: 24576 = 100% GPU, 32768 = 93% GPU (Partial-Offload).
GPU_NUM_CTX_LIMIT = 24576

# Obergrenze fuer den geloggten Response-Body (schuetzt RAM und MEDIUMTEXT-Spalte).
# Die Client-Antwort bleibt unveraendert, nur der DB-Eintrag wird bei Bedarf gekuerzt.
MAX_LOGGED_RESPONSE_BODY = 256 * 1024

# Obergrenze fuer nicht-streamende Backend-Responses. Verhindert, dass eine
# unerwartet grosse Response den Proxy-Speicher sprengt (Gegenstueck zu MAX_BODY_SIZE).
# ColBERT-Antworten enthalten pro Text bis zu 256x128 Float-Werte. Der
# Endpoint akzeptiert bis zu 128 Texte; der Proxy darf diesen Vertrag nicht
# mit einem kleineren Response-Limit brechen.
MAX_RESPONSE_SIZE = 128 * 1024 * 1024


def _truncate_for_log(text: str) -> str:
    if len(text) > MAX_LOGGED_RESPONSE_BODY:
        return text[:MAX_LOGGED_RESPONSE_BODY] + "... [truncated]"
    return text


def check_num_ctx_warning(body: bytes, path: str) -> Optional[str]:
    """Prueft ob num_ctx das GPU-Limit ueberschreitet. Gibt Warning-Text zurueck oder None.

    #611: Bei explizit gesetztem `options.num_gpu=0` (CPU-only Request) wird
    keine Warnung ausgegeben — die Warnung adressiert GPU-Partial-Offload und
    ist fuer reine CPU-Requests irrefuehrend. Suppression nur, wenn das aktuelle
    Ollama-Image num_gpu=0 laut Compat-Report tatsaechlich respektiert
    (num_gpu_zero_effective). Andernfalls bleibt die Warnung sichtbar, weil das
    Backend trotz num_gpu=0 partial-offloaden koennte.
    """
    if path not in STREAMING_PATHS or not body:
        return None
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return None
        options = data.get("options") or {}
        if not isinstance(options, dict):
            return None
        num_gpu = options.get("num_gpu")
        if (
            isinstance(num_gpu, (int, float))
            and not isinstance(num_gpu, bool)
            and num_gpu == 0
            and vram_lease.num_gpu_zero_effective()
        ):
            return None
        num_ctx = options.get("num_ctx", 0)
        if isinstance(num_ctx, int) and not isinstance(num_ctx, bool) and num_ctx > GPU_NUM_CTX_LIMIT:
            return (
                f"num_ctx={num_ctx} ueberschreitet GPU-Limit ({GPU_NUM_CTX_LIMIT}). "
                f"Modell wird teilweise auf CPU ausgelagert, Antwort kann deutlich laenger dauern."
            )
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


def check_streaming_requested(body: bytes, path: str) -> bool:
    """Pruefen ob Streaming angefragt ist (Default in Ollama: true)."""
    if path not in STREAMING_PATHS:
        return False
    if not body:
        # Ollama default ist stream=true fuer chat/generate
        return True
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return True
        # Ollama streamt per Default, nur wenn explizit stream=false nicht.
        # Nicht-bool Werte (null, Strings, Zahlen) als Default behandeln, damit
        # Routing-Entscheidung mit Backend-Verhalten uebereinstimmt.
        stream = data.get("stream", True)
        return stream if isinstance(stream, bool) else True
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Kaputter Body: nicht in den Streaming-Pfad routen. Ollama wuerde 400
        # (single JSON) zurueckgeben, der NDJSON-Handler findet keinen done-Marker
        # und loggt irrefuehrend "Stream ohne done-Marker beendet". Standard-Pfad
        # reicht das 400 transparent durch.
        return False


def create_proxy_app(
    request_store: RequestStore,
    routing_view: ProxyRoutingView = PROXY_ROUTING_VIEW,
) -> Starlette:
    """Erstellt die Starlette ASGI Proxy-App.

    Args:
        request_store: RequestStore-Instanz fuer Request-Logging und Broadcast.

    Returns:
        Starlette-App die als Reverse Proxy fungiert.
    """

    if not isinstance(routing_view, ProxyRoutingView):
        raise TypeError("routing_view must be a ProxyRoutingView")

    # Mandatory startup validator: discovery must never start with an
    # inconsistent, duplicated, or hash-invalid profile registry.
    EMBEDDING_REGISTRY.validate()
    request_embedding_registry = build_embedding_registry(routing_view.catalog)

    # #299: Request-Logging-Fehler duerfen den Proxy-Erfolg nicht kippen.
    # Wenn MariaDB nach dem Start ausfaellt oder update_request wirft, bekommt der
    # Client sonst 500 trotz erfolgreichem Backend. DB-Fehler nur loggen, weiterlaufen.
    async def _safe_log_add(record):
        try:
            await request_store.add_request(record)
        except Exception:
            import logging
            logging.getLogger("proxy").exception("request_store.add_request fehlgeschlagen")

    log_updates = set()

    async def _safe_log_update(request_id, **kwargs):
        async def update():
            try:
                await request_store.update_request(request_id, **kwargs)
            except Exception:
                import logging
                logging.getLogger("proxy").exception("request_store.update_request fehlgeschlagen")
        # A cancelled ASGI scope must not cancel the final database update.
        pending = asyncio.create_task(update())
        log_updates.add(pending)
        pending.add_done_callback(log_updates.discard)
        await asyncio.shield(pending)

    # Langlebiger httpx Client mit Connection-Pooling
    http_client = httpx.AsyncClient(
        base_url=OLLAMA_BACKEND,
        timeout=httpx.Timeout(
            connect=10.0,
            read=600.0,   # Ollama kann bei grossen Modellen lange brauchen
            write=30.0,
            # #441: 30s Queue-Wartezeit wenn Pool voll (max_connections=100 erreicht).
            # Vorher 10s war zu aggressiv — bei kurzen Streams bleibt kaum Zeit, einen
            # Slot freizumachen. 30s gibt realistischen Spielraum fuer Slot-Freigabe,
            # ohne minutenlang zu blockieren (read=600s belegt Slots bei Hang).
            pool=30.0,
        ),
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        ),
        follow_redirects=False,
    )

    # Embedding-Service Client (sentence-transformers)
    embed_client = httpx.AsyncClient(
        base_url=EMBED_BACKEND,
        timeout=httpx.Timeout(
            connect=5.0,
            read=120.0,
            write=30.0,
            pool=10.0,
        ),
        follow_redirects=False,
    )

    # DeBERTa Cross-Encoder Client (Reranking / NLI)
    deberta_client = httpx.AsyncClient(
        base_url=DEBERTA_BACKEND,
        timeout=httpx.Timeout(
            connect=5.0,
            read=300.0,
            write=30.0,
            pool=10.0,
        ),
        follow_redirects=False,
    )
    client_backends = {id(http_client): "ollama", id(embed_client): "kiron_embeddings",
                       id(deberta_client): "kiron_deberta"}

    def _native_request_operation(client, record, target_url, body, gpu_op, method):
        return NativeRequestOperation(backend=client_backends[id(client)], endpoint=target_url,
            model=extract_model_from_body(body) or record.model, body=body, gpu_operation=gpu_op,
            method=method, routing_view=routing_view)

    async def _admission_error_response(error, record, start_time):
        status = 409 if error.code == "resource_conflict" else 503
        detail = " " + json.dumps(error.details, sort_keys=True) if error.details is not None else ""
        response = _gpu_gate_response(vram_lease.GPUGateDecision(
            False, error.code, status_code=status, lease_state="admission"), error.details)
        await _safe_log_update(record.id, state="error", status_code=status,
            duration_ms=round((time.monotonic() - start_time) * 1000, 1),
            error_message=f"GPU admission blocked: {error.code}{detail}",
            response_body=response.body.decode("utf-8"))
        return response

    def _gpu_gate_response(decision: vram_lease.GPUGateDecision, admission_details=None) -> Response:
        exhausted = decision.reason == "resource_exhausted"
        payload = {
            "error": "Nicht genuegend freier GPU-Speicher" if exhausted else "GPU-Operation blockiert",
            "hint": ("Speicher freigeben und die Anfrage erneut senden." if exhausted else
                     "Eine andere GPU-Operation laeuft oder ihr Abschluss ist unklar."),
            "vram_lease": "active",
            "reason": decision.reason,
            "marker_kind": decision.marker_kind,
            "lease_state": decision.lease_state,
        }
        if admission_details is not None and exhausted:
            memory = {key: admission_details[key] for key in (
                "gpu_free_bytes", "gpu_requested_bytes", "headroom_bytes", "gpu_pending_bytes",
                "host_available_bytes", "host_requested_bytes", "host_pending_bytes",
            )}
            memory["gpu_required_bytes"] = (memory["gpu_requested_bytes"] + memory["headroom_bytes"]
                                            + memory["gpu_pending_bytes"])
            memory["host_required_bytes"] = memory["host_requested_bytes"] + memory["host_pending_bytes"]
            gpu_short = memory["gpu_required_bytes"] > memory["gpu_free_bytes"]
            host_short = memory["host_required_bytes"] > memory["host_available_bytes"]
            resource = "GPU- und Host-Speicher" if gpu_short and host_short else (
                "GPU-Speicher" if gpu_short else "Host-Speicher")
            free = memory["gpu_free_bytes"] if gpu_short else memory["host_available_bytes"]
            required = memory["gpu_required_bytes"] if gpu_short else memory["host_required_bytes"]
            pending = memory["gpu_pending_bytes"] if gpu_short else memory["host_pending_bytes"]
            payload.update(
                error=(f"{resource}: Admission vor Inferenzstart abgelehnt. "
                       f"{free / 1024**2:g} MiB frei, {required / 1024**2:g} MiB erforderlich "
                       f"(davon {pending / 1024**2:g} MiB offene Reservierungen)."),
                code="resource_exhausted", stage="admission", inference_started=False,
                vram_lease="not_acquired", memory=memory,
            )
        return Response(
            content=json.dumps(payload),
            status_code=decision.status_code,
            media_type="application/json",
            headers={"X-Kiron-VRAM-Lease": "blocked"},
        )

    def _embedding_backend_error_response(
        model_name: str,
        endpoint: str,
        *,
        status_code: int,
    ) -> Response:
        group = EMBEDDING_REGISTRY.require(model_name)
        profile = EMBEDDING_REGISTRY.default_profile(model_name, endpoint)
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "code": "embedding_backend_incompatible_or_unavailable",
                        "model": group.canonical_model_id,
                        "profile_id": (
                            profile["profile_id"] if profile is not None else None
                        ),
                        "field_path": "/backend",
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            status_code=status_code,
            media_type="application/json",
        )

    def _embedding_registry_error_response(
        error: EmbeddingContractError,
        model_name: str = "embedding-registry",
    ) -> Response:
        group = EMBEDDING_REGISTRY.resolve(model_name)
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "code": error.reason_code,
                        "model": (
                            group.canonical_model_id if group is not None else model_name
                        ),
                        "field_path": error.field_path,
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            status_code=503,
            media_type="application/json",
        )

    async def _ollama_model_loaded_gpu(model_name: str) -> bool:
        if not model_name:
            return False
        canonical = model_name if ":" in model_name else f"{model_name}:latest"
        try:
            resp = await http_client.get("/api/ps")
            if resp.status_code != 200:
                return False
            data = resp.json()
        except httpx.RequestError:
            # Transport-Fehler propagieren: Ollama unerreichbar → der Caller
            # darf KEINEN gpu_service_loading-Marker schreiben, der bis TTL
            # leakt und alle folgenden GPU-Operationen via gpu_gate_decision
            # blockiert. Aufgefangen in _begin_ollama_lazy_operation.
            raise
        except Exception:
            return False
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return False
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name not in (model_name, canonical):
                continue
            vram = item.get("size_vram")
            return isinstance(vram, int) and not isinstance(vram, bool) and vram > 0
        return False

    async def _service_health(client: httpx.AsyncClient) -> dict | None:
        try:
            resp = await client.get("/health")
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if resp.status_code == 200 or (
            resp.status_code == 503 and status in ("no_model", "loading")
        ):
            return data
        return None

    async def _overlay_embedding_tags(response_body: bytes) -> bytes:
        """Build complete canonical discovery or fail the whole request."""
        try:
            ollama_payload = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingContractError(
                "/ollama", "tags response is not valid JSON"
            ) from exc
        try:
            response = await embed_client.get("/api/tags")
        except httpx.RequestError as exc:
            raise EmbeddingContractError(
                "/kiron_embeddings", "tags endpoint is unavailable"
            ) from exc
        if response.status_code != 200:
            raise EmbeddingContractError(
                "/kiron_embeddings",
                f"tags endpoint returned HTTP {response.status_code}",
            )
        try:
            embedding_payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingContractError(
                "/kiron_embeddings", "tags response is not valid JSON"
            ) from exc
        merged = merge_model_tags(ollama_payload, embedding_payload)
        return json.dumps(
            merged,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    async def _overlay_embedding_show(
        response_body: bytes, model_name: str
    ) -> bytes:
        try:
            payload = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingContractError(
                "/show", "response is not valid JSON"
            ) from exc
        result = attach_show_capabilities(payload, model_name)
        return json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    async def _begin_lazy_operation(
        *,
        service_name: str,
        already_loaded,
    ) -> vram_lease.GPUServiceOperation:
        await vram_lease.gpu_service_ops_lock.acquire()
        try:
            hard_decision = await vram_lease.gpu_gate_decision(
                force=True,
                service_name=service_name,
            )
            if not hard_decision.allowed:
                vram_lease.gpu_service_ops_lock.release()
                return vram_lease.GPUServiceOperation(False, hard_decision)
            if await already_loaded():
                return vram_lease.GPUServiceOperation(
                    True,
                    vram_lease.GPUGateDecision(
                        allowed=True,
                        reason="already_loaded",
                        lease_state="inactive",
                        service_name=service_name,
                    ),
                    lock_acquired=True,
                )
            decision = await vram_lease.gpu_gate_decision(
                force=False,
                service_name=service_name,
            )
            if not decision.allowed:
                vram_lease.gpu_service_ops_lock.release()
                return vram_lease.GPUServiceOperation(False, decision)
            try:
                token = vram_lease.write_overlay_marker(
                    "gpu_service_loading",
                    ttl_s=vram_lease.GPU_SERVICE_LOADING_TTL_S,
                )
            except Exception as exc:
                vram_lease.gpu_service_ops_lock.release()
                return vram_lease.GPUServiceOperation(
                    False,
                    vram_lease.GPUGateDecision(
                        allowed=False,
                        reason=f"gpu_service_marker_write_failed:{type(exc).__name__}",
                        status_code=409,
                        lease_state="marker_write_failed",
                        service_name=service_name,
                    ),
                )
            for kind in ("startup", "shutdown"):
                if vram_lease.overlay_marker_active(kind):
                    vram_lease.clear_overlay_marker("gpu_service_loading", token)
                    vram_lease.gpu_service_ops_lock.release()
                    return vram_lease.GPUServiceOperation(
                        False,
                        vram_lease.GPUGateDecision(
                            allowed=False,
                            reason="gpu_overlay_active",
                            status_code=409,
                            marker_kind=kind,
                            lease_state="overlay",
                            service_name=service_name,
                        ),
                    )
            return vram_lease.GPUServiceOperation(
                True,
                decision,
                token=token,
                lock_acquired=True,
            )
        except BaseException:
            vram_lease.gpu_service_ops_lock.release()
            raise

    async def _begin_ollama_lazy_operation(
        request_path: str,
        model_name: str,
        request_body: bytes,
        managed_route: ProxyRoute | None,
    ) -> vram_lease.GPUServiceOperation | None:
        if vram_lease.is_native_unload(request_body, request_path):
            return None
        is_managed_ollama_embedding = (
            managed_route is not None
            and managed_route.endpoint is ModelEndpoint.EMBED
            and managed_route.backend is BackendType.OLLAMA
        )
        if (
            request_path not in STREAMING_PATHS
            and not is_managed_ollama_embedding
        ):
            return None
        if _explicit_verified_cpu_offload(request_body):
            return None
        try:
            return await _begin_lazy_operation(
                service_name="Ollama",
                already_loaded=lambda: _ollama_model_loaded_gpu(model_name),
            )
        except httpx.RequestError:
            # Ollama nicht erreichbar beim Already-Loaded-Check. Lazy-Op-Pfad
            # abbrechen, damit kein gpu_service_loading-Marker geschrieben wird.
            # Der nachfolgende send() schlaegt mit demselben Transport-Fehler
            # fehl und wird vom ConnectError/Timeout-Handler in proxy_handler
            # als 502/503/504 beantwortet.
            return None

    async def _begin_embedding_lazy_operation(normalized_model: str) -> vram_lease.GPUServiceOperation:
        async def _already_loaded() -> bool:
            data = await _service_health(embed_client)
            if not isinstance(data, dict):
                return False
            loaded_models = data.get("loaded_models")
            return (
                data.get("status") == "ok"
                and isinstance(loaded_models, list)
                and normalized_model in loaded_models
                and not data.get("loading_model")
            )

        return await _begin_lazy_operation(
            service_name="Embedding-Service",
            already_loaded=_already_loaded,
        )

    async def _begin_deberta_lazy_operation(resolved_model: str) -> vram_lease.GPUServiceOperation:
        async def _already_loaded() -> bool:
            data = await _service_health(deberta_client)
            return (
                data is not None
                and data.get("status") == "ok"
                and data.get("current_model") == resolved_model
                and not data.get("loading_model")
            )

        return await _begin_lazy_operation(
            service_name="DeBERTa-Service",
            already_loaded=_already_loaded,
        )

    async def proxy_handler(request: Request) -> Response:
        """Universeller Proxy-Handler fuer alle HTTP-Methoden und Pfade."""

        start_time = time.monotonic()
        client_ip = get_client_ip(request)
        path = request.url.path
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        method = request.method
        query_string = str(request.url.query) if request.url.query else ""

        # Request-Body lesen (max 4MB ≈ 1M Token)
        MAX_BODY_SIZE = 4 * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                clen = int(content_length)
            except ValueError:
                # Malformed Content-Length header — fall back to stream length check below
                clen = None
            if clen is not None:
                if clen < 0:
                    return Response(
                        content=json.dumps({"error": "Invalid Content-Length header"}).encode(),
                        status_code=400,
                        media_type="application/json",
                    )
                if clen > MAX_BODY_SIZE:
                    return Response(
                        content=json.dumps({"error": "Request body too large (max 4MB)"}).encode(),
                        status_code=413,
                        media_type="application/json",
                    )
        # Inkrementell lesen, damit chunked/header-lose Requests nicht unbegrenzt in den Speicher geladen werden
        body_chunks: list[bytes] = []
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > MAX_BODY_SIZE:
                return Response(
                    content=json.dumps({"error": "Request body too large (max 4MB)"}).encode(),
                    status_code=413,
                    media_type="application/json",
                )
            body_chunks.append(chunk)
        body = b"".join(body_chunks)

        # think: false injizieren wenn nicht explizit think: true
        body = inject_think_false(body, path)

        # Modellname und Streaming-Info aus Body extrahieren
        model = extract_model_from_body(body)
        requested_model = model
        is_streaming = check_streaming_requested(body, path)

        managed_endpoint = routing_view.endpoint(path)
        managed_route: ProxyRoute | None = None
        if managed_endpoint is not None:
            request_data = _body_json(body)
            supplied_model: object = None
            model_is_supplied = False
            if request_data is not None and "model" in request_data:
                supplied_model = request_data["model"]
                model_is_supplied = True
            if model_is_supplied:
                managed_route = routing_view.resolve(
                    supplied_model,
                    managed_endpoint,
                )
            elif request_data is not None and managed_endpoint in (
                ModelEndpoint.RERANK,
                ModelEndpoint.SCORE,
            ):
                managed_route = routing_view.request_default(managed_endpoint)

            if managed_route is None:
                available_models = list(
                    routing_view.available_models(managed_endpoint)
                )
                if managed_endpoint in (
                    ModelEndpoint.RERANK,
                    ModelEndpoint.SCORE,
                ):
                    error_message = (
                        f"DeBERTa-Modell '{supplied_model}' wird nicht "
                        "unterstuetzt."
                    )
                    log_message = f"Unbekanntes DeBERTa-Modell: {supplied_model}"
                    payload = {
                        "error": error_message,
                        "available_models": available_models,
                    }
                elif managed_endpoint is ModelEndpoint.EMBED_COLBERT:
                    error_message = (
                        f"ColBERT-Modell '{supplied_model}' wird nicht "
                        "unterstuetzt."
                    )
                    log_message = f"Unbekanntes ColBERT-Modell: {supplied_model}"
                    payload = {
                        "error": error_message,
                        "available_models": available_models,
                    }
                elif managed_endpoint is ModelEndpoint.EMBED_LATE:
                    error_message = (
                        f"Late Chunking ist fuer Modell '{supplied_model}' "
                        "nicht aktiv."
                    )
                    log_message = (
                        f"Late-Embed-Modell abgelehnt: {supplied_model}"
                    )
                    payload = {
                        "error": error_message,
                        "hint": (
                            "Re-Ingest nutzt Standard-Embedding via /api/embed."
                        ),
                        "available_models": available_models,
                    }
                else:
                    error_message = (
                        f"Embedding-Modell '{supplied_model}' ist auf dem "
                        "Ollama-Server nicht verfuegbar."
                    )
                    log_message = (
                        f"Unbekanntes Embedding-Modell: {supplied_model}"
                    )
                    payload = {
                        "error": error_message,
                        "available_models": available_models,
                        "hint": (
                            "Bitte den Server-Administrator kontaktieren, um "
                            "das Modell zu installieren."
                        ),
                    }
                request_body_text = body.decode("utf-8", errors="replace")
                record = RequestRecord(
                    client_ip=client_ip,
                    method=method,
                    path=path,
                    model=(
                        supplied_model if type(supplied_model) is str else ""
                    ),
                    request_size=len(body),
                    is_streaming=is_streaming,
                    state="active",
                    request_body=_truncate_for_log(request_body_text),
                )
                await _safe_log_add(record)
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=400,
                    duration_ms=round(duration_ms, 1),
                    error_message=log_message,
                )
                return Response(
                    content=json.dumps(payload),
                    status_code=400,
                    media_type="application/json",
                )

            requested_model = (
                supplied_model if type(supplied_model) is str else ""
            )

        # Vektor-Requestvertrag lokal und vor VRAM-Gate, Modell-Healthcheck oder
        # Backendkontakt erzwingen. Damit kann weder ein Fallback noch ein
        # Lade-/Unloading-Pfad einen fehlenden oder ungueltigen Rollentyp
        # verschlucken.
        if managed_endpoint in (
            ModelEndpoint.EMBED,
            ModelEndpoint.EMBED_LATE,
            ModelEndpoint.EMBED_COLBERT,
        ):
            request_data = _body_json(body)
            if request_data is not None:
                input_decision = resolve_profile_input_type(
                    managed_route.canonical_model_id,
                    managed_endpoint.value,
                    request_data.get("input_type"),
                    registry=request_embedding_registry,
                )
                if input_decision is not None and not input_decision.accepted:
                    request_body_text = body.decode("utf-8", errors="replace")
                    record = RequestRecord(
                        client_ip=client_ip,
                        method=method,
                        path=path,
                        model=requested_model,
                        request_size=len(body),
                        is_streaming=is_streaming,
                        state="active",
                        request_body=_truncate_for_log(request_body_text),
                    )
                    await _safe_log_add(record)
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=400,
                        duration_ms=round(duration_ms, 1),
                        error_message=input_decision.error_code,
                    )
                    return Response(
                        content=json.dumps(
                            input_type_error_payload(input_decision),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8"),
                        status_code=400,
                        media_type="application/json",
                    )

                if managed_route.backend is BackendType.OLLAMA and input_decision is not None:
                    profile = request_embedding_registry.profile(managed_route.profile_id)
                    try:
                        prepared = prepare_ollama_embedding(
                            request_data, profile, input_decision.canonical_input_type,
                        )
                    except EmbeddingRequestError as exc:
                        record = RequestRecord(
                            client_ip=client_ip, method=method, path=path,
                            model=requested_model, request_size=len(body),
                            is_streaming=is_streaming, state="active",
                            request_body=_truncate_for_log(body.decode("utf-8", errors="replace")),
                        )
                        await _safe_log_add(record)
                        await _safe_log_update(
                            record.id, state="error", status_code=400,
                            duration_ms=round((time.monotonic() - start_time) * 1000, 1),
                            error_message=exc.payload["error"]["code"],
                        )
                        return Response(
                            content=json.dumps(exc.payload), status_code=400,
                            media_type="application/json",
                        )
                    body = json.dumps(prepared, ensure_ascii=False).encode("utf-8")

        if managed_route is not None:
            body = _replace_model_in_body(
                body,
                managed_route.backend_model_name,
            )
            model = managed_route.backend_model_name

        # VRAM-Lease Intercept (#285): Docling beansprucht GPU waehrend
        # STARTING/RUNNING+active/STOPPED_DIRTY. Bytes-Helper entscheidet
        # anhand Policy (block/force_cpu/pass) ueber Outcome.
        lease_outcome = vram_lease.LeaseOutcome.PASS
        if method == "POST":
            body, lease_outcome = await vram_lease.apply_bytes(
                body,
                path,
                model,
                routing_view,
            )
        # #611: Warnung NACH apply_bytes, damit FORCE_CPU (num_gpu=0) den
        # irrefuehrenden GPU-Partial-Offload-Hinweis korrekt unterdrueckt.
        perf_warning = check_num_ctx_warning(body, path)
        request_size = len(body)

        # Request-Body als Text fuer Logging (analog response_body auf MAX_LOGGED_RESPONSE_BODY begrenzt,
        # damit MEDIUMTEXT nicht unbegrenzt waechst und sensible Inhalte im Body nicht in voller Laenge persistiert werden)
        request_body_text = body.decode("utf-8", errors="replace") if body else ""

        # RequestRecord erstellen und registrieren
        record = RequestRecord(
            client_ip=client_ip,
            method=method,
            path=path,
            model=requested_model,
            request_size=request_size,
            is_streaming=is_streaming,
            state="active",
            request_body=_truncate_for_log(request_body_text),
        )
        await _safe_log_add(record)

        # VRAM-Lease BLOCK: Request wird nicht an Ollama weitergeleitet.
        if lease_outcome == vram_lease.LeaseOutcome.BLOCK:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=503,
                duration_ms=round(duration_ms, 1),
                error_message="GPU-Gate aktiv — Request blockiert",
            )
            return Response(
                content=json.dumps({
                    "error": "GPU-Operation blockiert",
                    "hint": "Request spaeter wiederholen oder num_gpu=0 "
                            "explizit setzen.",
                    "vram_lease": "active",
                }),
                status_code=503,
                media_type="application/json",
                headers={"X-Kiron-VRAM-Lease": "blocked"},
            )

        # Response-Header fuer FORCE_CPU-Outcome bauen (wird an die
        # _handle_*-Helfer durchgereicht).
        lease_header_value: Optional[str] = None
        if lease_outcome == vram_lease.LeaseOutcome.FORCE_CPU:
            lease_header_value = "force-cpu"

        # Request-Headers vorbereiten (Hop-by-Hop + content-length entfernen)
        forward_headers = filter_headers(
            request.headers, HOP_BY_HOP_HEADERS | {"content-length"}
        )

        # Ziel-URL zusammenbauen
        target_url = path
        if query_string:
            target_url = f"{path}?{query_string}"

        # DeBERTa-Requests: route was resolved and rewritten before any gate.
        if (
            managed_route is not None
            and managed_route.backend is BackendType.KIRON_DEBERTA
        ):
            resolved_deberta = managed_route.backend_model_name
            gpu_op = await _begin_deberta_lazy_operation(resolved_deberta) if method == "POST" else None
            if gpu_op is not None and not gpu_op.allowed:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=gpu_op.decision.status_code,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"GPU-Gate blockiert: {gpu_op.decision.reason}",
                )
                return _gpu_gate_response(gpu_op.decision)
            try:
                return await _handle_standard_request(
                    http_client=deberta_client,
                    request_store=request_store,
                    record=record,
                    method=method,
                    target_url=target_url,
                    headers=forward_headers,
                    body=body,
                    start_time=start_time,
                    lease_header=lease_header_value,
                    gpu_op=gpu_op,
                )
            except httpx.ConnectError:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=503,
                    duration_ms=round(duration_ms, 1),
                    error_message="DeBERTa Cross-Encoder Service nicht erreichbar",
                )
                return Response(
                    content=json.dumps({
                        "error": "Service Unavailable",
                        "message": "DeBERTa Cross-Encoder Service nicht erreichbar. Laeuft kiron-deberta?",
                    }),
                    status_code=503,
                    media_type="application/json",
                )
            except httpx.PoolTimeout:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=503,
                    duration_ms=round(duration_ms, 1),
                    error_message="Proxy-Pool erschoepft (DeBERTa-Client)",
                )
                return Response(
                    content=json.dumps({
                        "error": "Service Unavailable",
                        "message": "Proxy-Verbindungspool zum DeBERTa-Service erschoepft. Spaeter erneut versuchen.",
                    }),
                    status_code=503,
                    media_type="application/json",
                )
            except httpx.TimeoutException:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=504,
                    duration_ms=round(duration_ms, 1),
                    error_message="DeBERTa Cross-Encoder Timeout",
                )
                return Response(
                    content=json.dumps({
                        "error": "Gateway Timeout",
                        "message": "DeBERTa Cross-Encoder hat nicht rechtzeitig geantwortet.",
                    }),
                    status_code=504,
                    media_type="application/json",
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=502,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Kommunikation mit DeBERTa fehlgeschlagen: {e}",
                )
                return Response(
                    content=json.dumps({
                        "error": "Bad Gateway",
                        "message": "Kommunikation mit DeBERTa Cross-Encoder fehlgeschlagen.",
                    }),
                    status_code=502,
                    media_type="application/json",
                )
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=500,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Interner Proxy-Fehler (DeBERTa): {e}",
                )
                return Response(
                    content=json.dumps({
                        "error": "Internal Server Error",
                        "message": "Ein interner Fehler ist im Proxy aufgetreten.",
                    }),
                    status_code=500,
                    media_type="application/json",
                )

        # ColBERT-Requests: always kiron-embeddings, never Ollama.
        if managed_endpoint is ModelEndpoint.EMBED_COLBERT:
            if managed_route is None:  # pragma: no cover - rejected above
                raise AssertionError("managed ColBERT route disappeared")
            gpu_op = await _begin_embedding_lazy_operation(
                managed_route.backend_model_name
            ) if method == "POST" else None
            if gpu_op is not None and not gpu_op.allowed:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=gpu_op.decision.status_code,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"GPU-Gate blockiert: {gpu_op.decision.reason}",
                )
                return _gpu_gate_response(gpu_op.decision)
            try:
                return await _handle_standard_request(
                    http_client=embed_client,
                    request_store=request_store,
                    record=record,
                    method=method,
                    target_url=target_url,
                    headers=forward_headers,
                    body=body,
                    start_time=start_time,
                    lease_header=lease_header_value,
                    gpu_op=gpu_op,
                )
            except httpx.ConnectError:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=503,
                    duration_ms=round(duration_ms, 1),
                    error_message="ColBERT Embedding-Service nicht erreichbar",
                )
                return _embedding_backend_error_response(
                    model,
                    path,
                    status_code=503,
                )
            except httpx.PoolTimeout:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=503,
                    duration_ms=round(duration_ms, 1),
                    error_message="Proxy-Pool erschoepft (ColBERT-Embedding-Client)",
                )
                return _embedding_backend_error_response(
                    model,
                    path,
                    status_code=503,
                )
            except httpx.TimeoutException:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=502,
                    duration_ms=round(duration_ms, 1),
                    error_message="ColBERT Embedding-Service Timeout",
                )
                return _embedding_backend_error_response(
                    model,
                    path,
                    status_code=502,
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=502,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Kommunikation mit ColBERT Embedding-Service fehlgeschlagen: {e}",
                )
                return _embedding_backend_error_response(
                    model,
                    path,
                    status_code=502,
                )
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    record.id,
                    state="error",
                    status_code=500,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Interner Proxy-Fehler (ColBERT): {e}",
                )
                return Response(
                    content=json.dumps({
                        "error": "Internal Server Error",
                        "message": "Ein interner Fehler ist im Proxy aufgetreten.",
                    }),
                    status_code=500,
                    media_type="application/json",
                )

        # Dense/Late requests use the exact Catalog-selected deployment.
        if managed_endpoint in (
            ModelEndpoint.EMBED,
            ModelEndpoint.EMBED_LATE,
        ):
            if managed_route is None:  # pragma: no cover - rejected above
                raise AssertionError("managed embedding route disappeared")
            normalized = managed_route.backend_model_name

            if managed_route.backend is BackendType.KIRON_EMBEDDINGS:
                # Kleine Modelle → Embedding-Service (schnell, FP16)
                gpu_op = await _begin_embedding_lazy_operation(normalized) if method == "POST" else None
                if gpu_op is not None and not gpu_op.allowed:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=gpu_op.decision.status_code,
                        duration_ms=round(duration_ms, 1),
                        error_message=f"GPU-Gate blockiert: {gpu_op.decision.reason}",
                    )
                    return _gpu_gate_response(gpu_op.decision)
                try:
                    return await _handle_standard_request(
                        http_client=embed_client,
                        request_store=request_store,
                        record=record,
                        method=method,
                        target_url=target_url,
                        headers=forward_headers,
                        body=body,
                        start_time=start_time,
                        lease_header=lease_header_value,
                        gpu_op=gpu_op,
                    )
                except httpx.ConnectError:
                    # A same-named Ollama model is not a compatibility proof.
                    # No current cross-backend pair has two verified equal IDs.
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=502,
                        duration_ms=round(duration_ms, 1),
                        error_message=(
                            "Embedding-Service nicht erreichbar; "
                            "kein verifizierter kompatibler Fallback"
                        ),
                    )
                    return _embedding_backend_error_response(
                        model,
                        path,
                        status_code=502,
                    )
                except httpx.PoolTimeout:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=503,
                        duration_ms=round(duration_ms, 1),
                        error_message="Proxy-Pool erschoepft (Embedding-Client)",
                    )
                    return _embedding_backend_error_response(
                        model,
                        path,
                        status_code=503,
                    )
                except httpx.TimeoutException:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=502,
                        duration_ms=round(duration_ms, 1),
                        error_message="Embedding-Service Timeout",
                    )
                    return _embedding_backend_error_response(
                        model,
                        path,
                        status_code=502,
                    )
                except httpx.RequestError as e:
                    # #465: Sonstige Netzwerk-/Protokollfehler (nicht ConnectError,
                    # nicht Timeout) -> 502 Bad Gateway ohne Fallback.
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=502,
                        duration_ms=round(duration_ms, 1),
                        error_message=f"Embedding-Service Netzwerkfehler: {e}",
                    )
                    return _embedding_backend_error_response(
                        model,
                        path,
                        status_code=502,
                    )
                except Exception as e:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=500,
                        duration_ms=round(duration_ms, 1),
                        error_message=f"Interner Proxy-Fehler (Embed): {e}",
                    )
                    return Response(
                        content=json.dumps({
                            "error": "Internal Server Error",
                            "message": "Ein interner Fehler ist im Proxy aufgetreten.",
                        }),
                        status_code=500,
                        media_type="application/json",
                    )

            elif managed_route.backend is BackendType.OLLAMA:
                # Catalog-managed GGUF deployment: fall through to Ollama.
                if managed_endpoint is not ModelEndpoint.EMBED:
                    raise AssertionError(
                        "Ollama deployment reached a non-dense endpoint"
                    )
            else:
                raise AssertionError(
                    "validated embedding route has an unsupported backend"
                )

        # /api/show follows the explicit canonical group. Service-owned groups
        # (including ColBERT) come from kiron-embeddings; Ollama-only groups are
        # augmented after their native show response.
        show_group = EMBEDDING_REGISTRY.resolve(model) if path == "/api/show" else None
        if (
            show_group is not None
            and show_group.service_model is None
            and show_group.ollama_model is not None
        ):
            body = _replace_model_in_body(body, show_group.ollama_model)
            model = show_group.ollama_model
        standard_client = http_client
        backend_name = "Ollama"
        backend_pool_label = "Ollama-Client"
        backend_unavailable_message = (
            "Ollama Backend nicht erreichbar. Laeuft Ollama auf Port 11435?"
        )
        backend_pool_message = (
            "Proxy-Verbindungspool zu Ollama erschoepft. Spaeter erneut versuchen."
        )
        backend_request_error_message = "Kommunikation mit Ollama Backend fehlgeschlagen."
        if show_group is not None and show_group.service_model is not None:
            standard_client = embed_client
            backend_name = "Embedding-Service"
            backend_pool_label = "Embedding-Client"
            backend_unavailable_message = "Embedding-Service nicht erreichbar."
            backend_pool_message = (
                "Proxy-Verbindungspool zum Embedding-Service erschoepft. "
                "Spaeter erneut versuchen."
            )
            backend_request_error_message = (
                "Kommunikation mit Embedding-Service fehlgeschlagen."
            )

        gpu_op = None
        if standard_client is http_client and method == "POST":
            gpu_op = await _begin_ollama_lazy_operation(
                path,
                model,
                body,
                managed_route,
            )
        if gpu_op is not None and not gpu_op.allowed:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=gpu_op.decision.status_code,
                duration_ms=round(duration_ms, 1),
                error_message=f"GPU-Gate blockiert: {gpu_op.decision.reason}",
            )
            return _gpu_gate_response(gpu_op.decision)

        standard_response_transform = None
        if method == "GET" and path == "/api/tags":
            standard_response_transform = _overlay_embedding_tags
        elif show_group is not None:
            standard_response_transform = lambda response_body: _overlay_embedding_show(
                response_body, model
            )

        try:
            if is_streaming:
                return await _handle_streaming_request(
                    http_client=http_client,
                    request_store=request_store,
                    record=record,
                    method=method,
                    target_url=target_url,
                    headers=forward_headers,
                    body=body,
                    start_time=start_time,
                    perf_warning=perf_warning,
                    lease_header=lease_header_value,
                    gpu_op=gpu_op,
                )
            else:
                return await _handle_standard_request(
                    http_client=standard_client,
                    request_store=request_store,
                    record=record,
                    method=method,
                    target_url=target_url,
                    headers=forward_headers,
                    body=body,
                    start_time=start_time,
                    perf_warning=perf_warning,
                    lease_header=lease_header_value,
                    gpu_op=gpu_op,
                    response_transform=standard_response_transform,
                )

        except EmbeddingContractError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=503,
                duration_ms=round(duration_ms, 1),
                error_message=str(e),
            )
            return _embedding_registry_error_response(e, model or "embedding-registry")

        except httpx.ConnectError as e:
            # Ollama nicht erreichbar
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Verbindung zu {backend_name} fehlgeschlagen: {e}",
            )
            return Response(
                content=json.dumps({
                    "error": "Bad Gateway",
                    "message": backend_unavailable_message,
                }),
                status_code=502,
                media_type="application/json",
            )

        except httpx.PoolTimeout:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=503,
                duration_ms=round(duration_ms, 1),
                error_message=f"Proxy-Pool erschoepft ({backend_pool_label})",
            )
            return Response(
                content=json.dumps({
                    "error": "Service Unavailable",
                    "message": backend_pool_message,
                }),
                status_code=503,
                media_type="application/json",
            )

        except httpx.TimeoutException as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=504,
                duration_ms=round(duration_ms, 1),
                error_message=f"Timeout bei {backend_name}-Anfrage: {e}",
            )
            return Response(
                content=json.dumps({
                    "error": "Gateway Timeout",
                    "message": f"{backend_name} hat nicht rechtzeitig geantwortet.",
                }),
                status_code=504,
                media_type="application/json",
            )

        except httpx.RequestError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Kommunikation mit {backend_name} fehlgeschlagen: {e}",
            )
            return Response(
                content=json.dumps({
                    "error": "Bad Gateway",
                    "message": backend_request_error_message,
                }),
                status_code=502,
                media_type="application/json",
            )

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                record.id,
                state="error",
                status_code=500,
                duration_ms=round(duration_ms, 1),
                error_message=f"Interner Proxy-Fehler: {e}",
            )
            return Response(
                content=json.dumps({
                    "error": "Internal Server Error",
                    "message": "Ein interner Fehler ist im Proxy aufgetreten.",
                }),
                status_code=500,
                media_type="application/json",
            )

    async def _handle_standard_request(
        http_client: httpx.AsyncClient,
        request_store: RequestStore,
        record: RequestRecord,
        method: str,
        target_url: str,
        headers: dict,
        body: bytes,
        start_time: float,
        perf_warning: Optional[str] = None,
        lease_header: Optional[str] = None,
        gpu_op: vram_lease.GPUServiceOperation | None = None,
        response_transform: Callable[[bytes], Awaitable[bytes]] | None = None,
    ) -> Response:
        """Nicht-Streaming Request verarbeiten und komplett weiterleiten."""

        backend_response = None
        operation = _native_request_operation(http_client, record, target_url, body, gpu_op, method)
        gpu_marker_no_start = False
        try:
            try:
                backend_request = http_client.build_request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )
                backend_response = await operation.send(http_client, backend_request)
            except AdmissionError as exc:
                return await _admission_error_response(exc, record, start_time)
            except (httpx.ConnectError, httpx.PoolTimeout):
                gpu_marker_no_start = True
                raise
            except Exception:
                if "backend_request" not in locals():
                    gpu_marker_no_start = True
                raise

            # Content-Length vorab pruefen, falls das Backend die Groesse bereits signalisiert.
            content_length_hdr = backend_response.headers.get("content-length")
            if content_length_hdr:
                try:
                    if int(content_length_hdr) > MAX_RESPONSE_SIZE:
                        if gpu_op is not None and gpu_op.token is not None:
                            gpu_op.clear_marker = True
                        duration_ms = (time.monotonic() - start_time) * 1000
                        await _safe_log_update(
                            record.id,
                            state="error",
                            status_code=502,
                            duration_ms=round(duration_ms, 1),
                            error_message=f"Backend-Response ueberschreitet Limit ({content_length_hdr} > {MAX_RESPONSE_SIZE} Bytes)",
                        )
                        return Response(
                            content=json.dumps({"error": f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)"}).encode(),
                            status_code=502,
                            media_type="application/json",
                        )
                except ValueError:
                    pass

            # Body inkrementell einlesen und bei Ueberschreitung abbrechen.
            body_chunks: list[bytes] = []
            total_size = 0
            async for chunk in operation.chunks(backend_response):
                total_size += len(chunk)
                if total_size > MAX_RESPONSE_SIZE:
                    if gpu_op is not None and gpu_op.token is not None:
                        gpu_op.clear_marker = True
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=502,
                        duration_ms=round(duration_ms, 1),
                        error_message=f"Backend-Response ueberschreitet Limit (> {MAX_RESPONSE_SIZE} Bytes beim Streamen)",
                    )
                    return Response(
                        content=json.dumps({"error": f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)"}).encode(),
                        status_code=502,
                        media_type="application/json",
                    )
                body_chunks.append(chunk)
            response_body = b"".join(body_chunks)
            if (
                response_transform is not None
                and 200 <= backend_response.status_code < 300
            ):
                response_body = await response_transform(response_body)
                if len(response_body) > MAX_RESPONSE_SIZE:
                    duration_ms = (time.monotonic() - start_time) * 1000
                    await _safe_log_update(
                        record.id,
                        state="error",
                        status_code=502,
                        duration_ms=round(duration_ms, 1),
                        error_message="Transformierte Backend-Response ueberschreitet Limit",
                    )
                    return Response(
                        content=json.dumps({
                            "error": (
                                "Backend response too large "
                                f"(max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)"
                            )
                        }).encode(),
                        status_code=502,
                        media_type="application/json",
                    )
            if gpu_op is not None and gpu_op.token is not None:
                if 200 <= backend_response.status_code < 300:
                    try:
                        parsed = json.loads(response_body) if response_body else {}
                        if not (isinstance(parsed, dict) and parsed.get("error")):
                            gpu_op.clear_marker = True
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # 2xx mit nicht-parsbarem Body ist Erfolg — Marker freigeben.
                        gpu_op.clear_marker = True
                elif 400 <= backend_response.status_code < 500:
                    gpu_op.clear_marker = True
        finally:
            await operation.close()

        duration_ms = (time.monotonic() - start_time) * 1000
        response_size = len(response_body)

        # Token-Count, Response-Text und ggf. Backend-Error extrahieren
        tokens_generated = 0
        response_text = ""
        backend_error_msg: Optional[str] = None
        if response_body:
            try:
                resp_data = json.loads(response_body)
                # Nicht-Dict JSON (Liste, Zahl, String, null) graceful behandeln,
                # sonst AttributeError und 500 statt Backend-Body an den Client.
                if isinstance(resp_data, dict):
                    err_field = resp_data.get("error")
                    if isinstance(err_field, str) and err_field:
                        backend_error_msg = err_field
                    eval_count = resp_data.get("eval_count")
                    if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                        tokens_generated = eval_count
                    # Response-Text extrahieren (/api/chat -> message.content, /api/generate -> response)
                    message = resp_data.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        response_text = message["content"]
                    elif isinstance(resp_data.get("response"), str):
                        response_text = resp_data["response"]
                    else:
                        response_text = response_body.decode("utf-8", errors="replace")
                else:
                    response_text = response_body.decode("utf-8", errors="replace")
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_text = response_body.decode("utf-8", errors="replace")

        # #1042: HTTP 4xx/5xx oder Backend-Error-Feld als state=error
        # klassifizieren — analog Streaming-Pfad und openai_api.py.
        backend_status = backend_response.status_code
        log_state = "error" if (backend_status >= 400 or backend_error_msg) else "completed"
        update_kwargs: dict = {
            "state": log_state,
            "status_code": backend_status,
            "duration_ms": round(duration_ms, 1),
            "response_size": response_size,
            "tokens_generated": tokens_generated,
            "response_body": _truncate_for_log(response_text),
        }
        if log_state == "error":
            update_kwargs["error_message"] = (
                backend_error_msg
                if backend_error_msg
                else f"Backend returned HTTP {backend_status}"
            )

        await _safe_log_update(record.id, **update_kwargs)

        # Response-Headers vorbereiten (Hop-by-Hop entfernen).
        # content-encoding/content-length entfernen, da httpx den Body bereits dekodiert hat
        # und die Originallaenge nicht mehr zum weitergereichten Body passt.
        response_headers = filter_headers(
            dict(backend_response.headers),
            HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"},
        )
        if perf_warning:
            response_headers["X-Performance-Warning"] = perf_warning
        if lease_header:
            response_headers["X-Kiron-VRAM-Lease"] = lease_header

        return Response(
            content=response_body,
            status_code=backend_response.status_code,
            headers=response_headers,
        )

    async def _handle_streaming_request(
        http_client: httpx.AsyncClient,
        request_store: RequestStore,
        record: RequestRecord,
        method: str,
        target_url: str,
        headers: dict,
        body: bytes,
        start_time: float,
        perf_warning: Optional[str] = None,
        lease_header: Optional[str] = None,
        gpu_op: vram_lease.GPUServiceOperation | None = None,
    ) -> StreamingResponse:
        """Streaming Request (NDJSON) verarbeiten und Chunks durchleiten."""

        operation = _native_request_operation(http_client, record, target_url, body, gpu_op, method)
        gpu_marker_no_start = False
        try:
            try:
                backend_request = http_client.build_request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )

                backend_response = await operation.send(http_client, backend_request)
            except AdmissionError as exc:
                await operation.close()
                return await _admission_error_response(exc, record, start_time)
            except (httpx.ConnectError, httpx.PoolTimeout):
                gpu_marker_no_start = True
                raise
            except Exception:
                if "backend_request" not in locals():
                    gpu_marker_no_start = True
                raise
        except BaseException:
            await operation.close()
            raise

        try:
            # Response-Headers vorbereiten.
            # content-encoding entfernen, da httpx via aiter_bytes bereits dekodiert;
            # content-length ist beim Streamen ohnehin ungueltig.
            response_headers = filter_headers(
                dict(backend_response.headers),
                HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"},
            )
            if perf_warning:
                response_headers["X-Performance-Warning"] = perf_warning
            if lease_header:
                response_headers["X-Kiron-VRAM-Lease"] = lease_header

            # Status-Code fuer Record setzen
            await _safe_log_update(
                record.id,
                status_code=backend_response.status_code,
            )
        except BaseException:
            # Connection-Leak verhindern wenn Exception vor StreamingResponse-Rueckgabe auftritt
            await operation.close()
            raise

        stream_started = False

        async def finalize_unstarted():
            if not stream_started:
                await _safe_log_update(record.id, state="error",
                    duration_ms=round((time.monotonic() - start_time) * 1000, 1),
                    error_message="Client disconnected before response body")

        async def stream_generator():
            """Generator der NDJSON-Chunks durchleitet und dabei parst."""
            nonlocal stream_started
            stream_started = True
            total_response_size = 0
            tokens_generated = 0
            last_line_data = None
            buffer = b""
            response_text_parts = []
            response_text_len = 0
            stream_corrupted = False

            # RAM-Schutz: Backend ohne Newlines darf Proxy nicht ins OOM treiben
            MAX_BUFFER_BYTES = 10 * 1024 * 1024

            def _accumulate_log_text(text: str) -> None:
                # RAM-Schutz: MAX_LOGGED_RESPONSE_BODY begrenzt den gespeicherten Text,
                # nicht den an den Client weitergereichten Stream.
                # +1 Zeichen Overshoot, damit _truncate_for_log den "[truncated]"-Marker
                # noch setzt (analog zum OpenAI-Pfad in openai_api.py).
                nonlocal response_text_len
                if response_text_len > MAX_LOGGED_RESPONSE_BODY:
                    return
                remaining = MAX_LOGGED_RESPONSE_BODY - response_text_len
                if len(text) > remaining:
                    response_text_parts.append(text[:remaining + 1])
                    response_text_len = MAX_LOGGED_RESPONSE_BODY + 1
                else:
                    response_text_parts.append(text)
                    response_text_len += len(text)

            try:
                async for chunk in operation.chunks(backend_response):
                    # NDJSON parsen: Chunks koennen unvollstaendige Zeilen enthalten.
                    # Buffer-Limit VOR yield pruefen, sonst leakt der ueberschreitende
                    # Chunk noch an den Client bevor der RuntimeError feuert.
                    buffer += chunk
                    if len(buffer) > MAX_BUFFER_BYTES:
                        raise RuntimeError(
                            f"Backend response buffer exceeded {MAX_BUFFER_BYTES} bytes without newline"
                        )
                    total_response_size += len(chunk)
                    yield chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            line_data = json.loads(line)
                            # Nicht ueberschreiben, falls die letzte vollstaendige Zeile bereits
                            # den done-Marker enthielt: ein nachgelagertes Fragment ohne done
                            # darf den Stream-Abschluss nicht aushebeln (analog Rest-Buffer-Guard
                            # unten, #1021).
                            if not (isinstance(last_line_data, dict) and last_line_data.get("done")):
                                last_line_data = line_data
                            # Nicht-Dict NDJSON-Zeilen (Listen, Zahlen, Strings, null)
                            # ignorieren, sonst AttributeError/TypeError im Stream.
                            if not isinstance(line_data, dict):
                                stream_corrupted = True
                                continue
                            # Text-Chunks akkumulieren
                            # /api/chat: message.content, /api/generate: response
                            message = line_data.get("message")
                            if isinstance(message, dict):
                                content = message.get("content")
                                if isinstance(content, str):
                                    _accumulate_log_text(content)
                            elif "response" in line_data:
                                resp = line_data["response"]
                                if isinstance(resp, str):
                                    _accumulate_log_text(resp)
                        except (json.JSONDecodeError, UnicodeDecodeError) as parse_err:
                            stream_corrupted = True
                            import logging
                            logging.getLogger("proxy").warning(
                                "NDJSON-Zeile nicht parsbar (Token-Count kann unvollstaendig sein): %s", parse_err
                            )

                # Restlichen Buffer verarbeiten
                if buffer.strip():
                    try:
                        rest_data = json.loads(buffer.strip())
                        if isinstance(rest_data, dict):
                            message = rest_data.get("message")
                            if isinstance(message, dict):
                                content = message.get("content")
                                if isinstance(content, str):
                                    _accumulate_log_text(content)
                            elif "response" in rest_data:
                                resp = rest_data["response"]
                                if isinstance(resp, str):
                                    _accumulate_log_text(resp)
                            # Nicht ueberschreiben, falls die letzte vollstaendige Zeile bereits
                            # den done-Marker enthielt: ein nachgelagertes Fragment ohne done
                            # darf den Stream-Abschluss nicht aushebeln.
                            if not (isinstance(last_line_data, dict) and last_line_data.get("done")):
                                last_line_data = rest_data
                        else:
                            stream_corrupted = True
                    except (json.JSONDecodeError, UnicodeDecodeError) as parse_err:
                        stream_corrupted = True
                        import logging
                        logging.getLogger("proxy").warning(
                            "NDJSON-Rest-Buffer nicht parsbar: %s", parse_err
                        )

                # Token-Count aus der letzten Zeile extrahieren
                # Die letzte NDJSON-Zeile mit "done": true enthaelt die Statistiken
                done_seen = bool(isinstance(last_line_data, dict) and last_line_data.get("done", False))
                # Ollama kann Error-Chunks mit done=true senden ({"error":"...","done":true}).
                # done allein darf nicht als Erfolg gewertet werden — sonst landet ein Backend-
                # Fehler als state=completed und der gpu_service_loading-Marker wird faelschlich
                # geloescht. Truthy-Check: error koennte null/"" sein. Pattern analog openai_api.py.
                err_field = last_line_data.get("error") if isinstance(last_line_data, dict) else None
                backend_error = bool(err_field)
                if done_seen and not backend_error:
                    eval_count = last_line_data.get("eval_count")
                    if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                        tokens_generated = eval_count

                response_text = "".join(response_text_parts)

                duration_ms = (time.monotonic() - start_time) * 1000
                if backend_error:
                    await _safe_log_update(
                        record.id,
                        state="error",
                        duration_ms=round(duration_ms, 1),
                        response_size=total_response_size,
                        response_body=_truncate_for_log(response_text),
                        error_message=f"Backend-Fehler: {err_field}",
                    )
                elif done_seen:
                    update = {
                        "state": "warning" if stream_corrupted else "completed",
                        "duration_ms": round(duration_ms, 1),
                        "response_size": total_response_size,
                        "tokens_generated": tokens_generated,
                        "response_body": _truncate_for_log(response_text),
                    }
                    if stream_corrupted:
                        update["error_message"] = "Stream enthielt korrupte NDJSON-Zeilen"
                    await _safe_log_update(record.id, **update)
                else:
                    await _safe_log_update(
                        record.id,
                        state="error",
                        duration_ms=round(duration_ms, 1),
                        response_size=total_response_size,
                        response_body=_truncate_for_log(response_text),
                        error_message="Stream ohne done-Marker beendet",
                    )

            except (asyncio.CancelledError, GeneratorExit):
                # Client-Disconnect - Record darf nicht auf state=active stehen bleiben (#183)
                duration_ms = (time.monotonic() - start_time) * 1000
                try:
                    # #899 (analog Exception-Handler unten): Wenn der Backend-Stream
                    # bereits mit done=true terminiert hat, war die Anfrage faktisch
                    # erfolgreich. Ein Client-Disconnect danach darf den Erfolg nicht
                    # zu state=error umschreiben und tokens_generated nicht verlieren.
                    done_after_terminal = (
                        isinstance(last_line_data, dict)
                        and last_line_data.get("done") is True
                        and not last_line_data.get("error")
                    )
                    if done_after_terminal:
                        eval_count = last_line_data.get("eval_count")
                        if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                            tokens_generated = eval_count
                        update = {
                            "state": "warning" if stream_corrupted else "completed",
                            "duration_ms": round(duration_ms, 1),
                            "response_size": total_response_size,
                            "tokens_generated": tokens_generated,
                            "response_body": _truncate_for_log("".join(response_text_parts)),
                        }
                        if stream_corrupted:
                            update["error_message"] = "Stream enthielt korrupte NDJSON-Zeilen"
                        await _safe_log_update(record.id, **update)
                    else:
                        await _safe_log_update(
                            record.id,
                            state="error",
                            duration_ms=round(duration_ms, 1),
                            response_size=total_response_size,
                            response_body=_truncate_for_log("".join(response_text_parts)),
                            error_message="Client disconnected",
                        )
                except Exception:
                    pass
                raise
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                # #899: Wenn aiter_bytes NACH dem terminalen done=true wirft (z.B.
                # waehrend Connection-Close), hat der Client bereits einen vollstaendigen
                # Stream gesehen. Den Erfolg nicht durch einen Transportfehler ueberschreiben
                # und keinen doppelten done-Chunk yielden.
                done_after_terminal = (
                    isinstance(last_line_data, dict)
                    and last_line_data.get("done") is True
                    and not last_line_data.get("error")
                )
                if done_after_terminal:
                    eval_count = last_line_data.get("eval_count")
                    if isinstance(eval_count, int) and not isinstance(eval_count, bool):
                        tokens_generated = eval_count
                    update = {
                        "state": "warning" if stream_corrupted else "completed",
                        "duration_ms": round(duration_ms, 1),
                        "response_size": total_response_size,
                        "tokens_generated": tokens_generated,
                        "response_body": _truncate_for_log("".join(response_text_parts)),
                    }
                    if stream_corrupted:
                        update["error_message"] = "Stream enthielt korrupte NDJSON-Zeilen"
                    await _safe_log_update(record.id, **update)
                else:
                    await _safe_log_update(
                        record.id,
                        state="error",
                        duration_ms=round(duration_ms, 1),
                        response_size=total_response_size,
                        response_body=_truncate_for_log("".join(response_text_parts)),
                        error_message=f"Streaming-Fehler: {e}",
                    )
                    # Client ueber Abbruch informieren — Status/Header sind bereits gesendet,
                    # daher Fehler als NDJSON-Chunk im Ollama-Format ausgeben.
                    # Vorher b"\n", damit eine ggf. offene Backend-Zeile geschlossen wird
                    # und der Fehler-Frame nicht als korrupte Mischzeile beim Client landet.
                    try:
                        yield b"\n"
                        yield (json.dumps({"error": "Streaming-Fehler", "done": True}) + "\n").encode("utf-8")
                    except Exception:
                        pass

            finally:
                await operation.close()

        try:
            return NativeStreamingResponse(
                content=stream_generator(),
                operation=operation,
                finalize=finalize_unstarted,
                status_code=backend_response.status_code,
                headers=response_headers,
            )
        except BaseException:
            await operation.close()
            raise

    # Catch-All Route fuer alle Pfade und Methoden
    async def catch_all(request: Request) -> Response:
        return await proxy_handler(request)

    app = Starlette(
        routes=[
            # #557: Starlettes path-Converter matched auch / (leerer Pfad), separate Route("/") redundant.
            Route("/{path:path}", endpoint=catch_all, methods=[
                "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS",
            ]),
        ],
    )

    # httpx Clients beim Shutdown schliessen
    async def shutdown_clients():
        import logging
        logger = logging.getLogger("proxy")
        if log_updates:
            await asyncio.wait(tuple(log_updates), timeout=10)
        for client_name, client in (
            ("http_client", http_client),
            ("embed_client", embed_client),
            ("deberta_client", deberta_client),
        ):
            try:
                await client.aclose()
            except Exception as exc:
                logger.warning("Fehler beim Schliessen von %s: %s", client_name, exc)

    app.add_event_handler("shutdown", shutdown_clients)

    # Store als App-State verfuegbar machen
    app.state.request_store = request_store
    app.state.http_client = http_client

    return app
