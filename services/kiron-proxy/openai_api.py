"""OpenAI-kompatibler API-Server - Starlette ASGI mit Auth + Format-Translation."""

import asyncio
import base64
import binascii
import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pymysql
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from request_store import RequestStore, RequestRecord
from api_key_store import ApiKeyStore
import vram_lease

_COMMON_SRC = Path(__file__).resolve().parents[1] / "kiron-common"
if _COMMON_SRC.exists() and str(_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(_COMMON_SRC))
try:
    from kiron_common.ollama_compat import ensure_think_false_dict
except ImportError:  # pragma: no cover - half-upgraded venv fallback
    def ensure_think_false_dict(payload, path):
        if payload.get("think") is not True:
            payload["think"] = False
        return payload


OLLAMA_BACKEND = "http://127.0.0.1:11435"

MAX_LOGGED_RESPONSE_BODY = 256 * 1024
MAX_RESPONSE_SIZE = 16 * 1024 * 1024

# Konsistent mit app.MODEL_NAME_PATTERN: Ollama-Tag-Konvention. Lehnt
# Control-Chars (Newlines, Null-Bytes) und alles ausserhalb der Whitelist
# ab — sonst landen ungueltige Werte in Logs (Log-Injection) und im
# Backend-Request.
_MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]+(/[a-zA-Z0-9._:-]+)*$')

_CHAT_COMPLETION_REQUEST_KEYS = frozenset({
    "model",
    "messages",
    "stream",
    "stream_options",
    "n",
    "tool_choice",
    "tools",
    "functions",
    "function_call",
    "response_format",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "frequency_penalty",
    "seed",
})


def _truncate_for_log(text: str) -> str:
    if len(text) > MAX_LOGGED_RESPONSE_BODY:
        return text[:MAX_LOGGED_RESPONSE_BODY] + "... [truncated]"
    return text


def _openai_error(message: str, error_type: str = "invalid_request_error", code: str = "invalid_api_key", status: int = 401, headers: dict[str, str] | None = None) -> JSONResponse:
    """OpenAI-kompatibles Error-Response."""
    if status == 401:
        # RFC 7235: 401 MUST include WWW-Authenticate. Bearer-Scheme triggert
        # keine Browser-Basic-Auth-Dialoge, bleibt OpenAI-kompatibel.
        merged = dict(headers) if headers else {}
        merged.setdefault("WWW-Authenticate", 'Bearer realm="api"')
        headers = merged
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": code}},
        status_code=status,
        headers=headers,
    )


def _sse_error_event(message: str, error_type: str = "server_error", code: str = "backend_error") -> str:
    """OpenAI-style Error-Event fuer SSE-Stream (#485).

    Schicke eine `data: {"error": {...}}`-Zeile bevor finish_reason=null + [DONE].
    Robuste Clients parsen das und zeigen dem User den Fehlergrund. Strict-Clients,
    die nur ChatCompletionChunk-Schema kennen, ignorieren das Event (kein choices-Feld).
    """
    payload = {"error": {"message": message, "type": error_type, "code": code}}
    return f"data: {json.dumps(payload)}\n\n"


def _chatcmpl_id() -> str:
    # 16 hex chars = 64 bit Entropie. 8 chars (32 bit) hatte Birthday-Bound bei
    # ~65k IDs und produzierte Kollisionen in Log-Korrelation bei aktivem Traffic.
    return "chatcmpl-" + uuid.uuid4().hex[:16]


def _unsupported_value_is_unset(name: str, value) -> bool:
    """True nur fuer explizite No-op-Werte unsupported OpenAI-Parameter."""
    if value is None:
        return True
    if name in ("tools", "functions"):
        return isinstance(value, list) and not value
    if name == "logit_bias":
        return isinstance(value, dict) and not value
    if name == "logprobs":
        return value is False
    if name == "top_logprobs":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
    if name == "presence_penalty":
        # OpenAI-Default ist 0 — viele Standard-Clients (inkl. aelterer
        # openai-python-Versionen) serialisieren den Default explizit. Analog
        # zu top_logprobs=0 als unset behandeln, sonst lehnt der Endpoint
        # legitime Default-Konfigurationen mit 400 ab.
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
    if name == "response_format":
        # #892: {"type": "text"} ist OpenAI-Default-Plain-Text-Modus und damit
        # ein No-op gegenueber dem Endpoint-Default. Aktive Modi
        # (json_object / json_schema) bleiben abgelehnt.
        return isinstance(value, dict) and value.get("type") == "text"
    return False


def _unsupported_request_parameter(data: dict) -> str | None:
    """Return the first request key this endpoint neither supports nor validates."""
    for key in data:
        if key not in _CHAT_COMPLETION_REQUEST_KEYS:
            return str(key)
    return None


# Bekannte Ollama done_reason -> OpenAI finish_reason. Unbekannte Werte
# werden geloggt und auf "stop" gemappt, um valides OpenAI-Schema zu liefern.
_OLLAMA_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
}


def _map_finish_reason(done_reason) -> str:
    if done_reason is None:
        return "stop"
    mapped = _OLLAMA_FINISH_REASON_MAP.get(done_reason)
    if mapped is None:
        logging.getLogger(__name__).warning(
            "Unknown Ollama done_reason=%r, mapping to 'stop'", done_reason
        )
        return "stop"
    return mapped


# #570: Request-Logging-Fehler duerfen den Chat-Erfolg nicht kippen — analog
# proxy.py:210-222. DB-/RequestStore-Fehler nur loggen, Client-Antwort unveraendert
# durchreichen, sonst maskiert ein RequestStore-Fehler die schon erzeugte 200-
# Antwort oder die eigentliche Backend-Error-Response.
async def _safe_log_add(request_store: RequestStore, record: RequestRecord) -> None:
    try:
        await request_store.add_request(record)
    except Exception:
        logging.getLogger(__name__).exception("request_store.add_request fehlgeschlagen")


async def _safe_log_update(request_store: RequestStore, request_id, **kwargs) -> None:
    try:
        await request_store.update_request(request_id, **kwargs)
    except Exception:
        logging.getLogger(__name__).exception("request_store.update_request fehlgeschlagen")


def _translate_messages_for_ollama(messages: list) -> tuple[list | None, str | None]:
    """OpenAI -> Ollama: Multimodal-Content (Array) in content+images uebersetzen.

    OpenAI: content = [{"type":"text","text":...}, {"type":"image_url","image_url":{"url":"data:..."}}]
    Ollama: content = "..." (string), images = ["base64...", ...] (auf Message-Ebene)

    Role-Mapping (#590): `developer` -> `system`, `function` -> `tool`. Ollama
    kennt weder `developer` noch `function`; ohne Mapping wuerde die Legacy-
    Rolle `function` mit 400/500 am Backend enden, und `developer` (OpenAI
    2024) wuerde schon im Validator mit 400 rausfallen.

    Returns (translated_messages, None) bei Erfolg oder (None, error_message) bei Fehler.
    """
    translated = []
    valid_roles = {"system", "user", "assistant", "tool", "developer", "function"}
    role_mapping = {"developer": "system", "function": "tool"}
    for msg in messages:
        if not isinstance(msg, dict):
            return None, "messages[] must contain only objects"
        # #420: role-Pflichtfeld und Wertebereich validieren, sonst reicht der
        # Proxy den Message-Stream an Ollama durch, das mit 400/500 antwortet.
        role = msg.get("role")
        if not isinstance(role, str) or role not in valid_roles:
            return None, f"message.role must be one of {sorted(valid_roles)}"
        mapped_role = role_mapping.get(role, role)
        # #673: tool-Messages ohne tool_call_id sind per OpenAI-Spec ungueltig;
        # ohne Pruefung wuerde Ollama mit 400/500 antworten.
        if mapped_role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                return None, "message.tool_call_id is required and must be a non-empty string for role=tool"
        content = msg.get("content")
        if content is None:
            # #951: assistant-Messages mit tool_calls duerfen per OpenAI-Spec
            # content=null haben (Conversation-Replays mit historischen
            # tool_calls). Auf "" mappen, damit Ollama die Message akzeptiert.
            # system/user/tool bleiben strikt - dort waere content=null
            # tatsaechlich ungueltig.
            if mapped_role != "assistant":
                return None, "message.content is required and cannot be null"
            new_msg = {k: v for k, v in msg.items() if k not in ("content", "images")}
            new_msg["role"] = mapped_role
            new_msg["content"] = ""
            translated.append(new_msg)
            continue
        if isinstance(content, str):
            # #799: images auf Message-Ebene ist Ollama-Native-Syntax und umgeht
            # die image_url-Validierung (base64/URL-Checks). Konsistent mit
            # Array-Pfad unten droppen, sonst koennen Clients ungeprueftes
            # Bild-Payload an Ollama smuggeln.
            if mapped_role != role or "images" in msg:
                new_msg = {k: v for k, v in msg.items() if k != "images"}
                new_msg["role"] = mapped_role
                translated.append(new_msg)
            else:
                translated.append(msg)
            continue
        if not isinstance(content, list):
            return None, "message.content must be a string or list of content parts"

        text_parts = []
        images = []
        for part in content:
            # #591: nicht-dict Parts und unbekannte Typen nicht mehr stumm
            # droppen — stummer Drop laesst unvollstaendige Prompts an Ollama
            # durch (z.B. audio + text => nur text, oder alles Audio => leer).
            if not isinstance(part, dict):
                return None, "content parts must be objects with a 'type' field"
            ptype = part.get("type")
            if ptype == "text":
                # Strikte Validierung: fehlendes/null/nicht-string text-Feld darf
                # nicht still zu "" werden (sonst gehen 0/false/dict-Werte unbemerkt
                # verloren oder werden via str() zu Prompt-Muell).
                text_val = part.get("text")
                if not isinstance(text_val, str):
                    return None, "content part of type 'text' requires a 'text' string"
                text_parts.append(text_val)
            elif ptype == "image_url":
                url_obj = part.get("image_url")
                if isinstance(url_obj, dict):
                    url = url_obj.get("url", "")
                elif isinstance(url_obj, str):
                    url = url_obj
                else:
                    url = ""
                if not isinstance(url, str) or not url:
                    return None, "image_url.url is required and must be a string"
                if url.startswith("data:image/") and ";base64," in url:
                    b64_payload = url.split(";base64,", 1)[1]
                    if not b64_payload:
                        return None, "image_url base64 payload is empty"
                    # #421: Pre-Validation - sonst wird ungueltiger Base64 an Ollama
                    # durchgereicht und dort als 500 gemeldet. b64decode(validate=True)
                    # wirft binascii.Error bei illegal chars oder falschem Padding.
                    try:
                        base64.b64decode(b64_payload, validate=True)
                    except binascii.Error:
                        return None, "image_url base64 payload is invalid"
                    images.append(b64_payload)
                else:
                    return None, "image_url must be a data:image/<type>;base64 URI (http/https URLs and non-image MIME types not supported)"
            else:
                return None, (
                    f"unsupported content part type: {ptype!r} "
                    f"(only 'text' and 'image_url' supported)"
                )

        # #825: Leeres content-Array (oder Array ohne text/image-Parts) wuerde
        # sonst zu '\n'.join([]) = '' werden und einen leeren Prompt an Ollama
        # durchreichen. OpenAI-Spec verlangt mind. einen Part im Array.
        if not text_parts and not images:
            return None, "message.content array must contain at least one part"

        new_msg = {k: v for k, v in msg.items() if k not in ("content", "images", "role")}
        new_msg["role"] = mapped_role
        new_msg["content"] = "\n".join(text_parts)
        if images:
            new_msg["images"] = images
        translated.append(new_msg)
    return translated, None


def create_openai_api_app(request_store: RequestStore, api_key_store: ApiKeyStore) -> Starlette:
    """Erstellt die OpenAI-kompatible API-App.

    Args:
        request_store: RequestStore fuer Request-Logging.
        api_key_store: ApiKeyStore fuer Bearer-Token-Authentifizierung.

    Returns:
        Starlette-App auf Port 11440.
    """

    http_client = httpx.AsyncClient(
        base_url=OLLAMA_BACKEND,
        timeout=httpx.Timeout(
            connect=10.0,
            read=600.0,
            write=30.0,
            pool=10.0,
        ),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
        ),
        follow_redirects=False,
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
        except Exception:
            return False
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return False
        for item in models:
            if not isinstance(item, dict):
                # Einzelner malformed Eintrag darf nicht den ganzen Geladen-Check
                # invalidieren — nachfolgende Eintraege koennen das Modell trotzdem
                # korrekt anzeigen. Sonst wuerde unnoetig ein GPU-Lease-Erwerb
                # getriggert obwohl das Modell bereits warm ist.
                continue
            if item.get("name") not in (model_name, canonical):
                continue
            vram = item.get("size_vram")
            return isinstance(vram, int) and not isinstance(vram, bool) and vram > 0
        return False

    async def _begin_openai_lazy_operation(model_name: str) -> vram_lease.GPUServiceOperation:
        await vram_lease.gpu_service_ops_lock.acquire()
        try:
            hard_decision = await vram_lease.gpu_gate_decision(
                force=True,
                service_name="Ollama",
            )
            if not hard_decision.allowed:
                vram_lease.gpu_service_ops_lock.release()
                return vram_lease.GPUServiceOperation(False, hard_decision)
            if await _ollama_model_loaded_gpu(model_name):
                return vram_lease.GPUServiceOperation(
                    True,
                    vram_lease.GPUGateDecision(
                        allowed=True,
                        reason="already_loaded",
                        lease_state="inactive",
                        service_name="Ollama",
                    ),
                    lock_acquired=True,
                )
            decision = await vram_lease.gpu_gate_decision(
                force=False,
                service_name="Ollama",
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
                        service_name="Ollama",
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
                            service_name="Ollama",
                        ),
                    )
            return vram_lease.GPUServiceOperation(
                True,
                decision,
                token=token,
                lock_acquired=True,
            )
        except Exception:
            vram_lease.gpu_service_ops_lock.release()
            raise

    async def _authenticate(request: Request) -> dict | Response | None:
        """Bearer-Token aus Authorization-Header pruefen.

        Returns:
            dict: gueltiger Key
            None: fehlender/ungueltiger Token (Caller -> 401)
            Response: DB-/Serverfehler (Caller gibt Response direkt zurueck)
        """
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None
        token = auth_header[7:].strip()
        if not token:
            return None
        try:
            return await asyncio.to_thread(api_key_store.validate_key, token)
        except pymysql.Error:
            import logging
            logging.getLogger(__name__).exception("API key validation failed: database error")
            return _openai_error(
                "Authentication service temporarily unavailable",
                error_type="server_error",
                code="auth_backend_unavailable",
                status=503,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception("API key validation failed")
            return _openai_error(
                "Internal authentication error",
                error_type="server_error",
                code="auth_internal_error",
                status=500,
            )

    async def models_handler(request: Request) -> Response:
        """GET /v1/models - Modelle im OpenAI-Format."""
        api_key = await _authenticate(request)
        if isinstance(api_key, Response):
            return api_key
        if not api_key:
            return _openai_error("Invalid API key")

        try:
            backend_request = http_client.build_request(method="GET", url="/api/tags")
            resp = await http_client.send(backend_request, stream=True)
        except httpx.TimeoutException as e:
            # #610: konsistent zu Chat-Pfaden — Timeout ist 504, nicht 502.
            return _openai_error(
                f"Backend timeout: {e}",
                error_type="server_error",
                code="timeout",
                status=504,
            )
        except Exception as e:
            return _openai_error(
                f"Backend error: {e}",
                error_type="server_error",
                code="backend_error",
                status=502,
            )

        try:
            if resp.status_code >= 400:
                return _openai_error(
                    f"Backend returned HTTP {resp.status_code}",
                    error_type="server_error",
                    code="backend_error",
                    status=502,
                )

            content_length_hdr = resp.headers.get("content-length")
            if content_length_hdr:
                try:
                    if int(content_length_hdr) > MAX_RESPONSE_SIZE:
                        return _openai_error(
                            f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)",
                            error_type="server_error",
                            code="backend_error",
                            status=502,
                        )
                except ValueError:
                    pass

            body_chunks: list[bytes] = []
            total_size = 0
            try:
                async for chunk in resp.aiter_bytes():
                    total_size += len(chunk)
                    if total_size > MAX_RESPONSE_SIZE:
                        return _openai_error(
                            f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)",
                            error_type="server_error",
                            code="backend_error",
                            status=502,
                        )
                    body_chunks.append(chunk)
            except httpx.TimeoutException as e:
                return _openai_error(
                    f"Backend timeout: {e}",
                    error_type="server_error",
                    code="timeout",
                    status=504,
                )
            except Exception as e:
                return _openai_error(
                    f"Backend error: {e}",
                    error_type="server_error",
                    code="backend_error",
                    status=502,
                )
            response_body = b"".join(body_chunks)
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass

        try:
            ollama_data = json.loads(response_body)
        except Exception as e:
            return _openai_error(
                f"Backend error: {e}",
                error_type="server_error",
                code="backend_error",
                status=502,
            )

        if not isinstance(ollama_data, dict):
            return _openai_error(
                "Invalid backend response",
                error_type="server_error",
                code="backend_error",
                status=502,
            )

        models_raw = ollama_data.get("models")
        if not isinstance(models_raw, list):
            return _openai_error(
                "Invalid backend response",
                error_type="server_error",
                code="backend_error",
                status=502,
            )
        models = []
        for m in models_raw:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not name or not isinstance(name, str):
                continue
            created = int(time.time())
            modified_at = m.get("modified_at")
            if modified_at:
                try:
                    created = int(datetime.fromisoformat(modified_at.replace("Z", "+00:00")).timestamp())
                except (ValueError, AttributeError, TypeError):
                    created = int(time.time())
            models.append({
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "ollama",
            })

        return JSONResponse({
            "object": "list",
            "data": models,
        })

    async def chat_completions_handler(request: Request) -> Response:
        """POST /v1/chat/completions - Chat im OpenAI-Format."""
        api_key = await _authenticate(request)
        if isinstance(api_key, Response):
            return api_key
        if not api_key:
            return _openai_error("Invalid API key")

        start_time = time.monotonic()

        # 1M Token ≈ 4MB Body-Limit
        max_body_size = 4 * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                clen = int(content_length)
            except (ValueError, TypeError):
                return _openai_error(
                    "Invalid Content-Length header",
                    code="invalid_request",
                    status=400,
                )
            if clen < 0:
                return _openai_error(
                    "Invalid Content-Length header",
                    code="invalid_request",
                    status=400,
                )
            if clen > max_body_size:
                return _openai_error(
                    "Request body too large (max 4MB)",
                    code="invalid_request",
                    status=413,
                )

        # Stream mit laufendem Zaehler: verhindert DoS bei chunked / fehlendem content-length
        body_chunks: list[bytes] = []
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > max_body_size:
                return _openai_error(
                    "Request body too large (max 4MB)",
                    code="invalid_request",
                    status=413,
                )
            body_chunks.append(chunk)
        body = b"".join(body_chunks)

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _openai_error(
                "Invalid JSON in request body",
                code="invalid_request",
                status=400,
            )

        if not isinstance(data, dict):
            return _openai_error(
                "Request body must be a JSON object",
                code="invalid_request",
                status=400,
            )

        unsupported_parameter = _unsupported_request_parameter(data)
        if unsupported_parameter is not None:
            return _openai_error(
                f"{unsupported_parameter} is not supported by this endpoint",
                code="unsupported_parameter",
                status=400,
            )

        model = data.get("model", "")
        messages = data.get("messages", [])
        # #654: Optionale Felder mit explizitem null wie nicht-gesetzt behandeln
        # (OpenAI-Standard-Clients serialisieren unset oft als null statt zu omitten).
        stream = data.get("stream", False)
        if stream is None:
            stream = False

        if not isinstance(model, str) or not model.strip():
            return _openai_error(
                "model is required and must be a non-empty string",
                code="invalid_request",
                status=400,
            )

        if not _MODEL_NAME_PATTERN.fullmatch(model):
            return _openai_error(
                "model contains invalid characters",
                code="invalid_request",
                status=400,
            )

        if not isinstance(messages, list) or not messages:
            return _openai_error(
                "messages is required and must be a non-empty list",
                code="invalid_request",
                status=400,
            )

        if not isinstance(stream, bool):
            return _openai_error(
                "stream must be a boolean",
                code="invalid_request",
                status=400,
            )

        include_usage = False
        stream_options = data.get("stream_options")
        if stream_options is not None:
            if not isinstance(stream_options, dict):
                return _openai_error(
                    "stream_options must be an object",
                    code="invalid_request",
                    status=400,
                )
            if not stream:
                return _openai_error(
                    "stream_options is only allowed when stream=true",
                    code="invalid_request",
                    status=400,
                )
            include_usage_val = stream_options.get("include_usage", False)
            if not isinstance(include_usage_val, bool):
                return _openai_error(
                    "stream_options.include_usage must be a boolean",
                    code="invalid_request",
                    status=400,
                )
            include_usage = include_usage_val

        # OpenAI -> Ollama Request-Translation
        translated_messages, translation_error = _translate_messages_for_ollama(messages)
        if translation_error is not None:
            return _openai_error(
                translation_error,
                code="invalid_request",
                status=400,
            )

        # #390: n>1 von Ollama nicht unterstuetzt (single-completion backend).
        # #654: explizites null wie nicht-gesetzt behandeln.
        if "n" in data and data["n"] is not None:
            n_val = data["n"]
            if not isinstance(n_val, int) or isinstance(n_val, bool) or n_val < 1:
                return _openai_error("n must be a positive integer", code="invalid_request", status=400)
            if n_val > 1:
                return _openai_error(
                    "n > 1 is not supported (Ollama backend limitation)",
                    code="unsupported_parameter",
                    status=400,
                )

        # tool_choice="none" bedeutet explizit "keine Tools verwenden" — entspricht
        # dem Default-Verhalten dieses Endpoints, also akzeptieren statt abzulehnen.
        if data.get("tool_choice") == "none":
            data.pop("tool_choice")

        # #554: tool_choice als Dict (auch leer {}) ist Tool-Spezifikation —
        # muss vor der generischen Schleife geprueft werden, da {} in (None, [], {}).
        if "tool_choice" in data and isinstance(data["tool_choice"], dict):
            return _openai_error(
                "tool_choice is not supported by this endpoint",
                code="unsupported_parameter",
                status=400,
            )

        # #391/#424/#594: Function-Calling / JSON-Mode / response_format /
        # presence_penalty / logprobs nicht unterstuetzt. Statt stillschweigend
        # zu ignorieren (Clients erwarten tool_calls bzw. choices[].logprobs im
        # Response) lehnen wir den Request explizit ab. logprobs=false /
        # top_logprobs=0 gilt als "nicht gesetzt" und wird akzeptiert.
        for unsupported in ("tools", "functions", "function_call", "tool_choice",
                            "response_format", "presence_penalty", "logit_bias",
                            "logprobs", "top_logprobs"):
            if (
                unsupported in data
                and not _unsupported_value_is_unset(unsupported, data[unsupported])
            ):
                return _openai_error(
                    f"{unsupported} is not supported by this endpoint",
                    code="unsupported_parameter",
                    status=400,
                )

        ollama_body = {
            "model": model,
            "messages": translated_messages,
            "stream": stream,
        }
        ensure_think_false_dict(ollama_body, "/api/chat")

        # Options-Mapping mit Typ-Validierung.
        # #654: explizites null wie nicht-gesetzt behandeln — viele OpenAI-Clients
        # serialisieren unset optionale Felder als null statt sie wegzulassen.
        options = {}
        if "temperature" in data and data["temperature"] is not None:
            if not isinstance(data["temperature"], (int, float)) or isinstance(data["temperature"], bool):
                return _openai_error("temperature must be a number", code="invalid_request", status=400)
            if not 0.0 <= data["temperature"] <= 2.0:
                return _openai_error("temperature must be between 0 and 2", code="invalid_request", status=400)
            options["temperature"] = data["temperature"]
        # #895: max_completion_tokens (OpenAI SDK v1+/Chat-Spec) und max_tokens
        # (deprecated, aber weiter unterstuetzt) werden beide auf
        # options.num_predict gemappt. Doppelangabe mit unterschiedlichen Werten
        # wird mit 400 abgelehnt, statt stillschweigend einen zu bevorzugen.
        # #698: Obergrenze 100000 gegen DoS via num_predict=10^9 — liegt
        # deutlich ueber Output-Limits aller deployten Modelle (Llama-3.x
        # ~8k, GPT-4o-Style ~16k) und blockt nur pathologische Werte.
        num_predict_val = None
        if "max_tokens" in data and data["max_tokens"] is not None:
            if not isinstance(data["max_tokens"], int) or isinstance(data["max_tokens"], bool):
                return _openai_error("max_tokens must be an integer", code="invalid_request", status=400)
            if data["max_tokens"] < 1:
                return _openai_error("max_tokens must be a positive integer", code="invalid_request", status=400)
            if data["max_tokens"] > 100000:
                return _openai_error("max_tokens must be at most 100000", code="invalid_request", status=400)
            num_predict_val = data["max_tokens"]
        if "max_completion_tokens" in data and data["max_completion_tokens"] is not None:
            if not isinstance(data["max_completion_tokens"], int) or isinstance(data["max_completion_tokens"], bool):
                return _openai_error("max_completion_tokens must be an integer", code="invalid_request", status=400)
            if data["max_completion_tokens"] < 1:
                return _openai_error("max_completion_tokens must be a positive integer", code="invalid_request", status=400)
            if data["max_completion_tokens"] > 100000:
                return _openai_error("max_completion_tokens must be at most 100000", code="invalid_request", status=400)
            if num_predict_val is not None and num_predict_val != data["max_completion_tokens"]:
                return _openai_error(
                    "max_tokens and max_completion_tokens are both set but differ; specify only one",
                    code="invalid_request", status=400,
                )
            num_predict_val = data["max_completion_tokens"]
        if num_predict_val is not None:
            options["num_predict"] = num_predict_val
        if "top_p" in data and data["top_p"] is not None:
            if not isinstance(data["top_p"], (int, float)) or isinstance(data["top_p"], bool):
                return _openai_error("top_p must be a number", code="invalid_request", status=400)
            if not 0.0 <= data["top_p"] <= 1.0:
                return _openai_error("top_p must be between 0 and 1", code="invalid_request", status=400)
            options["top_p"] = data["top_p"]
        if "stop" in data and data["stop"] is not None:
            stop_val = data["stop"]
            if isinstance(stop_val, str):
                if not stop_val:
                    return _openai_error("stop sequences cannot be empty", code="invalid_request", status=400)
                options["stop"] = [stop_val]
            elif isinstance(stop_val, list) and all(isinstance(s, str) for s in stop_val):
                if len(stop_val) > 4:
                    return _openai_error("stop must contain at most 4 sequences", code="invalid_request", status=400)
                if any(not s for s in stop_val):
                    return _openai_error("stop sequences cannot be empty", code="invalid_request", status=400)
                # Leere stop-Liste hat keinen semantischen Aequivalent im OpenAI-Default;
                # als unset behandeln statt undefined Backend-Verhalten zu provozieren.
                if stop_val:
                    options["stop"] = stop_val
            else:
                return _openai_error("stop must be a string or list of strings", code="invalid_request", status=400)
        if "frequency_penalty" in data and data["frequency_penalty"] is not None:
            if not isinstance(data["frequency_penalty"], (int, float)) or isinstance(data["frequency_penalty"], bool):
                return _openai_error("frequency_penalty must be a number", code="invalid_request", status=400)
            if not -2.0 <= data["frequency_penalty"] <= 2.0:
                return _openai_error("frequency_penalty must be between -2 and 2", code="invalid_request", status=400)
            # #425/#483 (wontfix): Mapping OpenAI -> Ollama ist semantisch inkonvertibel
            # (OpenAI: additiver Logit-Abzug * count, Ollama: multiplikativer Logit-Divisor).
            # Formel 1.0 + fp ist bewusste best-effort Design-Entscheidung; konservativere
            # Skalierung wuerde Clients die Kontrolle wegnehmen. Clients die das
            # Standard-Ollama-Verhalten wollen, sollen den Parameter weglassen
            # (Ollama-Default repeat_penalty=1.1).
            options["repeat_penalty"] = max(0.1, 1.0 + data["frequency_penalty"])
        # #423: seed -> Ollama options.seed (Reproduzierbarkeit).
        # #699: int64-Range matcht OpenAI-Spec; verhindert backend-spezifische 400/500
        # bei Werten ausserhalb der Ollama-int-Range.
        if "seed" in data and data["seed"] is not None:
            if not isinstance(data["seed"], int) or isinstance(data["seed"], bool):
                return _openai_error("seed must be an integer", code="invalid_request", status=400)
            if not -(2**63) <= data["seed"] <= 2**63 - 1:
                return _openai_error("seed must fit in a signed 64-bit integer", code="invalid_request", status=400)
            options["seed"] = data["seed"]

        if options:
            ollama_body["options"] = options

        # VRAM-Lease Intercept (#285): Dict-Helper operiert direkt auf
        # ollama_body["options"], setzt bei FORCE_CPU num_gpu=0 in place.
        lease_options = ollama_body.get("options")
        if not isinstance(lease_options, dict):
            lease_options = {}
        lease_outcome = await vram_lease.apply_options_dict(lease_options)
        if lease_options or lease_outcome == vram_lease.LeaseOutcome.FORCE_CPU:
            ollama_body["options"] = lease_options
        lease_header_value: str | None = None
        if lease_outcome == vram_lease.LeaseOutcome.FORCE_CPU:
            lease_header_value = "force-cpu"

        # Client-IP aus direkter Verbindung — XFF wird NICHT vertraut, weil der
        # OpenAI-API-Endpoint direkt auf 0.0.0.0:11440 bindet ohne Trusted Reverse-
        # Proxy davor (analog proxy.py get_client_ip / #598). Andernfalls koennte
        # jeder authentifizierte Client seine geloggte IP frei waehlen und damit
        # Audit-Logs, Top-Client-Auswertungen und Dashboard-Filter verfaelschen.
        if request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"

        # Request-Body als Text fuer Logging
        request_body_text = body.decode("utf-8", errors="replace")

        # RequestRecord erstellen
        record = RequestRecord(
            client_ip=client_ip,
            method="POST",
            path="/v1/chat/completions",
            model=model,
            request_size=len(body),
            is_streaming=stream,
            state="active",
            request_body=request_body_text,
        )
        await _safe_log_add(request_store, record)

        if lease_outcome == vram_lease.LeaseOutcome.BLOCK:
            duration_ms = (time.monotonic() - start_time) * 1000
            try:
                await request_store.update_request(
                    record.id, state="error", status_code=503,
                    duration_ms=round(duration_ms, 1),
                    error_message="GPU-Gate aktiv — Request verworfen",
                )
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "update_request im VRAM-Lease-Block-Zweig fehlgeschlagen"
                )
            return _openai_error(
                "GPU-Operation blockiert — Request verworfen",
                error_type="server_error",
                code="vram_lease_active",
                status=503,
            )

        ollama_json = json.dumps(ollama_body).encode()
        gpu_op = await _begin_openai_lazy_operation(model)
        if not gpu_op.allowed:
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=gpu_op.decision.status_code,
                duration_ms=round(duration_ms, 1),
                error_message=f"GPU-Gate blockiert: {gpu_op.decision.reason}",
            )
            return _openai_error(
                "GPU-Operation blockiert",
                error_type="server_error",
                code="vram_lease_active",
                status=gpu_op.decision.status_code,
            )

        if stream:
            return await _handle_streaming(
                http_client, request_store, record,
                ollama_json, model, start_time,
                lease_header=lease_header_value,
                include_usage=include_usage,
                gpu_op=gpu_op,
            )
        else:
            return await _handle_non_streaming(
                http_client, request_store, record,
                ollama_json, model, start_time,
                lease_header=lease_header_value,
                gpu_op=gpu_op,
            )

    async def _handle_non_streaming(
        http_client: httpx.AsyncClient,
        request_store: RequestStore,
        record: RequestRecord,
        ollama_json: bytes,
        model: str,
        start_time: float,
        lease_header: str | None = None,
        gpu_op: vram_lease.GPUServiceOperation | None = None,
    ) -> Response:
        """Non-streaming Chat Completion."""
        # #828: Lease-Header auch in Error-Responses durchreichen, damit Clients
        # VRAM-Lease-Status auch bei Backend-Fehlern verfolgen koennen.
        lease_headers = {"X-Kiron-VRAM-Lease": lease_header} if lease_header else None
        try:
            backend_request = http_client.build_request(
                method="POST",
                url="/api/chat",
                content=ollama_json,
                headers={"Content-Type": "application/json"},
            )
            resp = await http_client.send(backend_request, stream=True)
        except httpx.ConnectError as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend connection failed: {e}",
            )
            return _openai_error(
                "Backend not reachable", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )
        except httpx.TimeoutException as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=504,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend timeout: {e}",
            )
            return _openai_error(
                "Backend timeout", error_type="server_error",
                code="timeout", status=504, headers=lease_headers,
            )
        except httpx.RequestError as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend request error: {e}",
            )
            return _openai_error(
                "Backend request failed", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )
        except Exception as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=500,
                duration_ms=round(duration_ms, 1),
                error_message=f"Internal error: {e}",
            )
            return _openai_error(
                "Internal server error", error_type="server_error",
                code="internal_error", status=500, headers=lease_headers,
            )

        try:
            content_length_hdr = resp.headers.get("content-length")
            if content_length_hdr:
                try:
                    if int(content_length_hdr) > MAX_RESPONSE_SIZE:
                        duration_ms = (time.monotonic() - start_time) * 1000
                        await _safe_log_update(
                            request_store,
                            record.id, state="error", status_code=502,
                            duration_ms=round(duration_ms, 1),
                            error_message=f"Backend response exceeds limit ({content_length_hdr} > {MAX_RESPONSE_SIZE} bytes)",
                        )
                        return _openai_error(
                            f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)",
                            error_type="server_error", code="backend_error", status=502, headers=lease_headers,
                        )
                except ValueError:
                    pass

            body_chunks: list[bytes] = []
            total_size = 0
            try:
                async for chunk in resp.aiter_bytes():
                    total_size += len(chunk)
                    if total_size > MAX_RESPONSE_SIZE:
                        duration_ms = (time.monotonic() - start_time) * 1000
                        await _safe_log_update(
                            request_store,
                            record.id, state="error", status_code=502,
                            duration_ms=round(duration_ms, 1),
                            error_message=f"Backend response exceeds limit (> {MAX_RESPONSE_SIZE} bytes while streaming)",
                        )
                        return _openai_error(
                            f"Backend response too large (max {MAX_RESPONSE_SIZE // (1024 * 1024)}MB)",
                            error_type="server_error", code="backend_error", status=502, headers=lease_headers,
                        )
                    body_chunks.append(chunk)
            except httpx.TimeoutException as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    request_store,
                    record.id, state="error", status_code=504,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Backend timeout while reading: {e}",
                )
                return _openai_error(
                    "Backend timeout", error_type="server_error",
                    code="timeout", status=504, headers=lease_headers,
                )
            except httpx.RequestError as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                await _safe_log_update(
                    request_store,
                    record.id, state="error", status_code=502,
                    duration_ms=round(duration_ms, 1),
                    error_message=f"Backend read error: {e}",
                )
                return _openai_error(
                    "Backend request failed", error_type="server_error",
                    code="backend_error", status=502, headers=lease_headers,
                )
            response_body = b"".join(body_chunks)
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            if gpu_op is not None:
                if "response_body" in locals() and gpu_op.token is not None:
                    if 400 <= resp.status_code < 500:
                        gpu_op.clear_marker = True
                    elif 200 <= resp.status_code < 300:
                        # Leerer Body bei 2xx ist KEIN unambiguer Erfolg — Marker
                        # bleibt fail-closed gesetzt. Nur bei parsbarem JSON ohne
                        # error-Feld als Erfolg werten.
                        if response_body:
                            try:
                                parsed = json.loads(response_body)
                                if not (isinstance(parsed, dict) and parsed.get("error")):
                                    gpu_op.clear_marker = True
                            except Exception:
                                pass
                await vram_lease.finish_gpu_service_operation(gpu_op)

        duration_ms = (time.monotonic() - start_time) * 1000

        try:
            ollama_resp = json.loads(response_body)
        except Exception:
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message="Invalid JSON from backend",
            )
            return _openai_error(
                "Invalid response from backend", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )

        if not isinstance(ollama_resp, dict):
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message="Non-dict response from backend",
            )
            return _openai_error(
                "Invalid response from backend", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )

        error_msg = ollama_resp.get("error")
        if error_msg or resp.status_code >= 400:
            # Backend-Status nicht durchreichen: 429 -> 503, sonst -> 502.
            # Sonst wuerde ein Backend-401 bei OpenAI-SDKs Re-Auth/Logout triggern,
            # obwohl der Client-Key gueltig ist.
            if resp.status_code == 429:
                client_status, client_code = 503, "backend_unavailable"
            else:
                client_status, client_code = 502, "backend_error"
            message = error_msg if error_msg else f"Backend returned HTTP {resp.status_code}"
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=client_status,
                duration_ms=round(duration_ms, 1),
                error_message=str(message)[:500],
            )
            return _openai_error(
                str(message), error_type="server_error",
                code=client_code, status=client_status, headers=lease_headers,
            )

        # Ollama -> OpenAI Response-Translation
        # OpenAI-Spec verlangt content als String — non-string vom Backend (dict/list/zahl)
        # wuerde via json.dumps zu Schema-Verletzung fuehren und strikte Pydantic-Clients
        # (LangChain, openai-python) crashen. Defensiv auf "" fallen lassen.
        content = ""
        msg = ollama_resp.get("message", {})
        if isinstance(msg, dict):
            content_raw = msg.get("content", "")
            content = content_raw if isinstance(content_raw, str) else ""

        # Negative token counts vom Backend wuerden total_tokens unsinnig machen
        # (Billing-relevant). Defensiv auf 0 fallen lassen.
        pt_raw = ollama_resp.get("prompt_eval_count")
        prompt_tokens = pt_raw if isinstance(pt_raw, int) and not isinstance(pt_raw, bool) and pt_raw >= 0 else 0
        ct_raw = ollama_resp.get("eval_count")
        completion_tokens = ct_raw if isinstance(ct_raw, int) and not isinstance(ct_raw, bool) and ct_raw >= 0 else 0
        finish_reason = _map_finish_reason(ollama_resp.get("done_reason"))

        openai_response = {
            "id": _chatcmpl_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        response_json = json.dumps(openai_response)

        await _safe_log_update(
            request_store,
            record.id,
            state="completed",
            status_code=200,
            duration_ms=round(duration_ms, 1),
            response_size=len(response_json.encode("utf-8")),
            tokens_generated=completion_tokens,
            response_body=_truncate_for_log(content),
        )

        response_headers = {}
        if lease_header:
            response_headers["X-Kiron-VRAM-Lease"] = lease_header

        return Response(
            content=response_json,
            status_code=200,
            media_type="application/json",
            headers=response_headers or None,
        )

    async def _handle_streaming(
        http_client: httpx.AsyncClient,
        request_store: RequestStore,
        record: RequestRecord,
        ollama_json: bytes,
        model: str,
        start_time: float,
        lease_header: str | None = None,
        include_usage: bool = False,
        gpu_op: vram_lease.GPUServiceOperation | None = None,
    ) -> StreamingResponse:
        """Streaming Chat Completion (Ollama NDJSON -> OpenAI SSE)."""
        completion_id = _chatcmpl_id()
        # #828: Lease-Header auch in Error-Responses durchreichen, damit Clients
        # VRAM-Lease-Status auch bei Backend-Fehlern verfolgen koennen.
        lease_headers = {"X-Kiron-VRAM-Lease": lease_header} if lease_header else None

        try:
            backend_request = http_client.build_request(
                method="POST",
                url="/api/chat",
                content=ollama_json,
                headers={"Content-Type": "application/json"},
            )
            backend_response = await http_client.send(backend_request, stream=True)
        except httpx.ConnectError as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend connection failed: {e}",
            )
            return _openai_error(
                "Backend not reachable", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )
        except httpx.TimeoutException as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=504,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend timeout: {e}",
            )
            return _openai_error(
                "Backend timeout", error_type="server_error",
                code="timeout", status=504, headers=lease_headers,
            )
        except httpx.RequestError as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=502,
                duration_ms=round(duration_ms, 1),
                error_message=f"Backend request error: {e}",
            )
            return _openai_error(
                "Backend request failed", error_type="server_error",
                code="backend_error", status=502, headers=lease_headers,
            )
        except Exception as e:
            if gpu_op is not None:
                await vram_lease.finish_gpu_service_operation(gpu_op)
            duration_ms = (time.monotonic() - start_time) * 1000
            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=500,
                duration_ms=round(duration_ms, 1),
                error_message=f"Internal error: {e}",
            )
            return _openai_error(
                "Internal server error", error_type="server_error",
                code="internal_error", status=500, headers=lease_headers,
            )

        if backend_response.status_code != 200:
            MAX_ERROR_BODY = 64 * 1024
            error_body = b""
            try:
                async for chunk in backend_response.aiter_bytes():
                    error_body += chunk
                    if len(error_body) > MAX_ERROR_BODY:
                        error_body = error_body[:MAX_ERROR_BODY]
                        break
            except Exception:
                pass
            finally:
                # aclose() kann RemoteProtocolError/ReadTimeout werfen — ohne
                # try/except wuerde der gpu_op-Cleanup uebersprungen, Lock und
                # gpu_service_loading-Marker bleiben haengen. Pattern analog
                # zum sse_generator-finally Z.1683-1687.
                try:
                    await backend_response.aclose()
                except Exception:
                    pass
                if gpu_op is not None:
                    if 400 <= backend_response.status_code < 500 and gpu_op.token is not None:
                        gpu_op.clear_marker = True
                    await vram_lease.finish_gpu_service_operation(gpu_op)

            duration_ms = (time.monotonic() - start_time) * 1000
            error_msg = f"Backend returned status {backend_response.status_code}"
            try:
                error_data = json.loads(error_body)
                if isinstance(error_data, dict) and "error" in error_data:
                    error_msg = str(error_data["error"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Backend-Status nicht durchreichen: 429 -> 503, sonst -> 502.
            if backend_response.status_code == 429:
                client_status, client_code = 503, "backend_unavailable"
            else:
                client_status, client_code = 502, "backend_error"

            await _safe_log_update(
                request_store,
                record.id, state="error", status_code=client_status,
                duration_ms=round(duration_ms, 1),
                error_message=error_msg,
            )
            return _openai_error(
                error_msg, error_type="server_error",
                code=client_code, status=client_status, headers=lease_headers,
            )

        try:
            await request_store.update_request(record.id, status_code=200)
        except Exception as e:
            # #527: DB-Update am Stream-Start ist nicht kritisch — finaler State
            # wird vom sse_generator gesetzt. Frueher: re-raise -> Starlette
            # liefert rohen 500 ohne OpenAI-Error-Format, RequestStore-Record
            # bleibt 'active'. Jetzt: loggen + weitermachen.
            logging.getLogger(__name__).warning(
                "openai_api: update_request beim Stream-Start fehlgeschlagen "
                "(record=%s): %s — Stream startet trotzdem, finaler State im sse_generator.",
                record.id, e,
            )

        async def sse_generator():
            """Ollama NDJSON -> OpenAI SSE Translation."""
            # OpenAI-Konvention: selber created-Timestamp ueber alle Chunks einer
            # Completion. Einmal am Stream-Start berechnen.
            created = int(time.time())
            total_response_size = 0
            tokens_generated = 0
            prompt_tokens = 0
            buffer = b""
            response_text_parts = []
            response_text_len = 0
            first_chunk = True
            error_already_signaled = False  # #268: verhindert doppeltes DONE/update_request
            done_seen = False  # #296: backend muss done=true senden, sonst Stream abgebrochen
            stream_corrupted = False  # #596: Malformed NDJSON-Zeilen invalidieren den Stream

            MAX_BUFFER_BYTES = 10 * 1024 * 1024

            try:
                async for chunk in backend_response.aiter_bytes():
                    buffer += chunk
                    if len(buffer) > MAX_BUFFER_BYTES:
                        raise RuntimeError(
                            f"Backend response buffer exceeded {MAX_BUFFER_BYTES} bytes without newline"
                        )

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            line_data = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            import logging
                            logging.getLogger(__name__).warning(
                                "Malformed NDJSON-Zeile vom Backend verworfen: %s — line=%r",
                                e, line[:200],
                            )
                            # #596: Stream als korrupt markieren — selbst wenn
                            # spaeter done=true folgt, darf der Request nicht als
                            # erfolgreich gemeldet werden, weil hier Content/Usage
                            # verloren ging.
                            stream_corrupted = True
                            continue

                        # JSON erlaubt list/int/string/null als Top-Level. line_data.get()
                        # wuerde dann AttributeError werfen — analog zur JSONDecodeError-Behandlung
                        # als korrupt markieren, statt im generischen except Exception zu landen
                        # und Python-Internals ('list' object has no attribute 'get') zu leaken.
                        if not isinstance(line_data, dict):
                            import logging
                            logging.getLogger(__name__).warning(
                                "Non-dict NDJSON-Top-Level vom Backend verworfen — line=%r",
                                line[:200],
                            )
                            stream_corrupted = True
                            continue

                        is_done = line_data.get("done", False)

                        # Ollama kann Error-Chunks mit oder ohne done=true senden — done=true
                        # mit error-Feld darf nicht als erfolgreicher Stop emittiert werden.
                        # Truthy-Check: error koennte im done-Chunk auch null/"" sein.
                        err_field = line_data.get("error")
                        if err_field:
                            error_msg = str(err_field)
                            # #485: Error-Event mit OpenAI-style Fehler-Objekt bevor
                            # finish_reason=null, damit Clients den Grund erfahren.
                            err_event = _sse_error_event(f"Backend error: {error_msg}")
                            total_response_size += len(err_event.encode("utf-8"))
                            yield err_event
                            # #427: finish_reason=null signalisiert "nicht normal beendet"
                            # (vs. "stop" = erfolgreich). Clients koennen so Fehler von
                            # erfolgreicher Completion unterscheiden.
                            sse_data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None,
                                }],
                            }
                            if include_usage:
                                sse_data["usage"] = None
                            sse_line = f"data: {json.dumps(sse_data)}\n\n"
                            total_response_size += len(sse_line.encode("utf-8"))
                            yield sse_line
                            total_response_size += len(b"data: [DONE]\n\n")
                            yield "data: [DONE]\n\n"
                            error_already_signaled = True

                            duration_ms = (time.monotonic() - start_time) * 1000
                            try:
                                # #915: status_code=200 nachtragen, falls early-update (Z.1329)
                                # an DB-Glitch scheiterte — Backend lieferte HTTP-200 und
                                # StreamingResponse wurde bereits mit 200 emittiert.
                                await request_store.update_request(
                                    record.id, state="error", status_code=200,
                                    duration_ms=round(duration_ms, 1),
                                    tokens_generated=tokens_generated,
                                    response_size=total_response_size,
                                    error_message=f"Ollama error: {error_msg}",
                                )
                            except Exception:
                                import logging
                                logging.getLogger(__name__).exception(
                                    "update_request im Ollama-Error-Zweig fehlgeschlagen"
                                )
                            return

                        # Content IMMER extrahieren (auch bei done=true, siehe #243).
                        # Non-string content (dict/list) wuerde "".join(response_text_parts)
                        # mit TypeError abbrechen und einen erfolgreichen Stream als Error
                        # melden — defensiv auf "" fallen lassen.
                        content = ""
                        msg = line_data.get("message", {})
                        if isinstance(msg, dict):
                            content_raw = msg.get("content", "")
                            content = content_raw if isinstance(content_raw, str) else ""
                        if content and response_text_len < MAX_LOGGED_RESPONSE_BODY:
                            # Pro-Append slicen, damit der Buffer nicht weit ueber das
                            # Log-Limit hinaus waechst. +1 Zeichen Overshoot, damit
                            # _truncate_for_log den "[truncated]"-Marker noch setzt.
                            chunk = content[: MAX_LOGGED_RESPONSE_BODY - response_text_len + 1]
                            response_text_parts.append(chunk)
                            response_text_len += len(chunk)

                        # tokens_generated inkrementell mitfuehren falls Ollama es im Chunk mitschickt (#244)
                        # bool-Ausschluss + >=0 analog zum non-streaming Pfad (Z.880-883), da
                        # isinstance(True, int) True ist und True (=1) sonst tokens_generated kapern wuerde.
                        eval_count = line_data.get("eval_count")
                        if isinstance(eval_count, int) and not isinstance(eval_count, bool) and eval_count >= 0 and eval_count > tokens_generated:
                            tokens_generated = eval_count

                        prompt_eval_count = line_data.get("prompt_eval_count")
                        if isinstance(prompt_eval_count, int) and not isinstance(prompt_eval_count, bool) and prompt_eval_count >= 0 and prompt_eval_count > prompt_tokens:
                            prompt_tokens = prompt_eval_count

                        if is_done:
                            done_seen = True
                            if stream_corrupted:
                                # #744: Wenn eine vorherige malformed NDJSON-Zeile den
                                # Stream als korrupt markiert hat, NICHT den Erfolgs-
                                # finish_reason-chunk emittieren — der post-loop
                                # else-Branch emittiert sonst einen zweiten
                                # finish_reason-chunk (=null), was OpenAI-Spec verletzt
                                # (genau ein finish_reason pro choice).
                                # #1017: break statt continue — nach done=true keine
                                # weiteren Backend-Zeilen verarbeiten.
                                break
                            finish_reason = _map_finish_reason(line_data.get("done_reason"))
                            delta = {"content": content} if content else {}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False
                            sse_data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason,
                                }],
                            }
                            if include_usage:
                                sse_data["usage"] = None
                            sse_line = f"data: {json.dumps(sse_data)}\n\n"
                            total_response_size += len(sse_line.encode("utf-8"))
                            yield sse_line
                            # #1017: OpenAI-Spec verlangt genau einen finish_reason
                            # pro choice. Nach done=true keine weiteren NDJSON-Zeilen
                            # verarbeiten — falls Backend buggy ist und nach done=true
                            # noch delta- oder weitere done-Chunks sendet, wuerden die
                            # sonst als zusaetzliche Chunks emittiert.
                            break
                        else:
                            delta = {}
                            # role nur mitsenden wenn auch content kommt (#245) - sonst
                            # skip wir den leeren Chunk um role ohne content zu vermeiden
                            if first_chunk and content:
                                delta["role"] = "assistant"
                                first_chunk = False
                            if content:
                                delta["content"] = content

                            if not delta:
                                continue

                            sse_data = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None,
                                }],
                            }
                            if include_usage:
                                sse_data["usage"] = None
                            sse_line = f"data: {json.dumps(sse_data)}\n\n"
                            total_response_size += len(sse_line.encode("utf-8"))
                            yield sse_line

                    # #1017: Inner-Loop hat done=true gesehen und gebrochen — keine
                    # weiteren Backend-Chunks lesen, sonst koennten Trailing-Bytes
                    # (z.B. doppeltes done) zusaetzliche Chunks produzieren.
                    if done_seen:
                        break

                # Restlichen Buffer verarbeiten (letzter Chunk ohne Newline).
                # #1017: Wenn done=true bereits im Main-Loop gesehen wurde, ist
                # der Stream offiziell beendet — Trailing-Bytes (Backend-Bug oder
                # doppeltes done) duerfen keine zusaetzlichen Chunks erzeugen.
                if not done_seen and buffer.strip():
                    try:
                        line_data = json.loads(buffer.strip())
                        if not isinstance(line_data, dict):
                            # JSON erlaubt list/int/string/null als Top-Level — line_data.get()
                            # wuerde dann AttributeError werfen. Analog zur JSONDecodeError-Behandlung
                            # als korrupt markieren statt im generischen except Exception zu landen
                            # und Python-Internals an den Client zu leaken.
                            import logging
                            logging.getLogger(__name__).warning(
                                "Non-dict NDJSON-Top-Level im trailing buffer verworfen — buffer=%r",
                                buffer[:200],
                            )
                            stream_corrupted = True
                        else:
                            is_done = line_data.get("done", False)

                            # #528/#896: Ollama-Error-Chunk im Trailing-Buffer (mit oder
                            # ohne done=true, ohne Newline) muss wie im Main-Loop als
                            # SSE-Error an den Client gemeldet werden, sonst geht der
                            # Fehlertext verloren bzw. ein Error+done=true-Chunk wird
                            # faelschlich als Erfolg gestreamt. Truthy-Check analog Z.1382:
                            # error koennte im done-Chunk auch null/"" sein.
                            err_field = line_data.get("error")
                            if err_field:
                                error_msg = str(err_field)
                                err_event = _sse_error_event(f"Backend error: {error_msg}")
                                total_response_size += len(err_event.encode("utf-8"))
                                yield err_event
                                sse_data = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": None,
                                    }],
                                }
                                if include_usage:
                                    sse_data["usage"] = None
                                sse_line = f"data: {json.dumps(sse_data)}\n\n"
                                total_response_size += len(sse_line.encode("utf-8"))
                                yield sse_line
                                total_response_size += len(b"data: [DONE]\n\n")
                                yield "data: [DONE]\n\n"
                                error_already_signaled = True

                                duration_ms = (time.monotonic() - start_time) * 1000
                                try:
                                    # #915: status_code=200 nachtragen, falls early-update
                                    # an DB-Glitch scheiterte — Backend lieferte HTTP-200.
                                    await request_store.update_request(
                                        record.id, state="error", status_code=200,
                                        duration_ms=round(duration_ms, 1),
                                        tokens_generated=tokens_generated,
                                        response_size=total_response_size,
                                        error_message=f"Ollama error: {error_msg}",
                                    )
                                except Exception:
                                    import logging
                                    logging.getLogger(__name__).exception(
                                        "update_request im Ollama-Error-Zweig fehlgeschlagen"
                                    )
                                return

                            # Trailing-Buffer: gleiche String-Validierung wie im Main-Loop.
                            content = ""
                            msg = line_data.get("message", {})
                            if isinstance(msg, dict):
                                content_raw = msg.get("content", "")
                                content = content_raw if isinstance(content_raw, str) else ""
                            if content and response_text_len < MAX_LOGGED_RESPONSE_BODY:
                                # Pro-Append slicen, damit der Buffer nicht weit ueber das
                                # Log-Limit hinaus waechst. +1 Zeichen Overshoot, damit
                                # _truncate_for_log den "[truncated]"-Marker noch setzt.
                                chunk = content[: MAX_LOGGED_RESPONSE_BODY - response_text_len + 1]
                                response_text_parts.append(chunk)
                                response_text_len += len(chunk)

                            eval_count = line_data.get("eval_count")
                            if isinstance(eval_count, int) and not isinstance(eval_count, bool) and eval_count >= 0 and eval_count > tokens_generated:
                                tokens_generated = eval_count

                            prompt_eval_count = line_data.get("prompt_eval_count")
                            if isinstance(prompt_eval_count, int) and not isinstance(prompt_eval_count, bool) and prompt_eval_count >= 0 and prompt_eval_count > prompt_tokens:
                                prompt_tokens = prompt_eval_count

                            if is_done:
                                done_seen = True
                                # #744: gleicher Guard wie im Inner-Loop — bei korruptem
                                # Stream den Erfolgs-finish_reason-chunk nicht emittieren,
                                # sonst entsteht ein zweiter finish_reason-chunk im
                                # post-loop else-Branch.
                                if not stream_corrupted:
                                    finish_reason = _map_finish_reason(line_data.get("done_reason"))
                                    delta = {"content": content} if content else {}
                                    if first_chunk:
                                        delta["role"] = "assistant"
                                        first_chunk = False
                                    sse_data = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": finish_reason,
                                        }],
                                    }
                                    if include_usage:
                                        sse_data["usage"] = None
                                    sse_line = f"data: {json.dumps(sse_data)}\n\n"
                                    total_response_size += len(sse_line.encode("utf-8"))
                                    yield sse_line
                            else:
                                delta = {}
                                if first_chunk and content:
                                    delta["role"] = "assistant"
                                    first_chunk = False
                                if content:
                                    delta["content"] = content

                                if delta:
                                    sse_data = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }],
                                    }
                                    if include_usage:
                                        sse_data["usage"] = None
                                    sse_line = f"data: {json.dumps(sse_data)}\n\n"
                                    total_response_size += len(sse_line.encode("utf-8"))
                                    yield sse_line
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        # #296: Restbuffer-Parsefehler deutet auf abgeschnittenen Stream hin
                        import logging
                        logging.getLogger(__name__).warning(
                            "Malformed trailing NDJSON buffer — Stream vermutlich abgebrochen: %s — buffer=%r",
                            e, buffer[:200],
                        )
                        # #596: auch trailing-Korruption invalidiert den Stream.
                        stream_corrupted = True

                response_text = _truncate_for_log("".join(response_text_parts))
                duration_ms = (time.monotonic() - start_time) * 1000

                if done_seen and not stream_corrupted:
                    if include_usage:
                        usage_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": tokens_generated,
                                "total_tokens": prompt_tokens + tokens_generated,
                            },
                        }
                        usage_line = f"data: {json.dumps(usage_chunk)}\n\n"
                        total_response_size += len(usage_line.encode("utf-8"))
                        yield usage_line
                    total_response_size += len(b"data: [DONE]\n\n")
                    yield "data: [DONE]\n\n"
                    try:
                        # #915: status_code=200 nachtragen, falls early-update (Z.1329)
                        # an DB-Glitch scheiterte — sonst bleibt erfolgreicher Stream
                        # mit Default 0 trotz HTTP-200 im Store.
                        await request_store.update_request(
                            record.id,
                            state="completed",
                            status_code=200,
                            duration_ms=round(duration_ms, 1),
                            response_size=total_response_size,
                            tokens_generated=tokens_generated,
                            response_body=response_text,
                        )
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "update_request nach erfolgreichem Stream fehlgeschlagen"
                        )
                else:
                    # #296: Backend-Stream endete ohne done=true -> Fehler, nicht Erfolg
                    # #596: ODER der Stream enthielt malformed NDJSON-Zeilen — auch
                    # ein nachgereichtes done=true rettet einen verlustbehafteten
                    # Stream nicht.
                    # #485: Error-Event bevor finish_reason=null.
                    err_msg = (
                        "Backend stream contained malformed NDJSON"
                        if stream_corrupted
                        else "Backend stream ended without done=true"
                    )
                    err_event = _sse_error_event(err_msg)
                    total_response_size += len(err_event.encode("utf-8"))
                    yield err_event
                    # #427: finish_reason=null (nicht "stop") um Fehler signalisierbar zu machen.
                    sse_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }],
                    }
                    if include_usage:
                        sse_data["usage"] = None
                    sse_line = f"data: {json.dumps(sse_data)}\n\n"
                    total_response_size += len(sse_line.encode("utf-8"))
                    yield sse_line
                    total_response_size += len(b"data: [DONE]\n\n")
                    yield "data: [DONE]\n\n"
                    try:
                        # #915: status_code=200 nachtragen — Backend lieferte HTTP-200,
                        # StreamingResponse wurde bereits mit 200 emittiert; nur der
                        # Stream-Inhalt selbst ist defekt.
                        await request_store.update_request(
                            record.id,
                            state="error",
                            status_code=200,
                            duration_ms=round(duration_ms, 1),
                            response_size=total_response_size,
                            tokens_generated=tokens_generated,
                            response_body=response_text,
                            error_message=err_msg,
                        )
                    except Exception:
                        import logging
                        logging.getLogger(__name__).exception(
                            "update_request im abgebrochen-Zweig fehlgeschlagen"
                        )

            except (asyncio.CancelledError, GeneratorExit):
                # #330: aclose() des Async-Generators wirft GeneratorExit, nicht CancelledError —
                # ohne Behandlung wuerde der Request auf state=active stehen bleiben
                duration_ms = (time.monotonic() - start_time) * 1000
                try:
                    # #915: status_code=200 nachtragen — Client trennt erst, nachdem
                    # StreamingResponse mit HTTP-200 begonnen hat.
                    await request_store.update_request(
                        record.id,
                        state="error",
                        status_code=200,
                        duration_ms=round(duration_ms, 1),
                        response_size=total_response_size,
                        tokens_generated=tokens_generated,
                        response_body=_truncate_for_log("".join(response_text_parts)),
                        error_message="Client disconnected",
                    )
                except Exception:
                    pass
                raise

            except Exception as e:
                import logging
                logging.getLogger(__name__).exception("Streaming error")
                if not error_already_signaled:
                    # #485: Error-Event mit Exception-Info bevor finish_reason=null.
                    err_event = _sse_error_event(f"Streaming error: {e}")
                    total_response_size += len(err_event.encode("utf-8"))
                    yield err_event
                    # Error-Chunk an Client senden bevor der Stream endet
                    # #427: finish_reason=null (nicht "stop") fuer Fehler-Differenzierung.
                    sse_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }],
                    }
                    if include_usage:
                        sse_data["usage"] = None
                    sse_line = f"data: {json.dumps(sse_data)}\n\n"
                    total_response_size += len(sse_line.encode("utf-8"))
                    yield sse_line
                    total_response_size += len(b"data: [DONE]\n\n")
                    yield "data: [DONE]\n\n"

                    duration_ms = (time.monotonic() - start_time) * 1000
                    try:
                        # #915: status_code=200 nachtragen — Backend lieferte HTTP-200,
                        # StreamingResponse wurde bereits emittiert; Exception kam
                        # waehrend der Stream-Verarbeitung.
                        await request_store.update_request(
                            record.id,
                            state="error",
                            status_code=200,
                            duration_ms=round(duration_ms, 1),
                            response_size=total_response_size,
                            tokens_generated=tokens_generated,
                            response_body=_truncate_for_log("".join(response_text_parts)),
                            error_message=f"Streaming error: {e}",
                        )
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "update_request im Streaming-Error-Handler fehlgeschlagen"
                        )

            finally:
                try:
                    await backend_response.aclose()
                except Exception:
                    pass
                if gpu_op is not None:
                    if (
                        gpu_op.token is not None
                        and done_seen
                        and not stream_corrupted
                    ):
                        gpu_op.clear_marker = True
                    await vram_lease.finish_gpu_service_operation(gpu_op)

        stream_headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if lease_header:
            stream_headers["X-Kiron-VRAM-Lease"] = lease_header
        return StreamingResponse(
            content=sse_generator(),
            status_code=200,
            media_type="text/event-stream",
            headers=stream_headers,
        )

    app = Starlette(
        routes=[
            Route("/v1/models", endpoint=models_handler, methods=["GET"]),
            Route("/v1/chat/completions", endpoint=chat_completions_handler, methods=["POST"]),
        ],
    )

    async def shutdown_client():
        await http_client.aclose()

    app.add_event_handler("shutdown", shutdown_client)

    return app
