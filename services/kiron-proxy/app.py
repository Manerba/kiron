"""Kiron Dashboard - FastAPI Backend, Port 8505."""

import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.websockets import WebSocketState
import uvicorn
import httpx

from kiron_common.embedding_registry import MODEL_CATALOG, MODEL_STATE_VIEW
from kiron_common.local_model_registry import (
    ModelRegistrationError,
    ModelRegistrationService,
)
from kiron_common.local_model_registry.models import canonical_ollama_reference
from kiron_common.local_model_registry.composition import (
    DEFAULT_OLLAMA_BASE_URL,
    build_model_registration_service,
)
from kiron_common.model_catalog import BackendType
from kiron_common.ollama_compat import is_real_int
from kiron_common.gpu_admission import AdmissionError
from kiron_common.gpu_admission.ollama_lifecycle import OllamaLifecycleOperation

from request_store import RequestStore, RequestStoreReadError
from metrics import get_cached_payload
from history_db import MetricsDB
from high_res_history import HighResHistoryClosed, HighResHistoryService
from api_key_store import LastActiveApiKeyError
import vram_lease
from kiron_common.gpu_admission.native_contract import (
    COMPLETION_HEADER, OPERATION_HEADER, OVERLAY_HEADER,
)
import dashboard_runtime
from model_control import ModelControl
import ollama_recovery
from native_admission import make_store as ollama_admission_store
from routing_catalog import PROXY_ROUTING_VIEW
from selftest import router as selftest_router
from model_discovery import (
    InventoryShapeError,
    ServiceInventoryError,
    build_local_models_payload,
    service_huggingface_inventory,
)


if MODEL_STATE_VIEW.catalog is not MODEL_CATALOG:
    raise RuntimeError("proxy model state view must use the shared MODEL_CATALOG")

app = FastAPI(title="Kiron Dashboard")

# Verzeichnisse
static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
project_root = Path(__file__).resolve().parent.parent.parent

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(selftest_router)

# Globaler Request Store (wird von main.py gesetzt)
store: RequestStore = None

# Globale MetricsDB (wird von main.py gesetzt)
metrics_db: MetricsDB = None

# Eigenstaendige RAM-basierte 1m-Historie; wird von main.py gesetzt.
high_res_history_service: HighResHistoryService | None = None

# DBWorkTracker fuer SQLite-Reads aus Dashboard-Handlern (#713): cross-thread
# `db.close()` darf nicht waehrend laufender query_range/get_db_stats greifen.
db_work_tracker = None

# Globaler API Key Store (wird von main.py gesetzt)
api_key_store = None

OLLAMA_BASE_URL = DEFAULT_OLLAMA_BASE_URL
EMBEDDING_HEALTH_URL = "http://127.0.0.1:11436/health"
EMBEDDING_LOAD_URL = "http://127.0.0.1:11436/api/load"
EMBEDDING_UNLOAD_URL = "http://127.0.0.1:11436/api/unload"
EMBEDDING_SERVICE = "kiron-embeddings.service"
DEBERTA_HEALTH_URL = "http://127.0.0.1:11437/health"
DEBERTA_LOAD_URL = "http://127.0.0.1:11437/api/load"
DEBERTA_UNLOAD_URL = "http://127.0.0.1:11437/api/unload"
DEBERTA_SERVICE = "kiron-deberta.service"
DATA_ROOT = project_root / "data"
PROXY_DATA_DIR = DATA_ROOT / "kiron-proxy"
SHARED_DATA_DIR = DATA_ROOT / "shared"
RUNTIME_CONFIG_FILE = SHARED_DATA_DIR / "runtime_config.json"
SERVICE_CONTROL_HELPER = "/usr/local/sbin/kiron-service-control"
FIREWALL_HELPER = "/usr/local/sbin/kiron-maintenance-firewall"
EMBEDDING_MODEL_SLOTS_MIN = 1
EMBEDDING_MODEL_SLOTS_DEFAULT = 2
EMBEDDING_MODEL_SLOTS_MAX = 4
_runtime_config_file_lock = threading.RLock()
_SERVICE_CONTROL_ALLOWED = frozenset({
    ("restart", "kiron-proxy.service"),
    ("start", EMBEDDING_SERVICE),
    ("stop", EMBEDDING_SERVICE),
    ("restart", EMBEDDING_SERVICE),
    ("start", DEBERTA_SERVICE),
    ("stop", DEBERTA_SERVICE),
    ("restart", DEBERTA_SERVICE),
})
_FIREWALL_ALLOWED_ACTIONS = frozenset({"check", "insert", "delete"})
_FIREWALL_ALLOWED_CHAINS = frozenset({"INPUT", "DOCKER-USER"})
_FIREWALL_ALLOWED_PORTS = frozenset({5001, 11434, 11435, 11440})
_FIREWALL_ABSENT_EXIT = 10


async def _to_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _canonical_service_unit(unit: str) -> str:
    if not isinstance(unit, str) or not unit:
        raise ValueError("service unit must be a non-empty string")
    return unit if unit.endswith(".service") else f"{unit}.service"


def _service_control_command(verb: str, unit: str) -> list[str]:
    canonical_unit = _canonical_service_unit(unit)
    if (verb, canonical_unit) not in _SERVICE_CONTROL_ALLOWED:
        raise ValueError(f"service-control operation not allowed: {verb} {canonical_unit}")
    return ["sudo", "-n", SERVICE_CONTROL_HELPER, verb, canonical_unit]


def _firewall_command(action: str, chain: str, port: int) -> list[str]:
    if action not in _FIREWALL_ALLOWED_ACTIONS:
        raise ValueError(f"firewall action not allowed: {action}")
    if chain not in _FIREWALL_ALLOWED_CHAINS:
        raise ValueError(f"firewall chain not allowed: {chain}")
    if port not in _FIREWALL_ALLOWED_PORTS:
        raise ValueError(f"firewall port not allowed: {port}")
    return ["sudo", "-n", FIREWALL_HELPER, action, chain, str(port)]


async def _run_service_control(verb: str, unit: str, *, timeout: float) -> subprocess.CompletedProcess:
    return await _to_thread(
        subprocess.run,
        _service_control_command(verb, unit),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


async def _run_firewall_helper(action: str, chain: str, port: int) -> subprocess.CompletedProcess:
    try:
        return await _to_thread(
            subprocess.run,
            _firewall_command(action, chain, port),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired) as exc:
        cmd = ["sudo", "-n", FIREWALL_HELPER, action, chain, str(port)]
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(exc))


def _coerce_embedding_model_slots(value, *, default=EMBEDDING_MODEL_SLOTS_DEFAULT) -> int:
    if is_real_int(value):
        slots = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        slots = int(value)
    else:
        return default
    return max(EMBEDDING_MODEL_SLOTS_MIN, min(EMBEDDING_MODEL_SLOTS_MAX, slots))


def _runtime_config_default() -> dict:
    return {"embedding": {"model_slots": EMBEDDING_MODEL_SLOTS_DEFAULT}}


def _read_runtime_config_sync() -> dict:
    cfg = _runtime_config_default()
    try:
        raw = json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return cfg
    if not isinstance(raw, dict):
        return cfg
    embedding = raw.get("embedding")
    if not isinstance(embedding, dict):
        embedding = {}
    cfg.update({k: v for k, v in raw.items() if k != "embedding"})
    cfg["embedding"] = {
        **embedding,
        "model_slots": _coerce_embedding_model_slots(
            embedding.get("model_slots"),
            default=EMBEDDING_MODEL_SLOTS_DEFAULT,
        ),
    }
    return cfg


def _write_runtime_config_sync(cfg: dict) -> None:
    RUNTIME_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    with _runtime_config_file_lock:
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=RUNTIME_CONFIG_FILE.parent,
                prefix=f".{RUNTIME_CONFIG_FILE.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                json.dump(cfg, tmp, indent=2, ensure_ascii=False)
                tmp.write("\n")
            os.chmod(tmp_path, 0o640)
            os.replace(tmp_path, RUNTIME_CONFIG_FILE)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def _update_embedding_slots_config_sync(model_slots: int) -> dict:
    with _runtime_config_file_lock:
        cfg = _read_runtime_config_sync()
        embedding = cfg.get("embedding")
        if not isinstance(embedding, dict):
            embedding = {}
        embedding["model_slots"] = model_slots
        cfg["embedding"] = embedding
        _write_runtime_config_sync(cfg)
        return cfg


def _loaded_models_from_embedding_health(data: dict | None) -> set[str]:
    if not isinstance(data, dict):
        return set()
    raw_loaded = data.get("loaded_models")
    if isinstance(raw_loaded, list):
        return {item for item in raw_loaded if isinstance(item, str)}
    return set()


def _gpu_gate_response(decision: vram_lease.GPUGateDecision) -> JSONResponse:
    return JSONResponse(
        {
            "error": "GPU-Operation blockiert",
            "hint": "Eine andere GPU-Operation laeuft oder ihr Abschluss ist unklar.",
            "vram_lease": "active",
            "reason": decision.reason,
            "marker_kind": decision.marker_kind,
            "lease_state": decision.lease_state,
            "service": decision.service_name,
        },
        status_code=decision.status_code,
    )


async def _json_service_health(url: str, *, timeout: float = 2.0) -> tuple[bool, dict | None, int | None]:
    """Return process reachability separate from model readiness."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
    except Exception:
        return False, None, None
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return False, None, resp.status_code
    if not isinstance(data, dict):
        return False, None, resp.status_code
    status = data.get("status")
    reachable = (
        resp.status_code == 200
        or (
            resp.status_code == 503
            and status in ("no_model", "loading")
        )
    )
    return reachable, data, resp.status_code


async def _wait_service_reachable(url: str, deadline_s: float) -> tuple[bool, dict | None]:
    deadline = time.monotonic() + deadline_s
    while True:
        reachable, data, _ = await _json_service_health(
            url,
            timeout=min(2.0, max(0.1, deadline - time.monotonic())),
        )
        if reachable:
            return True, data
        if time.monotonic() >= deadline:
            return False, None
        await asyncio.sleep(vram_lease.GPU_SERVICE_HEALTH_POLL_S)


async def _restart_embedding_service_with_wait() -> tuple[bool, dict | None, str | None]:
    try:
        result = await _run_service_control("restart", EMBEDDING_SERVICE, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return False, None, f"Restart fehlgeschlagen: {exc}"
    except Exception as exc:
        return False, None, f"Restart fehlgeschlagen: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, None, f"Restart fehlgeschlagen: {detail}"
    ok, health = await _wait_service_reachable(
        EMBEDDING_HEALTH_URL,
        vram_lease.GPU_SERVICE_START_DEADLINE_S,
    )
    if not ok:
        return False, health, "Embedding-Service wurde neugestartet, ist aber nicht HTTP-bereit"
    return True, health, None


def _clear_marker_for_service_response(resp: httpx.Response, data_readable: bool) -> bool:
    if 200 <= resp.status_code < 300:
        return data_readable
    if 400 <= resp.status_code < 500:
        return data_readable
    return False


async def _db_to_thread(func, /, *args, **kwargs):
    if db_work_tracker is not None:
        return await db_work_tracker.to_thread(func, *args, **kwargs)
    return await _to_thread(func, *args, **kwargs)


MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]+(/[a-zA-Z0-9._:-]+)*$')

# Benchmark-Cache (persistiert als JSON-Datei)
BENCHMARKS_CACHE_FILE = PROXY_DATA_DIR / "benchmarks_cache.json"
BENCHMARK_REFRESH_TTL_S = 7 * 86400

# Statische Benchmarks fuer Modelle die nicht auf dem Open LLM Leaderboard sind
# Quelle: Offizielle Model Cards und Technical Reports
_STATIC_BENCHMARKS = {
    "qwen3": {
        "mmlu_pro": 56.7, "bbh": 78.4, "gpqa": 44.4, "ifeval": None, "humaneval": None,
        "source": "Qwen3 Technical Report (arXiv:2505.09388), Base-Model",
    },
    "qwen3.5": {
        "mmlu_pro": 82.5, "bbh": None, "gpqa": 81.7, "ifeval": 91.5, "humaneval": None,
        "source": "Qwen3.5-9B Model Card (HuggingFace), GPQA=Diamond",
    },
}

# Bekannte Mappings: Ollama-Basisname -> HuggingFace fullname
_OLLAMA_TO_HF_MAP = {
    "phi4": "microsoft/phi-4",
    "phi3": "microsoft/Phi-3-medium-128k-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "llama3": "meta-llama/Llama-3-8B-Instruct",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.3": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma2": "google/gemma-2-9b-it",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-v2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "codellama": "meta-llama/CodeLlama-7b-Instruct-hf",
    "starcoder2": "bigcode/starcoder2-15b",
    "command-r": "CohereForAI/c4ai-command-r-08-2024",
    "yi": "01-ai/Yi-1.5-9B-Chat",
    # Neuere Modelle
    "mistral-large": "mistralai/Mistral-Large-Instruct-2411",
    "mistral-large-3": "mistralai/Mistral-Large-Instruct-2411",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "ministral-3": "mistralai/Ministral-8B-Instruct-2410",
    "glm-4": "THUDM/glm-4-9b",
    "glm-4.6": "THUDM/glm-4-9b",
    "glm-4.7": "THUDM/glm-4-9b",
    "nemotron-3-nano": "nvidia/Nemotron-Mini-4B-Instruct",
    "nemotron-3-super": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "deepseek-r1:14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

# Bekannte Mappings: Ollama-Basisname -> EvalPlus Key
_OLLAMA_TO_EVALPLUS_MAP = {
    "phi3": "Phi-3-mini-4k-instruct",
    "llama3": "Llama3-8B-instruct",
    "llama3.1": "Llama3.1-8B-instruct",
    "llama3.3": "Llama3-70B-instruct",
    "qwen2.5-coder": "Qwen2.5-Coder-32B-Instruct",
    "deepseek-v2.5": "DeepSeek-V2.5 (Nov 2024)",
    "deepseek-v3": "DeepSeek-V3 (Nov 2024)",
    "deepseek-v3.1": "DeepSeek-V3 (Nov 2024)",
    "deepseek-v3.2": "DeepSeek-V3 (Nov 2024)",
    "deepseek-coder-v2": "DeepSeek-Coder-V2-Instruct",
    "codellama": "CodeLlama-7B",
    "mistral": "Mistral-7B-Instruct-v0.2",
    "mistral-large": "Mistral Large (Mar 2024)",
    "mistral-large-3": "Mistral Large (Mar 2024)",
}


_model_registration_service: ModelRegistrationService | None = None


def get_model_registration_service() -> ModelRegistrationService:
    global _model_registration_service
    if _model_registration_service is None:
        _model_registration_service = build_model_registration_service()
    return _model_registration_service


def set_model_registration_service(service: ModelRegistrationService) -> None:
    if not isinstance(service, ModelRegistrationService):
        raise TypeError("service must be a ModelRegistrationService")
    global _model_registration_service
    _model_registration_service = service


def set_store(request_store: RequestStore):
    global store
    store = request_store


def set_metrics_db(db: MetricsDB):
    global metrics_db
    metrics_db = db


def set_high_res_history_service(service: HighResHistoryService | None) -> None:
    global high_res_history_service
    high_res_history_service = service


def set_db_work_tracker(tracker):
    global db_work_tracker
    db_work_tracker = tracker


def set_api_key_store(aks):
    global api_key_store
    api_key_store = aks


async def _runtime_config_response_payload() -> dict:
    cfg = await _to_thread(_read_runtime_config_sync)
    desired_slots = _coerce_embedding_model_slots(
        cfg.get("embedding", {}).get("model_slots")
        if isinstance(cfg.get("embedding"), dict)
        else None
    )
    reachable, health, _ = await _json_service_health(EMBEDDING_HEALTH_URL)
    effective_slots = None
    loaded_models: list[str] = []
    loading_model = None
    current_model = None
    status = "down"
    if reachable and isinstance(health, dict):
        status = health.get("status") if isinstance(health.get("status"), str) else "unknown"
        raw_slots = health.get("model_slots")
        if isinstance(raw_slots, int) and not isinstance(raw_slots, bool):
            effective_slots = raw_slots
        current = health.get("current_model")
        current_model = current if isinstance(current, str) else None
        loading = health.get("loading_model")
        loading_model = loading if isinstance(loading, str) else None
        loaded_models = sorted(_loaded_models_from_embedding_health(health))
    return {
        "embedding": {
            "model_slots": desired_slots,
            "effective_model_slots": effective_slots,
            "pending_restart": effective_slots is not None and effective_slots != desired_slots,
            "service_running": reachable,
            "service_status": status,
            "current_model": current_model,
            "loaded_models": loaded_models,
            "loading_model": loading_model,
            "min_model_slots": EMBEDDING_MODEL_SLOTS_MIN,
            "max_model_slots": EMBEDDING_MODEL_SLOTS_MAX,
            "default_model_slots": EMBEDDING_MODEL_SLOTS_DEFAULT,
        }
    }


@app.get("/api/config/runtime")
async def get_runtime_config():
    return await _runtime_config_response_payload()


@app.post("/api/config/embedding")
async def update_embedding_config(body: dict | None = Body(default=None)):
    body = body if isinstance(body, dict) else {}
    raw_slots = body.get("model_slots")
    if not is_real_int(raw_slots):
        return JSONResponse(
            {"error": "model_slots muss eine ganze Zahl sein."},
            status_code=400,
        )
    model_slots = int(raw_slots)
    if not EMBEDDING_MODEL_SLOTS_MIN <= model_slots <= EMBEDDING_MODEL_SLOTS_MAX:
        return JSONResponse(
            {
                "error": (
                    f"model_slots muss zwischen {EMBEDDING_MODEL_SLOTS_MIN} "
                    f"und {EMBEDDING_MODEL_SLOTS_MAX} liegen."
                )
            },
            status_code=400,
        )

    try:
        await _to_thread(_update_embedding_slots_config_sync, model_slots)
    except OSError as exc:
        return JSONResponse(
            {"error": f"Runtime-Konfiguration konnte nicht gespeichert werden: {exc}"},
            status_code=500,
        )

    restart = body.get("restart") is True
    restart_info = {"requested": restart, "ok": None, "error": None}
    if restart:
        ok, health, error = await _restart_embedding_service_with_wait()
        restart_info = {"requested": True, "ok": ok, "error": error}
        if not ok:
            payload = await _runtime_config_response_payload()
            payload["restart"] = restart_info
            if health is not None:
                payload["restart"]["health"] = health
            return JSONResponse(payload, status_code=500)

    payload = await _runtime_config_response_payload()
    payload["restart"] = restart_info
    return payload


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Statische Dateien: Browser muss immer beim Server revalidieren."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ============================================================
# Dashboard-Authentifizierung (#1023/#1024/#1025)
# ============================================================
# Das Dashboard inkl. aller mutierenden /api-Endpunkte (apikeys-CRUD,
# /api/models/delete, /api/maintenance/toggle, service start/stop ...) lag
# ungeschuetzt auf 0.0.0.0:8505 — jeder LAN-Client konnte schreiben. Single-
# Tenant-LAN -> HTTP-Basic ueber die GANZE App. Default admin/admin, per Env
# (KIRON_DASHBOARD_USER / KIRON_DASHBOARD_PASSWORD) ueberschreibbar.
DASHBOARD_AUTH_USER = os.environ.get("KIRON_DASHBOARD_USER", "admin")
DASHBOARD_AUTH_PASSWORD = os.environ.get("KIRON_DASHBOARD_PASSWORD", "admin")
_DASHBOARD_AUTH_REALM = "Kiron Dashboard"


def _check_basic_auth(header_value: str) -> bool:
    """Authorization-Header gegen die Dashboard-Credentials pruefen."""
    scheme, _, encoded = (header_value or "").partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    user, sep, password = decoded.partition(":")
    if not sep:
        return False
    # Beide compare_digest-Aufrufe laufen immer (kein Timing-Orakel ueber das `and`).
    user_ok = secrets.compare_digest(user, DASHBOARD_AUTH_USER)
    pw_ok = secrets.compare_digest(password, DASHBOARD_AUTH_PASSWORD)
    return user_ok and pw_ok


class BasicAuthMiddleware:
    """Reine ASGI-Middleware: HTTP-Basic fuer das gesamte Dashboard.

    Kein BaseHTTPMiddleware, damit neben http auch der /ws/live-WebSocket-
    Handshake abgesichert ist. lifespan & sonstige Scopes passieren ungehindert.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or ())
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        if _check_basic_auth(auth_header):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # Handshake ablehnen, bevor accept() laeuft (1008 = policy violation).
            await send({"type": "websocket.close", "code": 1008})
            return

        body = b'{"error":"Authentifizierung erforderlich"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (
                        b"www-authenticate",
                        f'Basic realm="{_DASHBOARD_AUTH_REALM}"'.encode("latin-1"),
                    ),
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


app.add_middleware(BasicAuthMiddleware)


# ============================================================
# API-Endpunkte (muessen VOR dem Catch-All definiert werden)
# ============================================================

@app.get("/api/requests/recent")
async def get_recent_requests(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0, le=1000000),
    ip: str = Query(default=None, max_length=64, regex="^[0-9a-fA-F:.]+$"),
    model: str = Query(default=None, max_length=256),
    status: str = Query(default=None, regex="^(2xx|4xx|5xx)$"),
    active_only: bool = Query(default=False),
):
    """Letzte Requests mit optionalen Filtern."""
    if store is None:
        return JSONResponse({"error": "Store nicht initialisiert"}, status_code=503)
    try:
        results, total = await store.get_recent(
            limit=limit,
            offset=offset,
            ip_filter=ip,
            model_filter=model,
            status_filter=status,
            active_only=active_only,
        )
    except RequestStoreReadError:
        return JSONResponse({"error": "Request-Store nicht lesbar"}, status_code=503)
    return {"requests": results, "total": total}


@app.get("/api/requests/active")
async def get_active_requests():
    """Alle aktuell laufenden Requests."""
    if store is None:
        return JSONResponse({"error": "Store nicht initialisiert"}, status_code=503)
    results = store.get_active()
    return {"requests": results}


@app.get("/api/requests/{request_id}")
async def get_request_detail(request_id: str):
    """Einzelnen Request mit vollem Body (Prompt + Antwort) laden."""
    if store is None:
        return JSONResponse({"error": "Store nicht initialisiert"}, status_code=503)
    try:
        detail = await store.get_request_detail(request_id)
    except RequestStoreReadError:
        return JSONResponse({"error": "Request-Store nicht lesbar"}, status_code=503)
    if detail is None:
        return JSONResponse({"error": "Request nicht gefunden"}, status_code=404)
    return detail


@app.get("/api/metrics/system")
async def get_system_metrics():
    """Alle System-Metriken (CPU, RAM, GPU, Disk I/O, Ollama)."""
    payload = get_cached_payload()
    if payload is None:
        return JSONResponse(
            {"error": "metrics cache not ready", "retry_after_s": 2},
            status_code=503,
        )
    system_metrics = dict(payload["system"])
    system_metrics["maintenance"] = _maintenance_status_snapshot()
    return system_metrics


@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Zusammenfassung fuer Dashboard-KPIs."""
    if store is None:
        return JSONResponse({"error": "Store nicht initialisiert"}, status_code=503)
    summary = await store.get_summary()
    return summary


@app.get("/api/dashboard/info")
async def dashboard_info():
    """Dashboard-Informationen."""
    version = "0.0.0"
    version_file = project_root / "version.txt"
    if version_file.exists():
        version = version_file.read_text().strip()

    return {
        "version": version,
        "title": "Ollama Monitor",
    }


@app.post("/api/dashboard/restart")
async def restart_dashboard():
    """Startet den Dashboard-Service neu."""
    try:
        subprocess.Popen(
            _service_control_command("restart", "kiron-proxy.service"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        return JSONResponse(
            {"status": "error", "message": f"Restart fehlgeschlagen: {e}"},
            status_code=500,
        )
    return {"status": "restarting", "message": "Kiron-Proxy wird neu gestartet..."}


# ============================================================
# History API-Endpunkte
# ============================================================

@app.get("/api/history/metrics")
async def get_history_metrics(
    metrics: str = Query(default="cpu_usage,memory_usage", max_length=2000),
    from_ts: float = Query(default=None, alias="from"),
    to_ts: float = Query(default=None, alias="to"),
    max_points: int = Query(default=500, ge=10, le=2000),
):
    """Historische Metriken als Zeitreihen."""
    if metrics_db is None:
        return {"timestamps": [], "series": {}}

    now = time.time()
    if from_ts is None:
        from_ts = now - 86400  # Default: letzte 24h
    if to_ts is None:
        to_ts = now

    metric_names = [m.strip() for m in metrics.split(",") if m.strip()][:50]
    result = await _db_to_thread(
        metrics_db.query_range, metric_names, from_ts, to_ts, max_points
    )
    return result


@app.get("/api/history/stats")
async def get_history_stats():
    """DB-Statistiken fuer historische Metriken."""
    if metrics_db is None:
        return {"total_rows": 0, "db_size_mb": 0}

    stats = await _db_to_thread(metrics_db.get_db_stats)
    return stats


# ============================================================
# Model Manager API-Endpunkte
# ============================================================

def _validate_model_name(name: str) -> bool:
    """Prueft ob ein Modellname gueltig ist."""
    if not isinstance(name, str):
        return False
    return bool(name and MODEL_NAME_PATTERN.fullmatch(name))


def _json_backend_error(message: str, status_code: int = 502, code: str = "backend_error") -> JSONResponse:
    return JSONResponse({"error": message, "code": code}, status_code=status_code)


def _optional_int(value):
    return value if is_real_int(value) else None


def _canonical_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def _parse_ps_models(raw: object) -> tuple[dict[str, dict], bool]:
    """Return loaded model map and known-state flag."""
    if not isinstance(raw, dict):
        return {}, False
    models = raw.get("models")
    if not isinstance(models, list):
        return {}, False
    loaded: dict[str, dict] = {}
    for item in models:
        if not isinstance(item, dict):
            return {}, False
        name = item.get("name")
        if not isinstance(name, str) or not name:
            return {}, False
        loaded[name] = item
    return loaded, True


async def _verify_model_loaded(
    client: httpx.AsyncClient,
    name: str,
    *,
    require_cpu: bool = False,
    require_gpu: bool = False,
) -> JSONResponse | None:
    try:
        verify_resp = await client.get("/api/ps")
    except httpx.RequestError as exc:
        return _json_backend_error(f"Ollama /api/ps Verify fehlgeschlagen: {exc}")
    if verify_resp.status_code != 200:
        return _json_backend_error(
            f"Ollama /api/ps Verify lieferte HTTP {verify_resp.status_code}"
        )
    try:
        loaded, known = _parse_ps_models(verify_resp.json())
    except json.JSONDecodeError:
        return _json_backend_error("Ollama /api/ps Verify lieferte ungueltiges JSON")
    if not known:
        return _json_backend_error("Ollama /api/ps Verify-Shape ist unbekannt")
    canonical = _canonical_model_name(name)
    entry = loaded.get(name) or loaded.get(canonical)
    if entry is None:
        return _json_backend_error("Load konnte nicht per /api/ps bestaetigt werden")
    size_vram = entry.get("size_vram")
    if not is_real_int(size_vram):
        return _json_backend_error("Load-Verify hat keinen echten integer size_vram")
    if require_cpu and size_vram != 0:
        return _json_backend_error("CPU-Offload wurde nicht per /api/ps size_vram==0 bestaetigt")
    if require_gpu and size_vram <= 0:
        return _json_backend_error("GPU-Load wurde nicht per /api/ps size_vram>0 bestaetigt")
    return None


UNLOAD_VERIFY_DEADLINE_S = 2.0
UNLOAD_VERIFY_POLL_INTERVAL_S = 0.1


async def _verify_model_unloaded(client: httpx.AsyncClient, name: str) -> JSONResponse | None:
    canonical = _canonical_model_name(name)
    deadline = time.monotonic() + UNLOAD_VERIFY_DEADLINE_S
    while True:
        try:
            verify_resp = await client.get("/api/ps")
        except httpx.RequestError as exc:
            return _json_backend_error(f"Ollama /api/ps Unload-Verify fehlgeschlagen: {exc}")
        if verify_resp.status_code != 200:
            return _json_backend_error(
                f"Ollama /api/ps Unload-Verify lieferte HTTP {verify_resp.status_code}"
            )
        try:
            loaded, known = _parse_ps_models(verify_resp.json())
        except json.JSONDecodeError:
            return _json_backend_error("Ollama /api/ps Unload-Verify lieferte ungueltiges JSON")
        if not known:
            return _json_backend_error("Ollama /api/ps Unload-Verify-Shape ist unbekannt")
        if name not in loaded and canonical not in loaded:
            return None
        if time.monotonic() >= deadline:
            return _json_backend_error("Unload konnte nicht per /api/ps bestaetigt werden")
        await asyncio.sleep(UNLOAD_VERIFY_POLL_INTERVAL_S)


_REGISTRATION_STATUS_BY_CODE = {
    "invalid_provider": 400,
    "invalid_reference": 400,
    "invalid_loader": 400,
    "model_not_found": 404,
    "duplicate_model": 409,
    "loader_metadata_invalid": 422,
    "local_validation_failed": 502,
    "registry_corrupt": 500,
    "registry_access_failed": 503,
}


def _registration_error_response(error: ModelRegistrationError) -> JSONResponse:
    return JSONResponse(
        {
            "status": "error",
            "error": {"code": error.code, "message": str(error)},
        },
        status_code=_REGISTRATION_STATUS_BY_CODE.get(error.code, 500),
    )


def _catalog_ollama_reference_keys() -> frozenset[str]:
    keys: set[str] = set()
    for definition in MODEL_STATE_VIEW.for_backend(BackendType.OLLAMA):
        for name in definition.input_names:
            canonical = canonical_ollama_reference(name)
            if canonical is not None:
                keys.add(canonical.lower())
    return frozenset(keys)


@app.get("/api/models/registration-candidates")
async def get_model_registration_candidates():
    try:
        candidates = await _to_thread(
            get_model_registration_service().list_candidates
        )
    except ModelRegistrationError as exc:
        return _registration_error_response(exc)
    catalog_ollama = _catalog_ollama_reference_keys()
    visible = [
        dashboard_runtime.public_candidate(candidate)
        for candidate in candidates.candidates
        if not (
            candidate.runtime_provider is BackendType.OLLAMA
            and candidate.reference.lower() in catalog_ollama
        )
    ]
    return {"status": "ok", "candidates": visible,
            "runtime_profiles": [{"id": name, "projector_supported": policy.projector_sha256 is not None}
                                 for name, policy in get_model_registration_service().registration_profiles().items()],
            "errors": {provider.value: code for provider, code in candidates.errors.items()}}


@app.post("/api/models/register")
async def register_local_model(request: Request, body: object = Body(...)):
    if not dashboard_runtime.mutation_allowed(request):
        return JSONResponse({"error": {"code": "csrf_rejected", "message": "Aktion nur vom eigenen Dashboard erlaubt."}}, status_code=403)
    if (type(body) is not dict or not set(body).issubset(
            {"candidate_id", "loader", "projector_id", "runtime_profile"})
            or type(body.get("candidate_id")) is not str):
        return dashboard_runtime.error_response("invalid_request")
    try:
        service = get_model_registration_service()
        discovery = await _to_thread(service.list_candidates)
        candidates = {dashboard_runtime.candidate_id(item): item for item in discovery.candidates}
        candidate = candidates.get(body["candidate_id"])
        projector_id = body.get("projector_id")
        if projector_id is not None and type(projector_id) is not str:
            return dashboard_runtime.error_response("invalid_request")
        projector = candidates.get(projector_id) if projector_id is not None else None
        if (candidate is None or (projector_id is not None and (projector is None
                or projector.runtime_provider is not candidate.runtime_provider))):
            return dashboard_runtime.error_response("model_not_found")
        entry = await _to_thread(service.register_model,
            runtime_provider=candidate.runtime_provider, reference=candidate.reference,
            loader=body.get("loader"), runtime_profile=body.get("runtime_profile"),
            projector_reference=projector.reference if projector else None)
    except ModelRegistrationError as exc:
        return _registration_error_response(exc)
    return JSONResponse({"status": "registered", "model": dashboard_runtime.public_registration(entry)}, status_code=201)


@app.get("/api/models/runtime")
async def get_runtime_models(request: Request):
    return await dashboard_runtime.runtime_inventory(request)


@app.post("/api/models/runtime/action")
async def runtime_model_action(request: Request, body: object = Body(...)):
    return await dashboard_runtime.runtime_action(request, body)


def _model_control():
    return ModelControl(
        read_native=get_local_models, read_runtime=dashboard_runtime.runtime_inventory,
        runtime_action=dashboard_runtime.runtime_action,
        cpu_verified=vram_lease.num_gpu_zero_effective,
        read_ollama_state=lambda: ollama_recovery.status(ollama_admission_store()),
        handlers={"ollama_load": load_model, "ollama_cpu": load_model,
                  "ollama_unload": unload_model, "ollama_delete": delete_model,
                  "embedding_load": load_embedding_model, "embedding_unload": unload_embedding_model,
                  "embedding_warmup": warmup_colbert_model,
                  "deberta_load": load_deberta_model, "deberta_unload": unload_deberta_model},
    )


@app.get("/api/models/control")
async def get_model_controls(request: Request):
    payload, _ = await _model_control().inventory(request)
    return payload


@app.post("/api/models/control/action")
async def control_model(request: Request, body: object = Body(...)):
    return await _model_control().execute(request, body)


@app.get("/api/ollama/recovery")
async def ollama_recovery_status():
    return await _to_thread(ollama_recovery.status, ollama_admission_store())


@app.post("/api/ollama/recovery")
async def recover_ollama(request: Request, body: object = Body(...)):
    if not dashboard_runtime.mutation_allowed(request):
        return JSONResponse({"error": {"code": "csrf_rejected", "message": "Aktion nur vom eigenen Dashboard erlaubt."}}, status_code=403)
    if (type(body) is not dict or set(body) != {"revision"} or type(body["revision"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", body["revision"])):
        return dashboard_runtime.error_response("invalid_request")
    try:
        return await ollama_recovery.recover(ollama_admission_store(), body["revision"])
    except AdmissionError as exc:
        return JSONResponse({"error": {"code": exc.code, "message": str(exc)}}, status_code=409)
    except (OSError, ValueError):
        return JSONResponse({"error": {"code": "ollama_control_failed", "message": "Ollama-Wiederherstellung konnte nicht bestätigt werden."}}, status_code=503)


@app.get("/api/models/local")
async def get_local_models():
    """Catalog configuration plus injected local and atomic runtime inventory."""
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10) as client:
            tags_resp, ps_resp = await asyncio.gather(
                client.get("/api/tags"),
                client.get("/api/ps"),
                return_exceptions=True,
            )

        if isinstance(tags_resp, Exception):
            raise tags_resp
        if isinstance(ps_resp, Exception):
            ps_resp = None

        if tags_resp.status_code != 200:
            return JSONResponse(
                {"error": f"Ollama tags-Endpunkt lieferte HTTP {tags_resp.status_code}"},
                status_code=502,
            )
        try:
            tags_data = tags_resp.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "Ungueltige JSON-Antwort von Ollama"}, status_code=502)
        if not isinstance(tags_data, dict) or not isinstance(tags_data.get("models"), list):
            return JSONResponse({"error": "Ungueltige /api/tags-Shape von Ollama"}, status_code=502)

        async with httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=10,
        ) as show_client:
            async def _read_show(name: str):
                try:
                    response = await show_client.post(
                        "/api/show",
                        json={"model": name},
                    )
                    if response.status_code != 200:
                        return name, None
                    payload = response.json()
                    return name, payload if isinstance(payload, dict) else None
                except (httpx.RequestError, ValueError, TypeError):
                    return name, None

            show_results = await asyncio.gather(
                *(
                    _read_show(row["name"])
                    for row in tags_data["models"]
                    if isinstance(row, dict)
                    and type(row.get("name")) is str
                    and row["name"]
                )
            )
        show_by_name = {
            name: payload
            for name, payload in show_results
            if payload is not None
        }
        if ps_resp is not None and ps_resp.status_code == 200:
            try:
                ps_data = ps_resp.json()
            except json.JSONDecodeError:
                ps_data = None
        else:
            ps_data = None

        cached = get_cached_payload()
        system = cached.get("system") if isinstance(cached, dict) else None
        system = system if isinstance(system, dict) else {}
        embed_data = system.get("embedding")
        deberta_data = system.get("deberta")
        embed_reachable = (
            isinstance(embed_data, dict) and embed_data.get("running") is True
        )
        deberta_reachable = (
            isinstance(deberta_data, dict) and deberta_data.get("running") is True
        )
        service_reachability = {
            BackendType.KIRON_EMBEDDINGS: embed_reachable,
            BackendType.KIRON_DEBERTA: deberta_reachable,
        }
        reachable_backends = tuple(
            backend
            for backend in (
                BackendType.KIRON_EMBEDDINGS,
                BackendType.KIRON_DEBERTA,
            )
            if service_reachability[backend]
        )
        try:
            hf_revisions = service_huggingface_inventory(
                MODEL_STATE_VIEW,
                {
                    BackendType.KIRON_EMBEDDINGS: embed_data,
                    BackendType.KIRON_DEBERTA: deberta_data,
                },
                reachable_backends=reachable_backends,
            )
        except ServiceInventoryError as exc:
            return JSONResponse(
                {
                    "error": f"Verwaltetes Service-Inventar inkonsistent: {exc}",
                    "catalog_digest": MODEL_STATE_VIEW.catalog_digest,
                },
                status_code=503,
            )
        try:
            registrations = await _to_thread(
                get_model_registration_service().list_models
            )
        except ModelRegistrationError as exc:
            return _registration_error_response(exc)
        try:
            payload = build_local_models_payload(
                state_view=MODEL_STATE_VIEW,
                huggingface_revisions=hf_revisions,
                unavailable_huggingface_backends=frozenset(
                    backend
                    for backend, reachable in service_reachability.items()
                    if not reachable
                ),
                ollama_tag_rows=tags_data["models"],
                ollama_ps=ps_data,
                embedding_health=embed_data,
                embedding_reachable=embed_reachable,
                deberta_health=deberta_data,
                deberta_reachable=deberta_reachable,
                registrations=registrations,
                ollama_show_by_name=show_by_name,
                service_memory=system.get("service_memory"),
            )
            for row in payload.get("models", []):
                row.pop("reference", None)
            return payload
        except InventoryShapeError as exc:
            return JSONResponse(
                {"error": f"Ungueltiges lokales Modellinventar: {exc}"},
                status_code=502,
            )

    except httpx.ConnectError:
        return JSONResponse({"error": "Ollama nicht erreichbar"}, status_code=503)
    except httpx.TimeoutException:
        return JSONResponse({"error": "Ollama Timeout"}, status_code=504)
    except httpx.RequestError as exc:
        return _json_backend_error(f"Ollama Backend-Fehler: {exc}")


@app.post("/api/embedding/load")
async def load_embedding_model(body: dict):
    """Embedding-Modell wechseln (Proxy zu kiron-embeddings)."""
    model = body.get("model", "")
    if not _validate_model_name(model):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)
    force = body.get("force") is True
    async with vram_lease.gpu_service_operation(
        force=force,
        service_name="Embedding-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    EMBEDDING_LOAD_URL,
                    json={"model": model},
                    headers={OPERATION_HEADER: op.admission_id, OVERLAY_HEADER: op.token},
                )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom Embedding-Service"}, status_code=502)
            op.clear_marker = resp.headers.get(COMPLETION_HEADER) == op.admission_id
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "Embedding-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Embedding-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Embedding-Service Backend-Fehler: {exc}")


@app.post("/api/embedding/unload")
async def unload_embedding_model(body: dict):
    """Entlade genau ein Modell ueber den seriellen Embedding-Worker."""
    model = body.get("model")
    if set(body) != {"model"} or not _validate_model_name(model):
        return JSONResponse({"error": "Genau ein gueltiger Modellname ist erforderlich."}, status_code=400)
    route = next((route for route in PROXY_ROUTING_VIEW.routes
                  if route.backend is BackendType.KIRON_EMBEDDINGS and model in route.input_names), None)
    if route is None:
        return JSONResponse({"error": "Unbekanntes Embedding-Modell"}, status_code=400)
    async with vram_lease.gpu_service_operation(
        releases_resources=True, service_name="Embedding-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(EMBEDDING_UNLOAD_URL, json={"model": model})
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom Embedding-Service"}, status_code=502)
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            if 200 <= status < 300 and (type(data) is not dict or data.get("status") != "unloaded"
                    or data.get("model") != route.backend_model_name or type(data.get("already")) is not bool):
                return JSONResponse({"error": "Embedding-Service hat das Entladen nicht bestaetigt."}, status_code=502)
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            return JSONResponse({"error": "Embedding-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Embedding-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Embedding-Service Backend-Fehler: {exc}")


@app.post("/api/embedding/colbert/warmup")
async def warmup_colbert_model(body: dict):
    """ColBERT-Modell ueber den Token-Level-Endpoint warm laufen lassen."""
    model = body.get("model", "")
    if not _validate_model_name(model):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)
    force = body.get("force") is True
    async with vram_lease.gpu_service_operation(
        force=force,
        service_name="Embedding-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    "http://127.0.0.1:11436/api/embed_colbert",
                    json={
                        "model": model,
                        "input": ["warmup"],
                        "language": "de",
                    },
                )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    {"error": "Ungueltige JSON-Antwort vom Embedding-Service"},
                    status_code=502,
                )
            op.clear_marker = _clear_marker_for_service_response(resp, True)
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "Embedding-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Embedding-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Embedding-Service Backend-Fehler: {exc}")


@app.post("/api/models/load")
async def load_model(body: dict):
    """Modell in Speicher laden (optional CPU-only via gpu=false)."""
    name = body.get("name", "")
    if not _validate_model_name(name):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)

    gpu_value = body.get("gpu", True)
    if not isinstance(gpu_value, bool):
        return JSONResponse(
            {"error": "gpu muss boolean sein (true/false)"},
            status_code=400,
        )
    use_gpu = gpu_value
    force = body.get("force") is True

    payload = {"model": name, "prompt": "", "stream": False}
    if not use_gpu:
        effective_gate_active = await vram_lease.effective_snapshot()
        if not vram_lease.num_gpu_zero_effective():
            return JSONResponse(
                {
                    "error": "CPU-Offload ist fuer diese Ollama-Version nicht verifiziert",
                    "hint": "Gruenen Ollama-Compat-Report deployen oder spaeter erneut versuchen.",
                    "vram_lease": "active" if effective_gate_active else "unknown",
                },
                status_code=409,
        )
        payload["options"] = {"num_gpu": 0}

    async def _load_under_marker(op: vram_lease.GPUServiceOperation | None = None):
        response_received = load_ended = False
        try:
            operation_id = secrets.token_hex(16)
            async with OllamaLifecycleOperation(store=ollama_admission_store(),
                    owner="kiron-dashboard-load", generation="dashboard-load:" + operation_id,
                    operation_id=operation_id, deployment_id="dashboard-load:" + operation_id,
                    deadline_monotonic=time.monotonic() + 180) as lifecycle:
                async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=120) as client:
                    lifecycle.mark_started()
                    try:
                        resp = await client.post("/api/generate", json=payload)
                    except (httpx.ConnectError, httpx.PoolTimeout):
                        lifecycle.confirm_end()  # No backend request was sent.
                        raise
                    response_received = True
                    if resp.status_code != 200:
                        error_data = {}
                        data_readable = False
                        if resp.headers.get("content-type", "").startswith("application/json"):
                            try:
                                error_data = resp.json()
                                data_readable = isinstance(error_data, dict)
                            except json.JSONDecodeError:
                                error_data = {}
                        if op is not None:
                            op.clear_marker = _clear_marker_for_service_response(resp, data_readable)
                        status = resp.status_code if resp.status_code >= 400 else 502
                        return JSONResponse(
                            {"error": error_data.get("error", f"Ollama-Fehler (HTTP {resp.status_code})")},
                            status_code=status,
                        )
                    try:
                        load_result = resp.json()
                        load_ended = isinstance(load_result, dict) and load_result.get("done") is True
                    except (ValueError, UnicodeError):
                        load_ended = False
                    verify_error = await _verify_model_loaded(
                        client,
                        name,
                        require_cpu=not use_gpu,
                        require_gpu=use_gpu,
                    )
                    if verify_error is not None:
                        return verify_error
                if not load_ended:
                    return _json_backend_error("Ollama bestaetigte keinen vollstaendigen Ladeabschluss")
                lifecycle.confirm_end()
            if op is not None:
                op.clear_marker = load_ended
            return {"status": "loaded", "model": name, "gpu": use_gpu}

        except AdmissionError as exc:
            if op is not None and "lifecycle" in locals() and not lifecycle.started:
                op.clear_marker = True
            return JSONResponse({"error": ollama_recovery.conflict_message(ollama_admission_store(), exc), "code": exc.code}, status_code=409)
        except httpx.ConnectError:
            if op is not None:
                op.clear_marker = not response_received or load_ended
            return JSONResponse({"error": "Ollama nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Timeout beim Laden"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Ollama Backend-Fehler beim Laden: {exc}")

    if not use_gpu:
        return await _load_under_marker(None)

    async with vram_lease.gpu_service_operation(
        force=force,
        service_name="Ollama",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        return await _load_under_marker(op)


@app.post("/api/models/unload")
async def unload_model(body: dict):
    """Modell aus Speicher entladen."""
    name = body.get("name", "")
    if not _validate_model_name(name):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)
    force = body.get("force") is True

    async with vram_lease.gpu_service_operation(
        releases_resources=True,
        force=force,
        service_name="Ollama",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            operation_id = secrets.token_hex(16)
            async with OllamaLifecycleOperation(store=ollama_admission_store(),
                    owner="kiron-dashboard-unload", generation="dashboard:" + operation_id,
                    operation_id=operation_id, deployment_id="dashboard:" + operation_id,
                    deadline_monotonic=time.monotonic() + 60) as lifecycle:
                async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30) as client:
                    lifecycle.mark_started()
                    try:
                        resp = await client.post("/api/generate", json={
                            "model": name, "keep_alive": 0, "stream": False,
                        })
                    except (httpx.ConnectError, httpx.PoolTimeout):
                        lifecycle.confirm_end()  # No backend request was sent.
                        raise
                    if not (200 <= resp.status_code < 300):
                        data_readable = False
                        if resp.headers.get("content-type", "").startswith("application/json"):
                            try:
                                data_readable = isinstance(resp.json(), dict)
                            except json.JSONDecodeError:
                                data_readable = False
                        op.clear_marker = _clear_marker_for_service_response(resp, data_readable)
                        return JSONResponse(
                            {"error": f"Ollama-Fehler beim Entladen (HTTP {resp.status_code})"},
                            status_code=resp.status_code,
                        )
                    verify_error = await _verify_model_unloaded(client, name)
                    if verify_error is not None:
                        return verify_error
                    lifecycle.confirm_end()
            op.clear_marker = True
            return {"status": "unloaded", "model": name}

        except AdmissionError as exc:
            if "lifecycle" in locals() and not lifecycle.started:
                op.clear_marker = True
            return JSONResponse({"error": ollama_recovery.conflict_message(ollama_admission_store(), exc), "code": exc.code}, status_code=409)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "Ollama nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Timeout beim Entladen"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Ollama Backend-Fehler beim Entladen: {exc}")


@app.delete("/api/models/delete")
async def delete_model(body: dict):
    """Modell loeschen."""
    name = body.get("name", "")
    force = body.get("force") is True
    if not _validate_model_name(name):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)

    # #809: Loaded-Check und DELETE unter dem gleichen gpu_service_operation-Lock,
    # damit ein paralleler /api/models/load nicht zwischen ps-Check und DELETE
    # das Modell wieder in den Speicher bringt und die VRAM-Allokation
    # ohne Cleanup gerissen wird.
    async with vram_lease.gpu_service_operation(
        releases_resources=True,
        serialize_release=True,
        force=force,
        service_name="Ollama",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            operation_id = secrets.token_hex(16)
            async with OllamaLifecycleOperation(store=ollama_admission_store(),
                    owner="kiron-dashboard-delete", generation="dashboard-delete:" + operation_id,
                    operation_id=operation_id, deployment_id="dashboard-delete:" + operation_id,
                    deadline_monotonic=time.monotonic() + 60) as lifecycle:
                async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10) as client:
                    # Pruefen ob Modell geladen ist (#272: fail-close bei ps-Fehler)
                    ps_resp = await client.get("/api/ps")
                    loaded_map: dict[str, dict] = {}
                    ps_known = False
                    if ps_resp.status_code == 200:
                        try:
                            loaded_map, ps_known = _parse_ps_models(ps_resp.json())
                        except json.JSONDecodeError:
                            ps_known = False

                    # #272: Wenn /api/ps nicht verlaesslich ausgewertet werden kann, NICHT blind loeschen.
                    # Ohne force=true wird ein transienter Fehler auf 503 abgebildet, damit geladene
                    # Modelle nicht versehentlich aus dem Speicher gerissen werden.
                    if not ps_known and not force:
                        op.clear_marker = True
                        return JSONResponse({
                            "error": "Loaded-Check fehlgeschlagen",
                            "hint": "Ollama /api/ps nicht verlaesslich erreichbar. "
                                    "Retry oder force=true verwenden.",
                        }, status_code=503)

                    canonical = _canonical_model_name(name)
                    if (name in loaded_map or canonical in loaded_map) and not force:
                        op.clear_marker = True
                        return JSONResponse({
                            "error": "Modell ist geladen",
                            "loaded": True,
                            "hint": "Mit force=true trotzdem loeschen oder zuerst entladen",
                        }, status_code=409)

                    # Loeschen
                    lifecycle.mark_started()
                    try:
                        del_resp = await client.request("DELETE", "/api/delete", json={"model": name})
                    except (httpx.ConnectError, httpx.PoolTimeout):
                        lifecycle.confirm_end()  # No backend request was sent.
                        raise

                    if del_resp.status_code == 404:
                        lifecycle.confirm_end()
                        op.clear_marker = True
                        return JSONResponse({"error": "Modell nicht gefunden"}, status_code=404)
                    if not (200 <= del_resp.status_code < 300):
                        data_readable = False
                        if del_resp.headers.get("content-type", "").startswith("application/json"):
                            try:
                                data_readable = isinstance(del_resp.json(), dict)
                            except json.JSONDecodeError:
                                data_readable = False
                        op.clear_marker = _clear_marker_for_service_response(del_resp, data_readable)
                        return JSONResponse(
                            {"error": f"Ollama-Fehler beim Loeschen (HTTP {del_resp.status_code})"},
                            status_code=del_resp.status_code,
                        )

                    lifecycle.confirm_end()
                    op.clear_marker = True
                    return {"status": "deleted", "model": name}

        except AdmissionError as exc:
            if "lifecycle" in locals() and not lifecycle.started:
                op.clear_marker = True
            return JSONResponse({"error": ollama_recovery.conflict_message(ollama_admission_store(), exc), "code": exc.code}, status_code=409)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "Ollama nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Ollama Backend-Fehler beim Loeschen: {exc}")


# =============================================
# Docling Container-Steuerung
# =============================================

DOCLING_IMAGE = "quay.io/docling-project/docling-serve-cu130:v1.14.0"
DOCLING_CONTAINER = "docling-serve"

_docling_ops_lock = asyncio.Lock()
_docling_device_lock = _docling_ops_lock

# Docling Drain-Koordination (#278) — vor Device-Switch / Stop auf
# aktive Konvertierungen warten. LAN-only Lokal-Endpoint (kein Auth,
# Annahme wie Issue #260).
DOCLING_LIFECYCLE_URL = "http://127.0.0.1:5001/_internal/lifecycle"
DOCLING_START_URL = "http://127.0.0.1:5001/_internal/start"
DOCLING_STOP_URL = "http://127.0.0.1:5001/_internal/stop"
DRAIN_DEADLINE_S = 60.0
DRAIN_POLL_INTERVAL_S = 1.0
# docling-proxy `_wait_healthy` deckelt bei `HEALTH_CHECK_TIMEOUT_S=180`,
# plus VRAM-Free + docker start; Buffer fuer langsame Coldstarts (#560).
DOCLING_START_DEADLINE_S = 240.0
DOCLING_STOP_DEADLINE_S = 60.0


async def _gpu_service_lease_guard(force: bool, service: str) -> JSONResponse | None:
    decision = await vram_lease.gpu_gate_decision(force=force, service_name=service)
    if decision.allowed:
        return None
    return _gpu_gate_response(decision)


async def _docling_lifecycle_snapshot() -> tuple[bool, dict | None]:
    """Liest den Lifecycle-Snapshot vom Docling-Proxy.

    Rueckgabe: `(reachable, snapshot)`.
    - `(False, None)`: Proxy nicht erreichbar/Transportfehler. Drain-Fallback
      darf direkt stoppen.
    - `(True, None)`: Proxy antwortet, aber Lifecycle ist ungesund/ungueltig.
      Drain darf das nicht als "fertig" interpretieren.
    - `(True, dict)`: valider Snapshot.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(DOCLING_LIFECYCLE_URL)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        return False, None
    except Exception:
        return False, None
    if resp.status_code != 200:
        return True, None
    try:
        snap = resp.json()
    except (ValueError, TypeError):
        return True, None
    if not isinstance(snap, dict):
        return True, None
    return True, snap


async def _trigger_docling_proxy_start() -> tuple[int, dict]:
    """Triggert den Coldstart im Docling-Proxy (#560).

    Der Proxy fuehrt VRAM-Free, `docker start` und Health-Wait unter eigener
    Marker-/Lease-Verwaltung aus, sodass das VRAM-Gate ueber den gesamten
    Modellladevorgang gehalten wird. Das ersetzt den frueheren direkten
    `docker start` aus dem Dashboard, dessen Marker nach ~1-3s wieder gecleart
    war waehrend docling-serve weitere ~30-60s VRAM allokierte.

    Rueckgabe: `(status_code, body)`. Bei Transport-Fehlern wird ein 502/504
    plus `{"error": ...}` synthetisiert, sodass der Caller einheitlich
    JSONResponse erzeugen kann.
    """
    try:
        async with httpx.AsyncClient(
                timeout=DOCLING_START_DEADLINE_S) as client:
            resp = await client.post(DOCLING_START_URL)
    except httpx.TimeoutException:
        return 504, {"error": "Start: docling-proxy Timeout"}
    except httpx.ConnectError:
        return 502, {"error": "Start: docling-proxy nicht erreichbar"}
    except httpx.RequestError as exc:
        return 502, {"error": f"Start: docling-proxy Fehler: {exc}"}
    try:
        body = resp.json() if resp.content else {}
    except (ValueError, TypeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    return resp.status_code, body


async def _trigger_docling_proxy_stop(force: bool) -> tuple[int, dict]:
    """Stoppt Docling ueber die Proxy-State-Machine (#617)."""
    try:
        async with httpx.AsyncClient(timeout=DOCLING_STOP_DEADLINE_S) as client:
            resp = await client.post(DOCLING_STOP_URL, json={"force": force})
    except httpx.TimeoutException:
        return 504, {"error": "Stop: docling-proxy Timeout"}
    except httpx.ConnectError:
        return 502, {"error": "Stop: docling-proxy nicht erreichbar"}
    except httpx.RequestError as exc:
        return 502, {"error": f"Stop: docling-proxy Fehler: {exc}"}
    try:
        body = resp.json() if resp.content else {}
    except (ValueError, TypeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    return resp.status_code, body


async def _stop_docling_container_direct() -> dict | JSONResponse:
    """Fallback wenn der docling-proxy nicht erreichbar ist."""
    try:
        result = await _to_thread(
            subprocess.run,
            ["docker", "stop", "-t", "10", DOCLING_CONTAINER],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return _json_docker_error(e, "Stop")
    except Exception as e:
        return _json_docker_error(e, "Stop")
    if result.returncode != 0:
        if _stderr_has(result.stderr, "is not running", "no such container"):
            return {"status": "stopped", "already": True}
        return JSONResponse(
            {"error": f"Stop fehlgeschlagen: {result.stderr.strip()}"},
            status_code=500,
        )
    return {"status": "stopped"}


async def _drain_docling_for_stop(force: bool) -> JSONResponse | None:
    """Wartet, bis Docling-Proxy keine aktiven Konvertierungen mehr traegt.

    Drain-Bedingung (alle muessen gelten):
    - `active_requests == 0`
    - `state` NICHT in {"starting", "stopping"} — transiente Cold-Start-/
      Shutdown-Sequenzen duerfen nicht durch `docker stop` gesprengt werden.

    Returns:
    - `None` bei Erfolg (Drain-Done sofort oder innerhalb der Deadline,
      oder Lifecycle-Endpoint nicht erreichbar — Fallback).
    - `JSONResponse(409)` bei Drain-Timeout ohne `force`.
    - `None` bei Drain-Timeout mit `force=True` (Stop darf weiterlaufen).

    `last_request_age_s` / `backend_failed` werden vom Drain NICHT
    ausgewertet — rein diagnostische Felder fuer spaetere Heuristiken.
    """
    import logging
    _log = logging.getLogger(__name__)

    TRANSIENT = {"starting", "stopping"}

    def _drain_done(reachable: bool, snap: dict | None) -> bool:
        if not reachable:
            return True  # Fallback: Proxy unreachable -> nichts zu drainen
        if snap is None:
            return False
        if snap.get("active_requests", 0) != 0:
            return False
        return snap.get("state") not in TRANSIENT

    def _lifecycle_state(reachable: bool, snap: dict | None) -> str | None:
        if snap is not None:
            return snap.get("state")
        return "lifecycle-unhealthy" if reachable else "lifecycle-unreachable"

    first_reachable, first_snap = await _docling_lifecycle_snapshot()
    if _drain_done(first_reachable, first_snap):
        return None

    _log.info(
        "Docling-Drain gestartet: active=%s state=%s",
        first_snap.get("active_requests") if first_snap else None,
        _lifecycle_state(first_reachable, first_snap),
    )
    start = time.monotonic()
    deadline = start + DRAIN_DEADLINE_S
    reachable = first_reachable
    snap = first_snap
    while time.monotonic() < deadline:
        await asyncio.sleep(DRAIN_POLL_INTERVAL_S)
        reachable, snap = await _docling_lifecycle_snapshot()
        if _drain_done(reachable, snap):
            _log.info(
                "Docling-Drain fertig nach %.1fs (active=%s, state=%s)",
                time.monotonic() - start,
                snap.get("active_requests") if snap else None,
                _lifecycle_state(reachable, snap),
            )
            return None

    if snap is None:
        final_reachable, final_snap = await _docling_lifecycle_snapshot()
        reachable, snap = final_reachable, final_snap
    final_state = _lifecycle_state(reachable, snap)
    _log.warning(
        "Docling-Drain-Timeout nach %.1fs (active=%s, state=%s, force=%s)",
        time.monotonic() - start,
        snap.get("active_requests") if snap else None,
        final_state,
        force,
    )
    if force:
        return None
    return JSONResponse(
        {
            "error": "Aktive Docling-Konvertierungen — Drain-Deadline erreicht",
            "active_requests": snap.get("active_requests") if snap else None,
            "state": final_state,
            "drain_deadline_s": DRAIN_DEADLINE_S,
            "hint": "force=true im Body setzen um trotzdem zu stoppen",
        },
        status_code=409,
    )


def _docling_port_bound_localhost_only() -> bool | None:
    """Prueft, ob Docling-Container localhost-only auf 5001/tcp gebunden ist.

    Rueckgabe: True bei `127.0.0.1:5002`, False bei Wildcard/leer/falsch,
    None wenn nicht pruefbar (Container fehlt, docker-Fehler).
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f",
             "{{json .NetworkSettings.Ports}}", DOCLING_CONTAINER],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    except Exception:
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    if not raw or raw == "null":
        return None
    try:
        ports = json.loads(raw)
    except Exception:
        return None
    bindings = ports.get("5001/tcp") if isinstance(ports, dict) else None
    if not bindings:
        return False
    for b in bindings:
        host_ip = (b.get("HostIp") or "").strip()
        host_port = (b.get("HostPort") or "").strip()
        if host_port != "5002":
            return False
        if host_ip not in ("127.0.0.1", "localhost"):
            return False
    return True


def _docling_restart_policy_safe() -> bool | None:
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{json .HostConfig.RestartPolicy}}", DOCLING_CONTAINER],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        policy = json.loads((result.stdout or "").strip() or "{}")
    except Exception:
        return None
    if not isinstance(policy, dict):
        return None
    return policy.get("Name") in ("", "no", None)


def _json_docker_error(exc: BaseException, action: str) -> JSONResponse:
    """Mappt Docker-/Subprocess-Exceptions auf JSON-Fehler (F126)."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return JSONResponse(
            {"error": f"{action}: Docker-Timeout"},
            status_code=504,
        )
    if isinstance(exc, FileNotFoundError):
        return JSONResponse(
            {"error": f"{action}: docker nicht verfuegbar"},
            status_code=500,
        )
    if isinstance(exc, OSError):
        return JSONResponse(
            {"error": f"{action}: OS-Fehler: {exc}"},
            status_code=500,
        )
    return JSONResponse(
        {"error": f"{action}: unerwarteter Fehler: {exc}"},
        status_code=500,
    )


def _stderr_has(stderr: str, *needles: str) -> bool:
    """Idempotenz-Check: prueft ob stderr einen der Marker enthaelt (case-insensitiv)."""
    s = (stderr or "").lower()
    return any(n in s for n in needles)


def _json_service_control_error(exc: BaseException, action: str) -> JSONResponse:
    """Mappt Service-Control-/Subprocess-Exceptions auf JSON-Fehler."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return JSONResponse(
            {"error": f"{action}: service-control Timeout"},
            status_code=504,
        )
    if isinstance(exc, FileNotFoundError):
        return JSONResponse(
            {"error": f"{action}: service-control nicht verfuegbar"},
            status_code=500,
        )
    if isinstance(exc, OSError):
        return JSONResponse(
            {"error": f"{action}: OS-Fehler: {exc}"},
            status_code=500,
        )
    return JSONResponse(
        {"error": f"{action}: unerwarteter Fehler: {exc}"},
        status_code=500,
    )


@app.post("/api/docling/device")
async def set_docling_device(body: dict):
    """Docling Container mit neuem Device (cuda/cpu) neu erstellen.

    `unchanged` nur, wenn Device UND localhost-only Portbindung (127.0.0.1:5002)
    stimmen. Sonst Stop/Remove/Create (F96).
    Alle Docker-/Subprocess-Exceptions werden als JSONResponse gemappt (F126).

    Vor dem Recreate wird der Docling-Proxy via `_drain_docling_for_stop`
    angefragt, aktive Konvertierungen zu Ende zu fuehren (#278). `force=true`
    im Body umgeht den Drain.
    """
    device = body.get("device", "")
    if device not in ("cuda", "cpu"):
        return JSONResponse({"error": "Device muss 'cuda' oder 'cpu' sein"}, status_code=400)

    force = body.get("force") is True

    import logging
    _log = logging.getLogger(__name__)

    async with _docling_ops_lock:
        try:
            marker_token = vram_lease.write_overlay_marker("startup", ttl_s=300)
        except vram_lease.MarkerOwnershipError as exc:
            return JSONResponse(
                {"error": f"Konkurrierende Docling-Operation aktiv: {exc}"},
                status_code=409,
            )
        try:
            # Drain innerhalb des Locks, damit Drain-Entscheidung und Docker-Operationen
            # atomar gegen parallele Device-Switches sind (#TOCTOU-Fix).
            drain_response = await _drain_docling_for_stop(force=force)
            if drain_response is not None:
                return drain_response

            # Aktuelles Device + Portbindung pruefen
            try:
                env_result = await _to_thread(
                    subprocess.run,
                    ["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", DOCLING_CONTAINER],
                    capture_output=True, text=True, timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                return _json_docker_error(e, "Device-Inspect")
            except Exception as e:
                return _json_docker_error(e, "Device-Inspect")

            current_device = None
            if env_result.returncode == 0:
                for line in env_result.stdout.splitlines():
                    if line.startswith("DOCLING_DEVICE="):
                        current_device = line.split("=", 1)[1].strip()
                        break

            if current_device == device:
                try:
                    port_ok = await _to_thread(
                        _docling_port_bound_localhost_only)
                    restart_ok = await _to_thread(
                        _docling_restart_policy_safe)
                except Exception as e:
                    return _json_docker_error(e, "Device-Inspect")
                # Nur ueberspringen, wenn Portbindung und RestartPolicy eindeutig sicher sind.
                if port_ok is True and restart_ok is True:
                    return {"status": "unchanged", "device": device}

            # Stop via docling-proxy, damit dessen Lifecycle-State nicht stale
            # RUNNING bleibt (#617). Nur wenn der Proxy nicht erreichbar ist,
            # faellt das Dashboard auf direkten Docker-Stop zurueck.
            status, stop_body = await _trigger_docling_proxy_stop(force=force)
            if status == 502:
                direct_stop = await _stop_docling_container_direct()
                if isinstance(direct_stop, JSONResponse):
                    return direct_stop
            elif not (200 <= status < 300):
                return JSONResponse(
                    {
                        "error": stop_body.get("error")
                        or f"Stop fehlgeschlagen: HTTP {status}",
                    },
                    status_code=status if 400 <= status < 600 else 500,
                )

            # Remove -> Create (rm darf fehlschlagen wenn Container nicht existiert)
            try:
                rm_result = await _to_thread(
                    subprocess.run,
                    ["docker", "rm", DOCLING_CONTAINER],
                    capture_output=True, text=True, timeout=10,
                )
                if rm_result.returncode != 0:
                    _log.debug("docker rm %s: %s (ignoriert)", DOCLING_CONTAINER, rm_result.stderr.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                return _json_docker_error(e, "Device-Stop/Remove")
            except Exception as e:
                return _json_docker_error(e, "Device-Stop/Remove")

            cmd_create = [
                "docker", "create", "--name", DOCLING_CONTAINER,
                "-p", "127.0.0.1:5002:5001",
                "--restart=no",
                "--memory=14g", "--memory-swap=-1",
            ]
            if device == "cuda":
                cmd_create += ["--gpus", "all"]
            for env in [
                f"DOCLING_DEVICE={device}",
                "DOCLING_SERVE_OCR_BATCH_SIZE=16",
                "DOCLING_SERVE_LAYOUT_BATCH_SIZE=16",
                "DOCLING_SERVE_TABLE_BATCH_SIZE=16",
                "DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2",
                "DOCLING_SERVE_MAX_SYNC_WAIT=3600",
                "DOCLING_NUM_THREADS=4",
            ]:
                cmd_create += ["-e", env]
            cmd_create.append(DOCLING_IMAGE)

            try:
                result = await _to_thread(
                    subprocess.run, cmd_create,
                    capture_output=True, text=True, timeout=30,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                return _json_docker_error(e, "Container-Erstellung")
            except Exception as e:
                return _json_docker_error(e, "Container-Erstellung")

            if result.returncode != 0:
                return JSONResponse(
                    {"error": f"Container-Erstellung fehlgeschlagen: {result.stderr.strip()}"},
                    status_code=500,
                )
            return {"status": "recreated", "device": device}
        finally:
            vram_lease.clear_overlay_marker("startup", marker_token)


@app.post("/api/docling/start")
async def start_docling():
    """Docling Container starten (#560).

    Delegiert den Coldstart an den docling-proxy `/_internal/start`. Der
    Proxy haelt waehrend des kompletten Modellladevorgangs den VRAM-Marker
    und uebergibt den Lifecycle-Lease nahtlos an seine State-Machine.
    Das schliesst das frueher offene Fenster, in dem das Dashboard nach
    `docker start` (~1-3s) den Marker cleart, waehrend docling-serve fuer
    weitere ~30-60s VRAM allokiert.
    """
    async with _docling_ops_lock:
        status, body = await _trigger_docling_proxy_start()
    if status == 200:
        already = body.get("started") is False
        if already:
            return {"status": "started", "already": True}
        return {"status": "started"}
    return JSONResponse(
        {"error": body.get("error")
         or f"Start fehlgeschlagen: HTTP {status}"},
        status_code=status if 400 <= status < 600 else 500,
    )


@app.post("/api/docling/stop")
async def stop_docling(body: dict | None = Body(default=None)):
    """Docling Container stoppen.

    Vor dem Stop wird der Docling-Proxy via `_drain_docling_for_stop`
    angefragt, aktive Konvertierungen zu Ende zu fuehren (#278). `force=true`
    im Body umgeht den Drain. Body ist optional (Dashboard-JS ruft ohne Body).
    `Body(default=None)` macht das explizit, sonst liefert FastAPI 422 bei
    leerem Body.
    """
    # Direkte Testaufrufe `await stop_docling()` bekommen den FastAPI-Body-
    # Sentinel als Default — als "kein Body" behandeln.
    body = body if isinstance(body, dict) else {}
    force = body.get("force") is True
    async with _docling_ops_lock:
        try:
            marker_token = vram_lease.write_overlay_marker("shutdown", ttl_s=300)
        except vram_lease.MarkerOwnershipError as exc:
            return JSONResponse(
                {"error": f"Konkurrierende Docling-Operation aktiv: {exc}"},
                status_code=409,
            )
        try:
            drain_response = await _drain_docling_for_stop(force=force)
            if drain_response is not None:
                return drain_response

            status, stop_body = await _trigger_docling_proxy_stop(force=force)
            if status == 502:
                return await _stop_docling_container_direct()
            if not (200 <= status < 300):
                return JSONResponse(
                    {
                        "error": stop_body.get("error")
                        or f"Stop fehlgeschlagen: HTTP {status}",
                    },
                    status_code=status if 400 <= status < 600 else 500,
                )
            if stop_body.get("stopped") is False:
                return {"status": "stopped", "already": True}
            return {"status": "stopped"}
        finally:
            vram_lease.clear_overlay_marker("shutdown", marker_token)


# =============================================
# Embedding-Service-Steuerung
# =============================================

@app.post("/api/embedding/start")
async def start_embedding(body: dict | None = Body(default=None)):
    """Embedding-Service starten. Prozessstart ist lifecycle-only."""
    body = body if isinstance(body, dict) else {}
    reachable, data, _ = await _json_service_health(EMBEDDING_HEALTH_URL)
    if reachable:
        return {"status": "started", "already": True, "health": data}
    try:
        result = await _run_service_control("start", EMBEDDING_SERVICE, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return _json_service_control_error(e, "Start")
    except Exception as e:
        return _json_service_control_error(e, "Start")
    already = False
    if result.returncode != 0:
        if _stderr_has(result.stderr, "already active", "is already"):
            already = True
        else:
            return JSONResponse(
                {"error": f"Start fehlgeschlagen: {result.stderr.strip()}"},
                status_code=500,
            )
    ok, health = await _wait_service_reachable(
        EMBEDDING_HEALTH_URL,
        vram_lease.GPU_SERVICE_START_DEADLINE_S,
    )
    if not ok:
        return JSONResponse(
            {"error": "Embedding-Service wurde gestartet, ist aber nicht HTTP-bereit"},
            status_code=504,
        )
    response = {"status": "started", "health": health}
    if already:
        response["already"] = True
    return response


@app.post("/api/embedding/stop")
async def stop_embedding(body: dict | None = Body(default=None)):
    """Embedding-Service stoppen."""
    body = body if isinstance(body, dict) else {}
    async with vram_lease.gpu_service_operation(
        releases_resources=True,
        force=body.get("force") is True,
        service_name="Embedding-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            result = await _run_service_control("stop", EMBEDDING_SERVICE, timeout=15)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return _json_service_control_error(e, "Stop")
        except Exception as e:
            return _json_service_control_error(e, "Stop")
        if result.returncode != 0:
            if _stderr_has(result.stderr, "not loaded", "inactive", "not found"):
                op.clear_marker = True
                return {"status": "stopped", "already": True}
            # Auch bei echten Fehlern den Marker freigeben — sonst blockiert
            # der gpu_service_loading-Marker 300s alle GPU-Operationen, obwohl
            # das Service den GPU-Speicher nicht zwingend haelt.
            op.clear_marker = True
            return JSONResponse(
                {"error": f"Stop fehlgeschlagen: {result.stderr.strip()}"},
                status_code=500,
            )
        op.clear_marker = True
        return {"status": "stopped"}


# =============================================
# DeBERTa Cross-Encoder Service-Steuerung
# =============================================


@app.post("/api/deberta/start")
async def start_deberta(body: dict | None = Body(default=None)):
    """DeBERTa Cross-Encoder Service starten. Prozessstart ist lifecycle-only."""
    body = body if isinstance(body, dict) else {}
    reachable, data, _ = await _json_service_health(DEBERTA_HEALTH_URL)
    if reachable:
        return {"status": "started", "already": True, "health": data}
    try:
        result = await _run_service_control("start", DEBERTA_SERVICE, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return _json_service_control_error(e, "Start")
    except Exception as e:
        return _json_service_control_error(e, "Start")
    already = False
    if result.returncode != 0:
        if _stderr_has(result.stderr, "already active", "is already"):
            already = True
        else:
            return JSONResponse(
                {"error": f"Start fehlgeschlagen: {result.stderr.strip()}"},
                status_code=500,
            )
    ok, health = await _wait_service_reachable(
        DEBERTA_HEALTH_URL,
        vram_lease.GPU_SERVICE_START_DEADLINE_S,
    )
    if not ok:
        return JSONResponse(
            {"error": "DeBERTa-Service wurde gestartet, ist aber nicht HTTP-bereit"},
            status_code=504,
        )
    response = {"status": "started", "health": health}
    if already:
        response["already"] = True
    return response


@app.post("/api/deberta/stop")
async def stop_deberta(body: dict | None = Body(default=None)):
    """DeBERTa Cross-Encoder Service stoppen."""
    body = body if isinstance(body, dict) else {}
    async with vram_lease.gpu_service_operation(
        releases_resources=True,
        force=body.get("force") is True,
        service_name="DeBERTa-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            result = await _run_service_control("stop", DEBERTA_SERVICE, timeout=15)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return _json_service_control_error(e, "Stop")
        except Exception as e:
            return _json_service_control_error(e, "Stop")
        if result.returncode != 0:
            if _stderr_has(result.stderr, "not loaded", "inactive", "not found"):
                op.clear_marker = True
                return {"status": "stopped", "already": True}
            # Auch bei echten Fehlern den Marker freigeben — sonst blockiert
            # der gpu_service_loading-Marker 300s alle GPU-Operationen, obwohl
            # das Service den GPU-Speicher nicht zwingend haelt.
            op.clear_marker = True
            return JSONResponse(
                {"error": f"Stop fehlgeschlagen: {result.stderr.strip()}"},
                status_code=500,
            )
        op.clear_marker = True
        return {"status": "stopped"}


@app.post("/api/deberta/load")
async def load_deberta_model(body: dict):
    """DeBERTa-Modell wechseln (Proxy zu kiron-deberta)."""
    model = body.get("model", "")
    if not _validate_model_name(model):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)
    route = next((route for route in PROXY_ROUTING_VIEW.routes
                  if route.backend is BackendType.KIRON_DEBERTA and model in route.input_names), None)
    async with vram_lease.gpu_service_operation(
        force=body.get("force") is True,
        service_name="DeBERTa-Service",
        gpu_memory=route.gpu_memory if route else None,
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    DEBERTA_LOAD_URL,
                    json={"model": model},
                    headers={OPERATION_HEADER: op.admission_id, OVERLAY_HEADER: op.token},
                )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom DeBERTa-Service"}, status_code=502)
            op.clear_marker = resp.headers.get(COMPLETION_HEADER) == op.admission_id
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "DeBERTa-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "DeBERTa-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"DeBERTa-Service Backend-Fehler: {exc}")


@app.post("/api/deberta/unload")
async def unload_deberta_model(body: dict):
    """Genau das angegebene DeBERTa-Modell entladen und VRAM freigeben."""
    model = body.get("model")
    if set(body) != {"model"} or not _validate_model_name(model):
        return JSONResponse({"error": "Genau ein gueltiger Modellname ist erforderlich."}, status_code=400)
    route = next((route for route in PROXY_ROUTING_VIEW.routes
                  if route.backend is BackendType.KIRON_DEBERTA and model in route.input_names), None)
    if route is None:
        return JSONResponse({"error": "Unbekanntes DeBERTa-Modell"}, status_code=400)
    async with vram_lease.gpu_service_operation(
        releases_resources=True,
        service_name="DeBERTa-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(DEBERTA_UNLOAD_URL, json={"model": model})
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom DeBERTa-Service"}, status_code=502)
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            if 200 <= status < 300 and (type(data) is not dict or data.get("status") != "unloaded"
                    or data.get("model") != route.backend_model_name or type(data.get("already")) is not bool):
                return JSONResponse({"error": "DeBERTa-Service hat das Entladen nicht bestaetigt."}, status_code=502)
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            return JSONResponse({"error": "DeBERTa-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "DeBERTa-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"DeBERTa-Service Backend-Fehler: {exc}")


# =============================================
# Wartungsmodus (Maintenance Mode)
# =============================================

MAINTENANCE_STATE_FILE = PROXY_DATA_DIR / "maintenance_mode.json"
MAINTENANCE_PORTS = [
    {"port": 5001, "desc": "Docling"},
    {"port": 11434, "desc": "Ollama API"},
    {"port": 11435, "desc": "Ollama Backend (Docker)"},
    {"port": 11440, "desc": "OpenAI API"},
]
_maintenance_toggle_lock = asyncio.Lock()
_maintenance_transition: dict | None = None


def _get_maintenance_state() -> bool:
    """Liest den Wartungsmodus-Status aus der State-Datei.

    Akzeptiert nur Dict-Root mit `active`-Boolean. Alles andere (null, [],
    String, fehlendes/nicht-bool active) wird auf False gemappt, damit eine
    defekte State-Datei nicht /api/maintenance/status, /api/metrics/system,
    Toggle und WebSocket-Metrics ueber AttributeError lahmlegt.
    """
    try:
        if MAINTENANCE_STATE_FILE.exists():
            data = json.loads(MAINTENANCE_STATE_FILE.read_text())
            if isinstance(data, dict):
                active = data.get("active")
                if isinstance(active, bool):
                    return active
                import logging
                logging.getLogger(__name__).warning(
                    "maintenance_mode.json: 'active' ist kein bool (%r) — fallback False", active,
                )
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "maintenance_mode.json: Root ist kein dict (%s) — fallback False", type(data).__name__,
                )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        import logging
        logging.getLogger(__name__).warning("maintenance_mode.json nicht lesbar: %s", e)
    return False


def _set_maintenance_state(active: bool):
    """Schreibt den Wartungsmodus-Status in die State-Datei (atomar)."""
    MAINTENANCE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MAINTENANCE_STATE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({"active": active}))
        os.chmod(tmp, 0o640)
        tmp.replace(MAINTENANCE_STATE_FILE)
    except OSError as e:
        import logging
        logging.getLogger(__name__).error("maintenance_mode.json speichern fehlgeschlagen: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _set_maintenance_transition(current: bool, target: bool) -> None:
    global _maintenance_transition
    _maintenance_transition = {
        "transitioning": True,
        "pending": "activating" if target else "deactivating",
        "active": current,
        "target_active": target,
        "started_at": time.time(),
    }


def _clear_maintenance_transition() -> None:
    global _maintenance_transition
    _maintenance_transition = None


def _maintenance_status_snapshot() -> dict:
    status = {
        "active": _get_maintenance_state(),
        "transitioning": False,
        "pending": None,
        "target_active": None,
    }
    transition = _maintenance_transition
    if isinstance(transition, dict):
        status.update(transition)
    return status


async def restore_maintenance_state():
    """Stellt iptables-Regeln wieder her wenn Maintenance beim letzten Shutdown aktiv war."""
    import logging
    log = logging.getLogger(__name__)
    if _get_maintenance_state():
        log.warning("Maintenance-Modus war beim letzten Shutdown aktiv — stelle iptables-Regeln wieder her")
        result = await _apply_iptables_rules(True)
        if result["errors"]:
            log.error("Fehler beim Wiederherstellen der Maintenance-Regeln: %s", result["errors"])
        else:
            log.info("Maintenance-Regeln wiederhergestellt fuer Ports: %s",
                     [e["port"] for e in MAINTENANCE_PORTS])
    else:
        # #532: Stale ollama-monitor-maintenance-Regeln aus fruehem Crash aufraeumen.
        # Bei State=inactive duerfen keine DROP-Regeln in iptables stehen bleiben.
        result = await _apply_iptables_rules(False)
        if result["errors"]:
            log.warning("Fehler beim Aufraeumen stale Maintenance-Regeln: %s", result["errors"])


async def _apply_iptables_rules(enable: bool) -> dict:
    """Setzt oder entfernt iptables-Regeln fuer den Wartungsmodus (idempotent)."""
    errors = []

    def _completed_error(proc: subprocess.CompletedProcess) -> str:
        return (proc.stderr or proc.stdout or "").strip()

    async def _exists(chain: str, port: int):
        # #561: Tri-State (present, absent, error). Plain bool hat Permission/Timeout/
        # ENOENT als 'absent' interpretiert, sodass der Disable-Loop ohne Fehlereintrag
        # abbrach und State=false geschrieben wurde, obwohl DROP-Regeln noch aktiv sein
        # konnten. Absent wird nur bei expliziten Wrapper-Markern erkannt.
        proc = await _run_firewall_helper("check", chain, port)
        if proc.returncode == 0:
            return ("present", "")
        out = (proc.stdout or "").strip().lower()
        if proc.returncode == _FIREWALL_ABSENT_EXIT or out == "absent":
            return ("absent", "")
        err = _completed_error(proc)
        return ("error", err or "Firewall-Check fehlgeschlagen ohne stderr")

    for entry in MAINTENANCE_PORTS:
        port = entry["port"]
        for chain in ("INPUT", "DOCKER-USER"):
            if enable:
                # Nur einfuegen wenn Regel noch nicht existiert (idempotent).
                state, check_err = await _exists(chain, port)
                if state == "present":
                    continue
                if state == "error":
                    errors.append(f"{chain} port {port} (check): {check_err}")
                    continue
                result = await _run_firewall_helper("insert", chain, port)
                if result.returncode != 0:
                    errors.append(f"{chain} port {port}: {_completed_error(result)}")
            else:
                # Alle passenden Regeln entfernen (inkl. Duplikate aus fruheren Runs).
                # #281: Fehler beim Deaktivieren muessen genauso gesammelt werden wie
                # beim Aktivieren, sonst bleibt State=false aber Firewall hat noch Regeln.
                for _ in range(20):  # Schutz vor Endlos-Loop
                    state, exists_err = await _exists(chain, port)
                    if state == "error":
                        # #561: Check-Fehler im Disable-Pfad MUESSEN errors fuellen,
                        # sonst greift der Schutz in toggle_maintenance nicht.
                        errors.append(
                            f"{chain} port {port} (check): {exists_err}"
                        )
                        break
                    if state == "absent":
                        break
                    result = await _run_firewall_helper("delete", chain, port)
                    if result.returncode != 0:
                        errors.append(
                            f"{chain} port {port} (delete): {_completed_error(result)}"
                        )
                        break
                else:
                    # #576: Loop-Limit ohne break erreicht — Regel kann immer noch
                    # da sein. Ohne diesen Check meldet toggle_maintenance Erfolg
                    # bei >20 Duplikaten und schreibt state=inactive.
                    state, exists_err = await _exists(chain, port)
                    if state == "present":
                        errors.append(
                            f"{chain} port {port} (delete): >20 Duplikate, Loop-Limit erreicht"
                        )
                    elif state == "error":
                        errors.append(
                            f"{chain} port {port} (final check): {exists_err}"
                        )

    return {"errors": errors}


@app.get("/api/maintenance/status")
async def get_maintenance_status():
    """Wartungsmodus-Status abfragen."""
    status = _maintenance_status_snapshot()
    status["ports"] = MAINTENANCE_PORTS
    return status


@app.post("/api/maintenance/toggle")
async def toggle_maintenance():
    """Wartungsmodus ein-/ausschalten."""
    # #368: Serialisiere read-decide-apply-write gegen parallele Toggle-Requests.
    # #491 (wontfix): Reihenfolge "iptables vor State" ist bewusst — restore_maintenance_state()
    #   macht beim Start die Recovery-Invariante (state=active -> apply, state=inactive -> cleanup
    #   stale Regeln per #532). Damit ist jeder Crash zwischen den Schritten erholbar, egal welche
    #   Reihenfolge. OSError beim State-Schreiben rollt iptables zurueck (Zeile 1459-1476).
    async with _maintenance_toggle_lock:
        current = _get_maintenance_state()
        new_state = not current
        _set_maintenance_transition(current, new_state)

        try:
            result = await _apply_iptables_rules(new_state)

            if result["errors"] and new_state:
                # Bei Fehlern beim Aktivieren: Regeln zurueckrollen
                import logging
                log = logging.getLogger(__name__)
                log.error("iptables-Aktivierung fehlgeschlagen — rolle zurueck. Fehler: %s", result["errors"])
                try:
                    rollback = await _apply_iptables_rules(False)
                    if rollback["errors"]:
                        log.error("iptables-Rollback teilweise fehlgeschlagen: %s", rollback["errors"])
                    else:
                        log.info("iptables-Rollback erfolgreich")
                except Exception as rollback_exc:
                    log.exception("iptables-Rollback warf Exception: %s", rollback_exc)
                return JSONResponse(
                    {"error": "Fehler beim Setzen der Firewall-Regeln", "details": result["errors"]},
                    status_code=500,
                )

            # #281: Deaktivierungs-Fehler nicht verschlucken - State bleibt active bis wirklich alle
            # Regeln entfernt sind, sonst zeigt Dashboard inactive obwohl Ports noch geblockt sind.
            if result["errors"] and not new_state:
                import logging
                logging.getLogger(__name__).error(
                    "iptables-Deaktivierung teilweise fehlgeschlagen - State bleibt active: %s",
                    result["errors"],
                )
                return JSONResponse(
                    {"error": "Fehler beim Entfernen der Firewall-Regeln", "details": result["errors"]},
                    status_code=500,
                )

            try:
                _set_maintenance_state(new_state)
            except OSError as e:
                import logging
                log = logging.getLogger(__name__)
                log.error("maintenance_mode.json Schreibfehler — rolle iptables zurueck: %s", e)
                try:
                    rollback = await _apply_iptables_rules(current)
                    if rollback["errors"]:
                        log.error("iptables-Rollback teilweise fehlgeschlagen: %s", rollback["errors"])
                    else:
                        log.info("iptables-Rollback erfolgreich nach State-Schreibfehler")
                except Exception as rollback_exc:
                    log.exception("iptables-Rollback warf Exception: %s", rollback_exc)
                return JSONResponse(
                    {"error": "State-Datei nicht schreibbar", "details": str(e)},
                    status_code=500,
                )
            return {
                "active": new_state,
                "transitioning": False,
                "pending": None,
                "target_active": None,
                "ports": MAINTENANCE_PORTS,
            }
        finally:
            _clear_maintenance_transition()


# ============================================================
# Benchmark API-Endpunkte
# ============================================================

def _load_benchmarks_cache() -> dict:
    """Benchmark-Cache von Disk laden."""
    default = {"models": {}, "last_refresh": None, "sources": {}}
    if BENCHMARKS_CACHE_FILE.exists():
        try:
            data = json.loads(BENCHMARKS_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            import logging
            logging.getLogger(__name__).warning("benchmarks_cache.json nicht lesbar: %s", e)
            return default
        if isinstance(data, dict):
            return data
        import logging
        logging.getLogger(__name__).warning(
            "benchmarks_cache.json: Root ist kein dict (%s) — fallback default", type(data).__name__,
        )
    return default


_benchmarks_refresh_lock = asyncio.Lock()


def _valid_benchmark_timestamp(value) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _benchmark_source_timestamp(entry: dict, source: str) -> float:
    """Return source-specific timestamp with legacy global timestamp fallback.

    Existing cache entries only have `timestamp`; new entries maintain
    `hf_timestamp` and `ep_timestamp` so partial source failures do not mark the
    failed source fresh for seven days.
    """
    source_key = f"{source}_timestamp"
    if source_key in entry:
        return _valid_benchmark_timestamp(entry.get(source_key))
    return _valid_benchmark_timestamp(entry.get("timestamp"))


def _copy_benchmark_source_fields(
    entry: dict,
    prev: dict,
    *,
    source: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        if field in prev:
            entry[field] = prev[field]
    entry[f"{source}_timestamp"] = _benchmark_source_timestamp(prev, source)


def _save_benchmarks_cache(cache: dict) -> bool:
    """Benchmark-Cache auf Disk speichern (atomar). Gibt True bei Erfolg zurueck."""
    BENCHMARKS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BENCHMARKS_CACHE_FILE.with_suffix(BENCHMARKS_CACHE_FILE.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        os.chmod(tmp, 0o640)
        tmp.replace(BENCHMARKS_CACHE_FILE)
        return True
    except OSError as e:
        import logging
        logging.getLogger(__name__).warning("benchmarks_cache.json speichern fehlgeschlagen: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _extract_param_size_b(name: str, details: dict = None) -> float | None:
    """Parametergroesse in Milliarden aus Modellname oder Details extrahieren."""
    # Aus details (z.B. "8.2B" -> 8.2)
    if details:
        ps = details.get("parameter_size", "")
        m = re.match(r'([\d.]+)\s*B', ps, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.match(r'([\d.]+)\s*M', ps, re.IGNORECASE)
        if m:
            return float(m.group(1)) / 1000

    # Aus Tag-Name (z.B. "qwen3:8b" -> 8)
    m = re.search(r':(\d+(?:\.\d+)?)b$', name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Aus Modellname selbst (z.B. "nemotron-3-nano:30b")
    m = re.search(r'(\d+(?:\.\d+)?)b', name.split(":")[-1] if ":" in name else "", re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


async def _fetch_open_llm_leaderboard(search_term: str, max_results: int = 10) -> list[dict] | None:
    """Sucht im Open LLM Leaderboard nach Modellen.

    Returns None bei Fetch-Fehler (Timeout, HTTP-Fehler, malformed response),
    leere Liste bei erfolgreichem Fetch ohne Treffer.
    """
    url = "https://datasets-server.huggingface.co/search"
    params = {
        "dataset": "open-llm-leaderboard/contents",
        "config": "default",
        "split": "train",
        "query": search_term,
        "length": max_results,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not isinstance(data, dict):
                return None
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                return None
            results = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                row = r.get("row")
                if not isinstance(row, dict):
                    continue
                params_b_raw = row.get("#Params (B)")
                try:
                    params_b = float(params_b_raw) if params_b_raw is not None else None
                except (TypeError, ValueError):
                    params_b = None
                results.append({
                    "fullname": row.get("fullname", ""),
                    "params_b": params_b,
                    "mmlu_pro": row.get("MMLU-PRO"),
                    "gpqa": row.get("GPQA"),
                    "ifeval": row.get("IFEval"),
                    "bbh": row.get("BBH"),
                    "math_lvl5": row.get("MATH Lvl 5"),
                    "average": row.get("Average ⬆️"),
                })
            return results
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


_EVALPLUS_MAX_BYTES = 5 * 1024 * 1024


async def _fetch_evalplus() -> dict | None:
    """EvalPlus results.json abrufen (HumanEval Scores) mit Size-Limit (#213).

    Returns None bei Fetch-Fehler (Timeout, HTTP-Fehler, Size-Limit, malformed
    response), dict bei erfolgreichem Fetch.

    #579: httpx.timeout=15 ist Einzel-Read-Deadline; Slow-Drip-Upstream kann
    die Schleife theoretisch beliebig lange offenhalten. Wontfix — geplanter
    UI-Cache-Layer eliminiert die Live-Exposure.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            async with client.stream("GET", "https://evalplus.github.io/results.json") as resp:
                if resp.status_code != 200:
                    return None
                buffer = bytearray()
                async for chunk in resp.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > _EVALPLUS_MAX_BYTES:
                        import logging
                        logging.getLogger(__name__).warning(
                            "EvalPlus results.json > %d bytes - abgebrochen", _EVALPLUS_MAX_BYTES,
                        )
                        return None
                try:
                    data = json.loads(buffer.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return None
                return data if isinstance(data, dict) else None
    except (httpx.HTTPError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


def _normalize_for_match(s: str) -> str:
    """Normalisiert einen String fuer Fuzzy-Matching."""
    return s.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")


def _find_best_hf_match(
    results: list[dict], target_params_b: float | None, search_term: str = ""
) -> dict | None:
    """Bestes Match aus Leaderboard-Ergebnissen waehlen (nach Relevanz + Parametergroesse)."""
    if not results:
        return None

    # Normalisierter Suchbegriff fuer Relevanz-Check
    st = _normalize_for_match(search_term)

    # Mehrere Varianten des Suchbegriffs versuchen (mit/ohne trailing Versionsnummern)
    search_variants = [st]
    # Trailing Ziffern entfernen: "mistrallarge3" -> "mistrallarge"
    # #396: Min-Laenge 8 (war 6), damit kurze Treffer wie "mistral" (7) aus "mistral3"
    # nicht alle mistral-Varianten falsch matchen.
    stripped = re.sub(r'\d+$', '', st)
    if stripped and stripped != st and len(stripped) >= 8:
        search_variants.append(stripped)

    # Relevanz-Filter: fullname muss eine der Varianten enthalten
    relevant = []
    for r in results:
        fn = _normalize_for_match(r.get("fullname", ""))
        for variant in search_variants:
            if variant and variant in fn:
                relevant.append(r)
                break

    # Wenn kein relevantes Ergebnis, abbrechen (kein falsches Match)
    if not relevant:
        return None

    if target_params_b is None:
        return relevant[0]

    # Nach naechster Parametergroesse sortieren
    scored = []
    for r in relevant:
        p = r.get("params_b")
        if p is None:
            continue
        diff = abs(p - target_params_b)
        ratio = diff / max(target_params_b, 0.1)
        scored.append((ratio, r))

    if not scored:
        return relevant[0]

    scored.sort(key=lambda x: x[0])
    # Nur akzeptieren wenn Groesse innerhalb 50% liegt
    if scored[0][0] <= 0.5:
        return scored[0][1]
    return None


def _find_evalplus_match(
    evalplus_data: dict, ollama_base: str, params_b: float | None
) -> dict | None:
    """Bestes Match aus EvalPlus-Daten finden."""
    # Erst statisches Mapping pruefen
    if ollama_base in _OLLAMA_TO_EVALPLUS_MAP:
        key = _OLLAMA_TO_EVALPLUS_MAP[ollama_base]
        if key in evalplus_data:
            scores = evalplus_data[key].get("pass@1", {})
            return {
                "humaneval": scores.get("humaneval"),
                "humaneval_plus": scores.get("humaneval+"),
                "matched_key": key,
            }

    # Fuzzy-Suche: Ollama-Name in EvalPlus-Keys suchen
    # Kein Stripping von Versionsnummern - EvalPlus-Keys sind zu kurz/generisch
    ollama_norm = _normalize_for_match(ollama_base)

    best_match = None
    best_score = 0

    for key, data in evalplus_data.items():
        key_norm = _normalize_for_match(key)
        # Exakter Substring-Match (ohne Stripping)
        if ollama_norm in key_norm or key_norm.startswith(ollama_norm):
            score = len(ollama_norm) / max(len(key_norm), 1)
            if score > best_score:
                best_score = score
                scores = data.get("pass@1", {})
                best_match = {
                    "humaneval": scores.get("humaneval"),
                    "humaneval_plus": scores.get("humaneval+"),
                    "matched_key": key,
                }

    return best_match


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Gecachte Benchmark-Daten zurueckgeben."""
    cache = _load_benchmarks_cache()
    return cache


@app.post("/api/benchmarks/refresh")
async def refresh_benchmarks():
    """Benchmark-Daten von externen APIs abrufen und Cache aktualisieren.

    Sucht fuer jedes lokal installierte Ollama-Modell nach
    Benchmark-Scores im Open LLM Leaderboard und EvalPlus.
    """
    # Lock serialisiert load-modify-save gegen parallele Refreshes
    # (Doppelklick / paralleler API-Aufruf), die sonst Lost-Updates oder
    # OSError beim shared .tmp-Replace ausloesen koennten.
    async with _benchmarks_refresh_lock:
        return await _refresh_benchmarks_locked()


async def _refresh_benchmarks_locked():
    cache = await _to_thread(_load_benchmarks_cache)
    existing = cache.get("models", {})
    if not isinstance(existing, dict):
        existing = {}

    # Lokale Modelle abrufen
    local_models = []
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10) as client:
            resp = await client.get("/api/tags")
            if resp.status_code == 200:
                try:
                    local_models = resp.json().get("models", [])
                except json.JSONDecodeError:
                    local_models = []
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("benchmarks/refresh: local models fetch failed: %s", exc)

    # Alle Basis-Namen sammeln (ohne Duplikate)
    model_bases = {}
    for m in local_models:
        name = m.get("name") if isinstance(m, dict) else None
        if not name:
            continue
        try:
            base = name.split(":")[0]
            params = _extract_param_size_b(name, m.get("details", {}))
        except Exception:
            continue
        if base not in model_bases or params:
            model_bases[base] = params

    # Fuer jeden Basis-Namen Benchmarks parallel suchen (max 5 gleichzeitig)
    hf_searched = 0
    hf_found = 0
    ep_found = 0
    refreshed_ok = 0

    # Modelle filtern die einen Refresh brauchen
    to_refresh = {}
    now = time.time()
    for base_name, params_b in model_bases.items():
        refresh_hf = True
        refresh_ep = True
        if base_name in existing:
            cached_entry = existing[base_name] if isinstance(existing[base_name], dict) else {}
            hf_ts = _benchmark_source_timestamp(cached_entry, "hf")
            ep_ts = _benchmark_source_timestamp(cached_entry, "ep")
            refresh_hf = now - hf_ts >= BENCHMARK_REFRESH_TTL_S
            refresh_ep = now - ep_ts >= BENCHMARK_REFRESH_TTL_S
            if not refresh_hf and not refresh_ep:
                continue
        to_refresh[base_name] = {
            "params_b": params_b,
            "refresh_hf": refresh_hf,
            "refresh_ep": refresh_ep,
        }

    # EvalPlus ist eine globale Quelle; nur abrufen, wenn mindestens ein Modell
    # einen frischen EvalPlus-Versuch braucht. Sonst wuerde ein stale HF-Refresh
    # die bereits frische EvalPlus-Quelle unnoetig hammern (#769).
    needs_evalplus = any(item["refresh_ep"] for item in to_refresh.values())
    evalplus_data = await _fetch_evalplus() if needs_evalplus else None
    evalplus_status = "OK" if evalplus_data is not None else ("cached" if not needs_evalplus else "Fehler")

    sem = asyncio.Semaphore(5)
    counters_lock = asyncio.Lock()

    hf_fields = ("hf_match", "mmlu_pro", "gpqa", "ifeval", "bbh", "hf_params_b")
    ep_fields = ("humaneval", "humaneval_plus", "evalplus_match")

    async def _fetch_one(base_name, refresh_plan):
        nonlocal hf_searched, hf_found, ep_found, refreshed_ok
        async with sem:
            try:
                params_b = refresh_plan["params_b"]
                refresh_hf = refresh_plan["refresh_hf"]
                refresh_ep = refresh_plan["refresh_ep"]
                prev = existing.get(base_name, {})
                prev = prev if isinstance(prev, dict) else {}
                entry = {
                    "ollama_base": base_name,
                    "params_b": params_b if params_b is not None else prev.get("params_b"),
                }
                fetched_at = time.time()

                # 1. Open LLM Leaderboard
                if not refresh_hf:
                    _copy_benchmark_source_fields(entry, prev, source="hf", fields=hf_fields)
                else:
                    if base_name in _OLLAMA_TO_HF_MAP:
                        hf_name = _OLLAMA_TO_HF_MAP[base_name]
                        hf_results = await _fetch_open_llm_leaderboard(hf_name, 5)
                        match_term = hf_name
                    else:
                        hf_results = await _fetch_open_llm_leaderboard(base_name, 10)
                        match_term = base_name

                    if hf_results is None:
                        # Fetch-Fehler: bestehende Cache-Werte behalten, nicht ueberschreiben.
                        # hf_timestamp bleibt alt/0, damit HF beim naechsten Refresh erneut
                        # versucht wird, auch wenn EvalPlus erfolgreich war (#769).
                        _copy_benchmark_source_fields(entry, prev, source="hf", fields=hf_fields)
                    else:
                        async with counters_lock:
                            hf_searched += 1
                        match = _find_best_hf_match(hf_results, params_b, match_term)
                        if match:
                            async with counters_lock:
                                hf_found += 1
                            entry["hf_match"] = match["fullname"]
                            entry["mmlu_pro"] = round(match["mmlu_pro"], 1) if match.get("mmlu_pro") is not None else None
                            entry["gpqa"] = round(match["gpqa"], 1) if match.get("gpqa") is not None else None
                            entry["ifeval"] = round(match["ifeval"], 1) if match.get("ifeval") is not None else None
                            entry["bbh"] = round(match["bbh"], 1) if match.get("bbh") is not None else None
                            entry["hf_params_b"] = match.get("params_b")
                        else:
                            entry["hf_match"] = None
                            entry["mmlu_pro"] = None
                            entry["gpqa"] = None
                            entry["ifeval"] = None
                            entry["bbh"] = None
                            entry["hf_params_b"] = None
                        entry["hf_timestamp"] = fetched_at

                # 2. EvalPlus (HumanEval)
                if not refresh_ep:
                    _copy_benchmark_source_fields(entry, prev, source="ep", fields=ep_fields)
                else:
                    if evalplus_data is None:
                        # Fetch-Fehler: bestehende Cache-Werte behalten, nicht ueberschreiben.
                        # ep_timestamp bleibt alt/0, damit EvalPlus beim naechsten Refresh
                        # erneut versucht wird, ohne HF zu hammern.
                        _copy_benchmark_source_fields(entry, prev, source="ep", fields=ep_fields)
                    else:
                        ep_match = _find_evalplus_match(evalplus_data, base_name, params_b)
                        if ep_match:
                            async with counters_lock:
                                ep_found += 1
                            entry["humaneval"] = ep_match["humaneval"]
                            entry["humaneval_plus"] = ep_match.get("humaneval_plus")
                            entry["evalplus_match"] = ep_match["matched_key"]
                        else:
                            entry["humaneval"] = None
                            entry["humaneval_plus"] = None
                            entry["evalplus_match"] = None
                        entry["ep_timestamp"] = fetched_at

                # 3. Statische Benchmarks als Fallback
                if base_name in _STATIC_BENCHMARKS:
                    sb = _STATIC_BENCHMARKS[base_name]
                    for field in ("mmlu_pro", "bbh", "gpqa", "ifeval", "humaneval"):
                        if entry.get(field) is None and sb.get(field) is not None:
                            entry[field] = sb[field]
                    entry["static_source"] = sb.get("source", "")

                hf_ts = _benchmark_source_timestamp(entry, "hf")
                ep_ts = _benchmark_source_timestamp(entry, "ep")
                both_failed = (
                    refresh_hf and refresh_ep and hf_ts == 0 and ep_ts == 0
                )
                if both_failed and not prev:
                    return
                entry["timestamp"] = max(
                    hf_ts,
                    ep_ts,
                    _valid_benchmark_timestamp(prev.get("timestamp")),
                )

                existing[base_name] = entry
                async with counters_lock:
                    refreshed_ok += 1
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("benchmark refresh fuer %s fehlgeschlagen: %s", base_name, e)

    partial = False
    if to_refresh:
        # #447: Gesamt-Timeout 120s damit der Endpoint bei 50+ Modellen und langsamer
        # Upstream-Latenz nicht minutenlang blockiert. Per-Task-Writes in `existing`
        # sind atomar (Zeile 2065), bereits fertige Entries bleiben beim Cancel erhalten.
        # Nicht-fertige Modelle werden beim naechsten Refresh erneut versucht.
        tasks = [_fetch_one(name, refresh_plan) for name, refresh_plan in to_refresh.items()]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=120.0)
        except asyncio.TimeoutError:
            partial = True
            import logging
            logging.getLogger(__name__).warning(
                "refresh_benchmarks: Gesamt-Timeout 120s erreicht — %d/%d Modelle fertig, Rest wird beim naechsten Refresh erneut versucht",
                refreshed_ok, len(to_refresh),
            )

    cache["models"] = existing
    cache["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cache["sources"] = {
        "open_llm_leaderboard": f"{hf_found}/{hf_searched} Modelle gefunden",
        "evalplus": f"{ep_found} Modelle gefunden ({evalplus_status})",
    }
    if not await _to_thread(_save_benchmarks_cache, cache):
        return JSONResponse(
            {"error": "benchmarks_cache konnte nicht gespeichert werden"},
            status_code=500,
        )

    return {
        "status": "partial" if partial else "OK",
        "models_searched": hf_searched,
        "hf_matches": hf_found,
        "evalplus_matches": ep_found,
        "last_refresh": cache["last_refresh"],
    }


# ============================================================
# API Key Management Endpunkte
# ============================================================

@app.get("/api/apikeys")
async def list_api_keys():
    """Alle API-Keys auflisten."""
    if not api_key_store:
        return JSONResponse({"error": "API Key Store nicht initialisiert"}, status_code=503)
    keys = await _to_thread(api_key_store.list_keys)
    return {"keys": keys}


@app.post("/api/apikeys")
async def create_api_key(body: dict):
    """Neuen API-Key erstellen."""
    if not api_key_store:
        return JSONResponse({"error": "API Key Store nicht initialisiert"}, status_code=503)
    raw_name = body.get("name", "")
    if not isinstance(raw_name, str):
        return JSONResponse({"error": "Name muss ein String sein"}, status_code=400)
    name = raw_name.strip()
    if not name:
        return JSONResponse({"error": "Name ist erforderlich"}, status_code=400)
    if len(name) > 128:
        return JSONResponse({"error": "Name darf maximal 128 Zeichen lang sein"}, status_code=400)
    result = await _to_thread(api_key_store.create_key, name)
    return result


@app.put("/api/apikeys/{key_id}")
async def update_api_key(key_id: str, body: dict):
    """API-Key aktivieren/deaktivieren."""
    if not api_key_store:
        return JSONResponse({"error": "API Key Store nicht initialisiert"}, status_code=503)
    is_active = body.get("is_active")
    if is_active is None:
        return JSONResponse({"error": "is_active ist erforderlich"}, status_code=400)
    if not isinstance(is_active, bool):
        return JSONResponse({"error": "is_active muss ein Boolean sein"}, status_code=400)
    force = body.get("force") is True
    try:
        success = await _to_thread(api_key_store.set_active, key_id, is_active, force=force)
    except LastActiveApiKeyError as exc:
        return JSONResponse(
            {
                "error": str(exc),
                "hint": "Erst einen weiteren aktiven API-Key anlegen oder force=true setzen.",
            },
            status_code=409,
        )
    if not success:
        return JSONResponse({"error": "Key nicht gefunden"}, status_code=404)
    return {"status": "updated", "id": key_id, "is_active": is_active}


@app.delete("/api/apikeys/{key_id}")
async def delete_api_key(key_id: str, body: dict | None = Body(default=None)):
    """API-Key loeschen."""
    if not api_key_store:
        return JSONResponse({"error": "API Key Store nicht initialisiert"}, status_code=503)
    body = body if isinstance(body, dict) else {}
    force = body.get("force") is True
    try:
        success = await _to_thread(api_key_store.delete_key, key_id, force=force)
    except LastActiveApiKeyError as exc:
        return JSONResponse(
            {
                "error": str(exc),
                "hint": "Erst einen weiteren aktiven API-Key anlegen oder force=true setzen.",
            },
            status_code=409,
        )
    if not success:
        return JSONResponse({"error": "Key nicht gefunden"}, status_code=404)
    return {"status": "deleted", "id": key_id}


# ============================================================
# WebSocket - Echtzeit-Kanal
# ============================================================

async def _stream_high_res_history(
    websocket: WebSocket,
    service: HighResHistoryService,
    range_key: str,
) -> None:
    """Sendet einen Puffersnapshot und danach coalescte Einzelpunkte."""
    snapshot = service.snapshot_frame(range_key)
    await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
    samples = snapshot.get("samples")
    if isinstance(samples, list) and samples:
        last_sequence = int(samples[-1].get("sequence", 0))
    else:
        last_sequence = 0

    while True:
        sample = await service.wait_for_next(last_sequence)
        last_sequence = int(sample["sequence"])
        await websocket.send_text(
            json.dumps(service.point_frame(sample, range_key), separators=(",", ":"))
        )


async def _websocket_high_res_history(websocket: WebSocket, range_key: str) -> None:
    await websocket.accept()
    service = high_res_history_service
    if service is None or not service.running:
        await websocket.close(code=1013)
        return
    try:
        await _stream_high_res_history(websocket, service, range_key)
    except (WebSocketDisconnect, HighResHistoryClosed):
        pass
    except RuntimeError:
        # Starlette meldet Sends auf einer bereits geschlossenen Verbindung je
        # nach Disconnect-Zeitpunkt als RuntimeError statt WebSocketDisconnect.
        if websocket.application_state == WebSocketState.CONNECTED:
            raise
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await websocket.close()


@app.websocket("/ws/history/1m")
async def websocket_history_1m(websocket: WebSocket):
    """Inkrementeller 250-ms-Stream fuer die letzte Minute."""
    await _websocket_high_res_history(websocket, "1m")


@app.websocket("/ws/history/10m")
async def websocket_history_10m(websocket: WebSocket):
    """Inkrementeller 250-ms-Stream fuer die letzten zehn Minuten."""
    await _websocket_high_res_history(websocket, "10m")


async def _send_cached_metrics_frame(
    websocket: WebSocket,
    send_lock: asyncio.Lock | None = None,
) -> bool:
    """Sendet einen Metrics-Frame aus dem zentralen Cache, falls vorhanden."""
    payload = get_cached_payload()
    if payload is None:
        return False

    system_metrics = dict(payload["system"])
    system_metrics["maintenance"] = _maintenance_status_snapshot()
    frame = json.dumps({
        "type": "metrics",
        "data": {
            "system": system_metrics,
            "summary": payload["summary"],
        },
    }, default=str)
    if send_lock is None:
        await websocket.send_text(frame)
    else:
        async with send_lock:
            await websocket.send_text(frame)
    return True


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Echtzeit-WebSocket: Request-Events + System-Metriken."""
    await websocket.accept()

    if store is None:
        await websocket.close(code=1013)
        return
    queue = store.register_ws_client()
    # forward_request_events und push_metrics teilen sich die WS-Verbindung;
    # gleichzeitige send_text-Calls auf demselben ASGI-WebSocket sind ein
    # Runtime-Fehler. Lock serialisiert beide Sender pro Verbindung.
    send_lock = asyncio.Lock()

    async def forward_request_events():
        """Lese Request-Events aus der Queue und sende an Client."""
        try:
            while True:
                message = await queue.get()
                # #559: None-Sentinel signalisiert Eviction (QueueFull-Drop oder
                # explizites unregister) — Coroutine sauber beenden.
                if message is None:
                    # #613: WS explizit schliessen, sonst haengt der Client mit
                    # push_metrics weiter an einer Verbindung, die keine
                    # Request-Events mehr bekommt. Close triggert
                    # WebSocketDisconnect in der Main-Loop → Cleanup laeuft.
                    # #932: close() unter send_lock, sonst race mit push_metrics->send_text.
                    try:
                        async with send_lock:
                            await websocket.close()
                    except Exception:
                        pass
                    break
                async with send_lock:
                    await websocket.send_text(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("forward_request_events Fehler: %s", e)
            # Queue sofort abmelden, damit Broadcast sie nicht bis maxsize=100 fuellt (#534).
            store.unregister_ws_client(queue)
            # WebSocket schliessen, damit die Main-Loop via WebSocketDisconnect aufwacht und Cleanup laeuft.
            # #932: close() unter send_lock, sonst race mit push_metrics->send_text.
            try:
                async with send_lock:
                    await websocket.close()
            except Exception:
                pass

    async def push_metrics():
        """Alle 2 Sekunden System-Metriken + Summary an Client senden."""
        try:
            while True:
                await asyncio.sleep(2)
                try:
                    await _send_cached_metrics_frame(websocket, send_lock)
                except Exception as e:
                    # Fehler beim Metrics-Sammeln nicht propagieren, aber loggen
                    import logging
                    logging.getLogger(__name__).warning("push_metrics Fehler: %s", e)
                    # #959: Bei halbgeschlossener WS spammt sonst jeder Tick;
                    # raus, wenn Send-Seite nicht mehr CONNECTED ist.
                    if websocket.application_state != WebSocketState.CONNECTED:
                        break
        except asyncio.CancelledError:
            pass

    request_task = asyncio.create_task(forward_request_events())
    metrics_task = asyncio.create_task(push_metrics())

    try:
        # Warte auf eingehende Nachrichten (haelt Verbindung offen)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        # Erst Tasks canceln, dann unregister - sonst kann ein noch laufender
        # Broadcast die Queue nach unregister noch treffen (#217)
        request_task.cancel()
        metrics_task.cancel()
        await asyncio.gather(request_task, metrics_task, return_exceptions=True)
        store.unregister_ws_client(queue)


# ============================================================
# HTML-Routen (Catch-All fuer SPA-Routing, NACH API-Routen)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Startseite."""
    return FileResponse(templates_dir / "index.html")


@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str):
    """Catch-All fuer clientseitiges Routing (SPA)."""
    return FileResponse(templates_dir / "index.html")


# ============================================================
# Start (fuer Standalone-Betrieb)
# ============================================================

def main():
    """Dashboard standalone starten."""
    print("=" * 50)
    print("Ollama Monitor Dashboard")
    print("=" * 50)
    print("URL: http://localhost:8505")
    print("Strg+C zum Beenden")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8505,
        log_level="info",
    )


if __name__ == "__main__":
    main()
