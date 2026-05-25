"""Kiron Dashboard - FastAPI Backend, Port 8505."""

import asyncio
import base64
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from starlette.websockets import WebSocketState
import uvicorn
import httpx

from request_store import RequestStore, RequestStoreReadError
from metrics import get_cached_payload
from history_db import MetricsDB
from api_key_store import LastActiveApiKeyError
import vram_lease
from selftest import router as selftest_router

_COMMON_SRC = Path(__file__).resolve().parent.parent / "kiron-common"
if _COMMON_SRC.exists() and str(_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(_COMMON_SRC))
try:
    from kiron_common.ollama_compat import is_real_int
except ImportError:  # pragma: no cover - half-upgraded venv fallback
    def is_real_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

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

# DBWorkTracker fuer SQLite-Reads aus Dashboard-Handlern (#713): cross-thread
# `db.close()` darf nicht waehrend laufender query_range/get_db_stats greifen.
db_work_tracker = None

# Globaler API Key Store (wird von main.py gesetzt)
api_key_store = None

OLLAMA_BASE_URL = "http://127.0.0.1:11435"
EMBEDDING_HEALTH_URL = "http://127.0.0.1:11436/health"
EMBEDDING_LOAD_URL = "http://127.0.0.1:11436/api/load"
DEBERTA_HEALTH_URL = "http://127.0.0.1:11437/health"
DEBERTA_LOAD_URL = "http://127.0.0.1:11437/api/load"
DEBERTA_UNLOAD_URL = "http://127.0.0.1:11437/api/unload"


async def _to_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


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


# Registry-Cache fuer Featured-Modelle (Single-Worker async, kein Locking noetig).
# `timestamp` bleibt Wall-Time fuer den Disk-Cache; RAM-TTL laeuft monotonic.
_registry_cache = {"data": None, "timestamp": 0, "monotonic_timestamp": 0.0}

MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]+(/[a-zA-Z0-9._:-]+)*$')

# Benchmark-Cache (persistiert als JSON-Datei)
BENCHMARKS_CACHE_FILE = Path(__file__).parent / "benchmarks_cache.json"
BENCHMARK_REFRESH_TTL_S = 7 * 86400

# Registry-Cache (persistiert als JSON-Datei)
REGISTRY_CACHE_FILE = Path(__file__).parent / "registry_cache.json"

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


def _is_valid_registry_models(data) -> bool:
    # Verteidigt Cache + Consumer gegen Shape-Drift bei ollama.com / korrupte Disk-Caches.
    return isinstance(data, list) and all(isinstance(m, dict) for m in data)


def _load_registry_disk_cache() -> dict:
    """Registry-Cache von Disk laden.

    Normalisiert die Shape, sodass Consumer immer
    `{"registry": {"data": ..., "timestamp": <number>}, "tags": {<dict>}}`
    sehen. Korrupter Inhalt (Liste, String, fehlende Keys, falsche Typen)
    wird auf den leeren Default abgebildet, damit Available-Models und
    Tags-Endpoint nicht ueber AttributeError/TypeError abstuerzen.
    """
    default = {"registry": {"data": None, "timestamp": 0}, "tags": {}}
    if not REGISTRY_CACHE_FILE.exists():
        return default
    try:
        raw = json.loads(REGISTRY_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        import logging
        logging.getLogger(__name__).warning("registry_cache.json nicht lesbar: %s", e)
        return default
    if not isinstance(raw, dict):
        return default
    registry = raw.get("registry")
    if not isinstance(registry, dict):
        registry = {"data": None, "timestamp": 0}
    if not isinstance(registry.get("timestamp"), (int, float)):
        registry = {**registry, "timestamp": 0}
    tags = raw.get("tags")
    if not isinstance(tags, dict):
        tags = {}
    else:
        tags = {k: v for k, v in tags.items() if isinstance(v, dict) and isinstance(v.get("data"), dict)}
    return {"registry": registry, "tags": tags}


def _save_registry_disk_cache(cache: dict):
    """Registry-Cache auf Disk speichern (atomar)."""
    tmp = REGISTRY_CACHE_FILE.with_suffix(REGISTRY_CACHE_FILE.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        tmp.replace(REGISTRY_CACHE_FILE)
    except OSError as e:
        import logging
        logging.getLogger(__name__).warning("registry_cache.json speichern fehlgeschlagen: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


_registry_disk_lock = asyncio.Lock()


async def _update_registry_disk_cache(mutate):
    # Serialisiert load-modify-save gegen Lost-Updates bei concurrent Endpoints.
    async with _registry_disk_lock:
        cache = await _to_thread(_load_registry_disk_cache)
        mutate(cache)
        await _to_thread(_save_registry_disk_cache, cache)


def set_store(request_store: RequestStore):
    global store
    store = request_store


def set_metrics_db(db: MetricsDB):
    global metrics_db
    metrics_db = db


def set_db_work_tracker(tracker):
    global db_work_tracker
    db_work_tracker = tracker


def set_api_key_store(aks):
    global api_key_store
    api_key_store = aks


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
            ["systemctl", "restart", "kiron-proxy.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError) as e:
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


def _gb_or_none(value):
    value = _optional_int(value)
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


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


def _classify_model_type(name: str, family: str) -> str:
    """Modelltyp anhand Name und Familie bestimmen."""
    name_lower = name.lower()
    family_lower = family.lower()
    # Embedding-Modelle: Name enthaelt embed/gte/e5, oder Familie ist bert-basiert
    if ("embed" in name_lower or "bert" in family_lower
            or "/gte-" in name_lower or "/e5-" in name_lower):
        return "embedding"
    if "vl" in family_lower or "vision" in name_lower:
        return "vlm"
    return "llm"


@app.get("/api/models/local")
async def get_local_models():
    """Lokale Modelle mit Lade-Status und Embedding-Service-Info."""
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
        if ps_resp is not None and ps_resp.status_code == 200:
            try:
                ps_data = ps_resp.json()
            except json.JSONDecodeError:
                ps_data = None
        else:
            ps_data = None

        # Embedding-Service Status abfragen. 503 no_model/loading mit JSON
        # bedeutet: Prozess erreichbar, nur Modell nicht ready.
        embed_available = set()
        embed_current = None
        embed_loading = None
        embed_status = "down"
        embed_running = False
        reachable, embed_data, _ = await _json_service_health(EMBEDDING_HEALTH_URL)
        if reachable and isinstance(embed_data, dict):
            embed_running = True
            raw_status = embed_data.get("status")
            embed_status = raw_status if isinstance(raw_status, str) else "unknown"
            raw_current = embed_data.get("current_model")
            embed_current = raw_current if isinstance(raw_current, str) else None
            raw_loading = embed_data.get("loading_model")
            embed_loading = raw_loading if isinstance(raw_loading, str) else None
            raw_available = embed_data.get("available_models")
            if isinstance(raw_available, list):
                embed_available = {x for x in raw_available if isinstance(x, str)}

        # Geladene Modelle indexieren. Unknown bleibt fuer jedes Modell sichtbar.
        loaded, ps_known = _parse_ps_models(ps_data)

        models = []
        for m in tags_data.get("models", []):
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not isinstance(name, str) or not name:
                continue
            details = m.get("details", {})
            if not isinstance(details, dict):
                details = {}
            family = details.get("family", "")
            family = family if isinstance(family, str) else ""
            size_bytes = m.get("size", 0)
            size_gb = _gb_or_none(size_bytes)

            # Basename fuer Embedding-Service Matching (ohne :tag)
            basename = name.split(":")[0].rsplit("/", 1)[-1]
            embedding_loading = embed_running and basename == embed_loading
            lm = loaded.get(name)
            if not ps_known:
                load_state = "unknown"
                loaded_flag = False
            elif lm is not None:
                load_state = "loaded"
                loaded_flag = True
            else:
                load_state = "unloaded"
                loaded_flag = False

            entry = {
                "name": name,
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
                "family": family,
                "model_type": _classify_model_type(name, family),
                "size_gb": size_gb,
                "loaded": loaded_flag,
                "load_state": load_state,
                "embedding_capable": basename in embed_available,
                "embedding_active": embed_running and basename == embed_current,
                "embedding_loading": embedding_loading,
                "vram_gb": None if load_state == "unknown" else 0.0,
                "ram_gb": None if load_state == "unknown" else 0.0,
                "context_length": None,
                "expires_at": None,
            }

            if lm is not None:
                total = _optional_int(lm.get("size"))
                vram = _optional_int(lm.get("size_vram"))
                entry["vram_gb"] = _gb_or_none(vram)
                entry["ram_gb"] = (
                    round(max(0, total - vram) / (1024 ** 3), 2)
                    if total is not None and vram is not None
                    else None
                )
                entry["context_length"] = lm.get("context_length")
                entry["expires_at"] = lm.get("expires_at")

            models.append(entry)

        return {
            "models": models,
            "models_loaded_state": "known" if ps_known else "unknown",
            "embedding_service": {
                "running": embed_running,
                "status": embed_status,
                "current_model": embed_current,
                "loading_model": embed_loading,
                "available_models": sorted(embed_available),
            },
        }

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
                )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom Embedding-Service"}, status_code=502)
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
        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=120) as client:
                resp = await client.post("/api/generate", json=payload)
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
                verify_error = await _verify_model_loaded(
                    client,
                    name,
                    require_cpu=not use_gpu,
                    require_gpu=use_gpu,
                )
                if verify_error is not None:
                    return verify_error
            if op is not None:
                op.clear_marker = True
            return {"status": "loaded", "model": name, "gpu": use_gpu}

        except httpx.ConnectError:
            if op is not None:
                op.clear_marker = True
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
        force=force,
        service_name="Ollama",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=30) as client:
                resp = await client.post("/api/generate", json={
                    "model": name, "keep_alive": 0, "stream": False,
                })
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
            op.clear_marker = True
            return {"status": "unloaded", "model": name}

        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "Ollama nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "Timeout beim Entladen"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"Ollama Backend-Fehler beim Entladen: {exc}")


# Aktive Pull-Downloads (wird von Background-Tasks aktualisiert)
_active_pulls = {}
_MAX_ACTIVE_PULLS = 50
PULL_STATUS_FINISHED_TTL_S = 30


def _evict_active_pulls_if_full():
    """Entfernt aelteste finished Pulls wenn Dict das Limit ueberschreitet (#191)."""
    if len(_active_pulls) < _MAX_ACTIVE_PULLS:
        return
    finished = [
        (name, s.get("finished_at", 0))
        for name, s in _active_pulls.items()
        if s.get("finished_at")
    ]
    finished.sort(key=lambda x: x[1])
    for name, _ in finished[: max(1, len(_active_pulls) - _MAX_ACTIVE_PULLS + 1)]:
        _active_pulls.pop(name, None)


async def _do_pull(model_name: str):
    """Background-Task: Modell von Ollama pullen und _active_pulls aktualisieren."""
    state = _active_pulls.get(model_name)
    if not state:
        return
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None) as client:
            async with client.stream("POST", "/api/pull", json={"model": model_name, "stream": True}) as resp:
                if not (200 <= resp.status_code < 300):
                    # Body als Fehlermeldung lesen (begrenzt, falls sehr gross)
                    try:
                        body = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    except Exception:
                        body = ""
                    state["status"] = "error"
                    state["error"] = f"HTTP {resp.status_code}: {body}".rstrip(": ").strip()
                    return
                saw_success = False
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        state["status"] = "error"
                        state["error"] = data["error"] or "Ollama-Pull-Fehler ohne Details"
                        return
                    status_str = data.get("status", "")
                    state["status"] = status_str
                    if status_str == "success":
                        saw_success = True
                    if "total" in data:
                        state["total"] = data["total"]
                    if "completed" in data:
                        state["completed"] = data["completed"]
                    if "digest" in data:
                        state["digest"] = data.get("digest", "")
        if saw_success:
            state["status"] = "success"
        else:
            state["status"] = "error"
            state["error"] = "Ollama-Pull-Stream ohne success-Status beendet"
    except httpx.ConnectError:
        state["status"] = "error"
        state["error"] = "Ollama nicht erreichbar"
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
    finally:
        # Abgeschlossene Pulls nach 30s aufraeumen.
        # #832: monotonic statt time.time(), damit ein NTP-/Admin-Sprung der
        # Systemuhr rueckwaerts nicht (now - finished_at) negativ macht und
        # den Cleanup ewig blockiert.
        # Nur schreiben, wenn der Eintrag noch unsere Instanz ist — sonst
        # ueberschreibt ein paralleler Re-Pull seinen frischen state.
        if _active_pulls.get(model_name) is state:
            state["finished_at"] = time.monotonic()


@app.post("/api/models/pull")
async def pull_model(body: dict):
    """Neues Modell herunterladen (NDJSON-Streaming via Background-Task)."""
    model_name = body.get("name", "")
    if not _validate_model_name(model_name):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)

    # Background-Pull starten falls nicht bereits aktiv
    existing = _active_pulls.get(model_name)
    if not existing or existing.get("status") in ("success", "error"):
        # #276: Vor dem Eviction-Versuch pruefen ob noch Platz nach Evict da ist.
        # _evict_active_pulls_if_full entfernt nur finished Pulls - wenn alle 50 Slots
        # von laufenden Pulls belegt sind, wuerden weitere Background-Tasks gestartet.
        active_running = sum(
            1 for s in _active_pulls.values()
            if s.get("status") not in ("success", "error")
        )
        if active_running >= _MAX_ACTIVE_PULLS:
            return JSONResponse(
                {
                    "error": "Zu viele parallele Pulls",
                    "hint": f"Maximal {_MAX_ACTIVE_PULLS} gleichzeitige Downloads. "
                            "Bitte auf Abschluss warten oder einen laufenden Pull stoppen.",
                    "active_running": active_running,
                },
                status_code=429,
            )
        _evict_active_pulls_if_full()
        _active_pulls[model_name] = {
            "model": model_name,
            "status": "starting",
            "completed": 0,
            "total": 0,
            "digest": "",
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        asyncio.create_task(_do_pull(model_name))

    # Status als NDJSON an den Client streamen (liest aus _active_pulls)
    async def stream_status():
        prev_line = ""
        # #926: Referenz auf den state-dict halten. Wird der Eintrag waehrend
        # eines Sleeps von /pull/status (TTL-Cleanup) per pop entfernt, sieht
        # _active_pulls.get() None und der Loop bricht ab — der dict selbst
        # lebt aber weiter, weil _do_pull denselben dict mutiert. Nach dem
        # Loop reichen wir einen finalen Frame nach, falls wir das terminal
        # status (success/error) nicht mehr im Loop gesehen haben.
        last_state = None
        while True:
            state = _active_pulls.get(model_name)
            if not state:
                break
            last_state = state
            line_data = {"status": state["status"]}
            if state["total"]:
                line_data["total"] = state["total"]
                line_data["completed"] = state["completed"]
            if state.get("digest"):
                line_data["digest"] = state["digest"]
            if state.get("error"):
                line_data["error"] = state["error"]

            line = json.dumps(line_data)
            if line != prev_line:
                yield line + "\n"
                prev_line = line

            if state["status"] in ("success", "error"):
                break
            await asyncio.sleep(0.3)

        if last_state and last_state.get("status") in ("success", "error"):
            line_data = {"status": last_state["status"]}
            if last_state["total"]:
                line_data["total"] = last_state["total"]
                line_data["completed"] = last_state["completed"]
            if last_state.get("digest"):
                line_data["digest"] = last_state["digest"]
            if last_state.get("error"):
                line_data["error"] = last_state["error"]
            line = json.dumps(line_data)
            if line != prev_line:
                yield line + "\n"

    return StreamingResponse(stream_status(), media_type="application/x-ndjson")


@app.get("/api/models/pull/status")
async def get_pull_status():
    """Status aller aktiven Pull-Downloads."""
    # #832: finished_at ist monotonic (siehe _do_pull), Vergleich also gegen monotonic now.
    now = time.monotonic()
    # Abgeschlossene Pulls aufraeumen (nach 30s) - Snapshot gegen concurrent modification
    to_remove = [
        name for name, s in list(_active_pulls.items())
        if s.get("finished_at") and (now - s["finished_at"]) > PULL_STATUS_FINISHED_TTL_S
    ]
    for name in to_remove:
        _active_pulls.pop(name, None)

    active = {}
    for name, state in list(_active_pulls.items()):
        active[name] = {
            "model": name,
            "status": state["status"],
            "completed": state["completed"],
            "total": state["total"],
            "started_at": state["started_at"],
            "error": state.get("error"),
        }
    return {"active_pulls": active}


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
        force=force,
        service_name="Ollama",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
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
                del_resp = await client.request("DELETE", "/api/delete", json={"model": name})

                if del_resp.status_code == 404:
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

                op.clear_marker = True
                return {"status": "deleted", "model": name}

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


async def _docling_lifecycle_snapshot() -> dict | None:
    """Liest den Lifecycle-Snapshot vom Docling-Proxy.

    Rueckgabe: Snapshot-dict bei 200; `None` bei jeder Art Fehler
    (ConnectError, Timeout, Non-200, JSON-Decode). `None` signalisiert
    dem Drain-Loop: Proxy wahrscheinlich tot, Fallback = direkt stoppen.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(DOCLING_LIFECYCLE_URL)
            if resp.status_code != 200:
                return None
            return resp.json()
    except Exception:
        return None


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

    def _drain_done(snap: dict | None) -> bool:
        if snap is None:
            return True  # Fallback: Proxy unreachable -> nichts zu drainen
        if snap.get("active_requests", 0) != 0:
            return False
        return snap.get("state") not in TRANSIENT

    first_snap = await _docling_lifecycle_snapshot()
    if _drain_done(first_snap):
        return None

    _log.info(
        "Docling-Drain gestartet: active=%s state=%s",
        first_snap.get("active_requests") if first_snap else None,
        first_snap.get("state") if first_snap else None,
    )
    start = time.monotonic()
    deadline = start + DRAIN_DEADLINE_S
    snap = first_snap
    while time.monotonic() < deadline:
        await asyncio.sleep(DRAIN_POLL_INTERVAL_S)
        snap = await _docling_lifecycle_snapshot()
        if _drain_done(snap):
            _log.info(
                "Docling-Drain fertig nach %.1fs (active=%s, state=%s)",
                time.monotonic() - start,
                snap.get("active_requests") if snap else None,
                snap.get("state") if snap else None,
            )
            return None

    final = snap if snap is not None else (await _docling_lifecycle_snapshot() or {})
    _log.warning(
        "Docling-Drain-Timeout nach %.1fs (active=%s, state=%s, force=%s)",
        time.monotonic() - start,
        final.get("active_requests") if final else None,
        final.get("state") if final else None,
        force,
    )
    if force:
        return None
    return JSONResponse(
        {
            "error": "Aktive Docling-Konvertierungen — Drain-Deadline erreicht",
            "active_requests": final.get("active_requests") if final else None,
            "state": final.get("state") if final else None,
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


def _json_systemctl_error(exc: BaseException, action: str) -> JSONResponse:
    """Mappt systemctl-/Subprocess-Exceptions auf JSON-Fehler."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return JSONResponse(
            {"error": f"{action}: systemctl-Timeout"},
            status_code=504,
        )
    if isinstance(exc, FileNotFoundError):
        return JSONResponse(
            {"error": f"{action}: systemctl nicht verfuegbar"},
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

EMBEDDING_SERVICE = "kiron-embeddings.service"


@app.post("/api/embedding/start")
async def start_embedding(body: dict | None = Body(default=None)):
    """Embedding-Service starten. Prozessstart ist lifecycle-only."""
    body = body if isinstance(body, dict) else {}
    reachable, data, _ = await _json_service_health(EMBEDDING_HEALTH_URL)
    if reachable:
        return {"status": "started", "already": True, "health": data}
    try:
        result = await _to_thread(
            subprocess.run,
            ["systemctl", "start", EMBEDDING_SERVICE],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return _json_systemctl_error(e, "Start")
    except Exception as e:
        return _json_systemctl_error(e, "Start")
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
        force=body.get("force") is True,
        service_name="Embedding-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            result = await _to_thread(
                subprocess.run,
                ["systemctl", "stop", EMBEDDING_SERVICE],
                capture_output=True, text=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return _json_systemctl_error(e, "Stop")
        except Exception as e:
            return _json_systemctl_error(e, "Stop")
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

DEBERTA_SERVICE = "kiron-deberta"


@app.post("/api/deberta/start")
async def start_deberta(body: dict | None = Body(default=None)):
    """DeBERTa Cross-Encoder Service starten. Prozessstart ist lifecycle-only."""
    body = body if isinstance(body, dict) else {}
    reachable, data, _ = await _json_service_health(DEBERTA_HEALTH_URL)
    if reachable:
        return {"status": "started", "already": True, "health": data}
    try:
        result = await _to_thread(
            subprocess.run,
            ["systemctl", "start", DEBERTA_SERVICE],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return _json_systemctl_error(e, "Start")
    except Exception as e:
        return _json_systemctl_error(e, "Start")
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
        force=body.get("force") is True,
        service_name="DeBERTa-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            result = await _to_thread(
                subprocess.run,
                ["systemctl", "stop", DEBERTA_SERVICE],
                capture_output=True, text=True, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return _json_systemctl_error(e, "Stop")
        except Exception as e:
            return _json_systemctl_error(e, "Stop")
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
    async with vram_lease.gpu_service_operation(
        force=body.get("force") is True,
        service_name="DeBERTa-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    DEBERTA_LOAD_URL,
                    json={"model": model},
                )
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom DeBERTa-Service"}, status_code=502)
            op.clear_marker = _clear_marker_for_service_response(resp, True)
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
async def unload_deberta_model(body: dict | None = Body(default=None)):
    """DeBERTa-Modell entladen und VRAM freigeben."""
    body = body if isinstance(body, dict) else {}
    async with vram_lease.gpu_service_operation(
        force=body.get("force") is True,
        service_name="DeBERTa-Service",
    ) as op:
        if not op.allowed:
            return _gpu_gate_response(op.decision)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(DEBERTA_UNLOAD_URL)
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return JSONResponse({"error": "Ungueltige JSON-Antwort vom DeBERTa-Service"}, status_code=502)
            op.clear_marker = _clear_marker_for_service_response(resp, True)
            status = resp.status_code if 200 <= resp.status_code < 600 else 502
            return JSONResponse(data, status_code=status)
        except httpx.ConnectError:
            op.clear_marker = True
            return JSONResponse({"error": "DeBERTa-Service nicht erreichbar"}, status_code=503)
        except httpx.TimeoutException:
            return JSONResponse({"error": "DeBERTa-Service Timeout"}, status_code=504)
        except httpx.RequestError as exc:
            return _json_backend_error(f"DeBERTa-Service Backend-Fehler: {exc}")


# =============================================
# Wartungsmodus (Maintenance Mode)
# =============================================

MAINTENANCE_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "maintenance_mode.json"
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
    tmp = MAINTENANCE_STATE_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({"active": active}))
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

    async def _run(cmd: list):
        try:
            return await _to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired) as e:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(e))

    async def _exists(chain: str, args: list):
        # #561: Tri-State (present, absent, error). Plain bool hat Permission/Timeout/
        # ENOENT als 'absent' interpretiert, sodass der Disable-Loop ohne Fehlereintrag
        # abbrach und State=false geschrieben wurde, obwohl DROP-Regeln noch aktiv sein
        # konnten. Absent wird nur bei expliziten iptables-Markern erkannt.
        proc = await _run(["iptables", "-C", chain] + args)
        if proc.returncode == 0:
            return ("present", "")
        err = proc.stderr.strip()
        if "Bad rule" in err or "matching rule exist" in err or "No chain" in err:
            return ("absent", "")
        return ("error", err or "iptables -C fehlgeschlagen ohne stderr")

    for entry in MAINTENANCE_PORTS:
        port = entry["port"]
        rules = [
            ("INPUT", [
                "-p", "tcp", "--dport", str(port),
                "!", "-i", "lo",
                "-j", "DROP",
                "-m", "comment", "--comment", "ollama-monitor-maintenance",
            ]),
            ("DOCKER-USER", [
                "-p", "tcp", "--dport", str(port),
                "-j", "DROP",
                "-m", "comment", "--comment", "ollama-monitor-maintenance",
            ]),
        ]

        for chain, args in rules:
            if enable:
                # Nur einfuegen wenn Regel noch nicht existiert (idempotent).
                # Check-Fehler hier nicht separat sammeln: das nachfolgende -I scheitert
                # mit demselben Fehler und wird unten erfasst.
                state, _ = await _exists(chain, args)
                if state == "present":
                    continue
                result = await _run(["iptables", "-I", chain] + args)
                if result.returncode != 0:
                    if chain == "DOCKER-USER" and "No chain" in result.stderr:
                        continue
                    errors.append(f"{chain} port {port}: {result.stderr.strip()}")
            else:
                # Alle passenden Regeln entfernen (inkl. Duplikate aus fruheren Runs).
                # #281: Fehler beim Deaktivieren muessen genauso gesammelt werden wie
                # beim Aktivieren, sonst bleibt State=false aber Firewall hat noch Regeln.
                for _ in range(20):  # Schutz vor Endlos-Loop
                    state, exists_err = await _exists(chain, args)
                    if state == "error":
                        # #561: Check-Fehler im Disable-Pfad MUESSEN errors fuellen,
                        # sonst greift der Schutz in toggle_maintenance nicht.
                        errors.append(
                            f"{chain} port {port} (check): {exists_err}"
                        )
                        break
                    if state == "absent":
                        break
                    result = await _run(["iptables", "-D", chain] + args)
                    if result.returncode != 0:
                        if chain == "DOCKER-USER" and "No chain" in result.stderr:
                            break
                        errors.append(
                            f"{chain} port {port} (delete): {result.stderr.strip()}"
                        )
                        break
                else:
                    # #576: Loop-Limit ohne break erreicht — Regel kann immer noch
                    # da sein. Ohne diesen Check meldet toggle_maintenance Erfolg
                    # bei >20 Duplikaten und schreibt state=inactive.
                    state, exists_err = await _exists(chain, args)
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


@app.get("/api/models/available")
async def get_available_models():
    """Merged List: Registry-Modelle + lokale Modelle mit Source-Feld."""
    global _registry_cache

    now_wall = time.time()
    now_monotonic = time.monotonic()
    disk_cache = _load_registry_disk_cache()
    registry_models = None

    # 1. In-Memory-Cache pruefen (1 Stunde)
    if (
        _registry_cache["data"] is not None
        and (now_monotonic - _registry_cache.get("monotonic_timestamp", 0.0)) < 3600
    ):
        registry_models = _registry_cache["data"]
    else:
        # 2. Disk-Cache pruefen (1 Stunde)
        disk_reg = disk_cache.get("registry", {})
        disk_data = disk_reg.get("data") if isinstance(disk_reg, dict) else None
        disk_valid = _is_valid_registry_models(disk_data)
        disk_age = now_wall - disk_reg.get("timestamp", 0)
        if disk_valid and disk_age < 3600:
            registry_models = disk_data
            _registry_cache["data"] = registry_models
            _registry_cache["timestamp"] = disk_reg["timestamp"]
            # #954: Memory-TTL relativ zum Disk-Alter, sonst kann ein 30min
            # alter Disk-Cache nach Restart effektiv 90min stale werden.
            _registry_cache["monotonic_timestamp"] = now_monotonic - max(disk_age, 0.0)
        else:
            # 3. Von ollama.com fetchen
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get("https://ollama.com/api/tags")
                    if resp.status_code != 200:
                        raise httpx.HTTPStatusError(
                            f"ollama.com lieferte HTTP {resp.status_code}",
                            request=resp.request, response=resp,
                        )
                    registry_data = resp.json()
                    if not isinstance(registry_data, dict) or "models" not in registry_data:
                        raise ValueError("ollama.com /api/tags lieferte unerwartete Root-Shape")
                    fetched = registry_data.get("models")
                    if not _is_valid_registry_models(fetched):
                        raise ValueError("ollama.com /api/tags models-Feld hat unerwartete Shape")
                    registry_models = fetched
                    # In-Memory + Disk aktualisieren
                    _registry_cache["data"] = registry_models
                    _registry_cache["timestamp"] = now_wall
                    _registry_cache["monotonic_timestamp"] = now_monotonic
                    await _update_registry_disk_cache(
                        lambda c: c.__setitem__("registry", {"data": registry_models, "timestamp": now_wall})
                    )
            except Exception:
                # 4. Disk-Cache als Fallback (auch abgelaufen). Nicht in den
                # Memory-Cache uebernehmen: sonst wuerde die 1h-Frische-Pruefung
                # erneute Live-Fetches blockieren, obwohl ollama.com wieder
                # erreichbar sein koennte.
                if disk_valid:
                    registry_models = disk_data
                else:
                    registry_models = []

    # Lokale Modelle abrufen (mit Details fuer family/parameter_size)
    local_models_raw = []
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=5) as client:
            tags_resp = await client.get("/api/tags")
            if tags_resp.status_code == 200:
                try:
                    tags_data = tags_resp.json()
                    if not isinstance(tags_data, dict):
                        tags_data = {}
                    models_field = tags_data.get("models", [])
                    local_models_raw = models_field if isinstance(models_field, list) else []
                except json.JSONDecodeError as exc:
                    import logging
                    logging.getLogger(__name__).warning("Ollama /api/tags JSON-Decode-Fehler: %s", exc)
            else:
                import logging
                logging.getLogger(__name__).warning("Ollama /api/tags HTTP %d", tags_resp.status_code)
    except httpx.RequestError as exc:
        import logging
        logging.getLogger(__name__).warning("Ollama /api/tags nicht erreichbar: %s", exc)

    def _safe_size_bytes(v):
        if isinstance(v, bool):
            return 0
        if isinstance(v, (int, float)):
            return int(v) if v > 0 else 0
        return 0

    local_names = set()
    local_canonical_to_entry = {}
    local_base_names = {}
    for m in local_models_raw:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not name or not isinstance(name, str):
            continue
        local_names.add(name)
        canonical = _canonical_model_name(name)
        local_canonical_to_entry[canonical] = m
        base = name.split(":")[0]
        local_base_names.setdefault(base, m)

    # Registry-Modelle aufbauen
    registry_canonical_names = set()
    models = []
    for m in registry_models if isinstance(registry_models, list) else []:
        if not isinstance(m, dict):
            continue
        name = m.get("name", "")
        if not isinstance(name, str):
            name = ""
        base = name.split(":")[0]
        canonical = _canonical_model_name(name) if name else ""
        if canonical:
            registry_canonical_names.add(canonical)
        size_bytes = _safe_size_bytes(m.get("size", 0))
        size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else None

        # Exakter Match per canonical Name; sonst Base als Metadaten-Fallback
        # (parameter_size kann zwischen Varianten abweichen, aber besser als nichts).
        local_m = local_canonical_to_entry.get(canonical) if canonical else None
        if local_m is None:
            local_m = local_base_names.get(base)
        local_details = local_m.get("details") if local_m else None
        details = local_details if isinstance(local_details, dict) else {}

        models.append({
            "name": name,
            "description": m.get("description", ""),
            "size_gb": size_gb,
            "installed": canonical in local_canonical_to_entry,
            "source": "ollama",
            "family": details.get("family", ""),
            "parameter_size": details.get("parameter_size", ""),
        })

    # Lokale Modelle deren kanonischer Name NICHT in der Registry -> source "other"
    seen_other_canonicals = set()
    for m in local_models_raw:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not name or not isinstance(name, str):
            continue
        canonical = _canonical_model_name(name)
        if canonical not in registry_canonical_names and canonical not in seen_other_canonicals:
            seen_other_canonicals.add(canonical)
            raw_details = m.get("details", {})
            details = raw_details if isinstance(raw_details, dict) else {}
            size_bytes = _safe_size_bytes(m.get("size", 0))
            size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else None
            models.append({
                "name": name,
                "description": "",
                "size_gb": size_gb,
                "installed": True,
                "source": "other",
                "family": details.get("family", ""),
                "parameter_size": details.get("parameter_size", ""),
            })

    return {"models": models}


# Cache fuer Modell-Tags (pro Modellname, 1 Stunde)
_tags_cache = {}
_TAGS_CACHE_MAX = 1024
# Request-Coalescing: gleichzeitige Cache-Misses fuer denselben base_name
# teilen sich einen einzigen ollama.com-Fetch (sonst N parallele 5-MB-Buffer
# + N httpx-Clients pro Tab-Reload).
_tags_inflight: dict[str, asyncio.Future] = {}


def _tags_cache_set(key: str, entry: dict) -> None:
    """Insert + Eviction: abgelaufene Eintraege entfernen, bei Bedarf aelteste droppen."""
    now_monotonic = time.monotonic()
    for k in [
        k for k, v in _tags_cache.items()
        if (now_monotonic - v.get("monotonic_timestamp", 0.0)) >= 3600
    ]:
        _tags_cache.pop(k, None)
    if len(_tags_cache) >= _TAGS_CACHE_MAX:
        oldest = sorted(
            _tags_cache.items(),
            key=lambda kv: kv[1].get("monotonic_timestamp", 0.0),
        )
        for k, _ in oldest[: len(_tags_cache) - _TAGS_CACHE_MAX + 1]:
            _tags_cache.pop(k, None)
    cache_entry = dict(entry)
    cache_entry["monotonic_timestamp"] = now_monotonic
    _tags_cache[key] = cache_entry


async def _fetch_registry_tags_uncached(base_name: str, disk_tags: dict, now_wall: float):
    """Fetcht Tag-HTML von ollama.com, parst Tags und befuellt In-Memory- + Disk-Cache."""
    # #579: httpx.timeout=15 deckt nur Einzel-Reads, kein Gesamt-Deadline —
    # Slow-Drip-Upstream koennte die Schleife beliebig lange offenhalten.
    # Wontfix: geplanter UI-Cache-Layer eliminiert die Live-Exposure; bis dahin
    # bleibt der Disk-Cache-Fallback (unten) das Sicherheitsnetz.
    try:
        max_bytes = 5 * 1024 * 1024
        async with httpx.AsyncClient(timeout=15) as client:
            async with client.stream("GET", f"https://ollama.com/library/{base_name}") as resp:
                if resp.status_code == 404:
                    return JSONResponse({"error": "Modell nicht gefunden"}, status_code=404)
                if resp.status_code != 200:
                    # Transiente Upstream-Fehler (429, 5xx) -> raise fuer Cache-Fallback
                    raise httpx.HTTPStatusError(
                        f"ollama.com HTTP {resp.status_code}",
                        request=resp.request, response=resp,
                    )
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        # Disk-Cache-Fallback + Negative-Cache verhindern, dass
                        # jeder Folgeaufruf erneut bis 5 MB streamt, falls
                        # Upstream dauerhaft uebergrosse Antworten liefert.
                        if base_name in disk_tags and disk_tags[base_name].get("data"):
                            _tags_cache_set(base_name, disk_tags[base_name])
                            return disk_tags[base_name]["data"]
                        error_response = {"tags": [], "model": base_name, "error": "Antwort zu gross"}
                        _tags_cache_set(base_name, {"data": error_response, "timestamp": now_wall})
                        return error_response
                html = buf.decode("utf-8", errors="replace")
    except Exception:
        # Disk-Cache als Fallback (auch abgelaufen)
        if base_name in disk_tags and disk_tags[base_name].get("data"):
            _tags_cache_set(base_name, disk_tags[base_name])
            return disk_tags[base_name]["data"]
        return {"tags": [], "model": base_name, "error": "ollama.com nicht erreichbar"}

    # Tags aus HTML parsen (mobile Bloecke enthalten alle Infos)
    tag_pattern = re.compile(
        r'href="/library/(' + re.escape(base_name) + r':[^"]+)"[^>]*class="sm:hidden[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    tags = []
    seen = set()
    for tag_name, block_html in tag_pattern.findall(html):
        if tag_name in seen:
            continue
        seen.add(tag_name)

        text = re.sub(r'<[^>]+>', ' ', block_html)
        text = ' '.join(text.split())

        size_m = re.search(r'(\d+(?:\.\d+)?\s*(?:MB|GB|TB))', text)
        ctx_m = re.search(r'(\d+K)\s*context', text)
        # Input-Typ aus dem Textblock
        input_m = re.search(r'(?:Text(?:,\s*Image)?|Image)', text)

        tags.append({
            "tag": tag_name,
            "size": size_m.group(1) if size_m else None,
            "context": ctx_m.group(1) if ctx_m else None,
            "input_type": input_m.group(0) if input_m else "Text",
        })

    result = {"tags": tags, "model": base_name}
    _tags_cache_set(base_name, {"data": result, "timestamp": now_wall})

    # Disk-Cache aktualisieren (unter Lock gegen Lost-Updates)
    await _update_registry_disk_cache(
        lambda c: c.setdefault("tags", {}).__setitem__(base_name, {"data": result, "timestamp": now_wall})
    )

    return result


@app.get("/api/models/registry/{model_name:path}/tags")
async def get_registry_model_tags(model_name: str):
    """Verfuegbare Tags/Varianten eines Modells von ollama.com (cached, 1h)."""
    # Base-Name extrahieren (z.B. "qwen3:8b" -> "qwen3", "gemma3" -> "gemma3")
    base_name = model_name.split(":")[0]

    # Path-Traversal verhindern: httpx normalisiert /library/../x -> /x
    if not _validate_model_name(base_name) or any(seg in (".", "..") for seg in base_name.split("/")):
        return JSONResponse({"error": "Ungueltiger Modellname"}, status_code=400)

    now_wall = time.time()
    now_monotonic = time.monotonic()

    # 1. In-Memory-Cache pruefen
    if (
        base_name in _tags_cache
        and (now_monotonic - _tags_cache[base_name].get("monotonic_timestamp", 0.0)) < 3600
    ):
        return _tags_cache[base_name]["data"]

    # 2. Disk-Cache pruefen
    disk_cache = _load_registry_disk_cache()
    disk_tags = disk_cache.get("tags", {})
    if base_name in disk_tags and disk_tags[base_name].get("data") and (now_wall - disk_tags[base_name].get("timestamp", 0)) < 3600:
        _tags_cache_set(base_name, disk_tags[base_name])
        return disk_tags[base_name]["data"]

    # 3. Request-Coalescing: laeuft schon ein Fetch fuer denselben base_name,
    # warten alle Aufrufer auf das eine Resultat (verhindert Cache-Stampede).
    # Fetch laeuft als unabhaengige Task; shield isoliert sie von Cancel
    # einzelner Caller (Tab-Reload-Race darf Wartende nicht mit abbrechen).
    inflight = _tags_inflight.get(base_name)
    if inflight is not None:
        return await asyncio.shield(inflight)

    async def _coalesced_fetch():
        try:
            return await _fetch_registry_tags_uncached(base_name, disk_tags, now_wall)
        finally:
            _tags_inflight.pop(base_name, None)

    task = asyncio.ensure_future(_coalesced_fetch())
    _tags_inflight[base_name] = task
    return await asyncio.shield(task)


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
    tmp = BENCHMARKS_CACHE_FILE.with_suffix(BENCHMARKS_CACHE_FILE.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
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

    Sucht fuer jedes lokal installierte Modell und jeden Registry-Eintrag
    nach Benchmark-Scores im Open LLM Leaderboard und EvalPlus.
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

    # Registry-Modelle hinzufuegen (Benchmarks auch fuer nicht-installierte)
    registry_data = _registry_cache.get("data")
    if not _is_valid_registry_models(registry_data):
        disk_cache = await _to_thread(_load_registry_disk_cache)
        disk_reg = disk_cache.get("registry", {})
        disk_data = disk_reg.get("data") if isinstance(disk_reg, dict) else None
        registry_data = disk_data if _is_valid_registry_models(disk_data) else None
    if registry_data:
        for m in registry_data:
            if not isinstance(m, dict):
                continue
            name = m.get("name", "")
            base = name.split(":")[0] if isinstance(name, str) else ""
            if base and base not in model_bases:
                model_bases[base] = None

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
