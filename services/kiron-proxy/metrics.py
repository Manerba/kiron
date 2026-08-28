"""System Metrics Collector - CPU, RAM, GPU, Disk I/O, Ollama Status."""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
import psutil

from kiron_common.catalog_consistency import check_catalog_digests
from kiron_common.ollama_compat import is_real_int

from routing_catalog import PROXY_ROUTING_VIEW

logger = logging.getLogger(__name__)

# Zentraler Metrics-Producer (#585). Konsumenten lesen nur diese letzte
# erfolgreich gebaute Snapshot-Referenz.
_PRODUCER_INTERVAL_S = 2.0
_PRODUCER_INITIAL_TIMEOUT_S = 15.0
_cached_payload: dict | None = None

# Module-level state for disk I/O delta calculation
# Lock guards concurrent access between disk_io_sampler and direct get_all_metrics callers.
_last_disk_io: dict = {}
_disk_io_lock = threading.Lock()

# #586: Zentraler CPU-Sampler. Frueher setzte jeder Consumer den modul-
# globalen psutil-Cache zurueck - der 10s-Writer sah dadurch oft nur das
# Intervall seit dem letzten 2s-WebSocket-Call. Jetzt sampelt nur
# cpu_sampler() in main.py; alle Consumer lesen diesen Cache.
_cpu_percent_lock = threading.Lock()
_cpu_percent_cache: float | None = None
_cpu_percent_baseline_seen = False


# #589: Timeout-Guard fuer get_all_metrics().
# Gesamtbudget fuer eine Sammel-Runde; einzelne haengende Probes duerfen den
# Tick nicht unbegrenzt blockieren.
TOTAL_TIMEOUT_S = 7.0
# Best-effort Cancel/Drain nach Timeout; laeuft ausserhalb des Gesamtbudgets.
DRAIN_TIMEOUT_S = 0.5
# Fester Backoff nach wiederholten Probe-Timeouts.
_BACKOFF_S = 60.0
# Aufeinanderfolgende Timeouts neu gestarteter Probe-Versuche, die Backoff ausloesen.
_TIMEOUT_STRIKES = 3
# #671/#779: Aufeinanderfolgende Ticks mit haengendem in-flight Future, nach
# denen wir Backoff starten. Das Future bleibt referenziert, damit ein
# permanenter Haenger nicht pro Backoff-Zyklus einen weiteren Executor-Worker
# verbraucht.
_IN_FLIGHT_SKIP_STRIKES = 3

_STATE_OK = "ok"
_STATE_TIMEOUT = "timeout"
_STATE_BACKOFF = "backoff"

_timeout_logger = logging.getLogger("metrics.timeout_guard")

# Dedizierter Pool fuer blockierende Metrics-Probes. Lazy initialisiert damit
# Tests den Pool pro Test isolieren koennen (_metrics_executor = None setzen).
# Kein atexit-Handler: Prozess-Exit reicht, der Pool hat keine persistenten
# Ressourcen und wir wollen keine Shutdown-Reihenfolge-Abhaengigkeit.
_metrics_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_metrics_executor_lock = threading.Lock()

# Per-Probe Health-State: shared zwischen metrics_writer, REST und WebSocket.
# Entry-Shape:
#   timeouts: int - aufeinanderfolgende Timeouts neu gestarteter Probe-Versuche.
#                   In-Flight-Skips zaehlen NICHT.
#   backoff_until: float (time.monotonic) - Zeitstempel, ab dem wieder probiert werden darf.
#   last_state: str - "ok" | "timeout" | "backoff"; steuert state-change-Logging.
#   in_flight: Optional[concurrent.futures.Future] - noch laufendes Executor-Future
#              einer blockierenden Probe. Async-Probes nutzen das Feld nicht.
#   in_flight_skips: int - aufeinanderfolgende Ticks, in denen ein altes Future
#                          noch lief. Eskaliert bei _IN_FLIGHT_SKIP_STRIKES in
#                          Backoff. Das laufende Future bleibt als
#                          Outstanding-Limit stehen (#671/#779).
#   post_backoff_retry: bool - True wenn die aktuelle Arbeit der erste Retry nach
#                              abgelaufenem Backoff ist; Timeout dieses Versuchs
#                              startet sofort wieder Backoff.
_probe_health: dict[str, dict] = {}
_probe_health_lock = threading.Lock()

_CATALOG_SERVICES = ("kiron-embeddings", "kiron-deberta")


def catalog_consistency_diagnostic(
    embedding_status: object,
    deberta_status: object,
) -> dict:
    """Compare digests already collected by the periodic health probes."""

    reported = {
        "kiron-embeddings": (
            embedding_status.get("catalog_digest")
            if isinstance(embedding_status, dict)
            else None
        ),
        "kiron-deberta": (
            deberta_status.get("catalog_digest")
            if isinstance(deberta_status, dict)
            else None
        ),
    }
    report = check_catalog_digests(
        PROXY_ROUTING_VIEW.catalog_digest,
        reported,
        required_services=_CATALOG_SERVICES,
    ).to_dict()
    report["proxy_digest"] = PROXY_ROUTING_VIEW.catalog_digest
    return report


def get_cpu_metrics() -> dict:
    """Collect CPU metrics: usage, load average, core count, frequency.

    Jeder Aufruf ist einzeln abgesichert, damit eine transiente psutil/os-Exception
    (z.B. restricted Container ohne /proc/loadavg-Zugriff) nur das betroffene Feld
    auf None setzt statt den ganzen Tick zu verwerfen.
    """
    try:
        freq = psutil.cpu_freq()
        freq_mhz = round(freq.current, 1) if freq else None
    except (psutil.Error, OSError):
        freq_mhz = None

    try:
        load_1, load_5, load_15 = os.getloadavg()
        load_avg_1m = round(load_1, 2)
        load_avg_5m = round(load_5, 2)
        load_avg_15m = round(load_15, 2)
    except OSError:
        load_avg_1m = load_avg_5m = load_avg_15m = None

    with _cpu_percent_lock:
        usage_percent = _cpu_percent_cache

    try:
        core_count = psutil.cpu_count(logical=True)
    except psutil.Error:
        core_count = None

    return {
        "usage_percent": usage_percent,
        "load_avg_1m": load_avg_1m,
        "load_avg_5m": load_avg_5m,
        "load_avg_15m": load_avg_15m,
        "core_count": core_count,
        "freq_mhz": freq_mhz,
    }


def sample_cpu_percent() -> None:
    """Sampelt CPU-Auslastung und aktualisiert den gecachten Wert."""
    global _cpu_percent_baseline_seen, _cpu_percent_cache
    with _cpu_percent_lock:
        # #979: Erstaufruf mit interval=0.1 liefert sofort echten Wert. Mit
        # interval=None gibt psutil beim ersten Call immer 0.0 zurueck und der
        # Cache bliebe bis zum naechsten 10s-Tick None.
        interval = None if _cpu_percent_baseline_seen else 0.1
        try:
            value = psutil.cpu_percent(interval=interval)
        except (psutil.Error, OSError):
            return
        _cpu_percent_baseline_seen = True
        _cpu_percent_cache = value


def get_memory_metrics() -> dict:
    """Collect memory metrics: RAM and swap usage.

    Analog zu get_cpu_metrics jeden psutil-Call einzeln absichern, damit eine
    transiente Exception (z.B. restricted Container) nur das betroffene Feld
    auf None setzt statt den ganzen Tick zu verwerfen.
    """
    try:
        vm = psutil.virtual_memory()
    except (psutil.Error, OSError):
        vm = None

    try:
        swap = psutil.swap_memory()
    except (psutil.Error, OSError):
        swap = None

    return {
        "total_gb": round(vm.total / (1024 ** 3), 2) if vm is not None else None,
        "used_gb": round(vm.used / (1024 ** 3), 2) if vm is not None else None,
        "available_gb": round(vm.available / (1024 ** 3), 2) if vm is not None else None,
        "usage_percent": vm.percent if vm is not None else None,
        "swap_total_gb": round(swap.total / (1024 ** 3), 2) if swap is not None else None,
        "swap_used_gb": round(swap.used / (1024 ** 3), 2) if swap is not None else None,
    }


# #293: Zentraler Disk-I/O-Sampler.
# Frueher hat jeder get_all_metrics-Aufrufer (10s-Writer, 2s-WebSocket, REST-Endpoint)
# die globale Baseline _last_disk_io ueberschrieben - ein 2s-WebSocket-Read setzte die
# Baseline zurueck und der 10s-Writer sah dann nahe-0-Werte. Jetzt sampelt genau EIN
# periodischer Hintergrund-Task (sample_disk_io() im main.py), alle Consumer lesen nur
# den gecachten Snapshot via get_disk_io_metrics().
# #463: Initial None statt 0.0 — der erste sample_disk_io-Aufruf setzt nur
# die Baseline (_last_disk_io), der Snapshot bleibt bis zum zweiten Tick
# unberuehrt. Mit None erkennen Konsumenten (metrics_writer, WebSocket, REST)
# den "noch keine Daten"-Zustand und schreiben NULL in die DB statt falsche 0,
# was sonst als erster Ausreisser in Grafana/History erscheint.
_disk_io_snapshot: dict = {
    "read_mb_s": None,
    "write_mb_s": None,
    "read_iops": None,
    "write_iops": None,
}


def sample_disk_io() -> None:
    """Sampelt Disk-I/O-Counter und aktualisiert die gecachte Snapshot.

    Muss von genau einem periodischen Hintergrund-Task aufgerufen werden
    (typischer Intervall: 10s, siehe main.py). Consumer lesen die Werte
    mit get_disk_io_metrics() (liefert den letzten Snapshot).
    """
    global _last_disk_io, _disk_io_snapshot

    # psutil.disk_io_counters ausserhalb des Locks: Ein haengender Sampler
    # darf read-only Consumer (get_disk_io_metrics) nicht vom letzten
    # Snapshot abschneiden.
    try:
        counters = psutil.disk_io_counters()
    except (psutil.Error, OSError, RuntimeError):
        counters = None

    now = time.monotonic()

    with _disk_io_lock:
        if counters is None:
            # #463: Bei psutil-Ausfall None statt 0 — signalisiert "keine Daten"
            # statt "0 MB/s" an Konsumenten.
            _disk_io_snapshot = {
                "read_mb_s": None,
                "write_mb_s": None,
                "read_iops": None,
                "write_iops": None,
            }
            # Baseline verwerfen, damit der naechste erfolgreiche Sample eine frische 10s-Rate
            # berechnet statt den Delta ueber die gesamte Ausfallzeit zu mitteln.
            _last_disk_io = {}
            return

        current = {
            "read_bytes": counters.read_bytes,
            "write_bytes": counters.write_bytes,
            "read_count": counters.read_count,
            "write_count": counters.write_count,
            "timestamp": now,
        }

        if not _last_disk_io:
            _last_disk_io = current
            return  # Erster Sample: Baseline nur setzen, Snapshot bleibt bei 0

        elapsed = now - _last_disk_io["timestamp"]
        if elapsed <= 0:
            elapsed = 1.0

        read_mb_s = (current["read_bytes"] - _last_disk_io["read_bytes"]) / (1024 ** 2) / elapsed
        write_mb_s = (current["write_bytes"] - _last_disk_io["write_bytes"]) / (1024 ** 2) / elapsed
        read_iops = (current["read_count"] - _last_disk_io["read_count"]) / elapsed
        write_iops = (current["write_count"] - _last_disk_io["write_count"]) / elapsed

        _last_disk_io = current
        _disk_io_snapshot = {
            "read_mb_s": max(0.0, round(read_mb_s, 2)),
            "write_mb_s": max(0.0, round(write_mb_s, 2)),
            "read_iops": max(0.0, round(read_iops, 1)),
            "write_iops": max(0.0, round(write_iops, 1)),
        }


def get_disk_io_metrics() -> dict:
    """Liefert die letzte gecachte Disk-I/O-Rate (aus sample_disk_io)."""
    with _disk_io_lock:
        return dict(_disk_io_snapshot)


def get_cached_payload() -> dict | None:
    """Liefert die letzte erfolgreich produzierte Metrics-Payload-Referenz."""
    return _cached_payload


def get_gpu_metrics() -> dict:
    """Collect GPU metrics via nvidia-smi XML output.

    Returns a dict with GPU name, utilization, VRAM usage, temperature,
    and power draw. On failure (nvidia-smi unavailable, driver mismatch,
    etc.), returns a dict with an 'error' key instead of crashing.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {"error": "nvidia-smi not found. NVIDIA driver may not be installed."}
    except subprocess.TimeoutExpired:
        # #776: code/probe-Felder, damit der zentrale Done-Pfad das probe-interne
        # Timeout vom erfolgreichen Payload unterscheiden und in die Backoff-
        # Logik fuehren kann (sonst laeuft nvidia-smi im 2s-Tick endlos in den 5s-Timeout).
        return {
            "error": "nvidia-smi timed out after 5 seconds.",
            "code": "probe_timeout",
            "probe": "gpu",
        }
    except Exception as e:
        return {"error": f"Failed to run nvidia-smi: {e}"}

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "Unknown error"
        return {"error": f"nvidia-smi exited with code {result.returncode}: {stderr}"}

    try:
        root = ET.fromstring(result.stdout)
    except ET.ParseError as e:
        return {"error": f"Failed to parse nvidia-smi XML output: {e}"}

    # nvidia-smi liefert fuer einzelne Sensoren "N/A" oder "Not Supported";
    # _parse_num gibt None zurueck, damit fehlende Werte in history_db als NULL und
    # in Grafana als Daten-Luecke erscheinen statt als echte 0-Messung (#521).
    def _parse_num(elem, unit: str = "") -> Optional[float]:
        if elem is None or not elem.text:
            return None
        text = elem.text.replace(unit, "").strip() if unit else elem.text.strip()
        try:
            return float(text)
        except ValueError:
            return None

    try:
        gpu = root.find("gpu")
        if gpu is None:
            return {"error": "No GPU found in nvidia-smi output."}

        # GPU name
        name_elem = gpu.find("product_name")
        name = name_elem.text.strip() if name_elem is not None and name_elem.text else "Unknown"

        # GPU utilization
        utilization = gpu.find("utilization")
        gpu_util_percent = _parse_num(utilization.find("gpu_util"), "%") if utilization is not None else None

        # FB Memory Usage (VRAM)
        fb_memory = gpu.find("fb_memory_usage")
        if fb_memory is not None:
            vram_total_mb = _parse_num(fb_memory.find("total"), "MiB")
            vram_used_mb = _parse_num(fb_memory.find("used"), "MiB")
            vram_free_mb = _parse_num(fb_memory.find("free"), "MiB")
            if vram_total_mb and vram_used_mb is not None:
                vram_usage_percent = round((vram_used_mb / vram_total_mb) * 100, 1)
            else:
                vram_usage_percent = None
        else:
            vram_total_mb = vram_used_mb = vram_free_mb = vram_usage_percent = None

        # Temperature
        temperature = gpu.find("temperature")
        temperature_c = _parse_num(temperature.find("gpu_temp"), "C") if temperature is not None else None

        # Power
        power = gpu.find("gpu_power_readings")
        if power is None:
            power = gpu.find("power_readings")
        if power is not None:
            power_draw_w = _parse_num(power.find("power_draw"), "W")
            limit_elem = power.find("power_limit")
            if limit_elem is None:
                limit_elem = power.find("current_power_limit")
            power_limit_w = _parse_num(limit_elem, "W")
        else:
            power_draw_w = power_limit_w = None

        return {
            "name": name,
            "gpu_util_percent": gpu_util_percent,
            "vram_total_mb": vram_total_mb,
            "vram_used_mb": vram_used_mb,
            "vram_free_mb": vram_free_mb,
            "vram_usage_percent": vram_usage_percent,
            "temperature_c": temperature_c,
            "power_draw_w": power_draw_w,
            "power_limit_w": power_limit_w,
        }
    except AttributeError as e:
        return {"error": f"Failed to extract GPU metrics from XML: {e}"}


async def get_ollama_metrics() -> dict:
    """Collect Ollama API metrics: version, available models, loaded models.

    Queries the Ollama REST API at http://127.0.0.1:11435. On failure
    (Ollama not running, connection refused, etc.), returns a dict with
    an 'error' key instead of crashing.
    """
    base_url = "http://127.0.0.1:11435"

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            # Fire all three requests concurrently, einzelne Ausfaelle duerfen andere nicht blockieren
            tags_resp, ps_resp, version_resp = await asyncio.gather(
                client.get("/api/tags"),
                client.get("/api/ps"),
                client.get("/api/version"),
                return_exceptions=True,
            )

            _PARSE_FAILED = object()

            def _parse(resp):
                if isinstance(resp, Exception):
                    return None
                if getattr(resp, "status_code", 0) != 200:
                    return None
                try:
                    return resp.json()
                except (ValueError, json.JSONDecodeError):
                    return _PARSE_FAILED

            tags_parsed = _parse(tags_resp)
            ps_parsed = _parse(ps_resp)
            version_parsed = _parse(version_resp)

            # #322/#387/#777: Wenn alle drei Requests fehlschlagen (Exception,
            # non-2xx oder HTTP 200 mit unparsbarem Body), signalisieren wir das
            # explizit - sonst sieht das Dashboard "version=unknown, models=0" und
            # kann einen komplett toten Ollama nicht von einem leeren unterscheiden.
            def _is_failure(resp, parsed) -> bool:
                if isinstance(resp, BaseException):
                    return True
                if getattr(resp, "status_code", 0) != 200:
                    return True
                return parsed is _PARSE_FAILED

            all_pairs = [
                (tags_resp, tags_parsed),
                (ps_resp, ps_parsed),
                (version_resp, version_parsed),
            ]
            if all(_is_failure(r, p) for r, p in all_pairs):
                first_resp, first_parsed = all_pairs[0]
                if isinstance(first_resp, httpx.ConnectError):
                    return {"error": "Cannot connect to Ollama at 127.0.0.1:11435. Is Ollama running?"}
                if isinstance(first_resp, httpx.TimeoutException):
                    # #776-Pattern: code/probe-Felder, damit der zentrale Done-Pfad
                    # das probe-interne Timeout vom erfolgreichen Payload unterscheidet
                    # und Strikes/Backoff korrekt zaehlt (sonst laeuft Ollama im
                    # 2s-Tick endlos in den 5s-Timeout).
                    return {
                        "error": "Ollama API request timed out after 5 seconds.",
                        "code": "probe_timeout",
                        "probe": "ollama",
                    }
                if isinstance(first_resp, BaseException):
                    return {"error": f"Ollama API error: {type(first_resp).__name__}: {first_resp}"}
                if first_parsed is _PARSE_FAILED:
                    return {"error": "Ollama API returned HTTP 200 with unparseable JSON body."}
                return {"error": f"Ollama API returned non-2xx (status {getattr(first_resp, 'status_code', '?')})."}

            def _data_or(parsed, fallback):
                if parsed is None or parsed is _PARSE_FAILED:
                    return fallback
                return parsed

            # Version
            version_data = _data_or(version_parsed, {})
            version_raw = version_data.get("version") if isinstance(version_data, dict) else None
            version = version_raw if isinstance(version_raw, str) and version_raw else "unknown"

            # Available models
            tags_data = _data_or(tags_parsed, None)
            models_available_state = "unknown"
            models_available_count = None
            if isinstance(tags_data, dict) and isinstance(tags_data.get("models"), list):
                tag_entries = tags_data.get("models")
                if all(isinstance(item, dict) for item in tag_entries):
                    models_available_state = "known"
                    models_available_count = len(tag_entries)

            # Loaded models
            ps_data = _data_or(ps_parsed, None)
            models_loaded = []
            models_loaded_state = "unknown"
            models_loaded_count = None
            if isinstance(ps_data, dict) and isinstance(ps_data.get("models"), list):
                models_loaded_state = "known"
                for model in ps_data.get("models", []):
                    if not isinstance(model, dict):
                        models_loaded_state = "unknown"
                        models_loaded = []
                        break
                    name = model.get("name")
                    if not isinstance(name, str) or not name:
                        name = "unknown"
                    size_bytes = model.get("size")
                    vram_bytes = model.get("size_vram")
                    size_gb = round(size_bytes / (1024 ** 3), 2) if is_real_int(size_bytes) else None
                    vram_gb = round(vram_bytes / (1024 ** 3), 2) if is_real_int(vram_bytes) else None
                    vram_state = "known" if is_real_int(vram_bytes) else "unknown"
                    models_loaded.append({
                        "name": name,
                        "size_gb": size_gb,
                        "vram_gb": vram_gb,
                        "vram_state": vram_state,
                    })
                if models_loaded_state == "known":
                    models_loaded_count = len(models_loaded)

            return {
                "version": version,
                "models_available": models_available_count,
                "models_available_state": models_available_state,
                "models_available_count": models_available_count,
                "models_loaded": models_loaded,
                "models_loaded_state": models_loaded_state,
                "models_loaded_count": models_loaded_count,
            }

    except httpx.ConnectError:
        return {"error": "Cannot connect to Ollama at 127.0.0.1:11435. Is Ollama running?"}
    except httpx.TimeoutException:
        return {
            "error": "Ollama API request timed out after 5 seconds.",
            "code": "probe_timeout",
            "probe": "ollama",
        }
    except Exception as e:
        return {"error": f"Failed to query Ollama API: {e}"}


GPU_PROCESS_FIELDS = {
    "pid": "Linux process id",
    "process": "process executable/name reported by nvidia-smi",
    "label": "classified owner: ollama, docling, embedding, deberta, other",
    "vram_mb": "GPU memory in MiB, null when nvidia-smi cannot parse it",
    "vram_state": "known or unknown",
}

GPU_PROCESS_STATES = {
    "ok": "nvidia-smi returned one or more GPU compute processes",
    "empty": "nvidia-smi succeeded and reported no GPU compute processes",
    "unavailable": "nvidia-smi is not installed or not on PATH",
    "timeout": "the process probe exceeded its timeout budget",
    "backoff": "the probe is temporarily skipped after repeated failures",
    "in_flight": "a previous process probe is still running",
    "error": "nvidia-smi failed or the probe raised an unexpected error",
    "invalid_payload": "internal normalization rejected the probe payload",
}


def _gpu_processes_payload(
    state: str,
    data: list[dict] | None,
    *,
    error: dict | None = None,
) -> dict:
    payload = {
        "state": state,
        "description": "GPU compute processes from nvidia-smi",
        "fields": dict(GPU_PROCESS_FIELDS),
        "states": dict(GPU_PROCESS_STATES),
        "data": data,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _gpu_processes_error_payload(code: str, message: str) -> dict:
    return _gpu_processes_payload(
        code,
        None,
        error={"code": code, "message": message},
    )


def _normalize_gpu_processes_payload(value: object) -> dict:
    if isinstance(value, dict) and isinstance(value.get("state"), str):
        data = value.get("data")
        if data is not None and not isinstance(data, list):
            return _gpu_processes_error_payload(
                "invalid_payload",
                "gpu_processes data is not a list",
            )
        return value
    if isinstance(value, dict) and value.get("probe") == "gpu_processes":
        code = str(value.get("code") or "probe_error")
        message = str(value.get("error") or code)
        state = {
            "collection_timeout": "timeout",
            "probe_backoff": "backoff",
            "probe_in_flight": "in_flight",
        }.get(code, "error")
        return _gpu_processes_error_payload(state, message)
    if isinstance(value, list):
        return _gpu_processes_payload("ok" if value else "empty", value)
    return _gpu_processes_error_payload(
        "invalid_payload",
        "gpu_processes payload shape unknown",
    )


def _read_proc_cmdline(pid: int) -> str:
    """Liest /proc/<pid>/cmdline und gibt die Argumente space-separiert zurueck.

    Container-Prozesse (z.B. docling-serve aus quay.io/docling-project/...)
    erscheinen in nvidia-smi nur mit dem generischen Interpreter-Pfad
    (`/opt/app-root/bin/python3`). Der tatsaechliche Service-Name steckt
    aber im argv und ist via /proc lesbar (world-readable). Bei Fehler:
    leerer String, Caller faellt auf process_name zurueck.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _classify_gpu_process_label(process_name: str, pid: int) -> str:
    """Bestimmt das Service-Label fuer einen GPU-Prozess.

    Kombiniert nvidia-smi process_name mit /proc/<pid>/cmdline, damit
    Container-Prozesse mit generischem Interpreter-Pfad korrekt zugeordnet
    werden (sonst landet z.B. docling-serve im offiziellen Container als
    "other"). Reihenfolge der Pruefungen ist signifikant: ollama/docling
    zuerst, deberta vor embedding (kiron-deberta enthaelt nicht "embedding",
    aber sentence-transformers basiert teils auf deberta-Layern).
    """
    haystack = (process_name + " " + _read_proc_cmdline(pid)).lower()
    if re.search(r"\bollama(?:\b|_)", haystack):
        return "ollama"
    if "docling" in haystack:
        return "docling"
    if "kiron-deberta" in haystack or "deberta" in haystack or "cross_encoder" in haystack or "cross-encoder" in haystack:
        return "deberta"
    if "embedding" in haystack or "sentence" in haystack:
        return "embedding"
    return "other"


def get_gpu_process_metrics() -> dict:
    """Collect per-process GPU memory usage via nvidia-smi.

    Returns a self-describing payload with state, fields, data and optional
    error. Empty success and probe failures stay distinguishable (#647).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except FileNotFoundError:
        return _gpu_processes_error_payload(
            "unavailable",
            "nvidia-smi not found",
        )
    except subprocess.TimeoutExpired:
        return _gpu_processes_error_payload(
            "timeout",
            "nvidia-smi process query timed out",
        )
    except Exception as exc:
        return _gpu_processes_error_payload(
            "error",
            f"{type(exc).__name__}: {exc}",
        )

    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        return _gpu_processes_error_payload("error", msg[:500])
    if not result.stdout.strip():
        return _gpu_processes_payload("empty", [])

    processes = []
    malformed_lines = 0
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            malformed_lines += 1
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            malformed_lines += 1
            continue
        # process_name kann Kommas enthalten (Pfade) — pid am Anfang, MiB am Ende, Rest = Name.
        process_name = ",".join(parts[1:-1])
        try:
            vram_mb = float(parts[-1].replace("MiB", "").strip())
            vram_state = "known"
        except ValueError:
            vram_mb = None
            vram_state = "unknown"

        label = _classify_gpu_process_label(process_name, pid)

        processes.append({
            "pid": pid,
            "process": process_name,
            "label": label,
            "vram_mb": vram_mb,
            "vram_state": vram_state,
        })
    if not processes and malformed_lines > 0:
        return _gpu_processes_error_payload(
            "invalid_payload",
            f"nvidia-smi returned {malformed_lines} unparsable line(s)",
        )
    if malformed_lines > 0:
        logger.warning(
            "nvidia-smi returned %d unparsable line(s) alongside %d valid process(es)",
            malformed_lines, len(processes),
        )
    return _gpu_processes_payload("ok" if processes else "empty", processes)


async def get_embedding_status() -> dict:
    """Check embedding service status via health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://127.0.0.1:11436/health")
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                return {"running": False, "model": None, "available_models": None}
            if not isinstance(data, dict):
                return {"running": False, "model": None, "available_models": None}
            status = data.get("status")
            if resp.status_code == 200 or (
                resp.status_code == 503 and status in ("no_model", "loading")
            ):
                return {
                    "running": True,
                    "status": status or ("ok" if resp.status_code == 200 else "unknown"),
                    "model": data.get("current_model") or data.get("model"),
                    "loaded_models": data.get("loaded_models"),
                    "model_slots": data.get("model_slots"),
                    "loading_model": data.get("loading_model"),
                    "available_models": data.get("available_models"),
                    "available_colbert_models": data.get("available_colbert_models"),
                    "catalog_digest": data.get("catalog_digest"),
                    "model_states": data.get("model_states"),
                }
    except httpx.TimeoutException:
        # #776-Pattern: code/probe-Felder, damit der zentrale Done-Pfad das
        # probe-interne Timeout erkennt und Strikes/Backoff zaehlt (sonst laeuft
        # die embedding-Probe im 2s-Tick endlos in den 2s-Timeout).
        return {
            "running": False, "status": "down", "model": None,
            "loaded_models": None, "model_slots": None,
            "loading_model": None, "available_models": None,
            "available_colbert_models": None,
            "catalog_digest": None,
            "model_states": None,
            "error": "embedding health request timed out after 2 seconds.",
            "code": "probe_timeout",
            "probe": "embedding",
        }
    except Exception:
        pass
    return {
        "running": False,
        "status": "down",
        "model": None,
        "loaded_models": None,
        "model_slots": None,
        "loading_model": None,
        "available_models": None,
        "available_colbert_models": None,
        "catalog_digest": None,
        "model_states": None,
    }


def _docling_inspect_sync() -> dict:
    """Synchroner Docker-Inspect-Pfad fuer die docling-Probe.

    #589: Wird aus get_all_metrics() im Metrics-Executor aufgerufen, damit der
    blockierende Docker-Call nicht den asyncio-Default-Pool belegt.
    Rueckgabe ist formkompatibel mit get_docling_status().
    """
    container = "docling-serve"
    # #989: Beide Sub-Timeouts auf 3s, damit die Summe (max 6s) im
    # TOTAL_TIMEOUT_S=7s-Budget bleibt. Sonst blieb der Worker bis zu 10s
    # belegt, waehrend der In-Flight-Guard schon nach 7s Strikes vergibt.
    try:
        running_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True, text=True, timeout=3,
        )
        # #551: returncode-Check — Docker-Fehler (Daemon down, socket unreachable) ist nicht
        # dasselbe wie Container-Down. Wir behalten running=False fuer Dashboard-Kompatibilitaet,
        # loggen aber explizit, damit der Ausfall in den Logs nachvollziehbar ist.
        if running_result.returncode != 0:
            err = (running_result.stderr or "").strip()[:200]
            logger.warning(f"docker inspect {container} failed (rc={running_result.returncode}): {err}")
            return {"running": False, "device": "unknown", "container": container}
        running = running_result.stdout.strip().lower() == "true"
    except subprocess.TimeoutExpired:
        # #776-Pattern: probe-internes Timeout markieren, sonst zaehlt der zentrale
        # Done-Pfad keinen Strike und Backoff (3 Strikes -> 60s) wird umgangen.
        logger.warning(f"docker inspect {container} timed out after 3s")
        return {
            "running": False, "device": "unknown", "container": container,
            "error": "docker inspect timed out after 3 seconds",
            "code": "probe_timeout",
            "probe": "docling",
        }
    except Exception:
        return {"running": False, "device": "unknown", "container": container}

    device = "unknown"
    try:
        env_result = subprocess.run(
            ["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", container],
            capture_output=True, text=True, timeout=3,
        )
        for line in env_result.stdout.splitlines():
            if line.startswith("DOCLING_DEVICE="):
                device = line.split("=", 1)[1].strip()
                break
    except Exception:
        pass

    return {"running": running, "device": device, "container": container}


async def get_docling_status() -> dict:
    """Check Docker container status and device config for docling-serve.

    #589: Routet die blockierenden Docker-Inspect-Aufrufe in den Metrics-Executor
    statt in den asyncio-Default-Pool, damit ein Docker-Haenger nicht den
    Default-Pool saturiert. Aussen-Shape bleibt
    {"running": bool, "device": str, "container": "docling-serve"}.
    """
    executor = _get_metrics_executor()
    loop = asyncio.get_running_loop()
    cfut = executor.submit(_docling_inspect_sync)
    return await asyncio.wrap_future(cfut, loop=loop)


async def get_deberta_status() -> dict:
    """Check DeBERTa cross-encoder service status via health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://127.0.0.1:11437/health")
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                return {"running": False, "model": None, "available_models": None}
            if not isinstance(data, dict):
                return {"running": False, "model": None, "available_models": None}
            status = data.get("status")
            if resp.status_code == 200 or (
                resp.status_code == 503 and status in ("no_model", "loading")
            ):
                return {
                    "running": True,
                    "status": status or ("ok" if resp.status_code == 200 else "unknown"),
                    "model": data.get("current_model") or data.get("model"),
                    "loaded_models": data.get("loaded_models"),
                    "loading_model": data.get("loading_model"),
                    "available_models": data.get("available_models"),
                    "catalog_digest": data.get("catalog_digest"),
                    "model_states": data.get("model_states"),
                }
    except httpx.TimeoutException:
        # #776-Pattern: code/probe-Felder, damit der zentrale Done-Pfad das probe-interne
        # Timeout vom "down"-Payload unterscheiden und Strikes/Backoff korrekt zaehlen kann
        # (sonst laeuft die DeBERTa-Probe im 2s-Tick endlos in den 2s-Timeout).
        return {
            "running": False, "status": "down", "model": None, "loaded_models": None,
            "loading_model": None, "available_models": None, "catalog_digest": None,
            "model_states": None,
            "error": "DeBERTa health request timed out after 2 seconds",
            "code": "probe_timeout",
            "probe": "deberta",
        }
    except Exception:
        pass
    return {
        "running": False,
        "status": "down",
        "model": None,
        "loaded_models": None,
        "loading_model": None,
        "available_models": None,
        "catalog_digest": None,
        "model_states": None,
    }


def _get_metrics_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Liefert den dedizierten Metrics-Executor. Lazy-initialisiert, ohne atexit."""
    global _metrics_executor
    pool = _metrics_executor
    if pool is not None:
        return pool
    with _metrics_executor_lock:
        if _metrics_executor is None:
            _metrics_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="metrics-probe",
            )
        return _metrics_executor


def _health_entry(name: str) -> dict:
    """Liefert/initialisiert einen Health-Eintrag. Muss unter _probe_health_lock laufen."""
    entry = _probe_health.get(name)
    if entry is None:
        entry = {
            "timeouts": 0,
            "backoff_until": 0.0,
            "last_state": _STATE_OK,
            "in_flight": None,
            "in_flight_skips": 0,
            "post_backoff_retry": False,
        }
        _probe_health[name] = entry
    return entry


def _timeout_payload(name: str, is_list: bool) -> object:
    if is_list:
        return []
    return {"error": "collection timeout", "code": "collection_timeout", "probe": name}


def _backoff_payload(name: str, is_list: bool) -> object:
    if is_list:
        return []
    return {"error": "probe stuck, in backoff", "code": "probe_backoff", "probe": name}


def _in_flight_payload(name: str, is_list: bool) -> object:
    if is_list:
        return []
    return {"error": "probe still running", "code": "probe_in_flight", "probe": name}


def _transition_state(entry: dict, new_state: str, probe_name: str, detail: str = "") -> None:
    """State-change Logging: nur bei tatsaechlichem Wechsel eine Log-Zeile.

    Muss unter _probe_health_lock laufen. Die Log-Calls sind nicht-blockierend.
    """
    prev = entry["last_state"]
    if prev == new_state:
        return
    entry["last_state"] = new_state
    suffix = f" ({detail})" if detail else ""
    if new_state == _STATE_TIMEOUT:
        _timeout_logger.warning(f"{probe_name}: probe timeout{suffix}")
    elif new_state == _STATE_BACKOFF:
        _timeout_logger.warning(f"{probe_name}: backoff started{suffix}")
    elif new_state == _STATE_OK:
        if prev in (_STATE_TIMEOUT, _STATE_BACKOFF):
            _timeout_logger.info(f"{probe_name}: recovered{suffix}")


def _is_probe_internal_timeout(value: object) -> bool:
    """True wenn ein Probe-Ergebnis ein probe-internes Timeout signalisiert.

    Probes mit eigenem Sub-Timeout (z.B. nvidia-smi-subprocess timeout=5)
    liefern ein Error-Payload bevor TOTAL_TIMEOUT_S erreicht ist und landen
    in `done`. Ohne explizite Erkennung wuerden sie den Backoff umgehen, weil
    der Done-Pfad sonst Strikes/State auf OK zuruecksetzt (#776).
    """
    if not isinstance(value, dict):
        return False
    # gpu_processes-Form: state=="timeout" oder error.code=="timeout".
    if value.get("state") == "timeout":
        return True
    error = value.get("error")
    if isinstance(error, dict) and error.get("code") == "timeout":
        return True
    # Generischer probe_timeout-Code (z.B. von get_gpu_metrics).
    if value.get("code") == "probe_timeout":
        return True
    return False


def _apply_timeout_strike(entry: dict, name: str, now: float, detail_suffix: str = "") -> None:
    """Erhoeht Timeout-Strike-Counter und triggert ggf. Backoff.

    Wird sowohl vom zentralen Collection-Timeout-Pfad als auch vom Done-Pfad
    bei probe-internem Timeout (#776) aufgerufen. Muss unter _probe_health_lock laufen.
    """
    was_post_backoff_retry = entry.get("post_backoff_retry", False)
    entry["timeouts"] += 1

    if was_post_backoff_retry:
        entry["backoff_until"] = now + _BACKOFF_S
        entry["timeouts"] = 0
        entry["post_backoff_retry"] = False
        if entry["last_state"] != _STATE_BACKOFF:
            _transition_state(entry, _STATE_BACKOFF, name,
                              detail=f"post-backoff retry timed out{detail_suffix}")
        else:
            _timeout_logger.warning(
                f"{name}: backoff restarted (post-backoff retry timed out{detail_suffix})")
    elif entry["timeouts"] >= _TIMEOUT_STRIKES:
        entry["backoff_until"] = now + _BACKOFF_S
        entry["timeouts"] = 0
        _transition_state(entry, _STATE_BACKOFF, name,
                          detail=f"after {_TIMEOUT_STRIKES} timeouts{detail_suffix}")
    else:
        _transition_state(entry, _STATE_TIMEOUT, name)


# #589: Probe-Klassifikation. Blocking-Probes laufen im Metrics-Executor mit
# In-Flight-Guard und cancel(). Async-Probes laufen als asyncio.Task im Event
# Loop (httpx mit eigenen Timeouts). gpu_processes liefert ein
# selbstbeschreibendes Dict, damit Fehler nicht als leere Prozessliste
# erscheinen (#647).
_BLOCKING_PROBES: tuple[tuple[str, object, bool], ...] = (
    ("cpu", get_cpu_metrics, False),
    ("memory", get_memory_metrics, False),
    ("disk_io", get_disk_io_metrics, False),
    ("gpu", get_gpu_metrics, False),
    ("gpu_processes", get_gpu_process_metrics, False),
    ("docling", _docling_inspect_sync, False),
)

_ASYNC_PROBES: tuple[tuple[str, object], ...] = (
    ("ollama", get_ollama_metrics),
    ("embedding", get_embedding_status),
    ("deberta", get_deberta_status),
)


async def get_all_metrics() -> dict:
    """Collect all system metrics with a bounded total budget.

    #589: get_all_metrics() darf nicht unbegrenzt blockieren und nicht den
    asyncio-Default-Pool mit Metrics-Haengern saturieren.
    - Blocking Probes (cpu, memory, disk_io, gpu, gpu_processes, docling)
      laufen im dedizierten metrics-probe-Executor mit Per-Probe-In-Flight-Guard.
    - Async Probes (ollama, embedding, deberta) laufen als asyncio.Task.
    - Gesamtbudget TOTAL_TIMEOUT_S; Pending-Tasks werden mit DRAIN_TIMEOUT_S
      best-effort gecancelt. Haengende Probes liefern Timeout-, Backoff- oder
      In-Flight-Payloads.
    """
    loop = asyncio.get_running_loop()
    now = time.monotonic()

    # Executor ausserhalb von _probe_health_lock initialisieren, um
    # Lock-Ordering zu vermeiden.
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    tasks_by_name: dict[str, object] = {}
    cfut_by_name: dict[str, concurrent.futures.Future] = {}
    skipped: dict[str, object] = {}
    is_list_probe: dict[str, bool] = {name: is_list for name, _, is_list in _BLOCKING_PROBES}

    # Phase 1: Routing-Entscheidungen und Task-/Future-Erzeugung.
    # Executor.submit() und asyncio.create_task() sind nicht-blockierend, daher
    # ist es safe, den Lock waehrend der Submission zu halten.
    with _probe_health_lock:
        for name, func, is_list in _BLOCKING_PROBES:
            entry = _health_entry(name)

            # Cleanup: altes Future ist inzwischen fertig - Referenz freigeben,
            # Ergebnis wird bewusst verworfen (Sampling-Zeitpunkt gehoert zum
            # alten Tick) und der Skip-Counter zuruecksetzen.
            in_flight: Optional[concurrent.futures.Future] = entry["in_flight"]
            if in_flight is not None and in_flight.done():
                entry["in_flight"] = None
                entry["in_flight_skips"] = 0
                in_flight = None

            # Backoff-Guard hat Vorrang vor In-Flight-Eskalation: waehrend
            # Backoff laufen wir gar nichts und zaehlen auch keine Skips.
            if entry["backoff_until"] > now:
                skipped[name] = _backoff_payload(name, is_list)
                continue

            # In-Flight-Guard mit Hang-Recovery (#671/#779): nach
            # _IN_FLIGHT_SKIP_STRIKES eskalieren wir in Backoff, behalten das
            # alte Future aber als Outstanding-Limit. Ein permanenter Haenger
            # darf nicht nach jedem Backoff-Ende einen weiteren Worker belegen.
            if in_flight is not None:
                skips = entry.get("in_flight_skips", 0) + 1
                entry["in_flight_skips"] = skips
                if skips >= _IN_FLIGHT_SKIP_STRIKES:
                    entry["in_flight_skips"] = 0
                    entry["timeouts"] = 0
                    entry["post_backoff_retry"] = False
                    entry["backoff_until"] = now + _BACKOFF_S
                    _transition_state(
                        entry, _STATE_BACKOFF, name,
                        detail=f"after {_IN_FLIGHT_SKIP_STRIKES} in-flight skips",
                    )
                    skipped[name] = _backoff_payload(name, is_list)
                else:
                    if skips == 1:
                        # Erster Skip-Burst: einmalig sichtbar loggen, ohne
                        # jeden Folgetick zu spammen.
                        _timeout_logger.warning(
                            f"{name}: in-flight skip 1/"
                            f"{_IN_FLIGHT_SKIP_STRIKES} (probe still running)"
                        )
                    skipped[name] = _in_flight_payload(name, is_list)
                continue

            # Start new work. Post-Backoff-Retry markieren, falls letzter State
            # "backoff" war und das Fenster abgelaufen ist.
            if entry["last_state"] == _STATE_BACKOFF:
                entry["post_backoff_retry"] = True
                _timeout_logger.info(f"{name}: backoff ended, retry attempt")

            if executor is None:
                executor = _get_metrics_executor()
            try:
                cfut = executor.submit(func)
            except RuntimeError:
                # Pool shutdown / shutting down: post_backoff_retry zuruecksetzen,
                # damit der Marker nicht stale bleibt und der naechste Tick die Probe
                # nicht faelschlich als "post-backoff retry" interpretiert.
                entry["post_backoff_retry"] = False
                skipped[name] = _backoff_payload(name, is_list)
                continue
            entry["in_flight"] = cfut
            cfut_by_name[name] = cfut
            tasks_by_name[name] = asyncio.wrap_future(cfut, loop=loop)

        for name, factory in _ASYNC_PROBES:
            entry = _health_entry(name)

            if entry["backoff_until"] > now:
                skipped[name] = _backoff_payload(name, False)
                continue

            if entry["last_state"] == _STATE_BACKOFF:
                entry["post_backoff_retry"] = True
                _timeout_logger.info(f"{name}: backoff ended, retry attempt")

            tasks_by_name[name] = asyncio.create_task(factory())

    # Phase 2: Auf Probes warten mit Gesamtbudget.
    task_list = list(tasks_by_name.values())
    done: set = set()
    pending: set = set()
    if task_list:
        try:
            done, pending = await asyncio.wait(task_list, timeout=TOTAL_TIMEOUT_S)
        except asyncio.CancelledError:
            # Outer-Cancellation (z.B. WebSocket-Disconnect): best-effort
            # Cleanup; blockierende Futures bleiben in_flight markiert, damit
            # die naechste Runde den In-Flight-Guard sieht und keine Duplikate
            # einreiht.
            for t in task_list:
                try:
                    t.cancel()
                except Exception:
                    pass
            async_pending = [t for t in task_list if isinstance(t, asyncio.Task)]
            if async_pending:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*async_pending, return_exceptions=True),
                        timeout=DRAIN_TIMEOUT_S,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            # Blocking-Futures ausserhalb des Locks nochmal versuchen zu canceln;
            # bei RUNNING liefert cancel() False, die Referenz bleibt stehen.
            with _probe_health_lock:
                for name, cfut in cfut_by_name.items():
                    try:
                        cancelled = cfut.cancel()
                    except Exception:
                        cancelled = False
                    if cancelled:
                        entry = _health_entry(name)
                        if entry["in_flight"] is cfut:
                            entry["in_flight"] = None
            raise

    # Phase 3: Ergebnisse aggregieren und State aktualisieren.
    results: dict[str, object] = dict(skipped)
    async_pending_to_drain: list[asyncio.Task] = []

    with _probe_health_lock:
        for name, task in tasks_by_name.items():
            is_list = is_list_probe.get(name, False)
            entry = _health_entry(name)

            if task in done:
                val_is_internal_timeout = False
                try:
                    val = task.result()
                    results[name] = val
                    val_is_internal_timeout = _is_probe_internal_timeout(val)
                except Exception as e:
                    # Fertige Probe mit Exception ist kein Haenger; wir
                    # konvertieren in das existierende Error-Dict-Muster.
                    # KeyboardInterrupt/SystemExit propagieren bewusst nach oben.
                    if is_list:
                        results[name] = []
                    else:
                        results[name] = {"error": f"{type(e).__name__}: {e}"}

                if val_is_internal_timeout:
                    # #776: Probe meldet eigenes Sub-Timeout (z.B. nvidia-smi
                    # subprocess timeout=5). Behandeln wie central collection-
                    # timeout, sonst laeuft die Probe im 2s-Tick endlos in den
                    # Sub-Timeout und Backoff (3 Strikes -> 60s) wird umgangen.
                    entry["in_flight"] = None
                    entry["in_flight_skips"] = 0
                    _apply_timeout_strike(entry, name, now,
                                          detail_suffix=" (probe-internal)")
                else:
                    # Counter + Retry-Marker zuruecksetzen (Wert oder Exception).
                    entry["in_flight"] = None
                    entry["timeouts"] = 0
                    entry["in_flight_skips"] = 0
                    entry["post_backoff_retry"] = False
                    _transition_state(entry, _STATE_OK, name)
            else:
                # Timeout-Fall: pending Task/Future.
                results[name] = _timeout_payload(name, is_list)

                if isinstance(task, asyncio.Task):
                    try:
                        task.cancel()
                    except Exception:
                        pass
                    async_pending_to_drain.append(task)
                else:
                    cfut = cfut_by_name.get(name)
                    if cfut is not None:
                        try:
                            cancelled = cfut.cancel()
                        except Exception:
                            cancelled = False
                        if cancelled and entry["in_flight"] is cfut:
                            entry["in_flight"] = None
                            entry["in_flight_skips"] = 0

                _apply_timeout_strike(entry, name, now)

    # Phase 4: Pending Async-Tasks best-effort drain (ausserhalb des Locks).
    if async_pending_to_drain:
        try:
            await asyncio.wait_for(
                asyncio.gather(*async_pending_to_drain, return_exceptions=True),
                timeout=DRAIN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

    # Phase 5: Ergebnis-Dict bauen.
    gpu_processes_payload = _normalize_gpu_processes_payload(
        results.get("gpu_processes")
    )
    gpu_processes_data = gpu_processes_payload.get("data")
    if isinstance(gpu_processes_data, list):
        gpu_processes_list = gpu_processes_data
    else:
        gpu_processes_list = []

    if gpu_processes_payload.get("state") in ("ok", "empty"):
        gpu_process_vram_unknown_count = sum(
            1 for proc in gpu_processes_list
            if isinstance(proc, dict) and proc.get("vram_state") == "unknown"
        )
    else:
        gpu_process_vram_unknown_count = None

    embedding_status = results.get("embedding")
    deberta_status = results.get("deberta")
    catalog_consistency = catalog_consistency_diagnostic(
        embedding_status,
        deberta_status,
    )

    return {
        "cpu": results.get("cpu"),
        "memory": results.get("memory"),
        "disk_io": results.get("disk_io"),
        "gpu": results.get("gpu"),
        "ollama": results.get("ollama"),
        "gpu_processes": gpu_processes_payload,
        "gpu_process_vram_unknown_count": gpu_process_vram_unknown_count,
        "docling": results.get("docling"),
        "embedding": embedding_status,
        "deberta": deberta_status,
        "catalog_consistency": catalog_consistency,
        "timestamp": time.time(),
    }


async def _run_producer_tick(store) -> dict | None:
    """Berechnet einen zentralen Metrics-Snapshot und swapped ihn atomar ein."""
    global _cached_payload

    try:
        system_metrics = await get_all_metrics()
    except Exception:
        # Unerwartete Exception aus get_all_metrics (z.B. Programmierfehler):
        # Traceback fuer Production-Diagnose erhalten — sonst sieht
        # metrics_producer nur "tick failed" ohne Original-Stacktrace.
        logging.getLogger("metrics.producer").exception(
            "_run_producer_tick: get_all_metrics failed unexpectedly"
        )
        return None

    # store.get_summary haengt potentiell an MariaDB/PyMySQL. Ohne Deadline
    # blockiert ein einzelner Tick alle nachfolgenden ticks. Summary ist fuer
    # REST/WebSocket nuetzlich, aber nicht Voraussetzung fuer System-History.
    try:
        summary = await asyncio.wait_for(store.get_summary(), timeout=8.0)
    except Exception:
        summary = {}

    snapshot = {
        "system": system_metrics,
        "summary": summary,
        "produced_at_monotonic": time.monotonic(),
    }
    _cached_payload = snapshot
    return snapshot
