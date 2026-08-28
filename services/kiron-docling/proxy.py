#!/usr/bin/env python3
"""Docling-serve On-Demand Proxy.

Startet den docling-serve Container bei Bedarf, haelt ihn warm,
und stoppt ihn nach Inaktivitaet um GPU-VRAM freizugeben.

Vor dem Start werden grosse Ollama-Modelle (z.B. qwen3:8b) entladen,
damit docling genug VRAM bekommt. Kleine Modelle (Embeddings) bleiben.
Ollama-Modelle die waehrend der Ingestion angefragt werden (z.B. VLM)
fallen automatisch auf CPU-Offload zurueck.

Single-Worker uvicorn-Annahme: die State-Maschine lebt per-Prozess.
"""

import asyncio
from dataclasses import dataclass
import enum
import errno
import fcntl
import grp
import json
import logging
import os
import pwd
import re
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# --- Konfiguration ---
CONTAINER_NAME = "docling-serve"
EXPECTED_CONTAINER_IMAGE = "quay.io/docling-project/docling-serve-cu130:v1.14.0"
# PortBinding-Erwartung muss zu BACKEND_URL passen (#877): _wait_healthy
# spricht 127.0.0.1:5002 an; ohne Bindings-Check wuerde ein Drittprozess oder
# ein falsch gebundener Container faelschlich als gesund akzeptiert.
EXPECTED_CONTAINER_PORT_SPEC = "5001/tcp"
EXPECTED_HOST_BIND_IP = "127.0.0.1"
EXPECTED_HOST_BIND_PORT = "5002"
BACKEND_URL = f"http://{EXPECTED_HOST_BIND_IP}:{EXPECTED_HOST_BIND_PORT}"
OLLAMA_URL = "http://127.0.0.1:11435"
KIRON_ACTIVE_REQUESTS_URL = os.environ.get(
    "KIRON_ACTIVE_REQUESTS_URL",
    "http://127.0.0.1:8505/api/requests/active",
)
KIRON_DASHBOARD_USER = os.environ.get("KIRON_DASHBOARD_USER", "admin")
KIRON_DASHBOARD_PASSWORD = os.environ.get("KIRON_DASHBOARD_PASSWORD", "admin")
LISTEN_PORT = 5001
IDLE_TIMEOUT_S = 3600          # 60 Minuten Warmhaltephase
HEALTH_CHECK_TIMEOUT_S = 180   # Max. Wartezeit beim Starten (Cold-Start)
IDLE_CHECK_INTERVAL_S = 15     # Wie oft auf Idle pruefen
VRAM_VERIFY_TIMEOUT_S = 30     # Verify-Polling nach Unload
GPU_INFLIGHT_DRAIN_TIMEOUT_S = 60.0
GPU_INFLIGHT_DRAIN_POLL_S = 0.5
GPU_INFLIGHT_DRAIN_HTTP_TIMEOUT_S = 2.0
GPU_SERVICE_LOADING_DRAIN_TIMEOUT_S = 300.0
GPU_SERVICE_LOADING_DRAIN_POLL_S = 0.5
GPU_SERVICE_MARKER_SHORT_TTL_S = 5.0
SHUTDOWN_START_GRACE_S = 45    # Shutdown-Grace fuer laufenden Start-Task
SHUTDOWN_STOP_GRACE_S = 45     # Shutdown-Grace fuer laufenden Stop-Task
_DIRTY_RETRY_INTERVAL_S = 30.0  # Abstand zwischen Dirty-Retry-Stops
_DIRTY_RETRY_MAX_ATTEMPTS = 20  # ~10 min bis Aufgabe
WARM_REGATE_WAIT_TIMEOUT_S = 30.0
START_FAILURE_WINDOW_S: float = 600.0
START_FAILURE_THRESHOLD: int = 3
START_COOLDOWN_DURATION_S: float = 600.0
# docling braucht ~7 GiB aktiv. Bei 12 GiB GPU bleiben ~4.5 GiB
# als kumulatives Keep-Budget fuer alle Ollama-Modelle zusammen.
VRAM_BUDGET_BYTES = int(4.5 * 1024**3)
RUNTIME_MARKER_DIR = Path(os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron/vram"))
RUNTIME_MARKER_GROUP = "kiron-runtime"
RUNTIME_MARKER_FILE_OWNER_NAMES: frozenset[str] = frozenset({
    "kiron-proxy",
    "kiron-docling",
})
RUNTIME_MARKER_DIR_MODE = 0o2770
RUNTIME_MARKER_FILE_MODE = 0o660
STARTUP_MARKER_PATH = RUNTIME_MARKER_DIR / "docling-vram-startup.json"
SHUTDOWN_MARKER_PATH = RUNTIME_MARKER_DIR / "docling-vram-shutdown.json"
GPU_SERVICE_LOADING_MARKER_PATH = RUNTIME_MARKER_DIR / "gpu-service-loading.json"
MARKER_TTL_S = 300.0
_ASYNC_TRIGGER_PATHS: frozenset[str] = frozenset({
    "/v1alpha/convert/source/async",
    "/v1alpha/convert/file/async",
    "/v1/convert/source/async",
    "/v1/convert/file/async",
    "/v1/chunk/hybrid/source/async",
    "/v1/chunk/hybrid/file/async",
    "/v1/chunk/hierarchical/source/async",
    "/v1/chunk/hierarchical/file/async",
})


def _match_async_trigger_path(path: str) -> bool:
    # Trailing-Slash-Toleranz analog zu _STATUS_POLL_PATTERN/_RESULT_PATH_PATTERN
    if path in _ASYNC_TRIGGER_PATHS:
        return True
    if len(path) > 1 and path.endswith("/") and path[:-1] in _ASYNC_TRIGGER_PATHS:
        return True
    return False


_STATUS_POLL_PATTERN = re.compile(
    r"^/v1(?:alpha)?/status/poll/([A-Za-z0-9_-]+)/?$"
)
_RESULT_PATH_PATTERN = re.compile(
    r"^/v1(?:alpha)?/result/([A-Za-z0-9_-]+)/?$"
)
_TERMINAL_STATUS: frozenset[str] = frozenset({
    "success", "failure", "partial_success",
})
_GPU_INFLIGHT_PATHS: frozenset[str] = frozenset({
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "/api/rerank",
    "/api/score",
    "/v1/chat/completions",
})
JOURNAL_CRIT = 2
JOURNAL_ERR = 3
JOURNAL_WARNING = 4
TASK_MAX_AGE_S: float = 4 * 3600.0


SlotToken = int


def _journal_log(priority: int, message: str) -> None:
    """Schreibt mit systemd-kompatiblem Syslog-Priority-Prefix nach stderr."""
    print(f"<{priority}>{message}", file=sys.stderr, flush=True)


class VramGateError(RuntimeError):
    """Pre-start VRAM gate failed; caller must not start docling."""

    def __init__(self, message: str, *, side_effects_started: bool = False):
        super().__init__(message)
        self.side_effects_started = side_effects_started


# --- State-Modell ---

class State(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED_DIRTY = "stopped_dirty"
    SHUTDOWN = "shutdown"


_state: State = State.STOPPED
_active_requests: int = 0
_slot_generation: int = 0
_last_request_time: float = 0.0
_backend_failed: bool = False

_state_lock = asyncio.Lock()
_state_changed = asyncio.Condition(_state_lock)
_warm_regate_done: asyncio.Event = asyncio.Event()
_warm_regate_done.set()

_idle_watcher_task: asyncio.Task | None = None
_stop_task: asyncio.Task | None = None
_start_task: asyncio.Task | None = None
_dirty_retry_task: asyncio.Task | None = None
_start_failure_count: int = 0
_start_failure_window_start: float = 0.0
_start_cooldown_until: float = 0.0


@dataclass
class _TaskEntry:
    created_monotonic: float
    last_poll_monotonic: float
    generation: int = 0
    terminal: bool = False
    status: str | None = None


_tasks: dict[str, _TaskEntry] = {}
_tasks_lock: asyncio.Lock = asyncio.Lock()


async def _to_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# --- Container-Management ---

def _is_running() -> bool:
    """docker inspect als Best-Effort-Helfer fuer Lifecycle-Raender.

    Alle erwarteten und unerwarteten Fehler werden als False behandelt,
    damit Startup, Shutdown und Cleanup nie daran abbrechen (F132).
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() == "true"
    except subprocess.TimeoutExpired:
        _journal_log(JOURNAL_WARNING, "docker inspect Timeout")
        return False
    except (FileNotFoundError, OSError) as e:
        _journal_log(JOURNAL_WARNING, f"docker inspect nicht ausfuehrbar: {e}")
        return False
    except Exception as e:
        _journal_log(JOURNAL_WARNING, f"docker inspect unerwarteter Fehler: {e}")
        return False


def _has_expected_image() -> bool:
    """True nur wenn der Container exakt mit dem erwarteten Image erstellt wurde."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.Config.Image}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()[:200]
            _journal_log(
                JOURNAL_WARNING,
                f"Docker-Image-Inspect exit {r.returncode}: {stderr}",
            )
            return False
        image = (r.stdout or "").strip()
        if image == EXPECTED_CONTAINER_IMAGE:
            return True
        _journal_log(
            JOURNAL_WARNING,
            "Docling-Container-Image unerwartet: "
            f"{image!r} != {EXPECTED_CONTAINER_IMAGE!r}",
        )
        return False
    except subprocess.TimeoutExpired:
        _journal_log(JOURNAL_WARNING, "Docker-Image-Inspect Timeout")
        return False
    except (FileNotFoundError, OSError) as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Docker-Image-Inspect nicht ausfuehrbar: {e}",
        )
        return False
    except Exception as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Docker-Image-Inspect unerwarteter Fehler: {e}",
        )
        return False


def _has_expected_port_binding() -> bool:
    """True nur wenn der Container 5001/tcp auf 127.0.0.1:5002 mappt (#877).

    Ohne diese Pruefung wuerde _wait_healthy ein HTTP 200 von 127.0.0.1:5002
    auch dann akzeptieren, wenn ein Drittprozess auf dem Port lauscht oder der
    Container mit anderem Port-Mapping erstellt wurde.
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f",
             "{{json .HostConfig.PortBindings}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()[:200]
            _journal_log(
                JOURNAL_WARNING,
                f"Docker-PortBindings-Inspect exit {r.returncode}: {stderr}",
            )
            return False
        try:
            bindings = json.loads((r.stdout or "").strip() or "{}")
        except json.JSONDecodeError as e:
            _journal_log(
                JOURNAL_WARNING,
                f"Docker-PortBindings-Inspect JSON-Fehler: {e}",
            )
            return False
        if not isinstance(bindings, dict):
            _journal_log(
                JOURNAL_WARNING,
                "Docker-PortBindings-Inspect "
                f"unerwarteter Typ: {type(bindings).__name__}",
            )
            return False
        binding_list = bindings.get(EXPECTED_CONTAINER_PORT_SPEC)
        if not isinstance(binding_list, list) or not binding_list:
            _journal_log(
                JOURNAL_WARNING,
                "Docling-Container-PortBinding fehlt: "
                f"{EXPECTED_CONTAINER_PORT_SPEC} -> "
                f"{EXPECTED_HOST_BIND_IP}:{EXPECTED_HOST_BIND_PORT}",
            )
            return False
        for entry in binding_list:
            if not isinstance(entry, dict):
                continue
            host_ip = entry.get("HostIp")
            host_port = entry.get("HostPort")
            if (host_ip == EXPECTED_HOST_BIND_IP
                    and str(host_port) == EXPECTED_HOST_BIND_PORT):
                return True
        _journal_log(
            JOURNAL_WARNING,
            "Docling-Container-PortBinding unerwartet: "
            f"{binding_list!r} != "
            f"{EXPECTED_HOST_BIND_IP}:{EXPECTED_HOST_BIND_PORT}",
        )
        return False
    except subprocess.TimeoutExpired:
        _journal_log(JOURNAL_WARNING, "Docker-PortBindings-Inspect Timeout")
        return False
    except (FileNotFoundError, OSError) as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Docker-PortBindings-Inspect nicht ausfuehrbar: {e}",
        )
        return False
    except Exception as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Docker-PortBindings-Inspect unerwarteter Fehler: {e}",
        )
        return False


def _is_running_for_dirty() -> bool:
    """Konservativer Dirty-Inspect fuer Stop-Fehler-Entscheidungen.

    Anders als `_is_running()` ist dieser Helfer konservativ: ein unklarer
    Inspect-Fehler gilt als "laeuft noch", damit ein Stop-Fehler nicht
    faelschlich als sauberes STOPPED interpretiert wird.

    - `returncode=0, stdout=="true"` -> True
    - `returncode=0, stdout=="false"` -> False
    - `returncode!=0` mit "No such object" im stderr -> False
      (Container existiert nicht mehr)
    - `returncode!=0` mit anderem/unklarem Fehler -> True (konservativ)
    - `returncode=0` mit leerem/unerwartetem stdout -> True (konservativ)
    - Timeout, FileNotFoundError, OSError, sonstige Exception -> True
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            out = r.stdout.strip()
            if out == "true":
                return True
            if out == "false":
                return False
            _journal_log(
                JOURNAL_WARNING,
                f"Dirty-Inspect unerwartete Ausgabe: {out!r}",
            )
            return True
        stderr = (r.stderr or "").strip().lower()
        if "no such object" in stderr:
            return False
        _journal_log(
            JOURNAL_WARNING,
            f"Dirty-Inspect exit {r.returncode}: {stderr[:200]}",
        )
        return True
    except subprocess.TimeoutExpired:
        _journal_log(JOURNAL_WARNING, "Dirty-Inspect Timeout")
        return True
    except (FileNotFoundError, OSError) as e:
        _journal_log(JOURNAL_WARNING, f"Dirty-Inspect nicht ausfuehrbar: {e}")
        return True
    except Exception as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Dirty-Inspect unerwarteter Fehler: {e}",
        )
        return True


def _runtime_marker_group_gid() -> int | None:
    try:
        return grp.getgrnam(RUNTIME_MARKER_GROUP).gr_gid
    except KeyError:
        return None


def _runtime_marker_file_owner_uids() -> frozenset[int]:
    uids: set[int] = set()
    for name in RUNTIME_MARKER_FILE_OWNER_NAMES:
        try:
            uids.add(pwd.getpwnam(name).pw_uid)
        except KeyError:
            continue
    return frozenset(uids)


def _runtime_marker_dir_owner_uid() -> int:
    return 0


def _stat_is_safe_runtime_file(st: os.stat_result) -> tuple[bool, str]:
    if not stat.S_ISREG(st.st_mode):
        return False, "not_regular"
    if st.st_uid not in _runtime_marker_file_owner_uids():
        return False, "file_owner_not_allowed"
    runtime_gid = _runtime_marker_group_gid()
    if runtime_gid is None:
        return False, "runtime_group_missing"
    if st.st_gid != runtime_gid:
        return False, "file_group_not_runtime"
    if st.st_mode & stat.S_IRWXO:
        return False, "file_world_bits"
    if (st.st_mode & 0o777) != RUNTIME_MARKER_FILE_MODE:
        return False, "file_mode_not_0660"
    return True, "ok"


def _fd_is_safe_runtime_file(fd: int) -> tuple[bool, str]:
    return _stat_is_safe_runtime_file(os.fstat(fd))


def _runtime_file_identity(
    st: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
    ctime_ns = getattr(st, "st_ctime_ns", int(st.st_ctime * 1_000_000_000))
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode,
        st.st_uid,
        st.st_gid,
        st.st_size,
        mtime_ns,
        ctime_ns,
    )


def _open_existing_runtime_file(
    path: Path,
    flags: int,
) -> tuple[int | None, str, os.stat_result | None]:
    parent_ok, parent_reason = _path_is_safe_runtime_dir(path.parent)
    if not parent_ok:
        return None, parent_reason, None
    open_flags = flags | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, open_flags)
    except FileNotFoundError:
        return None, "missing", None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None, "symlink", None
        return None, f"open_failed:{exc}", None
    try:
        st = os.fstat(fd)
        ok, reason = _stat_is_safe_runtime_file(st)
        if not ok:
            os.close(fd)
            return None, reason, st
        return fd, "ok", st
    except Exception:
        os.close(fd)
        raise


def _path_is_safe_runtime_dir(path: Path) -> tuple[bool, str]:
    try:
        st = os.lstat(path)
    except OSError as exc:
        return False, f"parent_stat_failed:{exc}"
    if stat.S_ISLNK(st.st_mode):
        return False, "parent_symlink"
    if not stat.S_ISDIR(st.st_mode):
        return False, "parent_not_directory"
    if st.st_uid != _runtime_marker_dir_owner_uid():
        return False, "parent_owner_not_root"
    runtime_gid = _runtime_marker_group_gid()
    if runtime_gid is None:
        return False, "runtime_group_missing"
    if st.st_gid != runtime_gid:
        return False, "parent_group_not_runtime"
    if st.st_mode & stat.S_IRWXO:
        return False, "parent_world_bits"
    if (st.st_mode & 0o7777) != RUNTIME_MARKER_DIR_MODE:
        return False, "parent_mode_not_2770"
    return True, "ok"


def _path_is_safe_regular_file(path: Path) -> tuple[bool, str, os.stat_result | None]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False, "missing", None
    except OSError as exc:
        return False, f"stat_failed:{exc}", None
    if stat.S_ISLNK(st.st_mode):
        return False, "symlink", st
    if not stat.S_ISREG(st.st_mode):
        return False, "not_regular", st
    parent_ok, parent_reason = _path_is_safe_runtime_dir(path.parent)
    if not parent_ok:
        return False, parent_reason, st
    ok, reason = _stat_is_safe_runtime_file(st)
    return ok, reason, st


def _marker_payload_with_stat(
    path: Path,
) -> tuple[dict | None, bool, os.stat_result | None]:
    fd, reason, st = _open_existing_runtime_file(path, os.O_RDONLY)
    if fd is None:
        if reason == "missing":
            return None, False, None
        _journal_log(JOURNAL_WARNING, f"VRAM-Marker {path} unsafe: {reason}")
        return None, True, st
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = None
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _journal_log(
            JOURNAL_WARNING,
            f"VRAM-Marker {path} korrupt ({exc}) — wird ueberschrieben",
        )
        return None, False, st
    finally:
        if fd is not None:
            os.close(fd)
    if not isinstance(data, dict):
        _journal_log(
            JOURNAL_WARNING,
            f"VRAM-Marker {path} korrupt (non-dict) — wird ueberschrieben",
        )
        return None, False, st
    deadline = data.get("deadline_monotonic")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        return data, time.monotonic() < float(deadline), st
    created_wall = data.get("created_wall")
    ttl = data.get("ttl_s")
    if (
        isinstance(created_wall, (int, float))
        and not isinstance(created_wall, bool)
        and isinstance(ttl, (int, float))
        and not isinstance(ttl, bool)
    ):
        active = time.time() < (
            float(created_wall) + max(0.0, min(float(ttl), MARKER_TTL_S))
        )
        return data, active, st
    _journal_log(
        JOURNAL_WARNING,
        f"VRAM-Marker {path} ohne valide TTL-Felder — wird ueberschrieben",
    )
    return None, False, st


def _marker_payload(path: Path) -> tuple[dict | None, bool]:
    data, active, _ = _marker_payload_with_stat(path)
    return data, active


def _marker_active(path: Path) -> bool:
    _, active = _marker_payload(path)
    return active


def _marker_path(kind: str) -> Path:
    if kind == "startup":
        return STARTUP_MARKER_PATH
    if kind == "shutdown":
        return SHUTDOWN_MARKER_PATH
    raise ValueError(f"unknown vram marker kind: {kind!r}")


def _marker_lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _open_marker_lock(path: Path) -> int:
    lock_path = _marker_lock_path(path)
    lock_path.parent.mkdir(
        parents=True,
        mode=RUNTIME_MARKER_DIR_MODE,
        exist_ok=True,
    )
    parent_ok, parent_reason = _path_is_safe_runtime_dir(lock_path.parent)
    if not parent_ok:
        raise VramGateError(
            f"VRAM-Lock-Verzeichnis {lock_path.parent} unsafe: {parent_reason}"
        )
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    created = False
    for _ in range(2):
        try:
            lock_fd = os.open(
                lock_path,
                flags | os.O_CREAT | os.O_EXCL,
                RUNTIME_MARKER_FILE_MODE,
            )
            created = True
            break
        except FileExistsError:
            try:
                lock_fd = os.open(lock_path, flags)
                break
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise VramGateError(
                    f"VRAM-Lock {lock_path} unsafe: {exc}"
                ) from exc
        except OSError as exc:
            raise VramGateError(f"VRAM-Lock {lock_path} unsafe: {exc}") from exc
    else:
        raise VramGateError(f"VRAM-Lock {lock_path} unstable")
    try:
        if created:
            os.fchmod(lock_fd, RUNTIME_MARKER_FILE_MODE)
        ok, reason = _fd_is_safe_runtime_file(lock_fd)
        if not ok:
            raise VramGateError(f"VRAM-Lock {lock_path} unsafe: {reason}")
    except Exception:
        os.close(lock_fd)
        raise
    return lock_fd


def _write_vram_marker(
    kind: str = "startup", ttl_s: float = MARKER_TTL_S, token: str | None = None,
) -> str:
    token = token or uuid.uuid4().hex
    path = _marker_path(kind)
    path.parent.mkdir(parents=True, mode=RUNTIME_MARKER_DIR_MODE, exist_ok=True)
    payload = {
        "token": token,
        "kind": kind,
        "pid": os.getpid(),
        "created_wall": time.time(),
        "deadline_monotonic": time.monotonic() + ttl_s,
        "ttl_s": ttl_s,
    }
    lock_fd = _open_marker_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current, active = _marker_payload(path)
        if active:
            current_token = current.get("token") if isinstance(current, dict) else None
            if not isinstance(current_token, str) or current_token != token:
                raise VramGateError(f"aktiver fremder {kind}-Marker")
        tmp = path.with_name(f".{path.name}.{token}.tmp")
        fd: int | None = os.open(
            tmp,
            (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
            ),
            RUNTIME_MARKER_FILE_MODE,
        )
        try:
            os.fchmod(fd, RUNTIME_MARKER_FILE_MODE)
            ok, reason = _fd_is_safe_runtime_file(fd)
            if not ok:
                raise VramGateError(f"VRAM-Marker {tmp} unsafe: {reason}")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = None
                json.dump(payload, fh, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return token


def _clear_vram_marker(kind: str = "startup", token: str | None = None) -> None:
    if token is None:
        return
    path = _marker_path(kind)
    path.parent.mkdir(parents=True, mode=RUNTIME_MARKER_DIR_MODE, exist_ok=True)
    lock_fd = _open_marker_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data, active, marker_st = _marker_payload_with_stat(path)
        if data is None and active:
            return
        if data is not None and data.get("token") != token:
            return
        if marker_st is not None:
            try:
                current_st = os.lstat(path)
            except OSError:
                return
            if _runtime_file_identity(current_st) != _runtime_file_identity(marker_st):
                _journal_log(
                    JOURNAL_WARNING,
                    f"VRAM-Marker {path} changed before clear",
                )
                return
            ok, reason = _stat_is_safe_runtime_file(current_st)
            if not ok:
                _journal_log(
                    JOURNAL_WARNING,
                    f"VRAM-Marker {path} unsafe before clear: {reason}",
                )
                return
        try:
            path.unlink()
        except OSError:
            pass
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _restart_policy_safe() -> bool | None:
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{json .HostConfig.RestartPolicy}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        policy = json.loads((r.stdout or "").strip() or "{}")
    except Exception:
        return None
    if not isinstance(policy, dict):
        return None
    return policy.get("Name") in ("", "no", None)


def _remediate_restart_policy() -> bool:
    safe = _restart_policy_safe()
    if safe is True:
        return True
    if safe is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "update", "--restart=no", CONTAINER_NAME],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    except Exception:
        return False
    return r.returncode == 0


def _real_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _set_state(new: "State") -> None:
    """Sync-Helfer fuer State-Wechsel mit Diagnose-Logging.

    Caller MUSS `_state_lock` halten. No-op-Uebergaenge werden nicht
    geloggt, um Spam bei Reentry DIRTY->DIRTY zu vermeiden.
    """
    global _state
    old = _state
    if old is new:
        return
    print(f"  State: {old.value} -> {new.value}")
    _state = new


def _invalidate_generation_locked() -> None:
    """Invalidate all outstanding slot/task ownership for the current backend."""
    global _slot_generation, _active_requests, _last_request_time, _backend_failed
    _slot_generation += 1
    _active_requests = 0
    _backend_failed = False
    _last_request_time = 0.0
    _warm_regate_done.set()


async def _vram_lease_active() -> bool:
    """True solange Docling die GPU exklusiv beanspruchen soll (#285).

    Lease aktiv in:
      - STARTING (Window A+B: Cold-Start laeuft, VRAM wird allokiert)
      - STOPPED_DIRTY (Container koennte noch VRAM halten bis Retry erfolgreich)
      - STOPPING (#503: docker stop -t 10 + Daemon/Kernel-Cleanup = 5-15s
        in denen der Container noch ~7 GiB VRAM haelt; erst _finalize_stop
        setzt STOPPED/STOPPED_DIRTY)
      - RUNNING + active_requests > 0 (aktive Konvertierung)
      - RUNNING + non-terminale Async-Tasks der aktuellen Generation
        (#857: Async-Trigger geben den Slot nach _register_task_from_response
        frei, der Backend-Task konvertiert weiter — solange er nicht
        terminal ist, haelt Docling weiter VRAM)

    Lease NICHT aktiv in:
      - STOPPED (keine Docling-Nutzung)
      - RUNNING + active_requests == 0 ohne non-terminale Tasks (warm-hold)
      - SHUTDOWN (terminal)

    Liest `_state`/`_active_requests` und ruft `_has_nonterminal_tasks()`,
    das `_tasks_lock` haelt. Caller muss `_state_lock` halten — Lock-
    Ordering state_lock -> tasks_lock ist konsistent mit
    `_try_stop_if_idle`. Async, weil tasks_lock async ist.
    """
    if _state in (State.STARTING, State.STOPPED_DIRTY, State.STOPPING):
        return True
    if _state == State.RUNNING:
        if _active_requests > 0:
            return True
        if await _has_nonterminal_tasks():
            return True
    return False


def _in_cooldown_locked() -> bool:
    """True while start-failure cooldown is active. Caller holds _state_lock."""
    return time.monotonic() < _start_cooldown_until


def _record_start_failure_locked() -> None:
    """Increment start failures and enter cooldown at threshold.

    Caller holds `_state_lock`. Aufgerufen aus beiden Cold-Start-Failure-
    Pfaden (STOPPED und STOPPED_DIRTY), damit persistente Health-Failures
    ueber Dirty-Retry-Cycles hinweg im Cooldown-Counter erfasst werden.
    """
    global _start_failure_count
    global _start_failure_window_start
    global _start_cooldown_until
    now = time.monotonic()
    if (_start_failure_window_start <= 0.0
            or (now - _start_failure_window_start) > START_FAILURE_WINDOW_S):
        _start_failure_window_start = now
        _start_failure_count = 0
    _start_failure_count += 1
    if _start_failure_count >= START_FAILURE_THRESHOLD:
        _start_cooldown_until = now + START_COOLDOWN_DURATION_S
        logging.getLogger(__name__).warning(
            "#582: Start-Cooldown aktiv fuer %.0fs nach %d Failures "
            "im %.0fs-Fenster",
            START_COOLDOWN_DURATION_S,
            _start_failure_count,
            START_FAILURE_WINDOW_S,
        )


def _record_start_success_locked() -> None:
    """Reset start-failure counters after a successful cold start."""
    global _start_failure_count
    global _start_failure_window_start
    global _start_cooldown_until
    had_cooldown = _start_cooldown_until > 0.0
    _start_failure_count = 0
    _start_failure_window_start = 0.0
    _start_cooldown_until = 0.0
    if had_cooldown:
        logging.getLogger(__name__).info(
            "#582: Start erfolgreich, Start-Cooldown geloescht"
        )


def _gc_tasks_locked(now: float) -> None:
    """Evict task entries older than TASK_MAX_AGE_S.

    Terminal tasks: by `created_monotonic` (retention age).
    Non-terminal tasks: by `last_poll_monotonic` so an actively polled
    long-running job is not evicted while still being observed.

    Caller must hold `_tasks_lock`.
    """
    evict = []
    for task_id, entry in _tasks.items():
        ref = entry.created_monotonic if entry.terminal else entry.last_poll_monotonic
        if (now - ref) > TASK_MAX_AGE_S:
            evict.append(task_id)
    for task_id in evict:
        entry = _tasks.pop(task_id, None)
        if entry is not None and not entry.terminal:
            age = now - entry.last_poll_monotonic
            _journal_log(
                JOURNAL_WARNING,
                "#564 Task GC-Evict trotz non-terminal "
                f"task_id={task_id} stale_age={age:.0f}s",
            )


async def _register_task_from_response(path: str, body_bytes: bytes, token: SlotToken | None = None) -> None:
    """Register docling async task IDs from trigger responses.

    Lock-Ordering: Reads `_slot_generation` ohne `_state_lock`. Sicher weil:
    (1) Python int-Read ist unter dem GIL atomar — keine Torn-Reads.
    (2) Re-Check innerhalb `_tasks_lock` (vor dem _tasks-Insert) faengt
        zwischenzeitliche Generation-Bumps ab.
    (3) `_has_nonterminal_tasks` filtert per Generation, sodass Stale-Tasks
        aus einem Race-Fenster ignoriert und spaetestens nach
        TASK_MAX_AGE_S durch `_gc_tasks_locked` evicted werden.
    Gleiche Konvention in `_mark_result_task_success`, `_refresh_task_from_poll`,
    `_has_nonterminal_tasks`.
    """
    if token is None:
        token = _slot_generation
    if token != _slot_generation:
        return
    if not _match_async_trigger_path(path):
        return
    try:
        data = json.loads(body_bytes)
    except (ValueError, json.JSONDecodeError):
        return
    task_id = data.get("task_id") if isinstance(data, dict) else None
    if not isinstance(task_id, str) or not task_id:
        return
    now = time.monotonic()
    async with _tasks_lock:
        if token != _slot_generation:
            return
        _gc_tasks_locked(now)
        _tasks[task_id] = _TaskEntry(
            created_monotonic=now,
            last_poll_monotonic=now,
            generation=token,
        )
    print(f"  INFO: #564 Task registriert task_id={task_id}")


async def _mark_result_task_success(path: str, token: SlotToken | None = None) -> None:
    """Mark successful result downloads terminal without buffering the body."""
    if token is None:
        token = _slot_generation
    if token != _slot_generation:
        return
    match = _RESULT_PATH_PATTERN.match(path)
    if match is None:
        return
    task_id = match.group(1)
    now = time.monotonic()
    async with _tasks_lock:
        if token != _slot_generation:
            return
        _gc_tasks_locked(now)
        entry = _tasks.get(task_id)
        if entry is None:
            _tasks[task_id] = _TaskEntry(
                created_monotonic=now,
                last_poll_monotonic=now,
                generation=token,
                terminal=True,
                status="success",
            )
            print(f"  INFO: #564 Task terminal task_id={task_id} status=success")
            return
        if entry.generation != token:
            return
        entry.last_poll_monotonic = now
        entry.status = "success"
        if not entry.terminal:
            entry.terminal = True
            print(f"  INFO: #564 Task terminal task_id={task_id} status=success")


async def _refresh_task_from_poll(path: str, body_bytes: bytes, token: SlotToken | None = None) -> None:
    """Update task registry from small status-poll responses."""
    if token is None:
        token = _slot_generation
    if token != _slot_generation:
        return
    match = _STATUS_POLL_PATTERN.match(path)
    if match is None:
        return
    task_id = match.group(1)
    try:
        data = await asyncio.to_thread(json.loads, body_bytes)
    except (ValueError, json.JSONDecodeError):
        data = {}
    status = data.get("task_status") if isinstance(data, dict) else None
    if not isinstance(status, str):
        status = None
    terminal = status in _TERMINAL_STATUS if status is not None else False
    now = time.monotonic()
    async with _tasks_lock:
        if token != _slot_generation:
            return
        _gc_tasks_locked(now)
        entry = _tasks.get(task_id)
        if entry is None:
            _tasks[task_id] = _TaskEntry(
                created_monotonic=now,
                last_poll_monotonic=now,
                generation=token,
                terminal=terminal,
                status=status,
            )
            if terminal:
                print(f"  INFO: #564 Task terminal task_id={task_id} "
                      f"status={status}")
            return
        if entry.generation != token:
            return
        entry.last_poll_monotonic = now
        if status is not None:
            entry.status = status
            if terminal and not entry.terminal:
                entry.terminal = True
                print(f"  INFO: #564 Task terminal task_id={task_id} "
                      f"status={status}")


async def _has_nonterminal_tasks() -> bool:
    """Return True while any tracked async task is not terminal."""
    now = time.monotonic()
    generation = _slot_generation
    async with _tasks_lock:
        _gc_tasks_locked(now)
        return any(
            entry.generation == generation and not entry.terminal
            for entry in _tasks.values()
        )


def _active_gpu_requests_from_snapshot(data: object) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("requests"), list):
        raise VramGateError("GPU-In-Flight Drain: active-requests Shape unbekannt")
    active: list[dict] = []
    for item in data.get("requests", []):
        if not isinstance(item, dict):
            raise VramGateError("GPU-In-Flight Drain: requests[] Shape unbekannt")
        path = item.get("path")
        if not isinstance(path, str):
            continue
        normalized = path.rstrip("/") if len(path) > 1 else path
        if normalized not in _GPU_INFLIGHT_PATHS:
            continue
        if item.get("state", "active") != "active":
            continue
        active.append(item)
    return active


async def _guard_generation(expected_generation: SlotToken | None) -> None:
    if expected_generation is None:
        async with _state_lock:
            if _state == State.SHUTDOWN:
                raise VramGateError("shutdown")
            return
    async with _state_lock:
        if _slot_generation != expected_generation or _state != State.RUNNING:
            raise VramGateError("stale_regate")


async def _wait_gpu_service_loading_clear(
    deadline_s: float,
    expected_generation: SlotToken | None,
) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        await _guard_generation(expected_generation)
        if not _marker_active(GPU_SERVICE_LOADING_MARKER_PATH):
            return
        if time.monotonic() >= deadline:
            raise VramGateError("gpu_service_loading_timeout")
        await asyncio.sleep(GPU_SERVICE_LOADING_DRAIN_POLL_S)


async def _drain_inflight_gpu_requests(guard=None) -> None:
    """Drain GPU requests that passed the lease before marker write (#620)."""
    if not KIRON_ACTIVE_REQUESTS_URL:
        raise VramGateError(
            "GPU-In-Flight Drain nicht verfuegbar: "
            "KIRON_ACTIVE_REQUESTS_URL ist nicht konfiguriert"
        )
    deadline = time.monotonic() + GPU_INFLIGHT_DRAIN_TIMEOUT_S
    logged_wait = False
    while True:
        if guard is not None:
            await guard()
        try:
            async with httpx.AsyncClient(
                timeout=GPU_INFLIGHT_DRAIN_HTTP_TIMEOUT_S,
            ) as c:
                resp = await c.get(
                    KIRON_ACTIVE_REQUESTS_URL,
                    auth=(KIRON_DASHBOARD_USER, KIRON_DASHBOARD_PASSWORD),
                )
            if resp.status_code != 200:
                raise VramGateError(
                    "GPU-In-Flight Drain nicht verfuegbar: "
                    f"HTTP {resp.status_code}"
                )
            active = _active_gpu_requests_from_snapshot(resp.json())
        except VramGateError:
            raise
        except Exception as e:
            raise VramGateError(
                "GPU-In-Flight Drain nicht verfuegbar: "
                f"{type(e).__name__}: {e}"
            ) from e

        if not active:
            return
        if not logged_wait:
            names = ", ".join(
                str(item.get("model") or item.get("path") or "?")
                for item in active[:5]
            )
            print("  Warte auf aktive GPU-Requests vor Docling-Start: "
                  f"{len(active)} ({names})")
            logged_wait = True
        if time.monotonic() >= deadline:
            raise VramGateError(
                "GPU-In-Flight Drain Timeout: "
                f"{len(active)} aktive Request(s)"
            )
        await asyncio.sleep(GPU_INFLIGHT_DRAIN_POLL_S)


async def _prepare_vram_for_docling(expected_generation: SlotToken | None = None) -> None:
    async def _guard() -> None:
        await _guard_generation(expected_generation)

    await _wait_gpu_service_loading_clear(
        GPU_SERVICE_LOADING_DRAIN_TIMEOUT_S,
        expected_generation,
    )
    try:
        await _drain_inflight_gpu_requests(guard=_guard)
    except TypeError as exc:
        if "guard" not in str(exc):
            raise
        await _drain_inflight_gpu_requests()
    try:
        await _free_vram_for_docling(guard=_guard)
    except TypeError as exc:
        if "guard" not in str(exc):
            raise
        await _free_vram_for_docling()


async def _call_prepare_vram_for_docling(expected_generation: SlotToken | None) -> None:
    try:
        await _prepare_vram_for_docling(expected_generation=expected_generation)
    except TypeError as exc:
        if "expected_generation" not in str(exc):
            raise
        await _prepare_vram_for_docling()


def _ensure_retry_scheduler() -> None:
    """Startet den Dirty-Retry-Scheduler, falls noch keiner laeuft.

    Caller MUSS `_state_lock` halten. Idempotent: no-op wenn bereits ein
    laufender Task existiert. Genau EIN Task pro DIRTY-Periode.
    """
    global _dirty_retry_task
    task = _dirty_retry_task
    if task is not None and not task.done():
        return
    _dirty_retry_task = asyncio.create_task(_dirty_retry_loop())


async def _dirty_retry_loop() -> None:
    """Periodisch Stop wiederholen, bis Dirty-Inspect 'not running' liefert.

    Terminiert sich selbst bei State-Wechsel aus DIRTY oder nach
    `_DIRTY_RETRY_MAX_ATTEMPTS`. Wird ausschliesslich von
    `on_shutdown` extern gecancelt.
    """
    global _last_request_time, _backend_failed
    attempts = 0
    while True:
        await asyncio.sleep(_DIRTY_RETRY_INTERVAL_S)
        async with _state_lock:
            if _state != State.STOPPED_DIRTY:
                return
        attempts += 1
        try:
            still = await _to_thread(_is_running_for_dirty)
        except Exception as e:
            _journal_log(
                JOURNAL_WARNING,
                f"Dirty-Retry-Inspect Exception: {e}",
            )
            still = True
        if not still:
            async with _state_lock:
                if _state == State.STOPPED_DIRTY:
                    _invalidate_generation_locked()
                    _set_state(State.STOPPED)
                    _state_changed.notify_all()
            return
        try:
            stop_ok = await _run_stop_once()
        except Exception as e:
            _journal_log(
                JOURNAL_WARNING,
                f"Dirty-Retry-Stop Exception: {e}",
            )
            stop_ok = False
        await _finalize_stop(stop_ok)
        async with _state_lock:
            if _state != State.STOPPED_DIRTY:
                return
        if attempts >= _DIRTY_RETRY_MAX_ATTEMPTS:
            _journal_log(
                JOURNAL_CRIT,
                f"Dirty-Retry MAX ({_DIRTY_RETRY_MAX_ATTEMPTS}) erreicht "
                "— manueller Eingriff noetig",
            )
            return


async def _free_vram_for_docling(guard=None) -> None:
    """Unload enough Ollama VRAM for docling, fail-closed on unknown state."""
    side_effects_started = False

    async def _guard() -> None:
        if guard is not None:
            try:
                await guard()
            except VramGateError as exc:
                exc.side_effects_started = exc.side_effects_started or side_effects_started
                raise

    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=10.0) as c:
        await _guard()
        try:
            r = await c.get("/api/ps")
            r.raise_for_status()
            ps_data = r.json()
        except Exception as e:
            raise VramGateError(f"VRAM /api/ps fehlgeschlagen: {e}") from e
        if not isinstance(ps_data, dict) or not isinstance(ps_data.get("models"), list):
            raise VramGateError("VRAM /api/ps Shape unbekannt")

        models = []
        for item in ps_data.get("models", []):
            if not isinstance(item, dict):
                raise VramGateError("VRAM /api/ps models[] Shape unbekannt")
            name = item.get("name")
            vram = item.get("size_vram")
            if not isinstance(name, str) or not name:
                continue
            if not _real_int(vram):
                raise VramGateError(f"VRAM size_vram fuer {name} unbekannt")
            models.append({"name": name, "size_vram": vram})

        ordered = sorted(models, key=lambda x: x["size_vram"])
        kept_total = 0
        to_unload: list[str] = []
        for m in ordered:
            name = m["name"]
            vram = m["size_vram"]
            if kept_total + vram <= VRAM_BUDGET_BYTES:
                kept_total += vram
                print(f"  Behalte {name} ({vram / 1024**3:.1f} GiB)")
            else:
                to_unload.append(name)
                print(f"  Entlade {name} ({vram / 1024**3:.1f} GiB)")

        if not to_unload:
            return

        for name in to_unload:
            await _guard()
            try:
                side_effects_started = True
                r = await c.post(
                    "/api/generate",
                    json={"model": name, "keep_alive": 0, "stream": False},
                    timeout=15.0,
                )
                r.raise_for_status()
            except Exception as e:
                raise VramGateError(
                    f"Unload {name} fehlgeschlagen: {e}",
                    side_effects_started=side_effects_started,
                ) from e

        deadline = time.monotonic() + VRAM_VERIFY_TIMEOUT_S
        remaining_names = set(to_unload)
        consecutive_errors = 0
        while time.monotonic() < deadline:
            await _guard()
            try:
                r = await c.get("/api/ps")
                r.raise_for_status()
                verify_data = r.json()
                if not isinstance(verify_data, dict) or not isinstance(verify_data.get("models"), list):
                    raise VramGateError(
                        "VRAM-Verify Shape unbekannt",
                        side_effects_started=side_effects_started,
                    )
                loaded = set()
                for item in verify_data.get("models", []):
                    if not isinstance(item, dict):
                        raise VramGateError(
                            "VRAM-Verify models[] Shape unbekannt",
                            side_effects_started=side_effects_started,
                        )
                    name = item.get("name")
                    if isinstance(name, str):
                        loaded.add(name)
                remaining_names = remaining_names & loaded
                if not remaining_names:
                    return
                consecutive_errors = 0
            except VramGateError:
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise VramGateError(
                        f"VRAM-Verify abgebrochen nach 3 Fehlern: {e}",
                        side_effects_started=side_effects_started,
                    ) from e
            await asyncio.sleep(1.0)

        raise VramGateError(
            f"VRAM-Verify: {len(remaining_names)} Modelle noch geladen nach {VRAM_VERIFY_TIMEOUT_S}s",
            side_effects_started=side_effects_started,
        )


def _start() -> bool:
    """Startet den Container per `docker start`.

    Liefert True, wenn der Container erfolgreich gestartet wurde ODER bereits
    laeuft (F66). Das deckt out-of-band Starts ueber /api/docling/start ab.
    """
    if not _has_expected_image():
        _journal_log(
            JOURNAL_CRIT,
            "Docling-Container-Image konnte nicht validiert werden",
        )
        return False
    if not _remediate_restart_policy():
        _journal_log(
            JOURNAL_CRIT,
            "unsichere RestartPolicy konnte nicht remediated werden",
        )
        return False
    if not _has_expected_port_binding():
        _journal_log(
            JOURNAL_CRIT,
            "Docling-Container-PortBinding konnte nicht validiert werden",
        )
        return False
    try:
        subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if _is_running():
            print("  Hinweis: Container lief bereits (out-of-band start)")
            return True
        _journal_log(JOURNAL_ERR, f"docker start fehlgeschlagen: {stderr}")
        return False
    except subprocess.TimeoutExpired:
        # Daemon kann Start nach gekilltem CLI-Prozess noch vollenden —
        # sonst bleibt der Container unmanaged und belegt VRAM/Port.
        # Bounded nachpolling schliesst das Race-Fenster zwischen CLI-Kill
        # und Daemon-Completion, damit ein spaeter doch laufender Container
        # nicht mit STOPPED-State unmanaged zurueckbleibt.
        deadline = time.monotonic() + 15.0
        while True:
            if _is_running():
                print("  Hinweis: docker start Timeout, Container laeuft aber")
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        _journal_log(JOURNAL_ERR, "docker start Timeout")
        return False
    except (FileNotFoundError, OSError) as e:
        _journal_log(JOURNAL_ERR, f"docker start nicht ausfuehrbar: {e}")
        return False


def _stop() -> bool:
    try:
        r = subprocess.run(
            ["docker", "stop", "-t", "10", CONTAINER_NAME],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()[:200]
            _journal_log(
                JOURNAL_WARNING,
                f"docker stop exit {r.returncode}: {stderr}",
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        _journal_log(JOURNAL_WARNING, "docker stop Timeout")
        return False
    except (FileNotFoundError, OSError) as e:
        _journal_log(JOURNAL_WARNING, f"docker stop nicht ausfuehrbar: {e}")
        return False


async def _wait_healthy(timeout: float | None = None) -> bool:
    """Wartet bis docling-serve auf /health antwortet (Wall-Clock-Limit).

    Bricht frueh ab, wenn der Container clearly nicht mehr laeuft (z.B.
    Crash kurz nach `docker start` durch OOM, fehlende NVIDIA-Runtime oder
    Image-Fehler). Andernfalls wuerden /health-Connect-Errors bis
    HEALTH_CHECK_TIMEOUT_S geschluckt und Cold-Start-Caller minutenlang
    blockieren.
    """
    effective = timeout if timeout is not None else HEALTH_CHECK_TIMEOUT_S
    start = time.monotonic()
    deadline = start + effective
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            async with httpx.AsyncClient(timeout=min(2.0, remaining)) as c:
                r = await c.get(f"{BACKEND_URL}/health")
                if r.status_code == 200:
                    elapsed = time.monotonic() - start
                    print(f"  docling-serve bereit nach {elapsed:.1f}s")
                    return True
        except Exception:
            pass
        # Konservativer Inspect: nur ein eindeutig nicht-laufender Container
        # bricht frueh ab; transiente Inspect-Fehler -> weiter warten.
        try:
            still_running = await _to_thread(_is_running_for_dirty)
        except Exception as e:
            _journal_log(
                JOURNAL_WARNING,
                f"_wait_healthy Inspect Exception: {e}",
            )
            still_running = True
        if not still_running:
            elapsed = time.monotonic() - start
            _journal_log(
                JOURNAL_ERR,
                f"{CONTAINER_NAME} nicht mehr running nach "
                f"{elapsed:.1f}s (Container-Crash erkannt)",
            )
            return False
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


# --- Start-/Stop-Task-Serialisierung ---

async def _run_stop_once() -> bool:
    """Single-Flight Wrapper um _stop().

    Stellt sicher, dass Cleanup, idle watcher und Shutdown nie zwei parallele
    Docker-Stops ausloesen (F79). Wartende Caller teilen sich denselben Task
    per `asyncio.shield`, Cancellation des Callers cancelt den Stop nicht.
    """
    global _stop_task

    async def _runner() -> bool:
        try:
            return await _to_thread(_stop)
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"_stop() Exception: {e}")
            return False

    task = _stop_task
    if task is None or task.done():
        task = asyncio.create_task(_runner())
        _stop_task = task
    try:
        return await asyncio.shield(task)
    finally:
        if _stop_task is task and task.done():
            _stop_task = None


async def _run_start_once() -> bool:
    """Single-Flight Supervisor fuer die komplette Cold-Start-Sequenz (F134).

    Besitzt VRAM-Best-Effort, _start(), _wait_healthy(), Health-Failure-
    Cleanup via _run_stop_once() und die finale State-Transition aus
    STARTING. Caller warten per `asyncio.shield`; Caller-Cancellation
    cancelt den Supervisor nicht (F131).
    """
    global _start_task

    task = _start_task
    if task is None or task.done():
        task = asyncio.create_task(_start_supervisor())
        _start_task = task
    try:
        return await asyncio.shield(task)
    finally:
        if _start_task is task and task.done():
            _start_task = None


async def _start_supervisor() -> bool:
    """Die eigentliche Start-Sequenz unter Single-Flight-Ownership.

    Setzt den finalen State unter _state_lock mit SHUTDOWN-Gate (F31).
    BaseException (CancelledError/KeyboardInterrupt) wird nur fuer Cleanup
    beobachtet und danach unveraendert re-raised (F95).

    State-Writes erfolgen ausschliesslich im `finally`-Block, um einen
    F26-Race mit parallelen `ensure_running`-Callern auszuschliessen:
    `_finalize_stop` aus dem try-Block wuerde einen Zwischenzustand
    (STOPPED/STOPPED_DIRTY) veroeffentlichen, waehrend der eigene
    `_start_task` noch nicht `done()` ist — siehe
    Plan/`_start_supervisor`-Refactor.
    """
    global _state, _last_request_time, _backend_failed
    success = False
    had_started = False
    target_dirty = False
    count_start_failure = True
    clear_startup_marker = False
    vram_gate_passed = False
    marker_token: str | None = None
    try:
        try:
            marker_token = _write_vram_marker("startup", ttl_s=MARKER_TTL_S)
        except Exception as e:
            _journal_log(
                JOURNAL_ERR,
                f"VRAM-Startup-Marker konnte nicht gesetzt werden: {e}",
            )
            return False
        try:
            await _call_prepare_vram_for_docling(None)
        except VramGateError as e:
            _journal_log(
                JOURNAL_ERR,
                f"VRAM-Gate blockiert Docling-Start: {e}",
            )
            clear_startup_marker = not e.side_effects_started
            return False
        except Exception as e:
            _journal_log(JOURNAL_ERR, f"VRAM-Gate unerwarteter Fehler: {e}")
            return False
        vram_gate_passed = True

        # Docker-Start + Health
        try:
            started = await _to_thread(_start)
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"_start Exception: {e}")
            started = False
        had_started = started
        if started:
            try:
                healthy = await _wait_healthy()
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"_wait_healthy Exception: {e}")
                healthy = False
            if healthy:
                success = True
            else:
                _journal_log(
                    JOURNAL_ERR,
                    f"{CONTAINER_NAME} nicht bereit nach "
                    f"{HEALTH_CHECK_TIMEOUT_S}s",
                )
                # Health-Failure-Cleanup: best-effort stoppen (F67)
                try:
                    stop_ok = await _run_stop_once()
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"Health-Failure-Stop Exception: {e}",
                    )
                    stop_ok = False
                if not stop_ok:
                    try:
                        target_dirty = await _to_thread(
                            _is_running_for_dirty)
                    except Exception as e:
                        _journal_log(
                            JOURNAL_WARNING,
                            "Health-Failure-Dirty-Inspect "
                            f"Exception: {e}",
                        )
                        target_dirty = True
    except BaseException:
        # Cleanup nur beobachten, danach re-raise (F95).
        # Das finally-Block setzt den State — hier nur Container aufraeumen.
        if had_started:
            cleanup_completed = False
            stop_ok = False
            try:
                stop_ok = await asyncio.shield(_run_stop_once())
                cleanup_completed = True
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"Start-Abbruch-Cleanup Exception: {e}",
                )
                cleanup_completed = True
            except BaseException:
                # Cleanup selbst abgebrochen -> akzeptierter STOPPED-Fallback
                pass
            if cleanup_completed and not stop_ok:
                try:
                    target_dirty = await asyncio.shield(
                        _to_thread(_is_running_for_dirty))
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        "Start-Abbruch-Dirty-Inspect "
                        f"Exception: {e}",
                    )
                    target_dirty = True
                except BaseException:
                    # Inspect selbst abgebrochen -> STOPPED-Fallback
                    pass
        raise
    finally:
        # Finale State-Transition mit SHUTDOWN-Gate (F31)
        async with _state_lock:
            if _state == State.SHUTDOWN:
                # Shutdown-Race: terminalen State nie ueberschreiben
                success = False
            elif success:
                _set_state(State.RUNNING)
                _last_request_time = time.monotonic()
                _record_start_success_locked()
            elif target_dirty:
                _set_state(State.STOPPED_DIRTY)
                _last_request_time = 0.0
                _backend_failed = False
                if count_start_failure:
                    _record_start_failure_locked()
                _ensure_retry_scheduler()
            else:
                _set_state(State.STOPPED)
                _last_request_time = 0.0
                _backend_failed = False
                if count_start_failure:
                    _record_start_failure_locked()
            _state_changed.notify_all()
        # Marker erst nach State-Transition verwalten (#563): Auf dem
        # Erfolgspfad bleibt zwischen RUNNING-Set und der ersten
        # Slot-Reservation in ensure_running ein kurzes Fenster mit
        # state=RUNNING + active=0 bestehen, in dem der Lifecycle-Lease
        # False meldet. Den Marker mit Kurz-TTL refreshen ueberbrueckt
        # dieses Fenster ueber das Overlay; Auto-Expiry deckt Edge-
        # Cases (Cold-Start-Caller-Cancel, no further callers) ab.
        if marker_token is not None:
            if success:
                try:
                    _write_vram_marker(
                        "startup",
                        ttl_s=GPU_SERVICE_MARKER_SHORT_TTL_S,
                        token=marker_token,
                    )
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"Startup-Marker-Refresh fehlgeschlagen: {e}",
                    )
                    _clear_vram_marker("startup", marker_token)
            elif clear_startup_marker:
                _clear_vram_marker("startup", marker_token)
            elif vram_gate_passed and not target_dirty:
                _clear_vram_marker("startup", marker_token)
    return success


async def _finalize_stop(stop_ok: bool) -> None:
    """Zentrale Stop-Finalisierung mit Dirty-Inspect und Retry-Scheduler.

    Aufrufer: `ensure_running`-Drain, `_release_slot`-Drain,
    `on_startup`-Takeover, `_dirty_retry_loop`. **Nicht** vom
    `_start_supervisor`-try-Block — dort erfolgt der State-Set im finally
    via `target_dirty`-Flag (Race-Prevention).

    Precondition: `_active_requests == 0`. Der Helper schreibt
    `_active_requests` nicht. Caller hat `_state_lock` NICHT.

    Bei stop_ok=False: konservativer Dirty-Inspect. Wenn der Inspect-Helfer
    True liefert, Uebergang nach STOPPED_DIRTY + Retry-Scheduler starten.
    SHUTDOWN ist terminal und wird nie ueberschrieben.
    """
    global _last_request_time, _backend_failed
    still_running = False
    if not stop_ok:
        try:
            still_running = await _to_thread(_is_running_for_dirty)
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Dirty-Inspect Exception: {e}")
            still_running = True
    async with _state_lock:
        if _state == State.SHUTDOWN:
            _state_changed.notify_all()
            return
        if still_running:
            _set_state(State.STOPPED_DIRTY)
            _last_request_time = 0.0
            _backend_failed = False
            _ensure_retry_scheduler()
        else:
            _set_state(State.STOPPED)
            _last_request_time = 0.0
            _backend_failed = False
        _state_changed.notify_all()


# --- ensure_running ---

async def ensure_running() -> SlotToken | None:
    """Stellt sicher, dass der Container Requests annehmen kann.

    Rueckgabe:
    - int: Slot reserviert (`_active_requests += 1`, `_last_request_time`
      aktualisiert). Caller MUSS im proxy_handler genau einen Slot-Release
      ausloesen.
    - None: kein Slot reserviert; Caller liefert 503.

    Keine Exception-Propagation an proxy_handler (boolsche Semantik).
    BaseException (CancelledError etc.) wird nicht geschluckt.
    """
    global _state, _active_requests, _last_request_time, _backend_failed
    while True:
        drain_outside = False
        cold_start_outside = False
        will_own_regate = False
        needs_regate_wait = False
        slot_token: SlotToken | None = None
        drain_marker_token: str | None = None
        async with _state_lock:
            if _state == State.RUNNING and not _backend_failed:
                slot_token = _slot_generation
                if _active_requests == 0:
                    will_own_regate = True
                    _warm_regate_done.clear()
                    _active_requests = 1
                    _last_request_time = time.monotonic()
                elif not _warm_regate_done.is_set():
                    needs_regate_wait = True
                else:
                    _active_requests += 1
                    _last_request_time = time.monotonic()
                    return slot_token
            elif _state == State.RUNNING and _backend_failed:
                if _active_requests > 0:
                    await _state_changed.wait()
                    continue
                # letzter Slot ist Drain-Owner (F136)
                try:
                    drain_marker_token = _write_vram_marker("shutdown", ttl_s=MARKER_TTL_S)
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"Drain-Stop Shutdown-Marker fehlgeschlagen: {e}",
                    )
                    # #801: bei persistentem Marker-Failure (z.B. /run/kiron
                    # read-only) wuerde reines return None den State auf
                    # RUNNING+_backend_failed=True+_active_requests=0 stehen
                    # lassen, sodass jede folgende ensure_running denselben
                    # Pfad nimmt und 503-t. Stattdessen nach STOPPED_DIRTY
                    # transitionieren — Retry-Scheduler uebernimmt
                    # Container-Cleanup, DIRTY-lease ersetzt den
                    # fehlgeschlagenen Marker-Overlay.
                    _invalidate_generation_locked()
                    _set_state(State.STOPPED_DIRTY)
                    _ensure_retry_scheduler()
                    _state_changed.notify_all()
                    return None
                _invalidate_generation_locked()
                _state = State.STOPPING
                _state_changed.notify_all()
                drain_outside = True
            elif _state in (State.STARTING, State.STOPPING):
                await _state_changed.wait()
                continue
            elif _state == State.SHUTDOWN:
                return None
            elif _state == State.STOPPED_DIRTY:
                # Keine Slot-Vergabe und kein Cold-Start waehrend DIRTY;
                # Retry-Scheduler uebernimmt Recovery.
                return None
            elif _state == State.STOPPED:
                if _in_cooldown_locked():
                    return None
                _state = State.STARTING
                _state_changed.notify_all()
                cold_start_outside = True
            else:
                return None

        if will_own_regate:
            regate_token = slot_token
            marker_token: str | None = None
            existing_marker, existing_active = _marker_payload(STARTUP_MARKER_PATH)
            existing_token = (
                existing_marker.get("token")
                if isinstance(existing_marker, dict)
                else None
            )
            if (
                existing_active
                and isinstance(existing_marker, dict)
                and existing_marker.get("pid") == os.getpid()
                and isinstance(existing_token, str)
                and existing_token
            ):
                marker_token = existing_token
                try:
                    _write_vram_marker(
                        "startup",
                        ttl_s=GPU_SERVICE_MARKER_SHORT_TTL_S,
                        token=marker_token,
                    )
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        "Warm-Regate Startup-Marker-Refresh "
                        f"fehlgeschlagen: {e}",
                    )
                    async with _state_lock:
                        if regate_token == _slot_generation:
                            _active_requests = max(0, _active_requests - 1)
                        _warm_regate_done.set()
                        _state_changed.notify_all()
                    return None
            else:
                try:
                    marker_token = _write_vram_marker(
                        "startup",
                        ttl_s=GPU_SERVICE_MARKER_SHORT_TTL_S,
                    )
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"Warm-Regate Startup-Marker fehlgeschlagen: {e}",
                    )
                    async with _state_lock:
                        if regate_token == _slot_generation:
                            _active_requests = max(0, _active_requests - 1)
                        _warm_regate_done.set()
                        _state_changed.notify_all()
                    return None
            regate_task = asyncio.create_task(
                _call_prepare_vram_for_docling(regate_token)
            )

            async def _finish_regate(release_slot: bool) -> bool:
                global _active_requests
                async with _state_lock:
                    current = regate_token == _slot_generation
                    keep_slot = (
                        current
                        and not release_slot
                        and _state == State.RUNNING
                    )
                    release_slot = current and (release_slot or not keep_slot)
                    if release_slot:
                        _active_requests = max(0, _active_requests - 1)
                    _warm_regate_done.set()
                    _state_changed.notify_all()
                    return keep_slot

            try:
                await asyncio.shield(regate_task)
                if marker_token is not None:
                    try:
                        _write_vram_marker(
                            "startup",
                            ttl_s=GPU_SERVICE_MARKER_SHORT_TTL_S,
                            token=marker_token,
                        )
                    except Exception as e:
                        _journal_log(
                            JOURNAL_WARNING,
                            "Warm-Regate Startup-Marker-Refresh "
                            f"fehlgeschlagen: {e}",
                        )
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(regate_task)
                except VramGateError as e:
                    _journal_log(JOURNAL_WARNING, f"Warm-Regate VramGateError: {e}")
                except Exception as e:
                    _journal_log(JOURNAL_WARNING, f"Warm-Regate unerwartet: {e}")
                await _finish_regate(release_slot=True)
                if marker_token is not None:
                    _clear_vram_marker("startup", marker_token)
                raise
            except VramGateError as e:
                _journal_log(JOURNAL_WARNING, f"Warm-Regate VramGateError: {e}")
                await _finish_regate(release_slot=True)
                if marker_token is not None and not e.side_effects_started:
                    _clear_vram_marker("startup", marker_token)
                return None
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"Warm-Regate unerwartet: {e}")
                await _finish_regate(release_slot=True)
                if marker_token is not None:
                    _clear_vram_marker("startup", marker_token)
                return None
            keep = await _finish_regate(release_slot=False)
            if not keep and marker_token is not None:
                _clear_vram_marker("startup", marker_token)
            return regate_token if keep else None

        if needs_regate_wait:
            try:
                await asyncio.wait_for(
                    _warm_regate_done.wait(),
                    timeout=WARM_REGATE_WAIT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                _journal_log(
                    JOURNAL_WARNING,
                    "Warm-Regate-Wait Timeout "
                    f"({WARM_REGATE_WAIT_TIMEOUT_S:.0f}s) — retry",
                )
            continue

        if drain_outside:
            async def _drain_cleanup() -> None:
                stop_clean = False
                try:
                    stop_ok = await _run_stop_once()
                except Exception as e:
                    _journal_log(JOURNAL_WARNING, f"Drain-Stop Exception: {e}")
                    stop_ok = False
                await _finalize_stop(stop_ok)
                async with _state_lock:
                    stop_clean = _state == State.STOPPED
                if drain_marker_token is not None and stop_clean:
                    _clear_vram_marker("shutdown", drain_marker_token)

            # Shield gegen Caller-Cancellation: verhindert STOPPING-Hang,
            # wenn der Request waehrend des Drain-Stops abgebrochen wird
            # (F141).
            await asyncio.shield(_drain_cleanup())
            continue

        if cold_start_outside:
            try:
                started_ok = await _run_start_once()
            except BaseException:
                raise
            if not started_ok:
                return None
            # erfolgreich gestartet -> Loop zurueck, Warm-Reuse-Pfad holt Slot
            continue


# --- idle watcher ---

async def _try_stop_if_idle() -> None:
    """Pruefung + Transition unter Lock. Stop ausserhalb des Locks."""
    global _state, _last_request_time
    marker_token: str | None = None
    async with _state_lock:
        if _state != State.RUNNING:
            return
        if _active_requests > 0:
            return
        if _last_request_time <= 0.0:
            return
        if time.monotonic() - _last_request_time <= IDLE_TIMEOUT_S:
            return
        if await _has_nonterminal_tasks():
            return
        idle = time.monotonic() - _last_request_time
        print(f"Idle {idle:.0f}s > {IDLE_TIMEOUT_S}s — stoppe {CONTAINER_NAME}")
        # #771: Shutdown-Overlay vor STOPPING setzen, damit ein
        # kiron-proxy mit gecachtem lease=False nicht durch das 2s-
        # TTL-Fenster GPU-Requests passieren laesst, waehrend docker
        # stop noch VRAM freigibt.
        try:
            marker_token = _write_vram_marker("shutdown", ttl_s=MARKER_TTL_S)
        except Exception as e:
            # #802: idle-Stop darf nicht an Marker-Failure haengen, sonst
            # bleibt der Container bei persistentem /run/kiron-Failure
            # (read-only/voll) unendlich im RUNNING-State und haelt VRAM,
            # weil idle_watcher alle 15s denselben Versuch wiederholt.
            # Trade-off: bis zu 2s GPU-Request-Fenster ohne Overlay (#771).
            _journal_log(
                JOURNAL_WARNING,
                "idle-Stop Shutdown-Marker fehlgeschlagen, "
                f"fahre trotzdem fort: {e}",
            )
        _state = State.STOPPING
        _state_changed.notify_all()

    stop_clean = False
    try:
        try:
            stop_ok = await _run_stop_once()
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"idle-Stop Exception: {e}")
            stop_ok = False

        # Finalisierung via _finalize_stop: bei stop_ok=False + still_running
        # nach STOPPED_DIRTY (lease=True) und Dirty-Retry-Scheduler statt
        # RUNNING+active=0 (lease=False) zu veroeffentlichen, sonst kann ein
        # weiter VRAM haltender Container faelschlich als frei gemeldet werden.
        await _finalize_stop(stop_ok)
        async with _state_lock:
            stop_clean = _state == State.STOPPED
    finally:
        if marker_token is not None and stop_clean:
            _clear_vram_marker("shutdown", marker_token)


async def idle_watcher() -> None:
    """Background-Task: Stoppt Container nach Inaktivitaet."""
    while True:
        try:
            await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
            await _try_stop_if_idle()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _journal_log(
                JOURNAL_WARNING,
                f"idle_watcher-Iteration fehlgeschlagen: {e}",
            )


# --- Proxy-Handler ---

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade",
              "host", "content-length", "te", "trailer", "trailers",
              "proxy-authenticate", "proxy-authorization"}

http_client = httpx.AsyncClient(
    base_url=BACKEND_URL,
    timeout=httpx.Timeout(connect=10, read=3600, write=60, pool=10),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
)


async def _release_slot(
    token: SlotToken | None = None, *, ok: bool, backend_failed: bool,
) -> None:
    """Einheitlicher Slot-Release mit SHUTDOWN-Gate und Backend-Drain.

    `ok=True`: erfolgreicher Request/Stream -> `_last_request_time`
    aktualisieren (nur ausserhalb SHUTDOWN und ohne globalen Backend-Fehler).
    `backend_failed=True`: lokalen Backend-Fehler als global markieren;
    wenn danach keine aktiven Slots mehr laufen, Drain-Cleanup uebernehmen.
    """
    global _state, _active_requests, _last_request_time, _backend_failed

    do_drain = False
    marker_token: str | None = None
    async with _state_lock:
        if token is None:
            token = _slot_generation
        if token != _slot_generation:
            print(f"  INFO: stale Slot-Release ignoriert token={token} current={_slot_generation}")
            return
        _active_requests = max(0, _active_requests - 1)
        if backend_failed:
            _backend_failed = True
        if ok and not _backend_failed and _state != State.SHUTDOWN:
            _last_request_time = time.monotonic()
        # SHUTDOWN-Gate: erfolgreicher Release bei SHUTDOWN beruehrt State nicht
        if _state == State.SHUTDOWN:
            if backend_failed and _active_requests == 0:
                # auch im SHUTDOWN _backend_failed zuruecksetzen (F90)
                _backend_failed = False
            _state_changed.notify_all()
            return
        # Backend-Failure-Drain: letzter Slot startet Drain-Cleanup (F136)
        if _backend_failed and _active_requests == 0:
            # #771: Shutdown-Overlay vor STOPPING setzen, sonst sieht ein
            # kiron-proxy mit gecachtem lease=False bis zu 2s lang einen
            # falschen lease=False, waehrend docker stop noch VRAM freigibt.
            try:
                marker_token = _write_vram_marker("shutdown", ttl_s=MARKER_TTL_S)
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"Drain-Stop Shutdown-Marker fehlgeschlagen: {e}",
                )
                # #803: analog ensure_running #801 — reines return wuerde
                # RUNNING+_backend_failed=True+_active_requests=0 stehen
                # lassen, sodass idle_watcher und neue Requests denselben
                # Pfad nehmen und endlos 503-en. Nach STOPPED_DIRTY
                # transitionieren, damit der Retry-Scheduler Container-
                # Cleanup uebernimmt und DIRTY-lease den Overlay ersetzt.
                _invalidate_generation_locked()
                _set_state(State.STOPPED_DIRTY)
                _ensure_retry_scheduler()
                _state_changed.notify_all()
                return
            _invalidate_generation_locked()
            _state = State.STOPPING
            do_drain = True
        _state_changed.notify_all()

    if do_drain:
        async def _drain_cleanup() -> None:
            stop_clean = False
            try:
                try:
                    stop_ok = await _run_stop_once()
                except Exception as e:
                    _journal_log(JOURNAL_WARNING, f"Drain-Stop Exception: {e}")
                    stop_ok = False
                await _finalize_stop(stop_ok)
                async with _state_lock:
                    stop_clean = _state == State.STOPPED
            finally:
                if marker_token is not None and stop_clean:
                    _clear_vram_marker("shutdown", marker_token)

        # Shield gegen Caller-Cancellation: sonst bleibt _state in STOPPING
        # festgeschrieben, wenn der Caller waehrend _run_stop_once abgebrochen
        # wird (F141).
        await asyncio.shield(_drain_cleanup())


def _connection_specific_names(items) -> set[str]:
    """Header-Namen aus Connection-Header-Werten (RFC 7230 §6.1)."""
    extra: set[str] = set()
    for k, v in items:
        if k.lower() == "connection":
            for name in v.split(","):
                name = name.strip().lower()
                if name:
                    extra.add(name)
    return extra


def _filter_request_headers(request: Request) -> list[tuple[str, str]]:
    raw = getattr(request.headers, "raw", None)
    if raw:
        decoded = [(key.decode("latin-1"), value.decode("latin-1"))
                   for key, value in raw]
    else:
        decoded = list(request.headers.items())
    skip = HOP_BY_HOP | _connection_specific_names(decoded)
    return [(name, value) for name, value in decoded
            if name.lower() not in skip]


def _filter_response_headers(resp: httpx.Response) -> MutableHeaders:
    if hasattr(resp.headers, "multi_items"):
        items = list(resp.headers.multi_items())
    else:
        items = list(resp.headers.items())
    skip = HOP_BY_HOP | _connection_specific_names(items)
    raw = [
        (str(k).lower().encode("latin-1"), str(v).encode("latin-1"))
        for k, v in items
        if k.lower() not in skip
    ]
    return MutableHeaders(raw=raw)


async def _response_iter(resp: httpx.Response, token: SlotToken | None = None):
    """Streamt Response-Body; haelt den Slot bis zum finalen Cleanup.

    Fehlerpfade (F5/F30/F73/F75/F82):
    - RemoteProtocolError/ReadError -> backend_failed=True, re-raise
    - CancelledError/GeneratorExit -> keine Invalidation, re-raise
    - sonstige Exception -> re-raise, keine Invalidation
    - vollstaendiger Durchlauf -> ok=True
    """
    ok = True
    backend_failed = False
    if token is None:
        token = _slot_generation
    def _consume_release_exception(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Slot-Release Exception: {e}")
        except BaseException:
            pass

    async def _shielded_release() -> None:
        release_task = asyncio.create_task(
            _release_slot(token, ok=ok, backend_failed=backend_failed)
        )
        try:
            await asyncio.shield(release_task)
        except (asyncio.CancelledError, GeneratorExit):
            release_task.add_done_callback(_consume_release_exception)
            raise

    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    except (httpx.RemoteProtocolError, httpx.ReadError,
            httpx.TimeoutException) as e:
        ok = False
        backend_failed = True
        _journal_log(JOURNAL_WARNING, f"Stream-Backendfehler: {e}")
        raise
    except (asyncio.CancelledError, GeneratorExit):
        ok = False
        raise
    except Exception as e:
        ok = False
        _journal_log(JOURNAL_WARNING, f"Stream-Exception: {e}")
        raise
    finally:
        # Aeusseres try/finally garantiert Slot-Release auch bei BaseException
        # (SystemExit/KeyboardInterrupt) aus aclose() — sonst bleibt
        # _active_requests erhoeht haengen (F81/F123, F686).
        try:
            try:
                await resp.aclose()
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"resp.aclose() fehlgeschlagen: {e}",
                )
        finally:
            try:
                await _shielded_release()
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"Slot-Release Exception: {e}")


def _json_bytes(obj: dict) -> bytes:
    import json as _json
    return _json.dumps(obj).encode("utf-8")


async def proxy_handler(request: Request) -> Response:
    token = await ensure_running()
    if token is None:
        return Response(
            content=_json_bytes({"error": "docling-serve konnte nicht "
                                          "gestartet werden"}),
            status_code=503,
            media_type="application/json",
        )

    # Ab hier ist genau ein Slot reserviert. Jeder Pfad MUSS ihn genau einmal
    # freigeben (entweder im Handler oder im response_iter-finally).
    released = False

    try:
        # Pfad 0: Pre-send-Setup
        try:
            # #505: raw_path fuer byte-identischen Forward — ASGI scope["path"] und
            # request.url.path sind percent-decoded (%20 -> ' ', %2F -> '/'), was fuer
            # einen Reverse-Proxy falsch ist. raw_path enthaelt den Wire-Pfad als Bytes.
            raw_path = request.scope.get("raw_path")
            path = raw_path.decode("latin-1") if raw_path else request.url.path
            query = str(request.url.query) if request.url.query else ""
            target = f"{path}?{query}" if query else path
            headers = _filter_request_headers(request)
            req = http_client.build_request(
                method=request.method,
                url=target,
                headers=headers,
                content=request.stream(),
            )
        except ClientDisconnect:
            # Bau wird bei ClientDisconnect normalerweise nicht werfen; falls
            # doch, wie Pfad A ClientDisconnect behandeln.
            await _release_slot(token, ok=False, backend_failed=False)
            released = True
            return Response(status_code=499)
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Pre-send-Setup Fehler: {e}")
            await _release_slot(token, ok=False, backend_failed=False)
            released = True
            return Response(
                content=_json_bytes({"error": "Setup-Fehler"}),
                status_code=500,
                media_type="application/json",
            )

        # Pfad A: send()
        try:
            resp = await http_client.send(req, stream=True)
        except httpx.ConnectError as e:
            _journal_log(JOURNAL_WARNING, f"Backend ConnectError: {e}")
            await _release_slot(token, ok=False, backend_failed=True)
            released = True
            return Response(
                content=_json_bytes({"error": "docling-serve nicht erreichbar"}),
                status_code=502,
                media_type="application/json",
            )
        except httpx.PoolTimeout:
            # #638: PoolTimeout ist Subklasse von TimeoutException; muss VOR
            # dem TimeoutException-Branch gefangen werden. Lokale Pool-
            # Erschoepfung (max_connections), Backend ist gesund -> 503 mit
            # backend_failed=False, damit kein Drain-Stop ausgeloest wird.
            await _release_slot(token, ok=False, backend_failed=False)
            released = True
            return Response(
                content=_json_bytes({"error": "Proxy-Pool ueberlastet"}),
                status_code=503,
                media_type="application/json",
            )
        except httpx.TimeoutException:
            # Backend kann nach Client-Timeout weiterarbeiten -> als
            # backend_failed markieren, damit Drain-Cleanup den Container
            # stoppt und _last_request_time nicht stale bleibt (#566).
            await _release_slot(token, ok=False, backend_failed=True)
            released = True
            return Response(
                content=_json_bytes({"error": "docling-serve Timeout"}),
                status_code=504,
                media_type="application/json",
            )
        except (httpx.RemoteProtocolError, httpx.ReadError,
                httpx.WriteError) as e:
            _journal_log(JOURNAL_WARNING, f"Backend Protokollfehler: {e}")
            await _release_slot(token, ok=False, backend_failed=True)
            released = True
            return Response(
                content=_json_bytes({"error": "docling-serve Protokollfehler"}),
                status_code=502,
                media_type="application/json",
            )
        except ClientDisconnect:
            await _release_slot(token, ok=False, backend_failed=False)
            released = True
            return Response(status_code=499)
        except httpx.TransportError as e:
            _journal_log(JOURNAL_WARNING, f"Backend Transportfehler: {e}")
            await _release_slot(token, ok=False, backend_failed=True)
            released = True
            return Response(
                content=_json_bytes({"error": "docling-serve Transportfehler"}),
                status_code=502,
                media_type="application/json",
            )

        # Pfad B/C/HEAD: ab hier haben wir eine Response
        try:
            resp_headers = _filter_response_headers(resp)
            status = resp.status_code
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Response-Setup Fehler: {e}")
            try:
                await resp.aclose()
            except Exception as ce:
                _journal_log(
                    JOURNAL_WARNING,
                    f"resp.aclose() nach Setup-Fehler: {ce}",
                )
            except (asyncio.CancelledError, GeneratorExit):
                pass
            await _release_slot(token, ok=False, backend_failed=False)
            released = True
            return Response(
                content=_json_bytes({"error": "Response-Setup-Fehler"}),
                status_code=500,
                media_type="application/json",
            )

        # HEAD: bodylos, Slot synchron freigeben (F110)
        if request.method == "HEAD":
            # #504: HOP_BY_HOP strippt content-length global (noetig fuer Streaming-GET,
            # wo der Body unabhaengig vom Backend-Count rewritten wird). Fuer HEAD verlangt
            # RFC 7231 §4.3.2 aber die GET-aequivalenten Repraesentations-Metadaten inkl.
            # Content-Length — hier explizit wieder einfuegen, falls das Backend es gesetzt hat.
            backend_cl = resp.headers.get("content-length")
            try:
                await resp.aclose()
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"HEAD resp.aclose() Fehler: {e}")
            except (asyncio.CancelledError, GeneratorExit):
                pass
            await _release_slot(token, ok=True, backend_failed=False)
            released = True
            if backend_cl is not None:
                resp_headers["content-length"] = backend_cl
            return Response(status_code=status, headers=resp_headers)

        is_async_trigger = (
            request.method == "POST" and _match_async_trigger_path(path)
        )
        is_status_poll = (
            request.method == "GET"
            and _STATUS_POLL_PATTERN.match(path) is not None
        )
        if is_async_trigger or is_status_poll:
            try:
                body_bytes = await resp.aread()
            except httpx.TimeoutException:
                try:
                    await resp.aclose()
                except Exception as e:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"resp.aclose() nach Timeout: {e}",
                    )
                except (asyncio.CancelledError, GeneratorExit):
                    pass
                await _release_slot(token, ok=False, backend_failed=True)
                released = True
                return Response(
                    content=_json_bytes({"error": "docling-serve Timeout"}),
                    status_code=504,
                    media_type="application/json",
                )
            except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                _journal_log(JOURNAL_WARNING, f"Async-Task-Backendfehler: {e}")
                try:
                    await resp.aclose()
                except Exception as ce:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"resp.aclose() nach Async-Fehler: {ce}",
                    )
                except (asyncio.CancelledError, GeneratorExit):
                    pass
                await _release_slot(token, ok=False, backend_failed=True)
                released = True
                return Response(
                    content=_json_bytes({"error": "docling-serve Protokollfehler"}),
                    status_code=502,
                    media_type="application/json",
                )
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"Async-Task-Response-Read Fehler: {e}",
                )
                try:
                    await resp.aclose()
                except Exception as ce:
                    _journal_log(
                        JOURNAL_WARNING,
                        f"resp.aclose() nach Async-Read-Fehler: {ce}",
                    )
                except (asyncio.CancelledError, GeneratorExit):
                    pass
                await _release_slot(token, ok=False, backend_failed=False)
                released = True
                return Response(
                    content=_json_bytes({"error": "Response-Read-Fehler"}),
                    status_code=500,
                    media_type="application/json",
                )
            try:
                await resp.aclose()
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"Async-Task resp.aclose() Fehler: {e}",
                )
            except (asyncio.CancelledError, GeneratorExit):
                pass

            if 200 <= status < 300:
                if is_async_trigger:
                    await _register_task_from_response(path, body_bytes, token)
                else:
                    await _refresh_task_from_poll(path, body_bytes, token)
            await _release_slot(token, ok=True, backend_failed=False)
            released = True
            # httpx aread() dekodiert Content-Encoding (gzip/deflate/br)
            # automatisch; der Body ist hier bereits dekodiert. Den Header
            # mitzuleiten wuerde dem Client signalisieren, das Encoding noch
            # einmal entpacken zu muessen.
            if "content-encoding" in resp_headers:
                del resp_headers["content-encoding"]
            return Response(
                content=body_bytes,
                status_code=status,
                headers=resp_headers,
            )

        # Streaming: Slot-Ownership wandert zum Iterator
        if (
            request.method == "GET"
            and 200 <= status < 300
            and _RESULT_PATH_PATTERN.match(path) is not None
        ):
            await _mark_result_task_success(path, token)
        response = StreamingResponse(
            _response_iter(resp, token),
            status_code=status,
            headers=resp_headers,
        )
        released = True  # Slot gehoert ab jetzt dem Iterator
        return response
    except BaseException:
        # Falls im Handler selbst eine BaseException entweicht und wir den
        # Slot noch halten, hier freigeben (Defense-in-Depth).
        if not released:
            try:
                await asyncio.shield(_release_slot(token, ok=False,
                                                   backend_failed=False))
            except Exception as e:
                _journal_log(
                    JOURNAL_WARNING,
                    f"Emergency-Slot-Release Exception: {e}",
                )
            except BaseException:
                pass
        raise


async def catch_all(request: Request) -> Response:
    return await proxy_handler(request)


async def lifecycle_handler(request: Request) -> Response:
    """Interner Lifecycle-Snapshot fuer Dashboard-Drain-Koordination (#278).

    Atomarer Snapshot unter `_state_lock`, keine Side-Effects auf State,
    Counter oder Tasks. `Cache-Control: no-store` gegen versehentliches
    Caching durch Proxies/HTTP-Clients.

    **Security-Annahme**: LAN-only Deployment (siehe Issue #260 wontfix).
    Der Proxy bindet 0.0.0.0:5001, dieser Endpoint ist nicht auth-geschuetzt.
    Bei Public-Exposure muss der Endpoint Auth bekommen.
    """
    async with _state_lock:
        state_value = _state.value
        active = _active_requests
        last_ts = _last_request_time
        backend_failed = _backend_failed
        lease = await _vram_lease_active()
    age = (time.monotonic() - last_ts) if last_ts > 0 else None
    return JSONResponse(
        {
            "state": state_value,
            "active_requests": active,
            "last_request_age_s": age,
            "backend_failed": backend_failed,
            "vram_lease_active": lease,
        },
        headers={"Cache-Control": "no-store"},
    )


async def start_handler(request: Request) -> Response:
    """Interner Coldstart-Trigger fuer Dashboard-`/api/docling/start` (#560).

    Faehrt den Container in den State RUNNING und uebernimmt dabei VRAM-
    Marker und Lifecycle-Lease durch `_start_supervisor`. Reserviert KEINEN
    Request-Slot — Caller fuehrt keine Konvertierung aus.

    Idempotent:
    - RUNNING (sauber) -> 200, `started=false`.
    - RUNNING + `_backend_failed` -> warten bis Drain (STOPPING) angestossen,
      dann erneut bewerten. Konsistent mit `_acquire_slot` (1030-1033).
    - STARTING/STOPPING -> warten bis Transition fertig, dann erneut bewerten.
    - STOPPED -> Transition nach STARTING, `_run_start_once()`, finale State.
    - SHUTDOWN -> 503.
    - STOPPED_DIRTY -> 409 (Retry-Scheduler des Proxys handhabt Recovery).

    **Security-Annahme**: LAN-only (gleiches Modell wie /_internal/lifecycle).
    """
    global _state
    while True:
        async with _state_lock:
            if _state == State.RUNNING:
                if _backend_failed:
                    await _state_changed.wait()
                    continue
                return JSONResponse(
                    {"started": False, "state": _state.value},
                    status_code=200,
                    headers={"Cache-Control": "no-store"},
                )
            if _state in (State.STARTING, State.STOPPING):
                await _state_changed.wait()
                continue
            if _state == State.SHUTDOWN:
                return JSONResponse(
                    {"started": False, "state": _state.value,
                     "error": "shutdown"},
                    status_code=503,
                    headers={"Cache-Control": "no-store"},
                )
            if _state == State.STOPPED_DIRTY:
                return JSONResponse(
                    {"started": False, "state": _state.value,
                     "error": "stopped_dirty"},
                    status_code=409,
                    headers={"Cache-Control": "no-store"},
                )
            if _state == State.STOPPED:
                if _in_cooldown_locked():
                    return JSONResponse(
                        {"started": False, "state": _state.value,
                         "error": "start_cooldown"},
                        status_code=503,
                        headers={"Cache-Control": "no-store"},
                    )
                _set_state(State.STARTING)
                _state_changed.notify_all()
                break
            return JSONResponse(
                {"started": False, "state": _state.value,
                 "error": f"unexpected state: {_state.value}"},
                status_code=500,
                headers={"Cache-Control": "no-store"},
            )

    ok = await _run_start_once()
    async with _state_lock:
        final_state = _state.value
    if not ok:
        return JSONResponse(
            {"started": False, "state": final_state,
             "error": "start_failed"},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {"started": True, "state": final_state},
        status_code=200,
        headers={"Cache-Control": "no-store"},
    )


async def stop_handler(request: Request) -> Response:
    """Interner Stop-Trigger fuer Dashboard-`/api/docling/stop` (#617).

    Der Stop laeuft durch die lokale State-Machine statt direkt per Docker aus
    dem Dashboard. Damit koennen Lifecycle, VRAM-Lease und spaetere Starts den
    tatsaechlichen Zustand sehen.
    """
    global _state, _active_requests, _last_request_time, _backend_failed
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    force = payload.get("force") is True

    marker_token: str | None = None
    while True:
        async with _state_lock:
            if _state == State.SHUTDOWN:
                return JSONResponse(
                    {"stopped": False, "state": _state.value,
                     "error": "shutdown"},
                    status_code=503,
                    headers={"Cache-Control": "no-store"},
                )
            if _state == State.STOPPED:
                return JSONResponse(
                    {"stopped": False, "state": _state.value},
                    status_code=200,
                    headers={"Cache-Control": "no-store"},
                )
            if _state in (State.STARTING, State.STOPPING):
                await _state_changed.wait()
                continue
            if _state == State.RUNNING:
                if _active_requests > 0 and not force:
                    return JSONResponse(
                        {"stopped": False, "state": _state.value,
                         "active_requests": _active_requests,
                         "error": "active_requests"},
                        status_code=409,
                        headers={"Cache-Control": "no-store"},
                    )
                # #771: Shutdown-Overlay vor STOPPING setzen — sonst kann ein
                # kiron-proxy mit gecachtem lease=False bis zu 2s lang
                # GPU-Requests passieren lassen, waehrend docker stop noch
                # VRAM freigibt.
                try:
                    marker_token = _write_vram_marker("shutdown",
                                                      ttl_s=MARKER_TTL_S)
                except Exception as e:
                    # #804: User-expliziter Stop darf nicht an Marker-Failure
                    # haengen, sonst kann der Container nicht ueber Dashboard
                    # gestoppt werden, solange /run/kiron unbeschreibbar ist.
                    # Trade-off: bis zu 2s GPU-Request-Fenster ohne Overlay (#771).
                    _journal_log(
                        JOURNAL_WARNING,
                        "Control-Stop Shutdown-Marker fehlgeschlagen, "
                        f"fahre trotzdem fort: {e}",
                    )
                _invalidate_generation_locked()
                _set_state(State.STOPPING)
                _state_changed.notify_all()
                break
            if _state == State.STOPPED_DIRTY:
                # #771: Shutdown-Overlay vor STOPPING setzen — siehe oben.
                try:
                    marker_token = _write_vram_marker("shutdown",
                                                      ttl_s=MARKER_TTL_S)
                except Exception as e:
                    # #804: User-expliziter Stop darf nicht an Marker-Failure
                    # haengen, sonst kann der Container nicht ueber Dashboard
                    # gestoppt werden, solange /run/kiron unbeschreibbar ist.
                    # Trade-off: bis zu 2s GPU-Request-Fenster ohne Overlay (#771).
                    _journal_log(
                        JOURNAL_WARNING,
                        "Control-Stop Shutdown-Marker fehlgeschlagen, "
                        f"fahre trotzdem fort: {e}",
                    )
                _invalidate_generation_locked()
                _set_state(State.STOPPING)
                _state_changed.notify_all()
                break
            return JSONResponse(
                {"stopped": False, "state": _state.value,
                 "error": f"unexpected state: {_state.value}"},
                status_code=500,
                headers={"Cache-Control": "no-store"},
            )

    final_state: str | None = None
    try:
        try:
            stop_ok = await _run_stop_once()
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Control-Stop Exception: {e}")
            stop_ok = False
        await _finalize_stop(stop_ok)

        async with _state_lock:
            final_state = _state.value
    finally:
        if marker_token is not None and final_state == State.STOPPED.value:
            _clear_vram_marker("shutdown", marker_token)
    status = 200 if final_state == State.STOPPED.value else 503
    body = {"stopped": final_state == State.STOPPED.value,
            "state": final_state}
    if status != 200:
        body["error"] = "stop_failed"
    return JSONResponse(
        body,
        status_code=status,
        headers={"Cache-Control": "no-store"},
    )


app = Starlette(routes=[
    Route("/_internal/lifecycle", endpoint=lifecycle_handler,
          methods=["GET"]),
    Route("/_internal/start", endpoint=start_handler,
          methods=["POST"]),
    Route("/_internal/stop", endpoint=stop_handler,
          methods=["POST"]),
    Route("/{path:path}", endpoint=catch_all,
          methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
    Route("/", endpoint=catch_all,
          methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
])


# --- Startup/Shutdown ---

async def on_startup() -> None:
    """Uebernimmt laufenden Container nur mit erfolgreichem Health-Check (F1).

    Defekten Altcontainer stoppen; Startup darf daran nicht scheitern.
    Idle-Watcher wird in jedem Fall gestartet. Takeover laeuft auch ohne
    erfolgreichen VRAM-Marker, damit ein nach Crash-Restart laufender
    Container nicht unmanaged bleibt (#805).
    """
    global _idle_watcher_task, _state, _last_request_time

    marker_token: str | None = None
    try:
        marker_token = _write_vram_marker("startup", ttl_s=MARKER_TTL_S)
    except Exception as e:
        _journal_log(
            JOURNAL_WARNING,
            f"Startup-Marker fehlgeschlagen; Takeover ohne VRAM-Gate: {e}",
        )
    try:
        try:
            running = await _to_thread(_is_running)
        except Exception as e:
            _journal_log(JOURNAL_WARNING, f"Startup-Inspect fehlgeschlagen: {e}")
            running = False

        if running:
            image_ok = await _to_thread(_has_expected_image)
            if not image_ok:
                _journal_log(
                    JOURNAL_WARNING,
                    "Startup-Takeover: unerwartetes Image; stoppe Container",
                )
                healthy = False
            else:
                policy_ok = await _to_thread(_remediate_restart_policy)
                if not policy_ok:
                    _journal_log(
                        JOURNAL_WARNING,
                        "Startup-Takeover: unsichere RestartPolicy; "
                        "stoppe Container",
                    )
                    healthy = False
                else:
                    port_ok = await _to_thread(_has_expected_port_binding)
                    if not port_ok:
                        _journal_log(
                            JOURNAL_WARNING,
                            "Startup-Takeover: unerwartetes PortBinding; "
                            "stoppe Container",
                        )
                        healthy = False
                    else:
                        try:
                            healthy = await _wait_healthy(timeout=HEALTH_CHECK_TIMEOUT_S)
                        except Exception as e:
                            _journal_log(
                                JOURNAL_WARNING,
                                f"Takeover-Health Exception: {e}",
                            )
                            healthy = False
            if healthy:
                async with _state_lock:
                    _state = State.RUNNING
                    _last_request_time = time.monotonic()
                    _state_changed.notify_all()
                print(f"Uebernehme laufenden Container {CONTAINER_NAME}")
            else:
                try:
                    stop_ok = await _run_stop_once()
                except Exception as e:
                    _journal_log(JOURNAL_WARNING, f"Takeover-Stop Exception: {e}")
                    stop_ok = False
                await _finalize_stop(stop_ok)
                _journal_log(
                    JOURNAL_WARNING,
                    f"Bestehender Container {CONTAINER_NAME} nicht gesund, "
                    "Recovery ueber _finalize_stop",
                )
    finally:
        _clear_vram_marker("startup", marker_token)

    if _idle_watcher_task is None or _idle_watcher_task.done():
        _idle_watcher_task = asyncio.create_task(idle_watcher())


app.add_event_handler("startup", on_startup)


async def on_shutdown() -> None:
    """Terminaler Shutdown: SHUTDOWN setzen, Tasks abwickeln, Container stoppen.

    Reihenfolge (F10):
    1. _state = SHUTDOWN, notify_all
    2. laufenden getrackten Stop-Task festhalten
    3. dirty_retry und idle_watcher abbrechen
    4. laufenden getrackten Stop-Task bounded abwarten
    5. laufenden getrackten Start-Task bounded abwarten
    6. http_client.aclose()
    """
    global _idle_watcher_task, _dirty_retry_task, _state

    try:
        marker_token = _write_vram_marker("shutdown", ttl_s=MARKER_TTL_S)
    except Exception as e:
        _journal_log(
            JOURNAL_ERR,
            f"VRAM-Shutdown-Marker konnte nicht gesetzt werden: {e}",
        )
        marker_token = None
    stop_clean = True
    try:
        async with _state_lock:
            _state = State.SHUTDOWN
            _state_changed.notify_all()

        stop_task = _stop_task

        if _dirty_retry_task is not None:
            _dirty_retry_task.cancel()
            try:
                await _dirty_retry_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"Shutdown: dirty_retry Fehler: {e}")
            _dirty_retry_task = None
            if stop_task is None:
                stop_task = _stop_task

        if _idle_watcher_task is not None:
            _idle_watcher_task.cancel()
            try:
                await _idle_watcher_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                _journal_log(JOURNAL_WARNING, f"Shutdown: idle_watcher Fehler: {e}")
            _idle_watcher_task = None

        # Getrackten Stop-Task abwarten; falls keiner laeuft und Container noch,
        # selbst stoppen
        if stop_task is None:
            stop_task = _stop_task
        if stop_task is not None and not stop_task.done():
            try:
                stop_ok = await asyncio.wait_for(
                    asyncio.shield(stop_task),
                    timeout=SHUTDOWN_STOP_GRACE_S,
                )
                stop_clean = stop_clean and bool(stop_ok)
            except asyncio.TimeoutError:
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    "Shutdown: Stop-Task Timeout — Shutdown-Marker bleibt bis TTL",
                )
            except Exception as e:
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    f"Shutdown: Stop-Task Fehler: {e} — "
                    "Shutdown-Marker bleibt bis TTL",
                )
        elif stop_task is not None:
            if stop_task.cancelled():
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    "Shutdown: Stop-Task cancelled — "
                    "Shutdown-Marker bleibt bis TTL",
                )
            else:
                try:
                    stop_clean = stop_clean and bool(stop_task.result())
                except Exception as e:
                    stop_clean = False
                    _journal_log(
                        JOURNAL_CRIT,
                        f"Shutdown: Stop-Task Fehler: {e} — "
                        "Shutdown-Marker bleibt bis TTL",
                    )

        # Getrackten Start-Task bounded abwarten (F92/F99)
        start_task = _start_task
        if start_task is not None and not start_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(start_task),
                                        timeout=SHUTDOWN_START_GRACE_S)
            except asyncio.TimeoutError:
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    "Shutdown: Start-Task Timeout — "
                    "Shutdown-Marker bleibt bis TTL",
                )
            except Exception as e:
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    f"Shutdown: Start-Task Fehler: {e} — "
                    "Shutdown-Marker bleibt bis TTL",
                )

        # Falls der Container durch den Start-Task nun laeuft: best-effort stoppen
        try:
            still = await _to_thread(_is_running_for_dirty)
        except Exception as e:
            still = True
            stop_clean = False
            _journal_log(
                JOURNAL_CRIT,
                f"Shutdown: Dirty-Inspect Fehler: {e} — "
                "Shutdown-Marker bleibt bis TTL",
            )
        if still:
            try:
                print(f"Shutdown: stoppe {CONTAINER_NAME}...")
                stop_ok = await _run_stop_once()
                if not stop_ok:
                    stop_clean = False
                    _journal_log(
                        JOURNAL_CRIT,
                        "Shutdown: docker stop nicht bestaetigt — "
                        "Shutdown-Marker bleibt bis TTL",
                    )
            except Exception as e:
                stop_clean = False
                _journal_log(
                    JOURNAL_CRIT,
                    f"Shutdown: docker stop fehlgeschlagen: {e}",
                )

        try:
            await http_client.aclose()
        except Exception as e:
            _journal_log(
                JOURNAL_WARNING,
                f"Shutdown: http_client.aclose() Fehler: {e}",
            )
    except BaseException:
        stop_clean = False
        raise
    finally:
        _warm_regate_done.set()
        if stop_clean and marker_token is not None:
            _clear_vram_marker("shutdown", marker_token)
        elif not stop_clean:
            _journal_log(
                JOURNAL_CRIT,
                "Shutdown: Container-Status unklar — Shutdown-Marker "
                f"bleibt bis TTL ({MARKER_TTL_S:.0f}s)",
            )


app.add_event_handler("shutdown", on_shutdown)


if __name__ == "__main__":
    print("=" * 50)
    print("Docling On-Demand Proxy")
    print("=" * 50)
    print(f"Proxy:        http://0.0.0.0:{LISTEN_PORT}")
    print(f"Backend:      {BACKEND_URL}")
    print(f"Container:    {CONTAINER_NAME}")
    print(f"Idle-Timeout: {IDLE_TIMEOUT_S}s ({IDLE_TIMEOUT_S // 60} min)")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info")
