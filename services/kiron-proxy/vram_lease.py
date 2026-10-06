"""VRAM-Lease Helper-Modul fuer kiron-proxy (#285).

Gemeinsames Modul fuer proxy.py, openai_api.py, app.py. Stellt:
- `LeaseOutcome`-Enum (PASS/FORCE_CPU/BLOCK).
- `snapshot()` — async TTL-Cache + Fetch + Fallback mit Thundering-
  Herd-Lock.
- `apply_bytes(body, path, model)` — bytes-Intercept fuer proxy.py.
- `apply_options_dict(options)` — Dict-Intercept fuer openai_api.py.
- `lifespan_client()` — async-Context-Manager fuer den shared httpx-
  Client, vom main.py Startup/Shutdown aufgerufen.
- `VRAM_LEASE_POLICY` — Policy-Env mit Validierung beim Modul-Import.
"""

import asyncio
from dataclasses import dataclass
import errno
import enum
import fcntl
import grp
import json
import logging
import os
import pwd
import stat
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx

from kiron_common.gpu_admission import AdmissionError, RuntimeSecurity, runtime_lock
from kiron_common.gpu_admission.native_contract import NativeMemoryBudget

from kiron_common.model_catalog import BackendType, ModelEndpoint, ModelTask
from kiron_common.ollama_compat import ensure_num_gpu_zero, is_real_int

from routing_catalog import PROXY_ROUTING_VIEW, ProxyRoute, ProxyRoutingView

__all__ = [
    "LeaseOutcome",
    "VRAM_LEASE_POLICY",
    "DOCLING_LIFECYCLE_URL",
    "STREAMING_INTERCEPT_PATHS",
    "snapshot",
    "effective_snapshot",
    "apply_bytes",
    "apply_options_dict",
    "num_gpu_zero_effective",
    "runtime_capability_status",
    "invalidate_runtime_capability_cache",
    "GPU_SERVICE_LOADING_TTL_S",
    "GPU_SERVICE_START_DEADLINE_S",
    "GPU_SERVICE_HEALTH_POLL_S",
    "GPU_SERVICE_LOADING_DRAIN_TIMEOUT_S",
    "GPU_SERVICE_LOADING_DRAIN_POLL_S",
    "GPU_SERVICE_MARKER_SHORT_TTL_S",
    "GPU_SERVICE_LOADING_MARKER_PATH",
    "MarkerOwnershipError",
    "GPUGateDecision",
    "GPUServiceOperation",
    "gpu_gate_decision",
    "overlay_marker_active",
    "write_overlay_marker",
    "refresh_overlay_marker",
    "clear_overlay_marker",
    "gpu_service_operation",
    "begin_gpu_service_operation",
    "finish_gpu_service_operation",
    "gpu_service_ops_lock",
    "lifespan_client",
]

logger = logging.getLogger(__name__)


# --- Konstanten / Konfiguration ---

DOCLING_LIFECYCLE_URL = "http://127.0.0.1:5001/_internal/lifecycle"

# Pfade auf denen Chat/Generate-Intercept greift. Wird von proxy.py als
# Referenz fuer die STREAMING_PATHS-Menge reused.
STREAMING_INTERCEPT_PATHS = frozenset({"/api/chat", "/api/generate"})

# V1: Der fruehere Env-Bypass darf `/api/embed` nicht mehr lockern.
_EMBED_BLOCK_FALLBACK = True
if os.environ.get("KIRON_VRAM_LEASE_EMBED_BLOCK") == "0":
    logger.warning(
        "KIRON_VRAM_LEASE_EMBED_BLOCK=0 wird in V1 ignoriert; "
        "/api/embed bleibt bei aktivem Gate konservativ geblockt."
    )

_LEASE_CACHE_TTL_S = 2.0
_LEASE_FETCH_TIMEOUT_S = 0.5
_LEASE_UNREACHABLE_LOG_INTERVAL_S = 60.0
_PASS_WARNING_LOG_INTERVAL_S = 60.0

RUNTIME_CAPABILITY_PATH = os.environ.get(
    "KIRON_OLLAMA_COMPAT_RUNTIME",
    "/usr/lib/kiron/data/ollama_compat_runtime.json",
)
_RUNTIME_CAPABILITY_CACHE_TTL_S = 1.0
_runtime_capability_cache: dict[str, Any] = {
    "ts": 0.0,
    "stat_key": None,
    "status": None,
}

RUNTIME_MARKER_GROUP = "kiron-runtime"
RUNTIME_MARKER_FILE_OWNER_NAMES = frozenset({"kiron-proxy", "kiron-docling"})
RUNTIME_MARKER_DIR_MODE = 0o2770
RUNTIME_MARKER_FILE_MODE = 0o660
RUNTIME_MARKER_DIR = Path(os.environ.get("KIRON_RUNTIME_DIR", "/run/kiron/vram"))
STARTUP_MARKER_PATH = RUNTIME_MARKER_DIR / "docling-vram-startup.json"
SHUTDOWN_MARKER_PATH = RUNTIME_MARKER_DIR / "docling-vram-shutdown.json"
GPU_SERVICE_LOADING_MARKER_PATH = RUNTIME_MARKER_DIR / "gpu-service-loading.json"
_MARKER_MAX_TTL_S = 15 * 60.0

GPU_SERVICE_LOADING_TTL_S = 300.0
GPU_SERVICE_START_DEADLINE_S = 240.0
GPU_SERVICE_HEALTH_POLL_S = 0.5
GPU_SERVICE_LOADING_DRAIN_TIMEOUT_S = 300.0
GPU_SERVICE_LOADING_DRAIN_POLL_S = 0.5
GPU_SERVICE_MARKER_SHORT_TTL_S = 5.0


# --- Policy-Resolver ---

_POLICY_CHOICES = ("block", "force_cpu", "pass")


def _resolve_policy() -> str:
    raw = os.environ.get("KIRON_VRAM_LEASE_POLICY", "force_cpu")
    if raw not in _POLICY_CHOICES:
        logger.warning(
            "KIRON_VRAM_LEASE_POLICY=%r ist ungueltig (erwartet: %s) — "
            "fallback auf 'force_cpu'.",
            raw, "|".join(_POLICY_CHOICES),
        )
        return "force_cpu"
    return raw


VRAM_LEASE_POLICY = _resolve_policy()


# --- Outcome-Enum ---

class LeaseOutcome(enum.Enum):
    PASS = "pass"
    FORCE_CPU = "force_cpu"
    BLOCK = "block"


class MarkerOwnershipError(RuntimeError):
    """Raised when an active marker is owned by another process/token."""


@dataclass(frozen=True)
class GPUGateDecision:
    allowed: bool
    reason: str
    status_code: int = 200
    marker_kind: str | None = None
    lease_state: str = "inactive"
    service_name: str = "GPU-Service"


@dataclass
class GPUServiceOperation:
    allowed: bool
    decision: GPUGateDecision
    token: str | None = None
    clear_marker: bool = False
    lock_acquired: bool = False
    admission_store: Any = None
    admission_id: str | None = None
    admission_generation: str | None = None
    admission_heartbeat: Any = None
    cleanup_task: Any = None
    backend_session: Any = None


# --- Module-State ---

_lease_cache: dict[str, float | bool] = {"active": False, "ts": 0.0}
_lease_fetch_lock = asyncio.Lock()
gpu_service_ops_lock = asyncio.Lock()
_shared_client: httpx.AsyncClient | None = None
_SERVICE_OPERATION_EPOCH = "native-lifecycle:" + uuid.uuid4().hex

_last_unreachable_log_ts: float = 0.0
_last_pass_warning_log_ts: float = 0.0


def _log_unreachable_throttled(err: Exception | None = None) -> None:
    """Max 1 Warning pro 60s, damit Docling-Down nicht das Log flutet."""
    global _last_unreachable_log_ts
    now = time.monotonic()
    if now - _last_unreachable_log_ts < _LEASE_UNREACHABLE_LOG_INTERVAL_S:
        return
    _last_unreachable_log_ts = now
    logger.warning(
        "VRAM-Lease: Docling-Lifecycle (%s) unreachable: %s — "
        "fallback lease_active=False",
        DOCLING_LIFECYCLE_URL, err,
    )


def _log_pass_passthrough_throttled() -> None:
    """Max 1 Warning pro 60s bei Policy=pass + aktivem Lease."""
    global _last_pass_warning_log_ts
    now = time.monotonic()
    if now - _last_pass_warning_log_ts < _PASS_WARNING_LOG_INTERVAL_S:
        return
    _last_pass_warning_log_ts = now
    logger.warning(
        "VRAM-Lease aktiv, aber Policy=pass — Request passthrough "
        "(#285 Debug-Modus)."
    )


def _warn_runtime_capability(message: str) -> dict[str, Any]:
    return {"safe": False, "reason": message}


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
    if st.st_uid != 0:
        return False, "file_not_root_owned", st
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return False, "file_group_or_world_writable", st
    parent = path.parent
    try:
        pst = os.stat(parent)
    except OSError as exc:
        return False, f"parent_stat_failed:{exc}", st
    if pst.st_uid != 0:
        return False, "parent_not_root_owned", st
    if pst.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return False, "parent_group_or_world_writable", st
    return True, "ok", st


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
    if (st.st_mode & 0o7777) != RUNTIME_MARKER_FILE_MODE:
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


def _path_is_safe_runtime_file(path: Path) -> tuple[bool, str, os.stat_result | None]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False, "missing", None
    except OSError as exc:
        return False, f"stat_failed:{exc}", None
    if stat.S_ISLNK(st.st_mode):
        return False, "symlink", st
    parent_ok, parent_reason = _path_is_safe_runtime_dir(path.parent)
    if not parent_ok:
        return False, parent_reason, st
    ok, reason = _stat_is_safe_runtime_file(st)
    return ok, reason, st


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


def invalidate_runtime_capability_cache() -> None:
    _runtime_capability_cache["ts"] = 0.0
    _runtime_capability_cache["stat_key"] = None
    _runtime_capability_cache["status"] = None


def runtime_capability_status() -> dict[str, Any]:
    """Return safety status for runtime `num_gpu=0` handoff.

    Missing, malformed or unsafe files fail closed. The short cache is
    stat-key aware so replacing/removing the file invalidates a previous
    safe result quickly and usually on the next call.
    """
    path = Path(RUNTIME_CAPABILITY_PATH)
    ok, reason, st = _path_is_safe_regular_file(path)
    stat_key = None
    if st is not None:
        stat_key = (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, st.st_mode, st.st_uid)
    now = time.monotonic()
    cached = _runtime_capability_cache.get("status")
    if (
        cached is not None
        and _runtime_capability_cache.get("stat_key") == stat_key
        and now - float(_runtime_capability_cache.get("ts", 0.0)) < _RUNTIME_CAPABILITY_CACHE_TTL_S
    ):
        return dict(cached)

    if not ok:
        status = _warn_runtime_capability(reason)
    else:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            status = _warn_runtime_capability(f"read_or_json_failed:{exc}")
        else:
            if not isinstance(data, dict):
                status = _warn_runtime_capability("root_not_object")
            elif not isinstance(data.get("image_digest"), str) or not data.get("image_digest"):
                status = _warn_runtime_capability("missing_image_digest")
            elif not isinstance(data.get("report_path"), str) or not data.get("report_path"):
                status = _warn_runtime_capability("missing_report_path")
            elif not isinstance(data.get("num_gpu_zero_effective"), bool):
                status = _warn_runtime_capability("num_gpu_zero_effective_not_bool")
            elif data.get("num_gpu_zero_effective") is not True:
                status = _warn_runtime_capability("num_gpu_zero_effective_false")
            else:
                status = {
                    "safe": True,
                    "reason": "ok",
                    "image_digest": data.get("image_digest"),
                    "report_path": data.get("report_path"),
                }

    _runtime_capability_cache["ts"] = now
    _runtime_capability_cache["stat_key"] = stat_key
    _runtime_capability_cache["status"] = dict(status)
    return status


def num_gpu_zero_effective() -> bool:
    return bool(runtime_capability_status().get("safe"))


def _marker_payload_with_stat(
    path: Path,
) -> tuple[dict[str, Any] | None, bool, os.stat_result | None]:
    """Return marker payload and active flag.

    Missing or expired markers are inactive. Unsafe, unreadable or malformed
    markers are active fail-closed and have no trusted payload.
    """
    fd, reason, st = _open_existing_runtime_file(path, os.O_RDONLY)
    if fd is None:
        if reason == "missing":
            return None, False, None
        logger.warning("VRAM-Lease Marker %s unsafe: %s", path, reason)
        return None, True, st
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = None
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("VRAM-Lease Marker %s nicht lesbar: %s", path, exc)
        return None, True, st
    finally:
        if fd is not None:
            os.close(fd)
    if not isinstance(data, dict):
        return None, True, st
    deadline = data.get("deadline_monotonic")
    created_wall = data.get("created_wall")
    ttl = data.get("ttl_s")
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        return data, time.monotonic() < float(deadline), st
    if (
        isinstance(created_wall, (int, float))
        and not isinstance(created_wall, bool)
        and isinstance(ttl, (int, float))
        and not isinstance(ttl, bool)
    ):
        ttl_s = max(0.0, min(float(ttl), _MARKER_MAX_TTL_S))
        return data, time.time() < float(created_wall) + ttl_s, st
    return None, True, st


def _marker_payload(path: Path) -> tuple[dict[str, Any] | None, bool]:
    data, active, _ = _marker_payload_with_stat(path)
    return data, active


def _marker_active(path: Path) -> bool:
    _, active = _marker_payload(path)
    return active


def _overlay_active() -> bool:
    return (
        _marker_active(STARTUP_MARKER_PATH)
        or _marker_active(SHUTDOWN_MARKER_PATH)
        or _marker_active(GPU_SERVICE_LOADING_MARKER_PATH)
    )


def _marker_path(kind: str) -> Path:
    if kind == "startup":
        return STARTUP_MARKER_PATH
    if kind == "shutdown":
        return SHUTDOWN_MARKER_PATH
    if kind == "gpu_service_loading":
        return GPU_SERVICE_LOADING_MARKER_PATH
    raise ValueError(f"unknown overlay marker kind: {kind!r}")


def overlay_marker_active(kind: str) -> bool:
    return _marker_active(_marker_path(kind))


def _hard_overlay_marker_kind() -> str | None:
    for kind in ("gpu_service_loading", "startup", "shutdown"):
        if overlay_marker_active(kind):
            return kind
    return None


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
        raise MarkerOwnershipError(
            f"VRAM lock directory {lock_path.parent} unsafe: {parent_reason}"
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
                raise MarkerOwnershipError(
                    f"VRAM lock {lock_path} unsafe: {exc}"
                ) from exc
        except OSError as exc:
            raise MarkerOwnershipError(f"VRAM lock {lock_path} unsafe: {exc}") from exc
    else:
        raise MarkerOwnershipError(f"VRAM lock {lock_path} unstable")
    try:
        if created:
            os.fchmod(lock_fd, RUNTIME_MARKER_FILE_MODE)
        ok, reason = _fd_is_safe_runtime_file(lock_fd)
        if not ok:
            raise MarkerOwnershipError(f"VRAM lock {lock_path} unsafe: {reason}")
    except Exception:
        os.close(lock_fd)
        raise
    return lock_fd


def _write_marker_payload(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{payload['token']}.tmp")
    fd: int | None = os.open(
        tmp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        RUNTIME_MARKER_FILE_MODE,
    )
    try:
        os.fchmod(fd, RUNTIME_MARKER_FILE_MODE)
        ok, reason = _fd_is_safe_runtime_file(fd)
        if not ok:
            raise MarkerOwnershipError(f"VRAM marker {tmp} unsafe: {reason}")
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


def write_overlay_marker(
    kind: str = "startup", *, ttl_s: float = 300.0, token: str | None = None,
) -> str:
    path = _marker_path(kind)
    path.parent.mkdir(parents=True, mode=RUNTIME_MARKER_DIR_MODE, exist_ok=True)
    try:
        with runtime_lock(path.parent, security=_admission_security()):
            return write_overlay_marker_locked(kind, ttl_s=ttl_s, token=token)
    except AdmissionError as exc:
        raise MarkerOwnershipError(str(exc)) from exc


def clear_overlay_marker(kind: str = "startup", token: str | None = None) -> None:
    if token is None:
        return
    path = _marker_path(kind)
    try:
        with runtime_lock(path.parent, security=_admission_security()):
            clear_overlay_marker_locked(kind, token)
    except AdmissionError as exc:
        raise MarkerOwnershipError(str(exc)) from exc


def _admission_security() -> RuntimeSecurity:
    gid = _runtime_marker_group_gid()
    if gid is None:
        raise AdmissionError("resource_unknown", "runtime marker group missing")
    # The global lock may be created by any participating runtime service.
    system_writers = RuntimeSecurity.system().writer_uids
    return RuntimeSecurity(_runtime_marker_dir_owner_uid(), gid,
                           _runtime_marker_file_owner_uids() | system_writers)


def write_overlay_marker_locked(
    kind: str = "startup", *, ttl_s: float = 300.0, token: str | None = None,
) -> str:
    """Atomically write an owned effective-gate overlay marker.

    Active foreign, tokenless, malformed or unsafe markers fail closed and are
    never overwritten. Passing `token` refreshes only that owned marker.
    """
    token = token or uuid.uuid4().hex
    path = _marker_path(kind)
    path.parent.mkdir(parents=True, mode=RUNTIME_MARKER_DIR_MODE, exist_ok=True)
    payload = {
        "token": token,
        "kind": kind,
        "pid": os.getpid(),
        "created_wall": time.time(),
        "deadline_monotonic": time.monotonic() + max(1.0, min(ttl_s, _MARKER_MAX_TTL_S)),
        "ttl_s": max(1.0, min(ttl_s, _MARKER_MAX_TTL_S)),
    }
    lock_fd = _open_marker_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current, active = _marker_payload(path)
        if active:
            current_token = current.get("token") if isinstance(current, dict) else None
            if not isinstance(current_token, str) or current_token != token:
                raise MarkerOwnershipError(f"active {kind} marker is owned by another token")
        _write_marker_payload(path, payload)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    invalidate_runtime_capability_cache()
    return token


def refresh_overlay_marker(kind: str, token: str, *, ttl_s: float = 300.0) -> bool:
    try:
        write_overlay_marker(kind, ttl_s=ttl_s, token=token)
        return True
    except MarkerOwnershipError:
        return False


def clear_overlay_marker_locked(kind: str = "startup", token: str | None = None) -> None:
    """Remove overlay marker only if ownership token matches."""
    path = _marker_path(kind)
    if token is None:
        return
    path.parent.mkdir(parents=True, mode=RUNTIME_MARKER_DIR_MODE, exist_ok=True)
    lock_fd = _open_marker_lock(path)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data, active, marker_st = _marker_payload_with_stat(path)
        if data is None and active:
            return
        if data is None:
            return
        if data.get("token") != token:
            return
        if marker_st is not None:
            try:
                current_st = os.lstat(path)
            except OSError:
                return
            if _runtime_file_identity(current_st) != _runtime_file_identity(marker_st):
                logger.warning("VRAM-Lease Marker %s changed before clear", path)
                return
            ok, reason = _stat_is_safe_runtime_file(current_st)
            if not ok:
                logger.warning("VRAM-Lease Marker %s unsafe before clear: %s", path, reason)
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


# --- lifespan / shared client ---

@asynccontextmanager
async def lifespan_client():
    """Verwaltet den langlebigen httpx-Client zu Docling-Lifecycle.

    Aufruf aus main.py Startup. Setzt `_shared_client` waehrend des
    Kontextes und leert ihn beim Exit.
    """
    global _shared_client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_LEASE_FETCH_TIMEOUT_S,
            read=_LEASE_FETCH_TIMEOUT_S,
            write=_LEASE_FETCH_TIMEOUT_S,
            pool=_LEASE_FETCH_TIMEOUT_S,
        ),
        limits=httpx.Limits(
            max_connections=5,
            max_keepalive_connections=2,
        ),
    )
    _shared_client = client
    try:
        yield client
    finally:
        _shared_client = None
        try:
            await client.aclose()
        except Exception:
            logger.exception("VRAM-Lease client.aclose() fehlgeschlagen")


# --- snapshot (TTL-Cache + Thundering-Herd-Lock) ---

async def snapshot() -> bool:
    """Liefert den gecachten Lease-Zustand (True = Docling haelt GPU).

    Robust gegen `_shared_client is None` (vor lifespan_client-Startup
    oder nach Shutdown): liefert `False`, keine Exception, kein Log.

    TTL 2s; parallele Cache-Misses werden unter `_lease_fetch_lock`
    serialisiert — der Lifecycle-Endpoint bekommt genau EINEN Fetch
    statt N.
    """
    if _shared_client is None:
        return False
    now = time.monotonic()
    if now - float(_lease_cache["ts"]) < _LEASE_CACHE_TTL_S:
        return bool(_lease_cache["active"])

    async with _lease_fetch_lock:
        # Zweiter Check: paralleler Fetch hat moeglicherweise schon
        # waehrend wir auf das Lock warteten den Cache refreshed.
        now = time.monotonic()
        if now - float(_lease_cache["ts"]) < _LEASE_CACHE_TTL_S:
            return bool(_lease_cache["active"])
        try:
            resp = await _shared_client.get(DOCLING_LIFECYCLE_URL)
            if resp.status_code == 200:
                data = resp.json()
                _lease_cache["active"] = bool(data.get("vram_lease_active"))
                _lease_cache["ts"] = time.monotonic()
                return bool(_lease_cache["active"])
            _log_unreachable_throttled(
                RuntimeError(f"HTTP {resp.status_code}")
            )
        except Exception as e:
            _log_unreachable_throttled(e)
        _lease_cache["active"] = False
        _lease_cache["ts"] = time.monotonic()
        return False


async def effective_snapshot() -> bool:
    """Lifecycle snapshot plus crash/startup/shutdown overlays.

    `snapshot()` remains lifecycle-only and keeps its historical
    fail-open fallback for callers like the release watcher. Runtime GPU
    gates call this function.
    """
    if _overlay_active():
        return True
    return await snapshot()


async def gpu_gate_decision(
    force: bool = False,
    service_name: str = "GPU-Service",
) -> GPUGateDecision:
    """Shared hard-GPU gate decision without framework response objects."""
    hard_kind = _hard_overlay_marker_kind()
    if hard_kind is not None:
        return GPUGateDecision(
            allowed=False,
            reason="gpu_overlay_active",
            status_code=409,
            marker_kind=hard_kind,
            lease_state="overlay",
            service_name=service_name,
        )
    lease_active = await snapshot()
    if lease_active and not force and VRAM_LEASE_POLICY != "pass":
        return GPUGateDecision(
            allowed=False,
            reason="docling_lifecycle_lease_active",
            status_code=409,
            lease_state="lifecycle",
            service_name=service_name,
        )
    if lease_active and VRAM_LEASE_POLICY == "pass":
        _log_pass_passthrough_throttled()
    return GPUGateDecision(
        allowed=True,
        reason="allowed",
        status_code=200,
        lease_state="lifecycle" if lease_active else "inactive",
        service_name=service_name,
    )


def _marker_write_decision(
    service_name: str,
    exc: BaseException,
) -> GPUGateDecision:
    return GPUGateDecision(
        allowed=False,
        reason=f"gpu_service_marker_write_failed:{type(exc).__name__}",
        status_code=409,
        lease_state="marker_write_failed",
        service_name=service_name,
    )


@asynccontextmanager
async def gpu_service_operation(
    *,
    force: bool = False,
    service_name: str = "GPU-Service",
    marker_ttl_s: float = GPU_SERVICE_LOADING_TTL_S,
    releases_resources: bool = False,
    serialize_release: bool = False,
    gpu_memory: NativeMemoryBudget | None = None,
):
    """Serialize a marker-led GPU service operation in the proxy process.

    The marker is fail-closed by default. Callers set `op.clear_marker = True`
    only after an endpoint-specific, unambiguous success or no-start result.
    """
    op = await begin_gpu_service_operation(
        force=force,
        service_name=service_name,
        marker_ttl_s=marker_ttl_s,
        releases_resources=releases_resources,
        serialize_release=serialize_release,
        gpu_memory=gpu_memory,
    )
    try:
        yield op
    finally:
        await finish_gpu_service_operation(op)


async def begin_gpu_service_operation(
    *,
    force: bool = False,
    service_name: str = "GPU-Service",
    marker_ttl_s: float = GPU_SERVICE_LOADING_TTL_S,
    releases_resources: bool = False,
    serialize_release: bool = False,
    gpu_memory: NativeMemoryBudget | None = None,
) -> GPUServiceOperation:
    # Stop/unload/delete cannot allocate a new model and must remain available
    # when another provider owns admission. Delete alone keeps its atomic
    # loaded-check/delete sequence; stop/unload must interrupt a stuck request.
    if releases_resources:
        if serialize_release:
            await gpu_service_ops_lock.acquire()
        return GPUServiceOperation(True, GPUGateDecision(True, "resource_release",
                                   service_name=service_name), lock_acquired=serialize_release)
    await gpu_service_ops_lock.acquire()
    try:
        decision = await gpu_gate_decision(force=force, service_name=service_name)
    except BaseException:
        gpu_service_ops_lock.release()
        raise
    if not decision.allowed:
        gpu_service_ops_lock.release()
        return GPUServiceOperation(False, decision)
    try:
        token = write_overlay_marker(
            "gpu_service_loading",
            ttl_s=marker_ttl_s,
        )
    except Exception as exc:
        logger.warning("GPU-Service marker write failed for %s: %s", service_name, exc)
        gpu_service_ops_lock.release()
        return GPUServiceOperation(False, _marker_write_decision(service_name, exc))
    op = GPUServiceOperation(True, decision, token=token, lock_acquired=True)
    post_kind = None
    for kind in ("startup", "shutdown"):
        if overlay_marker_active(kind):
            post_kind = kind
            break
    if post_kind is not None:
        op.allowed = False
        op.decision = GPUGateDecision(
            allowed=False,
            reason="gpu_overlay_active",
            status_code=409,
            marker_kind=post_kind,
            lease_state="overlay",
            service_name=service_name,
        )
        op.clear_marker = True
        await finish_gpu_service_operation(op)
        return op
    try:
        # Lazy import avoids a module cycle; this creates no service at import.
        from native_admission import make_store, measure_memory
        store = make_store()
        if service_name == "Ollama":
            from kiron_common.gpu_admission.ollama_backend import OllamaBackendSession
            op.backend_session = OllamaBackendSession(store)
            await op.backend_session.__aenter__()
        operation_id = uuid.uuid4().hex
        # Model-specific loads reserve the Catalog budget. Other lifecycle
        # operations remain unmeasured and conflict with managed Prism residency.
        store.reserve(operation_id=operation_id, owner="kiron-proxy-lifecycle",
            generation=_SERVICE_OPERATION_EPOCH, deployment_id=f"native-lifecycle:{service_name}",
            kind="request", gpu_bytes=gpu_memory.additional_bytes(loading=True) if gpu_memory else 0,
            headroom_bytes=gpu_memory.headroom_bytes if gpu_memory else 0,
            host_bytes=0, measure=measure_memory,
            owned_overlays={"gpu-service-loading.json": token},
            backend_instance=op.backend_session.instance if op.backend_session else None,
            overlay_token=token,
            conflicting_resident_slots=("prism",))
        op.admission_store, op.admission_id = store, operation_id
        op.admission_generation = _SERVICE_OPERATION_EPOCH
        op.admission_heartbeat = asyncio.create_task(_heartbeat_service_operation(op))
    except AdmissionError as exc:
        op.allowed = False
        op.decision = GPUGateDecision(False, exc.code,
            status_code=409 if exc.code == "resource_conflict" else 503,
            lease_state="admission", service_name=service_name)
        op.clear_marker = True  # The caller has not started any backend work.
        await finish_gpu_service_operation(op)
    except BaseException:
        op.clear_marker = True
        await finish_gpu_service_operation(op)
        raise
    return op


async def _heartbeat_service_operation(op: GPUServiceOperation) -> None:
    try:
        while True:
            await asyncio.sleep(60)
            op.admission_store.heartbeat(op.admission_id, owner="kiron-proxy-lifecycle",
                                        generation=op.admission_generation)
    except AdmissionError:
        logger.error("Native lifecycle admission heartbeat became unknown")


async def _finish_gpu_service_operation(op: GPUServiceOperation) -> None:
    try:
        if op.admission_heartbeat is not None:
            op.admission_heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await op.admission_heartbeat
        if op.admission_id is not None:
            # clear_marker is an endpoint-specific end/no-start proof, not
            # merely return from an HTTP handler. Timeout/cancel stays unknown.
            op.admission_store.release(op.admission_id, owner="kiron-proxy-lifecycle",
                generation=op.admission_generation, confirmed_terminated=op.clear_marker,
                owned_overlays=({"gpu-service-loading.json": op.token}
                                if op.clear_marker and op.token else None))
    finally:
        try:
            if op.clear_marker and op.token is not None:
                clear_overlay_marker("gpu_service_loading", op.token)
        finally:
            if op.backend_session is not None:
                op.backend_session.close()
            if op.lock_acquired:
                op.lock_acquired = False
                gpu_service_ops_lock.release()


async def finish_gpu_service_operation(op: GPUServiceOperation) -> None:
    if op.cleanup_task is None:
        op.cleanup_task = asyncio.create_task(_finish_gpu_service_operation(op))
    await asyncio.shield(op.cleanup_task)


# --- Interne Helfer ---

def _num_gpu_explicit(options: dict) -> bool:
    """True wenn der Client num_gpu explizit gesetzt hat.

    Wichtig: 0 gilt als explizit (Client will bewusst CPU), 99 auch.
    """
    return "num_gpu" in options


def _explicit_num_gpu_zero(options: dict) -> bool:
    return is_real_int(options.get("num_gpu")) and options.get("num_gpu") == 0


def _evaluate_chat_generate_options(options: dict, *, explicit: bool) -> LeaseOutcome:
    if explicit:
        if _explicit_num_gpu_zero(options) and num_gpu_zero_effective():
            return LeaseOutcome.FORCE_CPU
        return LeaseOutcome.BLOCK
    if not num_gpu_zero_effective():
        return LeaseOutcome.BLOCK
    ensure_num_gpu_zero(options)
    return LeaseOutcome.FORCE_CPU


def _explicit_verified_cpu_offload_from_body(body: bytes) -> bool:
    try:
        data = json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    options = data.get("options")
    if not isinstance(options, dict):
        return False
    return _explicit_num_gpu_zero(options) and num_gpu_zero_effective()


def _embed_outcome(
    route: ProxyRoute,
    lease_active: bool,
    policy: str,
) -> LeaseOutcome:
    """Sub-Logik fuer /api/embed-Pfad (Pfad 2)."""
    if not lease_active:
        return LeaseOutcome.PASS
    if policy == "pass":
        _log_pass_passthrough_throttled()
        return LeaseOutcome.PASS
    if not (
        route.backend is BackendType.OLLAMA
        and route.task is ModelTask.EMBEDDING
        and route.endpoint is ModelEndpoint.EMBED
    ):
        # Service-owned Catalog deployments are not part of this intercept.
        return LeaseOutcome.PASS
    if policy == "block":
        return LeaseOutcome.BLOCK
    # force_cpu: Ollama honoriert num_gpu auf /api/embed nicht zuverlaessig
    # (F1). Konservativ auf Block zurueckfallen, ausser explizit deaktiviert.
    if _EMBED_BLOCK_FALLBACK:
        return LeaseOutcome.BLOCK
    return LeaseOutcome.FORCE_CPU


# --- Public Intercept-Helfer ---

def is_native_unload(body: bytes, path: str) -> bool:
    """Recognize only the no-inference Ollama generate release payload.

    Inference with keep_alive=0 must still reserve GPU resources. Unknown keys,
    options, nonempty prompts and chat messages cannot enter this exemption.
    """
    if path != "/api/generate":
        return False
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        return False
    return (isinstance(value, dict)
            and not set(value) - {"model", "keep_alive", "stream", "prompt", "think"}
            and isinstance(value.get("model"), str) and bool(value["model"])
            and type(value.get("keep_alive")) is int and value["keep_alive"] == 0
            and value.get("stream") is False and value.get("prompt", "") == ""
            and value.get("think", False) is False)


async def apply_bytes(
    body: bytes,
    path: str,
    model: str,
    routing_view: ProxyRoutingView = PROXY_ROUTING_VIEW,
) -> tuple[bytes, LeaseOutcome]:
    """Lease-Intercept fuer bytes-basierte Bodies (proxy.py-Pfad).

    Liefert (possibly-modified-body, outcome):
    - PASS      → Body unveraendert weiter, kein Header.
    - FORCE_CPU → Body mit `options.num_gpu=0` ergaenzt, Header
                  `X-Kiron-VRAM-Lease: force-cpu`.
    - BLOCK     → Caller liefert 503-JSON-Body, Header
                  `X-Kiron-VRAM-Lease: blocked`.

    Angewandte Matrix:
    - path nicht in STREAMING_INTERCEPT_PATHS ∪ {/api/embed} → PASS.
    - JSON-unparseable → PASS.
    - Client-`options.num_gpu` bereits gesetzt → PASS.
    - /api/embed only for an exact Catalog route whose selected deployment is
      backend=ollama, task=embedding, endpoint=/api/embed.
    - unknown or endpoint-foreign embedding names → PASS without consulting
      an overlay marker or lease snapshot; proxy.py returns their 400.
    """
    if not isinstance(routing_view, ProxyRoutingView):
        raise TypeError("routing_view must be a ProxyRoutingView")
    if is_native_unload(body, path):
        return body, LeaseOutcome.PASS
    normalized_path = path
    is_embed = normalized_path == "/api/embed"
    if normalized_path not in STREAMING_INTERCEPT_PATHS and not is_embed:
        return body, LeaseOutcome.PASS

    embed_route = None
    if is_embed:
        embed_route = routing_view.resolve(model, ModelEndpoint.EMBED)
        if embed_route is None:
            return body, LeaseOutcome.PASS

    hard_kind = _hard_overlay_marker_kind()
    if hard_kind is not None:
        if (
            normalized_path in STREAMING_INTERCEPT_PATHS
            and _explicit_verified_cpu_offload_from_body(body)
        ):
            return body, LeaseOutcome.PASS
        return body, LeaseOutcome.BLOCK

    lease_active = await snapshot()
    policy = VRAM_LEASE_POLICY

    if is_embed:
        if embed_route is None:  # pragma: no cover - guarded above
            raise AssertionError("resolved embedding route disappeared")
        outcome = _embed_outcome(embed_route, lease_active, policy)
        if outcome == LeaseOutcome.PASS:
            return body, LeaseOutcome.PASS
        if outcome == LeaseOutcome.BLOCK:
            return body, LeaseOutcome.BLOCK
        return body, LeaseOutcome.BLOCK

    # chat/generate
    if not lease_active:
        return body, LeaseOutcome.PASS
    if policy == "pass":
        _log_pass_passthrough_throttled()
        return body, LeaseOutcome.PASS
    if policy == "block":
        return body, LeaseOutcome.BLOCK
    # force_cpu
    try:
        data = json.loads(body) if body else {}
        if not isinstance(data, dict):
            return body, LeaseOutcome.PASS
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, LeaseOutcome.PASS

    options = data.get("options")
    if not isinstance(options, dict):
        options = {}
    explicit_num_gpu = _num_gpu_explicit(options)
    outcome = _evaluate_chat_generate_options(options, explicit=explicit_num_gpu)
    if outcome == LeaseOutcome.BLOCK:
        return body, LeaseOutcome.BLOCK
    data["options"] = options
    new_body = json.dumps(data, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
    return new_body, outcome


async def apply_options_dict(options: dict) -> LeaseOutcome:
    """Lease-Intercept fuer Dict-basierte Bodies (openai_api.py-Pfad).

    Ruft intern `snapshot()` — symmetrisch zu `apply_bytes`. Mutiert
    das uebergebene Dict in place: setzt `options["num_gpu"] = 0` bei
    FORCE_CPU, wenn der Client num_gpu nicht bereits explizit gesetzt
    hat. Gibt Outcome zurueck:

    - PASS      → Dict unveraendert, kein Header.
    - FORCE_CPU → Dict mutiert, Caller setzt Header
                  `X-Kiron-VRAM-Lease: force-cpu`.
    - BLOCK     → Dict unveraendert, Caller liefert 503.
    """
    hard_kind = _hard_overlay_marker_kind()
    if hard_kind is not None:
        if _explicit_num_gpu_zero(options) and num_gpu_zero_effective():
            return LeaseOutcome.PASS
        return LeaseOutcome.BLOCK

    lease_active = await snapshot()
    policy = VRAM_LEASE_POLICY
    if not lease_active:
        return LeaseOutcome.PASS
    if policy == "pass":
        _log_pass_passthrough_throttled()
        return LeaseOutcome.PASS
    if policy == "block":
        return LeaseOutcome.BLOCK
    # force_cpu
    return _evaluate_chat_generate_options(
        options,
        explicit=_num_gpu_explicit(options),
    )


# --- Body-Modifikation ---

def _inject_num_gpu_zero(body: bytes) -> bytes:
    """Setzt options.num_gpu=0 in JSON-Body. Unparseable → Original."""
    try:
        data = json.loads(body) if body else {}
        if not isinstance(data, dict):
            return body
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    options = data.get("options")
    if not isinstance(options, dict):
        options = {}
    if "num_gpu" not in options:
        options["num_gpu"] = 0
    data["options"] = options
    return json.dumps(data, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
