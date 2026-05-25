"""Kiron Proxy - Haupteinstiegspunkt. Startet Proxy + Dashboard + OpenAI API."""

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import sys
import threading
from pathlib import Path

import time

import httpx
import uvicorn

from request_store import RequestStore
from proxy import create_proxy_app
from app import app as dashboard_app, set_store, set_metrics_db, set_db_work_tracker, set_api_key_store, restore_maintenance_state, OLLAMA_BASE_URL
from history_db import MetricsDB
import metrics
from api_key_store import ApiKeyStore
from openai_api import create_openai_api_app
import vram_lease

_SERVER_BACKLOG = 2048
_SERVER_BINDINGS = (
    ("proxy", "0.0.0.0", 11434),
    ("dashboard", "0.0.0.0", 8505),
    ("openai", "0.0.0.0", 11440),
)

_COMMON_SRC = Path(__file__).resolve().parent.parent / "kiron-common"
if _COMMON_SRC.exists() and str(_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(_COMMON_SRC))
try:
    from kiron_common.ollama_compat import is_real_int
except ImportError:  # pragma: no cover
    def is_real_int(value):
        return isinstance(value, int) and not isinstance(value, bool)


def _socket_family(host: str) -> socket.AddressFamily:
    if ":" in host:
        return socket.AF_INET6
    return socket.AF_INET


def _close_sockets(sockets) -> None:
    for sock in sockets:
        with contextlib.suppress(OSError):
            sock.close()


def _prebind_tcp_sockets(
    bindings=_SERVER_BINDINGS,
    backlog: int = _SERVER_BACKLOG,
) -> dict[str, socket.socket]:
    """Bindet die Runtime-Ports vor DB-Side-Effects und gibt Uvicorn-Sockets zurueck."""
    bound: dict[str, socket.socket] = {}
    try:
        for name, host, port in bindings:
            sock = socket.socket(_socket_family(host), socket.SOCK_STREAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.listen(backlog)
                sock.setblocking(False)
                sock.set_inheritable(True)
            except OSError:
                _close_sockets((sock,))
                raise
            bound[name] = sock
    except Exception:
        _close_sockets(bound.values())
        raise
    return bound


async def _await_shielded_tick_task(tick_task: asyncio.Task):
    """Wartet einen Producer-Tick ab, ohne ihn bei Producer-Cancellation zu canceln."""
    try:
        return await asyncio.shield(tick_task)
    except asyncio.CancelledError:
        if not tick_task.done():
            try:
                await asyncio.shield(tick_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        elif not tick_task.cancelled():
            with contextlib.suppress(Exception):
                tick_task.exception()
        raise


def _consume_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.result()


class DBWorkTracker:
    """Trackt SQLite-Arbeit, die in Worker-Threads weiterlaufen kann (#644)."""

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    async def to_thread(self, func, /, *args, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        self._tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._tasks.discard(done_task)
            _consume_task_exception(done_task)

        task.add_done_callback(_done)
        return await asyncio.shield(task)

    async def drain(self, timeout: float) -> bool:
        tasks = [task for task in self._tasks if not task.done()]
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in done:
            _consume_task_exception(task)
        return not pending


async def _run_initial_producer_tick(store: RequestStore) -> tuple[bool, asyncio.Task | None]:
    """Best-effort Initial-Tick mit nicht-cancelndem Timeout-Watchdog."""
    log = logging.getLogger("metrics.producer")
    started = time.monotonic()
    task = asyncio.create_task(metrics._run_producer_tick(store))
    try:
        result = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=metrics._PRODUCER_INITIAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        if task.done():
            try:
                result = task.result()
            except Exception:
                log.warning(
                    "metrics_producer initial tick failed — service starting with empty cache",
                    exc_info=True,
                )
                return False, None
            if result is not None:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                log.info("metrics_producer initial tick complete (%dms)", elapsed_ms)
                return True, None
            log.warning("metrics_producer initial tick failed — service starting with empty cache")
            return False, None
        log.warning("metrics_producer initial tick timed out — service starting with empty cache")
        return False, task
    except asyncio.CancelledError:
        raise
    except Exception:
        log.warning(
            "metrics_producer initial tick failed — service starting with empty cache",
            exc_info=True,
        )
        return False, None

    if result is None:
        log.warning("metrics_producer initial tick failed — service starting with empty cache")
        return False, None

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info("metrics_producer initial tick complete (%dms)", elapsed_ms)
    return True, None


async def disk_io_sampler():
    """#293: Zentraler Disk-I/O-Sampler (alle 10s).

    Frueher setzte jeder Consumer (10s-Writer, 2s-WebSocket, REST) die Baseline
    zurueck - dadurch sah der Writer haeufig nahe-0-Werte. Jetzt sampelt nur
    dieser Task, alle Consumer lesen den gecachten Snapshot.
    """
    interval = 10.0
    # #978: Baseline + Warmup-Sample 1s spaeter, damit der Snapshot nicht 10s
    # lang None liefert (DB-History haette sonst NULL fuer die ersten Ticks).
    # #463-Vertrag (None = noch keine Daten) bleibt erhalten — der Warmup
    # ersetzt None nur frueher mit echten Werten.
    try:
        await asyncio.to_thread(metrics.sample_disk_io)
    except Exception:
        logging.getLogger(__name__).exception("disk_io_sampler baseline failed")
    await asyncio.sleep(1.0)
    try:
        await asyncio.to_thread(metrics.sample_disk_io)
    except Exception:
        logging.getLogger(__name__).exception("disk_io_sampler warmup failed")
    next_tick = time.monotonic() + interval
    while True:
        now = time.monotonic()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
            next_tick += interval
        else:
            next_tick = now + interval
        try:
            await asyncio.to_thread(metrics.sample_disk_io)
        except Exception:
            logging.getLogger(__name__).exception("disk_io_sampler failed")


async def cpu_sampler():
    """#586: Zentraler CPU-Sampler (alle 10s).

    Frueher setzte jeder Metrics-Consumer den psutil.cpu_percent-Cache zurueck.
    Jetzt sampelt nur dieser Task, alle Consumer lesen den gecachten Wert.
    """
    interval = 10.0
    next_tick = time.monotonic() + interval
    try:
        await asyncio.to_thread(metrics.sample_cpu_percent)
    except Exception:
        logging.getLogger(__name__).exception("cpu_sampler baseline failed")
    while True:
        now = time.monotonic()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
            next_tick += interval
        else:
            next_tick = now + interval
        try:
            await asyncio.to_thread(metrics.sample_cpu_percent)
        except Exception:
            logging.getLogger(__name__).exception("cpu_sampler failed")


async def metrics_producer(store, pending_initial_task: asyncio.Task | None = None):
    """Produziert alle 2 Sekunden einen zentralen Metrics+Summary-Cache-Snapshot."""
    log = logging.getLogger("metrics.producer")
    interval = metrics._PRODUCER_INTERVAL_S
    consecutive_errors = 0
    last_state = "error" if pending_initial_task is not None else "ok"

    def record_result(result: dict | None) -> None:
        nonlocal consecutive_errors, last_state
        if result is None:
            consecutive_errors += 1
            if last_state != "error":
                log.warning("metrics_producer tick failed — keeping last snapshot")
            last_state = "error"
            return

        if last_state == "error":
            log.info("metrics_producer recovered")
        last_state = "ok"
        consecutive_errors = 0

    if pending_initial_task is not None:
        try:
            result = await asyncio.wait_for(
                asyncio.shield(pending_initial_task),
                timeout=metrics._PRODUCER_INITIAL_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "metrics_producer pending initial tick still running — "
                "starting periodic loop"
            )
            pending_initial_task.add_done_callback(_consume_task_exception)
            result = None
        except asyncio.CancelledError:
            if not pending_initial_task.done():
                try:
                    await asyncio.shield(pending_initial_task)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
            elif not pending_initial_task.cancelled():
                _consume_task_exception(pending_initial_task)
            raise
        except Exception:
            result = None
        record_result(result)

    next_tick = time.monotonic() + interval
    while True:
        now = time.monotonic()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
            next_tick += interval
        else:
            await asyncio.sleep(0.5)
            next_tick = now + interval

        tick_task = asyncio.create_task(metrics._run_producer_tick(store))
        try:
            result = await _await_shielded_tick_task(tick_task)
        except Exception:
            result = None
        record_result(result)


async def metrics_writer(db: MetricsDB, db_work: DBWorkTracker | None = None):
    """Schreibt alle 10 Sekunden Metriken in SQLite."""
    if db_work is None:
        db_work = DBWorkTracker()
    log = logging.getLogger(__name__)
    interval = 10.0
    next_tick = time.monotonic() + interval
    last_written_produced_at: float | None = None
    cache_empty_logged = False
    last_state = "ok"
    while True:
        try:
            payload = metrics.get_cached_payload()
            if payload is None:
                if not cache_empty_logged:
                    log.warning("metrics_writer skipped: metrics cache not ready")
                    cache_empty_logged = True
            else:
                cache_empty_logged = False
                produced_at = payload["produced_at_monotonic"]
                if last_written_produced_at is None or produced_at > last_written_produced_at:
                    await db_work.to_thread(db.insert_metrics, payload["system"])
                    last_written_produced_at = produced_at
                    if last_state == "error":
                        log.info("metrics_writer recovered")
                        last_state = "ok"
        except Exception:
            # State-change-Logging: bei dauerhaftem DB-Fehler nicht jeden 10s-Tick
            # einen vollen Traceback emittieren, sondern nur beim Zustandswechsel.
            if last_state != "error":
                log.exception("metrics_writer failed")
                last_state = "error"
        now = time.monotonic()
        sleep_for = next_tick - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
            next_tick += interval
        else:
            # Slow-path: Mindestpause, damit Tick > 10s nicht in back-to-back
            # Iterationen ohne Event-Loop-Yield landet.
            await asyncio.sleep(0.5)
            next_tick = now + interval


async def _promote_cpu_resident_models() -> int:
    """Unloaded alle Ollama-Modelle die vollstaendig auf CPU liegen.

    Pro entladetes Modell sendet diese Funktion `keep_alive=0` an /api/generate.
    Ollama stellt den Befehl hinter laufende Requests in die Queue — keine
    Disruption. Der naechste Chat-Request zum entladenen Modell triggert
    Ollama's Lazy-Load; weil der VRAM-Lease inaktiv ist, kein num_gpu=0
    mehr injiziert wird, landet das Modell dann auf GPU.

    Nur `size_vram == 0` (vollstaendig CPU) wird promoted. Partial-Offloads
    (zu gross fuer VRAM) werden absichtlich nicht angefasst.

    Return: Anzahl der entladenen Modelle.
    """
    log = logging.getLogger(__name__)
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10) as client:
            ps_resp = await client.get("/api/ps")
            if ps_resp.status_code != 200:
                log.warning("promote: /api/ps lieferte HTTP %s", ps_resp.status_code)
                return 0
            try:
                ps_data = ps_resp.json()
            except json.JSONDecodeError:
                log.warning("promote: /api/ps JSON-Parse fehlgeschlagen")
                return 0

            if not isinstance(ps_data, dict) or not isinstance(ps_data.get("models"), list):
                log.warning("promote: /api/ps Shape unbekannt")
                return 0

            cpu_models = []
            for m in ps_data.get("models", []):
                if not isinstance(m, dict):
                    log.warning("promote: /api/ps models[] Shape unbekannt")
                    return 0
                name = m.get("name")
                if not isinstance(name, str) or not name:
                    continue
                size_vram = m.get("size_vram")
                if not is_real_int(size_vram):
                    log.warning("promote: %s ohne echten integer size_vram; fail-closed", name)
                    continue
                if size_vram == 0:
                    cpu_models.append(name)

            if not cpu_models:
                return 0

            log.info("promote: %d CPU-residente Modelle werden entladen: %s",
                     len(cpu_models), ", ".join(cpu_models))

            unloaded = 0
            for name in cpu_models:
                try:
                    r = await client.post("/api/generate",
                                          json={"model": name, "keep_alive": 0, "stream": False})
                    if r.status_code == 200:
                        unloaded += 1
                    else:
                        log.warning("promote: unload %s → HTTP %s", name, r.status_code)
                except Exception:
                    log.exception("promote: unload %s fehlgeschlagen", name)
            return unloaded
    except httpx.ConnectError:
        log.warning("promote: Ollama nicht erreichbar")
        return 0
    except Exception:
        log.exception("promote: unerwarteter Fehler")
        return 0


async def lease_release_watcher():
    """Erkennt VRAM-Lease-Freigabe (True→False) und promotet CPU-Modelle.

    Pollt alle 5s `vram_lease.snapshot()`. Bei Uebergang active→inactive wird
    `_promote_cpu_resident_models()` aufgerufen. Deaktivierbar mit
    `KIRON_LEASE_PROMOTE_ON_RELEASE=0`.
    """
    log = logging.getLogger(__name__)
    if os.environ.get("KIRON_LEASE_PROMOTE_ON_RELEASE", "1").strip() == "0":
        log.info("lease_release_watcher: deaktiviert via Env")
        return

    interval = 5.0
    # Startzustand lesen, um einen initialen False-Snapshot nicht als Release
    # misszuverstehen (z.B. wenn der Proxy hochfaehrt und Docling gerade
    # inaktiv ist).
    try:
        prev_active = await vram_lease.snapshot()
    except Exception:
        log.exception("lease_release_watcher: initialer snapshot fehlgeschlagen")
        prev_active = False

    while True:
        await asyncio.sleep(interval)
        try:
            active = await vram_lease.snapshot()
        except Exception:
            log.exception("lease_release_watcher: snapshot fehlgeschlagen")
            continue

        if prev_active and not active:
            log.info("VRAM-Lease freigegeben — pruefe CPU-residente Modelle")
            try:
                n = await _promote_cpu_resident_models()
                if n > 0:
                    log.info("promote: %d Modell(e) entladen (naechster "
                             "Request laed lazy auf GPU)", n)
            except Exception:
                log.exception("lease_release_watcher: promote fehlgeschlagen")

        prev_active = active


async def daily_retention(
    db: MetricsDB,
    cancel_event: threading.Event,
    db_work: DBWorkTracker | None = None,
):
    """Taegliche harte Retention alter Metrikdaten.

    Laeuft einmal direkt nach Start (#233/#911 - sonst koennen Restart-Schleifen
    die Retention dauerhaft unterbrechen), danach alle 24h.

    cancel_event wird beim Shutdown gesetzt (#383) - to_thread ist nicht
    unterbrechbar, daher prueft enforce_retention das Flag kooperativ.
    """
    if db_work is None:
        db_work = DBWorkTracker()
    while True:
        try:
            deleted = await db_work.to_thread(
                db.enforce_retention, 90, cancel_event
            )
            logging.getLogger(__name__).info("Retention: %d Datenpunkte entfernt", deleted)
        except Exception:
            logging.getLogger(__name__).exception("daily_retention failed")
        await asyncio.sleep(86400)


async def main():
    # Datenverzeichnis
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"FEHLER: Datenverzeichnis kann nicht erstellt werden: {data_dir}")
        print(f"Grund: {e}")
        raise SystemExit(1)

    # DB-Config laden
    db_config_path = data_dir / "db_config.json"
    if not db_config_path.exists():
        print(f"FEHLER: DB-Config nicht gefunden: {db_config_path}")
        print("Bitte db_config.json mit MariaDB-Credentials anlegen.")
        raise SystemExit(1)

    try:
        with open(db_config_path) as f:
            db_config = json.load(f)
        if not isinstance(db_config, dict):
            print(f"FEHLER: db_config.json: Top-Level muss ein Objekt sein, nicht {type(db_config).__name__}.")
            raise SystemExit(1)
        for required_key in ("host", "user", "password", "database"):
            if required_key not in db_config:
                print(f"FEHLER: db_config.json: Pflichtfeld '{required_key}' fehlt.")
                raise SystemExit(1)
            value = db_config[required_key]
            if not isinstance(value, str) or not value:
                print(f"FEHLER: db_config.json: Pflichtfeld '{required_key}' muss ein nicht-leerer String sein.")
                raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"FEHLER: db_config.json ist kein gueltiges JSON: {e}")
        raise SystemExit(1)
    except OSError as e:
        print(f"FEHLER: db_config.json konnte nicht gelesen werden: {e}")
        raise SystemExit(1)

    # #774: Ports vor allen DB-Side-Effects reservieren. Ein irrtuemlich
    # gestarteter zweiter Proxy-Prozess darf keine aktiven Requests abbrechen,
    # bevor er am Port-Konflikt scheitert.
    server_sockets = _prebind_tcp_sockets()
    server_bindings = {
        name: (host, port)
        for name, host, port in _SERVER_BINDINGS
    }

    # MariaDB-Init mit Retry/Backoff (Boot-Race: mariadb noch nicht ready)
    log = logging.getLogger(__name__)
    backoff_seconds = [2, 5, 10, 20, 30]
    store = RequestStore(db_config)
    api_key_store = ApiKeyStore(db_config)
    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + backoff_seconds):
        if delay:
            log.warning("MariaDB init fehlgeschlagen (%s) - Retry in %ds", last_err, delay)
            await asyncio.sleep(delay)
        try:
            store.init_db()
            api_key_store.init_db()
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is not None:
        log.error("MariaDB init nach %d Versuchen aufgegeben: %s", len(backoff_seconds) + 1, last_err)
        raise last_err

    set_store(store)
    set_api_key_store(api_key_store)

    # History-DB initialisieren (bleibt SQLite)
    db = MetricsDB(str(data_dir / "metrics.db"))
    db.init_db()
    set_metrics_db(db)

    # Proxy-App erstellen (leitet an Ollama weiter)
    proxy_app = create_proxy_app(store)

    # OpenAI-kompatible API erstellen
    openai_app = create_openai_api_app(store, api_key_store)

    # Server konfigurieren
    proxy_config = uvicorn.Config(
        proxy_app,
        host=server_bindings["proxy"][0],
        port=server_bindings["proxy"][1],
        backlog=_SERVER_BACKLOG,
        log_level="warning",
    )
    dashboard_config = uvicorn.Config(
        dashboard_app,
        host=server_bindings["dashboard"][0],
        port=server_bindings["dashboard"][1],
        backlog=_SERVER_BACKLOG,
        log_level="info",
    )
    openai_config = uvicorn.Config(
        openai_app,
        host=server_bindings["openai"][0],
        port=server_bindings["openai"][1],
        backlog=_SERVER_BACKLOG,
        log_level="warning",
    )

    proxy_server = uvicorn.Server(proxy_config)
    dashboard_server = uvicorn.Server(dashboard_config)
    openai_server = uvicorn.Server(openai_config)

    # #569: Uvicorn's Server.serve() installs a per-server SIGTERM/SIGINT
    # handler via signal.signal(). Process-wide handlers don't stack, so with
    # three embedded servers via asyncio.gather only the last-installed sees
    # the signal. The other two serve() loops never terminate, gather blocks,
    # and the finally cleanup runs only after systemd-SIGKILL. Disable
    # per-server capture and route shutdown through one handler.
    servers = (proxy_server, dashboard_server, openai_server)
    for _srv in servers:
        _srv.capture_signals = contextlib.nullcontext

    def _handle_shutdown(_sig, _frame):
        for srv in servers:
            if srv.should_exit:
                srv.force_exit = True
            else:
                srv.should_exit = True

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    version = "dev"
    version_file = Path(__file__).resolve().parent.parent.parent / "version.txt"
    if version_file.exists():
        try:
            version = version_file.read_text().strip()
        except OSError as e:
            log.warning("version.txt konnte nicht gelesen werden: %s", e)

    print("=" * 50)
    print(f"Kiron v{version}")
    print("=" * 50)
    print("Dashboard:   http://localhost:8505")
    print("Proxy:       http://localhost:11434 -> Ollama :11435")
    print("OpenAI API:  http://localhost:11440/v1/ (Bearer Auth)")
    print(f"Requests-DB: MariaDB {db_config['host']}:{db_config.get('port', 3306)}/{db_config['database']}")
    print("History-DB:  " + str(data_dir / "metrics.db"))
    print("Strg+C zum Beenden")
    print("=" * 50)

    # Maintenance-Modus wiederherstellen falls beim letzten Shutdown aktiv
    # #517: Optionales Recovery darf Startup nicht blockieren (z.B. iptables-Fehler).
    try:
        await restore_maintenance_state()
    except Exception:
        log.exception("restore_maintenance_state fehlgeschlagen - Startup faehrt ohne Maintenance-Restore fort")

    # #383: Cancellation-Flag fuer Retention-Thread (to_thread nicht unterbrechbar)
    retention_cancel = threading.Event()
    db_work = DBWorkTracker()
    set_db_work_tracker(db_work)

    _, pending_metrics_initial_task = await _run_initial_producer_tick(store)

    # Metrics-Producer, Metrics-Writer und Retention als Background-Tasks
    background_tasks = [
        asyncio.create_task(disk_io_sampler()),
        asyncio.create_task(cpu_sampler()),
        asyncio.create_task(metrics_producer(store, pending_metrics_initial_task)),
        asyncio.create_task(metrics_writer(db, db_work)),
        asyncio.create_task(daily_retention(db, retention_cancel, db_work)),
        asyncio.create_task(lease_release_watcher()),
    ]

    try:
        # VRAM-Lease: shared httpx-Client zu Docling-Lifecycle (#285)
        async with vram_lease.lifespan_client():
            # Alle Server parallel starten
            await asyncio.gather(
                proxy_server.serve(sockets=[server_sockets["proxy"]]),
                dashboard_server.serve(sockets=[server_sockets["dashboard"]]),
                openai_server.serve(sockets=[server_sockets["openai"]]),
            )
    finally:
        retention_cancel.set()
        for task in background_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            log.warning("Shutdown timeout: background tasks did not complete within 30s")
        drained = await db_work.drain(timeout=30.0)
        if not drained:
            # #713: db.close() schliesst Connections cross-thread - laeuft ein
            # SQLite-Worker noch, wuerde close() die laufende INSERT/SELECT/DELETE
            # zerschiessen (sqlite3.ProgrammingError, evtl. WAL-Verlust). Lieber
            # WAL-Checkpoint ueberspringen und beim naechsten Start recovern.
            log.warning("Shutdown timeout: SQLite worker threads still running, skipping close()")
        else:
            # #547: WAL-Checkpoint + Connection-Cleanup vor Process-Exit, damit
            # der naechste Startup nicht erst recovern muss.
            try:
                db.close()
            except Exception:
                log.exception("MetricsDB.close() fehlgeschlagen")
        _close_sockets(server_sockets.values())


if __name__ == "__main__":
    asyncio.run(main())
