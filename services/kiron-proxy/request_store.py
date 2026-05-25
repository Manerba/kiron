"""Request Store - MariaDB-persistenter Speicher fuer Ollama-Requests mit WebSocket-Broadcast."""

import asyncio
import json
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import pymysql


@dataclass
class RequestRecord:
    """Ein einzelner Request an Ollama."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    client_ip: str = ""
    method: str = "GET"
    path: str = ""
    model: str = ""
    status_code: int = 0
    duration_ms: float = 0.0
    request_size: int = 0
    response_size: int = 0
    tokens_generated: int = 0
    is_streaming: bool = False
    state: str = "active"  # active, completed, warning, error
    error_message: str = ""
    request_body: str = ""
    response_body: str = ""

    def to_dict(self):
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # Body-Felder nicht im Broadcast/Listen-View
        d.pop("request_body", None)
        d.pop("response_body", None)
        return d

    def to_dict_full(self):
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return d


class RequestStoreReadError(RuntimeError):
    """Raised when a RequestStore read fails because of a DB error.

    Signals to API handlers that the result is not "no data" but a backend
    failure — handlers should map this to 503, not 200/404.
    """


# Spaltenreihenfolge in der DB (ohne Body fuer Listen-Queries)
_COLUMNS_SUMMARY = [
    "id", "timestamp", "client_ip", "method", "path", "model",
    "status_code", "duration_ms", "request_size", "response_size",
    "tokens_generated", "is_streaming", "state", "error_message",
]

# Alle Spalten inkl. Body (fuer Inserts und Detail-Queries)
_COLUMNS_ALL = _COLUMNS_SUMMARY + ["request_body", "response_body"]


class RequestStore:
    """MariaDB-persistenter Request-Speicher mit WebSocket-Broadcast.

    Retention: Maximal MAX_RETAINED_REQUESTS Eintraege werden behalten.
    Aeltere Requests werden automatisch geloescht (Pruefung alle 50 Inserts).
    """

    MAX_RETAINED_REQUESTS = 1000

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self._active = {}  # id -> RequestRecord (laufende Requests)
        self._ws_clients = set()  # asyncio.Queue pro WebSocket-Client
        self._lock = asyncio.Lock()
        # Thread-side Lock fuer self._conn: schuetzt die geteilte PyMySQL-Verbindung,
        # auch wenn der asyncio.Lock durch CancelledError vorzeitig freigegeben wird
        # waehrend ein asyncio.to_thread-Worker noch laeuft (#866).
        self._db_lock = threading.Lock()
        self._conn = None
        self._insert_counter = 0  # Zaehler fuer Retention-Pruefung

    def _ensure_connection(self):
        """Verbindung sicherstellen, bei Bedarf reconnecten."""
        if self._conn is None:
            self._conn = pymysql.connect(
                host=self.db_config["host"],
                port=self.db_config.get("port", 3306),
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"],
                charset="utf8mb4",
                autocommit=False,
                connect_timeout=5,
                read_timeout=10,
                write_timeout=10,
            )
        else:
            try:
                self._conn.ping(reconnect=True)
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = pymysql.connect(
                    host=self.db_config["host"],
                    port=self.db_config.get("port", 3306),
                    user=self.db_config["user"],
                    password=self.db_config["password"],
                    database=self.db_config["database"],
                    charset="utf8mb4",
                    autocommit=False,
                    connect_timeout=5,
                    read_timeout=10,
                    write_timeout=10,
                )

    def init_db(self):
        """Tabelle und Indexe erstellen. Abgebrochene Requests bereinigen."""
        self._ensure_connection()
        try:
            with closing(self._conn.cursor()) as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS requests (
                        id VARCHAR(16) PRIMARY KEY,
                        timestamp DOUBLE NOT NULL,
                        client_ip VARCHAR(45) DEFAULT '',
                        method VARCHAR(10) DEFAULT 'GET',
                        path VARCHAR(512) DEFAULT '',
                        model VARCHAR(128) DEFAULT '',
                        status_code INT DEFAULT 0,
                        duration_ms DOUBLE DEFAULT 0.0,
                        request_size INT DEFAULT 0,
                        response_size INT DEFAULT 0,
                        tokens_generated INT DEFAULT 0,
                        is_streaming TINYINT DEFAULT 0,
                        state VARCHAR(16) DEFAULT 'active',
                        error_message TEXT DEFAULT '',
                        request_body MEDIUMTEXT,
                        response_body MEDIUMTEXT,
                        INDEX idx_req_ts (timestamp),
                        INDEX idx_req_state (state)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                self._conn.commit()

                # Body-Spalten nachrüsten falls Tabelle schon existiert (Migration)
                # MySQL-Fehlercode 1060 = "Duplicate column name" → erwartet, alles andere bricht ab.
                try:
                    cursor.execute("ALTER TABLE requests ADD COLUMN request_body MEDIUMTEXT AFTER error_message")
                    self._conn.commit()
                except pymysql.err.OperationalError as e:
                    if not e.args or e.args[0] != 1060:
                        raise

                try:
                    cursor.execute("ALTER TABLE requests ADD COLUMN response_body MEDIUMTEXT AFTER request_body")
                    self._conn.commit()
                except pymysql.err.OperationalError as e:
                    if not e.args or e.args[0] != 1060:
                        raise

                # Migration: id-Spalte von VARCHAR(8) auf VARCHAR(16) erweitern (idempotent)
                cursor.execute("ALTER TABLE requests MODIFY COLUMN id VARCHAR(16)")
                self._conn.commit()
        finally:
            # Stale-Active-Cleanup in eigenem Cursor-Kontext: läuft auch wenn eine
            # Migration mit unerwartetem Fehler propagiert (#1043). Best-effort —
            # bei DB-Fehler hier nur loggen, damit init_db die Migrations-Exception
            # weiter durchreichen kann.
            try:
                with closing(self._conn.cursor()) as cursor:
                    cursor.execute(
                        "UPDATE requests SET state = 'error', error_message = 'Abgebrochen (Neustart)' "
                        "WHERE state = 'active'"
                    )
                    self._conn.commit()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Stale-active cleanup Fehler: %s", e)

        # Retention: Alte Requests beim Start bereinigen
        self._cleanup_old_requests()

    def _cleanup_old_requests(self):
        """Loescht alle Requests ausser den neuesten MAX_RETAINED_REQUESTS.

        Wird alle 50 Inserts aufgerufen. Loescht nur abgeschlossene Requests
        (state != 'active'), um laufende Requests nicht zu verlieren.
        """
        try:
            with self._db_lock:
                self._ensure_connection()
                with closing(self._conn.cursor()) as cursor:
                    # Behalte die neuesten MAX_RETAINED_REQUESTS abgeschlossenen Eintraege,
                    # loesche nur nicht-aktive Requests, die nicht in dieser Menge sind.
                    # Subquery filtert ebenfalls auf state != 'active', damit ein voller
                    # Active-Bestand nicht alle abgeschlossenen Records aus dem Keep-Set draengt.
                    cursor.execute(
                        "DELETE FROM requests WHERE state != 'active' AND id NOT IN ("
                        "  SELECT id FROM ("
                        "    SELECT id FROM requests WHERE state != 'active' "
                        "    ORDER BY timestamp DESC, id DESC LIMIT %s"
                        "  ) AS keep"
                        ")",
                        (self.MAX_RETAINED_REQUESTS,),
                    )
                    deleted = cursor.rowcount
                    self._conn.commit()
            if deleted > 0:
                import logging
                logging.getLogger(__name__).info(
                    "Retention: %d alte Requests geloescht (behalte %d)",
                    deleted, self.MAX_RETAINED_REQUESTS,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Retention-Cleanup Fehler: %s", e)

    def _row_to_dict(self, row, columns=None):
        """DB-Zeile in Dict konvertieren."""
        cols = columns or _COLUMNS_SUMMARY
        d = dict(zip(cols, row))
        d["is_streaming"] = bool(d["is_streaming"])
        ts = d.get("timestamp")
        d["timestamp_iso"] = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if ts is not None else ""
        )
        return d

    def _add_request_db(self, record: RequestRecord):
        with self._db_lock:
            self._ensure_connection()
            with closing(self._conn.cursor()) as cursor:
                # VARCHAR/TEXT-Limits erzwingen, damit lange Werte im Strict-Mode nicht den Insert kippen
                path = (record.path or "")[:512]
                model = (record.model or "")[:128]
                error_message = (record.error_message or "")[:8000]
                try:
                    cursor.execute(
                        "INSERT INTO requests ({cols}) VALUES ({ph})".format(
                            cols=", ".join(_COLUMNS_ALL),
                            ph=", ".join(["%s"] * len(_COLUMNS_ALL)),
                        ),
                        [
                            record.id, record.timestamp, record.client_ip, record.method,
                            path, model, record.status_code, record.duration_ms,
                            record.request_size, record.response_size, record.tokens_generated,
                            int(record.is_streaming), record.state, error_message,
                            record.request_body, record.response_body,
                        ],
                    )
                    self._conn.commit()
                except Exception:
                    # Verhindert, dass die geteilte Verbindung mit offener Transaktion
                    # zurueckbleibt — sonst sehen folgende Reads einen halbfertigen Snapshot.
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass
                    raise

    async def add_request(self, record: RequestRecord):
        """Neuen Request in MariaDB speichern und an WebSocket-Clients broadcasten."""
        async with self._lock:
            await asyncio.to_thread(self._add_request_db, record)
            if record.state == "active":
                self._active[record.id] = record
            self._insert_counter += 1
            if self._insert_counter >= 50:
                self._insert_counter = 0
                await asyncio.to_thread(self._cleanup_old_requests)
        await self._broadcast({"type": "new_request", "data": record.to_dict()})

    def _update_request_db(self, set_parts, values, request_id):
        with self._db_lock:
            self._ensure_connection()
            with closing(self._conn.cursor()) as cursor:
                if set_parts:
                    cursor.execute(
                        f"UPDATE requests SET {', '.join(set_parts)} WHERE id = %s",
                        values,
                    )
                    self._conn.commit()
                cursor.execute(
                    "SELECT {cols} FROM requests WHERE id = %s".format(
                        cols=", ".join(_COLUMNS_SUMMARY)
                    ),
                    (request_id,),
                )
                row = cursor.fetchone()
        return row

    async def update_request(self, request_id: str, **kwargs):
        """Request in MariaDB aktualisieren (z.B. status, duration, tokens)."""
        async with self._lock:
            # SET-Klausel bauen
            set_parts = []
            values = []
            for key, value in kwargs.items():
                if key in _COLUMNS_ALL and key != "id":
                    if key == "is_streaming":
                        value = int(value)
                    set_parts.append(f"{key} = %s")
                    values.append(value)

            if set_parts:
                values.append(request_id)

            row = None
            try:
                row = await asyncio.to_thread(
                    self._update_request_db, set_parts, values, request_id
                )
            finally:
                # #680: Cleanup auch bei DB-Fehler, sonst leakt _active bei DB-Hickups.
                # init_db() raeumt DB-seitige stale 'active' beim naechsten Start.
                if request_id in self._active:
                    rec = self._active[request_id]
                    for key, value in kwargs.items():
                        if hasattr(rec, key):
                            setattr(rec, key, value)
                    if rec.state != "active":
                        del self._active[request_id]

        if row:
            await self._broadcast({"type": "update_request", "data": self._row_to_dict(row)})

    async def get_recent(self, limit=100, offset=0, ip_filter=None, model_filter=None, status_filter=None, active_only=False):
        """Requests mit Filtern und Pagination aus MariaDB. Gibt (Seite, Gesamtanzahl) zurueck."""
        async with self._lock:
            return await asyncio.to_thread(
                self._get_recent_sync, limit, offset, ip_filter, model_filter, status_filter, active_only
            )

    def _get_recent_sync(self, limit, offset, ip_filter, model_filter, status_filter, active_only):
        with self._db_lock:
            try:
                self._ensure_connection()
            except pymysql.err.Error as e:
                import logging
                logging.getLogger(__name__).warning("get_recent: DB nicht erreichbar")
                raise RequestStoreReadError("DB nicht erreichbar") from e
            where = []
            params = []

            if active_only:
                where.append("state = 'active'")
            if ip_filter:
                escaped = ip_filter.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where.append("client_ip LIKE %s ESCAPE '\\\\'")
                params.append(f"%{escaped}%")
            if model_filter:
                where.append("model = %s")
                params.append(model_filter)
            if status_filter:
                if status_filter == "2xx":
                    where.append("status_code >= 200 AND status_code < 300")
                elif status_filter == "4xx":
                    where.append("status_code >= 400 AND status_code < 500")
                elif status_filter == "5xx":
                    where.append("status_code >= 500 AND status_code < 600")

            where_clause = (" WHERE " + " AND ".join(where)) if where else ""
            cols = ", ".join(_COLUMNS_SUMMARY)

            try:
                with closing(self._conn.cursor()) as cursor:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM requests{where_clause}", params
                    )
                    total = cursor.fetchone()[0]

                    cursor.execute(
                        f"SELECT {cols} FROM requests{where_clause} ORDER BY timestamp DESC, id DESC LIMIT %s OFFSET %s",
                        params + [limit, offset],
                    )
                    rows = cursor.fetchall()
            except pymysql.err.Error as e:
                import logging
                logging.getLogger(__name__).warning("get_recent DB-Fehler: %s", e)
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise RequestStoreReadError("get_recent DB-Fehler") from e

        return [self._row_to_dict(r) for r in rows], total

    async def get_request_detail(self, request_id: str):
        """Einzelnen Request mit vollem Body laden."""
        async with self._lock:
            return await asyncio.to_thread(self._get_request_detail_sync, request_id)

    def _get_request_detail_sync(self, request_id: str):
        with self._db_lock:
            try:
                self._ensure_connection()
            except pymysql.err.Error as e:
                import logging
                logging.getLogger(__name__).warning("get_request_detail: DB nicht erreichbar")
                raise RequestStoreReadError("DB nicht erreichbar") from e
            try:
                with closing(self._conn.cursor()) as cursor:
                    cursor.execute(
                        "SELECT {cols} FROM requests WHERE id = %s".format(
                            cols=", ".join(_COLUMNS_ALL)
                        ),
                        (request_id,),
                    )
                    row = cursor.fetchone()
            except pymysql.err.Error as e:
                import logging
                logging.getLogger(__name__).warning("get_request_detail DB-Fehler: %s", e)
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise RequestStoreReadError("get_request_detail DB-Fehler") from e
        if row:
            return self._row_to_dict(row, columns=_COLUMNS_ALL)
        return None

    def get_active(self):
        """Alle aktuell laufenden Requests."""
        return [rec.to_dict() for rec in list(self._active.values())]

    async def get_summary(self):
        """Zusammenfassung fuer Dashboard-Tab (SQL-Aggregate)."""
        async with self._lock:
            return await asyncio.to_thread(self._get_summary_sync)

    def _get_summary_sync(self):
        empty = {
            "requests_today": 0,
            "active_now": len(self._active),
            "avg_duration_ms": 0,
            "error_rate": 0.0,
            "top_clients": [],
            "top_models": [],
        }
        with self._db_lock:
            try:
                self._ensure_connection()
            except pymysql.err.OperationalError:
                return empty
            now = time.time()
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

            try:
                with closing(self._conn.cursor()) as cursor:
                    cursor.execute(
                        "SELECT COUNT(*), SUM(CASE WHEN status_code >= 400 OR state = 'error' THEN 1 ELSE 0 END) "
                        "FROM requests WHERE timestamp >= %s",
                        (today_start,),
                    )
                    row = cursor.fetchone()
                    total_today = row[0] or 0
                    total_errors = row[1] or 0

                    cursor.execute(
                        "SELECT AVG(duration_ms) FROM requests "
                        "WHERE timestamp >= %s "
                        "AND state IN ('completed', 'warning') "
                        "AND duration_ms > 0",
                        (today_start,),
                    )
                    avg_row = cursor.fetchone()
                    avg_duration = avg_row[0] or 0

                    error_rate = (total_errors / total_today * 100) if total_today > 0 else 0

                    cursor.execute(
                        "SELECT client_ip, COUNT(*) as cnt FROM requests "
                        "WHERE timestamp >= %s AND client_ip != '' "
                        "GROUP BY client_ip ORDER BY cnt DESC LIMIT 10",
                        (today_start,),
                    )
                    top_clients_rows = cursor.fetchall()

                    cursor.execute(
                        "SELECT model, COUNT(*) as cnt, "
                        "AVG(CASE WHEN state IN ('completed', 'warning') AND duration_ms > 0 "
                        "THEN duration_ms END) as avg_dur "
                        "FROM requests WHERE timestamp >= %s AND model != '' "
                        "GROUP BY model ORDER BY cnt DESC LIMIT 10",
                        (today_start,),
                    )
                    top_models_rows = cursor.fetchall()
            except pymysql.err.Error as e:
                import logging
                logging.getLogger(__name__).warning("get_summary DB-Fehler: %s", e)
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                return empty

        top_clients = [{"ip": r[0], "count": r[1]} for r in top_clients_rows]
        top_models = [
            {"model": r[0], "count": r[1], "avg_duration_ms": round(r[2] or 0)}
            for r in top_models_rows
        ]

        return {
            "requests_today": total_today,
            "active_now": len(self._active),
            "avg_duration_ms": round(avg_duration),
            "error_rate": round(error_rate, 1),
            "top_clients": top_clients,
            "top_models": top_models,
        }

    def register_ws_client(self):
        """Neuen WebSocket-Client registrieren. Gibt Queue zurueck."""
        queue = asyncio.Queue(maxsize=100)
        self._ws_clients.add(queue)
        return queue

    def _evict_ws_client(self, queue):
        """#559: Client aus Set entfernen UND Consumer aufwecken via None-Sentinel.

        Ohne Sentinel haengt die WebSocket-Read-Coroutine (app.py forward_request_events)
        weiter in `await queue.get()` auf einer orphan Queue. Wenn die Queue voll ist,
        wird ein Element gedroppt um Platz fuer das Sentinel zu schaffen.
        """
        self._ws_clients.discard(queue)
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.put_nowait(None)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def unregister_ws_client(self, queue):
        """WebSocket-Client abmelden."""
        self._evict_ws_client(queue)

    async def _broadcast(self, message: dict):
        """Nachricht an alle WebSocket-Clients senden."""
        try:
            data = json.dumps(message, default=str)
        except (TypeError, ValueError) as e:
            import logging
            logging.getLogger(__name__).warning("Broadcast skipped: message not serializable (%s)", e)
            return
        dead_clients = set()
        for queue in list(self._ws_clients):
            try:
                queue.put_nowait(data)
            except Exception:
                dead_clients.add(queue)
        for client in dead_clients:
            self._evict_ws_client(client)
