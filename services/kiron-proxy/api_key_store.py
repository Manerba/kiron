"""API Key Store - MariaDB-persistenter Speicher fuer API-Key-Verwaltung."""

import hashlib
import logging
import secrets
import threading
import time
import uuid

import pymysql


class LastActiveApiKeyError(RuntimeError):
    """Raised when an operation would remove the last active API key."""


class ApiKeyStore:
    """MariaDB-persistenter API-Key-Speicher fuer OpenAI-kompatible API."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self._conn = None
        # validate_key laeuft via asyncio.to_thread in Worker-Threads,
        # waehrend Dashboard-Handler aus dem Event-Loop kommen — pymysql ist
        # nicht thread-safe, also serialisieren wir alle DB-Zugriffe.
        self._lock = threading.Lock()

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
                # Alte Connection explizit schliessen — sonst FD-Leak bei wiederholtem
                # DB-Flapping (pymysql cleant nicht zuverlaessig nach failed reconnect).
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
        """Tabelle und Indexe erstellen."""
        with self._lock:
            self._ensure_connection()
            with self._conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id VARCHAR(36) PRIMARY KEY,
                        name VARCHAR(128) NOT NULL,
                        key_hash VARCHAR(64) NOT NULL UNIQUE,
                        key_prefix VARCHAR(20) NOT NULL,
                        created_at DOUBLE NOT NULL,
                        last_used_at DOUBLE DEFAULT NULL,
                        is_active TINYINT DEFAULT 1,
                        INDEX idx_ak_hash (key_hash),
                        INDEX idx_ak_active (is_active)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                self._conn.commit()

                # Migration: key_prefix erweitern falls zu klein
                try:
                    cursor.execute("ALTER TABLE api_keys MODIFY COLUMN key_prefix VARCHAR(20) NOT NULL")
                    self._conn.commit()
                except pymysql.err.Error:
                    logging.getLogger(__name__).exception(
                        "ALTER TABLE api_keys MODIFY key_prefix failed — Schema bleibt im alten Zustand"
                    )
                    self._conn.rollback()
                    raise

    def _hash_key(self, raw_key: str) -> str:
        """SHA-256 Hash eines API-Keys berechnen."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def create_key(self, name: str) -> dict:
        """Neuen API-Key generieren und speichern. Gibt Plaintext einmalig zurueck."""
        key_id = str(uuid.uuid4())
        raw_key = "sk-" + secrets.token_hex(24)
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:12] + "..."
        now = time.time()

        with self._lock:
            self._ensure_connection()
            try:
                with self._conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO api_keys (id, name, key_hash, key_prefix, created_at, is_active) "
                        "VALUES (%s, %s, %s, %s, %s, 1)",
                        (key_id, name, key_hash, key_prefix, now),
                    )
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

        return {
            "id": key_id,
            "name": name,
            "key": raw_key,
            "key_prefix": key_prefix,
            "created_at": now,
        }

    def list_keys(self) -> list:
        """Alle Keys auflisten (ohne key_hash)."""
        with self._lock:
            self._ensure_connection()
            try:
                with self._conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, name, key_prefix, created_at, last_used_at, is_active "
                        "FROM api_keys ORDER BY created_at DESC"
                    )
                    rows = cursor.fetchall()
                    # autocommit=False: SELECT oeffnet eine Read-Transaktion, die ohne
                    # commit/rollback einen veralteten MVCC-Snapshot fuer nachfolgende
                    # SELECTs (insb. validate_key) auf der geteilten Connection halten wuerde.
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

        return [
            {
                "id": r[0],
                "name": r[1],
                "key_prefix": r[2],
                "created_at": r[3],
                "last_used_at": r[4],
                "is_active": bool(r[5]),
            }
            for r in rows
        ]

    def validate_key(self, raw_key: str) -> dict | None:
        """Key validieren. Gibt Key-Dict zurueck oder None bei ungueltigem Key."""
        if not isinstance(raw_key, str) or not raw_key:
            return None
        key_hash = self._hash_key(raw_key)

        with self._lock:
            self._ensure_connection()
            try:
                with self._conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, name, key_prefix FROM api_keys WHERE key_hash = %s AND is_active = 1",
                        (key_hash,),
                    )
                    row = cursor.fetchone()

                    if not row:
                        self._conn.commit()
                        return None

                    key_id, name, key_prefix = row

                    # last_used_at aktualisieren — TOCTOU-Schutz fuer multi-worker:
                    # Key koennte zwischen SELECT und UPDATE durch parallelen Worker
                    # geloescht ODER deaktiviert worden sein (process-lokaler Lock
                    # schuetzt nicht cross-process). is_active=1 im WHERE bricht auch
                    # die Deaktivierung ab — rowcount==0 -> Key ist weg/inaktiv,
                    # nicht authentisieren.
                    cursor.execute(
                        "UPDATE api_keys SET last_used_at = %s WHERE id = %s AND is_active = 1",
                        (time.time(), key_id),
                    )
                    if cursor.rowcount == 0:
                        self._conn.rollback()
                        return None
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

        return {
            "id": key_id,
            "name": name,
            "key_prefix": key_prefix,
        }

    def _has_other_active_key_locked(self, cursor, key_id: str) -> bool:
        # FOR UPDATE: ohne Row-Lock waere der Check unter MVCC nur ein Snapshot —
        # parallele Worker koennten beide "anderer aktiver Key existiert" sehen
        # und beide deaktivieren -> 0 aktive Keys.
        cursor.execute(
            "SELECT 1 FROM api_keys WHERE is_active = 1 AND id != %s LIMIT 1 FOR UPDATE",
            (key_id,),
        )
        return cursor.fetchone() is not None

    def set_active(self, key_id: str, active: bool, *, force: bool = False) -> bool:
        """Key aktivieren/deaktivieren."""
        with self._lock:
            self._ensure_connection()
            try:
                with self._conn.cursor() as cursor:
                    # Existenz pruefen, da rowcount bei No-op-UPDATE (Wert bereits gesetzt) 0 ist
                    # und sich damit nicht von "Key nicht gefunden" unterscheiden laesst.
                    # FOR UPDATE haelt den Row-Lock bis COMMIT, damit der "letzter aktiver Key"-
                    # Check nicht zwischen SELECT und UPDATE durch parallele Worker raceit.
                    cursor.execute(
                        "SELECT is_active FROM api_keys WHERE id = %s FOR UPDATE", (key_id,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        self._conn.commit()
                        return False
                    current_active = bool(row[0])
                    if current_active and not active and not force:
                        if not self._has_other_active_key_locked(cursor, key_id):
                            self._conn.commit()
                            raise LastActiveApiKeyError(
                                "Letzter aktiver API-Key kann nicht deaktiviert werden"
                            )
                    cursor.execute(
                        "UPDATE api_keys SET is_active = %s WHERE id = %s",
                        (int(active), key_id),
                    )
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise
        return True

    def delete_key(self, key_id: str, *, force: bool = False) -> bool:
        """Key loeschen."""
        with self._lock:
            self._ensure_connection()
            try:
                with self._conn.cursor() as cursor:
                    # FOR UPDATE: siehe set_active — ohne Row-Lock waere der
                    # "letzter aktiver Key"-Check unter MVCC nur ein Snapshot.
                    cursor.execute(
                        "SELECT is_active FROM api_keys WHERE id = %s FOR UPDATE", (key_id,)
                    )
                    row = cursor.fetchone()
                    if row is None:
                        self._conn.commit()
                        return False
                    current_active = bool(row[0])
                    if current_active and not force:
                        if not self._has_other_active_key_locked(cursor, key_id):
                            self._conn.commit()
                            raise LastActiveApiKeyError(
                                "Letzter aktiver API-Key kann nicht geloescht werden"
                            )
                    cursor.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))
                    affected = cursor.rowcount
                    self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise
        return affected > 0
