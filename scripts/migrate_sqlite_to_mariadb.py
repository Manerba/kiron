#!/usr/bin/env python3
"""Einmalige Migration: SQLite requests.db -> MariaDB ollama_monitor.requests

Migriert bestehende Request-Daten aus SQLite nach MariaDB.
Body-Felder (request_body, response_body) werden leer uebertragen,
da sie in der alten SQLite-DB nicht existierten.

Nutzung:
    python3 scripts/migrate_sqlite_to_mariadb.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import pymysql


def main():
    import os
    data_dir = Path(os.environ.get("KIRON_DATA_DIR", "/usr/lib/kiron/data"))
    sqlite_path = data_dir / "requests.db"
    config_path = data_dir / "db_config.json"

    if not sqlite_path.exists():
        print(f"SQLite-DB nicht gefunden: {sqlite_path}")
        print("Nichts zu migrieren.")
        return

    if not config_path.exists():
        print(f"DB-Config nicht gefunden: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        db_config = json.load(f)

    # SQLite oeffnen
    sqlite_conn = sqlite3.connect(str(sqlite_path), timeout=30)
    sqlite_conn.row_factory = sqlite3.Row

    # MariaDB oeffnen
    try:
        maria_conn = pymysql.connect(
            host=db_config["host"],
            port=db_config.get("port", 3306),
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            charset="utf8mb4",
            autocommit=False,
            connect_timeout=10,
            read_timeout=30,
        )
    except Exception:
        sqlite_conn.close()
        raise

    try:
        # Pruefen ob Ziel-Tabelle existiert — sonst schlaegt executemany mit
        # kryptischem pymysql.err.ProgrammingError fehl.
        with maria_conn.cursor() as check_cursor:
            check_cursor.execute("SHOW TABLES LIKE 'requests'")
            if check_cursor.fetchone() is None:
                print(
                    f"Tabelle 'requests' existiert nicht in Datenbank "
                    f"'{db_config['database']}'.",
                    file=sys.stderr,
                )
                print(
                    "Bitte zuerst kiron-proxy starten, damit "
                    "request_store.init_db() das Schema anlegt.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Spalten die in SQLite existieren
        sqlite_columns = [
            "id", "timestamp", "client_ip", "method", "path", "model",
            "status_code", "duration_ms", "request_size", "response_size",
            "tokens_generated", "is_streaming", "state", "error_message",
        ]

        # Anzahl vorab ermitteln (fuer Logging, ohne alle Rows in RAM zu laden)
        count_cursor = sqlite_conn.execute("SELECT COUNT(*) FROM requests")
        total_rows = count_cursor.fetchone()[0]

        if total_rows == 0:
            print("Keine Daten in SQLite gefunden.")
            return

        print(f"Migriere {total_rows} Requests von SQLite nach MariaDB...")

        # MariaDB-Spalten (inkl. leere Body-Felder)
        maria_columns = sqlite_columns + ["request_body", "response_body"]
        placeholders = ", ".join(["%s"] * len(maria_columns))
        insert_sql = f"INSERT IGNORE INTO requests ({', '.join(maria_columns)}) VALUES ({placeholders})"

        # Streaming-Cursor: fetchmany statt fetchall, damit grosse Alt-DBs nicht vollstaendig in RAM landen.
        # ORDER BY timestamp, id: id als Tiebreaker fuer deterministische Reihenfolge bei identischem timestamp.
        cursor_sqlite = sqlite_conn.execute(
            f"SELECT {', '.join(sqlite_columns)} FROM requests ORDER BY timestamp, id"
        )

        cursor_maria = maria_conn.cursor()
        # MariaDB default max_error_count=64 wuerde Warnings ueber 64 stumm verwerfen
        # - bei batch_size=500 koennte das die Pruefung auf unerwartete Warnings aushebeln.
        cursor_maria.execute("SET SESSION max_error_count = 1000")
        batch_size = 500
        migrated = 0
        skipped = 0
        batch_num = 0

        while True:
            batch = cursor_sqlite.fetchmany(batch_size)
            if not batch:
                break
            batch_num += 1
            values = []
            for row in batch:
                row_values = [row[col] for col in sqlite_columns]
                # MariaDB-VARCHAR/TEXT-Limits konsequent clampen (analog RequestStore._add_request_db).
                # pymysql.executemany kann Batches an max_stmt_length splitten; SHOW WARNINGS
                # zeigt danach nur Warnungen der LETZTEN Query — Truncation-Warnungen frueherer
                # physischer Inserts wuerden unbemerkt committed. Daher pre-insert clampen.
                row_values[0] = (row_values[0] or "")[:16]    # id VARCHAR(16)
                row_values[2] = (row_values[2] or "")[:45]    # client_ip VARCHAR(45)
                row_values[3] = (row_values[3] or "")[:10]    # method VARCHAR(10)
                row_values[4] = (row_values[4] or "")[:512]   # path VARCHAR(512)
                row_values[5] = (row_values[5] or "")[:128]   # model VARCHAR(128)
                row_values[12] = (row_values[12] or "")[:16]  # state VARCHAR(16)
                row_values[13] = (row_values[13] or "")[:8000]  # error_message TEXT
                # Leere Body-Felder hinzufuegen
                row_values.extend(["", ""])
                values.append(row_values)

            try:
                cursor_maria.executemany(insert_sql, values)
                inserted = cursor_maria.rowcount
                # 1062 = Duplikat-Key (erwartet). INSERT IGNORE degradiert sonst auch
                # Daten-/Schema-Fehler (Truncation, NOT-NULL) zu stummen Warnungen.
                cursor_maria.execute("SHOW WARNINGS")
                unexpected = [w for w in cursor_maria.fetchall() if w[1] != 1062]
                if unexpected:
                    raise RuntimeError(f"Unerwartete MariaDB-Warnungen: {unexpected}")
                maria_conn.commit()
            except Exception as e:
                maria_conn.rollback()
                print(f"FEHLER in Batch {batch_num}: {e}", file=sys.stderr)
                raise
            migrated += inserted
            skipped += len(batch) - inserted

        cursor_maria.close()

        print(f"Migration abgeschlossen:")
        print(f"  Migriert: {migrated}")
        print(f"  Uebersprungen (bereits vorhanden): {skipped}")
        print(f"  Gesamt in SQLite: {total_rows}")
    finally:
        sqlite_conn.close()
        maria_conn.close()


if __name__ == "__main__":
    main()
