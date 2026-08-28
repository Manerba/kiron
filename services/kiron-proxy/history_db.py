"""Historische Metriken - SQLite Verwaltung mit harter Retention."""

import os
import sqlite3
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

# #548: Feste Zone fuer Dashboard-Anzeigen — unabhaengig von der System-TZ.
_TZ_BERLIN = ZoneInfo("Europe/Berlin")


def _format_berlin(unix_ts: float) -> str:
    """Formatiert Unix-Timestamp als Europe/Berlin mit MEZ/MESZ-Suffix (#548)."""
    dt = datetime.fromtimestamp(unix_ts, tz=_TZ_BERLIN)
    suffix = "MESZ" if dt.dst() else "MEZ"
    return dt.strftime(f"%Y-%m-%d %H:%M {suffix}")


class MetricsDB:
    """SQLite-Datenbank fuer historische Metriken."""

    METRIC_COLS = [
        "cpu_usage", "cpu_load_1m", "memory_usage", "memory_used_gb",
        "gpu_util", "vram_usage", "vram_used_mb", "gpu_temp", "gpu_power_w",
        "disk_read_mb_s", "disk_write_mb_s",
    ]
    TEXT_COLS = {
        "gpu_probe_state": "TEXT",
    }

    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        # #547: Cross-Thread-Tracking fuer shutdown-Cleanup. threading.local()
        # exponiert nur die Connection des aufrufenden Threads — fuer den
        # WAL-Checkpoint beim Shutdown brauchen wir Zugriff auf alle.
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._closed = False

    def _get_conn(self):
        """Thread-lokale SQLite-Verbindung mit WAL-Mode."""
        if self._closed:
            raise RuntimeError("MetricsDB closed")
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=10000")
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                raise
            # _closed-Recheck und Append atomar unter dem Lock — sonst kann eine
            # parallel laufende close()-Iteration die frische Connection verpassen
            # und sie wuerde nie geschlossen.
            with self._all_conns_lock:
                if self._closed:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    raise RuntimeError("MetricsDB closed")
                self._local.conn = conn
                self._all_conns.append(conn)
        # #922: _closed-Recheck unter Lock auch fuer den Cache-Pfad. Ohne den
        # Lock kann close() in einem anderen Thread zwischen dem optimistischen
        # Pre-Check oben und dem Return die gecachte Connection schliessen, und
        # der Caller operiert auf einer geschlossenen Connection.
        with self._all_conns_lock:
            if self._closed:
                raise RuntimeError("MetricsDB closed")
            return self._local.conn

    def _reset_conn(self):
        """Thread-lokale Connection schliessen (z.B. nach Exception) - wird beim naechsten _get_conn neu aufgebaut."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            with self._all_conns_lock:
                try:
                    self._all_conns.remove(conn)
                except ValueError:
                    pass
        self._local.conn = None

    def close(self):
        """#547/#645: Shutdown-Cleanup mit sauberem WAL-Checkpoint.

        Connections sind cross-thread schliessbar (`check_same_thread=False`),
        damit der Main-Thread beim Shutdown auch Worker-Thread-Connections
        beenden kann. `_closed` blockt neue Connections danach fail-fast.
        """
        self._closed = True
        with self._all_conns_lock:
            conns = list(self._all_conns)
        failed = []
        for conn in conns:
            try:
                conn.close()
            except Exception:
                failed.append(conn)
        with self._all_conns_lock:
            self._all_conns = failed
        if failed:
            return
        try:
            chk = sqlite3.connect(self.db_path)
            try:
                chk.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                chk.close()
        except Exception:
            pass

    def init_db(self):
        """Tabelle und Index erstellen."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cpu_usage REAL,
                cpu_load_1m REAL,
                memory_usage REAL,
                memory_used_gb REAL,
                gpu_util REAL,
                vram_usage REAL,
                vram_used_mb REAL,
                gpu_temp REAL,
                gpu_power_w REAL,
                disk_read_mb_s REAL,
                disk_write_mb_s REAL,
                gpu_probe_state TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts ON metrics_history(timestamp)"
        )

        # Schema-Migration: fehlende Spalten aus METRIC_COLS nachziehen
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(metrics_history)").fetchall()}
        for col in self.METRIC_COLS:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE metrics_history ADD COLUMN {col} REAL")
        for col, col_type in self.TEXT_COLS.items():
            if col not in existing_cols:
                conn.execute(
                    f"ALTER TABLE metrics_history ADD COLUMN {col} {col_type}"
                )

        conn.commit()

    def insert_metrics(self, metrics_dict):
        """Flacht get_all_metrics() dict ab und fuegt Zeile ein."""
        cpu = metrics_dict.get("cpu") or {}
        mem = metrics_dict.get("memory") or {}
        gpu = metrics_dict.get("gpu") or {}
        disk = metrics_dict.get("disk_io") or {}

        # GPU-Fehler -> NULL
        has_gpu = "error" not in gpu
        gpu_probe_state = None
        if not has_gpu and isinstance(gpu.get("code"), str):
            gpu_probe_state = gpu.get("code")

        row = {
            "timestamp": metrics_dict.get("timestamp", time.time()),
            "cpu_usage": cpu.get("usage_percent"),
            "cpu_load_1m": cpu.get("load_avg_1m"),
            "memory_usage": mem.get("usage_percent"),
            "memory_used_gb": mem.get("used_gb"),
            "gpu_util": gpu.get("gpu_util_percent") if has_gpu else None,
            "vram_usage": gpu.get("vram_usage_percent") if has_gpu else None,
            "vram_used_mb": gpu.get("vram_used_mb") if has_gpu else None,
            "gpu_temp": gpu.get("temperature_c") if has_gpu else None,
            "gpu_power_w": gpu.get("power_draw_w") if has_gpu else None,
            "disk_read_mb_s": disk.get("read_mb_s"),
            "disk_write_mb_s": disk.get("write_mb_s"),
            "gpu_probe_state": gpu_probe_state,
        }

        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        conn = self._get_conn()
        try:
            conn.execute(
                f"INSERT INTO metrics_history ({col_names}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
            conn.commit()
        except sqlite3.Error:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            # Bei OperationalError (Disk-Full, Locked-DB) kann die Connection beschaedigt sein -
            # naechster Call baut via _reset_conn eine frische auf (#261)
            self._reset_conn()
            raise

    def query_range(self, metric_names, from_ts, to_ts, max_points=500):
        """Zeitreihen abfragen mit automatischem Downsampling.

        Wenn mehr als max_points Datenpunkte: GROUP BY Zeitfenster mit AVG.
        Return: {"timestamps": [...], "series": {"cpu_usage": [...], ...}}
        """
        # max_points >=1 erzwingen — Caller mit 0/negativ wuerde sonst ZeroDivisionError im Bucket-Schritt ausloesen.
        max_points = max(1, max_points)
        # Nur gueltige Spalten zulassen, Duplikate entfernen (series-Dict hat unique Keys).
        valid = list(dict.fromkeys(m for m in metric_names if m in self.METRIC_COLS))
        if not valid:
            return {"timestamps": [], "series": {}}

        conn = self._get_conn()

        try:
            # #908: COUNT und Datenabfrage in einer Read-Tx, damit beide Statements
            # denselben WAL-Snapshot lesen. Sonst kann ein nebenlaeufiger INSERT
            # (metrics_writer 10s-Kadenz) oder enforce_retention zwischen den
            # Statements committen, sodass Downsampling-Entscheidung und gelesene
            # Datenmenge auseinanderlaufen.
            conn.execute("BEGIN")
            try:
                # Anzahl Datenpunkte im Bereich zaehlen
                count = conn.execute(
                    "SELECT COUNT(*) FROM metrics_history WHERE timestamp >= ? AND timestamp <= ?",
                    (from_ts, to_ts),
                ).fetchone()[0]

                if count == 0:
                    return {"timestamps": [], "series": {m: [] for m in valid}}

                if count <= max_points:
                    # Volle Aufloesung
                    select_cols = ", ".join(valid)
                    rows = conn.execute(
                        f"SELECT timestamp, {select_cols} FROM metrics_history "
                        f"WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                        (from_ts, to_ts),
                    ).fetchall()
                else:
                    # Downsampling: Zeitfenster berechnen
                    time_range = to_ts - from_ts
                    if time_range < 0:
                        return {"timestamps": [], "series": {m: [] for m in valid}}
                    if time_range == 0:
                        avg_cols = ", ".join([f"AVG({m})" for m in valid])
                        rows = conn.execute(
                            f"SELECT AVG(timestamp), {avg_cols} FROM metrics_history "
                            f"WHERE timestamp >= ? AND timestamp <= ?",
                            (from_ts, to_ts),
                        ).fetchall()
                    else:
                        # Bucket-Index darf [0, max_points-1] sein. Teilen durch max_points
                        # liefert fuer timestamp == to_ts den Index max_points (1 zuviel),
                        # daher MIN(..., max_points-1) als Obergrenze.
                        bucket_size = time_range / max_points
                        last_bucket = max_points - 1

                        avg_cols = ", ".join([f"AVG({m})" for m in valid])
                        rows = conn.execute(
                            f"SELECT AVG(timestamp), {avg_cols} FROM metrics_history "
                            f"WHERE timestamp >= ? AND timestamp <= ? "
                            f"GROUP BY MIN(CAST((timestamp - ?) / ? AS INTEGER), ?) "
                            f"ORDER BY AVG(timestamp)",
                            (from_ts, to_ts, from_ts, bucket_size, last_bucket),
                        ).fetchall()

                timestamps = []
                series = {m: [] for m in valid}

                for row in rows:
                    ts = row[0]
                    if ts is None:
                        continue
                    timestamps.append(round(ts, 3))
                    for i, m in enumerate(valid):
                        val = row[i + 1]
                        series[m].append(round(val, 2) if val is not None else None)

                return {"timestamps": timestamps, "series": series}
            finally:
                # Read-Tx schliessen — auch fuer early returns. Bei Fehler im
                # Body bleibt die Tx u.U. offen und COMMIT scheitert; das
                # uebernimmt dann der aeussere except-Handler via rollback.
                try:
                    conn.execute("COMMIT")
                except sqlite3.Error:
                    pass
        except sqlite3.Error:
            # Analog zu insert_metrics/enforce_retention (#261): Bei Lock/Disk-Fehler
            # Transaktion schliessen und Connection resetten, damit Folge-Reads nicht
            # in offener Tx haengen.
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            self._reset_conn()
            raise

    def enforce_retention(self, age_days=90, cancel_event=None):
        """Loescht Messpunkte aelter als age_days in bounded Chunks.

        #775: Die dokumentierte 90-Tage-Retention ist hart. Alte signifikante
        Punkte duerfen nicht als Dead-Band-Anker dauerhaft erhalten bleiben,
        sonst waechst metrics.db bei schwankenden Metriken unbegrenzt.

        cancel_event: Optional threading.Event fuer kooperativen Abbruch (#383) -
        asyncio.to_thread ist nicht unterbrechbar, daher prueft die Schleife
        periodisch dieses Flag, damit Shutdown nicht auf grosse DB-Laeufe wartet.
        Cancel wirkt zwischen Chunks (Granularitaet ~CHUNK_SIZE Rows).

        Returns: Anzahl geloeschter Zeilen (bei Cancel: bis dahin committete).
        """
        cutoff = time.time() - (age_days * 86400)
        CHUNK_SIZE = 1000

        conn = self._get_conn()

        try:
            # #940: Sanity-Check gegen Wall-Clock-Spruenge (Boot vor NTP-Sync,
            # NTP-Korrektur in die Zukunft). Wenn der neueste Timestamp mehr als
            # eine Retention-Periode vor cutoff liegt, ist die Uhr vermutlich
            # kaputt — retention ueberspringen statt ALLE Daten zu loeschen. Bei
            # laufendem Service ist newest ~ now, der Gap also negativ; nur ein
            # massiver Clock-Skew ueberschreitet den Threshold.
            newest = conn.execute(
                "SELECT MAX(timestamp) FROM metrics_history"
            ).fetchone()[0]
            if newest is not None and cutoff - newest > age_days * 86400:
                return 0

            total_deleted = 0

            while True:
                # Kooperativer Abbruch zwischen Chunks (#383)
                if cancel_event is not None and cancel_event.is_set():
                    return total_deleted

                rows = conn.execute(
                    "SELECT id FROM metrics_history WHERE timestamp < ? "
                    "ORDER BY timestamp, id LIMIT ?",
                    (cutoff, CHUNK_SIZE),
                ).fetchall()
                if not rows:
                    return total_deleted

                conn.executemany(
                    "DELETE FROM metrics_history WHERE id = ?",
                    rows,
                )
                conn.commit()
                total_deleted += len(rows)
        except sqlite3.Error:
            # Analog zu insert_metrics: Bei Lock/Disk-Fehler Transaktion schliessen
            # und Connection resetten, damit Folge-Calls nicht in offener Tx haengen.
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            self._reset_conn()
            raise

    def get_db_stats(self):
        """Anzahl Datenpunkte, Zeitraum, DB-Groesse."""
        conn = self._get_conn()

        try:
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM metrics_history"
            ).fetchone()
        except sqlite3.Error:
            # Analog zu insert_metrics/query_range/enforce_retention (#261):
            # Bei Lock/Disk-Fehler Connection resetten, damit Folge-Aufrufe
            # nicht in einer offenen Tx haengen.
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            self._reset_conn()
            raise

        total_rows = row[0] or 0
        oldest = row[1]
        newest = row[2]

        # DB-Dateigroesse inkl. WAL+SHM (im WAL-Modus liegen aktuelle Daten dort)
        db_size_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                db_size_bytes += os.path.getsize(self.db_path + suffix)
            except OSError:
                pass
        db_size_mb = round(db_size_bytes / (1024 * 1024), 2)

        result = {
            "total_rows": total_rows,
            "db_size_mb": db_size_mb,
        }

        if oldest is not None:
            result["oldest"] = _format_berlin(oldest)
        if newest is not None:
            result["newest"] = _format_berlin(newest)

        return result
