"""Stdlib-Unittest fuer services/kiron-proxy/history_db.enforce_retention.

Ausfuehren:
    cd services/kiron-proxy && python -m unittest test_history_db

Tests laufen mit echter SQLite in tempfile.
"""

import concurrent.futures
import os
import tempfile
import time
import unittest
from unittest import mock

import history_db


class _CancelAfterNChecks:
    """Pseudo-Event, das nach N is_set()-Aufrufen True liefert."""

    def __init__(self, n_false_returns: int):
        self.remaining = n_false_returns

    def is_set(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return False
        return True


class RetentionTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = history_db.MetricsDB(self.path)
        self.db.init_db()

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suffix)
            except OSError:
                pass

    # -- Helpers ---------------------------------------------------------------

    def _insert_row(self, ts, **metrics):
        cols = ["timestamp"] + self.db.METRIC_COLS
        values = [ts] + [metrics.get(c) for c in self.db.METRIC_COLS]
        placeholders = ",".join(["?"] * len(cols))
        col_names = ",".join(cols)
        conn = self.db._get_conn()
        conn.execute(
            f"INSERT INTO metrics_history ({col_names}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    def _select_rows(self) -> list[tuple[int, float]]:
        conn = self.db._get_conn()
        return conn.execute(
            "SELECT id, timestamp FROM metrics_history ORDER BY id"
        ).fetchall()

    def _select_ids(self) -> set[int]:
        return {row[0] for row in self._select_rows()}

    # -- Tests -----------------------------------------------------------------

    def test_empty_table(self):
        self.assertEqual(self.db.enforce_retention(age_days=90), 0)

    def test_single_old_row_deleted(self):
        old_ts = time.time() - 91 * 86400
        self._insert_row(old_ts, cpu_usage=10.0)

        self.assertEqual(self.db.enforce_retention(age_days=90), 1)
        self.assertEqual(self._select_ids(), set())

    def test_recent_rows_untouched(self):
        for i in range(10):
            self._insert_row(time.time() - i, cpu_usage=float(i))

        self.assertEqual(self.db.enforce_retention(age_days=90), 0)
        self.assertEqual(len(self._select_ids()), 10)

    def test_retention_cutoff_is_strictly_older_than_age_days(self):
        fake_now = 2_000_000_000.0
        cutoff = fake_now - 90 * 86400
        self._insert_row(cutoff - 0.001, cpu_usage=1.0)
        self._insert_row(cutoff, cpu_usage=2.0)
        self._insert_row(cutoff + 0.001, cpu_usage=3.0)

        with mock.patch.object(history_db.time, "time", return_value=fake_now):
            self.assertEqual(self.db.enforce_retention(age_days=90), 1)

        self.assertEqual(self._select_ids(), {2, 3})

    def test_metric_changes_do_not_bypass_retention(self):
        old_ts = time.time() - 91 * 86400
        for i in range(50):
            self._insert_row(old_ts + i, cpu_usage=10.0 * (1.10 ** i))

        self.assertEqual(self.db.enforce_retention(age_days=90), 50)
        self.assertEqual(self._select_ids(), set())

    def test_chunked_retention_deletes_all_old_rows(self):
        old_ts = time.time() - 91 * 86400
        for i in range(2500):
            self._insert_row(old_ts + i, cpu_usage=10.0)

        self.assertEqual(self.db.enforce_retention(age_days=90), 2500)
        self.assertEqual(self._select_ids(), set())

    def test_cancel_before_first_chunk(self):
        old_ts = time.time() - 91 * 86400
        for i in range(500):
            self._insert_row(old_ts + i, cpu_usage=10.0)

        evt = _CancelAfterNChecks(0)
        self.assertEqual(self.db.enforce_retention(age_days=90, cancel_event=evt), 0)
        self.assertEqual(len(self._select_ids()), 500)

    def test_cancel_after_first_chunk_preserves_chunk1_deletes(self):
        old_ts = time.time() - 91 * 86400
        for i in range(2500):
            self._insert_row(old_ts + i, cpu_usage=10.0)

        evt = _CancelAfterNChecks(1)
        self.assertEqual(self.db.enforce_retention(age_days=90, cancel_event=evt), 1000)
        self.assertEqual(len(self._select_ids()), 1500)

    def test_mixed_old_and_recent_rows(self):
        old_ts = time.time() - 91 * 86400
        for i in range(50):
            self._insert_row(old_ts + i, cpu_usage=10.0)
        for i in range(20):
            self._insert_row(time.time() - i, cpu_usage=42.0)

        self.assertEqual(self.db.enforce_retention(age_days=90), 50)
        self.assertEqual(self._select_ids(), set(range(51, 71)))

    def test_repeated_retention_is_idempotent(self):
        old_ts = time.time() - 91 * 86400
        for i in range(100):
            self._insert_row(old_ts + i, cpu_usage=10.0)

        self.assertEqual(self.db.enforce_retention(age_days=90), 100)
        self.assertEqual(self.db.enforce_retention(age_days=90), 0)

    def test_clock_jump_into_future_does_not_wipe_db(self):
        """#940: Wall-Clock-Sprung in die Zukunft (z.B. NTP-Korrektur) darf
        nicht ALLE Daten loeschen. Retention ueberspringt diesen Lauf."""
        real_now = time.time()
        for i in range(10):
            self._insert_row(real_now - i, cpu_usage=10.0)

        # Clock springt 1 Jahr in die Zukunft
        fake_now = real_now + 365 * 86400
        with mock.patch.object(history_db.time, "time", return_value=fake_now):
            deleted = self.db.enforce_retention(age_days=90)

        self.assertEqual(deleted, 0)
        self.assertEqual(len(self._select_ids()), 10)

    def test_query_range_zero_width_downsampling_returns_single_bucket(self):
        ts = 1_800_000_000.123
        for value in (1.0, 2.0, 6.0):
            self._insert_row(ts, cpu_usage=value)

        result = self.db.query_range(
            ["cpu_usage"], from_ts=ts, to_ts=ts, max_points=2
        )

        self.assertEqual(result["timestamps"], [round(ts, 3)])
        self.assertEqual(result["series"]["cpu_usage"], [3.0])

    def test_init_db_migrates_gpu_probe_state_text_column(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            conn = history_db.sqlite3.connect(path)
            try:
                conn.execute("""
                    CREATE TABLE metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

            db = history_db.MetricsDB(path)
            try:
                db.init_db()
                conn = db._get_conn()
                rows = conn.execute("PRAGMA table_info(metrics_history)").fetchall()
                cols = {row[1]: row[2] for row in rows}
                self.assertEqual(cols["gpu_probe_state"].upper(), "TEXT")
            finally:
                db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(path + suffix)
                except OSError:
                    pass

    def test_close_handles_worker_thread_connection(self):
        metrics = {
            "timestamp": time.time(),
            "cpu": {"usage_percent": 12.0, "load_avg_1m": 0.5},
            "memory": {"usage_percent": 34.0, "used_gb": 5.0},
            "gpu": {},
            "disk_io": {},
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.db.insert_metrics, metrics).result(timeout=2.0)

        self.assertGreaterEqual(len(self.db._all_conns), 2)
        self.db.close()

        self.assertEqual(self.db._all_conns, [])
        wal_path = self.path + "-wal"
        if os.path.exists(wal_path):
            self.assertEqual(os.path.getsize(wal_path), 0)
        with self.assertRaises(RuntimeError):
            self.db.insert_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
