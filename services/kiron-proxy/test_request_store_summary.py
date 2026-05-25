"""Tests fuer RequestStore-Summary-Aggregate."""

import os
import sys
import time
import unittest
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from request_store import RequestStore  # noqa: E402


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.result = None

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        since = params[0] if params else 0
        rows = [row for row in self.rows if row["timestamp"] >= since]

        if normalized.startswith("SELECT COUNT(*), SUM"):
            self.result = (
                len(rows),
                sum(1 for row in rows if row["status_code"] >= 400),
            )
            return

        if normalized.startswith("SELECT AVG(duration_ms)"):
            states = self._success_states_from_sql(normalized)
            durations = [
                row["duration_ms"]
                for row in rows
                if row["state"] in states and row["duration_ms"] > 0
            ]
            self.result = (self._avg(durations),)
            return

        if normalized.startswith("SELECT client_ip"):
            counts = Counter(row["client_ip"] for row in rows if row["client_ip"])
            self.result = counts.most_common(10)
            return

        if normalized.startswith("SELECT model, COUNT(*)"):
            states = self._success_states_from_sql(normalized)
            grouped = defaultdict(list)
            for row in rows:
                if row["model"]:
                    grouped[row["model"]].append(row)
            model_rows = []
            for model, model_group in grouped.items():
                durations = [
                    row["duration_ms"]
                    for row in model_group
                    if row["state"] in states and row["duration_ms"] > 0
                ]
                model_rows.append((model, len(model_group), self._avg(durations)))
            self.result = sorted(model_rows, key=lambda row: row[1], reverse=True)[:10]
            return

        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result

    def close(self):
        pass

    @staticmethod
    def _avg(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def _success_states_from_sql(sql):
        if "state IN ('completed', 'warning')" in sql:
            return {"completed", "warning"}
        return {"completed"}


class _Connection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _Cursor(self.rows)

    def rollback(self):
        pass


class RequestStoreSummaryTests(unittest.TestCase):
    def test_warning_counts_as_success_duration_not_error_rate(self):
        now = time.time()
        rows = [
            {
                "timestamp": now,
                "client_ip": "192.0.2.10",
                "model": "qwen3:8b",
                "status_code": 200,
                "duration_ms": 100,
                "state": "completed",
            },
            {
                "timestamp": now,
                "client_ip": "192.0.2.10",
                "model": "qwen3:8b",
                "status_code": 200,
                "duration_ms": 300,
                "state": "warning",
            },
            {
                "timestamp": now,
                "client_ip": "192.0.2.11",
                "model": "qwen3:8b",
                "status_code": 200,
                "duration_ms": 900,
                "state": "error",
            },
            {
                "timestamp": now,
                "client_ip": "192.0.2.12",
                "model": "mistral:7b",
                "status_code": 500,
                "duration_ms": 500,
                "state": "error",
            },
        ]
        store = RequestStore({})
        store._conn = _Connection(rows)
        store._ensure_connection = lambda: None

        summary = store._get_summary_sync()

        self.assertEqual(summary["requests_today"], 4)
        self.assertEqual(summary["avg_duration_ms"], 200)
        self.assertEqual(summary["error_rate"], 25.0)
        qwen = next(row for row in summary["top_models"] if row["model"] == "qwen3:8b")
        self.assertEqual(qwen["count"], 3)
        self.assertEqual(qwen["avg_duration_ms"], 200)


if __name__ == "__main__":
    unittest.main()
