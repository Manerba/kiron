"""Tests fuer die Retention-Scheduler-Glue in main.py."""

import asyncio
import os
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as proxy_main  # noqa: E402


class _FakeDB:
    def __init__(self):
        self.calls = []

    def enforce_retention(self, age_days, cancel_event):
        self.calls.append((age_days, cancel_event.is_set()))
        return 7


class _InlineDBWork:
    async def to_thread(self, func, /, *args, **kwargs):
        return func(*args, **kwargs)


class DailyRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_retention_runs_before_first_sleep(self):
        db = _FakeDB()
        cancel_event = threading.Event()
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)
            raise asyncio.CancelledError()

        with mock.patch.object(proxy_main.asyncio, "sleep", fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await proxy_main.daily_retention(
                    db, cancel_event, db_work=_InlineDBWork()
                )

        self.assertEqual(db.calls, [(90, False)])
        self.assertEqual(sleeps, [86400])


if __name__ == "__main__":
    unittest.main()
