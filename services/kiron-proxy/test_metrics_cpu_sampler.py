"""Stdlib-Unittest fuer den zentralen CPU-Sampler (#586)."""

import importlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402


class CpuSamplerTests(unittest.TestCase):
    def setUp(self):
        self.metrics = importlib.reload(metrics)

    def tearDown(self):
        self.metrics._cpu_percent_cache = None
        self.metrics._cpu_percent_baseline_seen = False

    def test_get_cpu_metrics_returns_none_before_sample(self):
        self.assertIsNone(self.metrics._cpu_percent_cache)
        with mock.patch.object(
            self.metrics.psutil,
            "cpu_percent",
            side_effect=AssertionError("consumer must not sample cpu_percent"),
        ):
            got = self.metrics.get_cpu_metrics()
        self.assertIsNone(got["usage_percent"])

    def test_sample_cpu_percent_writes_cache(self):
        with mock.patch.object(self.metrics.psutil, "cpu_percent",
                               return_value=42.5):
            self.metrics.sample_cpu_percent()
        self.assertEqual(self.metrics._cpu_percent_cache, 42.5)

    def test_initial_sample_uses_blocking_interval(self):
        """#979: Erstaufruf blockiert mit interval=0.1 fuer echten Wert,
        sonst liefert psutil immer 0.0 und Cache bliebe 10s None."""
        intervals = []

        def fake_cpu_percent(interval=None):
            intervals.append(interval)
            return 12.5

        with mock.patch.object(self.metrics.psutil, "cpu_percent",
                               side_effect=fake_cpu_percent):
            self.metrics.sample_cpu_percent()
            self.metrics.sample_cpu_percent()
        self.assertEqual(intervals, [0.1, None])
        self.assertTrue(self.metrics._cpu_percent_baseline_seen)
        self.assertEqual(self.metrics._cpu_percent_cache, 12.5)

    def test_get_cpu_metrics_reads_cache_after_sample(self):
        self.metrics._cpu_percent_cache = 17.3
        got = self.metrics.get_cpu_metrics()
        self.assertEqual(got["usage_percent"], 17.3)

    def test_sample_cpu_percent_swallows_psutil_error(self):
        self.metrics._cpu_percent_baseline_seen = True
        self.metrics._cpu_percent_cache = 23.4
        with mock.patch.object(
            self.metrics.psutil,
            "cpu_percent",
            side_effect=self.metrics.psutil.Error("boom"),
        ):
            self.metrics.sample_cpu_percent()
        self.assertEqual(self.metrics._cpu_percent_cache, 23.4)


if __name__ == "__main__":
    unittest.main()
