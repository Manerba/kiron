"""Regression tests fuer Benchmark-Refresh source timestamps (#769)."""

import copy
import os
import sys
import time
import unittest
from unittest import mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import app as app_module  # noqa: E402


async def _direct_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


class _NoLocalModelsClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        raise RuntimeError("local ollama disabled in test")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _hf_result(name="unitmodel"):
    return [
        {
            "fullname": f"{name} instruct",
            "params_b": 7.0,
            "mmlu_pro": 80.12,
            "gpqa": 40.0,
            "ifeval": 70.0,
            "bbh": 60.0,
        }
    ]


def _evalplus_result(name="unitmodel", score=0.3):
    return {
        name: {
            "pass@1": {
                "humaneval": score,
                "humaneval+": score + 0.1,
            }
        }
    }


class BenchmarkPartialTimestampTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._old_registry_cache = dict(app_module._registry_cache)
        self._patchers = [
            mock.patch.object(app_module, "_to_thread", new=_direct_to_thread),
            mock.patch.object(app_module.httpx, "AsyncClient", _NoLocalModelsClient),
            mock.patch.object(
                app_module,
                "_load_registry_disk_cache",
                return_value={"registry": {"data": [], "timestamp": 0}, "tags": {}},
            ),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        app_module._registry_cache.clear()
        app_module._registry_cache.update(self._old_registry_cache)

    async def _refresh(self, *, cache, registry, hf_result, evalplus_result):
        saved = {}

        def save_cache(value):
            saved.clear()
            saved.update(copy.deepcopy(value))
            return True

        app_module._registry_cache["data"] = registry
        app_module._registry_cache["timestamp"] = time.time()
        app_module._registry_cache["monotonic_timestamp"] = time.monotonic()

        with mock.patch.object(
            app_module, "_load_benchmarks_cache", return_value=copy.deepcopy(cache)
        ), mock.patch.object(
            app_module, "_save_benchmarks_cache", side_effect=save_cache
        ), mock.patch.object(
            app_module, "_fetch_open_llm_leaderboard", new=mock.AsyncMock(return_value=hf_result)
        ) as hf_mock, mock.patch.object(
            app_module, "_fetch_evalplus", new=mock.AsyncMock(return_value=evalplus_result)
        ) as ep_mock:
            result = await app_module._refresh_benchmarks_locked()
        return result, saved, hf_mock, ep_mock

    async def test_full_success_sets_source_timestamps(self):
        before = time.time() - 1.0
        _, saved, hf_mock, ep_mock = await self._refresh(
            cache={"models": {}, "last_refresh": None, "sources": {}},
            registry=[{"name": "unitmodel:7b"}],
            hf_result=_hf_result(),
            evalplus_result=_evalplus_result(),
        )
        after = time.time() + 1.0

        entry = saved["models"]["unitmodel"]
        self.assertEqual(hf_mock.await_count, 1)
        self.assertEqual(ep_mock.await_count, 1)
        self.assertGreaterEqual(entry["hf_timestamp"], before)
        self.assertLessEqual(entry["hf_timestamp"], after)
        self.assertGreaterEqual(entry["ep_timestamp"], before)
        self.assertLessEqual(entry["ep_timestamp"], after)
        self.assertEqual(entry["timestamp"], max(entry["hf_timestamp"], entry["ep_timestamp"]))
        self.assertEqual(entry["hf_match"], "unitmodel instruct")
        self.assertEqual(entry["humaneval"], 0.3)

    async def test_hf_failure_does_not_mark_hf_fresh_or_refetch_fresh_evalplus(self):
        old = time.time() - app_module.BENCHMARK_REFRESH_TTL_S - 10
        cache = {
            "models": {
                "unitmodel": {
                    "ollama_base": "unitmodel",
                    "params_b": 7.0,
                    "timestamp": old,
                    "hf_timestamp": old,
                    "ep_timestamp": old,
                    "hf_match": "old-hf",
                    "mmlu_pro": 1.0,
                    "humaneval": 0.1,
                    "humaneval_plus": 0.2,
                    "evalplus_match": "unitmodel",
                }
            },
            "last_refresh": None,
            "sources": {},
        }

        _, saved1, _, ep_mock1 = await self._refresh(
            cache=cache,
            registry=[{"name": "unitmodel:7b"}],
            hf_result=None,
            evalplus_result=_evalplus_result(score=0.5),
        )
        entry1 = saved1["models"]["unitmodel"]
        self.assertEqual(ep_mock1.await_count, 1)
        self.assertEqual(entry1["hf_timestamp"], old)
        self.assertEqual(entry1["hf_match"], "old-hf")
        self.assertGreater(entry1["ep_timestamp"], old)
        self.assertEqual(entry1["humaneval"], 0.5)

        saved2 = {}

        def save_cache(value):
            saved2.clear()
            saved2.update(copy.deepcopy(value))
            return True

        app_module._registry_cache["data"] = [{"name": "unitmodel:7b"}]
        app_module._registry_cache["monotonic_timestamp"] = time.monotonic()
        with mock.patch.object(
            app_module, "_load_benchmarks_cache", return_value=copy.deepcopy(saved1)
        ), mock.patch.object(
            app_module, "_save_benchmarks_cache", side_effect=save_cache
        ), mock.patch.object(
            app_module, "_fetch_open_llm_leaderboard", new=mock.AsyncMock(return_value=_hf_result())
        ) as hf_mock2, mock.patch.object(
            app_module,
            "_fetch_evalplus",
            new=mock.AsyncMock(side_effect=AssertionError("EvalPlus must stay cached")),
        ) as ep_mock2:
            await app_module._refresh_benchmarks_locked()

        entry2 = saved2["models"]["unitmodel"]
        self.assertEqual(hf_mock2.await_count, 1)
        self.assertEqual(ep_mock2.await_count, 0)
        self.assertGreater(entry2["hf_timestamp"], entry1["hf_timestamp"])
        self.assertEqual(entry2["ep_timestamp"], entry1["ep_timestamp"])
        self.assertEqual(entry2["humaneval"], entry1["humaneval"])

    async def test_both_failed_keeps_prev_timestamps_and_skips_new_entry(self):
        old = time.time() - app_module.BENCHMARK_REFRESH_TTL_S - 10
        cache = {
            "models": {
                "unitmodel": {
                    "ollama_base": "unitmodel",
                    "params_b": 7.0,
                    "timestamp": old,
                    "hf_timestamp": old,
                    "ep_timestamp": old,
                    "hf_match": "old-hf",
                    "humaneval": 0.1,
                    "evalplus_match": "unitmodel",
                }
            },
            "last_refresh": None,
            "sources": {},
        }

        _, saved, hf_mock, ep_mock = await self._refresh(
            cache=cache,
            registry=[{"name": "unitmodel:7b"}, {"name": "newmodel:7b"}],
            hf_result=None,
            evalplus_result=None,
        )

        self.assertEqual(hf_mock.await_count, 2)
        self.assertEqual(ep_mock.await_count, 1)
        self.assertIn("unitmodel", saved["models"])
        self.assertNotIn("newmodel", saved["models"])
        entry = saved["models"]["unitmodel"]
        self.assertEqual(entry["hf_timestamp"], old)
        self.assertEqual(entry["ep_timestamp"], old)
        self.assertEqual(entry["timestamp"], old)


if __name__ == "__main__":
    unittest.main()
