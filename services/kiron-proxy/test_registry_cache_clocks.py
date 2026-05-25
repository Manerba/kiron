"""Cache-Clock-Regressionen fuer Registry-/Tags-Caches."""

import asyncio
import inspect
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as app_module  # noqa: E402


class RegistryCacheClockTests(unittest.TestCase):
    def setUp(self):
        self.old_tags_cache = app_module._tags_cache
        self.old_registry_cache = app_module._registry_cache
        app_module._tags_cache = {}
        app_module._registry_cache = {
            "data": None,
            "timestamp": 0,
            "monotonic_timestamp": 0.0,
        }

    def tearDown(self):
        app_module._tags_cache = self.old_tags_cache
        app_module._registry_cache = self.old_registry_cache

    def test_tags_cache_eviction_uses_monotonic_not_wall_time(self):
        app_module._tags_cache["old"] = {
            "data": {"tags": []},
            "timestamp": 10_000.0,
            "monotonic_timestamp": 0.0,
        }

        with mock.patch.object(app_module.time, "time", return_value=1.0), \
             mock.patch.object(app_module.time, "monotonic", return_value=3601.0):
            app_module._tags_cache_set(
                "new",
                {"data": {"tags": []}, "timestamp": 1.0},
            )

        self.assertNotIn("old", app_module._tags_cache)
        self.assertIn("new", app_module._tags_cache)
        self.assertEqual(app_module._tags_cache["new"]["timestamp"], 1.0)
        self.assertEqual(app_module._tags_cache["new"]["monotonic_timestamp"], 3601.0)

    def test_registry_memory_cache_uses_monotonic_timestamp(self):
        source = inspect.getsource(app_module.get_available_models)
        self.assertIn("now_monotonic", source)
        self.assertIn("monotonic_timestamp", source)
        self.assertNotIn('now - _registry_cache["timestamp"]', source)

    def test_disk_promotion_subtracts_disk_age_from_monotonic_anchor(self):
        # #954: Wenn ein 30min alter Disk-Cache nach Restart in den Memory-Cache
        # promoted wird, darf das Memory-TTL-Fenster nicht erneut auf 1h gesetzt
        # werden. Der monotonic_timestamp muss um das Disk-Alter zurueckdatiert
        # sein, damit die Memory-TTL-Pruefung das Disk-Alter beruecksichtigt.
        now_wall = 100_000.0
        now_mono = 5_000.0
        disk_age = 1_800.0  # 30 Minuten
        disk_models = [{"name": "qwen3:7b"}]
        fake_disk = {
            "registry": {"data": disk_models, "timestamp": now_wall - disk_age},
            "tags": {},
        }

        async def empty_local_models(*args, **kwargs):
            class _Resp:
                status_code = 599
            return _Resp()

        with mock.patch.object(app_module.time, "time", return_value=now_wall), \
             mock.patch.object(app_module.time, "monotonic", return_value=now_mono), \
             mock.patch.object(app_module, "_load_registry_disk_cache", return_value=fake_disk), \
             mock.patch.object(app_module.httpx, "AsyncClient") as fake_client:
            # Lokaler Ollama-Aufruf wird ignoriert; Disk-Promotion ist der Pfad,
            # den wir testen wollen.
            fake_client.return_value.__aenter__.return_value.get = empty_local_models
            asyncio.run(app_module.get_available_models())

        self.assertEqual(app_module._registry_cache["data"], disk_models)
        # monotonic_timestamp = now_mono - disk_age, sodass die naechste
        # Memory-Pruefung (now_mono - mono_ts) == disk_age sieht.
        self.assertAlmostEqual(
            app_module._registry_cache["monotonic_timestamp"],
            now_mono - disk_age,
        )


if __name__ == "__main__":
    unittest.main()
