"""Tests fuer Dashboard-API-Key-Handler."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as app_module  # noqa: E402


class _Store:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def set_active(self, key_id, active, *, force=False):
        self.calls.append(("set_active", key_id, active, force))
        if self.exc:
            raise self.exc
        return True

    def delete_key(self, key_id, *, force=False):
        self.calls.append(("delete_key", key_id, force))
        if self.exc:
            raise self.exc
        return True


class ApiKeyEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_store = app_module.api_key_store

    def tearDown(self):
        app_module.api_key_store = self.old_store

    async def test_deactivate_last_active_key_returns_409(self):
        store = _Store(app_module.LastActiveApiKeyError("last active"))
        app_module.api_key_store = store

        resp = await app_module.update_api_key("k1", {"is_active": False})

        self.assertEqual(resp.status_code, 409)
        body = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(body["error"], "last active")
        self.assertIn("force=true", body["hint"])

    async def test_deactivate_force_is_passed_to_store(self):
        store = _Store()
        app_module.api_key_store = store

        result = await app_module.update_api_key(
            "k1", {"is_active": False, "force": True}
        )

        self.assertEqual(result["status"], "updated")
        self.assertEqual(store.calls, [("set_active", "k1", False, True)])

    async def test_delete_last_active_key_returns_409(self):
        store = _Store(app_module.LastActiveApiKeyError("last active"))
        app_module.api_key_store = store

        resp = await app_module.delete_api_key("k1")

        self.assertEqual(resp.status_code, 409)
        body = json.loads(resp.body.decode("utf-8"))
        self.assertEqual(body["error"], "last active")
        self.assertIn("force=true", body["hint"])

    async def test_delete_force_is_passed_to_store(self):
        store = _Store()
        app_module.api_key_store = store

        result = await app_module.delete_api_key("k1", {"force": True})

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(store.calls, [("delete_key", "k1", True)])


if __name__ == "__main__":
    unittest.main()
