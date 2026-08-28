"""Tests fuer die Dashboard-Basic-Auth (#1023/#1024/#1025)."""

import base64
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as app_module  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402


def _basic(user, password):
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


class CheckBasicAuthTests(unittest.TestCase):
    """Reiner Credential-Vergleich ohne HTTP-Stack."""

    def test_correct_credentials(self):
        self.assertTrue(app_module._check_basic_auth(_basic("admin", "admin")))

    def test_wrong_password(self):
        self.assertFalse(app_module._check_basic_auth(_basic("admin", "nope")))

    def test_wrong_user(self):
        self.assertFalse(app_module._check_basic_auth(_basic("root", "admin")))

    def test_missing_header(self):
        self.assertFalse(app_module._check_basic_auth(""))

    def test_non_basic_scheme(self):
        self.assertFalse(app_module._check_basic_auth("Bearer sk-123"))

    def test_malformed_base64(self):
        self.assertFalse(app_module._check_basic_auth("Basic !!!nope"))

    def test_no_colon_separator(self):
        token = base64.b64encode(b"adminonly").decode("ascii")
        self.assertFalse(app_module._check_basic_auth(f"Basic {token}"))


class DashboardAuthIntegrationTests(unittest.TestCase):
    """End-to-end durch die ASGI-Middleware (TestClient)."""

    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_root_without_auth_is_401_with_challenge(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Basic", resp.headers.get("www-authenticate", ""))

    def test_root_with_auth_passes(self):
        resp = self.client.get("/", auth=("admin", "admin"))
        self.assertEqual(resp.status_code, 200)

    def test_api_endpoint_without_auth_is_401(self):
        # mutierender Pfad bleibt ohne Credentials verschlossen (#1025-Klasse)
        resp = self.client.get("/api/maintenance/status")
        self.assertEqual(resp.status_code, 401)

    def test_restart_endpoint_without_auth_is_401(self):
        resp = self.client.post("/api/dashboard/restart")
        self.assertEqual(resp.status_code, 401)

    def test_static_without_auth_is_401(self):
        resp = self.client.get("/static/js/app_core.js")
        self.assertEqual(resp.status_code, 401)

    def test_wrong_credentials_is_401(self):
        resp = self.client.get("/", auth=("admin", "wrong"))
        self.assertEqual(resp.status_code, 401)

    def test_websocket_without_auth_is_rejected(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws/live"):
                pass


if __name__ == "__main__":
    unittest.main()
