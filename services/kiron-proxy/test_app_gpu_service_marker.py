"""App-Level Regressionstests fuer GPU-Service-Marker-Lifecycle."""

import os
import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import httpx

PROXY_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def _load_local_app():
    spec = importlib.util.spec_from_file_location(
        "kiron_proxy_app_marker_app",
        PROXY_DIR / "app.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(PROXY_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(PROXY_DIR))
    return module


app_module = _load_local_app()


TEST_RUNTIME_DIR = Path(os.environ.get(
    "KIRON_TEST_RUNTIME_DIR",
    "/tmp/kiron-proxy-app-marker-test",
))


def _load_app():
    module = _load_local_app()
    TEST_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for path in TEST_RUNTIME_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    module.vram_lease.RUNTIME_MARKER_DIR = TEST_RUNTIME_DIR
    module.vram_lease.STARTUP_MARKER_PATH = (
        TEST_RUNTIME_DIR / "docling-vram-startup.json"
    )
    module.vram_lease.SHUTDOWN_MARKER_PATH = (
        TEST_RUNTIME_DIR / "docling-vram-shutdown.json"
    )
    module.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH = (
        TEST_RUNTIME_DIR / "gpu-service-loading.json"
    )
    module.vram_lease._lease_cache["active"] = False
    module.vram_lease._lease_cache["ts"] = 0.0
    module.vram_lease._shared_client = None
    return module


class _FakeRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeAsyncClient:
    def __init__(self, response=None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None):
        self.posts.append((url, json))
        if self.exc is not None:
            raise self.exc
        return self.response


class _ExplodingAsyncClient:
    async def __aenter__(self):
        raise AssertionError("backend must not be called")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _json_status(resp):
    return getattr(resp, "status_code", None)


class AppGpuServiceMarkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app_mod = _load_app()

    def tearDown(self):
        _load_app()

    async def test_force_embedding_load_does_not_bypass_startup_marker(self):
        token = self.app_mod.vram_lease.write_overlay_marker("startup", ttl_s=60)
        try:
            resp = await self.app_mod.load_embedding_model({
                "model": "nomic-embed-text",
                "force": True,
            })
        finally:
            self.app_mod.vram_lease.clear_overlay_marker("startup", token)

        self.assertEqual(resp.status_code, 409)

    async def test_marker_write_failure_skips_backend_call(self):
        with mock.patch.object(
            self.app_mod.vram_lease,
            "write_overlay_marker",
            side_effect=self.app_mod.vram_lease.MarkerOwnershipError("foreign"),
        ), mock.patch.object(
            self.app_mod.httpx,
            "AsyncClient",
            return_value=_ExplodingAsyncClient(),
        ):
            resp = await self.app_mod.load_embedding_model({
                "model": "nomic-embed-text",
            })

        self.assertEqual(resp.status_code, 409)
        self.assertIn("marker_write_failed", resp.body.decode())

    async def test_post_write_overlay_recheck_clears_loading_marker(self):
        allowed = self.app_mod.vram_lease.GPUGateDecision(
            allowed=True,
            reason="allowed",
            service_name="Embedding-Service",
        )

        def overlay_active(kind):
            return kind == "startup"

        with mock.patch.object(
            self.app_mod.vram_lease,
            "gpu_gate_decision",
            return_value=allowed,
        ), mock.patch.object(
            self.app_mod.vram_lease,
            "overlay_marker_active",
            side_effect=overlay_active,
        ), mock.patch.object(
            self.app_mod.httpx,
            "AsyncClient",
            return_value=_ExplodingAsyncClient(),
        ):
            resp = await self.app_mod.load_embedding_model({
                "model": "nomic-embed-text",
            })

        self.assertEqual(resp.status_code, 409)
        self.assertFalse(
            self.app_mod.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        )

    async def test_embedding_load_5xx_leaves_marker_ttl(self):
        fake = _FakeAsyncClient(
            httpx.Response(500, json={"error": "boom"})
        )
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", return_value=fake):
            resp = await self.app_mod.load_embedding_model({
                "model": "nomic-embed-text",
            })

        self.assertEqual(_json_status(resp), 500)
        self.assertTrue(
            self.app_mod.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        )

    async def test_embedding_load_clearable_4xx_clears_marker(self):
        fake = _FakeAsyncClient(
            httpx.Response(404, json={"error": "missing"})
        )
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", return_value=fake):
            resp = await self.app_mod.load_embedding_model({
                "model": "nomic-embed-text",
            })

        self.assertEqual(_json_status(resp), 404)
        self.assertFalse(
            self.app_mod.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        )

    async def test_embedding_stop_already_inactive_clears_marker(self):
        with mock.patch.object(
            subprocess,
            "run",
            return_value=_FakeRun(returncode=1, stderr="inactive"),
        ):
            resp = await self.app_mod.stop_embedding({})

        self.assertEqual(resp, {"status": "stopped", "already": True})
        self.assertFalse(
            self.app_mod.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        )

    async def test_deberta_unload_5xx_leaves_marker_ttl(self):
        fake = _FakeAsyncClient(
            httpx.Response(500, json={"error": "boom"})
        )
        with mock.patch.object(self.app_mod.httpx, "AsyncClient", return_value=fake):
            resp = await self.app_mod.unload_deberta_model({})

        self.assertEqual(_json_status(resp), 500)
        self.assertTrue(
            self.app_mod.vram_lease.GPU_SERVICE_LOADING_MARKER_PATH.exists()
        )


if __name__ == "__main__":
    unittest.main()
