"""Native dashboard unload transport, target identity and release-only admission."""
import unittest
from unittest import mock

import httpx

import app


class EmbeddingUnloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.post = mock.AsyncMock(return_value=httpx.Response(200, json={
            "status": "unloaded", "model": "mankei-326m-embedder", "already": False}))
        self.client = mock.AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.client.post = self.post
        patch = mock.patch.object(app.httpx, "AsyncClient", return_value=self.client)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_alias_target_confirmation_and_no_allocation_or_gpu_gate(self):
        with mock.patch.object(app.vram_lease, "gpu_gate_decision", side_effect=AssertionError("allocation gate called")), \
             mock.patch.object(app.vram_lease, "write_overlay_marker", side_effect=AssertionError("allocation marker written")):
            response = await app.unload_embedding_model({"model": "mankei-326m-embedder:latest"})
        self.assertEqual(response.status_code, 200)
        self.post.assert_awaited_once_with(app.EMBEDDING_UNLOAD_URL, json={"model": "mankei-326m-embedder:latest"})

    async def test_invalid_target_or_force_never_reaches_service(self):
        for body in ({}, {"model": None}, {"model": "unknown"}, {"model": "../../file"},
                     {"model": "mankei-326m-embedder", "force": True}):
            response = await app.unload_embedding_model(body)
            self.assertEqual(response.status_code, 400)
        self.post.assert_not_awaited()

    async def test_invalid_success_response_never_confirms_unload(self):
        for payload in ({"status": "ok"}, {"status": "unloaded", "model": "other", "already": False},
                        {"status": "unloaded", "model": "mankei-326m-embedder", "already": 0}, []):
            self.post.return_value = httpx.Response(200, json=payload)
            response = await app.unload_embedding_model({"model": "mankei-326m-embedder"})
            self.assertEqual(response.status_code, 502)
        self.post.return_value = httpx.Response(200, text="invalid JSON")
        self.assertEqual((await app.unload_embedding_model({"model": "mankei-326m-embedder"})).status_code, 502)

    async def test_backend_errors_and_timeout_are_preserved(self):
        for status in (409, 500, 503):
            self.post.return_value = httpx.Response(status, json={"error": "fixture"})
            response = await app.unload_embedding_model({"model": "mankei-326m-embedder"})
            self.assertEqual(response.status_code, status)
        for error, status in ((httpx.ConnectError("fixture"), 503), (httpx.ReadTimeout("fixture"), 504)):
            self.post.side_effect = error
            response = await app.unload_embedding_model({"model": "mankei-326m-embedder"})
            self.assertEqual(response.status_code, status)
