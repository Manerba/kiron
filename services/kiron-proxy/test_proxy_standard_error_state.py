"""Regressionstest fuer Standard-Handler-Finalisierung (#1042).

Standard-Pfad muss HTTP 4xx/5xx vom Backend als state=error klassifizieren —
analog Streaming-Pfad und openai_api.py.
"""

import os
import sys
import unittest
from unittest import mock

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proxy  # noqa: E402


class _Store:
    def __init__(self):
        self.records = []
        self.updates = []

    async def add_request(self, record):
        self.records.append(record)

    async def update_request(self, request_id, **kwargs):
        self.updates.append((request_id, kwargs))


class _BackendResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.closed = False

    async def aiter_bytes(self):
        yield self.body

    async def aclose(self):
        self.closed = True


class _ProxyHttpClient:
    def __init__(self, response):
        self.response = response

    def build_request(self, method, url, headers, content):
        return {
            "method": method,
            "url": url,
            "headers": headers,
            "content": content,
        }

    async def send(self, request, stream=False):
        return self.response

    async def aclose(self):
        pass


async def _pass_vram_lease(body, path, model, routing_view):
    del path, model, routing_view
    return body, proxy.vram_lease.LeaseOutcome.PASS


class ProxyStandardErrorStateTests(unittest.IsolatedAsyncioTestCase):
    async def _call_standard(self, body_bytes, status_code=200):
        store = _Store()
        backend_response = _BackendResponse(body_bytes, status_code=status_code)
        created_clients = []

        def client_factory(**kwargs):
            response = backend_response if not created_clients else _BackendResponse(b"")
            client = _ProxyHttpClient(response)
            created_clients.append(client)
            return client

        with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=client_factory):
            app = proxy.create_proxy_app(store)

        transport = httpx.ASGITransport(app=app)
        body = {
            "model": "qwen3:8b",
            "prompt": "hi",
            "stream": False,
            "options": {"num_gpu": 0},
        }
        with mock.patch.object(proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease), \
             mock.patch.object(proxy.vram_lease, "num_gpu_zero_effective", return_value=True):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post("/api/generate", json=body)

        final_update = next(kwargs for _, kwargs in reversed(store.updates) if "state" in kwargs)
        return response, final_update

    async def test_http_200_is_completed(self):
        body = b'{"response":"hello","done":true,"eval_count":3}'
        response, final_update = await self._call_standard(body, status_code=200)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(final_update["state"], "completed")
        self.assertEqual(final_update["status_code"], 200)
        self.assertEqual(final_update["tokens_generated"], 3)
        self.assertNotIn("error_message", final_update)

    async def test_http_503_is_error(self):
        body = b'{"error":"backend overloaded"}'
        response, final_update = await self._call_standard(body, status_code=503)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(final_update["state"], "error")
        self.assertEqual(final_update["status_code"], 503)
        self.assertIn("backend overloaded", final_update["error_message"])

    async def test_http_404_is_error(self):
        body = b'{"error":"model not found"}'
        response, final_update = await self._call_standard(body, status_code=404)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(final_update["state"], "error")
        self.assertEqual(final_update["status_code"], 404)
        self.assertIn("model not found", final_update["error_message"])

    async def test_http_500_without_error_field_uses_status_message(self):
        body = b'<html>internal server error</html>'
        response, final_update = await self._call_standard(body, status_code=500)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(final_update["state"], "error")
        self.assertEqual(final_update["status_code"], 500)
        self.assertEqual(
            final_update["error_message"],
            "Backend returned HTTP 500",
        )

    async def test_http_200_with_error_field_is_error(self):
        # Analog Streaming-Pfad: Backend kann mit HTTP 200 + error-Feld antworten.
        body = b'{"error":"out of memory","done":true}'
        response, final_update = await self._call_standard(body, status_code=200)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(final_update["state"], "error")
        self.assertEqual(final_update["status_code"], 200)
        self.assertIn("out of memory", final_update["error_message"])


if __name__ == "__main__":
    unittest.main()
