"""Regressionstests fuer Proxy-Streaming-Finalisierung (#704)."""

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
    def __init__(self, chunks, status_code=200, raise_after=None):
        self.chunks = chunks
        self.status_code = status_code
        self.headers = {"content-type": "application/x-ndjson"}
        self.closed = False
        self.raise_after = raise_after

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        if self.raise_after is not None:
            raise self.raise_after

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


async def _pass_vram_lease(body, path, model):
    return body, proxy.vram_lease.LeaseOutcome.PASS


class ProxyStreamWarningStateTests(unittest.IsolatedAsyncioTestCase):
    async def _call_stream(self, chunks, raise_after=None):
        store = _Store()
        backend_response = _BackendResponse(chunks, raise_after=raise_after)
        created_clients = []

        def client_factory(**kwargs):
            response = backend_response if not created_clients else _BackendResponse([])
            client = _ProxyHttpClient(response)
            created_clients.append(client)
            return client

        with mock.patch.object(proxy.httpx, "AsyncClient", side_effect=client_factory):
            app = proxy.create_proxy_app(store)

        transport = httpx.ASGITransport(app=app)
        body = {
            "model": "qwen3:8b",
            "prompt": "hi",
            "stream": True,
            "options": {"num_gpu": 0},
        }
        with mock.patch.object(proxy.vram_lease, "apply_bytes", side_effect=_pass_vram_lease), \
             mock.patch.object(proxy.vram_lease, "num_gpu_zero_effective", return_value=True):
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post("/api/generate", json=body)

        final_update = next(kwargs for _, kwargs in reversed(store.updates) if "state" in kwargs)
        return response, final_update

    async def test_valid_done_stream_completes(self):
        chunks = [
            b'{"response":"hel","done":false}\n',
            b'{"response":"lo","done":false}\n',
            b'{"done":true,"eval_count":7}\n',
        ]

        response, final_update = await self._call_stream(chunks)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"".join(chunks))
        self.assertEqual(final_update["state"], "completed")
        self.assertEqual(final_update["tokens_generated"], 7)
        self.assertEqual(final_update["response_body"], "hello")
        self.assertNotIn("error_message", final_update)

    async def test_corrupt_line_with_done_becomes_warning(self):
        chunks = [
            b'{"response":"ok","done":false}\n',
            b"not-json\n",
            b'{"response":"!","done":false}\n',
            b'{"done":true,"eval_count":3}\n',
        ]

        with self.assertLogs("proxy", level="WARNING") as logs:
            response, final_update = await self._call_stream(chunks)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"".join(chunks))
        self.assertTrue(any("NDJSON-Zeile nicht parsbar" in msg for msg in logs.output))
        self.assertEqual(final_update["state"], "warning")
        self.assertEqual(final_update["tokens_generated"], 3)
        self.assertEqual(final_update["response_body"], "ok!")
        self.assertEqual(
            final_update["error_message"],
            "Stream enthielt korrupte NDJSON-Zeilen",
        )

    async def test_stream_without_done_remains_error(self):
        chunks = [
            b'{"response":"partial","done":false}\n',
        ]

        response, final_update = await self._call_stream(chunks)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"".join(chunks))
        self.assertEqual(final_update["state"], "error")
        self.assertEqual(final_update["response_body"], "partial")
        self.assertEqual(
            final_update["error_message"],
            "Stream ohne done-Marker beendet",
        )

    async def test_error_chunk_with_done_finalizes_as_error(self):
        # #865: Ollama-Chunk {"error":"...","done":true} darf nicht als
        # completed geloggt werden — das Done-Feld allein reicht nicht.
        chunks = [
            b'{"response":"part","done":false}\n',
            b'{"error":"backend exploded","done":true}\n',
        ]

        response, final_update = await self._call_stream(chunks)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(final_update["state"], "error")
        self.assertIn("backend exploded", final_update["error_message"])

    async def test_transport_error_after_terminal_done_keeps_completed(self):
        # #899: Wirft aiter_bytes nach dem terminalen done=true (z.B. waehrend
        # Connection-Close), darf der erfolgreiche Stream nicht durch einen
        # Transportfehler in state=error umgewandelt werden und es darf kein
        # zweiter done-Chunk an den Client geyieldet werden.
        chunks = [
            b'{"response":"hel","done":false}\n',
            b'{"response":"lo","done":false}\n',
            b'{"done":true,"eval_count":7}\n',
        ]

        response, final_update = await self._call_stream(
            chunks, raise_after=ConnectionResetError("peer closed")
        )

        self.assertEqual(response.status_code, 200)
        # Originaler Body unveraendert — kein zusaetzlicher Error-Chunk.
        self.assertEqual(response.content, b"".join(chunks))
        self.assertEqual(final_update["state"], "completed")
        self.assertEqual(final_update["tokens_generated"], 7)
        self.assertEqual(final_update["response_body"], "hello")
        self.assertNotIn("error_message", final_update)

    async def test_trailing_chunk_after_done_keeps_completed(self):
        # #1021: Ein nachgelagerter NDJSON-Chunk ohne done darf einen bereits
        # gesehenen done=true-Marker im Loop-Pfad nicht aushebeln (analog zum
        # Rest-Buffer-Guard).
        chunks = [
            b'{"response":"hel","done":false}\n',
            b'{"response":"lo","done":false}\n',
            b'{"done":true,"eval_count":7}\n',
            b'{"response":"trailing","done":false}\n',
        ]

        response, final_update = await self._call_stream(chunks)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(final_update["state"], "completed")
        self.assertEqual(final_update["tokens_generated"], 7)
        self.assertNotIn("error_message", final_update)


if __name__ == "__main__":
    unittest.main()
