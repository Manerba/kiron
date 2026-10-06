"""Offline guardrails for exact body evidence; never contact a native backend."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import httpx

SPEC = importlib.util.spec_from_file_location('feature_probe', Path(__file__).with_name('probe-openai-features.py'))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, handler, path):
        trace = probe.NativeTrace(path)
        await trace.transport.aclose()
        trace.transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=trace, base_url='http://fixture', trust_env=False)
        self.addAsyncCleanup(client.aclose)
        return client

    async def test_original_bodies_and_clean_eof_recorded_without_bearer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            client = await self.make(lambda request: httpx.Response(200, content=b'data: exact\n\n'), path)
            response = await client.post('/v1/chat/completions', json={'prompt': 'fixture'},
                                         headers={'Authorization': 'Bearer never-store-this'})
            self.assertEqual(response.content, b'data: exact\n\n')
            self.assertEqual((path / 'native-001.response.raw').read_bytes(), response.content)
            self.assertTrue((path / 'native-001.eof').is_file())
            self.assertEqual(json.loads((path / 'native-001.request.json').read_text()), {
                'method': 'POST', 'path': '/v1/chat/completions', 'body': {'prompt': 'fixture'}})
            self.assertFalse(any(b'never-store-this' in item.read_bytes() for item in path.iterdir()))

    async def test_interrupted_body_has_no_false_eof(self):
        class Interrupted(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'partial'
                raise httpx.ReadError('interrupted')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            client = await self.make(lambda request: httpx.Response(200, stream=Interrupted()), path)
            with self.assertRaises(httpx.ReadError):
                await client.post('/tokenize', json={'content': [1, 2]})
            self.assertEqual((path / 'native-001.response.raw').read_bytes(), b'partial')
            self.assertFalse((path / 'native-001.eof').exists())

    async def test_oversized_body_stops_recording_before_overallocation(self):
        class Oversized(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(17):
                    yield b'x' * (1024 * 1024)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            client = await self.make(lambda request: httpx.Response(200, stream=Oversized()), path)
            with self.assertRaisesRegex(ValueError, 'limit exceeded'):
                await client.post('/tokenize', json={'content': [1]})
            self.assertEqual((path / 'native-001.response.raw').stat().st_size, 16 * 1024 * 1024)
            self.assertFalse((path / 'native-001.eof').exists())

