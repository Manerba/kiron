"""Bounded ASGI receive/send cleanup after success and request deadlines."""
import asyncio
import unittest
from unittest import mock

import openai_api
from test_openai_runtime_api import Keys, Records


def scope():
    return {'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
            'headers': [(b'authorization', b'Bearer test-key')], 'client': ('fixture', 1)}


class BoundaryCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_response_returns_despite_cancel_resistant_receive(self):
        entered, cancelled, release, exited = (asyncio.Event() for _ in range(4))
        first, sent = True, []

        async def receive():
            nonlocal first
            if first:
                first = False
                return {'type': 'http.request', 'body': b'{}'}
            entered.set()
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                raise RuntimeError('late receive failure must be observed')
            finally:
                exited.set()

        async def app(scope, receive, send):
            await receive()
            await entered.wait()
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'OK'})

        records = Records()
        boundary = openai_api._ApiBoundary(app, request_store=records, api_key_store=Keys())
        with mock.patch.object(openai_api, 'CANCEL_WAIT', .01):
            task = asyncio.create_task(boundary(scope(), receive, mock.AsyncMock(side_effect=sent.append)))
            try:
                await asyncio.wait_for(cancelled.wait(), .5)
                done, _ = await asyncio.wait({task}, timeout=.2)
                self.assertIn(task, done)
                task.result()
                self.assertFalse(exited.is_set())
                self.assertEqual(sent[-1]['body'], b'OK')
                self.assertEqual(records.updates[-1][1]['state'], 'completed')
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.wait_for(exited.wait(), .5)
            self.assertEqual(len(sent), 2)

    async def test_timeout_error_send_is_bounded_for_json_and_started_sse(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                first, sent, attempts = True, [], []
                blocked, cancelled, release = (asyncio.Event() for _ in range(3))

                async def receive():
                    nonlocal first
                    if first:
                        first = False
                        return {'type': 'http.request', 'body': b'{}'}
                    await asyncio.Event().wait()

                async def send(message):
                    if message['type'] == 'http.response.body' and not message.get('more_body', False):
                        attempts.append(message)
                        blocked.set()
                        try:
                            await release.wait()
                        except asyncio.CancelledError:
                            cancelled.set()
                            raise
                    sent.append(message)

                async def app(scope, receive, send):
                    await receive()
                    if streaming:
                        await send({'type': 'http.response.start', 'status': 200,
                                    'headers': [(b'content-type', b'text/event-stream')]})
                        await send({'type': 'http.response.body', 'body': b'data: partial\n\n', 'more_body': True})
                    await asyncio.Event().wait()

                records = Records()
                boundary = openai_api._ApiBoundary(app, request_store=records, api_key_store=Keys())
                with mock.patch.object(openai_api, 'REQUEST_TIMEOUT', .03), \
                     mock.patch.object(openai_api, 'ERROR_SEND_TIMEOUT', .02), \
                     mock.patch.object(openai_api, 'CANCEL_WAIT', .01):
                    task = asyncio.create_task(boundary(scope(), receive, send))
                    try:
                        await asyncio.wait_for(blocked.wait(), .5)
                        done, _ = await asyncio.wait({task}, timeout=.2)
                        self.assertIn(task, done)
                        task.result()
                        self.assertTrue(cancelled.is_set())
                        self.assertEqual(len(attempts), 1)
                        self.assertEqual(records.updates[-1][1]['state'], 'error')
                        self.assertFalse(any(message['type'] == 'http.response.body'
                            and not message.get('more_body', False) for message in sent))
                    finally:
                        release.set()
                        await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_resistant_error_header_cannot_send_a_late_body(self):
        entered, release, exited = (asyncio.Event() for _ in range(3))
        messages = []

        async def app(scope, receive, send):
            raise TimeoutError

        async def send(message):
            messages.append(message)
            self.assertEqual(message['type'], 'http.response.start')
            entered.set()
            try:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
            finally:
                exited.set()

        boundary = openai_api._ApiBoundary(app, request_store=Records(), api_key_store=Keys())
        with mock.patch.object(openai_api, 'ERROR_SEND_TIMEOUT', .02), \
             mock.patch.object(openai_api, 'CANCEL_WAIT', .01):
            task = asyncio.create_task(boundary(scope(), mock.AsyncMock(), send))
            try:
                await asyncio.wait_for(entered.wait(), .5)
                done, _ = await asyncio.wait({task}, timeout=.2)
                self.assertIn(task, done)
                task.result()
                self.assertFalse(exited.is_set())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.wait_for(exited.wait(), .5)
                await asyncio.sleep(.01)
            self.assertEqual(len(messages), 1)
