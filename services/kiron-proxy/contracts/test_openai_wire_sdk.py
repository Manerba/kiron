"""Pinned SDK validation of actual wire output; entirely in-memory transport."""

import json
import unittest

import httpx
from openai import AsyncOpenAI

from kiron_common.local_inference import (
    EventKind, FinishReason, InferenceEvent, InferenceResult, TextPart, TokenUsage,
)
from openai_wire import chat_events, serialize_completion


class WireSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_nonstream_and_stream_accept_actual_serialized_bytes(self):
        result = InferenceResult("internal", (TextPart("Hallo 世界"),), (), (), TokenUsage(17, 2), FinishReason.STOP)

        async def canonical():
            yield InferenceEvent(EventKind.STARTED, "internal")
            yield InferenceEvent(EventKind.TEXT_DELTA, "internal", text="Hallo ", output_item_index=0, part_index=0)
            yield InferenceEvent(EventKind.TEXT_DELTA, "internal", text="世界", output_item_index=0, part_index=0)
            yield InferenceEvent(EventKind.USAGE, "internal", usage=result.usage)
            yield InferenceEvent(EventKind.COMPLETED, "internal", finish_reason=FinishReason.STOP)

        output = b"".join([frame async for frame in chat_events(
            canonical(), "public/model:tag", "chatcmpl-fixed", 123, True)])

        async def handle(req):
            if json.loads(req.content).get("stream"):
                return httpx.Response(200, content=output, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=serialize_completion(result, "public/model:tag", "chatcmpl-fixed", 123))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(api_key="mock", base_url="http://fixture/v1", http_client=http,
                                   _strict_response_validation=True, max_retries=0) as client:
                response = await client.chat.completions.create(model="public/model:tag", messages=[{"role": "user", "content": "hello"}])
                self.assertEqual(response.choices[0].message.content, "Hallo 世界")
                events = await client.chat.completions.create(model="public/model:tag", messages=[{"role": "user", "content": "hello"}], stream=True)
                chunks = [chunk async for chunk in events]
                self.assertEqual(chunks[-1].usage.total_tokens, 19)
                self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
