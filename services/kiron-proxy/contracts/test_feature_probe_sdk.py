"""Pinned SDK behavior used by the native feature harness; no live requests."""
import importlib.util
import json
from pathlib import Path
import unittest

import httpx

SPEC = importlib.util.spec_from_file_location('feature_probe_sdk',
    Path(__file__).resolve().parents[3] / 'scripts/prism/probe-openai-features.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class SdkLengthTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_helper_length_preserves_final_exact_usage(self):
        import openai
        base = {'id': 'fixture', 'created': 1, 'model': 'fixture', 'object': 'chat.completion.chunk'}
        frames = [
            {**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]},
            {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
            {**base, 'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14,
                                            'completion_tokens_details': {'reasoning_tokens': 4}}},
        ]
        raw = ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames) + 'data: [DONE]\n\n'
        async with openai.AsyncOpenAI(api_key='fixture', base_url='http://fixture/v1', max_retries=0,
                _strict_response_validation=True, http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text=raw, headers={'content-type': 'text/event-stream'})))) as client:
            async with client.chat.completions.stream(model='fixture', messages=[{'role': 'user', 'content': 'fixture'}]) as stream:
                async for _ in stream:
                    pass
                final = await probe.final_completion(stream, allow_length=True)
                self.assertEqual(final.choices[0].finish_reason, 'length')
                self.assertEqual(final.usage.completion_tokens, 4)
                self.assertEqual(final.usage.completion_tokens_details.reasoning_tokens, 4)
                with self.assertRaises(openai.LengthFinishReasonError):
                    await probe.final_completion(stream)
