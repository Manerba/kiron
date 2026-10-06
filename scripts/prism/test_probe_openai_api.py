"""Closed error-terminal checks for the native context-overflow probe."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('public_api_probe', Path(__file__).with_name('probe-openai-api.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ContextProbeTests(unittest.TestCase):
    def test_fixed_prompt_and_exact_json_error(self):
        self.assertGreater(len(probe.CONTEXT_PROMPT.split()), 1024)
        self.assertLessEqual(len(probe.CONTEXT_PROMPT.encode()), 65536)
        error = {'error': {'message': 'context', 'type': 'invalid_request_error',
                           'param': None, 'code': 'context_length_exceeded'}}
        self.assertEqual(probe.context_error(json.dumps(error).encode(), streaming=False), error)
        for field, value in (('code', 'provider_unavailable'), ('type', 'server_error'), ('param', 'secret')):
            bad = deepcopy(error); bad['error'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.context_error(json.dumps(bad).encode(), streaming=False)

    def test_stream_requires_one_error_followed_by_done_without_success_or_usage(self):
        error = {'error': {'message': 'context', 'type': 'invalid_request_error',
                           'param': None, 'code': 'context_length_exceeded'}}
        role = {'object': 'chat.completion.chunk', 'choices': [
            {'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}
        frame = lambda value: b'data: ' + json.dumps(value).encode() + b'\n\n'
        done = b'data: [DONE]\n\n'
        self.assertEqual(probe.context_error(frame(role) + frame(error) + done, streaming=True), error)
        bad_chunks = [frame(error), frame(error) + frame(role) + done, frame(error) + frame(error) + done,
            frame({**role, 'usage': {'completion_tokens': 1}}) + frame(error) + done,
            frame({**role, 'choices': [{'index': 0, 'delta': {'content': 'invented'}, 'finish_reason': None}]}) + frame(error) + done,
            frame({**role, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}) + frame(error) + done]
        for raw in bad_chunks:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                probe.context_error(raw, streaming=True)


if __name__ == '__main__':
    unittest.main()
