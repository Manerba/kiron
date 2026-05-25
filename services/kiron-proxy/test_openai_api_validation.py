"""Validation-Tests fuer OpenAI-kompatible API."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai_api  # noqa: E402


class UnsupportedParameterUnsetTests(unittest.TestCase):
    def test_false_is_not_unset_for_tool_parameters(self):
        self.assertFalse(openai_api._unsupported_value_is_unset("tools", False))
        self.assertFalse(openai_api._unsupported_value_is_unset("functions", False))
        self.assertFalse(openai_api._unsupported_value_is_unset("response_format", False))

    def test_documented_noop_values_stay_unset(self):
        self.assertTrue(openai_api._unsupported_value_is_unset("tools", []))
        self.assertTrue(openai_api._unsupported_value_is_unset("logit_bias", {}))
        self.assertTrue(openai_api._unsupported_value_is_unset("logprobs", False))
        self.assertTrue(openai_api._unsupported_value_is_unset("top_logprobs", 0))
        self.assertFalse(openai_api._unsupported_value_is_unset("top_logprobs", False))

    def test_response_format_text_is_unset(self):
        self.assertTrue(
            openai_api._unsupported_value_is_unset(
                "response_format", {"type": "text"}
            )
        )
        self.assertFalse(
            openai_api._unsupported_value_is_unset(
                "response_format", {"type": "json_object"}
            )
        )
        self.assertFalse(
            openai_api._unsupported_value_is_unset(
                "response_format",
                {"type": "json_schema", "json_schema": {"name": "x"}},
            )
        )
        self.assertFalse(
            openai_api._unsupported_value_is_unset("response_format", {})
        )


class RequestParameterAllowlistTests(unittest.TestCase):
    def test_supported_chat_completion_keys_are_allowed(self):
        payload = {key: None for key in openai_api._CHAT_COMPLETION_REQUEST_KEYS}
        self.assertIsNone(openai_api._unsupported_request_parameter(payload))

    def test_unknown_parameter_is_rejected_even_when_null(self):
        self.assertEqual(
            openai_api._unsupported_request_parameter({
                "model": "llama3.1",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": None,
            }),
            "reasoning_effort",
        )

    def test_active_newer_openai_parameters_are_rejected(self):
        for name, value in (
            ("modalities", ["text", "audio"]),
            ("audio", {"voice": "alloy", "format": "wav"}),
            ("prediction", {"type": "content", "content": "x"}),
            ("store", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    openai_api._unsupported_request_parameter({
                        "model": "llama3.1",
                        "messages": [{"role": "user", "content": "hi"}],
                        name: value,
                    }),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
