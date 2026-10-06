"""Pure content-order and evidence-constraint checks, no provider processes."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace
import unittest

from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityStatus, ErrorCode, ImagePart,
    LocalInferenceError, Message, MessageRole, ParameterConstraint, TextPart,
)
import prism_vision
import ollama_vision


def image(detail="auto", source="image/png"):
    data = b"already validated image fixture"
    return ImagePart("image/png", data, hashlib.sha256(data).hexdigest(), 224, 224, detail,
                     source_media_type=source, source_sha256="a" * 64)


class PrismVisionTests(unittest.TestCase):
    def setUp(self):
        evidence = CapabilityEvidence("fixture", "a" * 64, "a" * 64, "b" * 64, None, "fixture", "c" * 64,
                                      "offline fixture", datetime.now(timezone.utc))
        constraints = {"formats": ParameterConstraint(allowed_values=("image/png", "image/jpeg", "image/webp")),
                       "detail": ParameterConstraint(allowed_values=("auto",)),
                       "images": ParameterConstraint(minimum=1, maximum=4),
                       "image_pixels": ParameterConstraint(minimum=1, maximum=16_000_000),
                       "total_pixels": ParameterConstraint(minimum=1, maximum=32_000_000),
                       "width": ParameterConstraint(minimum=1, maximum=8192),
                       "height": ParameterConstraint(minimum=1, maximum=8192),
                       "normalized_bytes": ParameterConstraint(minimum=1, maximum=64 * 1024**2)}
        self.capability = Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))
        self.deployment = SimpleNamespace(artifact_identity=SimpleNamespace(projector=object()))

    def test_adjacent_text_is_coalesced_without_newlines_around_ordered_images(self):
        picture = image()
        message = Message(MessageRole.USER, (TextPart(""), TextPart("alpha"), TextPart("beta"), picture,
            TextPart("middle"), TextPart(""), picture, TextPart("after"), TextPart("last")))
        content = prism_vision.message_content(message)
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual([part["text"] for part in content if part["type"] == "text"], ["alphabeta", "middle", "afterlast"])
        for part in content:
            if part["type"] == "image_url":
                prefix, data = part["image_url"]["url"].split(",")
                self.assertEqual(prefix, "data:image/png;base64")
                self.assertEqual(base64.b64decode(data, validate=True), picture.data)
                self.assertEqual(part["image_url"]["detail"], "auto")

    def test_image_only_empty_text_and_text_only_keep_exact_semantics(self):
        self.assertEqual(prism_vision.message_content(Message(MessageRole.USER, (TextPart("a"), TextPart("b")))), "ab")
        result = prism_vision.message_content(Message(MessageRole.USER, (TextPart(""), image(), TextPart(""))))
        self.assertEqual(result[0], {"type": "text", "text": ""})
        self.assertEqual(result[-1], {"type": "text", "text": ""})
        self.assertEqual(len(prism_vision.message_content(Message(MessageRole.USER, (image(),)))), 1)

    def test_projector_detail_original_format_and_budget_checked_without_backend(self):
        messages = (Message(MessageRole.USER, (image(source="image/webp"),)),)
        prism_vision.validate_images(messages, self.deployment, self.capability)
        no_projector = SimpleNamespace(artifact_identity=SimpleNamespace(projector=None))
        with self.assertRaises(LocalInferenceError) as raised:
            prism_vision.validate_images(messages, no_projector, self.capability)
        self.assertEqual(raised.exception.failure.code, ErrorCode.INVALID_CONFIGURATION)
        for field, value in (("detail", "high"), ("source_media_type", "image/jpeg")):
            modified = replace(messages[0].content[0], **{field: value})
            capability = self.capability
            if field == "source_media_type":
                capability = replace(capability, constraints={**capability.constraints,
                    "formats": ParameterConstraint(allowed_values=("image/png", "image/webp"))})
            with self.subTest(field=field), self.assertRaises(LocalInferenceError) as raised:
                prism_vision.validate_images((Message(MessageRole.USER, (modified,)),), self.deployment, capability)
            self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_VALUE)
        incomplete = replace(self.capability, constraints={})
        with self.assertRaises(LocalInferenceError) as raised:
            prism_vision.validate_images(messages, self.deployment, incomplete)
        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_PARAMETER)
        with self.assertRaises(LocalInferenceError):
            prism_vision.validate_images((Message(MessageRole.USER, (image(),) * 5),), self.deployment, self.capability)

    def test_raw_webp_is_not_delegated_to_native_ffmpeg(self):
        raw = replace(image(), media_type="image/webp")
        message = Message(MessageRole.USER, (raw,))
        with self.assertRaises(LocalInferenceError):
            prism_vision.validate_images((message,), self.deployment, self.capability)
        with self.assertRaises(LocalInferenceError):
            prism_vision.message_content(message)

    def test_ollama_does_not_invent_markers_or_separate_image_positions(self):
        text = Message(MessageRole.USER, (TextPart("alpha"), TextPart("beta")))
        self.assertEqual(ollama_vision.message_content(text), "alphabeta")
        with self.assertRaises(LocalInferenceError) as raised:
            ollama_vision.message_content(Message(MessageRole.USER, (TextPart("before"), image(), TextPart("after"))))
        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)


if __name__ == "__main__":
    unittest.main()
