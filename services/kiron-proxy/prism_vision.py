"""Pure Prism image capability checks and position-preserving native content.

Pin 9a9394a's common/chat.cpp adds separators between adjacent typed text
parts. Coalescing text runs prevents that change, while its native image parts
retain positions. Raw WebP is excluded: this build's MTMD_VIDEO=ON would route
it through external ffprobe/ffmpeg, outside the validated decoder boundary.
"""
import base64

from kiron_common.local_inference import (
    CapabilityStatus, ErrorCode, ImagePart, LocalInferenceError, MessageRole, RuntimeFailure, TextPart,
)
from openai_vision import MAX_NORMALIZED_BYTES


def _error(code, parameter="messages"):
    return LocalInferenceError(RuntimeFailure(code, "Image mapping is not verified for this deployment.", parameter))


def validate_images(messages, deployment, capability):
    """Caller first verifies capability evidence against its current implementation."""
    images = [part for message in messages for part in message.content if isinstance(part, ImagePart)]
    if not images:
        return
    if capability.status is not CapabilityStatus.SUPPORTED:
        raise _error(ErrorCode.UNSUPPORTED_CAPABILITY)
    if deployment.artifact_identity.projector is None:
        raise _error(ErrorCode.INVALID_CONFIGURATION)
    if any(message.role is not MessageRole.USER and any(isinstance(p, ImagePart) for p in message.content) for message in messages):
        raise _error(ErrorCode.INVALID_REQUEST)
    total_bytes = sum(len(image.data) for image in images)
    if total_bytes > MAX_NORMALIZED_BYTES:
        raise _error(ErrorCode.UNSUPPORTED_VALUE)

    def check(name, value):
        constraint = capability.constraints.get(name)
        if constraint is None:
            raise _error(ErrorCode.UNSUPPORTED_PARAMETER, name)
        if not constraint.accepts(value):
            raise _error(ErrorCode.UNSUPPORTED_VALUE, name)

    check("images", len(images))
    check("total_pixels", sum(image.width * image.height for image in images))
    check("normalized_bytes", total_bytes)
    for image in images:
        if image.media_type not in ("image/png", "image/jpeg"):
            raise _error(ErrorCode.UNSUPPORTED_VALUE, "formats")
        check("formats", image.source_media_type or image.media_type)
        check("detail", image.detail)
        check("image_pixels", image.width * image.height)
        check("width", image.width)
        check("height", image.height)


def message_content(message):
    if not any(isinstance(part, ImagePart) for part in message.content):
        return "".join(part.text for part in message.content)
    if message.role is not MessageRole.USER:
        raise _error(ErrorCode.INVALID_REQUEST)
    result, text = [], []
    for part in message.content:
        if isinstance(part, TextPart):
            text.append(part.text)
            continue
        if not isinstance(part, ImagePart) or part.media_type not in ("image/png", "image/jpeg"):
            raise _error(ErrorCode.UNSUPPORTED_VALUE)
        if text:
            result.append({"type": "text", "text": "".join(text)})
            text.clear()
        result.append({"type": "image_url", "image_url": {
            "url": f"data:{part.media_type};base64,{base64.b64encode(part.data).decode('ascii')}",
            "detail": part.detail,
        }})
    if text:
        result.append({"type": "text", "text": "".join(text)})
    return result
