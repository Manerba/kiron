"""Ollama text mapping; vision stays closed until native order is evidenced."""
from kiron_common.local_inference import ErrorCode, ImagePart, LocalInferenceError, RuntimeFailure


def validate_images(messages, deployment=None, capability=None):
    if any(isinstance(part, ImagePart) for message in messages for part in message.content):
        raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_CAPABILITY,
            "Ollama image positioning has no verified native mapping.", "messages"))


def message_content(message):
    validate_images((message,))
    return "".join(part.text for part in message.content)
