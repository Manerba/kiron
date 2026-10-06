"""Closed OpenAI embedding wire boundary; profile formatting stays in providers.

This module never selects a role, truncates text, reduces dimensions, normalizes
vectors or counts tokens. Those decisions belong to the resolved embedding
contract and its evidence-bound execution pipeline.
"""
import base64
from dataclasses import dataclass
import json
import math
import struct

from kiron_common.local_inference import EmbeddingRequest, EmbeddingResult, TokenUsage
from openai_wire import ApiError, _closed, _json_values, _model_id

MAX_INPUTS = 512
MAX_CHARACTERS = 32768
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def _invalid(param):
    return ApiError("Invalid embedding request value or structure.", "invalid_request", param)


def _protocol():
    return ApiError("The backend returned an invalid embedding result.", "backend_protocol_error",
                    status=502, error_type="server_error")


@dataclass(frozen=True, slots=True)
class ParsedEmbedding:
    model_id: str
    inputs: tuple[str, ...]
    encoding_format: str = "float"
    dimensions: int | None = None

    def to_request(self, model, context):
        return EmbeddingRequest(model, self.inputs, context, dimensions=self.dimensions)


def parse_embeddings(data):
    _json_values(data)
    _closed(data, {"model", "input", "encoding_format", "dimensions"}, ("model", "input"))
    model_id = _model_id(data["model"])
    value = data["input"]
    if type(value) is str:
        inputs = (value,)
    elif type(value) is list:
        if not 1 <= len(value) <= MAX_INPUTS:
            raise _invalid("input")
        token_ids = all(type(item) is int and item >= 0 for item in value)
        token_batches = all(type(item) is list and item and all(type(token) is int and token >= 0 for token in item)
                            for item in value)
        if token_ids or token_batches:
            raise ApiError("Token-ID embedding inputs are not supported.", "unsupported_capability", "input")
        inputs = tuple(value)
    else:
        raise _invalid("input")
    if any(type(text) is not str or not 1 <= len(text) <= MAX_CHARACTERS for text in inputs):
        raise _invalid("input")
    encoding = data.get("encoding_format", "float")
    if type(encoding) is not str or encoding not in ("float", "base64"):
        raise _invalid("encoding_format")
    dimensions = data.get("dimensions")
    if "dimensions" in data and (type(dimensions) is not int or dimensions <= 0):
        raise _invalid("dimensions")
    return ParsedEmbedding(model_id, inputs, encoding, dimensions)


def serialize_embeddings(result, *, request, encoding_format="float"):
    """Return exact ordered Float32 values and actual processed-token usage."""
    if (not isinstance(result, EmbeddingResult) or not isinstance(request, EmbeddingRequest)
            or result.request_id != request.context.request_id or encoding_format not in ("float", "base64")):
        raise _protocol()
    dimension = request.dimensions or request.model.profile_metadata.get("dimensions")
    if type(dimension) is not int or dimension <= 0 or len(result.vectors) != len(request.inputs):
        raise _protocol()
    if len(result.vectors) * dimension * 4 > MAX_OUTPUT_BYTES:
        raise _protocol()
    usage = result.usage
    if not isinstance(usage, TokenUsage):
        raise _protocol()
    try:
        TokenUsage(usage.input_tokens, usage.output_tokens, usage.cached_input_tokens, usage.reasoning_output_tokens)
        if usage.output_tokens != 0:
            raise ValueError("embedding output-token count")
        data = []
        for index, vector in enumerate(result.vectors):
            if len(vector) != dimension:
                raise ValueError("embedding dimension")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
                raise ValueError("embedding value")
            # Both encodings expose the same registered Float32 representation,
            # including its rounding and signed-zero bit pattern.
            encoded = struct.pack("<" + "f" * dimension, *vector)
            converted = struct.unpack("<" + "f" * dimension, encoded)
            if not all(math.isfinite(value) for value in converted):
                raise ValueError("Float32 overflow")
            embedding = base64.b64encode(encoded).decode("ascii") if encoding_format == "base64" else list(converted)
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        payload = {"object": "list", "model": request.model.api_model_id, "data": data,
                   "usage": {"prompt_tokens": usage.input_tokens, "total_tokens": usage.input_tokens}}
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError("embedding output limit")
        return payload
    except (ValueError, TypeError, OverflowError, struct.error, UnicodeError):
        raise _protocol() from None
