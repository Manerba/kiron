"""Execute a catalog-declared dense embedding contract before Ollama dispatch."""

from __future__ import annotations

from collections.abc import Mapping


class EmbeddingRequestError(ValueError):
    def __init__(self, code: str, field_path: str, profile_id: str):
        self.payload = {"error": {"code": code, "field_path": field_path,
                                  "profile_id": profile_id}}
        super().__init__(code)


def prepare_request(body: dict, profile: Mapping, role: str) -> dict:
    """Apply declared templates and pin vector-affecting Ollama options.

    The policy is opt-in through pipeline.parameters.request_options. Hardware
    placement and thread count can vary; context, batch size and dimensionality
    cannot silently select a different vector pipeline under the same profile.
    Ollama owns tokenization, overflow rejection, last-token pooling and L2.
    """
    fixed_options = profile["pipeline"]["parameters"].get("request_options")
    if fixed_options is None:
        return body

    def reject(code, field):
        raise EmbeddingRequestError(code, field, profile["profile_id"])

    allowed = {"model", "input", "input_type", "options", "keep_alive",
               "truncate", "dimensions"}
    for key in body.keys() - allowed:
        reject("embedding_profile_parameter_conflict", "/" + key)
    raw_input = body.get("input")
    texts = [raw_input] if isinstance(raw_input, str) else raw_input
    if not isinstance(texts, list) or not texts or any(type(t) is not str for t in texts):
        reject("invalid_embedding_input", "/input")
    if "truncate" in body and body["truncate"] is not False:
        reject("embedding_profile_parameter_conflict", "/truncate")
    if "dimensions" in body and (type(body["dimensions"]) is not int
                                  or body["dimensions"] != profile["dimensions"]):
        reject("embedding_profile_parameter_conflict", "/dimensions")
    options = body.get("options", {})
    if not isinstance(options, dict):
        reject("embedding_profile_parameter_conflict", "/options")
    for key, value in options.items():
        if key in fixed_options:
            valid = type(value) is int and value == fixed_options[key]
        elif key == "num_gpu":
            valid = type(value) is int and value >= 0
        elif key == "num_thread":
            valid = type(value) is int and value > 0
        else:
            valid = False
        if not valid:
            reject("embedding_profile_parameter_conflict", "/options/" + key)

    template = profile["pipeline"]["formatting"][role]["template"]
    if not isinstance(template, str) or template.count("{text}") != 1:
        raise ValueError("embedding profile requires exactly one {text} placeholder")
    formatted = [template.replace("{text}", text) for text in texts]
    return {**body, "input": formatted[0] if isinstance(raw_input, str) else formatted,
            "options": {**options, **fixed_options}, "truncate": False}
