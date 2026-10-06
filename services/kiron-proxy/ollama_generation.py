"""Pure Ollama generation mappings, gated separately by provider evidence."""
from kiron_common.local_inference import ErrorCode, LocalInferenceError, OutputFormatKind, RuntimeFailure
from kiron_common.local_inference._values import json_value
from kiron_common.local_inference.json_schema import compile_schema


def native_output_format(output_format):
    if output_format.kind is OutputFormatKind.TEXT:
        return {}
    if output_format.kind is OutputFormatKind.JSON_OBJECT:
        return {"format": "json"}
    return {"format": json_value(compile_schema(output_format.schema, strict=output_format.strict).expanded)}


def native_reasoning_options(reasoning):
    # Ollama has no verified separate token counter in the current profile.
    if reasoning.effort not in (None, "none") or reasoning.summary is not None:
        raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_CAPABILITY,
            "Exact Ollama reasoning usage is not verified"))
    return {"think": False}
