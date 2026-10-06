"""Pure Prism output-format mapping; capability evidence is checked by caller."""
from kiron_common.local_inference import ErrorCode, LocalInferenceError, OutputFormatKind, RuntimeFailure
from kiron_common.local_inference._values import json_value
from kiron_common.local_inference.json_schema import InstanceValidationError, compile_schema
from openai_generation import validate_output


def native_output_format(output_format):
    if output_format.kind is OutputFormatKind.TEXT:
        return {}
    # A nonempty object schema selects Prism's actual response grammar branch.
    # An empty schema from native json_object would not select that branch.
    schema = ({"type": "object"} if output_format.kind is OutputFormatKind.JSON_OBJECT
              else compile_schema(output_format.schema, strict=output_format.strict).expanded)
    native = {"name": output_format.name or "kiron_json_object", "schema": json_value(schema),
              "strict": output_format.strict}
    if output_format.description is not None:
        native["description"] = output_format.description
    return {"response_format": {"type": "json_schema", "json_schema": native}}


def validate_complete_output(text, output_format, finish_reason):
    try:
        validate_output(text, output_format, finish_reason)
    except InstanceValidationError:
        raise LocalInferenceError(RuntimeFailure(ErrorCode.PROVIDER_ERROR,
            "Provider output does not satisfy the requested format")) from None
