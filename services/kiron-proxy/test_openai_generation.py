import json

import pytest

from kiron_common.local_inference import FinishReason, OutputFormat, OutputFormatKind, ToolDefinition
from kiron_common.local_inference.json_schema import InstanceValidationError, SchemaError
from openai_generation import parse_output_format, validate_output, validate_request_schemas


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def test_public_format_preserves_original_schema_name_description_and_strict():
    fmt = parse_output_format({"type": "json_schema", "json_schema": {
        "name": "result", "description": "client annotation", "schema": SCHEMA, "strict": True}})
    assert fmt.kind is OutputFormatKind.JSON_SCHEMA and fmt.strict
    assert fmt.name == "result" and fmt.description == "client annotation"
    assert fmt.schema["properties"]["ok"]["type"] == "boolean"
    validate_output('{"ok":true}', fmt, FinishReason.STOP)
    for value in ('{"ok":1}', '{"ok":true,"extra":1}', '{}'):
        with pytest.raises(InstanceValidationError):
            validate_output(value, fmt, FinishReason.STOP)


@pytest.mark.parametrize("value", [None, False, {}, {"type": "json"},
    {"type": "text", "extra": None}, {"type": "json_object", "schema": {}},
    {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA, "strict": 1}},
    {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA, "description": None}}])
def test_invalid_format_is_rejected(value):
    with pytest.raises(SchemaError):
        parse_output_format(value)


def test_tools_and_output_share_one_byte_budget():
    large = {**SCHEMA, "description": "x" * 33000}
    fmt = parse_output_format({"type": "json_schema", "json_schema": {"name": "x", "schema": large}})
    with pytest.raises(SchemaError) as caught:
        validate_request_schemas((ToolDefinition("tool", large),), fmt)
    assert caught.value.schema_index == 1


@pytest.mark.parametrize("value", ['[]', '{"ok":true,"ok":false}', '{"ok":NaN}',
    '{"ok":1e999}', '{"ok":"\\ud800"}', '{', 'null', '```json\n{}\n```',
    '{"x":' + '[' * 33 + '0' + ']' * 33 + '}'])
def test_invalid_complete_json_never_passes(value):
    fmt = OutputFormat(OutputFormatKind.JSON_OBJECT)
    with pytest.raises(InstanceValidationError):
        validate_output(value, fmt, FinishReason.STOP)
    validate_output(value, fmt, FinishReason.LENGTH)


def test_json_object_and_non_strict_schema_still_validate_complete_output():
    validate_output('{"free":[1,true,null]}', OutputFormat(OutputFormatKind.JSON_OBJECT), FinishReason.STOP)
    fmt = parse_output_format({"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}})
    assert not fmt.strict
    with pytest.raises(InstanceValidationError):
        validate_output('{"ok":"invalid"}', fmt, FinishReason.STOP)
