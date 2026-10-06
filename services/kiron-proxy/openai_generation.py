"""Pure output-format validation shared by public generation endpoints."""
import json
import re

from kiron_common.local_inference import ErrorCode, FinishReason, OutputFormat, OutputFormatKind
from kiron_common.local_inference.json_schema import (
    InstanceValidationError, SchemaError, compile_schema, compile_schemas, validate_instance,
)

NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def parse_output_format(value):
    """SchemaError paths are relative to response_format/text.format."""
    if type(value) is not dict or type(value.get("type")) is not str:
        raise SchemaError(ErrorCode.INVALID_REQUEST)
    try:
        kind = OutputFormatKind(value["type"])
    except ValueError:
        raise SchemaError(ErrorCode.INVALID_REQUEST, ("type",)) from None
    allowed = {"type", "json_schema"} if kind is OutputFormatKind.JSON_SCHEMA else {"type"}
    if set(value) - allowed:
        raise SchemaError(ErrorCode.UNSUPPORTED_PARAMETER)
    if kind is not OutputFormatKind.JSON_SCHEMA:
        return OutputFormat(kind)
    wrapped = value.get("json_schema")
    if type(wrapped) is not dict or not {"name", "schema"} <= wrapped.keys():
        raise SchemaError(ErrorCode.INVALID_REQUEST, ("json_schema",))
    if set(wrapped) - {"name", "description", "schema", "strict"}:
        raise SchemaError(ErrorCode.UNSUPPORTED_PARAMETER, ("json_schema",))
    if type(wrapped["name"]) is not str or not NAME.fullmatch(wrapped["name"]):
        raise SchemaError(ErrorCode.INVALID_REQUEST, ("json_schema", "name"))
    if "description" in wrapped and type(wrapped["description"]) is not str:
        raise SchemaError(ErrorCode.INVALID_REQUEST, ("json_schema", "description"))
    strict = wrapped.get("strict", False)
    if type(strict) is not bool:
        raise SchemaError(ErrorCode.INVALID_REQUEST, ("json_schema", "strict"))
    try:
        compiled = compile_schema(wrapped["schema"], strict=strict)
    except SchemaError as exc:
        exc.path = ("json_schema", "schema", *exc.path)
        raise
    return OutputFormat(kind, compiled.source, wrapped["name"], strict, wrapped.get("description"))


def validate_request_schemas(tools, output_format):
    pairs = [(tool.parameters, tool.strict) for tool in tools]
    if output_format.kind is OutputFormatKind.JSON_SCHEMA:
        pairs.append((output_format.schema, output_format.strict))
    return compile_schemas(tuple(pairs))


def decode_output(text):
    """Decode an object without duplicate keys, nonfinite values or recursion."""
    if type(text) is not str:
        raise InstanceValidationError()
    try:
        if len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise InstanceValidationError()
        depth, quoted, escaped = 0, False, False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 32:
                    raise InstanceValidationError()
            elif char in "]}":
                depth -= 1

        def pairs(items):
            value = {}
            for key, child in items:
                if key in value:
                    raise InstanceValidationError()
                value[key] = child
            return value

        def invalid(_):
            raise InstanceValidationError()

        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
        # JSON floats may overflow without using a NaN/Infinity literal.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        if type(value) is not dict:
            raise InstanceValidationError()
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise InstanceValidationError() from None


def validate_output(text, output_format, finish_reason):
    if output_format.kind is OutputFormatKind.TEXT or finish_reason is FinishReason.LENGTH:
        return
    if finish_reason is not FinishReason.STOP:
        raise InstanceValidationError()
    value = decode_output(text)
    if output_format.kind is OutputFormatKind.JSON_SCHEMA:
        validate_instance(value, compile_schema(output_format.schema, strict=output_format.strict))
