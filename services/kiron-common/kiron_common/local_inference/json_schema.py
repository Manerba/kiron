"""Closed, bounded schema contract shared by tools and structured generation.

No resolver, network or filesystem access. The original schema stays immutable;
reference expansion is a separate provider representation. Instance errors have
no HTTP meaning: callers distinguish invalid replay from invalid model output.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType

from ._values import json_mapping
from .lifecycle import ErrorCode

MAX_SCHEMA_BYTES = 64 * 1024
MAX_DEPTH = 16
MAX_PROPERTIES = 128
MAX_ENUM_VALUES = 256
MAX_NODES = 4096
KEYWORDS = frozenset(("type", "properties", "required", "additionalProperties", "items",
                      "enum", "anyOf", "$defs", "$ref", "title", "description"))
TYPES = frozenset(("object", "array", "string", "number", "integer", "boolean", "null"))
# Recognized JSON Schema vocabulary outside this profile. Recognition does not
# enable these assertions or silently discard them from the provider schema.
UNSUPPORTED = frozenset((
    "$schema", "$id", "$anchor", "$dynamicRef", "$dynamicAnchor", "$comment",
    "allOf", "oneOf", "not", "if", "then", "else", "const", "multipleOf",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength",
    "maxLength", "pattern", "format", "prefixItems", "contains", "minContains",
    "maxContains", "minItems", "maxItems", "uniqueItems", "minProperties",
    "maxProperties", "patternProperties", "propertyNames", "dependentSchemas",
    "dependentRequired", "unevaluatedItems", "unevaluatedProperties", "readOnly",
    "writeOnly", "default", "examples", "deprecated", "contentEncoding",
    "contentMediaType", "contentSchema",
))


class SchemaError(ValueError):
    def __init__(self, code, path=(), *, schema_index=0):
        self.code, self.path, self.schema_index = code, tuple(path), schema_index
        super().__init__("Schema exceeds or violates the supported contract")


class InstanceValidationError(ValueError):
    def __init__(self, path=()):
        self.path = tuple(path)
        super().__init__("Value does not satisfy the schema")


@dataclass(frozen=True, slots=True)
class ValidatedSchema:
    source: Mapping
    expanded: Mapping
    strict: bool

    def __post_init__(self):
        if type(self.strict) is not bool:
            raise TypeError("strict must be Boolean")
        object.__setattr__(self, "source", json_mapping(self.source))
        object.__setattr__(self, "expanded", json_mapping(self.expanded))


def _error(path=(), code=ErrorCode.INVALID_REQUEST):
    raise SchemaError(code, path)


def _plain(value, depth=0):
    """Copy JSON values without accepting Python's nonfinite/bool-number aliases."""
    if depth > 32:
        _error()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _error()
        return {key: _plain(child, depth + 1) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child, depth + 1) for child in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _error()


def _equal(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _unsupported_keyword(key, value, path):
    """Reject malformed keyword values before classifying an unavailable feature."""
    strings = {"$schema", "$id", "$anchor", "$dynamicRef", "$dynamicAnchor", "$comment",
               "pattern", "format", "contentEncoding", "contentMediaType"}
    integers = {"minLength", "maxLength", "minContains", "maxContains", "minItems",
                "maxItems", "minProperties", "maxProperties"}
    numbers = {"multipleOf", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}
    booleans = {"uniqueItems", "readOnly", "writeOnly", "deprecated"}
    schemas = {"not", "if", "then", "else", "contains", "propertyNames",
               "unevaluatedItems", "unevaluatedProperties", "contentSchema"}
    schema_arrays = {"allOf", "oneOf", "prefixItems"}
    schema_maps = {"patternProperties", "dependentSchemas"}
    valid_schema = lambda item: type(item) in (dict, bool)
    if (key in strings and type(value) is not str
            or key in integers and (type(value) is not int or value < 0)
            or key in numbers and (type(value) not in (int, float) or key == "multipleOf" and value <= 0)
            or key in booleans and type(value) is not bool
            or key in schemas and not valid_schema(value)
            or key in schema_arrays and (type(value) is not list or not value or not all(map(valid_schema, value)))
            or key in schema_maps and (type(value) is not dict or not all(map(valid_schema, value.values())))
            or key == "examples" and type(value) is not list):
        _error(path)
    if key == "dependentRequired" and (type(value) is not dict or any(
            type(names) is not list or any(type(name) is not str for name in names)
            or len(set(names)) != len(names) for names in value.values())):
        _error(path)
    _error(path, ErrorCode.UNSUPPORTED_CAPABILITY)


def _encoded_size(value, remaining=MAX_SCHEMA_BYTES):
    """Count JSON bytes without serializing an expanded DAG into a large string."""
    if isinstance(value, Mapping):
        size = 2 + max(0, len(value) - 1)
        for key, child in value.items():
            size += _encoded_size(key, remaining - size) + 1
            size += _encoded_size(child, remaining - size)
    elif isinstance(value, (list, tuple)):
        size = 2 + max(0, len(value) - 1)
        for child in value:
            size += _encoded_size(child, remaining - size)
    else:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8"))
    if size > remaining:
        _error()
    return size


class _Compiler:
    def __init__(self):
        self.properties = self.enum_values = self.nodes = 0
        self.source_properties = self.source_enum_values = 0

    def source_budget(self, node):
        if type(node) is not dict:
            return  # Shape errors are reported by expand().
        properties, values = node.get("properties", {}), node.get("enum", [])
        self.source_properties += len(properties) if type(properties) is dict else 0
        self.source_enum_values += len(values) if type(values) is list else 0
        if self.source_properties > MAX_PROPERTIES or self.source_enum_values > MAX_ENUM_VALUES:
            _error()
        for name in ("properties", "$defs"):
            if type(node.get(name)) is dict:
                for child in node[name].values():
                    self.source_budget(child)
        for name in ("items", "additionalProperties"):
            self.source_budget(node.get(name))
        if type(node.get("anyOf")) is list:
            for child in node["anyOf"]:
                self.source_budget(child)

    def expand(self, node, root, strict, path=(), depth=1, stack=(), count=True, root_position=False):
        if not isinstance(node, dict):
            _error(path, ErrorCode.UNSUPPORTED_CAPABILITY if type(node) is bool else ErrorCode.INVALID_REQUEST)
        if depth > MAX_DEPTH:
            _error(path)
        self.nodes += 1
        if self.nodes > MAX_NODES:
            _error(path)
        for key in node:
            if key in UNSUPPORTED:
                _unsupported_keyword(key, node[key], (*path, key))
            elif key not in KEYWORDS:
                _error((*path, key), ErrorCode.UNSUPPORTED_PARAMETER)
        for key in ("title", "description"):
            if key in node and type(node[key]) is not str:
                _error((*path, key))
        definitions = node.get("$defs", {})
        if type(definitions) is not dict:
            _error((*path, "$defs"))
        # Validate unused definitions too; no unknown fields may hide in them.
        for name, definition in definitions.items():
            self.expand(definition, root, strict, (*path, "$defs", name), depth, stack,
                        count=False)
        if "$ref" in node:
            ref = node["$ref"]
            if type(ref) is not str:
                _error((*path, "$ref"))
            if not ref.startswith("#/$defs/"):
                _error((*path, "$ref"), ErrorCode.UNSUPPORTED_CAPABILITY)
            raw = ref[len("#/$defs/"):]
            if "/" in raw:
                _error((*path, "$ref"), ErrorCode.UNSUPPORTED_CAPABILITY)
            if re.search(r"~(?![01])", raw):
                _error((*path, "$ref"))
            name = raw.replace("~1", "/").replace("~0", "~")
            target = root.get("$defs", {}).get(name)
            if target is None:
                _error((*path, "$ref"))
            if ref in stack:
                _error((*path, "$ref"), ErrorCode.UNSUPPORTED_CAPABILITY)
            if set(node) - {"$ref", "$defs", "title", "description"}:
                _error(path, ErrorCode.UNSUPPORTED_CAPABILITY)
            result = self.expand(target, root, strict, path, depth, (*stack, ref), count, root_position)
            result = {**result, **{key: node[key] for key in ("title", "description") if key in node}}
            _encoded_size(result)
            return result
        kind = node.get("type")
        kinds = kind if type(kind) is list else [kind] if kind is not None else []
        if "type" in node and (not kinds or any(type(k) is not str or k not in TYPES for k in kinds)
                              or len(set(kinds)) != len(kinds)):
            _error((*path, "type"))
        properties = node.get("properties", {})
        required = node.get("required", [])
        if type(properties) is not dict or type(required) is not list:
            _error(path)
        if len(properties) > MAX_PROPERTIES:
            _error((*path, "properties"))
        if any(type(name) is not str for name in required) or len(set(required)) != len(required):
            _error((*path, "required"))
        if not set(required) <= properties.keys():
            _error((*path, "required"))
        if any(key in node for key in ("properties", "required", "additionalProperties")) and "object" not in kinds:
            _error(path, ErrorCode.UNSUPPORTED_CAPABILITY)
        if "items" in node and "array" not in kinds:
            _error((*path, "items"), ErrorCode.UNSUPPORTED_CAPABILITY)
        if "array" in kinds and "items" not in node:
            _error((*path, "items"), ErrorCode.UNSUPPORTED_CAPABILITY)
        if strict and "object" in kinds and (node.get("additionalProperties") is not False or set(required) != properties.keys()):
            _error(path)
        if strict and not kinds and "anyOf" not in node and "enum" not in node:
            _error(path, ErrorCode.UNSUPPORTED_CAPABILITY)
        if count:
            self.properties += len(properties)
            if self.properties > MAX_PROPERTIES:
                _error((*path, "properties"))
        result = {key: value for key, value in node.items() if key not in {"$defs", "properties", "items", "anyOf", "additionalProperties"}}
        if "properties" in node:
            result["properties"] = {name: self.expand(value, root, strict,
                (*path, "properties", name), depth + 1, stack, count) for name, value in properties.items()}
        if "items" in node:
            result["items"] = self.expand(node["items"], root, strict, (*path, "items"), depth + 1, stack, count)
        if "additionalProperties" in node:
            additional = node["additionalProperties"]
            result["additionalProperties"] = additional if type(additional) is bool else self.expand(
                additional, root, strict, (*path, "additionalProperties"), depth + 1, stack, count)
        if "anyOf" in node:
            alternatives = node["anyOf"]
            if type(alternatives) is not list or not alternatives:
                _error((*path, "anyOf"))
            if root_position:
                _error((*path, "anyOf"), ErrorCode.UNSUPPORTED_CAPABILITY)
            result["anyOf"] = [self.expand(value, root, strict, (*path, "anyOf", index),
                depth + 1, stack, count) for index, value in enumerate(alternatives)]
        if "enum" in node:
            values = node["enum"]
            if type(values) is not list or not values:
                _error((*path, "enum"))
            if len(values) > MAX_ENUM_VALUES:
                _error((*path, "enum"))
            if any(_equal(value, other) for index, value in enumerate(values) for other in values[:index]):
                _error((*path, "enum"))
            if count:
                self.enum_values += len(values)
                if self.enum_values > MAX_ENUM_VALUES:
                    _error((*path, "enum"))
        _encoded_size(result)
        return result


def compile_schemas(schemas):
    """Compile (schema, strict) pairs under one request-wide resource budget."""
    compiler, result, encoded_bytes, expanded_bytes = _Compiler(), [], 0, 0
    for index, pair in enumerate(schemas):
        try:
            schema, strict = pair
            if type(strict) is not bool or not isinstance(schema, Mapping):
                _error()
            source = _plain(schema)
            encoded_bytes += len(json.dumps(source, ensure_ascii=False, allow_nan=False,
                                             separators=(",", ":")).encode("utf-8"))
            if encoded_bytes > MAX_SCHEMA_BYTES:
                _error()
            compiler.source_budget(source)
            expanded = compiler.expand(source, source, strict, root_position=True)
            if expanded.get("type") != "object":
                _error(("type",))
            expanded_bytes += _encoded_size(expanded, MAX_SCHEMA_BYTES - expanded_bytes)
            result.append(ValidatedSchema(json_mapping(source), json_mapping(expanded), strict))
        except SchemaError as exc:
            exc.schema_index = index
            raise
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise SchemaError(ErrorCode.INVALID_REQUEST, schema_index=index) from None
    return tuple(result)


def compile_schema(schema, *, strict=False):
    return compile_schemas(((schema, strict),))[0]


def schema_features(schema):
    """Exact feature names for evidence-backed adapter constraints, not a grant."""
    if not isinstance(schema, ValidatedSchema):
        raise TypeError("expected a compiled schema")
    keywords, types, variants = set(), set(), set()
    def visit(node):
        keywords.update(node)
        kind = node.get("type", ())
        kinds = (kind,) if type(kind) is str else kind
        types.update(kinds)
        if "null" in kinds and len(kinds) > 1:
            variants.add("nullable")
        if "$ref" in node:
            variants.add("local_refs")
        if "anyOf" in node:
            variants.add("any_of")
        if "object" in kinds:
            additional = node.get("additionalProperties", True)
            if additional is True:
                variants.add("open_objects")
            elif isinstance(additional, Mapping):
                variants.add("schema_additional_properties")
        for name in ("properties", "$defs"):
            for child in node.get(name, {}).values():
                visit(child)
        for name in ("items", "additionalProperties"):
            if isinstance(node.get(name), Mapping):
                visit(node[name])
        for child in node.get("anyOf", ()):
            visit(child)
    visit(schema.source)
    return MappingProxyType({"keywords": tuple(sorted(keywords)), "types": tuple(sorted(types)),
                             "variants": tuple(sorted(variants))})


def _type(value, kind):
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, (list, tuple))
    if kind == "number":
        return type(value) in (int, float)
    if kind == "integer":
        return type(value) is int or type(value) is float and value.is_integer()
    return {"string": type(value) is str, "boolean": type(value) is bool, "null": value is None}[kind]


def _validate(value, schema, path):
    kinds = schema.get("type", ())
    kinds = (kinds,) if type(kinds) is str else kinds
    if kinds and not any(_type(value, kind) for kind in kinds):
        raise InstanceValidationError(path)
    if "enum" in schema and not any(_equal(value, candidate) for candidate in schema["enum"]):
        raise InstanceValidationError(path)
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                _validate(value, alternative, path)
                break
            except InstanceValidationError:
                pass
        else:
            raise InstanceValidationError(path)
    if isinstance(value, Mapping):
        if not set(schema.get("required", ())) <= value.keys():
            raise InstanceValidationError(path)
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for name, child in value.items():
            if name in properties:
                _validate(child, properties[name], (*path, name))
            elif additional is False:
                raise InstanceValidationError(path)
            elif isinstance(additional, Mapping):
                _validate(child, additional, (*path, name))
    if isinstance(value, (list, tuple)) and "items" in schema:
        for index, child in enumerate(value):
            _validate(child, schema["items"], (*path, index))


def validate_instance(value, schema):
    if not isinstance(schema, ValidatedSchema):
        raise TypeError("expected a compiled schema")
    try:
        value = _plain(value)
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (SchemaError, ValueError, UnicodeError, RecursionError):
        raise InstanceValidationError() from None
    _validate(value, schema.expanded, ())
