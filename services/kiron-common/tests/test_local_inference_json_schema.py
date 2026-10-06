"""One offline schema contract for function tools and structured generation."""
from copy import deepcopy
import pytest
from jsonschema import Draft202012Validator

from kiron_common.local_inference import ErrorCode
from kiron_common.local_inference.json_schema import (
    InstanceValidationError, SchemaError, compile_schema, compile_schemas, schema_features, validate_instance,
)


def obj(properties=None, **changes):
    properties = properties or {}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False, **changes}


def test_strict_nested_schema_roundtrip_matches_independent_validator():
    schema = obj({"status": {"type": "string", "enum": ["ok", "failed"]},
        "rows": {"type": "array", "items": {"$ref": "#/$defs/row"}},
        "optional": {"type": ["string", "null"]}},
        **{"$defs": {"row": obj({"id": {"type": "integer"}, "yes": {"type": "boolean"}})}})
    compiled = compile_schema(schema, strict=True)
    cases = [({"status": "ok", "rows": [{"id": 1, "yes": True}], "optional": None}, True),
             ({"status": "failed", "rows": [], "optional": "note"}, True),
             ({"status": "wrong", "rows": [], "optional": None}, False),
             ({"status": "ok", "rows": [{"id": True, "yes": 1}], "optional": None}, False),
             ({"status": "ok", "rows": []}, False),
             ({"status": "ok", "rows": [], "optional": None, "extra": 1}, False)]
    oracle = Draft202012Validator(schema)
    for value, valid in cases:
        assert oracle.is_valid(value) == valid
        if valid:
            validate_instance(value, compiled)
        else:
            with pytest.raises(InstanceValidationError):
                validate_instance(value, compiled)


def test_original_is_immutable_and_reference_free_plan_is_separate():
    schema = obj({"value": {"$ref": "#/$defs/a~1b~0c"}},
                 **{"$defs": {"a/b~c": {"type": "string", "enum": ["yes"]}}})
    original = deepcopy(schema)
    compiled = compile_schema(schema, strict=True)
    schema["$defs"]["a/b~c"]["enum"].append("no")
    assert compiled.source["properties"]["value"]["$ref"] == "#/$defs/a~1b~0c"
    assert compiled.expanded["properties"]["value"]["enum"] == ("yes",)
    assert "$defs" not in compiled.expanded
    assert original["$defs"]["a/b~c"]["enum"] == ["yes"]
    with pytest.raises(TypeError):
        compiled.source["type"] = "array"
    with pytest.raises(TypeError):
        compiled.expanded["properties"]["value"]["type"] = "integer"
    assert compile_schema(compiled.source, strict=True).expanded == compiled.expanded


def test_anyof_nullable_and_enum_values_preserve_json_type_semantics():
    compiled = compile_schema(obj({"value": {"anyOf": [{"type": "null"},
        {"type": "integer", "enum": [1, 2]}]}}), strict=True)
    for value in (None, 1, 1.0, 2):
        validate_instance({"value": value}, compiled)
    for value in (True, 3, "1"):
        with pytest.raises(InstanceValidationError):
            validate_instance({"value": value}, compiled)
    compiled = compile_schema(obj({"value": {"enum": [True, 1]}}))
    validate_instance({"value": 1}, compiled)
    validate_instance({"value": True}, compiled)


def test_non_strict_optional_and_additional_properties_schema():
    compiled = compile_schema(obj({"known": {"type": "string"}}, required=[],
                                  additionalProperties={"type": "integer"}))
    validate_instance({"extra": 3}, compiled)
    with pytest.raises(InstanceValidationError):
        validate_instance({"extra": "wrong"}, compiled)


@pytest.mark.parametrize("schema,code", [
    ({"type": "array", "items": {"type": "string"}}, ErrorCode.INVALID_REQUEST),
    (obj({"x": {"type": "mystery"}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"type": ["string", "string"]}}), ErrorCode.INVALID_REQUEST),
    (obj(required=["missing"]), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"enum": []}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"enum": [1, 1.0]}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"$ref": "#/$defs/missing"}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"$ref": "https://example.invalid/schema"}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"$ref": "file:///etc/passwd"}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"type": "string", "minLength": 1}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"type": "string", "pattern": "[a-z]+"}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"allOf": [{"type": "string"}]}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"type": "string", "minLength": -1}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"type": "string", "pattern": 3}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"allOf": []}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"allOf": [3]}}), ErrorCode.INVALID_REQUEST),
    (obj({"x": {"type": "string", "notAKeyword": True}}), ErrorCode.UNSUPPORTED_PARAMETER),
    (obj(**{"anyOf": [{"type": "object"}]}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj({"x": {"type": "array"}}), ErrorCode.UNSUPPORTED_CAPABILITY),
    (obj(**{"$defs": {"unused": {"unknown": True}}}), ErrorCode.UNSUPPORTED_PARAMETER),
])
def test_invalid_or_unsupported_schema_fails_with_typed_error(schema, code):
    with pytest.raises(SchemaError) as caught:
        compile_schema(schema)
    assert caught.value.code is code


def test_ref_cycles_and_expansion_bombs_are_bounded():
    cyclic = obj({"x": {"$ref": "#/$defs/a"}}, **{"$defs": {
        "a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}})
    with pytest.raises(SchemaError) as caught:
        compile_schema(cyclic)
    assert caught.value.code is ErrorCode.UNSUPPORTED_CAPABILITY
    definitions = {"v0": {"type": "string"}}
    for index in range(1, 12):
        definitions[f"v{index}"] = obj({name: {"$ref": f"#/$defs/v{index-1}"} for name in ("a", "b")})
    with pytest.raises(SchemaError):
        compile_schema(obj({"value": {"$ref": "#/$defs/v11"}}, **{"$defs": definitions}))


def test_anyof_definition_is_allowed_below_root_after_expansion():
    schema = obj({"value": {"$ref": "#/$defs/choice"}}, **{"$defs": {
        "choice": {"anyOf": [{"type": "string"}, {"type": "null"}]}}})
    compiled = compile_schema(schema, strict=True)
    validate_instance({"value": None}, compiled)
    validate_instance({"value": "yes"}, compiled)


def test_strict_checks_referenced_objects_and_does_not_rewrite_required():
    for schema in (obj(additionalProperties=True), obj({"x": {"type": "string"}}, required=[]),
                   obj({"x": {"$ref": "#/$defs/a"}}, **{"$defs": {"a": {"type": "object"}}})):
        with pytest.raises(SchemaError):
            compile_schema(schema, strict=True)


def test_request_wide_schema_byte_property_and_enum_budgets():
    long = obj(description="a" * 33000)
    with pytest.raises(SchemaError) as caught:
        compile_schemas(((long, False), (long, False)))
    assert caught.value.schema_index == 1
    many = obj({f"p{i}": {"type": "string"} for i in range(65)})
    with pytest.raises(SchemaError):
        compile_schemas(((many, False), (many, False)))
    enums = obj({"x": {"enum": list(range(129))}})
    with pytest.raises(SchemaError):
        compile_schemas(((enums, False), (enums, False)))
    boundary = obj({f"p{i}": {"type": "string"} for i in range(128)})
    compile_schema(boundary, strict=True)
    with pytest.raises(SchemaError):
        compile_schema(obj(**{"$defs": {"a": many, "b": many}}))


def test_boolean_numeric_aliases_nonfinite_and_invalid_unicode_are_rejected():
    for value in (float("nan"), float("inf"), "\ud800"):
        with pytest.raises(SchemaError):
            compile_schema(obj({"x": {"enum": [value]}}))
        with pytest.raises(InstanceValidationError):
            validate_instance({"x": value}, compile_schema(obj({"x": {}})))
    with pytest.raises(SchemaError):
        compile_schema(obj(), strict=1)


def test_feature_inventory_uses_schema_nodes_not_property_names():
    schema = obj({"notAKeyword": {"$ref": "#/$defs/choice"}}, **{"$defs": {
        "choice": {"anyOf": [{"type": ["string", "null"]},
                              {"type": "object", "additionalProperties": {"type": "number"}}]}}})
    features = schema_features(compile_schema(schema))
    assert features["types"] == ("null", "number", "object", "string")
    assert features["variants"] == ("any_of", "local_refs", "nullable", "schema_additional_properties")
    assert "notAKeyword" not in features["keywords"] and "$ref" in features["keywords"]
    with pytest.raises(TypeError):
        features["types"] = ()
    assert schema_features(compile_schema({"type": "object"}))["variants"] == ("open_objects",)


def test_expanded_bytes_are_bounded_before_freezing_and_across_schemas(monkeypatch):
    import kiron_common.local_inference.json_schema as compiler
    schema = obj({"x": {"anyOf": [{"$ref": "#/$defs/t"}] * 300}},
                 **{"$defs": {"t": {"type": "string", "description": "x" * 10000}}})
    # Only the unexpanded source is JSON-serialized. Never allocate a multi-MiB
    # serialization or immutable copy of the expanded reference graph.
    original_dumps = compiler.json.dumps
    def bounded_dumps(value, **kwargs):
        if isinstance(value, dict):
            assert value is not None and "$defs" in value
        return original_dumps(value, **kwargs)
    monkeypatch.setattr(compiler.json, "dumps", bounded_dumps)
    with pytest.raises(SchemaError):
        compile_schema(schema)
    small = obj({"x": {"anyOf": [{"$ref": "#/$defs/t"}] * 4}},
                **{"$defs": {"t": {"type": "string", "description": "x" * 10000}}})
    with pytest.raises(SchemaError) as caught:
        compile_schemas(((small, False), (small, False)))
    assert caught.value.schema_index == 1
