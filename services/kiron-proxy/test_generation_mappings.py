import pytest

from kiron_common.local_inference import FinishReason, LocalInferenceError, OutputFormat, OutputFormatKind, ReasoningOptions
from ollama_generation import native_output_format as ollama_format, native_reasoning_options
from prism_structured import native_output_format as prism_format, validate_complete_output


def test_json_object_selects_nonempty_prism_object_grammar_and_native_ollama_json():
    fmt = OutputFormat(OutputFormatKind.JSON_OBJECT)
    assert prism_format(fmt)["response_format"]["json_schema"]["schema"] == {"type":"object"}
    assert ollama_format(fmt) == {"format":"json"}
    assert prism_format(OutputFormat()) == ollama_format(OutputFormat()) == {}
    validate_complete_output('{"ok":true}',fmt,FinishReason.STOP)
    with pytest.raises(LocalInferenceError):
        validate_complete_output('[]',fmt,FinishReason.STOP)


def test_schema_mapping_uses_ref_free_copy_without_modifying_original():
    schema = {"type":"object","properties":{"x":{"$ref":"#/$defs/value"}},
              "required":["x"],"additionalProperties":False,"$defs":{"value":{"type":"string"}}}
    fmt = OutputFormat(OutputFormatKind.JSON_SCHEMA,schema,"result",True,"description")
    assert prism_format(fmt)["response_format"]["json_schema"]["schema"]["properties"]["x"] == {"type":"string"}
    assert "$ref" in fmt.schema["properties"]["x"]
    assert ollama_format(fmt)["format"]["properties"]["x"] == {"type":"string"}


def test_ollama_reasoning_stays_closed_without_exact_counter_evidence():
    assert native_reasoning_options(ReasoningOptions(effort="none")) == {"think":False}
    with pytest.raises(LocalInferenceError):
        native_reasoning_options(ReasoningOptions(effort="high"))
