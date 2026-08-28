from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = ROOT / "docs/adr-0009-embedding-contract-v1.md"
MATRIX_PATH = ROOT / "docs/embedding-model-profile-matrix-v1.md"
SCHEMA_PATH = ROOT / "docs/schemas/kiron-embedding-capabilities-v1.schema.json"
EXAMPLES_PATH = ROOT / "docs/schemas/kiron-embedding-capabilities-v1.examples.json"
FIXTURES_PATH = ROOT / "docs/schemas/kiron-embedding-compatibility-v1.fixtures.json"

ROLES = ("search_document", "search_query")
DOMAINS = {
    "search_document": "kiron.embedding.index_compatibility_id",
    "search_query": "kiron.embedding.query_compatibility_id",
}
JCS_INVALID_UNICODE_SCALAR = "jcs_invalid_unicode_scalar"
I_JSON_SAFE_INTEGER = 9_007_199_254_740_991


class JCSCanonicalizationError(ValueError):
    """Controlled fail-closed error for values outside the V1 JCS domain."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_valid_unicode_string(value: str) -> None:
    for code_point in map(ord, value):
        if 0xD800 <= code_point <= 0xDFFF:
            raise JCSCanonicalizationError(
                JCS_INVALID_UNICODE_SCALAR,
                f"lone surrogate U+{code_point:04X}",
            )


def _assert_unicode_scalars(value: object) -> None:
    """Reject surrogate code points in every JSON object name and string value."""

    if isinstance(value, str):
        _assert_valid_unicode_string(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _assert_valid_unicode_string(key)
            _assert_unicode_scalars(item)
    elif isinstance(value, list):
        for item in value:
            _assert_unicode_scalars(item)


def _assert_jcs_v1_value(value: object) -> None:
    """Validate the complete RFC 8785 domain reachable through the V1 schema.

    V1 deliberately restricts numbers to I-JSON-safe integers. All RFC 8785
    string, property-ordering, literal and UTF-8 rules remain applicable.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JCSCanonicalizationError(
                    "jcs_non_string_object_name",
                    f"object name has type {type(key).__name__}",
                )
            _assert_valid_unicode_string(key)
            _assert_jcs_v1_value(item)
    elif isinstance(value, list):
        for item in value:
            _assert_jcs_v1_value(item)
    elif isinstance(value, str):
        _assert_valid_unicode_string(value)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        if abs(value) > I_JSON_SAFE_INTEGER:
            raise JCSCanonicalizationError(
                "jcs_integer_outside_i_json_safe_range",
                str(value),
            )
    else:
        raise JCSCanonicalizationError(
            "jcs_v1_value_type_forbidden",
            type(value).__name__,
        )


def _utf16_sort_key(value: str) -> bytes:
    """Return unsigned UTF-16 code units in big-endian comparison order."""

    _assert_valid_unicode_string(value)
    return value.encode("utf-16-be")


def _jcs_quote(value: str) -> str:
    escapes = {
        0x08: "\\b",
        0x09: "\\t",
        0x0A: "\\n",
        0x0C: "\\f",
        0x0D: "\\r",
    }
    serialized = []
    for character in value:
        code_point = ord(character)
        if code_point in escapes:
            serialized.append(escapes[code_point])
        elif code_point <= 0x1F:
            serialized.append(f"\\u{code_point:04x}")
        elif character == '"':
            serialized.append('\\"')
        elif character == "\\":
            serialized.append("\\\\")
        else:
            serialized.append(character)
    return '"' + "".join(serialized) + '"'


def _serialize_jcs_v1(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_jcs_v1(item) for item in value) + "]"
    if isinstance(value, dict):
        ordered = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
        return "{" + ",".join(
            _jcs_quote(key) + ":" + _serialize_jcs_v1(item)
            for key, item in ordered
        ) + "}"
    raise AssertionError("_assert_jcs_v1_value must reject unsupported values")


def canonical_json(value: object) -> str:
    _assert_jcs_v1_value(value)
    return _serialize_jcs_v1(value)


def build_preimage(capabilities: dict, role: str, profile_index: int = 0) -> dict:
    if role not in ROLES:
        raise ValueError(f"unsupported compatibility role: {role}")
    profile = capabilities["profiles"][profile_index]
    artifact = profile["artifact"]
    pipeline = profile["pipeline"]
    role_formatting = pipeline["formatting"][role]
    limits = profile["max_input_tokens"]
    return {
        "artifact": {
            "auxiliary": sorted(
                (
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in artifact["auxiliary"]
                ),
                key=lambda item: item["path"],
            ),
            "manifest_digest": artifact["manifest_digest"],
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "weights": sorted(
                (
                    {"path": item["path"], "sha256": item["sha256"]}
                    for item in artifact["weights"]
                ),
                key=lambda item: item["path"],
            ),
        },
        "backend": {
            "implementation": profile["backend"]["implementation"],
            "implementation_revision": profile["backend"]["implementation_revision"],
            "type": profile["backend"]["type"],
        },
        "domain": DOMAINS[role],
        "pipeline": {
            "formatting": {
                "owner": profile["formatting_owner"],
                "parameters": role_formatting["parameters"],
                "template": role_formatting["template"],
            },
            "max_input_tokens": {
                "counting": limits["counting"],
                "limit": limits["by_role"][role],
                "overflow": limits["overflow"],
                "truncation": limits["truncation"],
                "unit": limits["unit"],
            },
            "normalization": {
                "dtype": pipeline["normalization"]["dtype"],
                "method": pipeline["normalization"]["method"],
                "scope": pipeline["normalization"]["scope"],
            },
            "parameters": pipeline["parameters"],
            "pooling": {
                "method": pipeline["pooling"]["method"],
                "parameters": pipeline["pooling"]["parameters"],
            },
            "role": role,
            "tokenizer": {
                "repository": pipeline["tokenizer"]["repository"],
                "revision": pipeline["tokenizer"]["revision"],
            },
        },
        "preimage_version": 1,
        "profile": {
            "dimensions": profile["dimensions"],
            "endpoint": profile["endpoint"],
            "kind": profile["kind"],
            "output_normalized": profile["output_normalized"],
            "profile_id": profile["profile_id"],
            "similarity": profile["similarity"],
        },
    }


def compatibility_vector(
    capabilities: dict, role: str, profile_index: int = 0
) -> dict:
    preimage = build_preimage(capabilities, role, profile_index)
    canonical = canonical_json(preimage)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "canonical_preimage": canonical,
        "sha256": digest,
        "id": f"sha256:{digest}",
    }


def _pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON Pointer must start with '/': {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _lookup(document: object, pointer: str) -> object:
    current = document
    for part in _pointer_parts(pointer):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]  # type: ignore[index]
    return current


def _parent(document: object, pointer: str) -> tuple[object, str]:
    parts = _pointer_parts(pointer)
    if not parts:
        raise ValueError("root replacement is not used by these artifacts")
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]  # type: ignore[index]
    return current, parts[-1]


def apply_operations(document: dict, operations: list[dict]) -> dict:
    result = copy.deepcopy(document)
    for operation in operations:
        op = operation["op"]
        parent, token = _parent(result, operation["path"])
        if op == "copy":
            value = copy.deepcopy(_lookup(result, operation["from"]))
        elif op in {"add", "replace"}:
            value = copy.deepcopy(operation["value"])
        elif op == "remove":
            if isinstance(parent, list):
                del parent[int(token)]
            else:
                del parent[token]  # type: ignore[index]
            continue
        else:
            raise ValueError(f"unsupported fixture operation: {op}")

        if isinstance(parent, list):
            if op in {"add", "copy"} and token == "-":
                parent.append(value)
            elif op in {"add", "copy"}:
                parent.insert(int(token), value)
            else:
                parent[int(token)] = value
        else:
            parent[token] = value  # type: ignore[index]
    return result


def _set_ids(capabilities: dict, profile_index: int = 0) -> None:
    profile = capabilities["profiles"][profile_index]
    profile["index_compatibility_id"] = compatibility_vector(
        capabilities, "search_document", profile_index
    )["id"]
    profile["query_compatibility_id"] = compatibility_vector(
        capabilities, "search_query", profile_index
    )["id"]


def base_capabilities() -> dict:
    capabilities = {
        "schema_version": 1,
        "canonical_model_id": "example/role-sensitive-embedder:1",
        "aliases": ["example-role-embedder", "role-embedder:1"],
        "profiles": [
            {
                "profile_id": "service-dense-v1",
                "kind": "dense",
                "endpoint": "/api/embed",
                "default_for_endpoint": True,
                "backend": {
                    "type": "kiron_embeddings",
                    "implementation": "example-sentence-encoder",
                    "implementation_revision": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
                },
                "artifact": {
                    "repository": "example/role-sensitive-embedder",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "manifest_digest": None,
                    "weights": [
                        {
                            "path": "model.safetensors",
                            "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                        }
                    ],
                    "auxiliary": [
                        {
                            "path": "adapter.safetensors",
                            "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
                        }
                    ],
                },
                "dimensions": 768,
                "max_input_tokens": {
                    "unit": "tokenizer_tokens",
                    "counting": "after_server_formatting_including_special_tokens",
                    "truncation": "right",
                    "overflow": "truncate",
                    "by_role": {
                        "search_document": 512,
                        "search_query": 384,
                    },
                },
                "output_normalized": True,
                "similarity": "cosine",
                "input_type": {
                    "supported": ["search_document", "search_query"],
                    "additional_task_roles": ["classification"],
                    "aliases": [
                        {"alias": "document", "canonical": "search_document"},
                        {"alias": "query", "canonical": "search_query"},
                    ],
                    "role_sensitive": True,
                    "required": False,
                    "default": "search_document",
                    "missing_role_behavior": "use_default",
                },
                "formatting_owner": "server",
                "pipeline": {
                    "tokenizer": {
                        "repository": "example/role-sensitive-embedder",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                    },
                    "formatting": {
                        "search_document": {
                            "template": "passage: {text}",
                            "parameters": {},
                        },
                        "search_query": {
                            "template": "query: {text}",
                            "parameters": {},
                        },
                    },
                    "pooling": {
                        "method": "mean",
                        "parameters": {"exclude_special_tokens": True},
                    },
                    "normalization": {
                        "method": "l2",
                        "dtype": "float32",
                        "scope": "vector",
                    },
                    "parameters": {
                        "instruction_version": 1,
                        "language": "de_DE",
                    },
                },
                "verification": {
                    "status": "verified",
                    "evidence": ["fixture:pinned-primary-and-runtime-witness"],
                    "blocking_reasons": [],
                },
                "index_compatibility_id": None,
                "query_compatibility_id": None,
            }
        ],
    }
    _set_ids(capabilities)
    return capabilities


MUTATION_SPECS = [
    ("base", "baseline", [], False, False),
    (
        "artifact_repository",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/artifact/repository", "value": "example/repacked-embedder"}],
        True,
        True,
    ),
    (
        "artifact_revision",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/artifact/revision", "value": "89abcdef0123456789abcdef0123456789abcdef"}],
        True,
        True,
    ),
    (
        "manifest_digest",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/artifact/manifest_digest", "value": "sha256:4444444444444444444444444444444444444444444444444444444444444444"}],
        True,
        True,
    ),
    (
        "weight_digest",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/artifact/weights/0/sha256", "value": "5555555555555555555555555555555555555555555555555555555555555555"}],
        True,
        True,
    ),
    (
        "auxiliary_digest",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/artifact/auxiliary/0/sha256", "value": "6666666666666666666666666666666666666666666666666666666666666666"}],
        True,
        True,
    ),
    (
        "backend_implementation_revision",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/backend/implementation_revision", "value": "sha256:7777777777777777777777777777777777777777777777777777777777777777"}],
        True,
        True,
    ),
    (
        "tokenizer_revision",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/tokenizer/revision", "value": "fedcba9876543210fedcba9876543210fedcba98"}],
        True,
        True,
    ),
    (
        "profile_id",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/profile_id", "value": "service-dense-v2"}],
        True,
        True,
    ),
    (
        "kind_and_endpoint",
        "positive_change",
        [
            {"op": "replace", "path": "/profiles/0/kind", "value": "late_chunking"},
            {"op": "replace", "path": "/profiles/0/endpoint", "value": "/api/embed_late"},
        ],
        True,
        True,
    ),
    (
        "dimensions",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/dimensions", "value": 512}],
        True,
        True,
    ),
    (
        "pooling",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/pooling/method", "value": "cls"}],
        True,
        True,
    ),
    (
        "normalization_dtype",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/normalization/dtype", "value": "float16"}],
        True,
        True,
    ),
    (
        "pipeline_parameters",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/parameters/instruction_version", "value": 2}],
        True,
        True,
    ),
    (
        "unicode_parameter_name_and_value",
        "positive_change",
        [
            {
                "op": "add",
                "path": "/profiles/0/pipeline/parameters/grüße",
                "value": "für KIara 😀",
            }
        ],
        True,
        True,
    ),
    (
        "output_normalized",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/output_normalized", "value": False}],
        True,
        True,
    ),
    (
        "similarity",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/similarity", "value": "dot_product"}],
        True,
        True,
    ),
    (
        "document_formatting",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/formatting/search_document/template", "value": "document: {text}"}],
        True,
        False,
    ),
    (
        "query_formatting",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/pipeline/formatting/search_query/template", "value": "question: {text}"}],
        False,
        True,
    ),
    (
        "document_token_limit",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/max_input_tokens/by_role/search_document", "value": 511}],
        True,
        False,
    ),
    (
        "query_token_limit",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/max_input_tokens/by_role/search_query", "value": 383}],
        False,
        True,
    ),
    (
        "overflow_policy",
        "positive_change",
        [{"op": "replace", "path": "/profiles/0/max_input_tokens/overflow", "value": "late_chunk_fallback"}],
        True,
        True,
    ),
    (
        "runtime_device",
        "negative_stability",
        [{"op": "add", "path": "/profiles/0/x_runtime", "value": {"device": "cuda", "dtype": "bfloat16"}}],
        False,
        False,
    ),
    (
        "runtime_load_and_queue",
        "negative_stability",
        [{"op": "add", "path": "/profiles/0/x_load", "value": {"loaded": True, "queue_depth": 7, "utilization_percent": 91}}],
        False,
        False,
    ),
    (
        "strict_enforcement_phase",
        "negative_stability",
        [
            {"op": "replace", "path": "/profiles/0/input_type/required", "value": True},
            {"op": "replace", "path": "/profiles/0/input_type/default", "value": None},
            {"op": "replace", "path": "/profiles/0/input_type/missing_role_behavior", "value": "reject"},
        ],
        False,
        False,
    ),
    (
        "model_aliases",
        "negative_stability",
        [{"op": "replace", "path": "/aliases", "value": ["new-display-alias"]}],
        False,
        False,
    ),
    (
        "verification_evidence",
        "negative_stability",
        [{"op": "replace", "path": "/profiles/0/verification/evidence", "value": ["fixture:replacement-evidence"]}],
        False,
        False,
    ),
]


JCS_UNICODE_GOLDEN_SPECS = (
    (
        "non_ascii_names_and_values",
        {"日本語": "東京", "grüße": "Grüße aus Köln"},
        '{"grüße":"Grüße aus Köln","日本語":"東京"}',
    ),
    (
        "utf16_order_differs_from_code_point_order",
        {"\ue000": "BMP private-use", "😀": "non-BMP"},
        '{"😀":"non-BMP","\ue000":"BMP private-use"}',
    ),
    (
        "composed_and_decomposed_strings_are_not_normalized",
        {"é": "composed: é", "e\u0301": "decomposed: e\u0301"},
        '{"e\u0301":"decomposed: e\u0301","é":"composed: é"}',
    ),
    (
        "rfc8785_property_sorting_sample",
        {
            "\u20ac": "Euro Sign",
            "\r": "Carriage Return",
            "\ufb33": "Hebrew Letter Dalet With Dagesh",
            "1": "One",
            "😀": "Emoji: Grinning Face",
            "\u0080": "Control",
            "ö": "Latin Small Letter O With Diaeresis",
        },
        '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
        '"😀":"Emoji: Grinning Face","דּ":"Hebrew Letter Dalet With Dagesh"}',
    ),
)

JCS_INVALID_UNICODE_SPECS = (
    (
        "lone_high_surrogate_in_object_name",
        '{"\\ud800":"invalid"}',
    ),
    (
        "lone_low_surrogate_in_string_value",
        '{"value":"\\udfff"}',
    ),
)


def _build_unicode_golden_vectors() -> list[dict]:
    vectors = []
    for name, value, expected in JCS_UNICODE_GOLDEN_SPECS:
        canonical = canonical_json(value)
        if canonical != expected:
            raise AssertionError(f"bad generated JCS Unicode vector for {name}")
        encoded = canonical.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        vectors.append(
            {
                "name": name,
                "value": value,
                "canonical_json": expected,
                "utf8_hex": encoded.hex(),
                "sha256": digest,
                "id": f"sha256:{digest}",
            }
        )
    return vectors


def _build_invalid_unicode_vectors() -> list[dict]:
    return [
        {
            "name": name,
            "json_source": json_source,
            "expected_error": JCS_INVALID_UNICODE_SCALAR,
        }
        for name, json_source in JCS_INVALID_UNICODE_SPECS
    ]


def build_fixture_artifact() -> dict:
    base = base_capabilities()
    base_index = compatibility_vector(base, "search_document")
    base_query = compatibility_vector(base, "search_query")
    vectors = []
    for name, classification, operations, index_changed, query_changed in MUTATION_SPECS:
        mutated = apply_operations(base, operations)
        index = compatibility_vector(mutated, "search_document")
        query = compatibility_vector(mutated, "search_query")
        vectors.append(
            {
                "name": name,
                "classification": classification,
                "operations": operations,
                "expected_change_from_base": {
                    "index": index_changed,
                    "query": query_changed,
                },
                "index": index,
                "query": query,
            }
        )
        if (index["id"] != base_index["id"]) != index_changed:
            raise AssertionError(f"bad generated index relationship for {name}")
        if (query["id"] != base_query["id"]) != query_changed:
            raise AssertionError(f"bad generated query relationship for {name}")
    return {
        "fixture_version": 1,
        "contract": "ADR-0009 compatibility preimage version 1",
        "canonicalization": "RFC 8785/JCS UTF-8 without BOM or trailing newline",
        "digest": "SHA-256 lowercase hex with sha256: ID prefix",
        "mutation_semantics": "Vectors exercise the hash projection; a mutated intermediate need not be a standalone schema example.",
        "base_capabilities": base,
        "vectors": vectors,
        "unicode_golden_vectors": _build_unicode_golden_vectors(),
        "invalid_unicode_vectors": _build_invalid_unicode_vectors(),
    }


def _unverified_ollama_profile() -> dict:
    profile = copy.deepcopy(base_capabilities()["profiles"][0])
    profile.update(
        {
            "profile_id": "ollama-dense-unverified-v1",
            "default_for_endpoint": False,
            "backend": {
                "type": "ollama",
                "implementation": "ollama",
                "implementation_revision": "0.18.0",
            },
            "artifact": {
                "repository": None,
                "revision": None,
                "manifest_digest": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
                "weights": [
                    {
                        "path": "sha256-9999999999999999999999999999999999999999999999999999999999999999",
                        "sha256": "9999999999999999999999999999999999999999999999999999999999999999",
                    }
                ],
                "auxiliary": [],
            },
            "output_normalized": None,
            "similarity": "unknown",
            "input_type": {
                "supported": ["search_document", "search_query"],
                "additional_task_roles": [],
                "aliases": [],
                "role_sensitive": None,
                "required": False,
                "default": None,
                "missing_role_behavior": "unknown",
            },
            "pipeline": {
                "tokenizer": {"repository": None, "revision": None},
                "formatting": {
                    "search_document": {"template": None, "parameters": {}},
                    "search_query": {"template": None, "parameters": {}},
                },
                "pooling": {"method": "unknown", "parameters": {}},
                "normalization": {
                    "method": "unknown",
                    "dtype": "unknown",
                    "scope": "unknown",
                },
                "parameters": {},
            },
            "verification": {
                "status": "unverified",
                "evidence": ["fixture:manifest-and-model-digest-only"],
                "blocking_reasons": [
                    "upstream_revision_and_pipeline_semantics_not_proven"
                ],
            },
            "index_compatibility_id": None,
            "query_compatibility_id": None,
        }
    )
    return profile


def _unverified_late_profile() -> dict:
    profile = copy.deepcopy(base_capabilities()["profiles"][0])
    profile.update(
        {
            "profile_id": "service-late-unverified-v1",
            "kind": "late_chunking",
            "endpoint": "/api/embed_late",
            "default_for_endpoint": True,
            "verification": {
                "status": "unverified",
                "evidence": ["fixture:pipeline-code-only"],
                "blocking_reasons": ["reference_vector_witness_missing"],
            },
            "index_compatibility_id": None,
            "query_compatibility_id": None,
        }
    )
    profile["max_input_tokens"]["overflow"] = "late_chunk_fallback"
    profile["pipeline"]["pooling"] = {
        "method": "late_chunk_mean",
        "parameters": {"exclude_prefix_and_special_tokens": True},
    }
    return profile


def _unverified_unknown_capabilities() -> dict:
    profile = _unverified_ollama_profile()
    profile["profile_id"] = "ollama-unknown-dense-v1"
    profile["default_for_endpoint"] = True
    profile["dimensions"] = 4096
    profile["max_input_tokens"] = {
        "unit": "tokenizer_tokens",
        "counting": "after_server_formatting_including_special_tokens",
        "truncation": "unknown",
        "overflow": "unknown",
        "by_role": {"search_document": None, "search_query": None},
    }
    return {
        "schema_version": 1,
        "canonical_model_id": "example/unverified-ollama:Q4_0",
        "aliases": [],
        "profiles": [profile],
    }


def build_examples_artifact() -> dict:
    provider = base_capabilities()

    strict = copy.deepcopy(provider)
    strict["profiles"][0]["input_type"].update(
        {"required": True, "default": None, "missing_role_behavior": "reject"}
    )

    role_independent = copy.deepcopy(provider)
    role_independent["canonical_model_id"] = "example/role-independent:1"
    role_independent["aliases"] = []
    independent_profile = role_independent["profiles"][0]
    independent_profile["profile_id"] = "service-role-independent-dense-v1"
    independent_profile["input_type"].update(
        {
            "role_sensitive": False,
            "required": False,
            "default": None,
            "missing_role_behavior": "no_op",
        }
    )
    independent_profile["pipeline"]["formatting"] = {
        "search_document": {"template": "{text}", "parameters": {}},
        "search_query": {"template": "{text}", "parameters": {}},
    }
    independent_profile["max_input_tokens"]["by_role"] = {
        "search_document": 512,
        "search_query": 512,
    }
    independent_profile["verification"]["evidence"] = [
        "fixture:role-independent-primary-and-runtime-witness"
    ]
    _set_ids(role_independent)

    multi_profile = copy.deepcopy(provider)
    multi_profile["canonical_model_id"] = "example/multi-backend:1"
    multi_profile["aliases"] = ["example-multi", "multi-backend"]
    service_profile = multi_profile["profiles"][0]
    service_profile["profile_id"] = "service-dense-v1"
    _set_ids(multi_profile)
    multi_profile["profiles"] = sorted(
        [
            _unverified_ollama_profile(),
            service_profile,
            _unverified_late_profile(),
        ],
        key=lambda item: item["profile_id"],
    )

    unverified_unknown = _unverified_unknown_capabilities()

    extension = copy.deepcopy(provider)
    extension["x_registry_epoch"] = 17
    extension["profiles"][0]["x_runtime"] = {
        "device": "cuda",
        "loaded": True,
        "queue_depth": 0,
    }

    unicode_parameters = copy.deepcopy(provider)
    unicode_parameters["canonical_model_id"] = "example/unicode-parameters:1"
    unicode_parameters["profiles"][0]["pipeline"]["parameters"].update(
        {
            "grüße": "für KIara 😀",
            "\ue000": "BMP private-use",
            "😀": "non-BMP",
            "é": "composed: é",
            "e\u0301": "decomposed: e\u0301",
        }
    )
    _set_ids(unicode_parameters)

    valid = [
        {"name": "role_sensitive_provider_phase", "value": provider},
        {"name": "role_sensitive_strict_phase_same_ids", "value": strict},
        {"name": "role_independent_canonical_roles_are_no_op", "value": role_independent},
        {"name": "merged_service_ollama_and_late_profiles", "value": multi_profile},
        {"name": "unverified_unknown_pipeline_has_null_ids", "value": unverified_unknown},
        {"name": "unknown_extensions_are_preserved_and_hash_irrelevant", "value": extension},
        {"name": "unicode_parameter_names_and_values", "value": unicode_parameters},
    ]

    invalid = [
        {
            "name": "unknown_schema_version",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/schema_version", "value": 2}],
            "validator": "schema",
            "reason_code": "unknown_schema_version",
        },
        {
            "name": "missing_profiles",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "remove", "path": "/profiles"}],
            "validator": "schema",
            "reason_code": "missing_required_field",
        },
        {
            "name": "verified_profile_has_null_index_id",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/profiles/0/index_compatibility_id", "value": None}],
            "validator": "schema",
            "reason_code": "verified_id_is_null",
        },
        {
            "name": "unverified_profile_claims_an_id",
            "base_valid": "unverified_unknown_pipeline_has_null_ids",
            "operations": [{"op": "replace", "path": "/profiles/0/index_compatibility_id", "value": provider["profiles"][0]["index_compatibility_id"]}],
            "validator": "schema",
            "reason_code": "unverified_id_is_non_null",
        },
        {
            "name": "kind_endpoint_mismatch",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/profiles/0/kind", "value": "late_chunking"}],
            "validator": "schema",
            "reason_code": "kind_endpoint_mismatch",
        },
        {
            "name": "role_independent_profile_cannot_require_role",
            "base_valid": "role_independent_canonical_roles_are_no_op",
            "operations": [{"op": "replace", "path": "/profiles/0/input_type/required", "value": True}],
            "validator": "schema",
            "reason_code": "role_independent_enforcement_invalid",
        },
        {
            "name": "noncanonical_rag_role",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/profiles/0/input_type/supported/1", "value": "query"}],
            "validator": "schema",
            "reason_code": "noncanonical_rag_role",
        },
        {
            "name": "verified_profile_has_unknown_tokenizer_revision",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/profiles/0/pipeline/tokenizer/revision", "value": None}],
            "validator": "schema",
            "reason_code": "verified_fact_unknown",
        },
        {
            "name": "duplicate_profile_id",
            "base_valid": "merged_service_ollama_and_late_profiles",
            "operations": [{"op": "copy", "from": "/profiles/0", "path": "/profiles/-"}],
            "validator": "invariant",
            "reason_code": "profile_id_not_unique",
        },
        {
            "name": "two_defaults_for_embed_endpoint",
            "base_valid": "merged_service_ollama_and_late_profiles",
            "operations": [{"op": "replace", "path": "/profiles/0/default_for_endpoint", "value": True}],
            "validator": "invariant",
            "reason_code": "endpoint_default_count",
        },
        {
            "name": "profiles_not_sorted",
            "base_valid": "merged_service_ollama_and_late_profiles",
            "operations": [{"op": "replace", "path": "/profiles/0/profile_id", "value": "zzz-unsorted-profile"}],
            "validator": "invariant",
            "reason_code": "profiles_not_sorted",
        },
        {
            "name": "duplicate_artifact_path",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "copy", "from": "/profiles/0/artifact/weights/0", "path": "/profiles/0/artifact/weights/-"}],
            "validator": "invariant",
            "reason_code": "artifact_paths_not_unique",
        },
        {
            "name": "role_independent_templates_differ",
            "base_valid": "role_independent_canonical_roles_are_no_op",
            "operations": [{"op": "replace", "path": "/profiles/0/pipeline/formatting/search_query/template", "value": "query: {text}"}],
            "validator": "invariant",
            "reason_code": "role_independent_pipeline_differs",
        },
        {
            "name": "role_independent_limits_differ",
            "base_valid": "role_independent_canonical_roles_are_no_op",
            "operations": [{"op": "replace", "path": "/profiles/0/max_input_tokens/by_role/search_query", "value": 511}],
            "validator": "invariant",
            "reason_code": "role_independent_pipeline_differs",
        },
        {
            "name": "verified_profile_has_stale_compatibility_ids",
            "base_valid": "role_sensitive_provider_phase",
            "operations": [{"op": "replace", "path": "/profiles/0/dimensions", "value": 769}],
            "validator": "invariant",
            "reason_code": "compatibility_id_mismatch",
        },
    ]
    return {
        "artifact_version": 1,
        "schema": "kiron-embedding-capabilities-v1.schema.json",
        "materialization": "Invalid examples apply RFC 6902-style operations to the named valid value; operations are ordered.",
        "valid": valid,
        "invalid": invalid,
    }


def behavioral_invariant_errors(capabilities: dict) -> list[str]:
    errors: list[str] = []
    try:
        _assert_unicode_scalars(capabilities)
    except JCSCanonicalizationError as error:
        return [error.reason_code]

    aliases = capabilities.get("aliases", [])
    if aliases != sorted(aliases):
        errors.append("aliases_not_sorted")

    profiles = capabilities.get("profiles", [])
    profile_ids = [profile["profile_id"] for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("profile_id_not_unique")
    if profile_ids != sorted(profile_ids):
        errors.append("profiles_not_sorted")

    endpoint_counts: dict[str, int] = {}
    for profile in profiles:
        endpoint = profile["endpoint"]
        endpoint_counts.setdefault(endpoint, 0)
        endpoint_counts[endpoint] += int(profile["default_for_endpoint"] is True)

        artifact_paths: list[str] = []
        for key in ("weights", "auxiliary"):
            paths = [item["path"] for item in profile["artifact"][key]]
            artifact_paths.extend(paths)
            if len(paths) != len(set(paths)):
                errors.append("artifact_paths_not_unique")
            if paths != sorted(paths):
                errors.append("artifact_paths_not_sorted")
        if len(artifact_paths) != len(set(artifact_paths)):
            errors.append("artifact_paths_not_unique")

        input_type = profile["input_type"]
        task_roles = input_type["additional_task_roles"]
        if len(task_roles) != len(set(task_roles)):
            errors.append("additional_task_roles_not_unique")
        if task_roles != sorted(task_roles):
            errors.append("additional_task_roles_not_sorted")
        role_aliases = [
            (item["alias"], item["canonical"]) for item in input_type["aliases"]
        ]
        alias_names = [item[0] for item in role_aliases]
        if len(alias_names) != len(set(alias_names)):
            errors.append("input_type_aliases_not_unique")
        if role_aliases != sorted(role_aliases):
            errors.append("input_type_aliases_not_sorted")
        if input_type["role_sensitive"] is False:
            formatting = profile["pipeline"]["formatting"]
            limits = profile["max_input_tokens"]["by_role"]
            if (
                formatting["search_document"] != formatting["search_query"]
                or limits["search_document"] != limits["search_query"]
            ):
                errors.append("role_independent_pipeline_differs")

        verification = profile["verification"]
        for key in ("evidence", "blocking_reasons"):
            values = verification[key]
            if len(values) != len(set(values)):
                errors.append(f"verification_{key}_not_unique")
            if values != sorted(values):
                errors.append(f"verification_{key}_not_sorted")

        if verification["status"] == "verified":
            index = compatibility_vector(
                capabilities, "search_document", profiles.index(profile)
            )["id"]
            query = compatibility_vector(
                capabilities, "search_query", profiles.index(profile)
            )["id"]
            if (
                profile["index_compatibility_id"] != index
                or profile["query_compatibility_id"] != query
            ):
                errors.append("compatibility_id_mismatch")

    if any(count != 1 for count in endpoint_counts.values()):
        errors.append("endpoint_default_count")
    return errors


class EmbeddingContractArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = _load(SCHEMA_PATH)
        cls.validator = Draft202012Validator(cls.schema)
        cls.examples = _load(EXAMPLES_PATH)
        cls.fixtures = _load(FIXTURES_PATH)

    def test_schema_is_valid_draft_2020_12(self) -> None:
        Draft202012Validator.check_schema(self.schema)
        self.assertEqual(
            self.schema["$id"],
            "https://kiron.local/schemas/kiron-embedding-capabilities-v1.schema.json",
        )

    def test_static_fixture_artifact_matches_reproducible_generator(self) -> None:
        self.assertEqual(self.fixtures, build_fixture_artifact())

    def test_fixture_preimages_digests_and_change_relations(self) -> None:
        base = self.fixtures["base_capabilities"]
        self.validator.validate(base)
        self.assertEqual(behavioral_invariant_errors(base), [])
        base_vector = self.fixtures["vectors"][0]
        self.assertEqual(base_vector["name"], "base")

        classifications = set()
        for vector in self.fixtures["vectors"]:
            classifications.add(vector["classification"])
            materialized = apply_operations(base, vector["operations"])
            for fixture_key, role in (
                ("index", "search_document"),
                ("query", "search_query"),
            ):
                expected = compatibility_vector(materialized, role)
                self.assertEqual(vector[fixture_key], expected, vector["name"])
                self.assertEqual(
                    hashlib.sha256(
                        vector[fixture_key]["canonical_preimage"].encode("utf-8")
                    ).hexdigest(),
                    vector[fixture_key]["sha256"],
                    vector["name"],
                )
                changed = vector[fixture_key]["id"] != base_vector[fixture_key]["id"]
                self.assertEqual(
                    changed,
                    vector["expected_change_from_base"][fixture_key],
                    vector["name"],
                )
        self.assertEqual(
            classifications, {"baseline", "positive_change", "negative_stability"}
        )
        self.assertEqual(
            base_vector["index"]["id"],
            "sha256:9269290d579a293846e7c448be0f6bda36c74b1f495e6e0c28bad2b7c5819ff3",
        )
        self.assertEqual(
            base_vector["query"]["id"],
            "sha256:60dd5bdc46aead3b19325d424ec265240010e0e66a600d736c7a1240d66370bf",
        )

    def test_jcs_unicode_golden_vectors(self) -> None:
        vectors = {
            vector["name"]: vector
            for vector in self.fixtures["unicode_golden_vectors"]
        }
        self.assertEqual(vectors, {
            vector["name"]: vector for vector in _build_unicode_golden_vectors()
        })

        for vector in vectors.values():
            with self.subTest(vector=vector["name"]):
                canonical = canonical_json(vector["value"])
                encoded = canonical.encode("utf-8")
                digest = hashlib.sha256(encoded).hexdigest()
                self.assertEqual(canonical, vector["canonical_json"])
                self.assertEqual(encoded.hex(), vector["utf8_hex"])
                self.assertEqual(digest, vector["sha256"])
                self.assertEqual(f"sha256:{digest}", vector["id"])

        utf16_vector = vectors["utf16_order_differs_from_code_point_order"]
        self.assertEqual(sorted(["\ue000", "😀"]), ["\ue000", "😀"])
        self.assertEqual(
            list(json.loads(utf16_vector["canonical_json"]).keys()),
            ["😀", "\ue000"],
        )

        normalization_vector = vectors[
            "composed_and_decomposed_strings_are_not_normalized"
        ]
        normalized_keys = list(
            json.loads(normalization_vector["canonical_json"]).keys()
        )
        self.assertEqual(normalized_keys, ["e\u0301", "é"])
        self.assertNotEqual(normalized_keys[0], normalized_keys[1])

    def test_lone_surrogates_fail_fast_with_controlled_error(self) -> None:
        for vector in self.fixtures["invalid_unicode_vectors"]:
            with self.subTest(vector=vector["name"]):
                value = json.loads(vector["json_source"])
                with self.assertRaises(JCSCanonicalizationError) as caught:
                    canonical_json(value)
                self.assertEqual(caught.exception.reason_code, vector["expected_error"])

        base = self.fixtures["base_capabilities"]
        for name, key, value in (
            ("object_name", "\ud800", "invalid"),
            ("string_value", "surrogate", "\udfff"),
        ):
            with self.subTest(location=name):
                malformed = copy.deepcopy(base)
                malformed["profiles"][0]["pipeline"]["parameters"][key] = value
                self.assertFalse(list(self.validator.iter_errors(malformed)))
                self.assertEqual(
                    behavioral_invariant_errors(malformed),
                    [JCS_INVALID_UNICODE_SCALAR],
                )
                with self.assertRaises(JCSCanonicalizationError) as caught:
                    compatibility_vector(malformed, "search_document")
                self.assertEqual(
                    caught.exception.reason_code,
                    JCS_INVALID_UNICODE_SCALAR,
                )

    def test_mutation_coverage_contains_normative_fields(self) -> None:
        names = {vector["name"] for vector in self.fixtures["vectors"]}
        self.assertTrue(
            {
                "artifact_repository",
                "artifact_revision",
                "manifest_digest",
                "weight_digest",
                "auxiliary_digest",
                "backend_implementation_revision",
                "tokenizer_revision",
                "profile_id",
                "kind_and_endpoint",
                "dimensions",
                "pooling",
                "normalization_dtype",
                "pipeline_parameters",
                "unicode_parameter_name_and_value",
                "output_normalized",
                "similarity",
                "document_formatting",
                "query_formatting",
                "document_token_limit",
                "query_token_limit",
                "overflow_policy",
                "runtime_device",
                "runtime_load_and_queue",
                "strict_enforcement_phase",
                "model_aliases",
            }.issubset(names)
        )

    def test_static_examples_match_reproducible_generator(self) -> None:
        self.assertEqual(self.examples, build_examples_artifact())

    def test_all_valid_examples_pass_schema_and_behavioral_invariants(self) -> None:
        for example in self.examples["valid"]:
            with self.subTest(example=example["name"]):
                self.validator.validate(example["value"])
                self.assertEqual(behavioral_invariant_errors(example["value"]), [])

        by_name = {item["name"]: item["value"] for item in self.examples["valid"]}
        provider = by_name["role_sensitive_provider_phase"]["profiles"][0]
        strict = by_name["role_sensitive_strict_phase_same_ids"]["profiles"][0]
        self.assertEqual(
            provider["index_compatibility_id"], strict["index_compatibility_id"]
        )
        self.assertEqual(
            provider["query_compatibility_id"], strict["query_compatibility_id"]
        )

    def test_schema_valid_unicode_parameters_receive_stable_ids(self) -> None:
        by_name = {item["name"]: item["value"] for item in self.examples["valid"]}
        capabilities = by_name["unicode_parameter_names_and_values"]
        self.validator.validate(capabilities)
        self.assertEqual(behavioral_invariant_errors(capabilities), [])

        profile = capabilities["profiles"][0]
        self.assertEqual(
            profile["index_compatibility_id"],
            "sha256:8795dd41f674da4b3e055919b5101d4db9aec5c1283a643e661e50e26f0ee6f9",
        )
        self.assertEqual(
            profile["query_compatibility_id"],
            "sha256:6f7a05493db54fb597c8b765bc6059b722694d481c01b8c8e8d8125ad3844f41",
        )
        for role, field in (
            ("search_document", "index_compatibility_id"),
            ("search_query", "query_compatibility_id"),
        ):
            vector = compatibility_vector(capabilities, role)
            self.assertEqual(vector["id"], profile[field])
            self.assertIn('"grüße":"für KIara 😀"', vector["canonical_preimage"])
            self.assertLess(
                vector["canonical_preimage"].index('"😀":"non-BMP"'),
                vector["canonical_preimage"].index('"\ue000":"BMP private-use"'),
            )

    def test_all_invalid_examples_fail_at_declared_layer(self) -> None:
        valid = {item["name"]: item["value"] for item in self.examples["valid"]}
        for example in self.examples["invalid"]:
            with self.subTest(example=example["name"]):
                materialized = apply_operations(
                    valid[example["base_valid"]], example["operations"]
                )
                schema_errors = list(self.validator.iter_errors(materialized))
                invariant_errors = behavioral_invariant_errors(materialized)
                if example["validator"] == "schema":
                    self.assertTrue(schema_errors, example["name"])
                else:
                    self.assertFalse(schema_errors, example["name"])
                    self.assertIn(example["reason_code"], invariant_errors)

    def test_unknown_extensions_are_hash_irrelevant_but_known_malformed_fails(self) -> None:
        valid = {item["name"]: item["value"] for item in self.examples["valid"]}
        base = valid["role_sensitive_provider_phase"]
        extended = valid["unknown_extensions_are_preserved_and_hash_irrelevant"]
        self.assertEqual(
            compatibility_vector(base, "search_document"),
            compatibility_vector(extended, "search_document"),
        )
        self.assertEqual(
            compatibility_vector(base, "search_query"),
            compatibility_vector(extended, "search_query"),
        )
        nested_extensions = copy.deepcopy(base)
        nested_extensions["profiles"][0]["pipeline"]["tokenizer"]["x_note"] = "ignored"
        nested_extensions["profiles"][0]["pipeline"]["pooling"]["x_note"] = "ignored"
        nested_extensions["profiles"][0]["pipeline"]["normalization"]["x_note"] = "ignored"
        self.assertEqual(
            compatibility_vector(base, "search_document"),
            compatibility_vector(nested_extensions, "search_document"),
        )
        self.assertEqual(
            compatibility_vector(base, "search_query"),
            compatibility_vector(nested_extensions, "search_query"),
        )
        malformed = copy.deepcopy(base)
        malformed["profiles"][0]["dimensions"] = "768"
        self.assertTrue(list(self.validator.iter_errors(malformed)))
        non_jcs_number = copy.deepcopy(base)
        non_jcs_number["profiles"][0]["pipeline"]["parameters"]["ratio"] = 0.5
        self.assertTrue(list(self.validator.iter_errors(non_jcs_number)))
        unsafe_integer = copy.deepcopy(base)
        unsafe_integer["profiles"][0]["dimensions"] = 9_007_199_254_740_992
        self.assertTrue(list(self.validator.iter_errors(unsafe_integer)))

    def test_adr_contains_each_normative_cross_system_decision(self) -> None:
        adr = ADR_PATH.read_text(encoding="utf-8")
        adr_flat = " ".join(adr.split())
        for required in (
            "RFC 8785 JSON Canonicalization Scheme (JCS)",
            "UTF-16 code units",
            "jcs_invalid_unicode_scalar",
            "no Unicode normalization",
            "kiron.embedding.index_compatibility_id",
            "kiron.embedding.query_compatibility_id",
            "missing_required_input_type",
            "unsupported_input_type",
            "embedding_discovery_registry_invalid",
            "embedding_backend_incompatible_or_unavailable",
            "required=false",
            "default=search_document",
            "required=true",
            "S6 gate",
            "maximum absolute element error at most",
            "cosine distance at most",
            "`1e-3`",
            "`1e-5`",
            "Same model name, alias, dimension",
        ):
            self.assertIn(required, adr_flat)

        for relative_target in (
            "docs/schemas/kiron-embedding-capabilities-v1.schema.json",
            "docs/embedding-model-profile-matrix-v1.md",
            "docs/schemas/kiron-embedding-capabilities-v1.examples.json",
            "docs/schemas/kiron-embedding-compatibility-v1.fixtures.json",
        ):
            self.assertTrue((ROOT / relative_target).is_file(), relative_target)

    def test_matrix_covers_every_audited_vector_profile_and_evidence_class(self) -> None:
        matrix = MATRIX_PATH.read_text(encoding="utf-8")
        profile_ids = {
            "kiron-nomic-dense-v1",
            "kiron-nomic-late-v1",
            "ollama-nomic-dense-v1",
            "kiron-mxbai-dense-v1",
            "ollama-mxbai-dense-v1",
            "kiron-snowflake-dense-v1",
            "ollama-snowflake-dense-v1",
            "kiron-bge-m3-dense-v1",
            "ollama-bge-m3-dense-v1",
            "kiron-mankei-dense-v1",
            "ollama-e5-mistral-dense-v1",
            "ollama-gte-qwen2-dense-v1",
            "kiron-colbert-xm-multivector-v1",
        }
        for profile_id in profile_ids:
            self.assertIn(f"`{profile_id}`", matrix)

        for required in (
            "Configured",
            "Installed",
            "Internal discovery",
            "Central discovery",
            "RAG verdict",
            "unverified/blocking",
            "/api/embed_late",
            "/api/embed_colbert",
            "first-wins",
            "reports only `completion`",
            "no pinned reference-vector witness",
            "no reference-vector witness",
        ):
            self.assertIn(required, matrix)

        for immutable_revision in (
            "e9b6763023c676ca8431644204f50c2b100d9aab",
            "b33106f585b9ce46904ad7443a3b52b7a63e231c",
            "d8fb21ca8d905d2832ee8b96c894d3298964346b",
            "5617a9f61b028005a4858fdac845db406aefb181",
            "86f562cf94d5510175b546e6e9156f99bbd790b5",
            "960de711799d210957d18df59c14c59a439b608a",
        ):
            self.assertIn(immutable_revision, matrix)


if __name__ == "__main__":
    if sys.argv[1:] == ["--generate-fixtures"]:
        print(json.dumps(build_fixture_artifact(), ensure_ascii=False, indent=2))
    elif sys.argv[1:] == ["--generate-examples"]:
        print(json.dumps(build_examples_artifact(), ensure_ascii=False, indent=2))
    else:
        unittest.main()
