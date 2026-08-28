from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct

import pytest
from jsonschema import Draft202012Validator

from kiron_common.embedding_contract import (
    EmbeddingContractError,
    JCSCanonicalizationError,
    canonical_hash_json,
    canonical_json,
    compatibility_vector,
    fallback_is_compatible,
    finalize_capabilities,
    validate_capabilities,
)
from kiron_common.embedding_registry import (
    EMBEDDING_REGISTRY,
    MODEL_CATALOG,
    EmbeddingModelGroup,
    EmbeddingProfileRegistry,
    attach_show_capabilities,
    build_embedding_registry,
    input_type_error_payload,
    merge_discovery_tags,
    resolve_profile_input_type,
)
from kiron_common.model_catalog import ModelCatalog


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    ROOT / "docs/schemas/kiron-embedding-compatibility-v1.fixtures.json"
)
SCHEMA_PATH = ROOT / "docs/schemas/kiron-embedding-capabilities-v1.schema.json"
EXAMPLES_PATH = ROOT / "docs/schemas/kiron-embedding-capabilities-v1.examples.json"
REGISTRY_IDS_PATH = ROOT / "docs/schemas/kiron-embedding-registry-v1.ids.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_manifest_documents() -> list[dict]:
    return [
        group.to_manifest_dict(schema_version=1)
        for group in MODEL_CATALOG.groups
    ]


def _non_embedding_manifest() -> dict:
    return {
        "schema_version": 1,
        "canonical_model_id": "catalog-only-reranker:latest",
        "aliases": ["catalog-only-reranker"],
        "deployments": [
            {
                "id": "catalog-only-reranker.service",
                "backend": {"type": "kiron_deberta", "parameters": {}},
                "artifact": {
                    "type": "huggingface",
                    "repository": "example/catalog-only-reranker",
                    "revision": "a" * 40,
                    "manifest_digest": None,
                    "trust_remote_code": False,
                    "weights": [
                        {"path": "model.safetensors", "sha256": "b" * 64}
                    ],
                    "auxiliary": [],
                    "metadata": {},
                },
                "routes": [{"task": "rerank", "endpoint": "/api/rerank"}],
                "loader": {"type": "cross_encoder", "parameters": {}},
                "metadata": {},
            }
        ],
        "profiles": [
            {
                "id": "catalog-only-reranker.primary",
                "deployment_id": "catalog-only-reranker.service",
                "task": "rerank",
                "endpoint": "/api/rerank",
                "default_for_endpoint": True,
                "metadata": {},
            }
        ],
        "request_defaults": [],
        "metadata": {},
    }


def _pointer_parts(pointer: str) -> list[str]:
    return [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer.removeprefix("/").split("/")
    ]


def _lookup(document: object, pointer: str) -> object:
    current = document
    for part in _pointer_parts(pointer):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _apply_operations(document: dict, operations: list[dict]) -> dict:
    result = copy.deepcopy(document)
    for operation in operations:
        parts = _pointer_parts(operation["path"])
        parent: object = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        token = parts[-1]
        if operation["op"] == "remove":
            if isinstance(parent, list):
                del parent[int(token)]
            else:
                del parent[token]
            continue
        if operation["op"] == "copy":
            value = copy.deepcopy(_lookup(result, operation["from"]))
        else:
            value = copy.deepcopy(operation["value"])
        if isinstance(parent, list):
            if token == "-":
                parent.append(value)
            elif operation["op"] in ("add", "copy"):
                parent.insert(int(token), value)
            else:
                parent[int(token)] = value
        else:
            parent[token] = value
    return result


def test_production_serializer_matches_all_54_frozen_compatibility_preimages():
    fixture = _load(FIXTURE_PATH)
    for case in fixture["vectors"]:
        capabilities = _apply_operations(
            fixture["base_capabilities"], case["operations"]
        )
        for role, expected_key in (
            ("search_document", "index"),
            ("search_query", "query"),
        ):
            assert compatibility_vector(capabilities, role) == case[expected_key], (
                case["name"],
                role,
            )


def test_production_serializer_matches_all_unicode_golden_vectors():
    fixture = _load(FIXTURE_PATH)
    for case in fixture["unicode_golden_vectors"]:
        canonical = canonical_json(case["value"])
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert canonical == case["canonical_json"], case["name"]
        assert canonical.encode("utf-8").hex() == case["utf8_hex"], case["name"]
        assert digest == case["sha256"], case["name"]
        assert f"sha256:{digest}" == case["id"], case["name"]


@pytest.mark.parametrize(
    ("binary64", "expected"),
    (
        ("0000000000000000", "0"),
        ("8000000000000000", "0"),
        ("0000000000000001", "5e-324"),
        ("8000000000000001", "-5e-324"),
        ("7fefffffffffffff", "1.7976931348623157e+308"),
        ("ffefffffffffffff", "-1.7976931348623157e+308"),
        ("4340000000000000", "9007199254740992"),
        ("c340000000000000", "-9007199254740992"),
        ("4430000000000000", "295147905179352830000"),
        ("44b52d02c7e14af5", "9.999999999999997e+22"),
        ("44b52d02c7e14af6", "1e+23"),
        ("44b52d02c7e14af7", "1.0000000000000001e+23"),
        ("444b1ae4d6e2ef4e", "999999999999999700000"),
        ("444b1ae4d6e2ef4f", "999999999999999900000"),
        ("444b1ae4d6e2ef50", "1e+21"),
        ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
        ("3eb0c6f7a0b5ed8d", "0.000001"),
        ("41b3de4355555553", "333333333.3333332"),
        ("41b3de4355555554", "333333333.33333325"),
        ("41b3de4355555555", "333333333.3333333"),
        ("41b3de4355555556", "333333333.3333334"),
        ("41b3de4355555557", "333333333.33333343"),
        ("becbf647612f3696", "-0.0000033333333333333333"),
        ("43143ff3c1cb0959", "1424953923781206.2"),
    ),
)
def test_complete_serializer_matches_rfc8785_number_samples(binary64, expected):
    value = struct.unpack(">d", bytes.fromhex(binary64))[0]
    assert canonical_json(value) == expected


def test_complete_and_hash_serializers_have_intentionally_distinct_number_domains():
    assert canonical_json({"ratio": 0.5}) == '{"ratio":0.5}'
    with pytest.raises(JCSCanonicalizationError) as error:
        canonical_hash_json({"ratio": 0.5})
    assert error.value.reason_code == "jcs_v1_value_type_forbidden"

    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(JCSCanonicalizationError) as error:
            canonical_json({"value": value})
        assert error.value.reason_code == "jcs_non_finite_number"


def test_production_serializer_rejects_lone_surrogates_with_controlled_code():
    fixture = _load(FIXTURE_PATH)
    for case in fixture["invalid_unicode_vectors"]:
        value = json.loads(case["json_source"])
        with pytest.raises(JCSCanonicalizationError) as error:
            canonical_json(value)
        assert error.value.reason_code == case["expected_error"]


def test_production_serializer_rejects_non_json_container_types():
    with pytest.raises(JCSCanonicalizationError) as error:
        canonical_json(("not", "a", "json", "array"))
    assert error.value.reason_code == "jcs_value_type_forbidden"


def test_schema_version_validation_matches_the_frozen_schema_types_and_value():
    schema = _load(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    base = copy.deepcopy(EMBEDDING_REGISTRY.groups[0].capabilities)

    for version in (True, False, "1", None, 0.5):
        candidate = copy.deepcopy(base)
        candidate["schema_version"] = version
        assert list(validator.iter_errors(candidate)), repr(version)
        with pytest.raises(EmbeddingContractError):
            validate_capabilities(candidate)

    for version in (1, 1.0):
        candidate = copy.deepcopy(base)
        candidate["schema_version"] = version
        assert not list(validator.iter_errors(candidate)), repr(version)
        validate_capabilities(candidate)


def test_open_float_extensions_are_preserved_and_hash_irrelevant():
    schema = _load(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    base = copy.deepcopy(EMBEDDING_REGISTRY.require("nomic-embed-text").capabilities)
    extended = copy.deepcopy(base)
    extended["x_runtime_score"] = 0.5
    extended["profiles"][0]["pipeline"]["tokenizer"]["x_ratio"] = 0.5

    assert not list(validator.iter_errors(extended))
    validate_capabilities(extended)
    finalized = finalize_capabilities(extended)
    assert finalized["x_runtime_score"] == 0.5
    assert finalized["profiles"][0]["pipeline"]["tokenizer"]["x_ratio"] == 0.5
    assert canonical_json(finalized) == canonical_json(extended)
    for index in range(len(base["profiles"])):
        for role in ("search_document", "search_query"):
            assert compatibility_vector(extended, role, index) == compatibility_vector(
                base, role, index
            )


def test_hash_parameters_remain_restricted_and_open_extensions_fail_jcs_edges():
    schema = _load(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    base = copy.deepcopy(EMBEDDING_REGISTRY.groups[0].capabilities)

    hash_float = copy.deepcopy(base)
    hash_float["profiles"][0]["pipeline"]["parameters"]["ratio"] = 0.5
    assert list(validator.iter_errors(hash_float))
    with pytest.raises(EmbeddingContractError, match="jcs_v1_value_type_forbidden"):
        validate_capabilities(hash_float)

    for value, reason in (
        (math.nan, "jcs_non_finite_number"),
        (math.inf, "jcs_non_finite_number"),
        ("\ud800", "jcs_invalid_unicode_scalar"),
    ):
        malformed = copy.deepcopy(base)
        malformed["x_invalid"] = value
        with pytest.raises(EmbeddingContractError, match=reason):
            validate_capabilities(malformed)


def test_all_s1_examples_match_schema_and_provider_validator_expectations():
    schema = _load(SCHEMA_PATH)
    examples = _load(EXAMPLES_PATH)
    validator = Draft202012Validator(schema)
    valid = {item["name"]: item["value"] for item in examples["valid"]}

    for name, candidate in valid.items():
        assert not list(validator.iter_errors(candidate)), name
        validate_capabilities(candidate)

    for example in examples["invalid"]:
        candidate = _apply_operations(
            valid[example["base_valid"]], example["operations"]
        )
        with pytest.raises(EmbeddingContractError):
            validate_capabilities(candidate)


def test_registry_contains_exactly_13_schema_valid_profiles():
    schema = _load(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    profiles = []
    for group in EMBEDDING_REGISTRY.groups:
        assert not list(validator.iter_errors(group.capabilities))
        assert group.capabilities["aliases"] == sorted(group.capabilities["aliases"])
        assert group.capabilities["profiles"] == sorted(
            group.capabilities["profiles"], key=lambda item: item["profile_id"]
        )
        profiles.extend(group.capabilities["profiles"])
    assert len(profiles) == EMBEDDING_REGISTRY.profile_count == 13
    assert sum(
        profile["verification"]["status"] == "verified" for profile in profiles
    ) == 3


def test_only_verified_profiles_have_recomputed_non_null_ids():
    for group in EMBEDDING_REGISTRY.groups:
        finalized = finalize_capabilities(group.capabilities)
        assert finalized == group.capabilities
        for profile in group.capabilities["profiles"]:
            verified = profile["verification"]["status"] == "verified"
            assert (profile["index_compatibility_id"] is not None) is verified
            assert (profile["query_compatibility_id"] is not None) is verified


def test_exact_production_profile_ids_match_the_frozen_registry_baseline():
    expected = _load(REGISTRY_IDS_PATH)
    actual = []
    for group in EMBEDDING_REGISTRY.groups:
        for profile in group.capabilities["profiles"]:
            actual.append(
                {
                    "canonical_model_id": group.canonical_model_id,
                    "profile_id": profile["profile_id"],
                    "verification_status": profile["verification"]["status"],
                    "index_compatibility_id": profile["index_compatibility_id"],
                    "query_compatibility_id": profile["query_compatibility_id"],
                }
            )
    actual.sort(key=lambda item: item["profile_id"])
    assert expected["fixture_version"] == 1
    assert expected["schema_version"] == 1
    assert expected["profile_count"] == EMBEDDING_REGISTRY.profile_count == 13
    assert actual == expected["profiles"]


def test_registry_is_derived_from_the_single_loaded_catalog_and_is_immutable():
    rebuilt = build_embedding_registry(MODEL_CATALOG)

    assert rebuilt.catalog_digest == MODEL_CATALOG.catalog_digest
    assert rebuilt.groups == EMBEDDING_REGISTRY.groups
    assert rebuilt.profile_count == EMBEDDING_REGISTRY.profile_count == 13

    detached = rebuilt.groups[0].capabilities
    detached["profiles"][0]["dimensions"] = 1
    assert rebuilt.groups[0].capabilities["profiles"][0]["dimensions"] != 1
    with pytest.raises(TypeError):
        rebuilt.groups[0]._capabilities["schema_version"] = 2
    with pytest.raises(AttributeError, match="immutable"):
        rebuilt._groups = ()


def test_pure_builder_reflects_a_controlled_manifest_change_without_model_code():
    manifests = _production_manifest_documents()
    target_manifest = next(
        manifest
        for manifest in manifests
        if manifest["canonical_model_id"] == "colbert-xm:latest"
    )
    target_profile = next(
        profile
        for profile in target_manifest["profiles"]
        if profile["id"] == "kiron-colbert-xm-multivector-v1"
    )
    target_profile["metadata"]["dimensions"] = 129
    target_profile["metadata"]["pipeline"]["parameters"][
        "projection_dimensions"
    ] = 129

    changed_catalog = ModelCatalog.from_manifests(manifests)
    changed_registry = build_embedding_registry(changed_catalog)
    changed_profile = changed_registry.profile(
        "kiron-colbert-xm-multivector-v1"
    )
    original_profile = EMBEDDING_REGISTRY.profile(
        "kiron-colbert-xm-multivector-v1"
    )

    assert changed_profile is not None
    assert original_profile is not None
    assert changed_profile["dimensions"] == 129
    assert changed_profile["pipeline"]["parameters"][
        "projection_dimensions"
    ] == 129
    assert changed_profile != original_profile
    assert changed_registry.catalog_digest != EMBEDDING_REGISTRY.catalog_digest


def test_non_embedding_catalog_groups_are_not_projected_as_embedding_models():
    expanded_catalog = ModelCatalog.from_manifests(
        [*_production_manifest_documents(), _non_embedding_manifest()]
    )
    expanded_registry = build_embedding_registry(expanded_catalog)

    assert len(expanded_catalog.groups) == 14
    assert len(expanded_registry.groups) == 8
    assert expanded_registry.profile_count == 13
    assert expanded_registry.resolve("catalog-only-reranker") is None
    assert expanded_registry.groups == EMBEDDING_REGISTRY.groups


def test_embedding_projection_rejects_unknown_profile_and_backend_metadata():
    manifest = _production_manifest_documents()[0]
    manifest["profiles"][0]["metadata"]["unexpected_adr_fact"] = True
    with pytest.raises(EmbeddingContractError, match="unknown fields"):
        build_embedding_registry(ModelCatalog.from_manifests([manifest]))

    manifest = _production_manifest_documents()[0]
    manifest["deployments"][0]["backend"]["parameters"][
        "unexpected_runtime_fact"
    ] = True
    with pytest.raises(EmbeddingContractError, match="unknown fields"):
        build_embedding_registry(ModelCatalog.from_manifests([manifest]))


def test_resolution_is_explicit_and_never_heuristic():
    assert EMBEDDING_REGISTRY.resolve("nomic-embed-text") is not None
    assert EMBEDDING_REGISTRY.resolve("nomic-embed-text:latest") is not None
    assert EMBEDDING_REGISTRY.resolve("vendor/nomic-embed-text:latest") is None
    assert EMBEDDING_REGISTRY.resolve("NOMIC-EMBED-TEXT") is None
    assert EMBEDDING_REGISTRY.resolve("e5-mistral-7b-instruct") is not None
    assert EMBEDDING_REGISTRY.resolve("something-e5-mistral-7b-instruct") is None


def test_runtime_registry_is_strict_only_for_role_sensitive_profiles():
    for group in EMBEDDING_REGISTRY.groups:
        for profile in group.capabilities["profiles"]:
            input_type = profile["input_type"]
            if input_type["role_sensitive"] is True:
                assert input_type["required"] is True
                assert input_type["default"] is None
                assert input_type["missing_role_behavior"] == "reject"
            else:
                assert input_type["required"] is False


def test_request_role_resolution_is_contract_driven_and_fail_closed():
    missing = resolve_profile_input_type("nomic-embed-text", "/api/embed", None)
    assert missing is not None and missing.accepted is False
    assert input_type_error_payload(missing) == {
        "error": {
            "code": "missing_required_input_type",
            "model": "nomic-embed-text:latest",
            "profile_id": "kiron-nomic-dense-v1",
            "field_path": "/input_type",
            "supported": ["search_document", "search_query"],
        },
        "valid_input_types": [
            "classification",
            "clustering",
            "search_document",
            "search_query",
        ],
    }

    for role in ("search_document", "search_query", "classification", "clustering"):
        decision = resolve_profile_input_type(
            "nomic-embed-text:latest", "/api/embed", role
        )
        assert decision is not None and decision.accepted is True
        assert decision.canonical_input_type == role

    alias = resolve_profile_input_type(
        "colbert-xm", "/api/embed_colbert", "query"
    )
    assert alias is not None and alias.accepted is True
    assert alias.canonical_input_type == "search_query"

    unsupported = resolve_profile_input_type(
        "mankei-326m-embedder", "/api/embed", "classification"
    )
    assert unsupported is not None and unsupported.accepted is False
    assert unsupported.error_code == "unsupported_input_type"

    independent = resolve_profile_input_type("bge-m3", "/api/embed", None)
    assert independent is not None and independent.accepted is True
    assert independent.canonical_input_type is None

    assert resolve_profile_input_type("bge-m3", "/api/embed_late", None) is None


def test_no_current_cross_backend_pair_is_an_implicit_fallback():
    for group in EMBEDDING_REGISTRY.groups:
        profiles = group.capabilities["profiles"]
        for source in profiles:
            for target in profiles:
                if source is not target and source["backend"] != target["backend"]:
                    assert fallback_is_compatible(source, target) is False


def test_registry_rejects_duplicate_aliases_and_inconsistent_ids():
    original = EMBEDDING_REGISTRY.groups[0]
    duplicate_alias_group = EmbeddingModelGroup(
        canonical_model_id="other:latest",
        aliases=(original.aliases[0],),
        service_model=None,
        ollama_model=None,
        capabilities=finalize_capabilities(
            {
                **copy.deepcopy(original.capabilities),
                "canonical_model_id": "other:latest",
                "aliases": [original.aliases[0]],
            }
        ),
    )
    with pytest.raises(EmbeddingContractError, match="duplicate explicit model name"):
        EmbeddingProfileRegistry((original, duplicate_alias_group))

    corrupted = copy.deepcopy(original.capabilities)
    verified = next(
        profile
        for profile in corrupted["profiles"]
        if profile["verification"]["status"] == "verified"
    )
    verified["index_compatibility_id"] = "sha256:" + "0" * 64
    bad_group = EmbeddingModelGroup(
        canonical_model_id=original.canonical_model_id,
        aliases=original.aliases,
        service_model=original.service_model,
        ollama_model=original.ollama_model,
        capabilities=corrupted,
    )
    with pytest.raises(EmbeddingContractError, match="normative preimage"):
        EmbeddingProfileRegistry((bad_group,))


def _service_tags() -> dict:
    return {
        "models": [
            {
                "name": group.service_model,
                "model": group.service_model,
                "service_extension": group.canonical_model_id,
                "kiron_capabilities": copy.deepcopy(group.capabilities),
            }
            for group in EMBEDDING_REGISTRY.service_groups
        ]
    }


def _ollama_tags() -> dict:
    return {
        "models": [
            {"name": "chat-model:latest", "model": "chat-model:latest", "size": 7},
            *(
                {
                    "name": group.ollama_model,
                    "model": group.ollama_model,
                    "ollama_extension": group.canonical_model_id,
                }
                for group in EMBEDDING_REGISTRY.ollama_groups
            ),
        ],
        "ollama_root_extension": True,
    }


def test_discovery_merge_keeps_one_complete_row_per_group_and_all_extensions():
    merged = merge_discovery_tags(_ollama_tags(), _service_tags())
    assert merged["ollama_root_extension"] is True
    assert merged["models"][0]["name"] == "chat-model:latest"
    vector_rows = merged["models"][1:]
    assert len(vector_rows) == len(EMBEDDING_REGISTRY.groups)
    assert {row["name"] for row in vector_rows} == {
        group.canonical_model_id for group in EMBEDDING_REGISTRY.groups
    }
    for row in vector_rows:
        group = EMBEDDING_REGISTRY.require(row["name"])
        assert row["kiron_capabilities"] == group.capabilities
        if group.service_model is not None:
            assert row["service_extension"] == group.canonical_model_id
        if group.ollama_model is not None:
            assert row["ollama_extension"] == group.canonical_model_id


def test_discovery_merge_and_show_preserve_open_numeric_capability_extensions():
    group = EMBEDDING_REGISTRY.require("nomic-embed-text")
    extended = copy.deepcopy(group.capabilities)
    extended["x_runtime_score"] = 0.5
    extended["profiles"][0]["pipeline"]["tokenizer"]["x_ratio"] = 0.5

    service = _service_tags()
    service_row = next(
        row for row in service["models"] if row["name"] == group.service_model
    )
    service_row["kiron_capabilities"] = copy.deepcopy(extended)
    merged = merge_discovery_tags(_ollama_tags(), service)
    tags_row = next(
        row for row in merged["models"] if row["name"] == group.canonical_model_id
    )
    shown = attach_show_capabilities(
        {"kiron_capabilities": copy.deepcopy(extended)}, group.canonical_model_id
    )

    assert tags_row["kiron_capabilities"]["x_runtime_score"] == 0.5
    assert (
        tags_row["kiron_capabilities"]["profiles"][0]["pipeline"]["tokenizer"][
            "x_ratio"
        ]
        == 0.5
    )
    assert canonical_json(tags_row["kiron_capabilities"]) == canonical_json(
        shown["kiron_capabilities"]
    )


def test_discovery_and_show_fail_closed_on_malformed_or_inconsistent_registry_data():
    service = _service_tags()
    service["models"][0]["kiron_capabilities"]["schema_version"] = 2
    with pytest.raises(EmbeddingContractError):
        merge_discovery_tags(_ollama_tags(), service)

    group = EMBEDDING_REGISTRY.groups[0]
    bad_show = {"kiron_capabilities": copy.deepcopy(group.capabilities)}
    bad_show["kiron_capabilities"]["profiles"][0]["profile_id"] = "changed"
    with pytest.raises(EmbeddingContractError):
        attach_show_capabilities(bad_show, group.canonical_model_id)

    unregistered_capability = _ollama_tags()
    unregistered_capability["models"][0]["kiron_capabilities"] = copy.deepcopy(
        group.capabilities
    )
    with pytest.raises(EmbeddingContractError, match="unregistered row"):
        merge_discovery_tags(unregistered_capability, _service_tags())


def test_show_capabilities_are_byte_identical_to_tags_capabilities():
    merged = merge_discovery_tags(_ollama_tags(), _service_tags())
    for row in merged["models"][1:]:
        shown = attach_show_capabilities({"details": {"family": "test"}}, row["name"])
        assert canonical_json(shown["kiron_capabilities"]) == canonical_json(
            row["kiron_capabilities"]
        )
