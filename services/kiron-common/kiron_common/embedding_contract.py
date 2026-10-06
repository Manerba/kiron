"""Normative KIron embedding capability and compatibility-ID primitives.

This module is production code shared by the discovery producers.  Complete
capability objects use the full RFC 8785/JCS I-JSON domain, while compatibility
preimages use ADR-0009's deliberately narrower safe-integer-only V1 domain.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = 1
ROLES = ("search_document", "search_query")
DOMAINS = {
    "search_document": "kiron.embedding.index_compatibility_id",
    "search_query": "kiron.embedding.query_compatibility_id",
}
I_JSON_SAFE_INTEGER = 9_007_199_254_740_991
JCS_INVALID_UNICODE_SCALAR = "jcs_invalid_unicode_scalar"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ENDPOINT_FOR_KIND = {
    "dense": "/api/embed",
    "late_chunking": "/api/embed_late",
    "multi_vector": "/api/embed_colbert",
}


class JCSCanonicalizationError(ValueError):
    """Controlled fail-closed error for an invalid JCS or V1-preimage value."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class EmbeddingContractError(ValueError):
    """A malformed capability object that must not be partially advertised."""

    reason_code = "embedding_discovery_registry_invalid"

    def __init__(self, field_path: str, detail: str) -> None:
        super().__init__(f"{self.reason_code} at {field_path}: {detail}")
        self.field_path = field_path
        self.detail = detail


def _fail(path: str, detail: str) -> None:
    raise EmbeddingContractError(path, detail)


def _assert_valid_unicode_string(value: str) -> None:
    for code_point in map(ord, value):
        if 0xD800 <= code_point <= 0xDFFF:
            raise JCSCanonicalizationError(
                JCS_INVALID_UNICODE_SCALAR,
                f"lone surrogate U+{code_point:04X}",
            )


def _assert_jcs_v1_value(value: object) -> None:
    if isinstance(value, Mapping):
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


def _assert_jcs_value(value: object) -> None:
    """Validate the complete RFC 8785/I-JSON value domain."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JCSCanonicalizationError(
                    "jcs_non_string_object_name",
                    f"object name has type {type(key).__name__}",
                )
            _assert_valid_unicode_string(key)
            _assert_jcs_value(item)
    elif isinstance(value, list):
        for item in value:
            _assert_jcs_value(item)
    elif isinstance(value, str):
        _assert_valid_unicode_string(value)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise JCSCanonicalizationError(
                "jcs_non_finite_number",
                repr(value),
            )
    else:
        raise JCSCanonicalizationError(
            "jcs_value_type_forbidden",
            type(value).__name__,
        )


def _utf16_sort_key(value: str) -> bytes:
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
    serialized: list[str] = []
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
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
        return "{" + ",".join(
            _jcs_quote(key) + ":" + _serialize_jcs_v1(item)
            for key, item in ordered
        ) + "}"
    raise AssertionError("_assert_jcs_v1_value must reject unsupported values")


def _serialize_jcs_number(value: int | float) -> str:
    """Return the ECMAScript-compatible RFC 8785 representation of a number."""

    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise JCSCanonicalizationError(
            "jcs_non_finite_number",
            repr(value),
        ) from exc
    if not math.isfinite(number):
        raise JCSCanonicalizationError(
            "jcs_non_finite_number",
            repr(value),
        )
    if number == 0:
        return "0"

    rendered = repr(number).lower()
    sign = ""
    if rendered.startswith("-"):
        sign = "-"
        rendered = rendered[1:]

    exponent_text = ""
    exponent = 0
    if "e" in rendered:
        rendered, raw_exponent = rendered.split("e", 1)
        exponent = int(raw_exponent)
        exponent_text = f"e{'+' if exponent >= 0 else ''}{exponent}"

    first, separator, last = rendered.partition(".")
    if last == "0":
        separator = ""
        last = ""

    if 0 < exponent < 21:
        first += last
        last = ""
        separator = ""
        exponent_text = ""
        zero_count = exponent - len(first) + 1
        if zero_count > 0:
            first += "0" * zero_count
    elif -7 < exponent < 0:
        last = first + last
        first = "0"
        separator = "."
        exponent_text = ""
        last = "0" * (-exponent - 1) + last

    return sign + first + separator + last + exponent_text


def _serialize_jcs(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, (int, float)):
        return _serialize_jcs_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_jcs(item) for item in value) + "]"
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
        return "{" + ",".join(
            _jcs_quote(key) + ":" + _serialize_jcs(item)
            for key, item in ordered
        ) + "}"
    raise AssertionError("_assert_jcs_value must reject unsupported values")


def canonical_hash_json(value: object) -> str:
    """Return JCS text for ADR-0009's restricted V1 hash-preimage domain."""

    _assert_jcs_v1_value(value)
    return _serialize_jcs_v1(value)


def canonical_json(value: object) -> str:
    """Return RFC 8785/JCS text for a complete I-JSON capability value."""

    _assert_jcs_value(value)
    return _serialize_jcs(value)


def build_preimage(
    capabilities: Mapping[str, Any], role: str, profile_index: int = 0
) -> dict[str, Any]:
    """Materialize ADR-0009's closed compatibility preimage."""

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
            "implementation_revision": profile["backend"][
                "implementation_revision"
            ],
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
    capabilities: Mapping[str, Any], role: str, profile_index: int = 0
) -> dict[str, str]:
    preimage = build_preimage(capabilities, role, profile_index)
    canonical = canonical_hash_json(preimage)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "canonical_preimage": canonical,
        "sha256": digest,
        "id": f"sha256:{digest}",
    }


def _require_keys(value: Mapping[str, Any], path: str, keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        _fail(path, f"missing required fields: {', '.join(missing)}")


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _require_hash_parameter_mapping(
    value: object, path: str
) -> Mapping[str, Any]:
    parameters = _require_mapping(value, path)
    try:
        _assert_jcs_v1_value(parameters)
    except JCSCanonicalizationError as exc:
        _fail(path, str(exc))
    return parameters


def _require_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be an array")
    return value


def _require_string(value: object, path: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")


def _require_sorted_unique_strings(value: object, path: str) -> list[str]:
    items = _require_list(value, path)
    for index, item in enumerate(items):
        _require_string(item, f"{path}/{index}")
    if len(items) != len(set(items)):
        _fail(path, "must not contain duplicates")
    if items != sorted(items):
        _fail(path, "must be sorted")
    return items


def _validate_digest_files(value: object, path: str) -> None:
    items = _require_list(value, path)
    paths: list[str] = []
    for index, raw in enumerate(items):
        item_path = f"{path}/{index}"
        item = _require_mapping(raw, item_path)
        _require_keys(item, item_path, ("path", "sha256"))
        _require_string(item["path"], f"{item_path}/path")
        if not isinstance(item["sha256"], str) or not _SHA256_HEX.fullmatch(
            item["sha256"]
        ):
            _fail(f"{item_path}/sha256", "must be a lowercase SHA-256 hex digest")
        paths.append(item["path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail(path, "digest files must have unique, path-sorted entries")


def _validate_input_type(raw: object, path: str) -> None:
    value = _require_mapping(raw, path)
    _require_keys(
        value,
        path,
        (
            "supported",
            "additional_task_roles",
            "aliases",
            "role_sensitive",
            "required",
            "default",
            "missing_role_behavior",
        ),
    )
    if value["supported"] != list(ROLES):
        _fail(f"{path}/supported", "must be the two canonical roles in order")
    additional = _require_sorted_unique_strings(
        value["additional_task_roles"], f"{path}/additional_task_roles"
    )
    if set(additional).intersection(ROLES):
        _fail(f"{path}/additional_task_roles", "must exclude canonical roles")

    aliases = _require_list(value["aliases"], f"{path}/aliases")
    alias_names: list[str] = []
    for index, raw_alias in enumerate(aliases):
        alias_path = f"{path}/aliases/{index}"
        alias = _require_mapping(raw_alias, alias_path)
        _require_keys(alias, alias_path, ("alias", "canonical"))
        _require_string(alias["alias"], f"{alias_path}/alias")
        if alias["canonical"] not in ROLES:
            _fail(f"{alias_path}/canonical", "must be a canonical role")
        alias_names.append(alias["alias"])
    if alias_names != sorted(alias_names) or len(alias_names) != len(set(alias_names)):
        _fail(f"{path}/aliases", "must be unique and sorted by alias")

    role_sensitive = value["role_sensitive"]
    required = value["required"]
    default = value["default"]
    behavior = value["missing_role_behavior"]
    if role_sensitive is not True and role_sensitive is not False and role_sensitive is not None:
        _fail(f"{path}/role_sensitive", "must be boolean or null")
    if not isinstance(required, bool):
        _fail(f"{path}/required", "must be boolean")
    if default not in (*ROLES, None):
        _fail(f"{path}/default", "must be a canonical role or null")
    if behavior not in ("use_default", "reject", "no_op", "unknown"):
        _fail(f"{path}/missing_role_behavior", "invalid behavior")
    if role_sensitive is True:
        provider_phase = (
            required is False
            and default == "search_document"
            and behavior == "use_default"
        )
        strict_phase = required is True and default is None and behavior == "reject"
        if not (provider_phase or strict_phase):
            _fail(
                path,
                "role-sensitive V1 profiles must use either the provider or strict phase",
            )
    elif role_sensitive is False:
        if required or default is not None or behavior != "no_op":
            _fail(path, "role-independent providers must use no_op without a default")
    elif required or default is not None or behavior != "unknown":
        _fail(path, "unknown role sensitivity must fail closed")


def _validate_limits(raw: object, path: str, verified: bool) -> None:
    value = _require_mapping(raw, path)
    _require_keys(value, path, ("unit", "counting", "truncation", "overflow", "by_role"))
    if value["unit"] != "tokenizer_tokens":
        _fail(f"{path}/unit", "invalid unit")
    if value["counting"] != "after_server_formatting_including_special_tokens":
        _fail(f"{path}/counting", "invalid counting rule")
    if value["truncation"] not in ("right", "none", "unknown"):
        _fail(f"{path}/truncation", "invalid truncation rule")
    if value["overflow"] not in ("truncate", "late_chunk_fallback", "reject", "unknown"):
        _fail(f"{path}/overflow", "invalid overflow rule")
    by_role = _require_mapping(value["by_role"], f"{path}/by_role")
    if set(by_role) != set(ROLES):
        _fail(f"{path}/by_role", "must contain exactly the canonical roles")
    for role in ROLES:
        limit = by_role[role]
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= I_JSON_SAFE_INTEGER
        ):
            _fail(f"{path}/by_role/{role}", "must be a positive safe integer or null")
        if verified and limit is None:
            _fail(f"{path}/by_role/{role}", "verified profiles require a limit")
    if verified and (value["truncation"], value["overflow"]) not in (
        ("right", "truncate"), ("right", "late_chunk_fallback"), ("none", "reject"),
    ):
        _fail(path, "verified profiles require a known input-limit policy")


def _validate_pipeline(raw: object, path: str, verified: bool) -> None:
    value = _require_mapping(raw, path)
    _require_keys(value, path, ("tokenizer", "formatting", "pooling", "normalization", "parameters"))
    tokenizer = _require_mapping(value["tokenizer"], f"{path}/tokenizer")
    _require_keys(tokenizer, f"{path}/tokenizer", ("repository", "revision"))
    for field in ("repository", "revision"):
        _require_string(
            tokenizer[field], f"{path}/tokenizer/{field}", nullable=not verified
        )

    formatting = _require_mapping(value["formatting"], f"{path}/formatting")
    if set(formatting) != set(ROLES):
        _fail(f"{path}/formatting", "must contain exactly the canonical roles")
    for role in ROLES:
        role_format = _require_mapping(
            formatting[role], f"{path}/formatting/{role}"
        )
        _require_keys(role_format, f"{path}/formatting/{role}", ("template", "parameters"))
        if role_format["template"] is not None and not isinstance(
            role_format["template"], str
        ):
            _fail(f"{path}/formatting/{role}/template", "must be string or null")
        if verified and role_format["template"] is None:
            _fail(f"{path}/formatting/{role}/template", "verified profiles require a template")
        _require_hash_parameter_mapping(
            role_format["parameters"], f"{path}/formatting/{role}/parameters"
        )

    pooling = _require_mapping(value["pooling"], f"{path}/pooling")
    _require_keys(pooling, f"{path}/pooling", ("method", "parameters"))
    if pooling["method"] not in (
        "mean",
        "cls",
        "last_token",
        "late_chunk_mean",
        "token_projection",
        "unknown",
    ):
        _fail(f"{path}/pooling/method", "invalid pooling method")
    _require_hash_parameter_mapping(
        pooling["parameters"], f"{path}/pooling/parameters"
    )
    if verified and pooling["method"] == "unknown":
        _fail(f"{path}/pooling/method", "verified profiles require known pooling")

    normalization = _require_mapping(value["normalization"], f"{path}/normalization")
    _require_keys(normalization, f"{path}/normalization", ("method", "dtype", "scope"))
    if normalization["method"] not in ("l2", "none", "unknown"):
        _fail(f"{path}/normalization/method", "invalid method")
    if normalization["dtype"] not in ("float32", "float16", "bfloat16", "unknown"):
        _fail(f"{path}/normalization/dtype", "invalid dtype")
    if normalization["scope"] not in ("vector", "token", "unknown"):
        _fail(f"{path}/normalization/scope", "invalid scope")
    if verified and "unknown" in normalization.values():
        _fail(f"{path}/normalization", "verified profiles require known normalization")
    _require_hash_parameter_mapping(value["parameters"], f"{path}/parameters")


def _validate_profile(raw: object, path: str) -> str:
    profile = _require_mapping(raw, path)
    _require_keys(
        profile,
        path,
        (
            "profile_id",
            "kind",
            "endpoint",
            "default_for_endpoint",
            "backend",
            "artifact",
            "dimensions",
            "max_input_tokens",
            "output_normalized",
            "similarity",
            "input_type",
            "formatting_owner",
            "pipeline",
            "verification",
            "index_compatibility_id",
            "query_compatibility_id",
        ),
    )
    profile_id = profile["profile_id"]
    if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
        _fail(f"{path}/profile_id", "invalid profile ID")
    if profile["kind"] not in _ENDPOINT_FOR_KIND:
        _fail(f"{path}/kind", "invalid profile kind")
    if profile["endpoint"] != _ENDPOINT_FOR_KIND[profile["kind"]]:
        _fail(f"{path}/endpoint", "does not match profile kind")
    if not isinstance(profile["default_for_endpoint"], bool):
        _fail(f"{path}/default_for_endpoint", "must be boolean")

    verification = _require_mapping(profile["verification"], f"{path}/verification")
    _require_keys(verification, f"{path}/verification", ("status", "evidence", "blocking_reasons"))
    if verification["status"] not in ("verified", "unverified"):
        _fail(f"{path}/verification/status", "invalid verification status")
    verified = verification["status"] == "verified"
    evidence = _require_sorted_unique_strings(
        verification["evidence"], f"{path}/verification/evidence"
    )
    blockers = _require_sorted_unique_strings(
        verification["blocking_reasons"], f"{path}/verification/blocking_reasons"
    )
    if verified and (not evidence or blockers):
        _fail(f"{path}/verification", "verified profiles need evidence and no blockers")
    if not verified and not blockers:
        _fail(f"{path}/verification/blocking_reasons", "unverified profiles need blockers")

    backend = _require_mapping(profile["backend"], f"{path}/backend")
    _require_keys(backend, f"{path}/backend", ("type", "implementation", "implementation_revision"))
    if backend["type"] not in ("kiron_embeddings", "ollama"):
        _fail(f"{path}/backend/type", "invalid backend type")
    _require_string(backend["implementation"], f"{path}/backend/implementation")
    _require_string(
        backend["implementation_revision"],
        f"{path}/backend/implementation_revision",
        nullable=not verified,
    )

    artifact = _require_mapping(profile["artifact"], f"{path}/artifact")
    _require_keys(artifact, f"{path}/artifact", ("repository", "revision", "manifest_digest", "weights", "auxiliary"))
    _require_string(artifact["repository"], f"{path}/artifact/repository", nullable=True)
    _require_string(artifact["revision"], f"{path}/artifact/revision", nullable=True)
    manifest = artifact["manifest_digest"]
    if manifest is not None and (
        not isinstance(manifest, str) or not _SHA256_ID.fullmatch(manifest)
    ):
        _fail(f"{path}/artifact/manifest_digest", "must be a SHA-256 ID or null")
    _validate_digest_files(artifact["weights"], f"{path}/artifact/weights")
    _validate_digest_files(artifact["auxiliary"], f"{path}/artifact/auxiliary")
    weight_paths = [item["path"] for item in artifact["weights"]]
    auxiliary_paths = [item["path"] for item in artifact["auxiliary"]]
    if set(weight_paths).intersection(auxiliary_paths):
        _fail(f"{path}/artifact", "weight and auxiliary paths must not overlap")
    if verified and (
        not artifact["weights"]
        or not (
            (artifact["repository"] and artifact["revision"])
            or artifact["manifest_digest"]
        )
    ):
        _fail(f"{path}/artifact", "verified profiles require pinned weights and provenance")

    dimensions = profile["dimensions"]
    if dimensions is not None and (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or not 1 <= dimensions <= I_JSON_SAFE_INTEGER
    ):
        _fail(f"{path}/dimensions", "must be a positive safe integer or null")
    if verified and dimensions is None:
        _fail(f"{path}/dimensions", "verified profiles require dimensions")
    if (
        profile["output_normalized"] is not True
        and profile["output_normalized"] is not False
        and profile["output_normalized"] is not None
    ):
        _fail(f"{path}/output_normalized", "must be boolean or null")
    if verified and profile["output_normalized"] is None:
        _fail(f"{path}/output_normalized", "verified profiles require normalization status")
    if profile["similarity"] not in ("cosine", "dot_product", "maxsim_cosine", "unknown"):
        _fail(f"{path}/similarity", "invalid similarity")
    if verified and profile["similarity"] == "unknown":
        _fail(f"{path}/similarity", "verified profiles require known similarity")

    _validate_limits(profile["max_input_tokens"], f"{path}/max_input_tokens", verified)
    _validate_input_type(profile["input_type"], f"{path}/input_type")
    if profile["formatting_owner"] != "server":
        _fail(f"{path}/formatting_owner", "must be server")
    _validate_pipeline(profile["pipeline"], f"{path}/pipeline", verified)

    index_id = profile["index_compatibility_id"]
    query_id = profile["query_compatibility_id"]
    if verified:
        if not isinstance(index_id, str) or not _SHA256_ID.fullmatch(index_id):
            _fail(f"{path}/index_compatibility_id", "verified profile requires a SHA-256 ID")
        if not isinstance(query_id, str) or not _SHA256_ID.fullmatch(query_id):
            _fail(f"{path}/query_compatibility_id", "verified profile requires a SHA-256 ID")
    elif index_id is not None or query_id is not None:
        _fail(path, "unverified profiles must have null compatibility IDs")
    return profile_id


def validate_capabilities(capabilities: object) -> None:
    """Validate schema-critical and ADR behavioral invariants fail-closed."""

    value = _require_mapping(capabilities, "/kiron_capabilities")
    _require_keys(value, "/kiron_capabilities", ("schema_version", "canonical_model_id", "aliases", "profiles"))
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, (int, float))
        or schema_version != SCHEMA_VERSION
    ):
        _fail("/kiron_capabilities/schema_version", "unsupported schema version")
    _require_string(value["canonical_model_id"], "/kiron_capabilities/canonical_model_id")
    aliases = _require_sorted_unique_strings(value["aliases"], "/kiron_capabilities/aliases")
    if value["canonical_model_id"] in aliases:
        _fail("/kiron_capabilities/aliases", "canonical ID must not be repeated as an alias")
    profiles = _require_list(value["profiles"], "/kiron_capabilities/profiles")
    if not profiles:
        _fail("/kiron_capabilities/profiles", "must contain at least one profile")

    profile_ids = [
        _validate_profile(profile, f"/kiron_capabilities/profiles/{index}")
        for index, profile in enumerate(profiles)
    ]
    if profile_ids != sorted(profile_ids) or len(profile_ids) != len(set(profile_ids)):
        _fail("/kiron_capabilities/profiles", "profile IDs must be unique and sorted")
    endpoints = {profile["endpoint"] for profile in profiles}
    for endpoint in endpoints:
        defaults = [
            profile
            for profile in profiles
            if profile["endpoint"] == endpoint and profile["default_for_endpoint"]
        ]
        if len(defaults) != 1:
            _fail(
                "/kiron_capabilities/profiles",
                f"endpoint {endpoint} must have exactly one default profile",
            )

    try:
        canonical_json(value)
    except JCSCanonicalizationError as exc:
        _fail("/kiron_capabilities", str(exc))

    for index, profile in enumerate(profiles):
        if profile["verification"]["status"] != "verified":
            continue
        expected_index = compatibility_vector(value, "search_document", index)["id"]
        expected_query = compatibility_vector(value, "search_query", index)["id"]
        if profile["index_compatibility_id"] != expected_index:
            _fail(
                f"/kiron_capabilities/profiles/{index}/index_compatibility_id",
                "does not match the normative preimage",
            )
        if profile["query_compatibility_id"] != expected_query:
            _fail(
                f"/kiron_capabilities/profiles/{index}/query_compatibility_id",
                "does not match the normative preimage",
            )


def finalize_capabilities(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    """Compute IDs for verified profiles, then run mandatory validation."""

    result = copy.deepcopy(dict(capabilities))
    profiles = result.get("profiles")
    if not isinstance(profiles, list):
        _fail("/kiron_capabilities/profiles", "must be an array")
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            _fail(f"/kiron_capabilities/profiles/{index}", "must be an object")
        verification = profile.get("verification")
        if isinstance(verification, dict) and verification.get("status") == "verified":
            try:
                profile["index_compatibility_id"] = compatibility_vector(
                    result, "search_document", index
                )["id"]
                profile["query_compatibility_id"] = compatibility_vector(
                    result, "search_query", index
                )["id"]
            except (KeyError, IndexError, TypeError, JCSCanonicalizationError) as exc:
                _fail(
                    f"/kiron_capabilities/profiles/{index}",
                    f"cannot construct compatibility preimage: {exc}",
                )
    validate_capabilities(result)
    return result


def capabilities_equal(left: object, right: object) -> bool:
    """Compare complete capability objects by their normative JCS bytes."""

    validate_capabilities(left)
    validate_capabilities(right)
    return canonical_json(left) == canonical_json(right)


def fallback_is_compatible(source: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """Return true only for an explicitly proven, byte-compatible fallback."""

    required = ("endpoint", "kind", "index_compatibility_id", "query_compatibility_id")
    if source.get("verification", {}).get("status") != "verified":
        return False
    if target.get("verification", {}).get("status") != "verified":
        return False
    if any(source.get(field) is None or target.get(field) is None for field in required[2:]):
        return False
    return all(source.get(field) == target.get(field) for field in required)
