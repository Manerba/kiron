"""Publish-job contract validation and Kitt-compatible signature checks."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


PUBLISH_JOB_SPEC_SCHEMA_VERSION = "kitt_publish_job_spec_v1"
PUBLISH_JOB_TYPE = "ollama_publish"
PUBLISH_RESULT_SCHEMA_VERSION = "publish_results.v1"
PUBLISH_SPEC_SIGNATURE_PURPOSE = "kitt_publish_spec_signature_v1"
PUBLISH_SPEC_SIGNATURE_ALGORITHM = "ed25519"
PUBLISH_REQUIRED_ARTIFACT_ROLES = frozenset(
    {"gguf", "ollama_modelfile", "run_lock"}
)

PUBLISH_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "job_type",
        "publish_job_uid",
        "publish_intent_uid",
        "model_version_uid",
        "training_run_uid",
        "source_worker_job_uid",
        "source_training_job_uid",
        "target",
        "approval",
        "artifacts",
        "lineage",
        "publish_spec_hash_sha256",
        "signature_envelope",
        "capability_snapshot_hash_sha256",
        "labels",
        "metadata",
    }
)
PUBLISH_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "job_type",
        "publish_job_uid",
        "publish_intent_uid",
        "model_version_uid",
        "training_run_uid",
        "target",
        "approval",
        "artifacts",
        "lineage",
        "publish_spec_hash_sha256",
        "signature_envelope",
        "capability_snapshot_hash_sha256",
    }
)
SIGNATURE_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "algorithm",
        "key_id",
        "publish_spec_hash_sha256",
        "signature_digest_sha256",
        "signed_payload_hash_sha256",
        "signature_base64url",
        "purpose",
    }
)
SIGNATURE_ENVELOPE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "algorithm",
        "key_id",
        "publish_spec_hash_sha256",
        "signature_digest_sha256",
        "signed_payload_hash_sha256",
        "signature_base64url",
    }
)

HEX_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
SAFE_MODEL_REF_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?"
    r"(?::[A-Za-z0-9][A-Za-z0-9_.-]{0,63})?$"
)
SAFE_MODEL_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$"
)
SAFE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAFE_KEY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
SECRET_KEY_RE = re.compile(
    r"(?i)(authorization|bearer|token|secret|credential|password|passwd|"
    r"private[_-]?key|api[_-]?key|access[_-]?key|session[_-]?key)"
)
RAW_TEXT_KEY_RE = re.compile(
    r"(?i)(^|[_:-])(raw[_:-]?)?(prompt|response|completion|cp[_:-]?text|"
    r"cp[_:-]?raw)([_:-]|$)"
)
DISALLOWED_STRING_RE = re.compile(r"(\s|\\|://|\?|#|\.\.)")
DSN_RE = re.compile(
    r"(?i)^[A-Za-z][A-Za-z0-9+.-]*://|"
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:[+]srv)?|redis|amqp|"
    r"amqps|mqtt|kafka|sqlite|mssql|oracle|file)://"
)
SECRET_VALUE_PARTS = (
    "authorization",
    "bearer",
    "token",
    "secret",
    "credential",
    "password",
    "passwd",
    "private_key",
    "private-key",
    "privatekey",
    "api_key",
    "api-key",
    "apikey",
    "access_key",
    "access-key",
    "accesskey",
    "session_key",
    "session-key",
    "sessionkey",
)


class PublishSpecError(ValueError):
    """Publish-spec validation failed with a public, redacted reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class PublishVerifyKey:
    key_id: str
    public_key: bytes
    active: bool = True
    not_before: datetime | None = None
    not_after: datetime | None = None

    def available_at(self, now: datetime) -> bool:
        if not self.active:
            return False
        if self.not_before is not None and now < _normalize_now(self.not_before):
            return False
        if self.not_after is not None and now >= _normalize_now(self.not_after):
            return False
        return True


@dataclass(frozen=True, slots=True)
class PublishVerifyKeyring:
    current: PublishVerifyKey
    previous: PublishVerifyKey | None = None

    def key_for_id(self, key_id: str, *, now: datetime | None = None) -> PublishVerifyKey:
        now = _normalize_now(now or datetime.now(timezone.utc))
        for key in (self.current, self.previous):
            if key is not None and key.key_id == key_id:
                if key.available_at(now):
                    return key
                raise PublishSpecError("publish_spec_signature_key_expired")
        raise PublishSpecError("publish_spec_signature_key_unknown")


def is_publish_job_spec(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == PUBLISH_JOB_SPEC_SCHEMA_VERSION
        and value.get("job_type") == PUBLISH_JOB_TYPE
    )


def validate_publish_job_spec_shape(job_spec: object) -> None:
    if not isinstance(job_spec, dict):
        raise PublishSpecError("publish_spec_shape_invalid")
    missing = PUBLISH_REQUIRED_TOP_LEVEL_KEYS - set(job_spec)
    if missing:
        raise PublishSpecError("publish_spec_shape_invalid")
    if set(job_spec) - PUBLISH_TOP_LEVEL_KEYS:
        raise PublishSpecError("publish_spec_shape_invalid")
    for key in job_spec:
        _reject_secret_key(key)
    if job_spec.get("schema_version") != PUBLISH_JOB_SPEC_SCHEMA_VERSION:
        raise PublishSpecError("publish_spec_shape_invalid")
    if job_spec.get("job_type") != PUBLISH_JOB_TYPE:
        raise PublishSpecError("publish_spec_job_type_invalid")
    for key in (
        "publish_job_uid",
        "publish_intent_uid",
        "model_version_uid",
        "training_run_uid",
        "capability_snapshot_hash_sha256",
    ):
        value = job_spec.get(key)
        if key.endswith("_hash_sha256"):
            if not _is_sha256(value):
                raise PublishSpecError("publish_spec_shape_invalid")
        elif not _is_safe_ref(value):
            raise PublishSpecError("publish_spec_shape_invalid")
    if "source_worker_job_uid" not in job_spec and "source_training_job_uid" not in job_spec:
        raise PublishSpecError("publish_spec_shape_invalid")
    for key in ("source_worker_job_uid", "source_training_job_uid"):
        if key in job_spec and not _is_safe_ref(job_spec[key]):
            raise PublishSpecError("publish_spec_shape_invalid")
    _validate_target(job_spec["target"])
    _validate_approval(job_spec["approval"])
    _validate_artifacts(job_spec["artifacts"])
    _validate_lineage(job_spec["lineage"])
    _validate_signature_envelope_shape(job_spec.get("signature_envelope"))
    if "labels" in job_spec:
        _validate_code_map(job_spec["labels"])
    if "metadata" in job_spec:
        _validate_code_map(job_spec["metadata"])


def verify_publish_spec_signature(
    job_spec: Mapping[str, Any],
    *,
    keyring: PublishVerifyKeyring | None,
    now: datetime | None = None,
) -> None:
    validate_publish_job_spec_shape(dict(job_spec))
    if keyring is None:
        raise PublishSpecError("publish_spec_signature_key_unavailable")
    envelope = job_spec.get("signature_envelope")
    if not isinstance(envelope, Mapping):
        raise PublishSpecError("publish_spec_signature_required")

    expected_publish_hash = publish_spec_hash(job_spec)
    if job_spec.get("publish_spec_hash_sha256") != expected_publish_hash:
        raise PublishSpecError("publish_spec_hash_mismatch")
    if envelope.get("publish_spec_hash_sha256") != expected_publish_hash:
        raise PublishSpecError("publish_spec_hash_mismatch")

    expected_payload_hash = signed_publish_payload_hash(
        job_spec,
        publish_spec_hash_sha256=expected_publish_hash,
    )
    if envelope.get("signed_payload_hash_sha256") != expected_payload_hash:
        raise PublishSpecError("publish_spec_signed_payload_hash_mismatch")

    key_id = envelope["key_id"]
    try:
        key = keyring.key_for_id(key_id, now=now)
    except PublishSpecError:
        raise
    except Exception as exc:
        raise PublishSpecError("publish_spec_signature_key_unavailable") from exc
    signature = _base64url_decode_signature(envelope.get("signature_base64url"))
    if hashlib.sha256(signature).hexdigest() != envelope.get("signature_digest_sha256"):
        raise PublishSpecError("publish_spec_signature_invalid")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(key.public_key)
        public_key.verify(signature, expected_payload_hash.encode("utf-8"))
    except (InvalidSignature, ValueError) as exc:
        raise PublishSpecError("publish_spec_signature_invalid") from exc


def publish_spec_hash(job_spec: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in job_spec.items()
        if key not in {"publish_spec_hash_sha256", "signature_envelope"}
    }
    return _canonical_json_hash(payload)


def signed_publish_payload_hash(
    job_spec: Mapping[str, Any],
    *,
    publish_spec_hash_sha256: str | None = None,
) -> str:
    publish_hash = publish_spec_hash_sha256 or publish_spec_hash(job_spec)
    payload = {
        "schema_version": PUBLISH_SPEC_SIGNATURE_PURPOSE,
        "algorithm": PUBLISH_SPEC_SIGNATURE_ALGORITHM,
        "publish_job_uid": job_spec.get("publish_job_uid"),
        "publish_intent_uid": job_spec.get("publish_intent_uid"),
        "model_version_uid": job_spec.get("model_version_uid"),
        "training_run_uid": job_spec.get("training_run_uid"),
        "source_worker_job_uid": job_spec.get("source_worker_job_uid"),
        "source_training_job_uid": job_spec.get("source_training_job_uid"),
        "publish_spec_hash_sha256": publish_hash,
    }
    return _canonical_json_hash(payload)


def signature_digest_sha256(
    *,
    signature: bytes,
) -> str:
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise PublishSpecError("publish_spec_signature_invalid")
    return hashlib.sha256(signature).hexdigest()


def build_signature_envelope(
    job_spec: Mapping[str, Any],
    *,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> dict[str, str]:
    """Build a secret-free Ed25519 envelope for tests and controlled fixtures."""

    if SAFE_KEY_ID_RE.fullmatch(key_id) is None:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    publish_hash = publish_spec_hash(job_spec)
    payload_hash = signed_publish_payload_hash(
        job_spec,
        publish_spec_hash_sha256=publish_hash,
    )
    signature = private_key.sign(payload_hash.encode("utf-8"))
    return {
        "schema_version": PUBLISH_SPEC_SIGNATURE_PURPOSE,
        "algorithm": PUBLISH_SPEC_SIGNATURE_ALGORITHM,
        "key_id": key_id,
        "publish_spec_hash_sha256": publish_hash,
        "signature_digest_sha256": signature_digest_sha256(signature=signature),
        "signed_payload_hash_sha256": payload_hash,
        "signature_base64url": _base64url_encode(signature),
    }


def publish_artifacts_by_role(job_spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = job_spec.get("artifacts")
    if not isinstance(artifacts, list):
        raise PublishSpecError("artifact_missing")
    result: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise PublishSpecError("artifact_missing")
        role = artifact.get("role")
        if not isinstance(role, str) or role in result:
            raise PublishSpecError("artifact_role_mismatch")
        result[role] = artifact
    return result


def publish_target_ref(job_spec: Mapping[str, Any]) -> str:
    target = job_spec.get("target")
    if not isinstance(target, Mapping) or not isinstance(target.get("target_ref"), str):
        raise PublishSpecError("publish_target_invalid")
    return target["target_ref"]


def source_worker_job_uid(job_spec: Mapping[str, Any]) -> str:
    value = job_spec.get("source_worker_job_uid") or job_spec.get("source_training_job_uid")
    if not isinstance(value, str):
        raise PublishSpecError("publish_spec_shape_invalid")
    return value


def _validate_target(value: object) -> None:
    if not isinstance(value, dict):
        raise PublishSpecError("publish_target_invalid")
    if set(value) != {"target_ref", "ollama_model_name", "ollama_tag"}:
        raise PublishSpecError("publish_target_invalid")
    target_ref = value["target_ref"]
    model_name = value["ollama_model_name"]
    tag = value["ollama_tag"]
    if (
        not isinstance(target_ref, str)
        or not isinstance(model_name, str)
        or not isinstance(tag, str)
        or SAFE_MODEL_REF_RE.fullmatch(target_ref) is None
        or SAFE_MODEL_NAME_RE.fullmatch(model_name) is None
        or SAFE_TAG_RE.fullmatch(tag) is None
    ):
        raise PublishSpecError("publish_target_invalid")
    if target_ref != f"{model_name}:{tag}":
        raise PublishSpecError("publish_target_invalid")
    for item in (target_ref, model_name, tag):
        if not _is_safe_model_string(item):
            raise PublishSpecError("publish_target_invalid")


def _validate_approval(value: object) -> None:
    if not isinstance(value, dict):
        raise PublishSpecError("publish_intent_required")
    if set(value) - {
        "eval_gate_status",
        "approval_status",
        "approved",
        "model_card_ref",
        "approval_ref",
        "operator_decision_ref",
        "human_approval_ref",
    }:
        raise PublishSpecError("publish_intent_required")
    approved = value.get("approved") is True or value.get("approval_status") == "approved"
    evaluated = value.get("eval_gate_status") in {"passed", "passed_with_warnings"}
    if not approved or not evaluated:
        raise PublishSpecError("publish_intent_not_approved")
    for key, child in value.items():
        if isinstance(child, bool):
            continue
        if not _is_safe_ref(child):
            raise PublishSpecError("publish_intent_required")


def _validate_artifacts(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise PublishSpecError("artifact_missing")
    by_role: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise PublishSpecError("artifact_missing")
        allowed = {
            "role",
            "artifact_ref",
            "sha256",
            "size_bytes",
            "source_worker_job_uid",
            "source_training_job_uid",
            "training_run_uid",
            "model_version_uid",
        }
        if set(item) - allowed:
            raise PublishSpecError("artifact_role_mismatch")
        role = item.get("role")
        if not isinstance(role, str) or role in by_role:
            raise PublishSpecError("artifact_role_mismatch")
        if role not in PUBLISH_REQUIRED_ARTIFACT_ROLES:
            raise PublishSpecError("artifact_role_mismatch")
        if not _is_safe_ref(item.get("artifact_ref")):
            raise PublishSpecError("artifact_missing")
        if not _is_sha256(item.get("sha256")):
            raise PublishSpecError("artifact_hash_mismatch")
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise PublishSpecError("artifact_size_mismatch")
        for ref_key in (
            "source_worker_job_uid",
            "source_training_job_uid",
            "training_run_uid",
            "model_version_uid",
        ):
            if ref_key in item and not _is_safe_ref(item[ref_key]):
                raise PublishSpecError("artifact_role_mismatch")
        by_role[role] = item
    if PUBLISH_REQUIRED_ARTIFACT_ROLES - set(by_role):
        raise PublishSpecError("artifact_missing")


def _validate_lineage(value: object) -> None:
    if not isinstance(value, dict):
        raise PublishSpecError("lineage_mismatch")
    allowed = {
        "parent_model_ref",
        "base_model_ref",
        "parent_model_version_uid",
        "lineage_kind",
        "expected_lineage",
    }
    if set(value) - allowed:
        raise PublishSpecError("lineage_mismatch")
    parent = value.get("parent_model_ref") or value.get("base_model_ref")
    if not isinstance(parent, str) or SAFE_MODEL_REF_RE.fullmatch(parent) is None:
        raise PublishSpecError("parent_mismatch")
    if not _is_safe_model_string(parent):
        raise PublishSpecError("parent_mismatch")
    for key, child in value.items():
        if key in {"parent_model_ref", "base_model_ref"}:
            continue
        if not _is_safe_ref(child):
            raise PublishSpecError("lineage_mismatch")


def _validate_signature_envelope_shape(value: object) -> None:
    if value is None:
        raise PublishSpecError("publish_spec_signature_required")
    if not isinstance(value, dict):
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    if SIGNATURE_ENVELOPE_REQUIRED_KEYS - set(value) or set(value) - SIGNATURE_ENVELOPE_KEYS:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    if value.get("schema_version") != PUBLISH_SPEC_SIGNATURE_PURPOSE:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    if "purpose" in value and value["purpose"] != PUBLISH_SPEC_SIGNATURE_PURPOSE:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    if value.get("algorithm") != PUBLISH_SPEC_SIGNATURE_ALGORITHM:
        raise PublishSpecError("publish_spec_signature_algorithm_unsupported")
    if not isinstance(value.get("key_id"), str) or SAFE_KEY_ID_RE.fullmatch(value["key_id"]) is None:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    for key in (
        "publish_spec_hash_sha256",
        "signature_digest_sha256",
        "signed_payload_hash_sha256",
    ):
        if not _is_sha256(value.get(key)):
            raise PublishSpecError("publish_spec_signature_shape_invalid")
    _base64url_decode_signature(value.get("signature_base64url"))


def _validate_code_map(value: object) -> None:
    if not isinstance(value, dict):
        raise PublishSpecError("publish_spec_shape_invalid")
    for key, child in value.items():
        if not isinstance(key, str) or SAFE_CODE_RE.fullmatch(key) is None:
            raise PublishSpecError("publish_spec_shape_invalid")
        _reject_secret_key(key)
        if isinstance(child, bool):
            continue
        if (
            isinstance(child, int)
            and not isinstance(child, bool)
            and 0 <= child <= 9_223_372_036_854_775_807
        ):
            continue
        if isinstance(child, str) and SAFE_CODE_RE.fullmatch(child) and _is_safe_string(child):
            continue
        raise PublishSpecError("publish_spec_shape_invalid")


def _canonical_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or value.strip() == "":
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise PublishSpecError("publish_spec_signature_shape_invalid") from exc
    if len(decoded) != 64:
        raise PublishSpecError("publish_spec_signature_shape_invalid")
    return decoded


def public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and HEX_SHA256_RE.fullmatch(value) is not None


def _is_safe_ref(value: object) -> bool:
    return isinstance(value, str) and SAFE_REF_RE.fullmatch(value) is not None and _is_safe_string(value)


def _is_safe_string(value: str) -> bool:
    if not value or DISALLOWED_STRING_RE.search(value):
        return False
    lowered = value.lower()
    if any(part in lowered for part in SECRET_VALUE_PARTS):
        return False
    return DSN_RE.search(value) is None


def _is_safe_model_string(value: str) -> bool:
    if not value or any(part in value for part in ("\\", "://", "?", "#", "..")):
        return False
    if value.startswith("/") or "//" in value:
        return False
    lowered = value.lower()
    if any(part in lowered for part in SECRET_VALUE_PARTS):
        return False
    return DSN_RE.search(value) is None


def _reject_secret_key(key: str) -> None:
    if SECRET_KEY_RE.search(key) or RAW_TEXT_KEY_RE.search(key):
        raise PublishSpecError("publish_spec_shape_invalid")


def _normalize_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
