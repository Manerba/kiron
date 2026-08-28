"""Sprint 7 job metadata validation and response helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import publish_contract


JOB_UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
CLIENT_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
CURSOR_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
LOGICAL_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
MODEL_REF_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}(/[A-Za-z0-9][A-Za-z0-9_.:-]{0,95})?$"
)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
SHORT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
LIMIT_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
LEASE_OWNER_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,127}$")

KITT_RUN_TYPES = {"sft"}
KITT_OUTPUT_ROLES = {
    "adapter",
    "gguf",
    "merged_weights",
    "metrics_jsonl",
    "ollama_modelfile",
    "run_lock",
    "training_log",
}
KITT_REQUIRED_OUTPUT_ROLES = {
    "adapter",
    "gguf",
    "merged_weights",
    "metrics_jsonl",
    "ollama_modelfile",
    "run_lock",
    "training_log",
}
KITT_JOB_SPEC_SCHEMA_VERSION = "kitt_job_spec_v1"
KITT_SUPPORTED_HYPERPARAMETERS = {
    "batch_size",
    "dataset_text_field",
    "eval_steps",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "lora_alpha",
    "lora_dropout",
    "lora_r",
    "lr_scheduler_type",
    "max_batch_size",
    "max_context_tokens",
    "learning_rate",
    "logging_steps",
    "micro_batch_size",
    "num_train_epochs",
    "optimizer",
    "packing",
    "max_seq_length",
    "max_sequence_length",
    "max_steps",
    "optim",
    "per_device_train_batch_size",
    "quantization_mode",
    "save_steps",
    "seed",
    "target_modules",
    "warmup_steps",
    "weight_decay",
}
KITT_TOP_LEVEL_JOB_SPEC_KEYS = {
    "schema_version",
    "run_uid",
    "run_type",
    "training_profile",
    "input_reference",
    "base_or_parent_model",
    "hyperparameters",
    "preprocessing",
    "output_roles",
    "capability_snapshot_hash_sha256",
    "lineage_risks",
    "labels",
    "metadata",
}
KITT_REQUIRED_JOB_SPEC_KEYS = {
    "schema_version",
    "run_uid",
    "run_type",
    "training_profile",
    "input_reference",
    "base_or_parent_model",
    "hyperparameters",
    "output_roles",
    "capability_snapshot_hash_sha256",
}
_SECRET_KEY_RE = re.compile(
    r"(?i)(authorization|bearer|token|secret|credential|password|passwd|"
    r"private[_-]?key|api[_-]?key|access[_-]?key|session[_-]?key|key[_-]?id)"
)
_RAW_TEXT_KEY_RE = re.compile(
    r"(?i)(^|[_:-])(raw[_:-]?)?(prompt|response|completion|cp[_:-]?text|cp[_:-]?raw)([_:-]|$)"
)
_SECRET_VALUE_PARTS = (
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
    "key_id",
    "key-id",
    "keyid",
)
_DISALLOWED_STRING_RE = re.compile(r"(\s|/|\\|://|\?|#|\.\.)")
_DSN_RE = re.compile(
    r"(?i)^[A-Za-z][A-Za-z0-9+.-]*://|"
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:[+]srv)?|redis|amqp|"
    r"amqps|mqtt|kafka|sqlite|mssql|oracle|file)://"
)


class JobValidationError(ValueError):
    """A job metadata value violates the Sprint 6 persistence allowlist."""


class JobConflictError(RuntimeError):
    pass


class JobNotFoundError(RuntimeError):
    pass


def validate_job_uid(job_uid: str) -> bool:
    return JOB_UID_RE.fullmatch(job_uid) is not None


def validate_idempotency_key(value: object) -> bool:
    return isinstance(value, str) and CLIENT_KEY_RE.fullmatch(value) is not None


def validate_reason_code(value: object) -> bool:
    return isinstance(value, str) and REASON_CODE_RE.fullmatch(value) is not None


def validate_cursor(value: object) -> bool:
    return isinstance(value, str) and CURSOR_RE.fullmatch(value) is not None


def validate_last_checkpoint_ref(value: object) -> bool:
    return isinstance(value, str) and _is_safe_logical_ref(value)


def validate_lease_owner(value: object) -> bool:
    return (
        isinstance(value, str)
        and LEASE_OWNER_RE.fullmatch(value) is not None
        and _is_safe_string(value)
    )


def canonicalize_job_spec(job_spec: dict[str, Any]) -> tuple[str, str]:
    validate_job_spec(job_spec)
    canonical = json.dumps(
        job_spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def job_spec_hash(job_spec: dict[str, Any]) -> str:
    return canonicalize_job_spec(job_spec)[1]


def validate_job_spec(job_spec: object) -> None:
    if not isinstance(job_spec, dict):
        raise JobValidationError("job_spec must be an object")
    if job_spec.get("schema_version") == KITT_JOB_SPEC_SCHEMA_VERSION:
        _validate_kitt_job_spec(job_spec)
        return
    if publish_contract.is_publish_job_spec(job_spec):
        try:
            publish_contract.validate_publish_job_spec_shape(job_spec)
        except publish_contract.PublishSpecError as exc:
            raise JobValidationError(exc.reason_code) from exc
        return
    raise JobValidationError("job_spec schema_version is invalid")


def kitt_job_spec(job: Any) -> dict[str, Any]:
    raw = getattr(job, "canonical_job_spec_json", None)
    if not isinstance(raw, str):
        raise JobValidationError("canonical job_spec is unavailable")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JobValidationError("canonical job_spec is invalid") from exc
    if not isinstance(data, dict):
        raise JobValidationError("canonical job_spec is invalid")
    if data.get("schema_version") != KITT_JOB_SPEC_SCHEMA_VERSION:
        raise JobValidationError("job_spec schema_version is invalid")
    _validate_kitt_job_spec(data)
    return data


def publish_job_spec(job: Any) -> dict[str, Any]:
    raw = getattr(job, "canonical_job_spec_json", None)
    if not isinstance(raw, str):
        raise JobValidationError("canonical job_spec is unavailable")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JobValidationError("canonical job_spec is invalid") from exc
    if not publish_contract.is_publish_job_spec(data):
        raise JobValidationError("job_spec job_type is invalid")
    try:
        publish_contract.validate_publish_job_spec_shape(data)
    except publish_contract.PublishSpecError as exc:
        raise JobValidationError(exc.reason_code) from exc
    return data


def job_type_from_spec(job_spec: Mapping[str, Any] | dict[str, Any]) -> str:
    if job_spec.get("schema_version") == KITT_JOB_SPEC_SCHEMA_VERSION:
        return "sft"
    if publish_contract.is_publish_job_spec(job_spec):
        return publish_contract.PUBLISH_JOB_TYPE
    raise JobValidationError("job_spec schema_version is invalid")


def job_type(job: Any) -> str:
    value = getattr(job, "job_type", None)
    if value in {"sft", publish_contract.PUBLISH_JOB_TYPE}:
        return value
    raw = getattr(job, "canonical_job_spec_json", None)
    if not isinstance(raw, str):
        raise JobValidationError("canonical job_spec is unavailable")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JobValidationError("canonical job_spec is invalid") from exc
    return job_type_from_spec(data)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def job_data(
    job: Any,
    *,
    execution_enabled: bool = False,
    publish_result: Mapping[str, Any] | None = None,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    data = {
        "job_uid": job.job_uid,
        "run_uid": job.run_uid,
        "job_type": job_type(job),
        "state": job.state,
        "execution_state_version": 2,
        "execution_enabled": execution_enabled,
        "job_spec_hash_sha256": job.job_spec_hash_sha256,
        "capability_hash_sha256": job.capability_hash_sha256,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "accepted_at": job.accepted_at,
        "queued_at": job.queued_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "terminal_at": job.terminal_at,
        "cancel_requested": bool(job.cancel_requested),
        "cancel_reason_code": job.cancel_reason_code,
        "resume_requested": bool(job.resume_requested),
        "resume_reason_code": job.resume_reason_code,
        "resume_count": job.resume_count,
        "resume_limit": job.resume_limit,
        "last_checkpoint_ref": job.last_checkpoint_ref,
        "attempt_count": job.attempt_count,
        "last_failure_code": job.last_failure_code,
        "last_failure_class": job.last_failure_class,
        "runner_kind": job.runner_kind,
        "lease_status": lease_status(job),
        "lease_expires_at": job.lease_expires_at,
    }
    if data["job_type"] == publish_contract.PUBLISH_JOB_TYPE:
        data["publish_status"] = _publish_status(job, publish_result)
        data["target_ref"] = _publish_target_ref(job)
        data["publish_result"] = dict(publish_result) if publish_result else None
    if artifacts:
        data["artifacts"] = [dict(artifact) for artifact in artifacts]
    return data


def logs_data(job: Any, *, execution_enabled: bool = False) -> dict[str, Any]:
    return {
        "job_uid": job.job_uid,
        "items": [],
        "next_cursor": None,
        "truncated": False,
        "execution_enabled": execution_enabled,
    }


def lease_status(job: Any, *, now: str | None = None) -> str:
    if not getattr(job, "lease_expires_at", None):
        return "none"
    now = utc_now() if now is None else now
    return "stale" if job.lease_expires_at < now else "active"


def _publish_status(job: Any, publish_result: Mapping[str, Any] | None) -> str:
    if publish_result:
        return "idempotent" if publish_result.get("idempotent") is True else "published"
    if getattr(job, "state", None) == "succeeded":
        return "publish_result_missing"
    if getattr(job, "state", None) == "failed":
        return "failed"
    if getattr(job, "state", None) == "canceled":
        return "canceled"
    return "pending"


def _publish_target_ref(job: Any) -> str | None:
    try:
        spec = publish_job_spec(job)
        return publish_contract.publish_target_ref(spec)
    except JobValidationError:
        return None


def _validate_code_map(value: object) -> None:
    if not isinstance(value, dict):
        raise JobValidationError("metadata values must be an object")
    for key, child in value.items():
        _validate_nested_key(key)
        if isinstance(child, bool):
            continue
        if (
            isinstance(child, int)
            and not isinstance(child, bool)
            and 0 <= child <= 9_223_372_036_854_775_807
        ):
            continue
        if isinstance(child, str) and SHORT_CODE_RE.fullmatch(child) and _is_safe_string(child):
            continue
        raise JobValidationError("metadata value is not a safe code")


def _validate_kitt_job_spec(job_spec: dict[str, Any]) -> None:
    missing = KITT_REQUIRED_JOB_SPEC_KEYS - set(job_spec)
    if missing:
        raise JobValidationError("job_spec is missing required fields")
    for key in job_spec:
        if not isinstance(key, str) or key not in KITT_TOP_LEVEL_JOB_SPEC_KEYS:
            raise JobValidationError("job_spec contains an unsupported field")
        _reject_secret_key(key)
    if job_spec["schema_version"] != KITT_JOB_SPEC_SCHEMA_VERSION:
        raise JobValidationError("job_spec schema_version is invalid")
    if not isinstance(job_spec["run_uid"], str) or not _is_safe_logical_ref(
        job_spec["run_uid"]
    ):
        raise JobValidationError("run_uid is invalid")
    if (
        not isinstance(job_spec["run_type"], str)
        or job_spec["run_type"] not in KITT_RUN_TYPES
    ):
        raise JobValidationError("run_type is invalid")
    capability_hash = job_spec["capability_snapshot_hash_sha256"]
    if not isinstance(capability_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", capability_hash
    ):
        raise JobValidationError("capability_snapshot_hash_sha256 is invalid")
    output_roles = job_spec["output_roles"]
    if (
        not isinstance(output_roles, list)
        or not output_roles
        or not KITT_REQUIRED_OUTPUT_ROLES.issubset(output_roles)
        or len(set(output_roles)) != len(output_roles)
        or any(
            not isinstance(role, str)
            or role not in KITT_OUTPUT_ROLES
            or not _is_safe_string(role)
            for role in output_roles
        )
    ):
        raise JobValidationError("output_roles is invalid")
    _validate_training_profile(job_spec["training_profile"])
    _validate_input_reference(job_spec["input_reference"])
    _validate_base_or_parent_model(job_spec["base_or_parent_model"])
    _validate_hyperparameters(job_spec["hyperparameters"])
    for key in (
        "preprocessing",
        "lineage_risks",
        "labels",
        "metadata",
    ):
        if key in job_spec:
            if key in {"labels", "metadata"}:
                _validate_code_map(job_spec[key])
            else:
                _validate_kitt_json_value(job_spec[key])


def _validate_training_profile(value: object) -> None:
    data = _required_object(value, label="training_profile")
    if set(data) != {"profile_uid", "version_label", "profile_hash_sha256"}:
        raise JobValidationError("training_profile is invalid")
    if not isinstance(data["profile_uid"], str) or not _is_safe_logical_ref(
        data["profile_uid"]
    ):
        raise JobValidationError("training_profile is invalid")
    if not isinstance(data["version_label"], str) or not _is_safe_logical_ref(
        data["version_label"]
    ):
        raise JobValidationError("training_profile is invalid")
    profile_hash = data["profile_hash_sha256"]
    if not isinstance(profile_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", profile_hash):
        raise JobValidationError("training_profile is invalid")


def _validate_input_reference(value: object) -> None:
    data = _required_object(value, label="input_reference")
    allowed = {"mode", "dataset_version_uid", "dataset_version_id", "target_phase"}
    if not set(data).issubset(allowed):
        raise JobValidationError("input_reference is invalid")
    if data["mode"] != "dataset_version":
        raise JobValidationError("input_reference is invalid")
    if not isinstance(data["dataset_version_uid"], str) or not UUID_RE.fullmatch(
        data["dataset_version_uid"]
    ):
        raise JobValidationError("input_reference is invalid")
    if "dataset_version_id" in data:
        dataset_version_id = data["dataset_version_id"]
        if (
            isinstance(dataset_version_id, bool)
            or not isinstance(dataset_version_id, int)
            or dataset_version_id < 1
        ):
            raise JobValidationError("input_reference is invalid")
    if "target_phase" in data and (
        not isinstance(data["target_phase"], str)
        or not _is_safe_logical_ref(data["target_phase"])
    ):
        raise JobValidationError("input_reference is invalid")


def _validate_base_or_parent_model(value: object) -> None:
    data = _required_object(value, label="base_or_parent_model")
    allowed = {"external_parent_ref", "parent_model_uid", "parent_model_version_id"}
    if not set(data).issubset(allowed):
        raise JobValidationError("base_or_parent_model is invalid")
    if not isinstance(data["external_parent_ref"], str) or not _is_safe_model_ref(
        data["external_parent_ref"]
    ):
        raise JobValidationError("base_or_parent_model is invalid")
    if "parent_model_uid" in data and (
        not isinstance(data["parent_model_uid"], str)
        or not UUID_RE.fullmatch(data["parent_model_uid"])
    ):
        raise JobValidationError("base_or_parent_model is invalid")
    if "parent_model_version_id" in data:
        version_id = data["parent_model_version_id"]
        if isinstance(version_id, bool) or not isinstance(version_id, int) or version_id < 1:
            raise JobValidationError("base_or_parent_model is invalid")


def _validate_hyperparameters(value: object) -> None:
    data = _required_object(value, label="hyperparameters")
    for key, child in data.items():
        _validate_nested_key(key)
        if key not in KITT_SUPPORTED_HYPERPARAMETERS:
            raise JobValidationError("hyperparameters key is unsupported")
        if child is None:
            raise JobValidationError("hyperparameters value is invalid")
        _validate_kitt_json_value(child)


def _required_object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise JobValidationError(f"{label} is invalid")
    return value


def _validate_kitt_json_value(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 9_223_372_036_854_775_807:
            return
        raise JobValidationError("job_spec integer is invalid")
    if isinstance(value, float):
        if value == value and value not in {float("inf"), float("-inf")}:
            return
        raise JobValidationError("job_spec number is invalid")
    if isinstance(value, str):
        if _is_safe_string(value):
            return
        raise JobValidationError("job_spec string is invalid")
    if isinstance(value, list):
        for item in value:
            _validate_kitt_json_value(item)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_nested_key(key)
            _validate_kitt_json_value(child)
        return
    raise JobValidationError("job_spec value is invalid")


def _validate_nested_key(key: object) -> None:
    if not isinstance(key, str) or LIMIT_KEY_RE.fullmatch(key) is None:
        raise JobValidationError("nested metadata key is invalid")
    _reject_secret_key(key)


def _reject_secret_key(key: str) -> None:
    if _SECRET_KEY_RE.search(key) or _RAW_TEXT_KEY_RE.search(key):
        raise JobValidationError("secret-like job_spec key is not persistable")


def _is_safe_logical_ref(value: str) -> bool:
    return LOGICAL_REF_RE.fullmatch(value) is not None and _is_safe_string(value)


def _is_safe_model_ref(value: str) -> bool:
    if MODEL_REF_RE.fullmatch(value) is None:
        return False
    if any(part in value for part in ("\\", "://", "?", "#", "..")):
        return False
    lowered = value.lower()
    if any(part in lowered for part in _SECRET_VALUE_PARTS):
        return False
    return _DSN_RE.search(value) is None


def _is_safe_string(value: str) -> bool:
    if not value or _DISALLOWED_STRING_RE.search(value):
        return False
    lowered = value.lower()
    if any(part in lowered for part in _SECRET_VALUE_PARTS):
        return False
    if _DSN_RE.search(value):
        return False
    return True
