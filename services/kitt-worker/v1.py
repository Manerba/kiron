"""ADR-0008 /v1 router for kitt-worker metadata endpoints."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Any

from fastapi import APIRouter, Request

import artifact_staging
import capabilities as capability_builder
import config as worker_config
import contract
import job_stubs
import monitoring
import publish_contract
import queue_store


PUT_JOB_BODY_LIMIT = 262144
ACTION_BODY_LIMIT = 8192
ARTIFACT_UPLOAD_BODY_LIMIT = 262144
ARTIFACT_UPLOAD_MAX_BYTES = 131072
ARTIFACT_UPLOAD_ROLES = frozenset({"ollama_modelfile", "run_lock"})

router = APIRouter(prefix="/v1")


@router.get("/health")
async def health(request: Request):
    contract.authenticate(request, required_scope="read")
    snapshot = _current_capability_snapshot(request)
    return contract.success_response(
        request,
        {
            "status": "ok",
            "service": "kitt-worker",
            "mode": "adr-0008-metadata-queue",
            "dispatch_enabled": snapshot["valid_for_scheduling"] is True,
            "execution_enabled": snapshot["execution_enabled"] is True,
        },
    )


@router.get("/heartbeat")
async def heartbeat(request: Request):
    contract.authenticate(request, required_scope="read")
    monitoring.record_heartbeat(request.app.state)
    snapshot = _current_capability_snapshot(request)
    operational_status = snapshot["operational_status"]
    return contract.success_response(
        request,
        {
            "status": "ok",
            "heartbeat_time": job_stubs.utc_now(),
            "dispatch_enabled": snapshot["valid_for_scheduling"] is True,
            "execution_enabled": snapshot["execution_enabled"] is True,
            "active_jobs": operational_status["active_jobs"],
            "queue_depth": operational_status["queue_depth"],
            "queue_status": operational_status["queue_status"],
        },
    )


@router.get("/capabilities")
async def capabilities(request: Request):
    contract.authenticate(request, required_scope="read")
    return contract.success_response(
        request,
        _current_capability_snapshot(request),
    )


@router.put("/jobs/{job_uid}")
async def put_job(job_uid: str, request: Request):
    contract.authenticate(request, required_scope="job_write")
    _validate_job_uid(job_uid)
    body = await _read_limited_body(request, limit=PUT_JOB_BODY_LIMIT)
    _require_json_content_type(request)
    payload = _parse_json_object(body, top_level_status=400)
    _reject_unknown_fields(payload, {"job_spec", "idempotency_key"}, label="job")
    job_spec = payload.get("job_spec")
    if not isinstance(job_spec, dict):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: job_spec",
        )
    if "idempotency_key" in payload and not job_stubs.validate_idempotency_key(
        payload["idempotency_key"]
    ):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: idempotency_key",
        )

    request_id = contract.request_id_from_headers(request)
    store = _queue(request)
    try:
        _verify_publish_job_spec_if_needed(request, job_spec)
        job = store.put_job(
            job_uid=job_uid,
            job_spec=job_spec,
            capability_hash_sha256=_current_capability_hash(request),
        )
    except job_stubs.JobConflictError as exc:
        raise contract.ContractError(409, contract.ERROR_JOB_CONFLICT) from exc
    except queue_store.QueueFullError as exc:
        monitoring.report_alert(
            request.app.state,
            "queue_full",
            active=True,
            reason_code="queue_full",
            request_id=request_id,
        )
        raise contract.ContractError(503, contract.ERROR_QUEUE_FULL) from exc
    except job_stubs.JobValidationError as exc:
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-6 contract: job_spec",
        ) from exc
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    except contract.ContractError:
        raise
    except Exception as exc:
        _set_queue_unavailable(request, "sqlite_unavailable")
        raise _queue_unavailable_contract_error("sqlite_unavailable") from exc

    return contract.success_response(
        request,
        _job_data(request, job),
        status_code=202,
        request_id=request_id,
    )


@router.put("/jobs/{job_uid}/artifacts/{role}")
async def put_job_artifact(job_uid: str, role: str, request: Request):
    contract.authenticate(request, required_scope="job_write")
    _validate_job_uid(job_uid)
    if role not in ARTIFACT_UPLOAD_ROLES:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_ROLE_MISMATCH)
    body = await _read_limited_body(request, limit=ARTIFACT_UPLOAD_BODY_LIMIT)
    _require_json_content_type(request)
    payload = _parse_json_object(body, top_level_status=400)
    _reject_unknown_fields(
        payload,
        {"sha256", "size_bytes", "content_base64"},
        label="artifact",
    )
    expected_sha256 = _artifact_upload_sha256(payload.get("sha256"))
    expected_size = _artifact_upload_size(payload.get("size_bytes"))
    data = _artifact_upload_bytes(payload.get("content_base64"))
    if len(data) != expected_size:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_SIZE_MISMATCH)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_HASH_MISMATCH)

    store = _queue(request)
    source_job = _get_job_or_404(request, job_uid)
    if source_job.job_type != "sft" or source_job.state != "succeeded":
        raise contract.ContractError(422, contract.ERROR_LINEAGE_MISMATCH)
    try:
        stager = artifact_staging.ArtifactStager.from_config(
            cfg=contract.worker_config(request),
            store=store,
        )
        result = stager.stage_bytes(
            job_uid=job_uid,
            role=role,
            data=data,
            expected_size_bytes=expected_size,
            expected_sha256=expected_sha256,
        )
    except artifact_staging.ArtifactStagingError as exc:
        raise contract.ContractError(503, contract.ERROR_ARTIFACT_MISSING) from exc
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc

    if result.artifact is None or result.status != "staged":
        failure_code = result.failure_code
        if failure_code == contract.ERROR_ARTIFACT_HASH_MISMATCH:
            raise contract.ContractError(422, contract.ERROR_ARTIFACT_HASH_MISMATCH)
        if failure_code == contract.ERROR_ARTIFACT_SIZE_MISMATCH:
            raise contract.ContractError(422, contract.ERROR_ARTIFACT_SIZE_MISMATCH)
        raise contract.ContractError(503, contract.ERROR_ARTIFACT_MISSING)
    return contract.success_response(
        request,
        {
            "job_uid": job_uid,
            "artifact": _artifact_public_data(result.artifact),
        },
        status_code=202,
    )


@router.get("/jobs/{job_uid}")
async def get_job(job_uid: str, request: Request):
    contract.authenticate(request, required_scope="read")
    _validate_job_uid(job_uid)
    return contract.success_response(
        request,
        _job_data(request, _get_job_or_404(request, job_uid)),
    )


@router.get("/jobs/{job_uid}/logs")
async def get_job_logs(job_uid: str, request: Request):
    contract.authenticate(request, required_scope="logs")
    _validate_job_uid(job_uid)
    _validate_logs_query(request)
    return contract.success_response(
        request,
        job_stubs.logs_data(
            _get_job_or_404(request, job_uid),
            execution_enabled=_execution_enabled(request),
        ),
    )


@router.post("/jobs/{job_uid}/cancel")
async def cancel_job(job_uid: str, request: Request):
    contract.authenticate(request, required_scope="cancel")
    _validate_job_uid(job_uid)
    payload = await _parse_optional_action_body(request, allowed_fields={"reason_code"})
    request_id = contract.request_id_from_headers(request)
    try:
        job = _queue(request).request_cancel(
            job_uid=job_uid,
            reason_code=payload.get("reason_code"),
        )
    except job_stubs.JobNotFoundError as exc:
        raise contract.ContractError(404, contract.ERROR_JOB_NOT_FOUND) from exc
    except job_stubs.JobValidationError as exc:
        raise contract.ContractError(422, contract.ERROR_VALIDATION_FAILED) from exc
    except queue_store.InvalidStateTransition as exc:
        raise contract.ContractError(409, contract.ERROR_JOB_CONFLICT) from exc
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    except contract.ContractError:
        raise
    except Exception as exc:
        _set_queue_unavailable(request, "sqlite_unavailable")
        raise _queue_unavailable_contract_error("sqlite_unavailable") from exc
    return contract.success_response(
        request,
        _job_data(request, job),
        status_code=202,
        request_id=request_id,
    )


@router.post("/jobs/{job_uid}/resume")
async def resume_job(job_uid: str, request: Request):
    contract.authenticate(request, required_scope="resume")
    _validate_job_uid(job_uid)
    payload = await _parse_optional_action_body(
        request,
        allowed_fields={"reason_code", "last_checkpoint_ref"},
    )
    request_id = contract.request_id_from_headers(request)
    try:
        job = _queue(request).request_resume(
            job_uid=job_uid,
            reason_code=payload.get("reason_code"),
            last_checkpoint_ref=payload.get("last_checkpoint_ref"),
        )
    except job_stubs.JobNotFoundError as exc:
        raise contract.ContractError(404, contract.ERROR_JOB_NOT_FOUND) from exc
    except queue_store.ResumeLimitExceeded as exc:
        raise contract.ContractError(
            409,
            contract.ERROR_RESUME_LIMIT_EXCEEDED,
        ) from exc
    except job_stubs.JobValidationError as exc:
        raise contract.ContractError(422, contract.ERROR_VALIDATION_FAILED) from exc
    except queue_store.InvalidStateTransition as exc:
        raise contract.ContractError(409, contract.ERROR_JOB_CONFLICT) from exc
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    except contract.ContractError:
        raise
    except Exception as exc:
        _set_queue_unavailable(request, "sqlite_unavailable")
        raise _queue_unavailable_contract_error("sqlite_unavailable") from exc
    return contract.success_response(
        request,
        _job_data(request, job),
        status_code=202,
        request_id=request_id,
    )


def _job_data(request: Request, job: queue_store.JobRecord) -> dict[str, Any]:
    return job_stubs.job_data(
        job,
        execution_enabled=_execution_enabled(request),
        publish_result=_publish_result_or_none(request, job),
        artifacts=_staged_artifacts_or_empty(request, job),
    )


def _staged_artifacts_or_empty(
    request: Request,
    job: queue_store.JobRecord,
) -> list[dict[str, Any]]:
    try:
        artifacts = _queue(request).list_staged_artifacts_for_job(job.job_uid)
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    return [_artifact_public_data(artifact) for artifact in artifacts]


def _artifact_public_data(artifact: queue_store.ArtifactRecord) -> dict[str, Any]:
    return {
        "role": artifact.role,
        "artifact_ref": artifact.artifact_ref,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "status": artifact.status,
        "verified_at": artifact.verified_at,
    }


def _artifact_upload_sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_HASH_MISMATCH)
    return value


def _artifact_upload_size(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > ARTIFACT_UPLOAD_MAX_BYTES
    ):
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_SIZE_MISMATCH)
    return value


def _artifact_upload_bytes(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_MISSING)
    try:
        data = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_MISSING) from exc
    if not data or len(data) > ARTIFACT_UPLOAD_MAX_BYTES:
        raise contract.ContractError(422, contract.ERROR_ARTIFACT_SIZE_MISMATCH)
    return data


def _publish_result_or_none(
    request: Request,
    job: queue_store.JobRecord,
) -> dict[str, Any] | None:
    if job.job_type != publish_contract.PUBLISH_JOB_TYPE:
        return None
    try:
        result = _queue(request).get_publish_result(job.job_uid)
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    if result is None:
        return None
    return result.as_public_dict()


def _verify_publish_job_spec_if_needed(request: Request, job_spec: dict[str, Any]) -> None:
    if not publish_contract.is_publish_job_spec(job_spec):
        return
    keyring = _current_publish_verify_keyring(request)
    try:
        publish_contract.verify_publish_spec_signature(job_spec, keyring=keyring)
    except publish_contract.PublishSpecError as exc:
        raise contract.ContractError(422, exc.reason_code) from exc


def _current_publish_verify_keyring(
    request: Request,
) -> publish_contract.PublishVerifyKeyring | None:
    cfg = contract.worker_config(request)
    if getattr(cfg, "publish_verify_keys_file", None) is not None:
        reason = worker_config.publish_verify_keyring_availability_error(cfg)
        if reason is not None:
            raise contract.ContractError(422, _publish_keyring_contract_error(reason))
        keyring = worker_config.load_publish_verify_keyring(cfg)
        if keyring is None:
            raise contract.ContractError(422, "publish_spec_signature_key_unavailable")
        return keyring
    return getattr(request.app.state, "publish_verify_keyring", None)


def _publish_keyring_contract_error(reason: str) -> str:
    if reason.startswith("publish_spec_signature_"):
        return reason
    return "publish_spec_signature_key_unavailable"


def _execution_enabled(request: Request) -> bool:
    return _current_capability_snapshot(request)["execution_enabled"] is True


def _queue(request: Request) -> queue_store.QueueStore:
    unavailable = getattr(request.app.state, "queue_unavailable", None)
    if unavailable is not None:
        raise _queue_unavailable_contract_error(unavailable.reason_code)
    store = getattr(request.app.state, "queue_store", None)
    if store is None:
        _set_queue_unavailable(request, "sqlite_unavailable")
        raise _queue_unavailable_contract_error("sqlite_unavailable")
    return store


def _get_job_or_404(request: Request, job_uid: str) -> queue_store.JobRecord:
    try:
        return _queue(request).get_job(job_uid)
    except job_stubs.JobNotFoundError as exc:
        raise contract.ContractError(404, contract.ERROR_JOB_NOT_FOUND) from exc
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        raise _queue_unavailable_contract_error(exc.reason_code) from exc
    except contract.ContractError:
        raise
    except Exception as exc:
        _set_queue_unavailable(request, "sqlite_unavailable")
        raise _queue_unavailable_contract_error("sqlite_unavailable") from exc


def _validate_job_uid(job_uid: str) -> None:
    if not job_stubs.validate_job_uid(job_uid):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: job_uid",
        )


def _validate_logs_query(request: Request) -> None:
    cursor = request.query_params.get("cursor")
    if cursor is not None and not job_stubs.validate_cursor(cursor):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: cursor",
        )
    limit_raw = request.query_params.get("limit")
    if limit_raw is None:
        return
    try:
        limit = int(limit_raw)
    except ValueError as exc:
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: limit",
        ) from exc
    if limit < 1 or limit > 200:
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: limit",
        )


async def _parse_optional_action_body(
    request: Request,
    *,
    allowed_fields: set[str],
) -> dict[str, Any]:
    body = await _read_limited_body(request, limit=ACTION_BODY_LIMIT)
    if body == b"":
        return {}
    _require_json_content_type(request)
    payload = _parse_json_object(body, top_level_status=422)
    _reject_unknown_fields(payload, allowed_fields, label="action")
    reason_code = payload.get("reason_code")
    if reason_code is not None and not job_stubs.validate_reason_code(reason_code):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-4 contract: reason_code",
        )
    checkpoint_ref = payload.get("last_checkpoint_ref")
    if checkpoint_ref is not None and not job_stubs.validate_last_checkpoint_ref(
        checkpoint_ref
    ):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            "request did not match ADR-0008 sprint-6 contract: last_checkpoint_ref",
        )
    return payload


async def _read_limited_body(request: Request, *, limit: int) -> bytes:
    body = await request.body()
    if len(body) > limit:
        raise contract.ContractError(413, contract.ERROR_PAYLOAD_TOO_LARGE)
    return body


def _require_json_content_type(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not media_type.endswith("+json"):
        raise contract.ContractError(415, contract.ERROR_UNSUPPORTED_MEDIA_TYPE)


def _parse_json_object(body: bytes, *, top_level_status: int) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise contract.ContractError(400, contract.ERROR_INVALID_REQUEST) from exc
    if not isinstance(payload, dict):
        raise contract.ContractError(
            top_level_status,
            contract.ERROR_INVALID_REQUEST
            if top_level_status == 400
            else contract.ERROR_VALIDATION_FAILED,
        )
    return payload


def _reject_unknown_fields(
    payload: dict[str, Any],
    allowed_fields: set[str],
    *,
    label: str,
) -> None:
    if any(key not in allowed_fields for key in payload):
        raise contract.ContractError(
            422,
            contract.ERROR_VALIDATION_FAILED,
            f"request did not match ADR-0008 sprint-6 contract: {label} fields",
        )


def _current_capability_hash(request: Request) -> str:
    return _current_capability_snapshot(request)["capability_hash_sha256"]


def _current_capability_snapshot(request: Request) -> dict[str, Any]:
    probes = getattr(request.app.state, "capability_probes", None)
    now_fn = getattr(request.app.state, "capability_clock", None)
    queue_status = _queue_status_for_capabilities(request)
    return capability_builder.build_capability_snapshot(
        cfg=contract.worker_config(request),
        active_jobs=_active_jobs_from_queue_status(queue_status),
        queue_status=queue_status,
        probes=probes,
        now_fn=now_fn,
    )


def _active_jobs_from_queue_status(queue_status: dict[str, Any]) -> int:
    value = queue_status.get("active_jobs", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _queue_status_for_capabilities(request: Request) -> dict[str, Any]:
    unavailable = getattr(request.app.state, "queue_unavailable", None)
    if unavailable is not None:
        stats = queue_store.unavailable_stats(unavailable.reason_code)
        _report_queue_stats(request, stats)
        return stats
    store = getattr(request.app.state, "queue_store", None)
    if store is None:
        stats = queue_store.unavailable_stats("sqlite_unavailable")
        _report_queue_stats(request, stats)
        return stats
    try:
        stats = store.queue_stats()
        _report_queue_stats(request, stats)
        return stats
    except queue_store.QueueUnavailable as exc:
        _set_queue_unavailable(request, exc.reason_code)
        stats = queue_store.unavailable_stats(exc.reason_code)
        _report_queue_stats(request, stats)
        return stats
    except Exception:
        _set_queue_unavailable(request, "sqlite_unavailable")
        stats = queue_store.unavailable_stats("sqlite_unavailable")
        _report_queue_stats(request, stats)
        return stats


def _set_queue_unavailable(request: Request, reason_code: str) -> None:
    request.app.state.queue_store = None
    request.app.state.queue_unavailable = queue_store.QueueUnavailableState(reason_code)
    monitoring.report_alert(
        request.app.state,
        "disk_full" if reason_code == "disk_full" else "queue_unavailable",
        active=True,
        reason_code=reason_code,
        source="queue",
    )


def _report_queue_stats(request: Request, stats: dict[str, Any]) -> None:
    codes = monitoring.alert_codes_from_queue_stats(stats)
    for code in ("queue_full", "queue_unavailable", "disk_full"):
        monitoring.report_alert(
            request.app.state,
            code,
            active=code in codes,
            reason_code=stats.get("queue_degraded_reason") or "queue_ready",
            source="queue",
        )


def _queue_unavailable_contract_error(reason_code: str) -> contract.ContractError:
    if reason_code == "disk_full":
        return contract.ContractError(503, contract.ERROR_DISK_FULL)
    return contract.ContractError(503, contract.ERROR_QUEUE_UNAVAILABLE)
