"""ADR-0008 response envelope and error handling for /v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import re
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import auth
import config


REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_REQUEST_ID_CREDENTIAL_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,64}\.[A-Za-z0-9_-]{32,256}"
    r"(?![A-Za-z0-9_-])"
)
_REQUEST_ID_SHA256_RE = re.compile(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])")
_REQUEST_ID_SECRET_WORD_RE = re.compile(
    r"(?i)(authorization|bearer|token|secret|credential|password|private[_-]?key|api[_-]?key)"
)

ERROR_INVALID_REQUEST = "invalid_request"
ERROR_VALIDATION_FAILED = "validation_failed"
ERROR_AUTH_REQUIRED = "auth_required"
ERROR_FORBIDDEN = "forbidden"
ERROR_NOT_FOUND = "not_found"
ERROR_METHOD_NOT_ALLOWED = "method_not_allowed"
ERROR_UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
ERROR_PAYLOAD_TOO_LARGE = "payload_too_large"
ERROR_JOB_NOT_FOUND = "job_not_found"
ERROR_JOB_CONFLICT = "job_conflict"
ERROR_QUEUE_FULL = "queue_full"
ERROR_QUEUE_UNAVAILABLE = "queue_unavailable"
ERROR_DISK_FULL = "disk_full"
ERROR_RESUME_LIMIT_EXCEEDED = "resume_limit_exceeded"
ERROR_PUBLISH_SPEC_SIGNATURE_REQUIRED = "publish_spec_signature_required"
ERROR_PUBLISH_SPEC_SIGNATURE_SHAPE_INVALID = "publish_spec_signature_shape_invalid"
ERROR_PUBLISH_SPEC_SIGNATURE_ALGORITHM_UNSUPPORTED = (
    "publish_spec_signature_algorithm_unsupported"
)
ERROR_PUBLISH_SPEC_HASH_MISMATCH = "publish_spec_hash_mismatch"
ERROR_PUBLISH_SPEC_SIGNED_PAYLOAD_HASH_MISMATCH = (
    "publish_spec_signed_payload_hash_mismatch"
)
ERROR_PUBLISH_SPEC_SIGNATURE_KEY_UNAVAILABLE = (
    "publish_spec_signature_key_unavailable"
)
ERROR_PUBLISH_SPEC_SIGNATURE_KEY_UNKNOWN = "publish_spec_signature_key_unknown"
ERROR_PUBLISH_SPEC_SIGNATURE_KEY_EXPIRED = "publish_spec_signature_key_expired"
ERROR_PUBLISH_SPEC_SIGNATURE_INVALID = "publish_spec_signature_invalid"
ERROR_PUBLISH_SPEC_SHAPE_INVALID = "publish_spec_shape_invalid"
ERROR_PUBLISH_SPEC_JOB_TYPE_INVALID = "publish_spec_job_type_invalid"
ERROR_PUBLISH_INTENT_REQUIRED = "publish_intent_required"
ERROR_PUBLISH_INTENT_NOT_APPROVED = "publish_intent_not_approved"
ERROR_PUBLISH_TARGET_INVALID = "publish_target_invalid"
ERROR_ARTIFACT_MISSING = "artifact_missing"
ERROR_ARTIFACT_ROLE_MISMATCH = "artifact_role_mismatch"
ERROR_ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
ERROR_ARTIFACT_SIZE_MISMATCH = "artifact_size_mismatch"
ERROR_LINEAGE_MISMATCH = "lineage_mismatch"
ERROR_PARENT_MISMATCH = "parent_mismatch"
ERROR_V1_DISABLED = "v1_disabled"
ERROR_CONTRACT_NOT_READY = "contract_not_ready"
ERROR_INTERNAL = "internal_error"

DIAGNOSTICS = {
    ERROR_INVALID_REQUEST: "request did not contain valid ADR-0008 JSON",
    ERROR_VALIDATION_FAILED: "request did not match ADR-0008 sprint-4 contract",
    ERROR_AUTH_REQUIRED: "authentication required for ADR-0008 contract",
    ERROR_FORBIDDEN: "authenticated principal lacks required scope",
    ERROR_NOT_FOUND: "ADR-0008 route was not found",
    ERROR_METHOD_NOT_ALLOWED: "HTTP method is not allowed for ADR-0008 route",
    ERROR_UNSUPPORTED_MEDIA_TYPE: "request content type is not supported",
    ERROR_PAYLOAD_TOO_LARGE: "request body exceeds ADR-0008 sprint-4 limit",
    ERROR_JOB_NOT_FOUND: "job was not found",
    ERROR_JOB_CONFLICT: "job uid exists with a different job spec hash",
    ERROR_QUEUE_FULL: "local worker queue is full",
    ERROR_QUEUE_UNAVAILABLE: "local worker queue is not available",
    ERROR_DISK_FULL: "local worker queue storage is full",
    ERROR_RESUME_LIMIT_EXCEEDED: "resume metadata limit was exceeded",
    ERROR_PUBLISH_SPEC_SIGNATURE_REQUIRED: "publish spec signature envelope is required",
    ERROR_PUBLISH_SPEC_SIGNATURE_SHAPE_INVALID: "publish spec signature envelope shape is invalid",
    ERROR_PUBLISH_SPEC_SIGNATURE_ALGORITHM_UNSUPPORTED: "publish spec signature algorithm is unsupported",
    ERROR_PUBLISH_SPEC_HASH_MISMATCH: "publish spec hash does not match canonical payload",
    ERROR_PUBLISH_SPEC_SIGNED_PAYLOAD_HASH_MISMATCH: "publish signed payload hash does not match canonical payload",
    ERROR_PUBLISH_SPEC_SIGNATURE_KEY_UNAVAILABLE: "publish spec signature verification key is unavailable",
    ERROR_PUBLISH_SPEC_SIGNATURE_KEY_UNKNOWN: "publish spec signature key id is unknown",
    ERROR_PUBLISH_SPEC_SIGNATURE_KEY_EXPIRED: "publish spec signature key is expired or disabled",
    ERROR_PUBLISH_SPEC_SIGNATURE_INVALID: "publish spec signature verification failed",
    ERROR_PUBLISH_SPEC_SHAPE_INVALID: "publish job spec shape is invalid",
    ERROR_PUBLISH_SPEC_JOB_TYPE_INVALID: "publish job type is invalid",
    ERROR_PUBLISH_INTENT_REQUIRED: "publish intent metadata is required",
    ERROR_PUBLISH_INTENT_NOT_APPROVED: "publish intent is not approved",
    ERROR_PUBLISH_TARGET_INVALID: "publish target is invalid",
    ERROR_ARTIFACT_MISSING: "required publish artifact is missing",
    ERROR_ARTIFACT_ROLE_MISMATCH: "publish artifact role does not match contract",
    ERROR_ARTIFACT_HASH_MISMATCH: "publish artifact hash does not match contract",
    ERROR_ARTIFACT_SIZE_MISMATCH: "publish artifact size does not match contract",
    ERROR_LINEAGE_MISMATCH: "publish lineage does not match contract",
    ERROR_PARENT_MISMATCH: "publish parent model does not match contract",
    ERROR_V1_DISABLED: "ADR-0008 contract is disabled",
    ERROR_CONTRACT_NOT_READY: "ADR-0008 contract is not ready",
    ERROR_INTERNAL: "internal server error while handling ADR-0008 contract",
}

_LOCAL_PATH_RE = re.compile(r"/(?:opt|usr/lib|run|etc|var|home)/[^\s'\"\)]*")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)
_MODULE_RE = re.compile(
    r"\b(?:Traceback|File|line|module|Exception|RuntimeError|ValueError)\b"
)
_DSN_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:[+]srv)?|redis|amqp|amqps|"
    r"mqtt|kafka|sqlite|mssql|oracle|file)://[^\s'\"\)]*",
    re.IGNORECASE,
)
_SIGNED_URL_RE = re.compile(
    r"\b(?:https?|wss?)://[^\s'\"\)]*[?&;][^\s'\"\)]*"
    r"(?:signature|token|secret|credential|password|passwd|api[_-]?key|"
    r"access[_-]?key|session[_-]?key|x-amz-signature|kid|key[_-]?id)"
    r"[^\s'\"\)]*",
    re.IGNORECASE,
)
_SECRET_WORD_RE = re.compile(
    r"(?i)\b(?:authorization|bearer|token|secret|credential|password|private[_-]?key|api[_-]?key)\b[^\s,;]*"
)
_RAW_PAYLOAD_FIELD_PATTERN = (
    r"(?:raw[_-]?prompt|raw[_-]?response|canonical_job_spec_json|"
    r"cp[_-]?raw(?:[_-]?text)?|artifact[_-]?(?:bytes|data|path|filename))"
)
_RAW_PAYLOAD_KEY_RE = re.compile(
    r"(?i)[\"']?" + _RAW_PAYLOAD_FIELD_PATTERN + r"[\"']?\s*[:=]\s*"
)
_RAW_PAYLOAD_WORD_RE = re.compile(
    r"(?i)\b" + _RAW_PAYLOAD_FIELD_PATTERN + r"\b"
)
_URL_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)([^/?#\s@]+)@")
_SENSITIVE_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&;][A-Za-z0-9_.:-]*"
    r"(?:signature|token|secret|credential|password|passwd|api[_-]?key|"
    r"access[_-]?key|session[_-]?key)"
    r"[A-Za-z0-9_.:-]*=)([^&#;\s]*)"
)
_SENSITIVE_KEY_VALUE_RE = re.compile(
    r"(?i)([\"']?\b[A-Za-z0-9_.:-]*"
    r"(?:authorization|bearer|token|secret|credential|password|passwd|"
    r"private[_-]?key|api[_-]?key|access[_-]?key|session[_-]?key)"
    r"[A-Za-z0-9_.:-]*\b[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^,\s;&)}\]]*)"
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContractError(Exception):
    status_code: int
    error_code: str
    diagnostic: str | None = None


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def request_id_from_headers(request: Request) -> str:
    raw = request.headers.get(REQUEST_ID_HEADER)
    if (
        raw is not None
        and raw == raw.strip()
        and REQUEST_ID_RE.fullmatch(raw)
        and not _request_id_looks_sensitive(raw)
    ):
        return raw
    return f"req_{uuid.uuid4().hex}"


def is_v1_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def has_malformed_job_path(path: str) -> bool:
    parts = path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "v1" or parts[1] != "jobs":
        return False
    if parts[-1] in {"logs", "cancel", "resume"}:
        return len(parts) > 4
    return len(parts) > 3


def success_response(
    request: Request,
    data: dict[str, Any],
    *,
    status_code: int = 200,
    request_id: str | None = None,
) -> JSONResponse:
    cfg = worker_config(request)
    request_id = request_id or request_id_from_headers(request)
    response = JSONResponse(
        {
            "ok": True,
            "worker_id": getattr(cfg, "worker_id", config.DEFAULT_WORKER_ID),
            "worker_contract_version": getattr(
                cfg,
                "worker_contract_version",
                config.DEFAULT_WORKER_CONTRACT_VERSION,
            ),
            "server_time": utc_now(),
            "request_id": request_id,
            "data": data,
        },
        status_code=status_code,
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


def error_response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    diagnostic: str | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    cfg = worker_config(request)
    request_id = request_id or request_id_from_headers(request)
    safe_diagnostic = sanitize_diagnostic(diagnostic or DIAGNOSTICS[error_code])
    response = JSONResponse(
        {
            "ok": False,
            "worker_id": getattr(cfg, "worker_id", config.DEFAULT_WORKER_ID),
            "worker_contract_version": getattr(
                cfg,
                "worker_contract_version",
                config.DEFAULT_WORKER_CONTRACT_VERSION,
            ),
            "server_time": utc_now(),
            "request_id": request_id,
            "data": None,
            "error_code": error_code,
            "diagnostic": safe_diagnostic,
        },
        status_code=status_code,
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    safe_path = sanitize_diagnostic(request.url.path)
    logger.warning(
        "kitt-worker contract error request_id=%s path=%s method=%s status_code=%s error_code=%s",
        request_id,
        safe_path,
        request.method,
        status_code,
        error_code,
    )
    return response


def sanitize_diagnostic(value: str) -> str:
    text = auth.redact_diagnostic(value)
    text = _redact_raw_payload_values(text)
    text = _DSN_RE.sub("<redacted-dsn>", text)
    text = _SIGNED_URL_RE.sub("<redacted-url>", text)
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _SENSITIVE_QUERY_PARAM_RE.sub(r"\1<redacted>", text)
    text = _SENSITIVE_KEY_VALUE_RE.sub(r"\1<redacted>", text)
    text = _TRACEBACK_RE.sub("<redacted-traceback>", text)
    text = _LOCAL_PATH_RE.sub("<redacted-path>", text)
    text = _SECRET_WORD_RE.sub("<redacted>", text)
    text = _MODULE_RE.sub("<redacted>", text)
    text = _RAW_PAYLOAD_WORD_RE.sub("<redacted-payload>", text)
    text = " ".join(text.split())
    if not text:
        text = DIAGNOSTICS[ERROR_INTERNAL]
    return text[:256]


def _request_id_looks_sensitive(value: str) -> bool:
    return (
        _REQUEST_ID_CREDENTIAL_RE.search(value) is not None
        or _REQUEST_ID_SHA256_RE.search(value) is not None
        or _REQUEST_ID_SECRET_WORD_RE.search(value) is not None
    )


def _redact_raw_payload_values(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    while True:
        match = _RAW_PAYLOAD_KEY_RE.search(text, cursor)
        if match is None:
            parts.append(text[cursor:])
            break
        parts.append(text[cursor : match.start()])
        parts.append("<redacted-payload>")
        cursor = _consume_raw_payload_value(text, match.end())
    return "".join(parts)


def _consume_raw_payload_value(text: str, start: int) -> int:
    pos = start
    length = len(text)
    while pos < length and text[pos].isspace():
        pos += 1
    if pos >= length:
        return pos
    if text[pos] in {"'", '"'}:
        return _consume_quoted_value(text, pos)
    if text[pos] in {"{", "["}:
        return _consume_balanced_value(text, pos)
    next_key = _RAW_PAYLOAD_KEY_RE.search(text, pos + 1)
    stop = length
    for marker in (",", ";", "\n", ")", "}", "]"):
        candidate = text.find(marker, pos)
        if candidate != -1:
            stop = min(stop, candidate)
    if next_key is not None:
        stop = min(stop, next_key.start())
    return stop


def _consume_quoted_value(text: str, start: int) -> int:
    quote = text[start]
    escaped = False
    for pos in range(start + 1, len(text)):
        char = text[pos]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return pos + 1
    return len(text)


def _consume_balanced_value(text: str, start: int) -> int:
    closers = {"{": "}", "[": "]"}
    stack = [closers[text[start]]]
    pos = start + 1
    while pos < len(text):
        char = text[pos]
        if char in {"'", '"'}:
            pos = _consume_quoted_value(text, pos)
            continue
        if char in closers:
            stack.append(closers[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return pos + 1
        pos += 1
    return len(text)


def worker_config(request: Request) -> object:
    return getattr(request.app.state, "worker_config", None)


def readiness_error(request: Request) -> ContractError | None:
    cfg = worker_config(request)
    if cfg is None:
        return ContractError(503, ERROR_CONTRACT_NOT_READY)
    if not getattr(cfg, "enable_v1", False):
        return ContractError(503, ERROR_V1_DISABLED)
    if getattr(cfg, "auth_mode", None) == "disabled":
        return ContractError(503, ERROR_V1_DISABLED)
    if getattr(cfg, "auth_mode", None) != "required":
        return ContractError(503, ERROR_CONTRACT_NOT_READY)
    if getattr(request.app.state, "credential_store", None) is None:
        return ContractError(503, ERROR_CONTRACT_NOT_READY)
    return None


def require_v1_ready(request: Request) -> None:
    problem = readiness_error(request)
    if problem is not None:
        raise problem


def authenticate(request: Request, *, required_scope: str) -> None:
    require_v1_ready(request)
    store = request.app.state.credential_store
    store.authenticate_headers(request.scope["headers"], required_scope=required_scope)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ContractError)
    async def handle_contract_error(request: Request, exc: ContractError):
        return error_response(
            request,
            status_code=exc.status_code,
            error_code=exc.error_code,
            diagnostic=exc.diagnostic,
        )

    @app.exception_handler(auth.AuthenticationError)
    async def handle_auth_error(request: Request, _exc: auth.AuthenticationError):
        return error_response(
            request,
            status_code=401,
            error_code=ERROR_AUTH_REQUIRED,
        )

    @app.exception_handler(auth.AuthorizationError)
    async def handle_authorization_error(request: Request, _exc: auth.AuthorizationError):
        return error_response(
            request,
            status_code=403,
            error_code=ERROR_FORBIDDEN,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        if not is_v1_path(request.url.path):
            return await request_validation_exception_handler(request, exc)
        problem = readiness_error(request)
        if problem is not None:
            return error_response(
                request,
                status_code=problem.status_code,
                error_code=problem.error_code,
                diagnostic=problem.diagnostic,
            )
        return error_response(
            request,
            status_code=422,
            error_code=ERROR_VALIDATION_FAILED,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        if not is_v1_path(request.url.path):
            return await http_exception_handler(request, exc)
        problem = readiness_error(request)
        if problem is not None:
            return error_response(
                request,
                status_code=problem.status_code,
                error_code=problem.error_code,
                diagnostic=problem.diagnostic,
            )
        if exc.status_code == 404:
            if has_malformed_job_path(request.url.path):
                return error_response(
                    request,
                    status_code=422,
                    error_code=ERROR_VALIDATION_FAILED,
                    diagnostic="request did not match ADR-0008 sprint-4 contract: job_uid",
                )
            return error_response(
                request,
                status_code=404,
                error_code=ERROR_NOT_FOUND,
            )
        if exc.status_code == 405:
            return error_response(
                request,
                status_code=405,
                error_code=ERROR_METHOD_NOT_ALLOWED,
            )
        return error_response(
            request,
            status_code=exc.status_code,
            error_code=ERROR_INVALID_REQUEST,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        if not is_v1_path(request.url.path):
            raise exc
        request_id = request_id_from_headers(request)
        return error_response(
            request,
            status_code=500,
            error_code=ERROR_INTERNAL,
            diagnostic=DIAGNOSTICS[ERROR_INTERNAL],
            request_id=request_id,
        )
