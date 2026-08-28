from __future__ import annotations

import logging
from pathlib import Path
import secrets
import string

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pytest
from starlette.testclient import TestClient

import auth


ALPHABET = string.ascii_letters + string.digits + "_-"


def _join(*parts: str) -> str:
    return "".join(parts)


def _random_text(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def _credential():
    kid = _random_text(12)
    secret = _random_text(48)
    credential = f"{kid}.{secret}"
    return credential, {
        "kid": kid,
        "token_hash_sha256": auth.credential_hash(credential),
        "scopes": ["read"],
    }


def _store(record: dict) -> auth.CredentialStore:
    return auth.parse_credentials_payload(
        {
            "version": 1,
            "current": record,
            "revoked_kids": [],
        }
    )


def _assert_absent(secret_value: str, text: str, label: str) -> None:
    if secret_value in text:
        pytest.fail(f"redaction leaked {label}", pytrace=False)


def _assert_equal_redacted(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        pytest.fail(f"redaction returned unexpected {label}", pytrace=False)


def test_redaction_helpers_remove_header_token_hash_and_paths():
    credential, record = _credential()
    credential_hash = record["token_hash_sha256"]
    credential_path = Path(_join("/", "etc", "/", "kiron", "/", "kitt-worker", "/", "credentials.json"))
    runtime_path = _join("/", "usr", "/", "lib", "/", "kiron", "/", "data", "/", "kitt-worker")
    auth_header = _join("Authorization", ":", " ", "Bea", "rer", " ")
    diagnostic = (
        f"{auth_header}{credential} hash {credential_hash} "
        f"path {credential_path} runtime {runtime_path}"
    )

    redacted = auth.redact_diagnostic(diagnostic)

    _assert_absent(credential, redacted, "credential")
    _assert_absent(credential_hash, redacted, "hash")
    _assert_absent(str(credential_path), redacted, "credential path")
    _assert_absent(runtime_path, redacted, "runtime path")
    _assert_equal_redacted(
        auth.redact_authorization_header(f"{auth_header}{credential}"),
        "Authorization: <redacted>",
        "authorization header",
    )
    _assert_equal_redacted(auth.redact_token(credential), "<redacted-token>", "token")
    _assert_equal_redacted(
        auth.redact_credential_path(credential_path),
        "<redacted-credential-path>",
        "credential path",
    )


def test_redaction_removes_malformed_bearer_suffixes():
    credential, record = _credential()
    malformed = _join("Authorization", ":", " ", "Bea", "rer", " ", credential, "!suffix-with-punctuation")

    redacted = auth.redact_diagnostic(malformed)

    _assert_absent(credential, redacted, "credential")
    _assert_absent("suffix-with-punctuation", redacted, "malformed bearer suffix")
    _assert_absent(record["token_hash_sha256"], redacted, "hash")


def test_redact_headers_masks_authorization_only():
    credential, _record = _credential()
    runtime_path = _join("/", "run", "/", "kiron", "/", "kitt-worker")

    redacted = auth.redact_headers(
        {
            "Authorization": f"{_join('Bea', 'rer')} {credential}",
            "X-Trace": f"path {runtime_path} {_join('to', 'ken')} {credential}",
        }
    )

    _assert_equal_redacted(redacted["Authorization"], "<redacted>", "authorization header")
    _assert_absent(credential, redacted["X-Trace"], "credential")
    _assert_absent(runtime_path, redacted["X-Trace"], "runtime path")


def test_auth_errors_do_not_include_token_material():
    credential, record = _credential()
    store = _store(record)
    mutated = f"{record['kid']}.{_random_text(48)}"

    with pytest.raises(auth.AuthenticationError) as exc_info:
        store.authenticate_headers(
            [(b"authorization", f"{_join('Bea', 'rer')} {mutated}".encode("ascii"))],
            required_scope="read",
        )

    error_text = str(exc_info.value)
    _assert_absent(credential, error_text, "credential")
    _assert_absent(mutated, error_text, "mutated credential")
    _assert_absent(record["token_hash_sha256"], error_text, "hash")


def test_redacted_diagnostics_do_not_leak_to_logs(caplog):
    credential, record = _credential()
    logger = logging.getLogger("kitt-worker-auth-test")
    credential_path = _join("/", "etc", "/", "kiron", "/", "kitt-worker", "/", "credentials.json")
    message = auth.redact_diagnostic(
        f"failed {_join('Bea', 'rer')} {credential} {record['token_hash_sha256']} {credential_path}"
    )

    with caplog.at_level(logging.ERROR, logger="kitt-worker-auth-test"):
        logger.error("auth failure: %s", message)

    _assert_absent(credential, caplog.text, "credential")
    _assert_absent(record["token_hash_sha256"], caplog.text, "hash")
    _assert_absent(credential_path, caplog.text, "credential path")


def test_synthetic_http_response_uses_generic_diagnostics():
    credential, record = _credential()
    store = _store(record)
    app = FastAPI()

    @app.get("/synthetic")
    async def synthetic(request: Request):
        try:
            store.authenticate_headers(request.scope["headers"], required_scope="logs")
        except auth.AuthenticationError:
            return JSONResponse({"detail": "authentication failed"}, status_code=401)
        except auth.AuthorizationError:
            return JSONResponse({"detail": "authorization failed"}, status_code=403)
        return {"ok": True}

    with TestClient(app) as client:
        auth_response = client.get(
            "/synthetic",
            headers={"Authorization": f"{_join('Bea', 'rer')} {credential}"},
        )
        bad_response = client.get(
            "/synthetic",
            headers={
                "Authorization": f"{_join('Bea', 'rer')} {record['kid']}.{_random_text(48)}"
            },
        )

    assert auth_response.status_code == 403
    assert bad_response.status_code == 401
    for response in (auth_response, bad_response):
        body = response.text
        _assert_absent(credential, body, "credential")
        _assert_absent(record["token_hash_sha256"], body, "hash")
        _assert_absent(_join("/", "etc", "/", "kiron"), body, "credential path")
