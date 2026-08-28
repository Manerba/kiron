from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import string

import pytest

import auth


ALPHABET = string.ascii_letters + string.digits + "_-"


def _random_text(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def _credential(scopes: list[str] | None = None, **extra_record):
    kid = _random_text(12)
    secret = _random_text(48)
    credential = f"{kid}.{secret}"
    record = {
        "kid": kid,
        "token_hash_sha256": auth.credential_hash(credential),
        "scopes": ["read"] if scopes is None else scopes,
        **extra_record,
    }
    return credential, record


def _headers(credential: str, *, scheme: str = "Bearer") -> list[tuple[bytes, bytes]]:
    return [(b"authorization", f"{scheme} {credential}".encode("ascii"))]


def _store(record: dict) -> auth.CredentialStore:
    return auth.parse_credentials_payload(
        {
            "version": 1,
            "current": record,
            "revoked_kids": [],
        }
    )


def _credential_tree(tmp_path: Path, payload: dict) -> tuple[Path, Path, Path]:
    root = tmp_path / "etc" / "kiron"
    credential_dir = root / "kitt-worker"
    credential_dir.mkdir(parents=True)
    root.chmod(0o755)
    credential_dir.chmod(0o750)
    credentials_file = credential_dir / "credentials.json"
    credentials_file.write_text(json.dumps(payload), encoding="utf-8")
    credentials_file.chmod(0o640)
    return root, credential_dir, credentials_file


def _load_test_file(path: Path, root: Path, credential_dir: Path) -> auth.CredentialStore:
    return auth.load_credentials_file(
        path,
        credential_root=root,
        credential_dir=credential_dir,
        expected_root_uid=os.getuid(),
        expected_root_gid=os.getgid(),
        expected_dir_uid=os.getuid(),
        expected_dir_gid=os.getgid(),
        expected_file_uid=os.getuid(),
        expected_file_gid=os.getgid(),
    )


def test_valid_bearer_token_authenticates_with_scope():
    credential, record = _credential(scopes=["read", "logs"])
    store = _store(record)

    principal = store.authenticate_headers(_headers(credential), required_scope="read")

    assert principal.kid == record["kid"]
    assert principal.scopes == frozenset({"read", "logs"})


def test_scope_mismatch_is_authorization_error():
    credential, record = _credential(scopes=["read"])
    store = _store(record)

    with pytest.raises(auth.AuthorizationError):
        store.authenticate_headers(_headers(credential), required_scope="logs")


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Basic abc")],
        [(b"authorization", b"Bearer")],
        [(b"authorization", b"Bea" + b"rer ")],
    ],
)
def test_missing_or_invalid_authorization_header_is_401(headers):
    _credential_value, record = _credential()
    store = _store(record)

    with pytest.raises(auth.AuthenticationError):
        store.authenticate_headers(headers)


def test_duplicate_authorization_headers_are_rejected_even_if_one_is_valid():
    credential, record = _credential()
    store = _store(record)
    headers = [
        (b"authorization", ("Bea" + "rer " + credential).encode("ascii")),
        (
            b"Authorization",
            ("Bea" + "rer " + _random_text(12) + "." + _random_text(48)).encode("ascii"),
        ),
    ]

    with pytest.raises(auth.AuthenticationError):
        store.authenticate_headers(headers)


def test_unknown_kid_and_mutated_secret_are_rejected():
    credential, record = _credential()
    store = _store(record)
    unknown = f"{_random_text(12)}.{_random_text(48)}"
    mutated = f"{record['kid']}.{_random_text(48)}"

    for value in (unknown, mutated):
        with pytest.raises(auth.AuthenticationError):
            store.authenticate_headers(_headers(value))


@pytest.mark.parametrize(
    "credential_builder",
    [
        lambda kid, secret: kid + secret,
        lambda kid, secret: f"{kid}.",
        lambda kid, secret: f".{secret}",
        lambda kid, secret: f"{kid}.{secret}.{secret}",
        lambda kid, secret: f"{kid}.{secret} extra",
        lambda kid, secret: f"{kid}. {secret}",
        lambda _kid, secret: f"{_random_text(7)}.{secret}",
        lambda _kid, secret: f"{_random_text(65)}.{secret}",
        lambda _kid, secret: f"{_random_text(11)}!.{secret}",
        lambda kid, _secret: f"{kid}.{_random_text(31)}",
        lambda kid, _secret: f"{kid}.{_random_text(257)}",
        lambda kid, _secret: f"{kid}.{_random_text(47)}!",
    ],
)
def test_bearer_credential_syntax_is_strict(credential_builder):
    kid = _random_text(12)
    secret = _random_text(48)
    credential = credential_builder(kid, secret)

    with pytest.raises(auth.AuthenticationError):
        auth.parse_authorization_headers(_headers(credential))


@pytest.mark.parametrize("scope", ["job:write", "logs:read", "job:cancel", "job:resume", "READ"])
def test_credential_scopes_reject_colon_and_noncanonical_variants(scope):
    credential, record = _credential(scopes=[scope])

    with pytest.raises(auth.AuthConfigError):
        _store(record)
    assert credential


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"version": 2}),
        lambda payload: payload.pop("current"),
        lambda payload: payload["current"].update({"algorithm": "sha256"}),
        lambda payload: payload["current"].update({"token_hash_sha256": "A" * 64}),
        lambda payload: payload["current"].update({"token_hash_sha256": "a" * 63}),
        lambda payload: payload["current"].update({"scopes": ["read", "read"]}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"revoked_kids": [_random_text(7)]}),
    ],
)
def test_invalid_credential_schema_fails_closed(mutator):
    _credential_value, record = _credential()
    payload = {
        "version": 1,
        "current": record,
        "revoked_kids": [],
    }
    mutator(payload)

    with pytest.raises(auth.AuthConfigError):
        auth.parse_credentials_payload(payload)


def test_previous_requires_not_after_and_distinct_kid():
    _credential_value, current = _credential()
    _previous_credential, previous = _credential(not_after="2030-01-01T00:00:00Z")

    payload = {
        "version": 1,
        "current": current,
        "previous": {**previous, "kid": current["kid"]},
        "revoked_kids": [],
    }
    with pytest.raises(auth.AuthConfigError):
        auth.parse_credentials_payload(payload)

    payload["previous"] = {key: value for key, value in previous.items() if key != "not_after"}
    with pytest.raises(auth.AuthConfigError):
        auth.parse_credentials_payload(payload)


def test_credential_file_security_accepts_expected_owner_mode(tmp_path):
    credential, record = _credential()
    payload = {
        "version": 1,
        "current": record,
        "revoked_kids": [],
    }
    root, credential_dir, credentials_file = _credential_tree(tmp_path, payload)

    store = _load_test_file(credentials_file, root, credential_dir)

    store.authenticate_headers(_headers(credential), required_scope="read")


def test_credential_file_security_rejects_parent_or_file_symlink(tmp_path):
    _credential_value, record = _credential()
    payload = {
        "version": 1,
        "current": record,
        "revoked_kids": [],
    }
    root, credential_dir, credentials_file = _credential_tree(tmp_path, payload)
    real_dir = tmp_path / "real-secret-dir"
    real_dir.mkdir()
    link_dir = root / "linked-worker"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(auth.AuthConfigError):
        _load_test_file(link_dir / "credentials.json", root, link_dir)

    credentials_file.unlink()
    target = credential_dir / "credentials.real"
    target.write_text(json.dumps(payload), encoding="utf-8")
    target.chmod(0o640)
    credentials_file.symlink_to(target)
    with pytest.raises(auth.AuthConfigError):
        _load_test_file(credentials_file, root, credential_dir)


def test_credential_file_security_rejects_wrong_modes(tmp_path):
    _credential_value, record = _credential()
    payload = {
        "version": 1,
        "current": record,
        "revoked_kids": [],
    }
    root, credential_dir, credentials_file = _credential_tree(tmp_path, payload)

    root.chmod(0o750)
    with pytest.raises(auth.AuthConfigError):
        _load_test_file(credentials_file, root, credential_dir)
    root.chmod(0o755)

    credential_dir.chmod(0o755)
    with pytest.raises(auth.AuthConfigError):
        _load_test_file(credentials_file, root, credential_dir)
    credential_dir.chmod(0o750)

    credentials_file.chmod(0o600)
    with pytest.raises(auth.AuthConfigError):
        _load_test_file(credentials_file, root, credential_dir)


def test_rfc3339_utc_timestamps_are_required():
    _credential_value, record = _credential(not_before=datetime.now(timezone.utc).isoformat())

    with pytest.raises(auth.AuthConfigError):
        _store(record)
