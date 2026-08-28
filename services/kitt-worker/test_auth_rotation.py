from __future__ import annotations

from datetime import datetime, timezone
import secrets
import string

import pytest

import auth


ALPHABET = string.ascii_letters + string.digits + "_-"
NOW = datetime(2026, 6, 14, 12, 0, 0, tzinfo=timezone.utc)


def _random_text(length: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def _record(scopes: list[str] | None = None, **extra_record):
    kid = _random_text(12)
    secret = _random_text(48)
    credential = f"{kid}.{secret}"
    return credential, {
        "kid": kid,
        "token_hash_sha256": auth.credential_hash(credential),
        "scopes": ["read"] if scopes is None else scopes,
        **extra_record,
    }


def _headers(credential: str) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", ("Bea" + "rer " + credential).encode("ascii"))]


def _store(current: dict, previous: dict | None = None, revoked_kids: list[str] | None = None):
    payload = {
        "version": 1,
        "current": current,
        "revoked_kids": [] if revoked_kids is None else revoked_kids,
    }
    if previous is not None:
        payload["previous"] = previous
    return auth.parse_credentials_payload(payload, now_fn=lambda: NOW)


def test_current_and_previous_tokens_are_accepted_within_rotation_window():
    current_credential, current = _record(scopes=["read", "job_write"])
    previous_credential, previous = _record(
        not_before="2026-06-14T11:00:00Z",
        not_after="2026-06-14T13:00:00Z",
    )
    store = _store(current, previous)

    assert store.authenticate_headers(_headers(current_credential), required_scope="job_write")
    assert store.authenticate_headers(_headers(previous_credential), required_scope="read")


def test_not_before_is_inclusive():
    current_credential, current = _record(not_before="2026-06-14T12:00:00Z")
    store = _store(current)

    assert store.authenticate_headers(_headers(current_credential), now=NOW)
    with pytest.raises(auth.AuthenticationError):
        store.authenticate_headers(
            _headers(current_credential),
            now=datetime(2026, 6, 14, 11, 59, 59, tzinfo=timezone.utc),
        )


def test_not_after_is_exclusive_for_current_and_previous():
    current_credential, current = _record(not_after="2026-06-14T12:00:00Z")
    _previous_credential, previous = _record(not_after="2026-06-14T12:00:00Z")
    store = _store(current, previous)

    with pytest.raises(auth.AuthenticationError):
        store.authenticate_headers(_headers(current_credential), now=NOW)


def test_previous_token_expires_at_not_after_boundary():
    _current_credential, current = _record()
    previous_credential, previous = _record(
        not_before="2026-06-14T11:00:00Z",
        not_after="2026-06-14T12:00:00Z",
    )
    store = _store(current, previous)

    with pytest.raises(auth.AuthenticationError):
        store.authenticate_headers(_headers(previous_credential), now=NOW)


def test_revoked_kids_win_over_current_and_previous():
    current_credential, current = _record()
    previous_credential, previous = _record(not_after="2026-06-14T13:00:00Z")
    store = _store(current, previous, revoked_kids=[current["kid"], previous["kid"]])

    for credential in (current_credential, previous_credential):
        with pytest.raises(auth.AuthenticationError):
            store.authenticate_headers(_headers(credential), now=NOW)


def test_duplicate_or_invalid_revoked_kids_fail_closed():
    _current_credential, current = _record()

    with pytest.raises(auth.AuthConfigError):
        _store(current, revoked_kids=[current["kid"], current["kid"]])

    with pytest.raises(auth.AuthConfigError):
        _store(current, revoked_kids=[_random_text(7)])


def test_invalid_time_window_fails_closed():
    _current_credential, current = _record(
        not_before="2026-06-14T12:00:00Z",
        not_after="2026-06-14T12:00:00Z",
    )

    with pytest.raises(auth.AuthConfigError):
        _store(current)


def test_naive_or_offset_times_fail_closed():
    for value in ("2026-06-14T12:00:00", "2026-06-14T14:00:00+02:00"):
        _current_credential, current = _record(not_before=value)
        with pytest.raises(auth.AuthConfigError):
            _store(current)
