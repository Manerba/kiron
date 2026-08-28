"""Bearer-token authentication for the kitt-worker data plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import grp
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Iterable, Mapping, Sequence


ALLOWED_SCOPES = frozenset({"read", "job_write", "logs", "cancel", "resume"})
MAX_CREDENTIAL_LENGTH = 384

DEFAULT_CREDENTIAL_ROOT = Path("/etc/kiron")
DEFAULT_CREDENTIAL_DIR = DEFAULT_CREDENTIAL_ROOT / "kitt-worker"
DEFAULT_CREDENTIAL_FILE = DEFAULT_CREDENTIAL_DIR / "credentials.json"

REQUIRED_ETC_ROOT_MODE = 0o755
REQUIRED_CREDENTIAL_DIR_MODE = 0o750
REQUIRED_CREDENTIAL_FILE_MODE = 0o640

KID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)

_FORBIDDEN_HASH_FIELDS = {"hash", "token_hash", "algorithm", "hash_algorithm"}
_RECORD_FIELDS = {"kid", "token_hash_sha256", "scopes", "not_before", "not_after"}
_TOP_LEVEL_FIELDS = {"version", "current", "previous", "revoked_kids"}
_PROHIBITED_CREDENTIAL_ROOTS = (
    Path("/opt/kiron"),
    Path("/usr/lib/kiron/data"),
    Path("/run/kiron"),
    Path("/usr/lib/kiron/services"),
)

_BEARER_VALUE_RE = re.compile(r"(?i)\bBearer\b[^\r\n]*")
_CREDENTIAL_VALUE_RE = re.compile(r"\b[A-Za-z0-9_-]{8,64}\.[A-Za-z0-9_-]{32,256}\b")
_SHA256_VALUE_RE = re.compile(r"\b[a-f0-9]{64}\b")
_LOCAL_PATH_RE = re.compile(
    r"/(?:etc/kiron/kitt-worker|usr/lib/kiron|run/kiron|opt/kiron)[^\s'\"\)]*"
)


class AuthConfigError(RuntimeError):
    """Credential configuration is absent, unsafe or invalid."""

    def __init__(self, message: str = "credential configuration is invalid") -> None:
        super().__init__(message)


class AuthenticationError(RuntimeError):
    """Request authentication failed with a generic public diagnostic."""

    def __init__(self) -> None:
        super().__init__("authentication failed")


class AuthorizationError(RuntimeError):
    """Request is authenticated but lacks the required scope."""

    def __init__(self) -> None:
        super().__init__("authorization failed")


@dataclass(frozen=True, slots=True)
class ParsedBearerCredential:
    kid: str
    credential: str


@dataclass(frozen=True, slots=True)
class CredentialRecord:
    kid: str
    token_hash_sha256: str
    scopes: frozenset[str]
    not_before: datetime | None = None
    not_after: datetime | None = None

    def is_active(self, now: datetime) -> bool:
        if self.not_before is not None and now < self.not_before:
            return False
        if self.not_after is not None and now >= self.not_after:
            return False
        return True


@dataclass(frozen=True, slots=True)
class AuthenticatedPrincipal:
    kid: str
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class CredentialStore:
    current: CredentialRecord
    previous: CredentialRecord | None
    revoked_kids: frozenset[str]
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def authenticate_headers(
        self,
        raw_headers: Sequence[tuple[bytes, bytes]],
        *,
        required_scope: str | None = None,
        now: datetime | None = None,
    ) -> AuthenticatedPrincipal:
        parsed = parse_authorization_headers(raw_headers)
        record = self._record_for_kid(parsed.kid)
        if parsed.kid in self.revoked_kids:
            raise AuthenticationError()

        effective_now = _normalize_now(now if now is not None else self.now_fn())
        if not record.is_active(effective_now):
            raise AuthenticationError()

        digest = hashlib.sha256(parsed.credential.encode("ascii")).hexdigest()
        if not hmac.compare_digest(digest, record.token_hash_sha256):
            raise AuthenticationError()

        if required_scope is not None:
            validate_scope_literal(required_scope)
            if required_scope not in record.scopes:
                raise AuthorizationError()

        return AuthenticatedPrincipal(kid=record.kid, scopes=record.scopes)

    def _record_for_kid(self, kid: str) -> CredentialRecord:
        if kid == self.current.kid:
            return self.current
        if self.previous is not None and kid == self.previous.kid:
            return self.previous
        raise AuthenticationError()


def parse_authorization_headers(
    raw_headers: Sequence[tuple[bytes, bytes]],
) -> ParsedBearerCredential:
    values: list[bytes] = []
    for name, value in raw_headers:
        try:
            header_name = name.decode("ascii")
        except UnicodeDecodeError as exc:
            raise AuthenticationError() from exc
        if header_name.lower() == "authorization":
            values.append(value)
    if len(values) != 1:
        raise AuthenticationError()

    try:
        header_value = values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise AuthenticationError() from exc

    if len(header_value) < len("Bearer ") + 1:
        raise AuthenticationError()
    if header_value[:6].lower() != "bearer" or header_value[6] != " ":
        raise AuthenticationError()
    credential = header_value[7:]
    if not credential or any(ch.isspace() for ch in credential):
        raise AuthenticationError()
    if len(credential) > MAX_CREDENTIAL_LENGTH:
        raise AuthenticationError()
    if credential.count(".") != 1:
        raise AuthenticationError()

    kid, secret = credential.split(".", 1)
    validate_kid(kid)
    if not SECRET_RE.fullmatch(secret):
        raise AuthenticationError()
    return ParsedBearerCredential(kid=kid, credential=credential)


def load_credentials_file(
    path: Path,
    *,
    expected_group: str = "kitt-worker",
    expected_root_uid: int = 0,
    expected_root_gid: int = 0,
    expected_dir_uid: int = 0,
    expected_dir_gid: int | None = None,
    expected_file_uid: int = 0,
    expected_file_gid: int | None = None,
    credential_root: Path = DEFAULT_CREDENTIAL_ROOT,
    credential_dir: Path = DEFAULT_CREDENTIAL_DIR,
    now_fn: Callable[[], datetime] | None = None,
) -> CredentialStore:
    validate_credentials_file_security(
        path,
        expected_group=expected_group,
        expected_root_uid=expected_root_uid,
        expected_root_gid=expected_root_gid,
        expected_dir_uid=expected_dir_uid,
        expected_dir_gid=expected_dir_gid,
        expected_file_uid=expected_file_uid,
        expected_file_gid=expected_file_gid,
        credential_root=credential_root,
        credential_dir=credential_dir,
    )
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthConfigError() from exc
    return parse_credentials_payload(payload, now_fn=now_fn)


def validate_credentials_file_security(
    path: Path,
    *,
    expected_group: str = "kitt-worker",
    expected_root_uid: int = 0,
    expected_root_gid: int = 0,
    expected_dir_uid: int = 0,
    expected_dir_gid: int | None = None,
    expected_file_uid: int = 0,
    expected_file_gid: int | None = None,
    credential_root: Path = DEFAULT_CREDENTIAL_ROOT,
    credential_dir: Path = DEFAULT_CREDENTIAL_DIR,
) -> None:
    path = Path(path)
    credential_root = Path(credential_root)
    credential_dir = Path(credential_dir)
    if not path.is_absolute():
        raise AuthConfigError()
    if path.parent != credential_dir:
        raise AuthConfigError()
    _reject_prohibited_credential_path(path)

    group_gid = None
    if expected_dir_gid is None or expected_file_gid is None:
        group_gid = _group_gid(expected_group)
    dir_gid = group_gid if expected_dir_gid is None else expected_dir_gid
    file_gid = group_gid if expected_file_gid is None else expected_file_gid

    root_st = _lstat_existing(credential_root)
    _require_directory(
        root_st,
        uid=expected_root_uid,
        gid=expected_root_gid,
        mode=REQUIRED_ETC_ROOT_MODE,
    )
    if root_st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise AuthConfigError()

    dir_st = _lstat_existing(credential_dir)
    _require_directory(
        dir_st,
        uid=expected_dir_uid,
        gid=dir_gid,
        mode=REQUIRED_CREDENTIAL_DIR_MODE,
    )

    file_st = _lstat_existing(path)
    if stat.S_ISLNK(file_st.st_mode):
        raise AuthConfigError()
    if not stat.S_ISREG(file_st.st_mode):
        raise AuthConfigError()
    if stat.S_IMODE(file_st.st_mode) != REQUIRED_CREDENTIAL_FILE_MODE:
        raise AuthConfigError()
    if file_st.st_uid != expected_file_uid or file_st.st_gid != file_gid:
        raise AuthConfigError()


def parse_credentials_payload(
    payload: Any,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> CredentialStore:
    if not isinstance(payload, dict):
        raise AuthConfigError()
    if set(payload) - _TOP_LEVEL_FIELDS:
        raise AuthConfigError()
    if payload.get("version") != 1:
        raise AuthConfigError()
    if "current" not in payload:
        raise AuthConfigError()

    current = _parse_record(payload["current"], require_not_after=False)
    previous = None
    if "previous" in payload:
        previous = _parse_record(payload["previous"], require_not_after=True)
        if previous.kid == current.kid:
            raise AuthConfigError()

    revoked_kids = _parse_revoked_kids(payload.get("revoked_kids", []))
    clock = now_fn if now_fn is not None else lambda: datetime.now(timezone.utc)
    return CredentialStore(
        current=current,
        previous=previous,
        revoked_kids=revoked_kids,
        now_fn=clock,
    )


def validate_kid(kid: str) -> None:
    if not isinstance(kid, str) or not KID_RE.fullmatch(kid):
        raise AuthenticationError()


def validate_scope_literal(scope: str) -> None:
    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise AuthConfigError()


def redact_authorization_header(value: object) -> str:
    text = "" if value is None else str(value)
    if text.lower().startswith("authorization:"):
        return "Authorization: <redacted>"
    return "Authorization: <redacted>"


def redact_token(_value: object) -> str:
    return "<redacted-token>"


def redact_credential_path(_value: object) -> str:
    return "<redacted-credential-path>"


def redact_headers(headers: Mapping[str, object] | Iterable[tuple[str, object]]) -> dict[str, str]:
    items = headers.items() if isinstance(headers, Mapping) else headers
    redacted: dict[str, str] = {}
    for key, value in items:
        if key.lower() == "authorization":
            redacted[key] = "<redacted>"
        else:
            redacted[key] = redact_diagnostic(str(value))
    return redacted


def redact_diagnostic(value: object) -> str:
    text = "" if value is None else str(value)
    text = _BEARER_VALUE_RE.sub("Bearer <redacted>", text)
    text = _CREDENTIAL_VALUE_RE.sub("<redacted-token>", text)
    text = _SHA256_VALUE_RE.sub("<redacted-sha256>", text)
    text = _LOCAL_PATH_RE.sub("<redacted-path>", text)
    return text


def credential_hash(credential: str) -> str:
    return hashlib.sha256(credential.encode("ascii")).hexdigest()


def _parse_record(raw: Any, *, require_not_after: bool) -> CredentialRecord:
    if not isinstance(raw, dict):
        raise AuthConfigError()
    fields = set(raw)
    if fields & _FORBIDDEN_HASH_FIELDS:
        raise AuthConfigError()
    if fields - _RECORD_FIELDS:
        raise AuthConfigError()

    kid = raw.get("kid")
    if not isinstance(kid, str) or not KID_RE.fullmatch(kid):
        raise AuthConfigError()

    token_hash = raw.get("token_hash_sha256")
    if not isinstance(token_hash, str) or not SHA256_HEX_RE.fullmatch(token_hash):
        raise AuthConfigError()

    scopes = _parse_scopes(raw.get("scopes"))
    not_before = _parse_optional_time(raw.get("not_before"), "not_before")
    not_after = _parse_optional_time(raw.get("not_after"), "not_after")
    if require_not_after and not_after is None:
        raise AuthConfigError()
    if not_before is not None and not_after is not None and not_after <= not_before:
        raise AuthConfigError()

    return CredentialRecord(
        kid=kid,
        token_hash_sha256=token_hash,
        scopes=scopes,
        not_before=not_before,
        not_after=not_after,
    )


def _parse_scopes(raw: Any) -> frozenset[str]:
    if not isinstance(raw, list) or not raw:
        raise AuthConfigError()
    scopes: list[str] = []
    for scope in raw:
        if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
            raise AuthConfigError()
        if scope in scopes:
            raise AuthConfigError()
        scopes.append(scope)
    return frozenset(scopes)


def _parse_optional_time(raw: Any, field: str) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not RFC3339_UTC_RE.fullmatch(raw):
        raise AuthConfigError()
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise AuthConfigError() from exc
    if parsed.tzinfo != timezone.utc:
        raise AuthConfigError()
    return parsed


def _parse_revoked_kids(raw: Any) -> frozenset[str]:
    if not isinstance(raw, list):
        raise AuthConfigError()
    revoked: list[str] = []
    for kid in raw:
        if not isinstance(kid, str) or not KID_RE.fullmatch(kid):
            raise AuthConfigError()
        if kid in revoked:
            raise AuthConfigError()
        revoked.append(kid)
    return frozenset(revoked)


def _normalize_now(now: datetime) -> datetime:
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise AuthConfigError()
    return now.astimezone(timezone.utc)


def _group_gid(expected_group: str) -> int:
    try:
        return grp.getgrnam(expected_group).gr_gid
    except KeyError as exc:
        raise AuthConfigError() from exc


def _lstat_existing(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise AuthConfigError() from exc


def _require_directory(st: os.stat_result, *, uid: int, gid: int, mode: int) -> None:
    if stat.S_ISLNK(st.st_mode):
        raise AuthConfigError()
    if not stat.S_ISDIR(st.st_mode):
        raise AuthConfigError()
    if stat.S_IMODE(st.st_mode) != mode:
        raise AuthConfigError()
    if st.st_uid != uid or st.st_gid != gid:
        raise AuthConfigError()


def _reject_prohibited_credential_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    for root in _PROHIBITED_CREDENTIAL_ROOTS:
        root_resolved = root.resolve(strict=False)
        if resolved == root_resolved or resolved.is_relative_to(root_resolved):
            raise AuthConfigError()
