"""Gate-closed worker-local artifact staging foundation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import os
from pathlib import Path
import shutil
import stat
from typing import Callable, Iterable

import job_stubs
import queue_store


SERVICE_USER = "kitt-worker"
SERVICE_GROUP = "kitt-worker"
REQUIRED_STAGING_MODE = 0o750
STAGED_FILE_MODE = 0o640
BLOCKER_UNAVAILABLE = "artifact_staging_unavailable"
BLOCKER_PRODUCER_GATE = "artifact_producer_gate_closed"


@dataclass(frozen=True, slots=True)
class ValidatedStagingRoot:
    root: Path
    tmp_dir: Path
    objects_dir: Path
    free_bytes: int


@dataclass(frozen=True, slots=True)
class StageResult:
    status: str
    artifact: queue_store.ArtifactRecord | None
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class CleanupResult:
    deleted_files: int = 0
    recovered_rows: int = 0
    failed_deletes: int = 0
    reason_code: str | None = None


class ArtifactStagingError(RuntimeError):
    def __init__(self, reason_code: str = BLOCKER_UNAVAILABLE) -> None:
        self.reason_code = _safe_reason(reason_code)
        super().__init__(self.reason_code)


class ArtifactStreamSizeExceeded(RuntimeError):
    pass


class ArtifactStager:
    def __init__(
        self,
        *,
        store: queue_store.QueueStore,
        root: ValidatedStagingRoot,
        quotas: queue_store.ArtifactQuotaSettings,
        cleanup_after_seconds: int = 0,
        disk_usage_fn: Callable[[Path], object] = shutil.disk_usage,
    ) -> None:
        self.store = store
        self.root = root
        self.quotas = quotas
        self.cleanup_after_seconds = max(0, int(cleanup_after_seconds))
        self._disk_usage_fn = disk_usage_fn
        self._active_artifacts: set[str] = set()

    @classmethod
    def from_config(
        cls,
        *,
        cfg: object,
        store: queue_store.QueueStore,
        expected_uid: int | None = None,
        expected_gid: int | None = None,
        disk_usage_fn: Callable[[Path], object] = shutil.disk_usage,
    ) -> "ArtifactStager":
        root = validate_staging_root(
            cfg,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            disk_usage_fn=disk_usage_fn,
        )
        return cls(
            store=store,
            root=root,
            quotas=quota_settings_from_config(cfg),
            cleanup_after_seconds=int(getattr(cfg, "artifact_cleanup_after_seconds", 0)),
            disk_usage_fn=disk_usage_fn,
        )

    def stage_bytes(
        self,
        *,
        job_uid: str,
        role: str,
        data: bytes,
        expected_size_bytes: int,
        expected_sha256: str | None = None,
    ) -> StageResult:
        if not isinstance(data, bytes):
            raise ArtifactStagingError("artifact_input_invalid")
        return self.stage_iterable(
            job_uid=job_uid,
            role=role,
            chunks=(data,),
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
        )

    def stage_iterable(
        self,
        *,
        job_uid: str,
        role: str,
        chunks: Iterable[bytes],
        expected_size_bytes: int,
        expected_sha256: str | None = None,
    ) -> StageResult:
        if expected_size_bytes is None:
            raise ArtifactStagingError("artifact_expected_size_required")
        if expected_sha256 is not None and not _is_sha256(expected_sha256):
            raise ArtifactStagingError("artifact_sha256_invalid")
        self._ensure_internal_dirs()
        free_bytes = _free_bytes(self.root.root, self._disk_usage_fn)
        reservation = self.store.reserve_artifact(
            job_uid=job_uid,
            role=role,
            expected_size_bytes=expected_size_bytes,
            quotas=self.quotas,
            staging_free_bytes=free_bytes,
        )
        if reservation.status == "quota_blocked":
            return StageResult(
                status=reservation.status,
                artifact=reservation,
                failure_code=reservation.failure_code,
            )

        self._active_artifacts.add(reservation.artifact_uid)
        temp_path = self.root.tmp_dir / f"{reservation.artifact_uid}.tmp"
        final_path = self.root.objects_dir / reservation.artifact_uid
        try:
            try:
                actual_sha256, actual_size = self._write_stream(
                    temp_path,
                    chunks,
                    expected_size_bytes=expected_size_bytes,
                )
            except ArtifactStreamSizeExceeded:
                _unlink_if_exists(temp_path)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_size_mismatch",
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)
            except OSError as exc:
                _unlink_if_exists(temp_path)
                if _is_disk_full(exc):
                    artifact = self.store.mark_artifact_disk_full(
                        artifact_uid=reservation.artifact_uid
                    )
                    return StageResult("disk_full", artifact, artifact.failure_code)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_write_failed",
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            if actual_size != expected_size_bytes:
                _unlink_if_exists(temp_path)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_size_mismatch",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                _unlink_if_exists(temp_path)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_hash_mismatch",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            try:
                os.replace(temp_path, final_path)
                os.chmod(final_path, STAGED_FILE_MODE)
            except OSError as exc:
                _unlink_if_exists(temp_path)
                _unlink_if_exists(final_path)
                if _is_disk_full(exc):
                    artifact = self.store.mark_artifact_disk_full(
                        artifact_uid=reservation.artifact_uid
                    )
                    return StageResult("disk_full", artifact, artifact.failure_code)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_finalize_failed",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            try:
                artifact = self.store.mark_artifact_staged(
                    artifact_uid=reservation.artifact_uid,
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
            except Exception:
                _unlink_if_exists(final_path)
                raise
            if artifact.artifact_uid != reservation.artifact_uid or artifact.status != "staged":
                _unlink_if_exists(final_path)
            return StageResult(artifact.status, artifact, artifact.failure_code)
        finally:
            self._active_artifacts.discard(reservation.artifact_uid)

    def stage_file(
        self,
        *,
        job_uid: str,
        role: str,
        source_path: Path,
        expected_size_bytes: int,
        expected_sha256: str | None = None,
    ) -> StageResult:
        if expected_size_bytes is None:
            raise ArtifactStagingError("artifact_expected_size_required")
        if expected_sha256 is not None and not _is_sha256(expected_sha256):
            raise ArtifactStagingError("artifact_sha256_invalid")
        self._ensure_internal_dirs()
        source = Path(source_path)
        try:
            source_stat = source.lstat()
        except OSError as exc:
            raise ArtifactStagingError("artifact_input_invalid") from exc
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
            raise ArtifactStagingError("artifact_input_invalid")

        free_bytes = _free_bytes(self.root.root, self._disk_usage_fn)
        same_device = source_stat.st_dev == self.root.root.stat().st_dev
        additional_staging_bytes = 0 if same_device else expected_size_bytes
        reservation = self.store.reserve_artifact(
            job_uid=job_uid,
            role=role,
            expected_size_bytes=expected_size_bytes,
            quotas=self.quotas,
            staging_free_bytes=free_bytes,
            additional_staging_bytes=additional_staging_bytes,
        )
        if reservation.status == "quota_blocked":
            return StageResult(
                status=reservation.status,
                artifact=reservation,
                failure_code=reservation.failure_code,
            )

        self._active_artifacts.add(reservation.artifact_uid)
        temp_path = self.root.tmp_dir / f"{reservation.artifact_uid}.tmp"
        final_path = self.root.objects_dir / reservation.artifact_uid
        try:
            try:
                actual_sha256, actual_size = _hash_file(source)
            except OSError:
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_read_failed",
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            if actual_size != expected_size_bytes:
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_size_mismatch",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_hash_mismatch",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            try:
                if same_device:
                    os.replace(source, final_path)
                else:
                    self._copy_file(source, temp_path, expected_size_bytes)
                    os.replace(temp_path, final_path)
                os.chmod(final_path, STAGED_FILE_MODE)
            except OSError as exc:
                _unlink_if_exists(temp_path)
                _unlink_if_exists(final_path)
                if _is_disk_full(exc):
                    artifact = self.store.mark_artifact_disk_full(
                        artifact_uid=reservation.artifact_uid
                    )
                    return StageResult("disk_full", artifact, artifact.failure_code)
                artifact = self.store.mark_artifact_verification_failed(
                    artifact_uid=reservation.artifact_uid,
                    failure_code="artifact_finalize_failed",
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
                return StageResult(artifact.status, artifact, artifact.failure_code)

            try:
                artifact = self.store.mark_artifact_staged(
                    artifact_uid=reservation.artifact_uid,
                    sha256=actual_sha256,
                    size_bytes=actual_size,
                )
            except Exception:
                _unlink_if_exists(final_path)
                raise
            if (
                artifact.artifact_uid != reservation.artifact_uid
                or artifact.status != "staged"
            ):
                _unlink_if_exists(final_path)
            return StageResult(artifact.status, artifact, artifact.failure_code)
        finally:
            self._active_artifacts.discard(reservation.artifact_uid)

    def cleanup(self) -> CleanupResult:
        self._ensure_internal_dirs()
        deleted = 0
        recovered = 0
        failed = 0

        for path in self._iter_temp_files():
            artifact_uid = path.name.removesuffix(".tmp")
            if artifact_uid in self._active_artifacts:
                continue
            try:
                self.store.get_artifact(artifact_uid)
            except job_stubs.JobNotFoundError:
                result = _remove_file(path)
                if result == "deleted":
                    deleted += 1
                elif result == "failed":
                    failed += 1
            except queue_store.QueueUnavailable:
                failed += 1
            else:
                continue

        for artifact in self.store.list_artifacts_for_cleanup():
            if artifact.artifact_uid in self._active_artifacts:
                continue
            temp_path = self.root.tmp_dir / f"{artifact.artifact_uid}.tmp"
            object_path = self.root.objects_dir / artifact.artifact_uid
            if artifact.status == "staging":
                if _is_recent(artifact.updated_at, self.cleanup_after_seconds):
                    continue
                temp_result = _remove_file(temp_path)
                object_result = _remove_file(object_path)
                if temp_result == "deleted":
                    deleted += 1
                if object_result == "deleted":
                    deleted += 1
                if temp_result == "failed" or object_result == "failed":
                    failed += int(temp_result == "failed") + int(object_result == "failed")
                    continue
                self.store.mark_artifact_verification_failed(
                    artifact_uid=artifact.artifact_uid,
                    failure_code="artifact_staging_recovered",
                )
                recovered += 1
            elif artifact.status in {"verification_failed", "quota_blocked", "disk_full"}:
                temp_result = _remove_file(temp_path)
                object_result = _remove_file(object_path)
                if temp_result == "deleted":
                    deleted += 1
                if object_result == "deleted":
                    deleted += 1
                if temp_result == "failed" or object_result == "failed":
                    failed += int(temp_result == "failed") + int(object_result == "failed")
            elif artifact.status == "cleanup_pending":
                object_result = _remove_file(object_path)
                if object_result == "deleted":
                    deleted += 1
                    self.store.mark_artifact_deleted(
                        artifact_uid=artifact.artifact_uid
                    )
                elif object_result == "failed":
                    failed += 1
                else:
                    self.store.mark_artifact_deleted(
                        artifact_uid=artifact.artifact_uid
                    )
                    recovered += 1

        for path in self._iter_object_files():
            try:
                artifact = self.store.get_artifact(path.name)
            except job_stubs.JobNotFoundError:
                if _unlink_if_exists(path):
                    deleted += 1
                else:
                    failed += 1
            except queue_store.QueueUnavailable:
                failed += 1
            else:
                if artifact.status in {
                    "verification_failed",
                    "quota_blocked",
                    "disk_full",
                    "deleted",
                }:
                    result = _remove_file(path)
                    if result == "deleted":
                        deleted += 1
                    elif result == "failed":
                        failed += 1

        return CleanupResult(
            deleted_files=deleted,
            recovered_rows=recovered,
            failed_deletes=failed,
            reason_code="artifact_cleanup_failed" if failed else None,
        )

    def _write_stream(
        self,
        temp_path: Path,
        chunks: Iterable[bytes],
        *,
        expected_size_bytes: int,
    ) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        fd = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            STAGED_FILE_MODE,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise OSError(errno.EINVAL, "invalid artifact chunk")
                    if size + len(chunk) > expected_size_bytes:
                        raise ArtifactStreamSizeExceeded()
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
        os.chmod(temp_path, STAGED_FILE_MODE)
        return digest.hexdigest(), size

    def _copy_file(
        self,
        source_path: Path,
        temp_path: Path,
        expected_size_bytes: int,
    ) -> None:
        with source_path.open("rb") as handle:
            self._write_stream(
                temp_path,
                iter(lambda: handle.read(1024 * 1024), b""),
                expected_size_bytes=expected_size_bytes,
            )

    def _ensure_internal_dirs(self) -> None:
        for path in (self.root.tmp_dir, self.root.objects_dir):
            _ensure_child_dir(self.root.root, path)

    def _iter_temp_files(self) -> Iterable[Path]:
        if not self.root.tmp_dir.exists():
            return ()
        return (
            path
            for path in self.root.tmp_dir.iterdir()
            if path.is_file()
            and path.name.endswith(".tmp")
            and _is_artifact_uid(path.name.removesuffix(".tmp"))
        )

    def _iter_object_files(self) -> Iterable[Path]:
        if not self.root.objects_dir.exists():
            return ()
        return (
            path
            for path in self.root.objects_dir.iterdir()
            if path.is_file() and _is_artifact_uid(path.name)
        )


def quota_settings_from_config(cfg: object) -> queue_store.ArtifactQuotaSettings:
    return queue_store.ArtifactQuotaSettings(
        max_bytes=int(getattr(cfg, "artifact_max_bytes")),
        job_quota_bytes=int(getattr(cfg, "artifact_job_quota_bytes")),
        total_quota_bytes=int(getattr(cfg, "artifact_total_quota_bytes")),
        min_free_bytes=int(getattr(cfg, "artifact_min_free_bytes")),
    )


def validate_staging_root(
    cfg: object,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    disk_usage_fn: Callable[[Path], object] = shutil.disk_usage,
) -> ValidatedStagingRoot:
    config_error = getattr(cfg, "artifact_config_error_code", None)
    if config_error is not None:
        raise ArtifactStagingError(config_error)
    raw_root = Path(getattr(cfg, "artifact_staging_dir"))
    try:
        st = raw_root.lstat()
    except OSError as exc:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE) from exc
    if stat.S_ISLNK(st.st_mode):
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    root = raw_root.resolve(strict=False)
    data_root = Path(getattr(cfg, "data_dir")).resolve(strict=False)
    if root == data_root or not root.is_relative_to(data_root):
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    for prohibited in (
        Path("/opt/kiron"),
        Path("/usr/lib/kiron/services"),
        Path("/run/kiron"),
        Path("/etc/kiron"),
    ):
        resolved = prohibited.resolve(strict=False)
        if root == resolved or root.is_relative_to(resolved):
            raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    _validate_dir(root, expected_uid=expected_uid, expected_gid=expected_gid)
    free_bytes = _free_bytes(root, disk_usage_fn)
    return ValidatedStagingRoot(
        root=root,
        tmp_dir=root / "tmp",
        objects_dir=root / "objects",
        free_bytes=free_bytes,
    )


def _validate_dir(
    path: Path,
    *,
    expected_uid: int | None,
    expected_gid: int | None,
) -> None:
    try:
        st = path.lstat()
    except OSError as exc:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    if stat.S_IMODE(st.st_mode) != REQUIRED_STAGING_MODE:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    if expected_uid is None:
        expected_uid = os.geteuid()
    if expected_gid is None:
        expected_gid = os.getegid()
    if expected_uid is not None and st.st_uid != expected_uid:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    if expected_gid is not None and st.st_gid != expected_gid:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    if (
        expected_uid is not None
        and os.geteuid() == expected_uid
        and not os.access(path, os.W_OK | os.X_OK, effective_ids=True)
    ):
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)


def _ensure_child_dir(root: Path, path: Path) -> None:
    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
    try:
        if path.exists():
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise ArtifactStagingError(BLOCKER_UNAVAILABLE)
        else:
            path.mkdir(mode=REQUIRED_STAGING_MODE)
        os.chmod(path, REQUIRED_STAGING_MODE)
    except OSError as exc:
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE) from exc


def _free_bytes(path: Path, disk_usage_fn: Callable[[Path], object]) -> int:
    try:
        usage = disk_usage_fn(path)
        free = getattr(usage, "free", None)
        if isinstance(free, bool) or not isinstance(free, int) or free < 0:
            raise OSError(errno.EIO, "invalid disk usage")
        return int(free)
    except OSError as exc:
        if _is_disk_full(exc):
            raise ArtifactStagingError("artifact_disk_full") from exc
        raise ArtifactStagingError(BLOCKER_UNAVAILABLE) from exc


def _remove_file(path: Path) -> str:
    try:
        path.unlink()
        return "deleted"
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "failed"


def _unlink_if_exists(path: Path) -> bool:
    return _remove_file(path) == "deleted"


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_disk_full(exc: OSError) -> bool:
    return getattr(exc, "errno", None) == errno.ENOSPC


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_artifact_uid(value: str) -> bool:
    return (
        len(value) == 41
        and value.startswith("artifact_")
        and all(char in "0123456789abcdef" for char in value[9:])
    )


def _is_recent(value: str | None, cleanup_after_seconds: int) -> bool:
    if cleanup_after_seconds <= 0 or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc) < parsed + timedelta(seconds=cleanup_after_seconds)


def _safe_reason(reason_code: str) -> str:
    if isinstance(reason_code, str) and reason_code in {
        BLOCKER_UNAVAILABLE,
        BLOCKER_PRODUCER_GATE,
        "artifact_quota_invalid",
        "artifact_cleanup_failed",
        "artifact_disk_full",
        "artifact_expected_size_required",
        "artifact_input_invalid",
        "artifact_sha256_invalid",
    }:
        return reason_code
    return BLOCKER_UNAVAILABLE
