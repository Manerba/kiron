"""Atomic, process-safe persistence for dynamic local model entries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator

from .codec import MAX_REGISTRY_BYTES, decode_registry, encode_registry
from .errors import (
    DuplicateModelError,
    RegistryAccessError,
    RegistryCorruptionError,
)
from .models import RegistryEntry


DEFAULT_REGISTRY_PATH = Path(
    "/usr/lib/kiron/data/shared/local-model-registry.json"
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_REGISTRY_MODE = 0o640
_LOCK_MODE = 0o640
_HOSTILE_OPEN_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ELOOP", None),
        getattr(errno, "EISDIR", None),
        getattr(errno, "ENXIO", None),
    )
    if value is not None
)


@dataclass(frozen=True, slots=True)
class RegistryFilePolicy:
    """Numeric owner policy; the default binds to the registry parent.

    Parent binding lets a root caller publish files for the existing service
    owner without resolving account or group names inside the common package.
    Explicit numeric ownership is available for controlled compositions.
    """

    owner_uid: int | None = None
    group_gid: int | None = None

    def __post_init__(self) -> None:
        if (self.owner_uid is None) != (self.group_gid is None):
            raise ValueError("owner_uid and group_gid must be configured together")
        for value in (self.owner_uid, self.group_gid):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("registry ownership ids must be non-negative integers")


@dataclass(frozen=True, slots=True)
class _ResolvedFilePolicy:
    owner_uid: int
    group_gid: int


def _raise_open_failure(error: OSError) -> None:
    if error.errno in _HOSTILE_OPEN_ERRNOS:
        raise RegistryCorruptionError() from None
    raise RegistryAccessError() from None


class RuntimeModelRegistry:
    """A separate runtime overlay; it never writes Catalog source manifests."""

    __slots__ = ("_file_policy", "_lock_path", "_path", "_readonly")

    def __init__(
        self,
        path: Path = DEFAULT_REGISTRY_PATH,
        *,
        file_policy: RegistryFilePolicy | None = None,
        readonly: bool = False,
    ) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            raise ValueError("registry path must be absolute")
        if resolved.name in ("", ".", ".."):
            raise ValueError("registry path must name a file")
        if file_policy is not None and type(file_policy) is not RegistryFilePolicy:
            raise TypeError("file_policy must be a RegistryFilePolicy")
        if type(readonly) is not bool:
            raise TypeError("readonly must be boolean")
        self._path = resolved
        self._lock_path = resolved.with_name(resolved.name + ".lock")
        self._file_policy = file_policy or RegistryFilePolicy()
        self._readonly = readonly

    @property
    def path(self) -> Path:
        return self._path

    def _resolved_file_policy(self) -> _ResolvedFilePolicy:
        parent = self._path.parent
        try:
            descriptor = os.open(
                parent,
                os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC,
            )
        except FileNotFoundError:
            raise
        except OSError:
            raise RegistryAccessError() from None
        try:
            try:
                info = os.fstat(descriptor)
            except OSError:
                raise RegistryAccessError() from None
            if not stat.S_ISDIR(info.st_mode):
                raise RegistryAccessError()
            return _ResolvedFilePolicy(
                owner_uid=(
                    info.st_uid
                    if self._file_policy.owner_uid is None
                    else self._file_policy.owner_uid
                ),
                group_gid=(
                    info.st_gid
                    if self._file_policy.group_gid is None
                    else self._file_policy.group_gid
                ),
            )
        finally:
            os.close(descriptor)

    @staticmethod
    def _descriptor_info(descriptor: int) -> os.stat_result:
        try:
            return os.fstat(descriptor)
        except OSError:
            raise RegistryAccessError() from None

    @classmethod
    def _verify_regular_descriptor(
        cls,
        descriptor: int,
        policy: _ResolvedFilePolicy,
        *,
        mode: int,
    ) -> os.stat_result:
        info = cls._descriptor_info(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegistryCorruptionError()
        if (
            info.st_uid != policy.owner_uid
            or info.st_gid != policy.group_gid
            or stat.S_IMODE(info.st_mode) != mode
        ):
            raise RegistryAccessError()
        return info

    @classmethod
    def _normalize_created_descriptor(
        cls,
        descriptor: int,
        policy: _ResolvedFilePolicy,
        *,
        mode: int,
    ) -> None:
        info = cls._descriptor_info(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegistryCorruptionError()
        try:
            if (
                info.st_uid != policy.owner_uid
                or info.st_gid != policy.group_gid
            ):
                if os.geteuid() != 0:
                    raise RegistryAccessError()
                os.fchown(descriptor, policy.owner_uid, policy.group_gid)
            os.fchmod(descriptor, mode)
        except RegistryAccessError:
            raise
        except OSError:
            raise RegistryAccessError() from None
        cls._verify_regular_descriptor(descriptor, policy, mode=mode)

    @staticmethod
    def _remove_created_path(path: Path, descriptor: int) -> None:
        try:
            descriptor_info = os.fstat(descriptor)
            path_info = os.lstat(path)
            if (
                descriptor_info.st_dev == path_info.st_dev
                and descriptor_info.st_ino == path_info.st_ino
            ):
                os.unlink(path)
        except OSError:
            pass

    def _open_lock(self, policy: _ResolvedFilePolicy) -> int:
        if self._readonly:
            try:
                descriptor = os.open(self._lock_path, os.O_RDONLY | _NOFOLLOW | _CLOEXEC | _NONBLOCK)
            except FileNotFoundError:
                raise RegistryAccessError() from None
            except OSError as error:
                _raise_open_failure(error)
            try:
                self._verify_regular_descriptor(descriptor, policy, mode=_LOCK_MODE)
                return descriptor
            except BaseException:
                os.close(descriptor)
                raise
        flags = os.O_RDWR | _NOFOLLOW | _CLOEXEC | _NONBLOCK
        for _attempt in range(8):
            created = False
            try:
                descriptor = os.open(
                    self._lock_path,
                    flags | os.O_CREAT | os.O_EXCL,
                    _LOCK_MODE,
                )
                created = True
            except FileExistsError:
                try:
                    descriptor = os.open(self._lock_path, flags)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    _raise_open_failure(error)
            except FileNotFoundError:
                raise
            except OSError as error:
                _raise_open_failure(error)
            try:
                if created:
                    self._normalize_created_descriptor(
                        descriptor,
                        policy,
                        mode=_LOCK_MODE,
                    )
                else:
                    self._verify_regular_descriptor(
                        descriptor,
                        policy,
                        mode=_LOCK_MODE,
                    )
                return descriptor
            except BaseException:
                if created:
                    self._remove_created_path(self._lock_path, descriptor)
                os.close(descriptor)
                raise
        raise RegistryAccessError()

    @contextmanager
    def _locked(
        self,
        *,
        exclusive: bool,
    ) -> Iterator[_ResolvedFilePolicy]:
        if self._readonly and exclusive:
            raise RegistryAccessError()
        try:
            policy = self._resolved_file_policy()
        except FileNotFoundError:
            if self._readonly:
                raise RegistryAccessError() from None
            raise
        descriptor = self._open_lock(policy)
        try:
            try:
                fcntl.flock(
                    descriptor,
                    fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
                )
            except OSError:
                raise RegistryAccessError() from None
            yield policy
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_unlocked(
        self,
        policy: _ResolvedFilePolicy,
    ) -> tuple[RegistryEntry, ...]:
        try:
            descriptor = os.open(
                self._path,
                os.O_RDONLY | _NOFOLLOW | _CLOEXEC | _NONBLOCK,
            )
        except FileNotFoundError:
            if self._readonly:
                raise RegistryAccessError() from None
            return ()
        except OSError as error:
            _raise_open_failure(error)
        try:
            info = self._verify_regular_descriptor(
                descriptor,
                policy,
                mode=_REGISTRY_MODE,
            )
            if (
                info.st_size < 1
                or info.st_size > MAX_REGISTRY_BYTES
            ):
                raise RegistryCorruptionError()
            chunks: list[bytes] = []
            remaining = info.st_size
            while remaining:
                try:
                    chunk = os.read(descriptor, min(remaining, 65536))
                except OSError:
                    raise RegistryAccessError() from None
                if not chunk:
                    raise RegistryCorruptionError()
                chunks.append(chunk)
                remaining -= len(chunk)
            try:
                trailing = os.read(descriptor, 1)
            except OSError:
                raise RegistryAccessError() from None
            if trailing:
                raise RegistryCorruptionError()
            return decode_registry(b"".join(chunks))
        finally:
            os.close(descriptor)

    def _write_unlocked(
        self,
        entries: tuple[RegistryEntry, ...],
        policy: _ResolvedFilePolicy,
    ) -> None:
        payload = encode_registry(entries)
        parent = self._path.parent
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".local-model-registry.",
                suffix=".tmp",
                dir=parent,
            )
        except FileNotFoundError:
            raise
        except OSError:
            raise RegistryAccessError() from None
        temporary_path = Path(temporary)
        try:
            self._normalize_created_descriptor(
                descriptor,
                policy,
                mode=_REGISTRY_MODE,
            )
            written = 0
            try:
                while written < len(payload):
                    written += os.write(descriptor, payload[written:])
                os.fsync(descriptor)
            except OSError:
                raise RegistryAccessError() from None
            self._verify_regular_descriptor(
                descriptor,
                policy,
                mode=_REGISTRY_MODE,
            )
            try:
                os.replace(temporary_path, self._path)
            except PermissionError:
                raise RegistryAccessError() from None
            self._verify_regular_descriptor(
                descriptor,
                policy,
                mode=_REGISTRY_MODE,
            )
            try:
                directory = os.open(
                    parent,
                    os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW | _CLOEXEC,
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                raise RegistryAccessError() from None
        finally:
            os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def list(self) -> tuple[RegistryEntry, ...]:
        with self._locked(exclusive=False) as policy:
            return self._read_unlocked(policy)

    def get(self, entry_id: object) -> RegistryEntry | None:
        if type(entry_id) is not str:
            return None
        return next((entry for entry in self.list() if entry.id == entry_id), None)

    def add(self, entry: RegistryEntry) -> RegistryEntry:
        if type(entry) is not RegistryEntry:
            raise TypeError("entry must be a RegistryEntry")
        with self._locked(exclusive=True) as policy:
            current = self._read_unlocked(policy)
            if any(
                existing.id == entry.id
                or (
                    existing.runtime_provider is entry.runtime_provider
                    and existing.reference == entry.reference
                )
                for existing in current
            ):
                raise DuplicateModelError()
            self._write_unlocked((*current, entry), policy)
        return entry


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "RegistryFilePolicy",
    "RuntimeModelRegistry",
]
