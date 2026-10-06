"""Bounded GGUF metadata inspection and FD-bound verification of local artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from itertools import chain
import os
from pathlib import Path
import stat
import struct
from types import MappingProxyType

from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType

from .errors import InvalidReferenceError, LoaderMetadataError, LocalModelNotFoundError
from .models import LocalArtifactFile, ValidatedLocalModel, _digest


DEFAULT_GGUF_MODEL_ROOT = Path("/usr/lib/kiron/data/gguf-models")
MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_METADATA_ITEMS = 4096
MAX_ARRAY_ITEMS = 1024 * 1024
MAX_FILE_BYTES = 256 * 1024 ** 3


@dataclass(frozen=True, slots=True)
class GGUFRegistrationPolicy:
    """Operator-approved artifact/profile pairing; never supplied by an API client."""

    model_sha256: str
    architecture: str
    projector_sha256: str | None = None

    def __post_init__(self):
        if not _digest(self.model_sha256) or (self.projector_sha256 is not None and not _digest(self.projector_sha256)):
            raise ValueError("policy requires pinned artifact hashes")
        if type(self.architecture) is not str or not self.architecture.isascii() or not self.architecture.isalnum():
            raise ValueError("policy requires an explicit architecture")


def _path(reference, root):
    if type(reference) is not str or len(reference) > 4096 or reference != reference.strip():
        raise InvalidReferenceError()
    if any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in reference):
        raise InvalidReferenceError()
    path = Path(reference)
    if not path.is_absolute() or str(path) != reference or ".." in path.parts or path.suffix != ".gguf":
        raise InvalidReferenceError()
    if root not in path.parents:
        raise InvalidReferenceError()
    return path


def _open_regular(path, *, root, owner_uid):
    """Traverse from / with openat+O_NOFOLLOW; retain the verified file descriptor."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for depth, part in enumerate(path.parts[1:-1], start=1):
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            if depth >= len(root.parts) - 1:
                directory = os.fstat(fd)
                if directory.st_uid != owner_uid or directory.st_mode & 0o022:
                    raise InvalidReferenceError()
        result = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    info = os.fstat(result)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or info.st_mode & 0o022
            or info.st_nlink != 1 or not 0 < info.st_size <= MAX_FILE_BYTES):
        os.close(result)
        raise InvalidReferenceError()
    return result, info


def read_gguf_metadata(stream):
    """Read only bounded GGUF v3 metadata from an already-held binary stream."""
    def read(size):
        if size < 0 or stream.tell() + size > MAX_METADATA_BYTES:
            raise ValueError("metadata budget exceeded")
        value = stream.read(size)
        if len(value) != size:
            raise ValueError("truncated metadata")
        return value

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

    def string():
        size = number("Q")
        if size > 4 * 1024 * 1024:
            raise ValueError("metadata string too large")
        return read(size).decode("utf-8")

    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind, depth=0):
        if kind == 8:
            return string()
        if kind == 9:
            subtype, count = number("I"), number("Q")
            if depth or count > MAX_ARRAY_ITEMS:
                raise ValueError("invalid metadata array")
            for _ in range(count):
                value(subtype, depth + 1)
            return None
        if kind not in formats:
            raise ValueError("unknown metadata type")
        return number(formats[kind])

    if read(4) != b"GGUF" or number("I") != 3:
        raise ValueError("unsupported GGUF header")
    tensors, count = number("Q"), number("Q")
    if not 0 < tensors <= MAX_ARRAY_ITEMS or not 0 < count <= MAX_METADATA_ITEMS:
        raise ValueError("invalid GGUF counts")
    result, seen = {}, set()
    for _ in range(count):
        key = string()
        if key in seen or len(key) > 1024:
            raise ValueError("duplicate or oversized metadata key")
        seen.add(key)
        item = value(number("I"))
        if key in ("general.architecture", "general.name", "clip.vision.projection_dim") or key.endswith(".embedding_length"):
            result[key] = item
    return result


def inspect_gguf(reference, *, root=DEFAULT_GGUF_MODEL_ROOT, expected_sha256, owner_uid=0):
    """Hash the opened file, not a reopened pathname; do not load any model tensors."""
    path = _path(reference, root)
    try:
        fd, before = _open_regular(path, root=root, owner_uid=owner_uid)
        with os.fdopen(fd, "rb") as stream:
            metadata = read_gguf_metadata(stream)
            stream.seek(0)
            digest = hashlib.sha256()
            count = 0
            while block := stream.read(1024 * 1024):
                count += len(block)
                if count > before.st_size:
                    raise ValueError("artifact changed")
                digest.update(block)
            after = os.fstat(stream.fileno())
        current_fd, current = _open_regular(path, root=root, owner_uid=owner_uid)
        os.close(current_fd)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
        if (count != before.st_size or signature(before) != signature(after)
                or signature(after) != signature(current) or digest.hexdigest() != expected_sha256):
            raise ValueError("artifact identity differs")
        return LocalArtifactFile(str(path), digest.hexdigest(), count), metadata
    except FileNotFoundError:
        raise LocalModelNotFoundError() from None
    except (OSError, UnicodeError, ValueError, struct.error):
        raise LoaderMetadataError() from None


class GGUFLocalValidator:
    def __init__(self, *, model_root=DEFAULT_GGUF_MODEL_ROOT, profiles=None, owner_uid=0):
        if not isinstance(model_root, Path) or not model_root.is_absolute() or ".." in model_root.parts:
            raise ValueError("GGUF root must be an absolute canonical path")
        if profiles is not None and not isinstance(profiles, Mapping):
            raise TypeError("GGUF profiles must be a mapping")
        policies = dict(profiles or {})
        if any(type(key) is not str or type(policy) is not GGUFRegistrationPolicy for key, policy in policies.items()):
            raise TypeError("invalid GGUF profile policy")
        self.root, self.profiles, self.owner_uid = model_root, MappingProxyType(policies), owner_uid

    def available_references(self):
        if not self.profiles or not self.root.exists():
            return ()
        # Discovery is descriptive; registration verifies descriptors and pinned hashes.
        from .validators import _canonical_local_directory
        _canonical_local_directory(str(self.root))
        result, examined = [], 0
        try:
            for parent in chain((self.root,), self.root.iterdir()):
                if parent.is_symlink() or not parent.is_dir():
                    continue
                for path in parent.iterdir():
                    examined += 1
                    if examined > MAX_METADATA_ITEMS:
                        raise LoaderMetadataError()
                    if path.suffix == ".gguf" and path.is_file() and not path.is_symlink():
                        result.append(str(path))
        except OSError:
            raise LoaderMetadataError() from None
        return tuple(sorted(set(result)))

    def validate(self, reference, *, runtime_profile, projector_reference=None,
                 expected_sha256=None, expected_projector_sha256=None):
        policy = self.profiles.get(runtime_profile) if type(runtime_profile) is str else None
        if policy is None or (expected_sha256 is not None and expected_sha256 != policy.model_sha256):
            raise LoaderMetadataError()
        primary, metadata = inspect_gguf(reference, root=self.root, expected_sha256=policy.model_sha256, owner_uid=self.owner_uid)
        if metadata.get("general.architecture") != policy.architecture:
            raise LoaderMetadataError()
        projector = None
        if projector_reference is not None:
            if policy.projector_sha256 is None or (expected_projector_sha256 is not None and expected_projector_sha256 != policy.projector_sha256):
                raise LoaderMetadataError()
            projector, projection = inspect_gguf(projector_reference, root=self.root, expected_sha256=policy.projector_sha256, owner_uid=self.owner_uid)
            dimension = metadata.get(f"{policy.architecture}.embedding_length")
            if (projection.get("general.architecture") != "clip" or type(dimension) is not int or dimension <= 0
                    or projection.get("clip.vision.projection_dim") != dimension or projector.reference == primary.reference):
                raise LoaderMetadataError()
        elif expected_projector_sha256 is not None:
            raise LoaderMetadataError()
        name = metadata.get("general.name")
        return ValidatedLocalModel(runtime_provider=BackendType.PRISM, artifact_origin=ArtifactType.LOCAL,
                                   artifact_format=ArtifactFormat.GGUF, reference=primary.reference,
                                   display_name=name if type(name) is str and name else Path(reference).stem,
                                   loader=LoaderType.PRISM_GGUF, sha256=primary.sha256, size_bytes=primary.size_bytes,
                                   projector=projector, runtime_profile=runtime_profile)
