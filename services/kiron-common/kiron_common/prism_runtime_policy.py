"""Root-controlled launch policy and immutable local artifact checks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType


class PolicyError(ValueError):
    pass


def immutable_path(path: Path, *, anchor: Path = Path("/"), directory=False):
    """Reject writable/symlink ancestors; the explicit anchor is trusted by composition."""
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(anchor):
        raise PolicyError("path outside its immutable root")
    for current in (anchor, *(anchor.joinpath(*path.relative_to(anchor).parts[:n])
                              for n in range(1, len(path.relative_to(anchor).parts) + 1))):
        info = current.lstat()
        if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
            raise PolicyError("path is not root-owned and immutable")
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise PolicyError("non-directory ancestor")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise PolicyError("unexpected file type")
    return info


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid, info.st_nlink)


def checked_open(path, sha256, size_bytes=None, *, anchor=Path("/")):
    """Keep the verified descriptor open through spawn; never follow a final link."""
    before = immutable_path(path, anchor=anchor)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if opened.st_nlink != 1 or file_identity(opened) != file_identity(before):
            raise PolicyError("artifact changed while opening")
        if size_bytes is not None and opened.st_size != size_bytes:
            raise PolicyError("artifact size mismatch")
        with os.fdopen(os.dup(fd), "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != sha256 or file_identity(os.fstat(fd)) != file_identity(opened):
            raise PolicyError("artifact digest or identity mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd)
        raise


@dataclass(frozen=True)
class Profile:
    id: str
    gpu_layers: int
    threads: int
    model_sha256: str
    architecture: str
    gpu_memory_bytes: int
    host_memory_bytes: int
    memory_headroom_bytes: int
    projector_sha256: str | None = None
    context: int = 1024
    batch: int = 128

    def __post_init__(self):
        # Exactly the bounded profiles measured in #286, never implicit full offload.
        if (type(self.gpu_layers) is not int or type(self.threads) is not int
                or (self.gpu_layers, self.threads) not in {(0, 2), (32, 2), (40, 4)}
                or self.context != 1024 or self.batch != 128):
            raise PolicyError("resource values outside the measured profile")
        for digest in (self.model_sha256, self.projector_sha256):
            if digest is not None and (type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest)):
                raise PolicyError("profile requires pinned artifact digests")
        if self.model_sha256 is None or not re.fullmatch("[A-Za-z0-9]+", self.architecture):
            raise PolicyError("profile requires model digest and architecture")
        for value in (self.gpu_memory_bytes, self.host_memory_bytes, self.memory_headroom_bytes):
            if type(value) is not int or value < 0:
                raise PolicyError("profile requires measured resource bytes")


@dataclass(frozen=True)
class Policy:
    runtime_root: Path
    binary: Path
    bundle_manifest: object
    library_dirs: tuple[Path, ...]
    artifact_roots: tuple[Path, ...]
    profiles: object
    port: int = 11442
    startup_timeout: float = 180
    health_timeout: float = 2
    drain_timeout: float = 30
    term_timeout: float = 10
    kill_timeout: float = 5
    anchor: Path = Path("/")

    def __post_init__(self):
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "bundle_manifest", MappingProxyType({
            key: MappingProxyType(dict(value)) for key, value in self.bundle_manifest.items()}))
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise PolicyError("invalid loopback port")
        for name, maximum in (("startup_timeout", 180), ("health_timeout", 5),
                              ("drain_timeout", 60), ("term_timeout", 15), ("kill_timeout", 5)):
            if not 0 < getattr(self, name) <= maximum:
                raise PolicyError("invalid lifecycle timeout")

    @classmethod
    def load(cls, path):
        immutable_path(path)
        if path.stat().st_size > 65536:
            raise PolicyError("policy is too large")
        with path.open() as stream:
            data = json.load(stream)
        allowed = {"schema_version", "runtime_root", "binary", "bundle_manifest", "library_dirs",
                   "artifact_roots", "profiles", "port", "startup_timeout", "health_timeout",
                   "drain_timeout", "term_timeout", "kill_timeout"}
        if set(data) - allowed or data.pop("schema_version") != 1:
            raise PolicyError("invalid policy schema")
        root = Path(data.pop("runtime_root"))
        if not root.is_relative_to("/usr/lib/kiron/runtimes/prism"):
            raise PolicyError("runtime must use the production bundle root")
        binary = root / data.pop("binary")
        libraries = tuple(root / value for value in data.pop("library_dirs"))
        if not binary.is_relative_to(root) or any(not item.is_relative_to(root) for item in libraries):
            raise PolicyError("binary/library outside bundle")
        profiles = {key: Profile(id=key, **value) for key, value in data.pop("profiles").items()}
        manifest = MappingProxyType(data.pop("bundle_manifest"))
        result = cls(root, binary, manifest, libraries,
                     tuple(Path(value) for value in data.pop("artifact_roots")),
                     MappingProxyType(profiles), **data)
        result.verify_bundle()
        for artifact_root in result.artifact_roots:
            if not artifact_root.is_relative_to("/usr/lib/kiron/data/gguf-models"):
                raise PolicyError("artifact root outside the managed GGUF tree")
            immutable_path(artifact_root, directory=True)
        return result

    def verify_bundle(self):
        immutable_path(self.runtime_root, anchor=self.anchor, directory=True)
        actual = set()
        for path in self.runtime_root.rglob("*"):
            if path.is_symlink():
                info = path.lstat()
                relative = str(path.relative_to(self.runtime_root))
                if (info.st_uid != 0 or not path.resolve().is_relative_to(self.runtime_root)
                        or self.bundle_manifest.get(relative) != {"link": os.readlink(path)}):
                    raise PolicyError("unapproved bundle link")
                immutable_path(path.resolve(), anchor=self.anchor)
                actual.add(relative)
            elif path.is_dir():
                immutable_path(path, anchor=self.anchor, directory=True)
            else:
                relative = str(path.relative_to(self.runtime_root))
                record = self.bundle_manifest.get(relative, {})
                fd = checked_open(path, record.get("sha256"), anchor=self.anchor)
                os.close(fd)
                actual.add(relative)
        if actual != set(self.bundle_manifest) or not os.access(self.binary, os.X_OK):
            raise PolicyError("bundle inventory or executable mismatch")
        immutable_path(self.binary, anchor=self.anchor)
        for directory in self.library_dirs:
            immutable_path(directory, anchor=self.anchor, directory=True)

    @property
    def runtime_revision(self):
        """Identity of the verified bundle and its selected executable/search paths.

        Composition must load/verify this policy before using the identity as
        evidence. Relocation of the same immutable bundle does not change it.
        """
        value = {
            "manifest": {name: dict(record) for name, record in self.bundle_manifest.items()},
            "binary": str(self.binary.relative_to(self.runtime_root)),
            "library_dirs": [str(path.relative_to(self.runtime_root)) for path in self.library_dirs],
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                             allow_nan=False).encode("ascii")
        return "sha256:" + hashlib.sha256(b"kiron.prism.runtime-bundle.v1\0" + encoded).hexdigest()

    def open_artifact(self, reference, sha256, size_bytes):
        path = Path(reference)
        if not any(path.is_relative_to(root) for root in self.artifact_roots):
            raise PolicyError("artifact outside managed roots")
        if not isinstance(sha256, str) or len(sha256) != 64 or type(size_bytes) is not int or size_bytes <= 0:
            raise PolicyError("artifact requires digest and size")
        return checked_open(path, sha256, size_bytes, anchor=self.anchor)

    def environment(self):
        return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                "LD_LIBRARY_PATH": ":".join(map(str, self.library_dirs))}

    def resource_profiles(self):
        from .local_inference.identity import ResourceProfile
        return MappingProxyType({profile.id: ResourceProfile(
            id=profile.id, context_tokens=profile.context, batch_size=profile.batch,
            ubatch_size=profile.batch, parallel_slots=1, threads=profile.threads,
            gpu_layers=profile.gpu_layers, projector_on_gpu=False,
            gpu_memory_bytes=profile.gpu_memory_bytes, host_memory_bytes=profile.host_memory_bytes,
            memory_headroom_bytes=profile.memory_headroom_bytes,
        ) for profile in self.profiles.values()})

    def registration_profiles(self):
        from .local_model_registry.gguf import GGUFRegistrationPolicy
        return MappingProxyType({profile.id: GGUFRegistrationPolicy(profile.model_sha256,
                                 profile.architecture, profile.projector_sha256)
                                 for profile in self.profiles.values()})

    def verify_metadata(self, profile, model_fd, projector_fd):
        from kiron_common.local_model_registry.gguf import read_gguf_metadata
        def metadata(fd):
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(fd), "rb") as stream:
                    return read_gguf_metadata(stream)
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
        model = metadata(model_fd)
        if model.get("general.architecture") != profile.architecture:
            raise PolicyError("GGUF architecture differs from the measured profile")
        if projector_fd is not None:
            projector = metadata(projector_fd)
            dimension = model.get(f"{profile.architecture}.embedding_length")
            if (projector.get("general.architecture") != "clip" or type(dimension) is not int
                    or dimension <= 0 or projector.get("clip.vision.projection_dim") != dimension):
                raise PolicyError("GGUF/projector dimensions do not match")

    def command(self, profile, alias, model_fd, projector_fd):
        argv = [str(self.binary), "--model", f"/proc/self/fd/{model_fd}", "--alias", alias,
                "--api-key", alias,
                "--host", "127.0.0.1", "--port", str(self.port), "--parallel", "1",
                "--ctx-size", str(profile.context), "--batch-size", str(profile.batch),
                "--ubatch-size", str(profile.batch), "--threads", str(profile.threads),
                "--threads-batch", str(profile.threads), "--n-gpu-layers", str(profile.gpu_layers),
                "--jinja", "--fit", "off", "--cache-ram", "0", "--no-warmup",
                "--no-context-shift", "--reasoning", "off", "--reasoning-format", "deepseek",
                "--no-agent", "--no-webui", "--no-ui-mcp-proxy"]
        if profile.gpu_layers == 0:
            argv += ["--device", "none"]
        argv += (["--mmproj", f"/proc/self/fd/{projector_fd}", "--no-mmproj-offload"]
                 if projector_fd is not None else ["--no-mmproj"])
        return argv
