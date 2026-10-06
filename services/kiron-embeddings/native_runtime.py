"""Read-only load identity proof for the explicit local inference transport."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat
import os
from uuid import uuid4

from kiron_common.local_inference import ArtifactIdentity, build_resolver_snapshot
from kiron_common.model_catalog import BackendType, LoaderType
from kiron_common.model_state import default_huggingface_hub_cache

SERVICE_EPOCH = "embedding:" + uuid4().hex
LOADER_AUXILIARY = frozenset({"config.json", "tokenizer.json", "tokenizer_config.json"})


def has_closed_loader_provenance(model):
    """Initial proven loader family, not inferred from a model name/repository."""
    return (model.loader_type is LoaderType.TRANSFORMERS_LAST_TOKEN
        and model.artifact.trust_remote_code is False
        and model.loader_parameters.tokenizer_use_fast is True
        and {item.path for item in model.artifact.weights} == {"model.safetensors"}
        and {item.path for item in model.artifact.auxiliary} == LOADER_AUXILIARY)


def _snapshot_files(snapshot):
    return frozenset(str(path.relative_to(snapshot)) for path in snapshot.rglob('*')
                     if path.is_symlink() or not path.is_dir())


def generation(snapshot):
    return {"boot_id": SERVICE_EPOCH, "process_id": f"{os.getpid()}:{snapshot['model_epoch']}"}


def state_payload(catalog, snapshot, revision):
    resolver = build_resolver_snapshot(catalog, ())
    proofs = snapshot.get("verified_artifacts", {})
    return {"version": 1, "generation": generation(snapshot), "service_revision": revision,
        "catalog_digest": catalog.catalog_digest,
        "accepting": snapshot.get("worker_accepting") is True and snapshot.get("worker_thread_alive") is True,
        "busy": snapshot.get("current_job") is not None or snapshot.get("queue_depth", 0) > 0,
        "device": snapshot.get("device"),
        "deployments": [{"deployment_id": item.id, "reference": item.reference,
            "artifact_fingerprint": item.artifact_identity.fingerprint,
            "configuration_fingerprint": item.configuration_fingerprint,
            "loaded": item.reference in snapshot.get("loaded_models", ())
                and proofs.get(item.reference) == item.artifact_identity.fingerprint}
            for item in resolver.deployments.values() if item.provider is BackendType.KIRON_EMBEDDINGS]}

@dataclass(frozen=True)
class _ArtifactFileProof:
    path: Path
    resolved: Path
    signature: tuple
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactProof:
    fingerprint: str
    files: tuple[_ArtifactFileProof, ...]
    snapshot: Path
    allowed_files: frozenset

    def recheck(self):
        """Rehash actual bytes; metadata is only an additional race rejection.

        Each check reads the complete declared payload in bounded blocks. This
        is a before/after construction check, not filesystem immutability: the
        operator must keep the loader inputs quiescent (the isolated native
        harness uses a read-only mount). A writer able to change and restore
        files between observations cannot be excluded by ordinary file reads.
        """
        if _snapshot_files(self.snapshot) != self.allowed_files:
            raise ValueError("embedding loader file inventory changed")
        for record in self.files:
            _verify_file(record)
        if _snapshot_files(self.snapshot) != self.allowed_files:
            raise ValueError("embedding loader file inventory changed")


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _verify_file(record):
    try:
        if record.path.resolve(strict=True) != record.resolved:
            raise ValueError("embedding artifact target changed")
        descriptor = os.open(record.resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or _signature(before) != record.signature:
                raise ValueError("embedding artifact changed while it was loaded")
            digest = hashlib.sha256()
            remaining = record.size_bytes
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("embedding artifact ended before its pinned size")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1):
                raise ValueError("embedding artifact exceeds its pinned size")
            after = os.fstat(stream.fileno())
            target = record.resolved.stat(follow_symlinks=False)
            if (_signature(after) != record.signature or not stat.S_ISREG(target.st_mode)
                    or _signature(target) != record.signature
                    or record.path.resolve(strict=True) != record.resolved):
                raise ValueError("embedding artifact changed while hashing")
            if digest.hexdigest() != record.sha256:
                raise ValueError("embedding artifact does not match its catalog digest")
    except OSError as exc:
        raise ValueError("embedding artifact could not be verified") from exc


def verify_local_artifact(artifact, *, cache=None):
    """Hash pinned local weights and loader files; no Hub, copies or downloads."""
    if not artifact.repository or not artifact.revision or not artifact.weights:
        raise ValueError("embedding artifact has no local pinned weight identity")
    root = Path(cache or default_huggingface_hub_cache()).resolve(strict=True)
    repository = root / ("models--" + artifact.repository.replace("/", "--"))
    snapshot = repository / "snapshots" / artifact.revision
    identity = ArtifactIdentity(artifact.type, artifact.format, repository=artifact.repository,
        revision=artifact.revision, manifest_digest=artifact.manifest_digest,
        files=(*artifact.weights, *artifact.auxiliary))
    records = []
    allowed = frozenset(item.path for item in identity.files)
    # Model cards are not consumed by this AutoModel/fast-tokenizer loader.
    if (snapshot / 'README.md').is_file():
        allowed |= {'README.md'}
    if _snapshot_files(snapshot) != allowed:
        raise ValueError("embedding snapshot contains undeclared loader files")
    for item in identity.files:
        path = snapshot / item.path
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repository):
            raise ValueError("embedding artifact is outside its local repository")
        info = resolved.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("embedding artifact is not a regular file")
        if item.size_bytes is not None and info.st_size != item.size_bytes:
            raise ValueError("embedding artifact does not match its catalog digest")
        records.append(_ArtifactFileProof(path, resolved, _signature(info), item.sha256, info.st_size))
    proof = ArtifactProof(identity.fingerprint, tuple(records), snapshot, allowed)
    proof.recheck()
    return proof
