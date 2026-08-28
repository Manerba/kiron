"""Runner interfaces for kitt-worker job execution."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import stat
from typing import Any
from urllib.parse import urlsplit
import zipfile

import artifact_staging
import config as worker_config
import job_stubs
import publish_contract
import queue_store
import sft_command


MIN_MODEL_ARTIFACT_BYTES = 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MIN_SAFETENSORS_TENSOR_BYTES = 4096
MIN_SAFETENSORS_NONZERO_BYTES = 64
MIN_SAFETENSORS_UNIQUE_BYTE_VALUES = 4
RUNNER_MANIFEST_SCHEMA_VERSION = "kiron_sft_runner_manifest_v1"
RUNNER_PROVENANCE_KIND = "kiron_sft_training_run"
RUNNER_PROVENANCE_VERSION = 1
RUNNER_ENTRYPOINT_KIND = "kiron_sft_trainer"
RUNNER_ENTRYPOINT_VERSION = "adr-0008.sft.v2"
_MODEL_ZIP_REQUIRED_FILES = {
    "adapter_config.json",
}
_MODEL_ZIP_WEIGHT_FILES = {
    "adapter_model.safetensors",
    "model.safetensors",
}
_MERGED_WEIGHTS_REQUIRED_FILES = {
    "config.json",
}
_MERGED_WEIGHTS_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
}
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(frozen=True, slots=True)
class RunnerArtifact:
    role: str
    path: Path


@dataclass(frozen=True, slots=True)
class RunnerOutcome:
    state: str
    failure_code: str | None = None
    failure_class: str | None = None
    artifacts: tuple[RunnerArtifact, ...] = ()
    runner_nonce_sha256: str | None = None
    publish_result: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _PublishContext:
    spec: dict[str, Any]
    workspace: Path
    target_ref: str
    publish_job_uid: str
    publish_spec_hash_sha256: str
    source_artifact_fingerprint_sha256: str
    provenance_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class _OllamaCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    canceled: bool = False


class PublishRunnerError(RuntimeError):
    def __init__(self, reason_code: str, failure_class: str = "publish") -> None:
        self.reason_code = reason_code
        self.failure_class = failure_class
        super().__init__(reason_code)


class FakeRunner:
    """Cooperative fake runner for executor tests; it performs no I/O."""

    def __init__(
        self,
        *,
        result: str = "succeeded",
        delay_seconds: float = 0.0,
        failure_code: str = "runner_failed",
        failure_class: str = "runner",
    ) -> None:
        if result not in {"succeeded", "failed"}:
            raise ValueError("fake runner result is invalid")
        self.result = result
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.failure_code = failure_code
        self.failure_class = failure_class
        self.started = False
        self.cancel_requested = False

    async def run(self, job: Any, cancel_event: asyncio.Event) -> RunnerOutcome:
        self.started = True
        deadline = asyncio.get_running_loop().time() + self.delay_seconds
        while True:
            if cancel_event.is_set():
                self.cancel_requested = True
                return RunnerOutcome("canceled")
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))
        if self.result == "failed":
            return RunnerOutcome(
                "failed",
                failure_code=self.failure_code,
                failure_class=self.failure_class,
            )
        return RunnerOutcome("succeeded")


class SftSubprocessRunner:
    """Executes a configured local SFT command and collects staged outputs."""

    _ARTIFACT_FILES = {
        "adapter": "adapter.zip",
        "gguf": "model.gguf",
        "manifest": "manifest.json",
        "merged_weights": "merged_weights.zip",
        "metrics_jsonl": "metrics.jsonl",
        "ollama_modelfile": "Modelfile",
        "run_lock": "run_lock.json",
        "training_log": "training_log.jsonl",
    }

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg

    async def run(self, job: Any, cancel_event: asyncio.Event) -> RunnerOutcome:
        proc: asyncio.subprocess.Process | None = None
        try:
            spec = job_stubs.kitt_job_spec(job)
            command = _command_argv(getattr(self.cfg, "sft_command", ""))
            job_dir, output_dir, spec_path = _prepare_job_workspace(self.cfg, job)
            catalog_path = _sft_resource_catalog_path(self.cfg)
        except Exception:
            return RunnerOutcome(
                "failed",
                failure_code="runner_config_invalid",
                failure_class="runner",
            )

        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(spec_path, 0o640)
        runner_nonce = secrets.token_urlsafe(32)
        runner_nonce_sha256 = _sha256_text(runner_nonce)

        env = _runner_env(
            job_uid=str(job.job_uid),
            run_uid=str(job.run_uid),
            spec_path=spec_path,
            output_dir=output_dir,
            runner_nonce=runner_nonce,
            catalog_path=catalog_path,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=job_dir,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            return RunnerOutcome(
                "failed",
                failure_code="runner_start_failed",
                failure_class="runner",
            )

        try:
            while True:
                if cancel_event.is_set():
                    await _terminate_process(proc, job_uid=str(job.job_uid))
                    return RunnerOutcome("canceled")
                try:
                    await asyncio.wait_for(proc.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            _kill_process_now(proc, job_uid=str(job.job_uid))
            with suppress(Exception):
                await proc.wait()
            raise

        if proc.returncode != 0:
            return RunnerOutcome(
                "failed",
                failure_code="runner_exit_failed",
                failure_class="runner",
            )

        try:
            artifacts, artifact_error = await asyncio.to_thread(
                _collect_artifacts,
                output_dir,
                spec,
                runner_nonce_sha256=runner_nonce_sha256,
            )
        except Exception:
            return RunnerOutcome(
                "failed",
                failure_code="runner_artifact_invalid",
                failure_class="artifact",
            )
        if artifact_error is not None:
            return RunnerOutcome(
                "failed",
                failure_code=artifact_error,
                failure_class="artifact",
            )
        if _required_artifact_missing(spec, artifacts):
            return RunnerOutcome(
                "failed",
                failure_code="runner_artifact_missing",
                failure_class="artifact",
            )
        return RunnerOutcome(
            "succeeded",
            artifacts=tuple(artifacts),
            runner_nonce_sha256=runner_nonce_sha256,
        )


class OllamaPublishRunner:
    """Publishes a verified GGUF/Modelfile bundle into a local Ollama tag."""

    def __init__(self, cfg: object, store: queue_store.QueueStore) -> None:
        self.cfg = cfg
        self.store = store

    async def run(self, job: Any, cancel_event: asyncio.Event) -> RunnerOutcome:
        try:
            context = await asyncio.to_thread(self._prepare_context, job)
        except PublishRunnerError as exc:
            return RunnerOutcome(
                "failed",
                failure_code=exc.reason_code,
                failure_class=exc.failure_class,
            )
        except Exception:
            return RunnerOutcome(
                "failed",
                failure_code="publish_verification_failed",
                failure_class="publish",
            )
        if cancel_event.is_set():
            return RunnerOutcome("canceled")

        show_before = await self._ollama_show(context, cancel_event)
        if show_before.canceled:
            return RunnerOutcome("canceled")
        if show_before.returncode == 0:
            current_digest = _ollama_digest_from_show(show_before.stdout, fallback="")
            existing = self.store.find_publish_result_by_target(
                target_ref=context.target_ref,
                publish_spec_hash_sha256=context.publish_spec_hash_sha256,
                source_artifact_fingerprint_sha256=(
                    context.source_artifact_fingerprint_sha256
                ),
            )
            if (
                existing is None
                or existing.provenance_fingerprint_sha256
                != context.provenance_fingerprint_sha256
                or not existing.ollama_digest
                or current_digest != existing.ollama_digest
            ):
                return RunnerOutcome(
                    "failed",
                    failure_code="duplicate_tag_conflict",
                    failure_class="publish",
                )
            return RunnerOutcome(
                "succeeded",
                publish_result={
                    **existing.as_public_dict(),
                    "idempotent": True,
                },
            )

        create_result = await self._ollama_create(context, cancel_event)
        if create_result.canceled:
            return RunnerOutcome(
                "failed",
                failure_code="publish_verification_unknown",
                failure_class="publish",
            )
        if create_result.returncode != 0:
            return RunnerOutcome(
                "failed",
                failure_code=_classify_ollama_failure(create_result.stderr),
                failure_class="publish",
            )

        show_after = await self._ollama_show(context, cancel_event)
        if show_after.canceled:
            return RunnerOutcome(
                "failed",
                failure_code="publish_verification_unknown",
                failure_class="publish",
            )
        if show_after.returncode != 0:
            return RunnerOutcome(
                "failed",
                failure_code="publish_verification_failed",
                failure_class="publish",
            )
        return RunnerOutcome(
            "succeeded",
            publish_result={
                "schema_version": publish_contract.PUBLISH_RESULT_SCHEMA_VERSION,
                "publish_job_uid": context.publish_job_uid,
                "target_ref": context.target_ref,
                "publish_spec_hash_sha256": context.publish_spec_hash_sha256,
                "source_artifact_fingerprint_sha256": (
                    context.source_artifact_fingerprint_sha256
                ),
                "provenance_fingerprint_sha256": context.provenance_fingerprint_sha256,
                "ollama_digest": _ollama_digest_from_show(
                    show_after.stdout,
                    fallback=context.provenance_fingerprint_sha256,
                ),
                "idempotent": False,
            },
        )

    def _prepare_context(self, job: Any) -> _PublishContext:
        spec = job_stubs.publish_job_spec(job)
        _verify_publish_signature_for_execution(self.cfg, spec)
        artifacts = _resolve_publish_artifacts(self.store, spec)
        root = artifact_staging.validate_staging_root(self.cfg)
        paths = {
            role: _artifact_object_path(root, artifact)
            for role, artifact in artifacts.items()
        }
        _verify_publish_artifact_files(paths, artifacts, spec)
        run_lock = _read_publish_run_lock(paths["run_lock"])
        _verify_run_lock(run_lock, spec, artifacts)
        modelfile_text = _verified_modelfile_text(paths["ollama_modelfile"])

        workspace = _prepare_publish_workspace(self.cfg, job)
        _copy_publish_file(paths["gguf"], workspace / "model.gguf")
        _write_publish_modelfile(modelfile_text, workspace / "Modelfile")

        source_fingerprint = _source_artifact_fingerprint(spec)
        publish_spec_hash = str(spec["publish_spec_hash_sha256"])
        provenance_fingerprint = _publish_provenance_fingerprint(
            spec=spec,
            source_artifact_fingerprint_sha256=source_fingerprint,
        )
        return _PublishContext(
            spec=spec,
            workspace=workspace,
            target_ref=publish_contract.publish_target_ref(spec),
            publish_job_uid=str(spec["publish_job_uid"]),
            publish_spec_hash_sha256=publish_spec_hash,
            source_artifact_fingerprint_sha256=source_fingerprint,
            provenance_fingerprint_sha256=provenance_fingerprint,
        )

    async def _ollama_show(
        self,
        context: _PublishContext,
        cancel_event: asyncio.Event,
    ) -> _OllamaCommandResult:
        return await self._run_ollama(
            ["show", context.target_ref],
            context,
            cancel_event,
        )

    async def _ollama_create(
        self,
        context: _PublishContext,
        cancel_event: asyncio.Event,
    ) -> _OllamaCommandResult:
        return await self._run_ollama(
            ["create", context.target_ref, "-f", "Modelfile"],
            context,
            cancel_event,
        )

    async def _run_ollama(
        self,
        args: list[str],
        context: _PublishContext,
        cancel_event: asyncio.Event,
    ) -> _OllamaCommandResult:
        try:
            command = [_ollama_binary(self.cfg), *args]
        except ValueError:
            return _OllamaCommandResult(127, b"", b"ollama unavailable")
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=context.workspace,
                env=_ollama_env(self.cfg),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            return _OllamaCommandResult(127, b"", b"ollama unavailable")
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                return _OllamaCommandResult(1, b"", b"no space left")
            return _OllamaCommandResult(1, b"", b"ollama start failed")

        communicate = asyncio.create_task(proc.communicate())
        try:
            while not communicate.done():
                if cancel_event.is_set():
                    await _terminate_process(proc, job_uid=context.publish_job_uid)
                    return _OllamaCommandResult(1, b"", b"", canceled=True)
                await asyncio.sleep(0.05)
            stdout, stderr = communicate.result()
            return _OllamaCommandResult(
                int(proc.returncode or 0),
                stdout[:1024 * 1024],
                stderr[:1024 * 1024],
            )
        except asyncio.CancelledError:
            _kill_process_now(proc, job_uid=context.publish_job_uid)
            with suppress(Exception):
                await proc.wait()
            raise


def build_runner(
    kind: str = "sft_subprocess",
    cfg: object | None = None,
    store: queue_store.QueueStore | None = None,
):
    if kind == "sft_subprocess" and cfg is not None:
        return SftSubprocessRunner(cfg)
    if kind == "ollama_publish" and cfg is not None and store is not None:
        return OllamaPublishRunner(cfg, store)
    if kind == "fake":
        return FakeRunner()
    raise ValueError("runner kind is not enabled")


def _verify_publish_signature_for_execution(cfg: object, spec: dict[str, Any]) -> None:
    if not hasattr(cfg, "publish_verify_keys_file"):
        return
    try:
        reason = worker_config.publish_verify_keyring_availability_error(cfg)
        if reason is not None:
            raise PublishRunnerError("publish_spec_signature_invalid", "publish")
        keyring = worker_config.load_publish_verify_keyring(cfg)
        publish_contract.verify_publish_spec_signature(spec, keyring=keyring)
    except PublishRunnerError:
        raise
    except worker_config.RuntimeConfigError as exc:
        raise PublishRunnerError("publish_spec_signature_invalid", "publish") from exc
    except publish_contract.PublishSpecError as exc:
        raise PublishRunnerError("publish_spec_signature_invalid", "publish") from exc


def _resolve_publish_artifacts(
    store: queue_store.QueueStore,
    spec: dict[str, Any],
) -> dict[str, queue_store.ArtifactRecord]:
    by_role = publish_contract.publish_artifacts_by_role(spec)
    result: dict[str, queue_store.ArtifactRecord] = {}
    source_job_uid = publish_contract.source_worker_job_uid(spec)
    _verify_publish_source_job(store, source_job_uid, spec)
    for role in publish_contract.PUBLISH_REQUIRED_ARTIFACT_ROLES:
        declared = by_role.get(role)
        if declared is None:
            raise PublishRunnerError("artifact_missing", "artifact")
        try:
            artifact = store.get_artifact_by_ref(str(declared["artifact_ref"]))
        except job_stubs.JobNotFoundError as exc:
            raise PublishRunnerError("artifact_missing", "artifact") from exc
        if artifact.status != "staged":
            raise PublishRunnerError("artifact_missing", "artifact")
        if artifact.job_uid != source_job_uid or artifact.role != role:
            raise PublishRunnerError("artifact_role_mismatch", "artifact")
        if artifact.sha256 != declared["sha256"]:
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        if artifact.size_bytes != declared["size_bytes"]:
            raise PublishRunnerError("artifact_size_mismatch", "artifact")
        for key in ("training_run_uid", "model_version_uid"):
            if key in declared and declared[key] != spec[key]:
                raise PublishRunnerError("lineage_mismatch", "artifact")
        result[role] = artifact
    return result


def _verify_publish_source_job(
    store: queue_store.QueueStore,
    source_job_uid: str,
    spec: dict[str, Any],
) -> None:
    try:
        source_job = store.get_job(source_job_uid)
        source_spec = job_stubs.kitt_job_spec(source_job)
    except (job_stubs.JobNotFoundError, job_stubs.JobValidationError) as exc:
        raise PublishRunnerError("lineage_mismatch", "artifact") from exc
    if source_job.job_type != "sft" or source_job.state != "succeeded":
        raise PublishRunnerError("lineage_mismatch", "artifact")
    if source_spec.get("run_uid") != spec.get("training_run_uid"):
        raise PublishRunnerError("lineage_mismatch", "artifact")
    lineage = spec.get("lineage")
    if not isinstance(lineage, dict):
        raise PublishRunnerError("lineage_mismatch", "artifact")
    source_parent = _base_model_ref(source_spec)
    expected_parent = lineage.get("parent_model_ref") or lineage.get("base_model_ref")
    if expected_parent is not None and source_parent != expected_parent:
        raise PublishRunnerError("parent_mismatch", "artifact")


def _artifact_object_path(
    root: artifact_staging.ValidatedStagingRoot,
    artifact: queue_store.ArtifactRecord,
) -> Path:
    path = (root.objects_dir / artifact.artifact_uid).resolve(strict=False)
    root_path = root.objects_dir.resolve(strict=False)
    if path == root_path or not path.is_relative_to(root_path):
        raise PublishRunnerError("artifact_missing", "artifact")
    return path


def _verify_publish_artifact_files(
    paths: dict[str, Path],
    artifacts: dict[str, queue_store.ArtifactRecord],
    spec: dict[str, Any],
) -> None:
    for role, path in paths.items():
        artifact = artifacts[role]
        if not _safe_publish_file(path):
            raise PublishRunnerError("artifact_missing", "artifact")
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise PublishRunnerError("artifact_missing", "artifact") from exc
        if size_bytes != artifact.size_bytes:
            raise PublishRunnerError("artifact_size_mismatch", "artifact")
        if _sha256_file(path) != artifact.sha256:
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
    gguf_path = paths["gguf"]
    try:
        with gguf_path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise PublishRunnerError("artifact_hash_mismatch", "artifact")
    except OSError as exc:
        raise PublishRunnerError("artifact_missing", "artifact") from exc
    _assert_publish_artifact_lineage(spec)


def _safe_publish_file(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(stat_result.st_mode) and not stat.S_ISLNK(stat_result.st_mode)


def _assert_publish_artifact_lineage(spec: dict[str, Any]) -> None:
    source_job_uid = publish_contract.source_worker_job_uid(spec)
    for declared in publish_contract.publish_artifacts_by_role(spec).values():
        if declared.get("source_worker_job_uid", source_job_uid) != source_job_uid:
            raise PublishRunnerError("lineage_mismatch", "artifact")
        if declared.get("training_run_uid", spec["training_run_uid"]) != spec["training_run_uid"]:
            raise PublishRunnerError("lineage_mismatch", "artifact")
        if declared.get("model_version_uid", spec["model_version_uid"]) != spec["model_version_uid"]:
            raise PublishRunnerError("lineage_mismatch", "artifact")


def _read_publish_run_lock(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishRunnerError("lineage_mismatch", "artifact") from exc
    if not isinstance(data, dict):
        raise PublishRunnerError("lineage_mismatch", "artifact")
    return data


def _verify_run_lock(
    data: dict[str, Any],
    spec: dict[str, Any],
    artifacts: dict[str, queue_store.ArtifactRecord],
) -> None:
    if data.get("schema_version") != "kitt_run_lock_v1":
        raise PublishRunnerError("lineage_mismatch", "artifact")
    if data.get("complete") is not True:
        raise PublishRunnerError("lineage_mismatch", "artifact")
    if data.get("release_eligible") is not True:
        raise PublishRunnerError("lineage_mismatch", "artifact")
    for key in ("blockers", "missing_fields"):
        value = data.get(key, [])
        if not isinstance(value, list) or value:
            raise PublishRunnerError("lineage_mismatch", "artifact")
    _match_required_ref(data, "job_uid", publish_contract.source_worker_job_uid(spec))
    if "source_worker_job_uid" in data:
        _match_required_ref(
            data,
            "source_worker_job_uid",
            publish_contract.source_worker_job_uid(spec),
        )
    _match_required_ref(data, "training_run_uid", str(spec["training_run_uid"]))
    _match_required_ref(data, "model_version_uid", str(spec["model_version_uid"]))

    lineage = spec["lineage"]
    parent_ref = lineage.get("parent_model_ref")
    base_ref = lineage.get("base_model_ref")
    if parent_ref is not None:
        _match_required_ref(data, "parent_model_ref", str(parent_ref), parent=True)
    elif base_ref is not None:
        _match_required_ref(data, "base_model_ref", str(base_ref), parent=True)
    else:
        raise PublishRunnerError("parent_mismatch", "artifact")
    _verify_run_lock_artifacts(data.get("artifacts"), artifacts)


def _match_required_ref(
    data: dict[str, Any],
    key: str,
    expected: str,
    *,
    parent: bool = False,
) -> None:
    if key not in data or data[key] != expected:
        if parent:
            raise PublishRunnerError("parent_mismatch", "artifact")
        raise PublishRunnerError("lineage_mismatch", "artifact")


def _verify_run_lock_artifacts(
    value: object,
    artifacts: dict[str, queue_store.ArtifactRecord],
) -> None:
    if value is None:
        raise PublishRunnerError("lineage_mismatch", "artifact")
    entries: dict[str, Any]
    if isinstance(value, dict):
        entries = value
    elif isinstance(value, list):
        entries = {
            str(item.get("role")): item
            for item in value
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        }
    else:
        raise PublishRunnerError("lineage_mismatch", "artifact")
    for role, artifact in artifacts.items():
        if role == "run_lock":
            continue
        entry = entries.get(role)
        if not isinstance(entry, dict):
            raise PublishRunnerError("lineage_mismatch", "artifact")
        if entry.get("sha256") != artifact.sha256:
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        if entry.get("size_bytes") != artifact.size_bytes:
            raise PublishRunnerError("artifact_size_mismatch", "artifact")


def _verified_modelfile_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PublishRunnerError("artifact_missing", "artifact") from exc
    if not text.strip() or "\x00" in text:
        raise PublishRunnerError("artifact_hash_mismatch", "artifact")
    safe_lines: list[str] = []
    from_seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            safe_lines.append(raw_line)
            continue
        upper = line.upper()
        if upper.startswith("FROM "):
            if from_seen or line not in {"FROM ./model.gguf", "FROM model.gguf"}:
                raise PublishRunnerError("artifact_hash_mismatch", "artifact")
            from_seen = True
            safe_lines.append("FROM ./model.gguf")
            continue
        if upper.startswith(("ADAPTER ", "INCLUDE ", "LICENSE ", "SYSTEM ", "TEMPLATE ")):
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        if any(part in line for part in ("\\", "://", "\t", "\r", "..")):
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        if line.startswith("/") or "/opt/" in line or "/usr/" in line or "/var/" in line:
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        lowered = line.lower()
        if any(part in lowered for part in ("authorization", "bearer", "token", "secret", "password")):
            raise PublishRunnerError("artifact_hash_mismatch", "artifact")
        safe_lines.append(raw_line)
    if not from_seen:
        raise PublishRunnerError("artifact_hash_mismatch", "artifact")
    return "\n".join(safe_lines).strip() + "\n"


def _prepare_publish_workspace(cfg: object, job: Any) -> Path:
    data_dir = Path(getattr(cfg, "data_dir")).resolve(strict=False)
    work_root = (data_dir / "work").resolve(strict=False)
    if work_root == data_dir or not work_root.is_relative_to(data_dir):
        raise PublishRunnerError("publish_verification_failed")
    workspace = (work_root / str(job.job_uid) / "publish").resolve(strict=False)
    if workspace == work_root or not workspace.is_relative_to(work_root):
        raise PublishRunnerError("publish_verification_failed")
    try:
        workspace.mkdir(mode=0o750, parents=True, exist_ok=True)
        os.chmod(workspace, 0o750)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise PublishRunnerError("disk_full") from exc
        raise PublishRunnerError("publish_verification_failed") from exc
    return workspace


def _copy_publish_file(source: Path, target: Path) -> None:
    temp = target.with_name(target.name + ".tmp")
    try:
        with suppress(OSError):
            temp.unlink()
        try:
            os.link(source, temp)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                raise
            shutil.copyfile(source, temp)
            os.chmod(temp, 0o640)
        os.replace(temp, target)
    except OSError as exc:
        with suppress(OSError):
            temp.unlink()
        if exc.errno == errno.ENOSPC:
            raise PublishRunnerError("disk_full") from exc
        raise PublishRunnerError("publish_verification_failed") from exc


def _write_publish_modelfile(text: str, target: Path) -> None:
    temp = target.with_name(target.name + ".tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.chmod(temp, 0o640)
        os.replace(temp, target)
    except OSError as exc:
        with suppress(OSError):
            temp.unlink()
        if exc.errno == errno.ENOSPC:
            raise PublishRunnerError("disk_full") from exc
        raise PublishRunnerError("publish_verification_failed") from exc


def _source_artifact_fingerprint(spec: dict[str, Any]) -> str:
    artifacts = publish_contract.publish_artifacts_by_role(spec)
    payload = [
        {
            "role": role,
            "artifact_ref": artifacts[role]["artifact_ref"],
            "sha256": artifacts[role]["sha256"],
            "size_bytes": artifacts[role]["size_bytes"],
        }
        for role in sorted(publish_contract.PUBLISH_REQUIRED_ARTIFACT_ROLES)
    ]
    return _json_hash(payload)


def _publish_provenance_fingerprint(
    *,
    spec: dict[str, Any],
    source_artifact_fingerprint_sha256: str,
) -> str:
    envelope = spec["signature_envelope"]
    payload = {
        "schema_version": publish_contract.PUBLISH_RESULT_SCHEMA_VERSION,
        "publish_job_uid": spec["publish_job_uid"],
        "publish_intent_uid": spec["publish_intent_uid"],
        "model_version_uid": spec["model_version_uid"],
        "training_run_uid": spec["training_run_uid"],
        "source_worker_job_uid": publish_contract.source_worker_job_uid(spec),
        "target_ref": publish_contract.publish_target_ref(spec),
        "publish_spec_hash_sha256": spec["publish_spec_hash_sha256"],
        "signed_payload_hash_sha256": envelope["signed_payload_hash_sha256"],
        "source_artifact_fingerprint_sha256": source_artifact_fingerprint_sha256,
    }
    return _json_hash(payload)


def _ollama_binary(cfg: object) -> str:
    value = getattr(cfg, "ollama_bin", None) or os.environ.get(
        "KITT_WORKER_OLLAMA_BIN",
        "ollama",
    )
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("ollama binary is invalid")
    return value


def _ollama_env(cfg: object | None = None) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
    home = _ollama_home(cfg)
    if home is not None:
        env["HOME"] = str(home)
    host = os.environ.get("OLLAMA_HOST")
    if host and _safe_ollama_host(host):
        env["OLLAMA_HOST"] = host
    return env


def _ollama_home(cfg: object | None) -> Path | None:
    if cfg is None:
        return None
    raw = getattr(cfg, "data_dir", None)
    if raw is None:
        return None
    try:
        path = Path(raw).resolve(strict=False)
        path.mkdir(mode=0o750, parents=True, exist_ok=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return path


def _safe_ollama_host(value: str) -> bool:
    if len(value) > 128 or "\x00" in value:
        return False
    lowered = value.lower()
    if any(part in lowered for part in ("authorization", "token", "secret", "password")):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    if parsed.path not in {"", "/"}:
        return False
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _classify_ollama_failure(stderr: bytes) -> str:
    lowered = stderr[:4096].decode("utf-8", errors="ignore").lower()
    if "ollama unavailable" in lowered or "start failed" in lowered:
        return "ollama_unavailable"
    if "no space" in lowered or "enospc" in lowered or "disk full" in lowered:
        return "disk_full"
    return "ollama_create_failed"


def _ollama_digest_from_show(stdout: bytes, *, fallback: str) -> str:
    try:
        data = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(data, dict):
        return fallback
    candidates = [
        data.get("digest"),
        data.get("model_digest"),
        data.get("modified_at"),
    ]
    details = data.get("details")
    if isinstance(details, dict):
        candidates.append(details.get("digest"))
    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        candidates.append(model_info.get("digest"))
    for value in candidates:
        if isinstance(value, str) and _safe_ollama_digest(value):
            return value[:128]
    return fallback


def _safe_ollama_digest(value: str) -> bool:
    return (
        1 <= len(value) <= 128
        and all(char.isalnum() or char in "_.:-" for char in value)
        and not any(part in value.lower() for part in ("authorization", "token", "secret", "password"))
    )


def _command_argv(command: str) -> list[str]:
    argv = sft_command.parse_trusted_sft_command(command)
    if argv is None:
        raise ValueError("sft command is empty")
    return argv


def _runner_env(
    *,
    job_uid: str,
    run_uid: str,
    spec_path: Path,
    output_dir: Path,
    runner_nonce: str,
    catalog_path: Path,
) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "KITT_JOB_UID": job_uid,
        "KITT_RUN_UID": run_uid,
        "KITT_JOB_SPEC_PATH": str(spec_path),
        "KITT_OUTPUT_DIR": str(output_dir),
        "KITT_RUNNER_NONCE": runner_nonce,
        "KITT_SFT_RESOURCE_CATALOG": str(catalog_path),
    }
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def _sft_resource_catalog_path(cfg: object) -> Path:
    raw = getattr(cfg, "sft_resource_catalog", None)
    if raw is not None:
        return Path(raw).resolve(strict=False)
    data_dir = Path(getattr(cfg, "data_dir")).resolve(strict=False)
    return data_dir / "sft_resources.json"


def _prepare_job_workspace(cfg: object, job: Any) -> tuple[Path, Path, Path]:
    data_dir = Path(getattr(cfg, "data_dir")).resolve()
    work_root = (data_dir / "work").resolve(strict=False)
    if not work_root.is_relative_to(data_dir):
        raise ValueError("work root escapes data dir")
    work_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    os.chmod(work_root, 0o750)
    job_dir = (work_root / job.job_uid).resolve(strict=False)
    if not job_dir.is_relative_to(work_root):
        raise ValueError("job workspace escapes work root")
    output_dir = job_dir / "output"
    output_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
    os.chmod(job_dir, 0o750)
    os.chmod(output_dir, 0o750)
    return job_dir, output_dir, job_dir / "job_spec.json"


async def _terminate_process(
    proc: asyncio.subprocess.Process,
    *,
    job_uid: str,
) -> None:
    if proc.returncode is not None:
        return
    descendant_pids = _runner_process_pids(proc.pid, job_uid=job_uid)
    _signal_process_tree(proc, signal.SIGTERM, descendant_pids=descendant_pids)
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        descendant_pids = _unique_pids(
            [*descendant_pids, *_runner_process_pids(proc.pid, job_uid=job_uid)]
        )
        _signal_process_tree(proc, signal.SIGKILL, descendant_pids=descendant_pids)
        await proc.wait()


def _kill_process_now(
    proc: asyncio.subprocess.Process | None,
    *,
    job_uid: str,
) -> None:
    if proc is None or proc.returncode is not None:
        return
    _signal_process_tree(
        proc,
        signal.SIGKILL,
        descendant_pids=_runner_process_pids(proc.pid, job_uid=job_uid),
    )


def _signal_process_tree(
    proc: asyncio.subprocess.Process,
    sig: signal.Signals,
    *,
    descendant_pids: list[int],
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)
    for pid in reversed(descendant_pids):
        with suppress(ProcessLookupError):
            os.kill(pid, sig)
    if sig == signal.SIGTERM:
        with suppress(ProcessLookupError):
            proc.terminate()
    else:
        with suppress(ProcessLookupError):
            proc.kill()


def _runner_process_pids(root_pid: int, *, job_uid: str) -> list[int]:
    return _unique_pids(
        [
            *_process_tree_pids(root_pid),
            *_job_env_pids(job_uid),
        ]
    )


def _process_tree_pids(root_pid: int) -> list[int]:
    parent_map: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == root_pid:
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            fields = raw.rsplit(")", 1)[1].strip().split()
            parent_map[pid] = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
    descendants: list[int] = []
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, ppid in parent_map.items() if ppid == parent]
        descendants.extend(children)
        frontier.extend(children)
    return _unique_pids(descendants)


def _job_env_pids(job_uid: str) -> list[int]:
    marker = f"KITT_JOB_UID={job_uid}".encode("utf-8", errors="strict")
    own_pid = os.getpid()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in {own_pid, 1}:
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker in environ.split(b"\0"):
            pids.append(pid)
    return _unique_pids(pids)


def _unique_pids(pids: list[int]) -> list[int]:
    return list(dict.fromkeys(pid for pid in pids if pid > 1))


def _collect_artifacts(
    output_dir: Path,
    spec: dict[str, Any],
    *,
    runner_nonce_sha256: str,
) -> tuple[list[RunnerArtifact], str | None]:
    artifacts: list[RunnerArtifact] = []
    for role, filename in SftSubprocessRunner._ARTIFACT_FILES.items():
        artifact = RunnerArtifact(role=role, path=output_dir / filename)
        error = validate_artifact_file(
            artifact,
            spec=spec,
            runner_nonce_sha256=runner_nonce_sha256,
        )
        if error == "runner_artifact_missing":
            continue
        if error is not None:
            return [], error
        artifacts.append(artifact)
    bundle_error = validate_artifact_bundle(
        tuple(artifacts),
        spec=spec,
        runner_nonce_sha256=runner_nonce_sha256,
    )
    if bundle_error is not None:
        return [], bundle_error
    return artifacts, None


def validate_artifact_file(
    artifact: RunnerArtifact,
    *,
    spec: dict[str, Any] | None = None,
    runner_nonce_sha256: str | None = None,
) -> str | None:
    path = artifact.path
    role = artifact.role
    if role not in SftSubprocessRunner._ARTIFACT_FILES:
        return "runner_artifact_invalid"
    try:
        stat_result = path.stat()
    except OSError:
        return "runner_artifact_missing"
    if not path.is_file() or path.is_symlink():
        return "runner_artifact_invalid"
    size_bytes = stat_result.st_size
    if size_bytes <= 0:
        return "artifact_empty"
    if role == "manifest":
        return _validate_manifest_artifact(
            path,
            spec=spec,
            size_bytes=size_bytes,
            runner_nonce_sha256=runner_nonce_sha256,
        )
    if role in {"adapter", "gguf", "merged_weights"} and size_bytes < MIN_MODEL_ARTIFACT_BYTES:
        return "runner_artifact_invalid"
    if role == "adapter" and not _is_supported_model_artifact(
        path,
        spec=spec,
    ):
        return "runner_artifact_invalid"
    if role == "merged_weights" and not _is_supported_merged_weights_artifact(path):
        return "runner_artifact_invalid"
    if role == "gguf":
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
        except OSError:
            return "runner_artifact_invalid"
        if magic != b"GGUF":
            return "runner_artifact_invalid"
    if role == "run_lock":
        return _validate_json_object_artifact(path, max_bytes=MAX_MANIFEST_BYTES)
    if role in {"metrics_jsonl", "training_log"}:
        return _validate_jsonl_artifact(path, max_bytes=MAX_MANIFEST_BYTES)
    if role == "ollama_modelfile":
        return _validate_text_artifact(path, max_bytes=MAX_MANIFEST_BYTES)
    return None


def validate_artifact_bundle(
    artifacts: tuple[RunnerArtifact, ...],
    *,
    spec: dict[str, Any] | None = None,
    runner_nonce_sha256: str | None = None,
) -> str | None:
    by_role = {artifact.role: artifact for artifact in artifacts}
    if spec is not None and _required_artifact_missing(spec, list(artifacts)):
        return "runner_artifact_missing"
    manifest_artifact = by_role.get("manifest")
    if manifest_artifact is None:
        return "runner_artifact_missing"
    manifest = _read_json_object(manifest_artifact.path)
    if manifest is None:
        return "runner_artifact_invalid"
    if (
        _validate_manifest_shape(
            manifest,
            spec=spec,
            runner_nonce_sha256=runner_nonce_sha256,
        )
        is not None
    ):
        return "runner_artifact_invalid"
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict) or not manifest_artifacts:
        return "runner_artifact_invalid"
    for role, artifact in by_role.items():
        if role == "manifest":
            continue
        entry = manifest_artifacts.get(role)
        if not isinstance(entry, dict):
            return "runner_artifact_invalid"
        try:
            size_bytes = artifact.path.stat().st_size
        except OSError:
            return "runner_artifact_missing"
        if entry.get("size_bytes") != size_bytes:
            return "runner_artifact_invalid"
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or expected_sha256 != _sha256_file(
            artifact.path
        ):
            return "runner_artifact_invalid"
    return None


def _validate_manifest_artifact(
    path: Path,
    *,
    spec: dict[str, Any] | None,
    size_bytes: int,
    runner_nonce_sha256: str | None,
) -> str | None:
    if size_bytes > MAX_MANIFEST_BYTES:
        return "runner_artifact_invalid"
    data = _read_json_object(path)
    if data is None:
        return "runner_artifact_invalid"
    return _validate_manifest_shape(
        data,
        spec=spec,
        runner_nonce_sha256=runner_nonce_sha256,
    )


def _validate_manifest_shape(
    data: dict[str, Any],
    *,
    spec: dict[str, Any] | None,
    runner_nonce_sha256: str | None,
) -> str | None:
    if data.get("schema_version") != RUNNER_MANIFEST_SCHEMA_VERSION:
        return "runner_artifact_invalid"
    if spec is not None and data.get("run_type") != spec.get("run_type"):
        return "runner_artifact_invalid"
    if spec is not None and data.get("run_uid") != spec.get("run_uid"):
        return "runner_artifact_invalid"
    if spec is not None and data.get("job_spec_hash_sha256") != job_stubs.job_spec_hash(
        spec
    ):
        return "runner_artifact_invalid"
    if spec is not None:
        if (
            not isinstance(runner_nonce_sha256, str)
            or data.get("runner_nonce_sha256") != runner_nonce_sha256
        ):
            return "runner_artifact_invalid"
    if spec is not None and _validate_training_provenance(
        data.get("training_provenance"),
        spec=spec,
    ) is not None:
        return "runner_artifact_invalid"
    trained_steps = data.get("trained_steps")
    if isinstance(trained_steps, bool) or not isinstance(trained_steps, int):
        return "runner_artifact_invalid"
    if trained_steps <= 0:
        return "runner_artifact_invalid"
    return None


def _validate_training_provenance(
    value: object,
    *,
    spec: dict[str, Any],
) -> str | None:
    if not isinstance(value, dict):
        return "runner_artifact_invalid"
    if value.get("kind") != RUNNER_PROVENANCE_KIND:
        return "runner_artifact_invalid"
    if value.get("provenance_version") != RUNNER_PROVENANCE_VERSION:
        return "runner_artifact_invalid"
    if value.get("base_model_ref") != _base_model_ref(spec):
        return "runner_artifact_invalid"
    if value.get("training_profile_hash_sha256") != spec["training_profile"].get(
        "profile_hash_sha256"
    ):
        return "runner_artifact_invalid"
    if value.get("input_reference_hash_sha256") != _json_hash(spec["input_reference"]):
        return "runner_artifact_invalid"
    trainer_entrypoint = value.get("trainer_entrypoint")
    if not isinstance(trainer_entrypoint, dict):
        return "runner_artifact_invalid"
    if trainer_entrypoint.get("kind") != RUNNER_ENTRYPOINT_KIND:
        return "runner_artifact_invalid"
    if trainer_entrypoint.get("version") != RUNNER_ENTRYPOINT_VERSION:
        return "runner_artifact_invalid"
    optimizer_steps = value.get("optimizer_steps")
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps <= 0
    ):
        return "runner_artifact_invalid"
    examples_seen = value.get("dataset_examples_seen")
    if (
        isinstance(examples_seen, bool)
        or not isinstance(examples_seen, int)
        or examples_seen <= 0
    ):
        return "runner_artifact_invalid"
    final_loss = value.get("final_train_loss")
    if isinstance(final_loss, bool) or not isinstance(final_loss, (int, float)):
        return "runner_artifact_invalid"
    if final_loss != final_loss or final_loss in {float("inf"), float("-inf")}:
        return "runner_artifact_invalid"
    return None


def _validate_json_object_artifact(path: Path, *, max_bytes: int) -> str | None:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        return "runner_artifact_missing"
    if size_bytes > max_bytes:
        return "runner_artifact_invalid"
    data = _read_json_object(path)
    return None if data is not None else "runner_artifact_invalid"


def _validate_jsonl_artifact(path: Path, *, max_bytes: int) -> str | None:
    try:
        size_bytes = path.stat().st_size
        if size_bytes > max_bytes:
            return "runner_artifact_invalid"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError):
        return "runner_artifact_invalid"
    if not lines:
        return "runner_artifact_invalid"
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return "runner_artifact_invalid"
        if not isinstance(data, dict):
            return "runner_artifact_invalid"
    return None


def _validate_text_artifact(path: Path, *, max_bytes: int) -> str | None:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "runner_artifact_invalid"
    if len(data.encode("utf-8")) > max_bytes or not data.strip():
        return "runner_artifact_invalid"
    if "\x00" in data:
        return "runner_artifact_invalid"
    return None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_supported_model_artifact(
    path: Path,
    *,
    spec: dict[str, Any] | None = None,
) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = {info.filename for info in infos}
            if not names or any(_unsafe_zip_name(name) for name in names):
                return False
            by_basename = {
                Path(info.filename).name: info for info in infos if info.file_size > 0
            }
            if not _MODEL_ZIP_REQUIRED_FILES.issubset(by_basename):
                return False
            config_info = by_basename["adapter_config.json"]
            try:
                config_data = json.loads(archive.read(config_info).decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False
            if not _valid_peft_adapter_config(config_data, spec=spec):
                return False
            for weight_name in _MODEL_ZIP_WEIGHT_FILES:
                weight_info = by_basename.get(weight_name)
                if weight_info is None:
                    continue
                try:
                    if _valid_safetensors_payload(archive.read(weight_info)):
                        return True
                except OSError:
                    return False
            return False
    except (OSError, zipfile.BadZipFile):
        return False


def _is_supported_merged_weights_artifact(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            names = {info.filename for info in infos}
            if not names or any(_unsafe_zip_name(name) for name in names):
                return False
            by_basename = {
                Path(info.filename).name: info for info in infos if info.file_size > 0
            }
            if not _MERGED_WEIGHTS_REQUIRED_FILES.issubset(by_basename):
                return False
            if not _MERGED_WEIGHTS_TOKENIZER_FILES.intersection(by_basename):
                return False
            weight_files = [
                info
                for info in infos
                if Path(info.filename).name.endswith(".safetensors")
            ]
            if not weight_files:
                return False
            try:
                config_data = json.loads(archive.read(by_basename["config.json"]).decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False
            return isinstance(config_data, dict)
    except (OSError, zipfile.BadZipFile):
        return False


def _unsafe_zip_name(name: str) -> bool:
    path = Path(name)
    return (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_peft_adapter_config(
    data: object,
    *,
    spec: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("peft_type") != "LORA":
        return False
    base_model = data.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not _safe_adapter_text(base_model):
        return False
    if spec is not None and base_model != _base_model_ref(spec):
        return False
    rank = data.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        return False
    alpha = data.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0:
        return False
    target_modules = data.get("target_modules")
    if isinstance(target_modules, str):
        return _safe_adapter_text(target_modules)
    if isinstance(target_modules, list) and target_modules:
        return all(
            isinstance(item, str) and _safe_adapter_text(item)
            for item in target_modules
        )
    return False


def _safe_adapter_text(value: str) -> bool:
    return bool(value) and "\x00" not in value and len(value) <= 256


def _base_model_ref(spec: dict[str, Any]) -> str | None:
    base = spec.get("base_or_parent_model")
    if not isinstance(base, dict):
        return None
    value = base.get("external_parent_ref")
    return value if isinstance(value, str) else None


def _json_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_safetensors_payload(payload: bytes) -> bool:
    if len(payload) <= 12:
        return False
    header_len = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_len <= 0 or header_len > MAX_MANIFEST_BYTES:
        return False
    header_end = 8 + header_len
    if header_end >= len(payload):
        return False
    try:
        header = json.loads(payload[8:header_end].rstrip(b" ").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    data_len = len(payload) - header_end
    total_tensor_bytes = 0
    nonzero_bytes = 0
    unique_values: set[int] = set()
    tensor_seen = False
    lora_a_seen = False
    lora_b_seen = False
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name:
            return False
        if not isinstance(value, dict):
            return False
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        dtype_size = _SAFETENSORS_DTYPE_BYTES.get(dtype)
        if dtype_size is None:
            return False
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            return False
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
        ):
            return False
        start, end = offsets
        if start < 0 or end <= start or end > data_len:
            return False
        expected_bytes = dtype_size
        for dim in shape:
            expected_bytes *= dim
        if expected_bytes <= 0 or end - start != expected_bytes:
            return False
        if len(shape) < 2:
            return False
        tensor_payload = payload[header_end + start : header_end + end]
        nonzero_bytes += sum(1 for byte in tensor_payload if byte != 0)
        unique_values.update(tensor_payload)
        lowered_name = name.lower()
        lora_a_seen = lora_a_seen or "lora_a" in lowered_name
        lora_b_seen = lora_b_seen or "lora_b" in lowered_name
        total_tensor_bytes += expected_bytes
        tensor_seen = True
    return (
        tensor_seen
        and lora_a_seen
        and lora_b_seen
        and total_tensor_bytes >= MIN_SAFETENSORS_TENSOR_BYTES
        and nonzero_bytes >= MIN_SAFETENSORS_NONZERO_BYTES
        and len(unique_values) >= MIN_SAFETENSORS_UNIQUE_BYTE_VALUES
    )


def _required_artifact_missing(
    spec: dict[str, Any],
    artifacts: list[RunnerArtifact],
) -> bool:
    roles = {artifact.role for artifact in artifacts}
    required = _required_artifact_roles(spec)
    return not required.issubset(roles)


def _required_artifact_roles(spec: dict[str, Any]) -> set[str]:
    required = {"manifest"}
    for role in spec.get("output_roles", []):
        if role in SftSubprocessRunner._ARTIFACT_FILES:
            required.add(role)
        else:
            required.add("unsupported_output_role")
    return required
