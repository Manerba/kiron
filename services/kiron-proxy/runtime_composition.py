"""Explicit composition of the local runtime, existing registry and transports."""
from __future__ import annotations

import asyncio
from dataclasses import fields
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import threading
import time

import httpx

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot
from kiron_common.local_inference import RuntimeImplementation, RuntimeTimeouts, build_resolver_snapshot
from kiron_common.local_model_registry import RuntimeModelRegistry
from kiron_common.model_catalog import BackendType
from kiron_common.ollama_compat import CompatResult, OllamaCapabilities
from kiron_common.prism_runtime_policy import Policy, PolicyError, file_identity, immutable_path

from prism_provider import PrismProvider
from runtime_service import RuntimeService
from runtime_control import PrismServiceControl
from runtime_capabilities import decode_provider_evidence


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_TIMEOUTS = RuntimeTimeouts(startup=180, readiness=180, first_token=180, idle=120,
                                   total=900, drain=30, stop=30)
logger = logging.getLogger(__name__)


class SnapshotResolver:
    def __init__(self, registry, resource_profiles, catalog=MODEL_CATALOG):
        self.registry, self.profiles, self.catalog = registry, dict(resource_profiles), catalog

    async def snapshot(self):
        entries = await asyncio.to_thread(self.registry.list)
        return build_resolver_snapshot(self.catalog, entries, resource_profiles=self.profiles)


def _owned_json(path: Path, *, root: Path, limit=1024 * 1024, expected_sha256=None):
    if not path.is_relative_to(root):
        raise ValueError("configuration is outside the runtime data root")
    immutable_path(path, anchor=root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("invalid runtime configuration file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit or file_identity(os.fstat(fd)) != file_identity(info):
            raise ValueError("runtime configuration changed while reading")
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("runtime configuration digest mismatch")
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate runtime configuration key")
                value[key] = item
            return value
        def invalid_constant(_):
            raise ValueError("invalid JSON constant in runtime configuration")
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant)
        except RecursionError as exc:
            raise ValueError("runtime configuration nesting exceeds decoder limit") from exc
        if type(value) is not dict:
            raise ValueError("runtime configuration must be an object")
        return value
    finally:
        os.close(fd)


def load_ollama_compatibility(runtime_path: Path, *, data_root: Path):
    """Consume the existing deploy gate's complete report; no positive defaults."""
    runtime = _owned_json(runtime_path, root=data_root)
    report_hash = runtime.get("report_sha256")
    if type(report_hash) is not str or len(report_hash) != 64:
        raise ValueError("Runtime handoff has no pinned compatibility report")
    report = _owned_json(Path(runtime["report_path"]), root=data_root, expected_sha256=report_hash)
    if (report.get("image_digest") != runtime.get("image_digest")
            or not isinstance(report.get("image_digest"), str) or not report["image_digest"]
            or report.get("report_status") != "passed" or report.get("upgrade_allowed") is not True
            or report.get("failures") != []):
        raise ValueError("Ollama compatibility report is not a passing matching deploy gate")
    observed = report.get("capabilities")
    if type(observed) is not dict:
        raise ValueError("Ollama compatibility evidence missing")
    values = {}
    for field in fields(OllamaCapabilities):
        value = observed.get(field.name)
        if type(value) is not dict or type(value.get("ok")) is not bool:
            values[field.name] = CompatResult(False, "fail", "evidence_missing", "Check is not present in the report")
        else:
            values[field.name] = CompatResult(value["ok"], value.get("severity", "fail"),
                                               "recorded_check", str(value.get("message", "")), value.get("data"))
    gpu_check = observed.get("num_gpu_zero_chat_generate")
    gpu_evidence = gpu_check.get("data") if type(gpu_check) is dict else None
    if (type(runtime.get("num_gpu_zero_effective")) is not bool or type(gpu_evidence) is not dict
            or (runtime["num_gpu_zero_effective"] and gpu_evidence.get("num_gpu_zero_effective") is not True)):
        raise ValueError("Runtime CPU-offload flag disagrees with its measured report")
    if not runtime["num_gpu_zero_effective"]:
        values["num_gpu_zero_chat_generate"] = CompatResult(False, "fail", "runtime_gate_closed", "CPU offload is not enabled by the runtime handoff")
    version_check = observed.get("version_endpoint")
    version_data = version_check.get("data") if type(version_check) is dict else None
    version = version_data.get("version") if type(version_data) is dict else None
    if type(version) is not str or not version:
        raise ValueError("Measured Ollama version missing")
    return OllamaCapabilities(**values), version, report["image_digest"]


class MemoryMeasurement:
    """Read actual free GPU memory on demand inside the admission transaction."""
    def __init__(self):
        self._lock = threading.Lock()
        self._nvml = self._handle = None

    def __call__(self):
        import psutil
        import pynvml
        with self._lock:
            try:
                if self._nvml is None:
                    pynvml.nvmlInit()
                    self._nvml = pynvml
                    if pynvml.nvmlDeviceGetCount() != 1:
                        raise AdmissionError("resource_unknown", "This admission policy requires exactly one GPU")
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                if self._handle is None:
                    raise AdmissionError("resource_unknown", "GPU measurement is not initialized")
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                return MemorySnapshot(int(memory.free), int(psutil.virtual_memory().available), time.monotonic())
            except pynvml.NVMLError as exc:
                raise AdmissionError("resource_unknown", "GPU memory cannot be measured") from exc

    def close(self):
        with self._lock:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
                self._nvml = self._handle = None


class ArtifactVerifier:
    def __init__(self, policy):
        self.policy, self._cache, self._lock = policy, {}, threading.Lock()

    def _verify(self, deployment):
        artifact = deployment.artifact_identity
        files = [(deployment.reference, artifact.sha256, artifact.size_bytes)]
        if artifact.projector:
            p = artifact.projector
            files.append((p.reference, p.sha256, p.size_bytes))
        with self._lock:
            for reference, digest, size in files:
                path = Path(reference)
                identity = file_identity(immutable_path(path, anchor=self.policy.anchor))
                key = (reference, digest, size)
                if self._cache.get(key) != identity:
                    fd = self.policy.open_artifact(reference, digest, size)
                    try:
                        self._cache[key] = file_identity(os.fstat(fd))
                    finally:
                        os.close(fd)
        return True

    async def __call__(self, deployment):
        try:
            return await asyncio.to_thread(self._verify, deployment)
        except (PolicyError, OSError):
            return False


def adapter_revision(filename):
    """Bind evidence to the complete parser, validation and DTO implementation."""
    import kiron_common.local_inference
    provider_files = {
        "prism_provider.py": ("prism_tools.py", "prism_vision.py", "prism_structured.py",
                              "prism_reasoning.py", "prism_reasoning_profile.py"),
        "ollama_provider.py": ("ollama_tools.py", "ollama_vision.py", "ollama_generation.py"),
        "embedding_provider.py": ("provider_embeddings.py",),
    }
    if filename not in provider_files:
        raise ValueError("unknown inference adapter")
    shared = ("provider_transport.py", "provider_features.py", "runtime_capabilities.py",
              "runtime_service.py", "runtime_composition.py", "openai_generation.py",
              "openai_tools.py", "openai_vision.py", "openai_wire.py", "openai_embeddings.py",
              "openai_responses.py", "openai_responses_stream.py")
    paths = [("proxy/" + name, Path(__file__).with_name(name))
             for name in (filename, *provider_files[filename], *shared)]
    common = Path(kiron_common.local_inference.__file__).parent
    paths.extend(("common/local_inference/" + path.name, path) for path in common.glob("*.py"))
    if filename == "prism_provider.py":
        paths.append(("common/prism_runtime_policy.py", common.parent / "prism_runtime_policy.py"))
        controller = Path(__file__).resolve().parent.parent / "kiron-prism"
        paths.extend(("prism/" + name, controller / name) for name in
                     ("main.py", "composition.py", "controller.py", "process.py", "admission.py"))
    digest = hashlib.sha256()
    for name, path in sorted(paths):
        digest.update(name.encode("ascii") + b"\0")
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def load_capability_evidence(data_root):
    evidence = {}
    for provider in (BackendType.OLLAMA, BackendType.PRISM, BackendType.KIRON_EMBEDDINGS):
        path = data_root / "local-inference-capabilities" / f"{provider.value}.json"
        if not (path.exists() or path.is_symlink()):
            continue
        try:
            evidence[provider] = decode_provider_evidence(_owned_json(path, root=data_root), provider)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Capability evidence for %s is unavailable: %s", provider.value, type(exc).__name__)
    return evidence


def build_runtime_service(*, data_root=DATA_DIR, registry=None, catalog=MODEL_CATALOG,
                          admission=None, measure=None, timeouts=DEFAULT_TIMEOUTS,
                          capability_evidence=None, service_control=None):
    """Construct clients once; no downloads, model loading or service mutation.

    Absent Prism policy means the optional provider is unconfigured. A present
    invalid policy raises instead of silently disabling it. Capability evidence
    is injected independently from registration and never invented here.
    """
    from ollama_provider import OllamaProvider
    policy_path = data_root / "prism-runtime-policy.json"
    policy = Policy.load(policy_path) if policy_path.exists() or policy_path.is_symlink() else None
    resolver = SnapshotResolver(registry or RuntimeModelRegistry(), policy.resource_profiles() if policy else {}, catalog)
    try:
        compatibility, version, digest = load_ollama_compatibility(data_root / "ollama_compat_runtime.json", data_root=data_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ollama runtime compatibility is unavailable: %s", exc)
        compatibility, version, digest = None, None, "unverified"
    evidence = load_capability_evidence(data_root) if capability_evidence is None else capability_evidence
    ollama_client = httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False,
                                    timeout=httpx.Timeout(timeouts.idle, connect=5), follow_redirects=False)
    providers = {BackendType.OLLAMA: OllamaProvider(client=ollama_client, resolver=resolver,
        implementation=RuntimeImplementation(digest, None, adapter_revision("ollama_provider.py")),
        compatibility=compatibility, expected_version=version, capabilities=evidence.get(BackendType.OLLAMA))}
    from embedding_provider import KironEmbeddingProvider
    embedding_evidence = evidence.get(BackendType.KIRON_EMBEDDINGS, {})
    revisions = {item.provider_revision for caps in embedding_evidence.values()
                 for cap in caps.by_name.values() for item in cap.evidence}
    embedding_revision = next(iter(revisions)) if len(revisions) == 1 else "unverified"
    embedding_client = httpx.AsyncClient(base_url="http://127.0.0.1:11436", trust_env=False,
        timeout=httpx.Timeout(timeouts.idle, connect=5), follow_redirects=False)
    providers[BackendType.KIRON_EMBEDDINGS] = KironEmbeddingProvider(client=embedding_client, resolver=resolver,
        implementation=RuntimeImplementation(embedding_revision, None, adapter_revision("embedding_provider.py")),
        catalog_digest=catalog.catalog_digest, capabilities=embedding_evidence)
    if policy is not None:
        control = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds="/run/kiron/prism/control.sock"),
            base_url="http://prism", trust_env=False, timeout=httpx.Timeout(timeouts.startup + timeouts.readiness, connect=5))
        inference = httpx.AsyncClient(base_url=f"http://127.0.0.1:{policy.port}", trust_env=False,
            timeout=httpx.Timeout(timeouts.idle, connect=5), follow_redirects=False)
        providers[BackendType.PRISM] = PrismProvider(control=control, inference=inference, resolver=resolver,
            implementation=RuntimeImplementation(policy.runtime_revision, None, adapter_revision("prism_provider.py")),
            capabilities=evidence.get(BackendType.PRISM), service_control=service_control or PrismServiceControl(),
            verify_artifact=ArtifactVerifier(policy))
    measurement = measure if measure is not None else MemoryMeasurement()
    return RuntimeService(resolver=resolver, providers=providers, admission=admission or AdmissionStore(),
                          measure=measurement, timeouts=timeouts,
                          measurement_close=measurement.close if measure is None else None)
