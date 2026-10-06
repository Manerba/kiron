"""Worker-owned, measured resident tickets. This module never loads a model.

Only tickets created by this process are renewed/rotated/released. Native cache
operations without a measured policy remain native operations, not API grants.
Reconciliation is called at a serial worker boundary, never from a health poll.
"""
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from kiron_common.gpu_admission import AdmissionError, AdmissionStore, MemorySnapshot
from kiron_common.local_inference import build_resolver_snapshot
from kiron_common.model_catalog import BackendType
from kiron_common.prism_runtime_policy import immutable_path

OWNER = "kiron-embeddings"
POLICY_PATH = Path("/usr/lib/kiron/data/embedding-residency-policy.json")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceSample:
    available: MemorySnapshot
    rss_bytes: int
    gpu_bytes: int


def measure_resources():
    """Fresh whole-host availability and this process's own RSS/CUDA usage."""
    def query(*args):
        result = subprocess.run(["/usr/bin/nvidia-smi", *args, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
        if len(result.stdout) > 65536:
            raise ValueError("GPU observation exceeds bound")
        return result.stdout.strip().splitlines()
    try:
        measured_at = time.monotonic()
        free = query("--query-gpu=memory.free")
        if len(free) != 1:
            raise ValueError("one GPU required")
        rows = [line.split(",") for line in query("--query-compute-apps=pid,used_memory")]
        own = [int(size.strip()) * 1024**2 for pid, size in rows if int(pid.strip()) == os.getpid()]
        if len(own) > 1:
            raise ValueError("ambiguous process GPU usage")
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        available = MemorySnapshot(int(free[0]) * 1024**2, int(mem["MemAvailable"].split()[0]) * 1024,
                                   measured_at)
        sample = ResourceSample(available, rss, own[0] if own else 0)
        if any(type(v) is not int or v < 0 for v in (available.gpu_free_bytes, available.host_available_bytes,
                                                   sample.rss_bytes, sample.gpu_bytes)) or rss == 0:
            raise ValueError("invalid memory observation")
        return sample
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        raise AdmissionError("resource_unknown", "embedding memory measurement unavailable") from exc


@dataclass(frozen=True)
class Profile:
    deployment_id: str
    reference: str
    artifact_fingerprint: str
    configuration_fingerprint: str
    device: str
    gpu_bytes: int
    host_bytes: int
    headroom_bytes: int
    load_timeout: int


def load_profiles(path, catalog):
    """Optional absent policy grants nothing; malformed/present policy fails shut."""
    try:
        path.lstat()
    except FileNotFoundError:
        return {}
    info = immutable_path(path)
    if info.st_nlink != 1 or info.st_size > 65536:
        raise ValueError("invalid embedding residency policy file")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate residency policy field")
            result[key] = value
        return result
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            raise ValueError("residency policy changed")
        raw = os.read(fd, 65537)
        after = os.fstat(fd)
        if (len(raw) > 65536 or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)):
            raise ValueError("residency policy exceeds size bound")
    finally:
        os.close(fd)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if (type(value) is not dict or set(value) != {"version", "profiles"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["profiles"]) is not list or not 1 <= len(value["profiles"]) <= 32):
        raise ValueError("invalid embedding residency policy")
    resolver = build_resolver_snapshot(catalog, ())
    profiles = {}
    fields = {"deployment_id", "artifact_fingerprint", "configuration_fingerprint", "device",
              "gpu_bytes", "host_bytes", "headroom_bytes", "load_timeout"}
    for row in value["profiles"]:
        if type(row) is not dict or set(row) != fields:
            raise ValueError("invalid measured residency profile fields")
        deployment = resolver.resolve_deployment(row["deployment_id"])
        if (deployment.provider is not BackendType.KIRON_EMBEDDINGS
                or row["artifact_fingerprint"] != deployment.artifact_identity.fingerprint
                or row["configuration_fingerprint"] != deployment.configuration_fingerprint
                or row["device"] not in {"cpu", "cuda"}):
            raise ValueError("residency policy identity differs from catalog")
        for name, minimum, maximum in (("gpu_bytes", 0, 12 * 1024**3), ("host_bytes", 1, 64 * 1024**3),
                                      ("headroom_bytes", 0, 12 * 1024**3), ("load_timeout", 1, 180)):
            if type(row[name]) is not int or not minimum <= row[name] <= maximum:
                raise ValueError("invalid measured residency resource bound")
        if (row["device"] == "cpu" and row["gpu_bytes"] != 0
                or row["device"] == "cuda" and row["gpu_bytes"] == 0):
            raise ValueError("device and GPU resource bounds disagree")
        key = deployment.reference, row["device"]
        if key in profiles:
            raise ValueError("duplicate embedding residency profile")
        profiles[key] = Profile(reference=deployment.reference, **row)
    return profiles


class ResidencyOwner:
    HEARTBEAT_SECONDS = 30
    TTL_SECONDS = 300

    def __init__(self, *, profiles, store, generation, measure=measure_resources, clock=time.monotonic):
        self.profiles, self.store, self.generation = dict(profiles), store, generation
        self.measure, self.clock = measure, clock
        self.residents, self.pending, self.observations = {}, {}, {}
        self.last_heartbeat = 0
        self.failed = False

    def before_load(self, name, device, snapshot, instance, *, load_parent: tuple[str, str] | None = None):
        if self.failed:
            raise AdmissionError("resource_unknown", "embedding owner needs confirmed cleanup")
        profile = self.profiles.get((name, device))
        if profile is None:
            if any(key[0] == name for key in self.profiles):
                self.unknown()
                raise AdmissionError("resource_unknown", "embedding device has no measured resource profile")
            return
        if name in self.pending:
            raise AdmissionError("operation_conflict", "embedding load already reserved")
        previous = self.residents.get(name)
        sample = self.measure()
        generation = self.generation(snapshot)
        ticket = self.store.reserve(operation_id="embedding-load-" + uuid4().hex, owner=OWNER,
            generation=generation, deployment_id=profile.deployment_id, kind="load",
            gpu_bytes=profile.gpu_bytes, host_bytes=profile.host_bytes, headroom_bytes=profile.headroom_bytes,
            measure=lambda: sample.available, ttl_seconds=self.TTL_SECONDS, allow_existing=False,
            unique_deployment=True, replacing_operation=previous.operation_id if previous else None,
            embedding_load_parent=load_parent)
        self.pending[name] = (ticket, profile, sample, instance, self.clock())

    def complete(self, snapshot, instances):
        """Only after a serial job/drop returns: no health-based end inference."""
        generation = self.generation(snapshot)
        verified = snapshot.get("verified_artifacts", {})
        loaded = set(snapshot["loaded_models"])
        try:
            sample = self.measure() if self.pending or self.residents else None
            for name, (ticket, profile, before, original, started) in tuple(self.pending.items()):
                replaced = instances.get(name) != original
                present = (name in loaded and verified.get(name) == profile.artifact_fingerprint
                           and snapshot["device"] == profile.device)
                if name in loaded and replaced and not present:
                    raise AdmissionError("resource_unknown", "new embedding instance has no matching artifact/device proof")
                if present and replaced:
                    if (self.clock() - started > profile.load_timeout
                            or max(0, sample.gpu_bytes - before.gpu_bytes) > profile.gpu_bytes
                            or max(0, sample.rss_bytes - before.rss_bytes) > profile.host_bytes):
                        raise AdmissionError("resource_exhausted", "embedding load exceeded its measured resource profile")
                    previous = self.residents.pop(name, None)
                    if previous:
                        self.store.release(previous.operation_id, owner=OWNER, generation=previous.generation,
                                           confirmed_terminated=True)
                    self.residents[name] = self.store.transition(ticket.operation_id, owner=OWNER,
                        expected_generation=ticket.generation, phase="resident", generation=generation)
                    self.observations[name] = {"rss_bytes": sample.rss_bytes, "gpu_bytes": sample.gpu_bytes,
                                              "measured_monotonic": sample.available.measured_monotonic}
                else:
                    # Failed load did not install this instance. The worker has
                    # returned, so this own attempted load has definitively ended.
                    self.store.release(ticket.operation_id, owner=OWNER, generation=ticket.generation,
                                       confirmed_terminated=True)
                del self.pending[name]
            for name, ticket in tuple(self.residents.items()):
                profile = self.profiles.get((name, snapshot["device"]))
                if name not in loaded:
                    self.store.release(ticket.operation_id, owner=OWNER, generation=ticket.generation,
                                       confirmed_terminated=True)
                    del self.residents[name]
                    self.observations.pop(name, None)
                elif profile is None or verified.get(name) != profile.artifact_fingerprint:
                    raise AdmissionError("resource_unknown", "resident embedding artifact/device became unconfirmed")
                else:
                    self.residents[name] = self.store.transition(ticket.operation_id, owner=OWNER,
                        expected_generation=ticket.generation, phase="resident", generation=generation)
            self.last_heartbeat = self.clock()
            if not self.residents and not self.pending:
                self.failed = False
        except BaseException:
            self.unknown()
            raise

    def heartbeat(self, snapshot):
        if self.failed or not self.residents or self.clock() - self.last_heartbeat < self.HEARTBEAT_SECONDS:
            return
        try:
            self.measure()  # A stale/unreadable GPU/RAM observation is not health.
            for name, ticket in self.residents.items():
                profile = self.profiles.get((name, snapshot["device"]))
                if (name not in snapshot["loaded_models"] or profile is None
                        or snapshot.get("verified_artifacts", {}).get(name) != profile.artifact_fingerprint
                        or ticket.generation != self.generation(snapshot)):
                    raise AdmissionError("resource_unknown", "embedding resident heartbeat identity changed")
                self.store.heartbeat(ticket.operation_id, owner=OWNER, generation=ticket.generation)
            self.last_heartbeat = self.clock()
        except BaseException:
            self.unknown()
            raise

    def unknown(self):
        self.failed = True
        for ticket in [*self.residents.values(), *(row[0] for row in self.pending.values())]:
            try:
                self.store.release(ticket.operation_id, owner=OWNER, generation=ticket.generation,
                                   confirmed_terminated=False)
            except AdmissionError:
                logger.exception("embedding residency could not be marked unknown")


def build_owner(catalog, generation, path=POLICY_PATH):
    profiles = load_profiles(path, catalog)
    return ResidencyOwner(profiles=profiles, store=AdmissionStore(), generation=generation) if profiles else None
