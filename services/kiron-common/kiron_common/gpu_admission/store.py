from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import grp
import json
import math
import os
from pathlib import Path
import pwd
import stat
import time
from typing import Callable, Iterator
from collections.abc import Mapping
import uuid


DEFAULT_ROOT = Path("/run/kiron/vram")
OVERLAYS = (
    "docling-vram-startup.json", "docling-vram-shutdown.json",
    "gpu-service-loading.json",
)
MAX_STATE_BYTES = 1024 * 1024


class AdmissionError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True, slots=True)
class RuntimeSecurity:
    directory_uid: int
    group_gid: int
    writer_uids: frozenset[int]

    @classmethod
    def system(cls) -> RuntimeSecurity:
        try:
            gid = grp.getgrnam("kiron-runtime").gr_gid
        except KeyError as exc:
            raise AdmissionError("resource_unknown", "kiron-runtime group missing") from exc
        uids = {0}
        for name in ("kiron-proxy", "kiron-docling", "kiron-prism", "kiron-embeddings", "kitt-worker"):
            try:
                uids.add(pwd.getpwnam(name).pw_uid)
            except KeyError:
                continue
        return cls(0, gid, frozenset(uids))


def _open_root(root: Path, security: RuntimeSecurity) -> int:
    # Reject symlink ancestors as well as a substituted leaf directory.
    if not root.is_absolute():
        raise AdmissionError("resource_unknown", "runtime root must be absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        st = os.fstat(fd)
        if (st.st_uid != security.directory_uid or st.st_gid != security.group_gid
                or stat.S_IMODE(st.st_mode) != 0o2770):
            raise AdmissionError("resource_unknown", "unsafe runtime directory ownership or mode")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_file(fd: int, security: RuntimeSecurity) -> None:
    st = os.fstat(fd)
    if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
            or st.st_uid not in security.writer_uids or st.st_gid != security.group_gid
            or stat.S_IMODE(st.st_mode) != 0o660):
        raise AdmissionError("resource_unknown", "unsafe runtime file ownership, links or mode")


@contextmanager
def _locked_directory(root: Path, security: RuntimeSecurity) -> Iterator[int]:
    directory = lock = None
    try:
        directory = _open_root(root, security)
        try:
            lock = os.open(".admission.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CREAT | os.O_EXCL,
                           0o660, dir_fd=directory)
            os.fchmod(lock, 0o660)
        except FileExistsError:
            lock = os.open(".admission.lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
        _check_file(lock, security)
        deadline = time.monotonic() + 3
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AdmissionError("resource_unknown", "GPU admission lock deadline exceeded")
                time.sleep(0.01)
        yield directory
    except OSError as exc:
        raise AdmissionError("resource_unknown", "GPU admission storage inaccessible") from exc
    finally:
        if lock is not None:
            os.close(lock)
        if directory is not None:
            os.close(directory)


@contextmanager
def runtime_lock(root: Path = DEFAULT_ROOT, *, security: RuntimeSecurity | None = None) -> Iterator[None]:
    """The same outer lock must wrap all existing overlay mutations.

    It is intentionally not reentrant. Never acquire it from a measurement or
    store callback, and always acquire it before a per-marker lock.
    """
    with _locked_directory(root, security or RuntimeSecurity.system()):
        yield


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    gpu_free_bytes: int
    host_available_bytes: int
    measured_monotonic: float


@dataclass(frozen=True, slots=True)
class Ticket:
    operation_id: str
    owner: str
    generation: str
    deployment_id: str
    kind: str  # load, unload, request, docling, training
    phase: str  # reserved, resident, draining, active, unknown
    gpu_bytes: int
    host_bytes: int
    heartbeat_monotonic: float
    ttl_seconds: float
    resident_slot: str | None = None
    lifecycle_domain: str | None = None
    lifecycle_model: str | None = None
    gpu_guard: bool = True
    backend_instance: str | None = None
    overlay_token: str | None = None


def _nonnegative(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_ticket(ticket: Ticket) -> None:
    if any(type(v) is not str or not v or len(v) > 512 for v in (
        ticket.operation_id, ticket.owner, ticket.generation, ticket.deployment_id,
    )):
        raise AdmissionError("resource_unknown", "invalid admission identity")
    if ticket.kind not in {"load", "unload", "request", "docling", "training"} or ticket.phase not in {
        "reserved", "resident", "draining", "active", "unknown",
    }:
        raise AdmissionError("resource_unknown", "invalid admission state")
    if not all(_nonnegative(v) for v in (ticket.gpu_bytes, ticket.host_bytes)):
        raise AdmissionError("resource_unknown", "invalid resource reservation")
    if (not _finite(ticket.heartbeat_monotonic) or not _finite(ticket.ttl_seconds)
            or not 0 < ticket.ttl_seconds <= 86400):
        raise AdmissionError("resource_unknown", "invalid admission freshness")
    if ticket.resident_slot is not None and (type(ticket.resident_slot) is not str or not ticket.resident_slot):
        raise AdmissionError("resource_unknown", "invalid resident slot")
    for value in (ticket.lifecycle_domain, ticket.lifecycle_model):
        if value is not None and (type(value) is not str or not value or len(value) > 512):
            raise AdmissionError("resource_unknown", "invalid lifecycle identity")
    if ticket.lifecycle_model is not None and ticket.lifecycle_domain is None:
        raise AdmissionError("resource_unknown", "lifecycle model requires a domain")
    if ticket.backend_instance is not None:
        from .ollama_backend import valid_instance
        if not valid_instance(ticket.backend_instance):
            raise AdmissionError("resource_unknown", "invalid Ollama backend instance")
    if ticket.overlay_token is not None and (type(ticket.overlay_token) is not str
            or len(ticket.overlay_token) != 32 or any(c not in "0123456789abcdef" for c in ticket.overlay_token)):
        raise AdmissionError("resource_unknown", "invalid admission overlay token")
    if type(ticket.gpu_guard) is not bool or (not ticket.gpu_guard and (
            ticket.kind != "request" or ticket.gpu_bytes or ticket.host_bytes or ticket.resident_slot
            or ticket.lifecycle_domain is None)):
        raise AdmissionError("resource_unknown", "invalid coordination-only request")


class AdmissionStore:
    def __init__(self, root: Path = DEFAULT_ROOT, *, security: RuntimeSecurity | None = None,
                 clock: Callable[[], float] = time.monotonic, boot_id: str | None = None) -> None:
        self.root = root
        self._security = security
        self.clock = clock
        self._boot_id = boot_id

    @property
    def security(self) -> RuntimeSecurity:
        return self._security or RuntimeSecurity.system()

    def _boot(self) -> str:
        return self._boot_id or Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def _read(self, directory: int) -> tuple[int, dict[str, Ticket]]:
        try:
            fd = os.open("admission.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return 0, {}
        try:
            _check_file(fd, self.security)
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(MAX_STATE_BYTES + 1)
            if len(raw) > MAX_STATE_BYTES:
                raise ValueError("state too large")
            state = json.loads(raw)
            if (type(state) is not dict or set(state) != {"schema_version", "boot_id", "revision", "tickets"}
                    or type(state["schema_version"]) is not int or state["schema_version"] != 3
                    or state["boot_id"] != self._boot() or not _nonnegative(state["revision"])
                    or type(state["tickets"]) is not list):
                raise ValueError("unknown admission schema or boot")
            tickets: dict[str, Ticket] = {}
            for row in state["tickets"]:
                if type(row) is not dict or set(row) != set(Ticket.__dataclass_fields__):
                    raise ValueError("invalid ticket fields")
                ticket = Ticket(**row)
                _validate_ticket(ticket)
                if ticket.operation_id in tickets:
                    raise ValueError("duplicate operation")
                tickets[ticket.operation_id] = ticket
            return state["revision"], tickets
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise AdmissionError("resource_unknown", "GPU admission state invalid") from exc
        finally:
            os.close(fd)

    def _write(self, directory: int, revision: int, tickets: dict[str, Ticket]) -> None:
        raw = json.dumps({"schema_version": 3, "boot_id": self._boot(), "revision": revision + 1,
                          "tickets": [asdict(t) for t in tickets.values()]}, allow_nan=False).encode()
        if len(raw) > MAX_STATE_BYTES:
            raise AdmissionError("resource_unknown", "admission state limit reached")
        name = f".admission-{uuid.uuid4().hex}.tmp"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660, dir_fd=directory)
        try:
            os.fchmod(fd, 0o660)
            _check_file(fd, self.security)
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(fd)
            os.replace(name, "admission.json", src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=directory)
            except FileNotFoundError:
                pass

    def _observed(self, ticket: Ticket) -> Ticket:
        age = self.clock() - ticket.heartbeat_monotonic
        return replace(ticket, phase="unknown") if age < 0 or age > ticket.ttl_seconds else ticket

    def snapshot(self) -> tuple[Ticket, ...]:
        with _locked_directory(self.root, self.security) as directory:
            _, tickets = self._read(directory)
            return tuple(self._observed(t) for t in tickets.values())

    def reserve(self, *, operation_id: str, owner: str, generation: str, deployment_id: str,
                kind: str, gpu_bytes: int, host_bytes: int, measure: Callable[[], MemorySnapshot],
                headroom_bytes: int = 0, ttl_seconds: float = 300, exclusive: bool = False,
                slot_limit: int | None = None, owned_overlays: Mapping[str, str] | None = None,
                resident_slot: str | None = None, conflicting_resident_slots: tuple[str, ...] = (),
                allow_existing: bool = True, lifecycle_domain: str | None = None,
                lifecycle_model: str | None = None, gpu_guard: bool = True,
                unique_deployment: bool = False, replacing_operation: str | None = None,
                backend_instance: str | None = None, overlay_token: str | None = None,
                embedding_load_parent: tuple[str, str] | None = None) -> Ticket:
        ticket = Ticket(operation_id, owner, generation, deployment_id, kind,
                        "active" if kind == "request" else "reserved", gpu_bytes, host_bytes,
                        self.clock(), ttl_seconds, resident_slot, lifecycle_domain, lifecycle_model, gpu_guard,
                        backend_instance, overlay_token)
        _validate_ticket(ticket)
        if type(allow_existing) is not bool:
            raise ValueError("allow_existing must be a boolean")
        if kind == "unload":
            raise ValueError("use begin_unload for a release operation")
        if (type(unique_deployment) is not bool or unique_deployment and kind != "load"
                or replacing_operation is not None and (not unique_deployment or not isinstance(replacing_operation, str))):
            raise ValueError("unique deployment/replacement requires a model load")
        if not gpu_guard and (headroom_bytes or exclusive or conflicting_resident_slots):
            raise ValueError("coordination-only requests cannot claim GPU policies")
        if not _nonnegative(headroom_bytes) or (slot_limit is not None and (
                type(slot_limit) is not int or slot_limit < 1)):
            raise ValueError("invalid admission limits")
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            prior = tickets.get(operation_id)
            if prior is not None:
                if not allow_existing:
                    raise AdmissionError("operation_conflict", "operation ID is already owned")
                # A replay must match the complete operation, not just its ID.
                if replace(prior, heartbeat_monotonic=ticket.heartbeat_monotonic) != ticket:
                    raise AdmissionError("operation_conflict", "operation ID reused with different parameters")
                if self._observed(prior).phase == "unknown":
                    raise AdmissionError("resource_unknown", "operation requires reconciliation")
                return prior
            observed = tuple(self._observed(t) for t in tickets.values())
            delegated_token = None
            if embedding_load_parent is not None:
                if (kind != "load" or owner != "kiron-embeddings" or not gpu_guard
                        or type(embedding_load_parent) is not tuple or len(embedding_load_parent) != 2
                        or any(type(value) is not str or len(value) != 32
                               or any(char not in "0123456789abcdef" for char in value)
                               for value in embedding_load_parent)):
                    raise ValueError("invalid embedding load parent")
                parent_id, parent_token = embedding_load_parent
                parent = next((item for item in observed if item.operation_id == parent_id), None)
                if (parent is None or parent.owner != "kiron-proxy-lifecycle"
                        or parent.deployment_id != "native-lifecycle:Embedding-Service"
                        or parent.kind != "request" or parent.phase != "active"
                        or parent.overlay_token != parent_token or not parent.gpu_guard):
                    raise AdmissionError("resource_conflict", "embedding load parent is not active")
                # The proxy holds the global loading marker while the serial
                # worker reserves the measured model. Only this live parent
                # authorizes a cross-process handoff of that exact marker.
                delegated_token = parent_token
            if unique_deployment:
                existing = [t for t in observed if t.deployment_id == deployment_id and t.kind in {"load", "unload"}]
                if replacing_operation is None:
                    if existing:
                        raise AdmissionError("resource_conflict", "deployment residency is already owned")
                elif (len(existing) != 1 or existing[0].operation_id != replacing_operation
                      or existing[0].owner != owner or existing[0].generation != generation
                      or existing[0].kind != "load" or existing[0].phase != "resident"):
                    raise AdmissionError("resource_conflict", "replacement does not own a confirmed resident")
            scoped = [t for t in observed if lifecycle_domain is not None
                      and t.lifecycle_domain == lifecycle_domain]
            if any(t.phase == "unknown" for t in scoped):
                raise AdmissionError("resource_unknown", "unconfirmed lifecycle work blocks admission")
            if any(t.kind == "unload" or (t.kind == "load" and t.phase != "resident") for t in scoped):
                raise AdmissionError("resource_conflict", "lifecycle domain is draining or loading")
            if kind == "load" and any(t.kind == "request" for t in scoped):
                raise AdmissionError("resource_conflict", "active lifecycle requests prevent model loading")
            if kind == "request" and any(t.kind == "request" for t in scoped):
                # Native inference may implicitly load, reconfigure, or unload
                # (keep_alive=0). A domain is deliberately a serialized lane.
                raise AdmissionError("capacity_exhausted", "lifecycle request slot is occupied")
            if kind == "request" and lifecycle_model is None and any(t.kind == "load" for t in scoped):
                raise AdmissionError("resource_conflict", "unidentified native request conflicts with managed residency")
            if gpu_guard:
                self._check_overlays(directory, owned_overlays, delegated_loading_token=delegated_token)
            gpu_observed = tuple(t for t in observed if t.gpu_guard) if gpu_guard else ()
            if any(t.phase == "unknown" for t in gpu_observed):
                raise AdmissionError("resource_unknown", "unconfirmed GPU operation blocks admission")
            if any(t.resident_slot in conflicting_resident_slots and t.kind == "load" for t in gpu_observed):
                raise AdmissionError("resource_conflict", "native request conflicts with a managed runtime slot")
            if resident_slot is not None:
                residents = [t for t in observed if t.resident_slot == resident_slot and t.kind == "load"]
                if kind == "load" and residents:
                    raise AdmissionError("resource_conflict", "runtime resident slot is already owned")
                if kind == "load" and any(t.kind == "request" for t in gpu_observed):
                    raise AdmissionError("resource_conflict", "active requests prevent a managed runtime load")
                if kind == "request" and (len(residents) != 1 or residents[0].phase != "resident"
                        or residents[0].generation != generation or residents[0].deployment_id != deployment_id):
                    raise AdmissionError("resource_conflict", "request generation does not own the runtime slot")
            if any(t.kind in {"docling", "training"} and t.phase != "resident" for t in gpu_observed) or (exclusive and gpu_observed):
                raise AdmissionError("resource_conflict", "GPU is reserved by another operation")
            if any(t.deployment_id == deployment_id and (t.phase == "draining" or t.kind == "unload")
                   for t in observed):
                raise AdmissionError("resource_conflict", "deployment is draining")
            if slot_limit is not None and sum(t.kind == "request" and t.deployment_id == deployment_id
                                              for t in observed) >= slot_limit:
                raise AdmissionError("capacity_exhausted", "inference slots are occupied")
            if not gpu_guard:
                tickets[operation_id] = ticket
                self._write(directory, revision, tickets)
                return ticket
            memory = measure()
            if (type(memory) is not MemorySnapshot or not _nonnegative(memory.gpu_free_bytes)
                    or not _nonnegative(memory.host_available_bytes) or not _finite(memory.measured_monotonic)
                    or not 0 <= self.clock() - memory.measured_monotonic <= 2):
                raise AdmissionError("resource_unknown", "fresh memory measurement unavailable")
            pending = [t for t in gpu_observed if t.phase not in {"resident", "draining"}]
            if (gpu_bytes + headroom_bytes + sum(t.gpu_bytes for t in pending) > memory.gpu_free_bytes
                    or host_bytes + sum(t.host_bytes for t in pending) > memory.host_available_bytes):
                raise AdmissionError("resource_exhausted", "insufficient unreserved memory", details={
                    "operation_id": operation_id,
                    "owner": owner,
                    "deployment_id": deployment_id,
                    "gpu_free_bytes": memory.gpu_free_bytes,
                    "gpu_requested_bytes": gpu_bytes,
                    "headroom_bytes": headroom_bytes,
                    "gpu_pending_bytes": sum(t.gpu_bytes for t in pending),
                    "host_available_bytes": memory.host_available_bytes,
                    "host_requested_bytes": host_bytes,
                    "host_pending_bytes": sum(t.host_bytes for t in pending),
                    "measured_monotonic": memory.measured_monotonic,
                    "reservations": [asdict(t) for t in gpu_observed],
                })
            tickets[operation_id] = ticket
            self._write(directory, revision, tickets)
            return ticket

    def _check_overlays(self, directory: int, owned_overlays: Mapping[str, str] | None,
                        *, delegated_loading_token: str | None = None):
        owned_overlays = dict(owned_overlays or {})
        if any(name not in OVERLAYS for name in owned_overlays):
            raise ValueError("unknown owned overlay")
        for name in OVERLAYS:
            try:
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                if name == "gpu-service-loading.json" and delegated_loading_token is not None:
                    raise AdmissionError("resource_conflict", "embedding load parent marker is missing")
                continue
            if name in owned_overlays and self._owns_overlay(directory, name, owned_overlays[name]):
                continue
            if (name == "gpu-service-loading.json" and delegated_loading_token is not None
                    and self._owns_overlay(directory, name, delegated_loading_token, process_owner=False)):
                continue
            raise AdmissionError("resource_conflict", "GPU lifecycle overlay requires cleanup")

    def activate_docling(self, operation_id: str, *, owner: str, generation: str,
                        owned_overlays: Mapping[str, str] | None = None) -> Ticket:
        """Warm admission acquires exclusivity without counting resident bytes twice."""
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            current = self._owned(tickets, operation_id, owner, generation)
            if current.kind != "docling" or self._observed(current).phase == "unknown":
                raise AdmissionError("resource_unknown", "Docling residency requires reconciliation")
            self._check_overlays(directory, owned_overlays)
            if current.phase in {"active", "reserved"}:
                return current
            if current.phase != "resident":
                raise AdmissionError("operation_conflict", "Docling cannot activate this state")
            if any(key != operation_id and t.gpu_guard for key, t in tickets.items()):
                raise AdmissionError("resource_conflict", "another GPU operation prevents Docling activation")
            updated = replace(current, phase="active", heartbeat_monotonic=self.clock())
            tickets[operation_id] = updated
            self._write(directory, revision, tickets)
            return updated

    def _owns_overlay(self, directory: int, name: str, token: str, *, process_owner: bool = True) -> bool:
        if type(token) is not str or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            return False
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            _check_file(fd, self.security)
            if process_owner and os.fstat(fd).st_uid != os.geteuid():
                return False
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(4097)
            if len(raw) > 4096:
                return False
            value = json.loads(raw)
            expected_kind = {"docling-vram-startup.json": "startup", "docling-vram-shutdown.json": "shutdown",
                             "gpu-service-loading.json": "gpu_service_loading"}[name]
            deadline, ttl = value.get("deadline_monotonic"), value.get("ttl_s")
            return (value.get("token") == token and type(value.get("pid")) is int and value["pid"] > 0
                    and (not process_owner or value["pid"] == os.getpid())
                    and value.get("kind") == expected_kind and _finite(deadline) and _finite(ttl)
                    and 0 < ttl <= 900 and 0 < deadline - self.clock() <= ttl)
        except (ValueError, TypeError, AttributeError):
            return False
        finally:
            os.close(fd)

    def begin_unload(self, operation_id: str, *, owner: str, generation: str,
                     deployment_id: str, ttl_seconds: float = 60, require_resident: bool = True,
                     allow_existing: bool = True, lifecycle_domain: str | None = None,
                     lifecycle_model: str | None = None, backend_instance: str | None = None,
                     overlay_token: str | None = None) -> Ticket:
        """Atomically stop new requests before waiting for existing work.

        Release operations remain possible under low memory, overlays and stale
        observations. They cannot release anything until termination is proven.
        An externally managed resident may have no load reservation. Its unload
        ticket only fences new work; it never establishes resource residency.
        """
        if type(require_resident) is not bool or type(allow_existing) is not bool:
            raise ValueError("unload policy flags must be boolean")
        ticket = Ticket(operation_id, owner, generation, deployment_id, "unload", "active",
                        0, 0, self.clock(), ttl_seconds, lifecycle_domain=lifecycle_domain,
                        lifecycle_model=lifecycle_model, backend_instance=backend_instance,
                        overlay_token=overlay_token)
        _validate_ticket(ticket)
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            if any(t.deployment_id == deployment_id and (t.owner != owner or t.generation != generation)
                   for t in tickets.values()):
                raise AdmissionError("operation_conflict", "deployment has foreign or previous-generation work")
            existing = tickets.get(operation_id)
            if existing is not None:
                if not allow_existing or replace(existing, heartbeat_monotonic=ticket.heartbeat_monotonic) != ticket:
                    raise AdmissionError("operation_conflict", "unload operation ID reused")
                return existing
            scoped = [self._observed(t) for t in tickets.values() if lifecycle_domain is not None
                      and t.lifecycle_domain == lifecycle_domain]
            if any(t.kind == "unload" or t.phase == "unknown"
                   or (t.kind == "load" and t.phase != "resident") for t in scoped):
                raise AdmissionError("operation_conflict", "lifecycle domain has unconfirmed or mutating work")
            # A different profile/alias cannot authorize deleting a resident
            # reservation. Unknown native identity conservatively overlaps all.
            if any(t.kind == "load" and (lifecycle_model is None or t.lifecycle_model is None
                       or t.lifecycle_model == lifecycle_model)
                   and (t.deployment_id != deployment_id or t.owner != owner or t.generation != generation)
                   for t in scoped):
                raise AdmissionError("operation_conflict", "lifecycle target has a foreign resident reservation")
            if not require_resident and any(t.deployment_id == deployment_id and t.kind == "unload"
                                            for t in tickets.values()):
                raise AdmissionError("operation_conflict", "deployment already has an unload operation")
            residents = [t for t in tickets.values() if t.owner == owner and t.kind == "load"
                         and t.deployment_id == deployment_id and t.generation == generation]
            if len(residents) > 1 or (require_resident and not residents):
                raise AdmissionError("operation_conflict", "matching resident reservation missing")
            if residents and require_resident:
                resident = residents[0]
                tickets[resident.operation_id] = replace(resident, phase="draining",
                                                         heartbeat_monotonic=self.clock())
            tickets[operation_id] = ticket
            self._write(directory, revision, tickets)
            return ticket

    def confirm_deployment_terminated(self, *, owner: str, generation: str, deployment_id: str) -> None:
        """Only the owning controller may call this after verified process-group end."""
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            remaining = {key: t for key, t in tickets.items() if not (
                t.owner == owner and t.generation == generation and t.deployment_id == deployment_id)}
            if len(remaining) != len(tickets):
                self._write(directory, revision, remaining)

    def begin_backend_recovery(self, operation_id: str, *, backend_instance: str,
                               expected: tuple[Ticket, ...]) -> Ticket:
        """Caller holds the exclusive backend lock; ordinary admission stays fenced."""
        ticket = Ticket(operation_id, "kiron-ollama-recovery", backend_instance, "service:ollama",
                        "unload", "active", 0, 0, self.clock(), 120,
                        lifecycle_domain="ollama", backend_instance=backend_instance)
        _validate_ticket(ticket)
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            current = tuple(self._observed(t) for t in tickets.values()
                            if t.lifecycle_domain == "ollama" or t.backend_instance is not None)
            if set(current) != set(expected) or operation_id in tickets:
                raise AdmissionError("operation_conflict", "Ollama-Zustand hat sich geändert")
            if not any(t.phase == "unknown" for t in current):
                raise AdmissionError("operation_conflict", "Keine unbestätigte Ollama-Operation vorhanden")
            if any(t.phase not in {"unknown", "resident"} for t in current):
                raise AdmissionError("ollama_busy", "Ollama hat noch aktive Operationen")
            if any(t.backend_instance is None or t.backend_instance.split("@", 1)[0]
                   != backend_instance.split("@", 1)[0] for t in current):
                raise AdmissionError("ollama_identity_unknown", "Reservierung gehört zu einer anderen oder unbekannten Ollama-Instanz")
            tickets[operation_id] = ticket
            self._write(directory, revision, tickets)
        return ticket

    def confirm_backend_terminated(self, fence: Ticket, expected: tuple[Ticket, ...]) -> None:
        """After Docker stop AND empty-cgroup proof, remove only the captured identities.

        The recovery fence remains until the new backend passed its health check.
        All epochs of this exact container ended when its process group emptied.
        """
        if (fence.owner != "kiron-ollama-recovery" or fence.kind != "unload"
                or fence.lifecycle_domain != "ollama" or fence.backend_instance != fence.generation):
            raise AdmissionError("operation_conflict", "Keine Ollama-Wiederherstellungssperre")
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            self._owned(tickets, fence.operation_id, fence.owner, fence.generation)
            for prior in expected:
                current = tickets.get(prior.operation_id)
                if current is not None and (current.owner, current.generation, current.backend_instance) != (
                        prior.owner, prior.generation, prior.backend_instance):
                    raise AdmissionError("operation_conflict", "Reservierungsidentität hat sich geändert")
                if (prior.backend_instance is None or prior.backend_instance.split("@", 1)[0]
                        != fence.backend_instance.split("@", 1)[0]):
                    raise AdmissionError("operation_conflict", "Fremde Ollama-Reservierung")
            for prior in expected:
                if prior.operation_id in tickets:
                    if prior.overlay_token:
                        self._clear_terminated_overlay(directory, "gpu-service-loading.json", prior.overlay_token)
                    del tickets[prior.operation_id]
            self._write(directory, revision, tickets)

    def transition(self, operation_id: str, *, owner: str, expected_generation: str,
                   phase: str, generation: str | None = None) -> Ticket:
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            current = self._owned(tickets, operation_id, owner, expected_generation)
            if phase not in {"resident", "unknown"}:
                raise ValueError("transition must confirm residency or mark unknown")
            if phase == "resident" and self._observed(current).phase == "unknown":
                raise AdmissionError("resource_unknown", "unknown operation requires confirmed termination")
            if phase == "resident" and current.kind not in {"load", "docling"}:
                raise ValueError("only model/container loads establish residency")
            updated = replace(current, phase=phase, generation=generation or current.generation,
                              heartbeat_monotonic=self.clock())
            _validate_ticket(updated)
            tickets[operation_id] = updated
            self._write(directory, revision, tickets)
            return updated

    def heartbeat(self, operation_id: str, *, owner: str, generation: str) -> Ticket:
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            current = self._owned(tickets, operation_id, owner, generation)
            if self._observed(current).phase == "unknown":
                raise AdmissionError("resource_unknown", "heartbeat cannot reconcile unknown backend state")
            updated = replace(current, heartbeat_monotonic=self.clock())
            tickets[operation_id] = updated
            self._write(directory, revision, tickets)
            return updated

    def reject_load(self, operation_id: str, *, owner: str, generation: str, deployment_id: str) -> None:
        """Controller proof that this exact Prism load never attempted a spawn.

        This is not a transport-error recovery shortcut. A controller must own
        the operation and prove it never launched work before calling it.
        """
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            if operation_id not in tickets:
                return
            current = self._owned(tickets, operation_id, owner, generation)
            if (current.deployment_id != deployment_id or current.kind != "load"
                    or current.resident_slot != "prism" or current.phase not in {"reserved", "unknown"}):
                raise AdmissionError("operation_conflict", "operation is not an unstarted Prism load")
            del tickets[operation_id]
            self._write(directory, revision, tickets)

    def release(self, operation_id: str, *, owner: str, generation: str, confirmed_terminated: bool,
                owned_overlays: Mapping[str, str] | None = None) -> None:
        overlays = dict(owned_overlays or {})
        if overlays and (confirmed_terminated is not True
                         or set(overlays) != {"gpu-service-loading.json"}):
            raise ValueError("overlay cleanup requires confirmed GPU-service termination")
        with _locked_directory(self.root, self.security) as directory:
            revision, tickets = self._read(directory)
            if operation_id not in tickets:
                return
            current = self._owned(tickets, operation_id, owner, generation)
            if confirmed_terminated is not True:
                tickets[operation_id] = replace(current, phase="unknown")
            else:
                # Writers hold this same outer lock. Remove the proven operation's
                # marker first: a crash before the state write leaves a blocking
                # ticket, never permission to start overlapping GPU work.
                for name, token in overlays.items():
                    self._clear_terminated_overlay(directory, name, token)
                del tickets[operation_id]
            self._write(directory, revision, tickets)

    def _clear_terminated_overlay(self, directory: int, name: str, token: str) -> None:
        if (type(token) is not str or len(token) != 32
                or any(c not in "0123456789abcdef" for c in token)):
            raise AdmissionError("operation_conflict", "invalid overlay cleanup identity")
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except FileNotFoundError:
            return  # A repeated cleanup is harmless.
        try:
            _check_file(fd, self.security)
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(4097)
            try:
                value = json.loads(raw) if len(raw) <= 4096 else None
            except (ValueError, UnicodeError):
                value = None
            if (not isinstance(value, dict) or value.get("token") != token
                    or value.get("kind") != "gpu_service_loading"):
                raise AdmissionError("operation_conflict", "GPU-service marker belongs to another operation")
            current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            opened = os.fstat(fd)
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise AdmissionError("operation_conflict", "GPU-service marker changed during cleanup")
            # The caller's end evidence, not TTL or a process-ID heuristic,
            # authorizes this token-bound cleanup, including after proxy restart.
            os.unlink(name, dir_fd=directory)
        finally:
            os.close(fd)

    @staticmethod
    def _owned(tickets: dict[str, Ticket], operation_id: str, owner: str, generation: str) -> Ticket:
        ticket = tickets.get(operation_id)
        if ticket is None or ticket.owner != owner or ticket.generation != generation:
            raise AdmissionError("operation_conflict", "operation owner or generation changed")
        return ticket
