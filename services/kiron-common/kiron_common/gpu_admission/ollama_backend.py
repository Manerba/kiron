"""Fixed Ollama container identity and a cross-process recovery barrier.

Every backend operation holds a shared lock until its transport is closed.
Recovery holds the exclusive lock through stop, end proof, cleanup and start.
Lock expiry and model residency are never request-end evidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re

from .store import AdmissionError, _check_file, _open_root


CONTAINER = "kiron-ollama"
_FORMAT = ('{"id":{{json .Id}},"started_at":{{json .State.StartedAt}},'
           '"running":{{json .State.Running}},"status":{{json .State.Status}},'
           '"pid":{{json .State.Pid}}}')


def valid_instance(value):
    return isinstance(value, str) and re.fullmatch(
        r"[0-9a-f]{64}@\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z", value) is not None


async def command(arguments, timeout=5):
    process = await asyncio.create_subprocess_exec("/usr/bin/docker", *arguments,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    pending = asyncio.create_task(process.communicate())
    try:
        done, _ = await asyncio.wait({pending}, timeout=timeout)
        if not done:
            raise AdmissionError("ollama_control_timeout", "Ollama-Verwaltung antwortet nicht rechtzeitig")
        stdout, stderr = pending.result()
        if process.returncode or len(stdout) + len(stderr) > 65536:
            raise AdmissionError("ollama_control_failed", "Ollama-Container konnte nicht verwaltet werden")
        return stdout.decode("utf-8", errors="strict")
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await asyncio.wait({pending}, timeout=2)
        if not pending.done():
            pending.cancel()
        pending.add_done_callback(lambda t: None if t.cancelled() else t.exception())


@dataclass(frozen=True)
class BackendState:
    container_id: str
    started_at: str
    running: bool
    status: str
    pid: int

    @property
    def instance(self):
        return self.container_id + "@" + self.started_at


class OllamaBackend:
    def __init__(self, *, run=None, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
        self.run = run or command
        self.proc_root, self.cgroup_root = proc_root, cgroup_root

    async def inspect(self, target=CONTAINER):
        try:
            value = json.loads(await self.run(["inspect", "--format", _FORMAT, target]))
            state = BackendState(value["id"], value["started_at"], value["running"], value["status"], value["pid"])
            if (not valid_instance(state.instance) or type(state.running) is not bool
                    or type(state.pid) is not int or state.pid < 0
                    or state.status not in {"running", "exited", "dead", "paused", "restarting", "created"}):
                raise ValueError("invalid container identity")
            return state
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AdmissionError("ollama_identity_unknown", "Ollama-Containeridentität ist unbekannt") from exc

    def process_group(self, state):
        # Resolve the *observed* host PID, and constrain it to this container.
        if not state.running or state.pid <= 0:
            return self.cgroup_root / "system.slice" / f"docker-{state.container_id}.scope"
        try:
            entries = (self.proc_root / str(state.pid) / "cgroup").read_text().splitlines()
            groups = [line[3:] for line in entries if line.startswith("0::")]
            expected = f"/system.slice/docker-{state.container_id}.scope"
            if groups != [expected]:
                raise ValueError("unexpected container process group")
            return self.cgroup_root / expected.lstrip("/")
        except (OSError, ValueError) as exc:
            raise AdmissionError("ollama_process_unknown", "Ollama-Prozessgruppe ist nicht bestätigt") from exc

    async def stop_and_verify(self, state):
        group = self.process_group(state)
        if state.running:
            await self.run(["stop", "--time", "10", state.container_id], timeout=20)
        stopped = await self.inspect(state.container_id)
        if (stopped.instance != state.instance or stopped.running or stopped.pid != 0
                or stopped.status not in {"exited", "dead"}):
            raise AdmissionError("ollama_stop_unconfirmed", "Ollama-Stopp ist nicht bestätigt")
        try:
            fields = dict(line.split() for line in (group / "cgroup.events").read_text().splitlines())
        except FileNotFoundError:
            fields = {"populated": "0"}
        except (OSError, ValueError) as exc:
            raise AdmissionError("ollama_stop_unconfirmed", "Ollama-Prozessende ist unbekannt") from exc
        if fields.get("populated") != "0":
            raise AdmissionError("ollama_stop_unconfirmed", "Ollama-Prozesse laufen noch")

    async def start_and_verify(self, state):
        await self.run(["start", state.container_id], timeout=20)
        current = await self.inspect()
        if (current.container_id != state.container_id or not current.running
                or current.status != "running" or current.pid <= 0 or current.instance == state.instance):
            raise AdmissionError("ollama_start_unconfirmed", "Ollama-Neustart ist nicht bestätigt")
        return current


class BackendLock:
    def __init__(self, store, *, exclusive=False):
        self.store, self.exclusive, self.fd = store, exclusive, None

    def acquire(self):
        directory = None
        try:
            directory = _open_root(self.store.root, self.store.security)
            try:
                self.fd = os.open(".ollama-backend.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CREAT | os.O_EXCL,
                                  0o660, dir_fd=directory)
                os.fchmod(self.fd, 0o660)
            except FileExistsError:
                self.fd = os.open(".ollama-backend.lock", os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            _check_file(self.fd, self.store.security)
            fcntl.flock(self.fd, (fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise AdmissionError("ollama_busy", "Ollama wird noch verwendet oder wiederhergestellt") from exc
        except (OSError, AdmissionError) as exc:
            self.close()
            if isinstance(exc, AdmissionError):
                raise
            raise AdmissionError("resource_unknown", "Ollama-Sperre ist nicht erreichbar") from exc
        finally:
            if directory is not None:
                os.close(directory)
        return self

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class OllamaBackendSession:
    def __init__(self, store, *, backend=None):
        self.lock = BackendLock(store)
        self.backend = backend or OllamaBackend()
        self.instance = None

    async def __aenter__(self):
        self.lock.acquire()
        try:
            state = await self.backend.inspect()
            if not state.running or state.status != "running" or state.pid <= 0:
                raise AdmissionError("ollama_unavailable", "Ollama läuft nicht")
            self.instance = state.instance
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.close()

    async def __aexit__(self, *exc):
        self.close()
