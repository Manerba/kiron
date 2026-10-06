"""Prism lifecycle owner; resolver and shared GPU admission are injected boundaries."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass
import os
import re
import time
from uuid import uuid4

from kiron_common.local_inference import RuntimeGeneration

from kiron_common.prism_runtime_policy import PolicyError
from process import NativeChild, require_free_port


class ControlError(Exception):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


@dataclass(frozen=True)
class Command:
    deployment_id: str
    snapshot_revision: str
    expected_generation: RuntimeGeneration
    operation_id: str

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != {
                "deployment_id", "snapshot_revision", "expected_generation", "operation_id"}:
            raise ControlError("invalid_request", 400)
        for name in ("deployment_id", "snapshot_revision", "operation_id"):
            if not isinstance(value[name], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value[name]):
                raise ControlError("invalid_request", 400)
        generation = value["expected_generation"]
        if not isinstance(generation, dict) or set(generation) != {"boot_id", "process_id"}:
            raise ControlError("invalid_request", 400)
        try:
            return cls(**{**value, "expected_generation": RuntimeGeneration(**generation)})
        except (ValueError, TypeError):
            raise ControlError("invalid_request", 400) from None


class Controller:
    """One mutation and one native model slot. No independent GPU lock lives here.

    resolver.snapshot() returns the current immutable ResolverSnapshot.
    admission.validate(op, deployment, generation, action), loaded(op, deployment,
    new_generation), drain(op, deployment, generation, deadline_monotonic), and
    terminated(op, deployment, generation), heartbeat(op, deployment, generation),
    rejected(op, deployment_id, expected_generation) for proven pre-spawn failures
    are async and must fail closed.
    """
    def __init__(self, policy, resolver, admission, *, child_factory=NativeChild,
                 port_check=require_free_port):
        self.policy, self.resolver, self.admission = policy, resolver, admission
        self.child_factory, self.port_check = child_factory, port_check
        self.generation = RuntimeGeneration(uuid4().hex, None)
        self.state, self.error = "unloaded", None
        self.deployment = self.child = self.alias = None
        self.snapshot_revision = None
        self._resident_operation = None
        self._admitted, self._closing = False, False
        self._active = None
        self._cleanup_task = None
        self._slots = None
        self._operations = OrderedDict()

    def observation(self):
        return {"provider": "prism", "generation": asdict(self.generation),
                "state": self.state, "health": "healthy" if self.error is None else "unhealthy",
                "deployment_id": self.deployment.id if self.deployment else None,
                "configuration_fingerprint": (self.deployment.configuration_fingerprint
                                              if self.deployment else None),
                "snapshot_revision": self.snapshot_revision, "backend_model": self.alias,
                "observed_at": time.time(), "error": self.error,
                **(self._slots or {"active_requests": 0 if self.child is None else None,
                                   "slot_task_id": None, "slots_observed_at": None})}

    async def health(self):
        child, generation = self.child, self.generation
        self._slots = None
        if child is not None and self._active is None:
            if not child.alive():
                self.state, self.error = "unknown", "process_exited"
                self._active = asyncio.create_task(self._stop_and_release(self._resident_operation))
                try:
                    await asyncio.shield(self._active)
                except Exception:
                    pass  # Unknown is intentionally retained after an unconfirmed cleanup.
                finally:
                    self._active = None
            elif self._admitted:
                ready = await child.ready(self.alias)
                if ready and child is self.child and generation == self.generation and self._active is None:
                    try:
                        await asyncio.wait_for(self.admission.heartbeat(self._resident_operation,
                                               self.deployment, generation), self.policy.health_timeout)
                    except Exception:
                        ready = False
                if child is self.child and generation == self.generation and self._active is None:
                    self.state = "loaded" if ready else "unknown"
                    self.error = None if ready else "provider_unavailable"
                    if ready:
                        slots = await child.slots()
                        if child is self.child and generation == self.generation and self._active is None:
                            self._slots = slots
        return self.observation()

    async def mutate(self, action, command):
        if action not in {"load", "unload"} or self._closing:
            raise ControlError("provider_unavailable", 503)
        signature = (action, command)
        previous = self._operations.get(command.operation_id)
        if previous:
            if previous[0] != signature:
                raise ControlError("operation_conflict")
            await asyncio.shield(previous[1])
            return await self.health()  # Never replay a stale historical 'loaded' observation.
        if self._active is not None:
            raise ControlError("resource_busy")
        if command.expected_generation != self.generation:
            raise ControlError("generation_conflict")
        task = asyncio.create_task(self._execute(action, command))
        self._active = task
        self._operations[command.operation_id] = (signature, task)
        while len(self._operations) > 128:
            self._operations.popitem(last=False)
        def finished(done):
            if self._active is done:
                self._active = None
            if not done.cancelled():
                done.exception()  # Observe failures even when the HTTP requester disconnected.
        task.add_done_callback(finished)
        await asyncio.shield(task)
        return self.observation()

    async def _snapshot(self, command):
        snapshot = await self.resolver.snapshot()
        if snapshot.revision != command.snapshot_revision:
            raise ControlError("snapshot_conflict")
        return snapshot

    def _profile(self, deployment):
        if (deployment.provider.value != "prism" or deployment.loader.value != "prism_gguf"
                or deployment.artifact_identity.format.value != "gguf" or deployment.resource_profile is None):
            raise ControlError("unsupported_capability", 400)
        actual = deployment.resource_profile
        profile = self.policy.profiles.get(actual.id)
        artifact = deployment.artifact_identity
        if (profile is None or actual.context_tokens != profile.context or actual.batch_size != profile.batch
                or actual.ubatch_size != profile.batch or actual.parallel_slots != 1 or actual.projector_on_gpu
                or actual.gpu_layers != profile.gpu_layers or actual.threads != profile.threads
                or actual.gpu_memory_bytes != profile.gpu_memory_bytes
                or actual.host_memory_bytes != profile.host_memory_bytes
                or actual.memory_headroom_bytes != profile.memory_headroom_bytes
                or artifact.sha256 != profile.model_sha256
                or (artifact.projector is not None and artifact.projector.sha256 != profile.projector_sha256)):
            raise ControlError("unsupported_profile", 400)
        return profile

    async def _open_artifact(self, *args):
        # Hashing immutable multi-GB files must not block the control event loop.
        task = asyncio.create_task(asyncio.to_thread(self.policy.open_artifact, *args))
        try:
            return await asyncio.shield(task)
        except BaseException:
            def discard(done):
                if not done.cancelled() and done.exception() is None:
                    os.close(done.result())
            task.add_done_callback(discard)
            raise

    async def _verify_metadata(self, profile, descriptors):
        copies = [os.dup(fd) for fd in descriptors]
        def verify():
            try:
                self.policy.verify_metadata(profile, copies[0], copies[1] if len(copies) == 2 else None)
            finally:
                for fd in copies:
                    os.close(fd)
        task = asyncio.create_task(asyncio.to_thread(verify))
        try:
            await asyncio.shield(task)
        except BaseException:
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            raise

    async def _execute(self, action, command):
        try:
            if action == "load":
                await self._load(command)
            else:
                await self._unload(command)
        except ControlError:
            raise
        except TimeoutError:
            raise ControlError("timeout", 504) from None
        except PolicyError:
            raise ControlError("model_unavailable", 503) from None
        except Exception:
            raise ControlError("provider_unavailable", 503) from None

    async def _load(self, command):
        descriptors = []
        deployment_id = command.deployment_id
        spawn_attempted = False
        try:
            async with asyncio.timeout(self.policy.startup_timeout):
                snapshot = await self._snapshot(command)
                deployment = snapshot.resolve_deployment(command.deployment_id)
                deployment_id = deployment.id
                profile = self._profile(deployment)
                if self.child is not None or self.deployment is not None:
                    if self.state == "loaded" and self.deployment == deployment:
                        return
                    raise ControlError("resource_busy")
                try:
                    self.port_check(self.policy.port)
                except OSError:
                    raise ControlError("backend_port_in_use") from None
                await asyncio.to_thread(self.policy.verify_bundle)
                artifact = deployment.artifact_identity
                descriptors.append(await self._open_artifact(deployment.reference, artifact.sha256, artifact.size_bytes))
                if artifact.projector:
                    projector = artifact.projector
                    descriptors.append(await self._open_artifact(projector.reference, projector.sha256, projector.size_bytes))
                await self._verify_metadata(profile, descriptors)
                # Detect a changed registration while hashing before any admission or spawn.
                current = await self._snapshot(command)
                if current.resolve_deployment(command.deployment_id) != deployment:
                    raise ControlError("snapshot_conflict")
                await self.admission.validate(command.operation_id, deployment, self.generation, "load")
                self.generation = RuntimeGeneration(self.generation.boot_id, uuid4().hex)
                self.deployment, self.snapshot_revision = deployment, snapshot.revision
                self._resident_operation = command.operation_id
                self.alias = f"kiron-prism-{self.generation.process_id}"
                self.state, self.error = "loading", None
                spawn_attempted = True
                self.child = self.child_factory(self.policy.command(profile, self.alias, descriptors[0],
                                                descriptors[1] if len(descriptors) == 2 else None),
                                                self.policy.environment(), descriptors, self.policy)
                while self.child.alive():
                    if await self.child.ready(self.alias):
                        await self.admission.loaded(command.operation_id, deployment, self.generation)
                        self._admitted, self.state, self.error = True, "loaded", None
                        return
                    await asyncio.sleep(0.05)
                raise ControlError("process_exited", 503)
        except BaseException:
            if self.deployment is not None and self._resident_operation == command.operation_id:
                await asyncio.shield(self._stop_and_release(command.operation_id))
            elif not spawn_attempted:
                # A received/serialized controller task can prove that it never
                # attempted work. The proxy cannot infer this from a timeout or
                # connection failure; only this callback clears that exact load.
                await asyncio.shield(self.admission.rejected(command.operation_id, deployment_id,
                                                            command.expected_generation))
            raise
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    async def _unload(self, command):
        async with asyncio.timeout(self.policy.drain_timeout):
            snapshot = await self._snapshot(command)
            deployment = snapshot.resolve_deployment(command.deployment_id)
            if self.deployment is None:
                return
            if self.deployment.id != deployment.id:
                raise ControlError("model_conflict")
            await self.admission.validate(command.operation_id, self.deployment, self.generation, "unload")
            await self.admission.drain(command.operation_id, self.deployment, self.generation,
                                       time.monotonic() + self.policy.drain_timeout)
        self.state = "unloading"
        await asyncio.shield(self._stop_and_release(command.operation_id))

    async def _stop_and_release(self, operation_id):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup(operation_id))
        task = self._cleanup_task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._cleanup_task = None

    async def _cleanup(self, operation_id):
        self.state = "unknown"
        self._slots = None
        try:
            if self.child is not None and not await self.child.stop():
                raise ControlError("process_cleanup_unconfirmed", 503)
            self.child = None
            if self.deployment is not None:
                await asyncio.wait_for(self.admission.terminated(operation_id, self.deployment, self.generation),
                                       self.policy.kill_timeout)
            self.deployment = self.alias = None
            self._admitted = False
            self.state, self.error = "unloaded", None
        except BaseException:
            self.state, self.error = "unknown", "cleanup_unconfirmed"
            raise

    async def close(self):
        self._closing = True
        active = self._active
        if active is not None:
            active.cancel()
            try:
                await active
            except (Exception, asyncio.CancelledError):
                pass
        if self.deployment is not None:
            try:
                await asyncio.wait_for(self.admission.drain(self._resident_operation, self.deployment,
                                        self.generation, time.monotonic() + self.policy.drain_timeout),
                                       self.policy.drain_timeout)
            except Exception:
                pass  # Service shutdown still terminates its own process tree after bounded drain.
            await self._stop_and_release(self._resident_operation)
