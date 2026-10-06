"""Request lifetime, resolver and admission owner for local inference providers."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import time

from starlette.responses import StreamingResponse

from kiron_common.gpu_admission import AdmissionError
from kiron_common.gpu_admission.ollama_backend import OllamaBackendSession
from kiron_common.gpu_admission.ollama_lifecycle import (
    OLLAMA_DOMAIN, OllamaLifecycleOperation, drain_ollama_requests,
)
from kiron_common.local_inference import (
    CapabilityName, EmbeddingRequest, EmbeddingResult, ErrorCode, EventIndexState, EventKind, InferenceResult, LifecycleResult, LocalInferenceError,
    ModelLifecycleOperation, ProviderHealth, ProviderObservation, RequestContext, RuntimeFailure,
)
from kiron_common.model_state import RuntimeState
from kiron_common.model_catalog import BackendType

from provider_transport import RequestRejected, failure, with_context
from provider_features import validate_features
from provider_embeddings import validate_embedding_request


logger = logging.getLogger(__name__)
OWNER = "kiron-proxy"


def generation_key(generation):
    return json.dumps([generation.boot_id, generation.process_id], separators=(",", ":"))


def check_context(context):
    if context.cancellation.is_set():
        raise failure(ErrorCode.CANCELLED, "Request cancelled")
    if context.deadline_monotonic <= time.monotonic():
        raise failure(ErrorCode.TIMEOUT, "Request deadline exceeded")


def chat_profile(capability, deployment):
    """Require at least one executable text request before publishing a model."""
    constraints = capability.constraints
    roles, limit, default = (constraints.get(name) for name in
                             ("roles", "max_output_tokens", "default_max_output_tokens"))
    if (roles is None or not any(roles.accepts(role) for role in ("system", "developer", "user", "assistant"))
            or limit is None or default is None or default.allowed_values is None
            or len(default.allowed_values) != 1):
        raise failure(ErrorCode.INVALID_CONFIGURATION, "Model has no executable text profile")
    budget = default.allowed_values[0]
    if (type(budget) is not int or not 1 <= budget <= 100000 or not limit.accepts(budget)
            or (deployment.resource_profile is not None and budget > deployment.resource_profile.context_tokens)):
        raise failure(ErrorCode.INVALID_CONFIGURATION, "Model default output budget is invalid")
    return constraints, budget


class RuntimeService:
    def __init__(self, *, resolver, providers, admission, measure, timeouts, measurement_close=None):
        self.resolver = resolver
        self.providers = dict(providers)
        self.admission, self.measure, self.timeouts = admission, measure, timeouts
        self._measurement_close = measurement_close
        self._locks = {key: asyncio.Lock() for key in self.providers}
        self._closed = False
        self._operations = set()

    def _provider(self, deployment):
        try:
            return self.providers[deployment.provider]
        except KeyError as exc:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Model provider is not configured") from exc

    async def observations(self, context):
        async def observe(key, provider):
            try:
                return key, await provider.health(context)
            except Exception as exc:
                error = exc.failure if isinstance(exc, LocalInferenceError) else RuntimeFailure(
                    ErrorCode.PROVIDER_UNAVAILABLE, "Provider health is unavailable")
                return key, ProviderObservation(key, None, datetime.now(timezone.utc), ProviderHealth.UNAVAILABLE, error=error)
        return dict(await asyncio.gather(*(observe(key, value) for key, value in self.providers.items())))

    async def resolve(self, public_model_id, context=None):
        pending = self.resolver.snapshot()
        snapshot = await pending if context is None else await with_context(pending, context)
        return snapshot.resolve(public_model_id)

    async def _addressable(self, models, context):
        """Join one frozen resolver snapshot to independent provider observations."""
        provider_ids = {model.deployment.provider for model in models}
        async def inspect(provider_id):
            provider = self.providers.get(provider_id)
            if provider is None:
                return provider_id, None
            try:
                observation = await with_context(provider.health(context), context)
                if observation.health not in {ProviderHealth.AVAILABLE, ProviderHealth.STARTABLE}:
                    return provider_id, None
                discovery = await with_context(provider.discover(context), context)
                return provider_id, None if discovery.error else (provider, observation, discovery)
            except Exception:
                check_context(context)
                return provider_id, None
        inspected = dict(await asyncio.gather(*(inspect(key) for key in provider_ids)))
        addressable = {}
        for model in models:
            deployment = model.deployment
            source = inspected[deployment.provider]
            if source is None or model.created is None:
                continue
            provider, observation, discovery = source
            identity = deployment.artifact_identity.content_key
            if identity is None or not any(item.installed and item.reference == deployment.reference
                    and item.artifact_identity.content_key == identity for item in discovery.models):
                continue
            resident = observation.models.get(deployment.id)
            if deployment.resource_profile is None and not (
                    resident is not None and resident.state is RuntimeState.LOADED
                    and resident.configuration_fingerprint == deployment.configuration_fingerprint):
                continue
            try:
                capabilities = await with_context(provider.capabilities(deployment), context)
            except Exception:
                check_context(context)
                continue
            chat_ready = capabilities.supports(CapabilityName.CHAT, deployment, provider.implementation)
            if chat_ready:
                try:
                    chat_profile(capabilities.by_name[CapabilityName.CHAT], deployment)
                except LocalInferenceError:
                    chat_ready = False
            if not chat_ready and not capabilities.supports(CapabilityName.EMBEDDINGS, deployment, provider.implementation):
                continue
            if not chat_ready:
                try:
                    self._validate_embedding(EmbeddingRequest(model, ("x",), context), capabilities, provider)
                    if deployment.provider is BackendType.KIRON_EMBEDDINGS:
                        tickets = await self._store("snapshot")
                        if not self._embedding_resident(tickets, deployment.id, observation.generation):
                            continue
                except (LocalInferenceError, ValueError, TypeError):
                    continue
            addressable[model.public_model_id] = {
                "id": model.api_model_id, "object": "model", "created": model.created,
                "owned_by": deployment.provider.value,
            }
        return addressable

    async def public_models(self, context):
        snapshot = await with_context(self.resolver.snapshot(), context)
        registered = tuple(model for model in snapshot.models.values() if model.created is not None)
        if not registered:
            return []
        available = await self._addressable(registered, context)
        if not available:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Registered models are currently unavailable")
        canonical = {row["id"]: row for row in available.values()}
        return [canonical[key] for key in sorted(canonical)]

    async def public_model(self, public_model_id, context):
        model = await self.resolve(public_model_id, context)
        return model, await self.public_model_for(model, context)

    async def public_model_for(self, model, context):
        """Check availability of exactly the definition already validated by the API."""
        available = await self._addressable((model,), context)
        if model.public_model_id not in available:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Registered model is currently unavailable", "model")
        return available[model.public_model_id]

    async def validate_chat(self, parsed, model, context):
        """Validate evidence and parameter semantics before any lifecycle mutation."""
        deployment = model.deployment
        provider = self._provider(deployment)
        capabilities = await with_context(provider.capabilities(deployment), context)
        for name in (CapabilityName.CHAT, *((CapabilityName.STREAMING,) if parsed.stream else ())):
            if not capabilities.supports(name, deployment, provider.implementation):
                raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Requested model capability is not verified", name.value)
        constraints, budget = chat_profile(capabilities.by_name[CapabilityName.CHAT], deployment)
        def check(name, value, parameter=None):
            constraint = constraints.get(name)
            if constraint is None:
                raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "Parameter mapping is not verified", parameter or name)
            if not constraint.accepts(value):
                raise failure(ErrorCode.UNSUPPORTED_VALUE, "Value is outside the verified model profile", parameter or name)
        for index, message in enumerate(parsed.messages):
            check("roles", message.role.value, f"messages[{index}].role")
        for name, value in parsed.explicit_parameters.items():
            if name == "reasoning_effort":
                continue  # Checked separately against CHAT none or REASONING efforts.
            elif name in {"max_tokens", "max_completion_tokens"}:
                check("token_budget", name, name)
                check("max_output_tokens", value, name)
            elif name == "stop":
                for stop in value:
                    check(name, stop)
            else:
                check(name, value)
        request = parsed.to_request(model, context, budget)
        if deployment.resource_profile is not None and request.options.max_output_tokens > deployment.resource_profile.context_tokens:
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Output budget exceeds the measured model context", "max_completion_tokens")
        self._validate_features(request, capabilities, provider)
        provider.validate_request(request, capabilities)
        return request

    @staticmethod
    def _validate_features(request, capabilities, provider):
        validate_features(request, capabilities, provider.implementation)

    async def validate_embedding(self, parsed, model, context):
        provider = self._provider(model.deployment)
        try:
            request = parsed.to_request(model, context)
        except (ValueError, TypeError):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding profile is not addressable", "model") from None
        capabilities = await with_context(provider.capabilities(model.deployment), context)
        self._validate_embedding(request, capabilities, provider)
        return request

    async def validate_response(self, parsed, model, context):
        request = await self.validate_chat(parsed.chat, model, context)
        provider = self._provider(model.deployment)
        capabilities = await with_context(provider.capabilities(model.deployment), context)
        fields = capabilities.by_name[CapabilityName.CHAT].constraints.get("usage_fields")
        if fields is None or fields.allowed_values is None or any(not fields.accepts(name)
                for name in ("cached_input_tokens", "reasoning_output_tokens")):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Responses usage details are not verified", "usage")
        return request

    @staticmethod
    def _validate_embedding(request, capabilities, provider):
        validate_embedding_request(request, capabilities, provider.implementation)
        validate = getattr(provider, "validate_embedding_request", None)
        if validate is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider has no verified embedding adapter", "embeddings")
        validate(request, capabilities)

    @staticmethod
    def _embedding_resident(tickets, deployment_id, generation):
        return generation is not None and any(ticket.kind == "load" and ticket.phase == "resident"
            and ticket.owner == "kiron-embeddings" and ticket.deployment_id == deployment_id
            and ticket.generation == generation_key(generation)
            and (ticket.gpu_bytes > 0 or ticket.host_bytes > 0) for ticket in tickets)

    @asynccontextmanager
    async def _lifecycle_lock(self, provider_id, context):
        lock = self._locks[provider_id]
        backend_session = None
        while True:
            if context.cancellation.is_set():
                raise failure(ErrorCode.CANCELLED, "Request cancelled while waiting for provider")
            remaining = context.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise failure(ErrorCode.TIMEOUT, "Provider lifecycle deadline exceeded")
            try:
                await asyncio.wait_for(lock.acquire(), min(remaining, 0.05))
                break
            except TimeoutError:
                continue
        try:
            if context.cancellation.is_set():
                raise failure(ErrorCode.CANCELLED, "Request cancelled while waiting for provider")
            if context.deadline_monotonic <= time.monotonic():
                raise failure(ErrorCode.TIMEOUT, "Provider lifecycle deadline exceeded")
            if provider_id is BackendType.OLLAMA:
                backend_session = OllamaBackendSession(self.admission)
                await backend_session.__aenter__()
            yield backend_session
        except AdmissionError as exc:
            raise failure(ErrorCode.CONFLICT, str(exc)) from exc
        finally:
            if backend_session is not None:
                backend_session.close()
            lock.release()

    async def start(self, provider_id, context):
        provider = self.providers.get(provider_id)
        if provider is None:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider is not configured")
        async with self._lifecycle_lock(provider_id, context):
            if provider_id is BackendType.OLLAMA:
                return await self._ollama_service_action(provider, context, stop=False)
            return await provider.start(context)

    async def stop(self, provider_id, context):
        provider = self.providers.get(provider_id)
        if provider is None:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider is not configured")
        async with self._lifecycle_lock(provider_id, context):
            if provider_id is BackendType.OLLAMA:
                return await self._ollama_service_action(provider, context, stop=True)
            observation = await provider.health(context)
            if observation.generation is None:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider generation is unknown")
            result = await provider.stop(observation.generation, context)
            if provider_id is BackendType.PRISM:
                # Fixed unit control verifies an empty cgroup; the per-spawn
                # native API token rejects delayed work from an old generation.
                tickets = await self._store("snapshot")
                residents = [t for t in tickets if t.owner == OWNER and t.resident_slot == "prism" and t.kind == "load"]
                for ticket in residents:
                    await self._store("confirm_deployment_terminated", owner=OWNER,
                                      generation=ticket.generation, deployment_id=ticket.deployment_id)
            return result

    async def _ollama_service_action(self, provider, context, *, stop):
        # This fixed controller is optional in OllamaProvider composition.
        # Reject its absence before acquiring a fence or touching the backend.
        if provider.service_control is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama service control is not configured")
        observation = await provider.health(context)
        if observation.generation is None:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama service generation is unknown")
        try:
            async with OllamaLifecycleOperation(store=self.admission, owner=OWNER,
                    operation_id=context.request_id, generation=generation_key(observation.generation),
                    deployment_id="service:ollama", deadline_monotonic=context.deadline_monotonic,
                    cancellation=context.cancellation) as operation:
                check_context(context)
                operation.mark_started()
                result = (await provider.stop(observation.generation, context) if stop
                          else await provider.start(context))
                expected_health = ProviderHealth.STARTABLE if stop else ProviderHealth.AVAILABLE
                if (not isinstance(result, LifecycleResult) or result.operation_id != context.request_id
                        or result.observation.provider is not BackendType.OLLAMA
                        or result.observation.health is not expected_health):
                    raise failure(ErrorCode.PROVIDER_ERROR, "Ollama service transition was not confirmed")
                operation.confirm_end()
                return result
        except asyncio.CancelledError:
            check_context(context)
            raise
        except AdmissionError as exc:
            check_context(context)
            raise failure(ErrorCode.CONFLICT, "Unconfirmed Ollama work prevents service transition") from exc

    async def _store(self, method, *args, **kwargs):
        try:
            return await asyncio.to_thread(getattr(self.admission, method), *args, **kwargs)
        except AdmissionError as exc:
            code = (ErrorCode.OVERLOADED if exc.code in {"resource_exhausted", "capacity_exhausted"}
                    else ErrorCode.PROVIDER_UNAVAILABLE if exc.code == "resource_unknown" else ErrorCode.CONFLICT)
            raise failure(code, str(exc)) from exc

    async def _reserve(self, **kwargs):
        # A filesystem transaction in a worker thread cannot be cancelled. Wait
        # for its result and undo an unstarted reservation on requester abort.
        pending = asyncio.create_task(self._store("reserve", allow_existing=False, **kwargs))
        try:
            return await asyncio.shield(pending)
        except BaseException:
            async def undo_unstarted():
                try:
                    ticket = await pending
                except Exception:
                    return
                await self._store("release", ticket.operation_id, owner=ticket.owner,
                                  generation=ticket.generation, confirmed_terminated=True)
            cleanup = asyncio.create_task(undo_unstarted())
            cleanup.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            await asyncio.shield(cleanup)
            raise

    async def load(self, model, context):
        """Load only this frozen model definition; a concurrent registry edit conflicts."""
        deployment = model.deployment
        provider = self._provider(deployment)
        async with self._lifecycle_lock(deployment.provider, context) as backend_session:
            snapshot = await self.resolver.snapshot()
            snapshot.resolve_deployment(deployment.id, expected_revision=model.snapshot_revision)
            observation = await provider.health(context)
            if observation.health is ProviderHealth.STARTABLE:
                if ModelLifecycleOperation.LOAD not in provider.model_lifecycle_operations:
                    raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider does not support model loading")
                observation = (await provider.start(context)).observation
            if observation.health is not ProviderHealth.AVAILABLE or observation.generation is None:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider is not available for model loading")
            existing = observation.models.get(deployment.id)
            if existing and existing.state is RuntimeState.LOADED:
                if existing.configuration_fingerprint != deployment.configuration_fingerprint:
                    raise failure(ErrorCode.CONFLICT, "Resident model configuration differs")
                return LifecycleResult(context.request_id, observation, False)
            if ModelLifecycleOperation.LOAD not in provider.model_lifecycle_operations:
                raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider does not support model loading")
            if any(item.state in {RuntimeState.LOADING, RuntimeState.LOADED, RuntimeState.UNLOADING, RuntimeState.UNKNOWN}
                   for item in observation.models.values()):
                raise failure(ErrorCode.CONFLICT, "Provider model slot is occupied or unconfirmed")
            profile = deployment.resource_profile
            if profile is None:
                raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "A measured load resource profile is required")
            generation = generation_key(observation.generation)
            await self._reserve(operation_id=context.request_id, owner=OWNER, generation=generation,
                              deployment_id=deployment.id, kind="load", **self._lifecycle_identity(deployment),
                              backend_instance=backend_session.instance if backend_session else None,
                              gpu_bytes=profile.gpu_memory_bytes,
                              host_bytes=profile.host_memory_bytes, headroom_bytes=profile.memory_headroom_bytes,
                              measure=self.measure, ttl_seconds=max(self.timeouts.startup + self.timeouts.readiness, 60),
                              resident_slot=deployment.provider.value if deployment.provider is BackendType.PRISM else None)
            try:
                result = await provider.load(deployment, snapshot_revision=model.snapshot_revision,
                                             expected_generation=observation.generation, context=context)
                loaded = result.observation.models.get(deployment.id)
                if (loaded is None or loaded.state is not RuntimeState.LOADED or loaded.generation is None
                        or loaded.configuration_fingerprint != deployment.configuration_fingerprint):
                    raise failure(ErrorCode.PROVIDER_ERROR, "Provider did not confirm the requested model")
                resident_key = generation_key(loaded.generation)
                tickets = await self._store("snapshot")
                ticket = next((t for t in tickets if t.operation_id == context.request_id), None)
                if ticket is None:
                    raise failure(ErrorCode.PROVIDER_ERROR, "Provider lost its load reservation")
                if ticket.phase != "resident":
                    await self._store("transition", context.request_id, owner=OWNER,
                                      expected_generation=generation, phase="resident", generation=resident_key)
                elif ticket.generation != resident_key:
                    raise failure(ErrorCode.CONFLICT, "Resident generation differs from admission")
                return result
            except BaseException:
                # The control request may continue after its requester disconnects.
                # Only controller reconciliation may confirm that load never ran.
                async def uncertain():
                    tickets = await self._store("snapshot")
                    ticket = next((t for t in tickets if t.operation_id == context.request_id), None)
                    if ticket and ticket.phase != "resident":
                        await self._store("release", ticket.operation_id, owner=OWNER,
                                          generation=ticket.generation, confirmed_terminated=False)
                await asyncio.shield(uncertain())
                raise

    async def unload(self, model, context):
        deployment = model.deployment
        provider = self._provider(deployment)
        if ModelLifecycleOperation.UNLOAD not in provider.model_lifecycle_operations:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider does not support model unloading")
        async with self._lifecycle_lock(deployment.provider, context) as backend_session:
            snapshot = await with_context(self.resolver.snapshot(), context)
            if snapshot.resolve_deployment(deployment.id, expected_revision=model.snapshot_revision) != deployment:
                raise failure(ErrorCode.CONFLICT, "Resolved deployment changed")
            observation = await provider.health(context)
            if observation.health is not ProviderHealth.AVAILABLE or observation.generation is None:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider is not available for model unloading")
            current = observation.models.get(deployment.id)
            if self._unloaded(deployment, observation):
                # A health-only no-op cannot prove that a queued request ended.
                # In particular it must not clear previous unknown tickets.
                return LifecycleResult(context.request_id, observation, False)
            if (current is None or current.state is not RuntimeState.LOADED
                    or current.configuration_fingerprint != deployment.configuration_fingerprint):
                raise failure(ErrorCode.CONFLICT, "Resident model identity is unconfirmed")
            generation = generation_key(observation.generation)
            await self._begin_unload(context.request_id, owner=OWNER, generation=generation,
                                     deployment_id=deployment.id, ttl_seconds=self.timeouts.drain + self.timeouts.stop,
                                     backend_instance=backend_session.instance if backend_session else None,
                                     require_resident=deployment.provider is not BackendType.OLLAMA,
                                     **self._lifecycle_identity(deployment))
            mutation_started = False
            try:
                if deployment.provider is BackendType.OLLAMA:
                    await self._drain_requests(deployment, generation, context)
                check_context(context)
                mutation_started = True
                result = await provider.unload(deployment, snapshot_revision=model.snapshot_revision,
                                               expected_generation=observation.generation, context=context)
                if (result.observation.health is not ProviderHealth.AVAILABLE
                        or result.observation.generation != observation.generation
                        or not self._unloaded(deployment, result.observation)):
                    raise failure(ErrorCode.PROVIDER_ERROR, "Provider did not confirm target unload")
            except BaseException:
                # Before provider invocation only this new unload operation is
                # known to be unstarted. Existing load/request tickets are kept.
                # A lost control reply is not evidence of model/request end.
                await asyncio.shield(self._store("release", context.request_id, owner=OWNER,
                                                  generation=generation, confirmed_terminated=not mutation_started))
                raise
            await self._store("confirm_deployment_terminated", owner=OWNER, generation=generation, deployment_id=deployment.id)
            return result

    async def _begin_unload(self, operation_id, **kwargs):
        pending = asyncio.create_task(self._store("begin_unload", operation_id, allow_existing=False, **kwargs))
        try:
            return await asyncio.shield(pending)
        except BaseException:
            async def undo_unstarted():
                try:
                    ticket = await pending
                except Exception:
                    return
                await self._store("release", ticket.operation_id, owner=ticket.owner,
                                  generation=ticket.generation, confirmed_terminated=True)
            cleanup = asyncio.create_task(undo_unstarted())
            cleanup.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            await asyncio.shield(cleanup)
            raise

    @staticmethod
    def _lifecycle_identity(deployment):
        if deployment.provider is not BackendType.OLLAMA:
            return {}
        identity = deployment.artifact_identity
        return {"lifecycle_domain": OLLAMA_DOMAIN,
                "lifecycle_model": identity.sha256 or (
                    identity.manifest_digest.removeprefix("sha256:") if identity.manifest_digest else None)}

    async def _drain_requests(self, deployment, generation, context):
        deadline = min(context.deadline_monotonic, time.monotonic() + self.timeouts.drain)
        if deployment.provider is BackendType.OLLAMA:
            try:
                await drain_ollama_requests(self.admission, deadline_monotonic=deadline,
                                            cancellation=context.cancellation)
            except asyncio.CancelledError:
                check_context(context)
                raise
            except AdmissionError as exc:
                check_context(context)
                raise failure(ErrorCode.CONFLICT, "Unconfirmed Ollama work prevents unloading") from exc
            return
        while True:
            check_context(context)
            requests = [ticket for ticket in await self._store("snapshot")
                        if ticket.deployment_id == deployment.id and ticket.kind == "request"]
            if not requests:
                return
            if any(ticket.generation != generation or ticket.phase == "unknown" for ticket in requests):
                raise failure(ErrorCode.CONFLICT, "Unconfirmed requests prevent model unloading")
            if time.monotonic() >= deadline:
                raise failure(ErrorCode.CONFLICT, "Model request drain deadline exceeded")
            await with_context(asyncio.sleep(min(.05, max(0, deadline - time.monotonic()))), context)

    @staticmethod
    def _unloaded(deployment, observation):
        target = observation.models.get(deployment.id)
        if target is not None:
            return (target.state is RuntimeState.UNLOADED and target.generation == observation.generation
                    and target.configuration_fingerprint == deployment.configuration_fingerprint)
        # The single-child Prism controller represents verified absence by an
        # empty model map. Other providers must explicitly observe the target.
        return deployment.provider is BackendType.PRISM and not observation.models

    async def prepare(self, request, *, streaming=False):
        backend_session = None
        try:
            if request.model.deployment.provider is BackendType.OLLAMA:
                backend_session = OllamaBackendSession(self.admission)
                await backend_session.__aenter__()
            operation = await self._prepare(request, streaming=streaming, backend_session=backend_session)
            operation.backend_session = backend_session
            return operation
        except BaseException as exc:
            if backend_session is not None:
                backend_session.close()
            if isinstance(exc, AdmissionError):
                raise failure(ErrorCode.CONFLICT, str(exc)) from exc
            raise

    async def _prepare(self, request, *, streaming, backend_session):
        if self._closed:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Runtime service is shutting down")
        provider = self._provider(request.model.deployment)
        capabilities = await provider.capabilities(request.model.deployment)
        embedding = isinstance(request, EmbeddingRequest)
        needed = {CapabilityName.EMBEDDINGS if embedding else CapabilityName.CHAT}
        if streaming:
            needed.add(CapabilityName.STREAMING)
        for capability in needed:
            if not capabilities.supports(capability, request.model.deployment, provider.implementation):
                raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, f"Model capability is not verified: {capability.value}")
        if embedding:
            self._validate_embedding(request, capabilities, provider)
        else:
            self._validate_features(request, capabilities, provider)
            provider.validate_request(request, capabilities)
        load_context = replace(request.context, request_id=f"load-{request.context.request_id}")
        loaded = await self.load(request.model, load_context)
        generation = loaded.observation.generation
        if generation is None:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Provider generation is unavailable")
        if embedding and request.model.deployment.provider is BackendType.KIRON_EMBEDDINGS:
            residents = await self._store("snapshot")
            if not self._embedding_resident(residents, request.model.deployment.id, generation):
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Embedding residency lacks measured shared admission")
        profile = request.model.deployment.resource_profile
        await self._reserve(operation_id=request.context.request_id, owner=OWNER,
                          generation=generation_key(generation), deployment_id=request.model.deployment.id,
                          kind="request", **self._lifecycle_identity(request.model.deployment),
                          backend_instance=backend_session.instance if backend_session else None,
                          gpu_bytes=0, host_bytes=0, measure=self.measure,
                          ttl_seconds=self.timeouts.total + self.timeouts.drain,
                          slot_limit=profile.parallel_slots if profile is not None else 1,
                          resident_slot=request.model.deployment.provider.value
                          if request.model.deployment.provider is BackendType.PRISM else None)
        bound_request = replace(request, execution_generation=generation)
        operation = RequestOperation(self, provider, bound_request, generation)
        self._operations.add(operation)
        return operation

    async def embed(self, request):
        operation = await self.prepare(request)
        try:
            operation.backend_started = True
            result = await operation.provider.embed(operation.request)
            if not isinstance(result, EmbeddingResult) or result.request_id != request.context.request_id:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider embedding response identity is invalid")
            expected = request.dimensions or request.model.profile_metadata["dimensions"]
            if len(result.vectors) != len(request.inputs) or any(len(vector) != expected for vector in result.vectors):
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider embedding shape is invalid")
            operation.confirmed_end = True
            return result
        except LocalInferenceError as exc:
            operation.confirmed_end = (isinstance(exc, RequestRejected)
                or exc.failure.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED)
            raise
        finally:
            await operation.close()

    async def chat(self, request):
        operation = await self.prepare(request)
        try:
            operation.backend_started = True
            result = await operation.provider.chat(operation.request)
            if not isinstance(result, InferenceResult) or result.request_id != operation.request.context.request_id:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider returned a different request identity")
            if result.usage.output_tokens > operation.request.options.max_output_tokens:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider exceeded the requested output token budget")
            operation.confirmed_end = True
            return result
        except LocalInferenceError as exc:
            # This code requires a pure rejection or fully consumed native
            # rejection response; interrupted transport must use another code.
            operation.confirmed_end = exc.failure.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED
            raise
        finally:
            await operation.close()

    async def aclose(self):
        if not self._closed:
            self._closed = True
            await asyncio.gather(*(operation.close() for operation in tuple(self._operations)))
            await asyncio.gather(*(provider.aclose() for provider in self.providers.values()))
            if self._measurement_close is not None:
                await asyncio.to_thread(self._measurement_close)


class RequestOperation:
    def __init__(self, service, provider, request, generation):
        self.service, self.provider, self.request, self.generation = service, provider, request, generation
        self.backend_started = self.confirmed_end = False
        self._cleanup = None
        self._iterator = None
        self.backend_session = None
        self._reader = None
        self._reader_error = None
        self._queue = asyncio.Queue(maxsize=4)
        self._detached = asyncio.Event()
        self._backend_context = None

    async def events(self):
        if self._iterator is not None or self._reader is not None:
            raise RuntimeError("request stream may be consumed only once")
        if self.request.model.deployment.provider is not BackendType.OLLAMA:
            async for event in self._validated_events(self.request):
                yield event
            return
        check_context(self.request.context)
        # Keep the original deadline, but own backend cancellation separately:
        # downstream disconnect starts a bounded drain instead of killing proof.
        self._backend_context = replace(self.request.context, cancellation=asyncio.Event())
        self._reader = asyncio.create_task(self._pump())
        try:
            while not self._reader.done() or not self._queue.empty():
                item = asyncio.create_task(self._queue.get())
                try:
                    await with_context(asyncio.wait({item, self._reader}, return_when=asyncio.FIRST_COMPLETED),
                                       self.request.context)
                    if item.done():
                        yield item.result()
                    elif self._reader.done():
                        break
                finally:
                    if not item.done():
                        item.cancel()
                    await asyncio.gather(item, return_exceptions=True)
            if self._reader_error is not None:
                raise self._reader_error
        finally:
            self._detached.set()

    async def _pump(self):
        try:
            request = replace(self.request, context=self._backend_context)
            async for event in self._validated_events(request):
                if self._detached.is_set():
                    continue
                put = asyncio.create_task(self._queue.put(event))
                detached = asyncio.create_task(self._detached.wait())
                try:
                    await asyncio.wait({put, detached}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in (put, detached):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(put, detached, return_exceptions=True)
        except BaseException as exc:
            self._reader_error = exc
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _validated_events(self, request):
        self._iterator = self.provider.stream(request)
        terminal = False
        completed = False
        rejection = None
        indices = EventIndexState()
        self.backend_started = True
        async for event in self._iterator:
            try:
                indices.accept(event)
            except (TypeError, ValueError) as exc:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider violated canonical output indices") from exc
            if terminal or event.request_id != self.request.context.request_id:
                self.confirmed_end = False
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider violated the event lifetime")
            if event.kind is EventKind.USAGE and event.usage.output_tokens > self.request.options.max_output_tokens:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider exceeded the requested output token budget")
            terminal = event.terminal
            completed = event.kind is EventKind.COMPLETED
            if event.kind is EventKind.FAILED and event.error.code is ErrorCode.CONTEXT_LENGTH_EXCEEDED:
                # Error serializers may stop at FAILED. Retain a definitive
                # rejection until source EOF so its ticket end proof is already
                # established when that terminal becomes publicly observable.
                rejection = event
                continue
            yield event
        if not terminal:
            raise failure(ErrorCode.PROVIDER_ERROR, "Provider stream has no terminal event")
        # A terminal frame followed by an exception or an interrupted iterator
        # does not prove the backend ended. Confirm only after clean EOF.
        self.confirmed_end = completed or rejection is not None
        if rejection is not None:
            yield rejection

    async def _finish(self):
        reader_ended = True
        if self._reader is not None:
            self._detached.set()
            done, _ = await asyncio.wait({self._reader}, timeout=min(30, self.service.timeouts.drain))
            if not done:
                self._backend_context.cancellation.set()
                self._reader.cancel()
                await asyncio.wait({self._reader}, timeout=1)
            reader_ended = self._reader.done()
            if reader_ended:
                await asyncio.gather(self._reader, return_exceptions=True)
        if self._iterator is not None and reader_ended:
            try:
                await asyncio.wait_for(self._iterator.aclose(), timeout=self.service.timeouts.drain)
            except (Exception, asyncio.CancelledError):
                pass
        confirmed = reader_ended and (self.confirmed_end or not self.backend_started)
        if not confirmed:
            context = RequestContext(self.request.context.request_id, time.monotonic() + self.service.timeouts.drain, asyncio.Event())
            try:
                confirmed = await asyncio.wait_for(self.provider.wait_request_end(self.request.model.deployment,
                    generation=self.generation, context=context), timeout=self.service.timeouts.drain)
            except (Exception, asyncio.CancelledError):
                confirmed = False
        try:
            await self.service._store("release", self.request.context.request_id, owner=OWNER,
                                      generation=generation_key(self.generation), confirmed_terminated=confirmed is True)
        finally:
            if self.backend_session is not None:
                self.backend_session.close()
        logger.info("Local inference request ended request_id=%s provider=%s confirmed=%s",
                    self.request.context.request_id, self.provider.provider.value, confirmed is True)
        self.service._operations.discard(self)

    async def close(self):
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._finish())
        await asyncio.shield(self._cleanup)


class RuntimeStreamingResponse(StreamingResponse):
    """Own the ticket even when the body never starts or header sending fails."""
    def __init__(self, content, *, operation, **kwargs):
        super().__init__(content, **kwargs)
        self.operation = operation

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.operation.close()
