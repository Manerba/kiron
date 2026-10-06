"""Evidence-bound native Ollama adapter; no discovery, clients or loads at import."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import ipaddress
import io
import json
import re
from uuid import uuid4

import httpx

from kiron_common.local_inference import (
    ArtifactIdentity, CapabilityName, CapabilitySet, CapabilityStatus, DeploymentObservation,
    DiscoveredModel, DiscoverySnapshot, EmbeddingResult, ErrorCode, EventKind, FinishReason,
    InferenceEvent, InferenceRequest, InferenceResult, LifecycleResult, LocalInferenceError, Message, MessageRole,
    ModelLifecycleOperation, OutputEventLayout, OutputFormatKind, ProviderHealth, ProviderObservation, RuntimeFailure, RuntimeGeneration,
    TextPart, TokenUsage, ToolChoiceKind,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType
from kiron_common.model_state import RuntimeState
from kiron_common.ollama_compat import (
    canonical_model_name, ensure_num_gpu_zero, ensure_think_false_dict,
    normalize_ps_response, normalize_tags_response,
)

from provider_transport import decode_provider_json, failure, with_context
from openai_generation import validate_output, validate_request_schemas
from kiron_common.local_inference.json_schema import SchemaError
from provider_features import validate_features
import ollama_generation
import ollama_tools
import ollama_vision


MAX_RESPONSE = 16 * 1024 * 1024
MAX_FRAME = 1024 * 1024
CONTEXT_OVERFLOW = "the input length exceeds the context length"


def _digest(value):
    if isinstance(value, str):
        value = value.removeprefix("sha256:")
        if re.fullmatch("[0-9a-f]{64}", value):
            return value
    return None


class OllamaProvider:
    """Native generation is unavailable: generation denotes our observed probe epoch.

    An unseen backend restart cannot be inferred from /api/version or /api/ps.
    Epochs rotate on observed outage/version change and owned lifecycle actions;
    they must never be advertised as native process IDs. Cancellation has no
    exact request-end proof, so wait_request_end deliberately returns False.
    The composition transfers client ownership; aclose closes it exactly once.
    """
    provider = BackendType.OLLAMA

    @property
    def model_lifecycle_operations(self) -> frozenset[ModelLifecycleOperation]:
        return frozenset((ModelLifecycleOperation.LOAD, ModelLifecycleOperation.UNLOAD))

    def __init__(self, *, client: httpx.AsyncClient, resolver, implementation,
                 capabilities=None, compatibility=None, expected_version=None,
                 service_control=None, embedding_formatter=None, options_policy=None,
                 keep_alive=None):
        try:
            local = ipaddress.ip_address(client.base_url.host).is_loopback
        except ValueError:
            local = False
        if client.base_url.scheme != "http" or not local or client.trust_env:
            raise ValueError("Ollama client requires fixed loopback HTTP and trust_env=False")
        if keep_alive is not None and (type(keep_alive) not in (int, str) or keep_alive == 0):
            raise ValueError("load keep_alive must not immediately unload the model")
        self.client, self.resolver, self.implementation = client, resolver, implementation
        self._capabilities = dict(capabilities or {})
        self.compatibility, self.expected_version = compatibility, expected_version
        self.service_control, self.embedding_formatter = service_control, embedding_formatter
        self.options_policy, self.keep_alive = options_policy, keep_alive
        self.generation = RuntimeGeneration("ollama-probe:" + uuid4().hex, None)
        self._version, self._available, self._closed = None, None, False
        self._resident_references = set()
        self._mutation = asyncio.Lock()

    def _compat(self, *names):
        if self.compatibility is None or any(not getattr(self.compatibility, name).ok for name in names):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama compatibility evidence is missing or failed")

    def validate_request(self, request, capabilities):
        """Pure validation also used before direct adapter calls touch the resolver."""
        validate_features(request, capabilities, self.implementation)
        self._sampling_options(request, capabilities)
        ollama_vision.validate_images(request.messages)
        if any(message.role not in {MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL}
               or message.reasoning for message in request.messages):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Message mapping is not verified", "messages")
        roles = capabilities.by_name[CapabilityName.CHAT].constraints.get("roles")
        if roles is None:
            raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "Message role mapping is not verified", "roles")
        if any(not roles.accepts(message.role.value) for message in request.messages):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Message role is outside the verified profile", "roles")
        try:
            validate_request_schemas(request.tools, request.options.output_format)
            ollama_generation.native_output_format(request.options.output_format)
        except SchemaError as exc:
            raise failure(exc.code, "Invalid or unsupported output schema", "response_format") from None
        ollama_tools.prepare_tools(request)
        ollama_generation.native_reasoning_options(request.options.reasoning)

    def _sampling_options(self, request, capabilities):
        deployment, options = request.model.deployment, request.options
        if not capabilities.supports(CapabilityName.CHAT, deployment, self.implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Chat mapping lacks matching evidence", "chat")
        resident_options = self._resident_options(deployment)
        if resident_options is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama resident profile is not verified", "chat")
        if options.max_output_tokens > resident_options["num_ctx"]:
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Output budget exceeds the measured context", "max_output_tokens")
        constraints = capabilities.by_name[CapabilityName.CHAT].constraints
        if deployment.resource_profile is not None and options.max_output_tokens > deployment.resource_profile.context_tokens:
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Output budget exceeds the measured context", "max_output_tokens")
        values = {"max_output_tokens": options.max_output_tokens}
        for name in ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty"):
            value = getattr(options.sampling, name)
            if value is not None:
                values[name] = value
        native = {}
        for name, value in values.items():
            if name in {"frequency_penalty", "presence_penalty"}:
                raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "No verified exact Ollama penalty mapping", name)
            constraint = constraints.get(name)
            if constraint is None:
                raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "Sampling mapping is not verified", name)
            if not constraint.accepts(value):
                raise failure(ErrorCode.UNSUPPORTED_VALUE, "Sampling value lacks matching evidence", name)
            native["num_predict" if name == "max_output_tokens" else name] = value
        if options.sampling.stop:
            constraint = constraints.get("stop")
            if constraint is None:
                raise failure(ErrorCode.UNSUPPORTED_PARAMETER, "Stop mapping is not verified", "stop")
            if any(not constraint.accepts(value) for value in options.sampling.stop):
                raise failure(ErrorCode.UNSUPPORTED_VALUE, "Stop sequence lacks matching evidence", "stop")
            native["stop"] = list(options.sampling.stop)
        return native

    def _new_epoch(self):
        self.generation = RuntimeGeneration("ollama-probe:" + uuid4().hex, None)

    def _execution_generation(self, generation):
        if generation is None or generation != self.generation:
            raise failure(ErrorCode.CONFLICT, "Ollama execution generation changed or was not bound")

    def _resident_options(self, deployment):
        """Exact measured residency constraints do not authorize a cold load.

        /api/ps proves CPU execution and context size. It cannot prove a GPU
        layer split, so GPU inference remains unverified by this adapter.
        """
        values = self._capabilities.get(deployment.id, CapabilitySet()).for_deployment(deployment, self.implementation)
        profiles = []
        for name in (CapabilityName.CHAT, CapabilityName.EMBEDDINGS):
            if not values.supports(name, deployment, self.implementation):
                continue
            constraints = values.by_name[name].constraints
            context = constraints.get("context_tokens")
            device = constraints.get("device")
            if (context is None or context.allowed_values is None or len(context.allowed_values) != 1
                    or type(context.allowed_values[0]) is not int or context.allowed_values[0] < 1
                    or not context.accepts(context.allowed_values[0])
                    or device is None or device.allowed_values != ("cpu",) or not device.accepts("cpu")):
                raise failure(ErrorCode.INVALID_CONFIGURATION, "Ollama evidence lacks an exact CPU resident profile")
            profiles.append({"num_ctx": context.allowed_values[0], "num_gpu": 0})
        if not profiles:
            return None
        profile = profiles[0]
        if any(value != profile for value in profiles):
            raise failure(ErrorCode.INVALID_CONFIGURATION, "Ollama resident evidence profiles disagree")
        resource = deployment.resource_profile
        if resource is not None and (resource.context_tokens != profile["num_ctx"] or resource.gpu_layers != 0):
            raise failure(ErrorCode.INVALID_CONFIGURATION, "Ollama resource and resident profiles disagree")
        return profile

    async def _confirm_resident(self, deployment, generation, context):
        self._execution_generation(generation)
        if self._resident_options(deployment) is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama resident profile is not verified")
        observation = await self.health(context)
        if observation.error is not None and observation.error.code in {ErrorCode.TIMEOUT, ErrorCode.CANCELLED}:
            raise LocalInferenceError(observation.error)
        self._execution_generation(generation)
        current = observation.models.get(deployment.id)
        if (observation.health is not ProviderHealth.AVAILABLE or current is None
                or current.state is not RuntimeState.LOADED):
            raise failure(ErrorCode.CONFLICT, "Ollama residency differs from the verified execution profile")

    async def _response(self, method, path, context, payload=None, *, execution_generation=None,
                        resident_deployment=None):
        request = self.client.build_request(method, path, json=payload)
        async def send():
            if execution_generation is not None:
                self._execution_generation(execution_generation)
            if resident_deployment is not None:
                await self._confirm_resident(resident_deployment, execution_generation, context)
            return await self.client.send(request, stream=True, follow_redirects=False)
        try:
            response = await with_context(send(), context)
        except httpx.TimeoutException as exc:
            raise failure(ErrorCode.TIMEOUT, "Ollama transport timeout") from exc
        except httpx.HTTPError as exc:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama transport unavailable") from exc
        if response.status_code != 200:
            try:
                if response.status_code == 400 and path in ("/api/chat", "/api/generate", "/api/embed"):
                    raw = bytearray()
                    async for block in self._blocks(response, context):
                        raw.extend(block)
                    try:
                        value = decode_provider_json(raw)
                    except ValueError:
                        value = None
                    if value == {"error": CONTEXT_OVERFLOW}:
                        raise failure(ErrorCode.CONTEXT_LENGTH_EXCEEDED, "Input exceeds the model context", "messages")
            except httpx.TimeoutException as exc:
                raise failure(ErrorCode.TIMEOUT, "Ollama error response timeout") from exc
            except httpx.HTTPError as exc:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama error response interrupted") from exc
            finally:
                await response.aclose()
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama request was rejected")
        return response

    async def _blocks(self, response, context):
        iterator, count = response.aiter_bytes().__aiter__(), 0
        while True:
            try:
                block = await with_context(iterator.__anext__(), context)
            except StopAsyncIteration:
                return
            count += len(block)
            if count > MAX_RESPONSE:
                raise failure(ErrorCode.PROVIDER_ERROR, "Ollama response limit exceeded")
            yield block

    async def _json(self, method, path, context, payload=None, *, execution_generation=None,
                    resident_deployment=None):
        response = await self._response(method, path, context, payload, execution_generation=execution_generation,
                                        resident_deployment=resident_deployment)
        try:
            content = bytearray()
            async for block in self._blocks(response, context):
                content.extend(block)
            value = decode_provider_json(content)
            if not isinstance(value, dict) or value.get("error"):
                raise ValueError("invalid native body")
            if execution_generation is not None:
                self._execution_generation(execution_generation)
            return value
        except httpx.TimeoutException as exc:
            raise failure(ErrorCode.TIMEOUT, "Ollama response timeout") from exc
        except httpx.HTTPError as exc:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama response interrupted") from exc
        except (ValueError, UnicodeError) as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama response") from exc
        finally:
            await response.aclose()

    async def _snapshot(self, context):
        return await with_context(self.resolver.snapshot(), context)

    async def _resolved(self, deployment, revision, context):
        snapshot = await self._snapshot(context)
        current = snapshot.resolve_deployment(deployment.id, expected_revision=revision)
        if current != deployment or current.provider is not self.provider:
            raise failure(ErrorCode.CONFLICT, "Resolved Ollama deployment changed")
        return snapshot

    async def discover(self, context):
        try:
            self._compat("tags_shape")
            value = await self._json("GET", "/api/tags", context)
            result = normalize_tags_response(value)
            if not result.ok or result.data["failures"]:
                raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama inventory")
            models = []
            for item in result.data["models"]:
                digest = _digest(item.get("digest"))
                if digest is None:
                    raise failure(ErrorCode.PROVIDER_ERROR, "Ollama inventory lacks artifact identity")
                models.append(DiscoveredModel(canonical_model_name(item["name"]),
                    ArtifactIdentity(ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST,
                                     sha256=digest, manifest_digest=digest), True))
            revision = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            return DiscoverySnapshot(self.provider, revision, datetime.now(timezone.utc), tuple(models))
        except LocalInferenceError as exc:
            return DiscoverySnapshot(self.provider, "unavailable", datetime.now(timezone.utc), error=exc.failure)

    async def health(self, context):
        snapshot = None
        try:
            self._compat("version_endpoint", "ps_shape", "ps_size_vram")
            snapshot = await self._snapshot(context)
            version = (await self._json("GET", "/api/version", context)).get("version")
            if not isinstance(version, str) or not version:
                raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama version")
            if self._version is not None and self._version != version:
                self._new_epoch()
            self._version = version
            if self.expected_version is not None and version != self.expected_version:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama differs from its compatibility evidence")
            result = normalize_ps_response(await self._json("GET", "/api/ps", context), require_size_vram=True)
            if not result.ok:
                raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama resident inventory")
            self._resident_references = {canonical_model_name(item.name) for item in result.data["models"]}
            models = {}
            for deployment in snapshot.deployments.values():
                if deployment.provider is not self.provider:
                    continue
                matches = [item for item in result.data["models"]
                           if canonical_model_name(deployment.reference) in {
                               canonical_model_name(item.name), canonical_model_name(item.model)}]
                state = RuntimeState.UNLOADED
                if matches:
                    expected = deployment.artifact_identity.sha256 or _digest(deployment.artifact_identity.manifest_digest)
                    matched = (len(matches) == 1 and expected is not None
                               and _digest(matches[0].raw.get("digest")) == expected)
                    if deployment.resource_profile is not None:
                        matched = matched and matches[0].raw.get("context_length") == deployment.resource_profile.context_tokens
                    try:
                        options = self._resident_options(deployment)
                    except LocalInferenceError:
                        matched = False
                    else:
                        if options is not None:
                            raw = matches[0].raw
                            matched = (matched and type(raw.get("context_length")) is int
                                       and raw["context_length"] == options["num_ctx"]
                                       and type(raw.get("size_vram")) is int and raw["size_vram"] == 0)
                    state = RuntimeState.LOADED if matched else RuntimeState.UNKNOWN
                models[deployment.id] = DeploymentObservation(deployment.id, state, self.generation,
                                                               deployment.configuration_fingerprint)
            self._available = True
            return ProviderObservation(self.provider, self.generation, datetime.now(timezone.utc),
                                       ProviderHealth.AVAILABLE, models)
        except LocalInferenceError as exc:
            if self._available is not False:
                self._new_epoch()
            self._available = False
            self._resident_references.clear()
            models = {} if snapshot is None else {
                deployment.id: DeploymentObservation(deployment.id, RuntimeState.UNKNOWN,
                    self.generation, deployment.configuration_fingerprint, exc.failure)
                for deployment in snapshot.deployments.values() if deployment.provider is self.provider}
            return ProviderObservation(self.provider, self.generation, datetime.now(timezone.utc),
                                       ProviderHealth.UNAVAILABLE, models, exc.failure)

    async def capabilities(self, deployment):
        values = self._capabilities.get(deployment.id, CapabilitySet()).for_deployment(deployment, self.implementation)
        try:
            self._resident_options(deployment)
        except LocalInferenceError:
            return CapabilitySet()
        implemented = {CapabilityName.CHAT, CapabilityName.STREAMING, CapabilityName.EMBEDDINGS,
                       CapabilityName.FUNCTION_TOOLS, CapabilityName.PARALLEL_TOOLS, CapabilityName.STRUCTURED_OUTPUT}
        return CapabilitySet({name: (capability if name in implemented else
                              replace(capability, status=CapabilityStatus.UNSUPPORTED))
                              for name, capability in values.by_name.items()})

    async def _require(self, deployment, name):
        capabilities = await self.capabilities(deployment)
        if not capabilities.supports(name, deployment, self.implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama capability lacks matching evidence", name.value)
        return capabilities.by_name[name]

    async def start(self, context):
        if self.service_control is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama service start is not configured")
        if self._mutation.locked():
            raise failure(ErrorCode.CONFLICT, "Ollama lifecycle operation already active")
        async with self._mutation:
            changed = await with_context(self.service_control.start(), context)
            if changed:
                self._new_epoch()
            return LifecycleResult(context.request_id, await self.health(context), changed)

    async def stop(self, expected_generation, context):
        if expected_generation != self.generation:
            raise failure(ErrorCode.CONFLICT, "Ollama probe generation changed")
        if self.service_control is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Ollama service stop is not configured")
        if self._mutation.locked():
            raise failure(ErrorCode.CONFLICT, "Ollama lifecycle operation already active")
        async with self._mutation:
            changed = await with_context(self.service_control.stop(), context)
            self._new_epoch()
            self._available = False
            # The injected fixed controller must prove container/service shutdown.
            return LifecycleResult(context.request_id, ProviderObservation(self.provider, self.generation,
                                   datetime.now(timezone.utc), ProviderHealth.STARTABLE), changed)

    async def _runtime_options(self, deployment, options, context):
        profile = deployment.resource_profile
        if profile is not None:
            options.update(num_ctx=profile.context_tokens, num_gpu=profile.gpu_layers)
        resident_options = self._resident_options(deployment)
        if resident_options is not None:
            options.update(resident_options)
        original = dict(options)
        if self.options_policy is not None:
            await with_context(self.options_policy(deployment, options, context), context)
        if ({key: value for key, value in options.items() if key != "num_gpu"}
                != {key: value for key, value in original.items() if key != "num_gpu"}
                or (options.get("num_gpu") != original.get("num_gpu") and options.get("num_gpu") != 0)):
            raise failure(ErrorCode.INVALID_CONFIGURATION, "Options policy may only force CPU execution")
        if options.get("num_gpu") == 0:
            self._compat("num_gpu_zero_chat_generate")
            ensure_num_gpu_zero(options)
        if set(options) - {"num_ctx", "num_gpu", "num_predict", "temperature", "top_p", "seed", "stop"}:
            raise failure(ErrorCode.INVALID_CONFIGURATION, "Unknown native runtime option")
        return options

    async def _lifecycle(self, deployment, revision, expected_generation, context, *, load):
        self._compat("generate_nonstream", "think_false", *( () if load else ("keep_alive_zero_unload",)))
        await self._resolved(deployment, revision, context)
        if self._mutation.locked():
            raise failure(ErrorCode.CONFLICT, "Ollama lifecycle operation already active")
        async with self._mutation:
            before = await self.health(context)
            if before.generation != expected_generation or before.health is not ProviderHealth.AVAILABLE:
                raise failure(ErrorCode.CONFLICT, "Ollama probe generation or health changed")
            state = before.models[deployment.id].state
            target = RuntimeState.LOADED if load else RuntimeState.UNLOADED
            if state is target:
                return LifecycleResult(context.request_id, before, False)
            if state is RuntimeState.UNKNOWN:
                raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama resident identity is unknown")
            if load and self._resident_references - {canonical_model_name(deployment.reference)}:
                raise failure(ErrorCode.CONFLICT, "Ollama load may not implicitly evict another deployment")
            payload = {"model": deployment.reference, "stream": False,
                       "options": await self._runtime_options(deployment, {}, context)}
            if not load or self.keep_alive is not None:
                payload["keep_alive"] = self.keep_alive if load else 0
            ensure_think_false_dict(payload, "/api/generate")
            await self._json("POST", "/api/generate", context, payload)
            while True:
                observation = await self.health(context)
                if observation.health is not ProviderHealth.AVAILABLE or observation.generation != expected_generation:
                    raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Ollama lifecycle verification became unavailable")
                if observation.models[deployment.id].state is target:
                    return LifecycleResult(context.request_id, observation, True)
                await with_context(asyncio.sleep(0.1), context)

    async def load(self, deployment, *, snapshot_revision, expected_generation, context):
        return await self._lifecycle(deployment, snapshot_revision, expected_generation, context, load=True)

    async def unload(self, deployment, *, snapshot_revision, expected_generation, context):
        return await self._lifecycle(deployment, snapshot_revision, expected_generation, context, load=False)

    async def _payload(self, request, *, generate=False, stream=False):
        self._execution_generation(request.execution_generation)
        deployment = request.model.deployment
        capabilities = await self.capabilities(deployment)
        # Generate keeps its raw prompt on the wire. The temporary canonical
        # view shares pure option checks without routing generation through chat.
        checked = InferenceRequest(request.model, (Message(MessageRole.USER, (TextPart(request.prompt),)),),
            request.options, request.context, execution_generation=request.execution_generation) if generate else request
        self.validate_request(checked, capabilities)
        self._compat("generate_nonstream" if generate else "chat_nonstream", "think_false")
        if stream:
            self._compat("stream_ndjson")
            if not capabilities.supports(CapabilityName.STREAMING, deployment, self.implementation):
                raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Streaming lacks matching evidence", "streaming")
        native_options = self._sampling_options(request, capabilities)
        snapshot = await self._resolved(deployment, request.model.snapshot_revision, request.context)
        if snapshot.resolve(request.model.public_model_id) != request.model:
            raise failure(ErrorCode.CONFLICT, "Resolved model address changed")
        options = request.options
        payload = {"model": deployment.reference, "stream": stream, "truncate": False, "shift": False,
                   "options": await self._runtime_options(deployment, native_options, request.context)}
        payload.update(ollama_generation.native_output_format(options.output_format))
        payload.update(ollama_generation.native_reasoning_options(options.reasoning))
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if generate:
            payload["prompt"] = request.prompt
        else:
            messages = [{"role": message.role.value, "content": ollama_vision.message_content(message)}
                        for message in request.messages]
            plan = ollama_tools.prepare_tools(request, messages=messages)
            payload.update(plan.payload)
            payload["messages"] = plan.messages
        return ensure_think_false_dict(payload, "/api/generate" if generate else "/api/chat")

    @staticmethod
    def _usage(value):
        return TokenUsage(value["prompt_eval_count"], value["eval_count"])

    @staticmethod
    def _content(value, reference, *, generate=False):
        metadata = {"model", "created_at", "done", "done_reason", "total_duration", "load_duration",
                    "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"}
        allowed = metadata | ({"response", "thinking", "context"} if generate else {"message"})
        if (type(value) is not dict or set(value) - allowed or type(value.get("done")) is not bool
                or canonical_model_name(value.get("model", "")) != canonical_model_name(reference)):
            raise ValueError("invalid model, status or backend error")
        if not value["done"] and value.get("done_reason") not in (None, ""):
            raise ValueError("early finish reason")
        if generate:
            text = value["response"]
            if "thinking" in value and value["thinking"] != "":
                raise ValueError("unexpected reasoning")
        else:
            message = value["message"]
            if (type(message) is not dict or set(message) - {"role", "content", "thinking", "tool_calls"}
                    or message["role"] != "assistant" or "thinking" in message and message["thinking"] != ""):
                raise ValueError("unexpected assistant payload")
            text = message["content"]
        if not isinstance(text, str):
            raise ValueError("invalid content")
        return text

    async def _completion(self, request, *, generate=False):
        payload = await self._payload(request, generate=generate)
        value = await self._json("POST", "/api/generate" if generate else "/api/chat", request.context, payload,
                                 execution_generation=request.execution_generation, resident_deployment=request.model.deployment)
        try:
            content = self._content(value, request.model.deployment.reference, generate=generate)
            if value["done"] is not True or value["done_reason"] not in {"stop", "length"}:
                raise ValueError("missing terminal status")
            calls = ()
            finish = FinishReason(value["done_reason"])
            if not generate:
                raw_calls = value["message"].get("tool_calls", [])
                plan = ollama_tools.prepare_tools(request)
                calls = ollama_tools.decode_tool_calls(raw_calls, plan, finish_reason=value["done_reason"])
                finish = ollama_tools.canonical_finish(value["done_reason"], bool(calls))
            validate_output(content, request.options.output_format, finish)
            return InferenceResult(request.context.request_id, (TextPart(content),) if content else (), (), calls,
                                   self._usage(value), finish)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama completion") from exc

    async def chat(self, request):
        return await self._completion(request)

    async def generate(self, request):
        return await self._completion(request, generate=True)

    async def _frames(self, response, context):
        pending = bytearray()
        async for block in self._blocks(response, context):
            pending.extend(block)
            while b"\n" in pending:
                line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                if len(line) > MAX_FRAME:
                    raise ValueError("frame limit")
                if line.strip():
                    yield decode_provider_json(line)
            if len(pending) > MAX_FRAME:
                raise ValueError("frame limit")
        if pending.strip():
            raise ValueError("unterminated native frame")

    async def stream(self, request):
        payload = await self._payload(request, stream=True)
        layout = OutputEventLayout()
        tools = ollama_tools.ToolStreamDecoder(ollama_tools.prepare_tools(request), layout=layout)
        yield InferenceEvent(EventKind.STARTED, request.context.request_id)
        response = None
        try:
            response = await self._response("POST", "/api/chat", request.context, payload,
                                            execution_generation=request.execution_generation,
                                            resident_deployment=request.model.deployment)
            terminal, native_error, structured = None, None, io.StringIO()
            async for frame in self._frames(response, request.context):
                self._execution_generation(request.execution_generation)
                if terminal is not None or native_error is not None:
                    raise ValueError("frame after terminal")
                if frame == {"error": CONTEXT_OVERFLOW, "status": 400}:
                    native_error = RuntimeFailure(ErrorCode.CONTEXT_LENGTH_EXCEEDED,
                                                  "Input exceeds the model context", "messages")
                    continue
                content = self._content(frame, request.model.deployment.reference)
                if content:
                    if request.options.output_format.kind is not OutputFormatKind.TEXT:
                        structured.write(content)
                    yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id, text=content, **layout.text())
                if "tool_calls" in frame["message"]:
                    for event in tools.feed(frame["message"]["tool_calls"]):
                        yield event
                if frame["done"]:
                    if frame["done_reason"] not in {"stop", "length"}:
                        raise ValueError("invalid finish reason")
                    terminal = frame
            if native_error is not None:
                raise LocalInferenceError(native_error)
            if terminal is None:
                raise ValueError("missing terminal frame")
            self._execution_generation(request.execution_generation)
            finish = ollama_tools.canonical_finish(terminal["done_reason"], tools.has_calls)
            validate_output(structured.getvalue(), request.options.output_format, finish)
            usage = self._usage(terminal)
            for event in tools.finish(terminal["done_reason"]):
                yield event
            yield InferenceEvent(EventKind.USAGE, request.context.request_id, usage=usage)
            yield InferenceEvent(EventKind.COMPLETED, request.context.request_id,
                                 finish_reason=finish)
        except LocalInferenceError as exc:
            cancelled = exc.failure.code is ErrorCode.CANCELLED
            yield InferenceEvent(EventKind.CANCELLED if cancelled else EventKind.FAILED,
                                 request.context.request_id, **({} if cancelled else {"error": exc.failure}))
        except httpx.TimeoutException:
            yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                                 error=RuntimeFailure(ErrorCode.TIMEOUT, "Ollama transport timeout"))
        except (httpx.HTTPError, KeyError, TypeError, ValueError, AttributeError):
            yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                                 error=RuntimeFailure(ErrorCode.PROVIDER_ERROR, "Invalid or interrupted Ollama stream"))
        finally:
            if response is not None:
                await response.aclose()

    async def embed(self, request):
        self._execution_generation(request.execution_generation)
        deployment = request.model.deployment
        snapshot = await self._resolved(deployment, request.model.snapshot_revision, request.context)
        if snapshot.resolve(request.model.public_model_id) != request.model:
            raise failure(ErrorCode.CONFLICT, "Resolved embedding role changed")
        await self._require(deployment, CapabilityName.EMBEDDINGS)
        if self.embedding_formatter is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding role formatting is not configured")
        inputs = await with_context(self.embedding_formatter(request.model, request.inputs), request.context)
        if (not isinstance(inputs, (tuple, list)) or len(inputs) != len(request.inputs)
                or any(not isinstance(value, str) for value in inputs)):
            raise failure(ErrorCode.INVALID_CONFIGURATION, "Invalid embedding formatter result")
        value = await self._json("POST", "/api/embed", request.context,
                                 {"model": deployment.reference, "input": list(inputs), "truncate": False,
                                  "options": await self._runtime_options(deployment, {}, request.context)},
                                 execution_generation=request.execution_generation, resident_deployment=deployment)
        try:
            if (canonical_model_name(value["model"]) != canonical_model_name(deployment.reference)
                    or len(value["embeddings"]) != len(inputs)):
                raise ValueError("embedding model or count")
            return EmbeddingResult(request.context.request_id, value["embeddings"], TokenUsage(value["prompt_eval_count"], 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Ollama embeddings") from exc

    async def wait_request_end(self, deployment, *, generation, context):
        return False  # /api/ps confirms residency, not completion of one cancelled request.

    async def aclose(self):
        if not self._closed:
            self._closed = True
            await self.client.aclose()
