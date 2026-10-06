"""Prism control and inference transport behind the common provider contract."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import time

import httpx

from kiron_common.local_inference import (
    CapabilityName, CapabilitySet, DeploymentObservation, DiscoveredModel, DiscoverySnapshot, ErrorCode,
    EventKind, FinishReason, InferenceEvent, InferenceResult, LifecycleResult,
    LocalInferenceError, ModelLifecycleOperation, OutputEventLayout, ProviderHealth, ProviderObservation,
    ReasoningKind, ReasoningPart, RuntimeFailure, RuntimeGeneration, TextPart, TokenUsage,
)
from kiron_common.model_catalog import BackendType
from kiron_common.model_state import RuntimeState

from provider_transport import bounded_lines, bounded_request, decode_provider_json, failure, with_context
import prism_tools
import prism_vision
import prism_structured
import prism_reasoning
from prism_reasoning_profile import reasoning_plan
from provider_features import validate_chat_request, validate_features


def _http_failure(status, content):
    """Recognize only the pinned server's complete context-error envelope."""
    if status == 400:
        try:
            body = decode_provider_json(content)
            error = body["error"]
            if (set(body) == {"error"} and type(error) is dict
                    and set(error) == {"code", "type", "message", "n_prompt_tokens", "n_ctx"}
                    and type(error["code"]) is int and error["code"] == 400
                    and error["type"] == "exceed_context_size_error"
                    and type(error["message"]) is str
                    and type(error["n_ctx"]) is int and error["n_ctx"] > 0
                    and type(error["n_prompt_tokens"]) is int
                    and error["n_prompt_tokens"] >= error["n_ctx"]):
                return failure(ErrorCode.CONTEXT_LENGTH_EXCEEDED, "Request exceeds the model context window")
        except (KeyError, TypeError, ValueError):
            pass
    code = (ErrorCode.CONFLICT if status == 409 else ErrorCode.TIMEOUT
            if status in {408, 504} else ErrorCode.PROVIDER_UNAVAILABLE)
    return failure(code, "Prism operation failed")


class PrismProvider:
    provider = BackendType.PRISM

    @property
    def model_lifecycle_operations(self) -> frozenset[ModelLifecycleOperation]:
        return frozenset((ModelLifecycleOperation.LOAD, ModelLifecycleOperation.UNLOAD))

    def __init__(self, *, control: httpx.AsyncClient, inference: httpx.AsyncClient,
                 resolver, implementation, capabilities=None, service_control=None, verify_artifact=None):
        self.control, self.inference, self.resolver = control, inference, resolver
        self.implementation = implementation
        self._capabilities = dict(capabilities or {})
        self.service_control = service_control
        self.verify_artifact = verify_artifact
        self._bindings = {}
        self._closed = False

    async def _request(self, client, method, path, context, **kwargs):
        if client is self.inference:
            # Buffered inference has no token traffic on which to apply the
            # client's streaming idle timeout. Use only the remaining request
            # budget; with_context still bounds the whole headers/body exchange
            # (including trickled bytes) and handles cancellation. Keep connect,
            # write and pool limits, and never mutate the shared stream client.
            remaining = context.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise failure(ErrorCode.TIMEOUT, "Request deadline exceeded")
            kwargs["timeout"] = httpx.Timeout(**{**client.timeout.as_dict(), "read": remaining})
        try:
            status, content = await bounded_request(client, method, path, context, limit=16 * 1024 * 1024, **kwargs)
        except httpx.TimeoutException as exc:
            raise failure(ErrorCode.TIMEOUT, "Prism transport timeout") from exc
        except httpx.HTTPError as exc:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism transport unavailable") from exc
        if status != 200:
            raise _http_failure(status, content)
        try:
            return decode_provider_json(content)
        except ValueError as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Malformed Prism response") from exc

    def _observation(self, data):
        try:
            if type(data) is not dict or data["provider"] != "prism":
                raise ValueError("provider")
            generation = RuntimeGeneration(**data["generation"])
            state = RuntimeState(data["state"])
            healthy = data["health"] == "healthy"
            if data["health"] not in {"healthy", "unhealthy"}:
                raise ValueError("health")
            stamp = data["observed_at"]
            if type(stamp) not in (int, float):
                raise ValueError("timestamp")
            models = {}
            self._bindings.clear()
            deployment_id = data["deployment_id"]
            if deployment_id is not None:
                fingerprint = data["configuration_fingerprint"]
                if state is RuntimeState.LOADED:
                    if not healthy or generation.process_id is None or not data["backend_model"]:
                        raise ValueError("loaded identity")
                    self._bindings[deployment_id] = (generation, data["backend_model"], fingerprint)
                models[deployment_id] = DeploymentObservation(deployment_id, state, generation, fingerprint)
            elif state not in {RuntimeState.UNLOADED, RuntimeState.UNKNOWN, RuntimeState.FAILED}:
                raise ValueError("deployment identity")
            return ProviderObservation(self.provider, generation, datetime.fromtimestamp(stamp, timezone.utc),
                ProviderHealth.AVAILABLE if healthy else ProviderHealth.UNAVAILABLE, models,
                None if healthy else RuntimeFailure(ErrorCode.PROVIDER_UNAVAILABLE, "Prism state unconfirmed"))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._bindings.clear()
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Prism observation") from exc

    async def health(self, context):
        try:
            return self._observation(await self._request(self.control, "GET", "/health", context))
        except LocalInferenceError as exc:
            if exc.failure.code is not ErrorCode.PROVIDER_UNAVAILABLE or self.service_control is None:
                raise
            if await with_context(self.service_control.status(), context) is not ProviderHealth.STARTABLE:
                raise
            return ProviderObservation(self.provider, None, datetime.now(timezone.utc), ProviderHealth.STARTABLE)

    async def discover(self, context):
        try:
            await self.health(context)
            snapshot = await self.resolver.snapshot()
            models = []
            for deployment in snapshot.deployments.values():
                if deployment.provider is self.provider:
                    installed = (False if self.verify_artifact is None else
                                 await with_context(self.verify_artifact(deployment), context))
                    models.append(DiscoveredModel(deployment.reference, deployment.artifact_identity, installed))
            return DiscoverySnapshot(self.provider, snapshot.revision, datetime.now(timezone.utc), models)
        except LocalInferenceError as exc:
            return DiscoverySnapshot(self.provider, "unavailable", datetime.now(timezone.utc), error=exc.failure)

    async def capabilities(self, deployment):
        return self._capabilities.get(deployment.id, CapabilitySet()).for_deployment(deployment, self.implementation)

    def validate_request(self, request, capabilities):
        validate_chat_request(request, capabilities, self.implementation)
        validate_features(request, capabilities, self.implementation)
        prism_vision.validate_images(request.messages, request.model.deployment, capabilities.by_name[CapabilityName.VISION])
        messages = [{"role": message.role.value, "content": prism_vision.message_content(message)}
                    for message in request.messages]
        prism_tools.prepare_tools(request, messages=messages)
        prism_structured.native_output_format(request.options.output_format)
        reasoning_plan(request, capabilities, self.implementation)

    async def start(self, context):
        if self.service_control is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider service start is not configured")
        changed = await with_context(self.service_control.start(), context)
        async def ready():
            while True:
                try:
                    observed = await self.health(context)
                    if observed.health is ProviderHealth.AVAILABLE:
                        return observed
                except LocalInferenceError as exc:
                    if exc.failure.code is not ErrorCode.PROVIDER_UNAVAILABLE:
                        raise
                await asyncio.sleep(.05)
        return LifecycleResult(context.request_id, await with_context(ready(), context), changed)

    async def stop(self, expected_generation, context):
        before = await self.health(context)
        if before.generation != expected_generation:
            raise failure(ErrorCode.CONFLICT, "Provider generation changed")
        if self.service_control is None:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Provider service stop is not configured")
        changed = await with_context(self.service_control.stop(), context)
        # The service controller must prove the unit/cgroup stopped, not merely
        # accept systemctl's request. Its API returns only after that proof.
        observation = ProviderObservation(self.provider, None, datetime.now(timezone.utc), ProviderHealth.STARTABLE)
        self._bindings.clear()
        return LifecycleResult(context.request_id, observation, changed)

    async def _mutate(self, action, deployment, snapshot_revision, expected_generation, context):
        before = self._bindings.get(deployment.id)
        data = await self._request(self.control, "POST", f"/{action}", context, json={
            "deployment_id": deployment.id, "snapshot_revision": snapshot_revision,
            "expected_generation": asdict(expected_generation), "operation_id": context.request_id,
        })
        observation = self._observation(data)
        after = observation.models.get(deployment.id)
        if action == "load" and (after is None or after.state is not RuntimeState.LOADED
                or after.configuration_fingerprint != deployment.configuration_fingerprint):
            raise failure(ErrorCode.PROVIDER_ERROR, "Load was not confirmed for the resolved deployment")
        if action == "unload" and (data["state"] != "unloaded" or data["deployment_id"] is not None
                or observation.health is not ProviderHealth.AVAILABLE or observation.generation != expected_generation):
            raise failure(ErrorCode.PROVIDER_ERROR, "Unload was not confirmed")
        return LifecycleResult(context.request_id, observation,
                               before != self._bindings.get(deployment.id))

    async def load(self, deployment, *, snapshot_revision, expected_generation, context):
        return await self._mutate("load", deployment, snapshot_revision, expected_generation, context)

    async def unload(self, deployment, *, snapshot_revision, expected_generation, context):
        return await self._mutate("unload", deployment, snapshot_revision, expected_generation, context)

    def _payload(self, request, *, stream=False):
        deployment = request.model.deployment
        capabilities = self._capabilities.get(deployment.id, CapabilitySet()).for_deployment(deployment, self.implementation)
        self.validate_request(request, capabilities)
        if stream:
            validate_chat_request(request, capabilities, self.implementation, stream=True)
        binding = self._bindings.get(deployment.id)
        if binding is None or binding[2] != deployment.configuration_fingerprint:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Resolved deployment is not confirmed loaded")
        if request.execution_generation is None or binding[0] != request.execution_generation:
            raise failure(ErrorCode.CONFLICT, "Request generation does not own the loaded model")
        options = request.options
        reasoning = reasoning_plan(request, capabilities, self.implementation)
        messages = [{"role": message.role.value, "content": prism_vision.message_content(message)}
                    for message in request.messages]
        plan = prism_tools.prepare_tools(request, messages=messages)
        payload = {"model": binding[1], "messages": plan.messages,
                   "max_tokens": options.max_output_tokens, "stream": stream, **plan.payload,
                   **prism_structured.native_output_format(options.output_format),
                   **prism_reasoning.native_reasoning_options(enabled=reasoning is not None and reasoning.enabled,
                       budget_tokens=reasoning.budget_tokens if reasoning is not None else None)}
        for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"):
            value = getattr(options.sampling, name)
            if value is not None:
                payload[name] = value
        if options.sampling.stop:
            payload["stop"] = list(options.sampling.stop)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload, plan, reasoning

    @staticmethod
    def _usage(value):
        try:
            if type(value) is not dict or type(value.get("prompt_tokens_details", {})) is not dict:
                raise ValueError("usage shape")
            prompt, completion = value["prompt_tokens"], value["completion_tokens"]
            usage = TokenUsage(prompt, completion,
                               cached_input_tokens=value.get("prompt_tokens_details", {}).get("cached_tokens"))
            if type(value["total_tokens"]) is not int or value["total_tokens"] != usage.total_tokens:
                raise ValueError("total")
            return usage
        except (KeyError, TypeError, ValueError) as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid exact token usage") from exc

    async def chat(self, request):
        payload, plan, reasoning_plan_value = self._payload(request)
        value = await self._request(self.inference, "POST", "/v1/chat/completions", request.context,
                                    json=payload, headers=self._generation_headers(request))
        try:
            if value["model"] != payload["model"]:
                raise ValueError("response model identity")
            choices = value["choices"]
            if (type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict
                    or type(choices[0]["index"]) is not int or choices[0]["index"] != 0):
                raise ValueError("choices")
            choice = choices[0]
            message = choice["message"]
            if type(message) is not dict or message["role"] != "assistant" or message.get("refusal") is not None:
                raise ValueError("unexpected content")
            reasoning = message.get("reasoning_content", "")
            if type(reasoning) is not str or (reasoning and (reasoning_plan_value is None or not reasoning_plan_value.enabled)):
                raise ValueError("unexpected reasoning")
            finish = FinishReason(choice["finish_reason"])
            calls = prism_tools.decode_tool_calls(message.get("tool_calls", []), plan, finish_reason=finish)
            content = message.get("content")
            if content is None and calls:
                content = ""
            if type(content) is not str:
                raise ValueError("invalid assistant text")
            prism_structured.validate_complete_output(content, request.options.output_format, finish)
            usage = await self._exact_usage(request, self._usage(value["usage"]), value.get("__verbose"),
                                             reasoning_plan_value, reasoning, content)
            return InferenceResult(request_id=request.context.request_id,
                                   content=(TextPart(content),) if content else (),
                                   reasoning=(ReasoningPart(ReasoningKind.TEXT, reasoning),) if reasoning else (),
                                   tool_calls=calls, finish_reason=finish, usage=usage)
        except (KeyError, TypeError, ValueError) as exc:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid Prism completion") from exc

    async def _exact_usage(self, request, usage, verbose, plan, reasoning_text, final_text):
        if plan is None:
            return usage
        # This is an original-ID lookup, not tokenization of emitted text. The
        # same generation bearer, context and still-held admission cover it.
        if (type(verbose) is not dict or type(verbose.get("tokens")) is not list
                or len(verbose["tokens"]) != usage.output_tokens or len(verbose["tokens"]) > 100000
                or any(type(token) is not int or token < 0 for token in verbose["tokens"])):
            raise failure(ErrorCode.PROVIDER_ERROR, "Original completion token IDs are unavailable")
        pieces = await self._request(self.inference, "POST", "/tokenize", request.context,
            json={"content": verbose["tokens"], "with_pieces": True, "add_special": False, "parse_special": False},
            headers=self._generation_headers(request))
        if not plan.enabled:
            return prism_reasoning.exact_disabled_reasoning_usage(verbose, usage,
                token_pieces=pieces.get("tokens", ()), rules=plan.rules,
                artifact_sha256=request.model.deployment.artifact_identity.sha256,
                template_revision=plan.rules.template_revision)
        return prism_reasoning.exact_reasoning_usage(verbose, usage, token_pieces=pieces.get("tokens", ()),
            reasoning_text=reasoning_text, final_text=final_text, rules=plan.rules,
            artifact_sha256=request.model.deployment.artifact_identity.sha256,
            template_revision=plan.rules.template_revision)

    async def stream(self, request):
        payload, plan, reasoning_plan_value = self._payload(request, stream=True)
        layout = OutputEventLayout()
        tools = prism_tools.ToolStreamDecoder(plan, layout=layout)
        yield InferenceEvent(EventKind.STARTED, request.context.request_id)
        usage = finish = verbose = None
        done = False
        text_parts, reasoning_parts = [], []
        total_bytes = 0
        response = None
        try:
            native = self.inference.build_request("POST", "/v1/chat/completions", json=payload,
                                                  headers=self._generation_headers(request))
            response = await with_context(self.inference.send(native, stream=True), request.context)
            try:
                if response.status_code != 200:
                    content = bytearray()
                    iterator = response.aiter_bytes(chunk_size=65536).__aiter__()
                    while True:
                        try:
                            chunk = await with_context(iterator.__anext__(), request.context)
                        except StopAsyncIteration:
                            break
                        content.extend(chunk)
                        if len(content) > 16 * 1024 * 1024:
                            raise failure(ErrorCode.PROVIDER_ERROR, "Prism error response limit exceeded")
                    raise _http_failure(response.status_code, bytes(content))
                iterator = bounded_lines(response, request.context).__aiter__()
                while True:
                    try:
                        line = await with_context(iterator.__anext__(), request.context)
                    except StopAsyncIteration:
                        if not done:
                            raise failure(ErrorCode.PROVIDER_ERROR, "Prism stream ended without a terminal frame")
                        text = "".join(text_parts)
                        prism_structured.validate_complete_output(text, request.options.output_format, finish)
                        usage = await self._exact_usage(request, usage, verbose, reasoning_plan_value,
                                                        "".join(reasoning_parts), text)
                        for event in tools.finish(finish):
                            yield event
                        yield InferenceEvent(EventKind.USAGE, request.context.request_id, usage=usage)
                        yield InferenceEvent(EventKind.COMPLETED, request.context.request_id, finish_reason=finish)
                        return
                    total_bytes += len(line.encode("utf-8")) + 1
                    if total_bytes > 16 * 1024 * 1024:
                        raise failure(ErrorCode.PROVIDER_ERROR, "Prism stream output limit exceeded")
                    if not line or line.startswith(":"):
                        continue
                    if done:
                        raise failure(ErrorCode.PROVIDER_ERROR, "Prism sent data after its terminal frame")
                    if not line.startswith("data: "):
                        raise failure(ErrorCode.PROVIDER_ERROR, "Malformed Prism stream framing")
                    if line == "data: [DONE]":
                        if usage is None or finish is None:
                            raise failure(ErrorCode.PROVIDER_ERROR, "Prism stream lacks final usage or finish reason")
                        done = True
                        continue
                    value = decode_provider_json(line[6:])
                    if value["model"] != payload["model"]:
                        raise ValueError("stream model identity")
                    if "__verbose" in value:
                        if verbose is not None or type(value["__verbose"]) is not dict:
                            raise ValueError("duplicate or invalid native token evidence")
                        verbose = value["__verbose"]
                    if value.get("usage") is not None:
                        if usage is not None:
                            raise ValueError("duplicate usage")
                        usage = self._usage(value["usage"])
                    choices = value["choices"]
                    if type(choices) is not list or len(choices) > 1:
                        raise ValueError("choices")
                    for choice in choices:
                        if (type(choice) is not dict or type(choice["index"]) is not int
                                or choice["index"] != 0 or finish is not None):
                            raise ValueError("choice order")
                        delta = choice["delta"]
                        if (type(delta) is not dict or delta.get("role", "assistant") != "assistant"
                                or delta.get("refusal") is not None):
                            raise ValueError("unexpected assistant delta")
                        thought = delta.get("reasoning_content")
                        if thought is not None:
                            if type(thought) is not str or (thought and (reasoning_plan_value is None or not reasoning_plan_value.enabled)):
                                raise ValueError("unexpected reasoning")
                            if thought:
                                reasoning_parts.append(thought)
                                yield InferenceEvent(EventKind.REASONING_DELTA, request.context.request_id,
                                                     text=thought, reasoning_kind=ReasoningKind.TEXT,
                                                     **layout.reasoning(ReasoningKind.TEXT))
                        if "tool_calls" in delta:
                            for event in tools.feed(delta["tool_calls"]):
                                yield event
                        content = delta.get("content")
                        if content is not None:
                            if type(content) is not str:
                                raise ValueError("invalid text delta")
                            if content:
                                text_parts.append(content)
                                yield InferenceEvent(EventKind.TEXT_DELTA, request.context.request_id,
                                                     text=content, **layout.text())
                        if choice.get("finish_reason") is not None:
                            finish = FinishReason(choice["finish_reason"])
            finally:
                await response.aclose()
        except LocalInferenceError as exc:
            yield InferenceEvent(EventKind.CANCELLED if exc.failure.code is ErrorCode.CANCELLED else EventKind.FAILED,
                                 request.context.request_id, **({} if exc.failure.code is ErrorCode.CANCELLED else {"error": exc.failure}))
        except httpx.TimeoutException:
            yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                                 error=RuntimeFailure(ErrorCode.TIMEOUT, "Prism transport timeout"))
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            yield InferenceEvent(EventKind.FAILED, request.context.request_id,
                                 error=RuntimeFailure(ErrorCode.PROVIDER_ERROR, "Invalid or interrupted Prism stream"))

    async def generate(self, request):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Prism raw prompt generation is not verified")

    @staticmethod
    def _generation_headers(request):
        generation = request.execution_generation
        if generation is None or generation.process_id is None:
            raise failure(ErrorCode.CONFLICT, "Prism request has no process generation")
        return {"Authorization": f"Bearer kiron-prism-{generation.process_id}"}

    async def embed(self, request):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "This Prism profile does not provide embeddings")

    async def wait_request_end(self, deployment, *, generation, context):
        # Until an exact slot-idle observation is implemented and tested, a
        # cancelled transport cannot release GPU work on its own.
        return False

    async def aclose(self):
        if not self._closed:
            self._closed = True
            await asyncio.gather(self.control.aclose(), self.inference.aclose())
