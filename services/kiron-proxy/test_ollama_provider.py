"""Canonical Ollama adapter checks over MockTransport; never contact a live model."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet,
    CapabilityStatus, EmbeddingRequest, EmbeddingRole, EventKind, FinishReason,
    GenerateRequest, GenerationOptions, InferenceRequest, LocalInferenceError,
    Message, MessageRole, ParameterConstraint, ProviderHealth, RequestContext,
    OutputFormat, OutputFormatKind, ReasoningKind, ReasoningOptions, ReasoningPart,
    ResolvedDeployment, ResolvedModel, ResolverSnapshot, RuntimeImplementation,
    SamplingOptions, TextPart, ToolDefinition, validate_event_sequence,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.model_state import RuntimeState
from kiron_common.ollama_compat import CompatResult, OllamaCapabilities

from ollama_provider import OllamaProvider
from provider_transport import decode_provider_json


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, blocks, *, stall=False):
        self.blocks, self.stall, self.closed = blocks, stall, False

    async def __aiter__(self):
        for block in self.blocks:
            yield block
        if self.stall:
            await asyncio.sleep(10)

    async def aclose(self):
        self.closed = True


class ProviderJsonTests(unittest.TestCase):
    def test_exact_depth_limit_unicode_and_duplicates_in_nested_fields(self):
        accepted = '{"nested":' + '[' * 31 + '0' + ']' * 31 + ',"text":"Grün 🌲"}'
        self.assertEqual(decode_provider_json(accepted.encode())["text"], "Grün 🌲")
        invalid = ['{"nested":' + '[' * 32 + '0' + ']' * 32 + '}',
            '{"meta":{"x":1,"x":2}}', '{"meta":Infinity}', '{"meta":-Infinity}',
            '{"meta":1e999}', '{"meta":"\\ud800"}', '{"\\ud800":0}', 'null', '[]']
        for source in invalid:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, '^Invalid provider JSON$'):
                decode_provider_json(source)
        for raw in (b'{"meta":"\xff"}', b'\xff\xfe{\x00}\x00', {"not": "encoded"}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_provider_json(raw)


class OllamaProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.identity = ArtifactIdentity(ArtifactType.OLLAMA, ArtifactFormat.OLLAMA_MANIFEST,
                                         sha256="a" * 64, manifest_digest="a" * 64)
        self.deployment = ResolvedDeployment("fixture.deployment", BackendType.OLLAMA, "fixture:latest",
                                             self.identity, LoaderType.OLLAMA, None, "b" * 64)
        self.model = ResolvedModel("fixture", self.deployment, None, None, "c" * 64)
        self.snapshot = ResolverSnapshot("c" * 64, {self.deployment.id: self.deployment}, {"fixture": self.model})
        self.resolver = SimpleNamespace(snapshot=mock.AsyncMock(side_effect=lambda: self.snapshot))
        self.implementation = RuntimeImplementation("fixture-ollama", "fixture-template", "fixture-parser")
        evidence = CapabilityEvidence("fixture-ollama", self.identity.fingerprint, "a" * 64, None,
                     "fixture-template", "fixture-parser", "b" * 64, "offline-fixture", datetime.now(timezone.utc))
        constraints = {"context_tokens": ParameterConstraint(allowed_values=(1024,)),
                       "device": ParameterConstraint(allowed_values=("cpu",)),
                       "max_output_tokens": ParameterConstraint(minimum=1, maximum=128),
                       "temperature": ParameterConstraint(minimum=0, maximum=2),
                       "top_p": ParameterConstraint(minimum=0, maximum=1),
                       "seed": ParameterConstraint(minimum=-(2**63), maximum=2**63-1),
                       "stop": ParameterConstraint(),
                       "roles": ParameterConstraint(allowed_values=("user", "assistant", "system", "tool"))}
        capability = Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))
        self.capabilities = CapabilitySet({name: capability for name in
                               (CapabilityName.CHAT, CapabilityName.STREAMING, CapabilityName.EMBEDDINGS)})
        self.requests, self.loaded, self.bad_ps = [], True, False
        self.custom_reads = False
        self.custom = None
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False,
                                        transport=httpx.MockTransport(self.respond))
        self.provider = OllamaProvider(client=self.client, resolver=self.resolver,
                        implementation=self.implementation, capabilities={self.deployment.id: self.capabilities},
                        compatibility=OllamaCapabilities(), expected_version="0.6.5")
        self.addAsyncCleanup(self.provider.aclose)

    def context(self, *, seconds=2):
        return RequestContext("fixture-operation", time.monotonic() + seconds, threading.Event())

    def request(self, **kwargs):
        kwargs.setdefault("execution_generation", self.provider.generation)
        return InferenceRequest(self.model, (Message(MessageRole.USER, (TextPart("hello"), TextPart(" world"))),),
                                kwargs.pop("options", GenerationOptions(16)), kwargs.pop("context", self.context()), **kwargs)

    def completion(self, *, content="answer", done=True):
        return {"model": "fixture:latest", "message": {"role": "assistant", "content": content},
                "done": done, **({"done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2} if done else {})}

    def enable_structured(self):
        from kiron_common.local_inference.json_schema import KEYWORDS, TYPES
        constraints = {"formats": ParameterConstraint(allowed_values=("json_object", "json_schema")),
            "strict": ParameterConstraint(allowed_values=(False, True)),
            "schema_keywords": ParameterConstraint(allowed_values=tuple(KEYWORDS)),
            "schema_types": ParameterConstraint(allowed_values=tuple(TYPES)),
            "schema_variants": ParameterConstraint(allowed_values=("local_refs", "nullable", "any_of", "open_objects", "schema_additional_properties"))}
        values = dict(self.capabilities.by_name)
        values[CapabilityName.STRUCTURED_OUTPUT] = replace(values[CapabilityName.CHAT], constraints=constraints)
        self.provider._capabilities[self.deployment.id] = CapabilitySet(values)

    def schema_options(self, *, strict=True):
        schema = {"type": "object", "properties": {"color": {"$ref": "#/$defs/color"}},
            "$defs": {"color": {"type": "string", "enum": ["red", "blue"]}},
            "required": ["color"], "additionalProperties": False}
        return GenerationOptions(16, output_format=OutputFormat(OutputFormatKind.JSON_SCHEMA, schema, "color", strict))

    def resident(self, **overrides):
        return {"name": "fixture:latest", "model": "fixture:latest", "digest": "a" * 64,
                "size": 1024, "size_vram": 0, "context_length": 1024, **overrides}

    def respond(self, request):
        self.requests.append(request)
        if self.custom is not None and (request.method == "POST" or self.custom_reads):
            response = self.custom(request)
            if response is not None:
                return response
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.6.5"})
        if request.url.path == "/api/ps":
            models = [self.resident()] if self.loaded else []
            if self.bad_ps:
                models = [self.resident(size_vram=True)]
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture", "digest": "a" * 64}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json=self.completion())
        if request.url.path == "/api/generate":
            value = json.loads(request.content)
            if "prompt" not in value:
                self.loaded = value.get("keep_alive") != 0
                return httpx.Response(200, json={"done": True, "done_reason": "load"})
            return httpx.Response(200, json={"model": "fixture:latest", "response": "answer", "done": True,
                                          "done_reason": "length", "prompt_eval_count": 3, "eval_count": 2})
        raise AssertionError(f"Unexpected fixture endpoint: {request.url.path}")

    async def test_text_chat_preserves_order_think_false_native_sampling_and_exact_usage(self):
        sampling = SamplingOptions(temperature=0.2, top_p=0.9, seed=42, stop=("END",))
        result = await self.provider.chat(self.request(options=GenerationOptions(16, sampling)))
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello world"}])
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"], {"num_predict": 16, "temperature": 0.2, "top_p": 0.9,
                                              "seed": 42, "stop": ["END"], "num_ctx": 1024, "num_gpu": 0})
        self.assertIs(payload["truncate"], False)
        self.assertIs(payload["shift"], False)
        self.assertEqual(result.content, (TextPart("answer"),))
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(result.finish_reason, FinishReason.STOP)

    async def test_generate_uses_prompt_operation_without_chat_rewriting(self):
        request = GenerateRequest(self.model, "raw ordered prompt", GenerationOptions(16), self.context(), self.provider.generation)
        result = await self.provider.generate(request)
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["prompt"], request.prompt)
        self.assertNotIn("messages", payload)
        self.assertIs(payload["think"], False)
        self.assertEqual(result.finish_reason, FinishReason.LENGTH)

    async def test_structured_chat_and_generate_send_exact_native_format_and_validate_output(self):
        self.enable_structured()
        for options in (GenerationOptions(16, output_format=OutputFormat(OutputFormatKind.JSON_OBJECT)), self.schema_options()):
            for generate in (False, True):
                def respond(req):
                    value = self.completion(content='{"color":"red"}')
                    if generate:
                        value["response"] = value.pop("message")["content"]
                    return httpx.Response(200, json=value)
                self.custom = respond
                request = (GenerateRequest(self.model, "raw prompt", options, self.context(), self.provider.generation)
                           if generate else self.request(options=options))
                result = await (self.provider.generate(request) if generate else self.provider.chat(request))
                self.assertEqual(result.content, (TextPart('{"color":"red"}'),))
                payload = json.loads(self.requests[-1].content)
                self.assertIs(payload["think"], False)
                if options.output_format.kind is OutputFormatKind.JSON_OBJECT:
                    self.assertEqual(payload["format"], "json")
                else:
                    self.assertNotIn("$ref", json.dumps(payload["format"]))
                    self.assertEqual(payload["format"]["properties"]["color"]["enum"], ["red", "blue"])
                if generate:
                    self.assertEqual(payload["prompt"], "raw prompt")
                    self.assertNotIn("messages", payload)

    async def test_invalid_structured_output_never_returns_success_but_length_stays_partial(self):
        self.enable_structured()
        for content in ('{"color":"green"}', '{"color":"red","extra":1}', '{"color":"red","color":"blue"}',
                        '[]', '{"color":NaN}', '{"color":1e999}', '{"color":'):
            self.custom = lambda req: httpx.Response(200, json=self.completion(content=content))
            with self.subTest(content=content), self.assertRaises(LocalInferenceError) as caught:
                await self.provider.chat(self.request(options=self.schema_options()))
            self.assertEqual(caught.exception.failure.code.value, "provider_error")
        value = {**self.completion(content='{"color":'), "done_reason": "length"}
        self.custom = lambda req: httpx.Response(200, json=value)
        result = await self.provider.chat(self.request(options=self.schema_options()))
        self.assertIs(result.finish_reason, FinishReason.LENGTH)
        self.assertEqual(result.content, (TextPart('{"color":'),))

    async def test_structured_stream_validates_after_eof_before_usage_or_completion(self):
        self.enable_structured()
        for output, succeeds in (('{"color":"red"}', True), ('{"color":"green"}', False), ('{"color":', False)):
            frames = self.wire(self.completion(content=output[:8], done=False), self.completion(content=output[8:]))
            stream = ByteStream([frames[:19], frames[19:]])
            self.custom = lambda req: httpx.Response(200, stream=stream)
            events = [event async for event in self.provider.stream(self.request(options=self.schema_options()))]
            self.assertEqual("".join(event.text or "" for event in events), output)
            self.assertEqual(events[-1].kind, EventKind.COMPLETED if succeeds else EventKind.FAILED)
            self.assertEqual(any(event.kind is EventKind.USAGE for event in events), succeeds)
            self.assertTrue(stream.closed)
        for extra in (self.wire(self.completion(content="")), b'{"bad":'):
            stream = ByteStream([self.wire(self.completion(content='{"color":"red"}')), extra])
            self.custom = lambda req: httpx.Response(200, stream=stream)
            events = [event async for event in self.provider.stream(self.request(options=self.schema_options()))]
            self.assertEqual(events[-1].kind, EventKind.FAILED)
            self.assertFalse(any(event.kind in (EventKind.USAGE, EventKind.COMPLETED) for event in events))
        stream = ByteStream([self.wire({**self.completion(content='{"color":'), "done_reason": "length"})])
        self.custom = lambda req: httpx.Response(200, stream=stream)
        events = [event async for event in self.provider.stream(self.request(options=self.schema_options()))]
        self.assertIs(events[-1].finish_reason, FinishReason.LENGTH)

    async def test_missing_format_evidence_and_unverified_combinations_fail_before_all_io(self):
        for options, fields in ((self.schema_options(), {}),
                (GenerationOptions(16, reasoning=ReasoningOptions(effort="low")), {})):
            with self.assertRaises(LocalInferenceError):
                await self.provider.chat(self.request(options=options, **fields))
        self.enable_structured()
        values = dict(self.provider._capabilities[self.deployment.id].by_name)
        values[CapabilityName.FUNCTION_TOOLS] = replace(values[CapabilityName.CHAT], constraints={
            "tool_choice": ParameterConstraint(allowed_values=("auto", "none")),
            "strict": ParameterConstraint(allowed_values=(False,)), "max_tools": ParameterConstraint(maximum=64)})
        values[CapabilityName.REASONING] = values[CapabilityName.CHAT]
        self.provider._capabilities[self.deployment.id] = CapabilitySet(values)
        history = (Message(MessageRole.ASSISTANT, (TextPart("answer"),), reasoning=(ReasoningPart(ReasoningKind.TEXT, "private"),)),)
        cases = [self.request(options=self.schema_options(), tools=(ToolDefinition("color", {"type": "object"}),)),
                 replace(self.request(), messages=history),
                 self.request(options=GenerationOptions(16, reasoning=ReasoningOptions(effort="low")))]
        for request in cases:
            with self.subTest(request=request), self.assertRaises(LocalInferenceError):
                await self.provider.chat(request)
            with self.assertRaises(LocalInferenceError):
                [event async for event in self.provider.stream(request)]
        self.resolver.snapshot.assert_not_awaited()
        self.assertEqual(self.requests, [])

    async def test_schema_feature_rejection_and_sampling_preflight_precede_resolver_and_policy(self):
        self.enable_structured()
        values = dict(self.provider._capabilities[self.deployment.id].by_name)
        structured = values[CapabilityName.STRUCTURED_OUTPUT]
        for name, constraint in (("formats", ParameterConstraint(allowed_values=("json_object",))),
                ("strict", ParameterConstraint(allowed_values=(False,))),
                ("schema_variants", ParameterConstraint(allowed_values=()))):
            values[CapabilityName.STRUCTURED_OUTPUT] = replace(structured, constraints={**structured.constraints, name: constraint})
            self.provider._capabilities[self.deployment.id] = CapabilitySet(values)
            with self.subTest(name=name), self.assertRaises(LocalInferenceError):
                await self.provider.chat(self.request(options=self.schema_options()))
        self.provider.options_policy = mock.AsyncMock()
        for request in (self.request(options=GenerationOptions(129)),
                        GenerateRequest(self.model, "raw", self.schema_options(), self.context(), self.provider.generation)):
            with self.assertRaises(LocalInferenceError):
                await (self.provider.generate(request) if isinstance(request, GenerateRequest) else self.provider.chat(request))
        self.resolver.snapshot.assert_not_awaited()
        self.provider.options_policy.assert_not_awaited()
        self.assertEqual(self.requests, [])

    async def test_native_json_is_strict_for_both_nonstream_and_ndjson(self):
        valid = json.dumps(self.completion()).encode()
        invalid = [b'{"model":"wrong",' + valid[1:], valid[:-1] + b',"extension":NaN}',
            valid[:-1] + b',"extension":1e999}', valid[:-1] + b',"extension":"\\ud800"}',
            valid[:-1] + b',"extension":' + b'['*33 + b'0' + b']'*33 + b'}', b'\xff', b'[]']
        for raw in invalid:
            with self.subTest(raw=raw):
                self.custom = lambda req: httpx.Response(200, content=raw)
                with self.assertRaises(LocalInferenceError) as caught:
                    await self.provider.chat(self.request())
                self.assertEqual(caught.exception.failure.code.value, "provider_error")
                stream = ByteStream([raw + b'\n'])
                self.custom = lambda req: httpx.Response(200, stream=stream)
                events = [event async for event in self.provider.stream(self.request())]
                self.assertEqual(events[-1].kind, EventKind.FAILED)
                self.assertFalse(any(event.kind in (EventKind.USAGE, EventKind.COMPLETED) for event in events))
                self.assertTrue(stream.closed)

    async def test_unknown_or_reasoning_native_content_is_never_silently_dropped(self):
        for field, value in (("thinking", "secret"), ("thinking", False), ("thinking", None),
                             ("images", []), ("reasoning", "secret"), ("unexpected", None)):
            body = self.completion()
            body["message"][field] = value
            self.custom = lambda req: httpx.Response(200, json=body)
            with self.subTest(field=field, value=value), self.assertRaises(LocalInferenceError):
                await self.provider.chat(self.request())
        body = self.completion()
        body["tool_calls"] = []
        self.custom = lambda req: httpx.Response(200, json=body)
        with self.assertRaises(LocalInferenceError):
            await self.provider.chat(self.request())

    async def test_capability_projection_never_grants_reasoning_or_vision_from_injected_flags(self):
        self.enable_structured()
        values = dict(self.provider._capabilities[self.deployment.id].by_name)
        for name in (CapabilityName.REASONING, CapabilityName.VISION, CapabilityName.FUNCTION_TOOLS, CapabilityName.PARALLEL_TOOLS):
            values[name] = values[CapabilityName.CHAT]
        self.provider._capabilities[self.deployment.id] = CapabilitySet(values)
        advertised = await self.provider.capabilities(self.deployment)
        for name in (CapabilityName.REASONING, CapabilityName.VISION):
            self.assertIs(advertised.by_name[name].status, CapabilityStatus.UNSUPPORTED)
        for name in (CapabilityName.FUNCTION_TOOLS, CapabilityName.PARALLEL_TOOLS, CapabilityName.STRUCTURED_OUTPUT):
            self.assertTrue(advertised.supports(name, self.deployment, self.implementation))
        self.assertEqual(self.requests, [])

    async def test_native_aggregate_response_limit_fails_and_closes_stream(self):
        for streaming in (False, True):
            stream = ByteStream([self.wire(self.completion(content="x"*30, done=False)),
                                 self.wire(self.completion(content="y"*30))])
            self.custom = lambda req: httpx.Response(200, stream=stream)
            with mock.patch("ollama_provider.MAX_RESPONSE", 200):
                if streaming:
                    events = [event async for event in self.provider.stream(self.request())]
                    self.assertEqual(events[-1].kind, EventKind.FAILED)
                    self.assertFalse(any(event.kind is EventKind.COMPLETED for event in events))
                else:
                    with self.assertRaises(LocalInferenceError):
                        await self.provider.chat(self.request())
            self.assertTrue(stream.closed)

    async def test_missing_or_mismatched_capability_evidence_blocks_before_io(self):
        for capability in (CapabilitySet(), self.capabilities.for_deployment(
                replace(self.deployment, configuration_fingerprint="d" * 64), self.implementation)):
            self.provider._capabilities[self.deployment.id] = capability
            with self.assertRaises(LocalInferenceError):
                await self.provider.chat(self.request())
        self.assertEqual(self.requests, [])

    async def test_failed_compatibility_is_not_bypassed_by_feature_evidence(self):
        self.provider.compatibility = replace(OllamaCapabilities(), think_false=CompatResult(False))
        with self.assertRaises(LocalInferenceError):
            await self.provider.chat(self.request())
        self.assertEqual(self.requests, [])

    async def test_no_penalty_approximation_or_unverified_tool_mapping(self):
        requests = [self.request(options=GenerationOptions(16, SamplingOptions(frequency_penalty=0.5))),
                    self.request(tools=(ToolDefinition("weather", {"type": "object"}),))]
        for request in requests:
            with self.assertRaises(LocalInferenceError):
                await self.provider.chat(request)
        self.assertEqual(self.requests, [])

    async def test_stale_snapshot_rejected_before_backend_io(self):
        self.snapshot = ResolverSnapshot("d" * 64, {self.deployment.id: self.deployment}, {})
        with self.assertRaises(LocalInferenceError):
            await self.provider.chat(self.request())
        self.assertEqual(self.requests, [])

    async def test_missing_or_stale_execution_generation_rejected_for_every_operation(self):
        async def collect(request):
            return [event async for event in self.provider.stream(request)]
        model = replace(self.model, profile_id="embedding", embedding_role=EmbeddingRole.QUERY)
        old = self.provider.generation
        self.provider._new_epoch()
        for generation in (None, old):
            chat = self.request(execution_generation=generation)
            generated = GenerateRequest(self.model, "prompt", GenerationOptions(16), self.context(), generation)
            embedded = EmbeddingRequest(model, ("text",), self.context(), generation)
            for method, request in ((self.provider.chat, chat), (collect, chat),
                                    (self.provider.generate, generated), (self.provider.embed, embedded)):
                with self.subTest(method=method, generation=generation), self.assertRaises(LocalInferenceError) as caught:
                    await method(request)
                self.assertEqual(caught.exception.failure.code.value, "conflict")
        self.assertEqual(self.requests, [])

    async def test_execution_epoch_changed_during_preparation_never_sends_native_request(self):
        request = self.request()
        async def rotate(_deployment, _options, _context):
            self.provider._new_epoch()
        self.provider.options_policy = rotate
        with self.assertRaises(LocalInferenceError) as caught:
            await self.provider.chat(request)
        self.assertEqual(caught.exception.failure.code.value, "conflict")
        self.assertEqual(self.requests, [])

    async def test_execution_epoch_change_during_response_rejects_completion(self):
        request = self.request()
        def rotated(native):
            self.provider._new_epoch()
            return httpx.Response(200, json=self.completion())
        self.custom = rotated
        with self.assertRaises(LocalInferenceError) as caught:
            await self.provider.chat(request)
        self.assertEqual(caught.exception.failure.code.value, "conflict")

    async def test_existing_force_cpu_policy_requires_compat_flag_and_cannot_change_sampling(self):
        async def force_cpu(deployment, options, context):
            options["num_gpu"] = 0
        self.provider.options_policy = force_cpu
        await self.provider.chat(self.request())
        self.assertEqual(json.loads(self.requests[-1].content)["options"]["num_gpu"], 0)
        async def corrupt(deployment, options, context):
            options["num_predict"] = 9000
        self.provider.options_policy = corrupt
        before = len(self.requests)
        with self.assertRaises(LocalInferenceError):
            await self.provider.chat(self.request())
        self.assertEqual(len(self.requests), before)

    async def test_positive_capability_requires_exact_resident_profile_without_coldload_grant(self):
        original = self.capabilities
        self.assertIsNone(self.deployment.resource_profile)
        for key, value in (("context_tokens", None), ("device", None),
                           ("context_tokens", ParameterConstraint(minimum=1, maximum=1024)),
                           ("context_tokens", ParameterConstraint(allowed_values=(True,))),
                           ("context_tokens", ParameterConstraint(allowed_values=(1024, 4096))),
                           ("device", ParameterConstraint(allowed_values=("gpu",)))):
            constraints = dict(original.by_name[CapabilityName.CHAT].constraints)
            if value is None:
                constraints.pop(key)
            else:
                constraints[key] = value
            caps = CapabilitySet({**original.by_name,
                CapabilityName.CHAT: replace(original.by_name[CapabilityName.CHAT], constraints=constraints)})
            self.provider._capabilities[self.deployment.id] = caps
            with self.subTest(key=key, value=value):
                effective = await self.provider.capabilities(self.deployment)
                self.assertFalse(effective.supports(CapabilityName.CHAT, self.deployment, self.implementation))
                observation = await self.provider.health(self.context())
                self.assertEqual(observation.models[self.deployment.id].state, RuntimeState.UNKNOWN)
                with self.assertRaises(LocalInferenceError):
                    await self.provider.chat(self.request())
                self.assertFalse(any(req.method == "POST" for req in self.requests))

    async def test_resident_context_device_drift_and_absence_reject_before_inference_post(self):
        self.custom_reads = True
        cases = ({"context_length": 4096}, {"size_vram": 512},
                 {"context_length": None}, {"context_length": True}, {"digest": "f" * 64}, None)
        for observed in cases:
            def changed(req):
                if req.url.path == "/api/ps":
                    return httpx.Response(200, json={"models": [] if observed is None else [self.resident(**observed)]})
            self.custom = changed
            self.requests.clear()
            with self.subTest(observed=observed):
                observation = await self.provider.health(self.context())
                self.assertIsNot(observation.models[self.deployment.id].state, RuntimeState.LOADED)
                with self.assertRaises(LocalInferenceError) as caught:
                    await self.provider.chat(self.request())
                self.assertEqual(caught.exception.failure.code.value, "conflict")
                self.assertFalse(any(req.method == "POST" for req in self.requests))

    async def test_residency_is_rechecked_after_option_policy_before_each_generation_post(self):
        self.custom_reads = True
        drift = False
        def observe(req):
            if req.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [self.resident(size_vram=512 if drift else 0)]})
        self.custom = observe
        async def change_after_earlier_health(deployment, options, context):
            nonlocal drift
            drift = True
        self.provider.options_policy = change_after_earlier_health
        for operation in ("chat", "generate", "stream"):
            drift = False
            healthy = await self.provider.health(self.context())
            self.assertIs(healthy.models[self.deployment.id].state, RuntimeState.LOADED)
            self.requests.clear()
            if operation == "stream":
                events = [event async for event in self.provider.stream(self.request())]
                self.assertIs(events[-1].kind, EventKind.FAILED)
                self.assertEqual(events[-1].error.code.value, "conflict")
            else:
                request = self.request() if operation == "chat" else GenerateRequest(
                    self.model, "prompt", GenerationOptions(16), self.context(), self.provider.generation)
                with self.assertRaises(LocalInferenceError) as caught:
                    await getattr(self.provider, operation)(request)
                self.assertEqual(caught.exception.failure.code.value, "conflict")
            self.assertFalse(any(req.method == "POST" for req in self.requests))

    async def test_native_context_rejection_requires_exact_body_and_complete_eof(self):
        from ollama_provider import CONTEXT_OVERFLOW
        for content, expected in (({"error": CONTEXT_OVERFLOW}, "context_length_exceeded"),
                                  ({"error": "private backend detail"}, "provider_unavailable")):
            self.custom = lambda req: httpx.Response(400, json=content)
            with self.subTest(content=content), self.assertRaises(LocalInferenceError) as caught:
                await self.provider.chat(self.request())
            self.assertEqual(caught.exception.failure.code.value, expected)
            self.assertNotIn("private backend detail", caught.exception.failure.message)
        error = self.wire({"error": CONTEXT_OVERFLOW, "status": 400})
        for raw, expected in ((error, "context_length_exceeded"),
                              (error + error, "provider_error"),
                              (error.rstrip(b"\n"), "provider_error")):
            stream = ByteStream([raw])
            self.custom = lambda req: httpx.Response(200, stream=stream)
            events = [event async for event in self.provider.stream(self.request())]
            self.assertEqual(events[-1].error.code.value, expected)
            self.assertFalse(any(e.kind in (EventKind.USAGE, EventKind.COMPLETED) for e in events))
            self.assertTrue(stream.closed)

    async def test_stream_read_timeout_before_or_after_error_frame_remains_timeout(self):
        from ollama_provider import CONTEXT_OVERFLOW
        class TimeoutStream(ByteStream):
            async def __aiter__(self):
                for block in self.blocks:
                    yield block
                raise httpx.ReadTimeout("private transport detail")
        for blocks in ([], [self.wire({"error": CONTEXT_OVERFLOW, "status": 400})],
                       [self.wire(self.completion(content="partial", done=False))]):
            stream = TimeoutStream(blocks)
            self.custom = lambda req: httpx.Response(200, stream=stream)
            events = [event async for event in self.provider.stream(self.request())]
            self.assertEqual(events[-1].error.code.value, "timeout")
            self.assertNotIn("private", events[-1].error.message)
            self.assertTrue(stream.closed)
        def header_timeout(req):
            raise httpx.ReadTimeout("private header detail")
        self.custom = header_timeout
        events = [event async for event in self.provider.stream(self.request())]
        self.assertEqual(events[-1].error.code.value, "timeout")

    async def test_resident_check_timeout_keeps_classification_before_generation_invalidation(self):
        self.custom_reads = True
        for path in ("/api/version", "/api/ps"):
            def timeout(req):
                if req.url.path == path:
                    raise httpx.ReadTimeout("private residency detail")
            self.custom = timeout
            for operation in ("chat", "generate", "stream"):
                self.requests.clear()
                with self.subTest(path=path, operation=operation):
                    if operation == "stream":
                        events = [event async for event in self.provider.stream(self.request())]
                        error = events[-1].error
                    else:
                        request = self.request() if operation == "chat" else GenerateRequest(
                            self.model, "prompt", GenerationOptions(16), self.context(), self.provider.generation)
                        with self.assertRaises(LocalInferenceError) as caught:
                            await getattr(self.provider, operation)(request)
                        error = caught.exception.failure
                    self.assertEqual(error.code.value, "timeout")
                    self.assertNotIn("private", error.message)
                    self.assertFalse(any(req.method == "POST" for req in self.requests))

    async def test_health_requires_pinned_resident_identity_and_real_vram_number(self):
        self.custom_reads = True
        self.loaded = True
        observation = await self.provider.health(self.context())
        self.assertEqual(observation.models[self.deployment.id].state, RuntimeState.LOADED)
        self.assertTrue(observation.generation.boot_id.startswith("ollama-probe:"))
        self.assertIsNone(observation.generation.process_id)
        self.custom = lambda req: httpx.Response(200, json={"models": [self.resident(digest="f" * 64)]}) if req.url.path == "/api/ps" else None
        observation = await self.provider.health(self.context())
        self.assertEqual(observation.models[self.deployment.id].state, RuntimeState.UNKNOWN)
        self.custom, self.bad_ps = None, True
        observation = await self.provider.health(self.context())
        self.assertEqual(observation.health, ProviderHealth.UNAVAILABLE)
        self.assertEqual(observation.models[self.deployment.id].state, RuntimeState.UNKNOWN)

    async def test_outage_rotates_probe_epoch_without_reporting_false_unloaded(self):
        self.custom_reads = True
        self.loaded = True
        before = await self.provider.health(self.context())
        def failed(req):
            raise httpx.ConnectError("offline", request=req)
        self.custom = failed
        after = await self.provider.health(self.context())
        self.assertNotEqual(before.generation, after.generation)
        self.assertEqual(after.models[self.deployment.id].state, RuntimeState.UNKNOWN)

    async def test_discovery_failure_is_an_isolated_typed_result(self):
        self.custom_reads = True
        inventory = await self.provider.discover(self.context())
        self.assertEqual(inventory.models[0].reference, "fixture:latest")
        self.custom = lambda req: httpx.Response(503)
        inventory = await self.provider.discover(self.context())
        self.assertIsNotNone(inventory.error)
        self.assertEqual(inventory.models, ())

    async def test_load_and_unload_are_idempotent_only_after_ps_confirmation(self):
        self.loaded = False
        generation = (await self.provider.health(self.context())).generation
        kwargs = dict(snapshot_revision=self.snapshot.revision, expected_generation=generation, context=self.context())
        result = await self.provider.load(self.deployment, **kwargs)
        self.assertTrue(result.changed)
        posts = len([req for req in self.requests if req.method == "POST"])
        self.assertFalse((await self.provider.load(self.deployment, **kwargs)).changed)
        self.assertEqual(len([req for req in self.requests if req.method == "POST"]), posts)
        result = await self.provider.unload(self.deployment, **kwargs)
        self.assertEqual(result.observation.models[self.deployment.id].state, RuntimeState.UNLOADED)
        payload = json.loads([req for req in self.requests if req.method == "POST"][-1].content)
        self.assertEqual(payload["keep_alive"], 0)
        self.assertIs(payload["think"], False)

    async def test_load_does_not_implicitly_evict_unregistered_resident(self):
        self.custom_reads = True
        self.custom = lambda req: httpx.Response(200, json={"models": [self.resident(name="foreign:latest", model="foreign:latest")]}) if req.url.path == "/api/ps" else None
        generation = (await self.provider.health(self.context())).generation
        with self.assertRaises(LocalInferenceError):
            await self.provider.load(self.deployment, snapshot_revision=self.snapshot.revision,
                                     expected_generation=generation, context=self.context())
        self.assertFalse(any(req.method == "POST" for req in self.requests))

    async def test_http200_unload_error_is_rejected_and_not_confirmed(self):
        self.loaded = True
        generation = (await self.provider.health(self.context())).generation
        self.custom = lambda req: httpx.Response(200, json={"error": "fixture backend rejection"}) if req.method == "POST" else None
        with self.assertRaises(LocalInferenceError):
            await self.provider.unload(self.deployment, snapshot_revision=self.snapshot.revision,
                                       expected_generation=generation, context=self.context())
        self.assertEqual(self.requests[-1].method, "POST")

    async def test_unload_waits_for_real_absence_and_has_finite_deadline(self):
        self.loaded = True
        generation = (await self.provider.health(self.context())).generation
        self.custom = lambda req: httpx.Response(200, json={"done": True}) if req.method == "POST" else None
        with self.assertRaises(LocalInferenceError) as error:
            await self.provider.unload(self.deployment, snapshot_revision=self.snapshot.revision,
                                       expected_generation=generation, context=self.context(seconds=0.03))
        self.assertEqual(error.exception.failure.code.value, "timeout")

    async def test_unload_done_keeps_polling_until_delayed_runner_disappears(self):
        generation = (await self.provider.health(self.context())).generation
        self.custom_reads = True
        polls = []
        acknowledged = False

        def delayed(request):
            nonlocal acknowledged
            if request.method == "POST":
                acknowledged = True
                return httpx.Response(200, json={"done": True})
            if request.url.path == "/api/ps" and acknowledged:
                polls.append(len(polls))
                return httpx.Response(200, json={
                    "models": [self.resident()] if len(polls) < 3 else []})

        self.custom = delayed
        result = await self.provider.unload(
            self.deployment, snapshot_revision=self.snapshot.revision,
            expected_generation=generation, context=self.context())
        self.assertEqual(len(polls), 3)
        self.assertTrue(result.changed)
        self.assertEqual(result.observation.models[self.deployment.id].state, RuntimeState.UNLOADED)
        self.assertEqual(len([r for r in self.requests if r.method == "POST"]), 1)

    def wire(self, *frames):
        return b"".join(json.dumps(frame).encode() + b"\n" for frame in frames)

    async def test_stream_reassembles_split_ndjson_and_emits_one_terminal_usage(self):
        data = self.wire(self.completion(content="hel", done=False), self.completion(content="lo"))
        stream = ByteStream([data[:13], data[13:47], data[47:]])
        self.custom = lambda req: httpx.Response(200, stream=stream)
        events = [event async for event in self.provider.stream(self.request())]
        validate_event_sequence(events)
        self.assertEqual("".join(event.text or "" for event in events), "hello")
        self.assertEqual([event.kind for event in events], [EventKind.STARTED, EventKind.TEXT_DELTA,
                         EventKind.TEXT_DELTA, EventKind.USAGE, EventKind.COMPLETED])
        self.assertTrue(stream.closed)

    async def test_truncated_stream_fails_without_completed_or_fabricated_usage(self):
        stream = ByteStream([self.wire(self.completion(content="partial", done=False))])
        self.custom = lambda req: httpx.Response(200, stream=stream)
        events = [event async for event in self.provider.stream(self.request())]
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertNotIn(EventKind.USAGE, [event.kind for event in events])
        self.assertTrue(stream.closed)

    async def test_cancellation_closes_transport_without_unloading_or_claiming_request_end(self):
        stream = ByteStream([self.wire(self.completion(content="partial", done=False))], stall=True)
        self.custom = lambda req: httpx.Response(200, stream=stream)
        context = self.context()
        events = []
        async for event in self.provider.stream(self.request(context=context)):
            events.append(event)
            if event.kind is EventKind.TEXT_DELTA:
                context.cancellation.set()
        self.assertEqual(events[-1].kind, EventKind.CANCELLED)
        self.assertTrue(stream.closed)
        self.assertFalse(any(req.url.path == "/api/generate" for req in self.requests))
        self.assertFalse(await self.provider.wait_request_end(self.deployment, generation=self.provider.generation,
                                                             context=context))

    async def test_stream_total_deadline_covers_waiting_for_response_headers(self):
        async def delayed(req):
            await asyncio.sleep(1)
            return httpx.Response(200, content=b"")
        await self.provider.aclose()
        client = httpx.AsyncClient(base_url="http://127.0.0.1:11435", trust_env=False,
                                  transport=httpx.MockTransport(delayed))
        self.provider.client, self.provider._closed = client, False
        events = [event async for event in self.provider.stream(self.request(context=self.context(seconds=0.03)))]
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertEqual(events[-1].error.code.value, "timeout")

    async def test_embedding_requires_explicit_role_formatter_and_exact_usage(self):
        model = ResolvedModel("embedding.query", self.deployment, "embedding-profile", EmbeddingRole.QUERY, self.snapshot.revision)
        self.snapshot = ResolverSnapshot(self.snapshot.revision, self.snapshot.deployments, {model.public_model_id: model})
        request = EmbeddingRequest(model, ("hello", "world"), self.context(), self.provider.generation)
        with self.assertRaises(LocalInferenceError):
            await self.provider.embed(request)
        self.assertEqual(self.requests, [])
        self.provider.embedding_formatter = mock.AsyncMock(return_value=("query: hello", "query: world"))
        self.custom = lambda req: httpx.Response(200, json={"model": "fixture:latest",
                 "embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 7})
        result = await self.provider.embed(request)
        self.assertEqual(result.usage.input_tokens, 7)
        self.assertEqual(result.usage.output_tokens, 0)
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["input"], ["query: hello", "query: world"])
        self.assertIs(payload["truncate"], False)
        for model in ("other:latest", None):
            with self.subTest(model=model):
                value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 7}
                if model is not None:
                    value["model"] = model
                self.custom = lambda req: httpx.Response(200, json=value)
                with self.assertRaises(LocalInferenceError):
                    await self.provider.embed(request)

    async def test_unconfigured_service_control_cannot_run_arbitrary_commands(self):
        with self.assertRaises(LocalInferenceError):
            await self.provider.start(self.context())
        with self.assertRaises(LocalInferenceError):
            await self.provider.stop(self.provider.generation, self.context())
        self.assertEqual(self.requests, [])

    async def test_close_owned_client_exactly_once(self):
        with mock.patch.object(self.client, "aclose", wraps=self.client.aclose) as close:
            await self.provider.aclose()
            await self.provider.aclose()
            close.assert_awaited_once()

    async def test_service_control_conflicts_with_existing_lifecycle_operation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def service_start():
            entered.set()
            await release.wait()
            return True

        self.provider.service_control = SimpleNamespace(start=service_start, stop=mock.AsyncMock())
        task = asyncio.create_task(self.provider.start(self.context()))
        await entered.wait()
        try:
            with self.assertRaises(LocalInferenceError):
                await self.provider.stop(self.provider.generation, self.context())
            with self.assertRaises(LocalInferenceError):
                await self.provider.load(self.deployment, snapshot_revision=self.snapshot.revision,
                                         expected_generation=self.provider.generation, context=self.context())
            self.provider.service_control.stop.assert_not_awaited()
        finally:
            release.set()
            await task


if __name__ == "__main__":
    unittest.main()
