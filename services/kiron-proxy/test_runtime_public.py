import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from kiron_common.local_inference import (
    ArtifactIdentity, Capability, CapabilityEvidence, CapabilityName, CapabilitySet,
    CapabilityStatus, DeploymentObservation, DiscoveredModel, DiscoverySnapshot,
    ErrorCode, GenerationOptions, InferenceRequest, LocalInferenceError, Message, ModelLifecycleOperation,
    MessageRole, OutputFormat, OutputFormatKind, ParameterConstraint, ProviderHealth, ProviderObservation, RequestContext,
    ReasoningOptions, ReasoningKind, ReasoningPart, SamplingOptions, ToolDefinition,
    ResolvedDeployment, ResolvedModel, ResolverSnapshot, ResourceProfile, RuntimeGeneration,
    RuntimeFailure, RuntimeImplementation, RuntimeTimeouts, TextPart,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType
from kiron_common.model_state import RuntimeState
from runtime_service import RuntimeService
from kiron_common.local_inference.json_schema import compile_schema, schema_features


class PublicRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = RequestContext("public", time.monotonic() + 2, asyncio.Event())
        self.implementation = RuntimeImplementation("runtime", None, "parser")
        self.generation = RuntimeGeneration("boot", "child")
        self.deployment = ResolvedDeployment("deployment", BackendType.PRISM, "/model.gguf",
            ArtifactIdentity(ArtifactType.LOCAL, ArtifactFormat.GGUF, "a" * 64, 100),
            LoaderType.PRISM_GGUF, ResourceProfile("measured", 1024, 128, 128, 1, 4, 40, False, 40, 20), "b" * 64)
        self.model = ResolvedModel("canonical", self.deployment, None, None, "c" * 64, created=123, canonical_model_id="canonical")
        self.alias = replace(self.model, public_model_id="alias", canonical_model_id="canonical")
        evidence = CapabilityEvidence("runtime", self.deployment.artifact_identity.fingerprint,
            "a" * 64, None, None, "parser", "b" * 64, "fixture test", datetime.now(timezone.utc))
        constraints = {
            "roles": ParameterConstraint(allowed_values=("user", "assistant", "system")),
            "max_output_tokens": ParameterConstraint(minimum=1, maximum=128),
            "default_max_output_tokens": ParameterConstraint(allowed_values=(16,)),
            "token_budget": ParameterConstraint(allowed_values=("max_tokens", "max_completion_tokens")),
            "temperature": ParameterConstraint(minimum=0, maximum=1),
        }
        self.capability = Capability(CapabilityStatus.SUPPORTED, constraints, (evidence,))
        self.capabilities = CapabilitySet({CapabilityName.CHAT: self.capability, CapabilityName.STREAMING: self.capability})
        self.observation = ProviderObservation(BackendType.PRISM, self.generation,
            datetime.now(timezone.utc), ProviderHealth.AVAILABLE, {})
        self.discovery = DiscoverySnapshot(BackendType.PRISM, "snapshot", datetime.now(timezone.utc),
            (DiscoveredModel("/model.gguf", self.deployment.artifact_identity, True),))
        self.provider = SimpleNamespace(implementation=self.implementation,
            model_lifecycle_operations=frozenset((ModelLifecycleOperation.LOAD, ModelLifecycleOperation.UNLOAD)),
            health=mock.AsyncMock(side_effect=lambda _: self.observation),
            discover=mock.AsyncMock(side_effect=lambda _: self.discovery),
            capabilities=mock.AsyncMock(side_effect=lambda _: self.capabilities), aclose=mock.AsyncMock(),
            validate_request=mock.Mock(),
            load=mock.AsyncMock(), chat=mock.AsyncMock())
        self.snapshot = ResolverSnapshot("c" * 64, {self.deployment.id: self.deployment},
            {model.public_model_id: model for model in (self.model, self.alias)})
        self.resolver = SimpleNamespace(snapshot=mock.AsyncMock(side_effect=lambda: self.snapshot))
        self.store = mock.Mock()
        self.service = RuntimeService(resolver=self.resolver, providers={BackendType.PRISM: self.provider},
            admission=self.store, measure=mock.Mock(), timeouts=RuntimeTimeouts(1, 1, 1, 1, 1, 1, 1))
        self.addAsyncCleanup(self.service.aclose)

    def parsed(self, **parameters):
        messages = (Message(MessageRole.USER, (TextPart("Hi"),)),)
        return SimpleNamespace(messages=messages, stream=False, explicit_parameters=parameters,
            to_request=lambda model, context, default: InferenceRequest(model, messages,
                GenerationOptions(parameters.get("max_completion_tokens", default)), context))

    async def test_list_canonicalizes_aliases_and_detail_returns_identical_stable_record(self):
        rows = await self.service.public_models(self.context)
        self.assertEqual(rows, [{"id": "canonical", "object": "model", "created": 123, "owned_by": "prism"}])
        model, row = await self.service.public_model("alias", self.context)
        self.assertEqual(model, self.alias)
        self.assertEqual(row, rows[0])
        self.store.reserve.assert_not_called()
        self.provider.load.assert_not_called()

    async def test_unknown_id_and_known_unavailable_are_distinct(self):
        with self.assertRaises(LocalInferenceError) as missing:
            await self.service.public_model("absent", self.context)
        self.assertEqual(missing.exception.failure.code, ErrorCode.MODEL_NOT_FOUND)
        self.discovery = replace(self.discovery, models=())
        with self.assertRaises(LocalInferenceError) as unavailable:
            await self.service.public_model("canonical", self.context)
        self.assertEqual(unavailable.exception.failure.code, ErrorCode.PROVIDER_UNAVAILABLE)

    async def test_empty_registration_is_empty_but_all_unavailable_is_error(self):
        self.provider.health.side_effect = RuntimeError("offline")
        with self.assertRaises(LocalInferenceError):
            await self.service.public_models(self.context)
        self.snapshot = ResolverSnapshot("c" * 64, {}, {})
        self.assertEqual(await self.service.public_models(self.context), [])

    async def test_one_provider_outage_does_not_hide_healthy_provider(self):
        other = replace(self.deployment, id="other", provider=BackendType.OLLAMA, loader=LoaderType.OLLAMA)
        other_model = replace(self.model, public_model_id="other", canonical_model_id="other", deployment=other)
        self.snapshot = ResolverSnapshot("c" * 64,
            {self.deployment.id: self.deployment, other.id: other}, {"canonical": self.model, "other": other_model})
        for error in (RuntimeError("offline"),
                      LocalInferenceError(RuntimeFailure(ErrorCode.TIMEOUT, "local timeout")),
                      LocalInferenceError(RuntimeFailure(ErrorCode.CANCELLED, "local cancellation"))):
            with self.subTest(error=error):
                self.service.providers[BackendType.OLLAMA] = SimpleNamespace(health=mock.AsyncMock(side_effect=error),
                    aclose=mock.AsyncMock())
                self.assertEqual([row["id"] for row in await self.service.public_models(self.context)], ["canonical"])

    async def test_wrong_artifact_and_stale_capability_evidence_are_not_published(self):
        self.discovery = replace(self.discovery, models=(replace(self.discovery.models[0],
            artifact_identity=replace(self.deployment.artifact_identity, sha256="d" * 64)),))
        with self.assertRaises(LocalInferenceError):
            await self.service.public_models(self.context)
        self.discovery = replace(self.discovery, models=(DiscoveredModel("/model.gguf", self.deployment.artifact_identity, True),))
        self.provider.implementation = RuntimeImplementation("changed runtime", None, "parser")
        with self.assertRaises(LocalInferenceError):
            await self.service.public_models(self.context)

    async def test_parameter_limits_and_roles_fail_before_lifecycle_or_inference(self):
        for parameters, code in (({"temperature": 1.5}, ErrorCode.UNSUPPORTED_VALUE),
                                  ({"presence_penalty": 0.0}, ErrorCode.UNSUPPORTED_PARAMETER),
                                  ({"max_completion_tokens": 129}, ErrorCode.UNSUPPORTED_VALUE)):
            with self.subTest(parameters=parameters), self.assertRaises(LocalInferenceError) as raised:
                await self.service.validate_chat(self.parsed(**parameters), self.model, self.context)
            self.assertEqual(raised.exception.failure.code, code)
        parsed = self.parsed()
        parsed.messages = (Message(MessageRole.DEVELOPER, (TextPart("Hi"),)),)
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.validate_chat(parsed, self.model, self.context)
        self.assertEqual(raised.exception.failure.parameter, "messages[0].role")
        self.provider.load.assert_not_called()
        self.provider.chat.assert_not_called()
        self.store.reserve.assert_not_called()

    async def test_explicit_profile_default_and_token_budget_semantics(self):
        request = await self.service.validate_chat(self.parsed(), self.model, self.context)
        self.assertEqual(request.options.max_output_tokens, 16)
        constraints = dict(self.capability.constraints)
        constraints["token_budget"] = ParameterConstraint(allowed_values=("max_completion_tokens",))
        self.capabilities = CapabilitySet({CapabilityName.CHAT: replace(self.capability, constraints=constraints)})
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.validate_chat(self.parsed(max_tokens=4), self.model, self.context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_VALUE)

    async def test_startable_provider_is_listed_without_starting_it(self):
        self.observation = ProviderObservation(BackendType.PRISM, None, datetime.now(timezone.utc), ProviderHealth.STARTABLE)
        self.assertEqual(len(await self.service.public_models(self.context)), 1)
        self.provider.load.assert_not_called()

    async def test_validated_snapshot_is_retained_when_registry_changes(self):
        model = await self.service.resolve("alias", self.context)
        await self.service.validate_chat(self.parsed(), model, self.context)
        self.provider.health.assert_not_called()
        self.provider.discover.assert_not_called()
        self.snapshot = ResolverSnapshot("d" * 64, {}, {})
        row = await self.service.public_model_for(model, self.context)
        self.assertEqual(row["id"], "canonical")
        self.assertEqual(self.resolver.snapshot.await_count, 1)

    async def test_deadline_and_cancellation_are_not_reclassified_as_provider_outage(self):
        async def blocked(_):
            await asyncio.Event().wait()
        self.provider.health.side_effect = blocked
        for cancelled, code in ((False, ErrorCode.TIMEOUT), (True, ErrorCode.CANCELLED)):
            with self.subTest(cancelled=cancelled):
                event = asyncio.Event()
                if cancelled:
                    event.set()
                context = RequestContext("bounded", time.monotonic() + .02, event)
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.public_models(context)
                self.assertEqual(raised.exception.failure.code, code)

    async def test_capability_failure_of_one_deployment_does_not_hide_another(self):
        other = replace(self.deployment, id="broken")
        other_model = replace(self.model, public_model_id="broken", canonical_model_id="broken", deployment=other)
        self.snapshot = ResolverSnapshot("c" * 64,
            {self.deployment.id: self.deployment, other.id: other}, {"canonical": self.model, "broken": other_model})
        async def capabilities(deployment):
            if deployment.id == "broken":
                raise ValueError("invalid evidence")
            return self.capabilities
        self.provider.capabilities.side_effect = capabilities
        self.assertEqual([row["id"] for row in await self.service.public_models(self.context)], ["canonical"])

    async def test_inexecutable_chat_profiles_are_not_advertised(self):
        invalid = [
            {key: value for key, value in self.capability.constraints.items() if key != omitted}
            for omitted in ("roles", "default_max_output_tokens", "max_output_tokens")
        ]
        invalid += [{**self.capability.constraints, "default_max_output_tokens": ParameterConstraint(allowed_values=(129,))},
                    {**self.capability.constraints, "roles": ParameterConstraint(allowed_values=("tool",))}]
        for constraints in invalid:
            with self.subTest(constraints=constraints):
                self.capabilities = CapabilitySet({CapabilityName.CHAT: replace(self.capability, constraints=constraints)})
                with self.assertRaises(LocalInferenceError):
                    await self.service.public_models(self.context)
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.validate_chat(self.parsed(), self.model, self.context)
                self.assertEqual(raised.exception.failure.code, ErrorCode.INVALID_CONFIGURATION)
        self.provider.load.assert_not_called()
        self.provider.chat.assert_not_called()

    def feature_parsed(self, *, output=None, effort=None, stop=(), tools=(), history=False):
        parsed = self.parsed()
        if history:
            parsed.messages = (Message(MessageRole.ASSISTANT, (TextPart("Answer"),),
                reasoning=(ReasoningPart(ReasoningKind.TEXT, "Prior thought"),)),)
        parsed.explicit_parameters = {} if effort is None else {"reasoning_effort": effort}
        parsed.to_request = lambda model, context, default: InferenceRequest(model, parsed.messages,
            GenerationOptions(default, sampling=SamplingOptions(stop=stop),
                              reasoning=ReasoningOptions(effort=effort), output_format=output or OutputFormat()),
            context, tools=tools)
        return parsed

    def grant_feature(self, name, constraints):
        self.capabilities = CapabilitySet({**self.capabilities.by_name,
            name: replace(self.capability, constraints=constraints)})

    def assert_no_runtime_io(self):
        self.provider.health.assert_not_called()
        self.provider.discover.assert_not_called()
        self.provider.load.assert_not_called()
        self.provider.chat.assert_not_called()
        self.store.reserve.assert_not_called()

    async def test_responses_require_both_usage_details_before_any_runtime_io(self):
        parsed = SimpleNamespace(chat=self.parsed())
        for allowed in (None, (), ("cached_input_tokens",), ("reasoning_output_tokens",)):
            with self.subTest(allowed=allowed):
                self.grant_feature(CapabilityName.CHAT, {**self.capability.constraints,
                    "usage_fields": ParameterConstraint(allowed_values=allowed)})
                with self.assertRaises(LocalInferenceError) as raised:
                    await self.service.validate_response(parsed, self.model, self.context)
                self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
                self.assert_no_runtime_io()
        self.grant_feature(CapabilityName.CHAT, {**self.capability.constraints,
            "usage_fields": ParameterConstraint(allowed_values=("cached_input_tokens", "reasoning_output_tokens"))})
        request = await self.service.validate_response(parsed, self.model, self.context)
        self.assertEqual(request.model, self.model)
        self.assert_no_runtime_io()

    async def test_structured_constraints_are_checked_before_service_or_direct_prepare_io(self):
        schema = {"type": "object", "properties": {"value": {"type": ["string", "null"]}},
                  "required": ["value"], "additionalProperties": False}
        output = OutputFormat(OutputFormatKind.JSON_SCHEMA, schema, "result", True)
        constraints = {"formats": ParameterConstraint(allowed_values=("json_schema", "json_object")),
                       "strict": ParameterConstraint(allowed_values=(True, False))}
        constraints.update({"schema_" + key: ParameterConstraint(allowed_values=value)
                            for key, value in schema_features(compile_schema(schema, strict=True)).items()})
        self.grant_feature(CapabilityName.STRUCTURED_OUTPUT, constraints)
        parsed = self.feature_parsed(output=output)
        request = await self.service.validate_chat(parsed, self.model, self.context)
        self.assertEqual(request.options.output_format, output)
        self.assert_no_runtime_io()
        for missing in ("formats", "strict", "schema_keywords", "schema_types", "schema_variants"):
            with self.subTest(missing=missing):
                self.grant_feature(CapabilityName.STRUCTURED_OUTPUT,
                                   {key: value for key, value in constraints.items() if key != missing})
                for invoke in (lambda: self.service.validate_chat(parsed, self.model, self.context),
                               lambda: self.service.prepare(request)):
                    with self.assertRaises(LocalInferenceError) as raised:
                        await invoke()
                    self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
                self.assert_no_runtime_io()
        self.grant_feature(CapabilityName.STRUCTURED_OUTPUT,
            {**constraints, "schema_types": ParameterConstraint(allowed_values=("object", "string"))})
        with self.assertRaises(LocalInferenceError) as raised:
            await self.service.validate_chat(parsed, self.model, self.context)
        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
        self.assert_no_runtime_io()

    async def test_active_reasoning_uses_its_own_explicit_effort_constraints(self):
        parsed = self.feature_parsed(effort="low")
        # CHAT does not contain reasoning_effort=low. Only REASONING grants it.
        for constraints, expected in (({}, ErrorCode.UNSUPPORTED_CAPABILITY),
            ({"efforts": ParameterConstraint()}, ErrorCode.UNSUPPORTED_CAPABILITY),
            ({"efforts": ParameterConstraint(allowed_values=("high",))}, ErrorCode.UNSUPPORTED_VALUE)):
            self.grant_feature(CapabilityName.REASONING, constraints)
            with self.assertRaises(LocalInferenceError) as raised:
                await self.service.validate_chat(parsed, self.model, self.context)
            self.assertEqual(raised.exception.failure.code, expected)
            self.assert_no_runtime_io()
        self.grant_feature(CapabilityName.REASONING, {"efforts": ParameterConstraint(allowed_values=("low",))})
        request = await self.service.validate_chat(parsed, self.model, self.context)
        self.assertEqual(request.options.reasoning.effort, "low")
        self.assert_no_runtime_io()

    async def test_each_tool_strict_value_requires_evidence_before_runtime_io(self):
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"], "additionalProperties": False}
        tools = (ToolDefinition("strict_lookup", schema, strict=True),
                 ToolDefinition("loose_lookup", schema, strict=False))
        for allowed in ((True,), (False,)):
            for ordered in (tools, tuple(reversed(tools))):
                with self.subTest(allowed=allowed, first=ordered[0].strict):
                    self.grant_feature(CapabilityName.FUNCTION_TOOLS, {
                        "tool_choice": ParameterConstraint(allowed_values=("auto",)),
                        "max_tools": ParameterConstraint(maximum=2),
                        "strict": ParameterConstraint(allowed_values=allowed)})
                    parsed = self.feature_parsed(tools=ordered)
                    request = parsed.to_request(self.model, self.context, 16)
                    rejected_index = next(index for index, tool in enumerate(ordered) if tool.strict not in allowed)
                    for invoke in (lambda: self.service.validate_chat(parsed, self.model, self.context),
                                   lambda: self.service.prepare(request)):
                        with self.assertRaises(LocalInferenceError) as raised:
                            await invoke()
                        self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_VALUE)
                        self.assertEqual(raised.exception.failure.parameter, f"tools[{rejected_index}].function.strict")
                    self.assert_no_runtime_io()
        self.grant_feature(CapabilityName.FUNCTION_TOOLS, {
            "tool_choice": ParameterConstraint(allowed_values=("auto",)),
            "max_tools": ParameterConstraint(maximum=2),
            "strict": ParameterConstraint(allowed_values=(True, False))})
        request = await self.service.validate_chat(self.feature_parsed(tools=tools), self.model, self.context)
        self.assertEqual(tuple(tool.strict for tool in request.tools), (True, False))
        self.assert_no_runtime_io()

    async def test_tool_history_without_definitions_does_not_invent_strict_false(self):
        from openai_wire import parse_chat
        self.grant_feature(CapabilityName.CHAT, {**self.capability.constraints,
            "roles": ParameterConstraint(allowed_values=("user", "assistant", "tool"))})
        self.grant_feature(CapabilityName.FUNCTION_TOOLS, {
            "tool_choice": ParameterConstraint(allowed_values=("none",)),
            "max_tools": ParameterConstraint(allowed_values=(0,))})
        parsed = parse_chat({"model": self.model.api_model_id, "messages": [
            {"role": "user", "content": "Look up the value."},
            {"role": "assistant", "tool_calls": [{"id": "historic_call", "type": "function",
                "function": {"name": "lookup", "arguments": '{"value":"x"}'}}]},
            {"role": "tool", "tool_call_id": "historic_call", "content": "result"}],
            "tools": [], "tool_choice": "none"})
        request = await self.service.validate_chat(parsed, self.model, self.context)
        self.assertEqual(request.tools, ())
        self.assert_no_runtime_io()

    async def test_rejected_tool_structured_reasoning_combinations_remain_closed(self):
        self.grant_feature(CapabilityName.REASONING, {"efforts": ParameterConstraint(allowed_values=("low",))})
        self.grant_feature(CapabilityName.STRUCTURED_OUTPUT,
                           {"formats": ParameterConstraint(allowed_values=("json_object",))})
        self.grant_feature(CapabilityName.FUNCTION_TOOLS, {
            "tool_choice": ParameterConstraint(allowed_values=("auto",)),
            "max_tools": ParameterConstraint(maximum=1), "strict": ParameterConstraint(allowed_values=(False,))})
        output = OutputFormat(OutputFormatKind.JSON_OBJECT)
        tool = ToolDefinition("lookup", {"type": "object"})
        for options in ({"effort": "low", "output": output}, {"effort": "low", "tools": (tool,)},
                        {"effort": "low", "stop": ("end",)}, {"output": output, "tools": (tool,)},
                        {"history": True}):
            with self.subTest(options=options):
                parsed = self.feature_parsed(**options)
                request = parsed.to_request(self.model, self.context, 16)
                for invoke in (lambda: self.service.validate_chat(parsed, self.model, self.context),
                               lambda: self.service.prepare(request)):
                    with self.assertRaises(LocalInferenceError) as raised:
                        await invoke()
                    self.assertEqual(raised.exception.failure.code, ErrorCode.UNSUPPORTED_CAPABILITY)
                self.assert_no_runtime_io()


if __name__ == "__main__":
    unittest.main()
