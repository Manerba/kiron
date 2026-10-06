"""Pinned SDK against the real ASGI boundary and RuntimeService, entirely offline.

Only provider execution, inventory, authentication storage and request logging
are fixtures. This tests KIron's text API wire contract, not live GPU capability.
Run in the isolated SDK venv with additional starlette==0.52.1/pymysql==1.2.3;
requirements-contract.txt remains the unchanged SDK dependency baseline.
"""
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import httpx
import openai

PROXY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROXY))
sys.path.insert(0, str(PROXY.parent / "kiron-common"))

from kiron_common.gpu_admission import AdmissionStore, MemorySnapshot, RuntimeSecurity
from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    DeploymentObservation, DiscoveredModel, DiscoverySnapshot, EventKind, FinishReason,
    InferenceEvent, InferenceResult, ParameterConstraint, ProviderHealth,
    ProviderObservation, ResourceProfile, RuntimeGeneration, RuntimeImplementation,
    RuntimeTimeouts, TextPart, TokenUsage, build_resolver_snapshot,
)
from kiron_common.model_catalog import BackendType, ModelCatalog
from kiron_common.model_state import RuntimeState
from openai_api import create_openai_api_app
from runtime_service import RuntimeService, generation_key


STAMP = datetime(2026, 9, 21, tzinfo=timezone.utc)
CREATED = 1758412800
PROFILE = ResourceProfile("fixture-prism-cuda40", 1024, 128, 128, 1, 4, 40, False, 40, 20)
MESSAGES = [{"role": "user", "content": "Hello"}]


def manifest(provider):
    prism = provider is BackendType.PRISM
    name = provider.value
    artifact = {"type": "local" if prism else "ollama", "format": "gguf" if prism else "ollama_manifest",
        "repository": None, "revision": None, "manifest_digest": None if prism else "sha256:" + "b" * 64,
        "trust_remote_code": False, "weights": [{"path": "/models/prism.gguf" if prism else "model",
            "sha256": "a" * 64, "size_bytes": 100 if prism else None}],
        "auxiliary": [], "projector": None, "metadata": {}}
    return {"schema_version": 2, "canonical_model_id": name, "aliases": [name + "-alias"],
        "deployments": [{"id": name + ".deployment", "backend": {"type": name,
            "parameters": {"model_name": name + ":latest"}}, "artifact": artifact,
            "routes": [{"task": "chat", "endpoint": "/v1/chat/completions"}],
            "loader": {"type": "prism_gguf" if prism else "ollama", "parameters": {}},
            "runtime_profile": PROFILE.id if prism else None, "metadata": {}}],
        "profiles": [{"id": name + ".chat", "deployment_id": name + ".deployment", "task": "chat",
            "endpoint": "/v1/chat/completions", "default_for_endpoint": True, "metadata": {"created": CREATED}}],
        "request_defaults": [], "metadata": {}}


class Provider:
    """Evidence and events are typed; IO calls are separately observable."""
    # This fixture supplies an already resident model and implements no mutations.
    model_lifecycle_operations = frozenset()

    def __init__(self, model, admission):
        self.model, self.admission = model, admission
        self.provider = model.deployment.provider
        self.generation = RuntimeGeneration("fixture-boot", self.provider.value + "-child")
        self.implementation = RuntimeImplementation("fixture-runtime", "fixture-template", "fixture-parser")
        artifact = model.deployment.artifact_identity
        evidence = CapabilityEvidence(self.implementation.provider_revision, artifact.fingerprint,
            artifact.sha256, None, self.implementation.template_revision, self.implementation.parser_revision,
            model.deployment.configuration_fingerprint, "offline-sdk-asgi-fixture", STAMP)
        self.chat_capability = Capability(CapabilityStatus.SUPPORTED, {
            "roles": ParameterConstraint(allowed_values=("system", "user", "assistant")),
            "max_output_tokens": ParameterConstraint(minimum=1, maximum=128),
            "default_max_output_tokens": ParameterConstraint(allowed_values=(16,)),
            "token_budget": ParameterConstraint(allowed_values=("max_tokens", "max_completion_tokens")),
            "temperature": ParameterConstraint(minimum=0, maximum=1),
        }, (evidence,))
        self.streaming, self.available, self.mode = True, True, "complete"
        self.io, self.requests = [], []

    async def capabilities(self, deployment):
        values = {CapabilityName.CHAT: self.chat_capability}
        if self.streaming:
            values[CapabilityName.STREAMING] = self.chat_capability
        return CapabilitySet(values)

    def validate_request(self, request, capabilities):
        pass

    async def health(self, context):
        self.io.append("health")
        if not self.available:
            return ProviderObservation(self.provider, None, STAMP, ProviderHealth.UNAVAILABLE)
        deployment = self.model.deployment
        return ProviderObservation(self.provider, self.generation, STAMP, ProviderHealth.AVAILABLE,
            {deployment.id: DeploymentObservation(deployment.id, RuntimeState.LOADED,
                self.generation, deployment.configuration_fingerprint)})

    async def discover(self, context):
        self.io.append("discover")
        deployment = self.model.deployment
        return DiscoverySnapshot(self.provider, "fixture", STAMP,
            (DiscoveredModel(deployment.reference, deployment.artifact_identity, True),))

    def _execution(self, kind, request):
        self.io.append(kind)
        self.requests.append(request)
        assert request.execution_generation == self.generation
        assert any(ticket.kind == "request" and ticket.operation_id == request.context.request_id
                   and ticket.deployment_id == self.model.deployment.id for ticket in self.admission.snapshot())

    async def chat(self, request):
        self._execution("chat", request)
        request_id = "foreign-request" if self.mode == "foreign_request_id" else request.context.request_id
        return InferenceResult(request_id, (TextPart("Hello world"),), (), (),
                               TokenUsage(5, 2), FinishReason.STOP)

    async def stream(self, request):
        self._execution("stream", request)
        request_id = request.context.request_id
        yield InferenceEvent(EventKind.STARTED, request_id)
        yield InferenceEvent(EventKind.TEXT_DELTA, request_id, text="Hello", output_item_index=0, part_index=0)
        if self.mode == "partial_eof":
            return
        yield InferenceEvent(EventKind.TEXT_DELTA, request_id, text=" world", output_item_index=0, part_index=0)
        yield InferenceEvent(EventKind.USAGE, request_id, usage=TokenUsage(5, 2))
        yield InferenceEvent(EventKind.COMPLETED, request_id, finish_reason=FinishReason.STOP)

    async def wait_request_end(self, deployment, *, generation, context):
        self.io.append("wait_end")
        return False

    async def aclose(self):
        pass


class Records:
    def __init__(self):
        self.records, self.updates = [], []

    async def add_request(self, record):
        self.records.append(record)

    async def update_request(self, request_id, **values):
        self.updates.append((request_id, values))


class OpenAISDKContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertEqual(openai.__version__, "2.29.0")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        root.chmod(0o2770)
        self.admission = AdmissionStore(root, security=RuntimeSecurity(
            os.geteuid(), os.getegid(), frozenset({os.geteuid()})))
        self.measure = lambda: MemorySnapshot(1000, 1000, time.monotonic())
        catalog = ModelCatalog.from_manifests([manifest(provider) for provider in (BackendType.OLLAMA, BackendType.PRISM)])
        self.snapshot = build_resolver_snapshot(catalog, (), resource_profiles={PROFILE.id: PROFILE})
        async def snapshot():
            return self.snapshot
        self.providers = {provider: Provider(self.snapshot.resolve(provider.value + ".chat"), self.admission)
                          for provider in (BackendType.OLLAMA, BackendType.PRISM)}
        self.runtime = RuntimeService(resolver=SimpleNamespace(snapshot=snapshot), providers=self.providers,
            admission=self.admission, measure=self.measure, timeouts=RuntimeTimeouts(2, 2, 2, 2, 5, .1, .1))
        self.addAsyncCleanup(self.runtime.aclose)
        prism = self.providers[BackendType.PRISM]
        generation = generation_key(prism.generation)
        self.admission.reserve(operation_id="resident-prism", owner="kiron-proxy", generation=generation,
            deployment_id=prism.model.deployment.id, kind="load", gpu_bytes=40, host_bytes=20,
            measure=self.measure, resident_slot="prism")
        self.admission.transition("resident-prism", owner="kiron-proxy", expected_generation=generation, phase="resident")
        self.records = Records()
        keys = SimpleNamespace(validate_key=lambda key: {"fixture": True} if key == "sdk-fixture-key" else None)
        app = create_openai_api_app(self.records, keys)
        app.state.local_inference = self.runtime
        self.sdk = openai.AsyncOpenAI(api_key="sdk-fixture-key", base_url="http://asgi.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app), trust_env=False),
            _strict_response_validation=True, max_retries=0)
        self.addAsyncCleanup(self.sdk.close)

    def request_tickets(self):
        return [ticket for ticket in self.admission.snapshot() if ticket.kind == "request"]

    def assert_no_provider_io(self):
        self.assertEqual({provider.value: value.io for provider, value in self.providers.items()},
                         {"ollama": [], "prism": []})
        self.assertEqual(self.request_tickets(), [])

    async def test_models_list_detail_and_alias_are_one_stable_canonical_record(self):
        first = await self.sdk.models.list()
        second = await self.sdk.models.list()
        self.assertEqual([model.id for model in first.data], ["ollama.chat", "prism.chat"])
        self.assertEqual(first.model_dump(), second.model_dump())
        for model in first.data:
            self.assertEqual(model.created, CREATED)
            self.assertEqual(model.owned_by, model.id.split(".")[0])
            detail = await self.sdk.models.retrieve(model.id)
            alias = await self.sdk.models.retrieve(model.owned_by + "-alias")
            self.assertEqual(detail.model_dump(), model.model_dump())
            self.assertEqual(alias.model_dump(), model.model_dump())
        self.assertEqual(self.request_tickets(), [])
        self.assertTrue(all(set(provider.io) <= {"health", "discover"} for provider in self.providers.values()))

    async def test_text_completions_use_both_providers_and_canonical_alias_id(self):
        for name, provider in self.providers.items():
            with self.subTest(provider=name.value):
                result = await self.sdk.chat.completions.create(model=name.value + "-alias", messages=MESSAGES,
                                                                max_completion_tokens=8, temperature=.5)
                self.assertEqual(result.model, name.value + ".chat")
                self.assertEqual(result.choices[0].message.content, "Hello world")
                self.assertEqual(result.choices[0].finish_reason, "stop")
                self.assertEqual((result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens), (5, 2, 7))
                self.assertEqual(provider.requests[-1].options.max_output_tokens, 8)
                self.assertEqual(self.request_tickets(), [])

    async def test_stream_helper_assembles_text_and_include_usage_for_both_providers(self):
        for provider in self.providers:
            with self.subTest(provider=provider.value):
                parts = []
                async with self.sdk.chat.completions.stream(model=provider.value + "-alias", messages=MESSAGES,
                        max_completion_tokens=8, stream_options={"include_usage": True}) as stream:
                    async for event in stream:
                        if event.type == "content.delta":
                            parts.append(event.delta)
                    final = await stream.get_final_completion()
                self.assertEqual("".join(parts), "Hello world")
                self.assertEqual(final.choices[0].message.content, "Hello world")
                self.assertEqual(final.choices[0].finish_reason, "stop")
                self.assertEqual(final.model, provider.value + ".chat")
                self.assertEqual(final.usage.total_tokens, 7)
                self.assertEqual(self.request_tickets(), [])

    async def test_sdk_authentication_and_unknown_model_errors_precede_provider_io(self):
        with self.assertRaises(openai.AuthenticationError) as auth:
            await self.sdk.with_options(api_key="invalid-fixture-key").models.list()
        self.assertEqual(auth.exception.code, "invalid_api_key")
        self.assertIn("Bearer", auth.exception.response.headers["www-authenticate"])
        with self.assertRaises(openai.NotFoundError) as missing:
            await self.sdk.models.retrieve("missing")
        self.assertEqual(missing.exception.code, "model_not_found")
        self.assert_no_provider_io()

    async def test_sdk_syntax_and_unsupported_features_do_not_reach_provider_io(self):
        cases = [({"messages": []}, "invalid_request"),
                 ({"n": 2}, "unsupported_parameter"),
                 ({"response_format": {"type": "json_object"}}, "unsupported_capability"),
                 ({"presence_penalty": 0.0}, "unsupported_parameter")]
        for changes, expected in cases:
            with self.subTest(changes=changes), self.assertRaises(openai.BadRequestError) as caught:
                await self.sdk.chat.completions.create(**{"model": "prism.chat", "messages": MESSAGES, **changes})
            self.assertEqual(caught.exception.code, expected)
            self.assert_no_provider_io()

    async def test_unverified_streaming_capability_blocks_before_provider_io(self):
        self.providers[BackendType.PRISM].streaming = False
        with self.assertRaises(openai.BadRequestError) as caught:
            await self.sdk.chat.completions.create(model="prism.chat", messages=MESSAGES, stream=True)
        self.assertEqual(caught.exception.code, "unsupported_capability")
        self.assert_no_provider_io()

    async def test_known_unavailable_model_is_503_and_other_provider_stays_listed(self):
        self.providers[BackendType.PRISM].available = False
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.models.retrieve("prism-alias")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.code, "provider_unavailable")
        self.assertEqual([model.id for model in (await self.sdk.models.list()).data], ["ollama.chat"])
        self.assertEqual(self.request_tickets(), [])

    async def test_foreign_completion_request_id_is_502_and_retains_unknown_work(self):
        self.providers[BackendType.PRISM].mode = "foreign_request_id"
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.chat.completions.create(model="prism.chat", messages=MESSAGES)
        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(caught.exception.code, "backend_protocol_error")
        self.assertEqual([ticket.phase for ticket in self.request_tickets()], ["unknown"])

    async def test_partial_stream_failure_raises_sdk_error_and_retains_unknown_work(self):
        self.providers[BackendType.PRISM].mode = "partial_eof"
        parts = []
        with self.assertRaises(openai.APIError) as caught:
            async with self.sdk.chat.completions.stream(model="prism.chat", messages=MESSAGES,
                    stream_options={"include_usage": True}) as stream:
                async for event in stream:
                    if event.type == "content.delta":
                        parts.append(event.delta)
        self.assertEqual("".join(parts), "Hello")
        self.assertEqual(caught.exception.code, "backend_protocol_error")
        self.assertEqual([ticket.phase for ticket in self.request_tickets()], ["unknown"])
        self.assertEqual(self.records.updates[-1][1]["state"], "error")


if __name__ == "__main__":
    unittest.main()
