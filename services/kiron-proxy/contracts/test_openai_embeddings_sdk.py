"""Pinned SDK embedding contract through the real API/runtime/admission boundary.

Provider execution and its evidence are explicit offline fixtures, never claims
about an installed model. Native formatting/tokenizer parity is a separate gate.
The SDK's strict List[float] model cannot parse base64 before its post-parser;
float uses strict validation, default/base64 use normal SDK parsing plus bytes.
"""
import base64
from dataclasses import replace
import os
from pathlib import Path
import struct
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
    CapabilityName, CapabilitySet, EmbeddingResult, EmbeddingRole, ErrorCode, LocalInferenceError, ParameterConstraint,
    RuntimeFailure, RuntimeTimeouts, TokenUsage, build_resolver_snapshot,
)
from kiron_common.model_catalog import BackendType, ModelCatalog
from openai_api import create_openai_api_app
from runtime_service import RuntimeService, generation_key
from provider_transport import RequestRejected
import test_openai_api_sdk as api_sdk


PROVIDERS = (BackendType.OLLAMA, BackendType.KIRON_EMBEDDINGS)


def embedding_manifest(provider):
    value = api_sdk.manifest(BackendType.OLLAMA)
    name = provider.value
    value.update(canonical_model_id=name, aliases=[name + "-alias"])
    deployment = value["deployments"][0]
    deployment["id"] = name + ".deployment"
    deployment["backend"] = {"type": name, "parameters": {"model_name": name + ":fixture"}}
    deployment["routes"] = [{"task": "embedding", "endpoint": "/api/embed"}]
    if provider is BackendType.KIRON_EMBEDDINGS:
        deployment["artifact"].update(type="huggingface", format="hf_weights",
            repository="fixture/embedding", revision="c" * 40, manifest_digest=None)
        deployment["loader"]["type"] = "sentence_transformers"
    value["profiles"] = [{"id": name + ".dense", "deployment_id": deployment["id"],
        "task": "embedding", "endpoint": "/api/embed", "default_for_endpoint": True,
        "metadata": {"created": api_sdk.CREATED, "kind": "dense", "dimensions": 3,
            "verification": {"status": "verified", "evidence": ["offline-sdk-fixture"]},
            "input_type": {"role_sensitive": False, "required": False,
                "missing_role_behavior": "no_op", "supported": ["search_query", "search_document"]}}}]
    return value


class EmbeddingProvider(api_sdk.Provider):
    def __init__(self, model, admission):
        super().__init__(model, admission)
        self.capability = replace(self.chat_capability, constraints={
            "profiles": ParameterConstraint(allowed_values=(model.profile_id,)),
            "roles": ParameterConstraint(allowed_values=(None, "search_query", "search_document")),
            "dimensions": ParameterConstraint(allowed_values=(3,)),
            "max_batch_size": ParameterConstraint(minimum=1, maximum=512),
            "max_input_characters": ParameterConstraint(minimum=1, maximum=32768),
        })

    async def capabilities(self, deployment):
        return CapabilitySet({CapabilityName.EMBEDDINGS: self.capability})

    def validate_embedding_request(self, request, capabilities):
        pass  # RuntimeService runs the shared evidence/profile checks itself.

    async def embed(self, request):
        self._execution("embed", request)
        if self.mode in {"rejected_before_execution", "unproved_conflict"}:
            error = RequestRejected if self.mode == "rejected_before_execution" else LocalInferenceError
            raise error(RuntimeFailure(ErrorCode.CONFLICT, "Embedding generation changed"))
        rid = "foreign-id" if self.mode == "foreign_id" else request.context.request_id
        vectors = tuple((index + .25, -.5, 1.) for index in range(len(request.inputs)))
        if self.mode == "wrong_dimension":
            vectors = tuple(row[:2] for row in vectors)
        return EmbeddingResult(rid, vectors, TokenUsage(17, 0))


class EmbeddingSdkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertEqual(openai.__version__, "2.29.0")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        path.chmod(0o2770)
        self.admission = AdmissionStore(path, security=RuntimeSecurity(
            os.geteuid(), os.getegid(), frozenset({os.geteuid()})))
        measure = lambda: MemorySnapshot(1000, 1000, time.monotonic())
        catalog = ModelCatalog.from_manifests([embedding_manifest(p) for p in PROVIDERS]
                                               + [api_sdk.manifest(BackendType.PRISM)])
        self.snapshot = build_resolver_snapshot(catalog, (), resource_profiles={api_sdk.PROFILE.id: api_sdk.PROFILE})
        async def snapshot():
            return self.snapshot
        self.providers = {p: EmbeddingProvider(self.snapshot.resolve(p.value + ".dense"), self.admission)
                          for p in PROVIDERS}
        self.providers[BackendType.PRISM] = api_sdk.Provider(self.snapshot.resolve("prism.chat"), self.admission)
        self.runtime = RuntimeService(resolver=SimpleNamespace(snapshot=snapshot), providers=self.providers,
            admission=self.admission, measure=measure, timeouts=RuntimeTimeouts(2, 2, 2, 2, 5, .1, .1))
        self.addAsyncCleanup(self.runtime.aclose)
        # The dedicated provider requires measured residency, even in a fixture.
        provider = self.providers[BackendType.KIRON_EMBEDDINGS]
        generation = generation_key(provider.generation)
        self.admission.reserve(operation_id="embedding-resident", owner="kiron-embeddings", generation=generation,
            deployment_id=provider.model.deployment.id, kind="load", gpu_bytes=40, host_bytes=20,
            measure=measure)
        self.admission.transition("embedding-resident", owner="kiron-embeddings", expected_generation=generation,
                                  phase="resident")
        self.records = api_sdk.Records()
        keys = SimpleNamespace(validate_key=lambda key: {"fixture": True} if key == "fixture-key" else None)
        self.app = create_openai_api_app(self.records, keys)
        self.app.state.local_inference = self.runtime
        self.sdk = self.client(strict=True)
        self.regular = self.client(strict=False)

    def client(self, *, strict):
        client = openai.AsyncOpenAI(api_key="fixture-key", base_url="http://asgi.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(self.app), trust_env=False),
            _strict_response_validation=strict, max_retries=0)
        self.addAsyncCleanup(client.close)
        return client

    def request_tickets(self):
        return [ticket for ticket in self.admission.snapshot() if ticket.kind == "request"]

    def assert_no_io(self):
        self.assertEqual({p.value: value.io for p, value in self.providers.items()},
                         {p.value: [] for p in self.providers})
        self.assertEqual(self.request_tickets(), [])

    async def test_strict_float_batch_order_roles_dimensions_and_usage(self):
        for provider_id in PROVIDERS:
            for suffix, role in (("", None), (".query", EmbeddingRole.QUERY), (".document", EmbeddingRole.DOCUMENT)):
                with self.subTest(provider=provider_id, role=role):
                    address = provider_id.value + ".dense" + suffix
                    result = await self.sdk.embeddings.create(model=address, input=[" raw query ", "世界"],
                                                               dimensions=3, encoding_format="float")
                    self.assertEqual(result.model, address)
                    self.assertEqual([row.index for row in result.data], [0, 1])
                    self.assertEqual([row.embedding for row in result.data], [[.25, -.5, 1.], [1.25, -.5, 1.]])
                    self.assertEqual((result.usage.prompt_tokens, result.usage.total_tokens), (17, 17))
                    request = self.providers[provider_id].requests[-1]
                    self.assertEqual(request.inputs, (" raw query ", "世界"))
                    self.assertIs(request.model.embedding_role, role)
                    self.assertEqual(request.dimensions, 3)
                    self.assertEqual(self.request_tickets(), [])

    async def test_sdk_default_and_explicit_base64_match_float32_wire_exactly(self):
        arguments = {"model": "kiron_embeddings.dense.query", "input": ["one", "two"]}
        decoded = await self.regular.embeddings.create(**arguments)
        encoded = await self.regular.embeddings.create(**arguments, encoding_format="base64")
        for float_row, encoded_row in zip(decoded.data, encoded.data):
            raw = base64.b64decode(encoded_row.embedding, validate=True)
            self.assertEqual(len(raw), 3 * 4)
            self.assertEqual(list(struct.unpack("<fff", raw)), float_row.embedding)
        # This is the pinned SDK's schema/post-parser ordering limitation,
        # not a server error or a silently different server encoding.
        with self.assertRaises(openai.APIResponseValidationError):
            await self.sdk.embeddings.create(**arguments)
        self.assertEqual(self.request_tickets(), [])

    async def test_discovery_and_role_ids_survive_a_fresh_resolver_snapshot(self):
        listed = await self.sdk.models.list()
        expected = {p.value + ".dense" + suffix for p in PROVIDERS for suffix in ("", ".query", ".document")}
        self.assertEqual({row.id for row in listed.data if row.id != "prism.chat"}, expected)
        before = {row.id: row.model_dump() for row in listed.data}
        for address in expected:
            self.assertEqual((await self.sdk.models.retrieve(address)).model_dump(), before[address])
        catalog = ModelCatalog.from_manifests([embedding_manifest(p) for p in PROVIDERS]
                                               + [api_sdk.manifest(BackendType.PRISM)])
        self.snapshot = build_resolver_snapshot(catalog, (), resource_profiles={api_sdk.PROFILE.id: api_sdk.PROFILE})
        self.assertEqual({row.id: row.model_dump() for row in (await self.sdk.models.list()).data}, before)

    async def test_negative_syntax_dimensions_and_chat_only_model_do_not_touch_providers(self):
        cases = [({"input": []}, "invalid_request"), ({"input": [1, 2]}, "unsupported_capability"),
            ({"dimensions": 2}, "unsupported_value"), ({"extra_body": {"input_type": "search_query"}}, "unsupported_parameter"),
            ({"user": "client"}, "unsupported_parameter"), ({"model": "prism.chat"}, "unsupported_capability")]
        for changes, code in cases:
            with self.subTest(changes=changes), self.assertRaises(openai.BadRequestError) as caught:
                await self.sdk.embeddings.create(**{"model": "kiron_embeddings.dense.query", "input": "text",
                    "encoding_format": "float", **changes})
            self.assertEqual(caught.exception.code, code)
            self.assert_no_io()

    async def test_foreign_result_identity_retains_unknown_admission(self):
        self.providers[BackendType.KIRON_EMBEDDINGS].mode = "foreign_id"
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual((caught.exception.status_code, caught.exception.code), (502, "backend_protocol_error"))
        self.assertEqual([ticket.phase for ticket in self.request_tickets()], ["unknown"])

    async def test_proven_rejection_releases_only_its_request_and_allows_next_turn(self):
        provider = self.providers[BackendType.KIRON_EMBEDDINGS]
        before = self.admission.snapshot()
        provider.mode = "rejected_before_execution"
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual((caught.exception.status_code, caught.exception.code), (503, "resource_busy"))
        self.assertEqual(self.admission.snapshot(), before)
        provider.mode = "complete"
        result = await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual(len(result.data), 1)
        self.assertEqual(self.admission.snapshot(), before)

    async def test_conflict_without_before_execution_proof_stays_unknown(self):
        self.providers[BackendType.KIRON_EMBEDDINGS].mode = "unproved_conflict"
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual((caught.exception.status_code, caught.exception.code), (503, "resource_busy"))
        self.assertEqual([ticket.phase for ticket in self.request_tickets()], ["unknown"])

    async def test_dedicated_embedding_discovery_requires_measured_shared_residency(self):
        provider = self.providers[BackendType.KIRON_EMBEDDINGS]
        self.admission.release("embedding-resident", owner="kiron-embeddings",
            generation=generation_key(provider.generation), confirmed_terminated=True)
        listed = await self.sdk.models.list()
        self.assertTrue(any(row.id.startswith("ollama.dense") for row in listed.data))
        self.assertFalse(any(row.id.startswith("kiron_embeddings.dense") for row in listed.data))
        with self.assertRaises(openai.InternalServerError) as caught:
            await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual((caught.exception.status_code, caught.exception.code), (503, "provider_unavailable"))
        self.assertNotIn("embed", provider.io)
        self.assertEqual(self.request_tickets(), [])

    async def test_discovery_checks_exact_profile_role_constraints(self):
        provider = self.providers[BackendType.KIRON_EMBEDDINGS]
        provider.capability = replace(provider.capability, constraints={**provider.capability.constraints,
            "roles":ParameterConstraint(allowed_values=(None,"search_document"))})
        with self.assertRaises(openai.BadRequestError) as caught:
            await self.sdk.embeddings.create(model="kiron_embeddings.dense.query", input="text", encoding_format="float")
        self.assertEqual(caught.exception.code, "unsupported_value")
        self.assert_no_io()
        ids = {row.id for row in (await self.sdk.models.list()).data}
        self.assertNotIn("kiron_embeddings.dense.query", ids)
        self.assertIn("kiron_embeddings.dense.document", ids)
        self.assertIn("kiron_embeddings.dense", ids)


if __name__ == "__main__":
    unittest.main()
