"""Dedicated native transport over MockTransport, no service/model execution."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import hashlib
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

from kiron_common.embedding_registry import MODEL_CATALOG
from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    EmbeddingRequest, LocalInferenceError, ParameterConstraint, RequestContext,
    RuntimeGeneration, RuntimeImplementation, build_resolver_snapshot,
)
from kiron_common.model_state import RuntimeState
from embedding_provider import KironEmbeddingProvider
from provider_transport import RequestRejected


class EmbeddingProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.snapshot = build_resolver_snapshot(MODEL_CATALOG, ())
        self.model = self.snapshot.resolve("kiron-bge-m3-dense-v1")
        self.deployment = self.model.deployment
        self.implementation = RuntimeImplementation("fixture-service", None, "fixture-adapter")
        self.generation = RuntimeGeneration("fixture-boot", "123:7")
        self.resolver = SimpleNamespace(snapshot=mock.AsyncMock(return_value=self.snapshot))
        evidence = CapabilityEvidence(self.implementation.provider_revision,
            self.deployment.artifact_identity.fingerprint, None, None, None,
            self.implementation.parser_revision, self.deployment.configuration_fingerprint,
            "offline fixture; not installed capability", datetime.now(timezone.utc))
        self.cap = Capability(CapabilityStatus.SUPPORTED, {
            "profiles": ParameterConstraint(allowed_values=(self.model.profile_id,)),
            "roles": ParameterConstraint(allowed_values=(None, "search_query", "search_document")),
            "dimensions": ParameterConstraint(allowed_values=(1024,)),
            "devices": ParameterConstraint(allowed_values=("cpu",)),
            "max_batch_size": ParameterConstraint(minimum=1, maximum=512),
            "max_input_characters": ParameterConstraint(minimum=1, maximum=32768)}, (evidence,))
        self.calls, self.mutate_state, self.mutate_result = [], None, None
        self.reject = None
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:18099", trust_env=False,
                                       transport=httpx.MockTransport(self.respond))
        self.provider = KironEmbeddingProvider(client=self.client, resolver=self.resolver,
            implementation=self.implementation, catalog_digest=MODEL_CATALOG.catalog_digest,
            capabilities={self.deployment.id: CapabilitySet({CapabilityName.EMBEDDINGS: self.cap})})
        self.addAsyncCleanup(self.provider.aclose)

    def request(self, **changes):
        return replace(EmbeddingRequest(self.model, ("  Grün original  ",),
            RequestContext("embed-fixture", time.monotonic() + 3, asyncio.Event()),
            execution_generation=self.generation), **changes)

    def respond(self, request):
        self.calls.append(request)
        if request.method == "POST" and self.reject is not None:
            return self.reject(request)
        gen = {"boot_id": self.generation.boot_id, "process_id": self.generation.process_id}
        if request.url.path == "/api/inference/state":
            value = {"version": 1, "generation": gen, "service_revision": "fixture-service",
                "catalog_digest": MODEL_CATALOG.catalog_digest, "accepting": True, "busy": False,
                "device": "cpu", "deployments": [{"deployment_id": self.deployment.id,
                    "reference": self.deployment.reference,
                    "artifact_fingerprint": self.deployment.artifact_identity.fingerprint,
                    "configuration_fingerprint": self.deployment.configuration_fingerprint, "loaded": True}]}
            if self.mutate_state:
                self.mutate_state(value)
        else:
            self.assertEqual(request.url.path, "/api/inference/embed")
            body = json.loads(request.content)
            value = {key: val for key, val in body.items() if key not in {"inputs", "dimensions"}}
            value.update(done=True, token_counting="forward_attention_mask_v1",
                         embeddings=[[.25] * 1024 for _ in body["inputs"]],
                         usage={"input_tokens": 13, "output_tokens": 0})
            if self.mutate_result:
                self.mutate_result(value)
        return httpx.Response(200, json=value)

    @staticmethod
    def rejection(request):
        body = json.loads(request.content)
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode()
        return {"version": 1, "request_id": body["request_id"], "generation": body["generation"],
            "request_sha256": hashlib.sha256(encoded).hexdigest(), "rejected": True,
            "error": {"code": "identity_conflict"}}

    async def test_exact_rejection_proof_confirms_no_execution(self):
        self.reject = lambda request: httpx.Response(409, json=self.rejection(request))
        with self.assertRaises(RequestRejected):
            await self.provider.embed(self.request())

    async def test_unbound_malformed_or_truncated_rejection_is_not_an_end_proof(self):
        changes = [lambda value: value.pop("rejected"),
            lambda value: value.update(rejected=False), lambda value: value.update(rejected=1),
            lambda value: value.update(version=True), lambda value: value.update(request_id="foreign"),
            lambda value: value.update(request_sha256="f" * 64),
            lambda value: value["generation"].update(process_id="foreign"),
            lambda value: value.update(extra=True),
            lambda value: value["error"].update(code="execution_failed")]
        for change in changes:
            with self.subTest(change=change):
                def respond(request):
                    value = self.rejection(request)
                    change(value)
                    return httpx.Response(409, json=value)
                self.reject = respond
                with self.assertRaises(LocalInferenceError) as caught:
                    await self.provider.embed(self.request())
                self.assertNotIsInstance(caught.exception, RequestRejected)

        class BrokenStream(httpx.AsyncByteStream):
            def __init__(self, value):
                self.value = value

            async def __aiter__(self):
                yield json.dumps(self.value).encode()
                raise httpx.ReadError("lost before EOF")

        self.reject = lambda request: httpx.Response(409, stream=BrokenStream(self.rejection(request)))
        with self.assertRaises(LocalInferenceError) as caught:
            await self.provider.embed(self.request())
        self.assertNotIsInstance(caught.exception, RequestRejected)

    async def test_resident_state_and_exact_result_bind_the_entire_native_identity(self):
        request = self.request()
        observation = await self.provider.health(request.context)
        self.assertIs(observation.models[self.deployment.id].state, RuntimeState.LOADED)
        result = await self.provider.embed(request)
        self.assertEqual(result.request_id, "embed-fixture")
        self.assertEqual(result.usage.input_tokens, 13)
        self.assertEqual(len(result.vectors[0]), 1024)
        self.assertEqual(json.loads(self.calls[-1].content)["inputs"], ["  Grün original  "])

    async def test_wrong_service_artifact_config_or_generation_shape_rejects_state(self):
        changes = [lambda x: x.update(service_revision="other"), lambda x: x.update(catalog_digest="other"),
            lambda x: x["deployments"][0].update(artifact_fingerprint="f" * 64),
            lambda x: x["deployments"][0].update(configuration_fingerprint="f" * 64),
            lambda x: x["generation"].update(unknown="extra")]
        for change in changes:
            with self.subTest(change=change):
                self.mutate_state = change
                with self.assertRaises(LocalInferenceError):
                    await self.provider.health(self.request().context)

    async def test_missing_end_wrong_identity_and_guessed_usage_fail_closed(self):
        changes = [lambda x: x.update(done=False), lambda x: x.update(request_id="foreign"),
            lambda x: x["generation"].update(process_id="old"),
            lambda x: x.update(token_counting="char_div_4"),
            lambda x: x["usage"].update(input_tokens=True), lambda x: x["usage"].update(input_tokens=0),
            lambda x: x.update(embeddings=[[.25]]), lambda x: x.update(undeclared=True)]
        for change in changes:
            with self.subTest(change=change):
                self.mutate_result = change
                with self.assertRaises(LocalInferenceError):
                    await self.provider.embed(self.request())

    async def test_unsupported_requests_and_coldload_do_not_contact_service(self):
        for request in (self.request(dimensions=3), self.request(inputs=("   ",)),
                        self.request(execution_generation=None)):
            with self.subTest(request=request), self.assertRaises(LocalInferenceError):
                await self.provider.embed(request)
        with self.assertRaises(LocalInferenceError):
            await self.provider.load(self.deployment, snapshot_revision=self.snapshot.revision,
                expected_generation=self.generation, context=self.request().context)
        self.assertFalse(await self.provider.wait_request_end(self.deployment,
            generation=self.generation, context=self.request().context))
        self.assertEqual(self.calls, [])

    async def test_unverified_or_wrong_device_is_never_a_loaded_observation(self):
        self.mutate_state = lambda x: x.update(accepting=False)
        observation = await self.provider.health(self.request().context)
        self.assertIs(observation.models[self.deployment.id].state, RuntimeState.UNKNOWN)
        self.mutate_state = None
        self.calls.clear()
        self.provider._capabilities = {}
        observation = await self.provider.health(self.request().context)
        self.assertIs(observation.models[self.deployment.id].state, RuntimeState.UNKNOWN)
        with self.assertRaises(LocalInferenceError):
            await self.provider.embed(self.request())
        self.assertEqual(len(self.calls), 1)
