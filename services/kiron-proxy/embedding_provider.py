"""Dedicated KIron Dense adapter with native identity and exact usage proofs.

Cold loads are deliberately unavailable until an explicit measured resource
profile and lifecycle implementation exist. Aborted requests have no end proof.
"""
from datetime import datetime, timezone
import ipaddress

import httpx

from kiron_common.local_inference import (
    CapabilityName, CapabilitySet, DeploymentObservation, DiscoveredModel, DiscoverySnapshot,
    EmbeddingResult, ErrorCode, LocalInferenceError, ModelLifecycleOperation, ProviderHealth, ProviderObservation,
    RuntimeFailure, RuntimeGeneration, TokenUsage,
)
from kiron_common.local_inference.embedding_native import TOKEN_COUNTING, request_fingerprint
from kiron_common.model_catalog import BackendType
from kiron_common.model_state import RuntimeState
from provider_embeddings import validate_embedding_request
from provider_transport import RequestRejected, bounded_request, decode_provider_json, failure, with_context


class KironEmbeddingProvider:
    provider = BackendType.KIRON_EMBEDDINGS

    @property
    def model_lifecycle_operations(self) -> frozenset[ModelLifecycleOperation]:
        return frozenset()

    def __init__(self, *, client, resolver, implementation, catalog_digest, capabilities=None):
        try:
            local = ipaddress.ip_address(client.base_url.host).is_loopback
        except ValueError:
            local = False
        if client.base_url.scheme != "http" or not local or client.trust_env or client.follow_redirects:
            raise ValueError("embedding client requires fixed loopback HTTP without environment or redirects")
        self.client, self.resolver, self.implementation = client, resolver, implementation
        self.catalog_digest = catalog_digest
        self._capabilities = dict(capabilities or {})
        self._closed = False

    async def capabilities(self, deployment):
        values = self._capabilities.get(deployment.id, CapabilitySet())
        return CapabilitySet({CapabilityName.EMBEDDINGS: values.by_name[CapabilityName.EMBEDDINGS]}).for_deployment(deployment, self.implementation)

    def validate_embedding_request(self, request, capabilities):
        validate_embedding_request(request, capabilities, self.implementation)
        if request.model.deployment.provider is not self.provider:
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Wrong embedding provider", "model")
        if request.dimensions not in (None, request.model.profile_metadata["dimensions"]):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Native dimension reduction is not implemented", "dimensions")
        if any(not text.strip() for text in request.inputs):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Native embedding requires a nonblank input", "input")

    async def _json(self, method, path, context, payload=None):
        try:
            status, raw = await bounded_request(self.client, method, path, context,
                limit=16 * 1024 * 1024, json=payload)
        except httpx.TimeoutException:
            raise failure(ErrorCode.TIMEOUT, "Embedding transport timeout") from None
        except httpx.HTTPError:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Embedding transport unavailable") from None
        if status == 409 and method == "POST" and path == "/api/inference/embed":
            expected = {"version": 1, "request_id": payload["request_id"],
                "generation": payload["generation"], "request_sha256": request_fingerprint(payload),
                "rejected": True, "error": {"code": "identity_conflict"}}
            try:
                proof = decode_provider_json(raw)
                if (proof != expected or type(proof["version"]) is not int
                        or proof["rejected"] is not True):
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise failure(ErrorCode.PROVIDER_ERROR, "Embedding rejection has no valid end proof") from None
            raise RequestRejected(RuntimeFailure(ErrorCode.CONFLICT, "Embedding generation or identity changed"))
        if status == 409:
            raise failure(ErrorCode.CONFLICT, "Embedding generation or identity changed")
        if status != 200:
            raise failure(ErrorCode.PROVIDER_UNAVAILABLE, "Embedding service rejected the request")
        try:
            return decode_provider_json(raw)
        except ValueError:
            raise failure(ErrorCode.PROVIDER_ERROR, "Invalid embedding service JSON") from None

    async def _state(self, context):
        raw = await self._json("GET", "/api/inference/state", context)
        try:
            if (set(raw) != {"version", "generation", "service_revision", "catalog_digest", "accepting", "busy", "device", "deployments"}
                    or type(raw["version"]) is not int or raw["version"] != 1
                    or raw["service_revision"] != self.implementation.provider_revision
                    or raw["catalog_digest"] != self.catalog_digest
                    or type(raw["accepting"]) is not bool or type(raw["busy"]) is not bool
                    or raw["device"] not in {"cpu", "cuda", "unknown"}
                    or type(raw["deployments"]) is not list or len(raw["deployments"]) > 512
                    or set(raw["generation"]) != {"boot_id", "process_id"}):
                raise ValueError()
            generation = RuntimeGeneration(**raw["generation"])
            snapshot = await with_context(self.resolver.snapshot(), context)
            entries = {}
            for row in raw["deployments"]:
                if (type(row) is not dict or set(row) != {"deployment_id", "reference", "artifact_fingerprint", "configuration_fingerprint", "loaded"}
                        or type(row["loaded"]) is not bool or row["deployment_id"] in entries):
                    raise ValueError()
                try:
                    deployment = snapshot.resolve_deployment(row["deployment_id"])
                except LocalInferenceError:
                    raise ValueError() from None
                if (deployment.provider is not self.provider or deployment.reference != row["reference"]
                        or deployment.artifact_identity.fingerprint != row["artifact_fingerprint"]
                        or deployment.configuration_fingerprint != row["configuration_fingerprint"]):
                    raise ValueError()
                entries[deployment.id] = (deployment, row["loaded"])
            return raw, generation, entries
        except (TypeError, ValueError, KeyError):
            raise failure(ErrorCode.PROVIDER_ERROR, "Embedding state identity is invalid") from None

    async def health(self, context):
        raw, generation, entries = await self._state(context)
        models = {}
        for deployment, loaded in entries.values():
            if not loaded:
                continue
            caps = await self.capabilities(deployment)
            devices = caps.by_name[CapabilityName.EMBEDDINGS].constraints.get("devices")
            state = (RuntimeState.LOADED if raw["accepting"] and devices is not None and devices.allowed_values is not None
                     and devices.accepts(raw["device"]) else RuntimeState.UNKNOWN)
            models[deployment.id] = DeploymentObservation(deployment.id, state, generation, deployment.configuration_fingerprint)
        return ProviderObservation(self.provider, generation, datetime.now(timezone.utc),
            ProviderHealth.AVAILABLE if raw["accepting"] else ProviderHealth.UNAVAILABLE, models)

    async def discover(self, context):
        _, _, entries = await self._state(context)
        return DiscoverySnapshot(self.provider, self.catalog_digest, datetime.now(timezone.utc),
            tuple(DiscoveredModel(deployment.reference, deployment.artifact_identity, loaded)
                  for deployment, loaded in entries.values()))

    async def embed(self, request):
        capabilities = await self.capabilities(request.model.deployment)
        self.validate_embedding_request(request, capabilities)
        if request.execution_generation is None:
            raise failure(ErrorCode.CONFLICT, "Embedding execution generation was not bound")
        snapshot = await with_context(self.resolver.snapshot(), request.context)
        deployment = snapshot.resolve_deployment(request.model.deployment.id, expected_revision=request.model.snapshot_revision)
        generation = {"boot_id": request.execution_generation.boot_id, "process_id": request.execution_generation.process_id}
        payload = {"version": 1, "request_id": request.context.request_id, "generation": generation,
            "catalog_digest": self.catalog_digest, "deployment_id": deployment.id,
            "profile_id": request.model.profile_id,
            "input_type": None if request.model.embedding_role is None else request.model.embedding_role.value,
            "artifact_fingerprint": deployment.artifact_identity.fingerprint,
            "configuration_fingerprint": deployment.configuration_fingerprint,
            "inputs": list(request.inputs), "dimensions": request.dimensions or request.model.profile_metadata["dimensions"]}
        raw = await self._json("POST", "/api/inference/embed", request.context, payload)
        try:
            identity = {key: value for key, value in payload.items() if key not in {"inputs", "dimensions"}}
            if (set(raw) != {*identity, "done", "token_counting", "embeddings", "usage"}
                    or any(raw[key] != value or type(raw[key]) is not type(value) for key, value in identity.items())
                    or raw["done"] is not True or raw["token_counting"] != TOKEN_COUNTING
                    or type(raw["usage"]) is not dict or set(raw["usage"]) != {"input_tokens", "output_tokens"}
                    or type(raw["embeddings"]) is not list
                    or len(raw["embeddings"]) != len(request.inputs)
                    or any(type(row) is not list or len(row) != payload["dimensions"] for row in raw["embeddings"])):
                raise ValueError()
            usage = TokenUsage(**raw["usage"])
            if usage.input_tokens < len(request.inputs):
                raise ValueError()
            return EmbeddingResult(request.context.request_id, tuple(map(tuple, raw["embeddings"])), usage)
        except (TypeError, ValueError, KeyError):
            raise failure(ErrorCode.PROVIDER_ERROR, "Embedding result identity or shape is invalid") from None

    async def wait_request_end(self, deployment, *, generation, context):
        return False  # Idle health cannot disprove a delayed or still-running job.

    async def load(self, deployment, *, snapshot_revision, expected_generation, context):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding cold load needs a measured runtime profile")

    async def start(self, context):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding lifecycle is not configured")

    async def stop(self, expected_generation, context):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding lifecycle is not configured")

    async def unload(self, deployment, **kwargs):
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Embedding lifecycle is not configured")

    async def aclose(self):
        if not self._closed:
            self._closed = True
            await self.client.aclose()
