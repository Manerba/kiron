"""Small explicit identity/end protocol for already admitted resident models."""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from kiron_common.local_inference import LocalInferenceError, build_resolver_snapshot
from kiron_common.local_inference.embedding import embedding_dimension
from kiron_common.local_inference.embedding_native import service_revision, TOKEN_COUNTING, request_fingerprint
from kiron_common.model_catalog import BackendType, ModelEndpoint
from native_runtime import generation, state_payload
from model_worker import StaleGenerationError


class NativeEmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: int
    request_id: str = Field(min_length=1, max_length=128)
    generation: dict[str, str]
    catalog_digest: str
    deployment_id: str
    profile_id: str
    input_type: str | None
    artifact_fingerprint: str
    configuration_fingerprint: str
    inputs: list[str] = Field(min_length=1, max_length=512)
    dimensions: int = Field(ge=1, le=65536)


@lru_cache(maxsize=1)
def revision():
    packages = ("sentence-transformers", "transformers", "torch", "numpy", "safetensors")
    return service_revision(Path(__file__).parent, versions={name: version(name) for name in packages})


def create_native_router(catalog, service_view, worker_getter):
    router = APIRouter(prefix="/api/inference")
    resolver = build_resolver_snapshot(catalog, ())

    def rejected(request):
        # The complete original request is bound to proof that the serial
        # worker did not execute it. This is not an idle-health end heuristic.
        return JSONResponse({"version": 1, "request_id": request.request_id,
            "generation": request.generation,
            "request_sha256": request_fingerprint(request.model_dump()),
            "rejected": True, "error": {"code": "identity_conflict"}}, status_code=409)

    @router.get("/state")
    def state():
        return state_payload(catalog, worker_getter().snapshot(), revision())

    @router.post("/embed")
    async def embed(request: NativeEmbeddingRequest):
        worker = worker_getter()
        snapshot = worker.snapshot()
        try:
            if request.version != 1 or request.generation != generation(snapshot) or request.catalog_digest != catalog.catalog_digest:
                raise ValueError("generation or catalog changed")
            suffix = {None: "", "search_query": ".query", "search_document": ".document"}[request.input_type]
            model = resolver.resolve(request.profile_id + suffix)
            deployment = model.deployment
            if (deployment.provider is not BackendType.KIRON_EMBEDDINGS
                    or deployment.id != request.deployment_id or model.profile_id != request.profile_id
                    or deployment.artifact_identity.fingerprint != request.artifact_fingerprint
                    or deployment.configuration_fingerprint != request.configuration_fingerprint
                    or embedding_dimension(model) != request.dimensions
                    or snapshot.get("verified_artifacts", {}).get(deployment.reference) != request.artifact_fingerprint
                    or deployment.reference not in snapshot.get("loaded_models", ())
                    or snapshot.get("worker_accepting") is not True or snapshot.get("worker_thread_alive") is not True):
                raise ValueError("resident profile changed")
            selected = service_view.require_runtime_model(deployment.reference).require_profile(ModelEndpoint.EMBED)
            if selected.profile_id != request.profile_id:
                raise ValueError("formatting profile differs")
            if any(not text.strip() or len(text) > 32768 for text in request.inputs):
                raise ValueError("invalid native input length")
            for text in request.inputs:
                text.encode("utf-8")
        except (LocalInferenceError, ValueError, TypeError, KeyError, UnicodeError):
            return rejected(request)
        try:
            result = await worker.encode(deployment.reference, request.inputs, request.input_type,
                expected_artifact=request.artifact_fingerprint, expected_epoch=snapshot["model_epoch"])
            if (type(result.execution_epoch) is not int
                    or result.execution_epoch != snapshot["model_epoch"]
                    or result.execution_artifact != request.artifact_fingerprint
                    or type(result.prompt_eval_count) is not int or result.prompt_eval_count < len(request.inputs)
                    or len(result.embeddings) != len(request.inputs)
                    or any(len(row) != request.dimensions for row in result.embeddings)):
                raise ValueError("native result evidence is inconsistent")
        except StaleGenerationError:
            return rejected(request)
        except Exception:
            return JSONResponse({"error": {"code": "embedding_execution_failed"}}, status_code=503)
        return {"version": 1, "request_id": request.request_id, "generation": request.generation,
            "catalog_digest": catalog.catalog_digest, "deployment_id": deployment.id,
            "profile_id": request.profile_id, "input_type": request.input_type,
            "artifact_fingerprint": request.artifact_fingerprint,
            "configuration_fingerprint": request.configuration_fingerprint,
            "done": True, "token_counting": TOKEN_COUNTING, "embeddings": result.embeddings,
            "usage": {"input_tokens": result.prompt_eval_count, "output_tokens": 0}}

    return router
