"""Small asynchronous provider boundary; implementations live in composition."""

from collections.abc import AsyncIterator
from enum import Enum
from typing import Protocol

from kiron_common.model_catalog import BackendType

from .capabilities import CapabilitySet
from .content import EmbeddingRequest, EmbeddingResult, GenerateRequest, InferenceRequest, InferenceResult
from .events import InferenceEvent
from .identity import ResolvedDeployment
from .lifecycle import DiscoverySnapshot, LifecycleResult, ProviderObservation, RequestContext, RuntimeGeneration


class ModelLifecycleOperation(str, Enum):
    """Implemented model mutations; service start/stop is a separate contract."""

    LOAD = "load"
    UNLOAD = "unload"


class RuntimeProvider(Protocol):
    @property
    def provider(self) -> BackendType: ...

    @property
    def model_lifecycle_operations(self) -> frozenset[ModelLifecycleOperation]:
        """Declared adapter support, separate from current health and admission."""
        ...

    async def discover(self, context: RequestContext) -> DiscoverySnapshot: ...
    async def health(self, context: RequestContext) -> ProviderObservation: ...
    async def capabilities(self, deployment: ResolvedDeployment) -> CapabilitySet: ...
    def validate_request(self, request: InferenceRequest, capabilities: CapabilitySet) -> None:
        """Pure mapping validation; no provider I/O, lifecycle or admission mutation."""
        ...
    async def start(self, context: RequestContext) -> LifecycleResult: ...
    async def stop(self, expected_generation: RuntimeGeneration, context: RequestContext) -> LifecycleResult: ...

    async def load(self, deployment: ResolvedDeployment, *, snapshot_revision: str,
                   expected_generation: RuntimeGeneration, context: RequestContext) -> LifecycleResult: ...

    async def unload(self, deployment: ResolvedDeployment, *, snapshot_revision: str,
                     expected_generation: RuntimeGeneration, context: RequestContext) -> LifecycleResult: ...

    async def chat(self, request: InferenceRequest) -> InferenceResult: ...
    def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceEvent]: ...
    async def generate(self, request: GenerateRequest) -> InferenceResult: ...
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...
    async def wait_request_end(self, deployment: ResolvedDeployment, *, generation: RuntimeGeneration,
                               context: RequestContext) -> bool:
        """Prove generation-bound backend completion within the drain budget.

        Transport closure alone is insufficient. False or timeout leaves the
        RuntimeService admission ticket in its blocking unknown state.
        """
        ...

    async def aclose(self) -> None: ...
