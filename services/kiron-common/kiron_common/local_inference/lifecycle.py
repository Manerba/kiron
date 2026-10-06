"""Generation-bound observations and operation context; no process management."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from kiron_common.model_catalog import BackendType
from kiron_common.model_state import RuntimeState

from ._values import finite, sha256, text, timestamp
from .identity import ArtifactIdentity


class ErrorCode(str, Enum):
    INVALID_CONFIGURATION = "invalid_configuration"
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNSUPPORTED_PARAMETER = "unsupported_parameter"
    UNSUPPORTED_VALUE = "unsupported_value"
    CONFLICT = "conflict"
    OVERLOADED = "overloaded"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    code: ErrorCode
    message: str
    parameter: str | None = None

    def __post_init__(self):
        if not isinstance(self.code, ErrorCode):
            raise TypeError("failure requires a typed code")
        text(self.message, "failure message")


class LocalInferenceError(Exception):
    def __init__(self, failure: RuntimeFailure):
        self.failure = failure
        super().__init__(failure.message)


class ResolverConfigurationError(LocalInferenceError, ValueError):
    def __init__(self, message: str):
        super().__init__(RuntimeFailure(ErrorCode.INVALID_CONFIGURATION, message))


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class RuntimeTimeouts:
    """Separate positive budgets in seconds, not transport configuration."""

    startup: float
    readiness: float
    first_token: float
    idle: float
    total: float
    drain: float
    stop: float

    def __post_init__(self):
        for name in ("startup", "readiness", "first_token", "idle", "total", "drain", "stop"):
            value = getattr(self, name)
            finite(value, name)
            if value <= 0:
                raise ValueError(f"{name} timeout must be positive")


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    deadline_monotonic: float
    cancellation: CancellationSignal

    def __post_init__(self):
        text(self.request_id, "request ID")
        finite(self.deadline_monotonic, "deadline")
        if not callable(getattr(self.cancellation, "is_set", None)):
            raise TypeError("cancellation must expose is_set()")


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    boot_id: str
    # Random spawn token, never an OS PID that can be reused.
    process_id: str | None = None

    def __post_init__(self):
        text(self.boot_id, "boot generation")
        if self.process_id is not None:
            text(self.process_id, "process generation")


class ProviderHealth(str, Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    STARTABLE = "startable"


@dataclass(frozen=True, slots=True)
class DeploymentObservation:
    deployment_id: str
    state: RuntimeState
    generation: RuntimeGeneration | None
    configuration_fingerprint: str | None
    error: RuntimeFailure | None = None

    def __post_init__(self):
        text(self.deployment_id, "deployment ID")
        if not isinstance(self.state, RuntimeState):
            raise TypeError("state requires RuntimeState")
        if self.generation is not None and not isinstance(self.generation, RuntimeGeneration):
            raise TypeError("invalid runtime generation")
        if self.configuration_fingerprint is not None:
            sha256(self.configuration_fingerprint, "configuration fingerprint")
        if self.state is RuntimeState.LOADED and (
            self.generation is None or self.configuration_fingerprint is None
        ):
            raise ValueError("loaded requires generation and configuration evidence")


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    provider: BackendType
    generation: RuntimeGeneration | None
    observed_at: datetime
    health: ProviderHealth
    models: Mapping[str, DeploymentObservation] = field(default_factory=dict)
    error: RuntimeFailure | None = None

    def __post_init__(self):
        timestamp(self.observed_at)
        if not isinstance(self.provider, BackendType) or not isinstance(self.health, ProviderHealth):
            raise TypeError("invalid provider observation")
        models = dict(self.models)
        for key, value in models.items():
            if not isinstance(value, DeploymentObservation) or key != value.deployment_id:
                raise ValueError("invalid deployment observation map")
            if value.state is RuntimeState.LOADED and (
                self.health is not ProviderHealth.AVAILABLE or value.generation != self.generation
            ):
                raise ValueError("loaded observation must belong to the healthy current generation")
        object.__setattr__(self, "models", MappingProxyType(models))


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    reference: str
    artifact_identity: ArtifactIdentity
    installed: bool

    def __post_init__(self):
        text(self.reference, "model reference")
        if not isinstance(self.artifact_identity, ArtifactIdentity) or type(self.installed) is not bool:
            raise TypeError("invalid discovered model")


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    provider: BackendType
    revision: str
    observed_at: datetime
    models: tuple[DiscoveredModel, ...] = ()
    error: RuntimeFailure | None = None

    def __post_init__(self):
        if not isinstance(self.provider, BackendType):
            raise TypeError("invalid discovery provider")
        text(self.revision, "discovery revision")
        timestamp(self.observed_at)
        models = tuple(self.models)
        if any(not isinstance(value, DiscoveredModel) for value in models):
            raise TypeError("invalid discovery model")
        object.__setattr__(self, "models", models)


@dataclass(frozen=True, slots=True)
class LifecycleResult:
    operation_id: str
    observation: ProviderObservation
    changed: bool

    def __post_init__(self):
        text(self.operation_id, "operation ID")
        if not isinstance(self.observation, ProviderObservation) or type(self.changed) is not bool:
            raise TypeError("invalid lifecycle result")
