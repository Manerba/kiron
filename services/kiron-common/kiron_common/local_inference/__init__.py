"""Pure local-inference contracts. No clients, services or discovery at import."""

from .capabilities import Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus, ParameterConstraint, RuntimeImplementation
from .content import (
    AssistantItem, AssistantTextItem, ContentPart, EmbeddingRequest, EmbeddingResult, FinishReason, GenerateRequest,
    GenerationOptions, ImagePart, InferenceRequest, InferenceResult, Message,
    MessageRole, OutputFormat, OutputFormatKind, ReasoningKind, ReasoningOptions,
    ReasoningPart, SamplingOptions, TextPart, TokenUsage, ToolCall, ToolChoice,
    ToolChoiceKind, ToolDefinition,
)
from .events import EventIndexState, EventKind, InferenceEvent, OutputEventLayout, TERMINAL_EVENTS, validate_event_sequence
from .identity import ArtifactFileReference, ArtifactIdentity, EmbeddingRole, ModelSource, ResolvedDeployment, ResolvedModel, ResourceProfile
from .lifecycle import (
    CancellationSignal, DeploymentObservation, DiscoveredModel, DiscoverySnapshot,
    ErrorCode, LifecycleResult, LocalInferenceError, ProviderHealth, ProviderObservation,
    RequestContext, ResolverConfigurationError, RuntimeGeneration, RuntimeTimeouts,
    RuntimeFailure,
)
from .provider import ModelLifecycleOperation, RuntimeProvider
from .resolver import ResolverSnapshot, build_resolver_snapshot
