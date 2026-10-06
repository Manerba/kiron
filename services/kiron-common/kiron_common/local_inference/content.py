"""Ordered, provider-neutral content and exact native usage values."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Mapping, TypeAlias

from ._values import finite, integer, json_mapping, sha256, text
from .identity import EmbeddingRole, ResolvedModel
from .lifecycle import RequestContext, RuntimeGeneration


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str

    def __post_init__(self):
        if type(self.text) is not str:
            raise TypeError("text content must be a string")


@dataclass(frozen=True, slots=True)
class ImagePart:
    media_type: str
    data: bytes
    sha256: str
    width: int
    height: int
    detail: str = "auto"
    source_media_type: str | None = None
    source_sha256: str | None = None

    def __post_init__(self):
        if self.media_type not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError("unsupported image media type")
        if type(self.data) is not bytes or not self.data:
            raise TypeError("image data must be immutable decoded bytes")
        sha256(self.sha256, "image sha256")
        if hashlib.sha256(self.data).hexdigest() != self.sha256:
            raise ValueError("image digest mismatch")
        integer(self.width, "image width", 1)
        integer(self.height, "image height", 1)
        if self.detail not in ("auto", "low", "high"):
            raise ValueError("unsupported image detail")
        if (self.source_media_type is None) != (self.source_sha256 is None):
            raise ValueError("source image media type and digest must be paired")
        if self.source_media_type is not None:
            if self.source_media_type not in ("image/png", "image/jpeg", "image/webp"):
                raise ValueError("unsupported source image media type")
            sha256(self.source_sha256, "source image sha256")


ContentPart: TypeAlias = TextPart | ImagePart


class MessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ReasoningKind(str, Enum):
    TEXT = "text"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class ReasoningPart:
    kind: ReasoningKind
    text: str

    def __post_init__(self):
        if not isinstance(self.kind, ReasoningKind) or type(self.text) is not str:
            raise TypeError("invalid reasoning part")


@dataclass(frozen=True, slots=True)
class ToolCall:
    index: int
    id: str
    name: str
    arguments: str
    complete: bool = True

    def __post_init__(self):
        integer(self.index, "tool call index")
        text(self.id, "tool call ID")
        text(self.name, "tool name")
        if type(self.arguments) is not str:
            raise TypeError("tool arguments must be JSON text")
        if type(self.complete) is not bool:
            raise TypeError("tool completeness must be explicit")
        # A length-limited result may retain partial argument text losslessly.
        if self.complete:
            json_mapping(json.loads(self.arguments))


@dataclass(frozen=True, slots=True)
class AssistantTextItem:
    """One assistant output message, retaining its ordered text parts."""
    content: tuple[TextPart, ...]

    def __post_init__(self):
        parts = tuple(self.content)
        if not parts or any(not isinstance(part, TextPart) for part in parts):
            raise TypeError("assistant text items require text parts")
        object.__setattr__(self, "content", parts)


AssistantItem: TypeAlias = AssistantTextItem | ToolCall | ReasoningPart


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: tuple[ReasoningPart, ...] = ()
    assistant_items: tuple[AssistantItem, ...] = ()

    def __post_init__(self):
        if not isinstance(self.role, MessageRole):
            raise TypeError("message role must be normalized")
        for name, accepted in (("content", (TextPart, ImagePart)), ("tool_calls", (ToolCall,)),
                               ("reasoning", (ReasoningPart,))):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, accepted) for value in values):
                raise TypeError(f"invalid {name}")
            object.__setattr__(self, name, values)
        items = tuple(self.assistant_items)
        if items and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant turns carry ordered output items")
        if any(not isinstance(item, (AssistantTextItem, ToolCall, ReasoningPart)) for item in items):
            raise TypeError("invalid assistant turn item")
        if items:
            projections = {
                "content": tuple(part for item in items if isinstance(item, AssistantTextItem) for part in item.content),
                "tool_calls": tuple(item for item in items if isinstance(item, ToolCall)),
                "reasoning": tuple(item for item in items if isinstance(item, ReasoningPart)),
            }
            for name, projected in projections.items():
                if getattr(self, name) and getattr(self, name) != projected:
                    raise ValueError("assistant turn projection differs from its ordered items")
                object.__setattr__(self, name, projected)
        elif self.role is MessageRole.ASSISTANT:
            # Chat has one content field before its call list. Responses supplies
            # explicit items instead; both become the same ordered turn contract.
            items = (*self.reasoning, *((AssistantTextItem(self.content),) if self.content else ()), *self.tool_calls)
        object.__setattr__(self, "assistant_items", items)
        if self.tool_call_id is not None:
            text(self.tool_call_id, "tool result call ID")
        if (self.role is MessageRole.TOOL) != (self.tool_call_id is not None):
            raise ValueError("only tool results require tool_call_id")
        if (self.tool_calls or self.reasoning) and self.role is not MessageRole.ASSISTANT:
            raise ValueError("only assistant messages carry calls/reasoning")
        if len({c.id for c in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("duplicate tool call ID")
        if len({c.index for c in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("duplicate tool call index")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    parameters: Mapping
    description: str = ""
    strict: bool = False

    def __post_init__(self):
        text(self.name, "tool name")
        if type(self.description) is not str or type(self.strict) is not bool:
            raise TypeError("invalid tool description/strict flag")
        object.__setattr__(self, "parameters", json_mapping(self.parameters))


class ToolChoiceKind(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class ToolChoice:
    kind: ToolChoiceKind = ToolChoiceKind.AUTO
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, ToolChoiceKind):
            raise TypeError("invalid tool choice")
        if (self.kind is ToolChoiceKind.NAMED) != (self.name is not None):
            raise ValueError("only a named tool choice carries a name")
        if self.name is not None:
            text(self.name, "chosen tool name")


class OutputFormatKind(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True, slots=True)
class OutputFormat:
    kind: OutputFormatKind = OutputFormatKind.TEXT
    schema: Mapping | None = None
    name: str | None = None
    strict: bool = False
    description: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, OutputFormatKind) or type(self.strict) is not bool:
            raise TypeError("invalid output format")
        is_schema = self.kind is OutputFormatKind.JSON_SCHEMA
        if is_schema != (self.schema is not None and self.name is not None):
            raise ValueError("JSON schema output requires schema and name")
        if not is_schema and (self.schema is not None or self.name is not None or self.strict or self.description is not None):
            raise ValueError("schema fields require JSON schema output")
        if self.description is not None and type(self.description) is not str:
            raise TypeError("output schema description must be text")
        if self.schema is not None:
            text(self.name, "output schema name")
            object.__setattr__(self, "schema", json_mapping(self.schema))


@dataclass(frozen=True, slots=True)
class ReasoningOptions:
    effort: str | None = None
    summary: str | None = None

    def __post_init__(self):
        if self.effort not in (None, "none", "minimal", "low", "medium", "high", "xhigh"):
            raise ValueError("invalid reasoning effort")
        if self.summary not in (None, "auto", "concise", "detailed"):
            raise ValueError("invalid reasoning summary")


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()

    def __post_init__(self):
        for name, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1),
                                    ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
                if not lower <= value <= upper:
                    raise ValueError(f"invalid {name}")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        stop = tuple(self.stop)
        if len(stop) > 4 or any(type(value) is not str or not value for value in stop):
            raise ValueError("invalid stop sequences")
        object.__setattr__(self, "stop", stop)


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_output_tokens: int
    sampling: SamplingOptions = field(default_factory=SamplingOptions)
    output_format: OutputFormat = field(default_factory=OutputFormat)
    reasoning: ReasoningOptions = field(default_factory=ReasoningOptions)
    parallel_tool_calls: bool = False

    def __post_init__(self):
        integer(self.max_output_tokens, "max output tokens", 1)
        if (not isinstance(self.sampling, SamplingOptions) or not isinstance(self.output_format, OutputFormat)
                or not isinstance(self.reasoning, ReasoningOptions) or type(self.parallel_tool_calls) is not bool):
            raise TypeError("invalid generation options")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    model: ResolvedModel
    messages: tuple[Message, ...]
    options: GenerationOptions
    context: RequestContext
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    execution_generation: RuntimeGeneration | None = None

    def __post_init__(self):
        _request_values(self)
        messages, tools = tuple(self.messages), tuple(self.tools)
        if not messages or any(not isinstance(value, Message) for value in messages):
            raise ValueError("inference requires ordered normalized messages")
        if any(not isinstance(value, ToolDefinition) for value in tools):
            raise TypeError("invalid tools")
        names = {value.name for value in tools}
        if len(names) != len(tools):
            raise ValueError("duplicate tool name")
        if not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool choice must be normalized")
        if self.tool_choice.kind is ToolChoiceKind.NAMED and self.tool_choice.name not in names:
            raise ValueError("chosen tool is not defined")
        if self.tool_choice.kind is ToolChoiceKind.REQUIRED and not tools:
            raise ValueError("required tool choice needs tools")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)


def _request_values(value):
    if (not isinstance(value.model, ResolvedModel) or not isinstance(value.options, GenerationOptions)
            or not isinstance(value.context, RequestContext)):
        raise TypeError("request requires resolved model, options and context")
    _execution_generation(value.execution_generation)


def _execution_generation(value):
    if value is not None and not isinstance(value, RuntimeGeneration):
        raise TypeError("execution generation must be a RuntimeGeneration")


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    model: ResolvedModel
    prompt: str
    options: GenerationOptions
    context: RequestContext
    execution_generation: RuntimeGeneration | None = None

    def __post_init__(self):
        _request_values(self)
        if type(self.prompt) is not str or not self.prompt:
            raise ValueError("generation requires a prompt")


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    model: ResolvedModel
    inputs: tuple[str, ...]
    context: RequestContext
    execution_generation: RuntimeGeneration | None = None
    dimensions: int | None = None

    def __post_init__(self):
        _execution_generation(self.execution_generation)
        if not isinstance(self.model, ResolvedModel) or not isinstance(self.context, RequestContext):
            raise TypeError("embedding requires resolved model and context")
        from .embedding import embedding_dimension
        if self.model.embedding_role is None:
            embedding_dimension(self.model)
        elif not isinstance(self.model.embedding_role, EmbeddingRole):
            raise ValueError("embedding requires an explicitly resolved role")
        if self.dimensions is not None:
            integer(self.dimensions, "embedding dimensions", 1)
        inputs = tuple(self.inputs)
        if not inputs or any(type(value) is not str for value in inputs):
            raise ValueError("embedding inputs must be text")
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None

    def __post_init__(self):
        integer(self.input_tokens, "input tokens")
        integer(self.output_tokens, "output tokens")
        for name, limit in (("cached_input_tokens", self.input_tokens), ("reasoning_output_tokens", self.output_tokens)):
            value = getattr(self, name)
            if value is not None:
                integer(value, name)
                if value > limit:
                    raise ValueError(f"{name} exceeds total")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class InferenceResult:
    request_id: str
    content: tuple[TextPart, ...]
    reasoning: tuple[ReasoningPart, ...]
    tool_calls: tuple[ToolCall, ...]
    usage: TokenUsage | None
    finish_reason: FinishReason

    def __post_init__(self):
        text(self.request_id, "request ID")
        for name, accepted in (("content", TextPart), ("reasoning", ReasoningPart), ("tool_calls", ToolCall)):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, accepted) for value in values):
                raise TypeError(f"invalid result {name}")
            object.__setattr__(self, name, values)
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("invalid usage")
        if not isinstance(self.finish_reason, FinishReason):
            raise TypeError("invalid finish reason")
        if self.finish_reason is FinishReason.TOOL_CALLS and not self.tool_calls:
            raise ValueError("tool_calls finish requires calls")
        if any(not call.complete for call in self.tool_calls) and self.finish_reason is not FinishReason.LENGTH:
            raise ValueError("incomplete tool arguments require length termination")
        if (len({call.id for call in self.tool_calls}) != len(self.tool_calls)
                or len({call.index for call in self.tool_calls}) != len(self.tool_calls)):
            raise ValueError("duplicate result tool identity")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    request_id: str
    vectors: tuple[tuple[float, ...], ...]
    usage: TokenUsage

    def __post_init__(self):
        text(self.request_id, "request ID")
        vectors = tuple(tuple(vector) for vector in self.vectors)
        if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("embedding vectors must have one nonzero dimension")
        for vector in vectors:
            for number in vector:
                finite(number, "embedding element")
        if not isinstance(self.usage, TokenUsage) or self.usage.output_tokens != 0:
            raise ValueError("embedding usage must contain only input tokens")
        object.__setattr__(self, "vectors", vectors)
