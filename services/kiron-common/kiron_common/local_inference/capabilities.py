"""Evidence-bound capability declarations, never inferred from model names."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType

from ._values import finite, json_mapping, sha256, text, timestamp
from .identity import ResolvedDeployment


class CapabilityName(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    VISION = "vision"
    FUNCTION_TOOLS = "function_tools"
    PARALLEL_TOOLS = "parallel_tools"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    EMBEDDINGS = "embeddings"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class RuntimeImplementation:
    provider_revision: str
    template_revision: str | None
    parser_revision: str | None

    def __post_init__(self):
        text(self.provider_revision, "provider revision")
        for name in ("template_revision", "parser_revision"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ParameterConstraint:
    allowed_values: tuple[str | int | float | bool | None, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    # Closed, evidence-backed variants such as formats/schema keywords are data,
    # not instructions for passing arbitrary parameters through to a provider.
    variants: tuple[str, ...] = ()

    def __post_init__(self):
        if self.allowed_values is not None:
            values = json_mapping({"values": tuple(self.allowed_values)})["values"]
            if any(isinstance(value, (Mapping, tuple)) for value in values):
                raise ValueError("allowed values must be JSON scalars")
            object.__setattr__(self, "allowed_values", values)
        for name in ("minimum", "maximum"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("invalid constraint interval")
        variants = tuple(self.variants)
        for item in variants:
            text(item, "constraint variant")
        object.__setattr__(self, "variants", variants)

    def accepts(self, value: object) -> bool:
        if self.allowed_values is not None and not any(
            type(value) is type(candidate) and value == candidate for candidate in self.allowed_values
        ):
            return False
        if self.minimum is not None or self.maximum is not None:
            try:
                finite(value, "parameter")
            except ValueError:
                return False
            if self.minimum is not None and value < self.minimum:
                return False
            if self.maximum is not None and value > self.maximum:
                return False
        return True


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    provider_revision: str
    artifact_fingerprint: str
    artifact_sha256: str | None
    projector_sha256: str | None
    template_revision: str | None
    parser_revision: str | None
    configuration_fingerprint: str
    test_reference: str
    observed_at: datetime

    def __post_init__(self):
        for name in ("provider_revision", "test_reference"):
            text(getattr(self, name), name)
        for name in ("artifact_fingerprint", "configuration_fingerprint"):
            sha256(getattr(self, name), name)
        for name in ("artifact_sha256", "projector_sha256"):
            if getattr(self, name) is not None:
                sha256(getattr(self, name), name)
        timestamp(self.observed_at)

    def matches(self, deployment: ResolvedDeployment, implementation: RuntimeImplementation) -> bool:
        artifact = deployment.artifact_identity
        projector_sha = artifact.projector.sha256 if artifact.projector else None
        return (artifact.content_key is not None
                and self.provider_revision == implementation.provider_revision
                and self.template_revision == implementation.template_revision
                and self.parser_revision == implementation.parser_revision
                and self.configuration_fingerprint == deployment.configuration_fingerprint
                and self.artifact_fingerprint == artifact.fingerprint
                and self.artifact_sha256 == artifact.sha256
                and self.projector_sha256 == projector_sha)


@dataclass(frozen=True, slots=True)
class Capability:
    status: CapabilityStatus = CapabilityStatus.UNVERIFIED
    constraints: Mapping[str, ParameterConstraint] = field(default_factory=dict)
    evidence: tuple[CapabilityEvidence, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, CapabilityStatus):
            raise TypeError("invalid capability status")
        constraints = dict(self.constraints)
        for name, value in constraints.items():
            text(name, "parameter name")
            if not isinstance(value, ParameterConstraint):
                raise TypeError("constraints require ParameterConstraint values")
        evidence = tuple(self.evidence)
        if any(not isinstance(value, CapabilityEvidence) for value in evidence):
            raise TypeError("invalid capability evidence")
        if self.status is CapabilityStatus.SUPPORTED and not evidence:
            raise ValueError("supported capability requires evidence")
        object.__setattr__(self, "constraints", MappingProxyType(constraints))
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    by_name: Mapping[CapabilityName, Capability] = field(default_factory=dict)

    def __post_init__(self):
        values = {name: Capability() for name in CapabilityName}
        for name, value in self.by_name.items():
            if not isinstance(name, CapabilityName) or not isinstance(value, Capability):
                raise TypeError("invalid capability mapping")
            values[name] = value
        object.__setattr__(self, "by_name", MappingProxyType(values))

    def supports(self, name: CapabilityName, deployment: ResolvedDeployment,
                 implementation: RuntimeImplementation) -> bool:
        capability = self.by_name[name]
        return (capability.status is CapabilityStatus.SUPPORTED
                and all(evidence.matches(deployment, implementation) for evidence in capability.evidence))

    def for_deployment(self, deployment: ResolvedDeployment,
                       implementation: RuntimeImplementation) -> "CapabilitySet":
        return CapabilitySet({name: (replace(value, status=CapabilityStatus.UNVERIFIED)
                                     if value.status is CapabilityStatus.SUPPORTED
                                     and not self.supports(name, deployment, implementation) else value)
                              for name, value in self.by_name.items()})
