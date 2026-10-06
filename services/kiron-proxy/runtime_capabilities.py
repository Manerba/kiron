"""Strict decoding of operator-installed test evidence, not model registrations.

Records grant nothing unless CapabilityEvidence matches the current artifact,
configuration, runtime binary/image and adapter source revision at execution.
"""
from datetime import datetime

from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    ParameterConstraint,
)


def _object(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("invalid capability evidence object")
    return value


def _constraint(value):
    _object(value, ("allowed_values", "minimum", "maximum", "variants"))
    if value["allowed_values"] is not None and type(value["allowed_values"]) is not list:
        raise ValueError("invalid capability allowed values")
    if type(value["variants"]) is not list:
        raise ValueError("invalid capability variants")
    return ParameterConstraint(**value)


def _evidence(value):
    _object(value, ("provider_revision", "artifact_fingerprint", "artifact_sha256", "projector_sha256",
        "template_revision", "parser_revision", "configuration_fingerprint", "test_reference", "observed_at"))
    if type(value["observed_at"]) is not str:
        raise ValueError("invalid capability observation timestamp")
    for name in ("template_revision", "parser_revision"):
        item = value[name]
        if item is not None and (type(item) is not str or not item.strip()):
            raise ValueError("invalid capability implementation revision")
    return CapabilityEvidence(**{**value, "observed_at": datetime.fromisoformat(value["observed_at"])})


def decode_provider_evidence(value, provider):
    """Decode one provider independently; malformed evidence never becomes green."""
    _object(value, ("version", "provider", "deployments"))
    if type(value["version"]) is not int or value["version"] != 1 or value["provider"] != provider.value:
        raise ValueError("capability report provider/version mismatch")
    deployments = value["deployments"]
    if type(deployments) is not dict or len(deployments) > 512:
        raise ValueError("invalid capability deployment map")
    result = {}
    for deployment_id, raw in deployments.items():
        if type(deployment_id) is not str or not deployment_id or len(deployment_id) > 256:
            raise ValueError("invalid capability deployment ID")
        if type(raw) is not dict or len(raw) > len(CapabilityName):
            raise ValueError("invalid capability map")
        capabilities = {}
        for name, capability in raw.items():
            name = CapabilityName(name)
            _object(capability, ("status", "constraints", "evidence"))
            if type(capability["constraints"]) is not dict or len(capability["constraints"]) > 64:
                raise ValueError("invalid capability constraints")
            if type(capability["evidence"]) is not list or len(capability["evidence"]) > 32:
                raise ValueError("invalid capability evidence list")
            capabilities[name] = Capability(CapabilityStatus(capability["status"]),
                {key: _constraint(item) for key, item in capability["constraints"].items()},
                tuple(_evidence(item) for item in capability["evidence"]))
        result[deployment_id] = CapabilitySet(capabilities)
    return result


def encode_provider_evidence(provider, deployments):
    """Produce reviewable report bytes from already typed successful test results."""
    return {"version": 1, "provider": provider.value, "deployments": {
        deployment: {name.value: {
            "status": capability.status.value,
            "constraints": {key: {
                "allowed_values": None if item.allowed_values is None else list(item.allowed_values),
                "minimum": item.minimum, "maximum": item.maximum, "variants": list(item.variants),
            } for key, item in capability.constraints.items()},
            "evidence": [{
                "provider_revision": evidence.provider_revision,
                "artifact_fingerprint": evidence.artifact_fingerprint,
                "artifact_sha256": evidence.artifact_sha256,
                "projector_sha256": evidence.projector_sha256,
                "template_revision": evidence.template_revision,
                "parser_revision": evidence.parser_revision,
                "configuration_fingerprint": evidence.configuration_fingerprint,
                "test_reference": evidence.test_reference,
                "observed_at": evidence.observed_at.isoformat(),
            } for evidence in capability.evidence],
        } for name, capability in values.by_name.items()}
        for deployment, values in deployments.items()}}
