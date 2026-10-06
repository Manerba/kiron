"""Decode an evidence-bound Prism effort/token policy without model heuristics.

The policy lives in the deployment's capability report, not a second registry.
The artifact hash binds its embedded template; template_sha256 additionally
identifies the exact template used when collecting the token-rule evidence.
"""
from dataclasses import dataclass
import re

from kiron_common.local_inference import CapabilityName, ErrorCode
from provider_transport import failure
from prism_reasoning import DISABLED_RULE_ID, RULE_ID, ReasoningTokenRules


@dataclass(frozen=True, slots=True)
class ReasoningPlan:
    budget_tokens: int | None
    rules: ReasoningTokenRules
    enabled: bool


def reasoning_plan(request, capabilities, implementation):
    effort = request.options.reasoning.effort
    if request.options.reasoning.summary is not None:
        raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Reasoning summaries are not verified", "reasoning.summary")
    deployment = request.model.deployment
    enabled = effort not in {None, "none"}
    if enabled:
        if not capabilities.supports(CapabilityName.REASONING, deployment, implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Reasoning mapping is not verified", "reasoning_effort")
        constraints = capabilities.by_name[CapabilityName.REASONING].constraints
        supported = constraints.get("efforts")
        if supported is None or supported.allowed_values is None or not supported.accepts(effort):
            raise failure(ErrorCode.UNSUPPORTED_VALUE, "Reasoning effort is not verified", "reasoning_effort")
    else:
        constraints = capabilities.by_name[CapabilityName.CHAT].constraints
        usage = constraints.get("usage_fields")
        if usage is None or usage.allowed_values is None or "reasoning_output_tokens" not in usage.allowed_values:
            return None
        if not capabilities.supports(CapabilityName.CHAT, deployment, implementation):
            raise failure(ErrorCode.UNSUPPORTED_CAPABILITY, "Exact disabled reasoning usage is not verified")

    def values(name):
        constraint = constraints[name]
        if (constraint.allowed_values is None or constraint.minimum is not None
                or constraint.maximum is not None or constraint.variants):
            raise ValueError("token policy requires exact values")
        return constraint.allowed_values

    def single(name, expected):
        items = values(name)
        if len(items) != 1 or type(items[0]) is not expected:
            raise ValueError("token policy requires one typed value")
        return items[0]

    def markers(name):
        ids, texts = values(name + "_ids"), values(name + "_texts")
        if len(ids) != len(texts) or len(ids) > 8:
            raise ValueError("invalid token marker list")
        return tuple(zip(ids, texts))

    try:
        if single("token_rule", str) != (RULE_ID if enabled else DISABLED_RULE_ID):
            raise ValueError("unknown token rule")
        template = single("template_sha256", str)
        if not re.fullmatch(r"[0-9a-f]{64}", template):
            raise ValueError("template identity")
        # An externally selected template would need its own proven policy.
        if implementation.template_revision not in {None, template}:
            raise ValueError("template override")
        budget = single("budget_tokens." + effort, int) if enabled else None
        prefix = single("generation_prefix", str)
        state = single("initial_state", str)
        if ((enabled and not 1 <= budget <= 100000) or len(prefix.encode("utf-8")) > 4096
                or state not in ({"prefilled", "generated"} if enabled else {"disabled"})):
            raise ValueError("invalid reasoning controls")
        rules = ReasoningTokenRules(deployment.artifact_identity.sha256, template, prefix, state,
            (single("opening_id", int), single("opening_text", str)),
            (single("closing_id", int), single("closing_text", str)), markers("eos"), markers("tool"))
        return ReasoningPlan(budget, rules, enabled)
    except (KeyError, TypeError, ValueError, UnicodeError):
        raise failure(ErrorCode.INVALID_CONFIGURATION, "Reasoning token policy is incomplete or invalid") from None
