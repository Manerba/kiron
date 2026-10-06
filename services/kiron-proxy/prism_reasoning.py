"""Exact, artifact-bound partition of ORIGINAL Prism completion token IDs.

This is the closed initial-block-token-ids-v1 rule, not a model-name heuristic.
Generated initial markers belong to reasoning; EOS and whitespace following the
closing marker do not. Prefill is input. Tool transitions, reopening, textual
markers assembled from ordinary tokens, and ambiguous stop truncation fail.
The caller fetches original-ID pieces under its existing generation/admission.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import re

from kiron_common.local_inference import ErrorCode, LocalInferenceError, RuntimeFailure, TokenUsage

RULE_ID = "initial-block-token-ids-v1"
DISABLED_RULE_ID = "disabled-token-ids-v1"
SPACE = b" \t\n\r\v\f"
MAX_PIECE_BYTES = 16 * 1024 * 1024


def _failure():
    raise LocalInferenceError(RuntimeFailure(ErrorCode.PROVIDER_ERROR,
        "Exact reasoning token partition could not be established"))


@dataclass(frozen=True, slots=True)
class ReasoningTokenRules:
    artifact_sha256: str
    template_revision: str
    generation_prefix: str
    initial_state: str  # prefilled, generated, or disabled
    opening: tuple[int, str]
    closing: tuple[int, str]
    eos: tuple[tuple[int, str], ...]
    tools: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        if (type(self.artifact_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256)
                or type(self.template_revision) is not str or not self.template_revision
                or type(self.generation_prefix) is not str
                or self.initial_state not in {"prefilled", "generated", "disabled"}):
            raise ValueError("invalid reasoning rule identity")
        for name in ("opening", "closing"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("eos", "tools"):
            object.__setattr__(self, name, tuple(tuple(value) for value in getattr(self, name)))
        tokens = (self.opening, self.closing, *self.eos, *self.tools)
        if not self.eos or any(len(value) != 2 or type(value[0]) is not int or value[0] < 0
                or type(value[1]) is not str or not value[1] for value in tokens):
            raise ValueError("invalid reasoning marker definitions")
        if len({value[0] for value in tokens}) != len(tokens):
            raise ValueError("reasoning token identities overlap")
        for _, value in tokens:
            value.encode("utf-8", errors="strict")


def native_reasoning_options(*, enabled, budget_tokens=None):
    """Caller selects enabled/budget only from its evidence-backed effort map."""
    if type(enabled) is not bool or (budget_tokens is not None and
            (type(budget_tokens) is not int or budget_tokens < 0)):
        raise ValueError("invalid native reasoning controls")
    result = {"reasoning_format": "deepseek", "chat_template_kwargs": {"enable_thinking": enabled},
              "return_tokens": True, "verbose": True}
    if not enabled:
        if budget_tokens is not None:
            raise ValueError("disabled reasoning must not carry a budget")
        result["reasoning_effort"] = "none"
    elif budget_tokens is not None:
        result["reasoning_budget_tokens"] = budget_tokens
    return result


def exact_reasoning_usage(verbose, usage, *, token_pieces, reasoning_text, final_text,
                          rules, artifact_sha256, template_revision):
    if not isinstance(rules, ReasoningTokenRules) or not isinstance(usage, TokenUsage):
        _failure()
    if artifact_sha256 != rules.artifact_sha256 or template_revision != rules.template_revision:
        _failure()
    try:
        ids = verbose["tokens"]
        settings = verbose["generation_settings"]
        if (not isinstance(verbose, Mapping) or not isinstance(settings, Mapping) or type(ids) is not list
                or any(type(value) is not int or value < 0 for value in ids)
                or len(ids) != usage.output_tokens or len(ids) > 100000
                or type(verbose["tokens_predicted"]) is not int
                or verbose["tokens_predicted"] != len(ids) or verbose["stop"] is not True
                or verbose["stop_type"] not in {"eos", "limit"}
                or type(settings.get("stream", False)) is not bool
                or settings["generation_prompt"] != rules.generation_prefix
                or type(reasoning_text) is not str or type(final_text) is not str
                or len(token_pieces) != len(ids)):
            _failure()
        decoded, piece_bytes = [], 0
        for token_id, entry in zip(ids, token_pieces):
            if type(entry["id"]) is not int or entry["id"] != token_id:
                _failure()
            piece = entry["piece"]
            if type(piece) is str:
                raw = piece.encode("utf-8", errors="strict")
            elif type(piece) is list and all(type(byte) is int and 0 <= byte <= 255 for byte in piece):
                raw = bytes(piece)
            else:
                _failure()
            piece_bytes += len(raw)
            if piece_bytes > MAX_PIECE_BYTES:
                _failure()
            decoded.append((token_id, raw))
        known = dict((token_id, text.encode()) for token_id, text in
                     (rules.opening, rules.closing, *rules.eos, *rules.tools))
        eos_ids, tool_ids = {value[0] for value in rules.eos}, {value[0] for value in rules.tools}
        state = "reasoning" if rules.initial_state == "prefilled" else rules.initial_state
        count, thought, final, emitted = 0, bytearray(), bytearray(), bytearray()
        for index, (token_id, piece) in enumerate(decoded):
            if token_id in known and piece != known[token_id]:
                _failure()
            if token_id in eos_ids:
                if index != len(ids) - 1:
                    _failure()
                continue
            emitted.extend(piece)
            if token_id in tool_ids:
                _failure()
            if token_id == rules.opening[0]:
                if state != "generated" or index != 0:
                    _failure()
                state = "reasoning"
                count += 1
            elif token_id == rules.closing[0]:
                if state != "reasoning":
                    _failure()
                count += 1
                state = "final"
            elif state == "reasoning":
                count += 1
                thought.extend(piece)
            else:
                state = "final"
                final.extend(piece)
        if verbose["stop_type"] == "eos" and (not ids or ids[-1] not in eos_ids):
            _failure()
        # Disallow textual marker spellings that bypass the pinned atomic IDs.
        for marker in known.values():
            if marker in thought or marker in final:
                _failure()
        # Qwen's initial p.space() and reasoning << content strip only leading
        # parser whitespace. Trailing content is preserved, never guessed away.
        if bytes(thought).lstrip(SPACE).decode("utf-8") != reasoning_text:
            _failure()
        if bytes(final).lstrip(SPACE).decode("utf-8") != final_text:
            _failure()
        if not settings.get("stream", False) and bytes(emitted).decode("utf-8") != verbose["content"]:
            _failure()
        return replace(usage, reasoning_output_tokens=count)
    except (KeyError, TypeError, ValueError, UnicodeError):
        _failure()


def exact_disabled_reasoning_usage(verbose, usage, *, token_pieces, rules,
                                   artifact_sha256, template_revision):
    """Prove zero from a closed template prefill and every original output ID.

    Native tool syntax is ordinary non-reasoning output here. Neither atomic
    reasoning delimiters nor their spelling across ordinary tokens may occur.
    This also handles a length-limited trailing UTF-8 fragment without counting
    retokenized text or pretending the unfinished fragment was emitted.
    """
    if (not isinstance(rules, ReasoningTokenRules) or rules.initial_state != "disabled"
            or not isinstance(usage, TokenUsage) or artifact_sha256 != rules.artifact_sha256
            or template_revision != rules.template_revision):
        _failure()
    try:
        ids = verbose["tokens"]
        settings = verbose["generation_settings"]
        if (not isinstance(settings, Mapping) or type(ids) is not list or len(ids) != usage.output_tokens or len(ids) > 100000
                or any(type(value) is not int or value < 0 for value in ids)
                or type(verbose["tokens_predicted"]) is not int or verbose["tokens_predicted"] != len(ids)
                or verbose["stop"] is not True or verbose["stop_type"] not in {"eos", "limit", "word"}
                or type(settings.get("stream", False)) is not bool
                or settings["generation_prompt"] != rules.generation_prefix
                or len(token_pieces) != len(ids)):
            _failure()
        output = bytearray()
        known = {token_id: value.encode("utf-8") for token_id, value in
                 (rules.opening, rules.closing, *rules.eos, *rules.tools)}
        eos_ids = {token_id for token_id, _ in rules.eos}
        for index, (token_id, entry) in enumerate(zip(ids, token_pieces)):
            if type(entry["id"]) is not int or entry["id"] != token_id:
                _failure()
            piece = entry["piece"]
            if type(piece) is str:
                raw = piece.encode("utf-8", errors="strict")
            elif type(piece) is list and all(type(byte) is int and 0 <= byte <= 255 for byte in piece):
                raw = bytes(piece)
            else:
                _failure()
            if token_id in known and raw != known[token_id]:
                _failure()
            if token_id in {rules.opening[0], rules.closing[0]} or len(output) + len(raw) > MAX_PIECE_BYTES:
                _failure()
            if token_id in eos_ids and index != len(ids) - 1:
                _failure()
            output.extend(raw)
        if verbose["stop_type"] == "eos" and (not ids or ids[-1] not in eos_ids):
            _failure()
        if rules.opening[1].encode("utf-8") in output or rules.closing[1].encode("utf-8") in output:
            _failure()
        return replace(usage, reasoning_output_tokens=0)
    except (KeyError, TypeError, ValueError, UnicodeError):
        _failure()
