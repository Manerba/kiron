"""Original native IDs plus adversarial partitions; no tokenizer or model run."""
from dataclasses import replace

import pytest
import prism_reasoning

from kiron_common.local_inference import ErrorCode, LocalInferenceError, TokenUsage
from prism_reasoning import ReasoningTokenRules, exact_reasoning_usage, native_reasoning_options


RULES = ReasoningTokenRules("a" * 64, "qwen-test-template", "assistant\n<think>\n", "prefilled",
    (100, "<think>"), (101, "</think>"), ((102, "<|im_end|>"),), ((103, "<tool_call>"),))


def check(pieces, *, thought="Thought", final="OK", rules=RULES, stop="eos", stream=False, **changes):
    raw = b"".join(value.encode() if type(value) is str else bytes(value)
                   for token_id, value in pieces if token_id not in dict(rules.eos))
    verbose = {"tokens": [token_id for token_id, _ in pieces], "tokens_predicted": len(pieces),
        "generation_settings": {"generation_prompt": rules.generation_prefix, "stream": stream},
        "stop": True, "stop_type": stop, "content": raw.decode("utf-8") if not stream else ""}
    verbose.update(changes)
    return exact_reasoning_usage(verbose, TokenUsage(9, len(pieces), 2),
        token_pieces=[{"id": token_id, "piece": value} for token_id, value in pieces],
        reasoning_text=thought, final_text=final, rules=rules,
        artifact_sha256=rules.artifact_sha256, template_revision=rules.template_revision)


def test_prefill_close_separator_and_eos_partition_matches_stream_and_nonstream():
    pieces = [(1, "Thought"), (101, "</think>"), (2, "\n\n"), (3, "OK"), (102, "<|im_end|>")]
    for stream in (False, True):
        usage = check(pieces, stream=stream)
        assert usage.reasoning_output_tokens == 2 and usage.output_tokens == 5
        assert usage.cached_input_tokens == 2 and usage.total_tokens == 14


def test_generated_initial_opening_is_reasoning_and_never_prefill():
    rules = replace(RULES, generation_prefix="assistant\n", initial_state="generated")
    pieces = [(100, "<think>"), (1, "\nThought"), (101, "</think>"), (3, "OK"), (102, "<|im_end|>")]
    assert check(pieces, rules=rules).reasoning_output_tokens == 3


def test_utf8_partial_pieces_count_individually_without_retokenizing():
    pieces = [(1, [0xE2]), (2, [0x82]), (3, [0xAC]), (101, "</think>"), (4, "OK"), (102, "<|im_end|>")]
    assert check(pieces, thought="€").reasoning_output_tokens == 4


def test_length_inside_reasoning_counts_all_generated_ids_without_inventing_close():
    assert check([(1, "Thought")], final="", stop="limit").reasoning_output_tokens == 1


def test_disabled_reasoning_has_proven_zero():
    rules = replace(RULES, initial_state="disabled", generation_prefix="assistant\n<think></think>\n")
    assert check([(3, "OK"), (102, "<|im_end|>")], thought="", rules=rules).reasoning_output_tokens == 0


def test_native_stream_flag_and_total_piece_bytes_are_bounded(monkeypatch):
    pieces = [(1, "Thought"), (101, "</think>"), (3, "OK"), (102, "<|im_end|>")]
    with pytest.raises(LocalInferenceError):
        check(pieces, stream="false")
    size = sum(len(piece.encode()) for _, piece in pieces)
    monkeypatch.setattr(prism_reasoning, "MAX_PIECE_BYTES", size)
    assert check(pieces).reasoning_output_tokens == 2
    monkeypatch.setattr(prism_reasoning, "MAX_PIECE_BYTES", size - 1)
    with pytest.raises(LocalInferenceError):
        check(pieces)


@pytest.mark.parametrize("pieces,changes", [
    ([(1,"Thought"),(101,"</think>"),(100,"<think>"),(102,"<|im_end|>")], {}),
    ([(1,"Thought"),(103,"<tool_call>"),(102,"<|im_end|>")], {}),
    ([(1,"Thought</think>"),(3,"OK"),(102,"<|im_end|>")], {}),
    ([(1,"Thought"),(101,"wrong"),(3,"OK"),(102,"<|im_end|>")], {}),
    ([(1,"Thought"),(101,"</think>"),(3,"OK"),(102,"<|im_end|>")], {"tokens_predicted":99}),
    ([(1,"Thought"),(101,"</think>"),(3,"OK"),(102,"<|im_end|>")], {"tokens":[True,101,3,102]}),
    ([(1,"Thought"),(101,"</think>"),(3,"OK"),(102,"<|im_end|>")], {"stop_type":"word"}),
    ([(1,"Thought"),(102,"<|im_end|>"),(3,"OK")], {}),
])
def test_ambiguous_or_unverified_partitions_fail_closed(pieces, changes):
    with pytest.raises(LocalInferenceError) as caught:
        check(pieces, **changes)
    assert caught.value.failure.code is ErrorCode.PROVIDER_ERROR


def test_artifact_or_template_mismatch_cannot_establish_usage():
    for identity in ({"artifact_sha256":"b"*64,"template_revision":RULES.template_revision},
                     {"artifact_sha256":RULES.artifact_sha256,"template_revision":"other"}):
        with pytest.raises(LocalInferenceError):
            exact_reasoning_usage({},TokenUsage(1,1),token_pieces=[],reasoning_text="",final_text="",rules=RULES,**identity)


def test_native_controls_never_inherit_none_for_enabled_reasoning():
    disabled = native_reasoning_options(enabled=False)
    assert disabled["reasoning_effort"] == "none" and disabled["chat_template_kwargs"]["enable_thinking"] is False
    enabled = native_reasoning_options(enabled=True, budget_tokens=32)
    assert "reasoning_effort" not in enabled and enabled["reasoning_budget_tokens"] == 32
    assert enabled["reasoning_format"] == "deepseek" and enabled["return_tokens"] and enabled["verbose"]


def test_recorded_35_original_ids_partition_to_30_reasoning_tokens():
    # Actual source-cuda40-smoke/07-reasoning + source-cuda40-extra/02-original_token_pieces.
    # Native response content/IDs/pieces retained here; this is not a new live gate.
    ids = [1596,1144,4087,1156,579,4145,6673,13,14235,1534,1132,1067,6970,9522,13,
           21902,220,16,22,9,16,24,283,220,18,17,18,13,198,248069,271,18,17,18,248046]
    pieces = ["We"," need"," answer"," user","'s"," simple"," math","."," Need"," final"," only",
        " result"," maybe"," brief","."," Compute"," ","1","7","*","1","9"," ="," ","3","2","3",
        ".","\n","</think>","\n\n","3","2","3","<|im_end|>"]
    rules = ReasoningTokenRules("3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        "fixture:pinned-qwen35-template", "<|im_start|>assistant\n<think>\n", "prefilled",
        (248068,"<think>"),(248069,"</think>"),((248046,"<|im_end|>"),),((248058,"<tool_call>"),))
    usage = check(list(zip(ids,pieces)), thought="".join(pieces[:29]), final="323", rules=rules)
    assert usage.output_tokens == 35 and usage.reasoning_output_tokens == 30


def test_disabled_id_proof_accepts_tool_syntax_and_partial_utf8_but_not_reasoning_markers():
    rules = replace(RULES, initial_state="disabled", generation_prefix="assistant\n<think></think>\n")
    def disabled(pieces, *, stop="eos"):
        verbose = {"tokens": [token for token, _ in pieces], "tokens_predicted": len(pieces),
            "generation_settings": {"stream": True, "generation_prompt": rules.generation_prefix},
            "stop": True, "stop_type": stop}
        return prism_reasoning.exact_disabled_reasoning_usage(verbose, TokenUsage(9, len(pieces), 2),
            token_pieces=[{"id": token, "piece": piece} for token, piece in pieces], rules=rules,
            artifact_sha256=rules.artifact_sha256, template_revision=rules.template_revision)
    assert disabled([(103, "<tool_call>"), (1, '{"city":"Berlin"}'), (102, "<|im_end|>")]).reasoning_output_tokens == 0
    assert disabled([(1, [0xE2])], stop="limit").reasoning_output_tokens == 0
    for pieces in ([(100, "<think>"), (102, "<|im_end|>")],
                   [(1, "<thi"), (2, "nk>"), (102, "<|im_end|>")],
                   [(1, "</think>"), (102, "<|im_end|>")],
                   [(102, "<|im_end|>"), (1, "after EOS")], [(1, "no EOS")]):
        with pytest.raises(LocalInferenceError):
            disabled(pieces)


@pytest.mark.parametrize("settings", [None, [], "invalid", 7, True, {}, {"stream": "false"}])
def test_malformed_settings_is_a_protocol_error_for_both_token_rules(settings):
    disabled = replace(RULES, initial_state="disabled", generation_prefix="assistant\n<think></think>\n")
    for rules, function, extra in (
        (RULES, prism_reasoning.exact_reasoning_usage, {"reasoning_text": "", "final_text": ""}),
        (disabled, prism_reasoning.exact_disabled_reasoning_usage, {}),
    ):
        verbose = {"tokens": [102], "tokens_predicted": 1, "stop": True,
                   "stop_type": "eos", "generation_settings": settings}
        with pytest.raises(LocalInferenceError) as raised:
            function(verbose, TokenUsage(9, 1), token_pieces=[{"id": 102, "piece": "<|im_end|>"}],
                rules=rules, artifact_sha256=rules.artifact_sha256,
                template_revision=rules.template_revision, **extra)
        assert raised.value.failure.code is ErrorCode.PROVIDER_ERROR
