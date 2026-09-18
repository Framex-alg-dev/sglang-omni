from sglang_omni.models.qwen3_omni.action_scoring import CandidateScore
from sglang_omni.serve.realtime.action.decision import (
    action_decision_candidates,
    action_decision_prompt,
    aggregate_action_decision,
)


def _score(candidate_id: str, value: float) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate_id,
        token_count=1,
        mean_logprob=value,
        mean_nll=-value,
        ppl=1.0,
        token_scores=[],
    )


def test_core_action_decision_uses_twelve_real_candidates_without_padding():
    candidates = action_decision_candidates()

    assert len(candidates) == 12
    assert len({candidate.candidate_id for candidate in candidates}) == 12


def test_visual_action_decision_uses_fifteen_real_candidates_without_padding():
    candidates = action_decision_candidates(include_visual=True)

    assert len(candidates) == 15
    assert len({candidate.candidate_id for candidate in candidates}) == 15
    assert [candidate.candidate_id for candidate in candidates[-3:]] == [
        "IV00",
        "IV01",
        "IV11",
    ]


def test_visual_safety_prompt_defers_exact_scope_and_preserves_mixed_speech():
    prompt = action_decision_prompt(include_visual=True)

    assert "模仿画面动作时，即使同时要求说话，仍选 IV01" in prompt
    assert "具体身体范围由统一意图解析" in prompt
    assert "IV02" not in prompt


def test_grouped_decision_compares_only_within_each_group():
    values = {
        "IB0": -4.0,
        "IB1": -0.1,
        "IB2": -3.0,
        "IB3": -2.0,
        "IF0": -1.0,
        "IF1": -0.2,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    assert decision.body_mode == "perform"
    assert decision.face_mode == "perform"
    assert decision.reaction_type == "none"
    assert decision.confident is True
    assert decision.allows_body is True


def test_prohibit_and_visual_answer_fail_closed():
    values = {
        "IB0": -3.0,
        "IB1": -2.0,
        "IB2": -0.1,
        "IB3": -4.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
        "IV00": -3.0,
        "IV01": -3.0,
        "IV11": -3.0,
    }
    values["IV11"] = -0.1

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        include_visual=True,
    )

    assert decision.body_mode == "prohibit"
    assert decision.visual_scope == "VISUAL_ANSWER"
    assert decision.allows_body is False


def test_visual_answer_blocks_an_otherwise_allowed_body_action():
    values = {
        "IB0": -3.0,
        "IB1": -0.1,
        "IB2": -4.0,
        "IB3": -5.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
        "IV00": -3.0,
        "IV01": -2.0,
        "IV11": -0.1,
    }

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        include_visual=True,
    )

    assert decision.body_mode == "perform"
    assert decision.visual_scope == "VISUAL_ANSWER"
    assert decision.body_gate_confident is True
    assert decision.allows_body is False


def test_ambiguous_face_does_not_block_clear_body_action():
    values = {
        "IB0": -3.0,
        "IB1": -0.1,
        "IB2": -4.0,
        "IB3": -5.0,
        "IF0": -0.10,
        "IF1": -0.11,
        "IR0": -0.1,
        "IR1": -0.11,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    assert decision.body_mode == "perform"
    assert decision.body_confident is True
    assert decision.face_confident is False
    assert decision.reaction_confident is False
    assert decision.confident is False
    assert decision.body_gate_confident is True
    assert decision.allows_body is True


def test_social_reaction_requires_body_and_reaction_confidence():
    values = {
        "IB0": -0.10,
        "IB1": -0.11,
        "IB2": -3.0,
        "IB3": -4.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -2.0,
        "IR1": -0.1,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    assert decision.body_mode == "none"
    assert decision.reaction_type == "greeting"
    assert decision.body_confident is False
    assert decision.reaction_confident is True
    assert decision.body_gate_confident is False
    assert decision.allows_body is False


def test_ambiguous_reaction_does_not_block_explicit_body_action():
    values = {
        "IB0": -3.0,
        "IB1": -0.1,
        "IB2": -4.0,
        "IB3": -5.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.10,
        "IR1": -0.11,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    assert decision.body_mode == "perform"
    assert decision.reaction_confident is False
    assert decision.body_gate_confident is True
    assert decision.allows_body is True
