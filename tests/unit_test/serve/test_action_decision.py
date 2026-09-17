from sglang_omni.models.qwen3_omni.action_scoring import CandidateScore
from sglang_omni.serve.realtime.action.decision import (
    action_decision_candidates,
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


def test_visual_action_decision_uses_twenty_four_real_candidates_without_padding():
    candidates = action_decision_candidates(include_visual=True)

    assert len(candidates) == 24
    assert len({candidate.candidate_id for candidate in candidates}) == 24


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
        **{f"IV{index:02d}": -3.0 for index in range(12)},
    }
    values["IV11"] = -0.1

    decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        include_visual=True,
    )

    assert decision.body_mode == "prohibit"
    assert decision.visual_scope == "V11"
    assert decision.allows_body is False
