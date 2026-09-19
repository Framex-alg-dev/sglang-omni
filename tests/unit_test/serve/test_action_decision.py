from sglang_omni.models.qwen3_omni.action_scoring import CandidateScore
from sglang_omni.serve.realtime.action.decision import (
    action_decision_candidates,
    action_decision_prompt,
    action_support_candidates,
    aggregate_action_decision,
    aggregate_action_support,
    aggregate_category_gate,
    category_gate_candidates,
    category_gate_prompt,
    category_gate_score_ids,
    fuse_category_gate_into_action_decision,
)


class _Category:
    def __init__(self, category_id: str, source_label: str, definition: str):
        self.category_id = category_id
        self.source_label = source_label
        self.short_definition = definition


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
    assert [candidate.suffix for candidate in candidates[:4]] == [
        "IB0=没有身体动作请求",
        "IB1=要求执行身体动作",
        "IB2=明确禁止身体动作",
        "IB3=询问身体动作能力",
    ]


def test_english_body_decision_uses_semantic_result_identifiers():
    candidates = action_decision_candidates(english=True)

    assert [candidate.suffix for candidate in candidates[:4]] == [
        "IB0=no body action request",
        "IB1=perform action now",
        "IB2=action explicitly prohibited",
        "IB3=ask action capability",
    ]


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


def test_category_gate_uses_independent_semantic_suffixes():
    categories = [
        _Category("13", "待机动作", "只用于自然待机"),
        _Category("32", "打招呼与告别", "用于挥手问候"),
    ]

    candidates = category_gate_candidates(categories)

    assert [candidate.candidate_id for candidate in candidates] == [
        "IC13",
        "IC32",
        "IC00",
        "ICN0",
    ]
    assert candidates[1].suffix == "IC32"
    assert category_gate_score_ids(categories) == {
        "IC13",
        "IC32",
        "IC00",
        "ICN0",
    }
    support = action_support_candidates()
    assert [candidate.candidate_id for candidate in support] == ["IS0", "IS1"]
    assert support[1].suffix == "IS1=不存在能实际完成请求的具体动作"
    prompt = category_gate_prompt(categories)
    assert "同一个物理 batch" in prompt
    assert "不得用待机类别" in prompt
    assert "IC32=category_id 32" in prompt


def test_category_gate_aggregation_selects_only_within_category_group():
    categories = [
        _Category("13", "待机动作", "只用于自然待机"),
        _Category("32", "打招呼与告别", "用于挥手问候"),
    ]
    scores = [
        _score("130", -0.01),
        _score("IC13", -2.0),
        _score("IC32", -0.2),
        _score("IC00", -3.0),
        _score("ICN0", -4.0),
    ]

    decision = aggregate_category_gate(scores, categories)

    assert decision.winner == "IC32"
    assert decision.category_id == "32"
    assert decision.unsupported is False
    assert decision.margin == 1.8


def test_category_gate_can_select_unsupported_category():
    categories = [_Category("13", "待机动作", "只用于自然待机")]

    decision = aggregate_category_gate(
        [
            _score("IC13", -2.0),
            _score("IC00", -0.1),
            _score("ICN0", -3.0),
        ],
        categories,
    )

    assert decision.winner == "IC00"
    assert decision.category_id is None
    assert decision.unsupported is True
    assert decision.no_action_request is False
    assert decision.action_request_margin == 2.9
    assert decision.support_margin == -1.9


def test_category_gate_can_resolve_ambiguous_body_none_without_serial_stage():
    categories = [_Category("32", "打招呼与告别", "用于挥手问候")]
    category_decision = aggregate_category_gate(
        [
            _score("IC32", -0.1),
            _score("IC00", -2.0),
            _score("ICN0", -3.0),
        ],
        categories,
    )
    values = {
        "IB0": -0.10,
        "IB1": -0.11,
        "IB2": -3.0,
        "IB3": -4.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }
    body_decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    fused = fuse_category_gate_into_action_decision(
        body_decision, category_decision, min_margin=0.1
    )

    assert fused.body_mode == "perform"
    assert fused.body_gate_confident is True
    assert fused.body_evidence_source == "category_gate"


def test_category_gate_can_confirm_low_margin_body_perform_without_serial_stage():
    categories = [_Category("32", "打招呼与告别", "用于挥手问候")]
    category_decision = aggregate_category_gate(
        [
            _score("IC32", -0.1),
            _score("IC00", -2.0),
            _score("ICN0", -3.0),
        ],
        categories,
    )
    values = {
        "IB0": -0.19,
        "IB1": -0.10,
        "IB2": -3.0,
        "IB3": -4.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }
    body_decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    fused = fuse_category_gate_into_action_decision(
        body_decision, category_decision, min_margin=0.1
    )

    assert body_decision.body_mode == "perform"
    assert body_decision.body_gate_confident is False
    assert fused.body_gate_confident is True
    assert fused.body_gate_margin == 2.9
    assert fused.body_evidence_source == "category_gate"


def test_category_gate_never_overrides_prohibition_or_capability_query():
    categories = [_Category("32", "打招呼与告别", "用于挥手问候")]
    category_decision = aggregate_category_gate(
        [
            _score("IC32", -0.1),
            _score("IC00", -2.0),
            _score("ICN0", -3.0),
        ],
        categories,
    )

    for body_winner in ("IB2", "IB3"):
        values = {
            "IB0": -3.0,
            "IB1": -2.0,
            "IB2": -0.1 if body_winner == "IB2" else -4.0,
            "IB3": -0.1 if body_winner == "IB3" else -4.0,
            "IF0": -0.1,
            "IF1": -2.0,
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

        fused = fuse_category_gate_into_action_decision(
            decision, category_decision, min_margin=0.1
        )

        assert fused is decision


def test_category_gate_does_not_override_confident_body_none():
    categories = [_Category("21", "姿态调整", "只用于躯干前倾")]
    category_decision = aggregate_category_gate(
        [
            _score("IC21", -0.1),
            _score("IC00", -1.0),
            _score("ICN0", -2.0),
        ],
        categories,
    )
    values = {
        "IB0": -0.1,
        "IB1": -0.5,
        "IB2": -3.0,
        "IB3": -4.0,
        "IF0": -0.1,
        "IF1": -2.0,
        "IR0": -0.1,
        "IR1": -2.0,
        "IR2": -3.0,
        "IR3": -4.0,
        "IR4": -5.0,
        "IR5": -6.0,
    }
    body_decision = aggregate_action_decision(
        [_score(candidate_id, value) for candidate_id, value in values.items()],
        min_margin=0.1,
    )

    fused = fuse_category_gate_into_action_decision(
        body_decision, category_decision, min_margin=0.1
    )

    assert body_decision.body_mode == "none"
    assert body_decision.groups["body"].margin == 0.4
    assert fused is body_decision


def test_action_support_is_a_separate_binary_group():
    decision = aggregate_action_support(
        [
            _score("130", -0.01),
            _score("IS0", -2.0),
            _score("IS1", -0.1),
        ]
    )

    assert decision.winner == "IS1"
    assert decision.supported is False
    assert decision.margin == 1.9


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
