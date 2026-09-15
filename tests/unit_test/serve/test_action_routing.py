from sglang_omni.serve.realtime.action.routing import (
    choose_category_width,
    resolve_unique_explicit_action,
)
from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
)


def _candidate(candidate_id: str, label: str) -> SessionActionCandidate:
    return SessionActionCandidate(
        candidate_id=candidate_id,
        action_id=candidate_id,
        source_label=label,
        short_definition=label,
        execution_binding={},
        category_id="01",
    )


def _category(*children: SessionActionCandidate) -> SessionActionCategory:
    return SessionActionCategory(
        category_id="01",
        source_label="测试类别",
        short_definition="测试类别",
        category_path=("测试",),
        children=children,
    )


def test_explicit_route_uses_catalog_alias_and_ignores_parenthetical_qualifier() -> None:
    candidate = _candidate("057", "整理衣领（轻轻拉正）")
    category = _category(candidate)

    route = resolve_unique_explicit_action("整理衣领。", [(category, candidate)])

    assert route is not None
    assert route.candidate.candidate_id == "057"
    assert route.matched_alias == "整理衣领"


def test_explicit_route_requires_equality_and_one_unique_candidate() -> None:
    first = _candidate("001", "挥手")
    second = _candidate("002", "挥手")
    category = _category(first, second)

    assert resolve_unique_explicit_action("请挥手", [(category, first)]) is None
    assert (
        resolve_unique_explicit_action(
            "挥手",
            [(category, first), (category, second)],
        )
        is None
    )


def test_category_width_uses_top1_only_for_confident_prewarmed_winner() -> None:
    decision = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=[
            ("57", -0.1, 1.1),
            ("37", -1.2, 3.3),
        ],
        overall_top_candidate_id="57",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )

    assert decision.effective_top_k == 1
    assert decision.adaptive_top1 is True
    assert decision.reason == "high_confidence_prewarmed_top1"


def test_category_width_keeps_recall_for_close_or_unsupported_first_score() -> None:
    scores = [("57", -0.1, 1.1), ("37", -0.2, 1.2)]
    close = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=scores,
        overall_top_candidate_id="57",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )
    unsupported_first = choose_category_width(
        configured_top_k=2,
        ranked_real_scores=scores,
        overall_top_candidate_id="00",
        adaptive_enabled=True,
        top_category_prewarmed=True,
        min_margin=0.8,
        max_ppl=8.0,
    )

    assert close.effective_top_k == 2
    assert close.reason == "category_margin_too_small"
    assert unsupported_first.effective_top_k == 2
    assert unsupported_first.reason == "unsupported_ranked_first"
