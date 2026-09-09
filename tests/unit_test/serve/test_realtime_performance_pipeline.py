# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.serve.realtime.performance.fusion import fuse_performance_decision
from sglang_omni.serve.realtime.performance.models import PerformanceDecision
from sglang_omni.serve.realtime.performance.pipeline import PerformancePipeline
from sglang_omni.serve.realtime.action.prompts import ActionPromptComponent
from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
)


class _Pipeline(PerformancePipeline):
    def __init__(self) -> None:
        self.categories = (
            SessionActionCategory(
                category_id="B019",
                source_label="基础表情",
                short_definition="独立脸部表情",
                category_path=("头部与视线动作", "基础表情"),
                children=(
                    SessionActionCandidate(
                        candidate_id="A154",
                        action_id="A154",
                        source_label="微笑",
                        short_definition="自然微笑",
                        execution_binding={},
                        category_id="B019",
                    ),
                    SessionActionCandidate(
                        candidate_id="A100",
                        action_id="A100",
                        source_label="非表情候选",
                        short_definition="不应进入表情分支",
                        execution_binding={},
                        category_id="B019",
                    ),
                ),
            ),
        )
        self.language = "zh"


def _expression(candidate_id: str = "A154") -> dict[str, object]:
    return {
        "category_id": "B019",
        "candidate_id": candidate_id,
        "expression_id": candidate_id,
        "label": "微笑",
        "description": "a natural smile",
        "apply": True,
    }


def _body_action(
    *, support_status: str = "supported", execute: bool = True
) -> dict[str, object]:
    return {
        "candidate_id": "A100",
        "action_id": "A100",
        "category_id": "B010",
        "execute": execute,
        "support_status": support_status,
        "fallback_applied": False,
    }


def _decision(
    scope: str,
    *,
    expression: dict[str, object] | None = None,
    expression_unsupported: bool = False,
) -> PerformanceDecision:
    return PerformanceDecision(
        request_scope=scope,  # type: ignore[arg-type]
        expression=expression,
        expression_unsupported=expression_unsupported,
        tts_instruction="自然",
        elapsed_ms=1.0,
    )


def test_expression_candidates_are_restricted_to_supported_b019_faces() -> None:
    pipeline = _Pipeline()

    assert [item.candidate_id for item in pipeline._expression_candidates()] == [
        "A154"
    ]


def test_body_child_merge_excludes_b019_candidate_even_when_duplicated_elsewhere() -> None:
    expression = SessionActionCandidate(
        candidate_id="A154",
        action_id="A154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="B019",
    )
    body = SessionActionCandidate(
        candidate_id="A100",
        action_id="A100",
        source_label="挥手",
        short_definition="挥手问候",
        execution_binding={},
        category_id="B010",
    )
    pipeline = ActionPromptComponent()
    pipeline.categories = (
        SessionActionCategory(
            category_id="B002",
            source_label="伴随动作",
            short_definition="伴随",
            category_path=("系统", "伴随"),
            children=(expression, body),
        ),
        SessionActionCategory(
            category_id="B019",
            source_label="基础表情",
            short_definition="独立表情",
            category_path=("头部", "基础表情"),
            children=(expression,),
        ),
    )

    merged = pipeline._child_candidates_for_categories(list(pipeline.categories))

    assert [candidate.candidate_id for candidate in merged] == ["A100"]


def test_body_fallback_skips_expression_duplicated_in_fallback_category() -> None:
    expression = SessionActionCandidate(
        candidate_id="A154",
        action_id="A154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="B002",
    )
    fallback = SessionActionCandidate(
        candidate_id="A132",
        action_id="A132",
        source_label="微调坐姿",
        short_definition="低扰身体动作",
        execution_binding={},
        category_id="B002",
    )
    expression_owner = SessionActionCandidate(
        candidate_id="A154",
        action_id="A154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="B019",
    )
    pipeline = ActionPromptComponent()
    pipeline.categories = (
        SessionActionCategory(
            category_id="B002",
            source_label="伴随动作",
            short_definition="伴随",
            category_path=("系统", "伴随"),
            children=(expression, fallback),
        ),
        SessionActionCategory(
            category_id="B019",
            source_label="基础表情",
            short_definition="独立表情",
            category_path=("头部", "基础表情"),
            children=(expression_owner,),
        ),
    )
    pipeline.fallback_category_ids = ("B002",)

    selected = pipeline._default_fallback_candidate()

    assert selected.candidate_id == "A132"


def test_joint_choices_cover_request_scope_with_and_without_expression() -> None:
    choices = PerformancePipeline._choices(["A154"])

    assert choices["P000"].scope == "none"
    assert choices["P199"].expression_unsupported is True
    assert choices["P201"].scope == "body_only"
    assert choices["P201"].expression_id == "A154"
    assert choices["P301"].scope == "both"


def test_tts_instruction_is_derived_from_turn_expression() -> None:
    instruction = _Pipeline()._tts_instruction("A154")

    assert "语速适中" in instruction
    assert "自然笑意" in instruction


def test_expression_only_ignores_body_failure_and_uses_not_required_barrier() -> None:
    expression = {
        "category_id": "B019",
        "candidate_id": "A154",
        "expression_id": "A154",
        "label": "微笑",
        "description": "a natural smile",
        "apply": True,
    }
    fused = fuse_performance_decision(
        action={
            "candidate_id": "fallback",
            "action_id": "fallback",
            "execute": False,
            "support_status": "unsupported",
        },
        action_error=RuntimeError("body branch failed"),
        performance=PerformanceDecision(
            request_scope="expression_only",
            expression=expression,
            expression_unsupported=False,
            tts_instruction="自然",
            elapsed_ms=1.0,
        ),
        expression_enabled=True,
    )

    assert fused.action_error is None
    assert fused.action is not None
    assert fused.action["support_status"] == "not_required"
    assert fused.action["execute"] is False
    assert fused.expression == expression


def test_disabled_expression_output_preserves_body_action_semantics() -> None:
    action = {
        "candidate_id": "fallback",
        "action_id": "fallback",
        "execute": False,
        "support_status": "unsupported",
    }
    fused = fuse_performance_decision(
        action=action,
        action_error=None,
        performance=PerformanceDecision(
            request_scope="expression_only",
            expression={
                "category_id": "B019",
                "candidate_id": "A154",
                "expression_id": "A154",
                "label": "微笑",
                "description": "a natural smile",
                "apply": True,
            },
            expression_unsupported=False,
            tts_instruction="自然",
            elapsed_ms=1.0,
        ),
        expression_enabled=False,
    )

    assert fused.action == action
    assert fused.action_error is None
    assert fused.expression is None


@pytest.mark.parametrize(
    (
        "scope",
        "expression",
        "expected_expression",
        "expected_action_status",
        "expected_execute",
    ),
    [
        pytest.param("none", None, False, "supported", True, id="no-expression-body"),
        pytest.param(
            "none", _expression(), True, "supported", True, id="optional-expression-body"
        ),
        pytest.param(
            "body_only", None, False, "supported", True, id="body-only"
        ),
        pytest.param(
            "body_only", _expression(), True, "supported", True, id="body-with-expression"
        ),
        pytest.param(
            "both", _expression(), True, "supported", True, id="explicit-combined"
        ),
    ],
)
def test_supported_expression_and_body_combinations(
    scope: str,
    expression: dict[str, object] | None,
    expected_expression: bool,
    expected_action_status: str,
    expected_execute: bool,
) -> None:
    fused = fuse_performance_decision(
        action=_body_action(),
        action_error=None,
        performance=_decision(scope, expression=expression),
        expression_enabled=True,
    )

    assert fused.action is not None
    assert fused.action["support_status"] == expected_action_status
    assert fused.action["execute"] is expected_execute
    assert (fused.expression is not None) is expected_expression


def test_no_expression_and_no_body_result_remains_empty() -> None:
    fused = fuse_performance_decision(
        action=None,
        action_error=None,
        performance=_decision("none"),
        expression_enabled=True,
    )

    assert fused.action is None
    assert fused.expression is None
    assert fused.action_error is None


def test_expression_only_scope_does_not_override_supported_body_action() -> None:
    body_action = _body_action()
    fused = fuse_performance_decision(
        action=body_action,
        action_error=None,
        performance=_decision("expression_only", expression=_expression()),
        expression_enabled=True,
    )

    assert fused.action == body_action
    assert fused.action_error is None
    assert fused.expression == _expression()


def test_optional_expression_is_suppressed_when_requested_body_is_unsupported() -> None:
    fused = fuse_performance_decision(
        action=_body_action(support_status="unsupported", execute=False),
        action_error=None,
        performance=_decision("body_only", expression=_expression()),
        expression_enabled=True,
    )

    assert fused.action is not None
    assert fused.action["support_status"] == "unsupported"
    assert fused.expression is None


@pytest.mark.parametrize(
    ("expression", "expression_unsupported"),
    [
        pytest.param(None, True, id="unsupported-expression"),
        pytest.param(None, False, id="missing-expression"),
    ],
)
def test_expression_only_without_supported_expression_is_unsupported(
    expression: dict[str, object] | None,
    expression_unsupported: bool,
) -> None:
    fused = fuse_performance_decision(
        action=_body_action(support_status="unsupported", execute=False),
        action_error=None,
        performance=_decision(
            "expression_only",
            expression=expression,
            expression_unsupported=expression_unsupported,
        ),
        expression_enabled=True,
    )

    assert fused.action is not None
    assert fused.action["support_status"] == "unsupported"
    assert fused.action["execute"] is False
    assert fused.action["reason_code"] == "expression_unsupported"
    assert fused.expression is None


@pytest.mark.parametrize(
    ("action", "action_error", "expression", "expression_unsupported"),
    [
        pytest.param(
            _body_action(support_status="unsupported", execute=False),
            None,
            _expression(),
            False,
            id="body-unsupported",
        ),
        pytest.param(None, None, _expression(), False, id="body-missing"),
        pytest.param(
            _body_action(support_status="unknown", execute=False),
            RuntimeError("body scoring failed"),
            _expression(),
            False,
            id="body-error",
        ),
        pytest.param(_body_action(), None, None, True, id="expression-unsupported"),
        pytest.param(_body_action(), None, None, False, id="expression-missing"),
    ],
)
def test_combined_request_fails_atomically_when_either_channel_is_unavailable(
    action: dict[str, object] | None,
    action_error: Exception | None,
    expression: dict[str, object] | None,
    expression_unsupported: bool,
) -> None:
    fused = fuse_performance_decision(
        action=action,
        action_error=action_error,
        performance=_decision(
            "both",
            expression=expression,
            expression_unsupported=expression_unsupported,
        ),
        expression_enabled=True,
    )

    assert fused.action is not None
    assert fused.action["support_status"] == "unsupported"
    assert fused.action["execute"] is False
    assert fused.action["reason_code"] == "combined_request_atomic_failure"
    assert fused.expression is None


def test_no_expression_change_still_generates_neutral_internal_tts_control() -> None:
    pipeline = _Pipeline()

    instruction = pipeline._tts_instruction(None)

    assert "语速适中" in instruction
    assert "当前回复内容一致" in instruction


def test_english_tts_control_uses_english_physical_voice_terms() -> None:
    pipeline = _Pipeline()
    pipeline.language = "en"

    instruction = pipeline._tts_instruction("A157")

    assert "surprised tone" in instruction
    assert "rising pitch" in instruction


def test_combined_request_is_atomic_when_body_action_is_unsupported() -> None:
    fused = fuse_performance_decision(
        action={
            "candidate_id": "fallback",
            "action_id": "fallback",
            "execute": False,
            "support_status": "unsupported",
        },
        action_error=None,
        performance=PerformanceDecision(
            request_scope="both",
            expression={
                "category_id": "B019",
                "candidate_id": "A154",
                "expression_id": "A154",
                "label": "微笑",
                "description": "a natural smile",
                "apply": True,
            },
            expression_unsupported=False,
            tts_instruction="自然",
            elapsed_ms=1.0,
        ),
        expression_enabled=True,
    )

    assert fused.action is not None
    assert fused.action["support_status"] == "unsupported"
    assert fused.action["reason_code"] == "combined_request_atomic_failure"
    assert fused.expression is None
