# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
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
                category_id="19",
                source_label="基础表情",
                short_definition="独立脸部表情",
                category_path=("头部与视线动作", "基础表情"),
                children=(
                    SessionActionCandidate(
                        candidate_id="154",
                        action_id="154",
                        source_label="微笑",
                        short_definition="自然微笑",
                        execution_binding={},
                        category_id="19",
                    ),
                    SessionActionCandidate(
                        candidate_id="100",
                        action_id="100",
                        source_label="非表情候选",
                        short_definition="不应进入表情分支",
                        execution_binding={},
                        category_id="19",
                    ),
                ),
            ),
        )
        self.language = "zh"
        self.action_language = "zh"
        self.action_locale = "zh-CN"


def _expression(candidate_id: str = "154") -> dict[str, object]:
    return {
        "category_id": "19",
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
        "candidate_id": "100",
        "action_id": "100",
        "category_id": "10",
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


def test_expression_candidates_are_restricted_to_supported_category_19_faces() -> None:
    pipeline = _Pipeline()

    assert [item.candidate_id for item in pipeline._expression_candidates()] == [
        "154"
    ]


def test_body_child_merge_excludes_category_19_candidate_even_when_duplicated_elsewhere() -> None:
    expression = SessionActionCandidate(
        candidate_id="154",
        action_id="154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="19",
    )
    body = SessionActionCandidate(
        candidate_id="100",
        action_id="100",
        source_label="挥手",
        short_definition="挥手问候",
        execution_binding={},
        category_id="10",
    )
    pipeline = ActionPromptComponent()
    pipeline.categories = (
        SessionActionCategory(
            category_id="02",
            source_label="伴随动作",
            short_definition="伴随",
            category_path=("系统", "伴随"),
            children=(expression, body),
        ),
        SessionActionCategory(
            category_id="19",
            source_label="基础表情",
            short_definition="独立表情",
            category_path=("头部", "基础表情"),
            children=(expression,),
        ),
    )

    merged = pipeline._child_candidates_for_categories(list(pipeline.categories))

    assert [candidate.candidate_id for candidate in merged] == ["100"]


def test_body_fallback_skips_expression_duplicated_in_fallback_category() -> None:
    expression = SessionActionCandidate(
        candidate_id="154",
        action_id="154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="02",
    )
    fallback = SessionActionCandidate(
        candidate_id="132",
        action_id="132",
        source_label="微调坐姿",
        short_definition="低扰身体动作",
        execution_binding={},
        category_id="02",
    )
    expression_owner = SessionActionCandidate(
        candidate_id="154",
        action_id="154",
        source_label="微笑",
        short_definition="自然微笑",
        execution_binding={},
        category_id="19",
    )
    pipeline = ActionPromptComponent()
    pipeline.categories = (
        SessionActionCategory(
            category_id="02",
            source_label="伴随动作",
            short_definition="伴随",
            category_path=("系统", "伴随"),
            children=(expression, fallback),
        ),
        SessionActionCategory(
            category_id="19",
            source_label="基础表情",
            short_definition="独立表情",
            category_path=("头部", "基础表情"),
            children=(expression_owner,),
        ),
    )
    pipeline.fallback_category_ids = ("02",)

    selected = pipeline._default_fallback_candidate()

    assert selected.candidate_id == "132"


def test_joint_choices_cover_request_scope_with_and_without_expression() -> None:
    choices = PerformancePipeline._choices(["154"])

    assert choices["P000"].scope == "none"
    assert choices["P199"].expression_unsupported is True
    assert choices["P201"].scope == "body_only"
    assert choices["P201"].expression_id == "154"
    assert choices["P301"].scope == "both"


def test_tts_instruction_is_derived_from_turn_expression() -> None:
    instruction = _Pipeline()._tts_instruction("154")

    assert "语速适中" in instruction
    assert "自然笑意" in instruction


def test_explicit_face_is_resolved_without_model_scoring() -> None:
    pipeline = _Pipeline()

    decision = pipeline._explicit_face_performance_decision("微笑")

    assert decision.request_scope == "expression_only"
    assert decision.expression_unsupported is False
    assert decision.expression is not None
    assert decision.expression["candidate_id"] == "154"
    assert decision.elapsed_ms == 0.0


def test_unknown_explicit_face_fails_closed_without_model_scoring() -> None:
    decision = _Pipeline()._explicit_face_performance_decision("不存在的表情")

    assert decision.request_scope == "expression_only"
    assert decision.expression is None
    assert decision.expression_unsupported is True
    assert decision.elapsed_ms == 0.0


@pytest.mark.asyncio
async def test_visual_deictic_expression_scores_the_current_user_camera() -> None:
    from sglang_omni.serve.realtime.turn_intent import TurnIntent

    pipeline = _Pipeline()
    pipeline.action_micro_batch_size = 64
    pipeline.model_name = "model"
    pipeline.session_id = "session"
    pipeline.session_instance_id = "instance"
    pipeline.action_profile = None
    pipeline.modalities = ["expression"]
    pipeline._action_prompt = lambda *, zh, en: zh
    requests = []

    async def score_action_request(turn, request):
        del turn
        requests.append(request)
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    token_count=1,
                    mean_logprob=(
                        -0.01 if candidate.candidate_id == "P101" else -10.0
                    ),
                    mean_nll=(
                        0.01 if candidate.candidate_id == "P101" else 10.0
                    ),
                    ppl=(1.01 if candidate.candidate_id == "P101" else 22026.0),
                    token_scores=[TokenScore(token_id=1, logprob=-0.01)],
                )
                for candidate in request.candidates
            ],
        )

    pipeline._score_action_request = score_action_request
    turn = SimpleNamespace(
        intent=TurnIntent(
            speech="none",
            text="",
            body="",
            body_mode="none",
            face="做出表情",
            history=False,
        ),
        request_base="visual-expression",
        turn_origin="user",
        text_role="user_input",
        trigger=None,
        turn_id="turn",
        trace_id="trace",
    )

    decision = await pipeline._infer_turn_performance(
        turn,
        [],
        current_text="请做出表情",
        images=["user-camera", "avatar-state"],
        image_roles=["user_camera", "avatar_state"],
    )

    request = requests[0]
    assert request.images == ["user-camera"]
    assert request.image_roles == ["user_camera"]
    assert "匹配可见脸部形态" in request.system_prompt
    assert decision.request_scope == "expression_only"
    assert decision.expression is not None
    assert decision.expression["candidate_id"] == "154"


def test_expression_only_ignores_body_failure_and_uses_not_required_barrier() -> None:
    expression = {
        "category_id": "19",
        "candidate_id": "154",
        "expression_id": "154",
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
                "category_id": "19",
                "candidate_id": "154",
                "expression_id": "154",
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


@pytest.mark.parametrize(
    ("body_id", "expression_id"),
    [
        pytest.param("460", "159", id="fear"),
        pytest.param("444", "160", id="aggrieved"),
        pytest.param("439", "161", id="sad"),
        pytest.param("153", "162", id="questioning"),
        pytest.param("442", "164", id="vulnerable"),
    ],
)
def test_expression_only_scope_suppresses_supported_body_action(
    body_id: str, expression_id: str,
) -> None:
    body_action = _body_action()
    body_action.update(candidate_id=body_id, action_id=body_id)
    expression = _expression(expression_id)
    fused = fuse_performance_decision(
        action=body_action,
        action_error=None,
        performance=_decision("expression_only", expression=expression),
        expression_enabled=True,
    )

    assert fused.action == {
        "candidate_id": "expression_only",
        "action_id": "no_action",
        "execute": False,
        "support_status": "not_required",
        "fallback_applied": False,
        "reason_code": "expression_only",
    }
    assert fused.action_error is None
    assert fused.expression == expression
    assert body_action["execute"] is True
    assert body_action["action_id"] == body_id


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
@pytest.mark.parametrize("body_supported", [False, True])
def test_expression_only_without_supported_expression_is_unsupported(
    expression: dict[str, object] | None,
    expression_unsupported: bool,
    body_supported: bool,
) -> None:
    fused = fuse_performance_decision(
        action=_body_action(
            support_status="supported" if body_supported else "unsupported",
            execute=body_supported,
        ),
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

    instruction = pipeline._tts_instruction("157")

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
                "category_id": "19",
                "candidate_id": "154",
                "expression_id": "154",
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
