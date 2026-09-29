from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
)
from sglang_omni.serve.realtime.performance.service import (
    register_performance_control,
)


class _Client:
    def __init__(self, winner: str = "P101", *, fail: bool = False) -> None:
        self.winner = winner
        self.fail = fail
        self.requests = []

    async def score_action_suffixes(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("model unavailable")
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=item.candidate_id,
                    token_count=1,
                    mean_logprob=(0.0 if item.candidate_id == self.winner else -10.0),
                    mean_nll=(0.0 if item.candidate_id == self.winner else 10.0),
                    ppl=(1.0 if item.candidate_id == self.winner else 22_026.0),
                    token_scores=[],
                )
                for item in request.candidates
            ],
        )


def _payload() -> dict[str, object]:
    return {
        "request_id": "performance-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "model": "qwen3-omni",
        "language": "zh",
        "action_locale": "zh-CN",
        "text": "笑着和我打招呼",
        "expressions": [
            {
                "expression_id": "154",
                "label": "微笑",
                "description": "自然微笑",
            }
        ],
        "action_profile": {
            "persona": {},
            "visual_behavior_preferences": "自然克制",
        },
    }


def _app(client: _Client, *, token: str = "secret") -> FastAPI:
    app = FastAPI()
    app.state.client = client
    register_performance_control(app, token=token)
    return app


def test_performance_control_reuses_staging_expression_and_tts_policy() -> None:
    model = _Client()
    client = TestClient(_app(model))

    response = client.post(
        "/v1/performance-control",
        headers={"Authorization": "Bearer secret"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "performance-1",
        "request_scope": "expression_only",
        "expression_id": "154",
        "expression_unsupported": False,
        "tts_instruction": "语气轻快温暖，音调略微上扬，语速适中，带有自然笑意",
        "degraded": False,
        "elapsed_ms": response.json()["elapsed_ms"],
    }
    assert model.requests[0].stage == "performance"


def test_performance_control_requires_its_private_token() -> None:
    response = TestClient(_app(_Client())).post(
        "/v1/performance-control",
        json=_payload(),
    )

    assert response.status_code == 401


def test_performance_control_matches_session_fail_soft_default() -> None:
    response = TestClient(_app(_Client(fail=True))).post(
        "/v1/performance-control",
        headers={"Authorization": "Bearer secret"},
        json=_payload(),
    )

    assert response.status_code == 200
    assert response.json()["degraded"] is True
    assert response.json()["expression_id"] is None
    assert "自然平和" in response.json()["tts_instruction"]
