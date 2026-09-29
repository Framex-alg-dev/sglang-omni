from __future__ import annotations

from sglang_omni.serve.realtime.visual_observation import (
    VISUAL_OBSERVATION_MIN_MEAN_LOGPROB_ENV,
    VISUAL_OBSERVATION_MIN_TOKEN_MARGIN_ENV,
    summarize_visual_observation_confidence,
)


def test_visual_observation_confidence_uses_same_forward_margin() -> None:
    confidence = summarize_visual_observation_confidence(
        [[-0.1, 11], [-0.2, 22]],
        [
            [[-0.1, 11], [-0.8, 12]],
            [[-0.2, 22], [-0.6, 23]],
        ],
    )

    assert confidence.available
    assert confidence.accepted
    assert confidence.mean_logprob == -0.15
    assert confidence.min_token_margin == 0.4


def test_visual_observation_confidence_can_fail_closed_without_second_call(
    monkeypatch,
) -> None:
    monkeypatch.setenv(VISUAL_OBSERVATION_MIN_MEAN_LOGPROB_ENV, "-0.5")
    monkeypatch.setenv(VISUAL_OBSERVATION_MIN_TOKEN_MARGIN_ENV, "0.2")

    confidence = summarize_visual_observation_confidence(
        [[-0.1, 11]],
        [[[-0.1, 11], [-0.2, 12]]],
    )

    assert not confidence.accepted
    assert confidence.rejection_reason == "token_margin_below_threshold"


def test_visual_observation_missing_confidence_stays_compatible_until_enforced(
    monkeypatch,
) -> None:
    monkeypatch.delenv(VISUAL_OBSERVATION_MIN_MEAN_LOGPROB_ENV, raising=False)
    monkeypatch.delenv(VISUAL_OBSERVATION_MIN_TOKEN_MARGIN_ENV, raising=False)

    confidence = summarize_visual_observation_confidence(None, None)

    assert not confidence.available
    assert confidence.accepted
