"""Focused checks for the deployed V10 role-specific audio budgets."""

import base64
import hashlib
import io
import wave

import pytest

from sglang_omni.serve.action_decision.contracts import (
    ActionDecisionRequest,
    DecisionChannel,
)
from sglang_omni.serve.action_decision.simple_audio import _clip
from sglang_omni.serve.action_decision.speech_audio import (
    AVATAR_CONTEXT_ROLE,
    AVATAR_FOCUS_ROLE,
    MAX_SPEECH_AUDIO_MS,
    MAX_USER_INPUT_AUDIO_MS,
    USER_CONTEXT_AUDIO_ROLE,
    validate_audio_request,
)
from sglang_omni.serve.action_decision.speech_window import MAX_AVATAR_AUDIO_MS
from sglang_omni.serve.streaming_request import StreamedMedia


def _wav(duration_ms: int) -> bytes:
    pcm = b"\x00\x00" * (duration_ms * 16)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    return output.getvalue()


def _media(role: str, start_ms: int, end_ms: int) -> StreamedMedia:
    payload = _wav(end_ms - start_ms)
    return StreamedMedia(
        role,
        "audio",
        start_ms,
        end_ms,
        "wav",
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        payload,
        role,
    )


def _clip_payload(duration_ms: int, role: str) -> StreamedMedia:
    return _clip(
        {
            "data": base64.b64encode(_wav(duration_ms)).decode(),
            "format": "wav",
        },
        role=role,
        start=0,
    )


def test_user_input_accepts_exactly_sixty_seconds() -> None:
    item = _clip_payload(60_000, "user_audio")
    assert item.end_ms - item.start_ms == MAX_USER_INPUT_AUDIO_MS


def test_user_input_rejects_more_than_sixty_seconds() -> None:
    with pytest.raises(ValueError, match="user_input audio duration.*at most 60 seconds"):
        _clip_payload(60_001, "user_audio")


@pytest.mark.parametrize(
    ("role", "field"),
    ((AVATAR_CONTEXT_ROLE, "avatar_history"), (AVATAR_FOCUS_ROLE, "avatar_current")),
)
def test_avatar_fields_have_independent_four_second_limits(
    role: str, field: str
) -> None:
    accepted = _clip_payload(4_000, role)
    assert accepted.end_ms - accepted.start_ms == MAX_AVATAR_AUDIO_MS
    with pytest.raises(ValueError, match=rf"{field} audio duration.*at most 4 seconds"):
        _clip_payload(4_001, role)


def test_full_sixty_plus_four_plus_four_request_is_accepted() -> None:
    request = ActionDecisionRequest(
        request_id="limit-68s",
        session_id="limit-68s",
        turn_id="limit-68s",
        decision_point_id="limit-68s",
        channel=DecisionChannel.BODY,
        character_prompt="沉稳、耐心的老师",
        runtime_context={
            "prompt_profile": "persona-audio-window-v2",
            "persona_revision": "v10-frozen",
            "decision_mode": "speech_window",
            "response_id": "limit-68s",
            "output_epoch": 0,
            "speech_window": {
                "window_id": "focus",
                "start_ms": 4_000,
                "end_ms": 8_000,
            },
        },
        media=(
            _media(USER_CONTEXT_AUDIO_ROLE, 0, 60_000),
            _media(AVATAR_CONTEXT_ROLE, 0, 4_000),
            _media(AVATAR_FOCUS_ROLE, 4_000, 8_000),
        ),
        simple_audio=True,
    )
    validate_audio_request(request)
    assert sum(item.end_ms - item.start_ms for item in request.media) == MAX_SPEECH_AUDIO_MS
