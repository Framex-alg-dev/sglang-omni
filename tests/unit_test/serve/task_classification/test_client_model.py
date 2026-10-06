from __future__ import annotations

import hashlib
import json

import pytest

from sglang_omni.client.types import CompletionResult
from sglang_omni.serve.task_classification.client_model import (
    SglangClientTaskClassificationModel,
)
from sglang_omni.serve.task_classification.contracts import (
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from sglang_omni.serve.task_classification.prompt import SYSTEM_PROMPT


class _Client:
    def __init__(self, output: str | None = None) -> None:
        self.calls = []
        self.output = output or json.dumps(
            {
                "route_token": "direct",
                "output_directive": "keep",
                "task_directive": "keep",
                "media_directive": "none",
                "request_cue": None,
            },
            separators=(",", ":"),
        )

    async def completion(self, request, *, request_id, audio_format="wav"):
        self.calls.append((request, request_id, audio_format))
        return CompletionResult(request_id=request_id, text=self.output)


def _audio(payload: bytes = b"audio") -> ClassificationMediaRef:
    return ClassificationMediaRef(
        media_id="audio-1",
        kind="audio",
        start_ms=0,
        end_ms=100,
        encoding="wav",
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )


@pytest.mark.asyncio
async def test_presents_text_with_exact_router_prompt_and_sampling() -> None:
    client = _Client()
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v1",
    )

    output = await model.classify(
        TaskClassificationRequest(
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            identity_epoch=1,
            input_revision=2,
            text="帮我订票",
            media=(),
        )
    )

    assert json.loads(output)["route_token"] == "direct"
    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content[1] == {
        "type": "text",
        "text": "[CURRENT_USER_TEXT]\n帮我订票",
    }
    assert request.sampling.temperature == 0.0
    assert request.sampling.top_p == 1.0
    assert request.sampling.max_new_tokens == 128
    assert request.sampling.stop == ["\n"]
    assert request.metadata["task"] == "turn_router"
    assert request.metadata["logical_request_id"] == "turn-1"
    assert "取消所有任务" in SYSTEM_PROMPT
    assert request.messages[1].content[-1]["text"].endswith(
        "符合 OUTPUT_SCHEMA 的单行 JSON："
    )


@pytest.mark.asyncio
async def test_presents_original_audio_with_music_route_contract_without_asr_text(
) -> None:
    client = _Client(
        json.dumps(
            {
                "route_token": "delegate",
                "output_directive": "keep",
                "task_directive": "keep",
                "media_directive": "none",
                "request_cue": {
                    "verb": "play",
                    "object": "a song",
                    "language": "en-US",
                },
            },
            separators=(",", ":"),
        )
    )
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v2",
    )

    output = await model.classify(
        TaskClassificationRequest(
            request_id="request-audio",
            session_id="session-1",
            turn_id="turn-1",
            identity_epoch=1,
            input_revision=2,
            text=None,
            media=(_audio(),),
        )
    )

    assert json.loads(output)["route_token"] == "delegate"
    request = client.calls[0][0]
    assert request.messages[0].content == SYSTEM_PROMPT
    assert request.messages[1].content[1] == {"type": "audio"}
    assert "歌曲能力" in request.messages[1].content[0]["text"]
    assert "不处理应用能力" in request.messages[1].content[0]["text"]
    assert request.metadata["audios"][0].startswith("data:audio/wav;base64,")
    assert "CURRENT_USER_TEXT" not in str(request.messages[1].content)


def test_music_and_weather_request_cues_are_part_of_prompt_contract() -> None:
    assert '“明天北京天气怎么样？” -> {"route_token":"delegate"' in SYSTEM_PROMPT
    assert '"verb":"查","object":"明天北京的天气"' in SYSTEM_PROMPT
    assert '“播放《青花瓷》。” -> {"route_token":"delegate"' in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "utterance",
    ("闭嘴，别再说了。", "别说了，但继续查天气。"),
)
def test_silence_controls_are_part_of_prompt_contract(utterance: str) -> None:
    assert utterance in SYSTEM_PROMPT
    assert '"route_token":"control"' in SYSTEM_PROMPT


def test_media_control_uses_the_deterministic_control_route() -> None:
    assert "媒体停止、暂停、继续/恢复时使用 control" in SYSTEM_PROMPT
    assert '“别唱了，停止播放。” -> {"route_token":"control"' in SYSTEM_PROMPT


def test_combined_silence_and_media_control_keeps_task() -> None:
    assert (
        "“不要回复，只把歌停掉。”" in SYSTEM_PROMPT
    )


def test_quoted_silence_is_not_a_control() -> None:
    assert (
        "“他刚才说‘闭嘴’是什么意思？”" in SYSTEM_PROMPT
    )


def test_rejects_visual_media() -> None:
    payload = b"image"
    with pytest.raises(ValueError, match="original user audio"):
        ClassificationMediaRef(
            media_id="image-1",
            kind="image",
            start_ms=0,
            end_ms=100,
            encoding="jpeg",
            checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )
