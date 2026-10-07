from __future__ import annotations

import hashlib
import json

import pytest

from sglang_omni.client.types import CompletionResult
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionSuffixScoreResult,
    CandidateScore,
    validate_action_suffix_request,
)
from sglang_omni.serve.task_classification.client_model import (
    SglangClientTaskClassificationModel,
)
from sglang_omni.serve.task_classification.contracts import (
    ClassificationMediaRef,
    TaskClassificationRequest,
)
from sglang_omni.serve.task_classification.prompt import SYSTEM_PROMPT
from sglang_omni.serve.task_classification.route_candidates import ROUTE_CANDIDATES


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


class _ScoringClient(_Client):
    def __init__(self, candidate_id: str, cue: str | None = None) -> None:
        super().__init__(cue)
        self.candidate_id = candidate_id
        self.score_calls = []

    async def score_action_suffixes(self, request):
        self.score_calls.append(request)
        scores = [
            CandidateScore(
                candidate_id=item.candidate_id,
                token_count=1,
                mean_logprob=(0.0 if item.candidate_id == self.candidate_id else -10.0),
                mean_nll=(0.0 if item.candidate_id == self.candidate_id else 10.0),
                ppl=(1.0 if item.candidate_id == self.candidate_id else 22_026.0),
                token_scores=[],
            )
            for item in ROUTE_CANDIDATES
        ]
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=False,
            scores=scores,
        )


class _FailingScoringClient(_Client):
    def __init__(self) -> None:
        super().__init__(
            json.dumps(
                {
                    "route_token": "direct",
                    "output_directive": "keep",
                    "task_directive": "keep",
                    "media_directive": "none",
                    "response_locale": "zh-CN",
                    "request_cue": None,
                },
                separators=(",", ":"),
            )
        )

    async def score_action_suffixes(self, _request):
        raise RuntimeError("scoring unavailable")


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


def test_route_candidate_catalog_is_the_exact_twenty_result_closed_set() -> None:
    coarse = {
        ("direct", "keep", "keep", "none"),
        ("delegate", "keep", "keep", "none"),
        ("control", "suppress_reply", "keep", "none"),
        ("control", "stop_current", "keep", "none"),
        ("control", "suppress_reply", "cancel_current", "none"),
        ("control", "suppress_reply", "cancel_all", "none"),
        ("control", "keep", "keep", "stop"),
        ("control", "keep", "keep", "pause"),
        ("control", "keep", "keep", "resume"),
        ("control", "suppress_reply", "keep", "stop"),
    }

    assert [item.candidate_id for item in ROUTE_CANDIDATES] == [
        f"R{index:03d}" for index in range(1, 21)
    ]
    assert {
        (
            item.route_token,
            item.output_directive,
            item.task_directive,
            item.media_directive,
        )
        for item in ROUTE_CANDIDATES
    } == coarse
    assert {item.response_locale for item in ROUTE_CANDIDATES} == {
        "zh-CN",
        "en-US",
    }


@pytest.mark.asyncio
async def test_scores_closed_direct_candidates_without_json_decode() -> None:
    selected = next(
        item
        for item in ROUTE_CANDIDATES
        if item.route_token == "direct" and item.response_locale == "zh-CN"
    )
    client = _ScoringClient(selected.candidate_id)
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v1",
    )

    output = json.loads(
        await model.classify(
            TaskClassificationRequest(
                request_id="request-score",
                session_id="session-1",
                turn_id="turn-1",
                identity_epoch=1,
                input_revision=2,
                text="你好",
                media=(),
            )
        )
    )

    assert output == {
        "route_token": "direct",
        "output_directive": "keep",
        "task_directive": "keep",
        "media_directive": "none",
        "response_locale": "zh-CN",
        "request_cue": None,
    }
    assert not client.calls
    score_request = client.score_calls[0]
    assert score_request.session_instance_id == "session-1"
    assert score_request.cache_static_system_only is True
    assert score_request.output_prompt == (
        "[OUTPUT]\n只输出一个候选标签（例如 R001）："
    )
    assert len(score_request.candidates) == 20
    assert all(item.suffix == item.candidate_id for item in score_request.candidates)
    validate_action_suffix_request(score_request)


@pytest.mark.asyncio
async def test_prewarm_scores_candidates_without_generating_a_cue() -> None:
    selected = next(item for item in ROUTE_CANDIDATES if item.route_token == "delegate")
    client = _ScoringClient(selected.candidate_id)
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v1",
    )

    await model.prewarm()

    assert len(client.score_calls) == 1
    assert not client.calls
    assert client.score_calls[0].cache_static_system_only is True


@pytest.mark.asyncio
async def test_delegate_scores_route_then_generates_only_request_cue() -> None:
    selected = next(
        item
        for item in ROUTE_CANDIDATES
        if item.route_token == "delegate" and item.response_locale == "en-US"
    )
    cue = json.dumps(
        {"verb": "check", "object": "tomorrow's weather", "language": "en-US"}
    )
    client = _ScoringClient(selected.candidate_id, cue)
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v1",
    )

    output = json.loads(
        await model.classify(
            TaskClassificationRequest(
                request_id="request-delegate",
                session_id="session-1",
                turn_id="turn-1",
                identity_epoch=1,
                input_revision=2,
                text="What's the weather tomorrow?",
                media=(),
            )
        )
    )

    assert output["route_token"] == "delegate"
    assert output["request_cue"] == {
        "verb": "check",
        "object": "tomorrow's weather",
        "language": "en-US",
    }
    assert len(client.calls) == 1
    cue_request = client.calls[0][0]
    assert cue_request.sampling.max_new_tokens == 64
    assert cue_request.metadata["session_instance_id"] == "session-1"


@pytest.mark.asyncio
async def test_scoring_failure_falls_back_to_single_json_generation() -> None:
    client = _FailingScoringClient()
    model = SglangClientTaskClassificationModel(
        client,
        model_id="shared-base",
        model_version="prompt-v1",
    )

    output = json.loads(
        await model.classify(
            TaskClassificationRequest(
                request_id="request-fallback",
                session_id="session-1",
                turn_id="turn-1",
                identity_epoch=1,
                input_revision=2,
                text="你好",
                media=(),
            )
        )
    )

    assert output["route_token"] == "direct"
    assert len(client.calls) == 1


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
    assert request.metadata["session_instance_id"] == "session-1"
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
