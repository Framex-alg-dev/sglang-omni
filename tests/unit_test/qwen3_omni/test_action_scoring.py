from __future__ import annotations

import asyncio
import math
import wave
from pathlib import Path

import pytest

from sglang_omni.client.client import Client

from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
    aggregate_candidate_score,
    score_candidate_from_runtime,
    align_suffix_logprobs,
    build_multimodal_cache_identity,
    build_suffix_batches,
    tokenize_suffixes,
    validate_action_suffix_request,
    validate_score_result,
)


class OffsetTokenizer:
    """Tiny tokenizer exposing HF-style offsets for contract tests."""

    def __init__(self):
        self.vocab = {"当": 1, "下": 2, "最": 3, "合": 4, "适": 5, "的": 6}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.vocab.setdefault(char, len(self.vocab) + 1) for char in text]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        if not return_offsets_mapping:
            return {"input_ids": self.encode(text)}
        ids = []
        offsets = []
        for index, char in enumerate(text):
            ids.append(self.vocab.setdefault(char, len(self.vocab) + 1))
            offsets.append((index, index + 1))
        return {"input_ids": ids, "offset_mapping": offsets}


def candidate(candidate_id: str, suffix: str) -> ActionScoreCandidate:
    return ActionScoreCandidate(candidate_id, suffix, "wave", {"body_side": "left"})


def test_packaged_audio_encoder_warmup_asset_format() -> None:
    asset = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni"
        / "assets"
        / "audio_encoder_warmup.wav"
    )
    with wave.open(str(asset), "rb") as source:
        assert source.getnchannels() == 1
        assert source.getsampwidth() == 2
        assert source.getframerate() == 16_000
        assert source.getnframes() == 24 * 16_000


def request(**kwargs) -> ActionSuffixScoreRequest:
    values = dict(
        request_id="turn-123",
        model="qwen3-omni",
        prefix="当下最合适的动作是：",
        language="zh",
        candidates=[candidate("left", "使用左手挥手")],
        audios=["/tmp/audio.wav"],
        images=[],
        sample_rate=16000,
        micro_batch_size=64,
    )
    values.update(kwargs)
    return ActionSuffixScoreRequest(**values)


def test_contract_validation_and_limits():
    validate_action_suffix_request(request())
    with pytest.raises(ValueError, match="candidates must not be empty"):
        validate_action_suffix_request(request(candidates=[]))
    with pytest.raises(ValueError, match="unique"):
        validate_action_suffix_request(request(candidates=[candidate("x", "左"), candidate("x", "右")]))
    with pytest.raises(ValueError, match="language"):
        validate_action_suffix_request(request(language="ja"))
    with pytest.raises(ValueError, match="punctuation"):
        validate_action_suffix_request(request(candidates=[candidate("x", "左手。")]))
    with pytest.raises(ValueError, match="request_id"):
        validate_action_suffix_request(request(request_id="bad id"))
    with pytest.raises(ValueError, match="micro_batch_size"):
        validate_action_suffix_request(request(micro_batch_size=0))
    validate_action_suffix_request(
        request(images=["image"], image_roles=["avatar_state"])
    )
    with pytest.raises(ValueError, match="image_roles must match images length"):
        validate_action_suffix_request(
            request(images=["image"], image_roles=["user_camera", "avatar_state"])
        )
    with pytest.raises(ValueError, match="image_roles must be a list"):
        validate_action_suffix_request(request(image_roles=[""]))


def test_turn_semantics_validation_and_metadata_propagation():
    proactive = request(
        turn_origin="proactive",
        text_role="character_reply",
        trigger="user_returned",
    )
    validate_action_suffix_request(proactive)
    omni = Client._build_action_scoring_request(proactive)

    assert omni.metadata["turn_origin"] == "proactive"
    assert omni.metadata["text_role"] == "character_reply"
    assert omni.metadata["trigger"] == "user_returned"
    action_spec = omni.params["action_scoring"]
    assert action_spec["turn_origin"] == "proactive"
    assert action_spec["text_role"] == "character_reply"
    assert action_spec["trigger"] == "user_returned"

    with pytest.raises(ValueError, match="text_role"):
        validate_action_suffix_request(
            request(turn_origin="proactive", text_role="user_input")
        )
    with pytest.raises(ValueError, match="only supported for proactive"):
        validate_action_suffix_request(request(trigger="user_returned"))


def test_tokenization_handles_suffix_boundary_and_special_tokens():
    tokenizer = OffsetTokenizer()
    tokenized = tokenize_suffixes(
        tokenizer,
        "当下",
        [candidate("left", " 使用左手")],
        special_token_ids=[999],
        terminal_token_id=999,
    )[0]
    assert tokenized.suffix_start_index == 2
    assert tokenized.suffix_token_ids == (*tuple(tokenized.full_input_ids[2:-1]), 999)
    assert tokenized.full_input_ids[-1] == 999
    assert tokenized.suffix_token_mask[-1] is True
    assert sum(tokenized.suffix_token_mask) == len(" 使用左手") + 1


def test_tokenization_does_not_assume_prefix_suffix_bpe_composition():
    class MergeTokenizer(OffsetTokenizer):
        def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
            if return_offsets_mapping:
                # The first suffix token spans the textual boundary, as a BPE
                # tokenizer can do for an English leading-space token.
                return {"input_ids": [10, 11], "offset_mapping": [(0, 3), (3, len(text))]}
            return super().__call__(text, add_special_tokens=add_special_tokens)

    item = tokenize_suffixes(MergeTokenizer(), "go", [candidate("x", " left")])[0]
    assert item.suffix_start_index == 0
    assert item.suffix_token_ids == (10, 11)


def test_tokenization_composes_suffix_after_multimodal_template_prefix():
    class OpaqueTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "prefix":
                return [101, 102]
            if text == "左手":
                return [201, 202]
            return [999]

    item = tokenize_suffixes(
        OpaqueTokenizer(),
        "prefix",
        [candidate("x", "左手")],
        prefix_token_ids=[101, 102],
    )[0]
    assert item.full_input_ids == (101, 102, 201, 202)
    assert item.suffix_token_ids == (201, 202)


def test_short_id_tokenization_composes_suffix_without_retokenizing_prefix():
    class ShortIdTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "A328":
                return [328]
            if text == "A329":
                return [329]
            raise AssertionError(f"unexpected text encoded: {text!r}")

    items = tokenize_suffixes(
        ShortIdTokenizer(),
        "ignored long multimodal prompt",
        [candidate("A328", "A328"), candidate("A329", "A329")],
        prefix_token_ids=[101, 102, 103],
        terminal_token_id=999,
        suffix_only=True,
    )

    assert [item.full_input_ids for item in items] == [
        (101, 102, 103, 328, 999),
        (101, 102, 103, 329, 999),
    ]
    assert all(item.suffix_start_index == 3 for item in items)
    assert all(item.suffix_token_ids[-1] == 999 for item in items)


def test_batches_cover_386_candidates_without_reordering():
    items = [
        type(
            "Item",
            (),
            {
                "candidate_id": str(index),
                "full_input_ids": (index,),
                "suffix_start_index": 0,
                "suffix_token_mask": (True,),
                "suffix_token_ids": (index,),
            },
        )()
        for index in range(386)
    ]
    batches = build_suffix_batches(items, 64)
    assert len(batches) == 7
    assert [len(batch.candidates) for batch in batches] == [64, 64, 64, 64, 64, 64, 2]
    assert [item.candidate_id for batch in batches for item in batch.candidates] == [str(i) for i in range(386)]


def test_math_counts_only_finite_suffix_tokens():
    scores = aggregate_candidate_score("left", [TokenScore(1, -0.1), TokenScore(2, -0.3)])
    assert scores.token_count == 2
    assert scores.mean_logprob == pytest.approx(-0.2)
    assert scores.mean_nll == pytest.approx(0.2)
    assert scores.ppl == pytest.approx(math.exp(0.2))
    with pytest.raises(ValueError, match="empty"):
        aggregate_candidate_score("empty", [])


def test_aggregate_math_counts_every_explicit_token_score():
    scores = aggregate_candidate_score(
        "left",
        [TokenScore(1, -0.1), TokenScore(2, -0.3), TokenScore(151645, -0.2)],
    )
    assert scores.token_count == 3
    assert scores.mean_logprob == pytest.approx(-0.2)
    assert scores.ppl == pytest.approx(math.exp(0.2))
    assert scores.token_scores[-1].token_id == 151645


def test_runtime_score_excludes_terminal_from_identifier_ppl():
    score = score_candidate_from_runtime(
        "B027",
        [100, 0, 2, 7, 151645],
        {},
        [-9.2338, -0.0056, -0.2165, -0.0035, -6.4764],
        terminal_token_id=151645,
    )

    assert [item.token_id for item in score.token_scores] == [100, 0, 2, 7]
    assert score.token_count == 4
    assert score.mean_logprob == pytest.approx(-2.36485)
    assert score.ppl == pytest.approx(math.exp(2.36485))


def test_runtime_score_excludes_terminal_on_continuation_only_backend():
    score = score_candidate_from_runtime(
        "B027",
        [10, 11, 151645],
        {10: -0.7},
        [-0.2, -6.0],
        terminal_token_id=151645,
    )

    assert [item.token_id for item in score.token_scores] == [10, 11]
    assert [item.logprob for item in score.token_scores] == [-0.7, -0.2]
    assert score.token_count == 2


def test_terminal_bias_cannot_flip_captured_category_ranking():
    terminal = 151645
    b000 = score_candidate_from_runtime(
        "B000",
        [100, 0, 0, 0, terminal],
        {},
        [-9.2338, -0.0056, -2.3386, -1.5422, -0.0122],
        terminal_token_id=terminal,
    )
    b027 = score_candidate_from_runtime(
        "B027",
        [100, 0, 2, 7, terminal],
        {},
        [-9.2338, -0.0056, -0.2165, -0.0035, -6.4764],
        terminal_token_id=terminal,
    )

    assert b027.mean_logprob > b000.mean_logprob
    assert b027.ppl < b000.ppl
    assert all(item.token_id != terminal for item in b000.token_scores)
    assert all(item.token_id != terminal for item in b027.token_scores)


def test_alignment_rejects_nan_and_preserves_first_suffix_token():
    item = tokenize_suffixes(OffsetTokenizer(), "当下", [candidate("x", "左手")])[0]
    aligned = align_suffix_logprobs(item, [-1.0, -2.0, -0.2, -0.4])
    assert [score.token_id for score in aligned] == list(item.suffix_token_ids)
    assert [score.logprob for score in aligned] == [-0.2, -0.4]
    with pytest.raises(ValueError, match="non-finite"):
        align_suffix_logprobs(item, [-1.0, -2.0, float("nan"), -0.4])


def test_cache_identity_includes_media_and_request_scope():
    first, first_digest = build_multimodal_cache_identity(
        prefix_token_ids=[1, 2], audios=["audio-a"], images=[], request_scope="turn-a"
    )
    second, second_digest = build_multimodal_cache_identity(
        prefix_token_ids=[1, 2], audios=["audio-b"], images=[], request_scope="turn-a"
    )
    third, third_digest = build_multimodal_cache_identity(
        prefix_token_ids=[1, 2], audios=["audio-a"], images=[], request_scope="turn-b"
    )
    assert len({first, second, third}) == 3
    assert len({first_digest, second_digest, third_digest}) == 3


def test_runtime_score_uses_shared_prefix_for_first_token():
    score = score_candidate_from_runtime(
        "left", [10, 11, 12], [0.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2, -0.3]
    )
    assert [item.token_id for item in score.token_scores] == [10, 11, 12]
    assert [item.logprob for item in score.token_scores[1:]] == [-0.2, -0.3]
    assert score.token_count == 3


def test_result_contract_requires_verified_cache_and_input_order():
    req = request(candidates=[candidate("left", "左手"), candidate("right", "右手")])
    good_scores = [
        CandidateScore(item.candidate_id, 1, -0.1, 0.1, math.exp(0.1), [TokenScore(1, -0.1)])
        for item in req.candidates
    ]
    validate_score_result(
        req,
        ActionSuffixScoreResult(req.request_id, req.model, True, good_scores),
    )
    with pytest.raises(RuntimeError, match="cached prefix"):
        validate_score_result(
            req,
            ActionSuffixScoreResult(req.request_id, req.model, False, good_scores),
        )


@pytest.mark.asyncio
async def test_action_score_warmup_is_sessionless_and_runs_both_stages() -> None:
    client = Client.__new__(Client)
    requests = []

    async def fake_score(request):
        requests.append(request)
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=request.candidates[0].candidate_id,
                    token_count=1,
                    mean_logprob=-0.1,
                    mean_nll=0.1,
                    ppl=1.105170,
                    token_scores=[TokenScore(token_id=1, logprob=-0.1)],
                )
            ],
        )

    client.score_action_suffixes = fake_score
    result = await client.warmup_action_score(
        model="qwen3-omni",
        category_count=3,
        child_count=2,
        timeout_s=1.0,
        audio_path="/warmup/audio.wav",
    )

    assert result["ready"] is True
    assert len(requests) == 2
    assert [request.stage for request in requests] == ["category", "child"]
    assert all(request.suffix_tokenization_mode == "short_id" for request in requests)
    assert all(request.session_id is None for request in requests)
    assert all(request.history == [] for request in requests)
    assert requests[0].audios == ["/warmup/audio.wav"]
    assert requests[1].audios == []
    assert all(request.images == [] for request in requests)
    assert [len(request.candidates) for request in requests] == [3, 2]
    assert result["audio_warmup_enabled"] is True


@pytest.mark.asyncio
async def test_action_score_flat_warmup_runs_one_single_stage() -> None:
    client = Client.__new__(Client)
    requests = []

    async def fake_score(request):
        requests.append(request)
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=request.candidates[0].candidate_id,
                    token_count=1,
                    mean_logprob=-0.1,
                    mean_nll=0.1,
                    ppl=1.105170,
                    token_scores=[TokenScore(token_id=1, logprob=-0.1)],
                )
            ],
        )

    client.score_action_suffixes = fake_score
    result = await client.warmup_action_score(
        model="qwen3-omni",
        category_count=3,
        child_count=5,
        selection_mode="flat_children",
        timeout_s=1.0,
        audio_path="/warmup/audio.wav",
    )

    assert result["ready"] is True
    assert result["selection_mode"] == "flat_children"
    assert len(requests) == 1
    assert requests[0].stage == "single"
    assert len(requests[0].candidates) == 5
    assert requests[0].prefix.endswith("action_id 是：")
    assert "固定具体动作集合" in requests[0].system_prompt
    assert requests[0].session_id is None
    assert requests[0].history == []
    assert requests[0].audios == ["/warmup/audio.wav"]
    assert requests[0].images == []
    assert result["audio_warmup_enabled"] is True


def test_action_score_warmup_prompt_supports_english() -> None:
    request = Client._build_action_warmup_request(
        request_id="warmup-en",
        model="qwen3-omni",
        stage="category",
        candidate_prefix="C",
        candidate_count=2,
        language="en",
    )

    assert request.language == "en"
    assert request.prefix.startswith("Select an action category")
    assert "digital-character action category classifier" in request.system_prompt


@pytest.mark.asyncio
async def test_action_score_warmup_failure_is_non_fatal() -> None:
    client = Client.__new__(Client)

    async def fake_score(request):
        raise RuntimeError("warmup unavailable")

    client.score_action_suffixes = fake_score
    result = await client.warmup_action_score(
        model="qwen3-omni",
        category_count=1,
        child_count=1,
        timeout_s=1.0,
    )

    assert result["ready"] is False
    assert "warmup unavailable" in result["error"]


def test_runtime_scoring_keeps_first_suffix_logprob_from_prefill():
    score = score_candidate_from_runtime(
        "left",
        [11, 12, 13],
        {11: -0.7, 12: -1.0},
        [-0.2, -0.3, -0.4],
    )
    assert [item.logprob for item in score.token_scores] == [-0.2, -0.3, -0.4]
    assert score.token_count == 3


def test_multiturn_session_context_preserves_history_media_and_avatar_state():
    score_request = request(
        request_id="turn-2",
        session_id="sess-42",
        history=[
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "我刚才抬起左手"}]},
            {"role": "assistant", "content": "我看到你抬起了左手。"},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "请看我现在的姿势"}]},
        ],
        history_audios=["/session/turn-1.wav"],
        history_images=["/session/turn-1.png"],
        audios=["/session/turn-2.wav"],
        images=["/session/turn-2.png"],
        avatar_state={"pose": "seated", "gaze": "camera", "left_hand": "raised"},
        system_prompt="固定动作候选集合：left=左手动作；right=右手动作。",
    )
    validate_action_suffix_request(score_request)
    omni = Client._build_action_scoring_request(score_request)

    assert [message["role"] for message in omni.inputs["messages"]] == [
        "system", "user", "assistant", "user"
    ]
    assert omni.inputs["messages"][0]["content"] == score_request.system_prompt
    assert "left_hand" not in omni.inputs["messages"][0]["content"]
    assert "数字人当前状态信息：" in omni.inputs["messages"][3]["content"][-1]["text"]
    assert "left_hand" in omni.inputs["messages"][3]["content"][-1]["text"]
    assert omni.inputs["messages"][1]["content"][0]["type"] == "audio"
    assert omni.inputs["messages"][3]["content"][0]["type"] == "image"
    assert omni.inputs["messages"][3]["content"][-1]["text"].endswith(score_request.prefix)
    assert sum(
        part.get("type") == "audio"
        for message in omni.inputs["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    ) == len(omni.inputs["audios"])
    assert sum(
        part.get("type") == "image"
        for message in omni.inputs["messages"]
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
    ) == len(omni.inputs["images"])
    assert omni.inputs["audios"] == ["/session/turn-1.wav", "/session/turn-2.wav"]
    assert omni.inputs["images"] == ["/session/turn-1.png", "/session/turn-2.png"]
    assert omni.metadata["session_id"] == "sess-42"
    assert omni.metadata["avatar_state"] == score_request.avatar_state
    action_spec = omni.params["action_scoring"]
    assert action_spec["history_message_count"] == 2
    assert action_spec["history_audio_count"] == 1
    assert action_spec["history_image_count"] == 0


def test_multiturn_context_requires_history_media_placeholders():
    with pytest.raises(ValueError, match="history audio placeholders"):
        validate_action_suffix_request(
            request(
                history=[{"role": "user", "content": "old turn"}],
                history_audios=["old.wav"],
            )
        )


def test_second_turn_keeps_first_turn_in_context():
    first_turn = request(
        request_id="turn-1",
        session_id="sess-42",
        prefix="用户刚刚说：你好。现在请判断动作：",
    )
    second_turn = request(
        request_id="turn-2",
        session_id="sess-42",
        history=[
            {"role": "user", "content": "你好。"},
            {"role": "assistant", "content": "你好，我在这里。"},
        ],
        prefix="用户现在说：请挥手。现在请判断动作：",
    )
    first_request = Client._build_action_scoring_request(first_turn)
    second_request = Client._build_action_scoring_request(second_turn)

    assert first_request.inputs["messages"][-1]["content"][-1]["text"] == first_turn.prefix
    assert second_request.inputs["messages"][:2] == second_turn.history
    assert second_request.inputs["messages"][-1]["content"][-1]["text"] == second_turn.prefix
    assert second_request.metadata["session_id"] == first_request.metadata["session_id"]


class BlockingActionCoordinator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.aborted: list[str] = []
        self.submit_order: list[str] = []

    async def submit(self, request_id, omni_request):
        del omni_request
        self.submit_order.append(request_id)
        if len(self.submit_order) == 1:
            self.started.set()
            await asyncio.Event().wait()
        return ActionSuffixScoreResult(
            request_id=request_id,
            model="qwen3-omni",
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id="left",
                    token_count=1,
                    mean_logprob=-0.1,
                    mean_nll=0.1,
                    ppl=1.105,
                    token_scores=[TokenScore(token_id=1, logprob=-0.1)],
                )
            ],
        )

    async def abort(self, request_id: str) -> bool:
        self.aborted.append(request_id)
        return True


@pytest.mark.asyncio
async def test_action_scoring_cancellation_aborts_and_releases_slot() -> None:
    coordinator = BlockingActionCoordinator()
    client = Client(coordinator)
    first = asyncio.create_task(
        client.score_action_suffixes(request(request_id="cancel-first"))
    )
    await asyncio.wait_for(coordinator.started.wait(), timeout=1)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert coordinator.aborted == ["cancel-first"]
    second = await asyncio.wait_for(
        client.score_action_suffixes(request(request_id="run-second")),
        timeout=1,
    )
    assert second.request_id == "run-second"
    assert coordinator.submit_order == ["cancel-first", "run-second"]
