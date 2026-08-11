from __future__ import annotations

import math

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
    score_candidate_from_runtime,
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


def test_math_includes_action_terminal_token_in_ppl():
    scores = aggregate_candidate_score(
        "left",
        [TokenScore(1, -0.1), TokenScore(2, -0.3), TokenScore(151645, -0.2)],
    )
    assert scores.token_count == 3
    assert scores.mean_logprob == pytest.approx(-0.2)
    assert scores.ppl == pytest.approx(math.exp(0.2))
    assert scores.token_scores[-1].token_id == 151645


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
    )

    assert result["ready"] is True
    assert len(requests) == 2
    assert [request.stage for request in requests] == ["category", "child"]
    assert all(request.session_id is None for request in requests)
    assert all(request.history == [] for request in requests)
    assert all(request.audios == [] and request.images == [] for request in requests)
    assert [len(request.candidates) for request in requests] == [3, 2]


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
    )
    validate_action_suffix_request(score_request)
    omni = Client._build_action_scoring_request(score_request)

    assert [message["role"] for message in omni.inputs["messages"]] == [
        "system", "user", "assistant", "user"
    ]
    assert omni.inputs["messages"][0]["content"].startswith("当前数字人状态：")
    assert "left_hand" in omni.inputs["messages"][0]["content"]
    assert omni.inputs["messages"][1]["content"][0]["type"] == "audio"
    assert omni.inputs["messages"][3]["content"][0]["type"] == "image"
    assert omni.inputs["messages"][3]["content"][-1]["text"] == score_request.prefix
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
