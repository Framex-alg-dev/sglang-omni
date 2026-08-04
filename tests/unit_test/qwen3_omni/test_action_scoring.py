from __future__ import annotations

import math

import pytest

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
    )[0]
    assert tokenized.suffix_start_index == 2
    assert tokenized.suffix_token_ids == tuple(tokenized.full_input_ids[2:])
    assert sum(tokenized.suffix_token_mask) == len(" 使用左手")


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


def test_runtime_scoring_keeps_first_suffix_logprob_from_prefill():
    score = score_candidate_from_runtime(
        "left",
        [11, 12, 13],
        {11: -0.7, 12: -1.0},
        [-0.2, -0.3, -0.4],
    )
    assert [item.logprob for item in score.token_scores] == [-0.2, -0.3, -0.4]
    assert score.token_count == 3

