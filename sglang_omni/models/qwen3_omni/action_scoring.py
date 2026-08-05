# SPDX-License-Identifier: Apache-2.0
"""Candidate suffix scoring primitives for Qwen3-Omni.

This module is deliberately the single owner of the scoring contract.  HTTP
and pipeline code should pass validated requests and consume the result; they
must not reimplement token spans, batching, or PPL arithmetic.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


Language = Literal["zh", "en"]

MAX_ACTION_CANDIDATES = 512
MAX_ACTION_SUFFIX_CHARS = 512
MAX_ACTION_HISTORY_MESSAGES = 256
MIN_MICRO_BATCH_SIZE = 1
MAX_MICRO_BATCH_SIZE = 256
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TERMINAL_PUNCTUATION = set(string.punctuation) | set(
    "，。！？；：、）》】」』〉》…—～．！？，；："
)


@dataclass(slots=True)
class ActionScoreCandidate:
    candidate_id: str
    suffix: str
    action_id: str | None = None
    execution_binding: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ActionSuffixScoreRequest:
    request_id: str
    model: str
    prefix: str
    language: Language
    candidates: list[ActionScoreCandidate]
    audios: list[str]
    images: list[str]
    sample_rate: int
    micro_batch_size: int = 64
    session_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    history_audios: list[str] = field(default_factory=list)
    history_images: list[str] = field(default_factory=list)
    avatar_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenScore:
    token_id: int
    logprob: float


@dataclass(slots=True)
class CandidateScore:
    candidate_id: str
    token_count: int
    mean_logprob: float
    mean_nll: float
    ppl: float
    token_scores: list[TokenScore]


@dataclass(slots=True)
class ActionSuffixScoreResult:
    request_id: str
    model: str
    prefix_cached: bool
    scores: list[CandidateScore]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateTokenization:
    """The exact token span used for one candidate.

    ``suffix_start_index`` is an index into ``full_input_ids``.  The mask is
    explicit because a tokenizer may emit a token spanning the textual
    prefix/suffix boundary, and because special tokens must never silently
    enter the PPL denominator.
    """

    candidate_id: str
    full_input_ids: tuple[int, ...]
    suffix_start_index: int
    suffix_token_mask: tuple[bool, ...]
    suffix_token_ids: tuple[int, ...]

    @property
    def suffix_token_count(self) -> int:
        return len(self.suffix_token_ids)


@dataclass(frozen=True, slots=True)
class SuffixBatch:
    index: int
    candidates: tuple[CandidateTokenization, ...]


@dataclass(slots=True)
class ActionScoringStats:
    """Execution counters shared by scheduler, tracing, and benchmarks."""

    queue_wait_ms: float = 0.0
    preprocessing_ms: float = 0.0
    image_encoder_ms: float = 0.0
    audio_encoder_ms: float = 0.0
    logical_prefix_request_count: int = 0
    physical_prefix_chunk_count: int = 0
    prefix_token_count: int = 0
    cached_prefix_token_count: int = 0
    candidate_prefix_recompute_tokens: int = 0
    suffix_batch_count: int = 0
    suffix_batch_sizes: list[int] = field(default_factory=list)
    suffix_batch_ms: list[float] = field(default_factory=list)
    aggregation_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_wait_ms": self.queue_wait_ms,
            "preprocessing_ms": self.preprocessing_ms,
            "image_encoder_ms": self.image_encoder_ms,
            "audio_encoder_ms": self.audio_encoder_ms,
            "logical_prefix_request_count": self.logical_prefix_request_count,
            "physical_prefix_chunk_count": self.physical_prefix_chunk_count,
            "prefix_token_count": self.prefix_token_count,
            "cached_prefix_token_count": self.cached_prefix_token_count,
            "candidate_prefix_recompute_tokens": self.candidate_prefix_recompute_tokens,
            "suffix_batch_count": self.suffix_batch_count,
            "suffix_batch_sizes": list(self.suffix_batch_sizes),
            "suffix_batch_ms": list(self.suffix_batch_ms),
            "aggregation_ms": self.aggregation_ms,
            "total_ms": self.total_ms,
        }


def validate_action_suffix_request(
    request: ActionSuffixScoreRequest,
) -> ActionSuffixScoreRequest:
    """Validate and normalize the public scoring contract."""
    if not isinstance(request.request_id, str) or not SAFE_REQUEST_ID.fullmatch(
        request.request_id
    ):
        raise ValueError("request_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    if not isinstance(request.model, str) or not request.model.strip():
        raise ValueError("model must be non-empty")
    if request.language not in ("zh", "en"):
        raise ValueError("language must be 'zh' or 'en'")
    if not request.prefix or not request.prefix.strip():
        raise ValueError("prefix must be non-empty")
    if not request.candidates:
        raise ValueError("candidates must not be empty")
    if len(request.candidates) > MAX_ACTION_CANDIDATES:
        raise ValueError(f"candidates must contain at most {MAX_ACTION_CANDIDATES} items")
    ids: set[str] = set()
    for candidate in request.candidates:
        if not candidate.candidate_id or candidate.candidate_id in ids:
            raise ValueError("candidate_id must be non-empty and unique")
        ids.add(candidate.candidate_id)
        if not candidate.suffix or not candidate.suffix.strip():
            raise ValueError(f"suffix must be non-empty: {candidate.candidate_id}")
        if len(candidate.suffix) > MAX_ACTION_SUFFIX_CHARS:
            raise ValueError(
                f"suffix is too long (>{MAX_ACTION_SUFFIX_CHARS} chars): "
                f"{candidate.candidate_id}"
            )
        if candidate.suffix[-1] in _TERMINAL_PUNCTUATION:
            raise ValueError(
                "suffix must not end with automatically excluded punctuation: "
                f"{candidate.candidate_id}"
            )
    if not isinstance(request.micro_batch_size, int) or not (
        MIN_MICRO_BATCH_SIZE <= request.micro_batch_size <= MAX_MICRO_BATCH_SIZE
    ):
        raise ValueError(
            f"micro_batch_size must be between {MIN_MICRO_BATCH_SIZE} and "
            f"{MAX_MICRO_BATCH_SIZE}"
        )
    if not isinstance(request.sample_rate, int) or request.sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    for name, media in (("audios", request.audios), ("images", request.images)):
        if not isinstance(media, list) or not all(isinstance(item, str) for item in media):
            raise ValueError(f"{name} must be a list of strings")
    if request.session_id is not None and not SAFE_REQUEST_ID.fullmatch(
        request.session_id
    ):
        raise ValueError("session_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    if not isinstance(request.history, list) or len(request.history) > MAX_ACTION_HISTORY_MESSAGES:
        raise ValueError(
            f"history must contain at most {MAX_ACTION_HISTORY_MESSAGES} messages"
        )
    history_audio_placeholders = 0
    history_image_placeholders = 0
    for message in request.history:
        if not isinstance(message, dict):
            raise ValueError("history messages must be objects")
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise ValueError("history message role must be a non-empty string")
        content = message.get("content")
        if isinstance(content, str) or content is None:
            continue
        if not isinstance(content, list):
            raise ValueError("history message content must be a string or list")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("history content parts must be objects")
            part_type = part.get("type")
            if part_type in ("audio", "input_audio"):
                history_audio_placeholders += 1
            elif part_type in ("image", "input_image", "image_url"):
                history_image_placeholders += 1
    for name, media in (("history_audios", request.history_audios), ("history_images", request.history_images)):
        if not isinstance(media, list) or not all(isinstance(item, str) for item in media):
            raise ValueError(f"{name} must be a list of strings")
    if history_audio_placeholders != len(request.history_audios):
        raise ValueError("history audio placeholders must match history_audios length")
    if history_image_placeholders != len(request.history_images):
        raise ValueError("history image placeholders must match history_images length")
    if not isinstance(request.avatar_state, dict):
        raise ValueError("avatar_state must be an object")
    return request


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    """Call common HF/tokenizer APIs without assuming a concrete tokenizer."""
    if hasattr(tokenizer, "encode"):
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=add_special_tokens)]
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(x) for x in encoded]


def _offset_encoded(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        return None
    if not isinstance(encoded, dict) or "input_ids" not in encoded:
        return None
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        offsets = offsets[0]
    if offsets is None or len(ids) != len(offsets):
        return None
    return [int(x) for x in ids], [(int(a), int(b)) for a, b in offsets]


def tokenize_suffixes(
    tokenizer: Any,
    prefix: str,
    candidates: Sequence[ActionScoreCandidate],
    *,
    prefix_token_ids: Sequence[int] | None = None,
    special_token_ids: Iterable[int] = (),
    terminal_token_id: int | None = None,
) -> list[CandidateTokenization]:
    """Tokenize full prefix+suffix strings and derive an explicit suffix mask.

    Offset mappings are preferred because BPE tokenization is not generally
    compositional at a text boundary.  For multimodal chat templates whose
    special tokens cannot be reconstructed from plain text, a non-whitespace
    action suffix is composed onto the authoritative prefix token IDs.
    ``terminal_token_id`` is appended explicitly and remains in the score mask
    even when it is also a tokenizer special token.
    """
    if not prefix:
        raise ValueError("prefix must be non-empty")
    special = {int(x) for x in special_token_ids}
    prefix_ids = (
        tuple(int(x) for x in prefix_token_ids)
        if prefix_token_ids is not None
        else tuple(_encode(tokenizer, prefix, add_special_tokens=False))
    )
    result: list[CandidateTokenization] = []
    for candidate in candidates:
        full_text = prefix + candidate.suffix
        encoded = _offset_encoded(tokenizer, full_text)
        if encoded is not None:
            full_ids_list, offsets = encoded
            boundary = len(prefix)
            mask = tuple(
                end > boundary and token_id not in special
                for token_id, (_, end) in zip(full_ids_list, offsets, strict=True)
            )
            suffix_ids = tuple(
                token_id for token_id, include in zip(full_ids_list, mask, strict=True) if include
            )
            suffix_start = next((i for i, include in enumerate(mask) if include), len(full_ids_list))
            if terminal_token_id is not None:
                full_ids_list.append(int(terminal_token_id))
                mask = (*mask, True)
                suffix_ids = (*suffix_ids, int(terminal_token_id))
            result.append(
                CandidateTokenization(
                    candidate_id=candidate.candidate_id,
                    full_input_ids=tuple(full_ids_list),
                    suffix_start_index=suffix_start,
                    suffix_token_mask=mask,
                    suffix_token_ids=suffix_ids,
                )
            )
            continue

        full_ids = tuple(_encode(tokenizer, full_text, add_special_tokens=False))
        if tuple(full_ids[: len(prefix_ids)]) != prefix_ids:
            # Multimodal chat templates can contain special tokens that the
            # plain tokenizer cannot reproduce from prompt text. For a CJK
            # action suffix, compose independently encoded suffix tokens onto
            # the authoritative model prefix IDs.
            if not candidate.suffix or candidate.suffix[0].isspace():
                raise ValueError(
                    "tokenizer does not expose offsets and prefix/suffix BPE boundary "
                    f"cannot be aligned for {candidate.candidate_id!r}"
                )
            suffix_only = tuple(_encode(tokenizer, candidate.suffix, add_special_tokens=False))
            if not suffix_only:
                raise ValueError(f"suffix tokenization is empty: {candidate.candidate_id!r}")
            full_ids = prefix_ids + suffix_only
        suffix_start = len(prefix_ids)
        mask = tuple(
            i >= suffix_start and token_id not in special
            for i, token_id in enumerate(full_ids)
        )
        suffix_ids = tuple(
            token_id for token_id, include in zip(full_ids, mask, strict=True) if include
        )
        if terminal_token_id is not None:
            full_ids = (*full_ids, int(terminal_token_id))
            mask = (*mask, True)
            suffix_ids = (*suffix_ids, int(terminal_token_id))
        result.append(
            CandidateTokenization(
                candidate_id=candidate.candidate_id,
                full_input_ids=full_ids,
                suffix_start_index=suffix_start,
                suffix_token_mask=mask,
                suffix_token_ids=suffix_ids,
            )
        )
    return result


def build_suffix_batches(
    tokenizations: Sequence[CandidateTokenization], micro_batch_size: int
) -> list[SuffixBatch]:
    if micro_batch_size < MIN_MICRO_BATCH_SIZE or micro_batch_size > MAX_MICRO_BATCH_SIZE:
        raise ValueError("micro_batch_size is outside the supported range")
    return [
        SuffixBatch(index=start // micro_batch_size, candidates=tuple(tokenizations[start : start + micro_batch_size]))
        for start in range(0, len(tokenizations), micro_batch_size)
    ]


def align_suffix_logprobs(
    tokenization: CandidateTokenization,
    logprobs: Sequence[float | None],
    *,
    logprob_start_index: int = 0,
    token_ids: Sequence[int] | None = None,
) -> list[TokenScore]:
    """Align input-token logprobs to the true suffix mask.

    ``logprobs`` may be a full-sequence array or an array beginning at
    ``logprob_start_index``.  The function intentionally rejects ambiguous
    lengths and token-id mismatches instead of silently shifting scores.
    """
    if logprob_start_index < 0:
        raise ValueError("logprob_start_index must be non-negative")
    full_len = len(tokenization.full_input_ids)
    if len(logprobs) == full_len:
        values = list(logprobs)
        value_offset = 0
    elif logprob_start_index + len(logprobs) == full_len:
        values = list(logprobs)
        value_offset = logprob_start_index
    else:
        raise ValueError(
            f"input logprob length {len(logprobs)} cannot cover full sequence "
            f"length {full_len} from {logprob_start_index}"
        )
    supplied_ids = list(token_ids) if token_ids is not None else None
    scores: list[TokenScore] = []
    for position, (token_id, include) in enumerate(
        zip(tokenization.full_input_ids, tokenization.suffix_token_mask, strict=True)
    ):
        if not include:
            continue
        value_index = position - value_offset
        if not 0 <= value_index < len(values) or values[value_index] is None:
            raise ValueError(f"missing input-token logprob at position {position}")
        if supplied_ids is not None:
            id_index = position if len(supplied_ids) == full_len else value_index
            if supplied_ids[id_index] != token_id:
                raise ValueError(
                    f"input-token logprob id mismatch at position {position}: "
                    f"expected {token_id}, got {supplied_ids[id_index]}"
                )
        score = float(values[value_index])
        if not math.isfinite(score):
            raise ValueError(f"non-finite input-token logprob at position {position}")
        scores.append(TokenScore(token_id=int(token_id), logprob=score))
    if not scores:
        raise ValueError(f"candidate {tokenization.candidate_id!r} has no suffix tokens")
    return scores


def aggregate_candidate_score(
    candidate_id: str,
    token_scores: Sequence[TokenScore],
) -> CandidateScore:
    if not token_scores:
        raise ValueError("cannot aggregate an empty suffix")
    total = math.fsum(score.logprob for score in token_scores)
    mean_logprob = total / len(token_scores)
    mean_nll = -mean_logprob
    ppl = math.exp(mean_nll)
    values = (mean_logprob, mean_nll, ppl, total)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite score for candidate {candidate_id!r}")
    return CandidateScore(
        candidate_id=candidate_id,
        token_count=len(token_scores),
        mean_logprob=mean_logprob,
        mean_nll=mean_nll,
        ppl=ppl,
        token_scores=list(token_scores),
    )


def validate_score_result(
    request: ActionSuffixScoreRequest,
    result: ActionSuffixScoreResult,
) -> None:
    expected = [candidate.candidate_id for candidate in request.candidates]
    actual = [score.candidate_id for score in result.scores]
    if result.request_id != request.request_id:
        raise ValueError("score result request_id does not match request")
    if result.model != request.model:
        raise ValueError("score result model does not match request")
    if actual != expected:
        raise ValueError("score result candidate order/set does not match request")
    if not result.prefix_cached:
        raise RuntimeError("action scoring did not obtain a verified cached prefix")
    for score in result.scores:
        if score.token_count <= 0 or not score.token_scores:
            raise ValueError(f"empty score for candidate {score.candidate_id!r}")
        if not all(
            math.isfinite(value)
            for value in (score.mean_logprob, score.mean_nll, score.ppl)
        ):
            raise ValueError(f"non-finite score for candidate {score.candidate_id!r}")


def _digest_media_ref(value: str) -> str:
    hasher = hashlib.sha256()
    path = Path(value)
    try:
        if path.is_file():
            stat = path.stat()
            hasher.update(f"file:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        else:
            hasher.update(value.encode("utf-8"))
    except OSError:
        # A media loader will report the actual access error later.  The cache
        # key still remains request-specific and never collapses to a shared
        # placeholder-only identity.
        hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()


def build_multimodal_cache_identity(
    *,
    prefix_token_ids: Sequence[int],
    audios: Sequence[str],
    images: Sequence[str],
    request_scope: str,
) -> tuple[str, str]:
    """Return a cache key and a safe digest for logs.

    ``request_scope`` isolates independent turns.  Media content digests are
    included even when the tokenized placeholder sequence is identical.
    """
    payload = {
        "scope": request_scope,
        "prefix": list(int(x) for x in prefix_token_ids),
        "audios": [_digest_media_ref(item) for item in audios],
        "images": [_digest_media_ref(item) for item in images],
    }
    encoded = repr(payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"qwen3-omni-action:{digest}", digest[:16]


class ActionScoringTimer:
    """Small monotonic timer helper for stage-level metrics."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0


def score_candidate_from_runtime(
    candidate_id: str,
    suffix_token_ids: Sequence[int],
    prefix_next_token_logits: Any,
    continuation_logprobs: Sequence[float],
) -> CandidateScore:
    """Combine shared-prefix next-token logits with suffix input logprobs."""
    if not suffix_token_ids:
        raise ValueError("candidate suffix has no tokens")
    logits = prefix_next_token_logits
    if isinstance(logits, dict):
        raw = [float(value) for value in continuation_logprobs]
        if len(raw) == len(suffix_token_ids):
            # The regular SGLang prefill path returns the input-token logprob
            # for every token from logprob_start_len, including token 0 of the
            # suffix.  It is already predicted by the shared prefix state.
            return aggregate_candidate_score(
                candidate_id,
                [
                    TokenScore(int(token_id), value)
                    for token_id, value in zip(suffix_token_ids, raw, strict=True)
                ],
            )
        if len(raw) != len(suffix_token_ids) - 1:
            raise ValueError("candidate continuation logprob count is incomplete")
        try:
            first = float(logits[int(suffix_token_ids[0])])
        except KeyError as exc:
            raise ValueError("prefix logprob for first suffix token is missing") from exc
        return aggregate_candidate_score(
            candidate_id,
            [TokenScore(int(suffix_token_ids[0]), first)]
            + [
                TokenScore(int(token_id), value)
                for token_id, value in zip(suffix_token_ids[1:], raw, strict=True)
            ],
        )
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu()
    if hasattr(logits, "tolist"):
        logits = logits.tolist()
    if isinstance(logits, list) and logits and isinstance(logits[0], list):
        logits = logits[0]
    maximum = max(float(value) for value in logits)
    log_denom = maximum + math.log(
        math.fsum(math.exp(float(value) - maximum) for value in logits)
    )
    raw = [float(value) for value in continuation_logprobs]
    if len(raw) == len(suffix_token_ids):
        # SGLang prefill-only input-token logprobs include the first suffix
        # token: its prediction is produced by the final shared-prefix state.
        token_scores = [
            TokenScore(int(token_id), value)
            for token_id, value in zip(suffix_token_ids, raw, strict=True)
        ]
    elif len(raw) == len(suffix_token_ids) - 1:
        # Compatibility path for a backend that returns only continuation
        # positions after the shared-prefix logits.
        first = float(logits[int(suffix_token_ids[0])]) - log_denom
        token_scores = [TokenScore(int(suffix_token_ids[0]), first)] + [
            TokenScore(int(token_id), value)
            for token_id, value in zip(suffix_token_ids[1:], raw, strict=True)
        ]
    else:
        raise ValueError(
            "candidate input-token logprob count is neither the full suffix "
            "nor the continuation-only suffix length"
        )
    return aggregate_candidate_score(candidate_id, token_scores)


__all__ = [
    "ActionScoreCandidate",
    "ActionScoringStats",
    "ActionScoringTimer",
    "ActionSuffixScoreRequest",
    "ActionSuffixScoreResult",
    "CandidateScore",
    "CandidateTokenization",
    "Language",
    "SuffixBatch",
    "TokenScore",
    "aggregate_candidate_score",
    "score_candidate_from_runtime",
    "align_suffix_logprobs",
    "build_multimodal_cache_identity",
    "build_suffix_batches",
    "tokenize_suffixes",
    "validate_action_suffix_request",
    "validate_score_result",
]
