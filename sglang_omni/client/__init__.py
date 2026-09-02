# SPDX-License-Identifier: Apache-2.0
"""Client package."""

from typing import TYPE_CHECKING

from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
    ClientError,
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
    Message,
    SamplingParams,
    SpeechResult,
    TokenScore,
    UsageInfo,
)

if TYPE_CHECKING:
    from sglang_omni.client.client import Client

__all__ = [
    "Client",
    "ActionScoreCandidate",
    "ActionSuffixScoreRequest",
    "ActionSuffixScoreResult",
    "CandidateScore",
    "TokenScore",
    "AbortLevel",
    "AbortResult",
    "ClientError",
    "CompletionAudio",
    "CompletionResult",
    "CompletionStreamChunk",
    "GenerateChunk",
    "GenerateRequest",
    "Message",
    "SamplingParams",
    "SpeechResult",
    "UsageInfo",
]


def __getattr__(name: str):
    """Load the pipeline-backed client only when callers request it."""
    if name == "Client":
        from sglang_omni.client.client import Client

        return Client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
