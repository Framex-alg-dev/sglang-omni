"""Typed contracts for independent body and expression inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sglang_omni.serve.streaming_request import StreamedMedia


class DecisionChannel(str, Enum):
    BODY = "body"
    EXPRESSION = "expression"


@dataclass(frozen=True)
class ActionDecisionRequest:
    request_id: str
    session_id: str
    turn_id: str
    decision_point_id: str
    channel: DecisionChannel
    text: str | None = None
    language: str = "zh-CN"
    character_prompt: str | None = None
    session_prompt: str | None = None
    reply_prefix: str | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    allowed_candidate_ids: tuple[str, ...] = ()
    excluded_candidate_ids: tuple[str, ...] = ()
    channel_enabled: bool = True
    prohibited: bool = False
    turn_origin: str = "user"
    media: tuple[StreamedMedia, ...] = ()


@dataclass(frozen=True)
class ActionDecision:
    contract_version: int
    request_id: str
    session_id: str
    turn_id: str
    decision_point_id: str
    channel: DecisionChannel
    outcome: str
    candidate_id: str
    action_id: str | None
    code: str
    token_ids: tuple[int, int]
    label: str
    occupies_channels: tuple[str, ...]
    deterministic: bool
    model_invoked: bool
    reason_code: str
    model_id: str
    model_version: str
    weight_version: str | None
    catalog_version: str
    catalog_hash: str
    mapping_version: str
    mapping_hash: str
    candidate_count: int
    elapsed_ms: float
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["channel"] = self.channel.value
        value["token_ids"] = list(self.token_ids)
        value["occupies_channels"] = list(self.occupies_channels)
        return value
