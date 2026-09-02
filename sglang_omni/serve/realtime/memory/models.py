"""Data contracts and shared policy primitives for session memory.

This module intentionally contains no model client or asyncio scheduler code.
Both the deterministic store and the extraction adapter depend on these
contracts, which keeps their dependency direction acyclic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .config import *  # noqa: F403
from .config import SessionMemoryConfig


MEMORY_OPERATION_ADD = "add"
MEMORY_OPERATION_SUPERSEDE = "supersede"
MEMORY_OPERATION_RETRACT = "retract"
MEMORY_OPERATION_NOOP = "noop"

THREAD_OPERATION_OPEN = "open"
THREAD_OPERATION_UPDATE = "update"
THREAD_OPERATION_RESOLVE = "resolve"
THREAD_OPERATION_REJECT = "reject"
THREAD_OPERATION_NOOP = "noop"
THREAD_OPERATIONS = frozenset(
    {
        THREAD_OPERATION_OPEN,
        THREAD_OPERATION_UPDATE,
        THREAD_OPERATION_RESOLVE,
        THREAD_OPERATION_REJECT,
        THREAD_OPERATION_NOOP,
    }
)

THREAD_STATUS_OPEN = "open"
THREAD_STATUS_RESOLVED = "resolved"
THREAD_STATUS_REJECTED = "rejected"

MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_SUPERSEDED = "superseded"
MEMORY_STATUS_RETRACTED = "retracted"

MEMORY_LIFECYCLE_SESSION = "session"
MEMORY_LIFECYCLE_UNTIL_REPLACED = "until_replaced"
MEMORY_LIFECYCLE_HISTORICAL = "historical"

MEMORY_ARTIFACT_KINDS = frozenset(
    {"none", "story", "copywriting", "translation", "explanation", "other"}
)
MEMORY_LIFECYCLES = frozenset(
    {
        MEMORY_LIFECYCLE_SESSION,
        MEMORY_LIFECYCLE_UNTIL_REPLACED,
        MEMORY_LIFECYCLE_HISTORICAL,
    }
)
MEMORY_OPERATIONS = frozenset(
    {
        MEMORY_OPERATION_ADD,
        MEMORY_OPERATION_SUPERSEDE,
        MEMORY_OPERATION_RETRACT,
        MEMORY_OPERATION_NOOP,
    }
)

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
SAFE_KEY_RE = re.compile(r"^[\w.:-]{1,96}$", re.UNICODE)
SENSITIVE_RE = re.compile(
    r"(?:密码|口令|验证码|银行卡|私钥|助记词|访问令牌|密钥|"
    r"password|passcode|verification\s*code|bank\s*card|private\s*key|"
    r"seed\s*phrase|access\s*token|api\s*key)",
    re.IGNORECASE,
)
PROMPT_INJECTION_RE = re.compile(
    r"(?:忽略.{0,12}(?:规则|指令|提示词)|泄露.{0,12}(?:规则|提示词)|"
    r"ignore.{0,20}(?:instruction|prompt)|system\s*prompt)",
    re.IGNORECASE,
)
EPISODE_HISTORY_CUE_RE = re.compile(
    r"(?:之前|刚才|上次|前面|先前|继续|接着|讲到|说到|聊到|讨论|"
    r"重复|再说一遍|previous|earlier|last\s+time|continue|carry\s+on|"
    r"where\s+did\s+we|what\s+did\s+we|repeat)",
    re.IGNORECASE,
)
ARTIFACT_HISTORY_CUE_RE = re.compile(
    r"(?:那个|那段|故事|笑话|文案|翻译|诗|介绍|that\s+one|story|joke|"
    r"copywriting|translation|poem|introduction)",
    re.IGNORECASE,
)
EXPLICIT_RETRACTION_RE = re.compile(
    r"(?:忘掉|忘记|别记|不要记|删除|清除|移除|不再叫|别再叫|别叫|"
    r"forget|delete|remove|do not remember|don't remember|stop calling)",
    re.IGNORECASE,
)

PROTECTED_PREDICATE_ALIASES: dict[str, str] = {
    "self_reported_name": "identity.self_reported_name",
    "user_name": "identity.self_reported_name",
    "username": "identity.self_reported_name",
    "姓名": "identity.self_reported_name",
    "名字": "identity.self_reported_name",
    "用户姓名": "identity.self_reported_name",
    "preferred_address": "preference.preferred_address",
    "preferred_name": "preference.preferred_address",
    "preferred_form_of_address": "preference.preferred_address",
    "偏好称呼": "preference.preferred_address",
    "希望称呼": "preference.preferred_address",
    "称呼偏好": "preference.preferred_address",
}
NON_DURABLE_PREDICATES = frozenset(
    {
        "request",
        "请求",
        "action_request",
        "动作请求",
        "command",
        "命令",
        "emotion",
        "情绪",
        "mood",
        "心情",
        "current_state",
        "当前状态",
    }
)


def canonical_predicate(predicate: str) -> str:
    normalized = predicate.strip()
    alias = PROTECTED_PREDICATE_ALIASES.get(normalized.casefold())
    return alias or normalized


def is_non_durable_predicate(predicate: str) -> bool:
    return predicate.strip().casefold() in {
        item.casefold() for item in NON_DURABLE_PREDICATES
    }


@dataclass(frozen=True, slots=True)
class SessionMemoryTurn:
    turn_id: str
    turn_seq: int
    user_text: str | None
    audios: tuple[str, ...]
    assistant_text: str | None
    reply_model_visible: bool
    reply_mode: str | None
    queued_at: float = 0.0


@dataclass(slots=True)
class SessionSemanticClaim:
    memory_id: str
    subject: str
    predicate: str
    value: str
    content: str
    source_turn_id: str
    source_authority: Literal["user_claim", "server_confirmed"]
    lifecycle: Literal["session", "until_replaced", "historical"]
    status: Literal["active", "superseded", "retracted"]
    created_turn_seq: int
    evidence: str = ""
    confidence: float = 1.0
    version: int = 1

    def as_context_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "source": self.source_authority,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "content": self.content,
            "lifecycle": self.lifecycle,
            "source_turn_id": self.source_turn_id,
            "created_turn_seq": str(self.created_turn_seq),
            "confidence": f"{self.confidence:.3f}",
        }


@dataclass(slots=True)
class SessionEpisodeRecord:
    turn_id: str
    turn_seq: int
    user_summary: str | None
    assistant_summary: str | None
    artifact_kind: Literal[
        "none", "story", "copywriting", "translation", "explanation", "other"
    ]
    model_visible: bool

    def as_context_dict(self) -> dict[str, str]:
        result = {"turn_id": self.turn_id, "artifact_kind": self.artifact_kind}
        if self.user_summary:
            result["user_summary"] = self.user_summary
        if self.assistant_summary:
            result["assistant_artifact_summary"] = self.assistant_summary
        return result


@dataclass(slots=True)
class SessionArtifactRecord:
    artifact_id: str
    turn_id: str
    turn_seq: int
    content_kind: str
    content: str
    summary: str | None

    def as_context_dict(self) -> dict[str, str]:
        result = {
            "artifact_id": self.artifact_id,
            "source": "assistant_artifact",
            "source_turn_id": self.turn_id,
            "content_kind": self.content_kind,
            "content": self.content,
        }
        if self.summary:
            result["summary"] = self.summary
        return result


@dataclass(slots=True)
class SessionOpenThread:
    thread_id: str
    content: str
    status: Literal["open", "resolved", "rejected"]
    source_turn_ids: tuple[str, ...]
    source_authority: Literal["user_supported", "server_confirmed"]
    created_turn_seq: int
    updated_turn_seq: int
    evidence: str = ""
    confidence: float = 1.0
    last_proactive_turn_seq: int = 0
    proactive_attempt_count: int = 0

    def as_context_dict(self) -> dict[str, str]:
        return {
            "thread_id": self.thread_id,
            "source": self.source_authority,
            "content": self.content,
            "status": self.status,
            "source_turn_ids": ",".join(self.source_turn_ids),
            "created_turn_seq": str(self.created_turn_seq),
            "updated_turn_seq": str(self.updated_turn_seq),
            "confidence": f"{self.confidence:.3f}",
        }


@dataclass(frozen=True, slots=True)
class ExtractedOpenThreadOperation:
    op: Literal["open", "update", "resolve", "reject", "noop"]
    thread_id: str = ""
    content: str = ""
    evidence: str = ""
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ExtractedMemoryOperation:
    op: Literal["add", "supersede", "retract", "noop"]
    subject: str = ""
    predicate: str = ""
    value: str = ""
    content: str = ""
    lifecycle: Literal["session", "until_replaced", "historical"] = (
        MEMORY_LIFECYCLE_SESSION
    )
    target_memory_ids: tuple[str, ...] = ()
    evidence: str = ""
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ExtractedTurnMemory:
    turn_id: str
    turn_seq: int
    user_summary: str | None
    assistant_summary: str | None
    artifact_kind: str
    operations: tuple[ExtractedMemoryOperation, ...]
    thread_operations: tuple[ExtractedOpenThreadOperation, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionMemoryContext:
    text: str
    claim_count: int
    episode_count: int
    artifact_count: int
    source_turn_ids: tuple[str, ...]
    open_thread_count: int = 0
    selected_claim_ids: tuple[str, ...] = ()
    selected_artifact_ids: tuple[str, ...] = ()
    selected_open_thread_ids: tuple[str, ...] = ()
    retrieval_mode: str = "none"


@dataclass(frozen=True, slots=True)
class SessionMemoryApplyStats:
    added: int = 0
    superseded: int = 0
    retracted: int = 0
    rejected: int = 0
    episode_count: int = 0
    artifact_count: int = 0
    opened_thread_count: int = 0
    updated_thread_count: int = 0
    closed_thread_count: int = 0
    rejected_operations: tuple[dict[str, Any], ...] = ()
    processed_through_turn_seq: int = 0
    complete_through_turn_seq: int = 0
    gap_count: int = 0
    compactable_turn_ids: tuple[str, ...] = ()
    store_revision: int = 0


class StaleSessionMemoryBatch(RuntimeError):
    """Raised when extraction was based on an obsolete store snapshot."""
