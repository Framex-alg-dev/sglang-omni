"""Stable wire-neutral contract for the model-1 turn router."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


CONTRACT_VERSION = 5
DEFAULT_BRAIN1_CAPABILITIES = (
    "普通聊天、简单问答、计算、下游用户视频理解；"
    "不处理应用能力、曲库或媒体播放查询"
)
DEFAULT_BRAIN2_CAPABILITIES = (
    "搜索、天气、票务、日历、业务服务、多步Agent；"
    "歌曲能力、曲库查询、所有立即唱歌和选歌请求、歌曲播放；"
    "唱歌请求不是普通对话或纯动作"
)


def _required(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    return normalized


class RouteToken(str, Enum):
    DIRECT = "direct"
    DELEGATE = "delegate"
    CONTROL = "control"


class OutputDirective(str, Enum):
    KEEP = "keep"
    STOP_CURRENT = "stop_current"
    SUPPRESS_REPLY = "suppress_reply"


class TaskDirective(str, Enum):
    KEEP = "keep"
    CANCEL_CURRENT = "cancel_current"
    CANCEL_ALL = "cancel_all"


class MediaDirective(str, Enum):
    NONE = "none"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"


class BrainRoute(str, Enum):
    BRAIN1 = "BRAIN1"
    BRAIN2 = "BRAIN2"
    CONTROL = "CONTROL"


ROUTE_BY_TOKEN = {
    RouteToken.DIRECT: BrainRoute.BRAIN1,
    RouteToken.DELEGATE: BrainRoute.BRAIN2,
    RouteToken.CONTROL: BrainRoute.CONTROL,
}


_ZH_CUE_QUESTION = re.compile(
    r"(?:[？?]|什么|哪些|哪种|怎么|如何|为什么|为何|谁|哪里|哪儿|多少|吗|么)$"
    r"|(?:什么|哪些|哪种|怎么|如何|为什么|为何|谁|哪里|哪儿|多少)"
)
_ZH_CUE_REQUEST_PREFIX = re.compile(
    r"^(?:你会|你能|你可以|您会|您能|您可以|我想|我要|我需要|帮我|请帮我|给我)"
)
_EN_CUE_QUESTION_PREFIX = re.compile(
    r"^(?:what|which|who|where|when|why|how|can\s+you|could\s+you|"
    r"would\s+you|will\s+you|do\s+you|are\s+you)\b",
    re.IGNORECASE,
)


def _validate_request_cue_object(*, verb: str, value: str, language: str) -> None:
    if value.casefold().startswith(verb.casefold()):
        raise ValueError("request cue object must not repeat its verb")
    if language == "zh-CN" and (
        _ZH_CUE_QUESTION.search(value) is not None
        or _ZH_CUE_REQUEST_PREFIX.search(value) is not None
    ):
        raise ValueError("request cue object must be a normalized noun phrase")
    if language == "en-US" and (
        "?" in value or _EN_CUE_QUESTION_PREFIX.search(value) is not None
    ):
        raise ValueError("request cue object must be a normalized noun phrase")


@dataclass(frozen=True)
class RequestCuePlan:
    """Grounded request acknowledgement prepared by the fast turn router."""

    verb: str
    object: str
    language: str

    def __post_init__(self) -> None:
        verb = _required("request cue verb", self.verb)
        object_text = _required("request cue object", self.object)
        if len(verb) > 24 or len(object_text) > 120:
            raise ValueError("request cue verb or object is too long")
        if any(char in verb or char in object_text for char in "\r\n{}"):
            raise ValueError("request cue verb and object must be plain text")
        if self.language not in {"zh-CN", "en-US"}:
            raise ValueError("request cue language is unsupported")
        _validate_request_cue_object(
            verb=verb,
            value=object_text,
            language=self.language,
        )
        object.__setattr__(self, "verb", verb)
        object.__setattr__(self, "object", object_text)


@dataclass(frozen=True)
class ClassificationMediaRef:
    """One original user-audio input; derived ASR and visual media are forbidden."""

    media_id: str
    kind: str
    start_ms: int
    end_ms: int
    encoding: str
    checksum: str
    payload: bytes

    def __post_init__(self) -> None:
        _required("media_id", self.media_id)
        if self.kind != "audio":
            raise ValueError("turn-router media must be original user audio")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("turn-router audio range is invalid")
        _required("audio encoding", self.encoding)
        _required("audio checksum", self.checksum)
        if not self.payload:
            raise ValueError("turn-router audio payload must not be empty")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.checksum) is None:
            raise ValueError("turn-router audio checksum must be sha256")
        actual = "sha256:" + hashlib.sha256(self.payload).hexdigest()
        if actual != self.checksum:
            raise ValueError("turn-router audio checksum does not match payload")


@dataclass(frozen=True)
class TaskClassificationRequest:
    request_id: str
    session_id: str
    turn_id: str
    identity_epoch: int
    input_revision: int
    text: str | None
    media: tuple[ClassificationMediaRef, ...]
    router_history: tuple[dict[str, Any], ...] = ()
    brain1_capabilities: str = DEFAULT_BRAIN1_CAPABILITIES
    brain2_capabilities: str = DEFAULT_BRAIN2_CAPABILITIES
    has_active_agent: bool = False
    pending_confirmation: bool = False
    follow_up_required: bool = False
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _required("request_id", self.request_id)
        _required("session_id", self.session_id)
        _required("turn_id", self.turn_id)
        if self.identity_epoch < 0 or self.input_revision < 0:
            raise ValueError("turn-router revisions must be non-negative")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported turn-router contract version")
        has_text = bool((self.text or "").strip())
        if has_text == bool(self.media):
            raise ValueError("turn-router request needs exactly one of text or audio")
        if len(self.media) > 1:
            raise ValueError("turn-router request accepts only one original audio input")
        if not all(isinstance(item, dict) for item in self.router_history):
            raise ValueError("router_history entries must be objects")
        _required("brain1_capabilities", self.brain1_capabilities)
        _required("brain2_capabilities", self.brain2_capabilities)


@dataclass(frozen=True)
class TaskClassificationResult:
    request_id: str
    session_id: str
    turn_id: str
    identity_epoch: int
    input_revision: int
    route_token: RouteToken
    route: BrainRoute
    output_directive: OutputDirective
    task_directive: TaskDirective
    media_directive: MediaDirective
    response_locale: str
    request_cue: RequestCuePlan | None
    model_id: str
    model_version: str
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("session_id", self.session_id),
            ("turn_id", self.turn_id),
            ("model_id", self.model_id),
            ("model_version", self.model_version),
        ):
            _required(name, value)
        if ROUTE_BY_TOKEN[self.route_token] is not self.route:
            raise ValueError("turn-router token and Brain route disagree")
        if self.response_locale not in {"zh-CN", "en-US"}:
            raise ValueError("turn-router response locale is unsupported")
        if (
            self.task_directive is not TaskDirective.KEEP
            and self.route_token is not RouteToken.CONTROL
        ):
            raise ValueError("task cancellation requires the control route")
        if (
            self.media_directive is not MediaDirective.NONE
            and self.route_token is not RouteToken.CONTROL
        ):
            raise ValueError("media control requires the control route")
        if (
            self.route_token is RouteToken.CONTROL
            and self.task_directive is TaskDirective.KEEP
            and self.output_directive is OutputDirective.KEEP
            and self.media_directive is MediaDirective.NONE
        ):
            raise ValueError(
                "control route requires an output, task, or media directive"
            )
        if self.route_token is RouteToken.DELEGATE and self.request_cue is None:
            raise ValueError("delegated route requires a request cue")
        if self.route_token is not RouteToken.DELEGATE and self.request_cue is not None:
            raise ValueError("only delegated routes may contain a request cue")
