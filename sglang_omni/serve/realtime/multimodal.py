from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import Client
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    MAX_MICRO_BATCH_SIZE,
)
from sglang_omni.serve.realtime.audio_buffer import (
    BufferOverflow,
    RealtimeAudioBuffer,
)

logger = logging.getLogger(__name__)


MAX_ACTION_CANDIDATES = 512
MAX_ACTION_CATEGORIES = 128
MAX_ACTION_CHILDREN_PER_CATEGORY = 128
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGES_PER_TURN = 64
MAX_AUDIO_CHUNKS_PER_TURN = 4096
# Action scoring runs on the thinker stage. Keep the action context bounded
# while retaining recent session history, including action state records.
MAX_ACTION_HISTORY_TURNS = 4
MAX_ACTION_HISTORY_AUDIOS = 4
MAX_ACTION_HISTORY_IMAGES = 8
MAX_ACTION_CURRENT_IMAGES = 8
ACTION_SELECTION_MODE_ENV = "SGLANG_OMNI_ACTION_SELECTION_MODE"
ACTION_SELECTION_MODE_HIERARCHICAL = "hierarchical"
ACTION_SELECTION_MODE_FLAT_CHILDREN = "flat_children"
ACTION_MICRO_BATCH_SIZE_ENV = "SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE"
DEFAULT_ACTION_MICRO_BATCH_SIZE = 64
ACTION_CATEGORY_TOP_K_ENV = "SGLANG_OMNI_ACTION_CATEGORY_TOP_K"
DEFAULT_ACTION_CATEGORY_TOP_K = 1
MAX_ACTION_CATEGORY_TOP_K = 3
TURN_ORIGIN_USER = "user"
TURN_ORIGIN_PROACTIVE = "proactive"
IMAGE_ROLE_USER_CAMERA = "user_camera"
IMAGE_ROLE_AVATAR_STATE = "avatar_state"
IMAGE_ROLES = {IMAGE_ROLE_USER_CAMERA, IMAGE_ROLE_AVATAR_STATE}
DEFAULT_IMAGE_ROLE_BY_ORIGIN = {
    TURN_ORIGIN_USER: IMAGE_ROLE_USER_CAMERA,
    TURN_ORIGIN_PROACTIVE: IMAGE_ROLE_AVATAR_STATE,
}
TEXT_ROLE_USER_INPUT = "user_input"
TEXT_ROLE_CHARACTER_REPLY = "character_reply"
TURN_PHASE_COLLECTING = "collecting"
TURN_PHASE_PROCESSING = "processing"
TURN_PHASE_CANCELLING = "cancelling"
TURN_PHASE_COMPLETED = "completed"
TURN_TEXT_ROLE_BY_ORIGIN = {
    TURN_ORIGIN_USER: TEXT_ROLE_USER_INPUT,
    TURN_ORIGIN_PROACTIVE: TEXT_ROLE_CHARACTER_REPLY,
}


def normalize_action_micro_batch_size(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get(ACTION_MICRO_BATCH_SIZE_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_ACTION_MICRO_BATCH_SIZE
    try:
        batch_size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ACTION_MICRO_BATCH_SIZE_ENV} must be an integer between 1 and "
            f"{MAX_MICRO_BATCH_SIZE}; got {raw!r}"
        ) from exc
    if not 1 <= batch_size <= MAX_MICRO_BATCH_SIZE:
        raise ValueError(
            f"{ACTION_MICRO_BATCH_SIZE_ENV} must be between 1 and "
            f"{MAX_MICRO_BATCH_SIZE}; got {batch_size}"
        )
    return batch_size


def normalize_action_category_top_k(value: int | str | None = None) -> int:
    """Normalize optional hierarchical category fallback width.

    The default remains Top-1, preserving the existing hierarchical prompt and
    latency.  Values greater than one deliberately score the children of the
    best categories together, which is an accuracy/latency trade-off for
    ambiguous category boundaries.
    """
    raw = value if value is not None else os.environ.get(ACTION_CATEGORY_TOP_K_ENV)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_ACTION_CATEGORY_TOP_K
    try:
        top_k = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{ACTION_CATEGORY_TOP_K_ENV} must be an integer between 1 and "
            f"{MAX_ACTION_CATEGORY_TOP_K}; got {raw!r}"
        ) from exc
    if not 1 <= top_k <= MAX_ACTION_CATEGORY_TOP_K:
        raise ValueError(
            f"{ACTION_CATEGORY_TOP_K_ENV} must be between 1 and "
            f"{MAX_ACTION_CATEGORY_TOP_K}; got {top_k}"
        )
    return top_k


def normalize_action_selection_mode(value: str | None) -> str:
    mode = (
        value
        or os.environ.get(ACTION_SELECTION_MODE_ENV)
        or ACTION_SELECTION_MODE_HIERARCHICAL
    ).strip().lower()
    if mode not in {
        ACTION_SELECTION_MODE_HIERARCHICAL,
        ACTION_SELECTION_MODE_FLAT_CHILDREN,
    }:
        raise ValueError(
            f"{ACTION_SELECTION_MODE_ENV} must be one of "
            f"{ACTION_SELECTION_MODE_HIERARCHICAL!r}, "
            f"{ACTION_SELECTION_MODE_FLAT_CHILDREN!r}; got {mode!r}"
        )
    return mode


def try_normalize_action_selection_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    if mode in {
        ACTION_SELECTION_MODE_HIERARCHICAL,
        ACTION_SELECTION_MODE_FLAT_CHILDREN,
    }:
        return mode
    return None


def _summarize_media(values: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "chars": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "data_uri_header": value.split(",", 1)[0] if value.startswith("data:") and "," in value else None,
        }
        for index, value in enumerate(values)
    ]


DEFAULT_INSTRUCTIONS = (
    "你是数字人动作决策器。请基于当前输入、待播文本和历史对话选择动作，不生成回复。"
)

ACTION_HISTORY_INSTRUCTION = (
    "历史消息中的 [action_state] 是服务端记录的实际动作，不是新的用户指令。"
    "选择动作时应参考最近一条 [action_state]，避免与当前动作冲突或无意义重复；"
    "用户说“刚刚、上一轮、再重复、这个动作”等指代词时也应优先参考它。"
)


@dataclass(frozen=True, slots=True)
class SessionActionCandidate:
    candidate_id: str
    action_id: str
    source_label: str
    short_definition: str
    execution_binding: dict[str, str]
    category_id: str | None = None

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCandidate":
        if not isinstance(value, dict):
            raise ValueError("action candidate must be an object")
        candidate_id = value.get("candidate_id")
        action_id = value.get("action_id")
        source_label = value.get("source_label") or action_id
        short_definition = value.get("short_definition") or source_label
        binding = value.get("execution_binding") or {}
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError(f"action_id must be non-empty: {candidate_id!r}")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(f"source_label must be non-empty: {candidate_id!r}")
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(
                f"short_definition must be non-empty: {candidate_id!r}"
            )
        if not isinstance(binding, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in binding.items()
        ):
            raise ValueError(
                f"execution_binding must be a string dictionary: {candidate_id!r}"
            )
        return cls(
            candidate_id=candidate_id.strip(),
            action_id=action_id.strip(),
            source_label=source_label.strip(),
            short_definition=short_definition.strip(),
            execution_binding=dict(binding),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
            "source_label": self.source_label,
            "short_definition": self.short_definition,
            "execution_binding": dict(self.execution_binding),
            **({"category_id": self.category_id} if self.category_id else {}),
        }



@dataclass(frozen=True, slots=True)
class SessionActionCategory:
    category_id: str
    source_label: str
    short_definition: str
    children: tuple[SessionActionCandidate, ...]

    @classmethod
    def from_payload(cls, value: Any) -> "SessionActionCategory":
        if not isinstance(value, dict):
            raise ValueError("action category must be an object")
        category_id = value.get("category_id")
        source_label = value.get("source_label") or category_id
        short_definition = value.get("short_definition") or source_label
        children = value.get("children")
        if not isinstance(category_id, str) or not category_id.strip():
            raise ValueError("category_id must be a non-empty string")
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError(f"category source_label must be non-empty: {category_id!r}")
        if not isinstance(short_definition, str) or not short_definition.strip():
            raise ValueError(f"category short_definition must be non-empty: {category_id!r}")
        if not isinstance(children, list) or not children:
            raise ValueError(f"category children must be a non-empty list: {category_id!r}")
        if len(children) > MAX_ACTION_CHILDREN_PER_CATEGORY:
            raise ValueError(f"category children must contain at most {MAX_ACTION_CHILDREN_PER_CATEGORY} items")
        parsed = []
        for child in children:
            item = SessionActionCandidate.from_payload(child)
            parsed.append(SessionActionCandidate(candidate_id=item.candidate_id, action_id=item.action_id, source_label=item.source_label, short_definition=item.short_definition, execution_binding=dict(item.execution_binding), category_id=category_id.strip()))
        return cls(category_id=category_id.strip(), source_label=source_label.strip(), short_definition=short_definition.strip(), children=tuple(parsed))

    def as_dict(self) -> dict[str, Any]:
        return {"category_id": self.category_id, "source_label": self.source_label, "short_definition": self.short_definition, "children": [item.as_dict() for item in self.children]}

@dataclass(slots=True)
class ImageFrame:
    seq: int
    timestamp_ms: int
    data_uri: str
    image_role: Literal["user_camera", "avatar_state"]


@dataclass(slots=True)
class ActionHistoryTurn:
    turn_id: str
    turn_origin: Literal["user", "proactive"]
    text_role: Literal["user_input", "character_reply"]
    messages: list[dict[str, Any]]
    audios: list[str]
    images: list[str]


@dataclass(slots=True)
class TurnBuffer:
    turn_id: str
    started_at: float
    audio: RealtimeAudioBuffer
    images: list[ImageFrame]
    audio_seqs: set[int]
    image_seqs: set[int]
    turn_origin: Literal["user", "proactive"]
    text_role: Literal["user_input", "character_reply"]
    trigger: str | None = None
    text: str | None = None
    avatar_state: dict[str, Any] | None = None
    audio_chunk_count: int = 0
    duplicate_audio_chunks: int = 0
    duplicate_image_frames: int = 0
    phase: Literal["collecting", "processing", "cancelling", "completed"] = (
        TURN_PHASE_COLLECTING
    )
    request_base: str | None = None
    current_request_id: str | None = None
    inference_task: asyncio.Task[None] | None = None


class MultimodalSession:
    """Manual-turn, multimodal session for audio chunks and image frames.

    This protocol deliberately lives next to, but separately from, the
    OpenAI-compatible /v1/realtime implementation. The latter remains
    audio/VAD compatible; this session owns explicit turn.start/commit
    boundaries and the fixed scheme-B action catalog.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        action_selection_mode: str | None = None,
        action_micro_batch_size: int | None = None,
        action_category_top_k: int | None = None,
        claim_session: Callable[[str, "MultimodalSession"], None],
        release_session: Callable[[str, "MultimodalSession"], None],
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(action_selection_mode)
        self.action_micro_batch_size = normalize_action_micro_batch_size(
            action_micro_batch_size
        )
        self.action_category_top_k = normalize_action_category_top_k(
            action_category_top_k
        )
        self.claim_session = claim_session
        self.release_session = release_session

        self.session_id: str | None = None
        self.language = "zh"
        self.instructions = DEFAULT_INSTRUCTIONS
        self.closed = False
        self.started = False
        self.active_turn: TurnBuffer | None = None
        self.used_turn_ids: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.history_audios: list[str] = []
        self.history_images: list[str] = []
        self.history_image_roles: list[str] = []
        self.history_turns: list[ActionHistoryTurn] = []
        self.candidates: list[SessionActionCandidate] = []
        self.categories: list[SessionActionCategory] = []
        self.candidate_by_id: dict[str, SessionActionCandidate] = {}
        self.action_system_prompt = ""
        self.action_catalog_hash = ""
        self.action_prefix_cache_namespace = ""
        self.action_prefix_prefilled = False
        self._prefilled_action_prefix_namespaces: set[str] = set()
        self.last_avatar_state: dict[str, Any] = {}
        self._send_lock = asyncio.Lock()
        # Detailed candidate scores are useful for diagnostics, but are not
        # needed by the action executor. Keep the production response small
        # unless the caller opts in at session.start.
        self.include_scores = False

    async def run(self) -> None:
        try:
            while not self.closed:
                try:
                    message = await self.websocket.receive()
                except WebSocketDisconnect:
                    logger.info(
                        "[SESSION_ACTION_REALTIME] client disconnected session_id=%s",
                        self.session_id,
                    )
                    break
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] != "websocket.receive":
                    continue
                if message.get("text") is None:
                    await self.send_error(
                        "invalid_request",
                        "binary_frames_not_supported",
                        "Use JSON events with base64 media payloads.",
                    )
                    continue
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError as exc:
                    await self.send_error(
                        "invalid_request", "invalid_json", str(exc)
                    )
                    continue
                if not isinstance(payload, dict):
                    await self.send_error(
                        "invalid_request",
                        "invalid_event",
                        "Top-level event must be a JSON object.",
                    )
                    continue
                try:
                    await self.dispatch(payload)
                except WebSocketDisconnect:
                    logger.info(
                        "[SESSION_ACTION_REALTIME] client disconnected during event "
                        "session_id=%s event=%s",
                        self.session_id,
                        payload.get("type"),
                    )
                    break
                except (BufferOverflow, ValueError, KeyError) as exc:
                    session_id = self._event_context_id(payload, "session_id") or self.session_id
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id if self.active_turn is not None else None
                    )
                    await self.send_error(
                        "invalid_request",
                        self._classify_error(payload, exc),
                        str(exc),
                        session_id=session_id,
                        turn_id=turn_id,
                    )
                except Exception as exc:
                    session_id = self._event_context_id(payload, "session_id") or self.session_id
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id if self.active_turn is not None else None
                    )
                    await self.send_error(
                        "server_error",
                        "server_error",
                        str(exc),
                        session_id=session_id,
                        turn_id=turn_id,
                    )
        finally:
            self.closed = True
            await self._cancel_active_turn(send_event=False)
            if self.session_id is not None:
                self.release_session(self.session_id, self)
            await self._close_websocket()

    async def _close_websocket(self) -> None:
        """Close only when both sides still allow a close frame.

        Starlette tracks peer state and application state separately. A failed
        server send can make application_state DISCONNECTED while client_state
        is still CONNECTED, so checking only client_state can attempt to send a
        second close frame and raise RuntimeError.
        """
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        try:
            await self.websocket.close()
        except (OSError, WebSocketDisconnect, RuntimeError):
            logger.debug(
                "[SESSION_ACTION_REALTIME] websocket already closed session_id=%s",
                self.session_id,
                exc_info=True,
            )

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        handlers = {
            "session.start": self.handle_session_start,
            "turn.start": self.handle_turn_start,
            "input_audio.append": self.handle_audio_append,
            "input_audio_buffer.append": self.handle_audio_append,
            "input_image.append": self.handle_image_append,
            "turn.text.update": self.handle_text_update,
            "turn.commit": self.handle_turn_commit,
            "turn.cancel": self.handle_turn_cancel,
            "session.close": self.handle_session_close,
        }
        handler = handlers.get(event_type)
        if handler is None:
            raise ValueError(f"unsupported event type: {event_type!r}")
        if event_type == "turn.commit":
            await self._dispatch_turn_commit(payload)
            return
        await handler(payload)

    async def _dispatch_turn_commit(self, event: dict[str, Any]) -> None:
        """Start committed-turn inference without blocking WebSocket input."""
        turn = self._require_collecting_turn(event)
        task = asyncio.create_task(
            self.handle_turn_commit(event),
            name=f"session-action-{self.session_id}-{turn.turn_id}",
        )
        turn.inference_task = task
        await asyncio.sleep(0)
        if task.done():
            await task

    async def handle_session_start(self, event: dict[str, Any]) -> None:
        if self.started:
            raise ValueError("session.start can only be sent once")
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        raw_candidates = event.get("action_candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("action_candidates must be a non-empty list")
        if len(raw_candidates) > MAX_ACTION_CANDIDATES:
            raise ValueError(f"action_candidates must contain at most {MAX_ACTION_CANDIDATES} items")

        nested = all(isinstance(x, dict) and "children" in x for x in raw_candidates)
        categories = [SessionActionCategory.from_payload(x) for x in raw_candidates] if nested else []
        if categories:
            if len(categories) > MAX_ACTION_CATEGORIES:
                raise ValueError(f"action categories must contain at most {MAX_ACTION_CATEGORIES} items")
            candidates = [child for category in categories for child in category.children]
            all_ids = [category.category_id for category in categories] + [x.candidate_id for x in candidates]
            if len(set(all_ids)) != len(all_ids):
                raise ValueError("action category and candidate IDs must be globally unique")
        else:
            candidates = [SessionActionCandidate.from_payload(x) for x in raw_candidates]
            candidate_ids = [x.candidate_id for x in candidates]
            if len(set(candidate_ids)) != len(candidate_ids):
                raise ValueError("action candidate IDs must be unique")
        if len(candidates) > MAX_ACTION_CANDIDATES:
            raise ValueError(f"action candidates must contain at most {MAX_ACTION_CANDIDATES} children")
        if not any(x.action_id == "no_action" for x in candidates):
            raise ValueError("action_candidates must include action_id=no_action")

        language = event.get("language", "zh")
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        instructions = event.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        include_scores = event.get("include_scores", False)
        if not isinstance(include_scores, bool):
            raise ValueError("include_scores must be a boolean")
        if event.get("input_audio_format", "pcm16") != "pcm16":
            raise ValueError("only pcm16 audio is supported")
        try:
            sample_rate = int(event.get("sample_rate", 16000))
            channels = int(event.get("channels", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_rate and channels must be integers") from exc
        if sample_rate != 16000:
            raise ValueError("only 16000 Hz audio is supported")
        if channels != 1:
            raise ValueError("only mono audio is supported")

        if "selection_mode" in event:
            requested_mode = event.get("selection_mode")
            selected_mode = try_normalize_action_selection_mode(requested_mode)
            if selected_mode is None:
                logger.warning(
                    "[SESSION_ACTION_REALTIME] invalid session.start selection_mode=%r "
                    "session_id=%s; using %s",
                    requested_mode,
                    session_id,
                    self.action_selection_mode,
                )
            else:
                self.action_selection_mode = selected_mode

        self.claim_session(session_id, self)
        self.session_id = session_id
        self.language = language
        if instructions is not None:
            self.instructions = instructions
        self.include_scores = include_scores
        self.candidates = candidates
        self.categories = categories
        self.candidate_by_id = {x.candidate_id: x for x in candidates}
        if categories and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL:
            self.action_system_prompt = self._build_category_system_prompt()
        else:
            self.action_system_prompt = self._build_action_system_prompt()
        canonical = json.dumps(
            ([category.as_dict() for category in categories] if categories else [x.as_dict() for x in candidates]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.action_catalog_hash = "sha256:" + hashlib.sha256(canonical).hexdigest()
        mode_namespace = (
            "hierarchical"
            if categories and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL
            else "flat_children"
        )
        self.action_prefix_cache_namespace = (
            f"{mode_namespace}:{self.action_catalog_hash}"
        )
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if callable(prefill):
            if categories and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL:
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.category_id,
                        suffix=item.category_id,
                        action_id=item.category_id,
                    )
                    for item in categories
                ]
                prefill_stage = "category"
            else:
                prefill_candidates = [
                    ActionScoreCandidate(
                        candidate_id=item.candidate_id,
                        suffix=item.candidate_id,
                        action_id=item.action_id,
                        execution_binding=dict(item.execution_binding),
                    )
                    for item in candidates
                ]
                prefill_stage = "single"
            self.action_prefix_prefilled = await prefill(
                model=self.model_name,
                system_prompt=self.action_system_prompt,
                candidates=prefill_candidates,
                prefix_cache_namespace=self.action_prefix_cache_namespace,
                stage=prefill_stage,
            )
            if self.action_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(
                    self.action_prefix_cache_namespace
                )
        self.started = True

        await self.send(
            {
                "type": "session.started",
                "session_id": self.session_id,
                "model": self.model_name,
                "action_catalog_hash": self.action_catalog_hash,
                "action_candidate_count": len(candidates),
                "action_category_count": len(categories),
                "action_selection_mode": self.action_selection_mode,
                "action_selection_stages": (
                    2 if categories and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL else 1
                ),
                "action_prefix_prefilled": self.action_prefix_prefilled,
            }
        )

    async def handle_turn_start(self, event: dict[str, Any]) -> None:
        self._require_started()
        if self.active_turn is not None:
            raise ValueError("another turn is already active")
        turn_id = event.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError(
                "turn_id must be generated by the caller and be a non-empty string"
            )
        turn_id = turn_id.strip()
        turn_origin, text_role, trigger = self._parse_turn_semantics(event)
        if turn_id in self.used_turn_ids:
            raise ValueError(f"turn_id has already been used: {turn_id}")
        self.used_turn_ids.add(turn_id)
        self.active_turn = TurnBuffer(
            turn_id=turn_id,
            started_at=time.perf_counter(),
            audio=RealtimeAudioBuffer(source_sr=16000, target_sr=16000),
            images=[],
            audio_seqs=set(),
            image_seqs=set(),
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
        )
        await self.send(
            {
                "type": "turn.started",
                "session_id": self.session_id,
                "turn_id": turn_id,
            }
        )

    async def handle_audio_append(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        seq = self._positive_int(event.get("seq"), "seq")
        if seq in turn.audio_seqs:
            turn.duplicate_audio_chunks += 1
            await self._ack_media(turn, "audio", seq, duplicate=True)
            return
        expected_seq = turn.audio_chunk_count + 1
        if seq != expected_seq:
            raise ValueError(
                f"audio seq must be monotonic starting at 1; expected {expected_seq}, got {seq}"
            )
        if turn.audio_chunk_count >= MAX_AUDIO_CHUNKS_PER_TURN:
            raise ValueError(
                f"audio chunk count exceeds {MAX_AUDIO_CHUNKS_PER_TURN}"
            )
        audio = event.get("audio")
        if not isinstance(audio, str) or not audio:
            raise ValueError("audio must be a non-empty base64 string")
        try:
            decoded = base64.b64decode(audio, validate=True)
        except Exception as exc:
            raise ValueError("audio is not valid base64") from exc
        if len(decoded) % 2:
            raise ValueError("PCM16 audio must contain an even number of bytes")
        turn.audio.append_b64(audio)
        turn.audio_seqs.add(seq)
        turn.audio_chunk_count += 1
        await self._ack_media(turn, "audio", seq)

    async def handle_image_append(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        seq = self._positive_int(event.get("seq"), "seq")
        image_role = event.get(
            "image_role", DEFAULT_IMAGE_ROLE_BY_ORIGIN[turn.turn_origin]
        )
        if not isinstance(image_role, str) or image_role not in IMAGE_ROLES:
            raise ValueError(
                "image_role must be 'user_camera' or 'avatar_state'"
            )
        if seq in turn.image_seqs:
            turn.duplicate_image_frames += 1
            stored_role = next(
                frame.image_role for frame in turn.images if frame.seq == seq
            )
            await self._ack_media(
                turn, "image", seq, duplicate=True, image_role=stored_role
            )
            return
        image = event.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError("image must be a non-empty base64 string or data URI")
        mime_type = event.get("mime_type", "image/jpeg")
        if mime_type not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("mime_type must be image/jpeg, image/png, or image/webp")
        if image.startswith("data:"):
            if ";base64," not in image:
                raise ValueError("image data URI must use base64 encoding")
            data_uri = image
            encoded = image.split(",", 1)[1]
        else:
            data_uri = f"data:{mime_type};base64,{image}"
            encoded = image
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("image is not valid base64") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
        timestamp_ms = self._nonnegative_int(
            event.get("timestamp_ms", 0), "timestamp_ms"
        )
        if len(turn.images) >= MAX_IMAGES_PER_TURN:
            raise ValueError(
                f"image frame count exceeds {MAX_IMAGES_PER_TURN}"
            )
        turn.images.append(
            ImageFrame(
                seq=seq,
                timestamp_ms=timestamp_ms,
                data_uri=data_uri,
                image_role=image_role,
            )
        )
        turn.image_seqs.add(seq)
        await self._ack_media(turn, "image", seq, image_role=image_role)

    async def handle_text_update(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        text = event.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError("text must be a string or null")
        turn.text = text
        await self.send(
            {
                "type": "turn.text.updated",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "text_present": bool(text),
            }
        )

    async def handle_turn_cancel(self, event: dict[str, Any]) -> None:
        turn = self._require_turn(event)
        await self._cancel_active_turn(send_event=True, expected_turn=turn)

    async def handle_session_close(self, event: dict[str, Any]) -> None:
        del event
        self.closed = True
        await self._cancel_active_turn(send_event=False)
        await self.send(
            {"type": "session.closed", "session_id": self.session_id}
        )


    async def _cancel_active_turn(
        self,
        *,
        send_event: bool,
        expected_turn: TurnBuffer | None = None,
    ) -> None:
        turn = self.active_turn
        if turn is None:
            return
        if expected_turn is not None and turn is not expected_turn:
            return

        cancel_started = time.perf_counter()
        if turn.phase == TURN_PHASE_PROCESSING:
            turn.phase = TURN_PHASE_CANCELLING
            request_id = turn.current_request_id
            abort = getattr(self.client, "abort", None)
            if request_id is not None and callable(abort):
                try:
                    await abort(request_id)
                except Exception:
                    logger.exception(
                        "[SESSION_ACTION_REALTIME] direct abort failed "
                        "session_id=%s turn_id=%s request_id=%s",
                        self.session_id,
                        turn.turn_id,
                        request_id,
                    )
            task = turn.inference_task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
            ):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        if self.active_turn is turn:
            self.active_turn = None
        turn.current_request_id = None
        turn.images.clear()
        turn.audio.clear()
        logger.info(
            "[SESSION_ACTION_REALTIME] turn cancelled session_id=%s turn_id=%s "
            "phase=%s elapsed_ms=%.3f",
            self.session_id,
            turn.turn_id,
            turn.phase,
            (time.perf_counter() - cancel_started) * 1000.0,
        )
        if send_event:
            await self.send(
                {
                    "type": "turn.cancelled",
                    "session_id": self.session_id,
                    "turn_id": turn.turn_id,
                }
            )

    async def handle_turn_commit(self, event: dict[str, Any]) -> None:
        turn = self._require_collecting_turn(event)
        turn_origin, text_role, trigger = self._parse_turn_semantics(event)
        if (turn_origin, text_role, trigger) != (
            turn.turn_origin,
            turn.text_role,
            turn.trigger,
        ):
            raise ValueError(
                "turn_origin, text_role, and trigger must match turn.start"
            )
        if turn_origin == TURN_ORIGIN_PROACTIVE and event.get("user_input") is not None:
            raise ValueError("user_input must be null or omitted for proactive turns")
        if "text" in event:
            text = event.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be a string or null")
            turn.text = text
        if "avatar_state" in event:
            state = event.get("avatar_state")
            if not isinstance(state, dict):
                raise ValueError("avatar_state must be an object")
            current_action_id = state.get("current_action_id")
            if current_action_id is not None and (
                not isinstance(current_action_id, str) or not current_action_id.strip()
            ):
                raise ValueError(
                    "avatar_state.current_action_id must be a non-empty string or null"
                )
            state_description = state.get("state_description")
            if state_description is not None and not isinstance(state_description, str):
                raise ValueError("avatar_state.state_description must be a string")
            turn.avatar_state = dict(state)
        current_audio = (
            turn.audio.to_full_wav_data_uri() if not turn.audio.is_empty() else None
        )
        current_image_frames = sorted(
            turn.images, key=lambda x: (x.timestamp_ms, x.seq)
        )
        current_images = [frame.data_uri for frame in current_image_frames]
        current_image_roles = [
            frame.image_role for frame in current_image_frames
        ]
        current_audio_list = [current_audio] if current_audio else []
        ingest_ms = (time.perf_counter() - turn.started_at) * 1000.0
        turn.phase = TURN_PHASE_PROCESSING
        turn.request_base = f"session-{self.session_id}-turn-{turn.turn_id}-action-{uuid.uuid4().hex}"

        turn_id = turn.turn_id

        try:
            await self.send(
                {
                    "type": "turn.committed",
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                    "image_roles": current_image_roles,
                }
            )
            if self.closed:
                await self._cancel_active_turn(send_event=False, expected_turn=turn)
                return
            commit_started = time.perf_counter()
            logger.info(
                "[SESSION_ACTION_REALTIME] turn.commit input session_id=%s turn_id=%s payload=%s",
                self.session_id,
                turn_id,
                json.dumps({
                    "text": turn.text,
                    "avatar_state": turn.avatar_state or self.last_avatar_state,
                    "turn_origin": turn.turn_origin,
                    "text_role": turn.text_role,
                    "trigger": turn.trigger,
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                    "image_roles": current_image_roles,
                    "audio": _summarize_media(current_audio_list),
                    "images": _summarize_media(current_images),
                    "history_turn_count": len(self.history_turns),
                    "candidate_count": len(self.candidates),
                    "action_catalog_hash": self.action_catalog_hash,
                }, ensure_ascii=False, default=str),
            )
            action_started = time.perf_counter()
            action, scores, action_timing, action_context = await self._score_action(
                current_audio_list, current_images, current_image_roles,
                turn.text, turn.avatar_state,
                turn_origin=turn.turn_origin, text_role=turn.text_role,
                trigger=turn.trigger, turn_id=turn_id, turn=turn,
                request_base=turn.request_base,
            )
            logger.info(
                "[SESSION_ACTION_REALTIME] action completed session_id=%s turn_id=%s elapsed_ms=%.3f top_action=%s",
                self.session_id, turn_id, (time.perf_counter() - action_started) * 1000.0, action.get("action_id"),
            )
            total_after_commit_ms = (time.perf_counter() - commit_started) * 1000.0
            if self.active_turn is not turn or turn.phase != TURN_PHASE_PROCESSING:
                return
            turn.phase = TURN_PHASE_COMPLETED
            if turn.avatar_state is not None:
                self.last_avatar_state = dict(turn.avatar_state)

            self._append_action_history(
                current_audio_list,
                current_images,
                current_image_roles,
                turn.text,
                turn_id=turn_id,
                turn_origin=turn.turn_origin,
                text_role=turn.text_role,
                action=action,
            )
            self.active_turn = None
            result = {
                "type": "turn.result",
                "session_id": self.session_id,
                "turn_id": turn_id,
                "action_catalog_hash": self.action_catalog_hash,
                "action": self._compact_action(action),
                "timing": {
                    "server_turn_ingest_ms": round(ingest_ms, 3),
                    "server_action_compute_ms": action_timing,
                    "server_total_after_commit_ms": round(
                        total_after_commit_ms, 3
                    ),
                },
            }
            if self.include_scores:
                result["scores"] = scores
                result["media_summary"] = {
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                    "user_camera_image_count": current_image_roles.count(
                        IMAGE_ROLE_USER_CAMERA
                    ),
                    "avatar_state_image_count": current_image_roles.count(
                        IMAGE_ROLE_AVATAR_STATE
                    ),
                    "received_image_count": len(turn.images),
                    "scored_image_count": action_context["scored_current_image_count"],
                    "text_present": bool(turn.text),
                    "duplicate_audio_chunks": turn.duplicate_audio_chunks,
                    "duplicate_image_frames": turn.duplicate_image_frames,
                    "action_context": action_context,
                }
            await self.send(result)
        except asyncio.CancelledError:
            logger.info(
                "[SESSION_ACTION_REALTIME] turn inference cancelled "
                "session_id=%s turn_id=%s request_id=%s",
                self.session_id, turn_id, turn.current_request_id,
            )
            raise
        except Exception as exc:
            if turn.phase == TURN_PHASE_CANCELLING:
                logger.warning(
                    "[SESSION_ACTION_REALTIME] cancelled turn cleanup failed "
                    "session_id=%s turn_id=%s",
                    self.session_id, turn_id, exc_info=True,
                )
                return
            logger.exception(
                "[SESSION_ACTION_REALTIME] turn failed session_id=%s turn_id=%s",
                self.session_id,
                turn_id,
            )
            turn.phase = TURN_PHASE_COMPLETED
            if self.active_turn is turn:
                self.active_turn = None
            message = str(exc)
            if "prefix selected-token logprobs are missing" in message:
                code = "action_score_logprob_unavailable"
            else:
                code = "action_score_failed"
            await self.send_error(
                "action_score_error",
                code,
                message,
                session_id=self.session_id,
                turn_id=turn_id,
            )

    def _build_bounded_action_context(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[str],
        list[str],
        list[str],
        dict[str, Any],
    ]:
        """Bound action-scoring media without changing normal chat history.

        History messages contain media placeholders, while the media arrays are
        flattened in the same order. This helper walks both together and drops
        whole old turns plus excess media placeholders atomically, so the
        preprocessor never sees a placeholder/media count mismatch.
        """
        if len(images) != len(image_roles):
            raise ValueError("images and image_roles must have the same length")
        if len(self.history_images) != len(self.history_image_roles):
            raise ValueError(
                "history_images and history_image_roles must have the same length"
            )
        # Proactive turns start with an assistant message, so role changes
        # cannot reliably identify turn boundaries.
        selected_turns = self.history_turns[-MAX_ACTION_HISTORY_TURNS:]
        selected_message_ids = {
            id(message) for turn in selected_turns for message in turn.messages
        }
        bounded_history: list[dict[str, Any]] = []
        bounded_history_audios: list[str] = []
        bounded_history_images: list[str] = []
        ignored_history_avatar_image_count = 0
        audio_index = 0
        image_index = 0

        for message in self.history:
            selected = id(message) in selected_message_ids
            content = message.get("content")
            if not selected:
                if isinstance(content, list):
                    audio_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "audio"
                    )
                    image_index += sum(
                        1
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "image"
                    )
                continue

            if not isinstance(content, list):
                bounded_history.append(dict(message))
                continue

            bounded_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    bounded_parts.append(part)
                    continue
                part_type = part.get("type")
                if part_type == "audio":
                    if audio_index < len(self.history_audios):
                        media = self.history_audios[audio_index]
                        if len(bounded_history_audios) < MAX_ACTION_HISTORY_AUDIOS:
                            bounded_history_audios.append(media)
                            bounded_parts.append({"type": "audio"})
                    audio_index += 1
                elif part_type == "image":
                    if image_index < len(self.history_images):
                        media = self.history_images[image_index]
                        image_role = self.history_image_roles[image_index]
                        if image_role == IMAGE_ROLE_AVATAR_STATE:
                            # A previous turn's avatar frame may be visually
                            # unrelated to the current rendered pose. Never
                            # expose it as evidence of current avatar state.
                            ignored_history_avatar_image_count += 1
                        elif len(bounded_history_images) < MAX_ACTION_HISTORY_IMAGES:
                            bounded_history_images.append(media)
                            bounded_parts.append(
                                self._image_role_text_part(image_role)
                            )
                            bounded_parts.append({"type": "image"})
                    image_index += 1
                else:
                    bounded_parts.append(dict(part))
            if not bounded_parts:
                bounded_parts = [
                    {"type": "text", "text": "（历史多媒体内容已裁剪）"}
                ]
            bounded_history.append(
                {**message, "content": bounded_parts}
            )

        bounded_images = images[-MAX_ACTION_CURRENT_IMAGES:]
        bounded_image_roles = image_roles[-MAX_ACTION_CURRENT_IMAGES:]
        bounded_audios = list(audios)
        truncated = (
            len(self.history_turns) > len(selected_turns)
            or len(self.history_audios) != len(bounded_history_audios)
            or len(self.history_images) != len(bounded_history_images)
            or len(images) != len(bounded_images)
        )
        context_summary = {
            "history_turn_count": len(selected_turns),
            "history_audio_count": len(bounded_history_audios),
            "history_image_count": len(bounded_history_images),
            "ignored_history_avatar_image_count": (
                ignored_history_avatar_image_count
            ),
            "received_current_image_count": len(images),
            "scored_current_image_count": len(bounded_images),
            "truncated": truncated,
        }
        return (
            bounded_history,
            bounded_history_audios,
            bounded_history_images,
            bounded_images,
            bounded_image_roles,
            context_summary,
        )

    @staticmethod
    def _image_role_label(image_role: str) -> str:
        if image_role == IMAGE_ROLE_USER_CAMERA:
            return "用户摄像头画面（用于观察用户及其环境）"
        return "数字人当前状态画面（用于观察数字人自身姿态）"

    @classmethod
    def _image_role_text_part(cls, image_role: str) -> dict[str, str]:
        return {
            "type": "text",
            "text": f"[image_role] {cls._image_role_label(image_role)}：",
        }

    @classmethod
    def _build_current_image_role_instruction(
        cls, image_roles: list[str]
    ) -> str:
        if not image_roles:
            return (
                "本轮未提供数字人最新状态图片。数字人当前视觉姿态未知；"
                "不得根据历史图片推断，也不得把用户摄像头画面当成数字人状态。"
                "如果动作选择依赖数字人当前视觉姿态，应选择 no_action。\n"
            )
        mapping = "；".join(
            f"图片{index}={cls._image_role_label(role)}"
            for index, role in enumerate(image_roles, start=1)
        )
        instruction = (
            f"当前图片角色（按模型接收顺序）：{mapping}。"
            "用户摄像头图片描述用户及其环境；数字人状态图片描述数字人自身。"
            "不得混淆两类图片中的人物、姿态或动作。\n"
        )
        if IMAGE_ROLE_AVATAR_STATE not in image_roles:
            instruction += (
                "本轮未提供数字人最新状态图片。数字人当前视觉姿态未知；"
                "不得根据历史图片或当前用户摄像头图片推断。"
                "如果动作选择依赖数字人当前视觉姿态，应选择 no_action。\n"
            )
        else:
            instruction += (
                "只有本轮标记为 avatar_state 的图片可用于判断数字人当前视觉姿态；"
                "不得使用历史数字人图片推断当前状态。\n"
            )
        return instruction

    @staticmethod
    def _with_current_proactive_text(
        history: list[dict[str, Any]], text: str | None, turn_origin: str
    ) -> list[dict[str, Any]]:
        if (
            turn_origin != TURN_ORIGIN_PROACTIVE
            or not isinstance(text, str)
            or not text.strip()
        ):
            return history
        return [*history, {"role": "assistant", "content": text.strip()}]

    @staticmethod
    def _build_turn_action_instruction(
        text: str | None,
        *,
        turn_origin: str,
        trigger: str | None,
        image_roles: list[str] | None = None,
    ) -> str:
        image_instruction = MultimodalSession._build_current_image_role_instruction(
            image_roles or []
        )
        if turn_origin == TURN_ORIGIN_PROACTIVE:
            trigger_text = (
                f"本轮主动触发原因：{trigger}。\n" if trigger is not None else ""
            )
            if not isinstance(text, str) or not text.strip():
                return (
                    image_instruction
                    + "本轮来源是 proactive，但未提供本轮待播文本。不要把历史中的 "
                    "assistant 消息当成本轮待播文本。\n"
                    + trigger_text
                    + "本轮动作决策不依赖当前语言文本。请仅根据 avatar_state、"
                    "触发原因、历史对话、历史动作和当前媒体选择最合适的动作；"
                    "动作必须满足 avatar_state.state_description 描述的场景目标、"
                    "动作要求和禁止项。avatar_state.current_action_id 表示当前或刚结束"
                    "的动作，后续动作必须与它自然衔接；除非 state_description 明确"
                    "要求重复，否则不要再次选择同一动作。没有满足这些约束的动作时"
                    "选择 no_action，不要生成新的回复。"
                )
            return (
                image_instruction
                + "本轮来源是 proactive。上一条 assistant 消息是数字人已经准备好、"
                "即将播放的文本，不是用户输入或用户动作请求。\n"
                + trigger_text
                + "上一条 assistant 消息与 avatar_state 是动作选择的共同核心约束。"
                "动作必须与待播文本的语义、语气和表达目标直接相关，同时满足 "
                "avatar_state.state_description 描述的场景目标、动作要求和禁止项；"
                "与任一约束冲突的候选动作不可选择。avatar_state.current_action_id "
                "表示当前或刚结束的动作，后续动作必须与它自然衔接；除非 "
                "state_description 明确要求重复，否则不要再次选择同一动作。\n"
                "请结合历史对话选择最合适的伴随动作，不要生成新的回复。即使待播"
                "文本没有直接要求动作，只要候选动作与其表达意图相关且满足数字人状态"
                "约束，也应选择它；没有同时满足文本和状态约束的动作时选择 no_action。"
            )
        text_prefix = (
            f"当前用户文本：{text.strip()}\n"
            if isinstance(text, str) and text.strip()
            else ""
        )
        return (
            image_instruction
            + "本轮来源是 user。当前文本和音频是用户输入；当前图片的来源以图片角色标注为准。\n"
            + text_prefix
            + "请理解用户意图，并根据历史对话和数字人状态选择合适的回应动作。"
            "如果用户没有动作意图且上下文中也没有合适动作，则选择 no_action。"
        )

    @staticmethod
    def _ensure_turn_processing(turn: TurnBuffer) -> None:
        if turn.phase != TURN_PHASE_PROCESSING:
            raise asyncio.CancelledError

    async def _score_action_request(
        self,
        turn: TurnBuffer,
        request: ActionSuffixScoreRequest,
    ) -> Any:
        self._ensure_turn_processing(turn)
        turn.current_request_id = request.request_id
        try:
            result = await self.client.score_action_suffixes(request)
        finally:
            if turn.current_request_id == request.request_id:
                turn.current_request_id = None
        self._ensure_turn_processing(turn)
        return result

    async def _score_action(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        if self.categories and self.action_selection_mode == ACTION_SELECTION_MODE_HIERARCHICAL:
            return await self._score_action_hierarchical(
                audios, images, image_roles, text, avatar_state,
                turn_origin=turn_origin, text_role=text_role,
                trigger=trigger, turn_id=turn_id, turn=turn,
                request_base=request_base,
            )
        return await self._score_action_flat(
            audios, images, image_roles, text, avatar_state,
            turn_origin=turn_origin, text_role=text_role,
            trigger=trigger, turn_id=turn_id, turn=turn,
            request_base=request_base,
        )

    async def _score_action_hierarchical(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history, action_history_audios, action_history_images,
            action_images, action_image_roles, action_context,
        ) = self._build_bounded_action_context(audios, images, image_roles)
        action_history = self._with_current_proactive_text(
            action_history, text, turn_origin
        )
        base = self._build_turn_action_instruction(
            text, turn_origin=turn_origin, trigger=trigger,
            image_roles=action_image_roles,
        )
        common = dict(
            model=self.model_name, language=self.language, audios=audios,
            images=action_images, sample_rate=16000,
            session_id=self.session_id, history=action_history,
            stage="category", logical_request_id=request_base,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            action_context_cache_key=request_base,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            history_audios=action_history_audios, history_images=action_history_images,
            avatar_state=dict(avatar_state or self.last_avatar_state),
        )
        category_request = ActionSuffixScoreRequest(
            request_id=request_base + "-category", prefix=base +
            "先选择最合适的动作类别。只输出 category_id，不要解释。下一步 category_id 是：",
            system_prompt=self._build_category_system_prompt(),
            candidates=[ActionScoreCandidate(
                candidate_id=item.category_id, suffix=item.category_id, action_id=item.category_id
            ) for item in self.categories],
            suffix_tokenization_mode="short_id",
            micro_batch_size=self.action_micro_batch_size,
            **common,
        )
        started = time.perf_counter()
        category_started = time.perf_counter()
        category_result = await self._score_action_request(turn, category_request)
        category_ms = round((time.perf_counter() - category_started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=category candidates=%d "
            "elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(category_request.candidates),
            category_ms,
            category_result.prefix_cached,
            json.dumps(category_result.stats, ensure_ascii=False, default=str),
        )
        category_by_id = {item.category_id: item for item in self.categories}
        category_ranked = sorted(category_result.scores, key=lambda item: item.mean_logprob, reverse=True)
        if not category_ranked or category_ranked[0].candidate_id not in category_by_id:
            raise ValueError("category action score did not return a valid category")
        selected_categories = [
            category_by_id[item.candidate_id]
            for item in category_ranked[: self.action_category_top_k]
            if item.candidate_id in category_by_id
        ]
        if not selected_categories:
            raise ValueError("category action score did not select a valid category")
        selected_category = selected_categories[0]
        selected_category_ids = [item.category_id for item in selected_categories]
        child_candidates = [
            child
            for category in selected_categories
            for child in category.children
        ]
        child_namespace = ",".join(selected_category_ids)
        if len(selected_categories) == 1:
            child_prefix = (
                f"已选择动作类别 {selected_category.category_id}。请只在该类别的子动作中选择一个。"
            )
            child_system_prompt = self._build_child_system_prompt(selected_category)
        else:
            child_prefix = (
                f"已选择动作类别候选 {', '.join(selected_category_ids)}。"
                "请只在这些类别的子动作中选择一个。"
            )
            child_system_prompt = self._build_child_system_prompt(selected_categories)
        child_namespace = (
            f"{self.action_prefix_cache_namespace}:child:{child_namespace}"
        )

        # Child catalogs depend on the category result, so they cannot be
        # prefetched at session.start. Warm the selected child catalog lazily
        # on its first use; subsequent turns reuse the same immutable prefix.
        child_prefix_prefilled = False
        prefill = getattr(self.client, "prefill_action_catalog", None)
        if callable(prefill) and child_namespace not in self._prefilled_action_prefix_namespaces:
            prefill_request_id = request_base + "-child-prefill"
            self._ensure_turn_processing(turn)
            turn.current_request_id = prefill_request_id
            try:
                child_prefix_prefilled = await prefill(
                    request_id=prefill_request_id,
                    model=self.model_name,
                    system_prompt=child_system_prompt,
                    candidates=[
                        ActionScoreCandidate(
                            candidate_id=item.candidate_id,
                            suffix=item.candidate_id,
                            action_id=item.action_id,
                            execution_binding=dict(item.execution_binding),
                        )
                        for item in child_candidates
                    ],
                    prefix_cache_namespace=child_namespace,
                    stage="child",
                )
            finally:
                if turn.current_request_id == prefill_request_id:
                    turn.current_request_id = None
            self._ensure_turn_processing(turn)
            if child_prefix_prefilled:
                self._prefilled_action_prefix_namespaces.add(child_namespace)

        action_common = {
            **common,
            "stage": "child",
            "micro_batch_size": self.action_micro_batch_size,
            "prefix_cache_namespace": child_namespace,
        }
        action_request = ActionSuffixScoreRequest(
            request_id=request_base + "-child", prefix=base + child_prefix
            + "只输出 action_id，不要解释。没有合适的伴随或回应动作时选择 no_action。下一步 action_id 是：",
            system_prompt=child_system_prompt,
            candidates=[ActionScoreCandidate(
                candidate_id=item.candidate_id, suffix=item.candidate_id,
                action_id=item.action_id, execution_binding=dict(item.execution_binding)
            ) for item in child_candidates],
            suffix_tokenization_mode="short_id",
            **action_common,
        )
        child_started = time.perf_counter()
        child_result = await self._score_action_request(turn, action_request)
        child_ms = round((time.perf_counter() - child_started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=child candidates=%d "
            "selected_category_ids=%s elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(action_request.candidates),
            ",".join(selected_category_ids),
            child_ms,
            child_result.prefix_cached,
            json.dumps(child_result.stats, ensure_ascii=False, default=str),
        )
        child_by_id = {item.candidate_id: item for item in child_candidates}
        ranked = sorted(child_result.scores, key=lambda item: item.mean_logprob, reverse=True)
        if not ranked or ranked[0].candidate_id not in child_by_id:
            raise ValueError("child action score did not return a valid candidate")

        def score_dict(score: Any, candidate: SessionActionCandidate) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id, "action_id": candidate.action_id,
                "category_id": candidate.category_id, "source_label": candidate.source_label,
                "short_definition": candidate.short_definition,
                "execution_binding": dict(candidate.execution_binding),
                "token_count": score.token_count,
                "mean_logprob": score.mean_logprob, "mean_nll": score.mean_nll, "ppl": score.ppl,
                "token_scores": [{"token_id": item.token_id, "logprob": item.logprob} for item in score.token_scores],
            }

        scores = [score_dict(score, child_by_id[score.candidate_id]) for score in ranked]
        top = scores[0]
        action = {
            "candidate_id": top["candidate_id"], "action_id": top["action_id"],
            "category_id": selected_category.category_id,
            "execution_binding": dict(top.get("execution_binding") or {}),
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"], "ppl": top["ppl"], "token_count": top["token_count"],
        }
        def compact_stage_score(score: Any) -> dict[str, Any]:
            return {"candidate_id": score.candidate_id, "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob, "mean_nll": score.mean_nll, "ppl": score.ppl,
                    "token_scores": [{"token_id": item.token_id, "logprob": item.logprob} for item in score.token_scores]}
        action_context.update({
            "selection_stages": 2,
            "selection_mode": ACTION_SELECTION_MODE_HIERARCHICAL,
            "logical_request_id": request_base,
            "selected_category_id": selected_category.category_id,
            "selected_category_ids": selected_category_ids,
            "category_top_k": self.action_category_top_k,
            "category_scores": [compact_stage_score(score) for score in category_ranked],
            "category_compute_ms": category_ms,
            "child_compute_ms": child_ms,
            "child_prefix_prefilled": child_prefix_prefilled,
            "child_prefix_cache_namespace": child_namespace,
        })
        return action, scores, round((time.perf_counter() - started) * 1000.0, 3), action_context

    async def _score_action_flat(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        trigger: str | None,
        turn: TurnBuffer,
        request_base: str,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_image_roles,
            action_context,
        ) = self._build_bounded_action_context(audios, images, image_roles)
        action_history = self._with_current_proactive_text(
            action_history, text, turn_origin
        )
        prefix = (
            self._build_turn_action_instruction(
                text,
                turn_origin=turn_origin,
                trigger=trigger,
                image_roles=action_image_roles,
            )
            + "请从固定候选集合中选择唯一一个最合适的动作。"
            "下一步 action_id 是："
        )
        candidates = [
            ActionScoreCandidate(
                candidate_id=item.candidate_id,
                suffix=item.candidate_id,
                action_id=item.action_id,
                execution_binding=dict(item.execution_binding),
            )
            for item in self.candidates
        ]
        request = ActionSuffixScoreRequest(
            request_id=request_base + "-single",
            model=self.model_name,
            prefix=prefix,
            system_prompt=self.action_system_prompt,
            language=self.language,
            candidates=candidates,
            suffix_tokenization_mode="short_id",
            audios=audios,
            images=action_images,
            sample_rate=16000,
            micro_batch_size=self.action_micro_batch_size,
            prefix_cache_namespace=self.action_prefix_cache_namespace,
            session_id=self.session_id,
            turn_origin=turn_origin,
            text_role=text_role,
            trigger=trigger,
            history=action_history,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=dict(avatar_state or self.last_avatar_state),
        )
        started = time.perf_counter()
        result = await self._score_action_request(turn, request)
        compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info(
            "[SESSION_ACTION_REALTIME] action stage completed "
            "session_id=%s turn_id=%s stage=flat candidates=%d "
            "elapsed_ms=%.3f prefix_cached=%s stats=%s",
            self.session_id,
            turn_id,
            len(request.candidates),
            compute_ms,
            result.prefix_cached,
            json.dumps(result.stats, ensure_ascii=False, default=str),
        )
        ranked = sorted(result.scores, key=lambda x: x.mean_logprob, reverse=True)
        scores: list[dict[str, Any]] = []
        for score in ranked:
            candidate = self.candidate_by_id[score.candidate_id]
            scores.append(
                {
                    "candidate_id": score.candidate_id,
                    "action_id": candidate.action_id,
                    **({"category_id": candidate.category_id} if candidate.category_id else {}),
                    "source_label": candidate.source_label,
                    "short_definition": candidate.short_definition,
                    "execution_binding": dict(candidate.execution_binding),
                    "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob,
                    "mean_nll": score.mean_nll,
                    "ppl": score.ppl,
                    "token_scores": [
                        {"token_id": item.token_id, "logprob": item.logprob}
                        for item in score.token_scores
                    ],
                }
            )
        top = scores[0]
        action = {
            "candidate_id": top["candidate_id"],
            "action_id": top["action_id"],
            **({"category_id": top["category_id"]} if top.get("category_id") else {}),
            "execution_binding": dict(top.get("execution_binding") or {}),
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"],
            "ppl": top["ppl"],
            "token_count": top["token_count"],
        }
        action_context.update(
            {
                "selection_stages": 1,
                "selection_mode": (
                    self.action_selection_mode if self.categories else "flat"
                ),
                "flattened_child_count": (
                    len(self.candidates) if self.categories else None
                ),
                "compute_ms": compute_ms,
            }
        )
        return action, scores, compute_ms, action_context

    @staticmethod
    def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
        """Return only fields required by the external action executor."""
        compact = {
            "action_id": action["action_id"],
            "candidate_id": action["candidate_id"],
        }
        if action.get("category_id"):
            compact["category_id"] = action["category_id"]
        compact["execute"] = action["execute"]
        execution_binding = action.get("execution_binding")
        if execution_binding:
            compact["execution_binding"] = dict(execution_binding)
        return compact

    def _append_action_history(
        self,
        audios: list[str],
        images: list[str],
        image_roles: list[str],
        text: str | None,
        *,
        turn_id: str,
        turn_origin: Literal["user", "proactive"],
        text_role: Literal["user_input", "character_reply"],
        action: dict[str, Any],
    ) -> None:
        candidate_id = str(action["candidate_id"])
        candidate = self.candidate_by_id[candidate_id]
        action_id = str(action["action_id"])
        if action_id == "no_action":
            status = "本轮数字人未执行动作"
        else:
            status = "本轮数字人已执行动作"
        action_state = (
            "[action_state] "
            f"turn_id={turn_id}；candidate_id={candidate.candidate_id}；"
            f"action_id={candidate.action_id}；"
            f"动作名称={candidate.source_label}；"
            f"动作描述={candidate.short_definition}；"
            f"状态={status}。"
        )

        if turn_origin == TURN_ORIGIN_USER:
            messages = [
                {
                    "role": "user",
                    "content": self._current_user_content(audios, images, text),
                },
                {"role": "assistant", "content": action_state},
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": self._current_character_content(
                        audios, images, text, action_state
                    ),
                }
            ]

        history_turn = ActionHistoryTurn(
            turn_id=turn_id,
            turn_origin=turn_origin,
            text_role=text_role,
            messages=messages,
            audios=list(audios),
            images=list(images),
        )
        self.history_turns.append(history_turn)
        self.history.extend(messages)
        self.history_audios.extend(audios)
        self.history_images.extend(images)
        self.history_image_roles.extend(image_roles)

    def _current_user_content(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
    ) -> Any:
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        if not parts:
            parts.append(
                {
                    "type": "text",
                    "text": "本轮没有文本输入，请根据当前会话内容选择动作。",
                }
            )
        return parts

    @staticmethod
    def _current_character_content(
        audios: list[str],
        images: list[str],
        text: str | None,
        action_state: str,
    ) -> Any:
        normalized_text = (
            text.strip() if isinstance(text, str) and text.strip() else None
        )
        if not audios and not images:
            return (
                f"{normalized_text}\n{action_state}"
                if normalized_text is not None
                else action_state
            )
        parts: list[dict[str, Any]] = []
        parts.extend({"type": "audio"} for _ in audios)
        parts.extend({"type": "image"} for _ in images)
        if normalized_text is not None:
            parts.append({"type": "text", "text": normalized_text})
        parts.append({"type": "text", "text": action_state})
        return parts

    def _no_action_candidate_id(self) -> str:
        for candidate in self.candidates:
            if candidate.action_id == "no_action":
                return candidate.candidate_id
        raise ValueError("session has no no_action candidate")

    def _build_category_system_prompt(self) -> str:
        lines = [
            "你是数字人动作类别识别器。",
            "以下类别集合在整个 session 内固定。只能输出一个 category_id。",
            ACTION_HISTORY_INSTRUCTION,
        ]
        # Category descriptions are opaque external metadata. Do not compress,
        # deduplicate, or rewrite them here; callers may optimize their wording
        # before session.start and the exact rendered catalog participates in
        # the catalog hash/prefix-cache identity.
        for item in self.categories:
            lines.append(f"{item.category_id}={item.source_label}；{item.short_definition}")
        no_action_categories = [item.category_id for item in self.categories if any(child.action_id == "no_action" for child in item.children)]
        if no_action_categories:
            lines.append("没有合适的伴随或回应动作时，选择包含 no_action 的类别：" + ", ".join(no_action_categories))
        lines.append("只输出 category_id，不要解释。")
        return "\n".join(lines)

    def _build_child_system_prompt(
        self, category: SessionActionCategory | list[SessionActionCategory]
    ) -> str:
        categories = category if isinstance(category, list) else [category]
        if len(categories) == 1:
            selected = categories[0]
            lines = [
                "你是数字人动作识别器。",
                f"当前已选动作类别：{selected.category_id}={selected.source_label}；{selected.short_definition}",
                "以下是该类别内固定的子动作集合，只能输出一个 action_id。",
                ACTION_HISTORY_INSTRUCTION,
            ]
        else:
            lines = [
                "你是数字人动作识别器。",
                "以下是候选动作类别及其固定子动作集合，只能输出一个 action_id。",
                ACTION_HISTORY_INSTRUCTION,
            ]
            for selected in categories:
                lines.append(
                    f"候选动作类别：{selected.category_id}={selected.source_label}；"
                    f"{selected.short_definition}"
                )
        for selected in categories:
            for item in selected.children:
                lines.append(
                    f"{item.candidate_id}={item.action_id}："
                    f"{item.source_label}；{item.short_definition}"
                )
        lines.extend(["没有合适的伴随或回应动作、状态冲突或需要避免重复时选择 no_action。", "只输出 action_id，不要解释。"])
        return "\n".join(lines)

    def _build_action_system_prompt(self) -> str:
        lines = [
            "你是数字人动作识别器。",
            "以下是本 session 固定的动作候选集合。",
            "候选 ID 是模型唯一允许输出的短 ID，必须从列表中选择一个。",
            ACTION_HISTORY_INSTRUCTION,
        ]
        for item in self.candidates:
            lines.append(
                f"{item.candidate_id}={item.action_id}："
                f"{item.source_label}；{item.short_definition}"
            )
        lines.extend(
            [
                f"没有合适的伴随或回应动作、状态冲突或需要避免重复时，选择 {self._no_action_candidate_id()} 对应的 no_action。",
                "只输出候选 ID，不要解释。",
            ]
        )
        return "\n".join(lines)

    async def _ack_media(
        self,
        turn: TurnBuffer,
        media_type: str,
        seq: int,
        duplicate: bool = False,
        image_role: str | None = None,
    ) -> None:
        payload = {
            "type": "input.ack",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "media_type": media_type,
            "seq": seq,
            "duplicate": duplicate,
        }
        if image_role is not None:
            payload["image_role"] = image_role
        await self.send(payload)

    def _require_started(self) -> None:
        if not self.started or self.session_id is None:
            raise ValueError("session.start must be sent first")

    @staticmethod
    def _parse_turn_semantics(
        event: dict[str, Any],
    ) -> tuple[
        Literal["user", "proactive"],
        Literal["user_input", "character_reply"],
        str | None,
    ]:
        turn_origin = event.get("turn_origin")
        text_role = event.get("text_role")
        expected_text_role = TURN_TEXT_ROLE_BY_ORIGIN.get(turn_origin)
        if expected_text_role is None:
            raise ValueError("turn_origin must be 'user' or 'proactive'")
        if text_role != expected_text_role:
            raise ValueError(
                f"text_role must be {expected_text_role!r} when "
                f"turn_origin is {turn_origin!r}"
            )
        trigger = event.get("trigger")
        if trigger is not None:
            if not isinstance(trigger, str) or not trigger.strip():
                raise ValueError("trigger must be a non-empty string or null")
            trigger = trigger.strip()
        if turn_origin == TURN_ORIGIN_USER and trigger is not None:
            raise ValueError("trigger is only supported for proactive turns")
        return turn_origin, text_role, trigger

    def _require_turn(self, event: dict[str, Any]) -> TurnBuffer:
        self._require_started()
        if self.active_turn is None:
            raise ValueError(
                "turn already committed or no active turn exists; send turn.start first"
            )
        turn_id = event.get("turn_id")
        if turn_id != self.active_turn.turn_id:
            raise ValueError("turn_id does not match the active turn")
        return self.active_turn

    def _require_collecting_turn(self, event: dict[str, Any]) -> TurnBuffer:
        turn = self._require_turn(event)
        if turn.phase != TURN_PHASE_COLLECTING:
            raise ValueError("turn already committed")
        return turn


    @staticmethod
    def _event_context_id(payload: dict[str, Any], field: str) -> str | None:
        value = payload.get(field)
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _classify_error(payload: dict[str, Any], exc: Exception) -> str:
        message = str(exc).lower()
        event_type = payload.get("type")
        if (
            "turn_origin" in message
            or "text_role" in message
            or "proactive turn" in message
            or "user_input" in message
        ):
            return "invalid_turn_semantics"
        if "session_id is already active" in message:
            return "duplicate_session_id"
        if "another turn is already active" in message:
            return "duplicate_active_turn"
        if "turn already committed" in message:
            return "turn_already_committed"
        if "turn_id" in message:
            return "invalid_turn_id"
        if "seq" in message or "sequence" in message:
            return "invalid_sequence"
        if "audio" in message or "pcm16" in message:
            return "invalid_audio"
        if "image" in message or "mime_type" in message:
            return "invalid_image"
        if "session.start can only" in message or event_type == "session.start":
            return "session_candidate_invalid"
        if "action_candidates" in message or "candidate" in message:
            return "session_candidate_invalid"
        if event_type in {
            "turn.commit",
            "input_audio.append",
            "input_audio_buffer.append",
            "input_image.append",
            "turn.cancel",
        } and "turn.start" in message:
            return "turn_already_committed"
        if "session.start" in message:
            return "session_not_started"
        return "invalid_event"

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._send_unlocked(payload)

    async def _send_unlocked(self, payload: dict[str, Any]) -> None:
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        try:
            await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except (OSError, WebSocketDisconnect):
            self.closed = True
            logger.info(
                "[SESSION_ACTION_REALTIME] client disconnected during send "
                "session_id=%s event=%s",
                self.session_id,
                payload.get("type"),
            )
        except RuntimeError as exc:
            if "close message has been sent" not in str(exc):
                raise
            self.closed = True
            logger.info(
                "[SESSION_ACTION_REALTIME] websocket already closed during send "
                "session_id=%s event=%s",
                self.session_id,
                payload.get("type"),
            )

    async def send_error(
        self,
        type_: str,
        code: str,
        message: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "error",
            "error": {"type": type_, "code": code, "message": message},
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if turn_id is not None:
            payload["turn_id"] = turn_id
        await self.send(payload)


class MultimodalSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        action_selection_mode: str | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(action_selection_mode)
        self.action_micro_batch_size = normalize_action_micro_batch_size()
        self.action_category_top_k = normalize_action_category_top_k()
        logger.info(
            "[SESSION_ACTION_REALTIME] action_selection_mode=%s",
            self.action_selection_mode,
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] action_micro_batch_size=%s",
            self.action_micro_batch_size,
        )
        logger.info(
            "[SESSION_ACTION_REALTIME] action_category_top_k=%s",
            self.action_category_top_k,
        )
        self.sessions: dict[str, MultimodalSession] = {}

    def create(self, websocket: WebSocket) -> MultimodalSession:
        return MultimodalSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
            action_selection_mode=self.action_selection_mode,
            action_micro_batch_size=self.action_micro_batch_size,
            action_category_top_k=self.action_category_top_k,
            claim_session=self.claim,
            release_session=self.release,
        )

    def claim(self, session_id: str, session: MultimodalSession) -> None:
        if session_id in self.sessions:
            raise ValueError(f"session_id is already active: {session_id}")
        self.sessions[session_id] = session

    def release(self, session_id: str, session: MultimodalSession) -> None:
        if self.sessions.get(session_id) is session:
            del self.sessions[session_id]

    def active_sessions(self) -> list[str]:
        return list(self.sessions)
