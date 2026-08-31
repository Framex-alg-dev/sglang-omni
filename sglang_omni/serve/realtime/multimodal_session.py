"""Multimodal realtime Session lifecycle and top-level orchestration."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.qwen3_omni.action_scoring import (
    MAX_MICRO_BATCH_SIZE,
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    CATEGORY_SEMANTIC_TAG_REPLY_ACCOMPANIMENT,
    CATEGORY_SEMANTIC_TAG_SILENT_ACCOMPANIMENT,
    CATEGORY_CONTEXT_POLICY,
    CATEGORY_CONTEXT_POLICY_EN,
    DEFAULT_ACTION_PROMPT_LOCALE,
    DIRECTION_REFERENCE_POLICY,
    DIRECTION_REFERENCE_POLICY_EN,
    UNSUPPORTED_CATEGORY_SCORE_ID,
    UNSUPPORTED_CHILD_SCORE_ID,
    UNSUPPORTED_DECISION_ID,
    GlobalActionCatalog,
    GlobalActionCatalogPrewarmStatus,
    child_unsupported_policy,
)
from sglang_omni.models.qwen3_omni.prompt_localization import (
    PROMPT_LANGUAGE_BY_LOCALE,
    localized_prompt,
)
from sglang_omni.preprocessing.image import prepare_image_bytes_for_wire
from sglang_omni.serve.realtime.audio_buffer import BufferOverflow, RealtimeAudioBuffer
from sglang_omni.serve.realtime.embedded_tts import (
    EmbeddedTTSConfig,
    EmbeddedTTSConnection,
)
from sglang_omni.serve.realtime.output_capabilities import (
    DEFAULT_OUTPUTS,
    SessionOutputCapabilities,
)
from sglang_omni.serve.realtime.components import compose_components
from sglang_omni.serve.realtime.memory import (
    SessionMemoryConfig,
    SessionMemoryScheduler,
    SessionMemoryStore,
    SessionMemoryTurn,
    build_memory_extraction_request,
    parse_memory_extraction,
)
from sglang_omni.utils.structured_logs import (
    get_structured_log_writer,
    new_trace_id,
)

if TYPE_CHECKING:
    from sglang_omni.client.client import Client

logger = logging.getLogger(__name__)


def emit_structured_log(log_type: str, event: str, **fields: Any) -> bool:
    """Route logs through the public facade so existing hooks keep working."""
    from sglang_omni.serve.realtime import multimodal as facade

    return facade.emit_structured_log(log_type, event, **fields)


from sglang_omni.serve.realtime.protocol.common import *  # noqa: F403
from sglang_omni.serve.realtime.protocol.common import (
    _action_timing_breakdown,
    _env_flag,
    _json_audit_fields,
    _summarize_media,
    _text_audit_fields,
)

from sglang_omni.serve.realtime.protocol.models import (
    ActionHistoryTurn,
    ExecutedActionRecord,
    ImageFrame,
    ProvisionalReplyState,
    ReplyHistoryRouteResult,
    ReplyHistoryTurn,
    ReplySpeechModeResult,
    ReplyTTSState,
    SessionActionCandidate,
    SessionActionCategory,
    SessionActionProfile,
    TurnBuffer,
)


from sglang_omni.serve.realtime.action import ActionPipeline


from sglang_omni.serve.realtime.reply import ReplyPipeline


from sglang_omni.serve.realtime.memory.controller import SessionMemoryController


from sglang_omni.serve.realtime.protocol.validation import ProtocolComponent


from sglang_omni.serve.realtime.turn_pipeline import TurnPipeline


@compose_components(
    TurnPipeline,
    ProtocolComponent,
    SessionMemoryController,
    ReplyPipeline,
    ActionPipeline,
)
class MultimodalSession:
    """Manual-turn, multimodal session for audio chunks and image frames.

    This session owns the ``/v1/session/realtime`` protocol, including
    explicit ``turn.start``/``turn.commit`` boundaries and the fixed
    scheme-B action catalog.
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
        global_action_catalog: GlobalActionCatalog | None = None,
        global_action_prewarm: GlobalActionCatalogPrewarmStatus | None = None,
        allow_unregistered_protocol_actions: bool = False,
        claim_session: Callable[[str, "MultimodalSession"], None],
        release_session: Callable[[str, "MultimodalSession"], None],
        request_resource_sample: Callable[..., bool] | None = None,
        embedded_tts_config: EmbeddedTTSConfig | None = None,
        embedded_tts_connector: Callable[..., Any] | None = None,
        session_memory_config: SessionMemoryConfig | None = None,
        session_memory_scheduler: SessionMemoryScheduler | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(
            action_selection_mode
        )
        self.action_micro_batch_size = normalize_action_micro_batch_size(
            action_micro_batch_size
        )
        self.action_category_top_k = normalize_action_category_top_k(
            action_category_top_k
        )
        self.global_action_catalog = global_action_catalog
        self.allow_unregistered_protocol_actions = allow_unregistered_protocol_actions
        self.embedded_tts_config = embedded_tts_config
        self.embedded_tts_connector = embedded_tts_connector
        self.global_action_prewarm = (
            global_action_prewarm or GlobalActionCatalogPrewarmStatus.not_run()
        )
        self.claim_session = claim_session
        self.release_session = release_session
        self.request_resource_sample = request_resource_sample
        self.log_full_instructions = _env_flag(FULL_INSTRUCTIONS_LOG_ENV)
        self.action_ready_tts_decoupled = _env_flag(
            ACTION_READY_TTS_DECOUPLED_ENV,
            default=True,
        )
        self.route_action_parallel = _env_flag(
            ROUTE_ACTION_PARALLEL_ENV,
            default=True,
        )
        self.session_instance_id = uuid.uuid4().hex
        self.session_memory_config = (
            session_memory_config
            if session_memory_config is not None and session_memory_config.enabled
            else None
        )
        self.session_memory_store = (
            SessionMemoryStore(self.session_memory_config)
            if self.session_memory_config is not None
            else None
        )
        self.session_memory_scheduler = (
            session_memory_scheduler
            if self.session_memory_config is not None
            else None
        )
        if (
            self.session_memory_config is not None
            and self.session_memory_scheduler is None
            and self.session_memory_config.write_enabled
        ):
            self.session_memory_scheduler = SessionMemoryScheduler(
                max_queued_sessions=self.session_memory_config.max_queued_sessions,
                max_concurrent_extractions=(
                    self.session_memory_config.max_concurrent_extractions
                ),
            )
        self._session_turn_sequence = 0
        self._session_memory_pending_turns: deque[SessionMemoryTurn] = deque()
        self._session_memory_request_id: str | None = None
        self._session_memory_running_turn_seqs: tuple[int, ...] = ()
        self._session_memory_attempts: dict[int, int] = {}
        self._session_memory_queue_overflow_count = 0
        self._session_memory_updated = asyncio.Event()
        self._session_memory_updated.set()

        self.session_id: str | None = None
        self.protocol_version: int | None = None
        self.locale = DEFAULT_ACTION_PROMPT_LOCALE
        self.language = "en"
        self.instructions = ""
        self.unsupported_action_text = ""
        self.action_profile: SessionActionProfile | None = None
        self.modalities: tuple[str, ...] = DEFAULT_MODALITIES
        self.output_capabilities = SessionOutputCapabilities.parse(
            list(DEFAULT_OUTPUTS)
        )
        self.embedded_tts: EmbeddedTTSConnection | None = None
        self.closed = False
        self.started = False
        self.active_turn: TurnBuffer | None = None
        self.used_turn_ids: set[str] = set()
        self.cancelled_turn_ids: set[str] = set()
        self.history: list[dict[str, Any]] = []
        self.history_audios: list[str] = []
        self.history_images: list[str] = []
        self.history_image_roles: list[str] = []
        self._image_preprocess_semaphore = asyncio.Semaphore(2)
        self.history_turns: list[ActionHistoryTurn] = []
        self.reply_history_turns: list[ReplyHistoryTurn] = []
        self.executed_action_history: list[ExecutedActionRecord] = []
        self.last_executed_action: ExecutedActionRecord | None = None
        self.last_user_executed_action: ExecutedActionRecord | None = None
        self.last_action_finished_candidate_id: str | None = None
        self.last_action_finished_action_id: str | None = None
        self.candidates: list[SessionActionCandidate] = []
        self.categories: list[SessionActionCategory] = []
        self.fallback_category_ids: tuple[str, ...] = ()
        self.candidate_by_id: dict[str, SessionActionCandidate] = {}
        self.action_system_prompt = ""
        self.action_catalog_hash = ""
        self.global_action_catalog_hash = (
            global_action_catalog.catalog_hash if global_action_catalog else ""
        )
        self.action_prefix_cache_namespace = ""
        self.action_prefix_prefilled = False
        self._prefilled_action_prefix_namespaces: set[str] = set()
        self.prewarm_child_category_ids: tuple[str, ...] = ()
        self.prewarmed_child_category_ids: list[str] = []
        self.last_avatar_state: dict[str, Any] = {}
        self._send_lock = asyncio.Lock()
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
                    await self.send_error("invalid_request", "invalid_json", str(exc))
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
                    session_id = (
                        self._event_context_id(payload, "session_id") or self.session_id
                    )
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id
                        if self.active_turn is not None
                        else None
                    )
                    await self.send_error(
                        "invalid_request",
                        self._classify_error(payload, exc),
                        str(exc),
                        session_id=session_id,
                        turn_id=turn_id,
                    )
                except Exception as exc:
                    session_id = (
                        self._event_context_id(payload, "session_id") or self.session_id
                    )
                    turn_id = self._event_context_id(payload, "turn_id") or (
                        self.active_turn.turn_id
                        if self.active_turn is not None
                        else None
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
            await self._shutdown_session_memory()
            if self.embedded_tts is not None:
                await self.embedded_tts.close()
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




























    def _prompt(self, *, zh: str, en: str) -> str:
        return localized_prompt(self.language, zh=zh, en=en)




    def _image_role_label(self, image_role: str) -> str:
        if image_role == IMAGE_ROLE_USER_CAMERA:
            return self._prompt(
                zh="用户摄像头画面（用于观察用户及其环境）",
                en="User camera view (for observing the user and their environment)",
            )
        return self._prompt(
            zh="数字人当前状态画面（用于观察数字人自身当前可视姿态和行为）",
            en=(
                "Current digital character state view (for observing the "
                "character's visible pose and behavior)"
            ),
        )

    def _image_role_text_part(self, image_role: str) -> dict[str, str]:
        return {
            "type": "text",
            "text": self._prompt(
                zh=f"[图片用途] {self._image_role_label(image_role)}：",
                en=f"[Image purpose] {self._image_role_label(image_role)}:",
            ),
        }

    @staticmethod
    def _has_structured_avatar_state(avatar_state: dict[str, Any]) -> bool:
        """Return whether reusable structured avatar state is actually present."""
        for key, value in avatar_state.items():
            if key in {"current_action_id", "state_description"} or value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, dict, set)) and not value:
                continue
            return True
        return False

    @classmethod
    def _avatar_state_source(
        cls,
        avatar_state: dict[str, Any],
        image_roles: list[str],
    ) -> Literal["image", "structured", "unknown"]:
        if IMAGE_ROLE_AVATAR_STATE in image_roles:
            return "image"
        if cls._has_structured_avatar_state(avatar_state):
            return "structured"
        return "unknown"

    def _build_avatar_state_instruction(
        self,
        source: Literal["image", "structured", "unknown"],
    ) -> str:
        """Build stage-neutral state guidance outside the static KV prefix."""
        if source == "image":
            return self._prompt(
                zh=(
                    "判断候选动作可执行性时，以本轮最新数字人照片中的当前可视姿态和行为"
                    "为准。\n"
                ),
                en=(
                    "When deciding whether a candidate action is executable, use the "
                    "visible pose and behavior in the latest current digital character "
                    "image from this interaction.\n"
                ),
            )
        if source == "structured":
            return self._prompt(
                zh=(
                    "本轮未提供数字人状态照片；判断候选动作可执行性时，以结构化 "
                    "数字人当前状态信息为准；不得从历史图片或用户摄像头画面"
                    "推断数字人状态。\n"
                ),
                en=(
                    "No current digital character state image is provided in this "
                    "interaction. Use the structured current character state when deciding "
                    "whether a candidate is executable. Do not infer the character state "
                    "from historical images or the user camera view.\n"
                ),
            )
        return self._prompt(
            zh=(
                "本轮没有可用的数字人当前状态信息；不得从历史图片或用户摄像头画面"
                "推断数字人状态。不依赖特定起始姿态的候选，不应仅因状态未知而被排除。\n"
            ),
            en=(
                "No current digital character state information is available in this "
                "interaction. Do not infer it from historical images or the user camera "
                "view. Do not exclude candidates that do not require a specific starting "
                "pose solely because the state is unknown.\n"
            ),
        )

    def _effective_avatar_state(
        self,
        avatar_state: dict[str, Any] | None,
        *,
        turn_origin: str,
        has_avatar_image: bool,
    ) -> dict[str, Any]:
        """Build model-visible current state without exposing prior-action identity."""
        explicit = dict(avatar_state or {})
        effective = {} if has_avatar_image else dict(self.last_avatar_state)
        for key, value in explicit.items():
            if key == "state_description":
                if turn_origin == TURN_ORIGIN_PROACTIVE:
                    effective[key] = value
            elif key != "current_action_id":
                effective[key] = value
        return effective

    def _persist_avatar_state(
        self,
        avatar_state: dict[str, Any] | None,
        *,
        has_avatar_image: bool,
    ) -> None:
        """Persist only reusable structured visual state across turns."""
        if has_avatar_image:
            self.last_avatar_state = {}
        elif avatar_state is not None:
            self.last_avatar_state = {
                key: value
                for key, value in avatar_state.items()
                if key not in {"current_action_id", "state_description"}
            }

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












































































































    async def send(self, payload: dict[str, Any]) -> bool:
        async with self._send_lock:
            return await self._send_unlocked(payload)

    async def _send_unlocked(self, payload: dict[str, Any]) -> bool:
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return False
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
            await self.websocket.send_text(encoded)
            turn_id = payload.get("turn_id")
            trace_id = None
            if self.active_turn is not None and turn_id == self.active_turn.turn_id:
                trace_id = self.active_turn.trace_id
            emit_structured_log(
                "protocol",
                "ws_event_sent",
                session_id=self.session_id,
                turn_id=turn_id,
                trace_id=trace_id,
                ws_event_type=payload.get("type"),
                payload_bytes=len(encoded.encode("utf-8")),
            )
            if payload.get("type") == "turn.action.ready":
                emit_structured_log(
                    "performance",
                    "turn_action_ready_sent",
                    session_id=self.session_id,
                    turn_id=turn_id,
                    trace_id=trace_id,
                    after_commit_ms=(
                        self._after_commit_ms(self.active_turn)
                        if self.active_turn is not None
                        and turn_id == self.active_turn.turn_id
                        else None
                    ),
                )
            elif payload.get("type") == "turn.result":
                emit_structured_log(
                    "performance",
                    "turn_result_sent",
                    session_id=self.session_id,
                    turn_id=turn_id,
                    trace_id=trace_id,
                    status=payload.get("status"),
                    outputs=payload.get("outputs") or payload.get("modalities"),
                )
            return True
        except (OSError, WebSocketDisconnect):
            self.closed = True
            logger.info(
                "[SESSION_ACTION_REALTIME] client disconnected during send "
                "session_id=%s event=%s",
                self.session_id,
                payload.get("type"),
            )
            return False
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
            return False

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
        emit_structured_log(
            "error",
            "websocket_error_sent",
            level="error",
            session_id=session_id or self.session_id,
            turn_id=turn_id,
            error_type=type_,
            error_code=code,
            error_message=message,
        )
        await self.send(payload)


class MultimodalSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        action_selection_mode: str | None = None,
        global_action_catalog: GlobalActionCatalog | None = None,
        global_action_prewarm: GlobalActionCatalogPrewarmStatus | None = None,
        allow_unregistered_protocol_actions: bool = False,
        embedded_tts_config: EmbeddedTTSConfig | None = None,
        embedded_tts_connector: Callable[..., Any] | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.action_selection_mode = normalize_action_selection_mode(
            action_selection_mode
        )
        self.action_micro_batch_size = normalize_action_micro_batch_size()
        self.action_category_top_k = normalize_action_category_top_k()
        self.global_action_catalog = global_action_catalog
        self.allow_unregistered_protocol_actions = allow_unregistered_protocol_actions
        self.embedded_tts_config = embedded_tts_config
        self.embedded_tts_connector = embedded_tts_connector
        self.global_action_prewarm = (
            global_action_prewarm or GlobalActionCatalogPrewarmStatus.not_run()
        )
        self.session_memory_config = SessionMemoryConfig.from_env()
        self.session_memory_scheduler = (
            SessionMemoryScheduler(
                max_queued_sessions=(
                    self.session_memory_config.max_queued_sessions
                ),
                max_concurrent_extractions=(
                    self.session_memory_config.max_concurrent_extractions
                ),
            )
            if self.session_memory_config.enabled
            and self.session_memory_config.write_enabled
            else None
        )
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
        logger.info(
            "[SESSION_ACTION_REALTIME] session_memory_enabled=%s "
            "write_enabled=%s read_enabled=%s "
            "batch_turns=%s max_pending_turns=%s max_retries=%s "
            "max_active_claims=%s max_injected_claims=%s "
            "max_queued_sessions=%s max_concurrent_extractions=%s",
            self.session_memory_config.enabled,
            self.session_memory_config.write_enabled,
            self.session_memory_config.read_enabled,
            self.session_memory_config.batch_turns,
            self.session_memory_config.max_pending_turns,
            self.session_memory_config.max_retries,
            self.session_memory_config.max_active_claims,
            self.session_memory_config.max_injected_claims,
            self.session_memory_config.max_queued_sessions,
            self.session_memory_config.max_concurrent_extractions,
        )
        self.sessions: dict[str, MultimodalSession] = {}
        self.resource_sample_requester: Callable[..., bool] | None = None

    def set_resource_sample_requester(
        self,
        requester: Callable[..., bool] | None,
    ) -> None:
        self.resource_sample_requester = requester

    def create(self, websocket: WebSocket) -> MultimodalSession:
        return MultimodalSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
            action_selection_mode=self.action_selection_mode,
            action_micro_batch_size=self.action_micro_batch_size,
            action_category_top_k=self.action_category_top_k,
            global_action_catalog=self.global_action_catalog,
            global_action_prewarm=self.global_action_prewarm,
            allow_unregistered_protocol_actions=(
                self.allow_unregistered_protocol_actions
            ),
            embedded_tts_config=self.embedded_tts_config,
            embedded_tts_connector=self.embedded_tts_connector,
            claim_session=self.claim,
            release_session=self.release,
            request_resource_sample=self.resource_sample_requester,
            session_memory_config=self.session_memory_config,
            session_memory_scheduler=self.session_memory_scheduler,
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

    def load_snapshot(self) -> dict[str, Any]:
        """Return aggregate session load without exposing session identifiers."""

        turn_phase_counts: dict[str, int] = {}
        modality_counts: dict[str, int] = {}
        active_turn_count = 0
        started_session_count = 0
        action_history_turn_count = 0
        reply_history_turn_count = 0
        session_memory_active_claim_count = 0
        session_memory_episode_count = 0
        session_memory_pending_turn_count = 0
        session_memory_running_turn_count = 0
        session_memory_gap_turn_count = 0
        session_memory_queue_overflow_count = 0
        for session in self.sessions.values():
            if session.started:
                started_session_count += 1
            modality_key = "+".join(session.modalities) or "not_started"
            modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1
            action_history_turn_count += len(session.history_turns)
            reply_history_turn_count += len(session.reply_history_turns)
            if session.session_memory_store is not None:
                session_memory_active_claim_count += len(
                    session.session_memory_store.active_claims()
                )
                session_memory_episode_count += len(
                    session.session_memory_store.episodes
                )
                session_memory_pending_turn_count += len(
                    session._session_memory_pending_turns
                )
                session_memory_running_turn_count += len(
                    session._session_memory_running_turn_seqs
                )
                session_memory_gap_turn_count += len(
                    session.session_memory_store.gap_turn_seqs
                )
                session_memory_queue_overflow_count += (
                    session._session_memory_queue_overflow_count
                )
            turn = session.active_turn
            if turn is not None:
                active_turn_count += 1
                turn_phase_counts[turn.phase] = turn_phase_counts.get(turn.phase, 0) + 1
        return {
            "active_session_count": len(self.sessions),
            "started_session_count": started_session_count,
            "active_turn_count": active_turn_count,
            "turn_phase_counts": turn_phase_counts,
            "modality_counts": modality_counts,
            "retained_action_history_turn_count": action_history_turn_count,
            "retained_reply_history_turn_count": reply_history_turn_count,
            "session_memory": {
                "enabled": self.session_memory_config.enabled,
                "write_enabled": self.session_memory_config.write_enabled,
                "read_enabled": self.session_memory_config.read_enabled,
                "active_claim_count": session_memory_active_claim_count,
                "episode_count": session_memory_episode_count,
                "pending_turn_count": session_memory_pending_turn_count,
                "running_turn_count": session_memory_running_turn_count,
                "gap_turn_count": session_memory_gap_turn_count,
                "queue_overflow_count": session_memory_queue_overflow_count,
                "scheduler": (
                    self.session_memory_scheduler.snapshot()
                    if self.session_memory_scheduler is not None
                    else {
                        "queued_session_count": 0,
                        "running_job_count": 0,
                        "dirty_session_count": 0,
                        "rejected_submission_count": 0,
                        "max_queued_sessions": (
                            self.session_memory_config.max_queued_sessions
                        ),
                        "max_concurrent_extractions": (
                            self.session_memory_config.max_concurrent_extractions
                        ),
                    }
                ),
            },
            "global_action_catalog": {
                "configured": self.global_action_catalog is not None,
                "catalog_hash": (
                    self.global_action_catalog.catalog_hash
                    if self.global_action_catalog is not None
                    else None
                ),
                "category_count": (
                    len(self.global_action_catalog.categories)
                    if self.global_action_catalog is not None
                    else 0
                ),
                "candidate_count": (
                    self.global_action_catalog.candidate_count
                    if self.global_action_catalog is not None
                    else 0
                ),
                "category_prefix_ready": self.global_action_prewarm.category_ready,
                "child_prefix_ready_count": len(
                    self.global_action_prewarm.ready_child_category_ids
                ),
                "child_prefix_failed_count": len(
                    self.global_action_prewarm.failed_child_category_ids
                ),
                "locale_prefix_statuses": {
                    locale: {
                        "category_prefix_ready": status.category_ready,
                        "child_prefix_ready_count": len(
                            status.ready_child_category_ids
                        ),
                        "child_prefix_failed_count": len(
                            status.failed_child_category_ids
                        ),
                    }
                    for locale, status in self.global_action_prewarm.by_locale.items()
                },
            },
        }
