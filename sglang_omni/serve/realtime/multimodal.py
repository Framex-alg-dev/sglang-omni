from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import Client
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
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
# Action scoring runs on the thinker stage, whose current deployment has an
# 8192-token context. Keep the action context bounded while retaining recent
# session history, including the assistant-side action state records.
MAX_ACTION_HISTORY_TURNS = 4
MAX_ACTION_HISTORY_AUDIOS = 4
MAX_ACTION_HISTORY_IMAGES = 8
MAX_ACTION_CURRENT_IMAGES = 8
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
    "你是一个有帮助的数字人助手。请根据用户当前的音频、图片和历史对话自然地回复。"
)

ACTION_HISTORY_INSTRUCTION = (
    "历史 assistant 消息中的 [action_state] 是服务端记录的上一轮实际动作，"
    "不是新的用户指令。用户说“刚刚、上一轮、再重复、这个动作”等指代词时，"
    "优先参考最近一条 [action_state]；如果 action_id=no_action，则保持不动作。"
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


@dataclass(slots=True)
class TurnBuffer:
    turn_id: str
    started_at: float
    audio: RealtimeAudioBuffer
    images: list[ImageFrame]
    audio_seqs: set[int]
    image_seqs: set[int]
    text: str | None = None
    avatar_state: dict[str, Any] | None = None
    audio_chunk_count: int = 0
    duplicate_audio_chunks: int = 0
    duplicate_image_frames: int = 0


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
        claim_session: Callable[[str, "MultimodalSession"], None],
        release_session: Callable[[str, "MultimodalSession"], None],
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
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
        self.candidates: list[SessionActionCandidate] = []
        self.categories: list[SessionActionCategory] = []
        self.candidate_by_id: dict[str, SessionActionCandidate] = {}
        self.action_system_prompt = ""
        self.action_catalog_hash = ""
        self.last_avatar_state: dict[str, Any] = {}

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
        await handler(payload)

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

        self.claim_session(session_id, self)
        self.session_id = session_id
        self.language = language
        if instructions is not None:
            self.instructions = instructions
        self.candidates = candidates
        self.categories = categories
        self.candidate_by_id = {x.candidate_id: x for x in candidates}
        self.action_system_prompt = (self._build_category_system_prompt() if categories else self._build_action_system_prompt())
        canonical = json.dumps(
            ([category.as_dict() for category in categories] if categories else [x.as_dict() for x in candidates]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.action_catalog_hash = "sha256:" + hashlib.sha256(canonical).hexdigest()
        self.started = True

        await self.send(
            {
                "type": "session.started",
                "session_id": self.session_id,
                "model": self.model_name,
                "action_catalog_hash": self.action_catalog_hash,
                "action_candidate_count": len(candidates),
                "action_category_count": len(categories),
                "action_selection_stages": 2 if categories else 1,
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
        )
        await self.send(
            {
                "type": "turn.started",
                "session_id": self.session_id,
                "turn_id": turn_id,
            }
        )

    async def handle_audio_append(self, event: dict[str, Any]) -> None:
        turn = self._require_turn(event)
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
        turn = self._require_turn(event)
        seq = self._positive_int(event.get("seq"), "seq")
        if seq in turn.image_seqs:
            turn.duplicate_image_frames += 1
            await self._ack_media(turn, "image", seq, duplicate=True)
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
            ImageFrame(seq=seq, timestamp_ms=timestamp_ms, data_uri=data_uri)
        )
        turn.image_seqs.add(seq)
        await self._ack_media(turn, "image", seq)

    async def handle_text_update(self, event: dict[str, Any]) -> None:
        turn = self._require_turn(event)
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
        self.active_turn = None
        await self.send(
            {
                "type": "turn.cancelled",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
            }
        )

    async def handle_session_close(self, event: dict[str, Any]) -> None:
        del event
        self.closed = True
        await self.send(
            {"type": "session.closed", "session_id": self.session_id}
        )

    async def handle_turn_commit(self, event: dict[str, Any]) -> None:
        turn = self._require_turn(event)
        if "text" in event:
            text = event.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("text must be a string or null")
            turn.text = text
        if "avatar_state" in event:
            state = event.get("avatar_state")
            if not isinstance(state, dict):
                raise ValueError("avatar_state must be an object")
            turn.avatar_state = dict(state)
        if turn.avatar_state is not None:
            self.last_avatar_state = dict(turn.avatar_state)

        current_audio = (
            turn.audio.to_full_wav_data_uri() if not turn.audio.is_empty() else None
        )
        current_images = [
            frame.data_uri
            for frame in sorted(
                turn.images, key=lambda x: (x.timestamp_ms, x.seq)
            )
        ]
        current_audio_list = [current_audio] if current_audio else []
        ingest_ms = (time.perf_counter() - turn.started_at) * 1000.0
        turn_id = turn.turn_id

        try:
            await self.send(
                {
                    "type": "turn.committed",
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                }
            )
            commit_started = time.perf_counter()
            logger.info(
                "[SESSION_ACTION_REALTIME] turn.commit input session_id=%s turn_id=%s payload=%s",
                self.session_id,
                turn_id,
                json.dumps({
                    "text": turn.text,
                    "avatar_state": turn.avatar_state or self.last_avatar_state,
                    "audio_chunk_count": turn.audio_chunk_count,
                    "image_frame_count": len(current_images),
                    "audio": _summarize_media(current_audio_list),
                    "images": _summarize_media(current_images),
                    "history_turn_count": len(self.history) // 2,
                    "candidate_count": len(self.candidates),
                    "action_catalog_hash": self.action_catalog_hash,
                }, ensure_ascii=False, default=str),
            )
            action_started = time.perf_counter()
            action, scores, action_timing, action_context = await self._score_action(
                current_audio_list, current_images, turn.text, turn.avatar_state, turn_id=turn_id
            )
            logger.info(
                "[SESSION_ACTION_REALTIME] action completed session_id=%s turn_id=%s elapsed_ms=%.3f top_action=%s",
                self.session_id, turn_id, (time.perf_counter() - action_started) * 1000.0, action.get("action_id"),
            )
            total_after_commit_ms = (time.perf_counter() - commit_started) * 1000.0
            self._append_action_history(
                current_audio_list,
                current_images,
                turn.text,
                turn_id=turn_id,
                action=action,
            )
            self.active_turn = None
            await self.send(
                {
                    "type": "turn.result",
                    "session_id": self.session_id,
                    "turn_id": turn_id,
                    "reply": None,
                    "action_catalog_hash": self.action_catalog_hash,
                    "action": action,
                    "scores": scores,
                    "media_summary": {
                        "audio_chunk_count": turn.audio_chunk_count,
                        "image_frame_count": len(current_images),
                        "received_image_count": len(turn.images),
                        "scored_image_count": action_context["scored_current_image_count"],
                        "text_present": bool(turn.text),
                        "duplicate_audio_chunks": turn.duplicate_audio_chunks,
                        "duplicate_image_frames": turn.duplicate_image_frames,
                        "action_context": action_context,
                    },
                    "timing": {
                        "server_turn_ingest_ms": round(ingest_ms, 3),
                        "server_action_compute_ms": action_timing,
                        "server_total_after_commit_ms": round(
                            total_after_commit_ms, 3
                        ),
                    },
                }
            )
        except Exception:
            logger.exception(
                "[SESSION_ACTION_REALTIME] turn failed session_id=%s turn_id=%s",
                self.session_id,
                turn_id,
            )
            self.active_turn = None
            raise

    def _build_bounded_action_context(
        self,
        audios: list[str],
        images: list[str],
    ) -> tuple[
        list[dict[str, Any]],
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
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []
        for message in self.history:
            if message.get("role") == "user" and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(message)
        if current_group:
            groups.append(current_group)

        selected_groups = groups[-MAX_ACTION_HISTORY_TURNS:]
        selected_message_ids = {
            id(message) for group in selected_groups for message in group
        }
        bounded_history: list[dict[str, Any]] = []
        bounded_history_audios: list[str] = []
        bounded_history_images: list[str] = []
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
                        if len(bounded_history_images) < MAX_ACTION_HISTORY_IMAGES:
                            bounded_history_images.append(media)
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
        bounded_audios = list(audios)
        truncated = (
            len(groups) > len(selected_groups)
            or len(self.history_audios) != len(bounded_history_audios)
            or len(self.history_images) != len(bounded_history_images)
            or len(images) != len(bounded_images)
        )
        context_summary = {
            "history_turn_count": len(selected_groups),
            "history_audio_count": len(bounded_history_audios),
            "history_image_count": len(bounded_history_images),
            "received_current_image_count": len(images),
            "scored_current_image_count": len(bounded_images),
            "truncated": truncated,
        }
        return (
            bounded_history,
            bounded_history_audios,
            bounded_history_images,
            bounded_images,
            context_summary,
        )

    async def _score_action(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        if not self.categories:
            return await self._score_action_flat(audios, images, text, avatar_state, turn_id=turn_id)
        return await self._score_action_hierarchical(audios, images, text, avatar_state, turn_id=turn_id)

    async def _score_action_hierarchical(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history, action_history_audios, action_history_images,
            action_images, action_context,
        ) = self._build_bounded_action_context(audios, images)
        text_prefix = (
            f"当前用户文本：{text.strip()}\n"
            if isinstance(text, str) and text.strip() else ""
        )
        base = (
            text_prefix
            + "请根据当前音频、图片、历史对话和数字人状态进行动作选择。"
        )
        request_base = f"session-{self.session_id}-turn-{turn_id or 'unknown'}-action-{uuid.uuid4().hex}"
        common = dict(
            model=self.model_name, language=self.language, audios=audios,
            images=action_images, sample_rate=16000, micro_batch_size=64,
            session_id=self.session_id, history=action_history,
            stage="category", logical_request_id=request_base,
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
            **common,
        )
        started = time.perf_counter()
        category_started = time.perf_counter()
        category_result = await self.client.score_action_suffixes(category_request)
        category_ms = round((time.perf_counter() - category_started) * 1000.0, 3)
        category_by_id = {item.category_id: item for item in self.categories}
        category_ranked = sorted(category_result.scores, key=lambda item: item.mean_logprob, reverse=True)
        if not category_ranked or category_ranked[0].candidate_id not in category_by_id:
            raise ValueError("category action score did not return a valid category")
        selected_category = category_by_id[category_ranked[0].candidate_id]
        action_common = {**common, "stage": "child"}
        action_request = ActionSuffixScoreRequest(
            request_id=request_base + "-child", prefix=base +
            f"已选择动作类别 {selected_category.category_id}。请只在该类别的子动作中选择一个。"
            "只输出 action_id，不要解释。没有明确动作指令时必须选择 no_action。下一步 action_id 是：",
            system_prompt=self._build_child_system_prompt(selected_category),
            candidates=[ActionScoreCandidate(
                candidate_id=item.candidate_id, suffix=item.candidate_id,
                action_id=item.action_id, execution_binding=dict(item.execution_binding)
            ) for item in selected_category.children],
            **action_common,
        )
        child_started = time.perf_counter()
        child_result = await self.client.score_action_suffixes(action_request)
        child_ms = round((time.perf_counter() - child_started) * 1000.0, 3)
        child_by_id = {item.candidate_id: item for item in selected_category.children}
        ranked = sorted(child_result.scores, key=lambda item: item.mean_logprob, reverse=True)
        if not ranked or ranked[0].candidate_id not in child_by_id:
            raise ValueError("child action score did not return a valid candidate")

        def score_dict(score: Any, candidate: SessionActionCandidate) -> dict[str, Any]:
            return {
                "candidate_id": score.candidate_id, "action_id": candidate.action_id,
                "category_id": candidate.category_id, "source_label": candidate.source_label,
                "short_definition": candidate.short_definition, "token_count": score.token_count,
                "mean_logprob": score.mean_logprob, "mean_nll": score.mean_nll, "ppl": score.ppl,
                "token_scores": [{"token_id": item.token_id, "logprob": item.logprob} for item in score.token_scores],
            }

        scores = [score_dict(score, child_by_id[score.candidate_id]) for score in ranked]
        top = scores[0]
        action = {
            "candidate_id": top["candidate_id"], "action_id": top["action_id"],
            "category_id": selected_category.category_id,
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"], "ppl": top["ppl"], "token_count": top["token_count"],
        }
        def compact_stage_score(score: Any) -> dict[str, Any]:
            return {"candidate_id": score.candidate_id, "token_count": score.token_count,
                    "mean_logprob": score.mean_logprob, "mean_nll": score.mean_nll, "ppl": score.ppl,
                    "token_scores": [{"token_id": item.token_id, "logprob": item.logprob} for item in score.token_scores]}
        action_context.update({
            "selection_stages": 2,
            "logical_request_id": request_base,
            "selected_category_id": selected_category.category_id,
            "category_scores": [compact_stage_score(score) for score in category_ranked],
            "category_compute_ms": category_ms,
            "child_compute_ms": child_ms,
            "media_input_reused_across_stages": True,
        })
        return action, scores, round((time.perf_counter() - started) * 1000.0, 3), action_context

    async def _score_action_flat(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
        avatar_state: dict[str, Any] | None,
        *,
        turn_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, dict[str, Any]]:
        (
            action_history,
            action_history_audios,
            action_history_images,
            action_images,
            action_context,
        ) = self._build_bounded_action_context(audios, images)
        text_prefix = (
            f"当前用户文本：{text.strip()}\n"
            if isinstance(text, str) and text.strip()
            else ""
        )
        prefix = (
            text_prefix
            + "请根据当前音频、图片、历史对话和数字人状态，"
            "从固定候选集合中选择唯一一个最合适的动作。"
            f"没有明确动作指令时请选择 {self._no_action_candidate_id()}。"
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
            request_id=f'session-{self.session_id}-turn-{turn_id or "unknown"}-action-{uuid.uuid4().hex}',
            model=self.model_name,
            prefix=prefix,
            system_prompt=self.action_system_prompt,
            language=self.language,
            candidates=candidates,
            audios=audios,
            images=action_images,
            sample_rate=16000,
            micro_batch_size=64,
            session_id=self.session_id,
            history=action_history,
            history_audios=action_history_audios,
            history_images=action_history_images,
            avatar_state=dict(avatar_state or self.last_avatar_state),
        )
        started = time.perf_counter()
        result = await self.client.score_action_suffixes(request)
        compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
        ranked = sorted(result.scores, key=lambda x: x.mean_logprob, reverse=True)
        scores: list[dict[str, Any]] = []
        for score in ranked:
            candidate = self.candidate_by_id[score.candidate_id]
            scores.append(
                {
                    "candidate_id": score.candidate_id,
                    "action_id": candidate.action_id,
                    "source_label": candidate.source_label,
                    "short_definition": candidate.short_definition,
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
            "execute": top["action_id"] != "no_action",
            "mean_logprob": top["mean_logprob"],
            "ppl": top["ppl"],
            "token_count": top["token_count"],
        }
        return action, scores, compute_ms, action_context

    def _append_action_history(
        self,
        audios: list[str],
        images: list[str],
        text: str | None,
        *,
        turn_id: str,
        action: dict[str, Any],
    ) -> None:
        self.history.append(
            {
                "role": "user",
                "content": self._current_user_content(audios, images, text),
            }
        )
        self.history_audios.extend(audios)
        self.history_images.extend(images)

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
        self.history.append({"role": "assistant", "content": action_state})

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
                    "text": "请根据当前会话内容进行回复。",
                }
            )
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
        for item in self.categories:
            lines.append(f"{item.category_id}={item.source_label}；{item.short_definition}")
        no_action_categories = [item.category_id for item in self.categories if any(child.action_id == "no_action" for child in item.children)]
        if no_action_categories:
            lines.append("没有明确动作指令时必须选择包含 no_action 的类别：" + ", ".join(no_action_categories))
        lines.append("只输出 category_id，不要解释。")
        return "\n".join(lines)

    def _build_child_system_prompt(self, category: SessionActionCategory) -> str:
        lines = [
            "你是数字人动作识别器。",
            f"当前已选动作类别：{category.category_id}={category.source_label}；{category.short_definition}",
            "以下是该类别内固定的子动作集合，只能输出一个 action_id。",
            ACTION_HISTORY_INSTRUCTION,
        ]
        for item in category.children:
            lines.append(f"{item.candidate_id}={item.action_id}：{item.source_label}；{item.short_definition}")
        lines.extend(["没有明确动作指令时必须选择 no_action。", "只输出 action_id，不要解释。"])
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
                f"没有明确动作指令时必须选择 {self._no_action_candidate_id()} 对应的 no_action。",
                "只输出候选 ID，不要解释。",
            ]
        )
        return "\n".join(lines)

    async def _ack_media(
        self, turn: TurnBuffer, media_type: str, seq: int, duplicate: bool = False
    ) -> None:
        await self.send(
            {
                "type": "input.ack",
                "session_id": self.session_id,
                "turn_id": turn.turn_id,
                "media_type": media_type,
                "seq": seq,
                "duplicate": duplicate,
            }
        )

    def _require_started(self) -> None:
        if not self.started or self.session_id is None:
            raise ValueError("session.start must be sent first")

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

    @staticmethod
    def _event_context_id(payload: dict[str, Any], field: str) -> str | None:
        value = payload.get(field)
        return value if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _classify_error(payload: dict[str, Any], exc: Exception) -> str:
        message = str(exc).lower()
        event_type = payload.get("type")
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
    def __init__(self, *, client: Client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.sessions: dict[str, MultimodalSession] = {}

    def create(self, websocket: WebSocket) -> MultimodalSession:
        return MultimodalSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
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
