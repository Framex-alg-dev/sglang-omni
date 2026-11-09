"""Stateless text/audio recall adapter with service-owned bookkeeping.

User text needs only text, channel and persona; audio also requires its speaker.
An omitted
candidate subset uses the full channel catalog. Offsets describe adjacent supplied clips, not playback
positions; the simple_audio flag keeps them out of model-visible evidence.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
import uuid
import wave
from typing import Any

from sglang_omni.serve.streaming_request import StreamedMedia

from .catalog import ActionCatalogRegistry
from .contracts import ActionDecisionRequest, DecisionChannel
from .speech_audio import MAX_SPEECH_AUDIO_MS, USER_CONTEXT_AUDIO_ROLE, validate_avatar_audio_clip
from .speech_window import PERSONA_AUDIO_PROFILE


_FIELDS = {"text", "audio", "audio_source", "channel", "character_prompt", "allowed_candidate_ids",
           "context_audio", "user_context", "user_context_audio", "last_dispatched_action", "request_id"}
_MAX_AUDIO_BYTES = 2 * 1024 * 1024


def _clip(raw: Any, *, role: str, start: int) -> StreamedMedia:
    if not isinstance(raw, dict) or set(raw) - {"data", "format"}:
        raise ValueError("audio must contain base64 data and optional format")
    data = raw.get("data")
    if not isinstance(data, str) or not data or len(data) > 4 * ((_MAX_AUDIO_BYTES + 2) // 3):
        raise ValueError("audio.data must be bounded nonempty base64")
    try:
        payload = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("audio.data is not valid base64") from exc
    if not payload or len(payload) > _MAX_AUDIO_BYTES:
        raise ValueError("decoded audio exceeds the size limit or is empty")
    encoding = raw.get("format", "wav")
    if encoding == "wav":
        try:
            with wave.open(io.BytesIO(payload), "rb") as audio:
                duration = audio.getnframes() * 1000 / audio.getframerate()
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise ValueError("audio.data must contain a decodable WAV") from exc
    elif encoding == "pcm_s16le":
        duration = len(payload) / 32.0
    else:
        raise ValueError("audio.format must be wav or pcm_s16le (16 kHz mono PCM16)")
    if not 0 < duration <= MAX_SPEECH_AUDIO_MS:
        raise ValueError("audio duration must be positive and at most 30 seconds")
    item = StreamedMedia(role, "audio", start, start + max(1, round(duration)), encoding,
                         "sha256:" + hashlib.sha256(payload).hexdigest(), payload, role)
    # Reuse the strict sample format and truncation checks, including user clips.
    validate_avatar_audio_clip(item)
    return item


def parse_simple_audio_request(
    raw: Any, registry: ActionCatalogRegistry,
) -> tuple[ActionDecisionRequest, str]:
    if not isinstance(raw, dict):
        raise ValueError("audio recall payload must be an object")
    if set(raw) - _FIELDS:
        raise ValueError("unsupported audio recall fields: " + ", ".join(sorted(set(raw) - _FIELDS)))
    channel = DecisionChannel(raw.get("channel"))
    user_text = raw.get("text")
    text_request = "text" in raw
    if text_request:
        if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 4000 or "\x00" in user_text:
            raise ValueError("text must be a nonempty user input string of at most 4000 characters")
        if set(raw) & {"audio", "audio_source", "context_audio", "user_context", "user_context_audio"}:
            raise ValueError("user text requires text alone; omit audio, audio_source, context_audio, user_context and user_context_audio")
        source = "user_text"
    else:
        source = raw.get("audio_source")
        if not isinstance(source, str) or source not in {"user_audio", "avatar_speech_audio"}:
            raise ValueError("audio_source must be user_audio or avatar_speech_audio")
    persona = raw.get("character_prompt")
    if not isinstance(persona, str) or not persona.strip() or len(persona) > 32_000 or "\x00" in persona:
        raise ValueError("character_prompt must be a bounded nonempty string")
    ids = raw.get("allowed_candidate_ids", [])
    if "allowed_candidate_ids" in raw and (not isinstance(ids, list) or not ids or len(ids) > 512
            or any(not isinstance(cid, str) or not cid or len(cid) > 256 for cid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("allowed_candidate_ids must be a nonempty list of unique registered IDs")
    public_id = raw.get("request_id", uuid.uuid4().hex)
    if not isinstance(public_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", public_id) is None:
        raise ValueError("request_id must be a bounded alphanumeric request identifier")
    internal_id = "simple-" + hashlib.sha256(public_id.encode()).hexdigest()
    user_context = raw.get("user_context")
    if user_context is not None and (not isinstance(user_context, str) or len(user_context) > 4000 or "\x00" in user_context):
        raise ValueError("user_context must be a bounded string")
    original_user_audio = raw.get("user_context_audio")
    if source == "user_audio" and (raw.get("context_audio") is not None or user_context or original_user_audio is not None):
        raise ValueError("context_audio, user_context and user_context_audio are only accepted for avatar speech")
    if original_user_audio is not None and user_context is not None:
        raise ValueError("user_context and user_context_audio are mutually exclusive")
    previous = raw.get("last_dispatched_action")
    if channel is DecisionChannel.EXPRESSION and previous is not None:
        raise ValueError("last_dispatched_action only describes the body channel")

    clips = []
    if raw.get("context_audio") is not None:
        clips.append(_clip(raw["context_audio"], role="avatar_speech_context", start=0))
    if not text_request:
        focus = _clip(raw.get("audio"), role=source, start=clips[-1].end_ms if clips else 0)
        clips.append(focus)
        if focus.end_ms > MAX_SPEECH_AUDIO_MS:
            raise ValueError("combined context and target audio exceeds 30 seconds")
    if original_user_audio is not None:
        # The original user turn has its own clock, separate from the reply.
        clips.insert(0, _clip(original_user_audio, role=USER_CONTEXT_AUDIO_ROLE, start=0))
        if sum(item.end_ms - item.start_ms for item in clips) > MAX_SPEECH_AUDIO_MS:
            raise ValueError("combined user context and avatar audio exceeds 30 seconds")
    context = {"prompt_profile": PERSONA_AUDIO_PROFILE,
               "persona_revision": "simple-audio-v1",
               "decision_mode": "speech_window" if source == "avatar_speech_audio" else "user_request"}
    if channel is DecisionChannel.BODY:
        context["last_dispatched_action"] = previous
    if source == "avatar_speech_audio":
        context.update(response_id=internal_id, output_epoch=0,
                       speech_window={"window_id": "focus", "start_ms": focus.start_ms, "end_ms": focus.end_ms})
    req = ActionDecisionRequest(
        request_id=internal_id, session_id=internal_id, turn_id=internal_id, decision_point_id=internal_id,
        channel=channel, text=user_text if text_request else user_context, character_prompt=persona,
        allowed_candidate_ids=tuple(ids), runtime_context=context, media=tuple(clips), simple_audio=True,
    )
    catalog = registry.resolve(req)
    real_ids = {entry.candidate_id for entry in catalog.entries if entry.kind == "action"}
    unsupported = set(ids) - real_ids
    if unsupported:
        raise ValueError("candidates are not registered in this channel: " + ", ".join(sorted(unsupported)))
    return req, public_id
