"""Stateless text/audio recall adapter with service-owned bookkeeping.

User text needs only text, channel and persona; audio also requires its speaker.
An omitted
candidate subset uses the full channel catalog. Offsets describe adjacent supplied clips, not playback
positions; the simple_audio flag keeps them out of model-visible evidence.
"""

from __future__ import annotations

import base64
from dataclasses import replace
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
from .requested_action import parse_requested_action


_FIELDS = {"text", "audio", "audio_source", "channel", "character_prompt", "allowed_candidate_ids",
           "context_audio", "user_context", "user_context_audio", "last_dispatched_action", "request_id", "requested_action"}
_MAX_AUDIO_BYTES = 2 * 1024 * 1024
_COMMON_FIELDS = {"channel", "character_prompt", "allowed_candidate_ids", "last_dispatched_action", "request_id", "requested_action"}
_NAMED_FIELDS = {"mode", "user_input", "avatar_current", "avatar_history"}


def _input_content(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) not in ({"text"}, {"audio"}):
        raise ValueError(f"{name} requires exactly one of text or audio")
    if "text" in value:
        text = value["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 4000 or "\x00" in text:
            raise ValueError(f"{name}.text must be nonempty and at most 4000 characters")
    elif not isinstance(value["audio"], dict):
        raise ValueError(f"{name}.audio must be an audio object")
    return value


def _normalize_named_request(raw: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Map the role-based public contract onto the existing evidence layout."""
    unknown = set(raw) - _COMMON_FIELDS - _NAMED_FIELDS
    if unknown:
        raise ValueError("named and legacy input fields cannot be mixed: " + ", ".join(sorted(unknown)))
    result = {key: value for key, value in raw.items() if key in _COMMON_FIELDS}
    mode = raw.get("mode")
    if mode == "user_reaction":
        if raw.get("avatar_current") is not None or raw.get("avatar_history") is not None:
            raise ValueError("user_reaction does not accept avatar_current or avatar_history")
        user = ({"text": ""} if raw.get("user_input") is None and raw.get("requested_action") is not None
                else _input_content(raw.get("user_input"), "user_input"))
        result.update(user)
        if "audio" in user:
            result["audio_source"] = "user_audio"
        return result, None, None
    if mode != "avatar_speech":
        raise ValueError("mode must be user_reaction or avatar_speech")
    current = _input_content(raw.get("avatar_current"), "avatar_current")
    result["audio_source"] = "avatar_speech_audio"
    if "audio" in current:
        result["audio"] = current["audio"]
    history_text = None
    if raw.get("avatar_history") is not None:
        history = _input_content(raw["avatar_history"], "avatar_history")
        if "audio" in history:
            result["context_audio"] = history["audio"]
        else:
            history_text = history["text"]
    if raw.get("user_input") is not None:
        user = _input_content(raw["user_input"], "user_input")
        result["user_context_audio" if "audio" in user else "user_context"] = user.get("audio", user.get("text"))
    return result, current.get("text"), history_text


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
    current_text = history_text = None
    if set(raw) & _NAMED_FIELDS:
        raw, current_text, history_text = _normalize_named_request(raw)
    if set(raw) - _FIELDS:
        raise ValueError("unsupported audio recall fields: " + ", ".join(sorted(set(raw) - _FIELDS)))
    channel = DecisionChannel(raw.get("channel"))
    requested_action = parse_requested_action(raw.get("requested_action"))
    user_text = raw.get("text")
    text_request = "text" in raw
    if text_request:
        if not isinstance(user_text, str) or (not user_text.strip() and requested_action is None) or len(user_text) > 4000 or "\x00" in user_text:
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
    if not text_request and current_text is None:
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
        context.update(response_id=internal_id, output_epoch=0)
        if current_text is None:
            context["speech_window"] = {"window_id": "focus", "start_ms": focus.start_ms, "end_ms": focus.end_ms}
    req = ActionDecisionRequest(
        request_id=internal_id, session_id=internal_id, turn_id=internal_id, decision_point_id=internal_id,
        channel=channel, text=user_text if text_request else user_context, character_prompt=persona,
        allowed_candidate_ids=tuple(ids), runtime_context=context, media=tuple(clips), simple_audio=True,
        avatar_current_text=current_text, avatar_history_text=history_text,
        requested_action=requested_action,
    )
    if ids or previous is not None:
        # Preserve the deployed transport-code adapter; model identities remain
        # registered candidate IDs, including prior actions outside the subset.
        lookup_context = {**context, "last_dispatched_action": None} if channel is DecisionChannel.BODY else context
        available = registry.resolve(replace(req, allowed_candidate_ids=(), runtime_context=lookup_context))
        by_code = {entry.code: entry.candidate_id for entry in available.entries if entry.kind == "action"}
        ids = [by_code.get(value, value) for value in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("allowed_candidate_ids must identify unique registered actions")
        if isinstance(previous, dict) and isinstance(previous.get("candidate_id"), str):
            previous = {**previous, "candidate_id": by_code.get(previous["candidate_id"], previous["candidate_id"])}
            req = replace(req, runtime_context={**context, "last_dispatched_action": previous})
        req = replace(req, allowed_candidate_ids=tuple(ids))
    catalog = registry.resolve(req)
    real_ids = {entry.candidate_id for entry in catalog.entries if entry.kind == "action"}
    unsupported = set(ids) - real_ids
    if unsupported:
        raise ValueError("candidates are not registered in this channel: " + ", ".join(sorted(unsupported)))
    return req, public_id
