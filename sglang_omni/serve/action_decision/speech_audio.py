"""Validate speaker and time boundaries for bounded avatar-audio decisions.

One request contains one avatar focus clip, optionally preceded by one avatar
context clip and one original-user context clip. Each clip is independently
decodable mono PCM16 WAV (or 16 kHz raw PCM16).
This contract does not implement a rolling buffer or persistent audio inference.
"""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING

from .speech_window import SpeechWindow

if TYPE_CHECKING:
    from .contracts import ActionDecisionRequest
    from sglang_omni.serve.streaming_request import StreamedMedia


AVATAR_FOCUS_ROLE = "avatar_speech_audio"
AVATAR_CONTEXT_ROLE = "avatar_speech_context"
USER_CONTEXT_AUDIO_ROLE = "user_context_audio"
AVATAR_AUDIO_ROLES = frozenset({AVATAR_CONTEXT_ROLE, AVATAR_FOCUS_ROLE})
SPEECH_WINDOW_AUDIO_ROLES = AVATAR_AUDIO_ROLES | {USER_CONTEXT_AUDIO_ROLE}
ACTION_AUDIO_ROLES = SPEECH_WINDOW_AUDIO_ROLES | {"user_audio"}
MAX_SPEECH_AUDIO_MS = 30_000


def validate_audio_request(request: ActionDecisionRequest) -> None:
    """Fail before model admission if audio could be attributed to the wrong speaker."""
    context = request.runtime_context
    revision = context.get("persona_revision")
    if not isinstance(revision, str) or not revision.strip() or len(revision) > 256 or "\x00" in revision:
        raise ValueError("persona-audio-window-v2 requires a bounded persona_revision")
    mode = context.get("decision_mode")
    if mode == "user_request":
        if any(item.evidence_role not in {"user_audio", "user_camera"} for item in request.media):
            raise ValueError("user_request only accepts original user media")
        if any(item.kind not in {"image", "audio"} for item in request.media):
            raise ValueError("user_request expects images/audio, not a video blob")
        if any(item.evidence_role != {"audio": "user_audio", "image": "user_camera"}[item.kind]
               for item in request.media):
            raise ValueError("user_request media speaker/kind mismatch")
        if sum(item.kind == "audio" for item in request.media) > 1 or sum(
            item.kind == "image" for item in request.media
        ) > 5:
            raise ValueError("user_request permits one user audio and five user frames")
        if not (request.text or "").strip() and not any(item.kind == "audio" for item in request.media):
            raise ValueError("user_request requires original user text or audio")
        if any(key in context for key in ("response_id", "output_epoch")):
            raise ValueError("user_request must not contain reply-window identity")
        return
    if mode != "speech_window":
        raise ValueError("persona-audio-window-v2 requires a valid decision_mode")
    response_id = context.get("response_id")
    if not isinstance(response_id, str) or not response_id.strip() or len(response_id) > 256 or "\x00" in response_id:
        raise ValueError("speech_window requires a bounded response_id")
    epoch = context.get("output_epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("speech_window requires a nonnegative output_epoch")
    window = SpeechWindow.parse(context.get("speech_window"), require_text=False)
    user_context = request.media[:1] if request.media and request.media[0].evidence_role == USER_CONTEXT_AUDIO_ROLE else ()
    if user_context and request.text is not None:
        raise ValueError("user_context and user_context_audio are mutually exclusive")
    avatar_media = request.media[len(user_context):]
    roles = [item.evidence_role for item in avatar_media]
    if roles not in ([AVATAR_FOCUS_ROLE], [AVATAR_CONTEXT_ROLE, AVATAR_FOCUS_ROLE]):
        raise ValueError("speech_window requires focus audio, optionally preceded by context audio")
    if any(item.kind != "audio" for item in request.media):
        raise ValueError("speech_window only accepts labeled avatar audio and optional original-user context audio")
    for item in request.media:
        validate_avatar_audio_clip(item)
    focus = request.media[-1]
    if focus.start_ms != window.start_ms or focus.end_ms != window.end_ms:
        raise ValueError("focus audio range must equal the target speech_window")
    if len(avatar_media) == 2 and avatar_media[0].end_ms != focus.start_ms:
        raise ValueError("context audio must immediately precede focus without overlap or gap")
    if sum(item.end_ms - item.start_ms for item in request.media) > MAX_SPEECH_AUDIO_MS:
        raise ValueError("combined user context/avatar context/focus audio exceeds 30 seconds")


def validate_avatar_audio_clip(item: StreamedMedia) -> None:
    """Verify actual samples match the claimed reply-audio time interval."""
    if type(item.start_ms) is not int or type(item.end_ms) is not int or item.start_ms < 0 or item.end_ms <= item.start_ms:
        raise ValueError("avatar audio range must contain nonnegative integer boundaries")
    encoding = item.encoding.strip().lower()
    if encoding in {"pcm_s16le", "pcm16", "audio/l16"}:
        if not item.payload or len(item.payload) % 2:
            raise ValueError("avatar PCM16 requires nonempty complete samples")
        duration_ms = len(item.payload) / 32.0
    elif encoding in {"wav", "audio/wav"}:
        try:
            with wave.open(io.BytesIO(item.payload), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16_000):
                    raise ValueError("avatar WAV must be 16 kHz mono PCM16")
                frames = audio.getnframes()
                if not frames or len(audio.readframes(frames)) != frames * 2:
                    raise ValueError("avatar WAV is empty or truncated")
                duration_ms = frames / 16.0
        except (wave.Error, EOFError) as exc:
            raise ValueError("invalid avatar WAV payload") from exc
    else:
        raise ValueError("avatar audio requires WAV or 16 kHz mono PCM16")
    if abs(duration_ms - (item.end_ms - item.start_ms)) > 1.0:
        raise ValueError("avatar audio sample duration does not match its time range")
