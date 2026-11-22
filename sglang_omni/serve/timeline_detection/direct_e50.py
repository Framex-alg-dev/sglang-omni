"""Checkpoint-1500 direct-E50 inference profile and product frequency policy."""

from __future__ import annotations

import base64
from collections import OrderedDict
import io
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
from PIL import Image

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .catalog import EVENT_ID_SET
from .contracts import TimelineMediaChunk
from .parser import ModelOutputError, normalize_event_ids
from .partial30 import PARTIAL30_EVENT_IDS, PARTIAL30_PROTOCOL, build_partial30_request
from .event1115 import EVENT1115_IDS, EVENT1115_PROTOCOL, event1115_names
from .window import EventDetectionWindow


DIRECT_E50_PROTOCOL = "direct-e50"
DIRECT_E50_OCCURRENCE_IDS = frozenset({"E22", "E38", "E44"})
DIRECT_E50_CONTROL_EVENT_IDS = frozenset({"E20", "E21"})
# These events are intentionally retained in detector artifacts and forwarded
# to the product boundary, where they are hidden from the user.  They must not
# reserve the global slot used by actionable events.
DIRECT_E50_SILENT_EVENT_IDS = frozenset(
    {
        "E1",
        "E3",
        "E4",
        "E12",
        "E16",
        "E17",
        "E18",
        "E25",
        "E26",
        "E29",
        "E30",
        "E31",
        "E35",
        "E39",
        "E41",
        "E42",
        "E43",
        "E45",
        "E48",
    }
)


@dataclass(frozen=True)
class DirectE50EventFrequencyDecision:
    event_id: str
    event_class: str
    disposition: str
    reason: str
    cooldown_remaining_ms: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_class": self.event_class,
            "disposition": self.disposition,
            "reason": self.reason,
            "cooldown_remaining_ms": self.cooldown_remaining_ms,
        }


@dataclass(frozen=True)
class DirectE50FrequencyDecision:
    accepted_event_ids: tuple[str, ...]
    event_decisions: tuple[DirectE50EventFrequencyDecision, ...]
    ordinary_latched: bool
    ordinary_clear_streak: int

    def to_record(self) -> dict[str, Any]:
        return {
            "accepted_event_ids": list(self.accepted_event_ids),
            "event_decisions": [
                decision.to_record() for decision in self.event_decisions
            ],
            "ordinary_latched": self.ordinary_latched,
            "ordinary_clear_streak": self.ordinary_clear_streak,
        }


@dataclass(frozen=True)
class DirectE50Profile:
    system_prompt: str
    user_prompt: str
    prompt_version: str
    max_tokens: int = 8
    video_fps: float = 2.0
    video_max_frames: int = 20
    video_max_pixels: int = 130_000
    cooldown_ms: int = 15_000
    control_cooldown_ms: int = 3_000
    confirmation_windows: int = 2
    rearm_absent_windows: int = 3
    silent_event_ids: frozenset[str] = DIRECT_E50_SILENT_EVENT_IDS
    protocol: str = DIRECT_E50_PROTOCOL
    event_ids: frozenset[str] | None = None

    @property
    def allowed_event_ids(self) -> frozenset[str]:
        if self.protocol == EVENT1115_PROTOCOL:
            return (
                self.event_ids
                if self.event_ids is not None
                else frozenset(event1115_names(self.system_prompt))
            )
        return PARTIAL30_EVENT_IDS if self.protocol == PARTIAL30_PROTOCOL else EVENT_ID_SET

    @classmethod
    def from_environment(
        cls,
        prompt_snapshot=None,
        *,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> DirectE50Profile | None:
        protocol = os.environ.get(
            "SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL", "v25-alias"
        ).strip()
        if protocol == "v25-alias" and prompt_snapshot is not None:
            raise ValueError("Agent timeline prompt requires direct-1115-v1")
        if protocol == "v25-alias":
            return None
        if protocol not in {DIRECT_E50_PROTOCOL, PARTIAL30_PROTOCOL, EVENT1115_PROTOCOL}:
            raise ValueError(
                "Unsupported SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL"
            )
        native_video = protocol in {PARTIAL30_PROTOCOL, EVENT1115_PROTOCOL}
        if prompt_snapshot is not None:
            if protocol != EVENT1115_PROTOCOL:
                raise ValueError("Agent timeline prompt requires direct-1115-v1")
            if (
                prompt_snapshot.binding.model_id != model_id
                or prompt_snapshot.binding.model_version != model_version
            ):
                raise ValueError("Agent timeline prompt serving model mismatch")
            from .event1115 import event1115_names_from_agent_snapshot
            system_prompt = prompt_snapshot.system_prompt
            agent_event_names = event1115_names_from_agent_snapshot(
                system_prompt,
                prompt_snapshot.event_registry_json,
                prompt_snapshot.event_codebook_csv,
            )
        else:
            system_prompt = _read_prompt(
                "SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH", "system",
                preserve_whitespace=native_video,
            )
            if protocol == EVENT1115_PROTOCOL:
                local_event_names = event1115_names(system_prompt)
        user_prompt = "" if native_video else _read_prompt(
            "SGLANG_OMNI_TIMELINE_USER_PROMPT_PATH", "user"
        )
        prompt_version = (
            prompt_snapshot.release_id
            if prompt_snapshot is not None
            else os.environ.get("SGLANG_OMNI_TIMELINE_PROMPT_VERSION", "").strip()
        )
        if not prompt_version:
            raise ValueError("direct-e50 requires a timeline prompt version")
        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=prompt_version,
            protocol=protocol,
            event_ids=(
                frozenset(agent_event_names)
                if prompt_snapshot is not None
                else frozenset(local_event_names)
                if protocol == EVENT1115_PROTOCOL
                else None
            ),
            max_tokens=_positive_int("SGLANG_OMNI_TIMELINE_MAX_TOKENS", 128 if native_video else 8),
            video_max_pixels=137625 if native_video else 130000,
            cooldown_ms=_positive_int(
                "SGLANG_OMNI_TIMELINE_EVENT_COOLDOWN_MS", 15_000
            ),
            control_cooldown_ms=_positive_int(
                "SGLANG_OMNI_TIMELINE_CONTROL_EVENT_COOLDOWN_MS", 3_000
            ),
            confirmation_windows=_positive_int(
                "SGLANG_OMNI_TIMELINE_CONFIRMATION_WINDOWS", 2
            ),
            rearm_absent_windows=_positive_int(
                "SGLANG_OMNI_TIMELINE_REARM_ABSENT_WINDOWS", 3
            ),
            silent_event_ids=_event_id_set(
                "SGLANG_OMNI_TIMELINE_SILENT_EVENT_IDS",
                DIRECT_E50_SILENT_EVENT_IDS,
            ),
        )


class DirectE50FrequencyGate:
    """Apply control, silent, and ordinary event frequency semantics.

    Every event must first appear in the configured number of consecutive
    inference windows.  A missing window resets that event's confirmation
    streak, so separate one-off guesses never combine into a trigger.

    Ordinary events share a global latch so the tail of one event cannot be
    published as a different event without an explicit absence run.  Control
    events bypass that latch so E20/E21 can always stop active output.  Silent
    events retain independent edge/cooldown suppression but never occupy or
    extend the ordinary global latch.
    """

    def __init__(
        self,
        *,
        cooldown_ms: int,
        confirmation_windows: int,
        rearm_absent_windows: int,
        control_cooldown_ms: int = 3_000,
        silent_event_ids: frozenset[str] = DIRECT_E50_SILENT_EVENT_IDS,
        allowed_event_ids: frozenset[str] = EVENT_ID_SET,
    ) -> None:
        if (
            cooldown_ms <= 0
            or control_cooldown_ms <= 0
            or confirmation_windows <= 0
            or rearm_absent_windows <= 0
        ):
            raise ValueError("direct E50 frequency limits must be positive")
        unknown_silent_ids = silent_event_ids - allowed_event_ids
        if unknown_silent_ids:
            raise ValueError(
                "direct E50 silent event ids are outside the catalog: "
                f"{sorted(unknown_silent_ids)}"
            )
        overlap = silent_event_ids & DIRECT_E50_CONTROL_EVENT_IDS
        if overlap:
            raise ValueError(
                "direct E50 events cannot be both control and silent: "
                f"{sorted(overlap)}"
            )
        self._cooldown_ms = cooldown_ms
        self._control_cooldown_ms = control_cooldown_ms
        self._confirmation_windows = confirmation_windows
        self._rearm_absent_windows = rearm_absent_windows
        self._silent_event_ids = silent_event_ids
        self._allowed_event_ids = allowed_event_ids
        self._independent_latched: set[str] = set()
        self._independent_absent_streaks: dict[str, int] = {}
        self._ordinary_latched = False
        self._ordinary_clear_streak = 0
        self._cooldown_until: dict[str, int] = {}
        self._confirmation_streaks: dict[str, int] = {}

    def accept(self, event_ids: list[str], *, end_ms: int) -> tuple[str, ...]:
        return self.evaluate(event_ids, end_ms=end_ms).accepted_event_ids

    def evaluate(
        self, event_ids: list[str], *, end_ms: int
    ) -> DirectE50FrequencyDecision:
        unknown_ids = set(event_ids) - self._allowed_event_ids
        if unknown_ids:
            raise ValueError(
                "direct E50 frequency input is outside the catalog: "
                f"{sorted(unknown_ids)}"
            )
        if end_ms < 0:
            raise ValueError("direct E50 frequency timestamp must be non-negative")

        present = set(event_ids)
        for event_id in tuple(self._confirmation_streaks):
            if event_id not in present:
                self._confirmation_streaks.pop(event_id, None)
        independent_ids = (
            DIRECT_E50_CONTROL_EVENT_IDS | self._silent_event_ids
        )
        for event_id in tuple(self._independent_latched):
            if event_id in present:
                self._independent_absent_streaks[event_id] = 0
                continue
            streak = self._independent_absent_streaks.get(event_id, 0) + 1
            if streak >= self._rearm_absent_windows:
                self._independent_latched.remove(event_id)
                self._independent_absent_streaks.pop(event_id, None)
            else:
                self._independent_absent_streaks[event_id] = streak

        ordinary_present = bool(present - independent_ids)
        ordinary_was_latched = self._ordinary_latched
        if ordinary_present:
            self._ordinary_clear_streak = 0
        elif self._ordinary_latched:
            self._ordinary_clear_streak += 1
            if self._ordinary_clear_streak >= self._rearm_absent_windows:
                self._ordinary_latched = False
                self._ordinary_clear_streak = 0

        accepted: list[str] = []
        accepted_independent: set[str] = set()
        ordinary_should_latch = False
        decisions: list[DirectE50EventFrequencyDecision] = []
        seen: set[str] = set()
        for event_id in event_ids:
            event_class = self._event_class(event_id)
            if event_id in seen:
                decisions.append(
                    DirectE50EventFrequencyDecision(
                        event_id=event_id,
                        event_class=event_class,
                        disposition="suppressed",
                        reason="duplicate_in_window",
                    )
                )
                continue
            seen.add(event_id)

            cooldown_until = self._cooldown_until.get(event_id, 0)
            cooldown_remaining_ms = max(0, cooldown_until - end_ms)
            if event_class != "ordinary" and event_id in self._independent_latched:
                disposition = "suppressed"
                reason = "continuous_event"
            elif event_class == "ordinary" and ordinary_was_latched:
                disposition = "suppressed"
                reason = "ordinary_window_latched"
            elif cooldown_remaining_ms:
                disposition = "suppressed"
                reason = "event_cooldown"
                if event_class == "ordinary":
                    ordinary_should_latch = True
            else:
                confirmation_streak = (
                    self._confirmation_streaks.get(event_id, 0) + 1
                )
                if confirmation_streak < self._confirmation_windows:
                    self._confirmation_streaks[event_id] = confirmation_streak
                    disposition = "suppressed"
                    reason = "awaiting_confirmation"
                else:
                    self._confirmation_streaks.pop(event_id, None)
                    disposition = "accepted"
                    reason = "accepted"
                    accepted.append(event_id)
                    if event_class == "ordinary":
                        ordinary_should_latch = True
                    else:
                        accepted_independent.add(event_id)
                    cooldown_ms = (
                        self._control_cooldown_ms
                        if event_class == "control"
                        else self._cooldown_ms
                    )
                    self._cooldown_until[event_id] = end_ms + cooldown_ms
            if reason not in {"awaiting_confirmation", "accepted"}:
                self._confirmation_streaks.pop(event_id, None)
            decisions.append(
                DirectE50EventFrequencyDecision(
                    event_id=event_id,
                    event_class=event_class,
                    disposition=disposition,
                    reason=reason,
                    cooldown_remaining_ms=(
                        cooldown_remaining_ms
                        if reason == "event_cooldown"
                        else 0
                    ),
                )
            )

        for event_id in accepted_independent:
            self._independent_latched.add(event_id)
            self._independent_absent_streaks[event_id] = 0
        if ordinary_should_latch:
            # Even a cooldown-suppressed ordinary event protects the tail from
            # being reclassified as a different actionable event.
            self._ordinary_latched = True

        return DirectE50FrequencyDecision(
            accepted_event_ids=tuple(accepted),
            event_decisions=tuple(decisions),
            ordinary_latched=self._ordinary_latched,
            ordinary_clear_streak=self._ordinary_clear_streak,
        )

    def _event_class(self, event_id: str) -> str:
        if event_id in DIRECT_E50_CONTROL_EVENT_IDS:
            return "control"
        if event_id in self._silent_event_ids:
            return "silent"
        return "ordinary"

    def clear(self) -> None:
        self._independent_latched.clear()
        self._independent_absent_streaks.clear()
        self._ordinary_latched = False
        self._ordinary_clear_streak = 0
        self._cooldown_until.clear()
        self._confirmation_streaks.clear()


def parse_direct_e50_output(
    raw_output: str, *, allowed_event_ids: frozenset[str] = EVENT_ID_SET,
) -> list[str]:
    """Validate the checkpoint's native E-code protocol without P-code fallback."""

    text = raw_output.strip()
    if text == "N":
        return []
    if not text or any(character.isspace() for character in text):
        raise ModelOutputError("direct E50 output must not contain whitespace")
    tokens = text.split(",")
    if len(tokens) > 16:
        raise ModelOutputError("direct E50 output contains too many tokens")
    invalid = next((token for token in tokens if token not in allowed_event_ids), None)
    if invalid is not None or "E27" in tokens:
        raise ModelOutputError(f"invalid direct E50 token: {invalid or 'E27'}")
    if "E8" in tokens and "E50" in tokens:
        raise ModelOutputError("E8 and E50 are mutually exclusive")
    counts = {token: tokens.count(token) for token in set(tokens)}
    invalid_repeat = next(
        (
            token
            for token, count in counts.items()
            if count > 1 and token not in DIRECT_E50_OCCURRENCE_IDS
        ),
        None,
    )
    if invalid_repeat is not None:
        raise ModelOutputError(f"direct E50 token cannot repeat: {invalid_repeat}")
    return normalize_event_ids(tokens)


def encode_hc_video_mp4(
    frames: tuple[TimelineMediaChunk, ...],
    *,
    fps: float,
    preset: str = "veryfast",
    frame_cache: OrderedDict[tuple[int, int], Image.Image] | None = None,
) -> bytes:
    """Encode selected right-aligned H+C frames like the evaluated live service."""

    if not frames:
        raise ValueError("direct E50 video requires at least one frame")
    decoded: list[Image.Image] = []
    owned: list[Image.Image] = []
    try:
        for item in frames:
            key = (item.stream_epoch, item.sequence)
            image = frame_cache.get(key) if frame_cache is not None else None
            if image is None:
                with Image.open(io.BytesIO(item.payload)) as source:
                    image = source.convert("RGB")
                if frame_cache is None or image.width * image.height > 1280 * 720:
                    owned.append(image)
                else:
                    frame_cache[key] = image
            else:
                frame_cache.move_to_end(key)
            decoded.append(image)
        width, height = decoded[0].size
        width -= width % 2
        height -= height % 2
        if width <= 0 or height <= 0:
            raise ValueError("direct E50 video frame dimensions are invalid")
        output = io.BytesIO()
        with av.open(output, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=Fraction(round(fps), 1))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "0", "preset": preset}
            for image in decoded:
                if image.size != (width, height):
                    image = image.resize((width, height))
                frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return output.getvalue()
    finally:
        for image in owned:
            image.close()
        if frame_cache is not None:
            # Pillow RGB storage can use four bytes per pixel. Trim after
            # encoding so an evicted image cannot still belong to this window.
            bytes_used = sum(image.width * image.height * 4 for image in frame_cache.values())
            while len(frame_cache) > 24 or bytes_used > 32 * 1024 * 1024:
                _, image = frame_cache.popitem(last=False)
                bytes_used -= image.width * image.height * 4
                image.close()


def build_direct_e50_prefill_request(
    *,
    profile: DirectE50Profile,
    model_id: str,
    session_id: str,
    logical_request_id: str,
) -> GenerateRequest:
    """Build a media-free request that warms the immutable training prefix."""

    return GenerateRequest(
        model=model_id,
        messages=[
            Message(role="system", content=profile.system_prompt),
            # Preserve the user-role boundary shared by live requests while
            # keeping dynamic window text and all media request-local.
            Message(role="user", content=[{"type": "text", "text": " "}]),
        ],
        sampling=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=42 if profile.protocol in {PARTIAL30_PROTOCOL, EVENT1115_PROTOCOL} else 0,
            max_new_tokens=1,
        ),
        stream=False,
        max_tokens=1,
        output_modalities=["text"],
        metadata={
            "task": "timeline_detection",
            "task_role": "timeline_detection",
            "logical_request_id": logical_request_id,
            "prompt_version": profile.prompt_version,
            "session_id": session_id,
            "session_instance_id": session_id,
            "input_layout": "static_system_prefix_only",
            "videos": [],
            "audios": [],
            "prefill_only": True,
        },
    )


def build_direct_e50_request(
    *,
    profile: DirectE50Profile,
    model_id: str,
    session_id: str,
    window: EventDetectionWindow,
    video_mp4: bytes,
    retry: bool,
    logical_request_id: str,
    contract_version: int,
    observer_epoch: int,
    stream_epoch: int,
) -> GenerateRequest:
    if profile.protocol in {PARTIAL30_PROTOCOL, EVENT1115_PROTOCOL}:
        return build_partial30_request(
            profile=profile, model_id=model_id, session_id=session_id,
            window=window, video_mp4=video_mp4,
            logical_request_id=logical_request_id, contract_version=contract_version,
            observer_epoch=observer_epoch, stream_epoch=stream_epoch,
        )
    history_seconds = window.history_duration_ms / 1_000
    call_seconds = window.end_ms / 1_000
    history_frames = len(window.history_frames)
    current_frames = len(window.current_frames)
    full_audio = (
        window.audio_wav is not None and window.audio_start_ms == window.start_ms
    )
    user_prompt = profile.user_prompt
    if retry:
        user_prompt += "\n只输出事件表中的合法E码或N，不输出解释。"
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "【连续H+C视频开始｜窗口截止"
                f"T={call_seconds:g}秒｜{profile.video_fps:g}fps右端对齐帧】\n"
            ),
        },
        {"type": "video"},
        {
            "type": "text",
            "text": (
                "\n【连续H+C视频结束｜前"
                f"{history_seconds:g}秒/{history_frames}帧=HISTORY H，只作上下文；"
                f"后3秒/{current_frames}帧=CURRENT C；"
                "完整C=(T-3s,T]均为本轮判定范围】\n"
                + (
                    "【同步音频｜覆盖完整H+C】\n"
                    if full_audio
                    else "【同步音频缺失】\n"
                )
            ),
        },
    ]
    if full_audio:
        content.append({"type": "audio"})
    # A media placeholder is surrounded by newlines in the training text.  For
    # video-only samples the preceding text already ends in a newline, so an
    # additional leading newline would create a prompt that was never trained.
    content.append(
        {
            "type": "text",
            "text": ("\n" if full_audio else "") + user_prompt,
        }
    )

    metadata: dict[str, Any] = {
        "task": "timeline_detection",
        "task_role": "timeline_detection",
        "logical_request_id": logical_request_id,
        "contract_version": contract_version,
        "observer_epoch": observer_epoch,
        "stream_epoch": stream_epoch,
        "prompt_version": profile.prompt_version,
        "session_id": session_id,
        "session_instance_id": session_id,
        "input_layout": "single_continuous_hc_native_video",
        "attribution_duration_ms": 3_000,
        "history_frame_count": history_frames,
        "current_frame_count": current_frames,
        "videos": [_data_url("video/mp4", video_mp4)],
        "video_fps": profile.video_fps,
        "video_max_frames": profile.video_max_frames,
        "video_max_pixels": profile.video_max_pixels,
    }
    if full_audio:
        metadata["audios"] = [_data_url("audio/wav", window.audio_wav or b"")]
    return GenerateRequest(
        model=model_id,
        messages=[
            Message(role="system", content=profile.system_prompt),
            Message(role="user", content=content),
        ],
        sampling=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_new_tokens=profile.max_tokens,
        ),
        stream=False,
        max_tokens=profile.max_tokens,
        output_modalities=["text"],
        metadata=metadata,
    )


def _read_prompt(environment_name: str, label: str, *, preserve_whitespace: bool = False) -> str:
    raw_path = os.environ.get(environment_name, "").strip()
    if not raw_path:
        raise ValueError(f"direct-e50 requires a {label} prompt path")
    value = Path(raw_path).read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"direct-e50 {label} prompt is empty")
    return value if preserve_whitespace else value.strip()


def _positive_int(environment_name: str, default: int) -> int:
    value = int(os.environ.get(environment_name, str(default)))
    if value <= 0:
        raise ValueError(f"{environment_name} must be positive")
    return value


def _event_id_set(
    environment_name: str, default: frozenset[str]
) -> frozenset[str]:
    raw_value = os.environ.get(environment_name)
    if raw_value is None:
        return default
    event_ids = frozenset(
        token.strip() for token in raw_value.split(",") if token.strip()
    )
    unknown_ids = event_ids - EVENT_ID_SET
    if unknown_ids:
        raise ValueError(
            f"{environment_name} contains unknown event ids: {sorted(unknown_ids)}"
        )
    return event_ids


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
