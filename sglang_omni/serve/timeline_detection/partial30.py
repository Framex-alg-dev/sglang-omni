"""Frozen Partial30 request format; event numbers retain the direct-E50 IDs."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any
import wave

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .catalog import EVENT_SUMMARIES
from .window import EventDetectionWindow

if TYPE_CHECKING:
    from .direct_e50 import DirectE50Profile


PARTIAL30_PROTOCOL = "direct-partial30"
PARTIAL30_EVENT_IDS = frozenset({
    "E0", "E2", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12",
    "E13", "E14", "E15", "E19", "E20", "E21", "E22", "E23", "E24",
    "E32", "E33", "E34", "E36", "E37", "E38", "E44", "E46", "E47",
    "E49", "E50",
})
PARTIAL30_EVENT_SUMMARIES = {
    **{code: EVENT_SUMMARIES[code] for code in PARTIAL30_EVENT_IDS},
    "E49": "频繁看手表",
}


def _aligned_audio(window: EventDetectionWindow, first_frame_ms: int) -> bytes | None:
    """Align audio zero with the first encoded video frame, as in training."""
    if window.audio_wav is None or window.audio_start_ms is None:
        return None
    if window.audio_start_ms > first_frame_ms:
        return None
    with wave.open(io.BytesIO(window.audio_wav), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1 or source.getframerate() != 16000:
            raise ValueError("Partial30 audio must be synchronized mono 16k PCM16")
        offset = (first_frame_ms - window.audio_start_ms) * 16
        count = (window.end_ms - first_frame_ms) * 16
        if count <= 0 or offset + count > source.getnframes():
            return None
        source.setpos(offset)
        pcm = source.readframes(count)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(pcm)
    return output.getvalue()


def build_partial30_request(
    *,
    profile: DirectE50Profile,
    model_id: str,
    session_id: str,
    window: EventDetectionWindow,
    video_mp4: bytes,
    logical_request_id: str,
    contract_version: int,
    observer_epoch: int,
    stream_epoch: int,
) -> GenerateRequest:
    tagged_frames = [*(('H', frame) for frame in window.history_frames),
                     *(('C', frame) for frame in window.current_frames)]
    if not 6 <= len(tagged_frames) <= 20:
        raise ValueError("Partial30 requires 6 to 20 ordered native video frames")
    rows = [
        f"F{index:02d}|{(frame.start_ms - window.end_ms) / 1000:.3f}|{region}"
        for index, (region, frame) in enumerate(tagged_frames, 1)
    ]
    first_frame_ms = tagged_frames[0][1].start_ms
    first_relative = (first_frame_ms - window.end_ms) / 1000
    request_window = (
        "<request_window>\n"
        "时间单位：秒；本轮请求末端为相对时间0。\n"
        f"输入帧数：{len(tagged_frames)}；H帧数：{len(window.history_frames)}；"
        f"C帧数：{len(window.current_frames)}；目标采样率：{profile.video_fps:g}fps。\n"
        "下面按模型实际输入顺序列出帧号、原始画面距本轮末端的时间、区域。\n"
        "帧号|相对时间|区域\n"
        + "\n".join(rows)
        + "\n</request_window>\n"
    )
    audio = _aligned_audio(window, first_frame_ms)
    content: list[dict[str, str]] = [
        {"type": "text", "text": request_window},
        {"type": "video"},
    ]
    if audio is not None:
        content.extend([
            {"type": "text", "text": (
                "\n【同步音频：从首个输入画面时刻"
                f"{first_relative:.3f}秒至本轮末端0秒】\n"
            )},
            {"type": "audio"},
        ])
    else:
        content.append({"type": "text", "text": "\n【本轮未提供有效音频】"})
    content.append({"type": "text", "text": (
        "\n结合完整H+C确认事件成立并定位触发点，只输出触发点属于C的合法E码。"
        "触发点只在H、尚未达到条件或证据不足时输出N。"
    )})
    metadata: dict[str, Any] = {
        "task": "timeline_detection", "task_role": "timeline_detection",
        "logical_request_id": logical_request_id,
        "session_id": session_id, "session_instance_id": session_id,
        "contract_version": contract_version, "observer_epoch": observer_epoch,
        "stream_epoch": stream_epoch, "prompt_version": profile.prompt_version,
        "input_layout": "single_continuous_hc_native_video",
        "frame_manifest": rows,
        "attribution_duration_ms": 3000,
        "history_frame_count": len(window.history_frames),
        "current_frame_count": len(window.current_frames),
        "videos": ["data:video/mp4;base64," + base64.b64encode(video_mp4).decode("ascii")],
        "video_fps": 2.0, "video_max_frames": 20,
        "video_min_pixels": 131072, "video_max_pixels": 137625,
        "use_audio_in_video": False,
        "audio_start_ms": first_frame_ms if audio is not None else None,
    }
    if audio is not None:
        metadata["audios"] = ["data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")]
    return GenerateRequest(
        model=model_id,
        messages=[Message(role="system", content=profile.system_prompt),
                  Message(role="user", content=content)],
        sampling=SamplingParams(temperature=0.0, top_p=1.0, seed=42,
                                max_new_tokens=profile.max_tokens),
        stream=False, max_tokens=profile.max_tokens,
        output_modalities=["text"], metadata=metadata,
    )
