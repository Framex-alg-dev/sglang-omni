"""Prompt-only 43-event timeline detector backed by the shared Omni client."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .contracts import (
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from .parser import ModelOutputError, parse_event_output
from .prompt import (
    EVENT_SUMMARIES,
    PROMPT_VERSION,
    RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from .window import EventDetectionWindow, EventWindowAssembler


logger = logging.getLogger(__name__)


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...


@dataclass(frozen=True)
class _WindowWork:
    generation: int
    stream_epoch: int
    window: EventDetectionWindow


@dataclass(frozen=True)
class _ModelFailure:
    error: BaseException


_ModelOutput = tuple[ObservationEvent, ...] | _ModelFailure


class _EventEdgeLatch:
    """Publish one rising edge and require stable absence before re-arming."""

    def __init__(self, *, rearm_absent_windows: int) -> None:
        if rearm_absent_windows <= 0:
            raise ValueError("event rearm window count must be positive")
        self._rearm_absent_windows = rearm_absent_windows
        self._latched: set[str] = set()
        self._absent_streaks: dict[str, int] = {}

    def accept(self, event_ids: list[str]) -> tuple[str, ...]:
        present = set(event_ids)
        for event_id in tuple(self._latched):
            if event_id in present:
                self._absent_streaks[event_id] = 0
                continue
            streak = self._absent_streaks.get(event_id, 0) + 1
            if streak >= self._rearm_absent_windows:
                self._latched.remove(event_id)
                self._absent_streaks.pop(event_id, None)
            else:
                self._absent_streaks[event_id] = streak

        accepted = tuple(
            event_id for event_id in event_ids if event_id not in self._latched
        )
        for event_id in present:
            self._latched.add(event_id)
            self._absent_streaks[event_id] = 0
        return accepted

    def clear(self) -> None:
        self._latched.clear()
        self._absent_streaks.clear()


class SglangClientTimelineDetectionModel:
    """Ingest continuously and coalesce event inference to the latest window."""

    def __init__(
        self,
        client: CompletionClient,
        start: TimelineSessionStart,
        *,
        model_version: str,
        max_attempts: int = 2,
        inference_interval_ms: int = 3_000,
        window_ms: int = 10_000,
        max_window_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not model_version.strip():
            raise ValueError("timeline model_version is required")
        if max_attempts <= 0:
            raise ValueError("timeline max_attempts must be positive")
        if inference_interval_ms not in {1_000, 3_000} or window_ms != 10_000:
            raise ValueError(
                "the 43-event detector requires a 1s or 3s cadence and 10s H/C window"
            )
        if max_window_bytes <= 0:
            raise ValueError("timeline max_window_bytes must be positive")
        self._client = client
        self._start = start
        self.model_id = start.model_id
        self.model_version = model_version
        self.prompt_version = PROMPT_VERSION
        self._attribution_duration_ms = inference_interval_ms
        self._max_attempts = max_attempts
        self._assembler = EventWindowAssembler(
            audio_format=start.audio_format,
            current_duration_ms=3_000,
            history_duration_ms=window_ms - 3_000,
            window_stride_ms=inference_interval_ms,
            max_buffer_bytes=max_window_bytes,
        )
        self._event_edges = _EventEdgeLatch(
            rearm_absent_windows=max(
                2,
                (3_000 + inference_interval_ms - 1) // inference_interval_ms,
            )
        )
        self._windows: asyncio.Queue[_WindowWork] = asyncio.Queue(maxsize=1)
        self._outputs: asyncio.Queue[_ModelOutput] = asyncio.Queue(maxsize=1)
        self._worker: asyncio.Task[None] | None = None
        self._generation = 0
        self._started = False
        self._closed = False

    async def start(self, request: TimelineSessionStart) -> None:
        if self._closed:
            raise RuntimeError("timeline model is closed")
        if request != self._start:
            raise ValueError("timeline model start contract changed")
        if self._started:
            return
        self._started = True
        self._worker = asyncio.create_task(
            self._run_inference(),
            name=f"timeline-inference-{request.session_id}",
        )

    async def append(self, chunk: TimelineMediaChunk) -> None:
        if not self._started or self._closed:
            raise RuntimeError("timeline model is not active")
        for window in self._assembler.append(chunk):
            self._replace_pending_window(
                _WindowWork(self._generation, chunk.stream_epoch, window)
            )

    async def next_observations(self) -> tuple[ObservationEvent, ...]:
        output = await self._outputs.get()
        self._outputs.task_done()
        if isinstance(output, _ModelFailure):
            raise RuntimeError("timeline event inference failed") from output.error
        return output

    async def discontinuity(self, event: TimelineDiscontinuity) -> None:
        self._generation += 1
        self._assembler.clear()
        self._event_edges.clear()
        _drain(self._windows)
        _drain(self._outputs)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._assembler.clear()
        worker = self._worker
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self._worker = None
        _drain(self._windows)
        _drain(self._outputs)

    def _replace_pending_window(self, work: _WindowWork) -> None:
        try:
            self._windows.put_nowait(work)
            return
        except asyncio.QueueFull:
            self._windows.get_nowait()
            self._windows.task_done()
        self._windows.put_nowait(work)

    async def _run_inference(self) -> None:
        try:
            while True:
                work = await self._windows.get()
                try:
                    observations = await self._detect(work)
                    if work.generation == self._generation:
                        self._replace_output(observations)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    if work.generation == self._generation:
                        self._replace_output(_ModelFailure(exc))
                finally:
                    self._windows.task_done()
        except asyncio.CancelledError:
            raise

    def _replace_output(self, output: _ModelOutput) -> None:
        try:
            self._outputs.put_nowait(output)
            return
        except asyncio.QueueFull:
            self._outputs.get_nowait()
            self._outputs.task_done()
        self._outputs.put_nowait(output)

    async def _detect(self, work: _WindowWork) -> tuple[ObservationEvent, ...]:
        last_error: ModelOutputError | None = None
        logical_request_id = (
            f"timeline:{self._start.session_id}:{self._start.observer_epoch}:"
            f"{work.stream_epoch}:{work.window.end_ms}"
        )
        for attempt in range(self._max_attempts):
            request = _generate_request(
                model_id=self.model_id,
                window=work.window,
                retry=attempt > 0,
                logical_request_id=logical_request_id,
                contract_version=self._start.contract_version,
                observer_epoch=self._start.observer_epoch,
                stream_epoch=work.stream_epoch,
                attribution_duration_ms=self._attribution_duration_ms,
            )
            result = await self._client.completion(
                request,
                request_id=f"{logical_request_id}:{attempt + 1}",
            )
            raw_output = str(result.text)
            try:
                event_ids = parse_event_output(raw_output)
            except ModelOutputError as exc:
                last_error = exc
                logger.warning(
                    "timeline invalid output request_id=%s attempt=%d prompt=%s error=%s",
                    logical_request_id,
                    attempt + 1,
                    PROMPT_VERSION,
                    exc,
                )
                continue
            accepted_event_ids = self._event_edges.accept(event_ids)
            return tuple(
                _observation(
                    event_id,
                    start=self._start,
                    stream_epoch=work.stream_epoch,
                    window=work.window,
                    model_id=self.model_id,
                    model_version=self.model_version,
                )
                for event_id in accepted_event_ids
            )
        raise ModelOutputError(
            f"model did not return usable event JSON after {self._max_attempts} "
            f"attempts: {last_error}"
        )


def _generate_request(
    *,
    model_id: str,
    window: EventDetectionWindow,
    retry: bool,
    logical_request_id: str,
    contract_version: int,
    observer_epoch: int,
    stream_epoch: int,
    attribution_duration_ms: int,
) -> GenerateRequest:
    history_seconds = window.history_duration_ms / 1_000
    total_seconds = (window.end_ms - window.start_ms) / 1_000
    mode = "冷启动" if window.history_duration_ms < 7_000 else "标准稳态"
    audio_status = window.audio_status.value
    attribution_seconds = attribution_duration_ms / 1_000
    attribution_start_ms = window.end_ms - attribution_duration_ms
    current_context_frames = tuple(
        frame
        for frame in window.current_frames
        if frame.start_ms < attribution_start_ms
    )
    attribution_frames = tuple(
        frame
        for frame in window.current_frames
        if frame.start_ms >= attribution_start_ms
    )
    audio_start_ms = window.audio_start_ms
    audio_duration_seconds = (
        (window.end_ms - audio_start_ms) / 1_000
        if audio_start_ms is not None
        else 0
    )
    full_audio = audio_start_ms == window.start_ms
    if window.audio_wav is None:
        audio_description = f"{audio_status}，本次为纯视觉判断"
    elif full_audio:
        audio_description = f"完整覆盖H+C共{total_seconds:g}秒"
    else:
        audio_description = f"完整覆盖C共{audio_duration_seconds:g}秒，H无音频"
    user_prompt = (
        f"本次为{mode}窗口：H={history_seconds:g}秒/"
        f"{len(window.history_frames)}张图片，C=3秒/"
        f"{len(window.current_frames)}张图片，A=窗口最后{attribution_seconds:g}秒，"
        f"音频状态={audio_description}。\n"
        f"{USER_PROMPT}"
    )
    if retry:
        user_prompt = f"{user_prompt}\n\n{RETRY_PROMPT}"
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": (
                f"【HISTORY H｜仅作历史基线｜{history_seconds:g}秒｜"
                f"{len(window.history_frames)}张图片开始】"
            ),
        },
        *({"type": "image"} for _ in window.history_frames),
        {
            "type": "text",
            "text": "【HISTORY H结束｜以下是CURRENT C；H仍可用于完整H+C取证】",
        },
        {
            "type": "text",
            "text": (
                "【CURRENT C前段｜只作取证、不可归属本轮｜"
                f"{len(current_context_frames)}张图片开始】"
            ),
        },
        *({"type": "image"} for _ in current_context_frames),
        {
            "type": "text",
            "text": (
                "【CURRENT C前段结束｜以下图片才是ATTRIBUTION A】"
            ),
        },
        {
            "type": "text",
            "text": (
                f"【ATTRIBUTION A｜窗口最后{attribution_seconds:g}秒｜"
                "只有首次确认点落在以下图片对应时段的事件才可输出｜"
                f"{len(attribution_frames)}张图片开始】"
            ),
        },
        *({"type": "image"} for _ in attribution_frames),
        {
            "type": "text",
            "text": "【ATTRIBUTION A结束｜此前已成立的事件不得重复输出】",
        },
    ]
    if window.audio_wav is not None:
        audio_boundary = (
            f"0-{history_seconds:g}秒为H，"
            f"{history_seconds:g}-{total_seconds:g}秒为C；"
            "使用完整H+C音频取证"
            if full_audio
            else "仅覆盖CURRENT C；H只有图片，不得推测H内声音"
        )
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"【同步音频完整覆盖{audio_duration_seconds:g}秒："
                        f"{audio_boundary}，"
                        "但只把确认点落入A的事件归给本轮】"
                    ),
                },
                {"type": "audio"},
            )
        )
    else:
        content.append(
            {
                "type": "text",
                "text": (
                    f"【本窗口没有完整同步音频，状态={audio_status}；"
                    "只依据有序图片，不得推测现场安静或存在任何声音】"
                ),
            }
        )
    content.append({"type": "text", "text": user_prompt})
    frames = (*window.history_frames, *window.current_frames)
    metadata: dict[str, Any] = {
        "task": "timeline_detection",
        "task_role": "timeline_detection",
        "logical_request_id": logical_request_id,
        "contract_version": contract_version,
        "observer_epoch": observer_epoch,
        "stream_epoch": stream_epoch,
        "prompt_version": PROMPT_VERSION,
        "evidence_mode": window.evidence_mode,
        "audio_status": audio_status,
        "audio_scope": (
            "history_current"
            if full_audio
            else "current_only"
            if window.audio_wav is not None
            else "none"
        ),
        "attribution_duration_ms": attribution_duration_ms,
        "current_context_image_count": len(current_context_frames),
        "attribution_image_count": len(attribution_frames),
        "images": [_data_url("image/jpeg", frame.payload) for frame in frames],
    }
    if window.audio_wav is not None:
        metadata["audios"] = [_data_url("audio/wav", window.audio_wav)]
    return GenerateRequest(
        model=model_id,
        messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=content),
        ],
        sampling=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_new_tokens=128,
        ),
        stream=False,
        max_tokens=128,
        output_modalities=["text"],
        metadata=metadata,
    )


def _observation(
    event_id: str,
    *,
    start: TimelineSessionStart,
    stream_epoch: int,
    window: EventDetectionWindow,
    model_id: str,
    model_version: str,
) -> ObservationEvent:
    event_key = (
        f"{start.session_id}:{start.identity_epoch}:{stream_epoch}:"
        f"{window.current_start_ms}:{window.end_ms}:{event_id}:"
        f"{window.evidence_mode}:{window.audio_status.value}:{PROMPT_VERSION}"
    )
    return ObservationEvent(
        observation_id="observation-" + hashlib.sha256(
            event_key.encode()
        ).hexdigest()[:24],
        session_id=start.session_id,
        identity_epoch=start.identity_epoch,
        stream_epoch=stream_epoch,
        event_type=event_id,
        summary=EVENT_SUMMARIES[event_id],
        evidence_start_ms=window.start_ms,
        evidence_end_ms=window.end_ms,
        model_id=model_id,
        model_version=model_version,
        evidence_mode=window.evidence_mode,
        audio_status=window.audio_status.value,
        observer_epoch=start.observer_epoch,
        contract_version=start.contract_version,
    )


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def _drain(queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        queue.task_done()
