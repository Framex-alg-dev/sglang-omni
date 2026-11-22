"""V25 frozen 49-event timeline detector backed by the shared Omni client."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import time
from collections import Counter, OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

from framex_event_policy.windows import DetectionWindow, RAW_WINDOW_MODE, WindowEvent
from PIL import Image, ImageOps

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

from .artifacts import TimelineArtifactRecorder, observation_records
from .diagnostics import log_diagnostic
from .contracts import (
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from .direct_e50 import (
    DirectE50FrequencyGate,
    DirectE50Profile,
    build_direct_e50_prefill_request,
    build_direct_e50_request,
    encode_hc_video_mp4,
    parse_direct_e50_output,
)
from .parser import ModelOutputError, parse_event_output
from .partial30 import PARTIAL30_EVENT_SUMMARIES, PARTIAL30_PROTOCOL
from .event1115 import (
    EVENT1115_PROTOCOL,
    event1115_names,
    event1115_names_from_agent_snapshot,
    parse_event1115_output,
)
from .catalog import EVENT_ID_SET, OCCURRENCE_EVENT_IDS
from .prompt import (
    EVENT_SUMMARIES,
    PROMPT_VERSION,
    RETRY_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from .window import EventDetectionWindow, EventWindowAssembler


logger = logging.getLogger(__name__)


MODEL_IMAGE_WIDTH = 448
MODEL_IMAGE_HEIGHT = 252
NORMALIZED_FRAME_CACHE_SIZE = 64
DIRECT_E50_PREFILL_TIMEOUT_SECONDS = 30.0


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...

    async def prefill_completion_prefix(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
    ) -> bool: ...


@dataclass(frozen=True)
class _WindowWork:
    generation: int
    stream_epoch: int
    window: EventDetectionWindow
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class _ModelFailure:
    error: BaseException


_DetectionResult = tuple[ObservationEvent, ...] | DetectionWindow
_ModelOutput = _DetectionResult | _ModelFailure


@dataclass(frozen=True)
class _QueuedOutput:
    generation: int
    output: _ModelOutput


class _EventEdgeLatch:
    """Publish one rising edge and require stable absence before re-arming."""

    def __init__(self, *, rearm_absent_windows: int) -> None:
        if rearm_absent_windows <= 0:
            raise ValueError("event rearm window count must be positive")
        self._rearm_absent_windows = rearm_absent_windows
        self._latched: set[str] = set()
        self._absent_streaks: dict[str, int] = {}
        self._occurrence_high_water: dict[str, int] = {}

    def accept(self, event_ids: list[str]) -> tuple[str, ...]:
        counts = Counter(event_ids)
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

        accepted: list[str] = []
        for event_id in event_ids:
            if event_id in OCCURRENCE_EVENT_IDS:
                continue
            if event_id not in self._latched:
                accepted.append(event_id)
        for event_id in OCCURRENCE_EVENT_IDS:
            current = counts.get(event_id, 0)
            previous = self._occurrence_high_water.get(event_id, 0)
            if current > previous:
                accepted.extend(event_id for _ in range(current - previous))
                self._occurrence_high_water[event_id] = current
            elif current == 0 and event_id not in self._latched:
                self._occurrence_high_water.pop(event_id, None)
        for event_id in present:
            self._latched.add(event_id)
            self._absent_streaks[event_id] = 0
        return tuple(accepted)

    def clear(self) -> None:
        self._latched.clear()
        self._absent_streaks.clear()
        self._occurrence_high_water.clear()


class _V25R12Compatibility:
    """Translate V25 occurrence semantics into the current R12 E contract."""

    _ROLLING_WINDOW_MS = 10_000

    def __init__(self) -> None:
        self._occurrences: dict[str, deque[int]] = {
            "E8": deque(),
            "E49": deque(),
        }
        self._aggregate_emitted: set[str] = set()

    def translate(self, event_ids: tuple[str, ...], *, end_ms: int) -> tuple[str, ...]:
        unknown = set(event_ids) - EVENT_ID_SET
        if unknown:
            raise ModelOutputError(
                f"compatibility input is outside the 50-event catalog: {sorted(unknown)}"
            )
        self._prune(end_ms)
        translated: list[str] = []
        for event_id in event_ids:
            if event_id not in self._occurrences:
                translated.append(event_id)
                continue
            occurrences = self._occurrences[event_id]
            occurrences.append(end_ms)
            if len(occurrences) == 1:
                if event_id == "E8":
                    translated.append("E8")
                continue
            if event_id not in self._aggregate_emitted:
                translated.append("E50" if event_id == "E8" else "E49")
                self._aggregate_emitted.add(event_id)
        result = tuple(translated)
        unknown = set(result) - EVENT_ID_SET
        if unknown:
            raise ModelOutputError(
                f"compatibility output is outside the 50-event catalog: {sorted(unknown)}"
            )
        return result

    def _prune(self, end_ms: int) -> None:
        cutoff = end_ms - self._ROLLING_WINDOW_MS
        for event_id, occurrences in self._occurrences.items():
            while occurrences and occurrences[0] <= cutoff:
                occurrences.popleft()
            if not occurrences:
                self._aggregate_emitted.discard(event_id)

    def clear(self) -> None:
        for occurrences in self._occurrences.values():
            occurrences.clear()
        self._aggregate_emitted.clear()


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
        direct_profile = (
            DirectE50Profile.from_environment(
                start.prompt_snapshot,
                model_id=start.model_id,
                model_version=model_version,
            )
            if start.prompt_snapshot is not None
            else DirectE50Profile.from_environment()
        )
        first_hit_1s = direct_profile is not None and direct_profile.protocol == EVENT1115_PROTOCOL
        allowed_intervals = {500} if first_hit_1s else {1_000, 3_000}
        if inference_interval_ms not in allowed_intervals or window_ms != 10_000:
            expected_intervals = ", ".join(
                str(value) for value in sorted(allowed_intervals)
            )
            raise ValueError(
                "timeline cadence must be one of "
                f"[{expected_intervals}] ms with a 10000 ms history window"
            )
        if max_window_bytes <= 0:
            raise ValueError("timeline max_window_bytes must be positive")
        self._client = client
        self._start = start
        self._direct_e50 = direct_profile
        self._raw_windows = start.result_mode == RAW_WINDOW_MODE
        if self._raw_windows and self._direct_e50 is None:
            raise ValueError("raw windows require a direct event model")
        self._inference_interval_ms = inference_interval_ms
        self._event1115_names = (
            event1115_names_from_agent_snapshot(
                direct_profile.system_prompt,
                start.prompt_snapshot.event_registry_json,
                start.prompt_snapshot.event_codebook_csv,
            )
            if first_hit_1s and start.prompt_snapshot is not None
            else event1115_names(direct_profile.system_prompt) if first_hit_1s
            else None
        )
        self.model_id = start.model_id
        self.model_version = model_version
        self.prompt_version = (
            self._direct_e50.prompt_version
            if self._direct_e50 is not None
            else PROMPT_VERSION
        )
        self._artifacts = TimelineArtifactRecorder.from_environment(
            start=start,
            model_version=model_version,
            prompt_version=self.prompt_version,
        )
        # V25/R10 was trained with the complete trailing C=(T-3s,T] as the
        # decision range, even though the service is called every second.
        self._attribution_duration_ms = 1_000 if first_hit_1s else 3_000
        self._max_attempts = max_attempts
        self._assembler = EventWindowAssembler(
            audio_format=start.audio_format,
            current_duration_ms=self._attribution_duration_ms,
            history_duration_ms=window_ms - self._attribution_duration_ms,
            current_frame_count=2 if first_hit_1s else 6,
            allow_frame_reuse=True,
            allow_stale_current_frame_fallback=not first_hit_1s,
            window_stride_ms=inference_interval_ms,
            max_buffer_bytes=max_window_bytes,
            window_event_sink=self._artifacts.record_window_event,
        )
        self._event_edges = None if self._raw_windows else (
            DirectE50FrequencyGate(
                cooldown_ms=self._direct_e50.cooldown_ms,
                control_cooldown_ms=self._direct_e50.control_cooldown_ms,
                confirmation_windows=self._direct_e50.confirmation_windows,
                rearm_absent_windows=self._direct_e50.rearm_absent_windows,
                silent_event_ids=self._direct_e50.silent_event_ids,
                **({"allowed_event_ids": self._direct_e50.allowed_event_ids} if first_hit_1s else {}),
            )
            if self._direct_e50 is not None
            else _EventEdgeLatch(
                rearm_absent_windows=max(
                    2,
                    (3_000 + inference_interval_ms - 1) // inference_interval_ms,
                )
            )
        )
        self._compatibility = _V25R12Compatibility()
        # Adjacent one-second windows retain 18 of their 20 selected frames.
        # Keep the deterministic JPEG normalization result per media sequence
        # so those retained frames do not repeat CPU decode/resize/encode work.
        # The cache is instance-local and is cleared on every stream epoch.
        self._normalized_frames: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self._decoded_video_frames: OrderedDict[tuple[int, int], Image.Image] = OrderedDict()
        self._windows: asyncio.Queue[_WindowWork] = asyncio.Queue(maxsize=1)
        self._outputs: asyncio.Queue[_QueuedOutput] = asyncio.Queue(maxsize=1)
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
        await self._prewarm_direct_e50_prefix()
        self._worker = asyncio.create_task(
            self._run_inference(),
            name=f"timeline-inference-{request.session_id}",
        )

    async def _prewarm_direct_e50_prefix(self) -> bool:
        profile = self._direct_e50
        prefill = getattr(self._client, "prefill_completion_prefix", None)
        if profile is None or not callable(prefill):
            return False
        request_id = (
            f"timeline:{self._start.session_id}:{self._start.observer_epoch}:"
            "prefix-prefill"
        )
        request = build_direct_e50_prefill_request(
            profile=profile,
            model_id=self.model_id,
            session_id=self._start.session_id,
            logical_request_id=request_id,
        )
        started = time.perf_counter()
        try:
            ready = bool(
                await asyncio.wait_for(
                    prefill(request, request_id=request_id),
                    timeout=DIRECT_E50_PREFILL_TIMEOUT_SECONDS,
                )
            )
        except Exception as exc:
            abort = getattr(self._client, "abort", None)
            if callable(abort):
                with suppress(Exception):
                    await abort(request_id)
            logger.warning(
                "timeline prefix prefill failed session_id=%s request_id=%s "
                "error=%s",
                self._start.session_id,
                request_id,
                type(exc).__name__,
                exc_info=True,
            )
            return False
        logger.info(
            "timeline prefix prefill complete session_id=%s request_id=%s "
            "prompt_version=%s prewarmed=%s elapsed_ms=%.1f",
            self._start.session_id,
            request_id,
            profile.prompt_version,
            ready,
            (time.perf_counter() - started) * 1_000,
        )
        return ready

    async def append(self, chunk: TimelineMediaChunk) -> None:
        if not self._started or self._closed:
            raise RuntimeError("timeline model is not active")
        for window in self._assembler.append(chunk):
            self._replace_pending_window(
                _WindowWork(self._generation, chunk.stream_epoch, window)
            )

    async def next_observations(self) -> _DetectionResult:
        while True:
            queued = await self._outputs.get()
            self._outputs.task_done()
            if queued.generation == self._generation:
                break
        output = queued.output
        if isinstance(output, _ModelFailure):
            raise RuntimeError("timeline event inference failed") from output.error
        return output

    async def discontinuity(self, event: TimelineDiscontinuity) -> None:
        log_diagnostic(
            "discontinuity", session_id=self._start.session_id,
            observer_epoch=self._start.observer_epoch,
            stream_epoch=event.new_stream_epoch,
            old_stream_epoch=event.old_stream_epoch, reason=event.reason,
            pending_windows=self._windows.qsize(),
        )
        self._artifacts.record_window_event(
            {
                "status": "discontinuity",
                "old_stream_epoch": event.old_stream_epoch,
                "new_stream_epoch": event.new_stream_epoch,
                "reason": event.reason,
            }
        )
        self._generation += 1
        self._assembler.clear()
        if self._event_edges is not None:
            self._event_edges.clear()
        self._compatibility.clear()
        self._normalized_frames.clear()
        self._decoded_video_frames.clear()
        _drain(self._windows)
        _drain(self._outputs)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._assembler.clear()
        self._normalized_frames.clear()
        self._decoded_video_frames.clear()
        worker = self._worker
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        self._worker = None
        _drain(self._windows)
        _drain(self._outputs)
        await asyncio.to_thread(self._artifacts.close)

    def _replace_pending_window(self, work: _WindowWork) -> None:
        self._log_work("window_enqueued", work)
        try:
            self._windows.put_nowait(work)
            return
        except asyncio.QueueFull:
            replaced = self._windows.get_nowait()
            self._windows.task_done()
            self._log_work(
                "window_skipped", replaced, reason="superseded_before_inference",
                replacement_window_id=work.window.end_ms,
                queue_wait_ms=(time.perf_counter() - replaced.enqueued_at) * 1000,
            )
            self._artifacts.record_window_event(
                {
                    "status": "skipped",
                    "reason": "superseded_before_inference",
                    "stream_epoch": replaced.stream_epoch,
                    "start_ms": replaced.window.start_ms,
                    "current_start_ms": replaced.window.current_start_ms,
                    "end_ms": replaced.window.end_ms,
                }
            )
        self._windows.put_nowait(work)

    async def _run_inference(self) -> None:
        try:
            while True:
                work = await self._windows.get()
                self._log_work(
                    "inference_started", work,
                    queue_wait_ms=(time.perf_counter() - work.enqueued_at) * 1000,
                )
                try:
                    observations = await self._detect(work)
                    if work.generation == self._generation:
                        await self._publish_output(work.generation, observations)
                    else:
                        self._log_work(
                            "inference_result_discarded", work,
                            reason="stream_generation_changed",
                        )
                except asyncio.CancelledError:
                    self._log_work("inference_cancelled", work)
                    raise
                except BaseException as exc:
                    self._log_work(
                        "inference_failed", work, error_type=type(exc).__name__,
                    )
                    if work.generation == self._generation:
                        await self._publish_output(work.generation, _ModelFailure(exc))
                finally:
                    self._windows.task_done()
        except asyncio.CancelledError:
            raise

    async def _publish_output(self, generation: int, result: _ModelOutput) -> None:
        output = _QueuedOutput(generation, result)
        if self._raw_windows:
            await self._outputs.put(output)
        else:
            self._replace_legacy_output(output)

    def _replace_legacy_output(self, output: _QueuedOutput) -> None:
        try:
            self._outputs.put_nowait(output)
            return
        except asyncio.QueueFull:
            self._outputs.get_nowait()
            self._outputs.task_done()
        self._outputs.put_nowait(output)

    def _log_work(self, stage: str, work: _WindowWork, **fields: Any) -> None:
        log_diagnostic(
            stage, session_id=self._start.session_id,
            observer_epoch=self._start.observer_epoch, stream_epoch=work.stream_epoch,
            window_id=work.window.end_ms,
            request_id=(
                f"timeline:{self._start.session_id}:{self._start.observer_epoch}:"
                f"{work.stream_epoch}:{work.window.end_ms}"
            ),
            **fields,
        )

    async def _detect(self, work: _WindowWork) -> _DetectionResult:
        last_error: ModelOutputError | None = None
        logical_request_id = (
            f"timeline:{self._start.session_id}:{self._start.observer_epoch}:"
            f"{work.stream_epoch}:{work.window.end_ms}"
        )
        input_started = time.perf_counter()
        frames = (*work.window.history_frames, *work.window.current_frames)
        if self._direct_e50 is not None:
            video_mp4 = encode_hc_video_mp4(
                frames,
                fps=self._direct_e50.video_fps,
                preset="ultrafast" if self._event1115_names is not None else "veryfast",
                frame_cache=self._decoded_video_frames if self._event1115_names is not None else None,
            )
            image_jpegs: tuple[bytes, ...] = ()
        else:
            video_mp4 = None
            image_jpegs = tuple(self._normalized_frame_jpeg(frame) for frame in frames)
        input_build_ms = (time.perf_counter() - input_started) * 1_000
        for attempt in range(self._max_attempts):
            if self._direct_e50 is not None:
                assert video_mp4 is not None
                request = build_direct_e50_request(
                    profile=self._direct_e50,
                    model_id=self.model_id,
                    session_id=self._start.session_id,
                    window=work.window,
                    video_mp4=video_mp4,
                    retry=attempt > 0,
                    logical_request_id=logical_request_id,
                    contract_version=self._start.contract_version,
                    observer_epoch=self._start.observer_epoch,
                    stream_epoch=work.stream_epoch,
                )
            else:
                request = _generate_request(
                    model_id=self.model_id,
                    window=work.window,
                    image_jpegs=image_jpegs,
                    retry=attempt > 0,
                    logical_request_id=logical_request_id,
                    contract_version=self._start.contract_version,
                    observer_epoch=self._start.observer_epoch,
                    stream_epoch=work.stream_epoch,
                    attribution_duration_ms=self._attribution_duration_ms,
                )
            self._log_work(
                "model_request_submitted", work, attempt=attempt + 1,
                prompt_version=self.prompt_version, model_version=self.model_version,
                start_ms=work.window.start_ms,
                current_start_ms=work.window.current_start_ms,
                first_frame_ms=frames[0].start_ms if frames else None,
                last_frame_ms=frames[-1].start_ms if frames else None,
                history_frames=len(work.window.history_frames),
                current_frames=len(work.window.current_frames),
                audio_status=work.window.audio_status.value,
                assembled_audio_start_ms=work.window.audio_start_ms,
                submitted_audio=bool(request.metadata.get("audios")),
                submitted_audio_start_ms=request.metadata.get("audio_start_ms"),
                submitted_videos=len(request.metadata.get("videos", [])),
                submitted_images=len(request.metadata.get("images", [])),
                input_build_ms=round(input_build_ms, 1),
            )
            inference_started = time.perf_counter()
            self._artifacts.record_attempt_input(
                stream_epoch=work.stream_epoch, window=work.window,
                attempt=attempt + 1, request=request,
                video_mp4=video_mp4, image_jpegs=image_jpegs,
            )
            try:
                result = await self._client.completion(
                    request,
                    request_id=f"{logical_request_id}:{attempt + 1}",
                )
            except (Exception, asyncio.CancelledError) as exc:
                self._artifacts.record_attempt_result(
                    stream_epoch=work.stream_epoch, window=work.window,
                    attempt=attempt + 1,
                    result={
                        "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "model_failure",
                        "error_type": type(exc).__name__,
                        "request_id": f"{logical_request_id}:{attempt + 1}",
                        "raw_output": None,
                    },
                )
                raise
            inference_ms = (time.perf_counter() - inference_started) * 1_000
            raw_output = str(result.text)
            try:
                if self._event1115_names is not None:
                    event_ids = parse_event1115_output(
                        raw_output,
                        allowed_event_ids=frozenset(self._event1115_names),
                    )
                elif self._direct_e50 is not None:
                    event_ids = parse_direct_e50_output(raw_output, allowed_event_ids=self._direct_e50.allowed_event_ids)
                else:
                    event_ids = parse_event_output(raw_output)
            except ModelOutputError as exc:
                last_error = exc
                self._artifacts.record_attempt_result(
                    stream_epoch=work.stream_epoch, window=work.window,
                    attempt=attempt + 1,
                    result={
                        "status": "invalid_output", "raw_output": raw_output,
                        "error_type": type(exc).__name__,
                        "request_id": f"{logical_request_id}:{attempt + 1}",
                        "inference_ms": inference_ms,
                    },
                )
                self._log_work(
                    "inference_invalid_output", work, attempt=attempt + 1,
                    raw_output=raw_output[:512], error_type=type(exc).__name__,
                    inference_ms=round(inference_ms, 1),
                )
                logger.warning(
                    "timeline invalid output request_id=%s attempt=%d prompt=%s error=%s",
                    logical_request_id,
                    attempt + 1,
                    self.prompt_version,
                    exc,
                )
                continue
            logger.info(
                "timeline inference complete request_id=%s attempt=%d "
                "input_layout=%s visual_preprocess=%s "
                "history_frames=%d current_frames=%d input_build_ms=%.1f "
                "inference_ms=%.1f total_ms=%.1f raw_output=%r",
                logical_request_id,
                attempt + 1,
                (
                    "single_continuous_hc_native_video"
                    if self._direct_e50 is not None
                    else "ordered_hc_images"
                ),
                (
                    f"dynamic-native-video(max_pixels={self._direct_e50.video_max_pixels})"
                    if self._direct_e50 is not None
                    else f"fixed-images({MODEL_IMAGE_WIDTH}x{MODEL_IMAGE_HEIGHT})"
                ),
                len(work.window.history_frames),
                len(work.window.current_frames),
                input_build_ms,
                inference_ms,
                input_build_ms + inference_ms,
                raw_output,
            )
            if self._raw_windows:
                frequency_decision = None
                accepted_event_ids = event_ids
            elif self._direct_e50 is not None:
                assert isinstance(self._event_edges, DirectE50FrequencyGate)
                frequency_decision = self._event_edges.evaluate(
                    event_ids, end_ms=work.window.end_ms
                )
                accepted_event_ids = frequency_decision.accepted_event_ids
                if event_ids:
                    logger.info(
                        "timeline frequency decision request_id=%s decisions=%s",
                        logical_request_id,
                        frequency_decision.to_record(),
                    )
            else:
                frequency_decision = None
                assert isinstance(self._event_edges, _EventEdgeLatch)
                accepted_event_ids = self._compatibility.translate(
                    self._event_edges.accept(event_ids),
                    end_ms=work.window.end_ms,
                )
            observations = tuple(
                _observation(
                    event_id,
                    start=self._start,
                    stream_epoch=work.stream_epoch,
                    window=work.window,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    prompt_version=self.prompt_version,
                    summary=(
                        self._event1115_names[event_id]
                        if self._event1115_names is not None
                        else PARTIAL30_EVENT_SUMMARIES[event_id]
                        if self._direct_e50 is not None and self._direct_e50.protocol == PARTIAL30_PROTOCOL
                        else None
                    ),
                )
                for event_id in accepted_event_ids
            )
            suppressed_counts = Counter(event_ids)
            suppressed_counts.subtract(Counter(accepted_event_ids))
            suppressed_event_ids = [
                event_id
                for event_id, count in suppressed_counts.items()
                for _ in range(max(0, count))
            ]
            detection_window = (
                DetectionWindow(
                    session_id=self._start.session_id,
                    identity_epoch=self._start.identity_epoch,
                    observer_epoch=self._start.observer_epoch,
                    stream_epoch=work.stream_epoch,
                    window_sequence=work.window.end_ms // self._inference_interval_ms,
                    evidence_start_ms=work.window.start_ms,
                    evidence_end_ms=work.window.end_ms,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    evidence_mode=work.window.evidence_mode,
                    audio_status=work.window.audio_status.value,
                    events=tuple(WindowEvent(event.event_type, event.summary) for event in observations),
                )
                if self._raw_windows else None
            )
            artifact_result = {
                "status": (
                    "raw_events" if self._raw_windows and event_ids
                    else "emitted"
                    if accepted_event_ids
                    else "frequency_suppressed"
                    if event_ids
                    else "no_event"
                ),
                "raw_output": raw_output,
                "parsed_event_ids": event_ids,
                "accepted_event_ids": list(accepted_event_ids),
                "suppressed_event_ids": suppressed_event_ids,
                "observations": observation_records(observations),
                "input_build_ms": input_build_ms,
                "inference_ms": inference_ms,
                "total_ms": input_build_ms + inference_ms,
                "usage": (
                    getattr(result, "usage").to_dict()
                    if getattr(result, "usage", None) is not None
                    else None
                ),
                "weight_version": getattr(result, "weight_version", None),
                "window_result": detection_window.to_wire() if detection_window is not None else None,
            }
            if frequency_decision is not None:
                artifact_result["frequency_gate"] = frequency_decision.to_record()
            self._log_work(
                "inference_result", work, attempt=attempt + 1,
                status=artifact_result["status"], raw_output=raw_output[:512],
                parsed_event_ids=event_ids,
                accepted_event_ids=list(accepted_event_ids),
                suppressed_event_ids=suppressed_event_ids,
                frequency_gate=(
                    frequency_decision.to_record() if frequency_decision is not None
                    else None
                ),
                input_build_ms=round(input_build_ms, 1),
                inference_ms=round(inference_ms, 1),
                total_ms=round(input_build_ms + inference_ms, 1),
                enqueued_to_result_ms=round(
                    (time.perf_counter() - work.enqueued_at) * 1000, 1
                ),
            )
            if self._artifacts.record_all:
                self._artifacts.record_attempt_result(
                    stream_epoch=work.stream_epoch, window=work.window,
                    attempt=attempt + 1, result=artifact_result,
                )
            elif event_ids:
                try:
                    self._artifacts.record_event_call(
                        stream_epoch=work.stream_epoch,
                        window=work.window,
                        attempt=attempt + 1,
                        request=request,
                        video_mp4=video_mp4,
                        image_jpegs=image_jpegs,
                        audio_wav=(
                            base64.b64decode(request.metadata["audios"][0].split(",", 1)[1])
                            if request.metadata.get("audios")
                            else None
                        ),
                        result=artifact_result,
                    )
                except Exception:
                    logger.exception(
                        "timeline event artifact enqueue failed "
                        "request_id=%s attempt=%d",
                        logical_request_id,
                        attempt + 1,
                    )
            if detection_window is not None:
                return detection_window
            return observations
        if self._event1115_names is not None and not self._raw_windows:
            logger.warning(
                "timeline window rejected invalid model output request_id=%s",
                logical_request_id,
            )
            return ()
        raise ModelOutputError(
            f"model did not return usable event JSON after {self._max_attempts} "
            f"attempts: {last_error}"
        )

    def _normalized_frame_jpeg(self, frame: TimelineMediaChunk) -> bytes:
        key = (frame.stream_epoch, frame.sequence)
        cached = self._normalized_frames.get(key)
        if cached is not None:
            self._normalized_frames.move_to_end(key)
            return cached
        normalized = _normalize_frame_jpeg(frame.payload)
        self._normalized_frames[key] = normalized
        if len(self._normalized_frames) > NORMALIZED_FRAME_CACHE_SIZE:
            self._normalized_frames.popitem(last=False)
        return normalized


def _generate_request(
    *,
    model_id: str,
    window: EventDetectionWindow,
    image_jpegs: tuple[bytes, ...],
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
        f"{len(window.current_frames)}张图片，完整C={attribution_seconds:g}秒均为本轮判定范围，"
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
            "text": "【HISTORY H结束｜与CURRENT C连续、无重叠无间隙】",
        },
        {
            "type": "text",
            "text": (
                f"【CURRENT C开始｜{attribution_seconds:g}秒｜"
                "完整C=(T-3s,T]均为本轮判定范围｜"
                f"{len(window.current_frames)}张图片开始】"
            ),
        },
        *({"type": "image"} for _ in window.current_frames),
        {
            "type": "text",
            "text": "【CURRENT C结束｜H只作取证，达标点落在完整C内即可输出】",
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
                        "完整CURRENT C均为本轮判定范围】"
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
        "input_layout": "ordered_hc_images",
        "attribution_duration_ms": attribution_duration_ms,
        "history_frame_count": len(window.history_frames),
        "current_frame_count": len(window.current_frames),
        "image_width": MODEL_IMAGE_WIDTH,
        "image_height": MODEL_IMAGE_HEIGHT,
        "images": [_data_url("image/jpeg", payload) for payload in image_jpegs],
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
            max_new_tokens=48,
        ),
        stream=False,
        max_tokens=48,
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
    prompt_version: str,
    summary: str | None = None,
) -> ObservationEvent:
    event_key = (
        f"{start.session_id}:{start.identity_epoch}:{stream_epoch}:"
        f"{window.current_start_ms}:{window.end_ms}:{event_id}:"
        f"{window.evidence_mode}:{window.audio_status.value}:{prompt_version}"
    )
    return ObservationEvent(
        observation_id="observation-" + hashlib.sha256(
            event_key.encode()
        ).hexdigest()[:24],
        session_id=start.session_id,
        identity_epoch=start.identity_epoch,
        stream_epoch=stream_epoch,
        event_type=event_id,
        summary=summary or EVENT_SUMMARIES[event_id],
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


def _normalize_frame_jpeg(payload: bytes) -> bytes:
    """Fit a camera frame into the V25 visual budget without cropping people."""

    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.pad(
            source.convert("RGB"),
            (MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT),
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0),
        )
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


def _drain(queue: asyncio.Queue[Any]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        queue.task_done()
