"""Lifecycle and evidence fencing for one continuous timeline model session."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)


class TimelineDetectionModel(Protocol):
    model_id: str
    model_version: str

    async def start(self, request: TimelineSessionStart) -> None: ...

    async def append(self, chunk: TimelineMediaChunk) -> None: ...

    async def next_observations(self) -> tuple[ObservationEvent, ...]: ...

    async def discontinuity(self, event: TimelineDiscontinuity) -> None: ...

    async def close(self) -> None: ...


class TimelineDetectionSession:
    """Validate ordered full-media input and fence model observations by epoch."""

    def __init__(self, start: TimelineSessionStart, model: TimelineDetectionModel) -> None:
        if model.model_id != start.model_id:
            raise ValueError("timeline start model_id does not match the loaded model")
        self._start = start
        self._model = model
        self._stream_epoch = start.stream_epoch
        self._last_sequence = start.next_sequence - 1
        self._max_media_end_ms = 0
        self._started = False
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("timeline session is closed")
        if self._started:
            return
        await self._model.start(self._start)
        self._started = True

    async def append(self, chunk: TimelineMediaChunk) -> None:
        await self.start()
        self._validate_chunk(chunk)
        await self._model.append(chunk)
        accepted_media_end_ms = max(self._max_media_end_ms, chunk.end_ms)
        self._last_sequence = chunk.sequence
        self._max_media_end_ms = accepted_media_end_ms

    async def next_observations(self) -> tuple[ObservationEvent, ...]:
        observations = await self._model.next_observations()
        for observation in observations:
            self._validate_observation(
                observation,
                accepted_media_end_ms=self._max_media_end_ms,
            )
        return observations

    async def discontinuity(self, event: TimelineDiscontinuity) -> None:
        await self.start()
        if event.session_id != self._start.session_id:
            raise ValueError("timeline discontinuity belongs to another session")
        if event.identity_epoch != self._start.identity_epoch:
            raise ValueError("timeline discontinuity has a stale identity epoch")
        if event.observer_epoch != self._start.observer_epoch:
            raise ValueError("timeline discontinuity has a stale observer epoch")
        if event.old_stream_epoch != self._stream_epoch:
            raise ValueError("timeline discontinuity has a stale stream epoch")
        await self._model.discontinuity(event)
        self._stream_epoch = event.new_stream_epoch
        self._last_sequence = 0
        self._max_media_end_ms = 0

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            await self._model.close()

    def _validate_chunk(self, chunk: TimelineMediaChunk) -> None:
        if chunk.session_id != self._start.session_id:
            raise ValueError("timeline chunk belongs to another session")
        if chunk.identity_epoch != self._start.identity_epoch:
            raise ValueError("timeline chunk has a stale identity epoch")
        if chunk.observer_epoch != self._start.observer_epoch:
            raise ValueError("timeline chunk has a stale observer epoch")
        if chunk.stream_epoch != self._stream_epoch:
            raise ValueError("timeline chunk has a stale stream epoch")
        if chunk.sequence != self._last_sequence + 1:
            raise ValueError("timeline chunks must be contiguous")

    def _validate_observation(
        self,
        event: ObservationEvent,
        *,
        accepted_media_end_ms: int,
    ) -> None:
        if event.session_id != self._start.session_id:
            raise ValueError("observation belongs to another session")
        if event.identity_epoch != self._start.identity_epoch:
            raise ValueError("observation has a stale identity epoch")
        if event.observer_epoch != self._start.observer_epoch:
            raise ValueError("observation has a stale observer epoch")
        if event.stream_epoch != self._stream_epoch:
            raise ValueError("observation has a stale stream epoch")
        if event.evidence_end_ms > accepted_media_end_ms:
            raise ValueError("observation cites media that the service has not received")
