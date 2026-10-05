from __future__ import annotations

import asyncio

import pytest

from sglang_omni.serve.timeline_detection.contracts import (
    MediaKind,
    ObservationEvent,
    TimelineDiscontinuity,
    TimelineMediaChunk,
    TimelineSessionStart,
)
from sglang_omni.serve.timeline_detection.session import TimelineDetectionSession


class _Model:
    model_id = "timeline-model"
    model_version = "2026-09-25"

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.observations = ()
        self.queue = asyncio.Queue()

    async def start(self, request) -> None:
        del request
        self.started += 1

    async def append(self, chunk):
        del chunk
        if self.observations:
            await self.queue.put(self.observations)

    async def next_observations(self):
        return await self.queue.get()

    async def discontinuity(self, event) -> None:
        del event

    async def close(self) -> None:
        self.closed += 1


def _start() -> TimelineSessionStart:
    return TimelineSessionStart(
        session_id="session-1",
        identity_epoch=3,
        stream_epoch=1,
        audio_format="pcm16/16000/mono",
        video_format="h264/annex-b",
        model_id="timeline-model",
        observer_epoch=4,
    )


def _chunk(sequence: int, *, stream_epoch: int = 1) -> TimelineMediaChunk:
    return TimelineMediaChunk(
        session_id="session-1",
        identity_epoch=3,
        stream_epoch=stream_epoch,
        sequence=sequence,
        kind=MediaKind.VIDEO,
        start_ms=(sequence - 1) * 40,
        end_ms=sequence * 40,
        encoding="h264/annex-b",
        payload=b"frame",
        observer_epoch=4,
    )


@pytest.mark.asyncio
async def test_requires_contiguous_full_media_sequences() -> None:
    session = TimelineDetectionSession(_start(), _Model())
    await session.append(_chunk(1))
    with pytest.raises(ValueError, match="contiguous"):
        await session.append(_chunk(3))
    await session.close()


@pytest.mark.asyncio
async def test_reconnected_session_resumes_at_declared_next_sequence() -> None:
    start = TimelineSessionStart(
        **{
            **_start().__dict__,
            "observer_epoch": 5,
            "next_sequence": 8,
        }
    )
    session = TimelineDetectionSession(start, _Model())
    resumed = TimelineMediaChunk(
        **{
            **_chunk(8).__dict__,
            "observer_epoch": 5,
        }
    )

    await session.append(resumed)
    with pytest.raises(ValueError, match="contiguous"):
        await session.append(
            TimelineMediaChunk(
                **{
                    **_chunk(10).__dict__,
                    "observer_epoch": 5,
                }
            )
        )
    await session.close()


@pytest.mark.asyncio
async def test_rejects_observations_that_cite_unsent_media() -> None:
    model = _Model()
    model.observations = (
        ObservationEvent(
            observation_id="observation-1",
            session_id="session-1",
            identity_epoch=3,
            observer_epoch=4,
            stream_epoch=1,
            event_type="E11",
            summary="视线移开",
            evidence_start_ms=20,
            evidence_end_ms=80,
            model_id=model.model_id,
            model_version=model.model_version,
        ),
    )
    session = TimelineDetectionSession(_start(), model)
    await session.append(_chunk(1))
    with pytest.raises(ValueError, match="has not received"):
        await session.next_observations()


@pytest.mark.asyncio
async def test_accepts_observation_from_the_current_media_chunk() -> None:
    model = _Model()
    model.observations = (
        ObservationEvent(
            observation_id="observation-current",
            session_id="session-1",
            identity_epoch=3,
            observer_epoch=4,
            stream_epoch=1,
            event_type="E06",
            summary="起身",
            evidence_start_ms=0,
            evidence_end_ms=40,
            model_id=model.model_id,
            model_version=model.model_version,
        ),
    )
    session = TimelineDetectionSession(_start(), model)

    await session.append(_chunk(1))
    observations = await session.next_observations()

    assert observations == model.observations
    await session.close()


@pytest.mark.asyncio
async def test_discontinuity_advances_epoch_and_resets_sequence() -> None:
    model = _Model()
    session = TimelineDetectionSession(_start(), model)
    await session.append(_chunk(1))
    await session.discontinuity(
        TimelineDiscontinuity(
            session_id="session-1",
            identity_epoch=3,
            observer_epoch=4,
            old_stream_epoch=1,
            new_stream_epoch=2,
            reason="camera_reconnected",
        )
    )
    await session.append(_chunk(1, stream_epoch=2))
    await session.close()
    await session.close()
    assert model.started == 1
    assert model.closed == 1
