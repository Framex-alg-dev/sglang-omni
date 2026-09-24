# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from sglang_omni.client.realtime_executor import (
    RemoteRealtimeModelClient,
    RoutedRealtimeModelClient,
    request_from_wire,
    request_to_wire,
)
from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    CompletionResult,
    CompletionStreamChunk,
    GenerateRequest,
    Message,
    SamplingParams,
    UsageInfo,
)
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
)
from sglang_omni.serve.internal_realtime_model_api import (
    register_internal_realtime_model_api,
)


class RecordingClient:
    def __init__(self, *, text: str = "ok", fail_before: bool = False) -> None:
        self.text = text
        self.fail_before = fail_before
        self.completions: list[str] = []
        self.streams: list[str] = []
        self.aborts: list[str] = []
        self.prefills: list[str] = []
        self.released_sessions: list[str] = []
        self.score_stages: list[str] = []

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        self.completions.append(request_id)
        if self.fail_before:
            raise ConnectionError("executor unavailable")
        return CompletionResult(request_id=request_id, text=self.text)

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        self.streams.append(request_id)
        if self.fail_before:
            raise ConnectionError("executor unavailable")
        yield CompletionStreamChunk(request_id=request_id, text=self.text)
        yield CompletionStreamChunk(
            request_id=request_id,
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        self.aborts.append(request_id)
        return AbortResult(success=True, level_applied=level)

    async def prefill_completion_prefix(
        self, request: GenerateRequest, *, request_id: str
    ) -> bool:
        self.prefills.append(request_id)
        if self.fail_before:
            raise ConnectionError("executor unavailable")
        return True

    async def release_session_cache(
        self, session_instance_id: str
    ) -> dict[str, Any]:
        self.released_sessions.append(session_instance_id)
        return {"released": True}

    async def score_action_suffixes(self, request):
        self.score_stages.append(request.stage)
        if self.fail_before:
            raise ConnectionError("executor unavailable")
        return ActionSuffixScoreResult(
            request_id=request.request_id,
            model=request.model,
            prefix_cached=True,
            scores=[
                CandidateScore(
                    candidate_id=request.candidates[0].candidate_id,
                    token_count=1,
                    mean_logprob=-0.1,
                    mean_nll=0.1,
                    ppl=1.1,
                    token_scores=[],
                )
            ],
        )

    def health(self) -> dict[str, Any]:
        return {"running": True}


def make_request(task: str, *, prepared_image: bool = False) -> GenerateRequest:
    metadata: dict[str, Any] = {
        "task": task,
        "session_id": "sess-test",
        "turn_id": "turn-test",
    }
    if prepared_image:
        metadata["images"] = [
            {
                "kind": "prepared_image_rgb",
                "width": 1,
                "height": 1,
                "pixel_bytes": b"\x01\x02\x03",
                "source_sha256": "source",
                "pixel_sha256": "pixel",
            }
        ]
    return GenerateRequest(
        model="model",
        messages=[Message(role="user", content="hello")],
        sampling=SamplingParams(temperature=0, max_new_tokens=8),
        metadata=metadata,
    )


def test_generate_request_wire_round_trip_preserves_prepared_image_bytes() -> None:
    request = make_request("session_reply", prepared_image=True)
    restored = request_from_wire(request_to_wire(request))
    assert restored.messages == request.messages
    assert restored.sampling == request.sampling
    assert restored.metadata == request.metadata


@pytest.mark.asyncio
async def test_routed_client_moves_only_allowlisted_reply_tasks() -> None:
    control = RecordingClient(text="control")
    reply = RecordingClient(text="reply")
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    reply_results = [
        await client.completion(
            make_request(task), request_id=f"reply-request-{index}"
        )
        for index, task in enumerate(
            (
                "session_reply",
                "session_pure_action_reply",
                "session_action_rejection",
                "session_turn_intent",
                "session_turn_intent_prewarm",
                "session_memory_extract",
                "session_visual_arithmetic_probe",
                "session_visual_arithmetic_prewarm",
            )
        )
    ]
    control_result = await client.completion(
        make_request("session_visual_gesture_probe"), request_id="control-request"
    )

    assert [result.text for result in reply_results] == ["reply"] * 8
    assert control_result.text == "control"
    assert reply.completions == [f"reply-request-{index}" for index in range(8)]
    assert control.completions == ["control-request"]


def make_score_request(stage: str) -> ActionSuffixScoreRequest:
    return ActionSuffixScoreRequest(
        request_id=f"score-{stage}", model="model", prefix="result:",
        language="en", candidates=[ActionScoreCandidate("R0", "R0")],
        audios=[], images=[], sample_rate=16000, stage=stage,
        session_id="sess-test", session_instance_id="instance-test",
    )


@pytest.mark.asyncio
async def test_routed_client_moves_reply_score_stages_only() -> None:
    control = RecordingClient()
    reply = RecordingClient()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    await client.score_action_suffixes(make_score_request("reply_history_route"))
    await client.score_action_suffixes(make_score_request("performance"))

    assert reply.score_stages == ["reply_history_route"]
    assert control.score_stages == ["performance"]


@pytest.mark.asyncio
async def test_reply_score_falls_back_to_control() -> None:
    control = RecordingClient()
    reply = RecordingClient(fail_before=True)
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    result = await client.score_action_suffixes(
        make_score_request("reply_speech_mode")
    )

    assert result.request_id == "score-reply_speech_mode"
    assert reply.score_stages == ["reply_speech_mode"]
    assert reply.aborts == ["score-reply_speech_mode"]
    assert control.score_stages == ["reply_speech_mode"]


@pytest.mark.asyncio
async def test_routed_stream_falls_back_before_first_output() -> None:
    control = RecordingClient(text="fallback")
    reply = RecordingClient(fail_before=True)
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    chunks = [
        chunk
        async for chunk in client.completion_stream(
            make_request("session_reply"), request_id="request-fallback"
        )
    ]

    assert "".join(chunk.text for chunk in chunks) == "fallback"
    assert reply.streams == ["request-fallback"]
    assert reply.aborts == ["request-fallback"]
    assert control.streams == ["request-fallback"]


@pytest.mark.asyncio
async def test_routed_stream_never_retries_after_visible_output() -> None:
    class PartialReply(RecordingClient):
        async def completion_stream(
            self, request, *, request_id, audio_format="wav"
        ):
            self.streams.append(request_id)
            yield CompletionStreamChunk(request_id=request_id, text="partial")
            raise ConnectionError("executor failed after output")

    control = RecordingClient(text="duplicate")
    reply = PartialReply()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]
    chunks: list[CompletionStreamChunk] = []

    with pytest.raises(ConnectionError, match="after output"):
        async for chunk in client.completion_stream(
            make_request("session_reply"), request_id="partial-reply"
        ):
            chunks.append(chunk)

    assert [chunk.text for chunk in chunks] == ["partial"]
    assert control.streams == []


@pytest.mark.asyncio
async def test_remote_executor_stream_and_abort_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_INTERNAL_MODEL_API", "1")
    monkeypatch.setenv("SGLANG_OMNI_INTERNAL_MODEL_TOKEN", "test-secret")
    local = RecordingClient(text="remote")
    app = FastAPI()
    app.state.client = local
    register_internal_realtime_model_api(app)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://executor",
    )
    remote = RemoteRealtimeModelClient(
        "http://executor",
        token="test-secret",
        client=http,
    )
    try:
        chunks = [
            chunk
            async for chunk in remote.completion_stream(
                make_request("session_reply", prepared_image=True),
                request_id="remote-stream",
            )
        ]
        result = await remote.completion(
            make_request("session_reply"), request_id="remote-completion"
        )
        score = await remote.score_action_suffixes(
            make_score_request("reply_history_route")
        )
        aborted = await remote.abort("remote-abort")
        released = await remote.release_session_cache("remote-session")
    finally:
        await http.aclose()

    assert "".join(chunk.text for chunk in chunks) == "remote"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 11
    assert result.text == "remote"
    assert score.request_id == "score-reply_history_route"
    assert score.scores[0].candidate_id == "R0"
    assert aborted.success is True
    assert released == {"released": True}
    assert local.streams == ["remote-stream"]
    assert local.completions == ["remote-completion"]
    assert local.score_stages == ["reply_history_route"]
    assert local.aborts == ["remote-abort"]
    assert local.released_sessions == ["remote-session"]


@pytest.mark.asyncio
async def test_remote_executor_rejects_wrong_token(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_INTERNAL_MODEL_API", "1")
    monkeypatch.setenv("SGLANG_OMNI_INTERNAL_MODEL_TOKEN", "right-secret")
    app = FastAPI()
    app.state.client = RecordingClient()
    register_internal_realtime_model_api(app)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://executor",
    )
    remote = RemoteRealtimeModelClient(
        "http://executor",
        token="wrong-secret",
        client=http,
    )
    try:
        with pytest.raises(Exception, match="HTTP 404"):
            await remote.completion(
                make_request("session_reply"), request_id="unauthorized"
            )
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_routed_abort_reaches_active_reply_executor() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingReply(RecordingClient):
        async def completion_stream(
            self, request, *, request_id, audio_format="wav"
        ):
            self.streams.append(request_id)
            started.set()
            await release.wait()
            yield CompletionStreamChunk(request_id=request_id, finish_reason="stop")

    control = RecordingClient()
    reply = BlockingReply()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    async def consume() -> None:
        async for _ in client.completion_stream(
            make_request("session_reply"), request_id="active-reply"
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    result = await client.abort("active-reply")
    release.set()
    await task

    assert result.success is True
    assert reply.aborts == ["active-reply"]
    assert control.aborts == []


@pytest.mark.asyncio
async def test_intentional_remote_abort_never_falls_back_to_control() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class AbortedReply(RecordingClient):
        async def completion_stream(
            self, request, *, request_id, audio_format="wav"
        ):
            self.streams.append(request_id)
            started.set()
            await release.wait()
            raise ConnectionError("remote request aborted")
            yield  # pragma: no cover

        async def abort(
            self,
            request_id: str,
            level: AbortLevel = AbortLevel.SOFT,
        ) -> AbortResult:
            self.aborts.append(request_id)
            release.set()
            return AbortResult(success=True, level_applied=level)

    control = RecordingClient(text="must-not-run")
    reply = AbortedReply()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    async def consume() -> None:
        async for _ in client.completion_stream(
            make_request("session_reply"), request_id="intentional-abort"
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    result = await client.abort("intentional-abort")

    assert result.success is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reply.aborts == ["intentional-abort"]
    assert control.streams == []


@pytest.mark.asyncio
async def test_marked_stream_cancel_teardown_error_never_falls_back() -> None:
    started = asyncio.Event()

    class BrokenClosingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            started.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            raise RuntimeError("aclose(): asynchronous generator is already running")

    class BrokenClosingReply(RecordingClient):
        def completion_stream(self, request, *, request_id, audio_format="wav"):
            self.streams.append(request_id)
            return BrokenClosingStream()

    control = RecordingClient(text="must-not-run")
    reply = BrokenClosingReply()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    async def consume() -> None:
        async for _ in client.completion_stream(
            make_request("session_reply"), request_id="marked-cancel"
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    await client.mark_intentional_abort("marked-cancel")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert reply.streams == ["marked-cancel"]
    assert control.streams == []


@pytest.mark.asyncio
async def test_reply_prefill_routes_and_releases_both_executors() -> None:
    control = RecordingClient()
    reply = RecordingClient()
    client = RoutedRealtimeModelClient(control, reply)  # type: ignore[arg-type]

    ready = await client.prefill_completion_prefix(
        make_request("session_reply"), request_id="reply-prefill"
    )
    await client.release_session_cache("session-instance")

    assert ready is True
    assert reply.prefills == ["reply-prefill"]
    assert control.prefills == []
    assert reply.released_sessions == ["session-instance"]
    assert control.released_sessions == ["session-instance"]
