# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from sglang_omni.client.client import Client, _PriorityAdmissionGate
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    ActionSuffixScoreResult,
    CandidateScore,
    TokenScore,
)
from sglang_omni.models.qwen3_omni.action_timing import (
    get_action_stage_timings,
    merge_action_stage_timings,
    record_action_stage_timing,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.async_jsonl import AsyncJSONLWriter


def _payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={},
            metadata={"task": "action_suffix_scoring"},
        ),
        data={},
    )


def test_action_stage_timings_merge_parallel_payloads() -> None:
    preprocessing = _payload("timing")
    image = _payload("timing")
    audio = _payload("timing")
    record_action_stage_timing(preprocessing, "preprocessing", wall_ms=4.5)
    record_action_stage_timing(image, "image_encoder", wall_ms=7.25)
    record_action_stage_timing(audio, "audio_encoder", wall_ms=8.75)

    merged = merge_action_stage_timings(
        [preprocessing, image, audio], preprocessing
    )

    assert merged == {
        "preprocessing": {"wall_ms": 4.5},
        "image_encoder": {"wall_ms": 7.25},
        "audio_encoder": {"wall_ms": 8.75},
    }
    assert get_action_stage_timings(preprocessing) == merged


def test_async_jsonl_writer_is_non_blocking_and_bounded(tmp_path) -> None:
    path = tmp_path / "action.jsonl"
    writer = AsyncJSONLWriter(path, max_queue_size=1)
    entered = threading.Event()
    release = threading.Event()
    original_append = writer._append

    def blocking_append(record):
        entered.set()
        release.wait(timeout=2.0)
        original_append(record)

    writer._append = blocking_append
    try:
        assert writer.write({"index": 1, "text": "你好"}) is True
        assert entered.wait(timeout=1.0)
        assert writer.write({"index": 2}) is True
        assert writer.write({"index": 3}) is False
        assert writer.dropped_records == 1
        release.set()
        writer.flush()
    finally:
        release.set()
        writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [{"index": 1, "text": "你好"}, {"index": 2}]


@pytest.mark.asyncio
async def test_action_admission_gate_prefers_lower_priority_and_preserves_fifo() -> None:
    gate = _PriorityAdmissionGate(1)
    await gate.acquire(0)
    admitted: list[str] = []

    async def enter(name: str, priority: int) -> None:
        async with gate.slot(priority):
            admitted.append(name)
            await asyncio.sleep(0)

    low_first = asyncio.create_task(enter("performance-first", 2))
    category = asyncio.create_task(enter("category", 0))
    low_second = asyncio.create_task(enter("performance-second", 2))
    await asyncio.sleep(0)
    await gate.release()
    await asyncio.gather(low_first, category, low_second)

    assert admitted == ["category", "performance-first", "performance-second"]


@pytest.mark.asyncio
async def test_client_adds_real_request_path_timings(monkeypatch) -> None:
    class FakeCoordinator:
        async def submit(self, request_id, request):
            del request
            return ActionSuffixScoreResult(
                request_id=request_id,
                model="qwen3-omni",
                prefix_cached=True,
                scores=[
                    CandidateScore(
                        candidate_id="A1",
                        token_count=1,
                        mean_logprob=-0.1,
                        mean_nll=0.1,
                        ppl=1.105,
                        token_scores=[TokenScore(token_id=1, logprob=-0.1)],
                    )
                ],
                stats={"preprocessing_ms": 2.0},
            )

        async def abort(self, request_id):
            del request_id

    monkeypatch.setattr(
        "sglang_omni.client.client._write_action_debug_record",
        lambda record: None,
    )
    client = Client(FakeCoordinator())
    request = ActionSuffixScoreRequest(
        request_id="timing-client",
        model="qwen3-omni",
        prefix="下一步 action_id 是：",
        language="zh",
        candidates=[
            ActionScoreCandidate(
                candidate_id="A1", suffix="A1", action_id="wave"
            )
        ],
        suffix_tokenization_mode="short_id",
        audios=[],
        images=[],
        sample_rate=16000,
    )

    result = await client.score_action_suffixes(request)

    assert result.stats["preprocessing_ms"] == 2.0
    for key in (
        "action_slot_wait_ms",
        "coordinator_pipeline_ms",
        "client_result_processing_ms",
        "client_total_ms",
    ):
        assert result.stats[key] >= 0.0
    assert result.stats["action_admission_priority"] == 10
    assert result.stats["action_admission_capacity"] >= 1


@pytest.mark.asyncio
async def test_session_catalog_prefill_places_session_prefix_before_turn_prompt(
    monkeypatch,
) -> None:
    class CaptureCoordinator:
        request = None

        async def submit(self, request_id, request):
            self.request = request
            return ActionSuffixScoreResult(
                request_id=request_id,
                model="qwen3-omni",
                prefix_cached=True,
                scores=[
                    CandidateScore(
                        candidate_id="B001",
                        token_count=1,
                        mean_logprob=-0.1,
                        mean_nll=0.1,
                        ppl=1.105,
                        token_scores=[TokenScore(token_id=1, logprob=-0.1)],
                    )
                ],
            )

        async def abort(self, request_id):
            del request_id

    monkeypatch.setattr(
        "sglang_omni.client.client._write_action_debug_record",
        lambda record: None,
    )
    coordinator = CaptureCoordinator()
    client = Client(coordinator)

    ready = await client.prefill_action_catalog(
        model="qwen3-omni",
        system_prompt="全局动作目录",
        session_instruction="会话人设、实体信息和动作偏好。",
        candidates=[
            ActionScoreCandidate(
                candidate_id="B001", suffix="B001", action_id="B001"
            )
        ],
        prefix_cache_namespace="category:session:test",
        stage="category",
        language="zh",
    )

    assert ready is True
    request = coordinator.request
    assert request is not None
    assert request.inputs["messages"][0] == {
        "role": "system",
        "content": "全局动作目录",
    }
    current = "".join(
        part.get("text", "")
        for part in request.inputs["messages"][-1]["content"]
    )
    assert current.index("会话人设、实体信息和动作偏好。") < current.index(
        "请根据当前输入选择动作"
    )
    scoring = request.params["action_scoring"]
    assert scoring["cache_static_system_only"] is False
    assert scoring["admission_priority"] == 3
