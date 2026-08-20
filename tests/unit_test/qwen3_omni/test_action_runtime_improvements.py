# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import threading

import pytest

from sglang_omni.client.client import Client
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
