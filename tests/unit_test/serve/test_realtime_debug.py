# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sglang_omni.serve.realtime import debug as realtime_debug
from sglang_omni.serve.realtime.debug import (
    REALTIME_LOG_DIR_ENV,
    load_realtime_session_debug,
    register_realtime_debug_routes,
)


def _write(root: Path, name: str, records: list[dict]) -> None:
    path = root / "2026-08-24" / "16" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _record(event: str, timestamp_ms: int, **fields) -> dict:
    return {
        "timestamp": f"2026-08-24T16:00:00.{timestamp_ms % 1000:03d}+08:00",
        "timestamp_unix_ms": timestamp_ms,
        "event": event,
        "level": "info",
        "session_id": "sess_debug_1",
        **fields,
    }


def _fixture_logs(root: Path) -> None:
    _write(
        root,
        "lifecycle_api_1_000.jsonl",
        [
            _record(
                "session_started",
                1_000,
                locale="zh-CN",
                modalities=["text", "action"],
                action_selection_mode="hierarchical",
                action_candidate_count=2,
                action_category_count=1,
            ),
            _record(
                "turn_started",
                1_100,
                turn_id="turn_debug_1",
                trace_id="trace-1",
                turn_origin="proactive",
                trigger="session_enter",
            ),
            _record(
                "turn_commit_received",
                2_000,
                turn_id="turn_debug_1",
                trace_id="trace-1",
                turn_origin="proactive",
            ),
            _record(
                "turn_completed",
                2_350,
                turn_id="turn_debug_1",
                trace_id="trace-1",
                status="completed",
            ),
        ],
    )
    _write(
        root,
        "diagnostic_api_1_000.jsonl",
        [
            _record(
                "session_instructions_received",
                1_010,
                instructions="自然回复。",
            )
        ],
    )
    _write(
        root,
        "reply_api_1_000.jsonl",
        [
            _record(
                "provided_reply_used",
                2_010,
                turn_id="turn_debug_1",
                output_text="你好。",
                first_delta_after_commit_ms=0.8,
                text_done_after_commit_ms=349.0,
                response_done_after_commit_ms=349.1,
            )
        ],
    )
    _write(
        root,
        "action_api_1_000.jsonl",
        [
            _record(
                "category_selected",
                2_220,
                turn_id="turn_debug_1",
                category_id="B027",
                category_label="打招呼与告别",
                support_status="supported",
            ),
            _record(
                "action_execution_assumed",
                2_348,
                turn_id="turn_debug_1",
                category_id="B027",
                candidate_id="A124",
                action_id="wave",
                source_label="单手挥手",
                execute=True,
            ),
        ],
    )
    _write(
        root,
        "performance_api_1_000.jsonl",
        [
            _record(
                "turn_timing",
                2_350,
                turn_id="turn_debug_1",
                status="completed",
                reply_first_delta_after_commit_ms=0.8,
                reply_text_done_after_commit_ms=349.0,
                reply_response_done_after_commit_ms=349.1,
                category_compute_ms=218.0,
                child_compute_ms=128.0,
                total_after_commit_ms=350.0,
                action_support_status="supported",
                action_fallback_applied=False,
            )
        ],
    )
    _write(
        root,
        "protocol_api_1_000.jsonl",
        [
            _record(
                "ws_event_sent",
                2_349,
                turn_id="turn_debug_1",
                ws_event_type="turn.action.ready",
            )
        ],
    )
    diagnostic_records = []
    rendered_records = []
    for stage, request_id, candidate_id in (
        ("category", "request-turn_debug_1-category", "B027"),
        ("child", "request-turn_debug_1-child", "A124"),
    ):
        diagnostic_records.extend(
            [
                _record(
                    "action_scoring_started",
                    2_020 if stage == "category" else 2_230,
                    full_logical_input={
                        "session_id": "sess_debug_1",
                        "request_id": request_id,
                        "logical_request_id": "logical-turn_debug_1",
                        "stage": stage,
                        "system_prompt": f"{stage} system",
                        "prefix": f"{stage} dynamic",
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": {
                            "session_id": "sess_debug_1",
                            "avatar_state": {
                                "state_description": "选择欢迎动作"
                            },
                        },
                    },
                ),
                _record(
                    "action_scoring_completed",
                    2_219 if stage == "category" else 2_347,
                    request_id=request_id,
                    scores=[
                        {
                            "candidate_id": candidate_id,
                            "ppl": 1.25,
                            "mean_logprob": -0.2,
                        }
                    ],
                ),
            ]
        )
        rendered_records.append(
            _record(
                "action_scoring_prompt_rendered",
                2_030 if stage == "category" else 2_240,
                request_id=request_id,
                full_prompt=f"rendered {stage} prompt",
                prompt_tokens=123,
            )
        )
    _write(root, "diagnostic_client_1_000.jsonl", diagnostic_records)
    _write(root, "diagnostic_preprocessing_2_000.jsonl", rendered_records)


def test_load_realtime_session_debug_aggregates_turn_prompts_and_timing(
    tmp_path: Path,
) -> None:
    _fixture_logs(tmp_path)

    result = load_realtime_session_debug(
        "sess_debug_1", log_root=tmp_path
    )

    assert result is not None
    assert result["session"]["locale"] == "zh-CN"
    assert result["session"]["instructions"] == "自然回复。"
    assert len(result["turns"]) == 1
    turn = result["turns"][0]
    assert turn["reply"]["text"] == "你好。"
    assert turn["reply"]["first_delta_after_commit_ms"] == 0.8
    assert turn["reply"]["prompt"] == {
        "available": False,
        "reason": "provided_reply",
        "configured_system_prompt": "自然回复。",
        "configured_system_prompt_applied": False,
    }
    assert turn["action"]["ready_after_commit_ms"] == 349
    assert turn["action"]["category_id"] == "B027"
    assert turn["action"]["candidate_id"] == "A124"
    assert turn["action"]["category_prompt"]["rendered_prompt"] == (
        "rendered category prompt"
    )
    assert turn["action"]["category_prompt"]["scores"][0][
        "candidate_id"
    ] == "B027"
    assert turn["action"]["child_prompt"]["dynamic_prompt"] == (
        "child dynamic"
    )


@pytest.mark.parametrize("session_id", ["", "../logs", "has space", "x" * 129])
def test_load_realtime_session_debug_rejects_invalid_session_id(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(ValueError, match="session_id must be"):
        load_realtime_session_debug(session_id, log_root=tmp_path)


@pytest.mark.asyncio
async def test_realtime_debug_routes_render_page_and_enforce_admin_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_logs(tmp_path)
    monkeypatch.setenv(REALTIME_LOG_DIR_ENV, str(tmp_path))
    fixture_result = load_realtime_session_debug(
        "sess_debug_1", log_root=tmp_path
    )
    monkeypatch.setattr(
        realtime_debug,
        "load_realtime_session_debug",
        lambda session_id: fixture_result
        if session_id == "sess_debug_1"
        else None,
    )

    async def _run_inline(function, *args):
        return function(*args)

    # Keep this route test deterministic in restricted environments where
    # repeated default-executor filesystem calls are not available.
    monkeypatch.setattr(realtime_debug.asyncio, "to_thread", _run_inline)
    app = FastAPI()
    register_realtime_debug_routes(app, "secret")
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        page = await client.get("/debug/realtime")
        assert page.status_code == 200
        assert "Realtime Session 调试" in page.text

        unauthorized = await client.get(
            "/debug/realtime/api/session/sess_debug_1"
        )
        assert unauthorized.status_code == 401

        response = await client.get(
            "/debug/realtime/api/session/sess_debug_1",
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200
        assert response.json()["turns"][0]["action"]["candidate_id"] == (
            "A124"
        )
