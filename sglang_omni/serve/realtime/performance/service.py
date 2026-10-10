"""Authenticated stateless access to the Session Realtime performance branch."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, WebSocket
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.websockets import WebSocketDisconnect

from sglang_omni.client import Client
from sglang_omni.models.qwen3_omni.action_scoring import ActionSuffixScoreRequest
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.performance.pipeline import PerformancePipeline
from sglang_omni.serve.realtime.protocol.common import FACIAL_EXPRESSION_CATEGORY_ID
from sglang_omni.serve.realtime.protocol.models import (
    SessionActionCandidate,
    SessionActionCategory,
    SessionActionProfile,
    TurnBuffer,
)
from sglang_omni.serve.streaming_request import (
    media_data_uri,
    receive_streamed_request,
    run_until_websocket_disconnect,
)


PERFORMANCE_CONTROL_TOKEN_ENV = "SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN"
logger = logging.getLogger(__name__)


class PerformanceExpressionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=512)
    description: str = Field(min_length=1, max_length=2_000)


class PerformanceControlRequest(BaseModel):
    """One committed Turn evaluated with the staging performance policy."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    session_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    language: Literal["zh", "en"]
    action_locale: Literal["zh-CN", "en-US"]
    text: str | None = Field(default=None, max_length=32_000)
    audios: list[str] = Field(default_factory=list, max_length=8)
    images: list[str] = Field(default_factory=list, max_length=8)
    image_roles: list[Literal["user_camera", "avatar_state"]] = Field(
        default_factory=list,
        max_length=8,
    )
    expressions: list[PerformanceExpressionInput] = Field(
        min_length=1,
        max_length=32,
    )
    action_profile: dict[str, object] = Field(default_factory=dict)
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    micro_batch_size: int = Field(default=64, ge=1, le=256)

    @model_validator(mode="after")
    def validate_turn(self) -> "PerformanceControlRequest":
        if not (self.text or "").strip() and not self.audios and not self.images:
            raise ValueError("performance control requires text, audio, or image input")
        if len(self.images) != len(self.image_roles):
            raise ValueError("images and image_roles must have equal length")
        expression_ids = [item.expression_id for item in self.expressions]
        if len(set(expression_ids)) != len(expression_ids):
            raise ValueError("performance expression ids must be unique")
        return self


class PerformanceControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    request_scope: Literal["none", "expression_only", "body_only", "both"]
    expression_id: str | None
    expression_unsupported: bool
    tts_instruction: str
    degraded: bool = False
    elapsed_ms: float


class _StatelessPerformancePipeline(PerformancePipeline):
    """Supply the small Session-owned surface required by PerformancePipeline."""

    def __init__(
        self,
        *,
        client: Client,
        request: PerformanceControlRequest,
    ) -> None:
        self.client = client
        self.model_name = request.model
        self.session_id = request.session_id
        self.session_instance_id = f"performance-{uuid.uuid4().hex}"
        self.language = request.language
        self.action_locale = request.action_locale
        self.action_language = "zh" if request.action_locale == "zh-CN" else "en"
        # This endpoint returns the instruction to D; it does not own embedded TTS.
        self.modalities: tuple[str, ...] = ()
        self.action_micro_batch_size = request.micro_batch_size
        self.action_profile = SessionActionProfile.from_payload(request.action_profile)
        self.categories = (
            SessionActionCategory(
                category_id=FACIAL_EXPRESSION_CATEGORY_ID,
                source_label="基础表情",
                short_definition="独立脸部表情",
                category_path=("头部与视线动作", "基础表情"),
                children=tuple(
                    SessionActionCandidate(
                        candidate_id=item.expression_id,
                        action_id=item.expression_id,
                        source_label=item.label,
                        short_definition=item.description,
                        execution_binding={},
                        category_id=FACIAL_EXPRESSION_CATEGORY_ID,
                    )
                    for item in request.expressions
                ),
            ),
        )

    def _action_prompt(self, *, zh: str, en: str) -> str:
        return zh if self.action_language == "zh" else en

    async def _score_action_request(
        self,
        turn: TurnBuffer,
        request: ActionSuffixScoreRequest,
        **_: Any,
    ) -> Any:
        del turn
        return await self.client.score_action_suffixes(request)

    async def decide(self, request: PerformanceControlRequest):
        turn = TurnBuffer(
            turn_id=request.turn_id,
            started_at=time.perf_counter(),
            audio=RealtimeAudioBuffer(source_sr=request.sample_rate),
            images=[],
            audio_seqs=set(),
            image_seqs=set(),
            turn_origin="user",
            text_role="user_input",
            text=(request.text or "").strip() or None,
            request_base=request.request_id,
            trace_id=request.request_id,
        )
        return await self._infer_turn_performance(
            turn,
            request.audios,
            current_text=turn.text,
            images=request.images,
            image_roles=list(request.image_roles),
        )


def register_performance_control(
    app: FastAPI,
    *,
    token: str | None = None,
) -> None:
    """Mount a private endpoint backed by the exact realtime PerformancePipeline."""

    configured_token = (
        token
        if token is not None
        else os.environ.get(PERFORMANCE_CONTROL_TOKEN_ENV, "")
    ).strip()

    @app.post(
        "/v1/performance-control",
        response_model=PerformanceControlResponse,
    )
    async def performance_control(
        request: PerformanceControlRequest,
        authorization: str | None = Header(default=None),
    ) -> PerformanceControlResponse:
        if not configured_token:
            raise HTTPException(
                status_code=503,
                detail="performance control is not configured",
            )
        supplied = (authorization or "").strip()
        expected = f"Bearer {configured_token}"
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

        return await _performance_response(app, request)

    @app.websocket("/v1/performance-control/realtime")
    async def performance_control_realtime(websocket: WebSocket) -> None:
        try:
            if not configured_token:
                await websocket.accept()
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "not_configured",
                        "detail": "performance control is not configured",
                    }
                )
                await websocket.close(code=1013)
                return
            streamed = await receive_streamed_request(
                websocket,
                expected_authorization=f"Bearer {configured_token}",
                max_media_items=16,
            )
            if streamed is None:
                return
            payload = dict(streamed.payload)
            audios = [
                media_data_uri(item)
                for item in streamed.media
                if item.kind == "audio"
            ]
            images = [
                media_data_uri(item)
                for item in streamed.media
                if item.kind in {"image", "video"}
            ]
            payload["audios"] = audios
            payload["images"] = images
            payload["image_roles"] = [
                item.evidence_role
                for item in streamed.media
                if item.kind in {"image", "video"}
            ]
            request = PerformanceControlRequest.model_validate(payload)
            response = await run_until_websocket_disconnect(
                websocket,
                _performance_response(app, request),
                abort=lambda: app.state.client.abort(request.request_id),
            )
            await websocket.send_json(
                {
                    "type": "response.completed",
                    "request_id": streamed.request_id,
                    "response": response.model_dump(),
                }
            )
        except WebSocketDisconnect:
            return
        except ValueError as exc:
            await websocket.send_json(
                {"type": "error", "code": "invalid_request", "detail": str(exc)}
            )
            await websocket.close(code=4400)


async def _performance_response(
    app: FastAPI,
    request: PerformanceControlRequest,
) -> PerformanceControlResponse:
    pipeline = _StatelessPerformancePipeline(
        client=app.state.client,
        request=request,
    )
    degraded = False
    try:
        decision = await pipeline.decide(request)
    except asyncio.CancelledError:
        raise
    except Exception:
        # Match the Session Realtime fail-soft policy: expression becomes
        # a no-op and speech uses the neutral language-specific instruction.
        logger.exception(
            "performance control failed; using staging neutral fallback",
            extra={"request_id": request.request_id},
        )
        decision = pipeline._default_performance_decision()
        degraded = True

    expression_id = (
        str(decision.expression.get("expression_id"))
        if decision.expression is not None
        else None
    )
    return PerformanceControlResponse(
        request_id=request.request_id,
        request_scope=decision.request_scope,
        expression_id=expression_id,
        expression_unsupported=decision.expression_unsupported,
        tts_instruction=decision.tts_instruction,
        degraded=degraded,
        elapsed_ms=decision.elapsed_ms,
    )
