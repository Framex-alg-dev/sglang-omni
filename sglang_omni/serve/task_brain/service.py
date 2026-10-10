"""Authenticated, bounded OpenAI-compatible task-brain endpoint."""

from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, ValidationError
from starlette.websockets import WebSocketDisconnect

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.protocol import ChatCompletionRequest
from sglang_omni.serve.streaming_request import (
    inject_openai_media,
    receive_streamed_request,
    run_until_websocket_disconnect,
)
from sglang_omni.serve.structured_output import response_json_schema


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...

    async def abort(self, request_id: str) -> Any: ...


class TaskBrainCompletionRequest(ChatCompletionRequest):
    """The narrow OpenAI subset accepted by D's task-brain adapter."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class TaskBrainServiceConfig:
    token: str
    model_id: str
    model_version: str
    max_concurrency: int = 2
    max_waiting: int = 32
    max_body_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if not all(
            value.strip() for value in (self.token, self.model_id, self.model_version)
        ):
            raise ValueError("task-brain token, model_id, and model_version are required")
        if min(self.max_concurrency, self.max_waiting, self.max_body_bytes) <= 0:
            raise ValueError("task-brain admission and body limits must be positive")


class TaskBrainQueueFull(RuntimeError):
    pass


class _BoundedAdmission:
    """Bound Brain work while leaving scheduler admission available to classifier."""

    def __init__(self, capacity: int, max_waiting: int) -> None:
        self._semaphore = asyncio.Semaphore(capacity)
        self._max_waiting = max_waiting
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiting = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[float]:
        queued_at = time.perf_counter()
        async with self._lock:
            if self._waiting >= self._max_waiting:
                raise TaskBrainQueueFull("task-brain admission queue is full")
            self._waiting += 1
        acquired = False
        accounted_active = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                self._active += 1
                accounted_active = True
            yield (time.perf_counter() - queued_at) * 1000.0
        finally:
            async with self._lock:
                if accounted_active:
                    self._active -= 1
                elif self._waiting > 0:
                    self._waiting -= 1
            if acquired:
                self._semaphore.release()

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {"active": self._active, "waiting": self._waiting}


def create_task_brain_app(
    client: CompletionClient,
    *,
    config: TaskBrainServiceConfig,
) -> FastAPI:
    """Expose only the completion contract needed by the D-owned Agent Harness."""

    app = FastAPI(title="sglang-omni-task-brain", version="1")
    admission = _BoundedAdmission(config.max_concurrency, config.max_waiting)

    def authenticate(request: Request) -> None:
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {config.token}"
        ):
            raise HTTPException(status_code=404, detail="not found")

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        authenticate(request)
        return JSONResponse(
            {
                "ok": True,
                "contract": "task-decision-v1",
                "model_id": config.model_id,
                "model_version": config.model_version,
                "admission": await admission.snapshot(),
            }
        )

    @app.post("/v1/chat/completions")
    async def complete(request: Request) -> JSONResponse:
        authenticate(request)
        body = await _read_body(request, config.max_body_bytes)
        try:
            completion = TaskBrainCompletionRequest.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            payload, queue_ms, elapsed_ms = await _complete_request(
                client,
                config=config,
                admission=admission,
                completion=completion,
            )
        except TaskBrainQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return JSONResponse(
            payload,
            headers={
                "Server-Timing": (
                    f"queue;dur={queue_ms:.3f},task_brain;dur={elapsed_ms:.3f}"
                )
            },
        )

    @app.websocket("/v1/chat/completions/realtime")
    async def complete_realtime(websocket: WebSocket) -> None:
        try:
            streamed = await receive_streamed_request(
                websocket,
                expected_authorization=f"Bearer {config.token}",
                max_total_bytes=config.max_body_bytes,
            )
            if streamed is None:
                return
            payload = inject_openai_media(streamed.payload, streamed.media)
            completion = TaskBrainCompletionRequest.model_validate(payload)
            request_id = completion.request_id or streamed.request_id
            completion = completion.model_copy(update={"request_id": request_id})
            response, _, _ = await run_until_websocket_disconnect(
                websocket,
                _complete_request(
                    client,
                    config=config,
                    admission=admission,
                    completion=completion,
                ),
                abort=lambda: client.abort(request_id),
            )
            await websocket.send_json(
                {
                    "type": "response.completed",
                    "request_id": streamed.request_id,
                    "response": response,
                }
            )
        except WebSocketDisconnect:
            return
        except HTTPException as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "invalid_request",
                    "detail": str(exc.detail),
                }
            )
            await websocket.close(code=4400)
        except TaskBrainQueueFull as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "overloaded",
                    "detail": str(exc),
                    "retryable": True,
                }
            )
            await websocket.close(code=4429)
        except (ValidationError, ValueError) as exc:
            await websocket.send_json(
                {"type": "error", "code": "invalid_request", "detail": str(exc)}
            )
            await websocket.close(code=4400)

    return app


async def _complete_request(
    client: CompletionClient,
    *,
    config: TaskBrainServiceConfig,
    admission: _BoundedAdmission,
    completion: TaskBrainCompletionRequest,
) -> tuple[dict[str, Any], float, float]:
    _validate_request(completion)
    request_id = completion.request_id or str(uuid.uuid4())
    started = time.perf_counter()
    async with admission.slot() as queue_ms:
        result = await client.completion(
            _generate_request(completion, config.model_id, request_id),
            request_id=request_id,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.model or config.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(result.text)},
                "finish_reason": getattr(result, "finish_reason", "stop"),
            }
        ],
    }
    usage = getattr(result, "usage", None)
    if usage is not None:
        payload["usage"] = usage.to_dict()
    return payload, queue_ms, elapsed_ms


async def _read_body(request: Request, max_body_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_body_bytes:
                raise HTTPException(status_code=413, detail="request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
    body = await request.body()
    if len(body) > max_body_bytes:
        raise HTTPException(status_code=413, detail="request body too large")
    return body


def _validate_request(request: TaskBrainCompletionRequest) -> None:
    if request.stream:
        raise HTTPException(status_code=422, detail="task brain does not stream")
    if request.modalities not in (None, ["text"]):
        raise HTTPException(status_code=422, detail="task brain returns text only")
    try:
        response_json_schema(request.response_format)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    if request.reasoning_effort not in (None, "none"):
        raise HTTPException(
            status_code=422,
            detail="local task brain supports reasoning_effort=none only",
        )


def _generate_request(
    request: TaskBrainCompletionRequest,
    model_id: str,
    request_id: str,
) -> GenerateRequest:
    stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
    metadata: dict[str, Any] = {
        "task": "task_brain",
        "task_role": "task_brain",
        "logical_request_id": request_id,
        "contract_version": 1,
    }
    if request.audios:
        metadata["audios"] = list(request.audios)
    if request.images:
        metadata["images"] = list(request.images)
    return GenerateRequest(
        model=request.model or model_id,
        messages=[Message(role=item.role, content=item.content) for item in request.messages],
        sampling=SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            top_k=request.top_k if request.top_k is not None else -1,
            min_p=request.min_p if request.min_p is not None else 0.0,
            repetition_penalty=(
                request.repetition_penalty
                if request.repetition_penalty is not None
                else 1.0
            ),
            stop=stop,
            seed=request.seed,
            max_new_tokens=request.effective_max_tokens,
            json_schema=response_json_schema(request.response_format),
        ),
        stream=False,
        max_tokens=request.effective_max_tokens,
        output_modalities=["text"],
        metadata=metadata,
    )
