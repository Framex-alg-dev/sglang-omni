"""Authenticated Action Omni service with exact two-token decoding."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.streaming_request import (
    StreamedMedia,
    media_data_uri,
    receive_streamed_request,
    run_until_websocket_disconnect,
)
from sglang_omni.utils.structured_logs import emit_structured_log

from .catalog import ActionCatalogRegistry, ActionEntry, ResolvedCatalog
from .contracts import ActionDecision, ActionDecisionRequest, DecisionChannel
from .prompt import build_system_prompt
from .speech_window import PERSONA_PROFILES
from .speech_audio import ACTION_AUDIO_ROLES
from .model_input import user_message_content
from .requested_action import parse_requested_action
from .recall_audit import input_snapshot, recall_variant


class CompletionClient(Protocol):
    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> Any: ...

    async def abort(self, request_id: str) -> Any: ...


@dataclass(frozen=True)
class ActionDecisionConfig:
    token: str
    model_id: str
    model_version: str
    mapping_path: str
    full_mapping_path: str
    product_catalog_path: str
    agent_policy_path: str
    max_body_bytes: int = 128 * 1024 * 1024
    idempotency_cache_size: int = 2048

    def __post_init__(self) -> None:
        required = (
            self.token,
            self.model_id,
            self.model_version,
            self.mapping_path,
            self.full_mapping_path,
            self.product_catalog_path,
            self.agent_policy_path,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError("action-decision configuration fields are required")
        if self.max_body_bytes <= 0 or self.idempotency_cache_size <= 0:
            raise ValueError("action-decision limits must be positive")


class ActionDecisionEngine:
    def __init__(
        self,
        client: CompletionClient,
        *,
        config: ActionDecisionConfig,
        registry: ActionCatalogRegistry | None = None,
        prompt_snapshot: dict[str, str] | None = None,
    ) -> None:
        self._prompt_snapshot = dict(prompt_snapshot) if prompt_snapshot is not None else None
        self._runtime_prompt_bundles = {}
        self._client = client
        self.config = config
        self.registry = registry or ActionCatalogRegistry(
            mapping_path=config.mapping_path,
            full_mapping_path=config.full_mapping_path,
            product_catalog_path=config.product_catalog_path,
            agent_policy_path=config.agent_policy_path,
        )
        self._lock = asyncio.Lock()
        self._completed: OrderedDict[str, tuple[str, ActionDecision]] = OrderedDict()
        self._inflight: dict[str, tuple[str, asyncio.Task[ActionDecision]]] = {}
        self._cancelled: OrderedDict[str, None] = OrderedDict()

    def _effective_prompt_snapshot(self, request):
        if self._prompt_snapshot is not None:
            return self._prompt_snapshot
        from .prompt import reference_prompt_profile, E67_REFERENCE_PROFILE
        reference = reference_prompt_profile(request)
        if reference is not None and reference != E67_REFERENCE_PROFILE:
            return None  # Preserve explicitly requested legacy reference profiles.
        profile = 'e67' if reference == E67_REFERENCE_PROFILE else 'generic'
        bundle = self._runtime_prompt_bundles.get(profile)
        return bundle.rules if bundle is not None else None

    async def decide(self, request: ActionDecisionRequest) -> ActionDecision:
        digest = _request_digest(request)
        async with self._lock:
            if request.request_id in self._cancelled:
                raise asyncio.CancelledError("logical_body_request_cancelled")
            completed = self._completed.get(request.request_id)
            if completed is not None:
                if completed[0] != digest:
                    raise ValueError("request_id was reused with different input")
                self._completed.move_to_end(request.request_id)
                emit_structured_log(
                    "action",
                    "action_decision_idempotent_replay",
                    component="action_decision",
                    request_id=request.request_id,
                    session_id=request.session_id,
                    turn_id=request.turn_id,
                    decision_point_id=request.decision_point_id,
                    channel=request.channel.value,
                )
                return completed[1]
            inflight = self._inflight.get(request.request_id)
            if inflight is not None:
                if inflight[0] != digest:
                    raise ValueError("request_id was reused with different input")
                task = inflight[1]
            else:
                task = asyncio.create_task(
                    self._decide_once(request),
                    name=f"action-decision:{request.channel.value}:{request.request_id}",
                )
                self._inflight[request.request_id] = (digest, task)
                task.add_done_callback(
                    lambda finished, current=request, current_digest=digest: (
                        asyncio.create_task(
                            self._settle(current, current_digest, finished)
                        )
                    )
                )
        try:
            decision = await asyncio.shield(task)
        except asyncio.CancelledError:
            emit_structured_log(
                "action",
                "action_decision_waiter_cancelled",
                component="action_decision",
                request_id=request.request_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
                decision_point_id=request.decision_point_id,
                channel=request.channel.value,
            )
            raise
        except Exception as exc:
            emit_structured_log(
                "error",
                "action_decision_failed",
                component="action_decision",
                request_id=request.request_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
                decision_point_id=request.decision_point_id,
                channel=request.channel.value,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            raise
        await self._settle(request, digest, task)
        return decision

    async def cancel(self, request_id: str) -> None:
        """Remove API waiters as well as requests already in model scheduling."""
        async with self._lock:
            self._cancelled[request_id] = None
            self._cancelled.move_to_end(request_id)
            while len(self._cancelled) > self.config.idempotency_cache_size:
                self._cancelled.popitem(last=False)
            entry = self._inflight.get(request_id)
            task = entry[1] if entry is not None else None
            if task is not None:
                task.cancel("logical_body_request_cancelled")
        try:
            await self._client.abort(request_id)
        finally:
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        emit_structured_log("action", "action_decision_cancel_terminal",
                            component="action_decision", request_id=request_id,
                            api_waiter_removed=True, model_abort_requested=True)

    async def _settle(
        self,
        request: ActionDecisionRequest,
        digest: str,
        task: asyncio.Task[ActionDecision],
    ) -> None:
        async with self._lock:
            current = self._inflight.get(request.request_id)
            if current is not None and current[1] is task:
                self._inflight.pop(request.request_id, None)
            if task.cancelled() or task.exception() is not None:
                return
            self._completed[request.request_id] = (digest, task.result())
            self._completed.move_to_end(request.request_id)
            while len(self._completed) > self.config.idempotency_cache_size:
                self._completed.popitem(last=False)

    async def _decide_once(self, request: ActionDecisionRequest) -> ActionDecision:
        started = time.perf_counter()
        expected_version = request.runtime_context.get("expected_model_version")
        if expected_version and expected_version != self.config.model_version:
            raise ValueError("Body frozen model release does not match the running model")
        expected_prompt = request.runtime_context.get("expected_prompt_sha256")
        if expected_prompt:
            from .prompt import _reference_system_prompt, reference_prompt_profile
            profile = reference_prompt_profile(request)
            if profile is None or hashlib.sha256(_reference_system_prompt(profile).encode()).hexdigest() != expected_prompt:
                raise ValueError("Body frozen prompt digest does not match the running prompt")
        catalog = self.registry.resolve(request)
        emit_structured_log(
            "action",
            "action_decision_started",
            component="action_decision",
            request_id=request.request_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            decision_point_id=request.decision_point_id,
            channel=request.channel.value,
            candidate_count=len(catalog.entries),
            recall_variant=recall_variant(request),
            requested_action=request.requested_action,
            media_count=len(request.media),
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            catalog_version=catalog.catalog_version,
            mapping_version=catalog.mapping_version,
        )
        if not request.channel_enabled or request.prohibited:
            entry = self.registry.control(request.channel)
            reason = "channel_disabled" if not request.channel_enabled else "channel_prohibited"
            return self._decision(
                request, catalog, entry, started, deterministic=True, reason_code=reason
            )
        persona_window = request.runtime_context.get("prompt_profile") in PERSONA_PROFILES
        if catalog.hard_bound is not None and not persona_window:
            return self._decision(
                request,
                catalog,
                catalog.hard_bound,
                started,
                deterministic=True,
                reason_code="required_action_binding",
            )

        content = user_message_content(request, catalog)
        system_prompt = build_system_prompt(request, catalog, prompt_snapshot=self._effective_prompt_snapshot(request))
        snapshot = await asyncio.to_thread(input_snapshot, request, catalog, system_prompt, content)
        emit_structured_log("action", "action_recall_model_input", component="action_decision",
                            request_id=request.request_id, session_id=request.session_id,
                            turn_id=request.turn_id, channel=request.channel.value,
                            model_id=self.config.model_id, model_version=self.config.model_version,
                            **snapshot)
        metadata: dict[str, Any] = {
            "task": "action_direct",
            "task_role": request.channel.value,
            "logical_request_id": request.decision_point_id,
            "session_instance_id": request.session_id,
            "contract_version": 1,
            "session_id": request.session_id,
            "turn_id": request.turn_id,
            "channel": request.channel.value,
            "output_modalities": ["text"],
        }
        from sglang_omni.utils.turn_media_scope import turn_media_scope
        media_owner = turn_media_scope(
            request.session_id, request.turn_id,
            request.runtime_context.get("identity_epoch"),
            request.runtime_context.get("input_revision"),
        )
        if media_owner is not None:
            metadata["media_cache_owner"] = media_owner
        audios = [media_data_uri(item) for item in request.media if item.kind == "audio"]
        images = [
            media_data_uri(item)
            for item in request.media
            if item.kind in {"image", "video"}
        ]
        if audios:
            metadata["audios"] = audios
        if images:
            metadata["images"] = images
            metadata["image_roles"] = [
                item.evidence_role
                for item in request.media
                if item.kind in {"image", "video"}
            ]
        legal_codes = (catalog.hard_bound.code,) if catalog.hard_bound is not None else tuple(entry.code for entry in catalog.entries)
        grammar = "(?:" + "|".join(re.escape(code) for code in legal_codes) + ")"
        result = await self._client.completion(
            GenerateRequest(
                model=self.config.model_id,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=content),
                ],
                sampling=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    seed=0,
                    max_new_tokens=2,
                    min_new_tokens=2,
                    ignore_eos=True,
                    regex=grammar,
                ),
                stream=False,
                max_tokens=2,
                output_modalities=["text"],
                metadata=metadata,
            ),
            request_id=request.request_id,
        )
        code = str(result.text)
        # Constrained backends normally return the assigned two-token code, but
        # some deployments decode control tokens to their canonical candidate
        # ID (for example ``IB0``). Both identify the same catalog entry.
        entry = catalog.by_code.get(code) or catalog.by_candidate_id.get(code)
        if entry is None or entry.code not in legal_codes:
            raise RuntimeError(f"action model returned code outside constrained catalog: {code!r}")
        usage = result.usage.to_dict() if result.usage is not None else None
        return self._decision(
            request,
            catalog,
            entry,
            started,
            deterministic=False,
            reason_code="model_selected",
            weight_version=result.weight_version,
            usage=usage,
        )

    def _decision(
        self,
        request: ActionDecisionRequest,
        catalog: ResolvedCatalog,
        entry: ActionEntry,
        started: float,
        *,
        deterministic: bool,
        reason_code: str,
        weight_version: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ActionDecision:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        outcome = _outcome(request.channel, entry.candidate_id)
        decision = ActionDecision(
            contract_version=1,
            request_id=request.request_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            decision_point_id=request.decision_point_id,
            channel=request.channel,
            outcome=outcome,
            candidate_id=entry.candidate_id,
            action_id=entry.action_id,
            code=entry.code,
            token_ids=entry.token_ids,
            label=entry.label,
            occupies_channels=entry.occupies_channels,
            deterministic=deterministic,
            model_invoked=not deterministic,
            reason_code=reason_code,
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            weight_version=weight_version,
            catalog_version=catalog.catalog_version,
            catalog_hash=catalog.catalog_hash,
            mapping_version=catalog.mapping_version,
            mapping_hash=catalog.mapping_hash,
            candidate_count=len(catalog.entries),
            elapsed_ms=round(elapsed_ms, 3),
            usage=usage,
        )
        emit_structured_log(
            "action",
            "action_decision_completed",
            component="action_decision",
            recall_variant=recall_variant(request),
            **decision.to_dict(),
        )
        return decision


def create_action_decision_app(
    client: CompletionClient,
    *,
    config: ActionDecisionConfig,
    performance_token: str | None = None,
) -> FastAPI:
    engine = ActionDecisionEngine(client, config=config)
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        from sglang_omni.serve.prompt_runtime import install_action_runtime
        await install_action_runtime(app, engine, include_performance=performance_token is not None)
        yield
    app = FastAPI(title="sglang-omni-action-decision", version="1", lifespan=lifespan)
    app.state.action_decision_engine = engine


    def authenticate(request: Request) -> None:
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {config.token}"
        ):
            raise HTTPException(status_code=404, detail="not found")

    from sglang_omni.serve.prompt_debug import install_action_debug
    install_action_debug(app, engine=engine, authenticate=authenticate, parser=_parse_request,
                         factory=lambda capture, snapshot: ActionDecisionEngine(capture, config=config, registry=engine.registry, prompt_snapshot=snapshot))

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        authenticate(request)
        runtime: dict[str, Any] = {}
        admission_load = getattr(client, "action_scoring_load", None)
        if callable(admission_load):
            runtime["workload_admission"] = admission_load()
        pipeline_health = getattr(client, "health", None)
        if callable(pipeline_health):
            runtime["pipeline"] = pipeline_health()
        return JSONResponse(
            {
                "ok": True,
                "contract": "action-decision.v1",
                "channels": ["body", "expression"],
                "model_id": config.model_id,
                "model_version": config.model_version,
                "catalog_version": engine.registry.catalog_version,
                "mapping_version": engine.registry.mapping_version,
                "runtime": runtime,
            }
        )

    @app.post("/v1/action-decision")
    async def decide(request: Request) -> JSONResponse:
        authenticate(request)
        body = await request.body()
        if len(body) > config.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        try:
            raw = json.loads(body)
            decision_request = _parse_request(raw, media=())
            decision = await engine.decide(decision_request)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(decision.to_dict())

    @app.websocket("/v1/action-decision/realtime")
    async def decide_realtime(websocket: WebSocket) -> None:
        request_id = ""
        try:
            streamed = await receive_streamed_request(
                websocket,
                expected_authorization=f"Bearer {config.token}",
                max_total_bytes=config.max_body_bytes,
                allowed_audio_roles=ACTION_AUDIO_ROLES | {"assistant_spoken"},
            )
            if streamed is None:
                return
            request_id = streamed.request_id
            payload = dict(streamed.payload)
            payload.setdefault("request_id", request_id)
            decision_request = _parse_request(payload, media=streamed.media)
            decision = await run_until_websocket_disconnect(
                websocket,
                engine.decide(decision_request),
                abort=lambda: engine.cancel(request_id),
            )
            await websocket.send_json(
                {
                    "type": "response.completed",
                    "request_id": request_id,
                    "response": decision.to_dict(),
                }
            )
        except WebSocketDisconnect:
            return
        except (TypeError, ValueError) as exc:
            await websocket.send_json(
                {"type": "error", "code": "invalid_request", "detail": str(exc)}
            )
            await websocket.close(code=4400)
        except Exception as exc:
            emit_structured_log(
                "error",
                "action_decision_failed",
                component="action_decision",
                request_id=request_id,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
            await websocket.send_json(
                {"type": "error", "code": "model_error", "detail": str(exc)}
            )
            await websocket.close(code=1011)

    @app.post("/v1/action-recall")
    async def recall_audio(request: Request) -> JSONResponse:
        authenticate(request)
        body = await request.body()
        if len(body) > config.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        from .simple_audio import parse_simple_audio_request
        try:
            decision_request, public_id = parse_simple_audio_request(json.loads(body), engine.registry)
            emit_structured_log("action", "action_recall_received", component="action_decision",
                                request_id=decision_request.request_id, public_request_id=public_id,
                                recall_variant=recall_variant(decision_request))
            decision = await engine.decide(decision_request)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse({"request_id": public_id, "channel": decision.channel.value,
                             "candidate_id": decision.candidate_id, "code": decision.code,
                             "label": decision.label, "outcome": decision.outcome,
                             "occupies_channels": list(decision.occupies_channels)})

    if performance_token is not None:
        normalized_performance_token = performance_token.strip()
        if not normalized_performance_token:
            raise ValueError("performance control token must not be empty")
        from sglang_omni.serve.performance_control.service import (
            register_performance_control,
        )

        app.state.client = client
        register_performance_control(app, token=normalized_performance_token)

    return app


def _parse_request(
    raw: Any,
    *,
    media: tuple[StreamedMedia, ...],
) -> ActionDecisionRequest:
    if not isinstance(raw, dict):
        raise ValueError("action decision payload must be an object")
    channel = DecisionChannel(_required_string(raw, "channel"))
    request_id = _required_string(raw, "request_id")
    if any(item.evidence_role == "assistant_spoken" for item in media) and not (
        channel is DecisionChannel.BODY and raw.get("turn_origin") == "assistant"
    ):
        raise ValueError("assistant audio requires an assistant body-action request")
    runtime_context = _object_field(raw, "runtime_context")
    return ActionDecisionRequest(
        request_id=request_id,
        session_id=_required_string(raw, "session_id"),
        turn_id=_required_string(raw, "turn_id"),
        decision_point_id=_required_string(raw, "decision_point_id"),
        channel=channel,
        text=_optional_string(raw, "text"),
        language=_optional_string(raw, "language") or "zh-CN",
        character_prompt=_optional_string(raw, "character_prompt"),
        session_prompt=_optional_string(raw, "session_prompt"),
        reply_prefix=_optional_string(raw, "reply_prefix"),
        runtime_context=runtime_context,
        allowed_candidate_ids=_string_array(raw, "allowed_candidate_ids"),
        excluded_candidate_ids=_string_array(raw, "excluded_candidate_ids"),
        channel_enabled=_boolean(raw, "channel_enabled", True),
        prohibited=_boolean(raw, "prohibited", False),
        turn_origin=_optional_string(raw, "turn_origin") or "user",
        media=media,
        requested_action=parse_requested_action(raw.get("requested_action")),
        # User-reaction media has no executor timeline. Match the frozen r1
        # text/audio/image renderer without exposing transport frame ordinals
        # as meaningful millisecond evidence. Speech windows retain timing.
        simple_audio=(runtime_context.get("prompt_profile") == "persona-audio-window-v2"
                      and runtime_context.get("decision_mode") == "user_request"),
    )


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _object_field(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return dict(value)


def _string_array(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a string array")
    return tuple(value)


def _boolean(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _outcome(channel: DecisionChannel, candidate_id: str) -> str:
    if candidate_id == "000":
        return "unsupported"
    if channel is DecisionChannel.BODY:
        return "no_action" if candidate_id == "IB0" else "execute"
    return "keep" if candidate_id == "IF0" else "apply"


def _request_digest(request: ActionDecisionRequest) -> str:
    value = {
        "request": {
            key: (item.value if isinstance(item, DecisionChannel) else item)
            for key, item in request.__dict__.items()
            if key != "media"
        },
        "media": [
            {
                "media_id": item.media_id,
                "kind": item.kind,
                "checksum": item.checksum,
                "start_ms": item.start_ms,
                "end_ms": item.end_ms,
                "evidence_role": item.evidence_role,
            }
            for item in request.media
        ],
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
