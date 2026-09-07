from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Any

from sglang_omni.serve.realtime.knowledge.client import (
    KnowledgeGatewayClient,
    KnowledgeGatewayError,
)
from sglang_omni.serve.realtime.knowledge.config import RealtimeKnowledgeConfig
from sglang_omni.serve.realtime.knowledge.models import (
    KnowledgeBinding,
    KnowledgeContext,
    KnowledgeEntityHint,
    KnowledgeEvidence,
    PreparedKnowledgeTurn,
)


class KnowledgeController:
    def __init__(
        self,
        config: RealtimeKnowledgeConfig,
        client: KnowledgeGatewayClient | None,
    ) -> None:
        self.config = config
        self.client = client
        self._turn_slots = asyncio.Semaphore(config.max_concurrency)

    async def prepare_turn(
        self,
        *,
        binding: KnowledgeBinding,
        session_id: str,
        turn_id: str,
        text: str,
        hints: tuple[KnowledgeEntityHint, ...] = (),
        recent_user_turns: list[str] | None = None,
        recent_assistant_turns: list[str] | None = None,
    ) -> PreparedKnowledgeTurn:
        started = time.perf_counter()
        deadline = time.monotonic() + self.config.turn_timeout_ms / 1000
        request_id = f"{session_id}:{turn_id}:knowledge"
        common = {
            "request_id": request_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "snapshot_id": binding.snapshot_id,
            "state_token": binding.state_token,
            "started_at": started,
            "deadline_at": deadline,
        }
        if binding.status != "ready" or self.client is None:
            return PreparedKnowledgeTurn(
                **common, preparation_id=None, payload={}, error_code="GATEWAY_DISABLED"
            )
        try:
            try:
                await asyncio.wait_for(
                    self._turn_slots.acquire(),
                    timeout=max(0.001, deadline - time.monotonic()),
                )
            except (asyncio.TimeoutError, TimeoutError):
                return PreparedKnowledgeTurn(
                    **common,
                    preparation_id=None,
                    payload={},
                    error_code="CLIENT_BULKHEAD_FULL",
                )
            try:
                remaining_ms = math.ceil((deadline - time.monotonic()) * 1000)
                prepare_ms = remaining_ms - self.config.commit_reserve_ms
                if prepare_ms < 10:
                    return PreparedKnowledgeTurn(
                        **common,
                        preparation_id=None,
                        payload={},
                        error_code="DEADLINE_EXCEEDED",
                    )
                payload = self._turn_payload(
                    binding=binding,
                    session_id=session_id,
                    turn_id=turn_id,
                    text=text,
                    hints=hints,
                    recent_user_turns=recent_user_turns,
                    recent_assistant_turns=recent_assistant_turns,
                    deadline_ms=min(2_000, prepare_ms),
                )
                result = await asyncio.wait_for(
                    self.client.prepare_turn(payload),
                    timeout=max(0.001, prepare_ms / 1000),
                )
            finally:
                self._turn_slots.release()
        except (asyncio.TimeoutError, TimeoutError):
            return PreparedKnowledgeTurn(
                **common,
                preparation_id=None,
                payload={},
                error_code="DEADLINE_EXCEEDED",
            )
        except KnowledgeGatewayError as exc:
            return PreparedKnowledgeTurn(
                **common,
                preparation_id=None,
                payload={},
                error_code=exc.code,
            )
        preparation_id = result.get("preparation_id")
        if not isinstance(preparation_id, str) or not preparation_id:
            return PreparedKnowledgeTurn(
                **common,
                preparation_id=None,
                payload={},
                error_code="INVALID_GATEWAY_RESPONSE",
            )
        return PreparedKnowledgeTurn(
            **common,
            preparation_id=preparation_id,
            payload=result,
        )

    async def commit_prepared_turn(
        self,
        *,
        binding: KnowledgeBinding,
        prepared: PreparedKnowledgeTurn,
    ) -> KnowledgeContext:
        if prepared.error_code or prepared.preparation_id is None or self.client is None:
            return self._degraded(
                binding, prepared.error_code or "GATEWAY_DISABLED", prepared.started_at
            )
        payload = {
            "request_id": prepared.request_id,
            "tenant_id": binding.tenant_id,
            "session_id": prepared.session_id,
            "turn_id": prepared.turn_id,
            "snapshot_id": prepared.snapshot_id,
            "state_token": prepared.state_token,
            "preparation_id": prepared.preparation_id,
        }
        remaining = prepared.deadline_at - time.monotonic()
        timeout = (
            remaining
            if remaining > 0
            else self.config.commit_recovery_timeout_ms / 1000
        )
        try:
            result = await asyncio.wait_for(
                self.client.commit_turn(payload), timeout=max(0.01, timeout)
            )
        except (asyncio.TimeoutError, TimeoutError):
            try:
                result = await asyncio.wait_for(
                    self.client.commit_turn(payload),
                    timeout=self.config.commit_recovery_timeout_ms / 1000,
                )
            except (
                asyncio.TimeoutError,
                TimeoutError,
                KnowledgeGatewayError,
            ) as exc:
                code = self._commit_error_code(exc)
                return self._degraded(binding, code, prepared.started_at)
        except KnowledgeGatewayError as exc:
            return self._degraded(
                binding, self._commit_error_code(exc), prepared.started_at
            )
        return self._context_from_result(binding, result, prepared.started_at)

    @staticmethod
    def _commit_error_code(exc: BaseException) -> str:
        if isinstance(exc, KnowledgeGatewayError) and exc.code in {
            "STATE_VERSION_CONFLICT",
            "PREPARATION_EXPIRED",
            "PREPARATION_MISMATCH",
        }:
            return exc.code
        return "COMMIT_OUTCOME_UNKNOWN"

    def _turn_payload(
        self,
        *,
        binding: KnowledgeBinding,
        session_id: str,
        turn_id: str,
        text: str,
        hints: tuple[KnowledgeEntityHint, ...],
        recent_user_turns: list[str] | None,
        recent_assistant_turns: list[str] | None,
        deadline_ms: int,
    ) -> dict[str, Any]:
        return {
            "request_id": f"{session_id}:{turn_id}:knowledge",
            "tenant_id": binding.tenant_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "snapshot_id": binding.snapshot_id,
            "state_token": binding.state_token,
            "input": {
                "text": text,
                "entity_hints": [item.as_dict() for item in hints],
            },
            "context": {
                "recent_user_turns": recent_user_turns or [],
                "recent_assistant_turns": recent_assistant_turns or [],
            },
            "limits": {
                "deadline_ms": deadline_ms,
                "max_evidence": self.config.max_evidence,
                "max_context_chars": self.config.max_context_chars,
            },
        }

    async def resolve_session(
        self,
        *,
        session_id: str,
        tenant_id: str | None,
        binding_id: str,
        required: bool,
        locale: str,
        binding_revision: int | None = None,
    ) -> KnowledgeBinding:
        resolved_tenant = tenant_id or self.config.default_tenant_id
        if not self.config.enabled or self.client is None:
            if required:
                raise ValueError("knowledge binding requires an enabled Knowledge Gateway")
            return KnowledgeBinding(
                binding_id=binding_id,
                required=False,
                tenant_id=resolved_tenant or "",
                snapshot_id="",
                state_token="",
                binding_revision=binding_revision,
                status="degraded",
            )
        if not resolved_tenant:
            raise ValueError(
                "knowledge binding requires an authenticated x-tenant-id header "
                "or a configured default tenant"
            )
        try:
            payload: dict[str, Any] = {
                    "request_id": f"{session_id}:knowledge-start",
                    "tenant_id": resolved_tenant,
                    "session_id": session_id,
                    "binding_id": binding_id,
                    "locale": locale,
                }
            if binding_revision is not None:
                payload["binding_revision"] = binding_revision
            result = await self.client.resolve_session(payload)
        except KnowledgeGatewayError:
            if required:
                raise
            return KnowledgeBinding(
                binding_id=binding_id,
                required=False,
                tenant_id=resolved_tenant,
                snapshot_id="",
                state_token="",
                binding_revision=binding_revision,
                status="degraded",
            )
        snapshot_id = result.get("snapshot_id")
        state_token = result.get("state_token")
        if (
            result.get("status") != "ready"
            or not isinstance(snapshot_id, str)
            or not snapshot_id
            or not isinstance(state_token, str)
            or not state_token
        ):
            if required:
                raise KnowledgeGatewayError(
                    "INVALID_GATEWAY_RESPONSE",
                    "gateway returned an invalid session binding",
                )
            return KnowledgeBinding(
                binding_id=binding_id,
                required=False,
                tenant_id=resolved_tenant,
                snapshot_id="",
                state_token="",
                binding_revision=binding_revision,
                status="degraded",
            )
        return KnowledgeBinding(
            binding_id=binding_id,
            required=required,
            tenant_id=resolved_tenant,
            snapshot_id=snapshot_id,
            state_token=state_token,
            binding_revision=binding_revision,
        )

    async def resolve_turn(
        self,
        *,
        binding: KnowledgeBinding,
        session_id: str,
        turn_id: str,
        text: str,
        hints: tuple[KnowledgeEntityHint, ...] = (),
        recent_user_turns: list[str] | None = None,
        recent_assistant_turns: list[str] | None = None,
    ) -> KnowledgeContext:
        started = time.perf_counter()
        if binding.status != "ready" or self.client is None:
            return self._degraded(binding, "GATEWAY_DISABLED", started)
        deadline = time.monotonic() + self.config.turn_timeout_ms / 1000
        try:
            try:
                await asyncio.wait_for(
                    self._turn_slots.acquire(),
                    timeout=max(0.001, deadline - time.monotonic()),
                )
            except (asyncio.TimeoutError, TimeoutError):
                return self._degraded(binding, "CLIENT_BULKHEAD_FULL", started)
            try:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    return self._degraded(binding, "DEADLINE_EXCEEDED", started)
                service_deadline_ms = min(
                    2_000,
                    max(10, math.ceil(remaining_seconds * 1000) - 100),
                )
                result = await asyncio.wait_for(
                    self.client.resolve_turn(
                        {
                            "request_id": f"{session_id}:{turn_id}:knowledge",
                            "tenant_id": binding.tenant_id,
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "snapshot_id": binding.snapshot_id,
                            "state_token": binding.state_token,
                            "input": {
                                "text": text,
                                "entity_hints": [item.as_dict() for item in hints],
                            },
                            "context": {
                                "recent_user_turns": recent_user_turns or [],
                                "recent_assistant_turns": recent_assistant_turns or [],
                            },
                            "limits": {
                                # Keep the Knowledge Service budget inside one
                                # absolute deadline, including bulkhead wait.
                                "deadline_ms": service_deadline_ms,
                                "max_evidence": self.config.max_evidence,
                                "max_context_chars": self.config.max_context_chars,
                            },
                        }
                    ),
                    timeout=remaining_seconds,
                )
            except (asyncio.TimeoutError, TimeoutError):
                return self._degraded(binding, "DEADLINE_EXCEEDED", started)
            finally:
                self._turn_slots.release()
        except KnowledgeGatewayError as exc:
            return self._degraded(binding, exc.code, started)
        return self._context_from_result(binding, result, started)

    async def script_event(
        self,
        *,
        binding: KnowledgeBinding,
        session_id: str,
        turn_id: str,
        script_id: str,
        event: str,
        script_version: int | None = None,
        checksum: str | None = None,
    ) -> KnowledgeBinding:
        if binding.status != "ready" or self.client is None:
            if binding.required:
                raise KnowledgeGatewayError(
                    "GATEWAY_DISABLED", "knowledge gateway is unavailable"
                )
            return binding
        if checksum is None:
            raise KnowledgeGatewayError(
                "SCRIPT_CHECKSUM_REQUIRED",
                "a checksum computed from the provided script text is required",
            )
        payload: dict[str, Any] = {
            "request_id": f"{session_id}:{turn_id}:script:{event}",
            "tenant_id": binding.tenant_id,
            "session_id": session_id,
            "snapshot_id": binding.snapshot_id,
            "state_token": binding.state_token,
            "script_id": script_id,
            "event": event,
        }
        if script_version is not None:
            payload["script_version"] = script_version
        payload["checksum"] = checksum
        try:
            result = await self.client.script_event(payload)
        except KnowledgeGatewayError:
            if binding.required:
                raise
            return binding
        state_token = result.get("state_token")
        if not isinstance(state_token, str) or not state_token:
            if binding.required:
                raise KnowledgeGatewayError(
                    "INVALID_GATEWAY_RESPONSE",
                    "gateway returned an invalid script event response",
                )
            return binding
        return KnowledgeBinding(
            binding_id=binding.binding_id,
            required=binding.required,
            tenant_id=binding.tenant_id,
            snapshot_id=binding.snapshot_id,
            state_token=state_token,
            binding_revision=binding.binding_revision,
            status=binding.status,
        )

    def _context_from_result(
        self,
        binding: KnowledgeBinding,
        result: dict[str, Any],
        started: float,
    ) -> KnowledgeContext:
        decision = result.get("decision")
        if decision not in {"SKIP", "RETRIEVE", "CLARIFY", "DEGRADED"}:
            return self._degraded(binding, "INVALID_GATEWAY_RESPONSE", started)
        evidence_items: list[KnowledgeEvidence] = []
        remaining_chars = self.config.max_context_chars
        raw_evidence = result.get("evidence", [])
        if not isinstance(raw_evidence, list):
            return self._degraded(binding, "INVALID_GATEWAY_RESPONSE", started)
        for item in raw_evidence[: self.config.max_evidence]:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            content = str(item["content"])[:remaining_chars]
            if not content:
                break
            try:
                authority = int(item.get("authority", 0))
            except (TypeError, ValueError):
                authority = 0
            metadata = item.get("metadata", {})
            evidence_items.append(
                KnowledgeEvidence(
                    evidence_id=str(item.get("evidence_id", "")),
                    source_type=str(item.get("source_type", "unknown")),
                    source_id=str(item.get("source_id", "")),
                    title=str(item.get("title", ""))[:512],
                    content=content,
                    authority=max(0, min(authority, 1_000)),
                    updated_at=(
                        str(item["updated_at"])
                        if item.get("updated_at") is not None
                        else None
                    ),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                )
            )
            remaining_chars -= len(content)
        clarification = result.get("clarification")
        return KnowledgeContext(
            decision=decision,
            reason=str(result.get("reason", "unknown")),
            result_id=str(result.get("result_id", f"kr_{uuid.uuid4().hex}")),
            state_token=(
                str(result.get("state_token"))
                if result.get("state_token")
                else binding.state_token
            ),
            snapshot_id=binding.snapshot_id,
            capabilities=tuple(result.get("capabilities", [])),
            evidence=tuple(evidence_items),
            clarification_reason=(
                str(clarification.get("reason"))
                if isinstance(clarification, dict) and clarification.get("reason")
                else None
            ),
            degraded_code=result.get("degraded_code"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _degraded(
        binding: KnowledgeBinding, code: str, started: float
    ) -> KnowledgeContext:
        return KnowledgeContext(
            decision="DEGRADED",
            reason="knowledge_gateway_unavailable",
            result_id=f"kr_{uuid.uuid4().hex}",
            state_token=binding.state_token,
            snapshot_id=binding.snapshot_id,
            degraded_code=code,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
