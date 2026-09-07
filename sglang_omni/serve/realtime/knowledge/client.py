from __future__ import annotations

from typing import Any

import httpx

from sglang_omni.serve.realtime.knowledge.config import RealtimeKnowledgeConfig


class KnowledgeGatewayError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class KnowledgeGatewayClient:
    def __init__(
        self,
        config: RealtimeKnowledgeConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=max(0.01, config.turn_timeout_ms / 1000),
                write=max(0.01, config.turn_timeout_ms / 1000),
                pool=config.connect_timeout_seconds,
            )
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self, payload: dict[str, Any]) -> dict[str, str]:
        headers = {"X-Request-ID": str(payload.get("request_id", ""))}
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            headers["X-Session-ID"] = session_id
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            headers["X-Turn-ID"] = turn_id
        if self.config.service_token:
            headers["Authorization"] = f"Bearer {self.config.service_token}"
        return headers

    async def resolve_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/knowledge/sessions:resolve", payload)

    async def resolve_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/knowledge/turns:resolve", payload)

    async def prepare_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/knowledge/turns:prepare", payload)

    async def commit_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/knowledge/turns:commit", payload, retry=True)

    async def script_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._post("/v1/knowledge/scripts:events", payload, retry=True)

    async def _post(
        self, path: str, payload: dict[str, Any], *, retry: bool = False
    ) -> dict[str, Any]:
        attempts = 2 if retry else 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    f"{self.config.url}{path}",
                    headers=self._headers(payload),
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                if attempt + 1 < attempts:
                    continue
                raise KnowledgeGatewayError(
                    "DEADLINE_EXCEEDED", str(exc), retryable=True
                ) from exc
            except httpx.HTTPError as exc:
                if attempt + 1 < attempts:
                    continue
                raise KnowledgeGatewayError(
                    "GATEWAY_UNAVAILABLE", str(exc), retryable=True
                ) from exc
            try:
                body = response.json()
            except ValueError as exc:
                raise KnowledgeGatewayError(
                    "INVALID_GATEWAY_RESPONSE", "gateway returned non-JSON response"
                ) from exc
            if response.is_error:
                error = body.get("error", {}) if isinstance(body, dict) else {}
                retryable = bool(error.get("retryable", False))
                if retryable and attempt + 1 < attempts:
                    continue
                raise KnowledgeGatewayError(
                    str(error.get("code", f"HTTP_{response.status_code}")),
                    str(error.get("message", "knowledge gateway request failed")),
                    retryable=retryable,
                )
            if not isinstance(body, dict):
                raise KnowledgeGatewayError(
                    "INVALID_GATEWAY_RESPONSE", "gateway response must be an object"
                )
            return body
        raise AssertionError("knowledge gateway retry loop exhausted")
