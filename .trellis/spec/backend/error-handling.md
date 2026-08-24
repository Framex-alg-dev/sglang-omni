# Error Handling

## Boundary and Domain Errors

Use Pydantic models for external configuration and request schemas (`sglang_omni/serve/protocol.py`, `sglang_omni_router/config.py`). Raise `ValueError` or `TypeError` for invalid internal arguments and protocol invariants, and `RuntimeError` for invalid lifecycle/runtime state; examples live in `sglang_omni/pipeline/stage/runtime.py` and `sglang_omni/pipeline/mp_runner.py`.

Use a narrow domain exception when a boundary must map failures consistently. TTS serving uses `SpeechAPIError` and helper constructors in `sglang_omni/serve/speech_errors.py`; router metadata and worker selection use `RouteMetadataError` and `NoEligibleWorkerError`. Catch these at the transport boundary, not deep inside helpers.

## HTTP and WebSocket Mapping

OpenAI-compatible speech endpoints return an `error` envelope with `message`, `type`, `param`, and `code`; reuse `sglang_omni/serve/speech_errors.py` for speech APIs. Other routes still use FastAPI `HTTPException` in places, so verify the neighboring endpoint's public contract instead of applying the speech envelope globally. `sglang_omni_router/proxy.py` explicitly maps invalid metadata/body to 400, oversized payloads to 413, unavailable workers to 503, and upstream transport failures to 502.

Do not expose tracebacks or arbitrary exception text in public 5xx payloads. Log unexpected failures with context and return a stable server error. WebSocket handlers should send protocol-shaped errors while the connection is writable and then close cleanly; follow `sglang_omni/serve/speech_ws.py` and `sglang_omni/serve/realtime/session.py`.

## Streaming, Cancellation, and Cleanup

Streaming code must have one explicit cleanup owner, and cleanup must be idempotent. `_RelayCleanup`/`_RelayResponse` in `sglang_omni_router/proxy.py` close upstream responses and release counters on completion, failure, or disconnect. `_ClosableStreamingResponse` in `sglang_omni/serve/openai_api.py` closes its iterator at the ASGI boundary.

Handle `asyncio.CancelledError`, `WebSocketDisconnect`, `KeyboardInterrupt`, and `SystemExit` separately from application errors. Clean up, then re-raise or end the connection; never turn cancellation into an ordinary 500. Use `try/finally` when one failed close could otherwise skip resource release.

Broad `except Exception` is acceptable only at a process, background-task, or transport ownership boundary where it records failure and guarantees teardown (`pipeline/stage_workers.py`, `pipeline/stage/runtime.py`, `sglang_omni_router/health.py`). Never silently swallow it.

## Tests

Test both external mapping and lifecycle side effects. See `tests/unit_test/serve/test_speech_ws.py` for WebSocket errors/cancellation, `tests/unit_test/router/test_app.py` for proxy cleanup failures, and `tests/unit_test/pipeline/test_stage.py` for routing/state errors. Use `pytest.raises(..., match=...)` for internal contracts and assert exact status/error fields for APIs.
