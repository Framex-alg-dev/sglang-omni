"""Run the CPU-only inference gateway from environment configuration."""

from __future__ import annotations

import os

import uvicorn

from .app import InferenceGatewayConfig, UpstreamStage, create_inference_gateway_app
from .speech_synthesis import GatewaySpeechConfig


def _bearer(name: str, *, fallback: str | tuple[str, ...] | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if not value and fallback:
        names = (fallback,) if isinstance(fallback, str) else fallback
        for candidate in names:
            value = os.environ.get(candidate, "").strip()
            if value:
                break
    return f"Bearer {value}" if value else ""


def _boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def main() -> None:
    speech_url = os.environ.get(
        "SGLANG_OMNI_GATEWAY_SPEECH_URL",
        "http://127.0.0.1:50001/v1/tts/stream",
    ).strip()
    config = InferenceGatewayConfig(
        stages={
            "classifier": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_CLASSIFIER_URL",
                    "ws://127.0.0.1:18006/classifier/v1/task-classification/realtime",
                ),
                _bearer("SGLANG_OMNI_TASK_CLASSIFICATION_TOKEN"),
            ),
            "brain": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_BRAIN_URL",
                    "ws://127.0.0.1:18006/brain/v1/chat/completions/realtime",
                ),
                _bearer("SGLANG_OMNI_TASK_BRAIN_TOKEN"),
            ),
            "reply": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_REPLY_URL",
                    "ws://127.0.0.1:18005/v1/chat/completions/realtime",
                )
            ),
            "body": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_BODY_URL",
                    "ws://127.0.0.1:18004/v1/action-decision/realtime",
                ),
                _bearer(
                    "SGLANG_OMNI_ACTION_DECISION_TOKEN",
                    fallback=(
                        "SGLANG_OMNI_INTERNAL_MODEL_TOKEN",
                        "SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN",
                    ),
                ),
            ),
            "expression": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_EXPRESSION_URL",
                    "ws://127.0.0.1:18004/v1/action-decision/realtime",
                ),
                _bearer(
                    "SGLANG_OMNI_ACTION_DECISION_TOKEN",
                    fallback=(
                        "SGLANG_OMNI_INTERNAL_MODEL_TOKEN",
                        "SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN",
                    ),
                ),
            ),
            "performance": UpstreamStage(
                os.environ.get(
                    "SGLANG_OMNI_GATEWAY_PERFORMANCE_URL",
                    "ws://127.0.0.1:18004/v1/performance-control/realtime",
                ),
                _bearer(
                    "SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN",
                    fallback=(
                        "SGLANG_OMNI_ACTION_DECISION_TOKEN",
                        "SGLANG_OMNI_INTERNAL_MODEL_TOKEN",
                    ),
                ),
            ),
        },
        request_timeout_seconds=float(
            os.environ.get("SGLANG_OMNI_GATEWAY_REQUEST_TIMEOUT_SECONDS", "60")
        ),
        receive_idle_timeout_seconds=float(
            os.environ.get("SGLANG_OMNI_GATEWAY_IDLE_TIMEOUT_SECONDS", "300")
        ),
        reply_speech=(
            GatewaySpeechConfig(
                endpoint=speech_url,
                timeout_seconds=float(
                    os.environ.get(
                        "SGLANG_OMNI_GATEWAY_SPEECH_TIMEOUT_SECONDS",
                        "60",
                    )
                ),
            )
            if speech_url
            else None
        ),
        adaptive_plain_reply_speech=_boolean(
            "SGLANG_OMNI_GATEWAY_ADAPTIVE_PLAIN_SPEECH",
            True,
        ),
        plain_reply_segment_max_chars=int(
            os.environ.get(
                "SGLANG_OMNI_GATEWAY_PLAIN_SEGMENT_MAX_CHARS",
                "120",
            )
        ),
        plain_reply_segment_max_delay_ms=float(
            os.environ.get(
                "SGLANG_OMNI_GATEWAY_PLAIN_SEGMENT_MAX_DELAY_MS",
                "160",
            )
        ),
    )
    app = create_inference_gateway_app(config)
    uvicorn.run(
        app,
        host=os.environ.get("SGLANG_OMNI_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("SGLANG_OMNI_GATEWAY_PORT", "18003")),
        log_level=os.environ.get("SGLANG_OMNI_GATEWAY_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
