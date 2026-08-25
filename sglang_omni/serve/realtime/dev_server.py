# SPDX-License-Identifier: Apache-2.0
"""Standalone server entry point for the Realtime development model."""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn

from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.dev_model import (
    DevRealtimeModelClient,
    DevRealtimeModelConfig,
    install_dev_model_error_handler,
)

logger = logging.getLogger(__name__)


async def serve_dev_realtime_model(
    config: DevRealtimeModelConfig,
    *,
    host: str,
    port: int,
    model_name: str,
    log_level: str,
    allowed_local_media_path: str | None = None,
    allowed_media_domains: list[str] | None = None,
) -> None:
    """Serve the real Realtime API without constructing a model pipeline."""
    if not config.enabled:
        raise ValueError(
            "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED must be true for the "
            "standalone development server"
        )
    logger.warning(
        "DEV FAKE REALTIME MODEL ENABLED: pipeline and model loading are bypassed; %s",
        config.log_summary(),
    )
    app = create_app(
        DevRealtimeModelClient(config),
        model_name=model_name,
        enable_realtime=False,
        allowed_local_media_path=allowed_local_media_path,
        allowed_media_domains=allowed_media_domains,
        architectures=[],
        enable_resource_monitor=False,
        allow_unregistered_protocol_actions=True,
    )
    install_dev_model_error_handler(app)
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_keep_alive=120,
    )
    await uvicorn.Server(uvicorn_config).serve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run /v1/session/realtime with the deterministic local model substitute."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="dev-realtime-model")
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error", "critical"),
        default="info",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(
        serve_dev_realtime_model(
            DevRealtimeModelConfig.from_env(),
            host=args.host,
            port=args.port,
            model_name=args.model_name,
            log_level=args.log_level,
        )
    )


if __name__ == "__main__":
    main()
