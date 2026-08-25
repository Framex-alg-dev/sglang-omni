# SPDX-License-Identifier: Apache-2.0
"""Standalone server entry point for the Realtime development model."""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from sglang_omni.serve.realtime.dev_model import (
    DevRealtimeModelClient,
    DevRealtimeModelConfig,
    install_dev_model_error_handler,
)
from sglang_omni.serve.realtime.multimodal import MultimodalSessionManager

logger = logging.getLogger(__name__)


def create_dev_app(client: DevRealtimeModelClient, *, model_name: str) -> FastAPI:
    """Build the narrow fake-model app without importing production services."""
    app = FastAPI(title="sglang-omni Realtime development model", version="0.1.0")
    app.state.client = client
    app.state.model_name = model_name
    manager = MultimodalSessionManager(
        client=client,
        model_name=model_name,
        allow_unregistered_protocol_actions=True,
    )
    app.state.multimodal_realtime_manager = manager

    @app.get("/health")
    async def health() -> JSONResponse:
        info = client.health()
        running = info.get("running", False)
        return JSONResponse(
            content={
                "status": "healthy" if running else "unhealthy",
                **info,
            },
            status_code=200 if running else 503,
        )

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "sglang-omni",
                        "root": model_name,
                    }
                ],
            }
        )

    @app.websocket("/v1/session/realtime")
    async def multimodal_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        await manager.create(websocket).run()

    install_dev_model_error_handler(app)
    return app


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
    del allowed_local_media_path, allowed_media_domains
    app = create_dev_app(DevRealtimeModelClient(config), model_name=model_name)
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
