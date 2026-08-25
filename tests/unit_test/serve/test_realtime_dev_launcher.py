# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_enabled_dev_model_bypasses_pipeline_runner(monkeypatch) -> None:
    from sglang_omni.serve import launcher

    class ForbiddenRunner:
        def __init__(self, pipeline_config) -> None:
            raise AssertionError("development mode must not construct a pipeline")

    app = object()
    serve = AsyncMock(return_value=None)
    captured: dict[str, object] = {}

    def fake_create_app(client, **kwargs):
        captured["client"] = client
        captured["kwargs"] = kwargs
        return app

    monkeypatch.setenv("SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED", "true")
    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", ForbiddenRunner)
    monkeypatch.setattr(launcher, "create_app", fake_create_app)
    monkeypatch.setattr(
        launcher,
        "install_dev_model_error_handler",
        lambda installed_app: captured.setdefault("error_handler_app", installed_app),
    )
    monkeypatch.setattr(launcher.uvicorn.Server, "serve", serve)

    await launcher._run_server(
        SimpleNamespace(name="fake-pipeline"),
        port=8123,
        enable_realtime=True,
    )

    serve.assert_awaited_once()
    assert captured["client"].health()["running"] is True
    assert captured["kwargs"]["enable_realtime"] is True
    assert captured["kwargs"]["model_name"] == "fake-pipeline"
    assert captured["error_handler_app"] is app


@pytest.mark.asyncio
async def test_enabled_dev_model_requires_realtime_endpoint(monkeypatch) -> None:
    from sglang_omni.serve import launcher

    monkeypatch.setenv("SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED", "true")
    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    with pytest.raises(ValueError, match="requires --enable-realtime"):
        await launcher._run_server(SimpleNamespace(name="fake"))


@pytest.mark.asyncio
async def test_disabled_dev_model_keeps_pipeline_path(monkeypatch) -> None:
    from sglang_omni.serve import launcher

    constructed: list[object] = []

    class MarkerRunner:
        def __init__(self, pipeline_config) -> None:
            constructed.append(pipeline_config)
            raise RuntimeError("production pipeline marker")

    monkeypatch.delenv("SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED", raising=False)
    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", MarkerRunner)
    config = SimpleNamespace(name="real-pipeline")
    with pytest.raises(RuntimeError, match="production pipeline marker"):
        await launcher._run_server(config)
    assert constructed == [config]
