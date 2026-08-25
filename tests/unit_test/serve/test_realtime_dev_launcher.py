# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.serve.realtime.dev_model import DevRealtimeModelConfig


@pytest.mark.asyncio
async def test_enabled_dev_model_bypasses_pipeline_runner(monkeypatch) -> None:
    from sglang_omni.serve import launcher

    class ForbiddenRunner:
        def __init__(self, pipeline_config) -> None:
            raise AssertionError("development mode must not construct a pipeline")

    captured: dict[str, object] = {}

    async def fake_serve(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setenv("SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED", "true")
    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", ForbiddenRunner)
    monkeypatch.setattr(launcher, "serve_dev_realtime_model", fake_serve)
    monkeypatch.setattr(
        launcher,
        "_load_production_action_catalog_support",
        lambda: (_ for _ in ()).throw(
            AssertionError("development mode must not load the action catalog")
        ),
    )

    await launcher._run_server(
        SimpleNamespace(name="fake-pipeline"),
        port=8123,
    )

    assert captured["config"].enabled is True
    assert captured["kwargs"]["model_name"] == "fake-pipeline"


@pytest.mark.asyncio
async def test_disabled_dev_model_keeps_pipeline_path(monkeypatch) -> None:
    from sglang_omni.serve import launcher

    constructed: list[object] = []
    startup_order: list[str] = []
    catalog = SimpleNamespace(
        catalog_version="test",
        catalog_hash="hash",
        categories=[],
        candidate_count=0,
    )

    def load_catalog_support():
        startup_order.append("catalog")
        return catalog, object()

    def find_port(host, port):
        del host
        startup_order.append("port")
        return port

    class MarkerRunner:
        def __init__(self, pipeline_config) -> None:
            startup_order.append("runner")
            constructed.append(pipeline_config)
            raise RuntimeError("production pipeline marker")

    monkeypatch.delenv("SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED", raising=False)
    monkeypatch.setattr(
        launcher, "_load_production_action_catalog_support", load_catalog_support
    )
    monkeypatch.setattr(launcher, "_find_available_port", find_port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", MarkerRunner)
    config = SimpleNamespace(name="real-pipeline")
    with pytest.raises(RuntimeError, match="production pipeline marker"):
        await launcher._run_server(config)
    assert constructed == [config]
    assert startup_order == ["catalog", "port", "runner"]


@pytest.mark.asyncio
async def test_standalone_dev_server_wires_real_app_without_pipeline(
    monkeypatch,
) -> None:
    from sglang_omni.serve.realtime import dev_server

    app = object()
    captured: dict[str, object] = {}

    def fake_create_app(client, **kwargs):
        captured["client"] = client
        captured["kwargs"] = kwargs
        return app

    class FakeServer:
        def __init__(self, config) -> None:
            captured["uvicorn_config"] = config

        async def serve(self) -> None:
            captured["served"] = True

    monkeypatch.setattr(dev_server, "create_app", fake_create_app)
    monkeypatch.setattr(
        dev_server,
        "install_dev_model_error_handler",
        lambda installed_app: captured.setdefault("error_handler_app", installed_app),
    )
    monkeypatch.setattr(dev_server.uvicorn, "Server", FakeServer)
    await dev_server.serve_dev_realtime_model(
        DevRealtimeModelConfig(enabled=True),
        host="127.0.0.1",
        port=8123,
        model_name="dev-model",
        log_level="info",
    )
    assert captured["client"].health()["running"] is True
    assert captured["kwargs"]["enable_realtime"] is False
    assert captured["kwargs"]["enable_resource_monitor"] is False
    assert captured["kwargs"]["allow_unregistered_protocol_actions"] is True
    assert captured["kwargs"]["model_name"] == "dev-model"
    assert captured["error_handler_app"] is app
    assert captured["served"] is True


@pytest.mark.asyncio
async def test_standalone_dev_server_requires_enabled_switch() -> None:
    from sglang_omni.serve.realtime.dev_server import serve_dev_realtime_model

    with pytest.raises(ValueError, match="must be true"):
        await serve_dev_realtime_model(
            DevRealtimeModelConfig(enabled=False),
            host="127.0.0.1",
            port=8123,
            model_name="dev-model",
            log_level="info",
        )
