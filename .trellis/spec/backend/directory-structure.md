# Backend Directory Structure

## Package Boundaries

- `sglang_omni/` is the core package. Keep public entry points in `cli/`, client contracts in `client/`, API adapters in `serve/`, inter-stage contracts in `proto/`, orchestration in `pipeline/`, scheduling in `scheduling/`, and transports in `comm/` and `relay/`.
- `sglang_omni/models/<model_name>/` owns model-specific topology, request builders, stages, and kernels. Follow packages such as `sglang_omni/models/qwen3_omni/` instead of adding model branches to generic serving code.
- `sglang_omni_router/` is a separate lightweight FastAPI reverse proxy. Configuration belongs in `config.py`, request metadata in `route_metadata.py`, selection in `selector.py`, forwarding in `proxy.py`, health state in `health.py`/`worker.py`, and startup in `serve.py`.
- `tests/unit_test/` mirrors the component under test. GPU/model integration tests belong in `tests/test_model/`; shared fakes belong in `tests/unit_test/fixtures/` or a local `helpers.py`.

## Serving and Pipeline Features

Place request/response schemas in `sglang_omni/serve/protocol.py`, not inline in route handlers. `sglang_omni/serve/openai_api.py` creates the app and registers routes. Extract cohesive stateful features as demonstrated by `speech_service.py`, `speech_ws.py`, `realtime/`, and `transcription_adapters/`. Route handlers should read validated inputs, obtain dependencies from `app.state`, call a client/service, and translate the result to the external protocol. Do not import model implementations into generic serving or router modules.

Use `sglang_omni/pipeline/runtime_config.py` for topology/configuration types, `coordinator.py` and `mp_runner.py` for lifecycle ownership, and `pipeline/stage/` for generic execution and stream routing. Wire-visible structures belong in `sglang_omni/proto/`; update their producers, consumers, serialization, and tests together.

Put helpers in the narrowest applicable package. Use `sglang_omni/utils/` only for cross-layer utilities; user-media normalization belongs in `preprocessing/`, while audio response encoding belongs in `client/audio.py`.

## Naming and Imports

- Use `snake_case.py`, `snake_case` functions/variables, `PascalCase` classes, and uppercase constants.
- Prefer explicit typed constructor dependencies, as in `RealtimeSessionManager` and `ProxyHandler`, over hidden globals.
- Keep `from __future__ import annotations` in modern modules and use built-in generics on Python 3.10+.
- Package `__init__.py` files may expose a small public surface or lazy imports; see `sglang_omni/serve/__init__.py` and `sglang_omni/pipeline/__init__.py`.
