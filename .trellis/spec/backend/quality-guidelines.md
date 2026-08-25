# Backend Quality Guidelines

## Baseline

The project targets Python 3.10+ and uses Black, isort, autoflake, AST/YAML/TOML checks, trailing-whitespace checks, private-key detection, and debug-statement rejection through `.pre-commit-config.yaml`. New Python code should be Black-compatible, use explicit imports, type public boundaries, and retain SPDX headers where surrounding files use them.

Runtime/test dependencies are declared in `pyproject.toml`, including compatibility-sensitive pinned ML packages. Keep generic layers independent from model-specific imports and use existing lazy-import patterns where import cost or optional availability matters.

## Required Properties

- Preserve async event-loop responsiveness: do not add blocking network, filesystem, queue, or GPU synchronization to FastAPI/streaming paths.
- Make lifecycle ownership explicit. Tasks, streams, subprocesses, queues, and counters require deterministic cancellation/close/release paths on partial startup and disconnect.
- Centralize typed wire contracts in `sglang_omni/serve/protocol.py`, `sglang_omni/proto/`, or `sglang_omni_router/route_metadata.py`; update every producer and consumer together.
- Preserve streaming order and terminal semantics. Reference `tests/unit_test/pipeline/test_stage.py`, `tests/unit_test/serve/test_speech_ws.py`, and `tests/unit_test/router/test_app.py`.
- Avoid module-level mutable runtime state. Attach state to app/session/runner objects and inject dependencies, as in `create_app`, `RealtimeSessionManager`, and `ProxyHandler`.

## Testing and Review

Add focused tests under the matching `tests/unit_test/<area>/` directory. Prefer deterministic fakes and `monkeypatch` over real models, networks, or GPUs. Use `TestClient` for route/WebSocket contracts and the async style already used by the neighboring test file.

Cover the happy path plus relevant validation, timeout, cancellation, disconnect, partial stream, cleanup, and idempotency cases. Mark hardware-heavy tests with the configured `gpu` or `benchmark` markers; model-serving integration belongs in `tests/test_model/`.

Run narrow affected tests first, then neighboring suites. Before handoff, run `pre-commit run --all-files` when feasible or at minimum relevant pytest targets and formatting checks.

Review package boundaries, error mapping, lifecycle cleanup, streaming terminal events, bounded/safe logs, backward-compatible configuration threading, and observable test coverage. Avoid bare `except`, silent broad exception handling, unowned fire-and-forget tasks, unbounded streaming buffers, `print`/debug breakpoints, and model downloads in ordinary control-plane unit tests.

## Pre-Implementation Git Checkpoint

Every Trellis task must have a local Git checkpoint after its planning artifacts are reviewed and before `task.py start` changes the task to `in_progress`.

- Commit only the reviewed task directory and other explicitly approved paths belonging to that task.
- List unrelated or unrecognized dirty files separately and leave them untouched.
- Obtain one confirmation for the exact commit message and file set before staging.
- Do not push, amend, stash, reset, clean, or use broad staging such as `git add .` for this checkpoint.
- Verify the checkpoint commit exists before implementation begins. If task changes overlap unrelated work and cannot be isolated, stop rather than creating an unsafe archive.

This checkpoint is a recoverable planning baseline. It does not replace the normal implementation, quality-check, final commit, archive, and journal steps.

## Development Substitute Boundaries

When a local development substitute intentionally implements only a narrow
client protocol, its transport surface must be narrowed at application setup as
well. New HTTP or WebSocket routes added to the shared app must not become
implicitly available in substitute mode and fail only after accepting a request.

- Keep the supported HTTP and WebSocket route sets explicit and test their exact
  contents when the shared application registers routes globally.
- Branch before production-only catalog loading, model imports, GPU discovery,
  pipeline construction, warmup, profiler setup, and runtime watchers.
- Keep production initialization after the development branch unchanged; test
  both enabled bypass behavior and disabled production behavior.
- Prefer a standalone development entry point when the production CLI must
  resolve model-specific configuration before reaching the shared launcher.
- A Session Realtime substitute must reuse the production `session.start`,
  Turn, media ACK, cancellation, provisional-reply, and terminal-event parser.
  Inject only narrow model/action capabilities; do not recreate the external
  WebSocket protocol in the fake client.
- Keep development action selection inside the Session whitelist. If the
  production global catalog is intentionally unavailable, any development-only
  inline-catalog projection must be opt-in and default off for production apps.

## Change Reporting

较大改动完成后必须在当前 `.trellis/tasks/<task>/` 下形成中文复盘报告，供开发者与组长
评审和后续接手。较大改动包括：新增或迁移公共接口、跨层数据流、启动/部署方式、环境
变量合同、并发或生命周期语义，或者修改多个核心模块的功能任务。

报告至少包含：目标与边界、最终链路、主要文件和行为变化、启用与停止步骤、验证证据、
已知限制、回滚或后续工作，并区分已提交内容与尚未提交内容。停止说明必须区分“终止
当前进程”和“禁用下次启动”；修改环境变量本身不能被描述为停止已运行服务。

小型局部修改不新增报告文档，在最终回复末尾用一段话说明修改文件、行为和验证结果即可。
