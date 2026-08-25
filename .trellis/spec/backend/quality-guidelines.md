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
