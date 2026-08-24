# Logging Guidelines

## Setup

Use standard-library logging and a module logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Do not configure global logging at import time. Entry points/process workers own configuration: `sglang_omni_router/serve.py` applies `dictConfig`, while `sglang_omni/pipeline/stage_workers.py` configures worker output.

## Levels and Context

- `debug`: high-frequency diagnostics and expected probe failures, as in `sglang_omni_router/health.py`.
- `info`: lifecycle transitions and low-frequency operational events, as in `pipeline/stage/runtime.py` and `serve/realtime/manager.py`.
- `warning`: recoverable degradation, timeout, rejected routing, or cleanup failure where service continues.
- `error`/`exception`: lost work, fatal background/process failure, or an unexpected boundary failure. Use `logger.exception` inside an exception handler when a traceback is needed.

Include stable correlation context: request/session ID, stage/process name, worker ID, route path, status, and elapsed time when available. Prefer lazy parameterized logging (`logger.info("Stage %s started", name)`) for new code, especially on hot paths.

## Safety and Volume

Never log API keys, authorization/cookie headers, raw request bodies, base64 media, audio samples, model weights, or full generated content. Log bounded metadata such as byte/chunk counts, model, format, duration, and IDs. Avoid per-token or per-frame `info` logs.

Do not log the same exception at every layer. The layer that handles, retries, or terminates the operation owns the log; lower helpers should raise a useful typed exception.
