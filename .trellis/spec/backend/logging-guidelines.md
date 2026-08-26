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

## Scenario: Realtime 关键时间线日志

### 1. Scope / Trigger

当 Realtime WebSocket、模型流、TTS 流或 Action 流新增性能埋点时，必须遵守本节。目标是在不改变
事件顺序、不对事件循环施加文件 I/O 背压的前提下，得到可关联的服务端时间线。

### 2. Signatures

- 记录器入口：`StructuredLogWriter.log(record_type, component, event, **fields)`。
- WebSocket 发送边界：`send(...) -> bool` 或等价返回值，表示事件是否真正发送。
- 时间字段：`timestamp_unix_ns: int`、`monotonic_ns: int`、`pid: int`。

### 3. Contracts

- 调用线程捕获时间后只能执行有界、非阻塞入队；不得在 async 热路径直接写文件。
- writer 队列已满时丢弃日志并累加 `dropped`，不反压业务流。
- `monotonic_ns` 只能在相同 `pid` 内相减；跨进程对齐使用 `timestamp_unix_ns`，不将其冒充为单调耗时。
- `*_sent` 事件必须由实际发送 owner 在发送成功后记录，不能在调用发送函数之前或无条件记录。
- 默认只记录边界、首包、终态和汇总；不逐文本 Delta、逐 PCM 块持久化。
- 禁止记录正文、媒体、URL/query、voice、headers、cookies 和 token。

### 4. Validation & Error Matrix

| 条件 | 要求 |
|---|---|
| WebSocket 已断开，发送返回 `false` | 不记录 `*_sent` |
| 队列已满 | 业务继续，`dropped` 增加 |
| writer 关闭 | 用独立关闭信号通知，drain 已接收记录，不往已满业务队列塞停止标记 |
| 耗时端点不同 PID 或缺失 | 派生耗时为 `null`，不估算 |
| 后台写入失败 | 累加 write error 健康计数，不抛回 Realtime 事件循环 |

### 5. Good/Base/Bad Cases

- Good：记录 commit、首文字、首 PCM 和 done，用同 PID 的 `monotonic_ns` 派生耗时。
- Base：关闭结构化时间线时，原 WebSocket 协议和输出顺序保持不变。
- Bad：在每个 Delta 内同步 `flush()`，或在 `await send(...)` 前记录 `response_done_sent`。

### 6. Tests Required

- writer：断言批量上限、时间上限、队列溢出不阻塞、健康计数和 `close()` drain。
- 时间派生：断言同 PID 单调差值正确，跨 PID 或缺点返回 `null`。
- Realtime：覆盖 text/audio/fusion/cancel/failure，断言外部事件顺序未变，断线发送不产生伪 `*_sent`。
- 安全：检查默认记录不含请求正文、文本 Delta、PCM 或凭证。

### 7. Wrong vs Correct

```python
# Wrong: 发送失败也会产生伪时间点
timeline.log(event="response_done_sent")
await websocket.send_json(payload)

# Correct: 实际发送 owner 确认成功后才记录
sent = await send(payload)
if sent:
    timeline.log(event="response_done_sent")
```
