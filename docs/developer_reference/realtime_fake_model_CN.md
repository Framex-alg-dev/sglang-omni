# Session Realtime 本地开发模型替身

本功能用于在没有 GPU、模型权重和 Pipeline 子进程时调试真实
`/v1/session/realtime` Session/Turn 协议。替身支持固定文本和固定动作，不实现
TTS、回复音频、PCM 输出或 `response.audio.*`。

## 启动

先安装项目依赖。PowerShell：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是固定回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "2"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
# 可选：必须出现在 session.start 的 allowed_candidates 中
$env:SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID = "ADEV"
python -m sglang_omni.serve.realtime.dev_server --host 127.0.0.1 --port 8000
```

独立入口不解析生产模型配置，不加载生产 action catalog，也不创建 runner、
coordinator、GPU warmup、profiler、watcher 或 resource monitor。开发 app 仅暴露：

- `GET /health`
- `GET /v1/models`
- `WS /v1/session/realtime`

旧 `/v1/realtime`、speech WebSocket 和其他 HTTP API 均不开放。关闭 fake 开关时，
正式 launcher 仍按 `action catalog -> port -> Pipeline runner` 的顺序启动。

## 输出模式

`session.start.outputs` 可使用 `['text']`、`['action']` 或
`['text', 'action']`。action 模式必须提供非空 `fallback_category_ids` 和
`allowed_candidates`。开发模式只在本 Session 白名单内选择候选；配置的
candidate ID 不在本轮评分白名单中会明确失败。

当前阶段若 outputs 包含 `audio`，服务返回 `unsupported_output`。这不是静默降级；
音频输出由后续 TTS 任务扩展同一协议。

## 进程外 smoke

另开终端运行 text、action 和融合模式：

```powershell
python scripts/realtime_fake_model_smoke.py --mode text --response-text "这是固定回复。"
python scripts/realtime_fake_model_smoke.py --mode action
python scripts/realtime_fake_model_smoke.py --mode fusion --response-text "这是固定回复。"
```

脚本发送 `session.start -> turn.start -> input.text.set -> turn.commit`，验证 ACK、
文本 Delta、`response.done`、`turn.action.ready` 和 `turn.result`；随后创建第二 Turn
并验证 `turn.cancelled`。成功输出示例：

```text
PASS: Session Realtime text and cancel (N events).
```

也可用 `--audio tests/data/query_to_draw.wav` 发送真实 16 kHz、单声道、PCM16 WAV
的 raw frames；新协议由 `turn.commit` 显式提交，不依赖 VAD。超时、乱序、服务
error 或正文/动作不符会打印 `FAIL` 并非零退出。

## 关闭

服务器联调或生产部署前必须关闭：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
```

也可删除所有 `SGLANG_OMNI_DEV_FAKE_*` 环境变量。默认关闭时不会改变正式服务的
multimodal/action 行为。
