# Realtime 本地开发模型替身

本功能用于在本地没有 GPU、模型权重和 Pipeline 子进程时，调试真实的
`/v1/realtime`、WebSocket、VAD、音频缓冲和文本事件链路。它只返回固定文本，
不是生产模型，也不包含 TTS、回复音频或 PCM 生成逻辑。

## 依赖与测试音频

先按项目安装文档安装当前项目依赖。独立开发服务器需要 FastAPI、Uvicorn、
Silero VAD 和 ONNX Runtime；进程外检查脚本还需要 `websockets`。这些依赖都已在
项目 `pyproject.toml` 中声明。该路径不读取模型权重，也不要求 GPU。

检查脚本默认使用仓库内的 `tests/data/query_to_draw.wav`。这是 16 kHz、单声道、
PCM16 的真实语音，脚本会自动追加 1 秒静音，让默认 VAD 可靠产生
`speech_stopped`。合成正弦波、全静音或任意非语音 PCM 不能保证触发默认 VAD，
因此不要用它们判断链路是否正常。也可以通过 `--audio` 指定满足相同格式的真实
语音 WAV。

## 启动本地开发服务器

PowerShell 中可直接复制以下命令：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是固定的助手回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT = "这是固定的用户转写。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "4"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
python -m sglang_omni.serve.realtime.dev_server --host 127.0.0.1 --port 8000
```

Linux/macOS shell 对应命令为：

```bash
export SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED=true
export SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT='这是固定的助手回复。'
export SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT='这是固定的用户转写。'
export SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE=4
export SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS=30
python -m sglang_omni.serve.realtime.dev_server --host 127.0.0.1 --port 8000
```

这里使用专门的 `dev_server` 入口，因为正式 `sgl-omni serve` 命令会先解析真实
Pipeline 配置并要求 `--model-path`。开发入口直接创建替身 client 和真实 FastAPI
应用，不构造 `MultiProcessPipelineRunner`、coordinator、profiler 或运行时 watcher，
也不会加载模型或初始化 GPU。它保留 `/health`、`/v1/models` 和
`/v1/realtime`。

如果已经有合法的正式 Pipeline 配置，也可以继续使用原服务命令并同时设置开发
开关及 `--enable-realtime`；launcher 会在构造 Pipeline runner 之前进入同一个
开发服务器函数。日常无模型本地调试优先使用上面的独立入口。

环境变量说明：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | `false` | 仅接受 `true` 或 `false` |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 固定中文回复 | response pass 输出 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT` | 固定中文转写 | transcription pass 输出 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | `4` | 每个文本 delta 的最大字符数，必须大于 0 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | `0` | 相邻文本 delta 的间隔，必须为非负整数 |

替身会校验 Realtime 层构造的请求，包括流式开关、仅文本输出、system/user
消息、音频 data URI 和请求 ID。若 prompt 无法识别为回复或转写 pass，会明确
报错，防止 Realtime 请求合同变化后测试仍然假通过。完整 app 中的其他模型推理、
语音生成和转写 HTTP 接口不受支持。
开发模式只放行 `/health`、`/v1/models` 和 WebSocket `/v1/realtime`。
其他 HTTP 接口会在进入公共路由处理器前统一返回 HTTP 501，并携带
`dev_fake_model_unsupported` 错误类型，不会因为 client 缺少方法而产生意外 500。

## 运行进程外检查

保持服务器运行，在另一个终端执行：

```powershell
python scripts/realtime_fake_model_smoke.py `
  --url ws://127.0.0.1:8000/v1/realtime `
  --response-text "这是固定的助手回复。" `
  --transcript-text "这是固定的用户转写。"
```

Linux/macOS shell：

```bash
python scripts/realtime_fake_model_smoke.py \
  --url ws://127.0.0.1:8000/v1/realtime \
  --response-text '这是固定的助手回复。' \
  --transcript-text '这是固定的用户转写。'
```

脚本会发送确定性的真实语音夹具，逐项检查 VAD、commit、response 文本流、
`response.done`、transcription 文本流及其先后顺序。成功时输出类似：

```text
PASS: Realtime fake model emitted the expected response and transcription (N events).
```

连接失败、单事件等待超时、事件缺失或固定文本不一致时会在标准错误中打印
`FAIL: ...` 并以非零状态退出，适合本地脚本或 CI 调用。使用自定义真实语音时增加
`--audio path/to/pcm16_16k_mono.wav`。

## 关闭方式

服务器联调或生产部署前必须关闭该功能：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
# 或彻底删除当前 PowerShell 会话中的整组开发变量：
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS
```

也可以删除整组 `SGLANG_OMNI_DEV_FAKE_MODEL_*` 环境变量。默认关闭时，launcher
继续执行原有 Pipeline 启动、profiler、运行时 watcher 和资源清理路径。
