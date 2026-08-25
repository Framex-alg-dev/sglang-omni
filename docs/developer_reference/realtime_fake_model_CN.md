# Realtime 本地开发模型替身

本功能用于在本地没有 GPU、模型权重和 Pipeline 子进程时，调试真实的
`/v1/realtime`、WebSocket、VAD、音频缓冲和文本事件链路。它只返回固定文本，
不是生产模型，也不包含 TTS、回复音频或 PCM 生成逻辑。

## 开启方式

在启动服务前设置以下环境变量，并且保留服务原有的 `--enable-realtime` 参数：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是固定的助手回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT = "这是固定的用户转写。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "4"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
```

然后按项目现有方式启动 `sglang-omni-server`。开启后，launcher 会在创建
`MultiProcessPipelineRunner` 之前进入开发分支，因此不会创建 coordinator、
加载模型或初始化 GPU。服务仍会启动真实 FastAPI 应用，并保留 `/health`、
`/v1/models` 和 `/v1/realtime`。

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

## 关闭方式

服务器联调或生产部署前必须关闭该功能：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
```

也可以删除整组 `SGLANG_OMNI_DEV_FAKE_MODEL_*` 环境变量。默认关闭时，launcher
继续执行原有 Pipeline 启动、profiler、运行时 watcher 和资源清理路径。
