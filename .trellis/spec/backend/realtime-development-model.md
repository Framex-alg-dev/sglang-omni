# Realtime 本地开发模型替身规范

## 1. 适用范围 / 触发条件

当开发者需要在没有 GPU、模型权重或 Pipeline 子进程的本地环境验证真实
`/v1/realtime`、WebSocket、VAD、音频缓冲和文本事件链路时，才启用开发模型替身。
该模式不是生产降级方案，不得承担 TTS、回复音频或通用 HTTP 推理能力。

## 2. 接口签名

Realtime 层只依赖以下窄接口：

```python
class RealtimeModelClient(Protocol):
    def completion_stream(
        self, request: GenerateRequest, *, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]: ...

    async def abort(self, request_id: str) -> AbortResult: ...
```

启动器必须在创建 `MultiProcessPipelineRunner` 之前解析开关。启用时直接把
`DevRealtimeModelClient` 注入 `create_app()`；关闭时继续执行原有 Pipeline 路径。

无模型本地调试使用独立入口，避免正式 CLI 仍需解析生产模型配置：

```powershell
python -m sglang_omni.serve.realtime.dev_server
python scripts/realtime_fake_model_smoke.py
```

独立入口可以导入项目通用模块，但不得创建或启动 runner、coordinator、GPU stage、
profiler 或 watcher。smoke 客户端是进程外 WebSocket 验证器，不得绕过真实
`/v1/realtime` 会话边界。

## 3. 合同

环境变量均使用 `SGLANG_OMNI_DEV_FAKE_MODEL_` 前缀：

| 环境变量 | 约束 |
|---|---|
| `ENABLED` | 仅接受 `true` / `false`，默认 `false` |
| `RESPONSE_TEXT` | 非空固定助手回复 |
| `TRANSCRIPT_TEXT` | 非空固定用户转写 |
| `CHUNK_SIZE` | 大于 0 的整数 |
| `CHUNK_INTERVAL_MS` | 非负整数 |

每个 `GenerateRequest` 必须是 `stream=True`、`output_modalities=["text"]`，包含
system/user 消息、非空 request ID，以及可解码的 Base64 音频 data URI。正文 chunk
使用 `modality="text"`、`finish_reason=None`；最后单独输出
`finish_reason="stop"`，不得重复正文。

开发模式只允许 HTTP `/health`、HTTP `/v1/models` 和 WebSocket
`/v1/realtime`。其他 HTTP 路由统一返回 501，错误类型为
`dev_fake_model_unsupported`。完整自定义文本不得写入日志。

## 4. 校验与错误矩阵

| 条件 | 行为 |
|---|---|
| 任一环境变量非法 | uvicorn 和 Pipeline 启动前抛出 `ValueError` |
| 开启替身但未开启 Realtime | 启动失败并说明需要 `--enable-realtime` |
| stream、modality、messages、audio 或 request ID 非法 | 抛出稳定的开发态请求合同错误 |
| prompt 无法区分 response/transcription pass | 明确失败，不选择默认正文 |
| 非白名单 HTTP 路由 | HTTP 501 + `dev_fake_model_unsupported` |
| 已知活动 request 被 abort | 尽快停止 generator，不输出 stop chunk |
| 未知 request 被 abort | 返回稳定未命中结果，不抛随机异常 |

## 5. Good / Base / Bad Cases

- Good：开启替身和 Realtime，发送有效 PCM 输入，经真实会话/VAD 边界收到固定
  transcription 和 response 文本事件，全程不构造 Pipeline。
- Base：未设置 `ENABLED`，启动器的 runner、watcher、profiler 和清理顺序与原路径一致。
- Bad：通过前缀判断接受空 payload 或非法 Base64；或者让 `/generate` 进入生产路由后
  因缺失 client 方法返回偶然的 HTTP 500。

## 6. 必需测试

- 配置单测：断言默认值、严格布尔/整数、空文本和边界值。
- client 单测：断言两种 pass、Unicode 切片、chunk 顺序、独立 stop chunk、abort。
- 请求合同单测：断言 roles、modalities、stream、request ID 和 Base64 data URI。
- launcher 单测：开启时 runner 构造函数不得被调用；关闭时仍调用原 runner。
- Realtime 集成测试：使用真实 `create_app()`、TestClient/WebSocket、音频 buffer 和
  确定性 VAD 替身，断言 transcription/response 事件顺序。
- HTTP 边界测试：断言白名单成功，代表性 chat/generate/speech/transcription 路由为 501。
- smoke 单测：fake WebSocket 覆盖连接、发送、严格事件顺序、delta 聚合、终态文本、
  超时、意外关闭和非零失败退出。
- 进程外 smoke：读取未压缩 16 kHz、单声道、PCM16、非空 WAV，只发送 raw frames
  并追加静音触发默认 VAD；不得把 WAV header 当作输入 PCM 发送。

## 7. Wrong vs Correct

### Wrong

```python
mp_runner = MultiProcessPipelineRunner(config)
if fake_enabled:
    client = FakeClient()  # runner 和 GPU 路径已经被触发
```

### Correct

```python
fake_config = DevRealtimeModelConfig.from_env()
if fake_config.enabled:
    return await run_dev_realtime_server(fake_config)

mp_runner = MultiProcessPipelineRunner(config)
```

开发替身必须在资源所有者创建前完成分流，才能保证本地模式真正不依赖模型与 GPU。
