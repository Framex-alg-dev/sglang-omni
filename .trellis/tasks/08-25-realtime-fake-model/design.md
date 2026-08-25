# Realtime 本地开发模型替身：技术设计

## 1. 替换位置

替身位于 serving 与 Pipeline 的边界，而不是模型包内部：

```text
真实模式：launch_server -> MultiProcessPipelineRunner -> Client -> create_app
开发模式：launch_server -> DevRealtimeModelClient ----------> create_app
```

环境变量在 `_run_server()` 入口解析；开发模式在实例化 `MultiProcessPipelineRunner` 前进入独立开发启动分支。这样保留真实 FastAPI、Realtime、VAD 和 WebSocket 链路，同时完全绕过模型和 GPU。

## 2. 模块边界

建议新增：

```text
sglang_omni/serve/realtime/dev_model.py
```

包含：

- `DevRealtimeModelConfig`：严格解析环境变量；
- `DevRealtimeModelClient`：实现 Realtime 所需的 `completion_stream()`、`abort()`；
- 请求分类与校验；
- 固定文本 chunk 状态机。

不继承真实 `Client`，因为它必须持有 `Coordinator`。应为 Realtime 引入窄 client Protocol，避免开发替身假装支持整个 Client API。

## 3. 状态机

```text
Idle -> Validating(request) -> Streaming(request_id, chunk_index) -> Finished
                    |                         |
                    +-> Failed               +-> Aborted
```

单次调用是独立 async generator：先完成全部合同校验，再按 Unicode 字符切片配置文本；每段按 interval 等待后返回 `CompletionStreamChunk`；最后返回单独的 stop chunk。

response/transcription 分类依据 `build_response_request()` 和 `build_transcription_request()` 的稳定 prompt 特征，集中在一个函数并有单测。未来 prompt 改动必须同步分类测试，不能静默选择默认输出。

## 4. 启动与生命周期

开发模式仍使用 uvicorn 和 `create_app()`，但不创建 runner、coordinator、profiler control 或 pipeline failure watcher。服务退出时只清理 Realtime sessions。

若完整 app 暴露其他端点，替身 client 对不支持操作返回明确的开发态 unsupported 错误。文档声明开发模式只保证 `/health`、`/v1/models` 和 `/v1/realtime`。

正式模式的 runner 启动、watcher 和 finally stop 应尽量保持原结构，通过早期分支隔离开发路径。

## 5. 环境变量与防误用

统一使用 `SGLANG_OMNI_DEV_FAKE_MODEL_` 前缀。enabled 默认 false，只有明确真值才能启用。启动时以 warning 显示：

```text
DEV FAKE REALTIME MODEL ENABLED: pipeline and model loading are bypassed
```

只记录响应字符数、chunk size 和 interval，不记录完整配置文本。服务器联调和生产部署必须移除 enabled 或设为 false。

## 6. 错误合同

| 条件 | 行为 |
|---|---|
| 环境变量非法 | uvicorn/Pipeline 启动前失败 |
| request 非 streaming/text-only | 开发态合同错误 |
| 缺少 audio metadata/messages | 开发态合同错误 |
| 无法判断 pass | 开发态合同错误，不使用默认分支 |
| abort 已知 request | 标记取消，generator 尽快结束 |
| abort 未知 request | 返回稳定未命中结果，不产生随机异常 |

## 7. 与 TTS 任务的关系

本任务完成后，TTS 任务可以在开发模式下继续使用真实 `/v1/realtime`，把确定性的文本 Delta 接到 fake/real TTS。模型替身自身永远不输出音频，也不感知 TTS 是否启用。

