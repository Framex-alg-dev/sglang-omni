# Realtime 本地开发模型替身

## 目标与价值

提供一个仅用于本地开发的 Realtime 模型替身。开发者通过一组环境变量启用后，服务不启动 `MultiProcessPipelineRunner`、不加载真实多模态模型、也不需要 GPU；替身校验 `RealtimeSession` 构造的模型请求，并按固定节奏流式输出固定文本。

该任务用于快速验证 `/v1/realtime` 的输入、VAD、会话、事件和后续外围逻辑，是“Realtime 多模态回复内嵌流式 TTS”任务的前置条件。本任务不开始 TTS 迁移。

## 当前事实

- `_run_server()` 当前总是先启动 `MultiProcessPipelineRunner`，然后构造真实 `Client`；仅替换 `completion_stream()` 不能避免模型/GPU 启动。
- `RealtimeSession` 对模型侧只依赖 `completion_stream()` 和 `abort()`，流式结果类型是 `CompletionStreamChunk`。
- `create_app()` 已支持注入 client-like 对象，现有 serving 单元测试广泛使用轻量 fake client。
- Realtime response pass 和 transcription pass 都调用 `completion_stream()`；替身需要按请求内容确定输出哪一组固定文本。

## 需求

### R1：开发态环境变量

至少提供以下环境变量，并由一个 typed config 集中解析：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | `false` | 启用本地模型替身 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 固定中文回复 | response pass 的输出正文 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_TRANSCRIPT_TEXT` | 固定用户转写 | transcription pass 的输出正文 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | 正整数 | 每个文本 Delta 的最大字符数 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | 非负整数 | 相邻 Delta 的模拟间隔 |

- 布尔值和数值解析必须严格、可测试，非法值在启动阶段失败。
- 未启用时不得改变现有真实 Pipeline 路径。
- 启用时启动日志必须显著标记开发替身已开启，但不得打印完整自定义文本。

### R2：绕过真实 Pipeline

- 开关开启时必须在创建和启动 `MultiProcessPipelineRunner` 之前分支。
- 不创建 coordinator、不加载模型权重、不初始化 GPU stage、不启动 pipeline watcher/profiler control。
- 仍使用真实 `create_app()`、`/v1/realtime`、`RealtimeSessionManager`、VAD、音频 buffer 和外部 WebSocket。
- 开关关闭时执行当前 `_run_server()` 路径，不产生行为变化。

### R3：请求校验状态机

替身收到每个 `GenerateRequest` 时必须校验：

- `stream is True`；
- `output_modalities == ["text"]`；
- `messages` 非空且包含 system/user 消息；
- `metadata["audios"]` 存在、是非空列表，且元素为受支持的音频 data URI；
- request ID 非空。

根据 system/user prompt 的已知特征区分 response pass 与 transcription pass。无法识别或合同不满足时抛出稳定、可测试的开发态请求错误，不静默返回文本。

### R4：固定流式输出

- response pass 按配置的 chunk size 和 interval 输出固定回复文本。
- transcription pass 按相同规则输出固定转写文本。
- 每个正文 chunk 使用 `modality="text"` 且 `finish_reason=None`。
- 最后单独输出 `finish_reason="stop"` 的终止 chunk，不重复完整正文。
- 输出顺序、chunk 切分和 request ID 必须确定，便于事件快照和集成测试。
- `abort(request_id)` 提供与真实 client 兼容的异步接口，支持外部断连 teardown 的调用路径。

### R5：作用域与安全边界

- 模型替身只保证 Realtime 所需接口；其他模型生成、speech、transcription HTTP endpoint 不属于本任务。
- 若沿用完整 `create_app()`，访问不支持端点必须返回明确的开发态 unsupported 错误，不能因缺失方法产生意外 500。
- 该路径是显式开发功能，文档必须说明正式服务器联调前关闭环境变量。
- 本任务禁止新增 TTS WebSocket client、TTS 配置、`response.audio.*` 事件或 PCM 处理。

### R6：测试与文档

- 单测覆盖配置解析、请求校验、两类 pass、chunk 切分、终止 chunk、abort 和非法输入。
- launcher 测试证明启用时不会构造/启动 `MultiProcessPipelineRunner`，关闭时仍走原路径。
- 使用真实 FastAPI Realtime + fake model 跑本地集成测试，验证音频输入经 VAD 后得到固定 response 和 transcript。
- 增加中文本地调试说明、环境变量示例和关闭方式。

## 范围外

- TTS client、TTS 协议、回复音频事件和 PCM 生成。
- 模拟模型推理质量、tokenizer、采样、工具调用或真实 usage。
- 替代生产模型、在正式环境默认启用、支持全部 HTTP API。
- 修改数字人后端或外部 TTS 服务。

## 验收标准

- [ ] 开关开启后，可在无 GPU、无模型权重、无 Pipeline 子进程的环境启动 `/v1/realtime`。
- [ ] 输入有效音频并触发 VAD 后，外部收到确定性的固定 response 文本和固定 transcription 文本。
- [ ] 替身会验证真实 `GenerateRequest` 合同，非法请求产生稳定错误。
- [ ] chunk 切分和模拟间隔由环境变量控制，终止 chunk 不重复正文。
- [ ] 开关关闭后，真实 Pipeline 启动路径与现有行为不变。
- [ ] 不支持的 API 返回明确错误，不出现缺失方法导致的意外异常。
- [ ] 代码和测试中没有任何 TTS 集成、音频回复或 PCM 逻辑。
- [ ] 中文文档能指导开发者开启本地替身，并在服务器联调前关闭。
