# Session Realtime 本地开发模型替身

## 目标与价值

提供仅用于本地开发的确定性模型替身，使开发者在没有 GPU、模型权重和 Pipeline 子进程时，仍能启动真实 FastAPI 服务并通过 `/v1/session/realtime` 完成 Session/Turn 调试。

本任务是内嵌 TTS 迁移的前置条件，但不实现 TTS。它负责替代模型与动作评分能力，验证新协议输入、固定文本流、固定动作结果和终态；后续 TTS 任务再增加 `outputs` 中的 `audio` 能力。

## 已确认事实

- 唯一 Realtime 接口为协议版本 1 的 `/v1/session/realtime`；旧 `/v1/realtime` 全面废弃。
- 新接口由客户端显式发送 `session.start`、`turn.start`、`input.*`、`turn.commit`，不依赖旧 VAD 自动提交。
- 当前正式接口支持 text、action 及融合模式，并包含 provisional reply、action ready、response done 和 turn result。
- TTS 将由 `session.start.outputs` 是否包含 `audio` 控制，不增加服务级 TTS 启用开关。
- 本任务完成时尚无 TTS，因此 `audio` 输出必须返回稳定的“能力尚未实现”错误，不能静默降级为 text。

## 需求

### R1：开发态配置

集中解析以下环境变量：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | `false` | 正式 launcher 中启用替身分支 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 固定中文回复 | text 输出正文 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | 正整数 | 文本 Delta 最大字符数 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | 非负整数 | 相邻 Delta 模拟间隔 |
| `SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID` | 空 | action 模式下优先返回的白名单 candidate；为空时确定性选择首个合法 candidate |

非法配置必须在启动阶段失败。日志只记录开关、长度和节奏，不记录正文、媒体或凭证。

### R2：无 Pipeline 启动

- 提供独立入口 `python -m sglang_omni.serve.realtime.dev_server`，不要求生产模型配置。
- 正式 launcher 的 fake 分支必须发生在 action catalog 加载、runner、coordinator、GPU、warmup、profiler 和 watcher 之前。
- 开关关闭时，合并后生产分支的 action catalog、端口探测、Pipeline 和资源生命周期顺序保持不变。
- fake app 只暴露 `/health`、`/v1/models` 和 WebSocket `/v1/session/realtime`；旧 `/v1/realtime` 及 speech WebSocket 不得注册。
- fake 模式禁用生产 resource monitor，避免本地 NVML/GPU 依赖。

### R3：Session 与 Turn 合同

- 第一条消息必须是协议版本 1 的 `session.start`。
- 本任务接受 `outputs=["text"]`、`["action"]`、`["text","action"]`。
- 包含 `audio` 时返回稳定的 `unsupported_output`，直到 TTS 任务实现该能力。
- 复用正式协议对 session ID、turn ID、候选白名单、输入序号、PCM16LE、图片和事件顺序的校验；替身不得创建第二套外部协议解析器。
- 支持重复 Turn；每个 Turn 输出确定且互不污染。

### R4：固定 text 与 action 输出

- text-only 使用正式 `response.created → response.text.delta* → response.text.done → response.done → turn.result` 事件。
- 固定文本按 Unicode 字符和配置节奏流式输出；终态正文严格等于所有 Delta 拼接。
- action-only 与融合模式使用 Session 白名单中的真实 candidate ID；配置 ID 不在白名单时明确失败。
- 融合模式走正式 provisional/promotion 状态机，至少覆盖 promoted 和 unsupported/fallback 的确定性场景。
- 替身实现新 `MultimodalSession` 实际需要的窄 client 能力，不伪装支持全部生产 Client API。

### R5：真正取消

- 收到 `turn.cancel` 后停止当前模型流和动作任务，释放 Turn 资源。
- 发送一次 `turn.cancelled`；该 `turn_id` 进入幂等终态。
- 取消后不得继续发送 `response.done`、`turn.action.ready` 或 `turn.result`。
- 客户端收到 `turn.cancelled` 后才能开始下一 Turn。
- WebSocket 断开与 `session.close` 必须幂等清理全部任务。

### R6：进程外 smoke 与文档

- smoke 客户端连接 `/v1/session/realtime`，发送固定 `session.start/turn.start/input.audio.append/turn.commit`。
- 默认读取仓库真实 16 kHz、单声道、PCM16 WAV，只发送 raw frames；允许指定文本输入以减少媒体依赖。
- smoke 至少验证 session/turn ACK、文本 Delta、response done、turn result、固定正文和取消流程。
- action smoke 使用有效白名单配置并验证固定 `turn.action.ready/turn.result.action`。
- 中文文档给出可复制启动与运行命令，并声明本任务不实现音频输出。

## 范围外

- TTS WebSocket、音频回复、`response.audio.*` 和 provider 配置。
- 兼容旧 `/v1/realtime`、旧 input audio buffer 事件和旧 transcription pass。
- 模拟真实模型质量、tokenizer、采样或完整 action scoring 数值。
- 修改数字人后端。

## 验收标准

- [ ] 无 GPU、模型权重和 Pipeline 子进程可启动 `/v1/session/realtime`。
- [ ] 旧 `/v1/realtime` 不再由开发替身暴露。
- [ ] text-only、action-only、text+action 均产生符合正式协议的确定性事件和 `turn.result`。
- [ ] `audio` 在 TTS 实施前被明确拒绝，不静默忽略。
- [ ] `turn.cancel` 真正停止当前 Turn，并只发送 `turn.cancelled` 终态。
- [ ] 开关关闭后生产 action/Pipeline 启动路径无行为变化。
- [ ] 进程外 smoke 可验证固定文本、固定动作、重复 Turn 和取消。
- [ ] 没有 TTS、回复 PCM 或 `response.audio.*` 实现。
