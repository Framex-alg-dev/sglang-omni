# `/v1/session/realtime` 内嵌流式 TTS 修改计划

## 1. 背景与目标

SGLang-Omni 是数字人后端依赖的多模态推理服务。数字人后端通过 WebSocket 提交用户文本、音频和图片，本服务生成回复文本和动作。目标是在模型仍只生成文本的前提下，将回复正文实时送入远程 TTS，并在同一条 WebSocket 上同时返回文字和 PCM 音频。

本方案全面采用已合并分支提供的协议版本 1：

```text
ws://<host>:<port>/v1/session/realtime
```

旧 `/v1/realtime`、旧 VAD 自动提交、`input_audio_buffer.*`、`response.audio_transcript.*` 和 `response.cancel` 均不再兼容。

## 2. 端到端目标链路

### 2.1 请求从谁发给谁

1. 数字人后端与 SGLang-Omni 建立 `/v1/session/realtime` WebSocket。
2. 数字人后端发送 `session.start`，通过 `outputs` 声明该 Session 需要 text、audio、action 中的哪些能力。
3. SGLang-Omni 校验协议版本、Session ID、输出组合、角色配置、动作白名单和输入音频格式，返回 `session.started`。
4. 数字人后端发送 `turn.start`，随后按需发送：
   - `input.text.set`；
   - `input.audio.append`（Base64 PCM16LE）；
   - `input.image.append`（用户摄像头或数字人当前画面）。
5. 数字人后端发送 `turn.commit`。SGLang-Omni 冻结输入并返回 `turn.committed`。

### 2.2 回复从谁返回给谁

1. SGLang-Omni 将冻结后的输入交给多模态模型，模型流式返回回复文本。
2. SGLang-Omni 将文本 Delta 通过 `response.text.delta` 返回数字人后端。
3. 如果 Session outputs 包含 audio，SGLang-Omni 同时把同一正文 Delta 发送给远程 TTS。
4. 远程 TTS 把 Base64 PCM 音频 Delta 返回 SGLang-Omni。
5. SGLang-Omni 校验后通过 `response.audio.delta` 立即转发给数字人后端。
6. 模型正文结束后，SGLang-Omni 发送 `response.text.done` 并向 TTS commit。
7. TTS 尾音频结束后，SGLang-Omni 发送 `response.audio.done`。
8. text/audio 回复分支全部完成后发送 `response.done`。
9. action（如请求）和回复分支都终止后发送唯一 Turn 终态 `turn.result`。

```text
数字人后端       SGLang-Omni          多模态模型          远程 TTS
    | session.start   |                    |                 |
    |---------------->|                    |                 |
    | session.started |                    |                 |
    |<----------------|                    |                 |
    | turn/input/commit                    |                 |
    |---------------->|---- Generate ----->|                 |
    | turn.committed  |                    |                 |
    |<----------------|                    |                 |
    |                 |<--- text delta ----|                 |
    | text.delta      |---- text append -------------------->|
    |<----------------|                    |                 |
    |                 |<------------------------- audio -----|
    | audio.delta     |                    |                 |
    |<----------------|                    |                 |
    |                 |<--- text EOF ------|                 |
    | text.done       |---- text commit -------------------->|
    |<----------------|                    |                 |
    |                 |<--------------------- tail audio -----|
    | audio.done      |                    |                 |
    | response.done   |                    |                 |
    | turn.result     |                    |                 |
    |<----------------|                    |                 |
```

## 3. outputs 是唯一 TTS 启用合同

暂定支持：

| outputs | 文本 | 在线 TTS 音频 | 动作 |
|---|---:|---:|---:|
| `["text"]` | 是 | 否 | 否 |
| `["text","audio"]` | 是 | 是 | 否 |
| `["action"]` | 否 | 否 | 是 |
| `["text","action"]` | 是 | 否 | 是 |
| `["text","audio","action"]` | 是 | 是 | 是 |

拒绝 `["audio"]` 和 `["audio","action"]`，因为在线音频来源于模型正文。

实现必须用集中能力模型解析 outputs，并生成 `text_enabled/audio_enabled/action_enabled`。不得在多个 handler 中硬编码组合。后续新增 video、viseme 等输出时，只扩展输出注册表、依赖图和表驱动测试。

不增加 `--realtime-tts-enabled` 或环境变量开关。TTS URL、voice、凭证、超时和队列上限仍属于服务部署配置，但它们不能决定某个 Session 是否需要 audio。

运维回滚方式是客户端从 outputs 移除 audio。若 provider 故障需要全局熔断，可增加只拒绝新 audio Session 的运维状态，但不能静默把已请求 audio 的 Session 降级为 text-only。

## 4. 外部事件合同

### 4.1 文本

沿用现有：

- `response.created`；
- `response.text.delta`；
- `response.text.done`；
- `response.done`。

### 4.2 音频

```json
{
  "type": "response.audio.delta",
  "session_id": "session-001",
  "turn_id": "turn-001",
  "response_id": "resp-001",
  "seq": 1,
  "delta": "<base64 pcm16le>",
  "audio": {
    "format": "pcm16le",
    "sample_rate_hz": 24000,
    "channels": 1
  }
}
```

`response.audio.done` 使用相同关联字段和最终 seq，不重复正文或音频。业务后端依赖事件 `type` 区分文字和音频，使用 session_id、turn_id、response_id 关联同一轮。

`response.done` 等待 text/audio；`turn.result.outputs` 增加 audio 状态：

```json
{
  "outputs": {
    "text": "completed",
    "audio": "completed",
    "action": "completed"
  }
}
```

## 5. 内部 TTS 连接

建议新增 `sglang_omni/serve/realtime/embedded_tts.py`：

- `EmbeddedTTSConfig`：provider 配置和脱敏摘要；
- `EmbeddedTTSConnection`：连接、ready、单 reader、关闭和复用；
- `EmbeddedTTSTurn`：文本队列、sender/reader、commit、cancel、音频缓冲；
- provider decoder：严格验证 JSON、事件顺序和 Base64。

文本队列和 provisional 音频缓冲必须有界。sender 与 reader 并发执行，确保首音频可以在模型正文结束前到达。一个内部连接只能有一个活动 Turn 和一个 reader；异常完成后关闭不可信连接。

## 6. text+audio+action 融合

已合并的新接口会在 action 判断前生成 provisional reply：

1. provisional text 立即送 TTS；
2. TTS 音频暂存在服务内，不能发送给数字人后端；
3. provisional promoted 后，先完成正式 response promotion，再释放音频缓冲并继续实时转发；
4. 正式重放中 `replayed_from_provisional=true` 的文本只用于显示和归档，不再次送 TTS；
5. provisional discarded 时取消 TTS、清空缓冲，不能泄漏任何在线音频；
6. action unsupported 使用客户端预录音频时，不调用在线 TTS。

必须限制 provisional 缓冲的最大字节数或时长。溢出按当前 Turn 失败处理，不允许无限内存增长。

## 7. 真正取消

数字人后端发送：

```json
{"type":"turn.cancel","turn_id":"turn-001"}
```

SGLang-Omni 必须：

1. 原子标记当前 Turn cancelling；
2. 取消模型、action、TTS sender/reader；
3. 清空未外发 provisional 音频；
4. 尝试发送内部 TTS cancel，连接不可信则关闭；
5. 等待资源收敛；
6. 只发送一次 `turn.cancelled`。

取消后不再发送 `response.done`、`turn.action.ready` 或 `turn.result`。正常完成与取消竞争时使用单一 terminal owner；迟到事件按 generation token 丢弃。客户端收到 `turn.cancelled` 后才能开始新 Turn。

## 8. 错误策略

首版使用 `fail_turn`，不静默降级：客户端明确请求 audio 时，TTS 失败不能伪装成 text-only 成功。

| 故障 | 处理 |
|---|---|
| outputs 非法 | Session 协议错误 |
| provider 未配置 | audio Session 建立失败 |
| 建连/ready/send/首音频/总超时 | 当前 Turn 失败，关闭连接 |
| 非法 Base64、乱序、重复 done | protocol error，关闭连接 |
| 模型失败 | 取消 TTS，Turn failed |
| action 部分失败 | 沿用现有 partial，并分别报告输出状态 |
| 外部 WebSocket 断开 | 不再发事件，清理全部任务 |

## 9. 本地开发前置任务

先把模型替身迁移到 `/v1/session/realtime`：

- 无 GPU/Pipeline 启动真实 Session/Turn 状态机；
- 固定 text、固定 action、融合 provisional；
- `turn.cancel` 真正取消；
- TTS 实施前明确拒绝 audio；
- 进程外 smoke 覆盖 text/action/text+action。

完成 TTS 后，同一 smoke 扩展 text+audio 和 text+audio+action，从而无需每轮 Docker/服务器联调即可验证大部分协议与生命周期。

## 10. 分阶段实施

1. 锁定 outputs 能力模型和无 audio 基线。
2. 实现 provider 配置与 fake TTS server。
3. 实现 `EmbeddedTTSTurn` 连接、并发、背压和协议校验。
4. 接入普通 text+audio 事件与终态。
5. 接入 action provisional 缓冲、promotion/discard。
6. 接入真正取消、断线和错误竞争。
7. 完成本地进程外测试、Linux 测试、Docker 和数字人联调。
8. 更新正式客户端指南、观测和回滚说明。

## 11. 验收清单

- [ ] 旧 `/v1/realtime` 和旧事件名已从本方案与实现移除。
- [ ] 五种 outputs 允许组合与两种拒绝组合有表驱动测试。
- [ ] 不请求 audio 时没有 TTS 连接、任务、事件和行为变化。
- [ ] text/audio Delta 真正并发，关联 ID 与 seq 正确。
- [ ] text done、audio done、response done、turn result 顺序正确。
- [ ] provisional 音频 promoted 前不外发，discarded 后不泄漏。
- [ ] `turn.cancel` 真正停止所有分支并只发送 `turn.cancelled`。
- [ ] provider 错误和超时不静默降级、无资源泄漏。
- [ ] fake、本地、Linux 完整测试和数字人联调通过。
- [ ] 日志和指标不泄露正文、音频或凭证。
