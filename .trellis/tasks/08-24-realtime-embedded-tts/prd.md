# Session Realtime 内嵌流式 TTS

## 目标与价值

在唯一的 `/v1/session/realtime` 协议版本 1 中增加 `audio` 输出能力。客户端在 `session.start.outputs` 中声明 `audio` 后，本服务继续向客户端流式返回模型正文，同时把正文 Delta 流式送入远程 TTS，并将 TTS PCM 音频通过同一条外部 WebSocket 返回。

TTS 是否启用由 Session 输出合同决定，不使用 `--realtime-tts-enabled` 或同义环境变量。服务部署配置只负责 TTS provider 地址、voice、超时和队列限制。

## 已确认决策

- 旧 `/v1/realtime`、旧 VAD 自动提交和旧事件不兼容、不保留。
- 支持组合：`["text"]`、`["text","audio"]`、`["action"]`、`["text","action"]`、`["text","audio","action"]`。
- `audio` 依赖 text；拒绝 `["audio"]` 和 `["audio","action"]`。
- 文本继续使用 `response.text.*`；新增 `response.audio.delta/done`，不使用 `response.audio_transcript.*`。
- `turn.cancel` 真正取消模型、TTS 和 action 工作，终态为一次 `turn.cancelled`。
- 设计与实现以已合并的 Session Realtime/action 分支为基线。

## 需求

### R1：可扩展输出能力模型

- 在 Session 协议的单一输出解析器中加入 `audio`，生成 `text_enabled`、`audio_enabled`、`action_enabled`。
- 校验非空、去重、已知类型和 `audio -> text` 依赖。
- `session.started.outputs` 回显规范化结果。
- 所有业务分支读取能力对象，不得散落硬编码组合。
- 未包含 audio 时不得创建 TTS Turn、连接 provider 或改变现有 text/action 事件。

### R2：部署配置

- 配置 TTS URL、voice、connect/ready/send/first-audio/turn timeout、有界文本队列和失败策略。
- 不提供 TTS enabled 开关；缺少 provider 配置不阻止服务启动。
- 只有收到包含 audio 的 `session.start` 时才校验 provider 配置；无效配置通过稳定 Session 错误拒绝该连接/Session。
- URL、凭证、完整文本和音频不得写入日志；endpoint 必须脱敏。

### R3：文字与音频事件

启用 audio 的正常顺序：

```text
response.created
response.text.delta × N
response.audio.delta × N        # 可与后续 text delta 交错
response.text.done
response.audio.delta × N        # TTS 尾音频
response.audio.done
response.done
turn.result
```

- 每个 `response.audio.delta` 包含 `session_id`、`turn_id`、`response_id`、单调 `seq`、Base64 PCM、format、sample rate 和 channels。
- PCM 只校验并转发，不转码、不重采样、不添加 WAV header。
- `response.text.done` 表示模型正文结束；此时向 TTS 恰好 commit 一次。
- `response.audio.done` 只能在最后一个有效音频块之后发送。
- `response.done` 等待 text 与 audio 分支完成；`turn.result` 仍是整个 Turn 终态。
- `turn.result.outputs.audio` 使用 `completed`、`failed`、`cancelled` 等与现有输出状态模型一致的值。

### R4：TTS 流式并发与背压

- 每个外部 Session 可复用一个内部 TTS WebSocket，但同一连接只允许一个 reader 和一个活动 TTS Turn。
- 模型 Delta 进入有界队列；发送 TTS 与接收音频并发运行，禁止先缓存完整文本。
- 首包音频允许在模型 text done 前外发。
- 任何 reader、sender、模型或外部发送失败都由同一个 Turn owner 收敛，不能产生孤儿任务。

### R5：action 融合与 provisional 音频

- text+audio+action 沿用正式 provisional reply 状态机。
- provisional 文本可以提前送入 TTS，但返回音频只在服务内有界缓冲，不向客户端播放。
- provisional promoted 后，复用同一 response/TTS Turn 并按序释放缓冲音频；标记 `replayed_from_provisional=true` 的文本不得重复送入 TTS。
- provisional discarded 时取消对应 TTS Turn并清空缓冲；不允许音频泄漏到正式 response。
- unsupported action 使用现有 `client_prerecorded_audio` 业务结果时，不调用在线 TTS，也不发送在线 audio delta。

### R6：真正取消与关闭

- `turn.cancel` 使当前 Turn token 失效，取消模型、action、TTS sender/reader 和缓冲任务。
- 尝试向内部 TTS 发送 cancel；连接状态不可信时关闭连接。
- 清空未外发 provisional 音频，发送一次 `turn.cancelled`。
- 取消后不得发送 response/audio done 或 turn result；迟到上游事件必须丢弃。
- `session.close`、WebSocket 断开和服务关闭执行幂等 teardown。

### R7：错误合同

| 场景 | 外部行为 |
|---|---|
| audio 组合非法 | `invalid_event_field` / 稳定 outputs 错误 |
| audio 请求但 provider 未配置 | Session 建立失败，稳定配置错误 |
| TTS 建连/ready/send/首包/总超时 | 当前 Turn audio failed；按失败策略决定整体 failed/partial |
| 非法 JSON、Base64、乱序、重复 done、提前关闭 | TTS protocol error，连接关闭 |
| 模型失败 | 取消 TTS，Turn failed |
| action 失败但 text/audio 完成 | 保留现有 partial 语义并返回各输出状态 |

首版失败策略采用 `fail_turn`：audio 请求中的 TTS 失败使当前 text/audio 回复分支失败，不静默降级成 text-only。

### R8：测试、观测与文档

- fake TTS server 覆盖正常、慢响应、错误、非法 Base64、乱序、提前关闭、无 done、取消和连接复用。
- 使用前置 fake model 在无 GPU 环境验证 text+audio 和 text+audio+action。
- 指标至少包含模型首文本、首个 TTS 文本、首音频、可播放 250ms、尾音频、response done 和 turn result。
- 日志携带 session_id、turn_id、response_id 和脱敏 provider，不记录正文/PCM。
- 中文文档更新 Session outputs、事件、取消、失败和本地 smoke。

## 范围外

- 修改 `/v1/audio/speech*` provider 路由或本地 TTS Pipeline。
- 音频转码、重采样、WAV 封装和播放器逻辑。
- 兼容旧 `/v1/realtime` 或双发旧文本事件。
- 修改数字人后端的动作执行或预录音频资产管理。

## 验收标准

- [ ] 不含 audio 的 Session 与合并分支当前行为和事件序列一致，且零 TTS 副作用。
- [ ] 五种允许组合通过；两种 audio-only 组合被稳定拒绝。
- [ ] text+audio 同时返回可关联文本和 PCM 音频，首音频可早于 text done。
- [ ] response.audio.done、response.done、turn.result 顺序符合 R3。
- [ ] text+audio+action 的 provisional 音频只在 promoted 后外发，discarded 不泄漏。
- [ ] `turn.cancel` 真正停止全部分支并只产生 `turn.cancelled`。
- [ ] TTS 协议错误、超时和失败没有任务/连接泄漏，也不静默降级。
- [ ] 无 GPU 本地集成、相关 pytest、格式和静态检查通过。
