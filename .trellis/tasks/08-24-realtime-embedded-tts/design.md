# Session Realtime 内嵌流式 TTS：技术设计

## 1. 总体架构

```text
数字人后端
  <-> /v1/session/realtime (Session/Turn owner)
        -> 模型 Client：text stream
        -> action scoring：action result
        -> EmbeddedTTSTurn：text queue <-> 远程 TTS WebSocket
                              -> PCM audio events
```

`MultimodalSession` 继续是外部 Session/Turn、response 和终态的唯一所有者。TTS adapter 不生成外层 ID、不决定 action、history 或 turn result；它只接受当前 Turn 的已确认/临时文本并返回经过校验的音频块。

## 2. 输出能力模型

新增集中类型 `SessionOutputCapabilities`（具体文件依现有协议类型位置确定）：

```python
@dataclass(frozen=True)
class SessionOutputCapabilities:
    outputs: tuple[str, ...]
    text_enabled: bool
    audio_enabled: bool
    action_enabled: bool
```

`session.start.outputs` 是能力的唯一事实来源。解析器负责类型、重复、允许值和组合约束：
当 outputs 包含 `audio` 时必须同时包含 `text`，因为数据链路是模型生成 text delta，
本服务把 text delta 发送给 TTS，TTS 再返回 audio delta。这不是“audio 转 text”。后续
新增输出时扩展注册表/约束表和测试矩阵，不修改 Turn handler 中的组合判断，也不增加
环境变量、CLI 参数或 provider 配置作为第二套启用入口。

## 3. 配置分层

- Session 配置：`outputs` 决定是否需要 audio。
- 服务配置：provider URL、voice、凭证、超时、queue/buffer 上限和失败策略。
- provider 配置允许服务启动时缺省；首个 audio Session 建立时验证并返回稳定错误。
- 不存在 TTS enabled 配置。

## 4. TTS adapter

建议模块：

```text
sglang_omni/serve/realtime/embedded_tts.py
```

主要对象：

- `EmbeddedTTSConfig`：部署配置与脱敏摘要；
- `EmbeddedTTSConnection`：单一内部 WebSocket reader、ready 和重连边界；
- `EmbeddedTTSTurn`：当前 turn 的有界文本队列、音频接收、commit、cancel 和 provisional buffer；
- provider event decoder：从 unknown JSON 严格校验事件和 Base64。

连接所有权与生命周期：

```text
MultimodalSession 创建
  -> 创建 EmbeddedTTSConnection manager（尚未联网）
  -> 首个 outputs 含 audio 的 Turn 借用连接
       -> Disconnected: connect + 等待 session.created
       -> Ready 且 voice 相同: 复用现有 WebSocket
       -> Ready 但 voice 不同: close old -> connect new
  -> Turn 正常完成: Streaming/Draining -> Ready，保留连接
  -> Turn cancel / timeout / protocol error: best-effort response.cancel -> Broken -> close
  -> 下一个 audio Turn: 从 Disconnected 延迟重建
  -> session.close / 外部断线 / 服务 teardown: manager.close()，永久 Closed
```

连接不能由 `EmbeddedTTSTurn` 创建并拥有，否则每个 Turn 都会重连；也不能放到 app
全局，否则不同外部 Session 会互相阻塞、串 voice、串 response 或在取消时误伤其他
Session。连接 manager 由 `MultimodalSession` 独占，Turn 只借用其当前连接。

参考实现 `ref/tts_migration_reference/realtime_ws_tts.py` 使用 connection lock 在整个
synthesis 期间串行化请求，并以 `_borrow_connection()` 按 voice 复用。迁移时保留该
语义，但把外层 `session_id` 与 provider session/response ID 分开记录，不得用 turn ID
冒充外部 Session ID。

参考代码不是可直接复制的生产模块，迁移时必须修正以下边界：

- 本项目支持 Python 3.10，不能直接依赖 Python 3.11 才提供的 `asyncio.timeout`；使用
  项目兼容的 `asyncio.wait_for`/timeout helper，并保持取消传播；
- `StreamingSpeech` 中的 text/audio queue 必须改为配置化有界队列，`put()` 自然形成
  背压，provisional PCM 另受最大字节数/时长限制；
- provider URL 中的 `session_id` 使用外部 `MultimodalSession.session_id`，Turn 通过
  provider response ID 和本地 turn/response 映射关联；
- 参考实现对 `response.audio.done` 宽松忽略，目标 decoder 仍需验证重复 done、
  audio-after-done 和最终 `response.done`，但不能把 audio done 当完整 Turn 终态。

状态机：

```text
Disconnected -> Connecting -> Ready -> Streaming -> Draining -> Ready
                     |           |          |            |
                     +---------> Broken <---+------------+
```

非正常 Turn 完成后连接进入 Broken 并关闭，禁止复用未知状态连接。

## 5. 普通 text+audio 数据流

1. `session.start` 解析 outputs；audio 能力触发 provider 配置校验。
2. `turn.commit` 创建 response 和 `EmbeddedTTSTurn`。
3. 每个模型正文 Delta：先形成正式 `response.text.delta`，再放入有界 TTS queue。
4. sender 发送文本；reader 并发接收 PCM，校验后立即发 `response.audio.delta`。
5. 模型 EOF：发送 `response.text.done`，向 TTS commit 一次。
6. TTS audio done：发送 `response.audio.done`。
7. text/audio 都完成：发送 `response.done`；action（如有）完成后发送 `turn.result`。

同一 send owner 串行化外部 WebSocket 事件，保证 seq 和终态顺序。

## 6. text+audio+action provisional

现有融合流程会先产生 provisional text，再由 action 支持性决定 promoted/discarded：

- provisional text delta 同时进入 TTS sender；
- TTS 音频进入按 `(turn_id, provisional_id)` 隔离的有界缓冲，不外发；
- promoted：复用 response_id，先发正式 promotion/response 事件，再按序释放 audio buffer；后续正式 Delta 继续实时合成；
- replayed provisional text 只用于客户端正式文本重放，不再次进入 TTS；
- discarded：发送内部 cancel，清空 buffer，不发 online audio；
- unsupported action 的预录音频分支保持现有协议，online TTS 不参与。

必须限制 provisional 音频最大字节/时长。超过上限按 fail_turn 处理，不能无限占用内存。

## 7. 事件合同

`response.audio.delta`：

```json
{
  "type": "response.audio.delta",
  "session_id": "session-1",
  "turn_id": "turn-1",
  "response_id": "resp-1",
  "seq": 1,
  "delta": "<base64 pcm16le>",
  "audio": {"format":"pcm16le","sample_rate_hz":24000,"channels":1}
}
```

`response.audio.done` 使用相同关联字段和最终 seq，不重复音频。`response.done` 的 response 内容继续以文本为审计正文；音频通过独立事件传输。`turn.result.outputs.audio` 记录分支状态。

## 8. 取消、竞争与幂等

每个 Turn 有 generation token 和单一 terminal owner：

- 业务端在外部 Session WebSocket 发送 `turn.cancel`，进入新合并分支已有的正式取消
  handler/terminal owner；
- handler 原子标记 cancelling，使 generation token 失效；
- 强制取消模型、action、TTS sender、TTS reader、文本 queue 等待者和 provisional
  buffer 任务，不等待正常生成结束；
- 向 provider 尽力发送内部 `response.cancel`，随后首版无条件关闭该 TTS 连接；
- 等待任务收敛后只发送一次 `turn.cancelled`；
- cancel 开始后，所有迟到 text/audio/action 事件都按 token 丢弃，不得发送
  `response.text.done`、`response.audio.done`、`response.done` 或 `turn.result`；
- 正常完成与取消竞争时，第一个获得终态所有权者决定结果，另一方不得再发送终态；
- 断线时不尝试发送事件，只清理资源。

首版采用保守连接恢复策略：活动 TTS Turn 被取消后，即使 `response.cancel` 发送成功也
关闭 provider WebSocket。原因是无法仅凭 send 成功证明 reader 已消费完旧 Turn 的迟到
事件；复用该连接可能把旧音频串入下一 Turn。正常完成且收到唯一 `response.done` 才将
连接重新标记为 Ready。

## 9. 错误与失败策略

首版 `fail_turn`。TTS 错误不能静默返回 text-only，因为客户端显式请求 audio。text/action 已部分完成时，外部错误与 `turn.result` 状态必须符合现有 partial/failed 模型；协议测试锁定具体矩阵。

所有超时分别配置且可观测。provider decoder 拒绝未知关键事件、非法 Base64、重复 done、audio-after-done、无 audio done 和提前关闭。

## 10. 兼容与回滚

- 不请求 audio：代码不建立 TTS 连接，现有 text/action 路径零事件变化。
- 请求 audio：若 provider 不可用，明确失败该 Session/Turn。
- 运维回滚无需服务级开关：客户端从 outputs 移除 audio；服务代码仍保留但零调用。
- 若必须紧急禁用 provider，应使用独立运维熔断配置，只拒绝新 audio Session，不把它设计成改变协议的第二个启用开关。

## 11. 观测

以 session/turn/response 为关联键记录：provider connect/ready、first text queued、commit、first audio、250ms audio、last audio、response done、turn result/cancelled。日志和 metrics 不包含正文、Base64、PCM 或凭证。

## 12. 已审批的时间戳观测设计

详细事件表以 `research/realtime_timestamp_observability_plan.md` 为准。实现遵守以下合同：

- 复用 `emit_structured_log()` 和现有按小时、类型、组件、PID 分片的 JSONL 格式；
- 在业务边界调用线程立即捕获绝对时间和单进程 monotonic 时间，writer 排队/落盘时间不得冒充
  事件发生时间；
- 默认只记录关键边界、首点、关键阈值和完成汇总，不逐文本 Delta、逐 PCM 块持久化；
- 逐 Delta/逐 PCM 只允许在显式诊断模式短时开启，且只记序号、长度、队列深度等元数据；
- writer 使用有界非阻塞队列，并以最多 100 条或最多等待 100ms 的批次聚合写入；
- 日志队列满时允许丢日志但不得反压 Realtime 链路，必须暴露 dropped/high-watermark/write-error；
- 默认日志相对关闭日志的 Realtime p99 增幅目标不超过 2%；
- 本次只实现本服务时间线，不修改数字人后端、WebSocket 握手协议或客户端连接追踪字段；
- GPU/权重/Pipeline ready 是服务级事件，Session 只记录本 Session 校验、Action/TTS manager 等
  资源 ready，禁止把 Session 初始化描述为每 Session 分配 GPU；
- 禁止记录正文、Base64、PCM、图像、完整 URL/query、voice、headers、Cookie 或凭据。
