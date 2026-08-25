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

解析器负责类型、重复、允许值和依赖图。当前依赖只有 `audio -> text`。后续新增输出时扩展注册表/依赖表和测试矩阵，不修改 Turn handler 中的组合判断。

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

- `turn.cancel` 原子标记 cancelling；
- 取消模型、action、sender、reader，清空 provisional buffer；
- 尝试 provider cancel，失败即关闭连接；
- 等待任务收敛后只发送一次 `turn.cancelled`；
- 正常完成与取消竞争时，第一个获得终态所有权者决定结果，另一方不得再发送终态；
- 断线时不尝试发送事件，只清理资源。

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
