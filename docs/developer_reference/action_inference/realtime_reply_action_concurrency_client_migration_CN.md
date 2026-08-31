# 回复、TTS 与动作并发改造：客户端检查与迁移说明

## 1. 变更目的

服务端进行了两项内部调度优化：

1. 历史需求分类与动作 Category 评分并行启动，减少动作分支等待历史分类的时间；
2. `turn.action.ready` 只等待 provisional 回复完成“提升或丢弃”的业务决议，不再等待 TTS
   完成或 TTS 取消。

协议版本、`session.start` 字段和事件结构均未变化。变化仅体现在合法事件顺序更灵活，客户端
必须分别驱动文本、音频和动作三个分支，不能互相等待。

## 2. 实际音频链路

当前链路是：

```text
Omni 流式文本
    -> 服务端内置 TTS 客户端
    -> 127.0.0.1:40001 TTS provider
    -> 服务端 provisional 音频缓冲/决议
    -> response.audio.delta
    -> 客户端直接播放
```

客户端不会收到 provisional 音频，也不需要自行调用 TTS。客户端收到的
`response.provisional.text.delta` 只是临时文本，不能拿去合成或播放。服务端只有在
provisional 被提升后，才把 TTS 已缓存或后续产生的正式音频通过 `response.audio.delta`
发送给客户端。

## 3. 客户端必须维护的状态

建议按 `turn_id` 维护以下独立状态：

| 状态 | 推荐值 | 用途 |
|---|---|---|
| `turn_terminal` | `open/result/cancelled/error` | 保证 Turn 只有一个终态 |
| `provisional_status` | `pending/promoted/discarded` | 决定临时文本是否成为正式回复 |
| `response_id` | 字符串或空 | 关联正式文本和音频 |
| `audio_state` | `idle/playing/done/stopped` | 管理服务端音频流 |
| `action_executed` | 布尔值 | 防止 `turn.result` 重复执行动作 |
| `unsupported_notice_played` | 布尔值 | 防止本地不支持提示重复播放 |

文本、音频、动作各自推进，不设置“动作 ready 后才能播放音频”或“音频开始后才能执行动作”
这样的跨分支门槛。

## 4. 事件处理规则

### 4.1 provisional 文本

- `response.provisional.created`：创建临时文本状态；
- `response.provisional.text.delta`：按 `(turn_id, provisional_id, seq)` 去重，可用于诊断或
  临时 UI，不进入正式消息、不进入历史、不调用客户端 TTS；
- `response.provisional.text.done`：只表示临时文本生成结束，不表示可以播放；
- `response.provisional.resolved`：一次性业务决议，必须作为不支持提示的唯一触发依据。

### 4.2 promoted

收到 `status="promoted"` 后：

- 接受随后的正式 `response.created`、`response.text.*` 和 `response.audio.*`；
- `replayed_from_provisional=true` 的文本是临时前缀的正式回放，展示和归档时需要去重；
- `reason="action_supported"` 表示动作受支持；
- `reason="language_required"` 表示动作可能不支持，但本轮有必须回答的语言意图，仍应播放
  服务端在线 TTS 音频；此时绝不能播放本地不支持提示。

### 4.3 discarded

收到 `status="discarded"` 后：

- 丢弃临时文本，不等待该 provisional 的在线音频；
- 只有 `reason="category_unsupported"` 或 `reason="child_unsupported"` 时，播放当前 Session
  对应的本地预录 `reply.unsupported_action_text` 音频；
- `reason="turn_cancelled"` 或 `reason="reply_failed"` 时不得播放不支持提示；
- 不得仅凭 `turn.action.ready.action.support_status="unsupported"` 播放提示，因为
  `LANGUAGE_REQUIRED` 场景会保留正常回复。

### 4.4 正式音频

- 收到 `response.audio.delta` 后立即按序送入当前 `response_id` 的播放器；
- 不等待 `turn.action.ready`；
- `response.audio.done` 表示该回复音频流结束；
- `response.done` 表示回复分支结束；
- `turn.result` 才是整个 Turn 的最终结果屏障。

### 4.5 动作

- 收到 `turn.action.ready` 后，`execute=true` 则立即提交动作播放器；
- 不等待首音频、`response.audio.done` 或 `response.done`；
- 使用 `turn_id` 幂等，`turn.result.action` 只用于核对和落库，不重复执行；
- `execute=false` 表示不启动新动作并保持当前状态；
- 若 Turn 已取消或已进入另一个终态，忽略迟到的 action/audio/text 事件。

## 5. 必须支持的事件顺序

### 5.1 动作先于首音频

```text
response.provisional.created
response.provisional.text.delta
response.provisional.resolved(promoted)
response.created
turn.action.ready
response.audio.delta
...
response.audio.done
response.done
turn.result
```

客户端应立即执行动作；首个正式音频到达后立即播放。

### 5.2 首音频先于动作

```text
response.provisional.created
response.provisional.text.delta
response.provisional.resolved(promoted)
response.created
response.text.delta(replayed_from_provisional=true)
response.audio.delta
turn.action.ready
...
turn.result
```

这是服务端在决议时已经缓存了 TTS 音频的正常情况。客户端应立即播放音频，不等待动作。

### 5.3 纯动作不支持

```text
response.provisional.created
response.provisional.resolved(discarded, child_unsupported)
turn.action.ready(support_status=unsupported)
turn.result(outputs.text=suppressed)
```

客户端在 `discarded` 时播放一次本地预录提示；不要启动在线 TTS，也不要把
`outputs.text="suppressed"` 当作失败。

### 5.4 语言意图保留、动作不支持

```text
response.provisional.created
response.provisional.resolved(promoted, language_required)
response.created
turn.action.ready(support_status=unsupported)
response.audio.delta
...
turn.result(outputs.text=completed)
```

客户端播放服务端音频，不播放本地不支持提示。

## 6. 取消和重连

- 发出 `turn.cancel` 后立即停止该 Turn 的本地音频播放并清空尚未播放的音频块；
- 等待 `turn.cancelled` 后再开始下一个 Turn；
- 对取消后迟到的 `response.*` 和 `turn.action.ready` 按 `turn_id` 丢弃；
- WebSocket 重连后创建新 `session_id`，重新发送完整 `session.start`，不能复用旧 Turn 状态；
- 不要因为 action 与 audio 的先后顺序变化主动重建 transport。

## 7. 客户端检查清单

- [ ] 客户端只播放服务端 `response.audio.delta`，不会对 provisional 文本自行调用 TTS；
- [ ] 音频播放不等待 `turn.action.ready`；
- [ ] 动作执行不等待首音频或音频 done；
- [ ] 不支持提示只由 `response.provisional.resolved=discarded` 的两个 unsupported reason 触发；
- [ ] `promoted + language_required + action unsupported` 不播放本地提示；
- [ ] `turn.action.ready` 按 `turn_id` 幂等，`turn.result` 不重复执行动作；
- [ ] provisional 文本与正式 replay 文本展示时去重；
- [ ] `response.audio.done`、`response.done`、`turn.result` 分别处理，不混为同一终态；
- [ ] 取消时停止播放、清缓存，并忽略该 Turn 的迟到事件；
- [ ] 自动化测试覆盖第 5 节四种合法顺序。

## 8. 服务端运行开关与观测

两项优化默认开启，可在需要快速回滚时设置为 `0` 并重启服务：

```text
SGLANG_OMNI_REALTIME_ACTION_READY_TTS_DECOUPLED=0
SGLANG_OMNI_REALTIME_ROUTE_ACTION_PARALLEL=0
```

服务端结构化日志的 `session_started` 会记录 `action_ready_tts_decoupled` 和
`route_action_parallel`。性能日志可对比：

- `action_category_ready`、`action_child_ready`；
- `turn_action_ready_sent.after_commit_ms`；
- TTS 首音频、`response.audio.done` 和 `turn_result_sent`；
- Category 与历史路由请求的音频编码 `cache_status` 是否为 `shared` 或 `hit`；
- 并发开启前后的 GPU queue wait、prefill 和总体 P50/P95/P99。

并发优化不保证所有 Turn 都达到固定 800 ms；它消除的是可避免的串行等待。若 GPU 排队上升
抵消收益，可只关闭 `SGLANG_OMNI_REALTIME_ROUTE_ACTION_PARALLEL`，保留 action ready 与 TTS
解耦。
