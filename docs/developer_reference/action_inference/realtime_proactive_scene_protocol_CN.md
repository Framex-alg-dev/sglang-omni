# Realtime 主动场景协议与服务端策略

## 目标

客户端只负责报告它能够可靠观测的场景，以及本轮动作的范围或限制。服务端负责场景语义、历史与记忆策略、回复生成、动作选择、去重、安全边界和失败降级。

客户端不需要维护“本轮选择的话题”“事实依据”“交流目标”等派生信息，也不应把一整套推理规则重复写入 Prompt。

## 客户端事件

已知主动场景使用 `turn.start.origin=proactive` 和稳定的 `trigger_type`：

- `session_enter`：用户首次进入当前会话。
- `idle_timeout`：满足客户端定义的空闲触发条件。
- `user_returned`：客户端确认用户此前离开互动画面，现在重新出现。
- `character_proactive`：角色自然发起一次交流。
- `session_ending`：当前会话即将结束。
- `action_finished`：上一动作已结束，仅选择后续视觉动作，不生成回复。

示例：

```json
{
  "type": "turn.start",
  "turn_id": "turn-proactive-1",
  "origin": "proactive",
  "trigger_type": "user_returned"
}
```

`turn.commit` 可按需携带：

```json
{
  "type": "turn.commit",
  "turn_id": "turn-proactive-1",
  "scene": {
    "context": "客户端已确认用户重新出现在当前互动画面中。",
    "reply_guidance": "语速轻快，只说一句。"
  },
  "action": {
    "guidance": "保持当前基础姿态，选择友好且清晰的欢迎动作。",
    "allowed_candidate_ids": ["A101", "A103"],
    "excluded_candidate_ids": ["A102"],
    "last_executed_action_id": "wave_once"
  },
  "avatar_state": {
    "pose": "<客户端实时确认的姿态，例如 seated 或 standing>"
  }
}
```

字段含义：

- `scene.context`：客户端确认的本轮环境或触发事实。只写事实，不写服务端工作流，不要求模型自行检索历史。
- `scene.reply_guidance`：本轮回复的长度、语气、语速等轻量偏好，不得重复全局人设和固定回复规则。
- `action.guidance`：不能完全用 ID 表达的本轮动作软约束。
- `action.allowed_candidate_ids`：本轮动作硬白名单；省略或空数组表示使用会话级候选全集。
- `action.excluded_candidate_ids`：本轮动作硬黑名单。
- `action.last_executed_action_id`：客户端实际完成的最近动作，用于主动动作去重；它不是强制排除项。
- `avatar_state`：客户端实时确认的当前姿态等结构化状态；例如坐姿传 `{"pose":"seated"}`，站姿传 `{"pose":"standing"}`。无法可靠确认时省略相应字段，不得沿用过期姿态或默认假定为坐姿。

硬白名单和硬黑名单必须引用当前 Session 已注册的 `candidate_id`，不得相互重叠，也不得共同导致空候选集合。客户端知道明确动作范围时应使用结构化 ID；不要依靠“禁止自然呼吸”之类自然语言触发字符串硬过滤。

## 服务端内置策略

即使客户端只发送 `trigger_type`，以下策略仍然完整生效：

| 场景 | 回复历史 | 会话记忆 | 回复目标 | 默认动作方向 |
| --- | --- | --- | --- | --- |
| `session_enter` | 不传 | 不使用 | 首次欢迎，不暗示“回来”或等待 | 挥手、点头、致意等首次问候 |
| `idle_timeout` | 不传 | 不使用 | 一句低打扰提醒，不猜沉默原因 | 小幅、低打扰提醒 |
| `user_returned` | 不传 | 不使用 | 可欢迎回来，不猜离开原因 | 挥手、致意、轻柔点头等重逢动作 |
| `character_proactive` | 不传原始历史 | 使用安全快照 | 最多承接一个未闭环事项；否则从人设自然开场 | 与最终回复一致的轻量动作 |
| `session_ending` | 不传 | 不使用 | 一句告别，不开启新话题 | 挥手、点头、致意等告别动作 |
| `action_finished` | 不传 | 不使用 | 不生成文本、不走 TTS | 仅做自然视觉衔接 |

服务端固定场景规则位于 system 权限；客户端场景补充作为低权限数据加入当前请求，不能覆盖服务端规则。主动 Turn 不携带最近 assistant 原始回复，因此不会因历史示例重复“欢迎回来”或空闲提醒。

## `character_proactive` 会话记忆

服务端在用户 Turn 完成后异步维护少量 `open_threads`：

- 仅记录用户明确支持、尚未完成且以后继续仍有价值的事项，例如等待结果、未完成任务或明确留下的悬念。
- 一次性请求、普通问题、短暂情绪和 assistant 自己提出的话题不会创建未闭环事项。
- 用户提供进展时更新，明确完成时关闭，拒绝继续时驳回。
- 主动生成读取最近的一致快照，不等待后台提取，不增加当前文本首包或音频首包的串行等待。
- 每轮最多注入一个未闭环事项；没有合适事项时使用人设开启轻量话题。
- 一个事项被成功主动提及后，在用户没有提供新进展前不会再次用于主动发起，避免连续追问。
- 旧 assistant 回复、普通对话摘要和可复用语言产物不会进入主动场景快照。

现有 Session Memory 调度器继续保证每个 Session 同时最多一个提取任务；后续用户 Turn 会合并到有界待处理批次，不会为每个 Turn 无限制创建后台任务。

## 动作选择与降级

服务端按以下优先级处理动作：

1. `allowed_candidate_ids` / `excluded_candidate_ids` 硬约束。
2. 客户端确认的当前姿态等可执行条件。
3. 服务端内置场景动作目标。
4. `action.guidance` 和角色视觉行为偏好。
5. 最近实际动作去重。

自然语言 guidance 只影响模型评分，不执行脆弱的关键词硬删除。类别召回和具体动作评分均只看到满足硬约束的候选；多个类别共享同一动作时仍按 `candidate_id` 去重。

若模型选择不支持或评分失败，服务端返回的兜底动作仍必须位于本轮硬约束内。协议校验阶段若发现硬约束使候选为空，则直接拒绝该 `turn.commit`，不会静默越界选择动作。

## 客户端处理要求

- 同一主动事件使用唯一 `turn_id`；未结束时不要重复提交同一 Turn。
- 只在真正确认用户离开后重新出现时发送 `user_returned`；首次进入必须发送 `session_enter`。
- `idle_timeout` 的计时、可见性和业务触发条件由客户端决定，服务端负责生成内容，不反推客户端状态。
- 客户端应以 `turn.result` 中的最终动作和状态为准，并将实际完成的动作 ID 在下一次主动 Turn 中通过 `last_executed_action_id` 回传。
- 客户端不得假设动作事件和音频事件的固定先后顺序；按 `turn_id` 聚合，并分别处理文本、音频、动作的完成状态。
- `reply.context` 保留用于兼容旧客户端。新主动场景优先使用 `scene.context` 和 `scene.reply_guidance`，不要同时在多个字段重复同一套规则。

## 兼容性与性能

- 不修改 `session.start`。
- 新增字段均为 `turn.commit` 可选字段；不发送时由服务端内置策略补全。
- 没有新增当前 Turn 的串行模型调用。
- `character_proactive` 只读取已完成的记忆快照；记忆提取仍在回复完成后异步执行。
- 动作 ID 硬过滤发生在请求构建前，不增加模型计算，且通常会减少候选数量。
