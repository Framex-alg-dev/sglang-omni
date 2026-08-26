# Session Realtime 动作推理说明

本文补充 action-only 与动作评分相关说明。客户端公共事件、字段和完整示例统一参见
[Realtime 回复与动作客户端接入指南](realtime_reply_action_client_guide.md)。

## Action-only Session

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "session-action-only",
  "outputs": ["action"],
  "locale": "zh-CN",
  "action": {
    "category_guidance": "优先低打扰类别。",
    "candidate_guidance": "避免大幅位移。",
    "fallback_category_ids": ["B008"],
    "allowed_candidates": [
      {"candidate_id": "A124"},
      {"candidate_id": "A125"},
      {"candidate_id": "A071"}
    ]
  }
}
```

action-only 与融合模式使用同一个两阶段动作选择逻辑：

1. 从当前 Session 白名单实际覆盖的类别或内部 `B000`（`UNSUPPORTED`）中选择一个结果；
   `B000` 只表示明确动作请求的目标语义类别不在当前 Session 中；
2. 普通类别只评分该类别下的白名单 Child，并允许返回内部 `A000`（`UNSUPPORTED`）；
   `A000` 只表示类别已存在、但白名单具体动作均无法满足明确请求；
3. 支持时返回具体动作；不支持时进入 `fallback_category_ids[0]` 并返回真实动作。

`fallback_category_ids` 是按优先级排列的非空类别数组，白名单必须在每个兜底类别下包含
至少一个真实动作；如果 `allowed_candidates` 缺省或为空，则自动使用兜底类别下的全部动作。
`B000`、`A000` 和 `UNSUPPORTED` 都是服务端保留值，客户端不得将其作为类别或动作上传。

客户端不再传递类别结构、名称、定义或 `action_id`。这些信息由服务端全局目录提供。

## 输入

用户 Turn：

```json
{"type":"turn.start","turn_id":"turn-001","origin":"user"}
```

随后可发送：

- `input.audio.append`：用户 PCM16LE 音频；
- `input.image.append`，`image_source=user_camera`：用户和环境画面；
- `input.image.append`，`image_source=avatar_current`：数字人当前画面；
- `input.text.set`：可选用户文本；
- `turn.commit.action.last_executed_action_id`：兼容字段；服务端仍接受并校验，但不再用于动作
  Prompt 或动作选择，客户端应省略；
- `turn.commit.avatar_state`：数字人当前结构化状态。

主动 Turn 可以在 `turn.commit.action.guidance` 中提供当前场景的临时动作目标和禁止项。

Category 和 Child 只使用本轮输入、当前数字人图片与当前结构化状态，不读取上一动作 ID、
动作名称、回复历史或跨轮动作历史。动画过渡和复位由动画引擎负责。

## 图片语义

- `user_camera` 只用于理解用户情绪、用户状态和用户环境；
- `avatar_current` 用于理解数字人景别、当前姿态、外观和可交互物体；
- 两类图片不得混用；
- 仅头肩或半身构图时，Category Prompt 会避免下肢、位移或全身大幅移动类别；
- Category 不以物体是否出现在数字人当前画面中作为选择物体交互类别的前置条件；用户明确
  指定的交互物体是否与候选动作匹配，由 Child 阶段判断。

## 动作结果

```json
{
  "type": "turn.result",
  "session_id": "session-action-only",
  "turn_id": "turn-001",
  "status": "completed",
  "outputs": {"action": "completed"},
  "action": {
    "candidate_id": "A124",
    "action_id": "A124",
    "category_id": "B027",
    "execution_binding": {},
    "execute": true
  },
  "timing": {}
}
```

客户端只根据 `execute` 决定是否播放。若 Session 打开
`diagnostics.include_action_scores=true`，结果会额外带 `scores` 与 `media_summary`，但这些
字段仅用于诊断，不能由客户端重新计算或覆盖服务端最终动作。

## 全局目录和 Session 隔离

- 全局动作目录、Category 静态 Prompt 和 Child 静态 Prompt 为进程级只读共享数据；
- Session 只保存自己的动作白名单、execution binding、人设和偏好；
- 单个 Session 的非法候选、异常或取消不得修改全局目录或其他 Session 状态；
- 全局预热只提供静态前缀缓存，不扩大任何 Session 的候选白名单。
