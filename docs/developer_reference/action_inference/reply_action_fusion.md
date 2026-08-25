# Realtime 回复与动作融合实现说明

本文记录服务端实现语义。客户端接入以
[Realtime 回复与动作客户端接入指南](realtime_reply_action_client_guide.md) 为唯一协议依据。

## 正式协议边界

- 协议版本：`protocol_version=1`，在 `session.start` 中必填；
- 当前输出：`text`、`action`；
- WebSocket 入口严格拒绝未知字段和旧事件名；
- 对外只使用紧凑动作白名单，服务端通过全局目录恢复类别和动作语义；
- 服务端内部仍可使用 `modalities`、`turn_origin`、`text_role` 等实现字段，但它们不是
  客户端协议。

## Session 配置映射

客户端配置：

```json
{
  "outputs": ["text", "action"],
  "character_profile": {},
  "reply": {
    "instructions": "...",
    "unsupported_action_text": "这个动作暂时做不了，我们换个互动方式吧"
  },
  "action": {
    "category_guidance": "...",
    "candidate_guidance": "...",
    "fallback_category_ids": ["B008"],
    "allowed_candidates": []
  }
}
```

作用域：

| 输入 | 回复 | Category | Child |
|---|---:|---:|---:|
| `character_profile` | 否 | 是 | 是 |
| `reply.instructions` | 是 | 否 | 否 |
| `reply.unsupported_action_text` | 仅作为不支持结果的历史回复事实 | 否 | 否 |
| `action.category_guidance` | 否 | 主要 | 背景 |
| `action.candidate_guidance` | 否 | 可行性参考 | 主要 |

`reply.instructions` 是回复模型唯一的 System Prompt 来源，服务端原样使用，不追加默认规则，
也不从 `character_profile` 生成回复 Prompt。客户端负责在其中提供完整人设、输出约束和兜底
规则；未传入时回复 System Prompt 为空。`character_profile` 仅作为 Category 和 Child 的动作
人设上下文。人设和偏好保存在单个 Session 内，不修改全局静态 Prompt、全局目录或其他 Session。
融合输出要求客户端提供非空 `reply.unsupported_action_text`。服务端保存文本和审计 hash，
不保存角色音频；角色音频由客户端预生成和管理。

## 动作白名单展开

`action.allowed_candidates` 只携带 `candidate_id` 和可选 `execution_binding`。服务端执行：

1. 校验 candidate ID 存在于只读全局目录；
2. 校验 `fallback_category_ids` 非空、无重复且全部存在于全局目录；
3. `allowed_candidates` 缺省或为空时，展开所有兜底类别下的全部动作；
4. 显式白名单非空时，校验每个兜底类别下至少包含一个真实可执行动作；
5. 按全局目录顺序恢复所属类别、动作 ID、名称和定义；
6. 只将展开后的 Session 类别和动作交给 PPL 评分；
7. 将 Session 私有 `execution_binding` 原样带入结果；自动展开时该字段为空对象。

全局 Category/Child 静态前缀跨 Session 共用；Session 白名单和偏好仍是私有动态数据，
不得写入全局共享状态。

## Turn 语义

对外协议只在 `turn.start` 定义一次 `origin`。服务端内部推导：

| `origin` | 内部文本角色 |
|---|---|
| `user` | `user_input` |
| `proactive` | `character_reply` |

`turn.commit` 中的输入隔离如下：

| 字段 | 分支 | 有效期 |
|---|---|---|
| `reply.context` | 回复 | 当前 Turn |
| `reply.provided_text` | 回复，同时可作为动作语义上下文 | 当前 proactive Turn |
| `action.last_executed_action_id` | 动作衔接 | 当前 Turn |
| `action.guidance` | 动作 | 当前 proactive Turn |
| `avatar_state` | 动作状态理解 | 当前 Turn/按现有状态策略持久化 |

`reply.context` 不进入动作 Prompt，`action.guidance` 不进入回复 Prompt。

图片也按分支隔离：回复只使用当前 Turn 时间最新的一张 `user_camera`，并把视觉依据放在
当前语音/文本之前；两类图片均不写入回复历史。`avatar_current` 只进入动作推理，每个 Turn
最多使用时间最新的一张，且不写入动作历史。Category 与 Child 使用同一组当前图片及同一个
动作上下文缓存键，复用图片编码结果。没有 `user_camera` 的 Turn 会向回复模型明确声明用户
画面缺失。如需回复数字人当前姿势，客户端通过 `reply.context` 提供可信文本状态。

动作衔接状态与用户指代使用两条独立记录：

- `action.last_executed_action_id` 及服务端最近一次动作表示数字人当前实际动作状态，只用于姿态衔接；
- 服务端另行保留最近一次由用户输入触发的动作。用户说“刚刚那个动作”“再做一次”等指代时，
  优先指向该用户触发动作，后续 `proactive` 动作不会覆盖它；仅在没有用户触发动作记录时，
  才回退到当前实际动作状态。

该区分完全由服务端维护，不新增客户端协议字段。

## 主动 Turn 的回复选择

- `reply.provided_text` 缺省或为 `null`：服务端生成回复；
- `reply.provided_text` 为非空字符串：直接流式返回客户端文本，不再生成；
- `reply.provided_text` 为 `""`：明确返回空回复，不再生成；
- provided text 与 `reply.context` 互斥。

空字符串通过单独的 `reply_provided` 状态区分，不能再用 `bool(text.strip())` 判断，否则会
把“明确静默”误判为“需要生成”。

## 并行流水线

融合 Turn 的主要顺序：

```text
turn.commit
  ├─ 临时回复流（生成回复或 provided reply）
  └─ Category 评分
       ├─ B000：丢弃临时回复 → 兜底真实动作
       └─ 普通类别：启动 Child 评分
            ├─ A000：丢弃临时回复 → 兜底真实动作
            └─ 普通动作：提升同一临时回复为正式回复

动作结果与已提升的正式回复汇总为 turn.result
```

回复生成不等待 Category/Child，但客户端在动作判断完成前只能预生成并缓存 TTS，不得播放。
提升复用同一个生成请求和已产生文本，不重新调用模型。Child 失败时保留已经成功的回复，并返回本 Session 最高优先级
兜底类别中的默认真实动作；Turn
最终返回 `status=partial`、`outputs.action=failed` 和仅包含 message 的
结构化错误。

Category 使用内部 `B000`、非兜底 Child 使用内部 `A000` 评分非执行型判断
`UNSUPPORTED`。Category 返回它时直接把
动作分支路由到 `fallback_category_ids[0]`；普通 Child 返回它时使用该类别的默认动作。
兜底类别的 Child 不加入 `UNSUPPORTED`，必须选择一个真实动作。以上情况都返回真实可执行
动作，并标记 `support_status=unsupported`、`fallback_applied=true`。

Category 或 Child 返回不支持时，不发送标准回复流，`turn.result.outputs.text=suppressed`，
`turn.result.reply` 只携带 `source=client_prerecorded_audio`、原因和历史记录状态，不携带文本。
服务端将 Session 配置的 `unsupported_action_text` 作为带类型的审计历史记录保留，但不会
把它作为普通 assistant 回复或动作 Prompt 文本提供给后续模型，避免后续支持动作的回复
模仿“不支持”文案；客户端播放与该文本绑定的角色预生成音频。

## 事件与终态

- 融合临时回复：`response.provisional.created` → provisional delta/done →
  `response.provisional.resolved`；
- 动作支持时：同一 ID 提升为 `response.created` → 标准 delta/done；提升前缀通过
  `replayed_from_provisional=true` 重放，供标准协议消费者使用；
- 动作不支持时：临时回复被丢弃，不发送标准回复事件；
- text-only 回复仍为 `response.created` → delta → `response.text.done` → `response.done`；
- 动作分支：`turn.action.ready`；
- Turn 终态：`turn.result`、顶层 `error` 或 `turn.cancelled`；
- text-only、action-only 和融合模式统一在 `turn.result` 返回 `status` 与 `outputs`。

成功推理的动作暂时视为已执行，写入 Session 动作事实，供后续 Turn 处理动作衔接和诊断；
Child 失败不覆盖上一条成功动作事实。服务端不会把该事实自动注入回复 Prompt，以免普通
回复复述或误报上一动作。若客户端确认用户正在询问历史动作，应通过本轮 `reply.context`
显式提供需要回答的动作事实。

对于只有身体动作指令、没有同时提出语言问题或交流内容的用户输入，是否输出空文本或
简短回应，以及动作复述、长度和结束条件，全部由客户端在 `reply.instructions` 中定义。
服务端不追加默认回复规则或正反例。Category 仍只输出类别，不承担额外的回复策略分类。

## 输入幂等与限制

- 音频 `seq` 从 1 严格递增；
- 图片 `seq` 在 Turn 内唯一；
- 同 seq 重传必须具有相同内容；图片还必须保持时间戳、媒体类型和来源一致；
- 冲突重传直接报错，不得静默 ACK；
- 音频约 60 秒、4096 块上限；图片 8 MiB/张、64 张/Turn；
- 客户端可以顺序发送媒体与 commit，不必逐 ACK 等待。

## 可观测性

协议日志记录客户端原始事件类型和字段名，推理日志记录归一化后的内部值。Session 启动
应记录协议版本、outputs、locale、动作目录 hash、人设/指令 hash；Turn 日志继续关联
`session_id`、`turn_id`、`trace_id`。

资源采样保持 20 秒周期，并在每个已提交 Turn 推理前后额外采样；采样失败不能影响 Turn。
