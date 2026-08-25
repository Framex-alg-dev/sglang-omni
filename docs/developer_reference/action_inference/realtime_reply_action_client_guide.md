# Realtime 回复与动作客户端接入指南

本文面向接入 `/v1/session/realtime` WebSocket 的客户端开发者，只描述正式协议版本 1
的接入方式、事件顺序和异常处理。服务端推理实现、Prompt 拼装和缓存细节不属于客户端
契约。

当前协议尚未上线，不提供旧字段或旧事件名兼容。客户端必须严格使用本文字段。

## 1. 基本约定

- WebSocket 地址：`ws://<host>:<port>/v1/session/realtime`。
- 只发送 JSON 文本帧；音频和图片放在 JSON 中，以原始 Base64 字符串传输。
- 第一条事件必须是 `session.start`，每条连接只能建立一个 Session。
- 一个 Session 同时只能有一个活跃 Turn。
- `session_id` 在服务当前活跃 Session 中唯一，最长 128 字符。
- `turn_id` 在同一 Session 中永久唯一，最长 128 字符；完成、失败或取消后都不能复用。
- 未声明的字段会被拒绝，不会被静默忽略。
- 当前正式协议版本固定为整数 `1`。

客户端发送的事件只有：

| 事件 | 用途 |
|---|---|
| `session.start` | 建立 Session，固定输出类型、人设和动作白名单 |
| `turn.start` | 开始一个用户或主动 Turn |
| `input.text.set` | 设置或清空用户 Turn 的文本输入 |
| `input.audio.append` | 追加 PCM16LE 音频块 |
| `input.image.append` | 追加用户摄像头或数字人当前图片 |
| `turn.commit` | 冻结输入并启动推理 |
| `turn.cancel` | 取消当前 Turn |
| `session.close` | 关闭 Session |

## 2. 推荐事件流程

```text
客户端                              服务端
  │── session.start ────────────────>│
  │<─ session.started ───────────────│
  │── turn.start ───────────────────>│
  │<─ turn.started ──────────────────│
  │── input.audio.append × N ───────>│
  │── input.image.append × N ───────>│
  │── input.text.set（可选）────────>│
  │── turn.commit ──────────────────>│
  │<─ turn.committed ────────────────│
  │<─ response.provisional.* ─────────│  融合模式：可预生成 TTS，但不可播放
  │<─ response.* / turn.action.ready │  动作确认支持后；相对顺序不固定
  │<─ turn.result ───────────────────│  Turn 最终结果
  │── 下一个 turn.start，或 close ──>│
```

同一条 WebSocket 上，消息顺序有保证。客户端可以按顺序发送所有媒体事件后立即发送
`turn.commit`，无需逐个等待 `input.ack`，从而减少网络往返。客户端仍应异步处理 ACK，
用于发现输入错误和实现背压。

## 3. 建立 Session

### 3.1 完整示例

下面示例显式选择中文；如果省略 `locale`，服务端会使用 `en-US`。

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "session-20260822-001",
  "outputs": ["text", "action"],
  "locale": "zh-CN",
  "character_profile": {
    "gender_expression": "男性",
    "visual_style": "写实",
    "role": "海洋科学家",
    "personality": "沉稳克制"
  },
  "reply": {
    "instructions": "自然、简洁地回复用户；用户只要求执行动作时，可以返回空文本或一个极短的自然回应，回应后立即结束，不要补充、复述或声称动作已经完成。",
    "unsupported_action_text": "这个动作暂时做不了，我们换个互动方式吧"
  },
  "action": {
    "category_guidance": "优先自然反馈和低打扰类别。",
    "candidate_guidance": "避免夸张舞蹈、快速位移和不符合当前构图的动作。",
    "fallback_category_ids": ["B008"],
    "allowed_candidates": [
      {
        "candidate_id": "A124",
        "execution_binding": {"asset_id": "motion_wave_single"}
      },
      {
        "candidate_id": "A125",
        "execution_binding": {"asset_id": "motion_wave_double"}
      },
      {
        "candidate_id": "A071",
        "execution_binding": {"asset_id": "motion_idle_breathing"}
      }
    ]
  },
  "input_audio": {
    "format": "pcm16le",
    "sample_rate_hz": 16000,
    "channels": 1
  },
  "diagnostics": {
    "include_action_scores": false
  }
}
```

### 3.2 字段说明

| 字段 | 必填 | 说明 |
|---|---:|---|
| `type` | 是 | 固定为 `session.start` |
| `protocol_version` | 是 | 当前只接受整数 `1` |
| `session_id` | 是 | 客户端生成的唯一 Session ID |
| `outputs` | 否 | `text`、`action` 的非空去重数组；缺省为当前版本全部支持类型 |
| `locale` | 否 | `zh-CN` 或 `en-US`，缺省 `en-US`；显式传 `null` 或其他值会被拒绝 |
| `character_profile` | 否 | 动作 Category 和 Child 共享的数字人人设；不进入回复 Prompt |
| `reply` | 否 | 回复专用 Session 配置 |
| `action` | 启用 action 时是 | 动作偏好和 Session 动作白名单 |
| `input_audio` | 否 | 输入音频格式；缺省即 PCM16LE、16 kHz、单声道 |
| `diagnostics` | 否 | 诊断输出开关 |

`outputs` 当前支持 `["text"]`、`["action"]` 和 `["text", "action"]`。生产客户端
建议始终显式发送，避免未来协议新增输出类型后产生意外行为。

`locale` 控制服务端生成的 Category、Child 和动作上下文 Prompt 使用中文还是英文。
客户端传入的 `reply.instructions`、人设、临时上下文、不支持文案，以及动作目录中
的名称和说明均保持原文，服务端不会自动翻译。客户端如果要求中文交互，应始终显式发送
`"locale": "zh-CN"`。

### 3.3 人设与约束作用域

| 字段 | 作用对象 | 有效期 |
|---|---|---|
| `character_profile` | 类别、动作 | 整个 Session |
| `reply.instructions` | 回复；作为完整 System Prompt 原样使用 | 整个 Session |
| `reply.unsupported_action_text` | 不支持动作时写入历史的文本，并对应客户端预生成音频 | 整个 Session |
| `action.category_guidance` | 类别选择为主 | 整个 Session |
| `action.candidate_guidance` | 类别内具体动作选择为主 | 整个 Session |
| `action.fallback_category_ids` | 不支持或异常时使用的类别，按数组顺序确定优先级 | 整个 Session |
| `turn.commit.reply.context` | 回复 | 当前 Turn |
| `turn.commit.action.guidance` | 动作 | 当前主动 Turn |

`character_profile` 支持 `gender_expression`、`visual_style`、`role`、`personality`
四个可选字符串字段，仅供动作推理使用。`reply.instructions` 最长 32768 字符，是回复模型
唯一的 System Prompt 来源；服务端不会拼接 `character_profile`，也不会添加默认或兜底 Prompt。
客户端如需人设、语言、输出格式或缺省行为，必须完整写入 `reply.instructions`。未传入时，
回复 System Prompt 为空。动作 guidance 和人设单字段最长 2048 字符。

当 `outputs` 同时包含 `text` 和 `action` 时，`reply.unsupported_action_text` 必填，必须是
1～2048 字符的非空字符串；其他输出组合不得传入。客户端应提前为当前角色生成与该文本
完全对应的音频并缓存。服务端只接收文本，不接收、存储或返回这段音频。

### 3.4 动作白名单

客户端可以通过 `allowed_candidates` 发送当前 Session 允许的 `candidate_id`，不再发送类别
名称、类别路径、动作名称、动作定义或 `action_id`。服务端从全局动作目录恢复这些权威信息。

每个元素只允许：

```json
{
  "candidate_id": "A124",
  "execution_binding": {"asset_id": "motion_wave_single"}
}
```

- `candidate_id` 必须存在于服务端全局目录，最多 512 个且不能重复；
- `execution_binding` 可选，必须是字符串到字符串的字典，动作结果会原样返回；
- `fallback_category_ids` 必填，必须是非空、无重复的类别 ID 数组，数组顺序即优先级；
- `allowed_candidates` 缺省或为空数组时，服务端自动使用所有兜底类别下的全部全局动作；
- 自动展开的动作 `execution_binding` 为 `{}`；客户端需要执行资源映射时应显式传入白名单；
- 显式传入非空白名单时，每个兜底类别下必须至少包含一个可执行动作；
- 服务端只会评分和返回白名单内候选，不会越界选择全局目录中的其他动作。

`UNSUPPORTED` 是服务端在 Category/Child 推理中使用的非执行型判断值；内部评分 ID 分别为
`B000` 和 `A000`。三者均为服务端保留值，客户端不得把它们加入类别目录或
`allowed_candidates`。当用户请求不受支持时，服务端进入
`fallback_category_ids[0]`，并从该类别的 Session 白名单中选择真实可执行动作。

可以配置多个兜底类别，例如 `["B008", "B009"]`。Category 正常判断仍可根据语义选择其中
任一类别；只有 `UNSUPPORTED`、Child 推理失败等兜底路径按数组优先级使用第一个类别。
因此第一个类别应当在任何场景下都安全、可执行，且其中第一个白名单动作应适合作为推理
异常时的确定性默认动作。

若客户端动作配置依赖某一目录版本，应核对 `session.started` 返回的
`global_action_catalog_version` 和 `global_action_catalog_hash`。

### 3.5 `session.started`

```json
{
  "type": "session.started",
  "protocol_version": 1,
  "session_id": "session-20260822-001",
  "model": "Qwen3-Omni",
  "outputs": ["text", "action"],
  "locale": "zh-CN",
  "action_candidate_count": 3,
  "action_category_count": 2,
  "session_action_catalog_hash": "sha256:...",
  "global_action_catalog_hash": "sha256:...",
  "global_action_catalog_version": "...",
  "fallback_category_ids": ["B008"],
  "action_prefix_prefilled": true,
  "unsupported_action_text_configured": true,
  "unsupported_action_text_sha256": "sha256:..."
}
```

收到该事件后才能开始 Turn。客户端至少校验版本、Session ID、outputs、候选数量，以及
业务依赖的全局动作目录版本/hash。融合模式还应确认
`unsupported_action_text_configured=true`，并可通过 SHA-256 校验服务端接收的文本与本地
预生成音频文案一致。

## 4. 开始 Turn

用户触发的被动 Turn：

```json
{"type":"turn.start","turn_id":"turn-user-001","origin":"user"}
```

被动场景可以只发送音频，不要求客户端提供 ASR 文本。

数字人主动 Turn：

```json
{
  "type": "turn.start",
  "turn_id": "turn-proactive-001",
  "origin": "proactive",
  "trigger_type": "user_returned"
}
```

- `origin` 只能是 `user` 或 `proactive`；
- `trigger_type` 只允许用于 proactive，最长 256 字符；
- `trigger_type` 是业务触发类型，详细临时约束放在 commit。

服务确认：

```json
{"type":"turn.started","session_id":"session-20260822-001","turn_id":"turn-user-001"}
```

## 5. 上传输入

### 5.1 文本输入

只用于 `origin=user`：

```json
{
  "type": "input.text.set",
  "turn_id": "turn-user-001",
  "text": "你可以给我打个招呼吗？"
}
```

该事件是覆盖设置，不是增量追加。再次发送会替换之前文本，发送 `null` 会清空文本。
文本最长 65536 字符。主动 Turn 的固定回复通过 `turn.commit.reply.provided_text` 发送。

服务确认事件为 `input.text.ack`，其中 `text_present` 表示当前文本是否非空。

### 5.2 音频输入

```json
{
  "type": "input.audio.append",
  "turn_id": "turn-user-001",
  "seq": 1,
  "data": "<raw-base64>"
}
```

- `data` 是裸 PCM16LE 字节的 Base64，不包含 WAV 头和 `data:` URI 头；
- `seq` 必须从 1 开始严格递增；
- 整个 Turn 最多约 60 秒、最多 4096 个音频块；
- 使用相同 `seq` 重传时，内容必须与已接受块完全一致；
- 相同 `seq`、不同内容会返回冲突错误。

### 5.3 图片输入

```json
{
  "type": "input.image.append",
  "turn_id": "turn-user-001",
  "seq": 1,
  "capture_timestamp_ms": 0,
  "media_type": "image/jpeg",
  "image_source": "user_camera",
  "data": "<raw-base64>"
}
```

| 字段 | 说明 |
|---|---|
| `seq` | 当前 Turn 图片唯一正整数序号 |
| `capture_timestamp_ms` | 当前 Turn 内的采集时间偏移，缺省 0 |
| `media_type` | `image/jpeg`、`image/png` 或 `image/webp` |
| `image_source` | `user_camera` 或 `avatar_current` |
| `data` | 不含 data URI 头的原始 Base64 |

图片最大 8 MiB，每个 Turn 最多 64 张。相同 `seq` 重传时，图片内容、时间戳、媒体类型
和 `image_source` 都必须一致。

省略 `image_source` 时，user Turn 默认 `user_camera`，proactive Turn 默认
`avatar_current`。生产客户端仍建议显式传入。

两类图片在服务端严格隔离：

- 回复只使用当前 Turn 时间最新的一张 `user_camera`；图片作为视觉依据排列在用户语音/文本
  之前，且不会写入后续回复历史；
- `user_camera` 仍可进入动作推理；
- `avatar_current` 只进入动作推理，每个 Turn 最多使用时间最新的一张，不进入回复或动作历史；
- Category 与 Child 使用同一张当前数字人图片，并复用相同的图片编码结果；
- 当前 Turn 没有 `user_camera` 时，回复模型会收到“未提供用户摄像头画面”的事实，不能声称
  看见用户或根据用户外观判断状态；
- 如果业务需要回答数字人当前姿势或状态，客户端应通过 `turn.commit.reply.context` 提供
  可信文本描述，不应依赖 `avatar_current` 图片进入回复模型。

### 5.4 媒体 ACK

```json
{
  "type": "input.ack",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-001",
  "media_type": "audio",
  "seq": 1,
  "duplicate": false
}
```

图片 ACK 还包含归一化后的 `image_source`。`duplicate=true` 表示这是内容一致的重传，
服务端没有重复写入。

## 6. 提交 Turn

### 6.1 被动 Turn

```json
{
  "type": "turn.commit",
  "turn_id": "turn-user-001",
  "reply": {"context": "当前在公开展厅，回复保持简短。"},
  "action": {"last_executed_action_id": "A015"},
  "avatar_state": {"pose": "standing", "gaze": "camera"}
}
```

### 6.2 主动 Turn：服务端生成回复

```json
{
  "type": "turn.commit",
  "turn_id": "turn-proactive-001",
  "reply": {"context": "用户刚刚回到镜头前，自然欢迎用户。"},
  "action": {
    "last_executed_action_id": "A015",
    "guidance": "选择轻量欢迎动作，避免重复上一动作。"
  },
  "avatar_state": {"pose": "standing"}
}
```

### 6.3 主动 Turn：客户端直接提供回复

```json
{
  "type": "turn.commit",
  "turn_id": "turn-proactive-002",
  "reply": {"provided_text": "欢迎回来。"},
  "action": {
    "last_executed_action_id": "A015",
    "guidance": "选择自然欢迎动作。"
  }
}
```

服务端不会再次生成或改写 `provided_text`，回复事件的 `source` 为 `provided`。动作可以
结合这段文本进行选择。在融合模式中，该文本仍先作为临时回复发送；若动作判断为不支持，
它会被抑制，不会成为正式回复。

如果主动 Turn 只应执行动作而不说话，必须显式发送空字符串：

```json
{"reply":{"provided_text":""}}
```

空字符串表示“客户端明确提供空回复”；缺省或 `null` 表示“需要服务端生成回复”。
`reply.provided_text` 与 `reply.context` 互斥。

### 6.4 commit 字段

| 字段 | 作用 |
|---|---|
| `reply.context` | 当前 Turn 的临时回复背景或约束，最长 8192 字符 |
| `reply.provided_text` | proactive Turn 的最终回复；空字符串表示明确静默 |
| `action.last_executed_action_id` | 客户端确认的数字人当前实际动作 `action_id`，用于姿态衔接，必须存在于全局目录 |
| `action.guidance` | 当前 proactive Turn 的临时动作指引，最长 8192 字符 |
| `avatar_state` | 当前数字人的结构化状态，序列化后最多 16384 字符 |

`avatar_state` 不允许再包含 `current_action_id` 或 `state_description`。它们已经分别替换为
`action.last_executed_action_id` 和 `action.guidance`。

主动 Turn 提供 `action.guidance` 时，服务端会将其标准化为本轮内部
`state_description`，并在 Category 和 Child Prompt 中作为最高优先级动作约束。其明确
给出的动作目标、要求和禁止项高于会话级人设与动作偏好、主动触发原因、待播文本、历史
动作、默认动作类别和其他通用选择规则。默认动作类别仅在不冲突且没有更具体动作目标时
使用。该字段不会扩展本次会话允许的类别或动作集合；客户端应确保允许的候选中至少存在
能够满足约束的动作，避免构造无可行候选的互斥条件。服务端会对“禁止、不得、避免、
must not、do not、avoid”等明确禁止语句执行确定性过滤：与禁止语义直接匹配的类别或
具体动作不会进入本轮 PPL 候选集合；其他偏好和自然语言目标仍由模型判断。

服务端以 `turn.committed` 确认输入冻结，其中包含实际接收的音频块数和图片数。

## 7. 回复流

### 7.1 仅文本模式

`outputs=["text"]` 时依次收到：`response.created`、零条或多条
`response.text.delta`、`response.text.done`、`response.done`。

```json
{
  "type": "response.text.done",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-001",
  "response_id": "resp-abcd",
  "text": "你好呀，今天过得怎么样？"
}
```

客户端可用 delta 实时展示，但最终以 `response.text.done.text` 或
`turn.result.reply.text` 为准。空回复也会正常发送 done，其中 `text` 为 `""`。

用户只提出动作指令时，回复是否为空、是否允许简短回应，以及回应长度和动作中性表达，
全部由客户端传入的 `reply.instructions` 决定。客户端必须支持空文本，不能把它当作回复
失败；回复不用于确认动作已经执行，动作仍以 `turn.action.ready` 为准。服务端不会追加
默认回复规则，也不会对模型输出做字符截断。

为避免“给我点个赞”被反向理解为用户已经给数字人点赞，客户端应在纯动作规则后加入：

```text
正确理解动作请求中的角色关系。用户说“给我、帮我、向我、对我做某个动作”时，是要求数字人对用户执行该动作，不表示用户已经对数字人做过该动作。不得把动作请求反写成对用户的感谢、确认或完成反馈。

以下示例只用于说明角色关系和回复边界，不是关键词匹配规则：
用户：“给我点个赞”
正确回复：“好呀”“可以”或空文本。
错误回复：“谢谢你的点赞，我收到了”“你已经给我点赞了”。
```

`response.done` 只表示回复分支完成，不是融合 Turn 终态；仍需等待 `turn.result`、顶层
`error` 或 `turn.cancelled`。

### 7.2 文本与动作融合模式

`outputs=["text","action"]` 时，回复和 Category 同时启动。最先收到的是临时回复流：

```json
{"type":"response.provisional.created","turn_id":"turn-user-001","provisional_id":"resp-abcd","response":{"id":"resp-abcd","status":"in_progress","source":"generated","provisional":true}}
```

随后是零条或多条临时文本：

```json
{"type":"response.provisional.text.delta","turn_id":"turn-user-001","provisional_id":"resp-abcd","response_id":"resp-abcd","seq":1,"delta":"你好呀"}
```

回复较短或动作较慢时，还可能先收到 `response.provisional.text.done`。客户端可以立即将
临时 delta 交给 TTS 生成，但只能缓存在当前 Turn 的临时音频区，不能播放、上屏为正式
消息或写入对话历史。

服务端随后只会发送一次 `response.provisional.resolved`：

- `status="promoted"`、`reason="action_supported"`：动作受支持，同一个
  `provisional_id` 被提升为正式回复。服务端发送标准 `response.created`，并把提升前已发的
  文本合并为一条带 `replayed_from_provisional=true` 的 `response.text.delta`，之后继续发送
  新的标准 delta 和 done。已消费临时流的新客户端不得再次对 replay delta 做 TTS；它只
  用于标准流兼容、正式展示和文本归档。
- `status="discarded"`、`reason="category_unsupported"` 或
  `reason="child_unsupported"`：立即丢弃临时文本和对应 TTS 缓冲，不会再收到该 Turn 的
  标准回复流；播放当前角色预生成的 `reply.unsupported_action_text` 音频。
- `status="discarded"`、`reason="turn_cancelled"`：丢弃缓冲且不要播放“不支持动作”音频。
- `status="discarded"`、`reason="reply_failed"`：丢弃缓冲且不要播放“不支持动作”音频，
  等待该 Turn 的顶层 `error` 并按失败流程处理。

推荐以 `(turn_id, provisional_id)` 建立临时 TTS 状态，并用 `seq` 去重。即使已经收到完整
临时文本，也必须等到 `promoted` 才能播放，因为 Child 仍可能返回 `A000`。

## 8. 动作结果与执行

动作可用时会提前发送：

```json
{
  "type": "turn.action.ready",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-001",
  "action": {
    "candidate_id": "A124",
    "action_id": "A124",
    "category_id": "B027",
    "execution_binding": {"asset_id": "motion_wave_single"},
    "execute": true,
    "support_status": "supported",
    "fallback_applied": false
  }
}
```

客户端只在 `execute=true` 时执行；按 `turn_id` 幂等，不能因为 `turn.result` 再次出现
同一 action 而重复执行。`execute=false` 表示保持当前状态。`turn.action.ready` 和回复
delta 的相对顺序不固定。

当前协议没有动作播放结果 ACK，服务端暂时把成功推理的动作视为后续 Turn 已执行动作。

## 9. Turn 终态

text-only、action-only 和融合模式统一返回 `status` 与 `outputs`：

```json
{
  "type": "turn.result",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-001",
  "status": "completed",
  "outputs": {"text": "completed", "action": "completed"},
  "reply": {
    "text": "你好呀，今天过得怎么样？",
    "source": "generated"
  },
  "action": {
    "candidate_id": "A124",
    "action_id": "A124",
    "category_id": "B027",
    "execution_binding": {"asset_id": "motion_wave_single"},
    "execute": true
  },
  "timing": {}
}
```

动作部分失败时，服务端保留回复并返回：

```json
{
  "type": "turn.result",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-002",
  "status": "partial",
  "outputs": {"text": "completed", "action": "failed"},
  "reply": {"text": "你好呀。", "source": "generated"},
  "action": {
    "candidate_id": "A071",
    "action_id": "A071",
    "category_id": "B008",
    "execute": true,
    "support_status": "unknown",
    "fallback_applied": true
  },
  "errors": {"action": {"message": "child scoring timeout"}},
  "timing": {}
}
```

客户端应正常保留回复、执行返回的兜底动作，并记录 `errors.action.message`。`timing`、`scores`、
`media_summary` 都是可扩展诊断字段，不能用于决定状态机。

动作识别为不支持时，`status` 仍为 `completed`，但动作包含
`support_status="unsupported"`、`fallback_applied=true`，实际 `category_id` 为本 Session
配置的最高优先级兜底类别。示例中的 B008 只是客户端配置结果，不是服务端写死值。

融合模式的不支持结果如下。此时没有正式文本，也没有 `reply.text`：

```json
{
  "type": "turn.result",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-003",
  "status": "completed",
  "outputs": {"text": "suppressed", "action": "completed"},
  "reply": {
    "source": "client_prerecorded_audio",
    "reason": "unsupported_action",
    "recorded_in_history": true
  },
  "action": {
    "candidate_id": "A071",
    "action_id": "A071",
    "category_id": "B008",
    "execute": true,
    "support_status": "unsupported",
    "fallback_applied": true
  }
}
```

客户端播放预生成音频；服务端把 `reply.unsupported_action_text` 作为该轮实际播放文案写入
带类型的审计历史，但不会作为普通 assistant 回复或动作 Prompt 文本提供给后续模型，避免
支持动作的后续回复错误复述“不支持”文案。终态不回传文本正文，客户端必须保留 Session
建立时的本地文案与音频映射。

## 10. 错误处理

```json
{
  "type": "error",
  "session_id": "session-20260822-001",
  "turn_id": "turn-user-003",
  "error": {
    "type": "invalid_request",
    "code": "invalid_event_field",
    "message": "turn.start contains unsupported fields: text_role"
  }
}
```

| code | 客户端处理 |
|---|---|
| `unsupported_protocol_version` | 协议版本不兼容，停止接入并升级客户端 |
| `invalid_event_field` | 字段缺失、拼错或仍在使用旧字段 |
| `invalid_event` | 事件名或数据结构错误 |
| `session_candidate_invalid` | Session 配置、动作白名单或目录 ID 无效 |
| `duplicate_session_id` | 更换 Session ID 或清理旧连接 |
| `duplicate_active_turn` | 等待当前 Turn 终态或先取消 |
| `invalid_turn_id` | 使用当前活跃且未使用过的 Turn ID |
| `invalid_sequence` | 媒体序号跳跃或重复内容冲突 |
| `invalid_audio` | 检查 PCM16LE、Base64、长度和 seq |
| `invalid_image` | 检查 Base64、媒体类型、大小和 seq |
| `turn_already_committed` | 不要再修改已 commit 的 Turn |
| `turn_processing_failed` | 当前 Turn 失败，使用新 turn_id 重试 |
| `server_error` | 记录 message，退避重试或重建连接 |

客户端必须用 `error.code` 做程序分支，不要解析 `message` 实现状态机。

## 11. 取消、关闭与重连

```json
{"type":"turn.cancel","turn_id":"turn-user-004"}
```

等待 `turn.cancelled` 后再开始新 Turn。正常结果和取消可能发生竞争，客户端应为每个 Turn
维护幂等终态，迟到事件不能重复落库或执行动作。

```json
{"type":"session.close","reason":"client_shutdown"}
```

断线后服务端不保证保留 Session。重连必须建立新 WebSocket、使用新 `session_id` 发送
完整 `session.start`，并为后续 Turn 使用新的 `turn_id`。

## 12. 客户端改造清单

- 增加必填 `protocol_version: 1`；
- `modalities` 改为 `outputs`；
- `language` 改为 `locale`，使用 `zh-CN`/`en-US`；缺省现为 `en-US`，中文客户端必须显式传
  `zh-CN`；
- 删除请求中的 `model`、`selection_mode`、顶层 `include_scores`；
- `instructions` 移到 `reply.instructions`，并由客户端提供完整回复 System Prompt；动作人设
  移到 `character_profile`，服务端不会将其注入回复；
- 融合模式在 `reply.unsupported_action_text` 传入角色“不支持动作”文案，并预生成完全对应
  的角色音频；核对 `session.started` 的 configured/hash；
- 动作偏好移到 `action.category_guidance`、`action.candidate_guidance`；
- `action_candidates` 改成 `action.allowed_candidates`，只传 ID 和 binding；
- 删除 `prewarm_child_category_ids`；
- `turn_origin` 改为 `origin`，删除 `text_role`；
- `trigger` 改为 `trigger_type`；
- `turn.text.update` 改为 `input.text.set`；
- `input_audio.append` 改为 `input.audio.append`，`audio` 改为 `data`；
- 删除 `input_audio_buffer.append`；
- `input_image.append` 改为 `input.image.append`；
- `timestamp_ms`、`mime_type`、`image_role`、`image` 分别改为
  `capture_timestamp_ms`、`media_type`、`image_source`、`data`；
- 图片来源枚举 `avatar_state` 改为 `avatar_current`；
- 删除 commit 中重复的 origin、role、trigger 和 `user_input`；
- `reply_context` 改为 `reply.context`；主动固定回复改为 `reply.provided_text`；
- 服务端不会自动把上一动作事实加入回复；若客户端确认用户正在询问历史动作，应在本轮
  `reply.context` 中显式传入需要回答的动作事实；普通 Turn 不要传入该信息；
- `avatar_state.current_action_id` 改为 `action.last_executed_action_id`；
- `avatar_state.state_description` 改为 `action.guidance`；
- 解析 `session.started.outputs` 和 `turn.result.outputs`；
- 支持 `response.provisional.*` 状态机：临时 delta 仅生成并缓存 TTS，`promoted` 后播放，
  `discarded` 后清空；按 `provisional_id` 和 `seq` 去重；
- 提升时忽略 `replayed_from_provisional=true` 的重复 TTS 输入，但仍用于正式文本展示/归档；
- Category/Child 不支持时播放当前角色的预生成音频，不等待或调用在线 TTS；
- 将 `outputs.text="suppressed"` 视为正常业务结果，不当作文本分支失败；
- 媒体 ACK 改为异步处理，不必逐包等待后再 commit；
- 对 `turn.action.ready` 和所有 Turn 终态实施 `turn_id` 幂等。
