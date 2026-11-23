# 41035 sglang-omni 接口字段说明｜2026-11-23

适用分支：`codex/41035-omni-integration-20261121`。

## 1. 接口地址

| 能力 | 接口 | 鉴权 |
| --- | --- | --- |
| Gateway | `WS /v1/inference-session`，默认端口 `18003` | 无，仅限可信内网 |
| Reply | `WS /v1/chat/completions/realtime`，通常端口 `18005` | 无，仅限可信内网 |
| Reply | `POST /v1/chat/completions` | 无，仅限可信内网 |
| Action Recall | `WS/POST /v1/action-recall`，当前 `10.199.0.7:18049` | `Authorization: Bearer <token>` |
| Action Decision | `WS /v1/action-decision/realtime`，当前端口 `18049` | `Authorization: Bearer <token>` |
| Event | `WS /v1/timeline`，当前 `10.199.0.15:18118` | `Authorization: Bearer <token>` |

## 2. Reply

### 2.1 WebSocket 首帧

```json
{
  "type": "request.start",
  "contract_version": 2,
  "request_id": "reply-turn-7",
  "payload": {
    "messages": [
      {"role": "system", "content": "Agent 管理的回复提示词"},
      {"role": "user", "content": "你好"}
    ],
    "stream": true,
    "modalities": ["text"]
  }
}
```

### 2.2 `payload` 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `messages` | `array<Message>` | 是 | - | OpenAI 格式消息列表 |
| `model` | `string` | 否 | Worker 默认模型 | 模型名 |
| `stream` | `boolean` | 否 | `false` | 是否返回增量 |
| `modalities` | `string[]` | 否 | `text` | 可选 `text`、`audio` |
| `temperature` | `number` | 否 | 模型默认 | 采样温度 |
| `top_p` | `number` | 否 | 模型默认 | nucleus sampling |
| `top_k` | `integer` | 否 | 模型默认 | top-k sampling |
| `min_p` | `number` | 否 | 模型默认 | min-p sampling |
| `repetition_penalty` | `number` | 否 | 模型默认 | 重复惩罚 |
| `max_tokens` | `integer` | 否 | 模型默认 | 最大输出 token |
| `max_completion_tokens` | `integer` | 否 | - | 设置时优先于 `max_tokens` |
| `stop` | `string|string[]` | 否 | - | 停止词 |
| `seed` | `integer` | 否 | - | 随机种子 |
| `response_format` | `object` | 否 | - | JSON/JSON Schema 输出约束 |
| `audio` | `object` | 否 | - | 音频输出配置，如 `voice`、`format` |
| `request_id` | `string` | 否 | 外层 ID | OpenAI 请求内 ID |
| `user` | `string` | 否 | - | 调用方用户标识 |
| `runtime_prompt_profile` | `"reply"` | 否 | - | 设置后 Worker 会再注入本地回复提示词 |
| `runtime_prompt_language` | `zh|en|zh-CN|en-US` | 否 | `zh-CN` | 本地提示词语言 |
| `log_context` | `object` | 否 | - | 仅实时 WebSocket 用于日志关联；HTTP 使用下述 Header |

`Message` 字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `role` | `string` | 是 | `system`、`user`、`assistant`、`tool` 等 |
| `content` | `any` | 否 | 文本或多模态内容 |
| `name` | `string` | 否 | 消息名称 |
| `tool_calls` | `array<object>` | 否 | 工具调用 |
| `tool_call_id` | `string` | 否 | 工具结果对应的调用 ID |

Agent 已经提供完整 system message 时，不传 `runtime_prompt_profile`。

HTTP 调用可用 `X-Request-ID`、`X-Root-Session-ID`、`X-Turn-ID`
传递日志关联 ID；响应回传最终采用的 `X-Request-ID`。

### 2.3 WebSocket 返回字段

| `type` | 字段 | 说明 |
| --- | --- | --- |
| `request.ready` | `request_id`、`contract_version=2` | 请求已建立 |
| `response.delta` | `request_id`、`response` | OpenAI 流式 chunk |
| `response.completed` | `request_id`、`response` | 成功；流式模式下 `response={}` |
| `error` | `request_id`、`code`、`detail` | 失败 |

## 3. Action Recall

接口：`WS/POST /v1/action-recall`。

### 3.1 请求字段

| 字段 | 类型 | 必填 | 约束/说明 |
| --- | --- | --- | --- |
| `request_id` | `string` | 否 | 1～128 字符；`[A-Za-z0-9._:-]`；省略时服务生成 |
| `mode` | `string` | 是 | `user_reaction`、`avatar_speech`、`action_request` |
| `channel` | `string` | 是 | `body` 或 `expression`；两通道分别调用 |
| `character_prompt` | `string` | 是 | 非空，最多 32,000 字符 |
| `allowed_candidate_ids` | `string[]` | 否 | 最多 512 个；非空、唯一、已注册；省略表示通道全目录 |
| `user_input` | `Input` | 见模式 | 用户输入/用户上下文/明确动作要求 |
| `avatar_history` | `Input` | 否 | 仅 `avatar_speech` 可用 |
| `avatar_current` | `Input` | `avatar_speech` 必填 | 数字人当前要说的内容 |
| `last_dispatched_action` | `object` | 否 | 兼容字段；V10 当前不用于推理 |
| `prompt_snapshot` | `object` | 否 | 试验字段，当前生产接入不要发送 |
| `log_context` | `object` | 否 | 仅用于日志关联；字段见第 8 节 |

`Input` 必须且只能包含 `text` 或 `audio`：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `text` | `string` | 非空，最多 4,000 字符 |
| `audio.data` | `string` | Base64 音频；解码后最大 2 MiB |
| `audio.format` | `string` | `wav` 或 `pcm_s16le`，默认 `wav`；16 kHz、单声道、PCM16 |

模式要求：

| `mode` | `user_input` | `avatar_history` | `avatar_current` |
| --- | --- | --- | --- |
| `user_reaction` | 必填，用户文本或音频 | 不允许 | 不允许 |
| `avatar_speech` | 可选，用户上一轮上下文 | 可选，数字人已说内容 | 必填，数字人当前内容 |
| `action_request` | 必填文本，格式 `名称: 定义` | 不允许 | 不允许 |

音频时长：

| 输入 | 最大时长 | WebSocket `evidence_role` |
| --- | ---: | --- |
| `user_reaction.user_input` | 60 秒 | `user_audio` |
| `avatar_speech.user_input` | 60 秒 | `user_context_audio` |
| `avatar_speech.avatar_history` | 4 秒 | `avatar_speech_context` |
| `avatar_speech.avatar_current` | 4 秒 | `avatar_speech_audio` |

Recall 不接收图片和视频。视觉手势使用第 4 节 Action Decision。

### 3.2 请求示例

```json
{
  "request_id": "turn-7-body",
  "mode": "user_reaction",
  "channel": "body",
  "character_prompt": "角色动作与表达风格",
  "allowed_candidate_ids": ["138", "189", "665"],
  "user_input": {"text": "你好呀"}
}
```

### 3.3 返回字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 原请求 ID |
| `channel` | `string` | `body` 或 `expression` |
| `candidate_id` | `string` | 召回候选 ID |
| `code` | `string` | 两 token 动作码 |
| `label` | `string` | 候选名称 |
| `outcome` | `string` | body：`execute/no_action/unsupported`；expression：`apply/keep/unsupported` |
| `occupies_channels` | `string[]` | 候选占用通道 |

## 4. Action Decision（视觉/完整多模态）

接口：`WS /v1/action-decision/realtime`。

### 4.1 `payload` 字段

| 字段 | 类型 | 必填 | 默认值/说明 |
| --- | --- | --- | --- |
| `request_id` | `string` | 是 | 本次模型请求 ID |
| `session_id` | `string` | 是 | 会话 ID |
| `turn_id` | `string` | 是 | 用户/数字人回合 ID |
| `decision_point_id` | `string` | 是 | 决策点 ID；可包含输入 revision |
| `channel` | `body|expression` | 是 | 两通道分别调用 |
| `text` | `string|null` | 否 | 当前文字证据 |
| `language` | `string` | 否 | `zh-CN` |
| `character_prompt` | `string|null` | 否 | 角色动作/表情风格 |
| `session_prompt` | `string|null` | 否 | 会话场景 |
| `reply_prefix` | `string|null` | 否 | 已生成回复前缀 |
| `runtime_context` | `object` | 否 | `{}`；运行时上下文 |
| `allowed_candidate_ids` | `string[]` | 否 | 允许候选 |
| `excluded_candidate_ids` | `string[]` | 否 | 排除候选 |
| `channel_enabled` | `boolean` | 否 | `true`；`false` 时不调用模型 |
| `prohibited` | `boolean` | 否 | `false`；`true` 时不调用模型 |
| `turn_origin` | `string` | 否 | `user`；数字人说话动作可用 `assistant` |
| `requested_action` | `object|null` | 否 | 明确目标动作 |
| `prompt_snapshot` | `object` | 否 | 试验字段，当前生产接入不要发送 |
| `log_context` | `object` | 否 | 仅用于日志关联；字段见第 8 节 |

### 4.2 媒体字段

`request.start` 后发送一组 `input.media` JSON，每条 JSON 后紧跟 binary frame：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `input.media` |
| `request_id` | `string` | 是 | 必须等于外层请求 ID |
| `media_id` | `string` | 是 | 本请求内唯一 |
| `kind` | `audio|image|video` | 是 | 媒体类型 |
| `start_ms` | `integer` | 是 | 大于等于 0 |
| `end_ms` | `integer` | 是 | 大于 `start_ms` |
| `encoding` | `string` | 是 | 如 `image/jpeg`、`pcm_s16le` |
| `checksum` | `string` | 是 | `sha256:<64位小写hex>` |
| `payload_bytes` | `integer` | 是 | 后续 binary 字节数 |
| `evidence_role` | `string` | 是 | 视觉用 `user_camera`；音频按服务允许角色 |

视觉手势使用：`kind=image`、`encoding=image/jpeg`、`evidence_role=user_camera`。

媒体发送完成后：

```json
{"type":"request.commit","request_id":"body-turn-7-r3"}
```

### 4.3 返回字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `contract_version` | `integer` | 当前为 `1` |
| `request_id/session_id/turn_id/decision_point_id` | `string` | 请求 fence |
| `channel` | `string` | `body` 或 `expression` |
| `outcome` | `string` | 执行、保持或不支持 |
| `candidate_id` | `string` | 候选 ID |
| `action_id` | `string|null` | 执行动作 ID |
| `code` | `string` | 两 token 动作码 |
| `token_ids` | `[integer,integer]` | 模型 token ID |
| `label` | `string` | 候选名称 |
| `occupies_channels` | `string[]` | 占用通道 |
| `deterministic` | `boolean` | 是否确定性返回 |
| `model_invoked` | `boolean` | 是否实际调用模型 |
| `reason_code` | `string` | 决策原因码 |
| `model_id/model_version` | `string` | 模型身份 |
| `catalog_version/mapping_version` | `string` | 目录与映射版本 |
| `candidate_count` | `integer` | 实际候选数 |
| `elapsed_ms` | `number` | 服务耗时 |
| `usage` | `object|null` | token 使用量 |

## 5. Event Timeline

接口：`WS /v1/timeline`。

### 5.1 `session.start` 字段

| 字段 | 类型 | 必填 | 默认值/说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `session.start` |
| `contract_version` | `integer` | 是 | 固定 `2` |
| `session_id` | `string` | 是 | 会话 ID |
| `identity_epoch` | `integer` | 是 | 身份代次，大于等于 0 |
| `observer_epoch` | `integer` | 否 | `0`；观察器代次 |
| `stream_epoch` | `integer` | 是 | 媒体连续流代次，大于等于 0 |
| `next_sequence` | `integer` | 否 | `1`；下一条媒体序号 |
| `audio_format` | `string` | 是 | 如 `pcm_s16le/16000/1` |
| `video_format` | `string` | 是 | 当前使用 `image/jpeg` |
| `model_id` | `string` | 是 | 当前 `event-1115-v1-fp8` |
| `result_mode` | `string` | 否 | 推荐 `raw_windows.v1`；兼容值 `observations` |
| `prompt_snapshot` | `object` | 否 | 试验字段，当前生产接入不要发送 |
| `root_session_id` | `string` | 否 | 全链路实时会话 ID；省略时兼容使用 `session_id` |
| `agent_session_id` | `string` | 否 | Agent 内部会话 ID；省略时兼容使用 `session_id` |

### 5.2 `media` 字段

每条媒体先发 JSON，再发 binary frame。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `media` |
| `session_id` | `string` | 是 | 必须匹配当前 session |
| `identity_epoch` | `integer` | 是 | 必须匹配当前 session |
| `observer_epoch` | `integer` | 否 | 默认 `0` |
| `stream_epoch` | `integer` | 是 | 必须匹配当前流 |
| `sequence` | `integer` | 是 | 音频和视频共用一条连续序列 |
| `kind` | `audio|video` | 是 | 媒体类型 |
| `start_ms` | `integer` | 是 | 大于等于 0 |
| `end_ms` | `integer` | 是 | 大于 `start_ms` |
| `encoding` | `string` | 是 | 视频 `image/jpeg`；音频 `pcm_s16le` |
| `payload_bytes` | `integer` | 是 | 最大 8 MiB；等于 binary 长度 |

接管回执：

```json
{"type":"media.ack","observer_epoch":0,"sequence":2}
```

`media.ack` 仅表示服务接管了字节，不表示模型完成推理。

### 5.3 `window.result` 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `string` | 固定 `window.result` |
| `contract_version` | `integer` | `2` |
| `status` | `string` | 当前为 `complete` |
| `session_id` | `string` | 会话 ID |
| `identity_epoch/observer_epoch/stream_epoch` | `integer` | 结果 fence |
| `window_sequence` | `integer` | 模型窗口序号 |
| `evidence_start_ms/evidence_end_ms` | `integer` | 窗口证据范围 |
| `model_id` | `string` | `event-1115-v1-fp8` |
| `model_version` | `string` | `1115_v1-checkpoint-6138-fp8` |
| `evidence_mode` | `string` | `audio_video` 或 `video_only` |
| `audio_status` | `string` | `complete/missing/gapped/behind` |
| `events` | `array<Event>` | 可以为空数组 |

`Event` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `event_type` | `string` | 事件编码，如 `E50` |
| `summary` | `string` | 模型事件摘要 |

### 5.4 流中断字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `discontinuity` |
| `session_id` | `string` | 是 | 会话 ID |
| `identity_epoch` | `integer` | 是 | 身份代次 |
| `observer_epoch` | `integer` | 否 | 默认 `0` |
| `old_stream_epoch` | `integer` | 是 | 旧流代次 |
| `new_stream_epoch` | `integer` | 是 | 必须大于旧值 |
| `reason` | `string` | 是 | 如 `rtc_reconnect` |

收到 `discontinuity.ack` 后，新流从 `sequence=1` 开始。

关闭：发送 `{"type":"session.close"}`，服务处理完已 ACK 媒体后返回 `session.closed`。

## 6. Inference Gateway

### 6.1 建立会话

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `session.open` |
| `contract_version` | `integer` | 是 | 固定 `4` |
| `session_id` | `string` | 是 | 会话 ID |
| `root_session_id` | `string` | 否 | 全链路实时会话 ID；省略时兼容使用 `session_id` |
| `agent_session_id` | `string` | 否 | Agent 内部会话 ID；省略时兼容使用 `session_id` |

返回 `session.ready`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `session_id` | `string` | 会话 ID |
| `contract_version` | `integer` | `4` |
| `availability_epoch` | `integer` | 上游可用性版本 |
| `stages` | `object<string,boolean>` | `classifier/brain/reply/body/expression/performance` 可用性 |

### 6.2 `media.put`

字段与第 4.2 节媒体字段相同，区别是 `type=media.put`，不带 `request_id`。
服务返回 `media.ack`，额外字段 `duplicate` 表示是否为完全相同的重复注册。

### 6.3 `stage.request`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `string` | 是 | 固定 `stage.request` |
| `request_id` | `string` | 是 | 会话内正在运行的请求不可重复 |
| `stage` | `string` | 是 | `classifier/brain/reply/body/expression/performance` |
| `payload` | `object` | 是 | 对应上游业务 payload |
| `media_refs` | `string[]` | 否 | 已通过 `media.put` 注册的媒体 ID |

`payload.log_context.request_id` 可传本次全链路日志 ID；它不替代外层
`request_id`，也不参与 stage 取消和去重。

返回事件：

| `type` | 关键字段 |
| --- | --- |
| `stage.accepted` | `request_id`、`stage` |
| `stage.delta` | `request_id`、`stage`、`response` |
| `stage.completed` | `request_id`、`stage`、`response` |
| `stage.error` | `request_id`、`code`、`detail`、`retryable` |
| `stage.cancelled` | `request_id` |

取消：`{"type":"stage.cancel","request_id":"..."}`。

## 7. 通用错误字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `string` | 固定 `error`，或 Gateway 的 `stage.error` |
| `request_id` | `string` | 有请求级 ID 时返回 |
| `code` | `string` | 如 `invalid_request`、`model_error`、`model_failure` |
| `detail` | `string` | 错误详情 |
| `retryable` | `boolean` | Gateway stage 错误是否建议重试 |

Action/Event WebSocket 鉴权失败关闭码为 `4404`；协议字段错误通常为 `4400`；内部模型错误通常为 `1011`。

## 8. 链路日志字段

关联 ID 均满足 `[A-Za-z0-9_.:-]{1,128}`。

| 字段 | 位置 | 说明 |
| --- | --- | --- |
| `request_id` | `log_context` | 一次全链路操作的日志 ID；不作为鉴权、计费、幂等或取消键 |
| `root_session_id` | `log_context` 或 `session.start/session.open` | 一次实时会话的全链路 ID |
| `turn_id` | `log_context`；Action 使用原 `turn_id` | 一轮用户/数字人交互 ID |
| `agent_session_id` | `log_context` 或 `session.start/session.open` | Agent 自身的会话 ID |

Action 的业务 `request_id` 仍是推理、取消和幂等键；日志 ID 放在
`log_context.request_id`，两者不会互相覆盖。旧 `session_id` 继续保留在接口中，
但日志内不写含义不明确的 `session_id` 字段。

Worker 日志分别写到：

| 服务 | 文件 |
| --- | --- |
| Reply | `/var/log/framex/omni-reply-worker/app.log` |
| Event | `/var/log/framex/omni-event-worker/app.log` |
| Action | `/var/log/framex/omni-action-worker/app.log` |

每条日志为单行 JSON，固定包含 `timestamp`、`level`、`service`、`env`、
`version`、`host`、`logger`、`event`。CLS 日志不记录用户原文、模型输出、
提示词、音视频内容、鉴权信息；Action 完整回放数据仅在显式配置独立回放目录时写入。

Reply HTTP 的 `http.request.completed` 在响应最后一个 body 成功发送后记录；
`outcome` 为 `completed/failed/cancelled`。流式响应中途断开或发送失败记为
`cancelled`，并且每个请求只记录一次终态。WebSocket 不经过该 HTTP middleware。
