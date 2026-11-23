# 41035 sglang-omni 统一分支接口说明

> 文档状态：按 `codex/41035-omni-integration-20261121` 分支、提交
> `6b43da4c` 整理，日期为 2026-11-23。
>
> 本文描述当前代码已经实现的接口，不代表下一阶段的统一接口方案。

快速导航：

- 第 1～3 节：架构、端点和两种 WebSocket 外层协议
- 第 4 节：Agent 会话级 Inference Gateway
- 第 5～7 节：Reply、Action、Event 业务接口
- 第 8 节：尚未稳定的 Prompt Snapshot 扩展
- 第 9～10 节：错误语义和 Agent 接入检查清单

## 1. 先说结论

当前完成的是**代码分支统一**，不是把 Reply、Action、Event 合并成一个模型、
一个进程或一套 wire protocol。

三个模型仍然独立部署：

| 能力 | 模型服务 | 主要 Agent 接口 | 当前模型身份 |
| --- | --- | --- | --- |
| Reply | OpenAI Reply Worker | Gateway `stage=reply`，或直连 `/v1/chat/completions/realtime` | 由 Reply 部署配置决定 |
| Action | Action V10 | `/v1/action-recall`；完整多模态场景也可经 Gateway 的 `body` / `expression` stage | `v10-boundary-ep1-final-fp8` |
| Event | Event 1115 Timeline | `/v1/timeline` | ID `event-1115-v1-fp8`；version `1115_v1-checkpoint-6138-fp8` |

当前建议的连接关系是：

```text
                         +--> Reply Worker /v1/chat/completions/realtime
Agent --> Inference GW --+--> Action Worker /v1/action-decision/realtime
  |                      +--> 其他 classifier / brain / performance stage
  |
  +--------------------------> Action Worker /v1/action-recall
  |
  +--------------------------> Event Worker /v1/timeline
```

Inference Gateway 用于会话内媒体复用和多 stage 编排；Action Recall 与 Timeline
仍是独立接口。不要把 Gateway v4、模型请求 v2、Timeline v2、Action Decision v1
理解成同一个 `contract_version`。

## 2. 服务和端点总览

| 服务 | 默认/当前端口 | 端点 | 鉴权 |
| --- | ---: | --- | --- |
| Inference Gateway | `18003` | `WS /v1/inference-session` | 当前代码未校验客户端 token，只应暴露在可信内网 |
| Inference Gateway | `18003` | `GET /live`、`/ready`、`/health` | 无 |
| Reply Worker | 通常 `18005` | `WS /v1/chat/completions/realtime` | 当前路由未校验 Bearer；通常只允许 Gateway 内部访问 |
| Reply Worker | 通常 `18005` | `POST /v1/chat/completions` | 当前路由未校验 Bearer |
| Reply Worker | 通常 `18005` | `GET /health`、`/v1/models` | 无 |
| Action V10 | 当前 `18049` | `WS/POST /v1/action-recall` | `Authorization: Bearer <token>` |
| Action V10 | 当前 `18049` | `WS /v1/action-decision/realtime` | `Authorization: Bearer <token>` |
| Action V10 | 当前 `18049` | `POST /v1/action-decision` | `Authorization: Bearer <token>` |
| Action V10 | 当前 `18049` | `GET /health` | `Authorization: Bearer <token>` |
| Event 1115 | 当前 `18118` | `WS /v1/timeline` | `Authorization: Bearer <token>` |
| Event 1115 | 当前 `18118` | `GET /health` | `Authorization: Bearer <token>` |

代码快照记录的 41035 地址为：

- Action：`ws://10.199.0.7:18049/v1/action-recall`
- Event：`ws://10.199.0.15:18118/v1/timeline`

地址属于部署配置，不属于协议。接入代码不应写死 IP。

## 3. 两种 WebSocket 外层协议

### 3.1 Gateway 会话协议：`inference-session-v4`

Gateway 连接是长会话连接。一份媒体先注册一次，之后多个 stage 用 `media_refs`
引用。

打开会话：

```json
{
  "type": "session.open",
  "contract_version": 4,
  "session_id": "sess_example"
}
```

成功响应：

```json
{
  "type": "session.ready",
  "contract_version": 4,
  "session_id": "sess_example",
  "availability_epoch": 0,
  "stages": {
    "body": true,
    "brain": true,
    "classifier": true,
    "expression": true,
    "performance": true,
    "reply": true
  }
}
```

Gateway 当前支持六个 stage：`classifier`、`brain`、`reply`、`body`、
`expression`、`performance`。

### 3.2 模型单请求协议：`request.start v2`

Reply Worker、Action Decision 以及 Action Recall 的 WebSocket 都使用相同的传输
外壳，但 `payload` 业务结构不同。

```text
client                         server
  | request.start v2             |
  |----------------------------->|
  | request.ready v2             |
  |<-----------------------------|
  | input.media JSON（可重复）    |
  | binary payload               |
  |----------------------------->|
  | input.media.ack              |
  |<-----------------------------|
  | request.commit               |
  |----------------------------->|
  | response.delta（Reply 可选）  |
  | response.completed / error   |
  |<-----------------------------|
```

首帧：

```json
{
  "type": "request.start",
  "contract_version": 2,
  "request_id": "req-1",
  "payload": {}
}
```

媒体头之后必须紧跟一个 binary WebSocket frame：

```json
{
  "type": "input.media",
  "request_id": "req-1",
  "media_id": "frame-1",
  "kind": "image",
  "start_ms": 0,
  "end_ms": 100,
  "encoding": "image/jpeg",
  "checksum": "sha256:<64位小写十六进制>",
  "payload_bytes": 12345,
  "evidence_role": "user_camera"
}
```

通用媒体约束：

| `kind` | 可用 `evidence_role` | 常用编码 |
| --- | --- | --- |
| `audio` | 接口允许的音频角色；Reply 常用 `user_audio` | `pcm_s16le`、`wav` |
| `image` | `user_camera`、`avatar_state` | `image/jpeg`、`image/png` |
| `video` | `user_camera` | 当前模型路径通常仍传采样后的图片帧 |

`avatar_state` 必须是 JPEG。服务会校验字节数和 SHA-256。

## 4. Inference Gateway 接口

### 4.1 注册媒体

Gateway 使用 `media.put`，而不是 `input.media`：

```json
{
  "type": "media.put",
  "media_id": "turn-7-frame-1",
  "kind": "image",
  "start_ms": 1200,
  "end_ms": 1300,
  "encoding": "image/jpeg",
  "checksum": "sha256:<digest>",
  "payload_bytes": 12345,
  "evidence_role": "user_camera"
}
```

发送 JSON 后紧跟 binary frame。成功返回：

```json
{
  "type": "media.ack",
  "media_id": "turn-7-frame-1",
  "kind": "image",
  "start_ms": 1200,
  "end_ms": 1300,
  "encoding": "image/jpeg",
  "checksum": "sha256:<digest>",
  "payload_bytes": 12345,
  "evidence_role": "user_camera",
  "duplicate": false
}
```

同一个 `media_id` 可以用完全相同的头和字节重复注册，响应中
`duplicate=true`；同 ID 不同内容会关闭会话。媒体可能因容量限制收到
`media.evicted`，之后不能再被新 stage 引用。

默认限制：单媒体 32 MiB、会话媒体总量 128 MiB、最多 64 项。

### 4.2 发起 stage

```json
{
  "type": "stage.request",
  "request_id": "reply-turn-7",
  "stage": "reply",
  "payload": {
    "model": "qwen3-omni-reply-gpu",
    "messages": [
      {"role": "system", "content": "应用侧完整回复规则"},
      {"role": "user", "content": "你好"}
    ],
    "stream": true,
    "modalities": ["text"]
  },
  "media_refs": ["turn-7-frame-1"]
}
```

立即接收：

```json
{
  "type": "stage.accepted",
  "request_id": "reply-turn-7",
  "stage": "reply"
}
```

后续可能收到：

| 事件 | 含义 |
| --- | --- |
| `stage.model_input` | 仅开启模型输入诊断时出现；不包含原始二进制 |
| `stage.delta` | 流式 Reply 增量，`response` 是 OpenAI chunk |
| `stage.completed` | stage 成功终态，`response` 为上游最终响应 |
| `stage.error` | stage 失败；包含 `code`、`detail`、`retryable` |
| `stage.cancelled` | 已运行请求被取消 |
| `stage.availability` | 上游可用性变化及 `availability_epoch` |

取消请求：

```json
{"type":"stage.cancel","request_id":"reply-turn-7"}
```

关闭会话：

```json
{"type":"session.close"}
```

Gateway 当前没有单独的成功关闭回执；发送后客户端应正常关闭 socket。

### 4.3 Gateway 的 Action stage

`stage=body` 或 `stage=expression` 时，Gateway 会强制把相同名称写入
`payload.channel`，然后转发到 Action Decision Worker。两通道仍然是两次独立
请求；Gateway 不替 Agent 做最终融合。

## 5. Reply 接口

### 5.1 推荐路径

Agent 优先通过 Gateway `stage=reply` 调用。直连 Worker 时使用：

```text
WS /v1/chat/completions/realtime
```

外层采用第 3.2 节的 `request.start v2`。最小 payload：

```json
{
  "model": "qwen3-omni-reply-gpu",
  "messages": [
    {"role": "system", "content": "完整的应用回复规则"},
    {"role": "user", "content": "用户输入"}
  ],
  "stream": true,
  "modalities": ["text"],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 512
}
```

主要字段：

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `messages` | 是 | OpenAI 风格消息；应用提示词也放在这里 |
| `model` | 否 | 省略时使用 Worker 默认模型 |
| `stream` | 否 | `true` 时返回 `response.delta` |
| `modalities` | 否 | 默认 `text`；可请求 `text`、`audio` |
| `temperature/top_p/top_k/min_p` | 否 | 采样参数 |
| `max_tokens` / `max_completion_tokens` | 否 | 后者优先 |
| `response_format` | 否 | 支持 JSON schema 约束 |
| `runtime_prompt_profile` | 否 | 内部扩展；见下面的提示词说明 |
| `runtime_prompt_language` | 否 | `zh-CN` / `en-US` 等 |

流式事件示例：

```json
{
  "type": "response.delta",
  "request_id": "reply-1",
  "response": {
    "id": "chatcmpl-reply-1",
    "choices": [{"index":0,"delta":{"content":"你好"},"finish_reason":null}]
  }
}
```

流式完成时：

```json
{"type":"response.completed","request_id":"reply-1","response":{}}
```

非流式完成时，`response` 是完整 OpenAI Chat Completion 对象。

### 5.2 Reply 提示词边界

- `runtime_prompt_profile` 省略：Worker 原样使用调用方提供的 `messages`。
- `runtime_prompt_profile="reply"`：Worker 会在最前面额外插入一条本地
  `reply_rules` system message。
- 如果提示词由 Agent 完整管理，Agent 应直接构造 `messages`，并避免同时设置
  `runtime_prompt_profile="reply"`，否则会出现两份 system policy。
- 当前 Reply 接口没有 Action/Event 所使用的 `prompt_snapshot` 字段，也没有
  prompt release 回执。

## 6. Action 接口

Action 有两层业务接口。二者使用同一个 V10 模型和目录，但请求结构不同。

### 6.1 `/v1/action-recall`：简化召回接口

这是当前 41035 动作/表情召回的主要接口，支持 HTTP 和可复用 WebSocket。
WebSocket 可在一条连接上并发最多 4 个 request。

每个决策点分别请求：

```json
{"channel":"body"}
```

和：

```json
{"channel":"expression"}
```

服务不会一次同时返回 body 与 expression。

#### user_reaction

文本示例：

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

音频可以内联：

```json
{
  "user_input": {
    "audio": {"data":"<base64 wav>","format":"wav"}
  }
}
```

也可以通过 `input.media` binary frame 上传，角色必须是 `user_audio`。

#### avatar_speech

用于数字人正在讲话时选择伴随动作或表情：

```json
{
  "request_id": "reply-9-expression",
  "mode": "avatar_speech",
  "channel": "expression",
  "character_prompt": "角色动作与表达风格",
  "allowed_candidate_ids": ["IF0", "655"],
  "user_input": {"text": "用户上一轮说的话"},
  "avatar_history": {"text": "数字人此前已说内容"},
  "avatar_current": {"text": "数字人当前要说内容"}
}
```

三类音频角色分别为：

| 字段 | binary `evidence_role` | 最大时长 |
| --- | --- | ---: |
| `user_input` | `user_context_audio` | 60 秒 |
| `avatar_history` | `avatar_speech_context` | 4 秒 |
| `avatar_current` | `avatar_speech_audio` | 4 秒 |

音频必须是 16 kHz、单声道、PCM16 WAV 或 `pcm_s16le`。文本与对应音频
互斥。三段音频合计最大 68 秒。

#### action_request

用于宿主明确指定下一刻目标动作/表情：

```json
{
  "request_id": "explicit-1",
  "mode": "action_request",
  "channel": "body",
  "character_prompt": "角色动作与表达风格",
  "allowed_candidate_ids": ["288", "000"],
  "user_input": {"text": "挥手：抬起右手左右摆动"}
}
```

`user_input.text` 必须是 `名称: 定义` 或 `名称：定义`。该模式不接受
`avatar_current`、`avatar_history`。

#### Recall 限制

- `request_id` 最长 128 字符，只允许字母、数字及 `._:-`。
- `character_prompt` 必需，最大 32,000 字符。
- `allowed_candidate_ids` 若传入，必须是非空、唯一、已注册候选，最多 512 个。
- 每个字段的文本最大 4,000 字符。
- WebSocket binary 模式最多 3 个媒体项，每项最大 2 MiB。
- 当前 Recall adapter 只接收文本和音频，不接收图片/视频。视觉手势等任务应走
  `/v1/action-decision/realtime` 的完整多模态接口。
- 同一 `request_id` 同一输入可复用结果；同 ID 不同输入会被拒绝。

#### Recall 响应

```json
{
  "request_id": "turn-7-body",
  "channel": "body",
  "candidate_id": "189",
  "code": "_x_y",
  "label": "动作名称",
  "outcome": "execute",
  "occupies_channels": ["body"]
}
```

`outcome`：

| channel | 普通候选 | 保持/不动作 | 不支持 |
| --- | --- | --- | --- |
| `body` | `execute` | `no_action`（`IB0`） | `unsupported`（`000`） |
| `expression` | `apply` | `keep`（`IF0`） | `unsupported`（`000`） |

### 6.2 `/v1/action-decision/realtime`：完整决策接口

该接口适合带用户画面、原始音频、明确 session/turn fence 的请求。外层仍是
`request.start v2`。

```json
{
  "request_id": "body-turn-7-r3",
  "session_id": "sess-1",
  "turn_id": "turn-7",
  "decision_point_id": "turn-7:revision-3",
  "channel": "body",
  "text": "模仿我刚才的手势",
  "language": "zh-CN",
  "character_prompt": "角色动作与表达风格",
  "session_prompt": "当前会话场景",
  "runtime_context": {
    "turn_origin": "user",
    "identity_epoch": 1,
    "input_revision": 3,
    "prompt_profile": "e67_v116_step695_fp8"
  },
  "allowed_candidate_ids": ["gesture-1", "gesture-2", "000"],
  "excluded_candidate_ids": [],
  "channel_enabled": true,
  "prohibited": false,
  "turn_origin": "user"
}
```

图片使用 `input.media`，`kind=image`、`evidence_role=user_camera`。完整响应
包含 `contract_version=1`、session/turn/decision fence、候选、code、token IDs、
model/catalog/mapping version、耗时以及是否真正调用模型。

`channel_enabled=false` 或 `prohibited=true` 时服务确定性返回控制项，不调用模型。

### 6.3 Action 健康检查

```http
GET /health
Authorization: Bearer <token>
```

关键返回：

```json
{
  "ok": true,
  "contract": "action-decision.v1",
  "channels": ["body", "expression"],
  "model_id": "v10-boundary-ep1-final-fp8",
  "model_version": "v10-boundary-ep1-final-fp8",
  "catalog_version": "taxonomy4-global210-review-v1",
  "mapping_version": "taxonomy4-global210-review-v1",
  "runtime": {}
}
```

## 7. Event Timeline 接口

### 7.1 建立连接

```text
WS /v1/timeline
Authorization: Bearer <token>
```

第一条消息必须是：

```json
{
  "type": "session.start",
  "contract_version": 2,
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "stream_epoch": 0,
  "next_sequence": 1,
  "audio_format": "pcm_s16le/16000/1",
  "video_format": "image/jpeg",
  "model_id": "event-1115-v1-fp8",
  "result_mode": "raw_windows.v1"
}
```

成功响应会原样回显 session fence：

```json
{
  "type": "session.ready",
  "contract_version": 2,
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "stream_epoch": 0,
  "next_sequence": 1,
  "result_mode": "raw_windows.v1"
}
```

`result_mode` 有两种：

| 值 | 返回 |
| --- | --- |
| `raw_windows.v1` | 每个模型窗口返回一次 `window.result`，可以是空事件数组 |
| `observations` | 只返回策略处理后的单条 `observation`；省略字段时的兼容默认值 |

Event 1115 当前建议使用 `raw_windows.v1`，由 Agent 统一做事件策略、去重和回复触发。

### 7.2 发送媒体

Timeline 的媒体头类型是 `media`，发送 JSON 后紧跟 binary frame：

```json
{
  "type": "media",
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "stream_epoch": 0,
  "sequence": 1,
  "kind": "video",
  "start_ms": 0,
  "end_ms": 500,
  "encoding": "image/jpeg",
  "payload_bytes": 48213
}
```

音频示例：

```json
{
  "type": "media",
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "stream_epoch": 0,
  "sequence": 2,
  "kind": "audio",
  "start_ms": 0,
  "end_ms": 500,
  "encoding": "pcm_s16le",
  "payload_bytes": 16000
}
```

服务返回：

```json
{"type":"media.ack","observer_epoch":0,"sequence":2}
```

注意：

- `sequence` 是音频和视频共享的一条全局连续序列，不能分别编号。
- `media.ack` 只表示服务已接管字节，不表示模型已经完成该窗口。
- 默认单 chunk 最大 8 MiB，等待队列最多 64 个 chunk、128 MiB。
- 视频 chunk 必须是 JPEG 帧；音频支持 PCM16 / `pcm_s16le` / `audio/l16`。
- 结果引用的 `evidence_end_ms` 不会超过服务已经接收的媒体时间。

### 7.3 原始窗口结果

```json
{
  "type": "window.result",
  "contract_version": 2,
  "status": "complete",
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "stream_epoch": 0,
  "window_sequence": 12,
  "evidence_start_ms": 1000,
  "evidence_end_ms": 11000,
  "model_id": "event-1115-v1-fp8",
  "model_version": "1115_v1-checkpoint-6138-fp8",
  "evidence_mode": "audio_video",
  "audio_status": "complete",
  "events": [
    {"event_type":"E50","summary":"检测到对应事件"}
  ]
}
```

空窗口是正常结果：

```json
{"type":"window.result","status":"complete","events":[]}
```

`evidence_mode` / `audio_status` 合法组合：

- `audio_video` + `complete`
- `video_only` + `missing|gapped|behind`

### 7.4 流中断

RTC 重连、媒体时钟重置或丢失连续性时，不应继续沿用旧 `stream_epoch`：

```json
{
  "type": "discontinuity",
  "session_id": "sess-1",
  "identity_epoch": 1,
  "observer_epoch": 0,
  "old_stream_epoch": 0,
  "new_stream_epoch": 1,
  "reason": "rtc_reconnect"
}
```

服务处理完成后返回：

```json
{"type":"discontinuity.ack","observer_epoch":0,"stream_epoch":1}
```

新 epoch 的下一条媒体从 `sequence=1` 开始。

### 7.5 正常关闭

```json
{"type":"session.close"}
```

服务会先处理所有已经 ACK 的媒体，再返回：

```json
{"type":"session.closed","observer_epoch":0}
```

## 8. Prompt Snapshot 扩展：当前实现状态

当前分支实现了可选 `prompt_snapshot`，但它仍处于接口试验阶段。根据本分支
review，现有 Action 多 profile 和现网 Event 1115 资产还不能无损迁移，因此生产
接入暂时不应发送该字段。

### 8.1 Action Snapshot

Action payload 可选字段：

```json
{
  "prompt_snapshot": {
    "consumer": "d_video_call.action.decision",
    "release_id": "不可变release ID，不能是active",
    "bundle_sha256": "<sha256>",
    "binding": {
      "assembly_contract": "d-action-decision-v1",
      "output_protocol": "d-action-decision-response-v1",
      "model_id": "v10-boundary-ep1-final-fp8",
      "model_version": "v10-boundary-ep1-final-fp8"
    },
    "system_prompt": "...",
    "system_prompt_sha256": "<sha256>"
  }
}
```

服务使用 snapshot 时，响应额外包含：

```json
{
  "prompt_release_id": "...",
  "prompt_bundle_sha256": "..."
}
```

### 8.2 Timeline Snapshot

`session.start` 可选字段：

```json
{
  "prompt_snapshot": {
    "consumer": "d_video_call.timeline.detection",
    "release_id": "不可变release ID",
    "bundle_sha256": "<sha256>",
    "binding": {
      "assembly_contract": "d-timeline-detection-v1",
      "output_protocol": "d-timeline-observation-v2",
      "model_id": "event-1115-v1-fp8",
      "model_version": "1115_v1-checkpoint-6138-fp8"
    },
    "system_prompt": "...",
    "event_registry_json": "...",
    "event_codebook_csv": "...",
    "item_sha256": {
      "system_prompt": "<sha256>",
      "event_registry_json": "<sha256>",
      "event_codebook_csv": "<sha256>"
    }
  }
}
```

服务使用 snapshot 时，`session.ready` 额外包含相同的
`prompt_release_id`、`prompt_bundle_sha256`。

下一阶段设计 Agent 接入时，需要先修订 snapshot 的模型绑定来源、Action 多
profile 表达方式和 Event 原始资产格式，再把这一节标记为稳定协议。

## 9. 错误与关闭语义

| 接口 | 非法鉴权 | 非法请求 | 模型/内部错误 |
| --- | --- | --- | --- |
| Gateway | 当前无鉴权 | `error` 后 WS 4400，或单 stage `stage.error` | `stage.error`；会话级错误 1011 |
| Reply WS | 当前无鉴权 | `error code=invalid_request`，WS 4400 | WS 1011 |
| Action Decision WS | WS 4404 | `error code=invalid_request`，WS 4400 | `error code=model_error`，WS 1011 |
| Action Recall WS | WS 4404 | 单 request `error code=invalid_request`，连接可继续 | 单 request `error code=model_error` |
| Timeline WS | WS 4404 | `error code=invalid_request`，WS 4400 | `error code=model_failure`，WS 1011 |

HTTP Action 鉴权失败刻意返回 404，避免暴露服务；非法 JSON/字段返回 422，过大
请求返回 413。

## 10. Agent 接入前需要确认的清单

1. Reply 走 Gateway 还是直连 Worker；同一部署只能明确选择一种主路径。
2. Reply 提示词由 Agent messages 完整提供时，不设置 `runtime_prompt_profile=reply`。
3. Action 的用户完成态 body、expression 是否都调用；两者不能互相取消。
4. 视觉手势走完整 Action Decision，不走只支持文本/音频的 Recall adapter。
5. Event 1115 使用 `raw_windows.v1`，事件策略由 Agent 处理。
6. Agent 严格维护 Timeline 的 identity/observer/stream epoch 和全局 sequence。
7. Prompt Snapshot 在完成兼容性修订前保持关闭。
8. Gateway 与 Reply Worker 当前没有 Agent-facing 鉴权，部署时必须有内网边界或补充鉴权。

## 11. 代码依据

- Gateway：`sglang_omni/serve/inference_gateway/app.py`
- 通用模型 WS 传输：`sglang_omni/serve/streaming_request.py`
- Reply：`sglang_omni/serve/openai_api.py`、`sglang_omni/serve/protocol.py`
- Action：`sglang_omni/serve/action_decision/`
- Timeline：`sglang_omni/serve/timeline_detection/`
- Raw window：`framex_event_policy/windows.py`
- Prompt Snapshot：`sglang_omni/serve/agent_prompt_snapshot.py`
- 当前部署代码来源：`deploy/41035_CODE_CONSOLIDATION.md`
