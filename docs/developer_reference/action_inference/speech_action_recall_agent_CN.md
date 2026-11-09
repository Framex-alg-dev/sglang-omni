# 动作与表情召回接口说明｜2026-11-09

## 接口定义

```http
POST http://10.199.0.4:18048/v1/action-recall
Content-Type: application/json
Authorization: Bearer <服务鉴权值>
```

## 请求参数

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| channel | string | 是 | body：身体动作；expression：表情 |
| character_prompt | string | 是 | 动态人设，非空，最多32000字符 |
| text | string | 条件必填 | 用户纯文本，最多4000字符 |
| audio | object | 条件必填 | 当前音频，结构见下文 |
| audio_source | string | 音频必填 | user_audio：用户；avatar_speech_audio：数字人 |
| context_audio | object | 否 | 数字人同一回复的前文音频；仅数字人音频请求可用 |
| user_context | string | 否 | 原用户文字语境，最多4000字符；与user_context_audio二选一 |
| user_context_audio | object | 否 | 原用户问题音频；仅数字人音频请求可用，省略不影响原调用 |
| allowed_candidate_ids | string[] | 否 | 当前通道候选ID列表；省略时使用通道全量候选 |
| last_dispatched_action | object / null | 否 | 上一条真实下发身体动作；仅body可用 |
| request_id | string | 否 | 请求标识，最多128字符；省略时自动生成 |

输入约束：纯文本请求使用text，不传audio、audio_source、context_audio、user_context、user_context_audio；音频请求使用audio和audio_source，不传text。

背景约束：user_context与user_context_audio均可省略，非null时二选一；二者仅用于avatar_speech_audio。user_context_audio是原用户背景，context_audio是数字人前文，audio是数字人当前片段。

候选约束：当前body有103个真实候选，expression有13个。指定候选时严格限定子集；空列表、null、重复、未知或跨通道ID返回422。候选ID不等于模型输出code。

调用约束：一次请求召回一个通道。身体与表情分别调用；相同音频可复用，request_id必须不同。同一request_id仅用于相同输入重试。

audio / context_audio / user_context_audio

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| data | string | 音频字节的Base64，不带data URI前缀 |
| format | string | 可选，默认wav；支持wav、pcm_s16le |

音频规格：16kHz、单声道、PCM16。每段须可独立解码；所有传入音频合计不超过30秒。context_audio应为同回复中紧邻当前片段之前的音频，不包含当前片段。

last_dispatched_action

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| candidate_id | string | 已登记身体动作ID，可不在本次候选子集中 |
| status | string | sent / running / completed / failed / cancelled |

未知上一动作时省略last_dispatched_action或传null；expression请求不传非null值。该字段表示真实下发记录，不表示上次模型预测。

## 请求示例

用户纯文本

```json
{
  "text": "请给我比一个三。",
  "channel": "body",
  "character_prompt": "耐心、自然的老师。"
}
```

用户音频

```json
{
  "audio_source": "user_audio",
  "channel": "expression",
  "character_prompt": "温柔、耐心的朋友。",
  "audio": {
    "data": "<用户音频Base64>",
    "format": "wav"
  }
}
```

数字人音频

```json
{
  "audio_source": "avatar_speech_audio",
  "channel": "expression",
  "character_prompt": "沉稳克制的商务顾问。",
  "user_context": "我不要选第二个方案，太冒险了。",
  "context_audio": {
    "data": "<数字人前文音频Base64>",
    "format": "wav"
  },
  "audio": {
    "data": "<数字人当前音频Base64>",
    "format": "wav"
  },
  "allowed_candidate_ids": [
    "154",
    "161",
    "162",
    "653"
  ],
  "request_id": "reply001-seg002-expression"
}
```

数字人音频（可选原用户音频背景）

```json
{
  "audio_source": "avatar_speech_audio",
  "channel": "expression",
  "character_prompt": "沉稳克制的商务顾问。",
  "user_context_audio": {
    "data": "<原用户问题音频Base64>",
    "format": "wav"
  },
  "context_audio": {
    "data": "<数字人前文音频Base64>",
    "format": "wav"
  },
  "audio": {
    "data": "<数字人当前音频Base64>",
    "format": "wav"
  }
}
```

数字人讲话时按当前片段多次HTTP调用；首段可省略context_audio。无需ASR、start_ms、end_ms或会话版本字段。

## 响应参数

```json
{
  "request_id": "reply001-seg002-expression",
  "channel": "expression",
  "candidate_id": "162",
  "code": "_l_i",
  "label": "疑惑",
  "outcome": "apply",
  "occupies_channels": []
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| request_id | string | 请求标识 |
| channel | string | body / expression |
| candidate_id | string | 选中候选ID |
| code | string | 模型输出编码 |
| label | string | 动作或表情名称 |
| outcome | string | execute：身体动作；apply：表情；unsupported：明确能力不支持 |
| occupies_channels | string[] | 额外占用通道，如face；无占用时为空数组 |

000 / _ctrl_a / unsupported仅表示明确要求的能力无匹配候选，不表示保持或中性表情。本接口不返回IB0/IF0保持编码。执行时点、去重、转场和通道冲突由执行端处理。

## 错误码

| HTTP状态 | 说明 |
| --- | --- |
| 200 | 请求成功 |
| 422 | 参数、音频或候选非法；文本音频混传；request_id输入冲突 |
| 413 | 请求体超过服务配置上限 |
| 404 | 鉴权失败或接口不存在 |
| 500 | 服务内部推理错误 |

```json
{
  "detail": "request_id was reused with different input"
}
```

文档来源：codex://threads/01a1a58c-0736-7963-9d27-8e3ab89ec2d7
