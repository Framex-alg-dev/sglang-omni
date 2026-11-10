# 动作与表情召回接口说明｜2026-11-10

## 接口定义

```http
POST http://10.0.0.45:18048/v1/action-recall
Content-Type: application/json
Authorization: Bearer <服务鉴权值>
```

一次请求召回一个身体动作或面部表情。用户输入、数字人当前片段、数字人前文使用三组固定字段，每组支持文本或音频。

## 请求参数

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| mode | string | 是 | user_reaction：用户说完后召回；avatar_speech：数字人讲话中召回 |
| channel | string | 是 | body：身体动作；expression：表情 |
| character_prompt | string | 是 | 动态人设，非空，最多32000字符 |
| user_input | Input | 条件必填 | 用户原始问题；user_reaction未传requested_action时必填，avatar_speech中为可选背景 |
| avatar_current | Input | 条件必填 | 数字人当前待表达片段；avatar_speech必传，user_reaction不传 |
| avatar_history | Input | 否 | 数字人同一回复中紧邻当前片段的前文；仅avatar_speech可用 |
| requested_action | object / null | 否 | 用户明确指定的下一刻动作或表情；结构见下文，可与两种mode配合使用 |
| allowed_candidate_ids | string[] | 否 | 当前通道候选ID；省略使用当前通道全量候选 |
| last_dispatched_action | object / null | 否 | 上一条真实下发身体动作；仅body可用 |
| request_id | string | 否 | 请求标识，最多128字符；省略自动生成 |

未传requested_action时，mode=user_reaction以user_input为本次判断依据，不传avatar_current和avatar_history；mode=avatar_speech只为avatar_current召回，user_input与avatar_history仅供理解，不为背景补做动作或表情，两组背景均可省略。

传入requested_action时，以它指定的下一刻明确目标为本次任务，已有对话和讲话片段仅作背景。user_reaction可省略user_input；avatar_speech仍需avatar_current。人设参与筛选，但不扩展候选能力。

Input结构

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| text | string | 与audio二选一；非空，最多4000字符 |
| audio | object | 与text二选一；结构为下述data、format |
| audio.data | string | 音频字节的Base64，不带data URI前缀；解码后每段最多2MiB |
| audio.format | string | 可选，默认wav；支持wav、pcm_s16le |

每个Input必须且只能包含text或audio之一，不接受空对象或同组文本音频混传。三组输入可以独立选择模态，例如用户音频＋数字人当前文本＋数字人前文文本。可选背景省略或传null均表示未提供。

音频规格：16kHz、单声道、PCM16，每段可独立解码。所有传入音频合计不超过30秒。纯文本不要求音频、ASR或start_ms/end_ms；模型按文字语义判断，不臆测语调。前文只传同一回复中当前片段之前的内容，不重复包含当前片段或未来片段。

候选约束：数量和编码由运行时目录决定。7号机C666使用[business111部署目录](../../../deploy/c666_pro5000_7_20261110/assets/action_catalog.json)，body有98个真实候选、expression有13个；仓库persona_window_v1为旧116码本的冻结资产，不能与C666权重混用。指定列表时严格限定子集；空列表、null、重复、未知或跨通道ID返回422。candidate_id是目录候选ID，不等于模型输出code。

last_dispatched_action

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| candidate_id | string | 已登记身体动作ID，可不在本次候选子集中 |
| status | string | sent / running / completed / failed / cancelled |

未知上一动作时省略last_dispatched_action或传null；expression请求不传非null值。该字段表示真实下发记录，不表示上次模型预测。身体和表情分别调用，相同输入可复用但request_id须不同；同一request_id仅用于相同输入重试。

## 指定下一刻动作或表情

requested_action必须且只能包含以下两个字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| name | string | 是 | 用户自定义名称，非空，最多256字符；可以与目录名称不同 |
| definition | string | 是 | 具体动作或表情定义，非空，最多4000字符；描述手型、单/双手、部位、方向和对象等约束 |

这是用户希望数字人下一刻执行的明确目标，不是修改目录适用场景。模型结合名称和定义匹配本次候选；没有真正匹配项时返回000，不创建新能力。该字段省略或传null时沿用普通召回。请求中的候选边界仍然有效，执行时点、打断和转场由执行端处理。

```json
{
  "mode": "user_reaction",
  "channel": "body",
  "character_prompt": "自然友好的数字人助手。",
  "requested_action": {
    "name": "点赞",
    "definition": "双手握拳，两个大拇指一起向上竖起。"
  },
  "allowed_candidate_ids": ["271", "138"],
  "request_id": "next-action-001"
}
```

C666正确匹配该例时返回candidate_id="271"、code="_v_o"。名称相同不代表运动约束相同；当前小样本诊断中，单手点赞仍曾被误选为双手点赞，不能将格式和候选约束通过视为语义匹配全部正确。

## 请求示例

用户文本召回

```json
{
  "mode": "user_reaction",
  "channel": "body",
  "character_prompt": "耐心自然的老师。",
  "user_input": {
    "text": "请给我比一个三。"
  }
}
```

用户音频召回

```json
{
  "mode": "user_reaction",
  "channel": "expression",
  "character_prompt": "温柔耐心的朋友。",
  "user_input": {
    "audio": {
      "data": "<用户原始问题Base64>",
      "format": "wav"
    }
  }
}
```

数字人当前文本，用户背景为音频、数字人前文为文本

```json
{
  "mode": "avatar_speech",
  "channel": "body",
  "character_prompt": "温柔耐心的朋友。",
  "user_input": {
    "audio": {
      "data": "<原用户问题Base64>",
      "format": "wav"
    }
  },
  "avatar_history": {
    "text": "我听到了你的担心。"
  },
  "avatar_current": {
    "text": "别着急，我们可以一步一步解决。"
  }
}
```

数字人当前音频，用户背景为文本、数字人前文为音频

```json
{
  "mode": "avatar_speech",
  "channel": "expression",
  "character_prompt": "沉稳克制的商务顾问。",
  "user_input": {
    "text": "我不要选第二个方案，太冒险了。"
  },
  "avatar_history": {
    "audio": {
      "data": "<数字人前文Base64>",
      "format": "wav"
    }
  },
  "avatar_current": {
    "audio": {
      "data": "<数字人当前片段Base64>",
      "format": "wav"
    }
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

数字人讲话时按当前片段多次HTTP调用。首段可以省略avatar_history，不要求背景每次必传；用户背景与数字人当前片段的身份不随模态改变。

## 响应参数

```json
{
  "request_id": "reply001-seg002-expression",
  "channel": "expression",
  "candidate_id": "162",
  "code": "_j_k",
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
| occupies_channels | string[] | 额外占用通道，如face；无占用为空数组 |

000 / _ctrl_a / unsupported仅表示明确要求的具体能力无匹配候选，不表示保持或中性表情。本接口不返回IB0/IF0保持编码。执行时点、去重、转场和通道冲突由执行端处理。

响应示例编码采用7号机C666码本；接入时以运行服务的目录与返回值为准。

## 回捞日志

服务记录普通召回／明确目标两种recall_variant、原始requested_action、人设、对话、实际system/user提示词、候选快照、模型版本和输出。设置SGLANG_OMNI_ACTION_REPLAY_ROOT时，将音频及画面按实际内容SHA256保存，日志记录路径和完整性标记，便于按request_id回放核查。未配置媒体保存目录时，媒体不会被标记为可完整回放。日志不记录服务鉴权值。

## 错误码

| HTTP状态 | 说明 |
| --- | --- |
| 200 | 请求成功 |
| 422 | 模式或输入非法；同组text/audio混传；新旧字段混传；候选非法；request_id输入冲突 |
| 413 | 请求体超过服务配置上限 |
| 404 | 鉴权失败或接口不存在 |
| 500 | 服务内部推理错误 |

```json
{
  "detail": "avatar_current requires exactly one of text or audio"
}
```

旧请求保持兼容：原text/audio/audio_source/user_context/user_context_audio/context_audio仍可独立按旧协议调用。新接入使用上述三组输入；同一次请求不能混用新旧字段。

代码仓库：https://github.com/Framex-alg-dev/sglang-omni/tree/codex/speech-action-windows

文档来源：codex://threads/01a1a58c-0736-7963-9d27-8e3ab89ec2d7
