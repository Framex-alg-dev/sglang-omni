# 多模态 Session Realtime（方案 B）

/v1/session/realtime 是服务内置的、面向数字人多轮对话的手动 turn WebSocket。它与
/v1/realtime 分开，后者保持现有音频 + server VAD 行为。

## Action-score 进程级预热

pipeline 启动完成后，服务在对外提供 HTTP/WebSocket 服务前执行一次进程级预热：

1. 使用约 60 个短 category suffix 执行一次类别评分；
2. 使用约 8 个短 child suffix 执行一次子动作评分；
3. 不携带 session_id、音频、图片、历史对话或数字人状态；
4. 丢弃预热分数，不写入 session history，也不会向客户端发送 turn.result。

预热只用于初始化 action-score 的 request builder、tokenizer、scheduler admission
和 thinker 执行路径。预热耗时计入服务启动时间，不计入任何用户 turn。

环境变量：

- SGLANG_OMNI_ACTION_WARMUP=0：关闭预热，默认开启；
- SGLANG_OMNI_ACTION_WARMUP_CATEGORY_COUNT：预热类别候选数，默认 60；
- SGLANG_OMNI_ACTION_WARMUP_CHILD_COUNT：预热子动作候选数，默认 8；
- SGLANG_OMNI_ACTION_WARMUP_TIMEOUT_S：预热超时时间，默认 30 秒。

预热失败不会阻止服务启动，服务会记录 [ACTION_WARMUP] failed 并继续提供服务；
此时首个真实 turn 仍可能包含 action-score 冷启动耗时。预热事件会写入动作诊断
JSONL，但不会被当作用户 session 请求统计。

## Session 初始化

客户端必须首先发送 session.start，并在整个 session 内固定动作候选集合：

    {
      "type": "session.start",
      "session_id": "session-001",
      "model": "Qwen3-Omni-30B-A3B-Instruct",
      "language": "zh",
      "input_audio_format": "pcm16",
      "sample_rate": 16000,
      "channels": 1,
      "action_candidates": [
        {
          "candidate_id": "a01",
          "action_id": "wave_left",
          "source_label": "左手挥手",
          "short_definition": "使用左手抬起并左右摆动"
        },
        {
          "candidate_id": "none",
          "action_id": "no_action",
          "source_label": "不做动作",
          "short_definition": "保持当前姿态"
        }
      ]
    }

候选列表必须包含 action_id=no_action。服务端返回 action_catalog_hash，
服务端随后返回：

    {"type":"session.started", "session_id":"session-001", "model":"Qwen3-Omni-30B-A3B-Instruct", "action_catalog_hash":"sha256:...", "action_candidate_count":2, "action_category_count":0, "action_selection_stages":1}

后续 turn 不重复发送候选列表。

> 说明：上面的 flat `action_candidates` 仅用于兼容旧接入方。新接入方应使用下面的两层类别格式；嵌套格式下类别 ID 与 child candidate_id 必须跨层全局唯一。

## 层级动作候选与两阶段选择

生产接入建议在 `session.start` 中传入两层候选。第一层是类别，第二层是类别下的具体动作；类别 ID 和 child candidate_id 必须跨层全局唯一：

    {
      "type": "session.start",
      "session_id": "session-001",
      "model": "Qwen3-Omni-30B-A3B-Instruct",
      "action_candidates": [
      {
        "category_id": "B1",
        "source_label": "基础姿态",
        "short_definition": "站立、坐下等姿态变化",
        "children": [
          {"candidate_id": "A1", "action_id": "A1", "source_label": "正式站立", "short_definition": "双脚并拢，脊背挺直双臂自然垂放"},
          {"candidate_id": "A0", "action_id": "no_action", "source_label": "不做动作", "short_definition": "保持当前姿态"}
        ]
      }
      ]
    }

一个外部 turn 内部按以下顺序执行两阶段，外部不会收到中间事件，也不需要创建第二个 session：

1. 第一阶段固定使用 session 的类别 system prompt，只对所有 `category_id` 的短 suffix 评分，选出一个类别。
2. 第二阶段只使用选中类别的 system prompt 和 children，只对 child `candidate_id` 评分，选出最终动作。
3. 服务端只返回一个 `turn.result`；其中 `action` 和 `scores` 是第二阶段结果，`media_summary.action_context` 额外包含 `selected_category_id`、`category_scores`、`category_compute_ms`、`child_compute_ms` 和 `selection_stages=2`。

阶段请求使用同一个 `logical_request_id`，诊断日志中的 `stage` 分别为 `category` 和 `child`。第二阶段 system prompt 是当前 turn 内部临时构造的，不会追加到 session 历史，因此多轮对话不会累积多个 system prompt。

两个阶段使用同一个 turn 的音频、图片、历史消息和数字人状态引用；服务端只向外暴露一个逻辑请求。`server_action_compute_ms` 是两阶段总耗时。旧版 flat `action_candidates` 仍兼容，但只执行单阶段。

实现层会执行两个内部 action-score pipeline pass；这不是两个 session，也不是两个外部 turn。两阶段传入同一份媒体和历史输入，媒体 encoder 是否命中内容缓存由服务端缓存层决定。

## 当前 turn

用户按住空格期间：

    {"type":"turn.start","turn_id":"turn-003"}

音频使用 Base64 原始 PCM16：

    {
      "type": "input_audio.append",
      "turn_id": "turn-003",
      "seq": 1,
      "audio": "<Base64 PCM16>"
    }

也兼容现有 Realtime 的事件名：

    input_audio_buffer.append

图片帧使用 Base64 图片数据和时间戳：

    {
      "type": "input_image.append",
      "turn_id": "turn-003",
      "seq": 1,
      "timestamp_ms": 1000,
      "mime_type": "image/jpeg",
      "image": "<Base64 image bytes>"
    }

文本是可选的。用户松开空格后提交：

    {
      "type": "turn.commit",
      "turn_id": "turn-003",
      "text": null,
      "avatar_state": {
        "pose": "seated",
        "gaze": "camera",
        "hands": "resting"
      }
    }

服务端按顺序合并当前 turn 的音频 chunk，把图片按时间戳排序，然后只计算动作。
当前版本不生成普通数字人回复，turn.result.reply 固定为 null。动作决策完成后，
服务端会把当前用户输入和实际动作写入 session 内部历史；下一轮动作 Prompt
可以使用这条历史处理“刚刚那个动作”“再重复一遍”等指代。

## 方案 B 动作评分

嵌套候选格式下，类别层和 child 层分别进行 suffix 评分。下面的 `a01`/`none` 示例是旧 flat 格式的兼容说明；新接入方应使用上面的类别和 children。

候选动作定义位于 session system prompt：

    a01=wave_left：左手挥手；使用左手抬起并左右摆动
    none=no_action：不做动作；保持当前姿态

当前 turn 的 suffix 只使用短 candidate_id：

    完整 prompt + a01
    完整 prompt + none

第一阶段对 category_id 排名；第二阶段只对选中类别的 child candidate_id 排名，第一阶段完整排序保存在 `media_summary.action_context.category_scores`，最终 child 排序保存在顶层 `scores`，最终再映射成真正的 action_id。短 ID 优先设计成单 token，但实际以 tokenizer 的 token_count 为准；PPL 分母包含实际评分的 suffix token（包括显式终止 token）。

对候选 token 的计算为：

    mean_logprob = sum(token_logprob) / token_count
    mean_nll = -mean_logprob
    ppl = exp(mean_nll)

## 返回示例

    {
      "type": "turn.result",
      "session_id": "session-001",
      "turn_id": "turn-003",
      "reply": null,
      "action_catalog_hash": "sha256:abcd1234",
      "action": {
        "candidate_id": "A1",
        "category_id": "B1",
        "action_id": "A1",
        "execute": true,
        "mean_logprob": -0.35,
        "ppl": 1.42,
        "token_count": 1
      },
      "media_summary": {
        "audio_chunk_count": 24,
        "image_frame_count": 6,
        "scored_image_count": 6,
        "text_present": false,
        "action_context": {
          "selection_stages": 2,
          "logical_request_id": "session-session-001-turn-turn-003-action-...",
          "selected_category_id": "B1",
          "category_compute_ms": 310.2,
          "child_compute_ms": 532.1,
          "category_scores": [{"candidate_id":"B1", "mean_logprob":-0.21, "ppl":1.23, "token_count":1, "token_scores":[]}]
        }
      },
      "timing": {
        "server_turn_ingest_ms": 5280.4,
        "server_action_compute_ms": 842.317,
        "server_total_after_commit_ms": 1250.6
      }
    }

server_action_compute_ms 只表示服务端动作评分耗时。客户端应额外测量从
发送 turn.commit 到收到 turn.result 的往返耗时；两者差值是非计算耗时
估算，不等同于纯网络耗时。

## 完整诊断日志

动作评分链路会将诊断记录追加到 JSONL 文件。默认路径为 `/tmp/sglang-omni-action-debug.jsonl`，也可以通过环境变量 `SGLANG_OMNI_ACTION_DEBUG_LOG_FILE` 指定路径。记录包含 `session_id`、带 `turn_id` 的 request_id、候选列表、system prompt、逻辑 messages、实际渲染后的 `full_prompt`、prompt token 数、阶段耗时、异常堆栈以及音频/图片数量、大小和 hash。原始媒体 Base64 不写入日志。

超时时间默认 120 秒，可通过 `SGLANG_OMNI_ACTION_SCORE_TIMEOUT_S` 调整。出现超时时，重点查看 `event=action_scoring_timeout`、`phase.name`、`phase.slot_wait_ms` 和 `phase.pipeline_ms`，可区分动作评分排队和模型流水线耗时。

JSONL 会分别记录 `stage=category` 和 `stage=child` 的 started、prompt_rendered、completed 或 failed 事件；两个事件通过 `logical_request_id` 关联。服务端控制台目前记录处理完成和回复字符数，但不记录完整 WebSocket 出站 `turn.result`。接入方应以实际收到的 `turn.result` 作为端到端成功依据，并在客户端保存该帧。

## 已知隐患

当前实现使用固定的 `MAX_IMAGES_PER_TURN = 64` 作为服务端保护阈值。
该数值不是 Qwen3-Omni 或 SGLang 的模型硬限制，也不是业务要求。它可能导致
长 turn 在图片帧较多时被拒绝。后续应改为可配置的图片抽帧和送模策略，并区分
`received_image_count` 与 `scored_image_count`。本隐患当前只记录，不改变现有行为。

## 当前限制

- 一个 session 同时只能有一个 active turn；
- 嵌套格式最多 128 个类别、单类别最多 128 个 children，所有 child 总数最多 512；
- 一个 turn 最多 4096 个音频 chunk；
- 一个 turn 最多 64 张图片；
- 单张图片最大 8 MiB；
- 音频固定为 16 kHz、mono、PCM16；
- session 只保存在进程内，连接断开后释放；
- 第一版将视频输入表示为按时间顺序到达的图片帧。

## 外部接入协议补充（正式约定）

### 标识、模型和候选集合

session_id、turn_id 和 seq 均由外部调用方生成。turn_id 必须提供，服务端不会自动生成；同一个 session 内每个 turn_id 必须唯一。音频和图片分别维护自己的 seq。session.start 中的 model 仅供调用方记录，服务端始终使用启动配置中的模型，单个 session 不能修改。

category_id、candidate_id 和 action_id 由外部调用方在 session.start 中传入，并在整个 session 内固定；嵌套格式下 category_id 与 child candidate_id 必须跨层全局唯一。候选列表只初始化一次，必须在某个 child 中包含 action_id=no_action。服务端返回 action_catalog_hash，外部调用方应保存并在每次 turn.result 中校验。

### 完整事件表

| 事件 | 方向 | 作用 |
|---|---|---|
| session.start | 客户端 -> 服务端 | 初始化 session 和固定候选 |
| session.started | 服务端 -> 客户端 | 返回候选 hash 和 session 确认 |
| turn.start | 客户端 -> 服务端 | 开始一个新 turn |
| turn.started | 服务端 -> 客户端 | 确认 turn 已创建 |
| input_audio.append | 客户端 -> 服务端 | 追加 PCM16 音频 chunk |
| input_image.append | 客户端 -> 服务端 | 追加图片帧 |
| input.ack | 服务端 -> 客户端 | 确认媒体序号已接收 |
| turn.text.update | 客户端 -> 服务端 | 更新可选文本 |
| turn.commit | 客户端 -> 服务端 | 结束输入并触发回复/动作计算 |
| turn.committed | 服务端 -> 客户端 | 确认开始处理 |
| turn.result | 服务端 -> 客户端 | 返回回复、动作、排序和耗时 |
| turn.cancel | 客户端 -> 服务端 | commit 前取消当前 turn |
| turn.cancelled | 服务端 -> 客户端 | 确认取消 |
| error | 服务端 -> 客户端 | 返回协议或处理错误 |
| session.close | 客户端 -> 服务端 | 主动关闭 session |

第一条有效事件必须是 session.start；一个 session 同时只能有一个 active turn。
WebSocket 只使用 JSON 文本帧，媒体使用 Base64，不接受二进制帧。服务端仍兼容
input_audio_buffer.append，其字段与 input_audio.append 相同。

### 序号与媒体规则

音频格式固定为原始 PCM16 little-endian、单声道、16000 Hz，不带 WAV 头。
音频 seq 从 1 开始，必须按 1、2、3... 递增到达。重复 seq 是幂等重试，
服务端不重复追加并以 input.ack 的 duplicate=true 确认。跳号或乱序返回
invalid_sequence。图片 seq 与音频 seq 独立；图片允许乱序，commit 时按
timestamp_ms 升序、再按 seq 升序排序。图片支持 image/jpeg、image/png、
image/webp，image 可以是 Base64 内容或 Base64 data URI。

turn.commit 后不能继续追加媒体或更新文本。turn.cancel 只保证取消尚未
commit 的 turn。连接断开会清理当前进程内 session 状态，不做持久化。

### turn.result 和计时

turn.result 必须包含 reply、action、scores、media_summary、timing 和
action_catalog_hash。scores 是第二阶段选中类别下全部 child 候选的排序（flat 兼容格式则是全部 flat 候选的排序），每项保留 mean_logprob、ppl、
token_count 和 token_scores。方案 B 将候选定义固定在 session system prompt，
suffix 只使用短 candidate_id；Top-1 映射为真实 action_id，action_id=no_action
时返回 execute=false。

timing 字段含义：

- server_turn_ingest_ms：turn.start 到收到 commit，包含当前 turn 媒体接收；
- server_action_compute_ms：服务端动作评分耗时；
- server_total_after_commit_ms：收到 commit 到 turn.result，包含动作评分及其服务端编排，
  不包含普通数字人回复生成，因为当前动作服务不生成回复。

外部服务应另外测量 commit 到 turn.result 的端到端时间。该时间与
server_total_after_commit_ms 的差值只能作为排队和传输开销估计。

media_summary 至少区分 audio_chunk_count、image_frame_count、
received_image_count、scored_image_count 和 text_present。当前版本默认
收到的图片全部送模，但 64 张图片是服务端保护阈值，不是模型硬限制。

### 错误协议

错误格式：

    {
      "type": "error",
      "session_id": "session-001",
      "turn_id": "turn-003",
      "error": {
        "type": "invalid_request",
        "code": "invalid_sequence",
        "message": "audio seq must be monotonic"
      }
    }

服务端能够确定时返回 session_id 和 turn_id；缺失的标识会省略。接入方需要处理：

| code | 含义 |
|---|---|
| session_not_started | session.start 尚未成功 |
| duplicate_session_id | session_id 已被另一个连接占用 |
| duplicate_active_turn | 当前 session 已有 active turn |
| invalid_turn_id | turn_id 缺失、为空或不匹配 |
| invalid_sequence | 音频 seq 缺失、跳号或乱序 |
| invalid_audio | PCM16、Base64 或音频参数非法 |
| invalid_image | 图片 Base64、类型或大小非法 |
| turn_already_committed | turn 已提交、已取消或不存在 |
| session_candidate_invalid | 候选缺失、重复、超限或缺少 no_action |
| server_error | 服务端生成或评分异常 |

协议解析还可能返回 invalid_json、invalid_event 或 binary_frames_not_supported。

## 错误诊断日志

当动作评分或模型预处理失败时，服务端会输出两类诊断日志：

- action scoring request failed; full logical input=...：记录 request/session
  标识、system prompt、动作 prefix、候选 suffix、逻辑 messages、数字人状态和
  音频/图片输入摘要；
- Qwen3-Omni prompt validation failed; full prompt diagnostics=...：记录模型
  chat template 展开后的完整 full_prompt、实际 prompt token 数、最大上下文、
  messages、动作参数和媒体摘要。

媒体摘要包含引用、数量、字符长度和 hash；不会直接打印音频或图片 Base64
原文。full_prompt 是本次模型真正使用的文本 prompt，出现上下文超长时可直接
用它定位是哪一轮历史或候选定义造成了超限。

当前 thinker 部署上下文为 8192 tokens。为避免多轮媒体历史无限增长，动作评分
使用独立的有界上下文：最多保留最近 4 个历史 turn、4 个历史音频、8 个历史
图片和当前 turn 最近 8 张图片。每个已完成 turn 的 assistant 历史会保存一条
[action_state]，包含 turn_id、candidate_id、action_id、动作名称和动作描述；
最近一轮的动作指代可以直接依赖该记录，超过有界历史的旧动作记录可能被裁剪。
turn.result.media_summary.action_context 会返回实际保留数量以及是否发生裁剪。
