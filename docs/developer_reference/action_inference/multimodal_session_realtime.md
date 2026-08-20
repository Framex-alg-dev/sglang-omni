# 多模态 Session Realtime（动作推理）

/v1/session/realtime 是服务内置的、面向数字人多轮对话的手动 turn WebSocket。它与
/v1/realtime 分开，后者保持现有音频 + server VAD 行为。

Session Realtime 只负责从当前角色在 `session.start` 中冻结的候选集合里选择
`action_id` 或 `no_action`。它不生成回复文本、动作脚本，也不执行 Motion。

## 接入方必改项

本节面向已经接入旧版 Session Realtime 的客户端，只列当前分支带来的增量迁移要求。
原有 `session.start`、候选目录、音频/图片追加、`action` 执行逻辑无需重做；
客户端需要调整以下内容：

1. `turn.start` 和 `turn.commit` 都必须发送 `turn_origin`、`text_role`，并保持一致；
2. 只允许 `user/user_input` 和 `proactive/character_reply` 两种组合，服务端不会自动纠正；
3. proactive Turn 的数字人待播文本可选；`trigger` 可选，但如果在 start 中提供，
   commit 中去除首尾空白后的值必须一致；
4. proactive Turn 的 `user_input` 必须为 `null` 或省略，服务端只把 `text` 当作待播文本；
5. 动作候选继续由调用方提供并在 Session 内冻结，flat 和 hierarchical 格式均继续支持，
   且候选中必须包含 `action_id=no_action`；
6. `current_action_id` 表示上一次已执行动作；主动 Turn 的 `state_description`
   表示本次主动场景说明。`pose`、`gaze`、`hands`、`conversation_phase` 等视觉或
   状态字段继续兼容；
7. `turn.result.action` 和 `execute` 仍是动作执行入口；`reply` 不再返回，`scores`
   和 `media_summary` 仍只在 `include_scores=true` 时返回。

未携带新字段的旧 Turn 请求会返回
`invalid_request / invalid_turn_semantics`，因此调用方需要同步升级 start 和 commit，
不能只改其中一个事件。

## Action-score 进程级预热

pipeline 启动完成后，服务在对外提供 HTTP/WebSocket 服务前执行一次进程级预热：

1. hierarchical 模式使用约 60 个短 category suffix 执行一次类别评分，再使用约
   8 个短 child suffix 执行一次子动作评分；
2. flat_children 模式只使用约 8 个短具体动作 suffix 执行一次 single 阶段评分，
   不执行 category 阶段；
3. 不携带 session_id、音频、图片、历史对话或数字人状态；
4. 丢弃预热分数，不写入 session history，也不会向客户端发送 turn.result。

预热只用于初始化 action-score 的 request builder、tokenizer、scheduler admission
和 thinker 执行路径。预热耗时计入服务启动时间，不计入任何用户 turn。

环境变量：

- SGLANG_OMNI_ACTION_WARMUP=0：关闭预热，默认开启；
- SGLANG_OMNI_ACTION_WARMUP_CATEGORY_COUNT：预热类别候选数，默认 60；
- SGLANG_OMNI_ACTION_WARMUP_CHILD_COUNT：hierarchical 模式的预热子动作数，
  或 flat_children 模式的具体动作数，默认 8；
- SGLANG_OMNI_ACTION_WARMUP_TIMEOUT_S：预热超时时间，默认 30 秒。
- SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE：子动作/flat suffix micro-batch，默认 64，范围 1–256。
- SGLANG_OMNI_ACTION_CATEGORY_TOP_K：hierarchical 类别兜底宽度，默认 1，范围 1–3。
  默认只评分类别 Top-1 下的 children；设置为 2 或 3 时，会将对应 Top-K 类别的
  children 合并到第二阶段评分，用于缓解类别边界模糊导致的动作漏选，但会增加第二阶段
  候选数和耗时。flat_children 不使用该配置。

预热失败不会阻止服务启动，服务会记录 [ACTION_WARMUP] failed 并继续提供服务；
此时首个真实 turn 仍可能包含 action-score 冷启动耗时。预热事件会写入动作诊断
JSONL，但不会被当作用户 session 请求统计。

## 动作选择模式

服务进程通过 SGLANG_OMNI_ACTION_SELECTION_MODE 提供动作选择默认策略，默认值为
hierarchical。候选仍由外部在 session.start 中传入；session.start 也可以传入
selection_mode，且优先级高于环境变量。非法的 session.start.selection_mode 不会
阻断 session，服务端会忽略它并继续使用环境变量或默认值：

- hierarchical：默认逻辑。先评分所有 category，再评分选中类别的 children，
  一个外部 turn 内部执行两个 action-score 阶段；
- flat_children：将所有类别的 children 按原始顺序拍平，所有具体动作放入同一个
  system prompt，只执行一次 action-score，直接返回具体动作。

示例：

~~~bash
SGLANG_OMNI_ACTION_SELECTION_MODE=hierarchical
SGLANG_OMNI_ACTION_SELECTION_MODE=flat_children
~~~

两种模式都保持同一个 session_id、turn_id、action catalog hash 和 session
历史；两者区别仅在内部候选评分路径。session.started 会返回实际的
action_selection_mode 和 action_selection_stages。flat_children 模式仍会在
action、scores 和 media_summary.action_context 中保留 child 的 category_id。

## Session 初始化

客户端必须首先发送 session.start，并在整个 session 内固定动作候选集合：

    {
      "type": "session.start",
      "session_id": "session-001",
      "model": "Qwen3-Omni-30B-A3B-Instruct",
      "include_scores": false,
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

    {"type":"session.started", "session_id":"session-001", "model":"Qwen3-Omni-30B-A3B-Instruct", "action_catalog_hash":"sha256:...", "action_candidate_count":2, "action_category_count":0, "action_selection_mode":"hierarchical", "action_selection_stages":1, "action_prefix_prefilled":true}

后续 turn 不重复发送候选列表。include_scores 可选，默认值为 false。

### session.start 的候选前缀预填充

session.start 成功后，服务端会尝试为本 session 的固定候选 system prompt 做一次
action-score 前缀预填充，并在 `session.started.action_prefix_prefilled` 返回结果：

- hierarchical：预填充包含全部类别的固定 system prompt；具体 child system prompt
  依赖本轮选中的类别，在第一次进入该类别时按需建立缓存；
- flat_children：预填充包含拍平后的全部 children，整个 session 只执行一个具体动作
  评分阶段，不建立类别阶段；
- 预填充请求不包含用户音频、图片、历史对话或当前数字人状态，只用于建立候选
  catalog 的 KV 前缀；预填充分数不会写入历史，也不会作为动作结果返回；
- 预填充失败不会阻断 session.start，`action_prefix_prefilled` 为 false，后续真实
  turn 仍会正常评分。

动作评分请求的 prompt 顺序固定为：

    固定候选 system prompt
    + 已完成的历史对话和历史媒体
    + 当前 turn 的音频、图片、文本和数字人状态
    + 当前 turn 的 action-score 指令
    + 短 candidate_id suffix

因此固定候选前缀可以跨 turn 复用。历史对话只有在它位于当前 turn 之前时才进入可复用
边界；当前 turn 的音频、图片、文本和状态不会被下一个 turn 错误复用。服务端会根据
实际渲染后的多模态 token 计算边界，而不是按字符数估算。
生产客户端通常保持默认值，只接收实际要执行的动作和服务端耗时；测试、问题
排查或离线分析时可以在 session.start 中设置 "include_scores": true，让每个
turn.result 额外返回完整候选排序和媒体摘要。一般候选下该开关只影响返回内容；
category 仅有一个 child 时，默认关闭诊断会启用直接返回快路，开启诊断则保留
child 评分以产生真实分数。所有实际执行的评分阶段都会记录 prompt 和诊断日志。

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

默认 hierarchical 模式下，一个外部 turn 内部按以下顺序执行两阶段，外部不会收到中间事件，也不需要创建第二个 session：

1. 第一阶段固定使用 session 的类别 system prompt，只对所有 `category_id` 的短 suffix 评分，选出一个类别。
2. 第二阶段使用选中类别的说明与 children，并确保全局 `no_action` 候选始终在
   Child 候选集合内；如果选中类别本身不含该候选，服务端自动追加。随后只对
   `candidate_id` 评分，选出最终动作。
3. 服务端只返回一个 `turn.result`；其中 `action` 始终返回最终动作。只有 include_scores=true 时才额外返回第二阶段的 `scores`，`media_summary.action_context` 额外包含 `selected_category_id`、`category_scores`、`category_compute_ms`、`child_compute_ms` 和 `selection_stages=2`。

阶段请求使用同一个 `logical_request_id`，诊断日志中的 `stage` 分别为 `category` 和 `child`。第二阶段 system prompt 是当前 turn 内部临时构造的，不会追加到 session 历史，因此多轮对话不会累积多个 system prompt。

实现层会执行两个内部 action-score pipeline pass；这不是两个 session，也不是两个外部 turn。两阶段传入同一份媒体和历史输入，媒体 encoder 是否命中内容缓存由服务端缓存层决定。

## Turn 来源与文本角色

`turn.start` 和 `turn.commit` 都必须携带相同的 `turn_origin` 和
`text_role`。只接受以下组合：

| turn_origin | text_role | text 语义 |
|---|---|---|
| `user` | `user_input` | 用户当前输入 |
| `proactive` | `character_reply` | 数字人已经准备好、即将播放的文本 |

组合不合法或 start/commit 不一致时返回
`invalid_request / invalid_turn_semantics`，不会静默纠正。

### 用户触发动作

用户按住空格期间：

    {
      "type": "turn.start",
      "turn_id": "turn-003",
      "turn_origin": "user",
      "text_role": "user_input"
    }

用户 Turn 可以追加 Base64 PCM16 音频和图片帧。音频事件示例：

    {
      "type": "input_audio.append",
      "turn_id": "turn-003",
      "seq": 1,
      "audio": "<Base64 PCM16>"
    }

也兼容事件名 `input_audio_buffer.append`。图片事件使用
`input_image.append`，并通过 `image_role` 明确图片主体：

    {
      "type": "input_image.append",
      "turn_id": "turn-003",
      "seq": 1,
      "timestamp_ms": 1710000000000,
      "image_role": "user_camera",
      "mime_type": "image/jpeg",
      "image": "<Base64 JPEG>"
    }

`image_role=user_camera` 表示用户摄像头画面，用于理解用户及其环境；
`image_role=avatar_state` 表示数字人最新视频帧，用于理解数字人自身当前可视姿态和行为。
两类图片可以出现在同一个 Turn，服务端会在排序后逐张绑定角色，并把角色随图片
记录到 Session。客户端应始终显式传入该字段，尤其不能只依靠
`turn_origin` 推断混合图片的来源。

为兼容旧客户端，字段省略时，user Turn 默认 `user_camera`，proactive Turn 默认
`avatar_state`。这只是兼容默认值，不适用于一个 Turn 中同时传入两类图片。

数字人状态图片只在其所属的当前 Turn 中有效。历史 Turn 的 `avatar_state` 图片不会
进入后续动作评分上下文，不能作为数字人当前姿态的证据。当前 Turn 收到数字人状态
图片时，只使用时间最新的一张表示数字人当前可视姿态和行为；结构化
`pose/gaze/hands` 等视觉字段不再注入，但仍保留本轮显式提供的
`current_action_id`，以及主动 Turn 的 `state_description`。没有数字人状态图片时，
优先使用本轮或 Session 继承的结构化视觉状态；两者都没有时明确标记当前状态未知。
状态未知不会单独排除不依赖特定起始姿态的候选。任何情况下都禁止从历史图片或用户
摄像头图片推断数字人当前状态。

收到 `input_image.append` 后，服务端会在后台提前完成压缩图片解码和 RGB 转换，
客户端协议不变。预处理结果以内部二进制信封传入正式多模态 processor；原始 data URI
仍用于 Session 历史和重放，因此不会改变图片顺序、角色、像素或动作评分语义。
每个 Turn 最多保留 8 个预处理任务、并发数为 2，RGB 结果单帧上限 32 MiB、Turn
总上限 64 MiB。图片无法解析、超过内存限制或没有被提前调度时，会自动回退到原始
data URI，由原有 commit 后链路处理。

该优化只把图片解码/RGB 转换移到用户仍在输入的时间段，不提前运行 Image Encoder，
也不改变 Flat/Hierarchical、Thinker 次数或候选评分方式。`turn.cancel`、连接断开和
`session.close` 会取消尚未完成的图片预处理任务。

提交示例：

    {
      "type": "turn.commit",
      "turn_id": "turn-003",
      "turn_origin": "user",
      "text_role": "user_input",
      "text": "你可以挥挥手吗？",
      "avatar_state": {
        "current_action_id": null,
        "pose": "seated"
      }
    }

`current_action_id` 可用于 user 和 proactive Turn；`state_description` 只用于描述
当前 proactive Turn。
为兼容现有接入，`pose`、`gaze`、`hands`、`conversation_phase`
及其他旧字段仍会保留；没有数字人状态图片时，它们作为结构化视觉状态进入模型上下文，
有数字人状态图片时则由最新图片替代。本轮显式提供的 `current_action_id` 仍用于动作衔接，
proactive Turn 显式提供的 `state_description` 仍用于约束本次推理。

### 系统主动动作

主动文本由外部 Qwen Omni 基于同一 session 历史生成后提交。它是数字人待播文本，
不是用户输入，也不是用户动作请求：

    {
      "type": "turn.start",
      "turn_id": "turn-proactive-001",
      "turn_origin": "proactive",
      "text_role": "character_reply",
      "trigger": "user_returned"
    }

    {
      "type": "turn.commit",
      "turn_id": "turn-proactive-001",
      "turn_origin": "proactive",
      "text_role": "character_reply",
      "trigger": "user_returned",
      "text": "Hello，你回来啦！",
      "user_input": null,
      "avatar_state": {
        "current_action_id": null,
        "state_description": "用户刚刚回来，动作应轻量友好，并避免重复最近动作。"
      }
    }

proactive Turn 的待播文本可选；`trigger` 可选，但 start 和 commit 必须一致。
提供待播文本时，服务根据其语义、语气、表达目标、历史对话、历史动作和
avatar_state 选择伴随动作。例如“Hello，你回来啦！”应优先匹配“挥手欢迎”等
候选，而不是因为文本没有直接动作指令就选择 `no_action`。服务不会基于该文本
生成新回复。

主动文本通常可以直接在 `turn.commit.text` 中提交，也可以先通过
`turn.text.update` 写入。若两处都提供，以 commit 中的 `text` 为最终文本。
只有本轮实际提供非空文本时，服务才新增一条本轮 assistant 消息，并在动作 Prompt 中
将它明确标记为待播文本；该规则不是所有 proactive Turn 的固定假设。
未提供文本或文本为空时，服务不会创建本轮 assistant 待播消息，也不会把上一条
assistant 历史误认为本轮待播文本；动作选择只使用 `avatar_state`、`trigger`、历史和媒体。
proactive Turn 即使通常没有本轮用户音频，也仍会读取同一 Session 中已经完成的
历史对话、历史媒体、历史动作和当前 `avatar_state`；调用方不需要在待播文本中
重复拼接这些历史。

两类 Turn 都只计算动作。用户输入以 user 角色写入历史；主动待播文本以 assistant
角色写入历史；实际动作以 `[action_state]` 一并记录，供后续 Turn 处理指代、
冲突和重复。

### avatar_state 兼容约定

`avatar_state` 是可选对象，推荐结构如下：

    {
      "current_action_id": "wave",
      "state_description": "用户刚刚回来；数字人准备友好欢迎，并避免重复最近动作。"
    }

- `current_action_id` 可以是非空字符串或 `null`，表示上一次已经执行的动作；
- `state_description` 可以是字符串，用自然语言解释当前主动场景，并给出本次动作
  推理的目标、指引、要求和禁止项；
- `pose`、`gaze`、`hands`、`conversation_phase` 以及调用方已有的其他字段保持兼容；
  没有数字人状态图片时作为结构化视觉状态使用，有图片时由最新图片替代；
- `avatar_state` 不用于扩展候选集合，模型最终仍只能选择 Session 已冻结的动作。

### `text`、`current_action_id` 和 `state_description` 的作用

这三个字段共同描述“本轮要表达什么”和“数字人当前允许怎样表达”，但职责不同：

| 字段 | 作用 | 是否必需 | 是否写入后续上下文 |
|---|---|---|---|
| `text` | 本轮需要理解的语言内容；角色由 `turn_origin/text_role` 决定 | user、proactive 均可选 | 成功后按 user 或 assistant 角色写入历史 |
| `current_action_id` | 上一次已经执行的动作，用于判断衔接和无意义重复 | 可选，可显式传 `null` | 不继承；省略时根据最近一条已执行的 `[action_state]` 判断 |
| `state_description` | 解释本次主动场景，并提供动作目标、指引、要求和禁止项 | proactive Turn 可选 | 不继承，仅当前 proactive Turn 使用 |

`text` 的具体语义：

- user Turn：表示用户当前说的话或输入的文字，与本轮音频、图片一起用于理解用户
  意图和动作请求；可以为空。成功后作为 user 历史保存；
- proactive Turn：表示数字人已经生成、即将播放的文本。提供时，服务根据其语义、
  语气和表达目标选择伴随动作，不把它当成用户指令，也不基于它生成新回复；
  成功后作为 assistant 历史保存；未提供时不创建本轮待播文本消息；
- 可以先用 `turn.text.update` 更新，`turn.commit.text` 如果存在则覆盖此前文本。

`current_action_id` 的具体语义：

- 非空字符串表示上一次已经执行的真实动作 ID，`null` 表示没有可报告的上一次动作；
- 模型用它判断下一动作能否自然衔接，并在没有明确重复要求时
  避免再次选择同一动作；
- 它只是上下文约束，不会强制服务返回该动作，也不会把不在 Session 候选目录中的
  ID 加入候选集合。

`state_description` 的具体语义：

- 它是不做结构化解析的当前主动场景说明，例如“用户刚回来”“本轮需要友好欢迎”
  “动作应轻量”“不要重复挥手”；
- proactive Turn 中，`text` 与 `state_description` 是共同的动作选择约束：
  动作既要匹配待播文本，也要满足状态中的目标和禁止项；没有同时满足两者的候选时
  应选择 `no_action`；
- user Turn 不使用 `state_description`。

状态继承规则：

- `pose/gaze/hands` 等可复用结构化状态可以在成功 Turn 后成为 Session 最近状态；
- `current_action_id` 和 `state_description` 都是当前 Turn 字段，不跨 Turn 继承；
- 当前 Turn 有数字人图片时，图片替代结构化视觉状态，并清除之前缓存的结构化视觉
  状态，避免下一 Turn 使用已经过期的姿态；
- 没有数字人图片且省略结构化视觉状态时，可以复用最近一次成功 Turn 的视觉状态；
- 取消或评分失败的 Turn 不更新最近状态；
- 这些字段属于当前 Turn 的动态上下文，不会改变固定候选目录或其 prefix cache identity。

## 方案 B 动作评分

嵌套候选格式下，类别层和 child 层分别进行 suffix 评分。下面的 `a01`/`none` 示例是旧 flat 格式的兼容说明；新接入方应使用上面的类别和 children。

候选动作定义位于 session system prompt。`source_label` 和 `short_definition` 是外部
传入的动作语义元数据，服务端按原文渲染，不压缩、不去重、不改写；外部调用方可以在
session.start 之前自行优化描述。类别描述的实际文本会参与 `action_catalog_hash` 和
prefix cache identity，因此同一 session 内必须保持不变。示例：

    candidate_id=a01｜动作=左手挥手｜说明=使用左手抬起并左右摆动
    candidate_id=none｜动作=不做动作｜说明=保持当前姿态

`action_id` 只用于服务端结果映射和动作执行，不再混入候选说明或模型输出目标；
模型在类别阶段只选择 `category_id`，在 Flat/Child 阶段只选择 `candidate_id`。
因此 `candidate_id`、`action_id` 与 `no_action` 不会以不同格式同时出现在选择指令中。

hierarchical 两次计算的固定部分如下；中间的历史、当前状态、媒体和 Turn 指令按本轮
动态补入：

    # Category / Child 共用的当前 Turn 动态指令
    ...
    没有候选满足全部约束时，选择本阶段定义的兜底项。

动态指令不写具体 `category_id` 或 `candidate_id`。准确的兜底 ID 只出现在对应阶段的
固定 system prompt 中，避免 Category 阶段被 Child 的 `candidate_id` 干扰。

    # Category system prompt
    你是数字人动作类别识别器。请从固定类别集合中选择一个 category_id。
    <明确时间顺序和优先级的历史动作规则>
    category_id=B1｜类别=基础姿态｜说明=站立、坐下等姿态变化
    ...
    没有候选动作满足输入与状态约束时，选择兜底 category_id=B0。

    # Category 当前 Turn 结尾
    最合适的 category_id：

    # Child system prompt
    你是数字人动作识别器。请从以下集合中选择一个 candidate_id。
    <明确时间顺序和优先级的历史动作规则>
    已选类别：category_id=B1｜类别=基础姿态｜说明=站立、坐下等姿态变化
    candidate_id=A1｜动作=正式站立｜说明=双脚并拢，脊背挺直双臂自然垂放
    candidate_id=A0｜动作=不做动作｜说明=保持当前姿态
    没有候选动作满足输入与状态约束，或需要避免冲突、重复时，选择兜底 candidate_id=A0。

    # Child 当前 Turn 结尾
    最合适的 candidate_id：

Category 和 Child 使用逐字相同的当前 Turn 动态指令，只在结尾分别追加
`category_id` 或 `candidate_id` 输出槽。静态 Category system catalog 继续使用
`hierarchical:<action_catalog_hash>` KV namespace，静态 Child catalog 继续使用
`hierarchical:<action_catalog_hash>:child:<selected_category_ids>` namespace；本次动态
Prompt 调整不进入静态 catalog KV 边界，因此不会使已预填充的固定目录前缀失效。
两阶段相同的 `action_context_cache_key` 继续复用当前 Turn 已解码的图片和音频；该缓存
复用的是多媒体预处理结果，不是跨不同 system prompt 的文本 KV。

当前 turn 的 suffix 只使用短 candidate_id：

    完整 prompt + a01
    完整 prompt + none

第一阶段对 category_id 排名；第二阶段对选中类别的 children 加全局 `no_action`
兜底候选的 candidate_id 排名。第一阶段完整排序保存在
`media_summary.action_context.category_scores`，最终 Child 排序保存在顶层
`scores`，再映射成真正的 action_id。短 ID 优先设计成单 token，但实际以
tokenizer 的 token_count 为准；PPL 分母包含实际评分的 suffix token（包括显式终止
token）。

对候选 token 的计算为：

    mean_logprob = sum(token_logprob) / token_count
    mean_nll = -mean_logprob
    ppl = exp(mean_nll)

## 返回示例

    {
      "type": "turn.result",
      "session_id": "session-001",
      "turn_id": "turn-003",
      "action_catalog_hash": "sha256:abcd1234",
      "action": {
        "candidate_id": "A1",
        "category_id": "B1",
        "action_id": "A1",
        "execute": true
      },
      "scores": [
        {
          "candidate_id": "A1",
          "category_id": "B1",
          "action_id": "A1",
          "mean_logprob": -0.35,
          "mean_nll": 0.35,
          "ppl": 1.42,
          "token_count": 1,
          "token_scores": []
        }
      ],
      "media_summary": {
        "audio_chunk_count": 24,
        "image_frame_count": 6,
        "user_camera_image_count": 5,
        "avatar_state_image_count": 1,
        "scored_image_count": 6,
        "text_present": false,
        "action_context": {
          "selection_stages": 2,
          "logical_request_id": "session-session-001-turn-turn-003-action-...",
          "selected_category_id": "B1",
          "selected_category_ids": ["B1"],
          "category_top_k": 1,
          "category_compute_ms": 310.2,
          "child_compute_ms": 532.1,
          "category_scores": [{"candidate_id":"B1", "mean_logprob":-0.21, "ppl":1.23, "token_count":1, "token_scores":[]}]
        }
      },
      "timing": {
        "server_turn_ingest_ms": 5280.4,
        "server_action_compute_ms": 842.317,
        "server_result_finalize_ms": 0.8,
        "server_total_after_commit_ms": 843.2,
        "action_breakdown": {
          "selection_mode": "hierarchical",
          "category": {"client": {}, "pipeline": {}, "scheduler": {}, "suffix": {}},
          "child": {"client": {}, "pipeline": {}, "scheduler": {}, "suffix": {}},
          "child_catalog_prefill_ms": 0.0
        }
      }
    }

server_action_compute_ms 只表示服务端动作评分耗时。客户端应额外测量从
发送 turn.commit 到收到 turn.result 的往返耗时；两者差值是非计算耗时
估算，不等同于纯网络耗时。

当 `include_scores=true` 时，`media_summary.action_context` 仍提供类别/子动作
阶段总耗时。`timing.action_breakdown` 无论是否开启 `include_scores` 都会返回，
并按 category/child 或 single 暴露以下真实阶段耗时：

- `timing.image_preprocessing.scheduled_count/prepared_count`：本 Turn 调度和成功复用的图片数；
- `timing.image_preprocessing.fallback_count/not_scheduled_count`：回退原始图片和未提前调度的图片数；
- `timing.image_preprocessing.worker_total_ms`：各图片后台 worker 耗时之和，任务并行时不能视为墙钟耗时；
- `timing.image_preprocessing.commit_wait_ms`：收到 commit 后等待尚未完成图片任务的墙钟耗时；
- `timing.image_preprocessing.prepared_bytes/statuses`：实际复用的 RGB 字节数和逐图片状态；
- `client_request_build_ms`：客户端构造 action-score 请求的耗时；
- `action_slot_wait_ms`：等待进程内 action-score 串行槽位的耗时；
- `coordinator_pipeline_ms`：Coordinator 从提交到返回结果的墙钟耗时；
- `preprocessing_ms`、`image_encoder_ms`、`audio_encoder_ms`、`mm_aggregate_ms`：多模态各阶段的实际墙钟耗时；
- `server_request_build_ms`：服务端从收到请求到构造出 prefix 请求的耗时；
- `scheduler_admission_ms`：prefix 请求构造完成后，到进入 scheduler 等待队列前的准入耗时；
- `scheduler_wait_ms` / `queue_wait_ms`：prefix 和各 suffix batch 等待 scheduler 的累计耗时；
- `prefix_prefill_ms`：共享动态多模态 prefix 真正开始执行到完成的耗时；
- `suffix_batch_queue_wait_ms`：每个 suffix batch 被 scheduler 选中前的等待耗时；
- `suffix_batch_ms`：每个 suffix batch 从入队到完成的总耗时；
- `client_result_processing_ms`：结果转换与校验耗时；
- `server_result_finalize_ms`：动作完成后构造历史和 `turn.result` 的耗时。

`queue_wait_ms` 是兼容字段，新的排查应优先使用上述拆分字段。

## 完整诊断日志

动作评分链路会将诊断记录追加到 JSONL 文件。默认路径为 `/tmp/sglang-omni-action-debug.jsonl`，也可以通过环境变量 `SGLANG_OMNI_ACTION_DEBUG_LOG_FILE` 指定路径。记录包含 `session_id`、带 `turn_id` 的 request_id、候选列表、system prompt、逻辑 messages、实际渲染后的 `full_prompt`、prompt token 数、阶段耗时、异常堆栈以及音频/图片数量、大小和 hash。原始媒体 Base64 不写入日志。

JSON 序列化、目录创建、跨进程文件锁和磁盘写入均在后台线程完成；请求线程只做
非阻塞入队。队列有界，过载时丢弃新诊断记录并输出累计 dropped_records 告警，
避免慢盘反向阻塞动作推理。

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
| turn.start | 客户端 -> 服务端 | 开始一个新 turn；必须声明 `turn_origin` 和 `text_role` |
| turn.started | 服务端 -> 客户端 | 确认 turn 已创建 |
| input_audio.append | 客户端 -> 服务端 | 追加 PCM16 音频 chunk |
| input_image.append | 客户端 -> 服务端 | 追加图片帧；用 `image_role` 区分用户摄像头与数字人状态 |
| input.ack | 服务端 -> 客户端 | 确认媒体序号已接收；图片 ACK 回显最终 `image_role` |
| turn.text.update | 客户端 -> 服务端 | 更新本轮文本；语义由 start 中的 `turn_origin` 决定 |
| turn.commit | 客户端 -> 服务端 | 回传相同 Turn 语义字段，结束输入并触发动作计算 |
| turn.committed | 服务端 -> 客户端 | 确认开始处理 |
| turn.result | 服务端 -> 客户端 | 返回动作和耗时；可选返回评分详情 |
| turn.cancel | 客户端 -> 服务端 | 取消 collecting 或 processing 状态的当前 turn |
| turn.cancelled | 服务端 -> 客户端 | 确认已停止后续处理；processing Turn 已触发底层 abort |
| error | 服务端 -> 客户端 | 返回协议或处理错误 |
| session.close | 客户端 -> 服务端 | 主动关闭 session |

第一条有效事件必须是 session.start；一个 session 同时只能有一个 active turn。
WebSocket 只使用 JSON 文本帧，媒体使用 Base64，不接受二进制帧。服务端仍兼容
input_audio_buffer.append，其字段与 input_audio.append 相同。

### Turn 字段约束

| turn_origin | text_role | `text` 的解释 | `trigger` | `user_input` |
|---|---|---|---|---|
| `user` | `user_input` | 用户本轮说的话或输入的文本 | 不允许 | 建议省略 |
| `proactive` | `character_reply` | 数字人已经生成、即将播放的文本 | 可选，start/commit 必须一致 | 必须为 `null` 或省略 |

`user_input` 仅是主动场景的兼容字段。服务端始终以 `turn_origin` 判断本轮来源，
不会根据 `user_input` 是否为空推断文本角色。

字段校验规则：

- start 和 commit 的 `turn_origin`、`text_role` 必须完全一致；`trigger` 去除
  首尾空白后的值必须一致；
- `trigger` 如果存在，必须是去除首尾空白后仍非空的字符串，并且只允许 proactive Turn；
- proactive 和 user Turn 的 `text` 均可为空；提供时必须是字符串，可以来自最近一次
  `turn.text.update`，也可以由 `turn.commit.text` 提供；
- 所有文本和媒体只用于动作选择，不会触发回复生成。

### Turn 生命周期与错误恢复

一个成功 Turn 的最短时序为：

    turn.start -> turn.started -> turn.commit -> turn.committed -> turn.result

接入方应按以下规则处理失败：

- `turn.start` 的语义字段非法时，Turn 尚未创建；修正后可以重新发送该 `turn_id`；
- `turn.commit` 在发送 `turn.committed` 前因语义字段、文本或 `avatar_state` 校验失败时，
  当前 Turn 仍处于 active 状态；可以修正后重新 commit，或发送 `turn.cancel`；
- 收到 `turn.committed` 后仍可发送 `turn.cancel`。服务端会停止后续评分阶段、
  abort 当前物理评分请求，并且不发送 `turn.result`、不写入动作历史；
- 发送 `turn.cancel` 后必须等待对应的 `turn.cancelled`，再发送下一个
  `turn.start`。取消清理期间旧 Turn 仍是 active turn；
- `turn.commit` 后输入已被冻结，不能继续追加音频或图片、更新文本或重复 commit；
- result 和 cancel 发生竞态时只有一个终态：先进入 completed 则返回 result，
  先进入 cancelling 则返回 cancelled；
- 收到 `turn.committed` 后若返回动作评分错误，该 Turn 已结束；应使用新的 `turn_id`
  创建下一 Turn；
- 只要某个 `turn.start` 已成功，同一 Session 内该 `turn_id` 就不能再次使用，
  即使该 Turn 后来被 cancel；
- 连接断开后当前 Session 历史会被释放。需要连续历史时，应保持同一 WebSocket
  Session，不要为每个 proactive Turn 新建连接。

### 序号与媒体规则

音频格式固定为原始 PCM16 little-endian、单声道、16000 Hz，不带 WAV 头。
音频 seq 从 1 开始，必须按 1、2、3... 递增到达。重复 seq 是幂等重试，
服务端不重复追加并以 input.ack 的 duplicate=true 确认。跳号或乱序返回
invalid_sequence。图片 seq 与音频 seq 独立；图片允许乱序，commit 时按
timestamp_ms 升序、再按 seq 升序排序。图片支持 image/jpeg、image/png、
image/webp，image 可以是 Base64 内容或 Base64 data URI。图片角色与图片一起排序，
不会因时间戳重排而错位。当前评分 Prompt 在所有当前图片之前只放一条压缩后的角色
说明，例如六张连续用户图片写为 `用户摄像头图片=1-6`，不再为每张图重复角色文本。
若同一 Turn 有多张 `avatar_state` 图片，动作评分只保留时间最新的一张，数字人的当前
可视状态完全以它为准，不再同时注入结构化视觉字段；本轮显式提供的
`current_action_id` 和 proactive Turn 的 `state_description` 仍保留。该图即使早于
很多用户摄像头图片，也不会被当前图片数量上限挤掉。没有数字人状态图片且存在可用
结构化状态时才注入结构化视觉字段；两者都没有时将当前状态标记为未知。
进入历史后，
历史用户摄像头图片会保留对应角色说明，历史数字人状态图片则从后续评分上下文中
移除，避免模型把过期数字人姿态当作当前状态。

turn.commit 后不能继续追加媒体、更新文本或重复 commit。turn.cancel 同时支持
尚未 commit 的 collecting Turn 和正在评分的 processing Turn。processing Turn
取消时会中止当前 Flat、category、child 或 child catalog prefill 请求，并阻止
后续评分阶段启动。WebSocket 断开或收到 session.close 时也会执行相同的后台
任务取消和底层 abort 清理，然后释放当前进程内的 session 状态。

底层 abort 是广播式取消；已经进入 GPU 执行的单个 kernel 可能要到可中断边界
才会停止，但排队请求、后续候选批次和后续层级阶段不会继续执行。

### turn.result 和计时

当前动作服务只执行动作推理，不生成数字人回复文本，因此默认 turn.result 使用精简结构：

    {
      "type": "turn.result",
      "session_id": "session-001",
      "turn_id": "turn-003",
      "action_catalog_hash": "sha256:abcd1234",
      "action": {
        "action_id": "A124",
        "candidate_id": "A124",
        "category_id": "B027",
        "execute": true
      },
      "timing": {
        "server_turn_ingest_ms": 1198.610,
        "server_action_compute_ms": 5511.866,
        "server_total_after_commit_ms": 5512.727
      }
    }

客户端应直接读取 turn.result.action，不需要从 scores 重新排序：

| 字段 | 是否返回 | 含义 | 客户端使用方式 |
|---|---|---|---|
| action_id | 必返 | 本轮最终选中的真实动作 ID，由候选中的 action_id 映射得到 | 当 execute=true 时，使用它查找或执行数字人动作 |
| candidate_id | 必返 | 本轮命中的候选 ID，是 session.start 中具体候选的唯一标识 | 用于记录、链路追踪和动作目录查找；不要根据它重新推断动作 |
| category_id | 嵌套候选时返回 | 候选所属的父类别 ID | 只用于分类、审计和动作目录定位，不参与动作执行；flat 候选没有该字段 |
| execute | 必返 | 服务端是否要求执行动作 | true 才执行；false 时必须跳过动画执行，通常对应 action_id=no_action |
| execution_binding | 有配置时返回 | 候选携带的执行绑定参数 | 原样传给动作播放器或执行器，不要由客户端重新构造 |

推荐处理逻辑：

    action = event["action"]
    if action["execute"]:
        executor.run(
            action_id=action["action_id"],
            candidate_id=action["candidate_id"],
            category_id=action.get("category_id"),
            execution_binding=action.get("execution_binding", {}),
        )
    else:
        # no_action：保持当前姿态，不调用动作播放器
        pass

字段之间的职责不同：action_id 决定执行什么动作，candidate_id 决定命中了
哪个候选，category_id 表示候选所属类别，execute 决定是否真正执行。客户端不应
仅因为 action_id 不是 no_action 就执行，而应以 execute 为最终开关。

对于 no_action，返回示例为：

    {
      "action_id": "no_action",
      "candidate_id": "A0",
      "execute": false
    }

默认不会返回 `reply`、`scores`、`media_summary`、`score_count` 或每个候选的
PPL/logprob。客户端不应从候选分数重新排序，直接使用 action 即可。

如果 session.start 设置 "include_scores": true，服务端会在同一个 turn.result
中额外返回：

- scores：最终评分阶段的完整候选排序；hierarchical 是选中类别下的
  children，flat_children 是全部具体动作；
- media_summary：当前 turn 的音频、图片、文本统计，以及 action_context；
- 每个候选的 mean_logprob、ppl、token_count 和 token_scores。

include_scores 不影响最终动作选择结果。对于 category 只有一个 child 的情况，
`include_scores=false` 会直接返回该 child，并跳过 child catalog prefill 与 child 模型评分；
`include_scores=true` 会保留 child 评分，以返回真实 PPL/logprob。快速路径只记录
category 阶段诊断日志，并在 `timing.action_breakdown.child` 标记 skipped。

timing 字段含义：

- server_turn_ingest_ms：turn.start 到收到 commit，包含当前 turn 媒体接收；
- server_action_compute_ms：服务端动作评分耗时；
- action_breakdown：category/child 或 single 的客户端、流水线、scheduler、suffix 分段耗时；
- server_result_finalize_ms：动作完成后构造历史和结果帧的耗时；
- server_total_after_commit_ms：收到 commit 到 turn.result，包含动作评分及其服务端编排，
  不包含普通数字人回复生成，因为当前动作服务不生成回复。

外部服务应另外测量 commit 到 turn.result 的端到端时间。该时间与
server_total_after_commit_ms 的差值只能作为排队和传输开销估计。

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
| invalid_turn_semantics | turn_origin/text_role 非法、start/commit 不一致，或 proactive 的 user_input 非空 |
| invalid_sequence | 音频 seq 缺失、跳号或乱序 |
| invalid_audio | PCM16、Base64 或音频参数非法 |
| invalid_image | 图片 Base64、类型或大小非法 |
| turn_already_committed | turn 已提交、已取消或不存在 |
| session_candidate_invalid | 候选缺失、重复、超限或缺少 no_action |
| action_score_failed | 当前 turn 动作评分或模型预处理失败，session 仍可继续创建下一 turn |
| action_score_logprob_unavailable | 当前 turn 无法取得 prefix 首 token 的有效 logprob，未伪造分数，session 仍可继续使用 |
| server_error | 非 turn 级的服务端异常 |

协议解析还可能返回 invalid_json、invalid_event 或 binary_frames_not_supported。
动作评分失败会在已经发送 turn.committed 后返回 action_score_error 类型的 error；
服务端会清理 active turn，不会把该异常升级为整个 session 的 server_error。
当 prefix selected-token logprob 字段缺失时，服务端依次尝试 SGLang 的
next-token selected logprob 或 next-token logits 计算候选首 token 概率；两者都不可用
时返回 action_score_logprob_unavailable，绝不使用虚假的默认分数。

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

每次 action-score 完成后，action_scoring_complete 诊断事件和主日志中的 stats.gpu
会记录实际执行评分的 Thinker GPU 资源，字段包括：

- device_id、device_name、tp_rank、tp_size；
- start/end 的进程显存 allocated、reserved 和 GPU 总显存 free/used；
- 本次评分期间的 process_max_memory_allocated/reserved；
- process_allocated_delta_bytes；
- NVML 采样的 gpu_utilization_percent、nvml_memory_utilization_percent、
  power_usage_watts 和 temperature_c。

其中进程显存是当前 Thinker/ActionScore 进程的显存，NVML used/free 是整张物理
GPU 的显存。start/end 是动作评分前后的采样，峰值显存统计从本次评分开始时重置。
这些数据同时写入 SGLANG_OMNI_ACTION_DEBUG_LOG_FILE 指向的 JSONL，不会返回给
生产客户端；include_scores 也不会影响 GPU 统计日志。

当前 thinker 和 preprocessing 部署上下文为 60000 tokens。为避免多轮媒体历史无限增长，动作评分
使用独立的有界上下文：最多保留最近 4 个历史 turn、4 个历史音频、8 个历史
图片和当前 turn 最近 8 张图片。每个已完成 turn 的 assistant 历史会保存一条
[action_state]，包含 turn_id、candidate_id、action_id、动作名称和动作描述；
最近一轮的动作指代可以直接依赖该记录，超过有界历史的旧动作记录可能被裁剪。
turn.result.media_summary.action_context 会返回实际保留数量以及是否发生裁剪。
