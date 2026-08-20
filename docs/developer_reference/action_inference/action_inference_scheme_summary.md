# Qwen3-Omni 多模态动作推理方案总览

本文总结当前 `sglang-omni` 内的数字人动作推理方案，供服务接入、调试和后续优化使用。
相关协议细节见 [multimodal_session_realtime.md](multimodal_session_realtime.md)，评分算法见
[action_suffix_scoring.md](action_suffix_scoring.md)，变更时间线见
[action_inference_change_history.md](action_inference_change_history.md)。

## 1. 目标和边界

服务在一个进程内维护一个多模态 session。外部服务通过 WebSocket 发送用户输入，或提交已经由
外部模型生成的数字人待播文本；服务只为当前 turn 选择动作，不生成普通数字人文本回复。

每个 turn 必须显式声明来源和文本角色：

| `turn_origin` | `text_role` | 当前 `text` 的语义 |
| --- | --- | --- |
| `user` | `user_input` | 用户当前输入，可与音频、图片一起提交 |
| `proactive` | `character_reply` | 数字人已经生成、即将播放的文本 |

主动 Turn 的文本可选，并可携带可选 `trigger` 说明触发原因。Turn 语义字段必须在
`turn.start` 和 `turn.commit` 中保持一致。

每次成功推理都会把本轮用户输入或主动待播文本与实际动作写入 session 内存历史，供后续 turn 理解：

```text
用户：请挥手。
服务动作：[action_state] action_id=A002；本轮数字人已执行动作
用户：再重复一遍刚刚的动作。
服务动作：A002
```

服务不做 session 持久化；断开连接后释放 session。本文不涉及外部动作播放器、骨骼绑定和动画渲染。

## 2. 候选目录和选择模式

`session.start` 初始化候选目录。候选目录在整个 session 内固定，并计算
`action_catalog_hash` 作为版本标识。每个具体动作至少包含：

- `candidate_id`：模型评分和返回排序使用的候选 ID；
- `action_id`：动作执行器使用的真实动作 ID；
- `source_label`：动作名称；
- `short_definition`：动作语义描述；
- `execution_binding`：可选的左右手、方向等执行参数；
- `category_id`：嵌套候选中的所属类别。

候选目录必须包含 `action_id=no_action`。它与其他动作一样参与评分；如果被选中，返回
`execute=false`，客户端保持当前姿态。

支持两种模式：

| 模式 | 固定候选内容 | 每个 turn 的计算 | 是否有类别阶段 |
| --- | --- | --- | --- |
| `hierarchical` | 所有一级类别及其描述 | 先评分类别，再评分选中类别的 children | 有，两阶段 |
| `flat_children` | 所有具体 children 及其描述 | 直接评分所有具体动作 | 无，一阶段 |

模式优先级为：

1. `session.start.selection_mode`；
2. 环境变量 `SGLANG_OMNI_ACTION_SELECTION_MODE`；
3. 默认值 `hierarchical`。

`session.start` 的 `selection_mode` 非法时，session 仍可正常启动，该值被忽略并继续使用环境变量
或默认值。环境变量只控制未被 session 覆盖的默认模式。

`hierarchical` 默认只选择类别 Top-1。可以通过 `SGLANG_OMNI_ACTION_CATEGORY_TOP_K` 设置为
1–3，在类别边界模糊时同时取多个类别的 children；Top-K 越大，准确率兜底能力越强，但第二阶段
候选数和耗时也会增加。 `flat_children` 不做类别评分，所有 children（包括 `no_action`）直接进入
同一次具体动作评分。

## 3. Prompt 结构和 KV cache

当前使用方案 B：候选语义定义放在固定 system prompt 中，模型真正评分的 suffix 只使用短
`candidate_id`。候选的 logit、token logprob、mean logprob、NLL 和 PPL 计算规则不变，变化点是
不再为每个候选 suffix 重复拼接长动作描述。

用户 Turn 的逻辑 prompt 顺序为：

```text
system: 固定动作候选目录
历史 user/assistant: 历史输入和服务端记录的 [action_state]
当前 user: 本轮音频、图片、文本、数字人状态和动作选择指令
```

主动 Turn 不会把待播文本伪装成用户输入，其逻辑顺序为：

```text
system: 固定动作候选目录
历史 user/assistant: 历史输入、待播文本和服务端记录的 [action_state]
当前 assistant: 本轮已经生成的待播文本
内部 user 指令: 结合待播文本、avatar_state 和历史选择伴随动作，不生成回复
```

未提供待播文本时省略“当前 assistant”消息，内部指令明确本轮不依赖当前语言文本，
只根据 `avatar_state`、`trigger`、历史和媒体选择动作。

`turn_origin`、`text_role` 和 `trigger` 同时进入 action-score metadata 和诊断信息。
它们不改变 suffix 的 logprob/PPL 公式，只决定当前文本的角色和动作选择指令。

这样候选目录在 session 内保持同一前缀，而当前 turn 的媒体和状态位于后面。`session.start` 会
根据选择模式预填充固定候选前缀：

- `hierarchical`：预填充类别候选阶段；
- `flat_children`：预填充所有具体动作的一阶段候选。

后续 turn 只在 token 完全一致且边界安全时复用固定候选目录以及已完成历史的 KV。当前 turn 的
音频、图片、数字人状态和最新动作指令不进入可复用边界，避免把上一轮动态内容错误复用到本轮。
日志中的 `prefix_cached`、`cached_prefix_token_count` 和 `candidate_prefix_recompute_tokens` 用于
确认缓存结果；理想状态是缓存命中且候选前缀重算为 0。

`hierarchical` 的第二阶段是同一个逻辑请求内的临时阶段：类别阶段完成后，服务根据 Top-K 类别
构造 children system prompt，再评分具体动作。它不创建第二个 session，也不会把第二阶段 system
prompt 追加到 session 历史。

## 4. Session 历史和媒体处理

成功的用户 Turn 会追加以下两条逻辑历史消息：

```text
user       当前 turn 的音频、图片和文本占位信息
assistant   [action_state] turn_id、candidate_id、action_id、动作名称、动作描述和执行状态
```

成功的主动 Turn 则追加一条 assistant 历史消息，把可选待播文本与本轮实际
`[action_state]` 放在一起；无待播文本时只记录 `[action_state]`。这样下一轮能够区分“用户说了什么”和“数字人刚刚说了什么”，
同时继续利用最近动作处理重复、冲突和指代。取消或评分失败的 Turn 不写入历史。

`[action_state]` 是服务端记录的实际动作，不是新的用户指令。动作 prompt 明确要求遇到“刚刚、上一轮、
再重复、这个动作”等指代时优先参考最近一条 action state。`no_action` 也会记录为“本轮未执行动作”。

动作评分最多保留最近 4 个已完成 Turn、4 个历史音频、8 个历史图片和当前 Turn
最近 8 张图片，防止上下文无限膨胀。裁剪按完整 Turn 和媒体占位符进行，保证消息中的
audio/image 数量与传给模型的媒体数组一致；这些限制不改变 Session 内候选目录。

一个 turn 可以包含多个音频 chunk 和图片帧：

- 音频：JSON 文本帧中的 Base64，16 kHz、单声道 PCM16，按 seq 顺序合并；
- 图片：JSON 文本帧中的 Base64 图片 bytes，按 timestamp/seq 排序；
- 文本：用户 Turn 和主动 Turn 均可选；主动 Turn 中表示数字人待播文本；
- 数字人状态：作为当前 turn 动态上下文，不污染固定候选前缀。推荐提供
  `current_action_id` 和 `state_description`，其他已有字段保持兼容。

`text` 与 `avatar_state` 的职责不同：

- `text` 决定本轮语言内容。user Turn 中是可选用户输入；proactive Turn 中是
  可选的数字人待播文本，提供时以 assistant 角色参与动作选择和后续历史；
- `current_action_id` 表示当前或刚结束的动作，用于动作衔接、冲突检查和避免
  无意义重复，但不会强制返回该动作或扩展候选集合；
- `state_description` 是自然语言场景约束，可描述表达目标、姿态、用户状态、
  必须满足的动作要求和禁止项；没有同时满足文本与状态约束的候选时选择 `no_action`；
- 成功 Turn 提供的非空 `avatar_state` 会成为 Session 最近状态；省略或传空对象时
  复用上次成功状态，取消或失败不会更新它。

## 5. Turn 生命周期和取消

正常时序为：

```text
turn.start -> turn.started -> turn.commit -> turn.committed -> turn.result
```

`turn.commit` 后 Turn 进入 processing，输入被冻结，但仍可以通过 `turn.cancel`
取消。服务会 abort 当前 flat、category、child 或 child catalog prefill 物理请求，
取消后台推理任务，阻止后续层级阶段启动，并在清理完成后返回 `turn.cancelled`。
被取消的 Turn 不返回 `turn.result`，也不更新动作历史和最后一次 `avatar_state`。

WebSocket 断开和 `session.close` 使用相同的后台任务取消与底层 abort 清理，但不会
额外发送 `turn.cancelled`。结果与取消发生竞态时只会产生一个 Turn 终态；客户端应等待
`turn.result`、`turn.cancelled` 或 Turn 级 `error` 后再启动下一 Turn。已成功 start 的
`turn_id` 即使随后取消也不能复用。

## 6. 返回结果和客户端使用

默认 `turn.result` 只返回动作执行所需的信息：

```json
{
  "type": "turn.result",
  "action": {
    "action_id": "A328",
    "candidate_id": "A328",
    "category_id": "B055",
    "execute": true,
    "execution_binding": {"body_side": "right"}
  },
  "timing": {
    "server_turn_ingest_ms": 120.4,
    "server_action_compute_ms": 318.6,
    "server_total_after_commit_ms": 319.1
  },
  "action_catalog_hash": "sha256:..."
}
```

客户端通常直接使用 `action`：

- `action_id`：动作执行器实际播放的动作；
- `candidate_id`：与本 session 候选目录、日志和评分结果对应的 ID；
- `category_id`：`hierarchical` 下实际选中的类别；`flat_children` 可为空或不返回；
- `execution_binding`：执行动作所需的附加参数；
- `execute=false`：不播放动作，保持当前姿态。

`server_turn_ingest_ms` 是服务接收并聚合当前 turn 输入的时间；
`server_action_compute_ms` 是动作评分耗时；
`server_total_after_commit_ms` 是从收到 `turn.commit` 到生成结果的服务端耗时。
它们都不包含客户端到服务端的网络传输时间。

只有 `session.start.include_scores=true` 时才返回完整 `scores` 排序、PPL、mean_logprob、token
scores 和媒体上下文摘要。生产动作执行通常不需要这些诊断字段。

## 7. 性能和资源优化

候选评分采用“共享 prefix + 短 ID suffix”：

1. 先构造并执行共享 prefix；
2. prefix 完成后再延迟构造 suffix；
3. 按 micro-batch 分批评分候选；
4. 聚合全部候选的 logprob/PPL 并返回 Top-1。

这样避免在 prefix 进入 scheduler 前，一次性构造所有完整候选请求。批大小由
`SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE` 控制，默认 64，允许范围 1–256；两种模式统一使用。
增大 batch 只能减少批次数，不能消除总候选计算量，并可能增加单批显存压力。

当前日志会记录：

- client/server request build；
- scheduler admission、scheduler wait 和 queue wait；
- prefix prefill；
- 每个 suffix batch 的等待和计算；
- 音频/图片预处理；
- GPU 显存起止值、峰值、利用率、功耗和温度；
- prefix cache 命中情况、候选数量、批次大小和最终动作。

服务启动时会执行与当前模式一致的 action-score 预热。预热不带业务 session、历史和媒体，不写入
业务历史，也不计入业务 turn 耗时；它只提前初始化 tokenizer、请求构造、scheduler admission、
Thinker 和 prefix cache 路径。

## 8. 上下文和部署配置

当前单 GPU 配置将 preprocessing 和 thinker 的 `max_seq_len` 设置为 60000。它是服务处理上下文
的上限，不是单个 turn 的独立长度；实际可用空间还受固定候选目录、历史、媒体展开 token、显存和
KV cache 影响。服务通过内置 `WS /v1/session/realtime` 路由提供能力，不依赖
`--enable-realtime` 开关。

## 9. 主要代码和验证入口

- Realtime session：`sglang_omni/serve/realtime/multimodal.py`
- 客户端请求和预填充：`sglang_omni/client/client.py`
- Prompt/媒体预处理：`sglang_omni/models/qwen3_omni/components/preprocessor.py`
- 候选构造：`sglang_omni/models/qwen3_omni/request_builders.py`
- 评分调度：`sglang_omni/scheduling/omni_scheduler.py`
- PPL/token score：`sglang_omni/models/qwen3_omni/action_scoring.py`
- Realtime 协议：`multimodal_session_realtime.md`
- 批量性能验证：`action_batch_benchmark.md`
- 历史耗时分析：`action_latency_history.md`
