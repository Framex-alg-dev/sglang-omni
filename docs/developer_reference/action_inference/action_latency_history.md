# 动作推理耗时演进记录

本文记录当前仓库中动作推理服务的成功请求耗时，并按日志发生时间与实现改动阶段关联，便于观察每次优化后的变化。

## 1. 统计口径

数据来源：

- 服务主日志：`logs/qwen3_omni_local.log`
- 动作调试明细：`logs/session_action_debug.jsonl`
- 统计对象：日志中出现 `action completed` 且成功返回 `turn.result` 的 turn
- 耗时口径：`action completed` 记录的服务端动作推理耗时；不是客户端从发送媒体到收到结果的端到端耗时

内部阶段耗时按以下方式理解：

- `category_total_ms`：hierarchical 模式类别阶段总耗时
- `child_total_ms`：hierarchical 模式具体动作阶段总耗时
- `queue_wait_ms`：请求进入调度器后等待 GPU 执行的时间
- `prefix_tokens` / `cached_prefix_tokens`：共享 prefix 的 token 数，以及候选 suffix 请求复用该共享 prefix 的 token 数；该字段不表示共享 prefix 本身全部来自启动预热
- `candidate_recompute_tokens`：候选计算阶段重新计算的 prefix token 数
- `batches`：候选 suffix 被拆分执行的批次

日志中的阶段统计只在较新的结构化日志中完整存在，因此早期部分请求只能记录 turn 总耗时。所有请求均使用当时服务实际运行的单 GPU 环境；不同 turn 的媒体数量、历史长度、候选数量和 GPU 调度状态并不完全一致，以下结果用于观察趋势，不应视为严格控制变量的 benchmark。

## 2. 各阶段汇总

| 阶段 | 时间范围 | 模式与主要实现 | 样本数 | 平均耗时 ms | 范围 ms | 主要结论 |
|---|---|---|---:|---:|---:|---|
| P0 | 18:38–19:55 | flat_children；完整动作描述 suffix，候选请求较早构造 | 7 | 5358.3 | 5145.2–5646.3 | 330 个候选导致大量 prefix 重复与 scheduler 排队 |
| P1 | 20:02–21:11 | hierarchical；类别阶段 + 子动作阶段 | 13 | 481.9 | 394.3–750.1 | 两阶段串行，但每阶段候选较少，明显低于旧 flat |
| P2 | 21:27 | flat_children；旧 flat 路径 | 3 | 4927.9 | 4819.6–5019.5 | 重启后仍维持约 5 秒，说明仅靠服务重启不能解决瓶颈 |
| P3 | 21:46 | flat_children；首次延迟构造候选 | 1 | 4111.5 | 4111.5–4111.5 | 减少了请求对象提前构造，但 tokenization / queue wait 仍很重 |
| P4 | 21:51–21:53 | flat_children；延迟构造 + short candidate_id suffix | 4 | 672.8 | 663.0–683.1 | 主要收益来自短 suffix 与共享 prefix KV cache |
| P5 | 22:02–22:07 | hierarchical；短 ID、延迟构造、阶段内 prefix 复用 | 2 | 229.9 | 221.8–238.1 | 合成请求中类别和子动作阶段均只产生一个候选批次 |
| P6 | 22:09–22:10 | hierarchical；最新实际多轮 session | 9 | 340.5 | 270.9–614.6 | 首个 turn 有预热/调度抖动；稳定后的 8 个 turn 平均 306.3 ms |

P6 去掉首个 turn 后的稳定区间为 270.9–326.8 ms。首个 turn 的类别阶段为 512.9 ms，其中 queue wait 为 378.8 ms；之后类别阶段降到 160.4–202.8 ms，说明主要是首次请求调度/预热开销，而不是每个 turn 固定存在的计算开销。

## 3. 改动与收益的关系

### P0：原始 flat_children

每个具体动作都参与一次动作候选评分。旧路径使用较长的动作描述 suffix，并在评分前较早构造全部候选请求。以 330 个候选为例，候选请求拆成 `64 + 64 + 64 + 64 + 64 + 10` 六批，但大量完整候选输入在进入 scheduler 前已经被准备，造成：

1. 生成大量 token ID 和请求对象；
2. 相同多模态历史 prefix 被重复带入候选请求；
3. scheduler queue wait 占据总耗时的大部分。

代表性内部数据：总计算约 5504 ms，queue wait 约 4673 ms，prefix 约 8674 tokens，candidate recompute 为 0。这里的 `candidate_recompute=0` 只说明请求执行阶段没有重新计算 prefix，不代表候选请求构造阶段没有产生大量重复输入。

### P1/P2：hierarchical 与旧 flat 对比

hierarchical 将 330 个具体动作先按类别缩小，再对选中类别的 children 评分。类别阶段通常约 59 个候选，子动作阶段通常只有几个到十几个候选，因此单 turn 通常落在 400–550 ms；首个 turn 可能因为 GPU 调度或预热达到 750 ms。

这不是消除了多模态处理，而是把一次大候选评分变成两个规模更小的评分。两个阶段仍然串行，所以 hierarchical 仍有两个模型阶段，但每个阶段的候选请求数量和排队压力小很多。

### P3：候选请求延迟构造

P3 只构造并执行共享 prefix，请求完成后再按批构造 suffix。目的在于避免在 prefix 进入 scheduler 之前就生成全部候选请求。耗时从约 4.9–5.6 秒下降到 4.11 秒，但 queue wait 仍约 3469 ms，说明延迟构造本身还不足以解决 tokenization 和候选 suffix 过长造成的压力。

### P4：short candidate_id suffix

P4 保留评分规则不变：仍然为每个候选计算目标 token 的 logit、token log-prob、平均 log-prob 和 PPL；改变的是候选 suffix 的表示方式：

```text
共享 prefix + "A124"
共享 prefix + "A125"
共享 prefix + "A126"
```

而不是：

```text
共享 prefix + "转头看左侧（头部与眼球同步，转向左侧注视）"
```

由于 suffix 只包含短 `candidate_id`，每批需要处理的候选 token 大幅减少，且共享 prefix 可以命中 KV cache。P4 总耗时降至约 0.66–0.68 秒，queue wait 约 45–51 ms，六批候选仍保留，但不再让每个候选重复携带长描述。

### P5/P6：hierarchical 的阶段内优化

hierarchical 中同样使用短 ID 和延迟构造：

```text
类别阶段：共享 prefix + "B001" / "B002" / ...
子动作阶段：共享 prefix + "A124" / "A125" / ...
```

类别阶段选出类别后，才构造该类别 children 的第二阶段请求。类别与子动作不能合并，因为第二阶段候选依赖第一阶段结果；但每个阶段内部都可以复用 prefix、减少 tokenization，并降低 scheduler 排队。

P5 合成请求的类别阶段为约 122–140 ms、子动作阶段为约 95–97 ms；P6 实际 session 中稳定 turn 的类别阶段为约 160–203 ms、子动作阶段为约 104–129 ms。P6 首个 turn 额外受到 queue wait 影响。

## 4. 按日志时间排序的全部成功 turn

下表保留所有成功完成的 turn，耗时单位为 ms。session 使用前 8 位显示，完整 ID 可在源日志中查询。

| # | 日志时间 | 阶段 | session | turn | 模式 | 总耗时 |
|---:|---|---|---|---|---|---:|
| 1 | 18:38:59 | P0 | `session_profile_04b64289` | `turn_0b38a734` | flat | 5512.4 |
| 2 | 19:37:36 | P0 | `session_profile_a8e66f00` | `turn_eda51658` | flat | 5537.0 |
| 3 | 19:37:59 | P0 | `session_profile_a8e66f00` | `turn_da7ef581` | flat | 5191.9 |
| 4 | 19:41:15 | P0 | `session_profile_f75b24ac` | `turn_b1899735` | flat | 5205.2 |
| 5 | 19:54:52 | P0 | `session_profile_902fc584` | `turn_abbcb837` | flat | 5646.3 |
| 6 | 19:55:19 | P0 | `session_profile_902fc584` | `turn_40cf2155` | flat | 5145.2 |
| 7 | 19:55:39 | P0 | `session_profile_902fc584` | `turn_58fd088d` | flat | 5269.8 |
| 8 | 20:02:31 | P1 | `session_profile_af5b2492` | `turn_1df93b4c` | hierarchical | 394.3 |
| 9 | 20:04:29 | P1 | `session_profile_af5b2492` | `turn_8f195870` | hierarchical | 406.2 |
| 10 | 20:04:40 | P1 | `session_profile_af5b2492` | `turn_959b34c3` | hierarchical | 414.8 |
| 11 | 20:04:48 | P1 | `session_profile_af5b2492` | `turn_2be8cf79` | hierarchical | 437.9 |
| 12 | 20:05:06 | P1 | `session_profile_af5b2492` | `turn_8bb66784` | hierarchical | 503.7 |
| 13 | 21:07:49 | diagnostic | `diag-session-hashlib` | `diag-turn-hashlib` | hierarchical | 190.1 |
| 14 | 21:10:50 | P1 | `session_profile_a05fe806` | `turn_f9cb51ee` | hierarchical | 750.1 |
| 15 | 21:10:59 | P1 | `session_profile_a05fe806` | `turn_d7c05ecf` | hierarchical | 443.4 |
| 16 | 21:11:07 | P1 | `session_profile_a05fe806` | `turn_335cf0fa` | hierarchical | 457.8 |
| 17 | 21:11:13 | P1 | `session_profile_a05fe806` | `turn_f3ee0609` | hierarchical | 491.8 |
| 18 | 21:11:22 | P1 | `session_profile_a05fe806` | `turn_6d10207b` | hierarchical | 551.3 |
| 19 | 21:11:29 | P1 | `session_profile_a05fe806` | `turn_62e7f02c` | hierarchical | 494.3 |
| 20 | 21:11:36 | P1 | `session_profile_a05fe806` | `turn_67ff78dc` | hierarchical | 457.8 |
| 21 | 21:11:46 | P1 | `session_profile_a05fe806` | `turn_b4e8de04` | hierarchical | 461.7 |
| 22 | 21:27:24 | P2 | `session_profile_18b2f6d8` | `turn_1a45f540` | flat | 4944.7 |
| 23 | 21:27:31 | P2 | `session_profile_18b2f6d8` | `turn_f079bc74` | flat | 4819.6 |
| 24 | 21:27:54 | P2 | `session_profile_18b2f6d8` | `turn_3714f254` | flat | 5019.5 |
| 25 | 21:46:45 | P3 | `lazy-flat-live` | `lazy-turn-330` | flat | 4111.5 |
| 26 | 21:51:17 | P4 | `lazy-flat-short-id-live` | `turn-lazy-001` | flat | 683.1 |
| 27 | 21:53:52 | P4 | `lazy-flat-short-id-repeat` | `turn-1` | flat | 663.0 |
| 28 | 21:53:53 | P4 | `lazy-flat-short-id-repeat` | `turn-2` | flat | 665.8 |
| 29 | 21:53:54 | P4 | `lazy-flat-short-id-repeat` | `turn-3` | flat | 679.1 |
| 30 | 22:02:12 | P5 | `hierarchical-optimization-live-3` | `turn-hier-001` | hierarchical | 221.8 |
| 31 | 22:07:27 | P5 | `hierarchical-optimization-live-final` | `turn-hier-001` | hierarchical | 238.1 |
| 32 | 22:09:57 | P6 | `session_profile_4eaf144c` | `turn_1e6002d5` | hierarchical | 614.6 |
| 33 | 22:10:05 | P6 | `session_profile_4eaf144c` | `turn_8a35473a` | hierarchical | 270.9 |
| 34 | 22:10:09 | P6 | `session_profile_4eaf144c` | `turn_1d5925d2` | hierarchical | 308.1 |
| 35 | 22:10:14 | P6 | `session_profile_4eaf144c` | `turn_a3c35e36` | hierarchical | 296.1 |
| 36 | 22:10:20 | P6 | `session_profile_4eaf144c` | `turn_4b42e3de` | hierarchical | 326.8 |
| 37 | 22:10:27 | P6 | `session_profile_4eaf144c` | `turn_4356ec81` | hierarchical | 318.1 |
| 38 | 22:10:34 | P6 | `session_profile_4eaf144c` | `turn_a9aae697` | hierarchical | 315.9 |
| 39 | 22:10:43 | P6 | `session_profile_4eaf144c` | `turn_5c5840e7` | hierarchical | 322.5 |
| 40 | 22:10:56 | P6 | `session_profile_4eaf144c` | `turn_e6be56c5` | hierarchical | 291.6 |

## 5. 代表性内部阶段数据

### 5.1 原始 flat 与延迟构造 flat

| 阶段/时间 | 总内部耗时 | queue wait | prefix tokens | 候选批次 | 备注 |
|---|---:|---:|---:|---|---|
| P0，18:38 | 5504 | 4673 | 8674 | 64×5+10 | 长 suffix、旧构造路径 |
| P2，21:27:24 | 4937 | 4475 | 约 8780 | 64×5+10 | 旧 flat 路径 |
| P3，21:46:45 | 4104 | 3469 | 6587 | 64×5+10 | 只做延迟构造 |
| P4，21:53:52 | 656 | 45 | 6786 | 64×5+10 | 延迟构造 + short ID |
| P4，21:53:53 | 659 | 45 | 6849 | 64×5+10 | 延迟构造 + short ID |
| P4，21:53:54 | 673 | 45 | 6912 | 64×5+10 | 延迟构造 + short ID |

P4 的日志显示 `cached_prefix_tokens == prefix_tokens`，`candidate_recompute_tokens == 0`。这说明候选 suffix 请求复用了本次评分请求已经计算出的共享 prefix，没有为每个候选重新计算 prefix；它不表示共享 prefix 本身全部来自启动预热。其余耗时主要来自六批短 suffix 的实际评分和少量调度开销。

### 5.2 hierarchical

| 时间 | 总耗时 | 类别阶段 | 类别 queue | 子动作阶段 | 子动作 queue | 候选数 |
|---|---:|---:|---:|---:|---:|---|
| 22:02:12 | 221.8 | 121.8 | 11.1 | 96.9 | 5.8 | 60 + 2 |
| 22:07:27 | 238.1 | 139.6 | 9.8 | 95.3 | 5.0 | 60 + 2 |
| 22:09:57 | 614.6 | 512.9 | 378.8 | 98.0 | 12.7 | 约 60 + 2 |
| 22:10:05 | 270.9 | 160.4 | 43.6 | 103.7 | 14.2 | 约 60 + 2 |
| 22:10:09 | 308.1 | 196.5 | 40.2 | 107.2 | 19.1 | 约 60 + 2 |
| 22:10:14 | 296.1 | 183.5 | 45.5 | 108.8 | 15.0 | 约 60 + 2 |
| 22:10:20 | 326.8 | 192.7 | 57.5 | 128.8 | 34.9 | 约 60 + 2 |
| 22:10:27 | 318.1 | 171.5 | 45.4 | 123.1 | 36.1 | 约 60 + 2 |
| 22:10:34 | 315.9 | 202.8 | 57.4 | 109.2 | 17.5 | 约 60 + 2 |
| 22:10:43 | 322.5 | 181.6 | 42.0 | 129.4 | 39.6 | 约 60 + 2 |
| 22:10:56 | 291.6 | 172.3 | 42.6 | 115.5 | 20.2 | 约 60 + 2 |

以上 hierarchical 请求的类别和子动作阶段中，候选 suffix 均复用了各自评分请求的共享 prefix，候选 prefix 重算为 0；Category 和 Child 的共享 prefix 仍分别执行 Prefill。两阶段必须串行，因此它的总耗时不会像 flat short ID 那样只包含一个评分阶段。

## 6. 结论

1. **最大收益来自 suffix 表示和请求构造方式的共同变化。** 单独做延迟构造只将约 5 秒降到约 4.1 秒；加入短 `candidate_id` suffix 后降到约 0.67 秒。
2. **hierarchical 的优势来自缩小候选集合。** 它需要类别评分和子动作评分两次，但每阶段候选较少，所以实际稳定耗时约 0.27–0.33 秒。
3. **候选级 KV cache 复用已经生效，但动态 Prefix Prefill 仍是主要开销。** 新路径中 `cached_prefix_tokens == prefix_tokens` 且 `candidate_recompute_tokens == 0`，只说明同一评分请求内的候选 suffix 不会重复计算共享 prefix。每个 Turn 的 Category 和 Child 仍分别对包含动态上下文的完整共享 prefix 执行 Prefill，启动预热只覆盖静态 System Prompt。
4. **首个 turn 需要单独看。** P6 首个 turn 的 378.8 ms 类别 queue wait 明显高于后续 40–58 ms，符合首次请求预热/调度抖动特征；不能用首个 turn 代表稳定态性能。
5. **当前比较仍有实验限制。** P0/P2/P3/P4/P5/P6 的媒体、历史长度、候选目录和服务状态不是严格相同，文档中的阶段关联用于定位优化方向，不替代固定输入下的正式 benchmark。

## 7. 2026-08-25 性能回归与待处理优化

状态：**历史基线，已由第 8 节取代**。本节记录 2026-08-25 当时实现和 `logs/realtime/2026-08-25/21/` 结构化日志中的性能基线，供后续优化与回归对比。这里的数据来自当时的回复与动作融合链路，不与 P0–P6 的旧实验条件直接横向比较。

### 7.1 当前基线

| 指标 | 当前结果 |
|---|---:|
| Category 平均耗时 | 约 430 ms |
| Child 平均耗时 | 约 239 ms |
| 动作就绪平均耗时 | 约 672 ms |
| 动作就绪 P50 | 约 682 ms |
| 动作就绪最大值 | 约 731 ms |

客户端需要同时拿到文本首包和最终动作后才能进入下一阶段，因此当前实际可推进屏障主要由动作就绪耗时决定。

代表性阶段拆分如下：

| 阶段 | Prefix Token | Prefix Prefill | 候选 suffix 评分 | 说明 |
|---|---:|---:|---:|---|
| Category | 约 5,000 | 约 229 ms | 约 129 ms | 64 个真实类别加 `B000` 时共有 65 个评分候选 |
| Child | 约 2,400 | 约 164 ms | 约 46 ms | 候选数量较少，但仍需单独计算完整 Child Prefix |

Category 完成后才能确定 Child 候选集合，当前两阶段串行执行，所以动作就绪耗时基本等于 Category、Child 及少量编排开销之和。

### 7.2 `prefix_cached` 的准确含义

当前调度器先计算一次共享 Prefix，再分批构造和执行候选 suffix。`prefix_cached=true` 的判定条件是每个候选 suffix 的 `candidate_cached_tokens` 均覆盖本次请求的 `prefix_token_count`。因此它只能证明：

- 候选 suffix 复用了本次评分请求已经计算出的共享 Prefix；
- 候选之间没有重复 Prefill 相同 Prefix；
- `candidate_prefix_recompute_tokens=0` 时，候选阶段没有重新计算 Prefix Token。

它不能证明启动时的全局预热覆盖了当前 Turn 的完整 Prefix。全局目录模式使用 `cache_static_system_only=True`，启动时只预热 Category 和 Child 的静态 System Prompt；当前音频或文本、`state_description`、`action_profile`、本次会话允许的类别和动作、默认动作类别、图片及结构化状态等动态内容仍需在每个 Turn 中参与 Prefix Prefill。跨 Turn 回复和动作历史现已从动作评分上下文移除。

### 7.3 主要耗时原因

1. **动态 Prefix 每轮重复 Prefill。** Category 和 Child 都会带入人设、允许范围、当前状态和约束；其中部分内容在会话内固定，部分规则在静态和动态 Prompt 中重复出现。跨 Turn 历史已移除，但其余动态内容仍需优化。
2. **Category 与 Child 不共享模型 KV。** 当前 `context_cache_status=hit` 主要体现 Child 复用了音频、图片等预处理结果；Child 仍使用自己的 System Prompt 和动态 Prompt，重新执行约 2.4k Token 的 Prefix Prefill。
3. **当时的 65 个 Category 产生尾批。** 默认 `micro_batch_size=64` 时，64 个真实类别加 `B000` 被拆成 `[64, 1]` 两批。当前目录已变为 52 个评分候选并能单批完成，因此该问题和 64→128 的调优建议已经失效。
4. **provisional 回复与 Category 竞争同一 GPU。** 用户 Turn 中两者并行可以提前回复首包、尽早启动临时 TTS，但也会竞争 Thinker 计算资源，使 Category 延迟增加；这是回复 TTFT 与动作就绪耗时之间的权衡。
5. **动作 Prompt 持续增长。** 最近增加了社交事件、自我介绍、对话反馈、左右坐标、显式动作优先和不支持动作等规则。静态部分会被预热，但会增加缓存占用；动态部分的重复会直接增加每轮 Prefill。

### 7.4 下午处理清单

| # | 待办 | 实施要点 | 状态 |
|---:|---|---|---|
| 1 | 动态 Prompt 去重 | 跨 Turn 动作历史已移除；继续合并重复的 `B000`/`A000`、`state_description` 和人设优先级规则，每项语义只保留一个权威定义 | 进行中 |
| 2 | 会话固定前缀缓存 | 将 `action_profile`、允许类别、默认动作类别等会话内固定内容划入稳定缓存边界，Turn 内只追加变化部分 | 待处理 |
| 3 | 调度策略对比 | 对比 Category 优先调度与现有 provisional 并行策略，量化回复 TTFT 和动作就绪耗时的取舍 | 待处理 |
| 4 | micro-batch 实测 | 在同一固定输入下对比 64 和 128，确认消除 `[64, 1]` 尾批后的净收益和显存影响 | 待处理 |
| 5 | Category/Child KV 续接 | 评估可共享的公共 Prefix 或可续接 KV，使 Child 复用当前 Turn 的模型 KV，而不只复用媒体编码 | 待处理 |
| 6 | 确定性 Child 快路径 | 定义只有一个真实候选或无需比较时跳过完整 Child PPL 的安全条件，并保留不支持动作判断的正确性 | 待处理 |

### 7.5 统一验证指标

每项优化必须使用相同输入和候选目录进行前后对比，并至少记录：

- Category/Child 的 `prefix_token_count`、`prefix_prefill_ms`；
- `suffix_batch_count`、`suffix_batch_sizes`、`suffix_batch_ms`；
- provisional 与正式回复的文本首包 TTFT；
- 动作就绪 P50、P95；
- GPU 利用率、显存占用、功耗，以及同卡竞争时的峰值；
- `context_cache_status`、`prefix_cached` 和 `candidate_prefix_recompute_tokens`，并按本节定义解释。

## 8. 2026-08-27 动作就绪延迟优化计划

状态：**原始规划，已由 8.5 的实际落地边界取代**。本节基于 `sess_fc1c4bae` 的结构化日志记录当时的候选方案。

### 8.1 当前基线与目标

稳定用户 Turn 的当前基线为：

| 指标 | 当前结果 | 验收目标 |
|---|---:|---:|
| 回复首包 P50 | 约 108 ms | 相对基线回归不超过 5 ms |
| 回复首包 P95 | 约 184 ms | 相对基线回归不超过 10 ms |
| Category 平均耗时 | 约 297 ms | 按阶段记录，不单独设硬门槛 |
| Child 平均耗时 | 约 146 ms | 按阶段记录，不单独设硬门槛 |
| 动作就绪 P50 | 约 452 ms | 不高于 400 ms |
| 动作就绪 P95 | 约 587 ms | 不高于 450 ms |
| `action_finished` 动作就绪 | 约 2.6 ms | P50 不高于 5 ms |

当前 Category 有 52 个评分候选，在 `micro_batch_size=64` 下已经单批完成；不再通过增大 micro-batch 解决历史上的 `[64, 1]` 尾批。`action_finished` 已跳过 Category 和 Child 评分，也不是本轮优化对象。

本次采用平衡策略：继续让 provisional 回复与 Category 并行，不通过延迟文本首包来换取动作速度；普通业务类别继续保留 Child PPL 和 `A000` 拒识能力。

### 8.2 分阶段实施

#### 阶段一：合并并发音频编码

- 为回复与 Category 的并发音频编码增加 singleflight：相同音频和预处理参数只编码一次，其他请求等待并复用结果。
- 缓存键必须包含音频内容、采样参数和预处理配置；失败和取消必须唤醒所有等待者。
- 回复和动作图片继续按角色隔离，只复用内容和用途完全一致的媒体。
- 预计减少 15–35 ms；保持 provisional 回复与 Category 的现有并发关系。

#### 阶段二：建立会话固定动作前缀缓存

将动作上下文拆成三层：

1. 全局固定：通用动作规则和全局目录版本；
2. 会话固定：语言、`action_profile`、允许类别、允许动作和兜底类别；
3. 本轮动态：当前音频、文本、图片、姿态、`state_description` 和指代锚点。

`session.start` 预填 Category、B001 和 B002 的会话固定前缀；普通 Child 第一次使用时按需预填，每个 Session 最多保留最近 8 个 Child 前缀。缓存命名空间包含 Session ID、语言、目录 Hash 和 Prompt Hash。Session 结束、失败或取消后必须释放对应引用，不得缓存跨 Turn 用户音频、图片、文本或状态。

#### 阶段三：复用 Category 到 Child 的本轮模型 KV

Category 和 Child 使用完全相同的公共前缀：

```text
通用动作规则
→ 会话动作人设
→ 当前状态与主动约束
→ 当前数字人图片
→ 当前用户音频和文本
```

Category 在公共前缀后追加允许类别、`B000` 规则和类别输出要求；Child 追加已选类别、具体候选、`A000` 规则和动作输出要求。B001 的回复首句只属于 Child 分支。

Category 完成后，Child 直接续用该 Turn 的公共前缀 KV。临时 KV 必须在 Child 完成、取消或失败后释放。该阶段预计减少 Child Prefix Prefill 的 50–75 ms，同时继续复用现有媒体编码结果。

#### 阶段四：候选后缀 Trie 评分

前三个阶段完成后，如果动作就绪 P50 仍高于 400 ms，再实施本阶段：

- 将候选 ID 的 Token 序列构造成 Trie，共同 Token 前缀只计算一次；
- 保持每个候选原有的 mean NLL/PPL 定义，不改变候选范围和最终决策规则；
- 新旧评分器的 mean log-prob 绝对误差不得超过 `1e-5`，Top-1 必须一致；
- 新评分器异常时自动回退到现有批量评分器。

预计额外减少 20–40 ms。该优化不能替代 `B000/A000`，也不能以关键词硬路由代替模型评分。

### 8.3 开关、观测与回滚

增加以下内部特性开关，初始默认关闭，不改变 WebSocket 公开协议：

- `SGLANG_OMNI_ACTION_SESSION_PREFIX_CACHE_ENABLED`
- `SGLANG_OMNI_ACTION_SHARED_TURN_KV_ENABLED`
- `SGLANG_OMNI_ACTION_SUFFIX_TRIE_ENABLED`

结构化日志需要补充：

- 全局、会话和本轮未缓存 Prefix Token 数；
- 会话固定前缀及 Category/Child 公共 KV 命中状态；
- 音频 singleflight 命中、等待和失败状态；
- Trie 唯一节点数、原始候选 Token 数和复用率；
- 各阶段显存、GPU 利用率、调度等待和释放结果。

任一阶段出现准确率回归、缓存污染、显存持续增长或延迟未改善时，只关闭对应内部开关，不回滚其他已验证阶段。

### 8.4 测试与发布顺序

- 使用固定候选、固定音频和固定图片执行 2 个预热 Turn 加至少 30 个正式 Turn，统计回复 TTFT、Category/Child Prefill、suffix 评分和动作就绪 P50/P95。
- 验证 concurrent audio singleflight 只执行一次编码，并覆盖失败、取消和不同采样参数不得复用的场景。
- 验证不同 Session、语言、目录 Hash、Prompt Hash 和 `action_profile` 之间不能复用错误 KV；Session 结束后缓存可回收。
- 验证 Category 和 Child 的公共 Token 边界完全一致，Child 能实际命中本轮 KV，B001 回复前缀仍只进入 Child。
- 回归 `reply_action_quality_cases.md`、`B000/A000`、B001/B002、明确动作、委婉动作、社交事件、自我介绍和“刚刚那个动作”指代。
- 先在 GPU6 依次开启音频复用、会话缓存和公共 KV；指标和准确率通过后再部署 GPU7。Trie 仅在前三项未达到 400 ms 目标时启用。

本计划只修改 `sglang-omni` 的缓存、调度和评分实现，不修改 `D_video_call`、动作目录、客户端事件或字段。

### 8.5 2026-09-08 实际落地边界

本轮实施采用保守版本，不为了性能重排或合并业务规则：

- Category Prompt 仅把 Session 固定的人设、实体和动作偏好移到当前
  `avatar_state/state_description` 之前；其余 Turn 规则、白名单、媒体、文本和输出提示顺序
  保持原语义。
- 服务启动继续预热全局 Category/Child 静态目录；`session.start` 在
  `session.started` 前同步扩展用户 Turn 的 Category Session 前缀。失败只退化为首轮 miss，
  不拒绝会话。普通 Child 的 Session 前缀在第一次真实评分时建立，由底层 KV 缓存淘汰机制
  管理。
- 不实施 8.2 阶段三所述的 Category→Child 本轮模型 KV 强制续接，因为两级 System Prompt
  和动态约束并不完全相同；当前仍只复用安全的 Session 前缀与已有媒体编码结果。
- 同一 Turn 的媒体仍按原有内容 Hash 和预处理参数进入编码缓存/同批 singleflight；不修改
  媒体保留轮数、选图、历史裁剪、可见性、跨 Turn 或跨 Session 生命周期。
- 评分准入默认容量为 2，并按优先级排队：历史路由和 Category 为 0，Child 与表现控制为 1，
  S0/S1 等二次判定为 2，Session 预填为 3。环境变量
  `SGLANG_OMNI_ACTION_SCORE_MAX_INFLIGHT` 可在 1–4 范围调整，设为 1 即恢复串行准入。
- Category 不等待回复历史路由；“再做一遍刚刚那个动作”使用独立的最近用户触发动作锚点。
  客户端后续提供的 `last_executed_action_id` 用于校正当前物理动作锚点，不把动作历史混入
  回复历史。
- provisional 提升/丢弃、纯动作不支持提示、语言意图保留、纯表情和组合表演的融合规则均
  保持不变；`turn.action.ready` 仍可先于 TTS 完成事件。

上线前必须分别测量容量 1 和 2 下的文本/音频首包、Category/Child、动作 ready、GPU queue
wait、显存与 P50/P95/P99；容量 2 若造成竞争或尾延迟回归，应先回退并发上限，而不是改变
Prompt 语义。

相关实现和算法说明：

- [动作 suffix 评分说明](action_suffix_scoring.md)
- [动作推理变更历史](action_inference_change_history.md)
- [动作推理调度实现](../../../sglang_omni/scheduling/omni_scheduler.py)
- [候选请求构造实现](../../../sglang_omni/models/qwen3_omni/request_builders.py)
- [动作评分实现](../../../sglang_omni/models/qwen3_omni/action_scoring.py)

P0–P6 最后一次旧记录的服务端口为 `18001`；第 7 节按结构化日志统计，不以监听端口作为筛选条件。本文只描述成功动作推理 turn，不包含失败、取消或尚未完成的请求。
