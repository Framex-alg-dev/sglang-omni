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
- `prefix_tokens` / `cached_prefix_tokens`：共享 prefix 的 token 数及命中 KV cache 的 token 数
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

P4 的日志显示 `cached_prefix_tokens == prefix_tokens`，`candidate_recompute_tokens == 0`。这说明 prefix KV cache 已命中；其余耗时主要来自六批短 suffix 的实际评分和少量调度开销。

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

以上 hierarchical 请求的类别和子动作阶段均命中 prefix cache，候选 prefix 重算为 0；两阶段仍必须串行，因此它的总耗时不会像 flat short ID 那样只包含一个评分阶段。

## 6. 结论

1. **最大收益来自 suffix 表示和请求构造方式的共同变化。** 单独做延迟构造只将约 5 秒降到约 4.1 秒；加入短 `candidate_id` suffix 后降到约 0.67 秒。
2. **hierarchical 的优势来自缩小候选集合。** 它需要类别评分和子动作评分两次，但每阶段候选较少，所以实际稳定耗时约 0.27–0.33 秒。
3. **KV cache 复用已经生效。** 新路径中 `cached_prefix_tokens == prefix_tokens` 且 `candidate_recompute_tokens == 0`，因此后续优化重点不是重复计算 prefix，而是减少候选批次、改善调度和降低多模态前处理开销。
4. **首个 turn 需要单独看。** P6 首个 turn 的 378.8 ms 类别 queue wait 明显高于后续 40–58 ms，符合首次请求预热/调度抖动特征；不能用首个 turn 代表稳定态性能。
5. **当前比较仍有实验限制。** P0/P2/P3/P4/P5/P6 的媒体、历史长度、候选目录和服务状态不是严格相同，文档中的阶段关联用于定位优化方向，不替代固定输入下的正式 benchmark。

相关实现和算法说明：

- [动作 suffix 评分说明](action_suffix_scoring.md)
- [动作推理变更历史](action_inference_change_history.md)
- [动作推理调度实现](../../../sglang_omni/scheduling/omni_scheduler.py)
- [候选请求构造实现](../../../sglang_omni/models/qwen3_omni/request_builders.py)
- [动作评分实现](../../../sglang_omni/models/qwen3_omni/action_scoring.py)

最后一次记录的服务端口为 `18001`；本记录只描述成功动作推理 turn，不包含失败、取消或尚未完成的请求。
