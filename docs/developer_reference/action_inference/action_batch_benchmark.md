# 动作推理 micro-batch 性能验证

本文记录 `SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE` 对 realtime 动作推理的实际验证结果。

## 1. 验证范围和输入

验证入口是 [qwen3_omni_action_batch_benchmark.py](../../../scripts/qwen3_omni_action_batch_benchmark.py)，通过正式的 `WS /v1/session/realtime` 协议执行，不直接调用内部 action-score API。

固定输入：

- 当前脚本动作目录：`sglang_omni/assets/character_action_global_catalog.json`
- 目录实际包含 386 个具体动作、64 个类别
- 额外加入一个 `no_action`，因此 flat_children 共 387 个 child 候选，hierarchical 共 65 个类别
- 音频：`/data/models/hehy/D_human_train/examples/test/bench20_examples/none_4.wav`
- 图片：`/data/models/xingmt/wan_export_step651_speedtest/ref.png`
- 当前文本：`你可以开心一点吗？`
- 数字人状态：`pose=seated, gaze=camera, hands=resting`
- 每组 3 个独立 session；音频按 200 ms PCM16 chunk 发送，当前 turn 发送 1 张图片
- `diagnostics.include_action_scores=true`，用于校验候选完整性和排序结果

下文 flat/hierarchical 对比数据是正式协议收敛前的历史测量。当前
`protocol_version=1` 固定使用 hierarchical，benchmark 脚本只上传全局目录的紧凑
candidate ID 白名单，不再接受 `--mode` 或上传动作语义目录。

为了验证 batch 大小本身，benchmark 使用短的外部 `candidate_id`（`A000`、`A001`……），同时保留目录中的长 canonical `action_id` 作为动作执行 ID。这样不会把长 action_id 的 token 数量误算成 batch 优化收益。

## 2. 实现改动

新增环境变量：

```text
SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE
```

行为：

- 默认值为 `64`，保持原有行为
- 支持范围为 `1–256`
- 服务启动创建 `MultimodalSessionManager` 时校验非法值；非法值不会创建 session
- batch 配置在 manager 初始化时解析，并固定传递给后续 session
- 不改变 prompt 顺序、候选排序规则、logit/PPL 计算公式或动作映射

相关代码：

- [realtime batch 配置](../../../sglang_omni/serve/realtime/multimodal.py)
- [micro-batch 分批执行](../../../sglang_omni/scheduling/omni_scheduler.py)
- [候选评分约束](../../../sglang_omni/models/qwen3_omni/action_scoring.py)

回归测试：

```bash
/home/ubuntu/miniconda3/envs/sglang-omni-local/bin/python -m pytest -q \
  tests/unit_test/qwen3_omni/test_multimodal_session.py \
  tests/unit_test/qwen3_omni/test_action_scoring.py
```

结果：`40 passed`。

## 3. 结果汇总

`server_action_compute_ms` 是服务端从动作评分开始到完成的耗时。每组首个请求可能包含 session catalog prefill 或调度预热，因此同时给出全部样本中位数和去掉首个样本后的稳定均值。

| 模式 | batch | 候选规模 | suffix 批次 | 全部中位数 ms | 稳定均值 ms | 稳定范围 ms | 结果 |
|---|---:|---:|---|---:|---:|---:|---|
| flat_children | 64 | 387 | 64×6 + 3 | 2751.1 | 2744.9 | 2706.1–2783.6 | 成功 |
| flat_children | 128 | 387 | 128×3 + 3 | 2683.5 | 2681.4 | 2679.3–2683.5 | 成功 |
| flat_children | 256 | 387 | 256 + 131 | 2742.9 | 2732.8 | 2722.7–2742.9 | 成功 |
| hierarchical | 64 | 65 + 12 | 64+1；12 | 1331.8 | 1330.6 | 1329.3–1331.8 | 成功 |
| hierarchical | 128 | 65 + 12 | 65；12 | 1328.2 | 1325.7 | 1323.3–1328.2 | 成功 |
| hierarchical | 256 | 65 + 12 | 65；12 | 1360.4 | 1357.2 | 1354.0–1360.4 | 成功 |

其中 hierarchical 的 `12` 是本次固定输入下类别选择后命中的 child 数量；不同输入可能命中其他类别，child 数会变化。

原始结果：

- [flat_children batch 64](../../../results/action_batch_benchmark_flat_children_64_short_id.json)
- [flat_children batch 128](../../../results/action_batch_benchmark_flat_children_128_short_id.json)
- [flat_children batch 256](../../../results/action_batch_benchmark_flat_children_256_short_id.json)
- [hierarchical batch 64](../../../results/action_batch_benchmark_hierarchical_64.json)
- [hierarchical batch 128](../../../results/action_batch_benchmark_hierarchical_128.json)
- [hierarchical batch 256](../../../results/action_batch_benchmark_hierarchical_256.json)

## 4. 内部日志对比

### flat_children

所有 flat 请求均满足：

- `prefix_tokens=24518`
- `cached_prefix_tokens=24518`
- `candidate_prefix_recompute_tokens=0`
- 返回完整 `387` 个候选分数
- 三个 batch 配置下 Top-1 都是同一个 canonical action：`act_6d0bc67bb4a3f62a1adfe1a8e5028f1a`

稳定样本的典型阶段数据：

| batch | suffix batch ms | queue wait ms | 内部 total ms | GPU 最大进程显存 |
|---:|---|---:|---:|---:|
| 64 | 137–152，最后一批约 50 | 392–413 | 2699–2776 | 约 47.4 GB |
| 128 | 270–272，最后一批约 48 | 391–400 | 2673–2677 | 约 47.4 GB |
| 256 | 542–543 + 319–326 | 404–405 | 2716–2736 | 约 47.5 GB |

batch 变大后，单批 suffix 计算时间近似随批大小增加：64 大约 138 ms，128 大约 271 ms，256 大约 543 ms。批次数减少了，但 GPU 总工作量没有消失，因此总耗时没有出现线性下降。

### hierarchical

所有 hierarchical 请求的类别和 child 阶段均满足 prefix cache 命中和 prefix 重算为 0：

- 类别阶段：`prefix_tokens=5524`
- child 阶段：`prefix_tokens=4791`
- 类别阶段 batch=64 时为 `[64, 1]`，batch=128/256 时为 `[65]`
- child 阶段固定为一个 `[12]` 批次
- 最终 `turn.result` 返回 12 个 child 分数
- 三个 batch 配置下 Top-1 都是同一个 canonical action：`act_d3f218c66bc42a96d7ec26b244eda120`

hierarchical 的首个样本分别出现约 1.8 秒的启动/调度抖动，后两个样本稳定在约 1.32–1.36 秒；这也是为什么不能只看单次结果。

## 5. 为什么 flat_children 没有因 batch 变大而超过 hierarchical

这次验证的关键不是 suffix 批次数，而是 prefix 规模：

```text
flat_children：system prompt 包含全部 387 个具体动作定义
               prefix_tokens ≈ 24518

hierarchical 类别阶段：system prompt 包含 65 个类别
                       prefix_tokens ≈ 5524

hierarchical child 阶段：只包含选中类别的 12 个 child
                         prefix_tokens ≈ 4791
```

flat_children 虽然使用短 `candidate_id` suffix，但所有具体动作的描述仍然位于固定候选 system prompt 中。即使 session 级 KV cache 已命中，这个约 24.5k token 的长前缀仍会增加当前 turn 请求的 KV 管理、调度和候选评分工作；增大 batch 只影响 suffix 的分批方式，不能减少这部分候选前缀负担。

因此本次实际结果是：

- flat_children batch=64 → 128：稳定均值只下降约 63 ms
- flat_children batch=128 → 256：稳定均值反而上升约 51 ms
- hierarchical 三组都约 1.33–1.36 秒，batch 变化很小
- flat_children 最快稳定均值约 2.68 秒，仍比 hierarchical 快约 1.35 秒

结论：在当前动作目录和多模态输入下，仅把 batch 提到 256 不能让 flat_children 超过 hierarchical。继续提升 batch 的收益已经被单批 GPU 计算、显存压力和 24.5k token prefix 主导的耗时抵消。

## 6. 资源与安全性结论

- 六组共 18 个 session 全部成功
- 没有 OOM、超时或 `action_score_failed`
- 所有 flat 请求返回完整 387 个候选；hierarchical 返回完整的最终 child 候选集合
- 所有请求 `cached_prefix_tokens == prefix_tokens`，候选 prefix 重算为 0
- batch=256 运行时 GPU 仍有约 3.9 GB 可用显存，但已经明显低于服务刚启动时的可用显存，不建议继续盲目增大
- batch=128 是 flat_children 中本次最好的稳定结果，但相对 batch=64 的收益很小
- hierarchical 当前推荐继续使用默认 batch=64；其类别阶段在本目录下有 65 个类别，batch=64 会多出一个大小为 1 的尾批，如果类别数量长期固定超过 64，可使用 128 消除该尾批，但实际收益很小

## 7. 后续优化方向

如果目标是让 flat_children 接近或超过 hierarchical，优先级应是：

1. 让外部候选目录提供稳定、短且真正用于评分的 `candidate_id`，canonical `action_id` 只用于执行映射；
2. 继续优化共享 prefix 的 session 级 KV 复用边界，确认当前 turn 的音频、图片和状态不会使静态 catalog prefix 失去命中；
3. 对 hierarchical 使用 `SGLANG_OMNI_ACTION_CATEGORY_TOP_K` 做可选类别兜底；
4. 在显存余量充足的独立环境再评估 512 以上 batch，不建议在当前单卡服务直接放开。

类别 `source_label` 和 `short_definition` 来自服务端全局目录。若需减少类别 Prompt，应更新
并重新验证服务端全局目录，而不是由客户端在 `session.start` 覆盖。

本次验证只比较动作推理服务耗时，不验证外部动作播放器、骨骼绑定或动画渲染。

## hierarchical 运行时诊断优化

hierarchical 的类别阶段默认 batch 为 128，以避免当前 65 个类别在 batch=64 时产生
`64+1` 尾批。
