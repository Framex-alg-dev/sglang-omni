# 单 Token 动作评分验证（2026-09-18）

## 结论

单 Token 动作评分链路已经能够正常工作，并在 `enforce` 模式下完全跳过候选 suffix batch；但是当前使用的任意两字母单 Token 映射不具备可用的零样本语义，不能启用 `enforce`。

当前线上配置保持为：

```text
SGLANG_OMNI_ACTION_SINGLE_TOKEN_MODE=shadow
```

## 验证集与结果

### 117 个标准动作

报告：`reports/text_action_single_token_canonical_shadow_fit_20260918.json`

| 评分路径 | Top-1 | Top-3 | Top-5 | 错误数 |
| --- | ---: | ---: | ---: | ---: |
| 原 suffix PPL | 51/117（43.59%） | 93/117（79.49%） | 97/117（82.91%） | 0 |
| 单 Token，未校准 | 1/117（0.85%） | 3/117（2.56%） | 6/117（5.13%） | 0 |

单 Token 结果几乎全部坍缩到候选 `220`（伸肘），说明 logits 主要反映别名 token 自身的语言模型先验，而不是动作语义。

### 8 个冒烟/边界用例

Shadow 报告：`reports/text_action_single_token_shadow_smoke_20260918.json`

Enforce 报告：`reports/text_action_single_token_enforce_smoke_20260918.json`

`enforce` 下 8/8 请求均完成且服务无错误，但 7 个有候选期望的用例 Top-1、Top-3、Top-5 均为 0，所有用例都选择了 `220`。因此链路正确不等于分类可用。

### 先验校准

使用 117 个标准动作拟合 action-only bias，产物：

```text
reports/action_single_token_map_calibrated_canonical_20260918.json
```

离线应用 bias 后，标准动作 Top-1/Top-3/Top-5 仅为 4.27%/9.40%/14.53%；8 个冒烟用例仍全部不命中。简单的类别先验校准无法补回缺失的语义映射。

## 性能路径核验

`enforce` 用例的调度统计确认：

```text
suffix.batch_count = 0
suffix.batch_sizes = []
suffix.batch_total_ms = 0
candidate_preparation.materialize_ms = 0
candidate_preparation.enqueue_ms = 0
candidate_preparation.queue_wait_ms = 0
```

即每轮只有一次 prefix forward，然后直接读取对应 token logits，不再创建或执行 142 个 suffix 请求。

8 个用例的 `turn.commit -> turn.result`：

| 模式 | P50 | P95/最大值 |
| --- | ---: | ---: |
| shadow（同时运行旧评分和新评分） | 893 ms | 2030 ms |
| enforce（仅单 Token） | 653 ms | 1121 ms |

这里的总时间仍受 intent/performance-control 等并行门控影响，不能把差值全部视为动作 scorer 本身的耗时。单 Token scorer 的典型 scheduler 部分约为 prefix prefill 49–63 ms，suffix 为 0 ms。

## 原因

当前映射把每个动作分配给模型词表中已有的任意两字母 token，例如：

| 动作 ID | 动作 | 别名 Token |
| --- | --- | --- |
| 220 | 伸肘 | `WI` |
| 270 | 点赞 | `DK` |
| 285 | 双手比心 | `OA` |
| 288 | 单手挥手 | `LI` |

这些 token 的 embedding 和输出权重从未针对动作分类训练。仅在 prompt 中声明映射表，不能保证模型在 prefix 最后一个位置把“点赞”转换为 `DK`；结果会被 token 固有频率和输出偏置主导。

## 下一步建议

1. 保持 `shadow`，不要启用当前映射的 `enforce`。
2. 首选在 prefix hidden state 上训练独立的 117/118 类 action classification head。推理仍是 batch 1、无 suffix batch，同时不受词表 token 语义限制。
3. 备选方案是加入动作专用 special tokens，并用覆盖标准表达、自然表达、否定、能力询问和多模态输入的数据做 LoRA/微调；仅扩展词表但不训练 embedding/lm_head 同样不可用。
4. 用同一套召回率测试作为上线门槛，并另外划分未参与训练/校准的验证集，避免在 117 个标准提示上过拟合。

建议最低门槛：新路径的 Top-1 不低于当前旧路径，Top-5 不低于 82.91%，且否定、禁止和 `UNSUPPORTED` 用例单独统计通过率。

## 回归测试

相关单元与调度测试共 393 项通过。测试结束时出现 `/tmp/sglang-omni-realtime-logs` 写权限告警，但 pytest 退出码为 0，与本次单 Token 逻辑无关。
