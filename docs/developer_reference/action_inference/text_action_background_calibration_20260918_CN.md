# 文本动作候选背景基线校准实验（2026-09-18）

## 结论

本次只做离线重排，没有修改或重启 18004 服务。

- 直接把候选 `mean_logprob` 按背景基线校准，可以显著改善已支持动作的排序。
- `alpha=0.5` 时，具有完整分数的 214 个已支持动作样本 Top-1 从
  `68/214 = 31.78%` 提升到 `185/214 = 86.45%`。
- 自然表达组 Top-1 从 `30/91 = 32.97%` 提升到 `76/91 = 83.52%`。
- 130、135 的错误 Top-1 次数分别从 67、10 降至 8、1；它们自身的明确正例仍保持
  Top-1。
- `UNSUPPORTED` 不能和普通动作共用同一个校准系数。统一使用 `alpha=0.5` 时，
  `UNSUPPORTED` 在 10 个不支持动作中召回为 0。
- 给 `UNSUPPORTED` 单独设置较小系数可以恢复召回，但误拒绝仍然明显。例如
  `action_alpha=0.6, unsupported_alpha=0.2` 时，不支持召回为 7/10，但会把 21 个
  已支持样本误判为不支持。

因此，背景校准适合用于“已支持动作之间的排序”；`UNSUPPORTED` 应继续使用独立的
decision gate、margin 或分类阈值，不能直接依赖统一候选排名。

## 数据与方法

### 原始分数

- 服务：`ws://127.0.0.1:18004/v1/session/realtime`
- 会话隔离：每个 case 单独创建 session
- 总用例：238
- 完成：238
- 运行错误：0
- 每个进入 scorer 的 turn 保存全部 118 个候选分数
- 有完整分数的 turn：227
- 含期望候选且可离线重排的 turn：224
- 因 action decision gate 未进入候选评分、但具有期望候选的 turn：8

原始全候选分数保存在：

`reports/text_action_recall_18004_full_scores.json`

### 为什么没有直接使用 neutral 文本

额外采集了 40 条没有身体动作要求的普通文本。40/40 均被 action decision gate 正确
判定为 `body_not_requested`，因此候选评分数为 0。该结果可用于验证门控，但无法用于
估计候选 scorer 的 neutral baseline。

原始门控结果保存在：

`reports/text_action_neutral_baseline_18004.json`

### 本次使用的背景基线

使用 117 条 canonical action 请求作为平衡背景，每个动作恰好一个明确正例。计算候选
`i` 的背景均值时排除它自己的正例：

```text
b_i = mean(score_i(x_j)), j 的期望动作不是 i
```

这样得到的是候选在“其他动作请求”中的平均吸引力，包含：

- candidate ID/token 先验；
- 描述过于通用造成的语义先验；
- 与其他动作之间的系统性混淆。

它不是纯粹的 token PMI，但比把“没有动作请求”的文本强行送入 scorer 更符合当前线上
链路。

采用中心化校准：

```text
centered_bias_i = b_i - median(b)
calibrated_i = raw_i - alpha_i * centered_bias_i
```

中心化在所有候选使用相同 `alpha` 时不改变排序；它允许为 `UNSUPPORTED` 设置独立系数。

## 重点候选的背景先验

| 候选 | 背景 mean_logprob | centered bias | 先验排名 |
| --- | ---: | ---: | ---: |
| `UNSUPPORTED` | -5.6804 | +2.9006 | 1 |
| 130 自然呼吸起伏 | -6.6493 | +1.9318 | 3 |
| 135 连续摇头 | -6.6570 | +1.9241 | 4 |

三者确实都是全局高先验候选。`UNSUPPORTED` 的背景吸引力最强，但它在真正不支持动作上
仍然经常只排第二，说明其原始分数无法简单用一个全局阈值与普通动作分开。

## 统一 alpha 回放

下表只统计有完整候选分数的 224 个带期望候选用例：

| alpha | 总 Top-1 | 总 Top-3 | 已支持动作 Top-1 | 不支持召回 | 已支持误判 UNSUPPORTED |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 68/224，30.36% | 154/224，68.75% | 68/214，31.78% | 0/10 | 64 |
| 0.25 | 172/224，76.79% | 208/224，92.86% | 172/214，80.37% | 0/10 | 2 |
| 0.50 | 185/224，82.59% | 203/224，90.63% | 185/214，86.45% | 0/10 | 0 |
| 0.75 | 171/224，76.34% | 192/224，85.71% | 171/214，79.91% | 0/10 | 0 |
| 1.00 | 154/224，68.75% | 177/224，79.02% | 154/214，71.96% | 0/10 | 0 |

完整 PMI 强度 `alpha=1` 已经过度校准；当前数据中 `alpha=0.5` 对已支持动作最好。

### 自然表达组

| alpha | Top-1 | Top-3 |
| ---: | ---: | ---: |
| 0.00 | 30/91，32.97% | 60/91，65.93% |
| 0.25 | 71/91，78.02% | 85/91，93.41% |
| 0.50 | 76/91，83.52% | 86/91，94.51% |
| 0.75 | 71/91，78.02% | 81/91，89.01% |
| 1.00 | 64/91，70.33% | 75/91，82.42% |

## 独立 UNSUPPORTED 系数

| action alpha | UNSUPPORTED alpha | 已支持 Top-1 | 不支持召回 | 不支持 precision | 已支持误拒绝 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.55 | 0.15 | 171/214，79.91% | 8/10，80% | 8/37，21.62% | 29 |
| 0.60 | 0.20 | 175/214，81.78% | 7/10，70% | 7/28，25.00% | 21 |

即使使用独立系数，`UNSUPPORTED` precision 仍然过低，所以暂不建议把这组参数直接上线。

## 建议

1. 下一步优先实现 shadow 模式，只记录 `raw_score`、`background_bias` 和
   `calibrated_score`，不改变线上动作。
2. 真实动作候选可先观察 `alpha=0.5`；保持 130、135 的明确正例和自然表达回归。
3. 将 `UNSUPPORTED` 从普通动作重排中拆出，使用 action decision、类别支持范围以及
   `UNSUPPORTED - best_supported` margin 联合判定。
4. 增加更多不支持动作与近边界支持动作。目前不支持样本只有 10 条，不足以确定生产阈值。
5. 若要测纯 candidate ID/token 先验，应执行“动作定义与候选 ID 随机置换”实验；本次背景
   基线还包含语义通用性，不应命名为纯 token PMI。

## 产物

- 全分数评测器：`scripts/evaluate_text_action_recall.py`
- 离线校准脚本：`scripts/calibrate_text_action_scores.py`
- neutral 门控用例：`tests/unit_test/fixtures/realtime_text_action_neutral_baseline_cases.json`
- 全候选原始分数：`reports/text_action_recall_18004_full_scores.json`
- neutral 门控原始结果：`reports/text_action_neutral_baseline_18004.json`
- 完整校准结果：`reports/text_action_recall_background_calibration_20260918.json`
