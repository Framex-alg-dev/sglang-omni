# avatar_state 图像编码提前执行：实现与 18004 实测

## 结论

在 GPU0、18004 服务上，`avatar_state` 提前编码已生效。相同代码、相同输入条件下，各运行 10 个 turn：

| 指标 | 未提前编码 | 提前编码 | 变化 |
| --- | ---: | ---: | ---: |
| 动作就绪均值 | 509.731 ms | 419.096 ms | **-90.635 ms（-17.78%）** |
| 动作就绪中位数 | 509.118 ms | 418.498 ms | **-90.620 ms** |
| 动作就绪样本 P95 | 522.275 ms | 424.723 ms | **-97.552 ms** |
| 动作就绪范围 | 501.312–522.275 ms | 413.946–424.723 ms | 最大值降低 97.552 ms |
| 评分客户端耗时均值 | 411.676 ms | 261.717 ms | **-149.959 ms（-36.43%）** |
| 动作评分内图像阶段均值 | 149.928 ms | 0.246 ms | **-149.682 ms（-99.84%）** |
| 图像 encoder cache hit | 0/10 | **10/10** | 全部命中 |
| `turn.result` 均值 | 510.187 ms | 484.523 ms | -25.663 ms（-5.03%） |

动作就绪没有获得完整的 150 ms，是因为图像阶段移出关键路径后，动作 suffix 评分更早与完整 intent 解码共享同一张 GPU。该测试中完整 intent 均值由 256.171 ms 增至 482.826 ms；动作仍更早发布，但最终 `turn.result` 会等待完整 intent，因此整轮收益小于动作就绪收益。

## 实现

当 collecting 阶段收到 `avatar_state` 时，服务立即：

1. 复用原有图片预处理任务，得到与正式动作请求完全一致的 RGB payload；
2. 提交只执行 image encoder 的私有预取请求；
3. 使用相同的 `session_id`、`session_instance_id`、图片内容和 `image_roles=["avatar_state"]` 写入现有 encoder cache；
4. 正式动作评分仍构建原请求，只是 image encoder 阶段按同一 key 命中缓存；
5. turn 取消或 avatar 帧被替换时，取消/abort 对应预取请求。

没有替换图片、减少图片或绕过正式动作评分，因此不会因预取本身改变模型输入。功能默认开启，可通过以下环境变量紧急关闭：

```text
SGLANG_OMNI_AVATAR_IMAGE_ENCODER_PREFETCH=0
```

新增的结构化日志事件：

```text
avatar_image_encoder_prefetch_scheduled
avatar_image_encoder_prefetch_submitted
avatar_image_encoder_prefetch_completed
avatar_image_encoder_prefetch_at_commit
avatar_image_encoder_prefetch_cancelled
avatar_image_encoder_prefetch_failed
```

## 严格 A/B 方法

- 日期：2026-09-20
- 服务：`sglang-omni-gpu0.service`，GPU0，端口 18004
- 两组均使用同一份当前代码；每次改变默认开关后重启服务
- 每组 10 个文本 turn，文本固定为“`双手比心`”
- 每个 turn 生成不同的 640×480 JPEG，避免跨 turn 图片缓存污染
- `avatar_state` 在 commit 前 1 秒发送
- 每个 turn 都重新核对服务端 `action_breakdown`
- 未提前编码组 10/10 为 `image_encoder.cache_status=compute`
- 提前编码组 10/10 为 `image_encoder.cache_status=hit` 且 `model_ms=0`

两组 20 个 turn 返回的 candidate ID 均为 `211`，证明 A/B 选择结果一致。不过 `211` 是“手臂交叠”，并非输入所期望的“双手比心” `285`；这是现存动作召回问题，本测试只验证预编码的耗时与结果等价性，不把该输出计为召回正确。

原始结果：

- [未提前编码 A/B 数据](./avatar_prefetch_ab_baseline_18004_20260920.json)
- [提前编码 A/B 数据](./avatar_prefetch_ab_enabled_18004_20260920.json)

## 分阶段均值

| 阶段 | 未提前编码 | 提前编码 | 变化 |
| --- | ---: | ---: | ---: |
| 动作 scorer client | 411.676 ms | 261.717 ms | -149.959 ms |
| prompt preprocessing | 16.023 ms | 14.788 ms | -1.235 ms |
| action image encoder | 149.928 ms | 0.246 ms | -149.682 ms |
| prefix prefill | 94.211 ms | 94.136 ms | -0.075 ms |
| scheduler wait | 22.485 ms | 17.649 ms | -4.836 ms |
| candidate materialize | 7.901 ms | 7.561 ms | -0.340 ms |
| candidate queue wait | 21.188 ms | 16.449 ms | -4.739 ms |
| suffix forward | 138.828 ms | 133.746 ms | -5.081 ms |
| 完整 intent | 256.171 ms | 482.826 ms | +226.655 ms |

`prefix prefill` 基本完全不变，说明主要收益确实来自图像编码缓存，而不是恰好获得了更高的 prefix 命中率。suffix、queue 的几毫秒变化属于调度波动，不应算作该方案的确定收益。

## 预取自身成本和提前量

在 commit 前 1 秒发送 avatar 的 10 个 turn 中：

| 指标 | 均值 | 中位数 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| 预处理＋encoder 预取完成 | 34.646 ms | 31.809 ms | 30.807 ms | 60.464 ms |
| commit 时已提前完成 | 967.172 ms | 970.022 ms | 941.522 ms | 970.947 ms |

该工作在用户仍在说话时完成，所以不计入 commit 后动作就绪关键路径。第一轮 60.464 ms 较高，后续稳定在约 31 ms。

## 边界时序

额外测试 avatar 与 commit 几乎同时到达：

| avatar 提前量 | turn 数 | 动作就绪均值 | 中位数 | 最大 | image cache hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 6 | 436.216 ms | 438.843 ms | 441.443 ms | 6/6 |
| 50 ms | 6 | 423.178 ms | 426.671 ms | 429.425 ms | 6/6 |
| 1000 ms | 10 | 419.096 ms | 418.498 ms | 424.723 ms | 10/10 |

0 ms 测试中，commit 时预取仍为 `submitted`，随后在 commit 后 39.450–43.843 ms 完成；动作评分到达 image encoder 阶段时仍成功命中缓存。相对未提前编码均值，0 ms 情况仍降低约 73.5 ms，但因为预取与 commit 后 intent 发生并发，收益低于正常提前量。

原始边界结果：

- [0 ms 提前量](./avatar_prefetch_zero_lead_18004_20260920.json)
- [50 ms 提前量](./avatar_prefetch_50ms_lead_18004_20260920.json)

## 风险和后续观察

1. **同 GPU 竞争**：图像预取本身约 31–35 ms。如果 avatar 只在 commit 附近到达，会短暂与 intent 竞争；正常 PTT 场景下 avatar 提前约 1 秒到达，此成本被完全隐藏。
2. **完整 intent 变慢**：动作 scorer 提前进入 Thinker 后，当前实测完整 intent 增加约 226.7 ms。动作就绪目标受益，但包含回复的复杂 turn 需要继续观察 TTFT 和整轮完成时间。
3. **缓存占用提前**：没有新增第二份 embedding；只是把同一缓存条目的创建时间提前。条目仍受现有 TTL、容量和 session owner 隔离约束。
4. **视觉路由组合不同**：该命中针对非视觉动作评分的精确 `[avatar_state]` 输入。若正式请求使用 `[avatar_state, user_camera...]`，组合 key 不同，不会错误复用 avatar-only embedding。
5. **准确率**：正式评分输入不变，因此预取不应改变准确率；本次 A/B 的动作结果完全一致，但测试文本本身暴露了 `双手比心 → 211` 的独立召回问题。

## 验证

- 新增 avatar 预取启动/缓存请求单测：通过
- 新增 collecting 阶段取消/abort 单测：通过
- 定向测试：`2 passed`
- 完整 `test_multimodal_session.py`：`345 passed, 11 failed`
- 11 个失败在实施前后相同，属于当前合并代码中既有的 greeting、category gate、background calibration 和 mixed numeric 测试失败；没有新增失败
- `compileall`：通过
- `git diff --check`：通过
- 18004 当前已重启并运行默认开启版本
