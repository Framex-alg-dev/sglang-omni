# sess_43d0e0d8 全部被动 turn 分析（2026-09-17）

## 范围与口径

以 `turn_origin=user` 为被动轮，并与 D_video_call 的 `user_question` 交叉核对，
本会话共 8 轮，全部来自 comment 文本，没有音频或图像输入。时间段为北京时间
12:59:53–13:00:52。

耗时以 sglang-omni 收到 `turn.commit` 为起点。动作评分是
`action_scoring_completed(stage=single).elapsed_ms`；动作就绪是
`turn_action_ready_sent.after_commit_ms`。首文本取 reply 的
`first_delta_after_commit_ms`，首音频取 `response_first_audio_delta_sent.after_commit_ms`。
整轮完成是服务端 `turn_timing.total_after_commit_ms`，不代表浏览器播放完成。

## 逐轮结果

| turn 后缀 | 用户输入 | 最终动作/表情 | single 评分 ms | 动作就绪 ms | 首文本 ms | 首音频 ms | 整轮 ms |
|---|---|---|---:|---:|---:|---:|---:|
| `37799bb1_2` | 你好 | 动作微笑154；表情微笑154 | 649.2 | 1065.4 | 2077.5 | 2266.3 | 2800.5 |
| `24718dc8_4` | 大笑一个 | 动作不需要；表情大笑155 | 862.3 | 1191.0 | — | — | 1293.3 |
| `89eb84ab_6` | 今天过得怎么样 | 微笑154 | 663.6 | 941.3 | 1583.8 | 1825.6 | 2821.4 |
| `99d96f1b_8` | 给我打个招呼 | 动作微笑154；表情微笑154 | 680.0 | 992.5 | 1621.1 | 1815.5 | 3297.2 |
| `99297082_10` | 给我比个心 | 双手比心285；表情微笑154 | 659.2 | 880.7 | 1746.7 | 1907.4 | 2055.0 |
| `94d6cea2_12` | 比个数字一 | **扮鬼脸158** | 637.9 | 859.6 | 1735.7 | 2194.7 | 2570.4 |
| `a7bc0d7d_14` | 比个数字二 | **害怕159**；表情微笑154 | 647.1 | 871.4 | — | — | 1183.1 |
| `633ca4bc_16` | 比个数字三 | **委屈160** | 680.1 | 910.7 | 1791.2 | 2420.5 | 2422.0 |

8 轮均为服务端 `completed`。single 评分均值 684.9 ms、中位数 661.4 ms、范围
637.9–862.3 ms；动作就绪均值 964.1 ms、中位数 926.0 ms、范围
859.6–1191.0 ms；整轮均值 2305.4 ms、中位数 2496.2 ms。

## 169 单批与缓存

新配置已实际生效。8 轮 single 评分均为 169 个候选、一个 suffix batch，
`suffix_batch_sizes=[169]`，没有退回 `[64,64,41]`。所有请求均
`prefix_cached=true`，复用 12,423 个 token，候选前缀重算为 0。

| turn | 意图解析 ms | 预处理 ms | prefix prefill ms | scheduler wait ms | suffix batch ms | 前缀命中率 |
|---|---:|---:|---:|---:|---:|---:|
| 你好 | 414.3 | 61.9 | 48.9 | 50.2 | 264.6 | 99.16% |
| 大笑一个 | 271.2 | 116.2 | 71.4 | 67.0 | 363.3 | 97.49% |
| 今天过得怎么样 | 275.9 | 63.1 | 72.1 | 51.7 | 281.8 | 97.49% |
| 给我打个招呼 | 310.8 | 62.1 | 72.1 | 50.8 | 273.1 | 97.48% |
| 给我比个心 | 219.9 | 61.5 | 72.1 | 51.8 | 262.8 | 97.48% |
| 比个数字一 | 220.1 | 59.5 | 59.3 | 51.8 | 264.9 | 98.01% |
| 比个数字二 | 222.8 | 63.2 | 62.9 | 52.2 | 262.4 | 97.81% |
| 比个数字三 | 229.0 | 95.5 | 62.6 | 52.8 | 271.3 | 97.89% |

single 的 suffix 单批耗时均值 280.5 ms、中位数 268.1 ms。作为参考，上一会话
`sess_8ff3c405` 使用 `[64,64,41]` 时，suffix 三批累计中位数为 260.4 ms，single
评分中位数为 631.9 ms；本会话分别为 268.1 ms 和 661.4 ms。两个会话输入集合和
运行状态并不完全相同，不能当作严格 A/B，但现有数据没有显示 169 单批加速。

两个会话重合的四个输入中，single 评分变化为：

- 大笑一个：610.6 → 862.3 ms（+251.7 ms；本轮还受到 expression-only 的
  performance 分支长等待影响）
- 今天过得怎么样：639.0 → 663.6 ms（+24.6 ms）
- 给我打个招呼：621.2 → 680.0 ms（+58.8 ms）
- 给我比个心：624.9 → 659.2 ms（+34.3 ms）

上一会话动作就绪均值 964.5 ms，本会话为 964.1 ms，基本没有变化。当前证据更支持
“169 只是把候选合成一批，没有减少总计算量”，而不是“减少两次分批即可明显降时延”。
8 轮没有 OOM 或评分失败；single 评分日志中的进程已分配显存增量平均约 46.1 MiB，
上一会话约 32.4 MiB。该字段没有重置峰值，不能直接解释为 batch 的完整峰值显存。

会话用户前缀预热 705.6 ms，主动前缀预热 134.8 ms，均成功。用户预热先复用
6,975 个公共 token，再计算 5,465 个 token，建立了后续每轮实际复用的 12,423-token
会话边界。

## 新增的 performance 评分

每轮 single 后还运行一次 12 候选的 `stage=performance` 评分，用于表情和 TTS
表现控制。因此本会话不是“每轮总共只评分一次”，而是每轮 1 次 single 加 1 次
performance。后续稳定轮 performance 通常 149–167 ms；首轮 180 ms。

“大笑一个”的 performance 评分达到 917.2 ms，prefix prefill 134.2 ms、scheduler
wait 186.5 ms。该轮属于 `expression_only`，动作就绪必须等待 performance 给出大笑155，
所以动作就绪升至 1191.0 ms。其他有身体动作的轮次先独立发布 body action，通常不等待
performance 才发送 `turn.action.ready`。

## 动作选择正确性

正确或可接受：

- “大笑一个”正确选择大笑155，且作为表情计划发布。
- “给我比个心”正确选择双手比心285。
- 普通问候/闲聊选择微笑154可作为陪伴表情，但当前 direct single 路径将基础表情也放入
  动作候选池；当意图没有标成 face 时，微笑可能先作为 `phase_1` 身体动作发布，再由
  performance 重复发布为 `facial-expression`。

明显错误：

- “给我打个招呼”仍只选微笑154，没有选择挥手。
- “比个数字一/二/三”应分别选择目录中存在的数字一258、数字二259、数字三260，实际却
  连续选择扮鬼脸158、害怕159、委屈160。结果恰好整体偏移 `-100`，不是普通的近义动作
  混淆，建议优先检查候选 ID 与 score 的对应、single 大批结果聚合以及数字 ID 提示词是否
  诱发了表面数字捷径。仅凭现有日志无法断定是 169 batch 导致；需要用同一提示分别跑
  batch 64 和 169，并记录完整 top-k 分数验证 composition invariance。

## 回复路径

三轮语言请求正常生成英文回复；五轮被判定为纯动作请求。其中：

- “大笑一个”生成原始短回复 `Alright, here's a big laugh for you. 😄`，因 `too_long`
  被丢弃，最终输出空文本、无音频。
- “比个数字二”生成 `Okay, I'll show you the number two.`，也因 `too_long` 被丢弃，
  最终输出空文本、无音频。
- “给我比个心”“比个数字一”“比个数字三”分别保留 `Sure!`、`Got it.`、`Okay.`；
  短回复生成耗时分别为 1068、1609、1402 ms，仍明显慢于动作选择。

因此整轮 1.18–3.30 秒主要还包含回复生成、语义校验和 TTS，不应当作动作推理耗时。

## 执行链：未发送原因已确认

8 个用户轮共创建 12 个动作/表情计划：7 个 `phase_1`、5 个
`facial-expression`。12 次全部得到 `sent=false`，随后全部记录
`prompt was not sent`。

本会话开始时 D_video_call 已明确记录：

- `dh_enabled=false`
- `server_dh_disabled`
- `reason=session_worker_disabled`

因此这次能够确认拒发的直接原因不是 sglang-omni 评分失败，而是数字人 session worker
未启用。服务端的 `action_execution_assumed` 仍只是模型侧历史记账，不是执行成功回执。

## 建议排查顺序

1. 修复数字一/二/三的 `-100` 连续错位。用完全相同的请求做 batch 64/169 对照，并输出
   top-k `candidate_id + mean_logprob`；若 top-k 随 batch 改变，则是批组成一致性问题，若
   两边都选 158/159/160，则重点检查 direct-single 提示词及纯数字 ID 方案。
2. 当前 169 单批没有可见延迟收益。保持 169 前应进行同输入、多轮、同并发 A/B；基于现有
   数据，128 或原 64 更可能是稳妥配置。
3. 启用或恢复 D_video_call 的数字人 session worker，再验证 prompt 实际送达，否则动作
   正确性只能停留在推理层。
4. 修正“打招呼”到挥手的意图/动作匹配，并明确基础表情在 direct single 中应走 body
   还是 facial-expression，避免同一微笑双通道发布。
5. 纯动作短回复仍走约 11.4k-token 请求，且 2/5 被 `too_long` 丢弃；应考虑固定短语或
   更短的专用提示路径。

## 日志证据

- D_video_call 用户输入：`/data/fanshide/D_video_call/backend/logs/experience/2026-09-17/{12,13}.jsonl`
- D_video_call 执行时间线：`/data/fanshide/D_video_call/backend/logs/latency_timeline/2026-09-17/{12,13}.jsonl`
- sglang-omni：`logs/realtime/2026-09-17/{12,13}/{performance,action,lifecycle,reply,protocol}_api_3603166_000.jsonl`

本报告仅分析日志，没有修改运行逻辑或配置。
