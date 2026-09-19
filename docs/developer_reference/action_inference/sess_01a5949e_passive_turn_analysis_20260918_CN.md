# sess_01a5949e 被动 turn 分析

分析时间：2026-09-18（Asia/Shanghai）

## 结论

该会话共包含 3 个来自评论输入的被动用户 turn，全部为纯文本输入；没有音频 chunk，也没有图片帧。三轮均完成，没有触发会话重连或服务不可用。

动作链路的延迟已经明显低于 400–450 ms 目标：sglang-omni 从 `turn.commit` 到发送 `turn.action.ready` 为 252.9–281.9 ms，平均 267.0 ms。117 个真实动作加 1 个 `UNSUPPORTED` 和 24 个 decision label，共 142 个物理候选，三轮均以单个 `[142]` batch 执行，没有拆批。

本会话的主要问题是准确性和稳定性，而不是性能：

- 第一次“`双手比心`”正确选择 285“`双手比心`”。
- “`点个赞`”错误选择 130“`自然呼吸起伏`”；正确候选 270“`点赞`”只落后 0.0983 mean logprob。
- 第二次完全相同的“`双手比心`”也错误选择 130；正确候选 285 只落后 0.1143 mean logprob。
- 相同输入第一次和第三次结果翻转。动作 scorer 已明确清空完整跨 turn 历史；本次扰动主要来自每个后续用户 turn 都会加入的“最近一次用户触发动作”指代锚点。130 的高先验仍足以覆盖目录中“不得替代明确动作请求”的限制。

## 会话配置

- 回复 locale：`en-US`
- 动作 locale：`zh-CN`
- 输出：text、audio、expression、action
- 真实动作数：117
- direct single 评分：118 个具体候选（含 `UNSUPPORTED`）
- 同批 decision label：24
- 每轮物理候选总数：142
- 数字人服务：禁用（`dh_enabled=false`）

因此中文动作命令使用中文动作 prompt 评分，但短回复为英文（`Got it.` / `Sure!`），符合本次 locale 配置。

session.start 已完成 intent 预热和多个 action prefix 预热。三轮实际评分均记录 `prefix_cached=true`、`token_blueprint_status=hit`、候选前缀重算为 0。

## 逐轮结果

下表所有时间均相对 sglang-omni 收到 `turn.commit`，单位为 ms。

| Turn 后缀 | 用户输入 | 输入模态 | 最终动作 | 正确性 | 主评分 | 动作就绪 | intent | performance 就绪 | 首音频 | 整轮完成 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e37a0e39_2` | 双手比心 | 文本 | 双手比心 285 | 正确 | 250.6 | 252.9 | 528.9 | 725.0 | 2341.3 | 2765.6 |
| `da7b8f42_4` | 点个赞 | 文本 | 自然呼吸起伏 130 | 错误，应为点赞 270 | 264.2 | 266.2 | 550.8 后 fallback | 776.7 | 1532.1 | 1983.2 |
| `c22f69cb_6` | 双手比心 | 文本 | 自然呼吸起伏 130 | 错误，应为双手比心 285 | 279.8 | 281.9 | 561.3 | 762.0 | 1980.7 | 2174.3 |
| **平均** | — | — | — | 1/3 正确 | **264.8** | **267.0** | **约 547** | **754.6** | **1951.4** | **2307.7** |

这里的“动作就绪”是早期独立发布的 `turn.action.ready`；`action_child_ready` 在 performance-control 之后才记录，不代表客户端必须等待到 725–777 ms 才获得动作。

## 动作关键路径

| Turn 后缀 | 预处理 | scheduler wait | prefix prefill | 候选构建 | 候选排队 | suffix `[142]` | 主评分总耗时 | 评分完成到动作发布 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e37a0e39_2` | 31.8 | 30.0 | 52.4 | 7.6 | 28.3 | 153.5 | 250.6 | 2.3 |
| `da7b8f42_4` | 29.0 | 31.1 | 66.4 | 7.1 | 29.1 | 156.7 | 264.2 | 2.0 |
| `c22f69cb_6` | 30.1 | 35.2 | 75.7 | 8.5 | 33.2 | 161.1 | 279.8 | 2.1 |
| **平均** | **30.3** | **32.1** | **64.8** | **7.7** | **30.2** | **157.1** | **264.8** | **2.1** |

这些内部统计存在异步重叠，不能把所有列简单相加得到主评分总耗时；尤其候选构建在 prefix 执行期间准备。

关键观察：

- 候选构建已经降到 7.1–8.5 ms，不再是主要瓶颈。
- scheduler wait 约 30–35 ms，候选 queue wait 约 28–33 ms，仍各有约 30 ms 的调度窗口。
- suffix forward 约 154–161 ms，是主评分中最大的单项成本。
- prefix prefill 从 52.4 ms 增至 75.7 ms，对应动态计算 token 从 107 增至 252；缓存前缀固定复用 11,681 token。
- 前缀命中率为 97.89%–99.09%，候选前缀重算为 0。
- 三轮都是单个 `[142]` 物理 batch，不存在显存不足导致的拆批。

## 候选准确性

### turn_e37a0e39_2：双手比心

具体动作前四名：

| 排名 | candidate | 动作 | mean logprob |
| ---: | --- | --- | ---: |
| 1 | 285 | 双手比心 | -4.6286 |
| 2 | 130 | 自然呼吸起伏 | -4.7544 |
| 3 | 135 | 连续摇头 | -4.7961 |
| 4 | 000 | UNSUPPORTED | -4.8342 |

虽然结果正确，但 285 仅领先 130 约 0.1259，优势很小。

decision gate 判断：body=`perform`、face=`perform`、reaction=`none`、visual=`V00`。body gate margin 为 0.9375，因此允许动作立即发布。

### turn_da7b8f42_4：点个赞

具体动作前三名：

| 排名 | candidate | 动作 | mean logprob |
| ---: | --- | --- | ---: |
| 1 | 130 | 自然呼吸起伏 | -2.8818 |
| 2 | 270 | 点赞 | -2.9801 |
| 3 | 000 | UNSUPPORTED | -3.1483 |

正确候选只落后 0.0983。decision gate 同时给出 body=`perform`、reaction=`greeting`，其中 greeting 与“点个赞”不符；face 也被判为 `perform`。body gate margin 仍有 0.625，因此错误的 130 在 266.2 ms 被当作 supported 发布。

完整 intent 在约 550.8 ms 生成后触发校验回退：

```text
explicit body task cannot contain implicit reaction
```

原始 intent 内容未被完整记录，因此不能断言具体生成了哪个隐式 reaction；但可确认它输出了互相冲突的“明确身体任务 + 隐式 reaction”。该 fallback 没有使 turn 失败，也没有触发重连，只使后续 reply/history 路径使用 fallback。

### turn_c22f69cb_6：双手比心

具体动作前三名：

| 排名 | candidate | 动作 | mean logprob |
| ---: | --- | --- | ---: |
| 1 | 130 | 自然呼吸起伏 | -3.0571 |
| 2 | 285 | 双手比心 | -3.1714 |
| 3 | 000 | UNSUPPORTED | -3.3109 |

同一句“`双手比心`”在第一轮由 285 领先 130，在第三轮却变为 130 领先 285 约 0.1143。两轮目录和静态 action prompt 相同。当前代码已将完整 action history、历史音频和历史图片清空；主要差异是第三轮动态 prefix 中额外加入了“最近一次用户触发动作”锚点，而该锚点正是第二轮误选后立即记录的 130：

- 第一轮 prefix token 11,788，动态计算 107 token。
- 第三轮 prefix token 11,933，动态计算 252 token。

这说明完整对话历史虽然已经隔离，但最近动作锚点仍足以改变候选排序。对于当前 turn 的明确、精确动作命令，不应无条件注入该锚点，更不应把仅完成推理、尚未收到客户端执行确认的动作作为后续指代事实。

## 决策和下游链路

三轮 body gate 均允许动作独立发布，且均记录：

- `waited_for_performance=false`
- `waited_for_reply=false`
- `support_status=supported`

因此 full intent 的约 529–561 ms、performance-control 的约 195–224 ms、纯动作短回复生成以及 TTS 都没有阻塞早期动作就绪。

三轮最终都产生一个下游 motion plan，但均立即记录：

```text
prompt was not sent for step 'phase_1'
```

该会话同时明确记录 `dh_enabled=false`，所以这是禁用数字人执行端后的预期下游结果，不代表 sglang-omni 推理失败，也不纳入动作推理耗时。

## 建议优先级

1. **先抑制 130 的错误高先验。** 130 只应在没有明确动作目标，或用户明确要求自然待机时可选。当前文字限制已经存在，但不足以改变模型排序；应考虑在 body=`perform` 且存在高置信明确动作时从候选集合硬排除 130，或对 130 增加规则性 penalty。
2. **为精确动作命令增加当前 turn 直匹配。** 对“点个赞”“双手比心”等与动作 label/alias 高度一致的命令，可在评分前产生受约束候选集，或者给精确 alias 命中候选确定性加分。
3. **消除最近动作锚点对普通命令的影响。** 完整 action history 已经不传入；还应只在本轮明确出现“刚才那个”“再做一次”等指代时加入最近动作锚点，并且锚点应来自客户端确认执行的动作，而不是服务端刚刚推理出的动作。
4. **修复 decision label 的语义误判。** 三轮都将 face 判为 `perform`，而完整 intent 对可用两轮均判 `has_face=false`；“点个赞”还被判为 greeting。虽然没有阻塞本次身体动作，但会降低门控可靠性。
5. **随后再优化约 60 ms 的调度空隙和约 157 ms 的 suffix。** 当前动作就绪已稳定低于 300 ms，准确性收益优先于继续压缩几十毫秒。

## 日志来源

- `logs/realtime/2026-09-18/00/action_api_2904714_000.jsonl`
- `logs/realtime/2026-09-18/00/performance_api_2904714_000.jsonl`
- `logs/realtime/2026-09-18/00/diagnostic_client_2904714_000.jsonl`
- `logs/realtime/2026-09-18/00/reply_api_2904714_000.jsonl`
- `/data/fanshide/D_video_call/backend/logs/experience/2026-09-18/00.jsonl`
- `/data/fanshide/D_video_call/backend/logs/latency_timeline/2026-09-18/00.jsonl`
