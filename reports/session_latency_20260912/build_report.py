from pathlib import Path
import json, statistics, csv
P=Path(__file__).resolve().parent
rs=json.loads((P/'turns.json').read_text()); summary=json.loads((P/'summary.json').read_text())
def fmt(x):return '—' if x is None else f'{x:.1f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in row)+' |' for row in rows])
intro='''# 两个会话的逐 turn 延迟分析

分析日期：2026-09-12（Asia/Shanghai）。本报告只分析日志和代码，未修改推理实现、配置或重启服务。

结论：应先消除纯表情、静默自动轮次对无关推理的等待，再优化短回复路径与 child 前缀缓存。当前用户轮次的动作平均约 0.76–0.93 秒，音频首包约 1 秒；其中可直接观察到的无关等待达到数百毫秒。不能把这些潜在收益简单相加，也不能把不同会话当作严格 A/B 测试。

## 1. 样本与计时口径

- `sess_dee23864`：47 个 start/commit/completed turn，18:44:13–19:00:20；用户 21、主动 26，其中 action_finished 18。服务 PID 1515362，表情融合修复部署前。
- `sess_939ac65d`：7 个 start/commit/completed turn，19:14:17–19:16:05；用户 3、主动 4，其中 action_finished 2。服务 PID 1643329，表情融合修复部署后。
- 共 54 个已记录 turn，全部有动作 ready 事件和完成记录；14 个有表情事件、30 个有音频。turn 编号不连续，以真实 start/commit 日志为准，不补造缺失编号。
- 数据来自 `logs/realtime/2026-09-12/{18,19}` 的 lifecycle/performance/protocol/action/reply/diagnostic JSONL，并用 `logs/sglang-omni.log` 中的 commit payload 补充用户文本。`events.jsonl` 是分析快照；后续新增 turn 不在本次样本中。
- 耗时使用同机 `monotonic_ns` 差值。开始指服务端收到 `turn.start`，commit 指收到 `turn.commit`；动作/表情终点是对应 WebSocket 发送日志，音频终点是首个 `response.audio.delta` 发送日志。
- 这是**服务端发送延迟**，不包含客户端录音开始前的时间、网络传输、浏览器解码/缓冲、实际播放或数字人渲染。动作 ready 也不代表动作已经完成播放。
- 无表情事件、无音频事件记为“—”，不记为 0。`not_required` 的动作 ready 是禁止身体动作的协议结果，不是执行动作。
- 汇总延迟均从 commit 起算；逐 turn 表同时提供 start 和 commit 两种口径。P95 使用线性插值，小样本（尤其新会话 3 个用户 turn）只作描述。
- 两个会话的角色指令哈希、动作 profile 哈希一致，但实际请求不同、缓存状态不同、纯短回复 prompt 长度也不同，无法从均值差异推断修复使性能变好或变坏。

## 2. 汇总

以下每个单元格为 **有效样本数 / 平均 / P50 / P95 / 最大**，耗时单位 ms。
'''
agg=[]
for sid in ['sess_dee23864','sess_939ac65d']:
 for group,zh in [('all','全部'),('user','用户'),('proactive','主动'),('expression_only','纯表情'),('action_finished','动作结束自动轮次')]:
  s=summary[sid+'/'+group];row=[sid,zh,str(s['turns'])]
  for k in ['commit_action_ms','commit_expression_ms','commit_audio_ms']:
   v=s[k];row.append('—' if not v['n'] else f"{v['n']} / {v['mean']:.1f} / {v['p50']:.1f} / {v['p95']:.1f} / {v['max']:.1f}")
  agg.append(row)
intro+= '\n'+table(['会话','分组','turn 数','动作 ready','表情 ready','音频首包'],agg)+'\n'
intro+='''
自动轮次占比很高，直接使用全会话均值会低估用户体感延迟。文本 turn 的 start→commit 通常约 1 ms；新会话唯一音频输入 `turn_0ca1ce49_20` 的输入阶段为 **1327.1 ms**，因此 start→动作为 **2326.1 ms**、start→音频为 **2325.4 ms**，而 commit→动作/音频仅 **999.0/998.3 ms**。不能把约 1.3 秒输入阶段当成模型推理延迟。

## 3. 逐 turn 完整清单

单位 ms。`C`=从 commit 起；`S`=从 start 起。自动轮次用 trigger 代替空用户文本。表中“表情就绪”是内部推理结束，“表情发送”是客户端事件发送。动作标签使用当前目录解释 ID，完整原始 ID 与结果状态见 CSV。
'''
parts=[intro]
for sid in ['sess_dee23864','sess_939ac65d']:
 rows=[]
 for r in rs:
  if r['session']!=sid:continue
  rows.append([r['turn'],r['text'] or ('[音频输入]' if r['audio_chunks'] else '['+str(r['trigger'])+']'),r['action_label'],r['expression_label'] or '—',fmt(r['ingest_ms']),fmt(r['category_compute_ms']),fmt(r['child_compute_ms']),fmt(r['commit_performance_ready_ms']),fmt(r['commit_action_ms'])+'/'+fmt(r['start_action_ms']),fmt(r['commit_expression_ms'])+'/'+fmt(r['start_expression_ms']),fmt(r['commit_audio_ms'])+'/'+fmt(r['start_audio_ms'])])
 parts.append('\n### '+sid+'\n\n'+table(['turn','输入/触发','动作','表情','输入阶段','Category计算','Child计算','表情/表现就绪 C','动作 C/S','表情发送 C/S','首音频 C/S'],rows))
parts.append('''

## 4. 链路与关键等待

```mermaid
flowchart LR
  S[turn.start] --> I[输入积累] --> C[turn.commit]
  C --> R[语言/历史路由]
  R --> G[短回复生成与校验 或 普通回复]
  C --> K[Category]
  K --> D[Child]
  C --> P[表现控制: scope/表情/TTS语气]
  G --> T[TTS文本队列]
  P --> T
  T --> PCM[TTS首PCM]
  D --> F[动作/表情融合与回复放行]
  P --> F
  F --> V[表情与动作 ready]
  PCM --> B[待放行音频缓冲]
  F --> B
  B --> A[音频首包发送]
```

部分系统伴随类别在 Category 后还等待实际回复前缀才能进入 Child。上图省略此条件边；它在“笑一个”的样本中发生。回复、Category、表现控制虽是并行协程，但共享评分准入与 GPU，不能将各分支 elapsed_ms 相加作为总延迟。

### 4.1 纯表情：正确性已修复，等待仍存在（最高优先级）

新会话 `turn_1022818d_13`（“笑一个”）的真实时间线：

| 事件 | commit 后 ms |
|---|---:|
| 语言路由完成 | 201.9 |
| Category 评分完成 | 364.0 |
| 表现控制完成，已确定 expression_only + 微笑 | 419.2 |
| 短回复校验通过，可用文本出现 | 525.1 |
| 系统伴随类别等待回复后完成路由，Child 开始 | 525.7 |
| TTS 首文本 append | 525.9 |
| TTS 首 PCM 到达并缓存 | 697.2 |
| Child 评分完成 | 1019.9 |
| 首音频发送（放行缓冲） | 1021.2 |
| 表情发送 / no_action ready | 1022.6 / 1022.6 |

这里身体动作最终已被正确禁止，但仍计算了 79 个 child 候选。**表情就绪→发送额外等 603.4 ms；首 PCM→发送额外等 324.0 ms。**前一轮修复只改变融合结果，没有提前结束动作依赖。

旧会话 11 个纯表情请求，表现控制平均 476.1 ms，表情发送平均 836.2 ms；额外等待平均 **360.1 ms**，范围 **159.4–636.2 ms**。动作融合后到发送通常不到数毫秒，因此主要等待在 Child/路由之前，不在 JSON 序列化或网络 send 调用。

优化方式：表现控制确认 expression_only 且表情输出开启后，立即构造 not_required/no_action、发送表情并放行合法回复。身体分支的结果不再是对外发送的前置条件；已提交的分支需通过现有请求取消机制清理，避免泄漏 GPU 工作。`both` 必须保留双通道原子性；audio-only 等没有表情输出的模式维持原有语义。

在保持其他时点不变的反事实估算下，新样本表情可从约 1023 ms 接近 419 ms，音频可从约 1021 ms 接近 697 ms；这是日志中可消除等待的量级，**尚不是优化后的实测结果**。此外提前取消无用评分可能改变 GPU 竞争，需回放测量。

### 4.2 自动 action_finished：动作已走快速路径，表现控制仍占住约 117 ms

20 个此类 turn 没有文本、没有音频，也没有表情事件。动作 Category/Child 评分已经跳过，选择静默伴随动作，但最终仍等待 48 候选的表现控制。旧会话平均动作 ready 116.7 ms、新会话 121.0 ms。典型轮次 Category ready 在约 0.8 ms，表现控制在约 113 ms，然后约 0.2 ms 内发送动作。

优化方式：严格限定在空输入、proactive、action_finished 且本轮不需要语音/表情的协议路径，使用默认表现控制，不发模型评分请求。可消除约 110 ms 等待并减少 20 次无效评分。idle_timeout、session_enter 和有文本的主动轮次仍需单独处理。不能仅凭“proactive”一刀切。

### 4.3 音频首包：短回复准备通常比 TTS 合成本身更慢

旧会话 17 个有音频的用户 turn：commit→首 TTS append 平均 **812.9 ms**；TTS response.created→首 PCM 平均 **170.5 ms**；commit→首音频平均 **998.7 ms**。新会话 3 个用户 turn 的对应指标为 **663.7 / 169.1 / 1018.0 ms**，后者包含两次明显的动作放行缓冲等待。

旧会话首个“看正面镜头”：路由 201.3 ms → 可用短回复 1158.5 ms → TTS 首音频 1324.8 ms。相同输入后续 7 次，动作 582–595 ms、音频 857–876 ms，说明首次长延迟中有明显冷态影响，但同一输入重复命中的结果不能代表整个动作空间。

23 个 PURE_ACTION turn 仍调用完整角色回复生成，prompt 约 **6097–7554 tokens**，多数最后只输出 3–4 tokens 的 Okay./好的。之后 19 次调用语义校验评分，平均 **93.5 ms**。对已有可用短回复的 19 个 turn，路由结束→可用文本为 **323–957 ms**，其中混合了生成、共享调度与校验，当前日志不能完整分离纯模型耗时。

优化方式：为纯动作/表情请求建立单独的短回复策略。若产品允许仅做表情，可以显式选择无口头回应，省去回复与 TTS；若必须回应，可评估按 locale/角色配置的小型确认语集合，或明显缩短独立短回复 prompt。生成式短回复若保留语义不确定性，不应直接删除校验。可信模板路径可避免再跑通用语义校验。预生成/缓存音频还需按 voice、语言和语气隔离，不能把中性音频复用给所有表情。

### 4.4 Child 冷前缀与候选规模：身体动作请求的主要优化空间

24 个用户 turn 的实际评分均值如下。各列存在重叠或未列出的协调时间，**不是可直接相加的分解式**。

| 阶段 | 总 elapsed | 准入槽等待 | 预处理 | prefix prefill | suffix 批次耗时 | 平均候选数 |
|---|---:|---:|---:|---:|---:|---:|
| Category | 377.0 | 0.01 | 82.9 | 97.4 | 107.6 | 51 |
| Child | 377.7 | 2.95 | 47.7 | 191.4 | 70.3 | 49.3 |
| 表现控制 | 460.5 | 209.2 | 7.5 | 91.3 | 75.0 | 48 |
| 语言/历史路由 | 210.8 | 0.02 | 6.9 | 87.6 | 104.6 | 4 |

Child 中 12 次 parent cache ratio=0，平均耗时 **479.5 ms**、prefill **300.6 ms**；另 12 次命中，平均 **275.9 / 82.1 ms**。这两组的候选、输入并不完全匹配，约 204 ms 差值是优化线索，不是因果收益承诺。Category 用户 prompt 约 9.5k tokens，平均已有约 93% parent token 命中；优先级低于 Child 的冷态问题。

具体例子：`turn_6d3a0974_25` 的 Child 726.0 ms，79 候选、2 批、prefix prefill 404.3 ms、suffix 160.9 ms，是样本中动作最慢的一轮。它本身为纯表情，优先应不等待该分支。真实身体请求 `turn_337d2f9b_62` 的 Child 492.0 ms、零命中；后续同文 `turn_7d9ce243_65` 仍零命中，但候选数从 43 变为 10，Child 降为 363.0 ms——不能把此变化归因于缓存。

代码中多类别 child namespace 含类别组合、locale、turn_origin 和完整 prompt hash，不同组合/排列及指令前缀会分裂缓存。后续应统计命中失败与这些维度的对应关系，再决定是否为稳定前缀预热，或将候选身份/固定描述与每轮信息拆开。不要仅去掉 hash，否则会破坏缓存正确性。较大候选集合可测试分批策略，但不能为延迟任意删掉正确动作。

### 4.5 评分准入：表现控制在用户轮次平均排队 209 ms

客户端评分日志的 `action_slot_wait_ms` 显示用户表现控制平均等待 **209.2 ms**，而 Category 几乎不等。当前 route/category 先提交，表现控制后提交；日志显示准入容量 2。代码也明确将 route/category 作为优先分支。

这对身体请求可能合理，但表情就是主输出时，表现控制也成了关键路径。先落地纯表情提前结束和静默快速路径，再测试表现控制优先级或合并 scope 与语言路由分类。合并请求必须保持当前意图分类质量，不能根据本样本推定可以无损。不要先盲目增大容量：GPU 仍共享，route 只有 4 候选也有约 105 ms suffix wall time，更多并发可能转移排队而非消除排队。

### 4.6 主动说话的表现控制等待约 63 ms

10 个 session_enter/idle_timeout 有语音的 turn，首文本出现后约 61–69 ms 才发送给 TTS，与表现控制结果到达对齐。这里是语气指令 future 的真实门槛。若产品允许这类固定主动语句使用预设语气，可单独评估省去模型表现控制；不能将此策略推广到用户明确要求的表情。

## 5. 没有音频、日志口径与观测限制

- 24 个无音频 turn 中，20 个是 action_finished 静默轮次；另 4 个是短回复被校验规则拒绝，不是 TTS 超时：
  - `turn_6d3a0974_25`：`*轻笑一声*`，markup_or_stage_direction。
  - `turn_95e18fe7_36`：`*无动作*`，markup_or_stage_direction。
  - `turn_7002c5a1_49`：长协议式回应，too_long。
  - `turn_dbadcc3c_52`：`[Silence]`，markup_or_stage_direction。
- 本次 session 关联日志未发现 warning/error，54 个 turn 都完成。正常完成不等于语义正确；旧会话附带错误动作的问题仍属于质量问题。
- `prefix_cached=true` 不能表示整个 prompt 命中：例如新会话首个 Category 的 parent cache ratio=0。应使用 `parent_radix_cached_token_count`、`parent_computed_token_count` 和 `parent_cache_hit_ratio`。
- `generation_ms` 在 `_run_pure_action_short_reply` 等 `_run_provided_reply` 返回后才记录，包含后续处理，不是纯模型生成耗时；本报告没有把它用于纯模型耗时结论。
- `tts_first_audio_received` 在 `audio_sink` 返回后才记录。直接转发时 WS 首包日志会比它早约 0.03–0.06 ms，这是埋点顺序，不是负网络耗时；在缓冲路径则仍能看出约 230/250/324 ms 的放行等待。
- 流式 TTS 的 response.created 可以早于文本 commit；因此 `commit_to_created_ms` 在 CSV 中允许为负，它不是 provider 排队时长。TTS 端模型排队、首文本聚合和推理内部耗时需要 provider 自身日志补充，不能从单个 created 事件硬拆。
- `tts_playable_250ms_ready` 是服务端累计 PCM 大小，既不保证音频已发给客户端，也不是浏览器播放时间。
- 当前评分样本最大单阶段 prompt **9742 tokens**，低于当前配置 `max_seq_len=60000`。这不是累积 token 数；本次没有上下文长度上限导致的延迟证据。历史技能文档里的 20000 是旧配置，不能套用。

## 6. 建议实施顺序与验收

| 顺序 | 改动/实验 | 数据支持的机会 | 验收重点 |
|---|---|---|---|
| P0 | expression_only 提前融合/发送/放行，取消无用身体评分 | 旧样本表情额外等均值 360 ms；新“笑一个”等 603 ms、音频缓冲 324 ms | 纯表情不执行动作；both 保持原子性；取消不泄漏请求；从表现就绪到发送不再依赖 child |
| P0 | 空 action_finished 跳过表现控制 | 20 次无输出意义的评分，每轮约 110 ms | 静默、不改变表情；动作池约束/避免重复不变；该路径模型评分次数为 0 |
| P1 | 独立短回复策略（无口头回应/可信短语/精简 prompt） | 6–7.5k prompt 换几 token，额外校验约 94 ms，4/23 回复被丢弃 | 回复质量、语言、人设与表情一致；明确产品是否需要口头确认；比较首音频而非仅生成时间 |
| P1 | Child 固定前缀/组合缓存，冷态与暖态分别评估 | 冷/暖 Child 平均 480/276 ms | 输出候选一致；不错误共享缓存；报告 computed tokens 与首次请求延迟 |
| P2 | route/performance 联合分类或调整准入优先级 | 表现控制用户轮次平均槽等待 209 ms | 表情/动作/语言分类准确率与全部渠道 P95 一起比较，不能只改善一支 |
| P2 | 补充 provider 与前端播放埋点 | 当前只能定位到服务端发送/PCM 门槛 | request/turn/trace 贯通，区分 GPU排队、TTS聚合、浏览器缓冲与播放 |

建议先单独实施两个 P0，再用相同输入集回放冷启动与热缓存，之后再测 P1；否则同时改 prompt、并发和调度会失去因果归因。回放应覆盖纯表情、身体动作、both、普通语言、主动欢迎、idle、action_finished、语音输入与失败取消场景。保存输出 ID/状态以及 start/commit→各 ready/首音频的分位数，不以重复“看正面镜头”的热缓存代替完整评估。

建议新增纯短回复的 submitted / raw_first_token / raw_done / validation_begin/end，表现控制 queued/admitted，PCM received-before-sink / buffered / released，以及前端 first_audio_received / scheduled / playback_started。当前日志足以确定上述主要等待，但尚不足以给出每个 GPU kernel 或 TTS provider 内部的精确归因。

## 7. 代码定位与附件

- `sglang_omni/serve/realtime/turn_pipeline.py:1102`：route/category/performance 启动顺序；约 1350–1518：等待表现控制与动作、融合、放行回复、发送表情/动作。
- `sglang_omni/serve/realtime/performance/fusion.py:41`：expression_only 覆盖动作的正确性规则；目前发生在动作完成后。
- `sglang_omni/serve/realtime/reply/provisional.py:334`：回复 promote 与缓冲音频放行。
- `sglang_omni/serve/realtime/reply/generation.py:939`：纯短回复生成、校验与记录时点。
- `sglang_omni/serve/realtime/embedded_tts.py:390`：TTS 等待表现指令；约 543 行：audio_sink 后才记首 PCM。
- `sglang_omni/serve/realtime/action/category.py:749`：child 前缀 namespace 与组合 hash。

附件：[逐 turn CSV](turns.csv)、[逐阶段 CSV](stages.csv)、[分组统计 JSON](summary.json)、[逐 turn JSON](turns.json)、[可重跑分析脚本](analyze.py)、[关键样本时间图](critical_paths.svg)。所有表格可由当前事件快照重建；CSV 保留毫秒小数，报告展示值四舍五入。
''')
parts.append('\n\n## 8. 跨仓库实施方案\n\n详见 [表情提前输出、音频放行与 Child 缓存优化实施方案](implementation_plan.md)，包含 Child 每次请求的身份/命中日志、D_video_call 的纯表情音频协调与前端状态修正，以及分批回归与验收。该文档是待实现方案，不代表运行服务已经应用这些优化。\n')
(P/'report.md').write_text('\n'.join(parts))
