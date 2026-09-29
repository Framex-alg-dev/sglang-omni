# 双流式意图 Lane 拆分 TODO

状态：**待实施**

记录日期：2026-09-20

范围：Realtime 被动用户 Turn 的统一意图解析；不修改动作目录、候选数量和动作播放器协议。

## 1. 背景与当前基线

当前统一意图模型以一个流式 JSON 返回视觉、身体动作、语言和反应字段。字段已按动作关键路径调整为：

```text
visual
→ body_intent
→ body_task / face_task
→ speech
→ reaction
→ 其余可选字段
```

最近四个音频动作请求的稳定实测基线为：

| 指标 | 当前耗时 |
|---|---:|
| 身体动作字段就绪 | 约 384 ms |
| 动作候选评分 | 约 339–345 ms |
| 动作最终就绪 | 约 390 ms |
| 完整意图 JSON 完成 | 约 451 ms |

动作候选评分和意图解析已经并行，动作最终就绪时间取两条关键路径的较大值，并包含少量编排与发布开销。当前意图字段仍略慢于候选评分，因此可以继续缩短动作侧意图输出；完成拆分后，主要瓶颈预计转移到候选评分。

## 2. 目标方案

把统一大 JSON 拆成两个独立、并行、持续流式输出的 Lane：

### 2.1 Action Lane

只负责动作与视觉决策：

```json
{
  "visual": "NO_CURRENT_VIEW",
  "body_intent": "perform",
  "body_task": "数字二手势",
  "face_task": null,
  "visual_answer_operation": null,
  "visual_answer_output": null
}
```

消费顺序保持为 `visual → body_intent → body_task/face_task → 视觉可选字段`。普通明确动作在最小权威字段稳定后即可进入动作评分或发布门控，不等待语言 Lane 完成。COPY、视觉识别和视觉算术必须等待对应规范化目标或视觉结果，不能仅凭 `body_intent` 提前发布。

### 2.2 Language Lane

只负责回复与交互语义：

```json
{
  "speech": "none",
  "text": "",
  "speech_independent_of_body": false,
  "reaction": "none"
}
```

语言 Lane 独立流式消费，可尽早启动回复/TTS，但无权新增、修改或取消身体动作。发布前仍需结合最终动作支持状态执行一致性门控，禁止在动作不支持时输出“好的、马上做、没问题”等执行承诺。

### 2.3 执行与调度要求

两条 Lane 必须：

1. 使用同一份冻结的当前 Turn 输入；
2. 共享相同的系统提示、会话提示、音频/图片编码和公共 Prefix KV；
3. 仅在公共 Prefix 后追加各自很短的 Lane 指令；
4. 原子入队并设置 ready barrier，避免被调度成两个独立 Prefill；
5. 各自维护流式解析器、超时、取消和终态；
6. `turn.cancel`、断线和 `session.close` 必须同时终止两条 Lane；
7. 不把空位补齐到最大 batch，实际请求数是多少就只运行多少。

简单发起两个普通生成请求不视为完成本 TODO，因为它可能重复 Prefill、媒体编码和 KV，并与动作候选争抢调度窗口。

## 3. 预计耗时收益

在“共享 Prefix + 原子入队 + 独立流式解析”全部成立时：

| 指标 | 当前 | 预计 | 预计收益 |
|---|---:|---:|---:|
| Action Lane 关键字段就绪 | 约 384 ms | 250–320 ms | 60–130 ms |
| 动作最终就绪 | 约 390 ms | 340–370 ms | 20–50 ms |
| Language Lane 完成 | 约 451 ms | 300–380 ms | 70–150 ms |

动作最终收益小于 Action Lane 自身收益，是因为候选评分仍需约 339–345 ms。拆分完成后，动作就绪的理论稳定下限主要由候选评分、scheduler wait 和发布开销决定。若要进一步稳定低于 340 ms，需要另行优化候选 suffix forward 和调度等待。

以上均为工程估算，不是验收结果。如果两条 Lane 没有真正共享 Prefix，动作就绪可能仅改善 0–20 ms，甚至因重复 Prefill和调度竞争回退约 50–200 ms。

## 4. 潜在问题与约束

### 4.1 两条 Lane 结论不一致

可能出现 Action Lane 要求执行动作，而 Language Lane 输出能力拒绝或相反承诺。必须规定：

- Action Lane 是身体动作、表情动作和视觉路由的唯一权威来源；
- Language Lane 不能改变动作意图；
- 语言发布前执行动作/语言一致性检查；
- Action Lane 缺字段、解析失败、超时或冲突时 fail closed，不执行动作。

### 4.2 过早发布不可撤销动作

只有最小权威字段组合完整且语义稳定后才能发布。`perform` 本身不足以执行；至少还要取得合法的规范化 `body_task` 或 `face_task`。视觉 COPY、视觉问答和算术继续等待视觉分支结果。

### 4.3 Prefix 或媒体被重复计算

如果现有普通请求接口不能表达“一份 Prefix + 两个生成 suffix”，需要增加原生共享 Prefix batch；不能用两个并发 `completion_stream` 假装共享。缓存键必须包含用户、Session、Prompt Hash、目录版本、输入媒体 Hash 和预处理配置，禁止跨用户或不兼容 Session 复用私有 KV。

### 4.4 Scheduler 竞争和批次碎片

典型 Turn 可能同时存在 159 个动作候选和 2 个意图 Lane，总真实请求数约 161，低于当前容量 200，但不同请求形态不保证进入同一个 CUDA 执行批次。必须确认：

- 两条意图 Lane 没有挤占或拆散动作候选的单物理批次；
- `candidate_queue_wait_ms` 没有明显增加；
- CUDA Graph 覆盖实际 batch size；
- 最大容量只作为上限，不创建虚假占位请求。

### 4.5 KV、显存和生命周期增长

公共 Prefix 应只保存一份，每条 Lane 只增加短 suffix 和少量生成状态。若观察到接近两倍的 Prefix KV，说明共享失败。任一 Lane 完成、失败或取消后都必须释放自己的状态；公共 Prefix 在两条 Lane 和动作评分都不再使用后释放。

### 4.6 超时和部分失败

- Action Lane 失败：不执行动作，语言可以走安全兜底；
- Language Lane 失败：已通过安全门控的动作可以继续，语言走固定兜底或保持静默；
- 任一 Lane 失败都不能触发服务重连、Session 登出或重复 Session 预热；
- Turn 终态必须等待必要的清理完成，但不能阻止已合法产生的 `turn.action.ready`。

## 5. 实施顺序

1. 增加内部特性开关，默认关闭；保留当前单 JSON 路径作为回滚方案。
2. 把现有统一 Prompt 拆成 Action/Language 两个最小 Schema，并保持字段流式顺序。
3. 实现同一 Turn 的公共 Prefix/媒体编码共享和两 suffix 原子入队。
4. 实现两个独立流式解析器，以及 Action Lane 权威、Language Lane 只读动作结果的一致性门控。
5. 完善取消、超时、部分失败和资源释放。
6. 先运行 shadow 模式：新 Lane 只记录结果，不影响线上动作与回复。
7. 通过准确率、延迟和资源验收后，再逐步启用 Action Lane 发布，最后启用 Language Lane。

## 6. 必须回归的语义场景

- 纯动作：`比个一`、`双手比心`、`挥挥手`；
- 说话与动作分离：`挥手说拒绝`、`说二比三`；
- 能力询问：`你能跳舞吗`；
- 禁止动作：`别跳舞`；
- 动作不支持：`你跳个舞`，不能保留执行承诺；
- 独立语言任务加动作：`告诉我天气，再跳个舞`；
- 纯表情、身体与表情组合；
- COPY 当前手势/表情；
- `这是几`、视觉算术及指定只说/只做手势/手势加语音；
- Turn 处理中取消、断线、Session 关闭；
- Action Lane 或 Language Lane 单独超时和格式错误。

## 7. 验收指标

使用固定模型、相同输入、相同159个候选和同一GPU0环境，对当前单 JSON 与双 Lane 各执行至少2轮预热和30轮正式请求，记录 P50/P95/P99：

- `action_lane_first_field_ms`、`body_task_ready_ms`、`action_lane_complete_ms`；
- `language_lane_first_field_ms`、`language_lane_complete_ms`；
- 动作评分、`candidate_queue_wait_ms`、动作最终就绪；
- 公共 Prefix Prefill次数、缓存 Token、媒体编码次数；
- 实际物理 batch size、CUDA Graph命中、峰值显存和GPU利用率；
- 两 Lane 冲突率、解析失败率、动作/回复准确率；
- 取消后残留请求数与 KV 释放结果。

性能目标：动作最终就绪 P50 进入 340–370 ms，P95 不因拆分恶化；完整意图/语言完成时间明显低于当前约451 ms。准确率、动作安全门控、取消正确性或资源释放任一项回归时，不启用新路径。
