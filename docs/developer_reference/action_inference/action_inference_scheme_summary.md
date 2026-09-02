# Qwen3-Omni 回复与动作推理方案总览

本服务通过 `/v1/session/realtime` 在同一个 Session 和 Turn 内处理回复与动作。客户端正式
协议见 [realtime_reply_action_client_guide.md](realtime_reply_action_client_guide.md)，融合实现
语义见 [reply_action_fusion.md](reply_action_fusion.md)。

## 能力边界

- 输出类型：`text`、`action`，可单独或同时启用；
- 输入：用户 PCM16LE 音频、用户摄像头图片、数字人当前图片、可选用户文本、结构化状态；
- Turn 类型：用户触发 `origin=user`，数字人主动触发 `origin=proactive`；
- Session 内保留有限多轮回复、媒体和已执行动作事实；断线后不持久化；
- 服务负责动作选择，不负责动画播放器和最终播放成功确认。

## 全局目录与 Session 白名单

服务启动时加载全局只读动作目录并预热 Category 与 Child 静态前缀。每个 Session 通过
`action.allowed_candidates` 发送全局候选的子集，只携带 `candidate_id` 和可选
`execution_binding`。

服务端根据全局目录恢复：

- 动作所属类别；
- `action_id`；
- 类别/动作名称和定义；
- Category 与 Child Prompt 语义。

Session 白名单不会修改全局目录。评分和结果都不能越过当前 Session 白名单。

## 两阶段选择

```text
当前 Session 实际覆盖的类别
        │
        ▼
Category PPL 在类别与 B000（UNSUPPORTED）中评分，选 Top-1
        │
        ▼
选中类别下的 Session 白名单动作
        │
        ▼
Child PPL 在类别动作与 A000（UNSUPPORTED）中评分
```

所有 Action Session 必须通过 `action.fallback_category_ids` 配置至少一个兜底类别。当前全局
目录以语义标签声明“语言表达伴随”和“静默低扰伴随”两个系统类别：融合 Session 必须同时
提供两类真实候选，action-only Session 至少提供静默类；静默类必须是首个 fallback，语言
表达伴随类不能作为 unsupported fallback。Category 的 `B000` 和普通业务 Child 的 `A000`
都映射为
`UNSUPPORTED`；其中 `B000` 只判断目标语义类别是否存在，不能用于判断类别内具体动作；
`A000` 只在类别已选定、但该类别下的 Session 具体动作无法满足明确请求时使用。它们只是
推理判断，不是目录动作。任一阶段选择
它时，服务端进入数组中的最高优先级类别并返回真实动作，`execute=true`，同时标记
`support_status=unsupported`、`fallback_applied=true`。

系统伴随类别不加入 `A000`。Category 初选任一系统类别后，服务端根据 provisional 回复的
实际结果做双向校正：非空回复进入语言表达伴随 Child，并把首句或最多 256 字符前缀作为主要
动作依据；空回复、回复失败或无需回复进入静默低扰 Child。明确动作、社交反应和主动场景指定
动作仍走普通业务类别，不等待回复前缀。

## 上下文和 Prompt 隔离

| 上下文 | 回复 | Category | Child |
|---|---:|---:|---:|
| `character_profile` | 是 | 是 | 是 |
| `reply.instructions` | 是 | 否 | 否 |
| `action.category_guidance` | 否 | 主要 | 背景 |
| `action.candidate_guidance` | 否 | 可行性参考 | 主要 |
| `reply.context` | 是 | 否 | 否 |
| `action.guidance` | 否 | 是 | 是 |
| 用户音频/图片/文本 | 是 | 是 | 是 |
| 数字人当前图片/状态 | 间接场景理解 | 是 | 是 |

回复与动作拥有独立 System Prompt。服务端不会把动作临时约束混入回复 Prompt，也不会把
回复临时约束混入动作 Prompt。

## 融合时序

回复生成与 Category 并行。普通业务类别选出后立即启动 Child；仅当 Category 选中系统伴随
类别时，动作分支等待首个非空回复 delta 或回复终态，并在首个 delta 后最多再等待 100 ms
补齐首句。回复不等待 Child，因此 Child 失败不会阻塞或改写已成功的回复；最终以
`status=partial`、`outputs.action=failed` 返回动作错误。

客户端主动提供 `reply.provided_text` 时，服务不再生成回复。空字符串是明确的静默回复，
不会被误判为缺省。

## 历史与动作事实

成功动作选择暂时视为实际执行并保存为 Session 动作事实，但该记录仅用于日志、调试和审计。
Category 和 Child 不读取跨 Turn 回复历史、上一动作 ID、动作名称或动作历史；动作衔接、复位
和重复动作的平滑处理由动画引擎负责。当前数字人图片以及 `pose`、手中物体等当前结构化状态
仍可进入本轮动作推理。

若用户询问“上一个动作是什么”，客户端应通过本轮 `reply.context` 提供已确认事实。若要支持
“再做一次刚才的动作”，客户端应把自身已确认的执行记录解析为明确动作目标；当前兼容字段
`action.last_executed_action_id` 不再影响动作选择。Child 失败或取消仍不会覆盖诊断记录中的
上一条成功动作事实。

## 结果

三种输出模式统一返回：

```json
{
  "type": "turn.result",
  "status": "completed",
  "outputs": {
    "text": "completed",
    "action": "completed"
  },
  "reply": {},
  "action": {},
  "timing": {}
}
```

动作可以通过 `turn.action.ready` 提前下发。客户端只依据 `execute` 执行，并按 `turn_id`
幂等，不能在 `turn.result` 再次执行同一动作。

## 缓存与性能

- 全局 Category/Child 静态前缀跨 Session 共享；
- 当前 Session 白名单和偏好动态追加，不污染全局缓存；
- 当前 Turn 媒体、状态和临时 guidance 位于动态上下文；
- 资源日志按 20 秒周期采样，并在 Turn 推理前后额外采样；
- 客户端可以顺序发送媒体和 commit，无需逐 ACK 等待。
