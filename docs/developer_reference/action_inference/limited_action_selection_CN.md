# 限定目录单阶段动作选择

生产服务默认使用 `SGLANG_OMNI_ACTION_CATALOG_MODE=limited`，加载
`sglang_omni/assets/character_limited_action_global_catalog.json`。
目录保留原有 categories 结构作为动作元数据，但不进行类别评分。
所有类别的动作按全局 candidate/action ID 去重后直接进入 `stage=single` 评分。
当前目录包含 11 类、168 个不同动作；加上 `000`（不支持）决策项，实际评分候选为 169 个。
`000` 不属于可执行动作，命中后返回 `UNSUPPORTED`、`execute=false`。

## 会话与协议

- 用户轮和主动轮均使用单阶段动作选择；基础表情也在候选集中。
- `action.allowed_candidates` 在该模式下只提供匹配动作的 `execution_binding`，
  不再缩减服务端限定动作集合；目录外的绑定被忽略，没有传入绑定的动作使用空绑定。
  因此调用方需要能够按返回的 action ID 执行动作，或为限定目录动作提供绑定。
- `action.fallback_category_ids` 可以省略；旧客户端传入的值不再参与选择。
  类别预热列表、类别 top-k、类别快捷路由不用于该模式。
- 每轮显式传入的动作允许/排除约束继续生效。
- 文本、音频、表情独立分支继续工作；表情分支不会提前取消本轮动作评分。
- 会话开始仍返回类别元数据用于兼容，但动作选择阶段数为 1。

## 预热与缓存

- 服务启动的评分 warmup 使用 `single` 和 169 个候选的规模。
- 公共目录 prefill 为中文/英文 × 用户/主动，共 4 个动作提示词前缀。
- `session.start` 在返回 `session.started` 前，预热当前语言的用户无相机、用户带相机、
  主动无相机三个动作前缀，加上该会话的不可变指令。
  不再预热 Category 或各类别 Child 前缀。
- prefill 与实际评分共用动作提示词、缓存 namespace 和 session instance ID。
  公共前缀校验器也注册这些动作提示词，允许公共目录跨会话复用；会话部分保持隔离。
- 提示词按目录内容、语言、轮次来源和摄像头规则区分。用户带相机的会话指令使用独立
  namespace，并在 `session.start` 同步预热；主动轮不使用用户图片动作规则。
- 预热失败只造成缓存未命中，不阻断会话。全局预热继续遵守单项超时与总预算配置。
- 全局诊断字段 `action_prefix_statuses` 提供四个动作前缀的预热状态；
  旧的 category/child ready 字段在限定模式下不代表动作前缀状态。

## 启用与回退

更新代码后重启服务生效。默认无需修改客户端两阶段参数。
如需回退，设置 `SGLANG_OMNI_ACTION_CATALOG_MODE=hierarchical` 后重启，
恢复完整目录与原有两阶段路径。
`SGLANG_OMNI_ACTION_CATALOG_PATH` 仍可显式覆盖当前模式所加载的文件；
若已有该环境变量，请确认它指向预期目录。

测试覆盖单阶段候选集、基础表情、不支持分支、回复融合、主动轮、
预热与实际请求的前缀一致性及预热失败降级。
实际延迟收益需在模型服务重启后测量：单次候选数量增加，不能仅凭减少一个阶段推断端到端耗时。

## 耗时日志

prefill 的 system prompt 包含完整动作集合，但仅用一个候选作为探针来建立共享前缀缓存，
不是逐动作执行，也不是逐一预先计算 169 项的评分。
日志统一使用 `action_count=168`、`candidate_count=169`、`probe_candidate_count=1`。

| 事件 | 用途 |
| --- | --- |
| `global_action_single_prewarm_completed` | 公共动作目录预热，每语言/轮次来源一条 |
| `session_action_single_prefill_completed` | 会话动作前缀预热，每轮次来源一条 |
| `action_scoring_completed`，`stage=single` | 实际动作评分，含 `elapsed_ms`、`prefix_cached`、`stats` |
| `action_prefix_cache_miss` | 实际评分未命中前缀缓存 |
| `turn_completed` | 完整轮次耗时 `total_after_commit_ms` |

两种预热事件包含 `elapsed_ms`、`prewarmed`、`request_id`、`catalog_hash`、
`prefix_cache_namespace`、语言/来源及底层 `stats`；会话事件另外包含会话 ID。
真实客户端将缓存命中状态和底层返回的 token、排队、prefill、suffix 分批统计写入 stats；
失败时 stats 可能为空，不能把缺失统计当作零耗时。

结构化 JSONL 默认存放于 `/tmp/sglang-omni-realtime-logs/<小时>/`，
可通过 `SGLANG_OMNI_REALTIME_LOG_DIR` 指定其他目录。
预热事件写入 `performance_*.jsonl`，动作评分事件写入 `action_*.jsonl`。
分析时按 `session_id`、`turn_id`、`request_id` 关联，分别统计首轮、后续轮次及缓存命中/未命中，
在相同输入类型和并发量下对比旧两阶段与新单阶段的 P50/P95。
