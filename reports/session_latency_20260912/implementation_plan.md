# 表情提前输出、音频放行与 Child 缓存优化实施方案

状态：2026-09-12 已完成第一批代码实现与本地回归，尚未重启服务或进行真实模型/数字人回放。纯表情提前输出、音频放行、D_video_call 配套、Child 观测，以及保持字节不变的前缀渲染/哈希缓存已实现。GPU KV 缓存布局调整和组合预热仍需根据新增日志单独评估，未实施，也未声称已取得冷/暖组约 204 ms 的收益。下文保留完整方案与后续实验边界。

## 1. 本次范围与责任

| 项目 | sglang-omni | D_video_call |
|---|---|---|
| 纯表情提前放行 | 表现控制判定后不再等待身体评分；输出最终 no_action；取消身体分支 | 保留表情先于 action.ready 的屏障；按 not_required 执行纯表情 |
| 纯表情音频提前放行 | 满足表情与回复合法性后 promote；排出已生成音频 | 纯表情不再占用身体动作的首音频同步等待；保持音频顺序与取消 |
| Child 前缀缓存优化 | 先记录身份和命中数据，再做可验证的固定前缀复用与有限预热 | 无缓存协议改动；仅接收现有动作/表情事件 |
| Child 观测 | 每次记录类别组合、候选、哈希、namespace、命中/计算 tokens、历史出现情况 | 不向客户端发送缓存内部信息；可用 turn/trace 对齐客户端延迟 |

不将空 action_finished 快速路径、短回复模板、route/performance 联合分类或准入容量调整混入这次实现，以便分别衡量收益。三个改动中的前两项共用一个结果判定流程，缓存优化独立推进。

## 2. 检查现有代码后的结论

### 服务端

- `sglang_omni/serve/realtime/turn_pipeline.py`：启动 route、身体动作、表现控制和回复分支；目前最终融合与表情/动作事件发送仍等待 action_task。
- `performance/fusion.py`：expression_only 的最终动作已经是 no_action/not_required，但只改变结果，没有解除等待。
- `reply/provisional.py::_promote_provisional_reply`：已有缓冲文本/音频按序放行能力，放行期间会发送音频。需明确表情/action ready 在 promote 之前。
- 当前 active_request_ids 是整轮集合。只停止身体评分需要补充分支所有权，不能对整轮集合调用 abort。

### D_video_call 客户端（包含接入后端和浏览器）

- `backend/character/providers/qwen_session_realtime.py` 已支持 not_required，并要求表情先于 action.ready；不要放宽此约束。
- `backend/qwen_realtime_client.py::_handle_session_realtime_expression_ready` 只保存表情；收到 action.ready 后才决定单独执行表情还是与身体动作合并，这是已有的正确屏障。
- `_handle_session_realtime_action_ready` 已在 not_required 时调用 `_schedule_session_realtime_expression`；没有表情则调用 offer_no_motion。已有重复 ready 检测。
- `_schedule_session_realtime_expression` 当前把表情通过 offer_motion 交给 `SessionInitialOutputCoordinator`。
- `backend/character/conversation/delivery/session_initial_output.py`：motion 提供后立即启动；若音频在 motion 在途时到达，可能等到 motion 返回或同步截止时间。当前 `.env.d/30-character.env` 的同步预算为 300 ms。这是上限，不是每轮固定增加 300 ms。
- `frontend/app.js` 收到音频就处理，不等待 turn.result；数字人启用时声音走数字人输出链路，否则按现有回退条件进入本地播放。
- 前端 action.ready 日志只区分 unsupported/其他，not_required 会显示成“支持；未命名动作”；回复动作面板也需要明确显示“不需要身体动作”。
- `character_conversation_runtime.py::dispatch_session_expression` 在已有计划运行时使用 FINISH_THEN_ACTION。因此服务端更早 ready 不保证脸部立即显示，当前动作排队是另一个需观测的时段。本方案保持此播放策略，不擅自打断已有动画。

## 3. 服务端：统一结果判定与纯表情提前输出

### 3.1 判定条件

仅当表现控制正常返回 request_scope=expression_only 且本会话启用了 expression 输出，才进入纯表情路径。不能凭文本关键词、空 action_id、短回复内容或 action_task 的完成顺序判断。

- 表情受支持：结果为表情 + action no_action / execute=false / support_status=not_required / fallback_applied=false。
- 表情不受支持：保持现有 expression_unsupported 语义，不返回成功表情、不执行身体动作；是否仍需语言回应沿现有规则处理。
- both：两个分支均满足要求后才成功，保持现有原子性。
- body_only / none、表现控制失败或超时：保持现有回退语义，不能因为模型异常伪造 expression_only。
- 没有 expression 输出的 audio-only 等模式：保持现有独立身体动作语义。

### 3.2 主流程如何改

1. 继续并行启动各分支。为本轮最终表现结果指定一个统一提交者，分支仅返回结果和诊断数据。
2. 统一提交者在表现控制完成时即可处理纯表情；不能把它放在“先 await 路由/回复，再检查表现控制”的串行尾部，否则仍可能受到无关慢分支影响。
3. 表现控制若判定纯表情，立即锁定最终动作结果。身体先完成但表现控制未完成时，先保存身体结果，不抢先对外提交。
4. 发送顺序为：turn.expression.ready（有合法表情时）→ turn.action.ready（最终 no_action/not_required）→ promote 合法回复/发送已缓冲音频。三者不等待身体分支完成。
5. 同一 turn 的最终动作只提交一次。后台晚到的身体结果、失败或类别回调不再覆盖结果，也不能另发 action.ready。
6. 不将无用身体结果写入已执行动作历史；表情仍按原表情通道处理。候选评分可保留在诊断日志，不能混入执行事实。
7. 终态 turn.result 与此前 ready 的动作、表情一致；它仍负责回复/TTS 和清理资源的最终收敛，不用于阻塞已经合法的首输出。

推荐将最终判定的类型和不变量放在现有 performance 组件，主流程负责调用与副作用。避免把发送、取消和回退规则复制到多个回调。

### 3.3 身体分支取消

建立明确的 body 分支请求归属，覆盖 category、child、按需 prefill 及该分支实际创建的其他子请求；不要依赖截取整个 turn 的 active_request_ids 猜测。

- 先锁定纯表情结果、阻止继续提交新的身体请求，再终止对应协程并 abort 已提交请求。
- 取消与完成竞态要幂等：已完成或不存在的请求也能安全清理；不要 abort reply、performance 或 TTS。
- 后台回收任务必须被 turn 持有和观察。记录取消请求与完成时点，失败或超时沿已有模型取消接口的能力收敛，不能产生无人观察的异常。
- 请求注册/注销使用 finally，准入槽必须释放；检查共享预热是否被其他请求使用，不能当成独占任务直接取消。
- 不调用“取消整个 turn”。只有用户插话、客户端断开等整轮取消条件才进入原整轮取消流程。
- 特别检查 category_unsupported 的早期 discard 回调：在 scope 未定时不能不可逆地丢弃之后应被纯表情路径接受的回复。scope 已确定为纯表情后，身体类别失败不再决定回复命运；其他模式保持已有语义。

## 4. 服务端：纯表情音频提前放行

使用上节同一个最终判定，不另起一套仅凭音频到达就 promote 的分支。

- 合法表情成立后，回复无需再等身体动作支持判定。
- 已有缓冲音频：通过现有 promote/发送出口顺序排出；新音频接在缓冲之后，不能绕过锁和 seq。
- TTS 尚未开始或还没有 PCM：正常继续生成。提前放行不保证此时立即有声音。
- 生成式纯短回复仍先做现有校验，再进入 TTS；本次不删校验、不引入模板、不改变 TTS 语气。
- 插话/取消优先于尚未发送的内容；每次排出音频前检查 turn 是否仍有效。
- 提前放行后已经发送的内容不能撤回；后续 TTS 异常通过现有 failed/partial 输出表达，不能回滚并重新发送另一套动作决定。
- 针对音频先到、表情先到、两个分支同周期完成、缓冲排出期间插话分别测试。

观察值参考：新“笑一个”表情 419 ms 就绪、PCM 697 ms 到达、约 1022 ms 才放行。优化目标是去掉表情约 603 ms、音频约 324 ms 的无关等待；这些不是改后实测承诺。

## 5. Child 观测：每次评分都有可关联的记录

### 5.1 事件组织

建议在现有结构化日志上增加 `child_cache_request` 与 `child_cache_result`。名称为方案约定，实施时与现有事件整合，避免重复打印大段 prompt。

- request 在请求提交前记录；包含身份、候选与是否曾出现。
- result 在成功/失败/取消时记录；关联 request_id，包含真实命中指标与终态。不能仅在正常返回时埋点。
- 纯表情使 Child 未提交时记 `child_scoring_skipped` 及原因，不伪造命中记录。客户端已有的纯表情协议无需新增缓存字段。

| 字段 | 含义与口径 |
|---|---|
| session_id / turn_id / trace_id / request_id | 与 route、performance、回复以及客户端事件对齐 |
| service_instance_id / process_epoch | 区分服务重启和统计窗口 |
| selected_category_ids | 按实际评分顺序记录，不排序改写 |
| candidate_ids_ordered / candidate_count | 最终参与评分的候选，包含实际加入的 unsupported 等哨兵 |
| candidate_set_hash / candidate_order_hash | 分别区分集合变化与仅顺序变化；实际评分顺序保持不变 |
| static_prompt_sha256 | 实际传入的 child_system_prompt 内容哈希 |
| session_instruction_sha256 | 实际会话指令哈希；空值采用固定、明确的编码规则 |
| prefix_cache_namespace / locale / turn_origin | 实际请求字段，不使用推测值；同时保留 language 便于核对 |
| model/tokenizer/rendering_identity | 模型与 tokenizer/模板版本信息；用现有可获得配置身份，拿不到时显式缺失 |
| cache_static_system_only / reusable_boundary_token_count | 实际复用范围，区分只缓存 system 与包含会话指令 |
| prefix_identity / scoring_identity | 见下文，两者用途不同 |
| identity_seen_before / seen_count_before / first_seen_at / last_seen_at | 同一 prefix_identity 在声明的观测窗口内是否提交过 |
| previous_status / previous_completed_at | 最近提交是否真正完成过，用于区别只见过取消请求 |
| parent_radix_cached_token_count | 评分端报告的 parent 已命中 tokens |
| parent_computed_token_count | 评分端实际计算 tokens |
| prefix_token_count / parent_cache_hit_ratio | 总长度与真实命中比例 |
| candidate_prefix_recompute_tokens | 候选阶段是否重新计算前缀，单独观察 |
| slot_wait_ms / preprocessing_ms / prefix_prefill_ms / suffix_ms / elapsed_ms | 延迟归因；明确字段是否重叠，避免盲目相加 |
| status / cancel_reason / error_type | completed/cancelled/failed；统计缺失字段留 null，不当成 0 |
| child_catalog_prefill_ms / prefill_status | 显式预热与本次评分成本分开，预热不等于评分成功 |

### 5.2 “缓存身份见过”不等于“缓存命中”

建立两种身份：

1. **prefix_identity**：实际可复用前缀的身份。包含实际 namespace、静态 prompt/会话指令哈希、复用边界策略、模型与渲染配置。若候选本身出现在 system prompt，它自然通过 prompt hash 参与身份。
2. **scoring_identity**：prefix_identity 加实际有序类别、候选及 suffix 编码/评分配置，用于定位为何同一前缀下评分工作量变化。

不要把所有动态候选约束无条件加进 prefix_identity，那会把可共享前缀误拆；也不要为了提高命中率删除现有 namespace 安全边界。观测身份不是替换底层 KV cache key 的指令。

先在单个 API 进程使用有容量/保留时间限制的记录表。字段明确 `seen_scope=process_window`，并带进程启动标识。重复判断与计数更新不跨 await，避免并发请求都记为第一次。表项过期/淘汰后，“未见过”只代表窗口内未见过；跨进程和跨重启汇总交给离线日志分析，不伪称全局首次。

即使 seen_before=true，KV 仍可能已淘汰、在另一个执行实例、此前仅取消，或实际渲染边界不同。因此只能归类为“重复身份仍未命中，需调查”，不能直接断言缓存 bug。底层没提供的信息不能据此补造。

### 5.3 汇总输出

按 prefix_identity、类别组合、origin/locale、首次/重复、成功/取消分别输出：次数、命中率、computed tokens、prefill/总延迟 P50/P95。另列“大候选集合”与“2 批评分”样本，防止将候选工作量下降误判为缓存收益。

新增埋点默认不打印原始用户文本、角色长指令或完整 prompt；记录 ID、哈希和数值即可。验证日志自身的开销、队列丢弃和保留窗口；debug 关闭时仍需能获得必要的轻量计数。

## 6. Child 缓存优化如何落地

先上观测，保证一次优化可以解释前后变化；随后应用以下保守顺序：

1. 固定描述与固定指令保持确定性渲染。只消除确认为非语义的格式漂移；默认不改变真实类别顺序、候选顺序、去重与类别归属。
2. 检查静态内容是否真的处于可缓存前缀，动态输入是否被放进 namespace 或静态段；若需移动指令位置，作为会改变模型输入的实验单独验证，不假设无损。
3. 对观测证明高频且重复的冷前缀使用现有 prefill 能力做有限预热。沿用准确身份，完成后才记 warmed；以低优先级运行，限制数量与缓存预算，不把大规模预热串进首请求。
4. 监测预热是否挤出其他缓存、提高 GPU 排队或启动时间。收益需扣除预热成本，分别报告冷态、暖态、混合真实请求与并发负载。
5. 若冷缓存主要由不断变化的类别组合造成，再做独立结构实验。不要直接排序类别组合、删除 hash 或缩小候选集合以制造命中提升。

本次目标是减少不必要的前缀重算，保持最终输出质量。当前冷/暖 Child 均值约 480/276 ms 仅为机会量级，不承诺普遍节省 204 ms。纯表情已不依赖 Child 后，缓存收益主要体现在身体动作和普通交互轮次。

## 7. D_video_call 的具体改动

### 7.1 Provider 适配器：协议保持不变

`backend/character/providers/qwen_session_realtime.py` 继续验证：

- expression.ready 必须先于 action.ready；
- not_required 不含 selected/fallback 可执行动作；
- turn.result 与先前输出一致。

服务端仍通过现有序列化层提供协议字段；不能把内部 no_action 字典原样发送成一个伪 selected_action。增加提前 ready、延后 result 的协议测试，不新增客户端必填参数。

### 7.2 接入后端：动作/表情仍统一判定

`backend/qwen_realtime_client.py` 保留“保存表情 → 收到 not_required action.ready → 调度表情”。不能收到 expression.ready 就无条件执行，否则 both 可能被提前拆开。

- 沿用单 turn 去重、过期 session/turn 丢弃机制。
- not_required 无身体动作 ID，不派发身体动作、不添加身体动作执行历史、不合成一个 no_action 动画。
- turn.result 到达时不重派表情或重复更新执行记录；增加对应回归。
- 现有 action_executed/scheduled 集合也被表情复用作“视觉输出已认领”。不要因变量名就当作身体动作执行事实。必要的表情/身体区分放在交付状态中，不做全仓重命名。

### 7.3 首音频协调：只对纯表情取消身体同步等待

在 `backend/character/conversation/delivery/session_initial_output.py` 给视觉输出增加明确的首音频等待策略，例如 `wait_for_visual_dispatch`（最终命名按代码约定）：

- 身体动作或 both：沿用当前同步预算。
- not_required + 合法表情：立即启动表情调度，但音频到达即排队发送，不等待表情 dispatch 返回。
- not_required 且无表情：保留 offer_no_motion。

由 `_schedule_session_realtime_expression` 传入纯表情策略，不全局把 `CHARACTER_AUDIO_MOTION_SYNC_THRESHOLD_MS` 改成 0。保持协调器对 motion/release/drain 任务的统一持有、finish 和 cancel，不能创建游离的表情任务。

表情 dispatch 出错需要观察和记录，不应无期限扣留已经合法的音频。音频串行 drain、audio_done 排在音频后、取消时丢弃未发送包等规则不变。

这项意味着不保证“脸部渲染完成再开口”，而是先发出表情调度，再允许语音流动。旧动画可能仍排在前面；不要为了隐藏延迟改成强制打断。记录实际排队/提交时间以区分两者。

### 7.4 浏览器：修复状态呈现，沿用播放链路

在 `frontend/app.js` 的 action.ready 日志和回复动作面板增加显式 not_required 分支：

- 动作显示“无需身体动作”，表情显示实际名称；不再显示“支持；未命名动作”或一直“等待动作”。
- ready 更新视觉决策状态，不代表整个 turn 已完成；不能因此提前结束语音、禁用后续音频或触发虚假的 action_finished。
- 继续收到 audio_delta 就处理，不等待 turn.result。数字人音视频链路与现有本地回退策略保持不变，不能为了降低浏览器计时强行本地播放，造成双重声音。
- 保留有效 turn 检查；旧 turn 音频、重复 ready、重连后的晚到事件不能污染当前轮次。

### 7.5 贯通埋点

复用已有时间线，增加或明确以下阶段：provider 表情到达、not_required ready 到达、表情 offer、dispatch 开始/返回、首音频到达接入后端、经过协调器后首包发出、数字人 feed_pcm、浏览器首包接收。

浏览器/数字人实际播放开始若已有可靠回调则复用；没有就标记不可观测，不拿 `audioState=playing` 或首包接收代替真实播放。

对齐使用 session/turn/request/trace 标识，各进程只计算自己的单调时钟耗时。不要将不同机器的 monotonic 时间直接相减。最终报告区分 Omni 提前放行收益、客户端协调等待、数字人动画排队与真实播放延迟。

## 8. 实施批次与验证

### 批次 A：纯表情与音频的统一放行

- sglang-omni：最终结果统一提交、身体请求归属、提前输出与取消清理。
- D_video_call：纯表情音频不等身体同步门槛、not_required 呈现与交付埋点。
- 保持协议兼容；旧客户端可接早到的 ready，但可能保留最多 300 ms 的本地协调等待。新客户端也必须接收旧服务端事件。

### 批次 B：Child 身份与命中观测

- request/result/skip/cancel 全路径齐备；轻量埋点不引入新的热路径模型调用。
- 建立可重复运行的按身份汇总脚本，基线与优化后使用相同统计口径。

### 批次 C：按观测结果做前缀复用与有限预热

- 独立变更、独立比较；不要同时改并发、类别 top-k、短回复策略。
- 若收益来自输出/候选减少，不能当作缓存优化成功。

| 回归场景 | 必须满足 |
|---|---|
| 表情快、身体评分被阻塞 | 表情与 no_action ready 能在身体释放前到达 |
| 身体先返回/两分支同时返回 | action.ready 只发一次，结果由 scope 决定 |
| 表情成功但身体报错或类别不支持 | 不撤回合法表情/回复；身体错误不再控制纯表情结果 |
| 表情不支持、both 某通道失败 | 保留既有失败/原子性语义 |
| expression 输出关闭或表现控制失败 | 不误用纯表情快速路径 |
| 音频早到且缓冲/音频晚到 | 音频顺序、seq、audio_done 一致，无重复排出 |
| 取消期间持续到音频/child 晚到 | 无旧输出、无执行历史污染、请求与准入槽正确释放 |
| D_video_call 表情 dispatch 阻塞 | 纯表情首音频仍发送；身体模式保留同步预算 |
| ready 早、result 晚、没有音频 | 客户端只执行一次表情，正确结束各输出状态 |
| namespace/locale/origin/模型/指令改变 | 身份有正确变化，不错误共享缓存 |
| 同身份并发、仅取消过、进程重启、记录表淘汰 | seen_before 口径正确，缺失 tokens 不记 0 |
| 首次/重复前缀与预热开关对比 | 记录真实 computed tokens、prefill、端到端 P50/P95 与动作输出差异 |

优先扩展现有测试：sglang-omni 的 performance 与 multimodal_session 测试；D_video_call 的 provider、session_realtime_turn_lifecycle、session_initial_output 及 frontend 呈现测试。异步竞态测试用可控事件阻塞分支，避免靠极短 sleep 偶然通过。

验收不以固定硬件毫秒阈值代替因果验证：首先证明表情发送、音频放行不依赖 child；然后以相同请求集实测服务端/客户端的分位数。旧样本纯表情额外等待均值 360 ms、新样本 603 ms/音频 324 ms 是改前依据，不是上线承诺。

实际开发时遵守 D_video_call/AGENTS.md，读取 Python 设计和变更检查文档；沿现有责任模块扩展，保留两个仓库当前无关工作区修改。按独立批次保留可回退边界，回退某项不撤销已修复的纯表情 no_action 正确性规则。


## 9. 本次实际实现与验证记录

### 已实现

- Omni 的表现控制任务可在慢 route、category/child 或身体取消清理尚未结束时发送纯表情与 no_action，再放行合法回复/音频。保留一般动作路径原有事件顺序和 TTS 解耦开关行为。
- 身体评分由自身任务取消，通过已有 ModelClient 的 CancelledError → 当前 request abort 清理；主 turn 终态观察清理完成，不取消其他模型请求。日志新增 expression_only_released/body_bypassed/body_cleanup_completed。
- Client 的视觉输出协调器增加 wait_for_motion 参数；纯表情传 false，身体动作保持默认 true。表情任务仍由协调器 finish/cancel 持有。
- 前端日志与动作面板增加 not_required 呈现；表情提供/dispatch 的时间线得到补充。数字人排队策略不变。
- Child 的 request/result 日志包含有序类别候选、各身份哈希、namespace/locale/origin、首次/重复和成功/失败/取消、真实 parent tokens。进程观测表最多 1024 项、保留时间 1 小时；窗口未见过不等于全局首次。缺失统计为 null。
- Child 静态文本使用最多 64 项的会话内渲染缓存，哈希最多 128 项复用；保持文本、候选顺序及 KV namespace 完全不变。这降低重复 CPU 构造成本，不等于优化了 GPU 冷前缀。
- 新增汇总入口：`python scripts/summarize_child_cache.py logs/realtime/2026-09-12`。成功请求的分位数与失败/取消次数分开统计。旧日志没有新事件时返回空列表。

### 验证

- 新增阻塞 category、child、route（含 route/action 串行模式）的四种竞态回归：表情 ready 先到、action ready 唯一、慢取消不阻塞首音频、表情→动作→音频顺序成立。
- 新增观察表身份隔离、TTL/容量淘汰、并发提交乱序完成、缺失 tokens、前缀文本与候选顺序不变测试。
- D_video_call 接入 provider、生命周期与首输出协调共 118 项通过；Node 22 对 app.js 的语法检查通过。
- Omni 最新聚焦回归 233 项通过，排除两个此前已确认的提示词断言失败。
- Omni 第一轮扩大测试 726 passed / 5 failed：两个此前已确认的提示词断言；一个已有组件列表断言；两个 SOCKS 环境缺少 socksio。没有修改这些不相关逻辑。
- D_video_call 扩大测试 116 项中 110 passed / 6 failed；6 项均属已有前端契约断言，使用移除本次 app.js 修改的内存基线重跑，仍是相同 6 项失败。
- 本地 fake adapter 验证了事件依赖和资源清理；尚未测量真实模型首包改善，也未验证浏览器/数字人实际渲染耗时。需要重启对应服务后收集新日志再回放。

没有新增 D_video_call/backend 根目录模块；交付策略仍由 character/conversation/delivery/session_initial_output.py 拥有。两个仓库的已有无关修改保持原样。
