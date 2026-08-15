# Qwen3-Omni 动作推理能力变更总览

本文统一记录当前分支从基线提交
10dd097eaae52b33eadccd437947ed758520c31a（包含该提交）到当前 HEAD 的完整
动作推理能力实现，以及本次开发形成的相关代码、测试、协议和运行配置。

## 实现范围

- 分支：feature/action-suffix-scoring
- 基线：10dd097 chore(qwen3): preserve single-gpu low-latency profile
- 当前 HEAD：c0b59b9 add catalog-backed action scoring comparisons
- 提交范围：15 个提交。

这次改动整体目标不是增加一个孤立的接口，而是在 sglang-omni 内形成一套
Qwen3-Omni 数字人动作推理能力：接收多模态对话上下文，比较动作候选，
返回动作排序和执行信息，并支持多轮 session、动作历史、日志分析和部署。

## 能力总览

当前实现包含以下能力：

1. 增加 action suffix scoring，对候选 suffix 计算 mean_logprob、mean_nll、
   PPL 和逐 token 分数。
2. 对一个逻辑请求只做一次多模态前缀处理，再批量评分所有候选。
3. 通过 KV/prefix cache 复用共享前缀，并将音频/图片内容纳入 cache identity。
4. 支持 session 历史、历史音频、历史图片和数字人状态。
5. 支持 flat 候选和“动作类别 → 具体动作”的两阶段选择。
6. 支持 action-only 多模态 session：当前 turn 不生成普通回复，但会记录最终动作，
   供下一轮理解“刚刚那个动作”“再重复一遍”等指代。
7. 支持音频 chunk、图片帧、文本和动作候选的外部接入协议。
8. 增加动作目录驱动的 smoke 测试、A/B/C 评分方案比较和日志分析 skill。
9. 增加 Qwen3-Omni 单 GPU 配置、Docker runtime 支持和 action-score 预热。

## 变更时间线

| 提交 | 变更内容 |
| --- | --- |
| 10dd097 | 保留 Qwen3-Omni 单 GPU 低延迟 profile，增加单 GPU 配置、阶段显存预算和相关 launcher/stage 调整。 |
| 0e44455 | 建立 action suffix scoring contract/API，增加 client 类型、请求/响应协议、候选校验、PPL 计算和基础单测。 |
| 86d957c | 将 action-score 接入 Thinker 调度器，增加 request builder、action-score 阶段、候选批处理、结果聚合和输出处理。 |
| c9d72df | 修正 action-score cache identity，使音频/图片内容而不是文件名参与媒体缓存身份。 |
| a77417c | 增加 action suffix scoring 开发文档，说明请求格式、PPL、KV cache、取消和 smoke 验证。 |
| f7e7e8e | 清理 action-scoring 回归测试格式。 |
| 7908ec5 | 使 Docker runtime 自包含，并补充 model worker、placement、请求取消和调度支持。 |
| bcab45e | 支持多轮上下文：session_id、历史消息、历史音频/图片、avatar state，以及上下文相关的 API 和测试。 |
| ccca02e | 增加原子动作 conversation smoke 脚本。 |
| e0a578d | 稳定 smoke 测试候选构造，减少候选不确定性。 |
| 485f508 | 对齐原子动作描述，使候选定义和测试语义一致。 |
| 81d9390 | 将动作场景改为更接近真实用户的自然表达。 |
| 43520c5 | 增加固定 action ID 分类测试，用于区分模型理解失败和自然语言 suffix 概率偏置。 |
| a9b45a2 | 避免多模态动作测试重复发送相同媒体输入。 |
| c0b59b9 | 使用动作目录生成候选，增加 A/B/C action-score 对比脚本、catalog 驱动 smoke 调整及 catalog 单测。 |

## 动作推理核心链路

~~~text
HTTP /v1/action-scores 或 WS /v1/session/realtime
        │
        ▼
预处理
  ├─ 渲染 system、历史消息、当前 turn
  ├─ 合并音频和图片
  ├─ 校验 max_seq_len
  └─ 生成包含媒体内容的 cache identity
        │
        ▼
Thinker prefix request
  ├─ 一次多模态前缀 prefill
  ├─ 使用 dummy token 获取前缀末端分布
  └─ 验证 prefix_cached
        │
        ▼
候选 suffix requests
  ├─ 复用共享前缀 KV
  ├─ 按 micro-batch 评分候选
  └─ 聚合 token logprob / PPL
        │
        ▼
动作结果
  └─ 返回完整排序、token scores、耗时和执行绑定
~~~

候选 token 的计算规则：

~~~text
mean_logprob = sum(token_logprob) / token_count
mean_nll     = -mean_logprob
ppl          = exp(mean_nll)
~~~

action-score 是 prefill/评分路径，不进入普通文本 decode、Talker 或 Code2Wav。

当前实现的主要代码位置：

- [action_scoring.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/models/qwen3_omni/action_scoring.py)
- [request_builders.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/models/qwen3_omni/request_builders.py)
- [omni_scheduler.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/scheduling/omni_scheduler.py)
- [client.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/client/client.py)
- [action_suffix_scoring.md](/home/ubuntu/fanshide/sglang-omni/docs/developer_reference/action_inference/action_suffix_scoring.md)

## Session Realtime 能力

新增 [multimodal.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/serve/realtime/multimodal.py)，并在
[openai_api.py](/home/ubuntu/fanshide/sglang-omni/sglang_omni/serve/openai_api.py) 中注册：

~~~text
WS /v1/session/realtime
~~~

该路由不依赖旧的 --enable-realtime 开关。协议事件包括：

- session.start / session.started；
- turn.start / turn.started；
- input_audio.append；
- input_image.append；
- turn.text.update；
- turn.commit / turn.committed；
- turn.result；
- turn.cancel / turn.cancelled；
- session.close 和 error。

session 状态保存在进程内，不做持久化。一个 session 同时只有一个 active turn。
用户松开空格并提交 turn.commit 后，服务只计算动作，turn.result.reply 为 null。
服务会把当前用户输入和实际选中的动作写入内部历史，供后续 turn 处理动作指代。

音频使用 Base64 PCM16，图片使用 Base64 图片 bytes 和 timestamp。一个 turn 可以包含
多个音频 chunk 和图片帧；服务按音频 seq 合并，图片按 timestamp/seq 排序。

## 两阶段动作选择

嵌套候选使用以下逻辑：

~~~text
一个外部 turn
  ├─ 阶段一：固定类别 system prompt，评分所有 category_id
  └─ 阶段二：临时构造选中类别的 system prompt，只评分该类别 children
~~~

这仍然是一个逻辑请求、一个 session、一个外部 turn，不会创建第二个 session。
第二阶段 system prompt 是当前 turn 内部临时构造的，不会追加到 session 历史。

方案 B 中，动作候选列表固定在 system prompt，suffix 只使用短 candidate_id。
最终结果再映射到真实 action_id 和 execution_binding。这样可以把一个基础动作
和它的执行参数分开，例如：

~~~json
{
  "candidate_id": "A15L",
  "action_id": "wave_one_hand",
  "source_label": "单手挥手-左手",
  "execution_binding": {
    "body_side": "left"
  }
}
~~~

模型选择的是 A15L，播放器执行的是 wave_one_hand 加 body_side=left。

## API、诊断和启动预热

本次实现新增或调整了：

- ActionScoreRequest.system_prompt，用于固定候选 system prompt；
- ActionScoreResponse.timing.server_action_compute_ms；
- WebSocket turn.result 的 action-only timing 和 media_summary.action_context；
- action-score prompt rendered 日志，记录实际 full_prompt、messages、prompt token
  数、媒体数量/hash、候选和 metadata；
- prompt 超限日志，记录 full prompt 和媒体诊断后再抛出校验错误；
- 进程级 action-score 预热：hierarchical 模式启动时分别用约 60 个 category 和
  约 8 个 child 候选执行两阶段真实评分路径；flat_children 模式只执行一次
  single 具体动作评分；预热不带 session、历史和媒体，不写入业务 session；
- prefix 首 token logprob 增加 SGLang next-token selected logprob / logits fallback，
  仍只使用真实模型概率；两类数据都不可用时返回结构化的当前 turn 错误；
- 动作评分异常按 turn 隔离，清理 active turn 并保留 session，后续 turn 可以继续提交；
- 预热环境变量：
  SGLANG_OMNI_ACTION_WARMUP、
  SGLANG_OMNI_ACTION_WARMUP_CATEGORY_COUNT、
  SGLANG_OMNI_ACTION_WARMUP_CHILD_COUNT、
  SGLANG_OMNI_ACTION_WARMUP_TIMEOUT_S。

预热用于初始化 request builder、tokenizer、scheduler admission 和 Thinker 执行
路径。预热失败不会阻止服务启动，也不会计入业务 session。

## 测试和实验能力

- [qwen3_omni_atomic_action_smoke.py](/home/ubuntu/fanshide/sglang-omni/scripts/qwen3_omni_atomic_action_smoke.py)
  从动作目录加载候选，支持音频、图片、文本、session 历史和 action ID 分类。
- [qwen3_omni_action_scheme_comparison.py](/home/ubuntu/fanshide/sglang-omni/scripts/qwen3_omni_action_scheme_comparison.py)
  比较：
  - A：自然语言 short_definition suffix；
  - B：固定候选列表 + 短 candidate_id suffix；
  - C：source_label suffix。
- 方案比较脚本的最终 turn 只加入最新用户输入，不伪造最新 assistant 回复。
- [test_action_scoring.py](/home/ubuntu/fanshide/sglang-omni/tests/unit_test/qwen3_omni/test_action_scoring.py)
  增加 system prompt、两阶段选择、预热和失败非致命等覆盖。
- [test_multimodal_session.py](/home/ubuntu/fanshide/sglang-omni/tests/unit_test/qwen3_omni/test_multimodal_session.py)
  覆盖 session 初始化、多媒体 chunk、图片排序、turn 历史、commit/cancel、错误、
  动作结果和耗时。
- tests/data/actions 作为外部测试资源，动作目录可以独立变化，不随服务代码固化。
- 新增日志分析 skill：
  [.claude/skills/analyze-qwen3-omni-action-logs](/home/ubuntu/fanshide/sglang-omni/.claude/skills/analyze-qwen3-omni-action-logs)
  用于统计成功 turn、category/child 耗时、prompt tokens、媒体数量、prefix cache
  和动作排序结果。

## 配置和部署

[config_single_gpu.yaml](/home/ubuntu/fanshide/sglang-omni/deploy/config_single_gpu.yaml)
本次实现将 preprocessing 和 thinker 的 max_seq_len 固定为 60000：

~~~yaml
stage_overrides:
  preprocessing:
    runtime:
      max_seq_len: 60000
  thinker:
    runtime:
      max_seq_len: 60000
~~~

同时保留单 GPU 的 image/audio encoder、Thinker、Talker 和 Code2Wav 显存预算。
Dockerfile、模型 worker、placement 和相关 launcher 修改使 action-score 运行时可以
在自包含 Docker 环境中启动。

## 协议和返回结果

正式接入说明位于
[multimodal_session_realtime.md](/home/ubuntu/fanshide/sglang-omni/docs/developer_reference/action_inference/multimodal_session_realtime.md)。

当前 turn.result 重点字段：

- reply：当前为 null；
- action：Top-1 candidate、action_id、execute、PPL 和执行绑定；
- scores：全部候选排序和 token scores；
- media_summary：音频 chunk、图片帧和两阶段上下文；
- timing.server_turn_ingest_ms；
- timing.server_action_compute_ms；
- timing.server_total_after_commit_ms；
- action_catalog_hash。

session.start 初始化候选目录后，后续 turn 不重复发送候选。候选目录固定，
no_action 也是普通候选；若 Top-1 是 no_action，则 execute=false。

## 验证记录

本次预热实现完成后已执行：

~~~text
py_compile：通过
git diff --check：通过
pytest tests/unit_test/qwen3_omni/test_action_scoring.py
       tests/unit_test/qwen3_omni/test_multimodal_session.py：25 passed
~~~

一次本地启动中的预热结果：

~~~text
category：311.945 ms
child：86.780 ms
total：398.953 ms
ready：true
~~~

预热不计入业务 session；启动后 health 的 total_requests 仍为 0。

## 需要继续关注的事项

1. max_seq_len=60000 同时限制 preprocessing 和 thinker。提高该值仍需重新评估
   KV cache、显存和并发，不能只修改一个 YAML 数值。
2. MAX_IMAGES_PER_TURN=64 是服务保护值，不是模型硬限制；长 turn 的图片抽帧策略
   仍需配置化。
3. session 状态只在单进程内保存；多副本部署必须使用 session 粘性路由，或增加
   外部 session 状态存储。
4. PPL 是候选 suffix 的续写概率指标，不等同于绝对动作置信度；应保留完整 scores、
   token scores 和 top-1 margin。
5. 当前测试验证动作评分链路和协议行为，不验证外部动作播放器、骨骼绑定或动画
   渲染是否真正执行。


## Session 级候选前缀预填充和历史 KV 边界

动作候选在 `session.start` 中固定后，服务端为 hierarchical 的类别候选或
flat_children 的全部具体候选建立稳定的 catalog cache namespace，并通过
`session.started.action_prefix_prefilled` 报告预填充是否成功。动作 prompt 的固定
catalog 位于最前面，已完成历史位于其后，当前 turn 位于最后。预处理阶段使用实际
多模态 token 计算“固定候选 + 已完成历史”的安全边界；跨 turn 的 parent prefix
不会复用当前 turn 的音频、图片或数字人状态。

## flat_children 候选构造优化

在方案 B 的短 `candidate_id` suffix 场景下，候选 suffix 采用独立编码模式，直接拼接到
已生成的共享 prefix token 后，不再为每个候选重复编码整段多模态 prompt。共享 prefix
请求先进入 Thinker scheduler；prefix 完成并取得首个 suffix token 的 logprob 后，服务端
才按 `micro_batch_size` 延迟创建候选请求。上一批完成后再创建下一批，避免在 prefix 入队
前一次性构造全部候选的 `prefix + suffix` 请求。

该改动不改变 prompt 语义、候选排序或 PPL 公式，只减少 CPU 侧 tokenization、请求对象
构造和 scheduler queue wait。描述性 suffix 仍使用精确边界 tokenization，不使用短 ID 优化。

## hierarchical 类别兜底与描述元数据边界

当前 realtime 动作链路保持 action-only：turn.commit 只执行动作评分，不进入普通
文本、语音或回复生成链路；每轮实际动作以 `[action_state]` assistant 历史记录，
供后续 turn 处理“刚刚那个动作”等指代。

hierarchical 默认仍为类别 Top-1 后评分选中类别 children，保持原有 prompt、候选
顺序和 PPL/logit 计算不变。新增可选环境变量
`SGLANG_OMNI_ACTION_CATEGORY_TOP_K`（默认 1，范围 1–3）：设置为大于 1 时，
将类别阶段 Top-K 类别的 children 合并到第二阶段评分，用于缓解类别边界模糊时的
候选漏选；该模式会增加第二阶段候选量和耗时。

类别的 `source_label` 和 `short_definition` 属于外部动作目录提供的语义元数据。服务端
按传入文本原样渲染，不做压缩、去重或改写；其内容参与 action catalog hash 和
prefix cache identity。需要压缩类别描述时，应在 session.start 之前由外部目录完成，
并重新验证动作准确率和 cache 命中。
