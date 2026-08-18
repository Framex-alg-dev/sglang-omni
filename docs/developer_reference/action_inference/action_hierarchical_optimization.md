# hierarchical 动作推理优化总结

本文集中说明 Qwen3-Omni 动作推理 hierarchical 模式的完整链路、已经实施的优化、实际收益和剩余瓶颈。

相关文档：

- [动作推理方案总览](action_inference_scheme_summary.md)
- [动作 suffix 评分](action_suffix_scoring.md)
- [动作耗时演进](action_latency_history.md)
- [micro-batch 性能验证](action_batch_benchmark.md)
- [Realtime 外部接入协议](multimodal_session_realtime.md)

## 1. 当前 hierarchical 链路

一个外部 turn 对应一个 session 内的一个逻辑请求，内部顺序执行两个评分阶段：

~~~text
当前 turn 音频、图片、文本、数字人状态
        +
session 最近历史和动作历史
        │
        ▼
category 阶段
  固定 system prompt：所有一级类别
  suffix：B001、B002、...
  选择 Top-K 类别
        │
        ▼
child 阶段
  临时 system prompt：Top-K 类别下的 children
  suffix：A001、A002、...
  选择最终具体动作
        │
        ▼
action_id / candidate_id / category_id / execute
~~~

这不是两个 session，也不会向外部暴露两个 turn。child 阶段的 system prompt 只在当前逻辑请求内部构造，不追加到 session 历史。

当前语义保持不变：

- 候选动作定义放在 system prompt；
- suffix 只使用短 candidate_id；
- 每个候选都计算 token logprob、mean logprob、NLL 和 PPL；
- PPL 越低、mean logprob 越高，候选排名越靠前；
- no_action 和其他动作一起参与评分；
- no_action 被选中时返回 execute=false；
- 不生成普通数字人回复，只保存动作结果到 session 历史。

## 2. 优化目标

hierarchical 的主要目标不是消除两阶段逻辑，而是减少每个阶段的无效工作：

1. 减少候选 suffix 的 token 数量；
2. 避免在共享 prefix 进入 scheduler 前构造全部候选请求；
3. 复用 session 级固定候选 prefix 的 KV cache；
4. 复用同一 turn 的音频、图片和历史媒体上下文；
5. 避免 child 候选在类别尚未确定时提前预热；
6. 降低多模态预处理、编码器和 scheduler 的重复开销；
7. 让每个阶段的耗时和缓存状态可以被日志验证。

## 3. 已实施的优化

### 3.1 方案 B：短 candidate_id suffix

category 和 child 都使用短 ID 作为 suffix：

~~~text
category：共享上下文 + B051
child：   共享上下文 + A328
~~~

不再对每个候选重复拼接完整动作描述。动作描述仍保留在 system prompt 中，因此动作语义没有丢失。

收益：

- 每个候选 suffix token 数显著减少；
- 不同候选之间的 token 长度更稳定；
- PPL 计算仍完整覆盖 suffix token；
- 减少候选请求构造和 scheduler 排队压力；
- 不改变动作语义和动作映射。

历史测试中，完整描述 suffix 的 flat 路径约为 5 秒级；切换到短 ID 后，flat 路径下降到约 0.6–0.7 秒级。hierarchical 因为每阶段候选较少，通常稳定在数百毫秒级。

### 3.2 共享 prefix 和 KV cache

候选 system prompt 放在动态上下文前面：

~~~text
system：固定候选目录
历史：  历史用户输入和动作状态
当前：  当前 turn 媒体、文本、数字人状态和动作指令
~~~

session.start 时：

- hierarchical 预填充所有 category 的固定候选 prefix；
- flat_children 预填充所有具体动作候选 prefix。

后续请求通过 prefix_cache_namespace 复用固定候选内容。缓存边界不包含当前 turn 的媒体、状态和最新指令，避免动态内容污染共享 KV。

验证字段：

- prefix_cached=true；
- cached_prefix_token_count；
- candidate_prefix_recompute_tokens=0。

### 3.3 候选请求延迟构造

候选评分的执行顺序是：

1. 先构造一个共享 prefix 请求；
2. prefix 完成后取得候选首 token 分布；
3. 再按 micro-batch 延迟构造 suffix 请求；
4. 每批完成后聚合 token logprob 和 PPL；
5. 最后返回完整候选排序。

这样避免在 prefix 进入 scheduler 前一次性生成全部 prefix+suffix token 数组，也避免大量无效候选请求提前占用 CPU 和队列。

该优化对 flat_children 更明显；hierarchical 的每阶段候选数量较少，但两个阶段内部仍使用同一机制。

### 3.4 micro-batch 控制

环境变量：

~~~text
SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE
~~~

规则：

- 默认值为 64；
- 合法范围为 1–256；
- flat_children 和 hierarchical 统一使用；
- category 阶段和 child 阶段分别按该值分批；
- 不改变候选顺序、prompt、PPL 或动作映射。

hierarchical 的 category 和 child 通常已经各自一批，因此继续增大 batch 的收益有限。batch 主要影响批次数，不能消除两个阶段本身的模型计算，还可能增加单批显存压力。

### 3.5 category prefix 在 session.start 预填充

category 候选目录在整个 session 内固定，因此 session.start 会预填充类别 prefix。

这部分预热：

- 不携带业务音频、图片和历史；
- 不产生动作结果；
- 不写入 session 历史；
- 只初始化 tokenizer、scheduler、Thinker 和固定类别 prompt 的 KV cache 路径。

### 3.6 child prefix 延迟缓存

child 候选依赖 category 结果，不能在 session.start 预填充所有类别的 children，否则会为未选中的类别浪费 prefill、显存和 KV cache。

当前流程：

1. category 阶段完成；
2. 根据 Top-K category_id 构造 child namespace；
3. 第一次使用该 namespace 时执行 child prefix prefill；
4. 将 namespace 记录到当前 session；
5. 后续 turn 复用该 child prefix，不重复 prefill。

诊断结果中会记录：

~~~json
{
  "child_prefix_prefilled": true,
  "child_prefix_cache_namespace": "hierarchical:...:child:B051"
}
~~~

同一 namespace 的后续 turn 中 child_prefix_prefilled 为 false。

第一次使用某个 child namespace 仍需支付一次 prefill 成本；优化目标是避免所有类别提前支付，并在后续 turn 摊薄成本。

### 3.7 category/child 同 turn 多模态上下文复用

同一个 turn 的 category 和 child 请求具有相同的：

- 当前音频；
- 当前图片帧；
- 历史音频；
- 历史图片；
- 历史消息媒体占位关系；
- 当前 turn 的多模态输入集合。

两个阶段使用同一个 action_context_cache_key。category 预处理后，在 preprocessing 进程内暂存已加载的媒体对象；child 阶段直接复用，避免重复媒体加载和解码。

缓存特征：

- category 写入；
- child 一次性读取并消费；
- 最多保留 4 个逻辑上下文；
- child 使用后释放；
- 音频、图片和视频派生音频的 cache identity 保持一致；
- category 和 child 的文本 prompt 仍分别构造，因为两者的 system prompt 不同。

这项优化复用的是同一 turn 的完整媒体上下文，不是把 category 文本 prompt 直接复制成 child prompt，因此不会改变两阶段的文本语义。

日志中可观察：

~~~text
action_context_cache_status=miss   # category 首次处理
action_context_cache_status=hit    # child 复用
~~~

### 3.8 音频和图片编码器缓存

image/audio encoder 还有独立的输出缓存：

- category 阶段首次计算 encoder output；
- child 阶段使用相同媒体 cache key 时命中 encoder output；
- 不重复执行同一 turn 的音频或图片 encoder；
- cache key 包含媒体内容摘要和关键预处理参数，不只依赖文件名。

典型日志：

~~~text
audio_encoder cache_status=compute
audio_encoder cache_status=hit
~~~

因此 child 仍需构造自己的文本 prompt，但不再重复运行同一 turn 的媒体 encoder。

### 3.9 action-only 路径

动作推理模式下，Thinker 只计算动作候选，不触发：

- 普通文本 decode；
- Talker；
- Code2Wav；
- 数字人语音回复链路。

动作结果会写入 session 历史，支持后续 turn 理解“刚刚那个动作”“再重复一遍”等指代。

### 3.10 历史上下文裁剪

动作评分上下文保留有限的最近内容：

- 最近若干逻辑 turn；
- 最近音频；
- 最近图片；
- 每个 turn 的动作状态；
- 当前 turn 的受限媒体。

裁剪以完整 turn 和媒体占位符为单位，保证消息中的 audio/image 数量与媒体数组一致，避免历史 token 无限膨胀。

当前上下文窗口为 60000，但实际可用空间还受候选 system prompt、媒体展开 token、历史和 KV cache 影响。

### 3.11 action-score 预热和运行时诊断

服务启动时会对当前模式执行 action-score 预热：

- hierarchical：预热 category 路径；
- flat_children：预热一阶段具体动作路径；
- 不带业务 session、历史和媒体；
- 不计入业务 turn 耗时；
- 失败不会阻止服务启动。

运行时日志记录：

- client/server request build；
- scheduler admission/wait；
- prefix prefill；
- suffix batch 数量和耗时；
- audio/image preprocessing；
- encoder cache hit/miss；
- prefix cache 状态；
- candidate prefix recompute tokens；
- GPU 显存和阶段耗时；
- category/child 各自的动作结果和 Top-1。

## 4. 实际耗时结论

近期实际 session 日志中，常规多模态 hierarchical turn 大致表现为：

| 项目 | 典型范围 |
|---|---:|
| 完整动作推理 | 约 280–355 ms |
| category 阶段 | 约 180–240 ms |
| child 阶段 | 约 100–140 ms |

不同 session 的音频长度、图片数量、历史长度和 scheduler 状态不同，因此该范围用于趋势观察，不是严格 benchmark。

首个 turn 不一定总是更慢。有些 session 会出现首次多模态路径或 scheduler 预热；也有些 session 首个 turn 已处于稳定状态。应分别统计首个 turn、稳定态 median 和 p95。

hierarchical 通常比 flat_children 快，是因为它将候选空间拆成“所有类别”和“选中类别下的少量 children”。flat_children 需要一次评分全部具体动作，system prompt 也包含全部具体动作定义。

## 5. 当前仍存在的成本

以下成本尚未完全消除：

1. category 和 child 的 Thinker 评分必须串行，因为 child 候选依赖 category 结果；
2. category 和 child 的 system prompt 不同，不能直接共享完整文本 KV；
3. 即使媒体对象和 encoder output 复用，child 仍需构造自己的文本 prompt；
4. 第一次命中某个 child namespace 仍需支付一次 child prefix prefill；
5. category Top-K 大于 1 会扩大 child 候选集合和 child prompt；
6. scheduler queue wait 仍可能受同一 GPU 上其他 Thinker 请求影响；
7. 当前媒体上下文缓存是进程内、一次性消费，不是跨进程持久化缓存；
8. session 历史越长，prompt token 和 KV 管理成本越高。

## 6. 后续可优化方向

### 6.1 先拆分真实耗时

继续把以下阶段分别记录并关联到同一个 logical request：

~~~text
turn ingest
  ├─ media aggregation
  ├─ preprocessing
  ├─ audio/image encoder
  ├─ mm aggregate
  ├─ Thinker scheduler queue
  ├─ prefix prefill
  ├─ suffix batch
  └─ result aggregation / IPC
~~~

当前最应该确认的是 category 阶段中未被现有统计覆盖的等待时间，而不是只看 server_action_compute_ms。

### 6.2 减少 scheduler queue wait

可以评估：

- action-score 请求独立调度优先级；
- 动作评分与普通生成请求隔离；
- 为 action-only 请求使用独立 Thinker worker/GPU；
- 分别统计 media preprocessing、encoder 和 Thinker queue；
- 降低同一 GPU 上其他 stage 对 Thinker 的阻塞。

### 6.3 category 单 child 快速路径

如果某个 category 只有一个确定 child，可以考虑跳过 child 模型评分，直接返回该 child。

必须同时满足：

- category 选择已经有效；
- children 数量确实为 1；
- no_action 规则不会被绕过；
- 不违反“没有明确动作时选择 no_action”的约束；
- 有充分测试证明准确率不下降。

### 6.4 模糊类别 Top-K 兜底

当前默认 SGLANG_OMNI_ACTION_CATEGORY_TOP_K=1。对于类别边界模糊的输入，可以根据 category Top-1 与 Top-2 分差动态决定：

- 分差大：只计算 Top-1 children；
- 分差小：同时计算 Top-2 children；
- 最后仍只返回一个 action。

代价是 child 候选更多、耗时更高。

### 6.5 更深层的上下文/特征复用

当前已经复用：

- 同一 turn 的已加载媒体对象；
- 音频/image encoder output；
- 固定候选目录 KV prefix。

还可以评估是否安全复用：

- multimodal aggregate 结果；
- 预处理后的 media token layout；
- category 到 child 的共享 hidden state；
- 同一 turn 的完整 input embedding。

这需要确保 child system prompt 的文本位置变化不会破坏 M-RoPE、媒体 token 对齐和 suffix logprob 计算。不能仅凭 token 数量相同就直接复用。

### 6.6 外部动作目录预处理

服务端不自动压缩 category 的 source_label 和 short_definition。如果存在重复路径或冗余文本，建议由外部动作目录生成阶段完成：

- 保留能区分类别的最短描述；
- 保证 category_id 和 child_id 全局唯一；
- 固定排序；
- 重新计算 catalog hash；
- 用准确率测试确认压缩没有引入类别混淆。

## 7. 正确性约束

任何后续性能优化都必须保持：

- category 候选完整；
- child 候选只来自选中 category Top-K；
- no_action 始终参与评分；
- category/child 候选顺序稳定；
- prefix_cached=true；
- candidate_prefix_recompute_tokens=0；
- token scores 非空；
- 相同输入下 Top-1 不发生无解释漂移；
- 音频和图片占位符数量与媒体数组一致；
- 当前 turn 用户输入位于最新位置；
- 最新 turn 不伪造 assistant 回复；
- 动作历史进入后续 turn；
- 不触发普通回复生成链路。

## 8. 验证方式

代码回归测试：

~~~bash
/home/ubuntu/miniconda3/envs/sglang-omni-local/bin/python -m pytest -q \
  tests/unit_test/qwen3_omni/test_action_scoring.py \
  tests/unit_test/qwen3_omni/test_multimodal_session.py
~~~

当前实现验证结果：

~~~text
46 passed
py_compile passed
git diff --check passed
~~~

实时验证时建议记录：

- server_action_compute_ms；
- category/child stage elapsed；
- category/child queue wait；
- prefix_cached；
- candidate_prefix_recompute_tokens；
- action_context_cache_status；
- child_prefix_prefilled；
- audio/image encoder cache status；
- suffix batch 数量和大小；
- GPU 显存起止值和峰值；
- Top-1 action 和完整候选排序。

## 9. 代码位置

- Realtime 两阶段逻辑和 child prefix 缓存：sglang_omni/serve/realtime/multimodal.py
- action request 和多模态上下文构造：sglang_omni/client/client.py
- action request 数据结构和 PPL：sglang_omni/models/qwen3_omni/action_scoring.py
- 多模态预处理和同 turn context cache：sglang_omni/models/qwen3_omni/components/preprocessor.py
- 候选 prefix/suffix 请求构造：sglang_omni/models/qwen3_omni/request_builders.py
- scheduler 分批和 prefix cache：sglang_omni/scheduling/omni_scheduler.py
- hierarchical 回归测试：tests/unit_test/qwen3_omni/test_multimodal_session.py
