# 单 token 动作评分

## 目标

直接动作模式把每个逻辑动作和独立决策标签映射到一个已有词表 token。Thinker 完成共享
prefix 后，只读取这些 token 的 next-token logprob，不再创建候选 suffix 请求。

- 逻辑 `candidate_id`、`action_id` 和执行绑定保持不变。
- 当前受限目录为 117 个动作，加 `000` 和 24 个 IB/IF/IR/IV 标签，共 142 个映射。
- 实际请求只探测本轮真实候选。关闭视觉决策组时为 130 个，不补空位。
- enforce 模式的物理请求数为 1，suffix batch 数为 0。

## 配置

`SGLANG_OMNI_ACTION_SINGLE_TOKEN_MODE` 支持：

- `off`：原 suffix-PPL 路径。
- `shadow`：新旧路径同时评分，旧路径仍为权威结果；日志记录逐标签新分数和各组 winner。
- `enforce`：单 token 结果成为权威结果，跳过全部 suffix 请求。

可用 `SGLANG_OMNI_ACTION_SINGLE_TOKEN_MAP` 指向自定义映射；未设置时使用打包的
`sglang_omni/assets/action_single_token_map.json`。加载时会校验目录 hash、候选全集、映射
hash 和唯一性；模型进程首次请求还会逐项校验文本确实编码为指定的一个非 special token。

## 生成与校验映射

```bash
.venv/bin/python scripts/generate_action_selection_token_map.py \
  --model-path /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --catalog sglang_omni/assets/character_limited_action_global_catalog.json \
  --output sglang_omni/assets/action_single_token_map.json

.venv/bin/python scripts/generate_action_selection_token_map.py \
  --model-path /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --catalog sglang_omni/assets/character_limited_action_global_catalog.json \
  --output sglang_omni/assets/action_single_token_map.json --check
```

目录或 tokenizer 变化时必须重新生成，不能沿用旧映射。

## 校准

shadow 结果的 `stats.single_token_shadow_scores` 保存应用当前 bias 后的所有分数；同时记录
action、IB、IF、IR、IV 各组的新旧 winner。收集覆盖真实流量分布的 shadow 样本后，可生成
组内先验校准：

```bash
.venv/bin/python scripts/calibrate_action_selection_tokens.py \
  --mapping sglang_omni/assets/action_single_token_map.json \
  --logs logs/realtime/.../*.jsonl \
  --output /tmp/action_single_token_map.calibrated.json \
  --version calibration-v1 \
  --min-samples 100
```

先用独立评测集验证校准文件，再通过 `SGLANG_OMNI_ACTION_SINGLE_TOKEN_MAP` 加载。不要用同一批
样本同时拟合和验收。

## 切换门槛与回滚

建议至少检查动作 top-1、unsupported、IB/IF/IR/IV 各组准确率和关键安全门控召回率；并分别
统计文本、音频、相机输入。达到门槛后把模式切到 `enforce`。出现准确率回退或映射校验失败时，
切回 `shadow` 或 `off` 并重启服务；逻辑动作协议不需要回滚。
