# 基于部署 tokenizer 的动作单-token ID 映射

本文说明如何使用实际部署的 Qwen3-Omni tokenizer，为所有 category 和 action
离线生成、验证并版本化一套全局唯一的单-token 短 ID。该流程不修改 tokenizer
词表或模型权重。

## 为什么不能继续假设 A000 是单 token

部署配置 `deploy/config_single_gpu.yaml` 中的 FP8 Thinker 和 Instruct checkpoint
使用相同的 Qwen2Tokenizer。实测结果如下：

~~~text
A000 -> [32, 15, 15, 15]
A328 -> [32, 18, 17, 23]
B051 -> [33, 15, 20, 16]
~~~

因此 `A000/B051` 是短字符串，但不是单 token。新的离线工具从实际 tokenizer
词表中筛选满足以下全部条件的字符串：

- 已存在于模型词表，不调用 `tokenizer.add_tokens`；
- 独立编码恰好得到一个 token ID；
- 单 token 解码后与原字符串完全一致；
- 不是 BOS、EOS、PAD 或其他特殊 token；
- 无前后空白且符合配置的 ID 正则；
- category_id 与 child candidate_id 在同一池中全局唯一。

默认策略使用 `[A-Z]{2}`，并排除 `token_id < 3000` 的高频片段。当前部署
tokenizer 中共有 484 个可用 ID。现有 canonical catalog 生成 65 个类别和
387 个动作（包含工具补充的 no_action 类别与动作），合计使用 452 个，剩余 32 个。

## 生成映射和 realtime catalog

在仓库根目录执行：

~~~bash
python scripts/qwen3_omni_action_token_mapping.py generate \
  --model-path /data/models/Qwen3-Omni-30B-A3B-FP8 \
  --verify-model-path /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --catalog tests/data/actions/character_action_catalog.json \
  --mapping-output results/qwen3_omni_action_single_token_mapping.json \
  --catalog-output results/qwen3_omni_action_single_token_catalog.json
~~~

`mapping-output` 是审计和升级使用的 manifest，包含：

- tokenizer 词表 fingerprint 与特殊 token ID；
- source catalog SHA-256；
- ID 正则、token ID 范围和分配 salt；
- stable_key、short_id、token_id、category/action 类型；
- active/retired 状态；
- assignment SHA-256、容量和剩余数量。

`catalog-output` 包含可直接放入 `session.start.action_candidates` 的两层目录：

~~~python
document = json.load(open("results/qwen3_omni_action_single_token_catalog.json"))
session_start["action_candidates"] = document["action_candidates"]
~~~

业务 `action_id` 保持 canonical catalog 中的永久 ID；只有模型实际评分的
`category_id/candidate_id` 会替换成单-token ID。

## 验证已有映射

上线、模型升级或 tokenizer 文件变更后必须执行：

~~~bash
python scripts/qwen3_omni_action_token_mapping.py validate \
  --model-path /data/models/Qwen3-Omni-30B-A3B-FP8 \
  --verify-model-path /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --catalog tests/data/actions/character_action_catalog.json \
  --mapping results/qwen3_omni_action_single_token_mapping.json \
  --runtime-catalog results/qwen3_omni_action_single_token_catalog.json
~~~

以下任一情况都会返回非零退出码：

- tokenizer fingerprint 改变；
- 某个 ID 不再是一个 token，或 token ID 发生变化；
- 使用了特殊 token 或超出声明的 token ID 范围；
- category/action 出现 short_id、token_id 或 stable_key 冲突；
- source catalog SHA-256 改变；
- active stable_key 与 canonical catalog 不一致；
- runtime catalog 不是由当前 mapping 和 source catalog 生成。

## 目录增量更新

更新 canonical catalog 时传入上一版 manifest：

~~~bash
python scripts/qwen3_omni_action_token_mapping.py generate \
  --model-path /data/models/Qwen3-Omni-30B-A3B-FP8 \
  --catalog path/to/new_catalog.json \
  --previous-mapping path/to/previous_mapping.json \
  --mapping-output path/to/new_mapping.json \
  --catalog-output path/to/new_runtime_catalog.json
~~~

工具会保留已有 stable_key 的分配。已删除动作会变为 `active=false`，其 short_id
仍被保留，不会分配给新动作。这避免客户端日志、历史 session 和离线数据中的旧 ID
在新目录中指向另一个动作。

## 正确性和上线门禁

单-token 映射只解决 suffix 长度和全局唯一性，不自动保证动作准确率。不同预训练
token 具有不同先验概率，即使 prompt 中提供了完整动作定义，仍可能引入排序偏置。
生产上线前必须使用固定评测集比较：

- 原短 ID 与单-token ID 的 category Top-1；
- 最终 action Top-1 和 no_action 准确率；
- category/child 的 mean_logprob 与错误分布；
- `server_action_compute_ms`、suffix batch 和端到端 p50/p95。

当前 scorer 会在短 ID 后显式追加 terminal token。因此 ID 本身虽然只有一个 token，
实际参与评分的 suffix 通常仍包含 `ID token + terminal token` 两个 token。

禁止仅使用 `tokenizer.add_tokens` 创建 `<ACT_001>`。没有同步扩展和训练模型 embedding
及输出头时，新 token 的 logprob 不具备可靠可比性。
