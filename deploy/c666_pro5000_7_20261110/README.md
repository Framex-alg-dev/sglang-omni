# C666 动作与表情服务资产｜2026-11-10

7号机 `10.0.0.45:18048` 使用 `foundation-C-step666-fp8`，服务目录为 `/data/xingmt/c666_action_20261110`。

本目录保存当前已验证的 C666 business111 目录、输出映射及 r1 场景定义。`assets/action_catalog.json` 的每个具体动作都有 `output_code`，表示模型输出字符串，与 `candidate_id` 分开。例如双手点赞 `candidate_id=271`、`output_code=_v_o`；疑惑 `candidate_id=162`、`output_code=_j_k`。当前共98个身体动作和13个表情。`000/_ctrl_a` 是不支持控制输出，不是具体动作。

旧 `sglang_omni/assets/persona_window_v1` 是 E68-A/E67 的116码本冻结资产，保留用于旧模型和回归测试，不能用作 C666 部署目录。C666 也不能加载旧版场景文件：场景中的编码和目录、映射哈希必须配套。

部署准备（在独立发布目录内完成，勿覆盖运行中的代码）：

```bash
release_root=/data/xingmt/c666_action_20261110
cp -a deploy/c666_pro5000_7_20261110/assets/. "$release_root/assets/"
cp -a deploy/c666_pro5000_7_20261110/assets/. "$release_root/code/sglang_omni/assets/persona_window_v1/"
cp -a deploy/c666_pro5000_7_20261110/usage_scenarios/. "$release_root/code/sglang_omni/assets/"
```

运行脚本、配置和用户级 systemd 单元随附。模型权重及现有鉴权文件不入库；单元沿用 `/data/xingmt/e68_a_action_20261109/service.env` 的认证配置。模型使用 GPU6、`max_running_requests=1`，禁用 radix/shared cache。不要将示例目录与其他版本权重混用。

同一接口 `POST /v1/action-recall` 支持身体／表情、用户文本／音频、数字人当前及历史文本／音频，以及 `mode=action_request` 时使用 `user_input.text` 指定下一刻目标。完整参数见 [接口说明](../../docs/developer_reference/action_inference/speech_action_recall_agent_CN.md)。未改 Agent 服务。

变基验证：`codex/speech-action-windows` 基于 `codex/character-action-sequences@84fd3d8`。独立测试实例使用当前 C666 权重和本目录资产，6种实际文本／音频调用通过；16条指定目标诊断正确15条，单手点赞误选双手点赞仍是已知语义问题。35条视觉回归输出与同约束解码的源 C666 逐条一致：计算15/16、模仿5/7、识别9/12。这些是固定回归结果，不是独立泛化指标。
