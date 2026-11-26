# Prompt 旧入口退役

2026-11-04：先提交并标记 `rollback/prompts-18-20261104`，再清理旧入口。Knowledge 18 个具体发布版本与启动配置基线位于 Knowledge 仓库 `prompt-baselines/2026-11-04-before-prompt-retirement`；凭据保存在仓库外受限目录。

- stateless Performance 迁至 `sglang_omni/serve/performance_control/`；保留规则、评分请求、结果解析和服务协议。
- embedded TTS 迁至 `serve/embedded_tts.py`，文本／缓冲工具迁至 `serve/tts_text.py`、`serve/tts_buffer.py`，保留推理网关的真实依赖。
- 迁出共享依赖并核查全部入向引用后，删除旧 `serve/realtime/` 会话包、`/v1/realtime`、管理／调试路由、文件热覆盖、CLI 开关、无消费者的可选 Reply 注入和对应旧测试／入口示例。
- `/v1/chat/completions/realtime`、`/v1/action-decision/realtime`、`/v1/performance-control/realtime` 是新版内部推理传输协议，保留。外部 TTS 服务自身的 `/api-ws/v1/realtime` 也保留。
- 明确拒绝已经退役的 `runtime_prompt_profile` 请求参数，不暗中注入第二份 system。新版宿主仍发送的 `runtime_prompt_language` 语言字段继续接受，它不会触发规则注入。
- 训练／模型规则与有独立协议的模型文件不在本次删除范围；历史研究记录按其日期归档，不再作为接入说明。
- AI News、角色第 4／6 章、Knowledge 重启加载快照方式均不变。

## E67 动作素材与补跑（2026-11-04）

端口 18004 的实际监听进程、`deploy/launch_dual_gpu_dev.sh`、在线健康与 Prompt 状态共同确认，当前 E67 素材根目录为：

```text
/data/xingmt/model_repo/action_prediction_model/assets/e67_v116
```

使用其中的 `action_pair_token_map_current.json`、`action_pair_token_map_full.json`、`action_catalog.json`、`agent_action_policy.json`。模型目录为 `/data/xingmt/action_omni_output/fp8_overlay_e67_v116_step695`，目录版本 `dmd-action1000-reviewed-116`。四份文件的摘要、在线绑定及 tokenizer 对完整映射的校验见 Knowledge `prompt-baselines/e67-assets-revalidation.json`。

测试原先默认上一级 `assets`，导致三个跨模块测试找不到素材。`tests/unit_test/serve/test_action_decision_service.py` 的共享测试配置现与部署路径一致，仍支持显式 `ACTION_ARTIFACT_ROOT` 覆盖。没有复制／修改正式素材、生产代码、启动配置或 Prompt 发布选择。测试配置修复不需要重启模型服务。

三项无跳过补跑：

```bash
.venv/bin/python -m pytest -q \
  tests/unit_test/serve/test_prompt_runtime.py::test_normal_action_and_performance_use_frozen_releases_until_restart \
  tests/unit_test/serve/test_prompt_debug.py::test_action_debug_uses_real_engine_and_keeps_reference_prefix_immutable \
  tests/unit_test/serve/test_prompt_debug.py::test_e67_uses_character_behavior_in_final_model_request_and_keeps_snapshot_isolated
```

结果 3 passed。相关动作／Prompt／Performance／网关／TTS／协议回归 267 passed，包含上述三项，不累加。真实 E67 原生 HTTP、单次草稿调试及 Knowledge → D → Gateway → E67 均通过，调试不改变已加载固定快照，不以动作准确率验收。

扩展检查另发现 `test_global_action_catalog.py` 中两项既有全局目录测试期望 117，而其仓库目录实际为 82 项、其中英文定义 27 项。已在清理前 `rollback/prompts-18-20261104` 独立复现同样失败；素材和 loader 摘要及两个测试函数 AST 均未改变。它们不读取 E67 外置素材目录，本次不修改该目录或放宽测试。原始扩展检查结果为 295 passed、2 failed，不能称全仓测试通过。

## 全局目录旧断言修复（2026-11-04）

已追溯目录来源：`e322f40f` 将 117 项变为 54 项；`6d1d8850` 更新为当前 21 类／82 项，其中 27 项显式英文表达。端口 18005 的实际启动日志、当前默认 loader 和文件均对应 `sha256:fe53df8527dc8074d59c0836ddfa26fac8981357520e838116b6c7b449f245a5`。原两项失败是旧 117 项测试契约未更新，未发现该进程加载了错误素材。新版 E67 端口 18004 继续使用独立 116 项目录。

修复不是仅改数字：`tests/unit_test/fixtures/limited_action_catalog_contract.json` 固定精确候选／类别 ID、文件 SHA、规范化目录摘要、27 项英文规则身份和召回样例覆盖；测试逐项校验中英文、用户／主动来源、已有回退和最终候选行，验证启动 loader、预填充及公共／私有前缀边界。生产素材、规则正文和服务配置未修改。

同源的旧 117 项召回样例也已更新：当前 171 项 = 82 个目录正样例 + 67 条自然表达 + 12 个意图边界 + 10 个不支持样例；32 条不在当前目录的旧自然表达保留在 `excluded_natural_cases`，没有转成臆测的负例，原 238 项版本仍在回滚标签中。生成器原始自然表达表未删除。相关缓存测试移除了对已删除 Router v5 私有常量的引用，按当前 v1.0 生成式 Router 验证其未被误认作公共目录前缀。

相关回归 314 passed；最后加强候选行检查后，46 项专项复跑通过（包含于相关模块，不累计）。在线执行四组原生目录 prefix prefill 均 ready，中文与英文各对 83 项（82 + `000`）完成真实模型评分，均命中前缀缓存、分数有限。没有触发数字人执行动作或第 6 章主动会话；不评价动作准确率。首次验证中文用户预填充在 40 秒上限超时，后续使用 120 秒验收上限通过；生产 0.001 秒启动预热预算保持原值，不能将验证成功视作所有启动都会完成预热。

完整来源与运行证据见 Knowledge `docs/global_catalog_revalidation_CN.md` 和 `prompt-baselines/global-catalog-revalidation.json`。本次仅更新测试、样例生成器和说明，无需重启服务；18 个 Knowledge 固定版本、AI News、第 4／6 章保持原样。
