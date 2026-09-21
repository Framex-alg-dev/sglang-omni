# 回复与动作提示词的中英文维护

回复提示词使用会话 `language`；动作和表情选择提示词使用 `action_language` / `action_locale`。两者独立选择，不根据当前用户输入临时切换。客户端提供的人设、当前请求、引用内容和上下文按原文保留。

## 运行时覆盖

目录由 `SGLANG_OMNI_RUNTIME_PROMPT_DIR` 指定，默认是仓库 `runtime/`。以下六类文件均按语言隔离：

| 规则 | 中文文件 | 英文文件 |
| --- | --- | --- |
| 通用回复 | `reply_rules.txt` | `reply_rules.en-US.txt` |
| 通用动作 | `action_rules.txt` | `action_rules.en-US.txt` |
| 主动回复 | `proactive_reply_rules.txt` | `proactive_reply_rules.en-US.txt` |
| 主动动作 | `proactive_action_rules.txt` | `proactive_action_rules.en-US.txt` |
| 用户图片回复 | `user_image_reply_rules.txt` | `user_image_reply_rules.en-US.txt` |
| 用户图片动作 | `user_image_action_rules.txt` | `user_image_action_rules.en-US.txt` |

旧文件名按中文处理。英文文件不存在或为空时，使用仓库内英文默认规则，**不会回退到中文覆盖文件**。已有英文会话若此前依赖中文覆盖文件，需要另外维护英文文件。文件内容没有自动翻译，更新一份不会同步修改另一份。

主动规则仍支持 `[session_enter]`、`[idle_timeout]`、`[user_returned]`、`[character_proactive]`、`[session_ending]` 分节。缺少的节使用对应语言的默认规则。

现有管理接口支持可选查询参数 `locale=zh-CN` 或 `locale=en-US`：

- `GET /admin/runtime-prompts?locale=en-US` 获取英文默认内容和覆盖内容。
- `PUT /admin/runtime-prompts/reply_rules?locale=en-US`，请求体仍是 `{"content": "..."}`。
- 不传 `locale` 时仍编辑中文；鉴权方式保持现有管理接口约定。

管理界面未增加语言切换控件；英文规则可通过文件或上述接口维护。

`examples/runtime_prompts/` 保存本次补充的英文图片规则示例，包含本地中文规则中的手势触发表。部署时按需复制到实际 runtime 目录。这些示例不会自动替换仓库默认策略，也不会自动覆盖已有 runtime 文件。后续修改中文触发表时需同步审核英文版本。

代码需要服务重启后加载；加载新代码后，runtime 内容在后续请求构建时读取。模型目录及预热前缀在服务启动时加载，修改目录后也需要重启。

## 动作目录

当前 `character_limited_action_global_catalog.json` 的 25 个类别、117 个动作已补齐英文名称和基础描述，117 个动作都有英文用户响应与主动表达。

`prompt_text_by_locale.en-US` 含 `label` 和 `short_definition`，仅用于模型提示词展示。`expressions_by_locale.en-US` 保留原有结构，并补齐原先缺失的 37 个动作。

原始 `source_label`、`short_definition`、`category_path`、`candidate_id`、`action_id` 均保持原值。名称精确匹配、数字手势识别、执行映射和返回动作标识继续使用原字段。会话对象中的 `prompt_label` / `prompt_definition` 由服务端目录解析产生，不接受客户端用它们改写标准语义。

目录内容与生成的提示词会参与已有哈希计算，因此英文内容变化会使相关缓存标识变化；无需手工修改动作 ID 或客户端执行绑定。

新增动作时应同时补充英文展示字段与两种表达，并运行 `tests/unit_test/serve/test_prompt_localization.py`。旧目录或自定义目录没有展示字段时仍可加载，沿用原文；本次未翻译完整历史目录中的全部动作。

## 本次边界

动作拒绝回复的固定规则、头像标签和独立语言任务提示已按回复语言选择；原有拒绝语气、身体动作边界和失败兜底保持原有行为。

独立意图识别、独立视觉分类器的中文内部协议本次不调整。历史动作记录、外部注入的人设和上下文也不保证整段都是英文；本次完整性测试针对当前限定目录及回复／动作规则的语言选择，不能视为全仓库无中文残留的证明。

单元测试检查语言选择和最终提示词文本，不替代真实模型对语言遵循率、动作准确率及拒绝语气的验收。
