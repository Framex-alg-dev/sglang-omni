# Prompt 旧入口退役

2026-11-04：先提交并标记 `rollback/prompts-18-20261104`，再清理旧入口。Knowledge 18 个具体发布版本与启动配置基线位于 Knowledge 仓库 `prompt-baselines/2026-11-04-before-prompt-retirement`；凭据保存在仓库外受限目录。

- stateless Performance 迁至 `sglang_omni/serve/performance_control/`；保留规则、评分请求、结果解析和服务协议。
- embedded TTS 迁至 `serve/embedded_tts.py`，文本／缓冲工具迁至 `serve/tts_text.py`、`serve/tts_buffer.py`，保留推理网关的真实依赖。
- 迁出共享依赖并核查全部入向引用后，删除旧 `serve/realtime/` 会话包、`/v1/realtime`、管理／调试路由、文件热覆盖、CLI 开关、无消费者的可选 Reply 注入和对应旧测试／入口示例。
- `/v1/chat/completions/realtime`、`/v1/action-decision/realtime`、`/v1/performance-control/realtime` 是新版内部推理传输协议，保留。外部 TTS 服务自身的 `/api-ws/v1/realtime` 也保留。
- 明确拒绝已经退役的 `runtime_prompt_profile` 请求参数，不暗中注入第二份 system。新版宿主仍发送的 `runtime_prompt_language` 语言字段继续接受，它不会触发规则注入。
- 训练／模型规则与有独立协议的模型文件不在本次删除范围；历史研究记录按其日期归档，不再作为接入说明。
- AI News、角色第 4／6 章、Knowledge 重启加载快照方式均不变。
