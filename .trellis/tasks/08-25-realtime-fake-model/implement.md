# Realtime 本地开发模型替身：实施计划

## 阶段 0：实施前本地 Git 存档

- [ ] 用户批准本任务最终规划。
- [ ] 仅暂存本任务目录和明确属于本任务的已确认文件，不混入 TTS 任务或其他未提交文档。
- [ ] 展示提交信息与精确文件列表并获得确认。
- [ ] 创建本地 checkpoint commit，不 push；确认成功后才执行 `task.py start`。

## 阶段 1：配置与状态机

- [ ] 实现严格的环境变量 config 与脱敏摘要。
- [ ] 实现窄 client Protocol 和 `DevRealtimeModelClient`。
- [ ] 实现 GenerateRequest 校验、pass 分类、固定 chunk 输出和 stop chunk。
- [ ] 实现 abort bookkeeping 和 async generator 清理。
- [ ] 单测覆盖所有合法/非法状态，不启动 FastAPI 或 Pipeline。

验证：运行新增 dev model 单元测试；检查所有 chunk 拼接严格等于配置文本。

## 阶段 2：无 Pipeline 启动分支

- [ ] 在 `_run_server()` 创建 runner 前解析 enabled。
- [ ] 提取必要的共享 uvicorn serving 辅助函数，避免复制整段生命周期代码。
- [ ] 开发模式直接构造 client + app；不创建 runner、coordinator、profiler 或 watcher。
- [ ] 正式模式保持当前启动与清理顺序。
- [ ] 测试启用/关闭两条分支的构造调用和资源清理。

验证：用构造时抛错的 runner fake 证明开发模式完全未触碰 Pipeline。

## 阶段 3：本地 Realtime 集成

- [ ] 使用真实 `create_app()` 和 TestClient/WebSocket 测试完整会话。
- [ ] 输入测试 PCM 并触发 VAD，断言固定 response 和 transcription 事件顺序。
- [ ] 验证请求校验看到 audio metadata、messages 和 text-only modality。
- [ ] 验证 WebSocket 断开清理和重复 turn 的确定性。
- [ ] 验证不支持端点返回明确错误。

## 阶段 4：中文文档

- [ ] 增加环境变量示例、启动命令、测试音频用法和关闭方式。
- [ ] 明确“仅用于开发、不是 TTS 迁移、服务器联调前关闭”。
- [ ] 在 TTS 任务中将本任务列为前置质量门。

## 最终检查

- [ ] 无 GPU、模型和 Pipeline 也能完成 Realtime 固定文本回路。
- [ ] disabled 路径零行为变化。
- [ ] 没有新增 TTS、PCM 或音频回复逻辑。
- [ ] 环境变量、错误、日志和异步生命周期符合 backend spec。
- [ ] 相关 pytest 与 pre-commit 检查通过。
