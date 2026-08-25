# Session Realtime 本地开发模型替身：实施计划

## 阶段 0：重新规划与本地存档

- [ ] 用户审核并批准本次协议迁移后的 PRD、design 和 implement。
- [ ] 不提交当前基于旧 `/v1/realtime` 的未完成适配；实施时按新设计替换。
- [ ] 创建精确的本地 checkpoint，不包含 TTS 文档和其他用户文件，不 push。

## 阶段 1：协议与路由迁移

- [ ] 开发路由从 `/v1/realtime` 改为唯一 `/v1/session/realtime`。
- [ ] 删除旧 RealtimeSession/VAD/transcription pass 专用开发测试与 smoke 逻辑。
- [ ] 复用正式 outputs 校验；本阶段允许 text/action/text+action，明确拒绝 audio。
- [ ] 开发模式关闭 resource monitor，并拒绝所有非 allowlist HTTP/WS。

验证：路由集合、协议版本、非法 outputs 和旧路径回归测试。

## 阶段 2：固定模型与动作能力

- [ ] 从 `MultimodalSession` 调用点提取窄 client Protocol。
- [ ] 固定文本按配置分块输出，保持 request ID、stop 和取消语义。
- [ ] 固定动作从已验证 Session 白名单确定性选择。
- [ ] text+action 复用正式 provisional promoted/discarded 状态机。
- [ ] `turn.cancel` 真正取消所有活动任务并只产生 `turn.cancelled`。

验证：client 单测和真实 Session WebSocket 集成测试。

## 阶段 3：启动路径

- [ ] 独立 dev server 无模型配置启动新接口。
- [ ] launcher fake 分支发生在 action catalog、runner、GPU、warmup、profiler 和 watcher 之前。
- [ ] disabled 分支保持 `catalog -> port -> runner` 生产顺序。
- [ ] 生命周期与重复关闭幂等。

验证：构造即失败的生产依赖证明 fake 路径未触达资源。

## 阶段 4：进程外 smoke

- [ ] 重写脚本为 `session.start/turn.start/input.* /turn.commit`。
- [ ] 覆盖 text、action、text+action、重复 Turn 和真正取消。
- [ ] 校验 ACK、delta、action ready、response done 和 turn result。
- [ ] 超时、乱序、服务 error 和意外关闭非零退出。

## 阶段 5：中文文档

- [ ] 更新独立启动命令、outputs 示例、测试输入和成功输出。
- [ ] 明确旧接口废弃、本阶段 audio 未实现、TTS 由后续任务增加。
- [ ] 更新 TTS 前置质量门。

## 最终检查

- [ ] 无 GPU/Pipeline 完成新 Session/Turn 固定结果回路。
- [ ] 三种非 audio 输出模式与取消均可验证。
- [ ] 生产启动路径和新 action 功能无回归。
- [ ] 代码中无旧 `/v1/realtime` 开发兼容和 TTS 实现。
- [ ] 相关 pytest、格式和静态检查通过；平台依赖阻塞有明确记录。
