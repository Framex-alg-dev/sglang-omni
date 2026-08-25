# Session Realtime 内嵌流式 TTS：实施计划

## 前置门：新接口 fake model

- [ ] `08-25-realtime-fake-model` 已完成并归档。
- [ ] 无 GPU 环境可通过 `/v1/session/realtime` 验证 text/action/text+action、重复 Turn 和真正取消。
- [ ] 旧 `/v1/realtime` 开发代码和 smoke 已删除。
- [ ] 开始本任务前创建精确本地 checkpoint，不 push。

## 阶段 A：锁定 Session 输出合同

- [ ] 在现有协议解析器中加入 audio 和集中 `SessionOutputCapabilities`。
- [ ] 实现五种允许组合与两种拒绝组合的表驱动测试。
- [ ] `session.started.outputs` 回显规范化值。
- [ ] 为 text/action 无 audio 路径建立完整事件快照基线。
- [ ] 更新客户端协议文档中的 outputs、事件和错误码。

验证：协议单测；无 audio 快照与合并分支基线一致。

## 阶段 B：provider 配置与 fake TTS

- [ ] 实现 provider URL、voice、凭证、timeout、queue/buffer 上限配置。
- [ ] 删除/禁止 TTS enabled 参数；audio Session 才校验 provider readiness。
- [ ] 实现可编程 fake TTS WebSocket server。
- [ ] 覆盖 ready、delta、done、cancel、错误和连接复用协议。
- [ ] 日志脱敏测试。

验证：adapter 单测，不启动模型、GPU 或外网。

## 阶段 C：EmbeddedTTSTurn

- [ ] 实现单 reader 的内部 WebSocket 连接状态机。
- [ ] 实现有界文本 queue、并发 sender/reader 和一次 commit。
- [ ] 严格解码音频事件、Base64 与 done 顺序。
- [ ] 实现 cancel、Broken 连接关闭和幂等 teardown。
- [ ] 添加 connect/ready/send/first-audio/turn timeout。

验证：正常、慢服务、背压、非法协议、提前关闭、无 done、重复 done、取消。

## 阶段 D：普通 text+audio 编排

- [ ] 在 `MultimodalSession` Turn owner 中按 audio capability 创建 TTS Turn。
- [ ] 模型 Delta 同时驱动现有 text event 和 TTS queue。
- [ ] 音频到达即按 seq 外发 `response.audio.delta`。
- [ ] text EOF 后发送 text done 并 commit TTS。
- [ ] audio done 后发送 audio done；两个分支完成后发送 response done。
- [ ] `turn.result.outputs.audio` 与 reply/action 状态统一聚合。

验证：首音频早于 text done、尾音频、空正文、外部发送失败和无 audio 零副作用。

## 阶段 E：action provisional 集成

- [ ] provisional 文本提前送 TTS，音频写入有界隔离缓冲。
- [ ] promoted 后按序释放缓冲并继续实时输出。
- [ ] `replayed_from_provisional` 不重复送入 TTS。
- [ ] discarded/unsupported/cancelled 清理 TTS 和缓冲，禁止音频泄漏。
- [ ] 预录 unsupported audio 分支不调用在线 TTS。

验证：promoted、discarded、unsupported、action failure、buffer overflow 和 race 测试。

## 阶段 F：真正取消与错误收敛

- [ ] `turn.cancel` 竞争安全地取消 model/action/TTS 全部任务。
- [ ] 只发送一次 `turn.cancelled`，取消后无其他终态。
- [ ] session.close、断线、服务关闭幂等清理。
- [ ] 实现 fail_turn 错误矩阵及 partial/failed 输出状态。
- [ ] 迟到上游事件按 generation token 丢弃。

验证：各取消检查点、正常完成竞争、provider cancel 失败和重复 teardown。

## 阶段 G：本地与服务器验证

- [ ] 扩展 fake model smoke 支持 text+audio 和 text+audio+action。
- [ ] 运行真实 FastAPI + fake model + fake TTS 进程外测试。
- [ ] 验证事件关联、seq、Base64 PCM 和完整终态。
- [ ] 在 Linux/服务器完整依赖环境运行相关 pytest、pre-commit 和类型检查。
- [ ] Docker 联调数字人后端，确认 outputs 移除 audio 即可回滚且不会重复 TTS。

## 阶段 H：文档与观测

- [ ] 更新组长交流设计文档和正式客户端接入指南。
- [ ] 给出五种 outputs 示例、audio event schema、取消和错误处理。
- [ ] 增加延迟、buffer、失败、取消指标与脱敏日志。
- [ ] 删除旧接口、旧开关、旧事件名和 no-op cancel 描述。

## 最终验收

- [ ] outputs 能力解析集中且便于扩展。
- [ ] audio 未请求时无 TTS 连接、任务、事件和延迟回归。
- [ ] audio 请求时文本/音频真正并发，完成顺序正确。
- [ ] provisional 音频不提前播放且不泄漏。
- [ ] turn.cancel 真正取消并幂等终止。
- [ ] 错误不静默降级，资源无泄漏。
- [ ] fake、本地、Linux 完整测试与文档通过。
