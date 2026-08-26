# Session Realtime 内嵌流式 TTS：实施计划

## 前置门：新接口 fake model

- [x] `08-25-realtime-fake-model` 已完成并归档。
- [x] 无 GPU 环境可通过 `/v1/session/realtime` 验证 text/action/text+action、重复 Turn 和真正取消。
- [x] 旧 `/v1/realtime` 开发代码和 smoke 已删除。
- [ ] 开始本任务前创建精确本地 checkpoint，不 push。

## 阶段 A：锁定 Session 输出合同

- [ ] 把 `session.start.outputs` 设为 text/audio/action 能力的唯一控制源；删除或拒绝任何
  环境变量、CLI、provider readiness 或默认值对 TTS 启用状态的影响。
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
- [ ] 锁定 provider 协议：建连等待 `session.created`，输入使用
  `input_text_buffer.append/commit`，取消使用 `response.cancel`，完整 Turn 以
  `response.done` 为准；`response.audio.done` 仅作提示。
- [ ] 日志脱敏测试。

验证：adapter 单测，不启动模型、GPU 或外网。

## 阶段 C：EmbeddedTTSTurn

- [ ] 实现 Session-owned、首次 audio Turn 延迟建连的单 reader WebSocket 状态机。
- [ ] 实现同 Session/同 voice 跨 Turn 复用；voice 变化重建；不同 Session 绝不共享。
- [ ] 使用 Python 3.10 兼容的 timeout/cancel 机制，不直接复制 `asyncio.timeout`。
- [ ] 实现有界文本 queue、并发 sender/reader 和一次 commit。
- [ ] 严格解码音频事件、Base64 与 done 顺序。
- [ ] 实现 cancel 后强制丢弃连接、Broken 连接关闭、Session close/断线幂等 teardown；
  正常 Turn 完成保留 Ready 连接。
- [ ] 添加 connect/ready/send/first-audio/turn timeout。

验证：正常、慢服务、背压、非法协议、提前关闭、无 done、重复 done、取消、同 Session
两 Turn 仅一次 connect、跨 Session 两次 connect、voice 改变/取消后重连。

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

- [ ] 复用新合并分支的 `turn.cancel` handler/terminal owner，把 TTS sender、reader、
  queue 等待者和 provisional buffer 纳入同一取消集合。
- [ ] 收到业务端 `turn.cancel` 后立即失效 token，向 provider 尽力发送内部
  `response.cancel`，并关闭连接；不得等待模型/TTS 正常结束。
- [ ] 只发送一次 `turn.cancelled`，取消后无其他终态。
- [ ] session.close、断线、服务关闭幂等清理。
- [ ] 实现 fail_turn 错误矩阵及 partial/failed 输出状态。
- [ ] 迟到上游事件按 generation token 丢弃。

验证：各取消检查点、正常完成竞争、provider cancel 失败和重复 teardown。

## 阶段 G：本地与服务器验证

- [x] 扩展 fake model smoke 支持 text+audio 和 text+audio+action。
- [x] 运行真实 FastAPI + fake model + fake TTS 进程外测试。
- [x] 验证事件关联、seq、Base64 PCM 和完整终态，以及取消后无迟到成功终态。
- [ ] 在 Linux/服务器完整依赖环境运行相关 pytest、pre-commit 和类型检查。
- [ ] Docker 联调数字人后端，确认 outputs 移除 audio 即可回滚且不会重复 TTS。

Windows 本地手工验证使用三个终端：

```powershell
# 终端 1：启动 fake TTS provider
.venv\Scripts\python.exe scripts\realtime_fake_tts_provider.py --host 127.0.0.1 --port 8765

# 终端 2：启动 fake model Realtime 服务；TTS 仍只由 session.start.outputs 控制
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED='true'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT='这是本地 TTS smoke 固定回复'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS='20'
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server --host 127.0.0.1 --port 8000 --realtime-tts-url ws://127.0.0.1:8765 --realtime-tts-voice smoke

# 终端 3：分别验证普通音频回复和 action 融合音频回复
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py --mode text-audio --response-text '这是本地 TTS smoke 固定回复'
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py --mode fusion-audio --response-text '这是本地 TTS smoke 固定回复'
```

终止当前验证进程：在终端 1 和终端 2 分别按 `Ctrl+C`。这只停止当前进程；若下次启动不希望配置
provider，省略 `--realtime-tts-url` 和 `--realtime-tts-voice` 即可。是否请求音频始终由
`session.start.outputs` 决定，不存在 TTS enabled 开关。

## 阶段 H：文档与观测

- [x] 更新组长交流设计文档和任务复盘报告；正式客户端接入指南待服务器联调后补充。
- [x] 给出五种 outputs 示例、audio event schema、取消和错误处理。
- [ ] 增加延迟、buffer、失败、取消指标与脱敏日志。
- [x] 从本任务设计文档中删除旧接口、旧开关、旧事件名和 no-op cancel 描述。

## 最终验收

- [x] outputs 能力解析集中且便于扩展。
- [x] audio 未请求时无 TTS 连接、任务、事件和延迟回归。
- [x] audio 请求时文本/音频真正并发，完成顺序正确。
- [x] provisional 音频不提前播放且不泄漏。
- [x] turn.cancel 真正取消并幂等终止。
- [x] 错误不静默降级，资源无泄漏。
- [ ] fake、本地、Linux 完整测试与文档通过。

## 阶段 I：关键时间戳与非阻塞持久化（进行中）

- [x] 扩展结构化日志 schema，保留现有 `timestamp/timestamp_unix_ms`，新增
  `timestamp_unix_ns/monotonic_ns`，锁定跨版本兼容测试。
- [x] 将 writer 改为有界异步批量写：最多 100 条或 100ms；按目标分片合并 bytes；正常关闭
  drain，队列满时丢弃并统计，禁止阻塞事件循环。
- [x] 增加 writer batch、队列高水位、dropped、written、write error 的健康指标和单元测试。
- [ ] 在生产 launcher/Pipeline 边界记录进程启动、Pipeline begin/end、model pipeline ready、Client
  ready、app ready 和 server listening；fake 路径明确记录 `backend=fake`。除缺少可靠
  Uvicorn 监听成功回调的 `server_listening` 外均已完成。
- [x] 在 `/v1/session/realtime` 记录服务端 upgrade received/accepted、Session start/validation/resources
  ready/started、disconnect 和 cleanup；不修改客户端或数字人后端协议。
- [x] 记录 Turn start/ack、首输入、commit/ack、输入汇总、图像预处理和模型请求 build 边界。
- [x] 记录模型 submit、首 chunk、首正文、首正文外发、模型文本完成和 abort 边界；默认不逐 Delta。
- [x] 记录 TTS connect/ready/reuse、首文本 queue/append、commit、首 PCM、首 PCM 外发、250ms
  可播放、stream done、audio done、cancel/failure；默认不逐 PCM。
- [x] 记录 Action begin/category/child/ready，以及 response done、turn result、cancel 分段和失败 owner。
- [x] 扩展 debug 汇总器，生成 commit→首字、首字→TTS append、TTS 首包、commit→首音频、
  250ms 可播放、response/turn 总耗时和 cancel 总耗时。
- [ ] 使用 fake model/fake TTS 注入固定延迟，验证派生时间误差；覆盖时钟回拨、跨进程 monotonic
  禁止相减、队列溢出、滚动、正常 drain 和异常有界丢失。
- [ ] 对日志关闭、默认日志、诊断日志做对照压测，默认日志的目标 Realtime p99 增幅不超过 2%。
- [x] 更新配置测试手册和任务报告，明确日志字段、目录、批量/丢弃语义与已知限制。

验证门：相关 writer/Realtime 单测、Black/isort/compileall/diff-check；Linux 环境运行压测并保留
原始命令、样本量、p50/p95/p99、日志 writer 健康统计。数字人后端时间线不属于本阶段实现。
