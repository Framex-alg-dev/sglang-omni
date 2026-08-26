# Session Realtime 内嵌流式 TTS 改动报告

## 1. 背景与结果

数字人后端原先只能从 SGLang-Omni 接收流式回复文字，再由业务侧合成语音。本次改动把远程 TTS
编排内嵌到唯一的 `WS /v1/session/realtime` Session/Turn 状态机中：模型继续生成正文，本服务
同时返回文字、向 TTS 发送正文，并把 PCM 音频通过同一条业务 WebSocket 返回。

普通 text+audio、text+audio+action provisional、真正取消、连接复用和本地进程外验证均已实现。
Linux、真实 provider、Docker 和数字人后端联调尚未完成，不能视为已上线验收。

## 2. 范围

本次包含：

- `session.start.outputs` 扩展 audio，并成为唯一启用源；
- Session-owned 远程 TTS WebSocket adapter；
- 流式正文、PCM、action provisional 和 Turn 终态编排；
- 真正取消、失败收敛、有界队列/缓冲；
- 无 GPU fake model、fake TTS provider、自动化和进程外 smoke。

不包含 PCM 转码/重采样/WAV 封装、本地 TTS 模型、播放器，以及数字人后端动作执行和预录音频
资产修改。

## 3. 主要模块与文件

| 文件 | 职责 |
|---|---|
| `sglang_omni/serve/realtime/output_capabilities.py` | 集中校验和规范化 outputs |
| `sglang_omni/serve/realtime/embedded_tts.py` | Session-owned 连接、Turn、队列、超时、解码和关闭 |
| `sglang_omni/serve/realtime/multimodal.py` | text/audio/action、provisional、终态、取消和失败 owner |
| `sglang_omni/serve/realtime/dev_server.py` | 无 GPU fake model 及本地 TTS 配置装配 |
| `sglang_omni/serve/launcher.py`、`sglang_omni/cli/serve.py` | 生产 TTS 配置传递 |
| `scripts/realtime_fake_tts_provider.py` | 可独立启停、支持复用和取消的 fake provider |
| `scripts/realtime_fake_model_smoke.py` | 进程外 Session Realtime 协议验证 |
| `tests/unit_test/serve/test_embedded_tts.py` | adapter 协议、复用、超时、取消和错误测试 |
| `tests/unit_test/serve/test_realtime_embedded_tts.py` | 普通及 provisional 跨层集成测试 |

## 4. 协议合同

TTS 是否参与只由 `session.start.outputs` 决定。允许五种组合：

```json
["text"]
["text", "audio"]
["action"]
["text", "action"]
["text", "audio", "action"]
```

拒绝 `["audio"]`、`["audio","action"]`，也拒绝空、重复、未知或空白项。不存在 TTS enabled 环境
变量或 CLI 开关；URL 和 voice 只提供连接信息，不能主动启用 Session 音频。

音频事件示例：

```json
{
  "type": "response.audio.delta",
  "session_id": "session-001",
  "turn_id": "turn-001",
  "response_id": "resp-001",
  "seq": 1,
  "delta": "<base64 pcm16le>",
  "audio": {"format": "pcm16le", "sample_rate_hz": 24000, "channels": 1}
}
```

业务端用 `type` 区分文字和音频，以三个 ID 关联回复，以 seq 排序 PCM。成功顺序为：

```text
response.created
response.text.delta / response.audio.delta（可交错）
response.text.done
response.audio.done
response.done
turn.result
```

三输出中 provisional 文本提前合成，但 PCM 在 promotion 前只缓存在服务内；promoted 后释放一次，
replay 文本不重复合成；discarded、unsupported 或 cancelled 均不泄漏在线 PCM。

## 5. 生命周期与取消

每个外部 Session 独占一个 TTS manager。包含 audio 的 Session 建立时只校验配置，第一个 audio
Turn 才连接并等待 `session.created`。同 Session、同 voice、正常完成的连接跨 Turn 复用；不同
Session 隔离，voice 改变重建。正常完成保留连接；取消、超时、协议错误和 Session close 幂等关闭。

业务端取消：

```json
{"type":"turn.cancel","turn_id":"turn-001"}
```

本服务立即取消模型、action、TTS producer/sender/reader、队列等待和 provisional 缓冲，尽力向
provider 发送内部 `response.cancel`，随后关闭连接。外部只收到一次 `turn.cancelled`，不会再收到
该 Turn 的 text/audio done、response done 或 turn result。

## 6. Windows 本地启停与验证

打开三个 PowerShell 终端：

```powershell
# 终端 1：启动 fake TTS
.venv\Scripts\python.exe scripts\realtime_fake_tts_provider.py --host 127.0.0.1 --port 8765

# 终端 2：启动 fake model Realtime 服务
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED='true'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT='这是本地 TTS smoke 固定回复'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS='20'
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 8000 `
  --realtime-tts-url ws://127.0.0.1:8765 `
  --realtime-tts-voice smoke

# 终端 3：验证普通和三输出
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py --mode text-audio `
  --response-text '这是本地 TTS smoke 固定回复'
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py --mode fusion-audio `
  --response-text '这是本地 TTS smoke 固定回复'
```

在终端 1、终端 2 分别按 `Ctrl+C` 停止当前两个进程。修改环境变量只影响下次启动，不能替代停止
当前进程。

## 7. 已通过验证

- outputs 表驱动及无 audio 零副作用；
- handshake、append/commit、Base64/JSON/事件顺序错误；
- 同 Session 两 Turn 复用、跨 Session 隔离、voice 改变重连；
- cancel/timeout/protocol error 关闭及 Session close 幂等；
- 普通 text+audio 的正文、PCM、seq、关联字段和终态顺序；
- TTS 在模型暂停时失败会立即 abort 模型；
- provisional promotion 不提前泄漏、replay 不重复合成；
- discard、unsupported、cancel 和 buffer overflow 清理；
- TTS 核心定向测试合跑 `58 passed`；阶段 G smoke/provider 定向测试为 `12 passed`；
- Windows 独立进程 smoke：text+audio 22 个事件、triple 28 个事件均通过；
- Black、isort、compileall 和 `git diff --check` 通过。

## 8. 环境限制与服务器联调清单

Windows 完整 launcher 测试被项目既有 Unix `fcntl` 依赖阻塞，不影响独立 dev server 和 smoke。
仍需在 Linux/服务器完成：

- [ ] 完整依赖环境运行相关及邻近 pytest；
- [ ] 运行 pre-commit 和类型检查；
- [ ] 连接真实 TTS，验证凭证、握手、首包、尾包、超时和网络断开；
- [ ] 构建 Docker 并检查配置传递；
- [ ] 与数字人后端验证五种 outputs 和两个拒绝组合；
- [ ] 验证业务端按 type/三个 ID/seq 分流文字和 PCM；
- [ ] 验证 triple promotion 不提前播放音频；
- [ ] 验证 `turn.cancel` 后只出现 `turn.cancelled`；
- [ ] 验证 outputs 移除 audio 后无 provider 连接和音频事件。

## 9. 回滚说明

- 协议级停用：数字人后端从新 Session 的 outputs 移除 audio。本服务代码仍在，但该 Session 不创建
  TTS Turn、不连接 provider、不产生音频事件。这是首选的低风险停用方式。
- 停止当前服务：正常终止进程，再按目标配置重启。修改环境变量本身不会停止已运行进程。
- 代码回滚：回退本次代码提交并重新构建、部署镜像，适用于实现必须撤回的情况；它与 outputs 移除
  audio 不是同一操作。

## 10. 风险

- 真实 provider 事件细节、代理设备和网络抖动仍需服务器验证；
- provisional 缓冲有界，但上限需要结合真实时延和并发量调优；
- TTS 故障会使当前 Turn 失败，不静默降级，业务端需处理明确错误；
- 取消或异常后保守断连重建，会增加下一 audio Turn 的建连延迟；
- 正式延迟、buffer、失败、取消 metrics 尚未实现，当前不能声称已有完整线上可观测性。

## 11. 阶段 I：服务端关键时间线

结构化 JSONL schema 保留 `timestamp`、`timestamp_unix_ms`，新增：

- `timestamp_unix_ns`：跨进程或与外部系统对齐的绝对时间；
- `monotonic_ns`：同一进程内计算阶段耗时，禁止跨 PID 相减。

writer 调用线程只捕获时间并执行有界 `put_nowait()`，不等待文件系统。后台线程最多聚合 100 条或
等待 100ms，按目标小时、类型、组件、PID、分片合并写入。正常 `close()` 会 drain 已接收记录；
队列满时丢弃日志而不反压 Realtime。健康信息包含 queue size/capacity/high-watermark、written、
dropped、write errors、batch count、最大 batch records/bytes。

可配置：

| 环境变量 | 默认值 |
|---|---:|
| `SGLANG_OMNI_REALTIME_LOG_QUEUE_SIZE` | 8192 |
| `SGLANG_OMNI_REALTIME_LOG_MAX_FILE_MB` | 128 |
| `SGLANG_OMNI_REALTIME_LOG_BATCH_MAX_RECORDS` | 100 |
| `SGLANG_OMNI_REALTIME_LOG_BATCH_MAX_DELAY_MS` | 100 |

当前已覆盖服务端 WebSocket upgrade/accept（明确 `backend=production/fake`）、TTS Turn、connect、
reuse、ready、首 append、commit、provider response、首 PCM、累计 250ms、外部首 PCM、audio done、
response done、cancel 和 fail。默认不逐正文 Delta、逐 PCM 落盘，新增记录只包含 ID、计数、字节和
阶段，不包含正文、媒体、URL/query、voice、headers/cookies/token。

debug 汇总器使用同 PID monotonic 点派生 commit→首字、首字→TTS append、TTS 首包、
commit→外部首音频、commit→250ms、response/turn 总耗时和 cancel 总耗时；缺点或跨 PID 时返回
null，不使用绝对时间假装单调耗时。

生产 `server_listening` 尚未记录：当前 Uvicorn 装配没有已确认且同时适用于 production/fake 的
监听成功回调。本阶段不使用调用 `serve()` 前的时间冒充端口已监听。该点需在 Linux 集成阶段通过
可靠 server startup hook 补齐。Windows 未执行 p99 对照压测；“默认日志 p99 增幅不超过 2%”仍是
Linux 验证门，不是当前已通过结论。

质量审查额外修正了两个时序风险：日志队列已满时，`close()` 改用独立关闭事件通知
writer，避免为写入停止标记而无限阻塞；所有 `*_sent` 埋点均以 WebSocket 实际发送成功为
条件，避免断线后仍记录伪成功时间。首文本外发埋点也位于 TTS 队列入队之前，不将
TTS 背压等待混入模型到文本外发的耗时。

本阶段 Windows 回归结果：相关 59 项单元测试全部通过，Black、isort、`compileall` 和
`git diff --check` 通过；仅有项目现存的 Starlette/httpx 弃用警告。
