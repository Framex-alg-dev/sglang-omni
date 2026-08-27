# `/v1/session/realtime` 内嵌流式 TTS 实现说明与后续计划

## 1. 背景、目标与边界

SGLang-Omni 是数字人后端依赖的多模态推理服务。数字人后端向本服务提交用户文本、音频和图片，
本服务调用多模态模型生成回复文本并按需判断动作。本次改动在模型仍生成文本的前提下，把正文实时
送入远程 TTS，再将 PCM 音频通过同一条业务 WebSocket 返回。

当前只支持协议版本 1 的接口：

```text
WS /v1/session/realtime
```

本文后续提到的所有业务事件均属于该接口。业务端取消使用 `turn.cancel`；本服务向 TTS provider
取消使用内部 `response.cancel`，两者属于不同连接和协议层。

本次不包含音频转码、重采样、WAV 封装、播放器逻辑，也不修改数字人后端动作执行和预录音频管理。

## 2. 端到端链路

### 2.1 请求从谁发给谁

1. 数字人后端连接 `WS /v1/session/realtime`。
2. 数字人后端发送 `session.start`，通过 `outputs` 声明本 Session 的输出能力。
3. 本服务校验协议、Session ID、outputs、回复配置和动作白名单，返回 `session.started`。
4. 数字人后端发送 `turn.start`，再按需发送 `input.text.set`、`input.audio.append`、
   `input.image.append`。
5. 数字人后端发送 `turn.commit`；本服务冻结 Turn 输入并返回 `turn.committed`。

### 2.2 回复从谁返回给谁

1. 本服务把冻结输入交给多模态模型，模型流式返回正文 Delta。
2. 本服务通过 `response.text.delta` 把正文返回数字人后端。
3. 若 outputs 包含 audio，本服务同时把正文 Delta 按序放入有界 TTS 输入队列。
4. 远程 TTS 返回 PCM；本服务校验 JSON、事件顺序和 Base64，再通过
   `response.audio.delta` 返回数字人后端。
5. 模型 EOF 后，本服务发送 `response.text.done` 并向 TTS 恰好 commit 一次。
6. TTS 尾音频结束后，本服务发送 `response.audio.done`。
7. text/audio 都完成后发送 `response.done`；action 分支也收敛后发送唯一 Turn 终态
   `turn.result`。

```text
数字人后端       SGLang-Omni          多模态模型          远程 TTS
    | session.start   |                    |                 |
    |---------------->|                    |                 |
    | session.started |                    |                 |
    |<----------------|                    |                 |
    | turn/input/commit                    |                 |
    |---------------->|---- Generate ----->|                 |
    | turn.committed  |                    |                 |
    |<----------------|                    |                 |
    |                 |<--- text delta ----|                 |
    | text.delta      |---- append ------------------------->|
    |<----------------|                    |                 |
    |                 |<------------------------- PCM delta -|
    | audio.delta     |                    |                 |
    |<----------------|                    |                 |
    |                 |<--- text EOF ------|                 |
    | text.done       |---- commit ------------------------->|
    |<----------------|                    |                 |
    |                 |<-------------------------- tail PCM -|
    | audio.done      |                    |                 |
    | response.done   |                    |                 |
    | turn.result     |                    |                 |
    |<----------------|                    |                 |
```

首个 audio delta 可以早于 text done，二者可以交错；成功终态满足
`response.text.done < response.audio.done < response.done < turn.result`。

## 3. outputs 是唯一启用源

TTS 是否参与某个 Session，只由第一条 `session.start.outputs` 决定。provider URL、voice 和超时
配置不能隐式启用 TTS，也没有 `--realtime-tts-enabled` 或 TTS enabled 环境变量。

| `session.start.outputs` | 文本 | 在线 TTS | 动作 |
|---|---:|---:|---:|
| `["text"]` | 是 | 否 | 否 |
| `["text","audio"]` | 是 | 是 | 否 |
| `["action"]` | 否 | 否 | 是 |
| `["text","action"]` | 是 | 否 | 是 |
| `["text","audio","action"]` | 是 | 是 | 是 |

拒绝 `["audio"]` 和 `["audio","action"]`，因为在线音频来源是模型正文；空数组、重复项、未知值
和空白值也会被拒绝。集中 `SessionOutputCapabilities` 推导 text/audio/action 能力，
`session.started.outputs` 回显规范化顺序。

不包含 audio 的 Session 不创建 TTS manager/Turn、不连接 provider、不产生音频事件。服务启动时
允许缺少 TTS 配置，只有包含 audio 的 `session.start` 才会稳定失败。

## 4. 外部音频事件合同

```json
{
  "type": "response.audio.delta",
  "session_id": "session-001",
  "turn_id": "turn-001",
  "response_id": "resp-001",
  "seq": 1,
  "delta": "<base64 pcm16le>",
  "audio": {
    "format": "pcm16le",
    "sample_rate_hz": 24000,
    "channels": 1
  }
}
```

数字人后端通过事件 `type` 区分文字和音频，用 `session_id + turn_id + response_id` 关联同一回复；
seq 在一个 response 内从 1 单调递增。PCM 只校验和转发，不转码、不重采样、不添加 WAV header。

`response.audio.done` 携带相同关联 ID 和最终 seq，不重复音频。成功 Turn 的
`turn.result.outputs` 分别报告请求分支的状态；未真实完成 audio 时不能标记为 completed。

## 5. 普通 text+audio

1. 正式 `response.created` 后启动 TTS Turn，provider WebSocket 仍按需延迟建立。
2. 模型 Delta 先成为正式 text delta，再进入有界 TTS 文本队列，队列满时自然背压。
3. TTS sender/reader 并发，PCM 到达后立即按 seq 外发。
4. 模型 EOF 产生 text done，迭代器结束驱动唯一一次 provider commit。
5. provider 完成 audio done 和 response done 后，本服务依次发送外部 audio done、response done、
   turn result。
6. 模型或 TTS 任一侧失败会立即取消另一侧；即使模型永久暂停，TTS 失败也会立即 abort 模型，
   不等待下一个模型 Delta。

## 6. text+audio+action provisional

1. provisional 文本实时送 TTS，每段只发送一次。
2. promotion 判定前，PCM 只进入按当前 Turn/response 隔离的有界缓冲，绝不外发。
3. promoted 后先完成 provisional resolved 和正式 response 边界，再按 seq 释放缓冲；后续 PCM 实时
   外发。
4. `replayed_from_provisional=true` 的正式文本只用于重放和归档，不再次送 TTS。
5. discarded 时立即取消 TTS、清空缓冲、关闭当前 provider 连接，不泄漏在线音频。
6. unsupported action 走 `client_prerecorded_audio` 时也不外发在线 TTS 音频。
7. provisional PCM 受最大字节和 PCM 时长双重限制，溢出会使 Turn 失败并关闭连接。

## 7. TTS WebSocket 生命周期

每个外部 `MultimodalSession` 独占一个 `EmbeddedTTSConnection`，不同 Session 不共享连接。

```text
Session 建立
  -> outputs 含 audio：校验配置并创建 manager（尚未联网）
  -> 首个 audio Turn：lazy connect，等待 provider session.created
  -> 正常 Turn 完成：连接回到 Ready，供同 Session/同 voice 下一 Turn 复用
  -> voice 改变：关闭旧连接，再建立新连接
  -> cancel/timeout/protocol error：尽力发 response.cancel，关闭不可信连接
  -> 下一 audio Turn：重新 lazy connect
  -> session.close/业务 WebSocket 断开：幂等释放 manager 和连接
```

同一 provider 连接只允许一个 reader 和一个活动 TTS Turn。正常完成保留健康连接；取消或异常后
保守关闭，以免旧 Turn 的迟到音频污染下一 Turn。

provider 协议：建连等待 `session.created`；文本使用 `input_text_buffer.append`；正文结束使用
`input_text_buffer.commit`；取消使用 `response.cancel`；完整成功必须收到 audio done 后的
response done。

## 8. 真正取消与错误策略

业务端发送：

```json
{"type":"turn.cancel","turn_id":"turn-001"}
```

本服务立即使 Turn token 失效，统一取消模型、action、TTS producer/sender/reader、队列等待者和
provisional PCM 缓冲；向 provider 尽力发送内部 `response.cancel` 后关闭连接。收敛后只发送一次
`turn.cancelled`。取消后不再发送该 Turn 的 text/audio done、response done 或 turn result，迟到
事件被丢弃。

首版采用 `fail_turn`，不会把明确请求 audio 的 Session 静默降级为 text-only：

| 故障 | 当前行为 |
|---|---|
| outputs 非法 | Session 协议错误 |
| audio Session 缺少 provider 配置 | Session 建立失败 |
| connect/ready/send/首音频/总超时 | Turn 失败并关闭连接 |
| 非法 JSON/Base64、乱序、重复 done、提前关闭 | protocol error，Turn 失败并关闭连接 |
| 模型失败 | 取消 TTS，Turn 失败 |
| TTS 失败且模型暂停 | 立即 abort 模型，Turn 失败 |
| 外部断线或 `session.close` | 不再发事件，幂等释放 Session 资源 |

## 9. 部署配置

生产装配支持 provider URL、voice、connect/ready/send/first-audio/turn timeout、文本队列最大 chunk、
单音频块和单 Turn 最大字节、provisional 最大字节和最大 PCM 时长。`websockets` 最低版本为 13。

这些参数描述如何连接 provider，不是 enabled 开关。日志不得包含 provider URL 凭证、完整正文、
Base64 或 PCM。当前已有稳定错误和关联 ID，完整延迟、buffer、失败、取消 metrics 尚未实现。

## 10. Windows 本地进程外验证

无需 GPU、真实模型、外网或 Docker，使用三个 PowerShell 终端：

```powershell
# 终端 1：fake TTS provider
.venv\Scripts\python.exe scripts\realtime_fake_tts_provider.py --host 127.0.0.1 --port 8765

# 终端 2：fake model Session Realtime 服务
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED='true'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT='这是本地 TTS smoke 固定回复'
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS='20'
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 8000 `
  --realtime-tts-url ws://127.0.0.1:8765 `
  --realtime-tts-voice smoke

# 终端 3：普通与 provisional
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --mode text-audio --response-text '这是本地 TTS smoke 固定回复'
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --mode fusion-audio --response-text '这是本地 TTS smoke 固定回复'
```

停止当前验证：在终端 1、终端 2 分别按 `Ctrl+C`。省略两个 `--realtime-tts-*` 参数表示下次启动
不配置 provider；某个 Session 不需要音频时，从其 outputs 移除 audio。

本地已实际通过两种进程外 smoke。Linux 完整依赖、真实 provider、Docker 和数字人后端仍待验证。

## 11. 完成情况与后续计划

已完成 outputs 能力解析、Session-owned adapter、连接复用、普通 text+audio、triple provisional、
真正取消、错误收敛、fake provider、自动化测试和 Windows 进程外 smoke。

待完成：

- Linux/服务器完整 pytest、pre-commit 和类型检查；
- 真实 provider 的网络、首包、尾包和超时验证；
- Docker 与数字人后端联调五种 outputs、取消、断线和回滚；
- 正式延迟、buffer、失败和取消指标及脱敏日志；
- 依据真实联调结果补充正式客户端接入指南。
