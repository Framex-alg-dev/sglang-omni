# Session Realtime、fake model 与 TTS 配置测试手册

## 1. 先区分四类控制入口

| 控制对象 | 入口 | 是否影响已运行进程 | 作用 |
|---|---|---|---|
| 使用真实模型还是 fake model | `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | 否，需重启 | 替换底层模型 Client |
| 使用哪个 TTS provider | 服务启动参数 `--realtime-tts-url/voice` | 否，需重启 | 配置 provider，但不主动启用 audio |
| 本 Session 是否请求 TTS | WebSocket 首条 `session.start.outputs` | 只影响该 Session | `outputs` 包含 `audio` 才进入 TTS |
| 客户端连接哪个本服务 | smoke 的 `--url` 或业务端 WebSocket 配置 | 新连接生效 | 修改 `/v1/session/realtime` 地址 |

fake TTS 不是本服务内的 enabled 开关。`scripts/realtime_fake_tts_provider.py` 是一个独立
WebSocket provider 进程；把本服务的 `--realtime-tts-url` 指向它，就能替代真实 TTS。

## 2. fake model

### 2.1 环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | `false` | `true` 时绕过生产 Pipeline/GPU |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 内置固定文本 | fake 回复正文，不能为空 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | `4` | 每个模型文本 Delta 的字符数，至少 1 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | `0` | Delta 间隔，可用于延迟/取消测试 |
| `SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID` | 空 | 可选固定 Action，必须属于 Session 白名单 |

PowerShell 示例：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是固定流式回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "2"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
$env:SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID = "ADEV"
```

启动独立开发服务：

```powershell
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 18080 --log-level debug
```

关闭当前进程：在该前台终端按 `Ctrl+C`。只修改环境变量不会改变已经运行的进程。

下次使用生产模型时：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
```

然后停止开发服务并通过生产 launcher 重新启动。不要把 `dev_server` 当作生产模型入口。

## 3. fake TTS

终端 1 启动独立 provider：

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_tts_provider.py `
  --host 127.0.0.1 --port 8765 --log-level info
```

终端 2 启动 fake model，并把 TTS URL 指向该 provider：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是固定流式回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "2"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
$env:SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID = "ADEV"

.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 18080 `
  --realtime-tts-url "ws://127.0.0.1:8765" `
  --realtime-tts-voice "smoke" `
  --log-level debug
```

两个 `--realtime-tts-*` 参数必须同时提供。配置它们只表示 provider 可用；只有客户端的
`session.start.outputs` 包含 `audio` 时，本 Session 才会连接 TTS。

停止时分别在两个前台终端按 `Ctrl+C`。

## 4. 真实 TTS 与 SSH 隧道

建立本地端口转发：

```powershell
ssh -N -L 51000:127.0.0.1:50001 -p 246 user@124.221.190.139
```

该命令需要持续运行。`Test-NetConnection 127.0.0.1 -Port 51000` 只证明本地 SSH listener
存在，不能证明远端 TTS 容器正在监听 `50001`。

启动本服务时配置完整 provider WebSocket 路径：

```powershell
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 18080 `
  --realtime-tts-url "ws://127.0.0.1:51000/api-ws/v1/realtime" `
  --realtime-tts-voice "benchmark_qwen_cherry_zh" `
  --log-level debug
```

若远端容器的 path、voice 或端口不同，以真实部署为准。URL 和 voice 改动后必须重启本服务。

## 5. TTS 参数

生产 CLI 当前支持：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--realtime-tts-url` | 未配置 | provider WebSocket 完整 URL |
| `--realtime-tts-voice` | 未配置 | voice；与 URL 成对提供 |
| `--realtime-tts-connect-timeout-seconds` | 10 | 建连超时 |
| `--realtime-tts-ready-timeout-seconds` | 10 | 等待 `session.created` |
| `--realtime-tts-send-timeout-seconds` | 10 | 单次发送超时 |
| `--realtime-tts-first-audio-timeout-seconds` | 10 | 首段文本成功发送后等待首音频 |
| `--realtime-tts-turn-timeout-seconds` | 30 | 单轮 TTS 总超时 |
| `--realtime-tts-text-queue-max-chunks` | 64 | 文本队列上限，满时自然背压 |
| `--realtime-tts-max-audio-chunk-bytes` | 1048576 | 单音频块上限 |
| `--realtime-tts-max-turn-audio-bytes` | 33554432 | 单轮累计 PCM 上限 |
| `--realtime-tts-provisional-audio-max-bytes` | 8388608 | provisional PCM 上限 |
| `--realtime-tts-provisional-audio-max-milliseconds` | 10000 | provisional 实际音频时长上限 |

不存在 `--realtime-tts-enabled`。如果 URL/voice 已配置但 `outputs=["text"]`，不会创建 provider
连接。若 outputs 请求 audio 但服务没有配置 provider，Session 会明确失败，不会静默降级。

## 6. 修改本服务 WebSocket 地址

业务客户端连接地址固定为：

```text
ws://<host>:<port>/v1/session/realtime
```

smoke 通过 `--url` 修改：

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode text
```

`outputs` 不在 URL 或请求头中，而在连接成功后的第一条 JSON：

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "session-001",
  "outputs": ["text", "audio"],
  "locale": "zh-CN",
  "reply": {"instructions": "请简短回复"}
}
```

当前允许：

```text
text
action
text + action
text + audio
text + audio + action
```

拒绝 `audio` 和 `audio + action`，因为 audio 必须由模型文本驱动。

## 7. smoke 测试矩阵

确保 `--response-text` 与 fake model 环境变量完全一致：

```powershell
$reply = "这是固定流式回复。"
```

### 7.1 纯文本

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode text --response-text $reply --timeout 30
```

预期：PASS；无任何 `response.audio.*`，也不连接 TTS。

### 7.2 Action

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode action --timeout 30
```

预期：PASS；收到 `turn.action.ready` 和 completed `turn.result`。

### 7.3 文本 + Action

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode fusion --response-text $reply --timeout 30
```

预期：PASS；provisional 被晋升，文本只形成一个正式回复。

### 7.4 文本 + 音频

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode text-audio --response-text $reply --timeout 30
```

预期：PASS；音频 seq 连续、Base64 PCM16LE 可解码，完成顺序正确。

### 7.5 文本 + 音频 + Action

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode fusion-audio --response-text $reply --timeout 30
```

预期：PASS；provisional PCM 在晋升前不外发，晋升后一次释放，取消后无迟到终态。

smoke 每种模式还会执行一次主动 `turn.cancel`，验证最终只出现 `turn.cancelled`。

## 8. 结构化日志开关与位置

当前结构化日志由以下环境变量控制：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_OMNI_REALTIME_LOG_DIR` | Linux: `/tmp/sglang-omni-realtime-logs` | JSONL 根目录 |
| `SGLANG_OMNI_REALTIME_LOG_TIMEZONE` | `Asia/Shanghai` | 日志时区 |
| `SGLANG_OMNI_REALTIME_LOG_QUEUE_SIZE` | `8192` | 异步日志队列容量 |
| `SGLANG_OMNI_REALTIME_LOG_MAX_FILE_MB` | `128` | 单个分片上限 |
| `SGLANG_OMNI_SERVICE_INSTANCE_ID` | hostname-PID | 服务实例关联 ID |
| `SGLANG_OMNI_REALTIME_LOG_FULL_INSTRUCTIONS` | 关闭 | 高风险诊断项；可能记录完整指令，生产不建议开启 |

Windows 本地建议显式设置目录：

```powershell
$env:SGLANG_OMNI_REALTIME_LOG_DIR = "E:\Company_data\sglang-omni\.local\realtime-logs"
$env:SGLANG_OMNI_REALTIME_LOG_TIMEZONE = "Asia/Shanghai"
```

日志按以下目录组织：

```text
<root>/<YYYY-MM-DD>/<HH>/<log_type>_<component>_<pid>_<segment>.jsonl
```

修改环境变量只影响随后启动的新进程。

## 9. 启停与端口检查

前台进程统一用 `Ctrl+C` 停止。不要通过修改 enabled 环境变量来“停止”已经运行的服务。

检查端口：

```powershell
Get-NetTCPConnection -LocalPort 18080,8765,51000 -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,State,OwningProcess
```

若必须停止后台进程，先核对精确 PID，再执行：

```powershell
Stop-Process -Id <verified-pid>
```

正式上传服务器前至少确认：

- fake model 已关闭；
- 生产 launcher 而非 `dev_server` 启动；
- TTS URL/voice 指向真实 provider；
- 数字人后端连接 `/v1/session/realtime`；
- 不需要 audio 的 Session 从 outputs 移除 audio；
- 数字人后端不会再次调用独立 TTS，避免重复合成。

