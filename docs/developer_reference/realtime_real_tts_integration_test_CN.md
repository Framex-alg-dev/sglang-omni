# Session Realtime 真实 TTS 联调测试手册

## 1. 测试目标

本手册用于在 Windows 本机上联调以下链路：

```text
smoke 客户端
  -> ws://127.0.0.1:18080/v1/session/realtime
  -> 本地 fake model 流式生成文本
  -> 本服务的 Embedded TTS adapter
  -> ws://127.0.0.1:51000/api-ws/v1/realtime
  -> SSH 隧道
  -> 服务器 127.0.0.1:40001
  -> 真实 TTS provider
  -> PCM 音频块按原 Session WebSocket 返回 smoke 客户端
```

本轮不加载 GPU 多模态模型，使用 fake model 产生可重复的固定文本，仅验证模型之后的
Realtime、TTS、音频转发和取消链路。

## 2. 已确认配置

| 用途 | 地址/参数 |
|---|---|
| 真实 TTS 公网地址 | `ws://124.221.190.139:40001/api-ws/v1/realtime` |
| SSH 登录地址 | `user@124.221.190.139` |
| SSH 端口 | `246` |
| SSH 隧道本地端口 | `51000` |
| SSH 隧道远程目标 | `127.0.0.1:40001` |
| 本地 Session Realtime 服务 | `ws://127.0.0.1:18080/v1/session/realtime` |
| TTS WebSocket path | `/api-ws/v1/realtime` |
| voice | `benchmark_qwen_cherry_zh` |

> 注意：`50001` 不是本次真实 TTS WebSocket 的目标端口。SSH 隧道必须转发到远程
> `40001`。

## 3. 测试前清理

关闭之前启动的本服务和旧 SSH 隧道，确保旧进程没有继续占用端口。

在 PowerShell 中检查：

```powershell
Get-NetTCPConnection -LocalPort 18080,51000 -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,State,OwningProcess
```

如果有旧进程，先查看进程名称：

```powershell
Get-Process -Id <PID>
```

确认就是本次测试的旧 `ssh.exe` 或 Python 服务后，优先回到它的前台窗口按 `Ctrl+C`
停止。

## 4. 终端一：建立 SSH 隧道

打开第一个 PowerShell，执行：

```powershell
ssh -N -L 51000:127.0.0.1:40001 -p 246 user@124.221.190.139
```

输入密码后，命令正常情况下不会输出内容，也不会退出。该窗口必须在整个联调期间保持运行。

另开一个 PowerShell 检查本地 listener：

```powershell
Test-NetConnection 127.0.0.1 -Port 51000
```

预期：

```text
TcpTestSucceeded : True
```

`True` 只证明 SSH listener 存在，下一步还要验证远程 WebSocket 路由。

## 5. WebSocket 握手预检

在普通 PowerShell 窗口执行：

```powershell
curl.exe --http1.1 -i --max-time 5 `
  -H "Connection: Upgrade" `
  -H "Upgrade: websocket" `
  -H "Sec-WebSocket-Version: 13" `
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" `
  "http://127.0.0.1:51000/api-ws/v1/realtime?voice=benchmark_qwen_cherry_zh&session_id=manual-probe"
```

预期首行：

```text
HTTP/1.1 101 Switching Protocols
```

`curl` 可能在 5 秒后提示超时，这是因为 WebSocket 已经建立而 `curl` 没有继续按业务协议交互。
只要首行是 `101 Switching Protocols`，握手预检就算通过。

如果是 `403 Forbidden`：

- 确认隧道目标是 `40001`，不是 `50001`；
- 停止旧 SSH 进程后重建隧道；
- 在服务器本机用相同 URL 直接探测 `127.0.0.1:40001`。

本步不通过时不要启动后续 smoke，否则只会再次得到 `InvalidStatus` / `embedded TTS
turn failed`。

## 6. 终端二：启动本地 Session Realtime 服务

打开第二个 PowerShell，进入项目：

```powershell
cd E:\Company_data\sglang-omni
```

配置 fake model：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是连接真实TTS的固定流式回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "2"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "30"
```

配置结构化时间线日志：

```powershell
$env:SGLANG_OMNI_REALTIME_LOG_DIR = "E:\Company_data\sglang-omni\.local\realtime-logs"
$env:SGLANG_OMNI_REALTIME_LOG_TIMEZONE = "Asia/Shanghai"
```

启动服务：

```powershell
.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 `
  --port 18080 `
  --realtime-tts-url "ws://127.0.0.1:51000/api-ws/v1/realtime" `
  --realtime-tts-voice "benchmark_qwen_cherry_zh" `
  --log-level debug
```

注意：

- TTS URL 使用本地隧道地址 `127.0.0.1:51000`，不直接填公网地址；
- 代码会自动在 URL 上追加 `voice` 和当前外部 `session_id`；
- 启动参数只配置 provider，是否合成音频仍由 `session.start.outputs` 决定；
- 本轮 smoke 使用 `outputs=["text","audio"]`，因此会进入真实 TTS。

在另一窗口检查本服务 HTTP health：

```powershell
curl.exe http://127.0.0.1:18080/health
```

预期返回 fake realtime model 正在运行的 JSON。

## 7. 终端三：执行 text + audio 主测试

打开第三个 PowerShell，进入项目：

```powershell
cd E:\Company_data\sglang-omni
```

执行：

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --url "ws://127.0.0.1:18080/v1/session/realtime" `
  --mode text-audio `
  --response-text "这是连接真实TTS的固定流式回复。" `
  --text "请回复这段固定文本。" `
  --timeout 30
```

成功预期：

```text
PASS: Session Realtime text-audio and cancel (... events).
```

smoke 会验证：

- Session 和 Turn 正常建立；
- 收到完整的固定文本；
- 收到至少一个 `response.audio.delta`；
- Base64 PCM 可解码且非空；
- audio `seq` 连续；
- `response.text.done` / `response.audio.done` / `response.done` / `turn.result` 顺序合法；
- 紧接着的取消用例最终只收敛到 `turn.cancelled`。

文本 Delta 和音频 Delta 可以交错到达，这是正常流式行为。首个音频块可以早于
`response.text.done`。

## 8. 时间线日志验证

列出最新日志：

```powershell
Get-ChildItem .local\realtime-logs -Recurse -Filter *.jsonl |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 20 FullName,Length,LastWriteTime
```

搜索本轮 Session：

```powershell
Get-ChildItem .local\realtime-logs -Recurse -Filter *.jsonl |
  Select-String 'dev-smoke-text-audio'
```

成功链路应能看到以下关键点（中间允许有其他事件）：

```text
tts_turn_started
tts_connect_begin
tts_connect_completed
tts_ready_received
tts_first_text_appended
tts_commit_sent
tts_first_pcm_received
response_first_audio_sent
tts_stream_completed
response_audio_done_sent
response_done_sent
turn_result_sent
```

如果同一 Session 的后续 audio Turn 复用连接，应看到 `tts_connection_reused`，不应再次出现该
Session 的 `tts_connect_begin`。

## 9. 失败定位表

| 现象 | 优先判断 | 处理 |
|---|---|---|
| `bind ... 51000: Permission denied` | 本地 51000 已被占用或 Windows 保留 | 查看 `Get-NetTCPConnection -LocalPort 51000` 及 PID |
| `Test-NetConnection` 为 `False` | SSH 隧道未建立 | 检查 SSH 窗口、密码和进程 |
| WebSocket 预检返回 `403` | 多数是隧道仍指向旧 `50001` 或远程路由拒绝 | 重建 `51000 -> 40001` 隧道 |
| `ConnectionRefusedError` | 隧道或远程 40001 未监听 | 先在服务器检查 `40001` |
| `InvalidStatus` | WebSocket 握手得到非 101 | 重跑第 5 节并查看远程日志 |
| `ready timed out` | 连接建立但没有 `session.created` | 查看 TTS provider 协议和模型初始化 |
| `first_audio timed out` | 文本已发送但无 PCM | 查看 voice、TTS 模型与 provider error 事件 |
| `protocol error` | provider 事件类型、Base64 或 done 顺序不符合预期 | 保留服务端日志并对照协议 |
| 收到文本但最终 Turn 失败 | TTS 分支失败，首版策略是 `fail_turn` | 查看 `tts_turn_failed.phase` |

排查失败时，优先保留：

1. 本地 Session Realtime 服务窗口的完整 traceback；
2. `.local/realtime-logs` 中本轮 Session 的 `error` 和 `performance` JSONL；
3. TTS 容器同一时间段的日志。

## 10. 停止与恢复

按以下顺序停止：

1. 终端三 smoke 测试自动退出；
2. 在终端二按 `Ctrl+C` 停止本地 Session Realtime 服务；
3. 在终端一按 `Ctrl+C` 停止 SSH 隧道。

若后续要恢复真实多模态模型，在新 PowerShell 中设置：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
```

然后使用生产 launcher 重新启动。`dev_server` 仅用于本地无 GPU 联调，不代表真实模型部署。

## 11. 本轮通过标准

同时满足以下条件，才可认为真实 TTS 主链路联调通过：

- [ ] SSH 隧道明确是 `51000 -> 127.0.0.1:40001`；
- [ ] WebSocket 预检返回 `101 Switching Protocols`；
- [ ] 本地 `/health` 正常；
- [ ] `text-audio` smoke 输出 `PASS`；
- [ ] 文本内容完整，音频 PCM 非空且 seq 连续；
- [ ] `response.audio.done` 在最后一个音频块之后；
- [ ] `response.done` 在文本和音频均完成后；
- [ ] 取消用例收敛到唯一 `turn.cancelled`；
- [ ] 时间线包含建连、ready、首文本、首 PCM、audio done 和 response done；
- [ ] 服务端无孤儿 task、未捕获异常或连接泄漏告警。
