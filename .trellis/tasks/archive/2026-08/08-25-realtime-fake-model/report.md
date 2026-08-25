# Session Realtime 本地 fake-model 实施复盘报告

## 1. 报告摘要

本任务为后续内嵌 TTS 迁移提供一个不依赖 GPU、模型权重和 Pipeline 子进程的本地
调试底座。当前已经可以在 Windows 本机启动真实 FastAPI/WebSocket 服务，通过唯一的
`/v1/session/realtime` protocol v1 接口验证 text、action、text+action、重复 Turn
和真正取消。

本任务不实现 TTS、回复音频、PCM 输出或 `response.audio.*`。当 `outputs` 包含
`audio` 时，当前服务明确返回 `unsupported_output`，等待后续 TTS 任务扩展。

截至 2026-08-25：

- 第一阶段迁移已经形成本地提交 `8dee03f feat: migrate fake model to Session Realtime`；
- Windows 本机可运行化、依赖隔离和 smoke 偏序修复尚未提交；
- 本机服务当前已在 `127.0.0.1:8000` 实际启动并验证通过；
- 未向远程仓库 push。

## 2. 目标与最终链路

```text
本地 smoke / 数字人后端测试客户端
  -> WS /v1/session/realtime
  -> session.start(protocol_version=1, outputs=[...])
  -> turn.start
  -> input.text.set / input.audio.append / input.image.append
  -> turn.commit
  -> MultimodalSession 正式协议与 Turn 状态机
  -> DevRealtimeModelClient 固定文本 / 固定 action 评分
  -> response.* / turn.action.ready / turn.result
```

fake-model 只替换生产 `Client/Pipeline` 所提供的模型生成和动作评分能力。外部协议解析、
输入校验、Session/Turn 状态机、provisional reply、取消、事件发送与终态仍复用正式实现。

开发 app 仅暴露：

- `GET /health`；
- `GET /v1/models`；
- `WS /v1/session/realtime`。

旧 `/v1/realtime`、speech WebSocket 和其他生产 HTTP API 不在开发 app 中注册。

## 3. 主要修改

### 3.1 全面迁移到 Session Realtime

- 放弃开发态旧 `/v1/realtime`、VAD 自动提交和 transcription pass；
- 使用显式 `session.start -> turn.start -> input.* -> turn.commit`；
- 支持 `outputs=['text']`、`['action']`、`['text','action']`；
- `audio` 在 TTS 实施前稳定返回 `unsupported_output`；
- 通过事件类型区分文本和动作，后续音频也将使用独立 `response.audio.*` 事件。

### 3.2 确定性模型替身

- `DevRealtimeModelClient` 按环境变量输出固定 Unicode 文本；
- 文本按字符数切分为流式 delta，可配置块大小和相邻块间隔；
- action 只从 Session 已验证白名单中确定性选择；
- text+action 复用正式 provisional promotion/discard 状态机；
- 窄 client 只实现正式 Session 实际需要的能力，不伪装完整生产 Client。

### 3.3 真正取消

`turn.cancel` 不再只是记账。服务会取消当前文本、动作、图片和推理任务，abort 活动模型
请求，释放 Turn 资源，并且只发送一次 `turn.cancelled`。取消后不得继续发送
`response.done`、`turn.action.ready` 或 `turn.result`。重复取消、迟到取消、断线和
`session.close` 均按幂等语义处理。

### 3.4 开发启动路径隔离

- fake launcher 分支发生在生产 action catalog、Pipeline、GPU、warmup、profiler、
  watcher 和 resource monitor 之前；
- 新增独立入口 `python -m sglang_omni.serve.realtime.dev_server`；
- 新增窄化 `create_dev_app()`，不再通过完整 `openai_api.create_app()` 导入 speech、
  vocoder、Pipeline 和模型运行时；
- `Client` 和 Realtime 包导出改为按需加载，仅用于类型标注的生产 Client 移入
  `TYPE_CHECKING`；
- `cache_key.py` 的 PyTorch 变为严格的可选导入：仅 torch 本体不存在时跳过 Tensor
  分支，torch 的传递依赖损坏仍会正常抛错。

因此 Windows 本机不需要安装 torch、CUDA、SGLang、模型权重或完整 Pipeline 依赖。

### 3.5 smoke 校验完善

smoke 支持 `text`、`action`、`fusion` 三种模式，并在每次正常 Turn 后创建第二个
Turn 验证取消。

融合模式的文本和 action 是并发分支，事件不能被错误地强制为一个全序列。当前脚本
按以下偏序校验：

```text
turn.committed < response.created < response.text.delta* < response.text.done
response.text.done < response.done < turn.result
turn.committed < turn.action.ready < turn.result
```

`turn.action.ready` 可以出现在任意两个文本 delta 之间；但任何晚于
`response.text.done` 的文本 delta 都会被拒绝。

## 4. 环境变量

| 变量 | 默认值/作用 |
|---|---|
| `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` | 默认 `false`；必须为 `true` 才能启动独立服务 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT` | 固定文本回复 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE` | 每个文本 delta 的最大 Unicode 字符数 |
| `SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS` | 相邻文本 delta 的模拟间隔，单位毫秒 |
| `SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID` | 可选 action 偏好，仍受 Session 白名单约束 |

非法布尔值、非正 chunk size、负间隔、空回复或非法 action ID 会在启动或协议校验阶段
明确失败，不会静默回退。

## 5. Windows 本机安装与启动

### 5.1 安装轻量依赖

不要执行 `uv sync`，它会解析完整项目的 Linux/CUDA 依赖。使用仓库 `.venv` 安装本地
开发服务所需的轻量依赖：

```powershell
uv venv --seed .venv
uv pip install --python .venv\Scripts\python.exe `
  "fastapi>=0.110" "uvicorn>=0.23" "websockets>=12" `
  "pydantic>=2" "numpy>=1.24" "pillow>=10" "xxhash>=3" `
  "msgspec" "pybase64>=1.4" "python-multipart" "httpx" "tzdata"
```

若需要运行本报告中的定向测试，再安装：

```powershell
uv pip install --python .venv\Scripts\python.exe pytest pytest-asyncio
```

### 5.2 启动服务

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "true"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT = "这是本机 fake model 的固定流式回复。"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE = "3"
$env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS = "80"
$env:SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID = "ADEV"

.venv\Scripts\python.exe -m sglang_omni.serve.realtime.dev_server `
  --host 127.0.0.1 --port 8000
```

启动成功后：

- 健康检查：`http://127.0.0.1:8000/health`；
- 模型列表：`http://127.0.0.1:8000/v1/models`；
- WebSocket：`ws://127.0.0.1:8000/v1/session/realtime`。

### 5.3 运行 smoke

另开 PowerShell：

```powershell
.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --mode text --response-text "这是本机 fake model 的固定流式回复。"

.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py --mode action

.venv\Scripts\python.exe scripts\realtime_fake_model_smoke.py `
  --mode fusion --response-text "这是本机 fake model 的固定流式回复。"
```

脚本成功时打印 `PASS`，超时、事件缺失、真实乱序、正文/action 不一致、服务 error 或
连接异常时打印 `FAIL` 并以非零状态退出。

## 6. 停止与禁用

### 6.1 停止当前服务进程

如果服务在当前 PowerShell 前台运行，按 `Ctrl+C`。

如果服务由其他终端或后台进程启动，先确认 8000 端口的监听进程，再停止精确 PID：

```powershell
$connection = Get-NetTCPConnection -LocalPort 8000 -State Listen
$connection | Select-Object LocalAddress, LocalPort, OwningProcess
Stop-Process -Id $connection.OwningProcess
```

不要按模糊进程名批量结束所有 Python 进程。

### 6.2 禁用后续 fake 启动

以下操作只影响当前 PowerShell 后续启动，不会停止已经运行的服务：

```powershell
$env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED = "false"
```

也可以删除当前终端中的全部 fake 环境变量：

```powershell
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT -ErrorAction SilentlyContinue
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS -ErrorAction SilentlyContinue
Remove-Item Env:SGLANG_OMNI_DEV_FAKE_ACTION_CANDIDATE_ID -ErrorAction SilentlyContinue
```

生产或服务器联调时必须关闭 fake 开关，以恢复真实 action catalog、Pipeline 和模型路径。

## 7. 验证结果

本机实际验证结果：

- `GET /health`：`status=healthy`；
- text smoke：通过，固定文本和取消均正确；
- action smoke：通过，固定 action 和取消均正确；
- fusion smoke：通过，文本/action 并发交错和取消均正确；
- smoke 偏序单测：`7 passed`；
- 本机可运行化相关非生产定向测试：`36 passed`；
- `compileall`：通过；
- `git diff --check`：通过。

生产 launcher 的两个 Windows 导入测试未作为本地验收门槛。原因是生产链依赖
Linux-only `fcntl`；Windows 本机验证使用独立 `dev_server`，Linux 服务器联调再覆盖
完整 launcher 路径。

## 8. 已知限制与后续工作

- 当前没有 TTS 和 `response.audio.*`；
- 当前 fake action 只验证协议与状态机，不模拟真实模型评分质量；
- Windows 不验证生产 launcher、Pipeline 或 GPU 生命周期；
- 后续 TTS 接入后，需要复用本报告的偏序校验原则：text、action、audio 各自分支内部
  有序，共同以 `turn.result` 为终态屏障；
- 上传 Linux 服务器后仍需补一次生产 launcher 开关分流和完整数字人后端联调。

## 9. 涉及的主要文件

- `sglang_omni/serve/realtime/dev_model.py`：固定文本与 action 模型替身；
- `sglang_omni/serve/realtime/dev_server.py`：独立窄化开发服务；
- `sglang_omni/serve/realtime/multimodal.py`：正式 Session/Turn、融合与取消；
- `sglang_omni/serve/launcher.py`：生产/fake 启动分流；
- `sglang_omni/client/__init__.py`：生产 Client 延迟导入；
- `sglang_omni/preprocessing/cache_key.py`：PyTorch 可选导入；
- `scripts/realtime_fake_model_smoke.py`：进程外 text/action/fusion/cancel 验证；
- `docs/developer_reference/realtime_fake_model_CN.md`：日常使用说明；
- `.trellis/spec/backend/realtime-development-model.md`：可执行代码规范；
- `tests/unit_test/serve/`、`tests/unit_test/preprocessing/test_cache_key.py`：回归测试。
