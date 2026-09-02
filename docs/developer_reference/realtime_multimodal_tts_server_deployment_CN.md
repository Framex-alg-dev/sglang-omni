# Session Realtime 多模态模型 + 真实 TTS 服务器部署手册

## 1. 目标和阶段

本手册将部署分成两个独立验证阶段：

1. **服务器源码运行**：只迁移已提交代码，在服务器上启动真实多模态模型和外部流式
   TTS；本机通过 SSH 本地端口转发访问服务器，用已完成的静态网页验证文本和完整音频。
2. **Docker 部署**：只在第一阶段通过后构建镜像，使宿主机业务后端通过
   `ws://127.0.0.1:18001/v1/session/realtime` 连接容器中的服务。

两个阶段的核心链路相同：

```text
业务客户端 / 手工测试网页
  -> /v1/session/realtime
  -> 真实多模态模型（音频/图像/文本 -> 文本 Delta）
  -> Embedded TTS adapter
  -> ws://127.0.0.1:40001/api-ws/v1/realtime
  -> 真实 TTS（文本 Delta -> PCM16LE 24kHz 单声道）
  -> response.text.* + response.audio.*
```

> 这里的模型使用 `--text-only` Pipeline，因为模型只负责生成文本，音频回复由已集成的
> 外部 TTS 产生。`--text-only` 不等于只支持文本输入；Qwen3-Omni Thinker 仍可处理项目已支持的
> 音频、图像和文本输入。

## 2. 部署变量

执行前先按实际环境确认下表：

| 变量 | 示例 | 说明 |
|---|---|---|
| `SERVER` | `user@124.221.190.139` | SSH 服务器 |
| `SSH_PORT` | `246` | SSH 端口 |
| `DEPLOY_ROOT` | `/data/services/sglang-omni` | 服务器部署目录 |
| `MODEL_PATH` | `/data/models/Qwen3-Omni-30B-A3B-Instruct` | 服务器模型目录或 HF model ID |
| `SERVICE_PORT` | `18001` | Session Realtime 服务端口 |
| `TTS_URL` | `ws://127.0.0.1:40001/api-ws/v1/realtime` | 服务器同机真实 TTS |
| `TTS_VOICE` | `benchmark_qwen_cherry_zh` | 已验证 voice |
| `LOG_ROOT` | `/data/logs/sglang-omni-realtime` | 结构化日志持久化目录 |

本手册假设 TTS 在多模态服务同一台 Linux 宿主机上监听 `127.0.0.1:40001`。

## 3. 部署前必查项

### 3.1 代码必须来自 Git 提交

不要直接打包当前脏工作区，否则可能把临时文件、未审核配置或本机修改后的 `uv.lock`
带到服务器。

本机执行：

```powershell
git status --short
git log -1 --oneline
```

部署包应只基于已确认的 commit。记录：

```powershell
$deployCommit = git rev-parse HEAD
$deployCommit
```

### 3.2 服务器硬件和运行时

服务器检查：

```bash
nvidia-smi
docker version
git --version
uv --version
```

真实模型所需 GPU 数量、显存、CUDA、UCX 和驱动版本必须与服务器已验证的 Qwen3-Omni
部署一致。必须在启动前明确模型目录和 GPU 拓扑，不要盲目修改显存比例。

### 3.3 真实 TTS

服务器本机检查：

```bash
curl --fail --show-error http://127.0.0.1:40001/health
```

再执行 WebSocket Upgrade 预检：

```bash
curl --http1.1 -i --max-time 5 \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "http://127.0.0.1:40001/api-ws/v1/realtime?voice=benchmark_qwen_cherry_zh&session_id=deploy-probe"
```

必须看到：

```text
HTTP/1.1 101 Switching Protocols
```

## 4. 阶段一：迁移已提交源码

### 4.1 本机生成 Git bundle

Git bundle 只包含已提交历史，不会携带本机未提交文件：

```powershell
cd E:\Company_data\sglang-omni
git bundle create sglang-omni-deploy.bundle main
git bundle verify sglang-omni-deploy.bundle
```

上传：

```powershell
scp -P 246 sglang-omni-deploy.bundle user@124.221.190.139:~/
```

### 4.2 服务器从 bundle 创建独立目录

为避免覆盖旧部署，每次用 commit 创建新目录：

```bash
DEPLOY_COMMIT=$(git ls-remote ~/sglang-omni-deploy.bundle refs/heads/main | awk '{print $1}')
DEPLOY_ROOT=/data/services/sglang-omni

mkdir -p "$DEPLOY_ROOT/releases"
git clone ~/sglang-omni-deploy.bundle "$DEPLOY_ROOT/releases/$DEPLOY_COMMIT"
cd "$DEPLOY_ROOT/releases/$DEPLOY_COMMIT"
git switch --detach "$DEPLOY_COMMIT"
git status --short
```

`git status --short` 应为空。在第一阶段通过前，不删除旧 release 目录。

> 如果服务器已有公司代码迁移机制，可替换 bundle/scp，但仍必须部署一个明确 commit，
> 不得直接同步脏工作区。

## 5. 阶段一：安装和启动真实模型

### 5.1 创建环境

在新 release 目录中：

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv sync --locked
```

如果服务器已有与当前 commit/lock 一致的可复用环境，可使用该环境，但要记录 Python、
PyTorch、CUDA、SGLang 和 `sglang-omni` commit。

### 5.2 禁用 fake model

生产模型启动前必须清理 fake model 开关：

```bash
unset SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED
unset SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT
unset SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE
unset SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS
```

不建议只设为空字符串；直接 `unset` 可避免解析歧义。

### 5.3 日志目录

```bash
LOG_ROOT=/data/logs/sglang-omni-realtime
mkdir -p "$LOG_ROOT"

export SGLANG_OMNI_REALTIME_LOG_DIR="$LOG_ROOT"
export SGLANG_OMNI_REALTIME_LOG_TIMEZONE=Asia/Shanghai
export SGLANG_OMNI_SERVICE_INSTANCE_ID="baremetal-$(hostname)-$DEPLOY_COMMIT"
```

### 5.4 启动命令

先设置实际模型路径：

```bash
MODEL_PATH=/data/models/Qwen3-Omni-30B-A3B-Instruct
```

前台首次启动，便于直接看到模型加载错误：

```bash
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --text-only \
  --model-name qwen3-omni \
  --host 127.0.0.1 \
  --port 18001 \
  --realtime-tts-url "ws://127.0.0.1:40001/api-ws/v1/realtime" \
  --realtime-tts-voice "benchmark_qwen_cherry_zh" \
  --log-level info
```

说明：

- `/v1/session/realtime` 不使用旧 `--enable-realtime` 开关；不要为了新 Session 接口加该参数；
- `--realtime-tts-url/voice` 只配置 provider；客户端 `session.start.outputs` 包含 `audio` 时才连接 TTS；
- 服务只绑定 `127.0.0.1:18001`，阶段一不对公网开放；
- 启动可能需要较长时间，必须等待 Pipeline、Client 和 API 均 ready，不以进程存在冒充就绪。

如服务器 GPU 拓扑需要 TP、colocate、显存比例或指定 config，应使用服务器已验证的 Qwen3-Omni
启动参数替换上述最小命令；TTS、host 和 port 参数保持。

## 6. 阶段一：服务器本机预检

另开 SSH 窗口：

```bash
curl --fail --show-error http://127.0.0.1:18001/health
```

检查端口：

```bash
ss -lntp | grep ':18001'
```

检查 fake model 未被误启用：

- health 不应包含 `dev-fake-realtime-model`；
- 启动日志不应包含 `DEV FAKE REALTIME MODEL ENABLED`；
- 应看到真实 Pipeline/model client ready 日志。

若 health 未就绪，不要建立本地联调隧道。

## 7. 阶段一：转发到本机网页联调

### 7.1 本机建立 Session Realtime 隧道

先停止占用本机 `18080` 的旧 `dev_server`，然后在本机 PowerShell 执行：

```powershell
ssh -N -L 18080:127.0.0.1:18001 -p 246 user@124.221.190.139
```

这条隧道的方向是：

```text
本机 127.0.0.1:18080
  -> SSH
  -> 服务器 127.0.0.1:18001
```

它不会暴露服务器端口到公网。

检查：

```powershell
curl.exe http://127.0.0.1:18080/health
```

返回应与服务器本地 `18001/health` 一致。

### 7.2 本机启动静态页面

在本机项目目录执行：

```powershell
.venv\Scripts\python.exe -m http.server 8080 --directory scripts
```

浏览器打开：

```text
http://127.0.0.1:8080/realtime_tts_manual_test.html
```

页面 WebSocket 地址保持：

```text
ws://127.0.0.1:18080/v1/session/realtime
```

点击：

1. **连接**；
2. 输入测试文本；
3. **发送并合成**；
4. 等待完整文本和 `response.audio.done`；
5. 点击页面音频播放器。

该页面当前只发送文本输入，但回复端完整经过真实多模态模型和真实 TTS。若要验证用户音频
或图像输入，需要另行扩展网页，不应将本轮文本联调误称为已验证全部输入模态。

## 8. 阶段一验收门

全部满足后才进入 Docker：

- [ ] 服务器运行明确 commit，源码工作区干净；
- [ ] fake model 环境变量已清理；
- [ ] TTS `40001` health 和 WebSocket 101 通过；
- [ ] 真实模型 Pipeline 和 API ready；
- [ ] 服务器 `127.0.0.1:18001/health` 通过；
- [ ] 本机 `18080 -> 服务器 18001` 隧道通过；
- [ ] 网页收到非固定 fake 文本，且内容符合真实模型输入；
- [ ] 网页收到非空 PCM，完整收到后可正常播放；
- [ ] 日志包含 model first text、TTS first append、first PCM、audio done、response done；
- [ ] 模型、TTS、外部 WebSocket 无孤儿任务、连接泄漏或未捕获异常。

## 9. 阶段二：构建部署镜像

### 9.1 构建前确认

只使用阶段一已通过的 release 目录：

```bash
cd "/data/services/sglang-omni/releases/$DEPLOY_COMMIT"
git status --short
git rev-parse HEAD
```

`git status --short` 必须为空，`HEAD` 必须等于阶段一记录的 commit。

### 9.2 使用项目 Dockerfile 构建

项目已有 `docker/Dockerfile`，基于 `lmsysorg/sglang-omni:dev`，安装 UCX，并根据 `uv.lock` 安装
锁定依赖和当前代码。

```bash
IMAGE="sglang-omni-realtime:${DEPLOY_COMMIT}"

DOCKER_BUILDKIT=1 docker build \
  --progress plain \
  -f docker/Dockerfile \
  -t "$IMAGE" \
  .
```

构建可能需要较长时间和稳定的镜像/Python 包网络。不要用 `--no-cache` 作为默认方案。

记录镜像：

```bash
docker image inspect "$IMAGE" --format '{{.Id}}'
docker image ls "$IMAGE"
```

## 10. 阶段二：启动 Docker 服务

### 10.1 推荐网络模式

本部署推荐 Linux `--network host`，原因是：

- 容器内的 `127.0.0.1:40001` 可直接访问同宿主机 TTS；
- 服务绑定 `127.0.0.1:18001` 后，宿主机业务后端可直接连接；
- 无需另外做 `-p` 端口映射或修改 TTS URL。

`--network host` 只适用于 Linux Docker。

### 10.2 持久化目录

```bash
MODEL_PATH=/data/models/Qwen3-Omni-30B-A3B-Instruct
LOG_ROOT=/data/logs/sglang-omni-realtime

mkdir -p "$LOG_ROOT"
```

模型以只读方式挂载；日志目录必须可写。

### 10.3 启动命令

先停止阶段一的裸进程，确认 `18001` 已释放：

```bash
ss -lntp | grep ':18001' || true
```

启动容器：

```bash
docker run -d \
  --name sglang-omni-realtime \
  --restart unless-stopped \
  --gpus all \
  --shm-size 32g \
  --ipc host \
  --network host \
  --privileged \
  -v "$MODEL_PATH:/models/qwen3-omni:ro" \
  -v "$LOG_ROOT:/var/log/sglang-omni-realtime" \
  -e SGLANG_OMNI_REALTIME_LOG_DIR=/var/log/sglang-omni-realtime \
  -e SGLANG_OMNI_REALTIME_LOG_TIMEZONE=Asia/Shanghai \
  -e SGLANG_OMNI_SERVICE_INSTANCE_ID="docker-$(hostname)-$DEPLOY_COMMIT" \
  "$IMAGE" \
  sgl-omni serve \
    --model-path /models/qwen3-omni \
    --text-only \
    --model-name qwen3-omni \
    --host 127.0.0.1 \
    --port 18001 \
    --realtime-tts-url "ws://127.0.0.1:40001/api-ws/v1/realtime" \
    --realtime-tts-voice "benchmark_qwen_cherry_zh" \
    --log-level info
```

项目官方环境示例使用 `--privileged`。若公司安全基线要求去掉该参数，必须在相同 GPU/UCX
环境下重新验证，不能在未测试时直接宣称等价。

## 11. Docker 验证

### 11.1 启动日志

```bash
docker logs -f --tail 300 sglang-omni-realtime
```

等待真实 Pipeline/model client/API ready。

### 11.2 health 和端口

```bash
curl --fail --show-error http://127.0.0.1:18001/health
ss -lntp | grep ':18001'
```

### 11.3 重复本机网页联调

保持：

```powershell
ssh -N -L 18080:127.0.0.1:18001 -p 246 user@124.221.190.139
```

重新用页面连接：

```text
ws://127.0.0.1:18080/v1/session/realtime
```

确认容器版的文本和音频与阶段一行为一致。

### 11.4 业务后端连接

宿主机上的业务后端使用：

```text
ws://127.0.0.1:18001/v1/session/realtime
```

客户端首条业务消息必须是 `session.start`。需要文字和音频时：

```json
{
  "type": "session.start",
  "protocol_version": 1,
  "session_id": "business-session-unique-id",
  "outputs": ["text", "audio"],
  "locale": "zh-CN",
  "reply": {
    "instructions": "请简短回复。"
  }
}
```

`outputs` 不在 HTTP 请求头或 WebSocket URL 中。

> 如果业务后端本身也是 bridge 网络容器，它的 `127.0.0.1` 指向自己，不是宿主机。
> 要求使用 `ws://127.0.0.1:18001` 时，业务后端必须也使用 host network、共享同一网络
> namespace，或由部署系统提供等价的 localhost 共享方式。否则应改用宿主机网关或容器服务名。

## 12. Docker 阶段验收门

- [ ] 镜像 tag 与阶段一 commit 一致；
- [ ] 容器不包含宿主机脏工作区内容；
- [ ] 容器内 fake model 未启用；
- [ ] 容器可访问宿主机 TTS `127.0.0.1:40001`；
- [ ] 宿主机 `127.0.0.1:18001/health` 通过；
- [ ] 网页文本 + 音频用例通过；
- [ ] 业务后端可建立 `/v1/session/realtime` WebSocket；
- [ ] 同 Session 顺序 audio Turn 可观测到 TTS WebSocket 复用；
- [ ] `turn.cancel` 会终止模型和 TTS，且只返回一次 `turn.cancelled`；
- [ ] 宿主机日志目录已持久化，容器重启后仍保留；
- [ ] 业务后端已明确文本 Delta、音频 Delta、done、result 和 error 处理。

## 13. 日志和故障定位

### 13.1 容器 stdout/stderr

```bash
docker logs --since 10m sglang-omni-realtime
```

### 13.2 结构化时间线

```bash
find /data/logs/sglang-omni-realtime -type f -name '*.jsonl' -mmin -10 -print
```

主链路应包含：

```text
session_start_received
turn_commit_received
model_stream_submit_begin
model_first_text_delta_received
response_first_text_delta_sent
tts_connect_begin / tts_connection_reused
tts_session_ready
tts_first_append_sent
tts_first_audio_received
response_first_audio_delta_sent
tts_stream_completed
response_audio_done_sent
response_done_sent
turn_result_sent
```

### 13.3 常见错误

| 现象 | 优先检查 |
|---|---|
| 回复仍是固定 fake 文本 | `SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED` 是否被宿主机、systemd 或 Docker env 注入 |
| 服务进程存在但 health 不通 | Pipeline 仍在加载、worker 已崩溃、端口不一致 |
| `InvalidStatus` / HTTP 403 | TTS URL 是否误指向 `50001`；正确是 `40001` |
| 容器内 TTS connection refused | 是否遗漏 `--network host`，TTS 是否只绑定宿主机 loopback |
| 业务后端使用 localhost 失败 | 后端是否在另一 bridge 容器中 |
| 收到文本但 Turn 失败 | 查看 `tts_turn_failed.phase`，首版 TTS 失败策略为 `fail_turn` |
| 音频无法播放 | 核对 PCM16LE、24kHz、mono，以及 Base64 和 seq |

## 14. 重启、停止和回滚

停止：

```bash
docker stop sglang-omni-realtime
```

启动：

```bash
docker start sglang-omni-realtime
```

重启：

```bash
docker restart sglang-omni-realtime
```

查看当前镜像：

```bash
docker inspect sglang-omni-realtime --format '{{.Config.Image}}'
```

回滚时不覆盖或删除旧镜像：

1. 记录新容器日志和结构化日志；
2. 停止并将新容器改名，保留现场；
3. 使用上一个已验证 commit tag 按相同 `docker run` 参数重建容器；
4. 重新执行 health 和最小 text+audio 联调。

不要为回滚使用 `latest` 或未记录镜像 ID。

## 15. 交付记录

每次部署应留存：

```text
Git commit:
Docker image tag:
Docker image ID:
Model path / model revision:
GPU / driver / CUDA:
TTS URL and voice:
Service bind address:
Structured log directory:
Stage-one validation time:
Docker validation time:
Business backend validation result:
Known limitations:
Rollback image tag:
```

只有这份记录和两个阶段验收清单都完整，才可将容器交给业务后端长期使用。
