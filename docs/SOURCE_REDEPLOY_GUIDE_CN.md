# SGLang-Omni v3 从源码重新部署指南

本文记录如何在当前服务器环境中，从源码重新部署并验证 `sglang-omni-b3f53c9-v3`。目标是运行真实 Qwen3-Omni FP8 模型，并连接宿主机已有的真实 TTS WebSocket 服务。

## 1. 当前部署基线

| 项目 | 值 |
|---|---|
| 源码目录 | `/data/luozhijie/sglang-omni-b3f53c9-v3` |
| 源码提交 | `b3f53c916d253d2cd6b58afbfb83b84835b5f6a0` |
| FP8 模型 | `/data/models/Qwen3-Omni-30B-A3B-FP8` |
| Chat template 模型 | `/data/models/Qwen3-Omni-30B-A3B-Instruct` |
| TTS WebSocket | `ws://127.0.0.1:40001/api-ws/v1/realtime` |
| 默认音色 | `benchmark_qwen_cherry_zh` |
| 模型服务端口 | `18001`，可通过启动参数改为 `18101` 等空闲端口 |
| 测试 GPU | 宿主机 GPU 5 |
| Docker 镜像 | `sglang-omni-realtime:b3f53c9` |
| Docker 容器名 | `sglang-omni-realtime-v3` |

注意：GPU 编号以宿主机为准。容器仅暴露一张 GPU 时，程序日志中的 `gpu_id=0` 对应宿主机传入的那张卡，这是正常映射。

## 2. 部署前检查

不要直接假设端口、GPU 或容器名空闲。先执行只读检查：

```bash
nvidia-smi
ss -lntp | grep -E ':(18001|18101|40001)\b' || true
docker ps -a --filter name=sglang-omni-realtime-v3
curl --max-time 5 http://127.0.0.1:40001/health
```

如果目标端口或 GPU 正被他人使用，应选择其他空闲资源，不要停止或删除无法确认归属的进程和容器。

## 3. Python 环境准备

### 3.1 使用已经准备好的 v3 环境

当前源码目录已有 `.venv` 时：

```bash
cd /data/luozhijie/sglang-omni-b3f53c9-v3
source .venv/bin/activate
```

检查命令和 Python import 是否指向 v3：

```bash
command -v sgl-omni
python -c 'import sglang_omni; print(sglang_omni.__file__)'
```

输出路径应位于 `sglang-omni-b3f53c9-v3` 或当前环境的 v3 安装位置，不能仍然指向 v2。

### 3.2 从 v2 复用环境

v2 与当前 v3 的 `pyproject.toml` 和 `uv.lock` 相同，因此可以复用依赖环境：

```bash
cd /data/luozhijie
cp -a --reflink=auto \
  sglang-omni-6919c65-v2/.venv \
  sglang-omni-b3f53c9-v3/.venv

cd /data/luozhijie/sglang-omni-b3f53c9-v3
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install \
  --no-deps --no-build-isolation --editable .
```

复制环境后必须重新执行 editable install，否则入口脚本和包路径可能仍指向 v2。

如果未来两个版本的锁文件不再一致，不应复用旧环境。可先比较：

```bash
sha256sum \
  /data/luozhijie/sglang-omni-6919c65-v2/uv.lock \
  /data/luozhijie/sglang-omni-b3f53c9-v3/uv.lock
```

## 4. 源码目录中的适配

### 4.1 Blackwell 运行环境变量

启动服务前设置：

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
```

这些配置用于当前 RTX PRO 5000 72GB Blackwell 环境。成功启动时日志应显示 Blackwell 架构、FlashInfer 后端以及项目提供的 kernel configs 已加载。

### 4.2 Docker 构建上下文排除项

`.dockerignore` 中增加：

```text
.nv/
.triton/
```

同时保留已有的 `.venv/`、`.cache/`、`.git/` 等排除项，防止把虚拟环境、运行缓存和源码历史打进镜像。

### 4.3 测试网页调整

文件：

```text
scripts/realtime_tts_manual_test.html
```

当前调整包括：

- WebSocket 默认地址改为 `ws://127.0.0.1:18180/v1/session/realtime`；
- 页面说明改为真实 Qwen3-Omni + 真实 TTS；
- 等待事件超时由 10 秒增加到 30 秒；
- 默认输入改为简短中文测试文本。

这里的 `18180` 是 Windows 本地 SSH 隧道端口，不是服务器模型服务实际端口。

## 5. 原生方式启动验证

建议先用源码环境验证，再构建 Docker。以下示例使用 GPU 5 和服务器端口 18101；实际部署前必须确认二者空闲：

```bash
cd /data/luozhijie/sglang-omni-b3f53c9-v3
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=5 \
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
SGLANG_OMNI_QWEN3_CHAT_TEMPLATE_MODEL=/data/models/Qwen3-Omni-30B-A3B-Instruct \
sgl-omni serve \
  --model-path /data/models/Qwen3-Omni-30B-A3B-FP8 \
  --text-only \
  --model-name qwen3-omni \
  --host 127.0.0.1 \
  --port 18101 \
  --mem-fraction-static 0.82 \
  --realtime-tts-url ws://127.0.0.1:40001/api-ws/v1/realtime \
  --realtime-tts-voice benchmark_qwen_cherry_zh \
  --log-level info
```

关键参数说明：

- `--text-only`：模型负责文本回复，音频交给外部真实 TTS；
- `--mem-fraction-static 0.82`：适配当前 72GB GPU；
- `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`：只使用服务器本地模型；
- `SGLANG_OMNI_QWEN3_CHAT_TEMPLATE_MODEL`：指定 Instruct 模型中的模板资源；
- `--host 127.0.0.1`：仅允许宿主机本地或 SSH 隧道访问。

服务首次启动会加载 FP8 权重并捕获 CUDA Graph，可能需要数分钟。等待日志出现：

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:<端口>
```

健康检查：

```bash
curl http://127.0.0.1:18101/health
```

健康响应应至少包含：

```json
{"status":"healthy","running":true}
```

## 6. 构建 Docker 镜像

容器文件位于：

```text
/data/luozhijie/sglang-omni-v3-container
```

Dockerfile 继承已经包含相同依赖锁、CUDA 和 Python 依赖的 v2 镜像：

```dockerfile
FROM sglang-omni-realtime:6919c65-r1
```

构建时只复制并重新安装 v3 项目本身，从而复用已有大体积依赖层：

```bash
cd /data/luozhijie/sglang-omni-b3f53c9-v3

DOCKER_BUILDKIT=0 docker build \
  --build-arg SOURCE_COMMIT=b3f53c916d253d2cd6b58afbfb83b84835b5f6a0 \
  -f /data/luozhijie/sglang-omni-v3-container/Dockerfile \
  -t sglang-omni-realtime:b3f53c9 \
  .
```

当前服务器曾遇到 BuildKit 读取本地基础镜像层的问题，因此这里使用 classic builder。若环境中的 BuildKit 已能稳定使用本地镜像，也可以移除 `DOCKER_BUILDKIT=0`。

构建后检查：

```bash
docker image inspect sglang-omni-realtime:b3f53c9 \
  --format 'ID={{.Id}} SIZE={{.Size}} REVISION={{index .Config.Labels "org.opencontainers.image.revision"}} CMD={{json .Config.Cmd}}'
```

## 7. 启动 Docker 容器

默认启动：

```bash
sudo /data/luozhijie/sglang-omni-v3-container/run-container.sh
```

脚本默认使用：

- 容器名 `sglang-omni-realtime-v3`；
- 宿主机 GPU 5；
- 服务端口 18001；
- FP8 模型和 Instruct 模型只读挂载；
- `--network host`，以访问宿主机 40001 TTS；
- `--restart unless-stopped`；
- 日志目录 `/data/luozhijie/sglang-omni-v3-container/logs`。

如需使用 18101：

```bash
sudo SERVICE_PORT=18101 \
  /data/luozhijie/sglang-omni-v3-container/run-container.sh
```

如需选择其他 GPU 或容器名：

```bash
sudo GPU_DEVICE=5 \
  SERVICE_PORT=18101 \
  CONTAINER_NAME=sglang-omni-realtime-v3-test \
  /data/luozhijie/sglang-omni-v3-container/run-container.sh
```

脚本发现同名容器已存在时会直接拒绝执行，不会覆盖或删除已有容器。若只是停止过的同名容器，也需要先明确其归属，再由负责人决定是重新启动、改名还是删除。

查看启动状态：

```bash
docker ps --filter name=sglang-omni-realtime-v3
docker logs --tail 200 -f sglang-omni-realtime-v3
curl http://127.0.0.1:18001/health
```

## 8. Windows 网页测试

在服务器启动静态网页服务：

```bash
cd /data/luozhijie/sglang-omni-b3f53c9-v3/scripts
python3 -m http.server 18102 --bind 127.0.0.1
```

假设模型服务监听服务器 18001，在 Windows PowerShell 中建立隧道：

```powershell
ssh -N `
  -L 18180:127.0.0.1:18001 `
  -L 18102:127.0.0.1:18102 `
  用户名@服务器地址
```

浏览器访问：

```text
http://127.0.0.1:18102/realtime_tts_manual_test.html
```

如果服务器模型服务使用 18101，只需把 PowerShell 第一条转发改为：

```powershell
-L 18180:127.0.0.1:18101
```

网页中的 WebSocket 地址仍保持 `ws://127.0.0.1:18180/v1/session/realtime`。

## 9. 停止服务

先确认容器身份：

```bash
docker inspect sglang-omni-realtime-v3 \
  --format 'NAME={{.Name}} IMAGE={{.Config.Image}} STATUS={{.State.Status}} ID={{.Id}}'
```

确认无误后只停止，不删除：

```bash
docker stop sglang-omni-realtime-v3
```

检查状态和端口：

```bash
docker ps -a --filter name=sglang-omni-realtime-v3
ss -lntp | grep -E ':(18001|18101|18102)\b' || true
```

不要使用模糊名称批量停止容器，也不要删除无法确认归属的容器、镜像、进程或文件。

## 10. 已验证结果与常见现象

当前镜像和启动参数已经完成以下验证：

- FP8 模型权重正常加载；
- Blackwell kernel configs 正常安装和识别；
- CUDA Graph 捕获完成；
- 全部 pipeline stages 启动；
- `/health` 返回 healthy；
- `/v1/session/realtime` 网页测试通过；
- 真实 TTS 连接和语音播放通过。

启动日志中的以下现象不是启动失败：

- 容器日志显示 `gpu_id=0`：单卡映射后的容器内编号；
- `rope_parameters` 未识别字段警告：当前测试未阻止模型运行；
- 缺少 down-projection 单独调优配置：会提示性能可能不是最优，但可正常运行；
- 首次 CUDA Graph 捕获耗时较长：等待启动完成后再做健康检查。

如果 `curl` 立即返回 connection refused，优先检查实际启动端口：

```bash
ss -lntp | grep sgl-omni
docker logs --tail 100 sglang-omni-realtime-v3
```

不要仅根据历史端口号判断服务是否启动。
