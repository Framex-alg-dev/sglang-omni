# Omni 当前问题与 Infra 交接（2026-09-17 实测）

## 1. 结论

当前服务没有经过代理或外部网络。业务服务 `30070` 通过
`ws://127.0.0.1:18008/v1/session/realtime` 连接 Omni，Omni 运行在宿主机物理
GPU 6。

当前最值得 Infra 关注的不是动作 PPL 的 prefix，而是回复生成路径：最近一次真实
Turn 的回复输入为 11,600 tokens，其中只命中 4,822 tokens 缓存，仍需 prefill
6,778 tokens。该 generation 请求又与 12 个 performance 候选处于同一个物理 batch，
从正式提交 reply 请求到首 token 约 625 ms；从 `turn.commit` 计算的官方首 token
延迟为 1,306 ms。

动作路径的 prefix 已基本优化到位：最近一次 Turn 的动作 prompt 为 13,361 tokens，
其中缓存命中 12,898 tokens，仅计算 463 tokens，缓存命中率约 96.5%。

## 2. 本次真实测试 Case

### 2.1 日志定位

- 测试时间：`2026-09-17 18:39:37 CST`
- Omni Session：`sess_02361a02`
- Turn：`turn_8c4340f1_20`
- Trace：`sess_02361a02:turn_8c4340f1_20:6c5dc60d7dfb4d07af327e2791359212`
- 服务实例：`sglang-omni-xmt-gpu6`
- API PID：`1149627`
- Worker PID：`1150557`

对应日志目录：

```text
/data/fanshide/sglang-omni-xmt/logs/realtime/2026-09-17/18/
```

### 2.2 用户输入和模型输出

用户面对摄像头做数字二手势，并说：

```text
这个是数字几。
```

commit 时服务收到：

| 输入 | 数量 |
|---|---:|
| 直接文本 | 0 字符 |
| 音频块 | 40 |
| 图片帧 | 7 |
| 图片角色 | 6 张 `user_camera` + 1 张 `avatar_state` |

回复请求实际转发 1 段聚合音频和 6 张 `user_camera` 图片；动作评分请求使用 1 段
聚合音频和 2 张图片。

结果：

- ASR：`这个是数字几。`
- 动作：`candidate_id=259`，`数字二手势`，识别正确；
- 回复：`加起来是数字二。`，数字正确但措辞错误；
- completion：7 tokens。

措辞错误不是 Infra/GPU 问题。当前 Turn 的客户端回复规则包含
`加起来是数字【实际计算结果】`，模型把单个数字识别问题错误套入了加法模板。

### 2.3 端到端时间线

以下均以 `turn.commit` 为零点：

| 事件 | 延迟 |
|---|---:|
| visual scope gate 完成 | 84 ms |
| ASR 最终文本 | 205 ms |
| intent 完成 | 304 ms |
| 动作结果发布 | 681 ms（D_video 记录 685 ms） |
| reply 首 token | 1,306 ms |
| reply 文本完成 | 1,359 ms |
| TTS 首个音频块 | 1,551 ms |
| `response.done` | 1,881 ms |
| 整个 `turn.result` | 1,881 ms |

首 token 的主要分段：

```text
commit
  ├─ 0 ~ 304 ms：视觉 gate + intent
  ├─ 304 ~ 681 ms：动作评分并发布动作
  └─ 681 ~ 1306 ms：reply prefill / 调度 / 首 token，约 625 ms
```

### 2.4 动作请求 token 和耗时

本轮不是“模仿这个手势”的专用 gesture route，而是普通问答，最终评分完整动作目录：

| 指标 | 值 |
|---|---:|
| 候选数 | 169 |
| prompt tokens | 13,361 |
| reusable/cached prefix | 12,898 |
| 实际计算 prefix | 463（scheduler scheduled 464） |
| prefix cache hit ratio | 96.5% |
| prefix prefill | 106.2 ms |
| candidate suffix tokens | 676 |
| candidate suffix batch | 170.3 ms |
| action score total | 371.3 ms |
| action scoring API elapsed | 374.9 ms |

动作 prompt 大，但绝大部分已经缓存；当前它不是首 token 的最大剩余问题。

### 2.5 回复请求 token 和耗时

| 指标 | 值 |
|---|---:|
| prompt tokens | 11,600 |
| cached tokens | 4,822 |
| 本轮实际 prefill tokens | 6,778 |
| cache hit ratio | 41.6% |
| completion tokens | 7 |
| reply submit 到首 token | 约 625 ms |
| commit 到首 token（官方 TTFT） | 1,306 ms |
| 文本流持续时间 | 52 ms |

reply 动态部分包含本轮音频、6 张摄像头图片、当前场景规则，以及约 20K 字符的外部
Knowledge evidence。该 Knowledge 与“这是数字几”无关，但仍被注入回复请求。日志显示
reply history route 为 `CURRENT_ONLY`，历史共 6 Turn、实际转发 0 Turn，因此 6,778 个
动态 tokens 不是历史对话造成的。

Worker 在 `18:39:38.464` 选择了一个 13-request 的物理 batch：

- 12 个 performance PPL candidate；
- 1 个 generation reply，range `4822..11600`，即 6,778 个新 tokens。

该 batch 与首 token 的等待区间高度重合，应重点检查调度和同卡竞争。

## 3. Prefill 当前状态

### 3.1 配置

当前工作树把 Thinker 的：

```yaml
chunked_prefill_size: 8192
```

从 1024 调到了 8192。该配置是单次调度允许处理的 prefill chunk 上限，不等于把所有
动态输入提前计算。本轮 reply 的 6,778 个新 tokens 因此可以在一个 chunk 内处理；动作
请求只需计算 463 tokens，也只需要一个 chunk。

### 3.2 已做的 gesture prefix prewarm

“模仿这个手势”路径的 143 条视觉动作定义，已从每轮动态 prompt 移到 Session 稳定
instruction，并在 `session.start` 增加 gesture 专用 prewarm。

实测变化：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 动态计算 tokens | 5,307 | 约 519～663 |
| prefix prefill | 725 ms | 约 125～133 ms |
| commit 到 action ready | 约 1,153 ms | 约 519～529 ms |

代价是 Session 启动多一次 gesture prewarm，新增约 1.42 秒；当前 Session 总启动时间约
3.65～3.69 秒，原来约 2.28 秒。它只改善 gesture route；本次“这个是数字几”走的是
普通全目录路径，因此没有命中 gesture 专用 namespace。

### 3.3 Infra 建议优先检查

1. generation 是否应高于 performance PPL candidate；可否把 12 候选 performance 延后到
   reply 首 token 之后，或设置更低 admission priority。
2. 单 GPU 同时承载 169-candidate action、12-candidate performance 和 reply generation
   时的物理 batch 策略是否合理。
3. 对 reply prompt 做 prefix-cache 分段统计，确认 4,822-token cache boundary 为什么无法
   覆盖更多稳定 persona / system 内容。
4. 对比以下两组：纯 text+video（关闭 action/performance）与四模态业务 Session，量化
   action/performance 对 reply TTFT 的影响。
5. `chunked_prefill_size=8192` 在双 Session 并发下做显存和 P99 soak；当前 GPU 6 实测约
   59.6/73.4 GiB 已用，尚未出现 OOM，但不能只依据单 Session 判断。

应用层同时应处理：对无关问题不注入 20K 字符 Knowledge、减少回复模型使用的摄像头帧数、
压缩角色 prompt。这三项会直接减少 reply 的 6,778 个动态 tokens。

## 4. 当前部署方式

### 4.1 拓扑

```text
D_video_call :30070
  -> ws://127.0.0.1:18008/v1/session/realtime
     -> Qwen3-Omni-30B-A3B-FP8（物理 GPU 6）
     -> ws://127.0.0.1:40001/api-ws/v1/realtime（外部 TTS）
```

Omni 与业务端均走 loopback；systemd 单元明确清除了所有 HTTP/HTTPS/ALL proxy 环境变量。

### 4.2 systemd

- Unit：`/etc/systemd/system/sglang-omni-xmt-gpu6.service`
- 运行用户：`user`
- 工作目录：`/data/fanshide/sglang-omni-xmt`
- 物理 GPU：`CUDA_VISIBLE_DEVICES=6`
- 模型：`/data/models/Qwen3-Omni-30B-A3B-FP8`
- Chat template：`/data/models/Qwen3-Omni-30B-A3B-Instruct`
- 配置：`deploy/config_single_gpu.yaml`
- API：`0.0.0.0:18008`
- TTS：`ws://127.0.0.1:40001/api-ws/v1/realtime`
- 日志：`logs/sglang-omni-gpu6.log` 和 `logs/realtime/<date>/<hour>/`

实际 ExecStart：

```bash
/data/fanshide/sglang-omni-xmt/.venv/bin/sgl-omni serve \
  --config /data/fanshide/sglang-omni-xmt/deploy/config_single_gpu.yaml \
  --host 0.0.0.0 \
  --port 18008 \
  --log-level info \
  --realtime-tts-url ws://127.0.0.1:40001/api-ws/v1/realtime \
  --realtime-tts-voice benchmark_qwen_cherry_zh
```

常用运维命令：

```bash
systemctl status sglang-omni-xmt-gpu6.service
systemctl restart sglang-omni-xmt-gpu6.service
curl http://127.0.0.1:18008/health
tail -f /data/fanshide/sglang-omni-xmt/logs/sglang-omni-gpu6.log
```

当前分支为 `codex/action-ppl-visual-routing-20260917`，基线提交为
`3fdd6ef2431849913225a3536dc560235c1e72e9`。服务正在直接运行未提交工作树，包含
`chunked_prefill_size=8192` 和 gesture prefix prewarm；Infra 复现时必须保留这些改动，
不能只 checkout 基线提交后启动。

## 5. 文字 + 视频直连 Case

仓库提供独立脚本：

```bash
cd /data/fanshide/sglang-omni-xmt
.venv/bin/python scripts/session_realtime_text_video_smoke.py \
  --url ws://127.0.0.1:18008/v1/session/realtime \
  --video tests/data/draw.mp4 \
  --text '请简要描述视频中发生了什么。' \
  --max-frames 4 \
  --frame-interval 0.5
```

该脚本执行：

```text
session.start outputs=[text]
turn.start
input.text.set
input.image.append × N（从 MP4 抽 JPEG 帧）
turn.commit
等待 response.text.delta 和 turn.result
```

它直接连接 18008，显式设置 `proxy=None`，不经过 30070，不启用 action、performance、
expression 或 TTS，适合 Infra 测纯 text+video 的首 token 基线。再与上面的真实四模态
Turn（TTFT 1,306 ms）比较，即可定位额外延迟来自基础 generation prefill，还是业务并发
分支。

## 6. 其他已知问题（避免混淆）

1. D_video_call 注册 524 candidates / 51 categories，而 Omni 全局目录为 168 / 11；未知
   fallback 曾导致 30070 主动断开。客户端现已忽略不可执行的未知 fallback，但目录版本仍应统一。
2. “这是数字几”回复成“加起来是数字二”属于客户端回复规则误匹配，不是模型部署或网络问题。
3. gesture 定义迁入 prefix 后的动作精度仍需要固定数据集做 A/B；目前只有现场样本和单元测试。
