# Realtime 双 GPU 拆分（GPU0 控制面，GPU1 回复面）

## 目标与边界

公网仍只暴露 GPU0 的 `18004`。GPU0 上的 Realtime Gateway 是唯一的会话所有者，继续负责：

- WebSocket、Session/Turn 生命周期与事件顺序；
- 动作目录、动作评分、动作型视觉判定、performance 与表情/TTS 指令；
- 历史、知识绑定、会话记忆的状态编排与最终 history commit；
- TTS、turn cancel、资源释放以及最终结果的 exactly-once 发布。

GPU1 的 `18005` 只绑定 `127.0.0.1`，承接回复侧全部模型计算：回复生成、turn intent、历史路由、语音模式、纯动作回复校验、记忆提取和回复型视觉算术。它不创建第二个 Realtime Session，也不独立维护历史或发送客户端事件；这些有顺序和一致性要求的状态仍由 GPU0 编排。

## 请求归属

| 模型任务 | 执行端 | 说明 |
| --- | --- | --- |
| `action_suffix_scoring` 及动作安全标签 | GPU0 | 保持动作关键路径与权威门控不变 |
| `session_turn_intent` / prewarm | GPU1 | 回复侧意图计算不再占用 GPU0；动作在 enforce 直选模式下不等待它启动 |
| `reply_history_route` / `reply_speech_mode` / `pure_action_reply_validation` | GPU1 | 通过私有 action-score RPC 执行回复侧短分类 |
| `session_memory_extract` | GPU1 | 记忆抽取模型计算迁移；记忆状态写入仍由 GPU0 串行提交 |
| `session_visual_arithmetic_probe` / prewarm | GPU1 | 属于回复理解，不阻塞 GPU0 动作关键路径 |
| `session_visual_gesture_probe` | GPU0 | 直接服务动作选择，留在动作关键路径 |
| `performance` score、表情与 TTS 指令 | GPU0 | 等动作计算实际完成后才获准运行，避免争抢动作算力 |
| `session_reply` | GPU1 | 普通、历史、知识和主动回复均使用已物化 prompt |
| `session_pure_action_reply` | GPU1 | 生成短社交回应，语义校验也在 GPU1 |
| `session_action_rejection` | GPU1 | 不支持动作时的自然语言回复 |
| provided reply | 不调用回复模型 | 继续由 GPU0 Gateway 直接进入 TTS |

任务和短分类 stage 都使用显式 allowlist。未识别的新任务一律留在 GPU0，避免新增安全或动作控制调用被意外迁移。

## 兼容和故障语义

1. GPU1 连接失败、HTTP 错误或在首个输出前失败时，同一个请求 ID 自动在 GPU0 重试。
2. GPU1 已经产生任何流式输出后不再重试，避免重复文本、重复 TTS 和重复客户端事件。
3. 活跃请求记录实际执行端；turn cancel 精确中止该端。注册/清理竞态下同时向两端发送 abort。
4. Session 关闭时同时释放 GPU0、GPU1 的 session-scoped KV cache。
5. GPU1 的私有接口默认不存在；只有设置 `SGLANG_OMNI_INTERNAL_MODEL_API=1` 才注册，并强制 Bearer token。
6. 图片的 prepared RGB bytes 使用 msgpack 二进制透传，不重新 JPEG 编解码；音频、metadata、logprobs 和 usage 保持原结构。

## Session 预热

`session.start` 并行预热普通回复、turn intent 和回复型视觉算术前缀，这些请求送到 GPU1；动作和视觉手势前缀留在 GPU0。GPU1 不可用时预热可降级到 GPU0，Session 仍能启动。

## 推测回复与等价性门控

GPU0 支持 `SGLANG_OMNI_SPECULATIVE_REPLY_MODE=off|shadow|enforce`：

- `off`：完全使用原权威回复时序；
- `shadow`：从 `turn.commit` 立即在 GPU1 启动 current-only 回复，但文本只保存在服务端，不发送 provisional/text/audio 事件，不进入 TTS 或会话历史；原 intent、history、knowledge、action 和权威回复链路不变；
- `enforce`：仍先运行原权威判定，只在严格的 current-only 普通语言轮采用已生成结果。相机帧本身不再阻止早启动；如果完整 intent 判定为视觉语义请求，仍会丢弃该结果。任何历史需求、知识注入、动作/表情/反应、纯动作、provided/proactive、动态上下文或视觉路由都会丢弃推测结果并重新运行原权威回复。

由于当前系统尚未承载生产流量，本分支部署文件直接使用目标态 `enforce`。推测输出在权威门控完成前保持私有，不进入客户端、TTS 或历史；门控拒绝后，普通、历史、知识、纯动作短回应和动作拒绝等所有生成型回复仍在 GPU1 执行。动作评分和动作型视觉手势始终在 GPU0；回复 intent、历史/语音路由及回复校验在 GPU1。需要对照验证时可临时切回 `shadow`；结构化日志 `speculative_reply_shadow_comparison` 只记录长度、SHA-256、exact match 和归一化相似度，不记录额外明文。

### 已知问题：History 路由拒绝推测回复时，TTS 取消会阻塞权威回复

2026-09-24 使用 `character_4ef8a49978444c9aba77fc7293fb5984` 的定向回归发现：History route 通常约 `0.7s` 即可判定需要历史，但当前 `enforce` 实现会为 current-only 推测回复提前创建私有 TTS。门控拒绝该回复时，`_discard_provisional_reply(..., wait_for_cleanup=True)` 同步等待 `_abort_reply_tts()`；远端 `cancel_active_turn()` 最长约 `10s` 才返回，因此 History 权威回复要到约 `11.2–11.7s` 才产生公开音频。动作首事件仍约 `0.4–0.5s`，问题只在回复/TTS 路径。

复现证据：`reports/dual_gpu_dev_three_fixes_targeted_20260924.json`，测试会话 `latency-461a6c9e409048aab3014e3e28d268f0`。

推荐修复由两部分组成：

1. 保留 speculative 文本生成，但在 History route 完成前不创建或写入 speculative TTS；只有门控采用 current-only 结果后才启动 TTS。History/knowledge 等拒绝路径直接丢弃缓存文本并为权威回复启动 TTS。
2. 将 TTS cancel 改为有界快速取消：先让旧 generation 失效并停止向客户端发布旧音频，等待 `100–300ms`；超时则淘汰旧连接并后台清理，权威回复使用独立的预热连接。不能只把 `wait_for_cleanup=True` 改成 `False`，因为新旧回复当前共享 `EmbeddedTTS` 与 `_turn_lock`，异步清理可能取消新回复或继续阻塞它。

验收至少覆盖 History、普通 current-only、连续 turn 和用户打断；要求无旧音频泄漏、无 `control_fallback`、动作延迟不回退，并将 History 首音频恢复到约 `1.5–2s`（最终阈值以稳定环境的 p95 为准）。该项当前仅记录，尚未实现。

## 部署

1. 创建不入库的共享 token 文件：

   ```bash
   cp deploy/realtime-executors.env.example deploy/realtime-executors.env
   # 将占位值替换为随机 secret
   chmod 600 deploy/realtime-executors.env
   ```

2. 用仓库中的私有执行器配置替换并重启现有 GPU1 服务：

   ```bash
   sudo install -o root -g root -m 0644 \
     deploy/sglang-omni-gpu1.service \
     /etc/systemd/system/sglang-omni-gpu1.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now sglang-omni-gpu1.service
   ```

3. 将本分支的 GPU0 unit 安装到 systemd，随后重启 GPU0：

   ```bash
   sudo install -o root -g root -m 0644 \
     deploy/sglang-omni-gpu0.service \
     /etc/systemd/system/sglang-omni-gpu0.service
   sudo systemctl daemon-reload
   sudo systemctl restart sglang-omni-gpu0.service
   ```

启动顺序建议先 GPU1、后 GPU0，但不是硬依赖；GPU0 在 GPU1 启动、重启或故障期间仍能回退为原单 GPU 功能。

## 验证

```bash
systemctl is-active sglang-omni-gpu1.service
systemctl is-active sglang-omni-gpu0.service
curl -sS http://127.0.0.1:18004/health
```

Realtime 结构化日志应出现：

- 普通回复：`realtime_model_request_routed executor=reply`；
- 回复 intent/history/speech/memory 请求：`executor=reply`；
- 动作评分、performance 和视觉手势请求：`executor=control`；
- GPU1 故障降级：`executor=control_fallback`；
- session.start：`reply_prefix_prefilled=true`。
- enforce 普通 current-only 轮次：`speculative_reply_started`、`speculative_reply_resolved adopted=true`，且只有一个 `session_reply` 模型请求；
- enforce 历史、知识、纯动作或视觉轮次：`speculative_reply_resolved adopted=false`，随后权威生成请求仍应出现 `realtime_model_request_routed executor=reply`。

功能回归至少覆盖：普通文本/音频回复、动作-only、动作加短回复、不支持动作、相机手势、视觉算术、provided reply、主动事件、知识回复、历史追问、会话记忆、cancel、断线和 GPU1 故障回退。

## 设计边界

“回复逻辑迁移”指回复侧模型计算迁移到 GPU1，不是把 Session/History/Knowledge 的可变状态复制到第二个进程。后者会引入双写、事件乱序和 cancel 竞态；当前实现由 GPU0 生成完整请求、GPU1 只计算、GPU0 单点提交状态，因此保持原回复语义和事件顺序。
