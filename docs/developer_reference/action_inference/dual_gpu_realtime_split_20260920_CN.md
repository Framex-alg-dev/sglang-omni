# Realtime 双 GPU 拆分（GPU0 控制面，GPU1 回复面）

## 目标与边界

公网仍只暴露 GPU0 的 `18004`。GPU0 上的 Realtime Gateway 是唯一的会话所有者，继续负责：

- WebSocket、Session/Turn 生命周期与事件顺序；
- 动作目录、动作评分、intent、视觉判定、performance 与表情；
- 历史路由、知识绑定、会话记忆与最终 history commit；
- TTS、turn cancel、资源释放以及最终结果的 exactly-once 发布。

GPU1 的 `18005` 只绑定 `127.0.0.1`，接收 GPU0 已经完整构造好的模型请求。它不创建第二个 Realtime Session，也不独立维护历史或发送客户端事件。

## 请求归属

| 模型任务 | 执行端 | 说明 |
| --- | --- | --- |
| `action_suffix_scoring` 及动作安全标签 | GPU0 | 保持动作关键路径与权威门控不变 |
| `session_turn_intent` | GPU0 | 保持意图、安全和视觉路由语义不变 |
| reply-history / performance / memory extraction | GPU0 | 第一阶段不改变原有决策顺序 |
| `session_visual_gesture_probe` / `session_visual_arithmetic_probe` | GPU0 | 属于动作控制，不与长回复争用 GPU1 |
| `session_reply` | GPU1 | 普通、历史、知识和主动回复均使用已物化 prompt |
| `session_pure_action_reply` | GPU1 | 生成短社交回应；语义校验仍在 GPU0 |
| `session_action_rejection` | GPU1 | 不支持动作时的自然语言回复 |
| provided reply | 不调用回复模型 | 继续由 GPU0 Gateway 直接进入 TTS |

任务路由使用显式 allowlist。未识别的新任务一律留在 GPU0，避免新增安全或控制调用被意外迁移。

## 兼容和故障语义

1. GPU1 连接失败、HTTP 错误或在首个输出前失败时，同一个请求 ID 自动在 GPU0 重试。
2. GPU1 已经产生任何流式输出后不再重试，避免重复文本、重复 TTS 和重复客户端事件。
3. 活跃请求记录实际执行端；turn cancel 精确中止该端。注册/清理竞态下同时向两端发送 abort。
4. Session 关闭时同时释放 GPU0、GPU1 的 session-scoped KV cache。
5. GPU1 的私有接口默认不存在；只有设置 `SGLANG_OMNI_INTERNAL_MODEL_API=1` 才注册，并强制 Bearer token。
6. 图片的 prepared RGB bytes 使用 msgpack 二进制透传，不重新 JPEG 编解码；音频、metadata、logprobs 和 usage 保持原结构。

## Session 预热

`session.start` 现在并行预热普通用户回复的静态 system prefix。路由客户端根据 `task=session_reply` 将该预热送到 GPU1；intent、视觉和动作前缀继续预热 GPU0。GPU1 不可用时预热可降级到 GPU0，Session 仍能启动。

## 推测回复与等价性门控

GPU0 支持 `SGLANG_OMNI_SPECULATIVE_REPLY_MODE=off|shadow|enforce`：

- `off`：完全使用原权威回复时序；
- `shadow`：从 `turn.commit` 立即在 GPU1 启动 current-only 回复，但文本只保存在服务端，不发送 provisional/text/audio 事件，不进入 TTS 或会话历史；原 intent、history、knowledge、action 和权威回复链路不变；
- `enforce`：仍先运行原权威判定，只在严格的 current-only 普通语言轮采用已生成结果。任何历史需求、知识注入、相机输入、动作/表情/反应、纯动作、provided/proactive 或动态上下文都会丢弃推测结果并重新运行原权威回复。

部署文件默认使用 `shadow`。结构化日志 `speculative_reply_shadow_comparison` 只记录长度、SHA-256、exact match 和归一化相似度，不记录额外明文。观察准确率达标后才可将 GPU0 unit 改为 `enforce`；动作评分、intent 和路由模型始终在 GPU0 上运行，推测分支不参与这些决策。

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
- intent/控制请求：`executor=control`（显式 completion 路由时）；
- GPU1 故障降级：`executor=control_fallback`；
- session.start：`reply_prefix_prefilled=true`。
- shadow 轮次：`speculative_reply_started` 和 `speculative_reply_shadow_comparison`，且 `adopted=false`。

功能回归至少覆盖：普通文本/音频回复、动作-only、动作加短回复、不支持动作、相机手势、视觉算术、provided reply、主动事件、知识回复、历史追问、会话记忆、cancel、断线和 GPU1 故障回退。

## 后续优化

- 将 history/knowledge route 合并为低成本控制输出；
- GPU1 回复 prompt blueprint 与更细粒度的静态 KV 预热；
- 在验证准确率前不启用当前未校准的 single-token 动作映射。

这些优化会改变执行顺序或决策边界，应在双 GPU 等价性回归完成后单独灰度。
