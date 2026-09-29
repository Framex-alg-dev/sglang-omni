# Action Omni 双通道推理协议

## 服务边界

GPU0/18004 使用 `SGLANG_OMNI_SERVICE_ROLE=action-decision`，只提供动作决策，
不拥有回复、TTS、会话状态或数字人投递。Body 与 Expression 使用同一份 Action
Omni 权重，但每个决策点发起两次互相独立的请求：

- `channel=body`：74 个产品身体动作 + `IB0`（不执行）+ `000`（不支持）。
- `channel=expression`：8 个基础表情 + `IF0`（保持）+ `000`（不支持）。

CPU Gateway 的五个阶段为 `classifier`、`brain`、`reply`、`body`、
`expression`。`body` 和 `expression` 都转发到
`/v1/action-decision/realtime`；Gateway 会写入并校验对应的 `channel`。

## 请求

WebSocket 使用通用的 `request.start` / `input.media` / `request.commit`
传输协议。`request.start.payload` 至少包含：

```json
{
  "request_id": "action:turn-1:initial:body",
  "session_id": "sess-1",
  "turn_id": "turn-1",
  "decision_point_id": "turn-1:initial",
  "channel": "body",
  "text": "用户本轮文本或 ASR 辅助文本",
  "runtime_context": {},
  "channel_enabled": true,
  "prohibited": false
}
```

音频和多张图片通过二进制媒体帧发送。服务会同时把媒体占位符及
`metadata.audios/images` 交给模型，不把 ASR 当作原始音频的替代品。

## 输出约束

候选来自版本化产品目录、两-token 映射和受信任 Agent 绑定。服务在启动时验证
完整映射与模型 tokenizer；推理时同时使用：

- 候选 code 的服务端正则约束；
- `min_new_tokens=2`、`max_new_tokens=2`；
- `ignore_eos=true`；
- 输出 code 到映射表的严格反查。

因此 `max_tokens=2` 不再是唯一保护。模型返回目录外内容会作为服务错误处理，
不会被猜测或模糊匹配。

## 确定性分支与融合

通道关闭/禁止时直接返回 `IB0` 或 `IF0`，不调用模型。注册过的
`required_action_candidate_id` 也直接返回；绑定另一通道时不会污染当前通道。

响应携带 `occupies_channels`。D 在收到两次结果后负责最终融合：身体动作占用
`face` 时，普通表情被抑制，显式高优先级表情延迟发布。任一通道失败时只对该
通道做 fail-soft，不能取消另一通道或回复/TTS。

`request_id` 在服务内幂等；同 ID 同输入复用结果，同 ID 不同输入拒绝。结构化
日志记录 session、Turn、决策点、通道、候选数、目录/映射/模型版本、媒体数、
最终 code、token IDs、耗时、确定性分支及失败原因。
