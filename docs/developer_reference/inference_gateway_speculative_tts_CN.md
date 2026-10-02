# Inference Gateway speculative TTS 协议

Inference Session v2 的 reply stage 可携带 `speech_speculation`：

```json
{
  "type": "stage.request",
  "stage": "reply",
  "request_id": "response-1:text",
  "payload": {"stream": true},
  "media_refs": [],
  "speech_speculation": {
    "request_id": "response-1:speech-speculation",
    "voice": "voice-1",
    "instruction": "自然地说",
    "generation_id": "generation-1",
    "output_epoch": "3",
    "text_mode": "reply_envelope_v1"
  }
}
```

Gateway 接受 reply stage 时创建独立 Embedded TTS Turn。`reply_envelope_v1` 模式会先剥离 unified reply 的 JSON plan 与 `<<TEXT>>` 标记，再把自然语言 Delta 增量送给 TTS；标记可以紧跟 JSON，也可以位于下一行，并允许跨 Delta 分片。Performance 得出最终风格后，客户端先发送：

部署时，普通完整文本回退仍使用 `http://127.0.0.1:50001/v1/tts/stream`；增量 Embedded TTS 必须使用 `ws://127.0.0.1:40001/api-ws/v1/realtime`。50001 不提供该 WebSocket 协议。

```json
{
  "type": "speech.configure",
  "request_id": "response-1:speech-speculation",
  "instruction": "温暖地说",
  "generation_id": "generation-1",
  "output_epoch": 3
}
```

reply 完成且客户端决定采用该文本时发送规范 commit hash：

```json
{
  "type": "speech.commit",
  "request_id": "response-1:speech-speculation",
  "text_hash": "sha256:<canonical metadata hash>",
  "voice": "voice-1",
  "instruction": "温暖地说",
  "generation_id": "generation-1",
  "output_epoch": 3
}
```

规范 hash 的输入是最终文本 SHA-256、音色、指令、generation 和 output epoch 的稳定 JSON。只有五项全部匹配，gateway 才返回 `speech.accepted`、有序的 `speech.audio.delta` 和 `speech.audio.done`。commit 前不会发送 PCM。缓存不存在时返回 `speculation_unavailable`，元数据不匹配时返回 `speculation_mismatch`；调用方应改用普通 `speech.request`。

Gateway 在接受 reply stage 时立即预留 speculation request id，直到该 stage 失败、取消或把所有权交给 TTS task。预留期间其他 stage、普通 TTS 和 speculation 都不得复用该 id。单次 provisional PCM 和整个 session 的 provisional PCM 都有独立字节预算。

`request.cancel` 或 `speech.cancel` 会取消模型或 speculation；reply 错误、预算溢出、连接关闭和 commit 超时同样回收底层任务与缓存。生成与公开 PCM 始终分离：首个文本 Delta 可以开始计算，但任何音频都不能在 commit 前离开 gateway。

Performance 控制默认使用与 Action Omni 相同的 Bearer token；如果显式配置独立的 `SGLANG_OMNI_PERFORMANCE_CONTROL_TOKEN`，必须同时配置 gateway 与 18004 服务。两端未显式配置时按 action token、internal token 的顺序回退，避免部署后静默降级。
