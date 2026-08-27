# TTS 移植参考包

本目录是当前项目独立 TTS 链路的精简副本，用于把相同逻辑迁移到多模态服务。
它不是新的生产入口，也不会被当前后端导入。

## 文件与来源

| 文件 | 作用 | 原始实现 |
| --- | --- | --- |
| `contracts.py` | 请求、结果、音频 sink、provider 接口 | `speech_synthesis.py` |
| `realtime_ws_tts.py` | 自部署 WebSocket 协议、连接复用、取消、PCM delta | `realtime_ws_tts_adapter.py` |
| `streaming_bridge.py` | 多模态文本 delta 有序传递、可选切块、PCM 暂存 | `streaming_speech.py` |

移植版删除本项目配置全局变量和 JSONL 日志依赖，以可选 `trace_sink` 代替；
协议、生命周期、取消和错误语义均保留。

## 输入输出

- 输入：按生成顺序调用 `StreamingSpeech.push_text(delta)`。
- 请求：发送多条 `input_text_buffer.append`，最后发送一次
  `input_text_buffer.commit`。
- 返回：消费多条 `response.audio.delta`；`delta` 为 base64 编码的 24 kHz、
  单声道、PCM16LE，并以 `response.done` 结束。
- 输出：解码后的 PCM 立即交给 `audio_sink(bytes)`。多模态服务应把它转换为
  自己的 direct-audio 返回事件。

## 必须保留的规则

1. 每个多模态会话创建一个 synthesizer，不要每个 turn 新建。
2. 一个实例串行执行 turn，FIFO iterator 保证 append 有序，无需额外排序协议。
3. 同一音色复用连接；音色变化时重建。
4. 用户挂断时先尽力发送 `response.cancel`，再丢弃连接并传播取消。
5. 会话结束调用 `close()`；错误、超时、协议异常后不复用旧连接。
6. `response.audio.done` 仅是提示，完整 turn 仍以 `response.done` 为准。

## 接线示例

```python
synthesizer = RealtimeWsSpeechSynthesizer(
    RealtimeWsTtsConfig("ws://127.0.0.1:40001/api-ws/v1/realtime", 30)
)
speech = StreamingSpeech(
    request=SpeechSynthesisRequest(turn_id, "", voice, "zh-CN"),
    synthesizer=synthesizer,
)

await speech.push_text(text_delta)  # 每个多模态正文 delta
await speech.deliver(
    SpeechSynthesisRequest(turn_id, final_text, voice, "zh-CN"),
    audio_sink=send_direct_audio_delta,
)

# 挂断
await speech.cancel()
await synthesizer.close()
```

`send_direct_audio_delta` 是目标服务需要适配的输出边界，参数已经是原始 PCM。

## 首包监控

`SynthesisTrace` 保留 `adapter_started`、`connection_ready`、
`request_started`、`input_committed`、`first_chunk_received`、`completed`。

```python
trace.elapsed_ms("request_started", "first_chunk_received")
```

## 有意排除

- `reply_delivery.py`：本后端的编排、重试和数字人投递。
- `digital_human_audio_adapter.py`：浏览器/数字人输出，不属于 TTS。
- `dashscope_tts_adapter.py`：百炼 SDK 与本项目监控，不是目标自部署协议。
- `local_cosyvoice_tts_adapter.py`：旧 HTTP 完整文本路径。

依赖为 Python 3.11+ 和支持 `websockets.asyncio.client` 的 `websockets`。
