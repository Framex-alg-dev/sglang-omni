# 多模态回复服务内化自部署 TTS：开发交接说明

## 1. 目标与边界

当前链路是：

```text
多模态模型生成文本 Delta
  -> 本项目后端接收文本
  -> 本项目后端调用独立 TTS
  -> 本项目后端接收 PCM 音频
  -> 浏览器 / 数字人
```

目标链路是：

```text
多模态模型生成文本 Delta
  -> 多模态服务内部立即流式调用 TTS
  -> 多模态服务对外同时返回文本事件和音频事件
  -> 本项目后端直接消费音频
  -> 浏览器 / 数字人
```

本次改造应发生在部署多模态模型的 Gateway/Realtime 服务中。本项目后端不再发起第二次
TTS 请求。多模态服务对外协议应继续保持现有 Qwen Realtime 兼容事件，尽量不让浏览器、
数字人和会话控制代码感知内部变化。

本文只描述当前项目已经验证过的自部署 TTS 调用方式，以及多模态服务对外必须满足的兼容
约束。TTS 服务端当前实现位于第五台服务器：

```text
/data/hanning/model/qwen3_omni_inference/gateway/realtime_tts.py
```

## 2. 当前可用 TTS 接口

### 2.1 地址

```text
公网：ws://124.221.190.139:40001/api-ws/v1/realtime
内网：ws://10.0.0.14:40001/api-ws/v1/realtime
```

如果多模态服务和 TTS 位于同一内网，应使用内网地址，避免公网绕行。推荐建连时在 query
中固定音色和会话 ID：

```text
ws://10.0.0.14:40001/api-ws/v1/realtime
  ?voice=benchmark_qwen_cherry_zh
  &session_id=<multimodal-session-id>
```

常用音色：

- `benchmark_qwen_cherry_zh`
- `benchmark_cosyvoice_female_zh`
- `benchmark_cosyvoice_male_zh`
- `prep_male_short4435_rerun`

当前入口没有鉴权和 TLS。正式跨机器使用时至少应通过内网和防火墙限制来源；暴露到公网前
应增加鉴权并升级为 `wss://`。

### 2.2 建连与输入事件

建连成功后，TTS 首先返回：

```json
{
  "type": "session.created",
  "session": {
    "id": "multimodal-session-id",
    "voice": "benchmark_qwen_cherry_zh"
  }
}
```

多模态模型每产生一段可见正文，就立即原样追加给 TTS：

```json
{
  "type": "input_text_buffer.append",
  "text": "你好，"
}
```

可以连续发送多个 append：

```json
{
  "type": "input_text_buffer.append",
  "text": "这是实时语音合成。"
}
```

本轮文本彻底结束后必须提交：

```json
{
  "type": "input_text_buffer.commit"
}
```

`commit` 不能省略。服务虽然会在遇到 `。！？!?；;`、换行或累计约 60 字时提前分句并
开始合成，但没有标点的尾部文本只能通过 commit 刷出。

不要等待完整回复再一次性 append。应在多模态模型生成正文 Delta 时同时执行：

1. 向外部客户端发送正文 transcript Delta；
2. 将同一段纯正文 append 给内部 TTS；
3. 并发接收并立即转发 TTS 音频 Delta。

工具参数、JSON 控制结构、思维链、动作标记和其他非朗读内容不能送入 TTS。只有最终用户
可见、应该被朗读的正文才能 append。

### 2.3 输出事件

TTS 正常输出顺序：

```text
response.created
response.audio.delta  # 多条
response.audio.done
response.done
```

音频块事件：

```json
{
  "type": "response.audio.delta",
  "response_id": "resp_xxx",
  "delta": "<base64 PCM>"
}
```

`delta` 解码后的固定格式：

| 属性 | 值 |
|---|---|
| 采样率 | 24000 Hz |
| 声道 | 单声道 |
| 编码 | PCM16LE / signed 16-bit little-endian |
| 常见块大小 | 12000 bytes |
| 常见块音频时长 | 250 ms |

计算关系：

```text
audio_duration_ms = pcm_bytes / (24000 * 1 * 2) * 1000
```

多模态 Gateway 应原样转发 base64 PCM，不能包装 WAV header、转成 MP3、改变采样率或把
多个块攒到响应结束后再发送。

### 2.4 取消与错误

用户打断、挂断或上游 turn 被取消时发送：

```json
{
  "type": "response.cancel"
}
```

收到 TTS `error` 事件时，本轮必须终止，记录错误阶段并通知外部客户端。连接状态不可信时
应关闭并重建，不能把失败连接直接放回连接池。

## 3. 连接复用规则

当前本项目的已验证实现是“一条多模态会话持有一条 TTS WebSocket”：

- 首轮建立连接并等待 `session.created`；
- 后续 turn 在音色不变时复用同一连接；
- 同一连接上的多个 turn 串行执行；
- 必须等上一轮 `response.done` 后才能在同一连接提交下一轮；
- 音色变化时重建连接，或者在业务验证后使用 `session.update`；
- 会话挂断时关闭连接；
- 超时、协议错误、非法 base64 或无音频完成时丢弃连接；
- 不要让多个协程并发读同一个 WebSocket。

如果多模态服务允许同一业务 session 内并行生成多个 response，应给并行 response 分配独立
TTS 连接，或者在 Gateway 中显式排队。当前 TTS 协议没有客户端提供的 turn sequence 字段，
不能依赖收到后再排序来解决并发串流问题。

## 4. 多模态服务内部推荐状态机

每个外部 Realtime session 建议维护：

```text
Disconnected
  -> Connecting
  -> Ready
  -> Synthesizing(turn A)
  -> Ready
  -> Synthesizing(turn B)
  -> Ready
  -> Closed
```

单轮推荐流程：

```text
1. 创建外部 response_id
2. 对外发送 response.created
3. 多模态模型开始生成正文
4. 每个正文 Delta：
   a. 对外发送 response.audio_transcript.delta
   b. 对内发送 input_text_buffer.append
5. 内部 TTS 一旦返回 response.audio.delta：
   a. 校验 base64
   b. 对外立即发送 response.audio.delta
6. 正文生成结束：
   a. 对外发送 response.audio_transcript.done
   b. 对内发送 input_text_buffer.commit
7. 等内部 TTS response.done
8. 对外发送 response.audio.done
9. 对外发送 response.done
```

发送文本和接收音频必须并发进行。如果先等待模型生成完全文，再开始读取音频，会重新引入
等待全文和缓冲区拥塞，失去本次内化 TTS 的主要收益。

外部 `response_id` 应由多模态服务自己管理。内部 TTS 的 `response_id` 只用于关联和日志，
不要直接覆盖外部多模态 response ID。

## 5. 本项目后端要求的外部事件

多模态服务内化 TTS 后，本项目后端应配置：

```dotenv
QWEN_REALTIME_PROVIDER=internal
QWEN_CHARACTER_DIRECT_AUDIO=true
```

`QWEN_CHARACTER_DIRECT_AUDIO=true` 是必要条件。否则角色链路会忽略多模态服务返回的
`response.audio.delta`，并可能再次调用独立 TTS，造成重复合成或无声。

多模态服务至少应保持以下事件：

### 5.1 回复文本

```json
{
  "type": "response.audio_transcript.delta",
  "response_id": "resp_outer_xxx",
  "delta": "你好，"
}
```

```json
{
  "type": "response.audio_transcript.done",
  "response_id": "resp_outer_xxx",
  "transcript": "你好，这是完整回复。"
}
```

最终 `transcript` 必须等于所有 delta 拼接后的用户可见正文。后端依赖它完成字幕、最终回复
校验和角色交付。

### 5.2 回复音频

```json
{
  "type": "response.audio.delta",
  "response_id": "resp_outer_xxx",
  "delta": "<base64 PCM16LE 24kHz mono>"
}
```

```json
{
  "type": "response.audio.done",
  "response_id": "resp_outer_xxx"
}
```

本项目后端收到音频 Delta 后会：

- 原样转发给浏览器；
- base64 解码后送入数字人；
- 按 24 kHz PCM16 计算音频时长；
- 在 `response.audio.done` 后执行本轮收尾。

因此，音频格式和 done 事件顺序属于接口契约，不能随意改变。

## 6. 最小 Python 接入骨架

以下代码展示内部 TTS client 的关键行为。生产实现还需要接入项目自身的任务所有权、日志、
超时和外部事件发送器。

```python
import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from urllib.parse import urlencode

from websockets.asyncio.client import connect


SendOuter = Callable[[dict], Awaitable[None]]


class EmbeddedTtsSession:
    def __init__(self, base_url: str, voice: str, session_id: str) -> None:
        query = urlencode({"voice": voice, "session_id": session_id})
        self.url = f"{base_url}?{query}"
        self.ws = None
        self.lock = asyncio.Lock()

    async def connect(self) -> None:
        self.ws = await connect(self.url, max_size=None)
        created = json.loads(await self.ws.recv())
        if created.get("type") != "session.created":
            await self.close()
            raise RuntimeError(f"expected session.created, got {created!r}")

    async def synthesize(
        self,
        text_deltas: AsyncIterator[str],
        *,
        outer_response_id: str,
        send_outer: SendOuter,
    ) -> None:
        async with self.lock:
            if self.ws is None:
                await self.connect()

            async def send_text() -> str:
                parts = []
                async for delta in text_deltas:
                    if not delta:
                        continue
                    parts.append(delta)
                    await send_outer({
                        "type": "response.audio_transcript.delta",
                        "response_id": outer_response_id,
                        "delta": delta,
                    })
                    await self.ws.send(json.dumps({
                        "type": "input_text_buffer.append",
                        "text": delta,
                    }, ensure_ascii=False))
                transcript = "".join(parts)
                await send_outer({
                    "type": "response.audio_transcript.done",
                    "response_id": outer_response_id,
                    "transcript": transcript,
                })
                await self.ws.send(json.dumps({
                    "type": "input_text_buffer.commit"
                }))
                return transcript

            async def receive_audio() -> None:
                while True:
                    event = json.loads(await self.ws.recv())
                    event_type = event.get("type")
                    if event_type == "response.audio.delta":
                        encoded = event.get("delta")
                        if not isinstance(encoded, str):
                            raise RuntimeError("TTS audio delta is not a string")
                        base64.b64decode(encoded, validate=True)
                        await send_outer({
                            "type": "response.audio.delta",
                            "response_id": outer_response_id,
                            "delta": encoded,
                        })
                    elif event_type == "error":
                        raise RuntimeError(f"TTS failed: {event!r}")
                    elif event_type == "response.done":
                        return

            sender = asyncio.create_task(send_text())
            receiver = asyncio.create_task(receive_audio())
            try:
                await asyncio.gather(sender, receiver)
            except BaseException:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                await self.close()
                raise

            await send_outer({
                "type": "response.audio.done",
                "response_id": outer_response_id,
            })

    async def cancel(self) -> None:
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "response.cancel"}))

    async def close(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            await ws.close()
```

注意：示例故意没有发送外层 `response.created` 和 `response.done`，因为它们应由多模态
Gateway 的既有 response 生命周期统一管理，而不是由 TTS client 擅自创建。

## 7. 超时、背压与故障策略

建议至少设置：

- 建连超时；
- 等待 `session.created` 超时；
- 首音频块超时；
- 单轮总超时；
- 外部 WebSocket 写入超时；
- 文本生产速度高于 TTS 消费速度时的有界队列。

不要使用无限队列缓存模型文本或音频。如果外部客户端持续无法消费音频，应取消本轮 TTS，
而不是无限积压 PCM。

错误恢复建议：

| 故障 | 处理 |
|---|---|
| 建连失败 | 本轮失败；有限次数退避重试 |
| 首事件不是 `session.created` | 关闭连接；协议错误 |
| TTS 返回 `error` | 关闭连接；向外返回 provider error |
| 非法 base64 | 关闭连接；不得继续复用 |
| `response.done` 前没有音频 | 按合成失败处理 |
| 用户打断 | 发送 `response.cancel`，停止转发旧 turn 音频 |
| 外部会话挂断 | 取消模型生成、TTS sender/receiver，并关闭 TTS 连接 |

## 8. 必须记录的监控时间戳

为了验证内化确实降低延迟，建议每个 turn 记录单调时钟：

```text
multimodal_request_started
first_text_delta_generated
tts_connection_ready
first_text_chunk_sent_to_tts
text_input_committed
first_tts_audio_chunk_received
first_250ms_audio_received
last_tts_audio_chunk_received
response_done
```

并记录：

- 外部 session ID、turn ID、外部 response ID、内部 TTS response ID；
- 每个输入文本块的字符数和发送时刻；
- 每个音频块的字节数和到达时刻；
- 是否复用连接、音色、TTS 服务地址（不要记录凭证）；
- 取消、超时、错误阶段。

核心指标：

```text
模型请求到首字 = first_text_delta_generated - multimodal_request_started
首字到 TTS 首音频 = first_tts_audio_chunk_received - first_text_delta_generated
总体首音频 = first_tts_audio_chunk_received - multimodal_request_started
可播放 250ms = first_250ms_audio_received - multimodal_request_started
相邻音频包间隔 = audio_chunk[i].arrived - audio_chunk[i-1].arrived
```

必须同时保留“首字到 TTS 首音频”和“总体首音频”，否则无法区分多模态模型变慢与 TTS
变慢。

## 9. 联调验收清单

### 协议正确性

- 建连后首先收到 TTS `session.created`；
- 文本 Delta 顺序与多模态模型生成顺序一致；
- 只向 TTS 发送可朗读正文；
- 每轮必发一次 commit；
- 音频是 24 kHz、mono、PCM16LE；
- 外部 transcript 拼接结果与最终正文一致；
- 外部音频 Delta 使用当前多模态 response ID；
- `response.audio.done` 在最后一个音频块之后；
- 外部 `response.done` 在文本和音频均结束后。

### 性能与生命周期

- 第一轮建立 TTS WebSocket；
- 第二轮复用同一连接，建连耗时应接近 0；
- 文本未生成完时即可收到首个音频块；
- 用户打断后不再转发旧 turn 音频；
- 会话挂断后没有遗留生成任务和 WebSocket；
- TTS 故障不会导致下一轮误收上一轮音频。

### 本项目后端联调

- 使用 `QWEN_CHARACTER_DIRECT_AUDIO=true`；
- 浏览器能收到文字 Delta 和最终文字；
- 浏览器能播放音频；
- 数字人能收到并消费相同 PCM；
- 没有触发独立角色 TTS 日志；
- 首音频、250 ms 可播放和相邻包间隔日志完整。

## 10. 当前实现参考

- TTS 协议与已验证客户端：
  [`backend/character/conversation/delivery/realtime_ws_tts_adapter.py`](../../../../backend/character/conversation/delivery/realtime_ws_tts_adapter.py)
- 后端直接消费多模态音频：
  [`backend/qwen_realtime_client.py`](../../../../backend/qwen_realtime_client.py)
- 自部署多模态配置模板：
  [`.env.example.d/10-qwen.internal.env`](../../../../.env.example.d/10-qwen.internal.env)
- 原始 TTS 接入说明：
  [`ref/接入文档.md`](../../../../ref/接入文档.md)

