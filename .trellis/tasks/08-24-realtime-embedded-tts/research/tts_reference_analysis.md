# 后端 TTS 参考模块迁移分析

## 来源与用途

参考代码位于 `ref/tts_migration_reference/`：

- `contracts.py`：provider-neutral 请求、结果、audio sink 和 synthesizer Protocol；
- `realtime_ws_tts.py`：自部署 TTS WebSocket 协议、连接复用、取消与 PCM 解码；
- `streaming_bridge.py`：增量文本队列、PCM 暂存、deliver/cancel；
- `README.md`：原后端链路的迁移说明。

这些文件是行为参考，不是可直接复制的生产入口。

## 必须保留的合同

1. 每个外部 `MultimodalSession` 创建一个 synthesizer/连接管理器。
2. 第一个 audio Turn 延迟连接 provider；同 Session、同 voice、连接健康时跨 Turn 复用。
3. connection lock 覆盖完整 synthesis，一个连接只有一个 reader 和一个活动 Turn。
4. 输入按生成顺序发送多条 `input_text_buffer.append`，正文结束后只发送一次
   `input_text_buffer.commit`。
5. provider PCM 来自 `response.audio.delta.delta` 的 Base64；完整 provider Turn 以
   `response.done` 结束，`response.audio.done` 只作提示。
6. 取消时尽力发送 `{"type":"response.cancel"}`；错误、超时、协议异常后丢弃连接。
7. Session 结束调用 `close()`；正常 Turn 完成保留健康连接。

## 目标项目必须修正的差异

- 参考实现要求 Python 3.11+ 并使用 `asyncio.timeout`；本项目支持 Python 3.10，需用
  兼容 timeout 机制并保持 `CancelledError` 传播。
- 参考实现的 text/audio queue 无界；目标实现必须配置上限并产生背压，provisional PCM
  还需最大字节数或时长限制。
- 参考 `_session_url()` 使用 `request.turn_id` 填 provider `session_id`；目标实现必须
  使用外部 Session ID，另外维护 turn/response/provider response ID 映射。
- 参考实现取消后直接丢弃连接，这一点首版保留。仅 send cancel 成功不能证明旧 Turn
  的迟到事件已排空，贸然复用会导致下一 Turn 串音频。
- 参考实现的 PCM buffer 用于“最终文本确认后交付”；目标实现还要支持普通
  text+audio 的实时外发，以及 text+audio+action 的 provisional 隔离缓冲。
- provider 错误细节不能把完整事件、正文、PCM、URL query 或凭证写入日志/外部错误。

## 连接所有权结论

```text
MultimodalSession
  owns EmbeddedTTSConnectionManager
    owns at most one provider WebSocket
      borrowed by one EmbeddedTTSTurn at a time
```

不能由 Turn 拥有连接，否则无法跨 Turn 复用；不能由 app 全局拥有连接，否则不同外部
Session 会互相阻塞、串 voice/response，并在取消时误伤其他 Session。

## 必需验证

- 同一 Session 两个顺序 audio Turn、同 voice：provider connect 恰好一次；
- 不同 Session：各自 connect，连接对象不同；
- voice 改变：关闭旧连接再 connect；
- 正常 `response.done`：连接回到 Ready 并可复用；
- cancel、timeout、非法 JSON/Base64、重复 done、audio-after-done、提前关闭：连接进入
  Broken 并关闭，下一 Turn 新建连接；
- `session.close`、外部断线、重复 teardown：close 幂等且无 reader/sender 孤儿任务；
- 不请求 audio：不创建 manager 的网络连接、不导入或调用 provider。

