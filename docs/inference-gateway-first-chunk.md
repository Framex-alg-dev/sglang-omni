# Configurable first TTS audio packet

`session.open` v2/v3 accepts optional `speech_first_chunk_ms`: an integer from 40 to 1000. It is fixed for the lifetime of that gateway session and applies to ordinary, whole-text speculative, adaptive, and incremental speculative synthesis. `session.ready` echoes the value when configured so a caller can detect an old gateway. Omit the field to preserve existing behavior, including 250 ms framing and no model hop override.

`speech.request` may override the session value with `first_chunk_ms` for one ordinary request. Explicit `null` uses legacy behavior for that request. This does not mutate the session default or another session. Speculative speech uses the frozen session value. Its hash contains an additional `first_chunk_ms` field only when configured; legacy hashes are byte-for-byte unchanged. A commit with a mismatching target is rejected.

For HTTP TTS, the gateway forwards the same integer as `first_chunk_ms` and requires `X-TTS-First-Chunk-Ms` to acknowledge it. The first emitted PCM16 mono packet is `48 * first_chunk_ms` bytes at 24 kHz. Thus 334 ms is 16032 bytes. Later packets retain the configured ordinary frame size (12000 bytes = 250 ms by default). Remainders carry forward without truncation or zero padding; EOS can flush a short final packet. Adaptive multi-segment replies share one response framer: a short sentence does not flush a short first packet. The model first-hop override remains active until enough PCM has accumulated to emit the first response packet, then later segments use their ordinary model hop. Only the overall response EOS flushes a short remainder. Pending first-packet PCM counts against both response and session audio budgets, remains accounted after a commit, and is released on cancellation or failure.

For incremental speech through WS TTS, the embedded client updates `session.first_chunk_ms` and verifies `session.updated` before sending text. Reusing a connection for an unconfigured request explicitly resets a previous value to `null`. Fresh default connections send no additional update. Configuration follows the connection when a warm standby is adopted and resets when that connection is discarded.

This setting is a requested audio duration. The model may round its native generation hop upward; the gateway frames that native output without fabricating audio. It is separate from the digital-human Worker's window and credit policy.

## Deployment status

This is an isolated source patch, not a deployed feature. Deploy the matching inference gateway (18003), WS speech gateway (40001), and model TTS server (50001) together before enabling a client target. Explicit configuration requires acknowledgements on both gateway and model boundaries; an old provider is deliberately rejected rather than silently retaining its old first-hop policy. No live configuration or service restart is included.
