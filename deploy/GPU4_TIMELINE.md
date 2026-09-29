# GPU4 timeline service

GPU4 runs one standalone Qwen3-Omni timeline pipeline on port 18008. Its
continuous audio/video windows are intentionally isolated from classifier,
Brain, action and reply inference.

| Capability | Endpoint | Scheduling |
| --- | --- | --- |
| timeline observer | `WS /v1/timeline` | video-driven sliding-window inference every 3 seconds |

Timeline contract v2 treats video as the window clock. A complete synchronized
audio span is attached when available; after a 350 ms grace period, missing,
gapped, or lagging audio produces a pure-image request instead of blocking the
window. The service never synthesizes silence for missing audio.

Install without starting:

1. Copy `config_gpu4_timeline.example.yaml` to `config_gpu4_timeline.yaml`.
2. Copy `gpu4-timeline.env.example` to `gpu4-timeline.env`, replace the token,
   and run `chmod 600`.
3. Generate and configure the immutable timeline model release manifest.
4. Install `sglang-omni-gpu4-timeline.service` as a systemd unit.
5. Verify physical GPU4 is free and port 18008 is unused before starting.
