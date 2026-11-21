# 41035 sglang-omni code consolidation

This branch captures the code paths used by the 41035 Agent without changing
their external interfaces.

## Frozen baseline

- Agent commit: `e5858427ded8dd573e4b1e416c9a7f7a4fca3f0c`.
- sglang-omni base: `bebe4bc22e02461027a145b9ac9da0959161e427`.
- Reply model: `/data/models/Qwen3-Omni-30B-A3B-FP8` on the local Reply service.
- Action model: `/data/xingmt/action_omni_output/fp8_overlay_e67_v116_step695`
  on the local Action service.
- Event endpoint used by 41035: `ws://10.199.0.9:18118/v1/timeline`.
- Event identity: `event-1115-v1-fp8`, model version
  `1115_v1-checkpoint-6138-fp8`, protocol `direct-1115-v1`.

## Included code

1. The previously uncommitted Reply, Action, and inference-gateway runtime
   changes from `/data/fanshide/sglang-omni-dev`, including Action prefix
   prefill and Reply model-input diagnostics.
2. The E66/E67 frozen Action prompts required by the corresponding runtime
   prompt profiles. These were present in the active checkout but hidden by
   the repository-wide `*.txt` ignore rule.
3. The Event 1115/raw-window source snapshot copied on 2026-11-20, including
   the timeline service, frozen codebook validation, raw-window result path,
   video frame preservation, startup prefix prefill, bounded diagnostics, and
   timeline artifact recording.
4. `framex_event_policy` 0.1.1 source extracted from the exact wheel consumed
   by the 41035 Agent. The wheel SHA-256 is
   `3e48594da60fe35b12fd8a99bf0cbd05d3e472f53c47315db0e207a9a03308d8`.

## Deliberately excluded

- `codex/speech-action-windows` and its V10 Action Recall contract.
- The snapshot's unrelated FP8 scale inversion, image-cache placement, and
  task-classifier whitespace changes.
- Any Reply/Action/Event interface unification.
- Secrets, authentication tokens, model weights, recorded user media, and
  timeline artifact contents.

## Provenance limitation

The Reply and Action checkout has a known Git base. The Event 1115 deployment
snapshot does not expose an original Git commit, so this branch records the
snapshot itself as the source of truth. The observed isolated launch evidence
uses port 18118, a 500 ms inference interval, and the model/protocol identities
listed above. Its private token and deployment payload are intentionally not
copied.
