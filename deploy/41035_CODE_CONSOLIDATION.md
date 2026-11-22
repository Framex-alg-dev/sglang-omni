# 41035 sglang-omni code consolidation

This branch captures the code paths used by the 41035 Agent without changing
their external interfaces.

## Current baseline

- sglang-omni base: `bebe4bc22e02461027a145b9ac9da0959161e427`.
- Reply: the existing 41035 realtime Reply code path is retained; model and
  endpoint selection remain independent Agent deployment configuration.
- Action endpoint used by 41035: `ws://10.199.0.7:18049/v1/action-recall`.
- Action identity: `v10-boundary-ep1-final-fp8`, contract
  `action-decision.v1`, with separate `body` and `expression` channels.
- Event endpoint used by 41035: `ws://10.199.0.15:18118/v1/timeline`.
- Event identity: `event-1115-v1-fp8`, model version
  `1115_v1-checkpoint-6138-fp8`, protocol `direct-1115-v1`.

## Included code

1. The previously uncommitted Reply and inference-gateway runtime changes from
   `/data/fanshide/sglang-omni-dev`, including Reply model-input diagnostics.
2. The exact deployed Action V10 `action_decision` source, package resources,
   runtime catalog, token mappings, scenario resources, configuration, and a
   secret-free launch entrypoint. This includes the `body` and `expression`
   inference paths and role-specific 60s/4s/4s audio limits.
3. The E66/E67 frozen Action prompts retained for older prompt profiles.
4. The Event 1115/raw-window source snapshot copied on 2026-11-20, including
   the timeline service, frozen codebook validation, raw-window result path,
   video frame preservation, startup prefix prefill, bounded diagnostics, and
   timeline artifact recording.
5. `framex_event_policy` 0.1.1 source extracted from the exact wheel consumed
   by the 41035 Agent. The wheel SHA-256 is
   `3e48594da60fe35b12fd8a99bf0cbd05d3e472f53c47315db0e207a9a03308d8`.
6. Agent-owned prompt snapshot consumption for Action and Event. Action accepts
   the immutable snapshot on both `/v1/action-decision` and
   `/v1/action-recall`; Event accepts it in `session.start` and acknowledges
   the exact release and bundle hash in `session.ready`. Both consumers reject
   model/protocol binding or content-hash drift before inference.

## Prompt ownership

- Reply prompt assembly is already supplied by the Agent Reply snapshot path.
- Action policy text is supplied by the Agent. sglang-omni only appends
  request-scoped persona, channel and catalog data. Legacy callers without a
  snapshot retain the captured local prompt fallback during migration.
- Event system prompt, E0..E95 registry and CSV codebook are one Agent-owned
  snapshot. The service validates all three together and uses that snapshot for
  the session model profile; reconnects must carry the same snapshot.
- sglang-omni no longer downloads the Action prompt release independently at
  service startup. Release selection and pinning belong to the Agent process.

## Deliberately excluded

- The snapshot's unrelated FP8 scale inversion, image-cache placement, and
  task-classifier whitespace changes.
- Any Reply/Action/Event interface unification.
- Secrets, authentication tokens, model weights, recorded user media, and
  timeline artifact contents.

## Provenance limitation

The Reply checkout has a known Git base. The Action V10 and Event 1115
deployments do not expose one authoritative Git commit, so this branch records
their byte-verified runtime snapshots as the source of truth. The observed
Event launch uses port 18118, a 500 ms inference interval, and the
model/protocol identities listed above. Private tokens, model weights, recorded
media, and deployment payloads are intentionally not copied.
