# Qwen3-Omni action-log metrics

## Sources and success criteria

- `logs/qwen3_omni_local.log` contains commit payload summaries and realtime
  action completion lines.
- `logs/session_action_debug.jsonl` contains `action_scoring_prompt_rendered`
  and `action_scoring_completed` records for each stage.
- Nested action selection uses one logical turn with two internal stages:
  `category` followed by `child`. It is not two sessions.
- For an action-only report, require action completion plus all expected stage
  completions. A normal `reply completed` record is not required.

## Timing semantics

`category_ms` and `child_ms` come from debug completion records. The
realtime `action completed elapsed_ms` is the action-service total and includes
stage scheduling overhead. It is the primary action latency metric.

The WebSocket response has three timing fields:

- `server_turn_ingest_ms`: `turn.start` to `turn.commit` receipt;
- `server_action_compute_ms`: action scoring time, including both nested
  stages;
- `server_total_after_commit_ms`: commit receipt to result creation, including
  reply generation and action scoring.

The current service console does not log the complete outbound `turn.result`,
so `server_total_after_commit_ms` must be read from the client-received frame.

## Token semantics

The configured limit is `stage_overrides.preprocessing.runtime.max_seq_len` and
`stage_overrides.thinker.runtime.max_seq_len`, currently 20000 in
`deploy/config_single_gpu.yaml`. Compare this limit with the maximum single
stage `prompt_tokens` and any completion budget if one is used. Do not compare
it with the sum of prompt tokens over all turns.

The action context keeps `MAX_ACTION_HISTORY_TURNS = 4`. A session may report
many historical turns while the actual action prompt contains only the last
four groups plus the current turn.

For content composition, the tokenizer-based report separates:

1. system candidate instructions;
2. historical assistant text;
3. current action-selection instruction;
4. a residual containing media expansion plus chat/message structure.

The residual is deliberately not labeled as pure audio or pure image. The
rendered prompt shows media placeholders, while actual media expansion is
handled by the multimodal preprocessor. Current logs expose media counts and
hashes, not a per-media token breakdown.

## Recommended report language

Say “the request succeeded through the action pipeline” separately from “the
top action is semantically correct.” Preserve full raw scores when the latter
is questionable. For audio-only input, say that the user's utterance is in
audio and is not visible as text in `full_prompt`; do not claim that the audio
was transcribed unless an ASR field or external transcript is present.
