---
name: analyze-qwen3-omni-action-logs
description: Analyze Qwen3-Omni realtime action-scoring logs in this repository. Use when asked for successful action requests, category/child latency, prefix-cache behavior, session prompt tokens, audio/image participation, token composition, or whether a session approaches the configured 20000-token limit.
---

# Analyze Qwen3-Omni Action Logs

Use the bundled parser to inspect `logs/qwen3_omni_local.log` and
`logs/session_action_debug.jsonl` without changing service state. The default
report is action-only: it excludes normal assistant reply latency from the
action timing summary.

## Workflow

1. Resolve the session. Use `--session-id` when the user names one; otherwise
   choose the session with the latest `turn.commit input` record.
2. Parse server records for committed turns and `[SESSION_ACTION_REALTIME]
   action completed` records.
3. Parse debug JSONL stage records. A nested catalog has `category` and
   `child` stages; a flat catalog has `single`.
4. Treat a turn as action-successful when the server has an action-completed
   record and every expected scoring stage has a completed record. Do not
   require `reply completed` unless the user explicitly asks for reply timing.
5. Report per turn and aggregate category/child/single latency, action total
   latency, selected action, prompt tokens, candidate counts, prefix cache
   status, and current/history audio and image counts when present.
6. Compare the maximum single-stage `prompt_tokens` with the configured
   `max_seq_len`. Never compare the sum of tokens processed across turns with
   the context limit.

Run from the repository root:

```bash
python .claude/skills/analyze-qwen3-omni-action-logs/scripts/analyze_action_logs.py
```

Useful options:

```bash
python .claude/skills/analyze-qwen3-omni-action-logs/scripts/analyze_action_logs.py \
  --session-id session_profile_... --json

PATH=/home/ubuntu/miniconda3/envs/sglang-omni-local/bin:$PATH \
python .claude/skills/analyze-qwen3-omni-action-logs/scripts/analyze_action_logs.py \
  --token-breakdown --model-path /data/models/Qwen3-Omni-30B-A3B-FP8
```

## Interpretation rules

- `action_total_ms` is the realtime action-completed duration. For nested
  selection it should be close to `category_ms + child_ms` plus scheduling
  overhead.
- `server_total_after_commit_ms` is a response field, not normally present in
  the service console log. Do not invent it from reply/action logs; report
  `reply_ms + action_total_ms` only as an estimate when needed.
- `prompt_tokens` is a single-stage input length. The cumulative sum of
  category and child prompt tokens across a session is workload, not one
  context length.
- The action context keeps at most four recent history turns. The session's
  `history_turn_count` in commit logs can exceed the history actually present
  in the action prompt.
- In an audio-only session, the current utterance is not visible as text in
  `full_prompt`; it appears as audio media. Report images as zero only when
  both current and bounded history image counts are zero.
- Without a model tokenizer, report `media plus message-structure residual`
  as a combined value. Do not call that residual pure audio or pure image
  tokens. With `--token-breakdown`, explain that media expansion is inferred
  from the logged prompt count minus text-token counts.
- A successful action request can still have a semantically wrong top action.
  Preserve raw candidate scores and distinguish link success from selection
  quality.

## Output expectations

Lead with the session id, successful turn count, maximum single-stage prompt
tokens, and whether it is below `max_seq_len`. Then show a compact per-turn
table and aggregate min/average/max timings. If token composition is requested,
show category and child stages separately and state the audio/image evidence
and approximation limits.

Read [references/metrics.md](references/metrics.md) when the request involves
token composition, context limits, or reconciling service logs with
`turn.result` timing fields.
