# Qwen3-Omni Action Suffix Scoring

`POST /v1/action-scores` implements candidate continuation scoring for
`Qwen3-Omni-30B-A3B-Instruct`. It is a prefill-only path: one logical request
preprocesses the multimodal prefix once, runs the Thinker once, then evaluates
all candidate suffixes in micro-batches. Talker, Code2Wav, and ordinary decode
are not entered.

## Request contract

The request contains a non-empty text `prefix`, optional `audios`/`images`, a
`language` (`zh` or `en`), and 1–512 ordered candidates. A candidate has a
unique `candidate_id`, a suffix of at most 512 characters, and optional
`action_id`/`execution_binding`. Suffixes ending in terminal punctuation are
rejected because punctuation is excluded from the action score denominator.

`sample_rate` must be positive. `micro_batch_size` defaults to 64 and is
limited to 256. The realtime session route uses the startup environment
variable `SGLANG_OMNI_ACTION_MICRO_BATCH_SIZE` to set this value for both
flat_children and hierarchical action stages; an omitted variable keeps 64.
The service serializes scoring requests with a global
concurrency limit of one and applies a 120 second end-to-end timeout. A timeout or
client cancellation aborts the logical request and all physical candidate
requests.

## Session and multi-turn context

Each turn may include `session_id`, the ordered `history` messages, `history_audios`, `history_images`, and an `avatar_state` object. The caller sends the complete context on every request; `session_id` is a correlation key and does not store conversation state inside the server. Historical audio/image entries must have explicit structured parts in `history`, for example `{"type":"audio"}` or `{"type":"image"}`, with matching entries in `history_audios`/`history_images`. The current turn remains in `prefix`, `audios`, and `images`. The service combines historical and current media in message order, so the action decision can use prior audio, text, images, and the current digital-human state.

The Realtime Session adapter also attaches internal `turn_origin`, `text_role`,
and optional `trigger` metadata to scoring requests. The only valid pairs are
`user/user_input` and `proactive/character_reply`. User-turn content is treated
as user input. For a proactive turn, the already-generated character reply is
inserted as the latest assistant message before the internal candidate-selection
instruction is appended. Both paths use the complete bounded history and
`avatar_state`, score only the supplied suffixes, and never enter ordinary reply
generation. These fields are Realtime Session metadata and are not currently
part of the public `POST /v1/action-scores` request schema.

Example shape:

```json
{
  "session_id": "session-42",
  "history": [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "我刚才抬起左手"}]},
    {"role": "assistant", "content": "我看到你抬起了左手。"}
  ],
  "history_audios": ["/data/session/turn-1.wav"],
  "avatar_state": {"pose": "seated", "gaze": "camera", "left_hand": "raised"},
  "prefix": "请基于当前对话判断下一步动作：",
  "audios": ["/data/session/turn-2.wav"],
  "images": []
}
```

## Scoring semantics

The Thinker request obtains the shared prefix state and the log probability for
the first token of every candidate. Candidate requests reuse the same
multimodal cache identity and score input-token log probabilities from
`logprob_start_len = prefix_token_count`. Tokenization uses offsets when
available, preserves a token that crosses the text boundary, and excludes
special tokens. The executed candidate sequence keeps the explicit ChatML
assistant-turn terminator, but aggregation removes that trailing terminator so
the returned score measures only the supplied candidate suffix.

For candidate tokens with log probabilities `l_1 ... l_n`, the response returns
`mean_logprob = sum(l_i) / n`, `mean_nll = -mean_logprob`, and
`ppl = exp(mean_nll)`, together with each token ID and log probability. Results
are emitted in the original candidate order. Here `n`, `token_count`, and
`token_scores` cover candidate suffix tokens only and exclude the trailing
ChatML terminator. A result is accepted only when the shared prefix was verified
in the KV cache; otherwise the request fails with `prefix_cached=false`.

### Flat-children latency path

Scheme B uses short identifier suffixes such as `A328` and sets
`action_scoring.suffix_tokenization_mode` to `short_id`. The service encodes each short
suffix independently and appends it to the already authoritative prefix token IDs. This
avoids re-tokenizing the long multimodal prompt once per candidate. Descriptive suffixes
must keep the default `exact` mode because a BPE token can cross the prefix/suffix text
boundary.

The scheduler also constructs candidate `Req` objects lazily. It first admits one shared
prefix request, waits for its prefix logprob/KV state, and then creates only the next
micro-batch of candidate requests. After that batch completes, its objects are released
and the next batch is built. Therefore a catalog with 330 candidates still uses batches
of `64, 64, 64, 64, 64, 10`, but it does not allocate 330 full prefix-plus-suffix
requests before the shared prefix enters the scheduler. This changes CPU-side request
construction and queueing, not the scoring formula or the prompt semantics.

## Cache and media safety

The cache identity hashes prefix token IDs, request scope, and the contents of
audio/image files (or their string values). File paths alone are never used as
media identity. This prevents stale multimodal embeddings from crossing
requests that happen to reuse a filename.

## Internal route

```text
preprocessing -> image/audio encoder(s) -> mm_aggregate -> thinker
                                                        |
                       action_suffix_scoring ---------+-> action_score (terminal)

ordinary text: ... -> thinker -> decode
ordinary speech: ... -> thinker -> decode/talker -> code2wav
```

The terminal stage is an identity stage: the scheduler attaches the score
result to the Thinker payload, and the stage returns it without invoking a
second model.

## Validation before starting

Run the pure contract tests and compile checks before launching the model:

```bash
python -m pytest -q \
  tests/unit_test/qwen3_omni/test_action_scoring.py
python -m py_compile \
  sglang_omni/models/qwen3_omni/action_scoring.py \
  sglang_omni/models/qwen3_omni/request_builders.py \
  sglang_omni/models/qwen3_omni/config.py \
  sglang_omni/scheduling/omni_scheduler.py
```

### Response example

```json
{
  "request_id": "turn-123",
  "model": "Qwen3-Omni-30B-A3B-Instruct",
  "prefix_cached": true,
  "timing": {
    "server_action_compute_ms": 842.317
  },
  "stats": {
    "logical_prefix_request_count": 1,
    "prefix_physical_prefill_chunk_count": 1,
    "cached_prefix_token_count": 128,
    "candidate_prefix_recompute_tokens": 0,
    "suffix_batch_count": 2
  },
  "scores": [
    {
      "candidate_id": "left",
      "token_count": 4,
      "mean_logprob": -0.42,
      "mean_nll": 0.42,
      "ppl": 1.52,
      "token_scores": [
        {"token_id": 123, "logprob": -0.31}
      ]
    }
  ]
}
```

The timing.server_action_compute_ms field is measured inside the
/v1/action-scores handler around the server-side action scoring call. It includes
server-side queueing, media preprocessing, prefix handling, and candidate
scoring, but does not include the client's upload/download time. The client can
measure the full HTTP round trip using client_round_trip_ms, from immediately
before sending the request until the response body is received. The difference
client_round_trip_ms - server_action_compute_ms is only an estimated
non-compute overhead: it may also include client JSON serialization,
server/request handling, and response serialization, so it should not be
interpreted as pure network latency.

`prefix_cached` is a correctness gate, not an advisory metric. A cache miss or partial match is returned as a failed request because recomputing the multimodal prefix would violate the latency and cost contract.

### SGLang logprob and KV lifecycle

The prefix physical request appends one dummy token, sets `max_new_tokens=0`, and asks for selected logprobs for the first token of each candidate. This makes the final shared-prefix hidden state available while allowing SGLang to cache only the original prefix. Candidate requests contain `prefix + suffix`, use the same cache key, and set `logprob_start_len` to the original prefix length. The scheduler keeps only the continuation positions from input-token logprobs; the first suffix token always comes from the shared-prefix distribution.

Do not replace this with generated-token logprobs or with a second multimodal prefill. The latter is both slower and unsafe when media placeholders have the same token sequence but different content.

The repository pins the SGLang dependency in `pyproject.toml`. Before a GPU smoke test, verify the installed version exposes `Req.return_logprob`, `Req.logprob_start_len`, selected input-token logprobs, and prefill-only requests. If those APIs are absent, update the SGLang dependency or its source fork; do not monkey-patch a running process.

### Observability and cancellation

Structured events include queue entry, suffix batch enqueue, first emit, and model-path end. The response `stats` object reports logical prefix count, physical prefill chunk count, cached prefix tokens, candidate prefix recomputation, suffix batch sizes, and stage timings when available. Logs and cache labels use a short digest; raw media paths, audio bytes, and image contents are not logged.

The client serializes MIS calls. Timeout and cancellation first cancel the coordinator task, then abort the logical request; the scheduler recursively removes waiting, running, and candidate requests. Internal candidate IDs are never sent through the external request-finished abort callback.

For `WS /v1/session/realtime`, `turn.commit` runs scoring in a background task
so the socket can still receive `turn.cancel`. The adapter tracks the currently
active flat, category, child, or first-use child-catalog-prefill request ID,
aborts that request, cancels the orchestration task, and prevents later stages,
`turn.result`, history updates, and `avatar_state` updates. A cancelled Turn is
complete only after `turn.cancelled` is received. Socket disconnect and
`session.close` perform the same cleanup without emitting that event.

### GPU smoke and benchmark checklist

The following commands are intentionally dry-run checks until the operator explicitly authorizes a launch:

```bash
# Check model/config/GPU prerequisites without starting a server
python -m py_compile sglang_omni/models/qwen3_omni/*.py
python -m pytest -q tests/unit_test/qwen3_omni/test_action_scoring.py

# After explicit launch approval, use the single-GPU profile as the baseline
# bash scripts/launch_server.sh --config deploy/config_single_gpu.yaml
```

For a GPU smoke test, send one multimodal prefix with two candidates and assert: one logical prefix request, `prefix_cached=true`, exact candidate order, non-empty token scores, and `candidate_prefix_recompute_tokens=0`. For the benchmark matrix use 1, 64, 128, 256, and 512 candidates, micro-batches 32/64/128, and audio-only/image-only/audio+image inputs. Record p50/p95 total latency, queue wait, encoder time, physical prefix chunks, suffix batch count, and peak GPU memory.

The service is intentionally not started by the repository preparation work.
