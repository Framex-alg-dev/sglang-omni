# Deployment TODOs

## Session Realtime follow-ups

- Implement text/action fusion for `selection_mode=flat_children`; the first
  fused version intentionally requires hierarchical selection and category
  Top-K 1. Existing action-only flat selection remains supported.
- Define reply audio/TTS events, fixed-text TTS, and Code2Wav integration after
  the text/action protocol is stable.
- Add external action execution acknowledgement so the service can distinguish
  an inferred action from a successfully played action.
- Add retention, compression, and offline P50/P95/P99 aggregation for hourly
  structured realtime logs.

## Multi-session concurrency hardening

### Status

Deferred. The global action catalog and cross-session static-prefix prewarm are
implemented without changing the current concurrency model. Keep action
suffix scoring globally serialized until a dedicated multi-session capacity and
isolation test has been completed.

### Current behavior and risks

- Multiple WebSocket sessions may be active at the same time, but each session
  permits only one active turn.
- Reply requests may overlap in the shared model pipeline. Action category and
  child scoring across all sessions currently share one global
  `asyncio.Semaphore(1)` and therefore queue serially.
- There is no configured maximum active-session count, per-session request-rate
  limit, or explicit fair action-admission policy. One busy or slow session can
  increase queue wait and tail latency for other sessions.
- Action context uses a bounded recent-history view, but the session currently
  retains older turn objects and their audio/image payloads until disconnect.
  Long-running or abandoned sessions can therefore increase API-process memory.
- Shared static-prefix cache eviction can cause performance interference across
  sessions. It must never cause candidate, history, media, execution-binding,
  cancellation, or error-state leakage between sessions.

### Required follow-up

1. Add configurable limits for active sessions, per-session request rate, input
   buffering, and queued/in-flight work, with structured overload errors.
2. Prune or compact retained turn history and media once they fall outside the
   inference history window; clean up all state on disconnect, cancellation,
   timeout, and failure.
3. Measure concurrent-session P50/P95/P99 reply TTFT, reply completion, action
   slot wait, category/child compute, queue depth, GPU utilization, KV-cache
   occupancy/eviction, and API-process memory.
   - Use the periodic `resource` structured log as the common time series for
     active sessions/turns, action admission waiting/in-flight counts, API
     process CPU/RSS, host load/memory, log-disk capacity, and per-visible-GPU
     utilization, memory, power, temperature, clocks, and compute-process GPU
     memory.
   - Correlate resource samples with lifecycle/reply/action/performance events
     by timestamp and service instance. Keep the sampling interval configurable
     and record it in `resource_monitor_started`.
4. Verify fairness between sessions before making action-scoring concurrency
   configurable above one. Raising the semaphore alone is not an accepted
   solution.
5. Add isolation tests for disjoint session candidate whitelists, histories,
   media, execution bindings, errors, turn cancellation, disconnect cleanup,
   shared-prefix cache miss/rewarm, and duplicate session IDs.
6. Ensure shared prefix rewarm is single-flight and cannot be cancelled by the
   session that first observes a miss. Global catalog and prompt objects must
   remain immutable after startup.
7. If runtime catalog hot reload is introduced, build a new immutable catalog
   generation and atomically switch only new sessions after its prefixes are
   ready. Do not mutate prompts/hashes used by existing sessions in place.

### Non-goal for the global catalog prewarm change

Do not change action-scoring concurrency, introduce a new admission scheduler,
or claim a supported concurrent-session capacity as part of the global catalog
prewarm work. That change may share only immutable catalog prompts and their KV
prefixes; every session's whitelist, history, media, mappings, tasks, errors,
and cancellation state must remain isolated.

## Qwen3-Omni: continuity-safe Code2Wav streaming crossfade

### Status

Not implemented. Do not add the historical patch script to startup and do not
change the current Code2Wav output until the behavior below is implemented and
validated against the current scheduler.

### Problem

Qwen3-Omni Code2Wav decodes each streaming window with left context and then
trims the re-decoded context. The first sample of the new chunk is not
guaranteed to be continuous with the last sample of the previous chunk. This
can produce audible clicks or residual electrical noise at chunk boundaries,
especially with very small `stream_chunk_size` values.

This is separate from the repeated-WAV-header problem. API clients that append
streaming chunks must still request raw PCM with `audio.format=pcm`; crossfade
must not be used to hide container/header bytes being interpreted as samples.

The historical deployment used a configurable 240-sample crossfade (10 ms at
24 kHz). That implementation targeted an older scheduler and cannot be copied
verbatim into the current code.

### Target code

Primary implementation:

- `sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py`
- `Code2WavStreamState`
- `Code2WavScheduler.decode_delta()` for the ordinary per-request path
- `Code2WavScheduler.run_step()` for the optional batched path
- the stream-finalization and abort cleanup inherited from
  `StreamingVocoderBase`

Tests belong with the existing Code2Wav suites:

- `tests/unit_test/qwen3_omni/test_code2wav.py`
- `tests/unit_test/qwen3_omni/test_code2wav_batching.py`
- `tests/unit_test/qwen3_omni/test_streaming.py` when API-level coverage is
  needed

### Required behavior

1. Add a `stream_crossfade_samples` Code2Wav factory/config argument.
   - Default should be chosen explicitly after audio regression testing; the
     historical production value was `240`.
   - `0` must disable crossfade and preserve the current behavior.
   - Negative values must be rejected or normalized consistently with the
     scheduler's other integer parameters.
2. Store pending audio tail state per request in `Code2WavStreamState`; never
   keep it in scheduler-global state shared by requests.
3. Route both `decode_delta()` and every row emitted by `run_step()` through one
   common helper so batching cannot silently bypass crossfade.
4. On an intermediate chunk:
   - retain at most `stream_crossfade_samples` samples from the previous chunk;
   - blend that tail with the beginning of the next chunk using deterministic
     complementary fades;
   - handle chunks shorter than the requested crossfade without dropping or
     duplicating invalid ranges.
5. On finalization, emit all remaining pending samples exactly once. A request
   that finishes after one short chunk must still produce audio.
6. On cancellation, failure, or scheduler cleanup, discard that request's
   pending state without affecting another request.
7. Keep output dtype, device/CPU transfer behavior, sample rate metadata, first
   audio event semantics, profiler events, batching decisions, and CUDA Graph
   eligibility unchanged except where a documented crossfade-specific metric
   is deliberately added.
8. Do not attempt to solve the known duration loss caused by
   `stream_chunk_size=1` in this task. Crossfade addresses boundary continuity;
   chunk-size/duration behavior is a separate issue.

### Design constraint

Apply crossfade after Code2Wav has produced and context-trimmed each request's
waveform, not inside the model forward pass. The helper should accept one
request's NumPy waveform plus its `Code2WavStreamState` and return the samples
ready to emit. This keeps the audio operation independent of whether the
waveform came from eager, CUDA Graph, single-request, or batched execution.

Before implementation, confirm how `StreamingVocoderBase.on_stream_done()`
invokes final decode and cleanup. Do not add a second finalization path that can
flush the pending tail twice.

### Acceptance criteria

- With `stream_crossfade_samples=0`, existing Code2Wav unit tests remain
  unchanged and emitted samples are identical to the current implementation.
- With crossfade enabled, a deterministic synthetic discontinuity demonstrates
  that the boundary jump is reduced according to the selected fade.
- Single-request and batched execution produce the same per-request crossfade
  result for identical decoded chunks.
- Interleaved requests prove that pending tails cannot leak across request IDs.
- Tests cover: first chunk, multiple chunks, final flush, a chunk shorter than
  the configured crossfade, cancellation/cleanup, streaming output, and
  non-streaming accumulation.
- No sample is emitted twice and no pending tail is lost at normal completion.
- A real Qwen3-Omni audio comparison records boundary-jump statistics, total
  sample count/duration, time to first audio, and an audible regression check
  against `stream_crossfade_samples=0`.

### Non-goals

- Changing `stream_chunk_size` defaults or the low-latency deployment profile.
- Fixing duration loss from very small Code2Wav chunks.
- Changing WAV/PCM API defaults.
- Reworking Code2Wav batching, CUDA Graph capture, or left-context decoding.
- Reintroducing runtime source-patching scripts.
