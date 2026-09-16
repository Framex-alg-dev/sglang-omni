# Action-first user Turns

## Policy

Only user Turns use independent body/expression support. With reliable shared
intent, body scoring publishes `turn.action.ready` immediately, before performance
or reply completion. Facial-only intent retains its channel guard; failed shared
intent conservatively waits for scope classification. `turn.result` still joins
all inference branches. Proactive behavior is unchanged.

The complete-reply numeric 0–10 override is disabled for user Turns. Explicit
number gestures and visual imitation remain available. Ordinary accompaniment
uses shared speech intent instead of waiting for a generated reply prefix.

## Startup

Admission is two sessions per manager/Thinker deployment, including initialization
and cleanup. The production deployment must have one admission manager per shared
Thinker; independent API workers do not constitute a distributed admission gate.
`session_busy` rejects the third session. Cleanup has a 30-second deadline.

Before `session.started`, warm the available intersection of these exact labels:
基础表情、躯干前后动作、单臂抬起、指向类、展示类、符号化手势、打招呼与告别、强调类、
鼓励与庆祝、身体触碰、手指精细动作。

Each prefix is the real contextual single-category user prompt and session
instruction, in the session action locale. Ordinary contextual routing removes
camera frames, so the warmed instruction uses `has_user_camera=False` even when
the input stream contains camera frames. Visual-definition and multi-category
variants populate on demand. Reply/silent accompaniment categories are attempted
after the original eleven, using only remaining token budget; an optional
prefill rejection does not reject the session. Public startup prefixes remain
shared; session extensions are scoped by session instance and instruction hash.

Warmups are sequential per session, priority 30, with a 90-second aggregate
deadline and an 85,000-token full-prefix budget per session. Two
admitted sessions therefore budget at most 170,000 tokens for this warmup, not
the total deployment KV usage. Each request checks its remaining token allowance
before GPU admission; missing token accounting, failure or budget overflow rejects
startup and normal session cleanup releases scoped resources. Unavailable labels
are logged and not substituted. Prefixes are not pinned and may be evicted.

## Scheduling and observability

After trusted shared user intent, a one-second action-first window starts body
scoring before reply-history classification, reply generation and performance
inference are admitted. Body completion (including failure) releases the window
early. Deadline expiry releases the other branches without cancelling body work;
turn cancellation retains the existing owned-task cleanup. Facial-only turns,
provided replies, missing intent and turns without action output bypass the
window. This is a per-turn admission gate, not a deployment-wide GPU reservation;
another session may still contend for compute. Audio latency may increase.

Candidate requests inherit stage, admission priority, turn origin and logical
request/session identity from their parent prefix. Only scheduling metadata is
copied, not mutable prefix lifecycle state. Physical batch logs include these
identifiers for both prefix and candidate requests.

The single-GPU deployment profile limits chunked prefill to 1,024 tokens.
Queued category/child work is isolated from newly admitted reply/performance
prefills when no chunk is already active. A deferred request aged 250 ms permits
normal mixed admission, preventing indefinite starvation. Already running kernels
and chunks are not preempted; these are scheduling opportunities, not a wall-time
latency guarantee.

`turn_branch_started/finished` logs owned task lifecycles.
`body_action_independently_published` identifies the early body boundary.
`gpu_physical_batch_selected` records physical batch IDs, request IDs, stage,
prefix/candidate role and token ranges. Set `SGLANG_OMNI_LOG_GPU_BATCH_MEMBERS=0`
to disable detailed membership records. Logs use the bounded asynchronous writer.
Startup logs include exact categories, prefix namespaces, token accounting,
budget, elapsed time and capacity reservation/release.

Multi-category user prompts canonicalize category/candidate ordering, so the same
candidate set with reversed category rank reuses its namespace. No candidate is
removed and the execution-category ranking remains unchanged. A session-local
negative cache skips recently failed exact intent shortcuts for the same
candidate/body-task pair for 30 seconds (at most 32 entries). It skips only the
speculative attempt: normal category scoring, state filters and unsupported
validation still execute. No cross-session action decisions are cached.

`scripts/evaluate_action_priority_latency.py` runs the same ordered inputs with
one and two sessions, records workload hashes, cold/repeat iteration numbers,
event latencies and action timing breakdowns. Its default profile is synthetic,
not the production character. Use `--session` and `--cases` for role-specific
replay. The reported metric is client commit-send to event-receive, not playback
or server-only time. Small-sample P95 uses nearest rank and is not a load SLO.

## Validation and deployment

### GPU0 / 18004 visual-routing validation

Visual camera turns now use the same shared semantic intent parse as other
turns. The coarse V01–V10 classifier no longer replaces speech/body/face.
The body keeps unresolved visual references only for actual imitation requests;
explicit named actions retain their targets and simultaneous speech remains intact.
Visual category recall selects at most two real categories, without automatic
expansion to every category in the named range. This can reduce visual recall and
must be evaluated separately from speed.

With scoped Radix caching, generated visual-combination prompts are not public
catalogs. Their public boundary may therefore be zero. The exact media-free
session-instruction boundary may still be reused within the same session when
preprocessing verifies its token prefix and the position check succeeds.
Media and subsequent tokens remain request-private; cross-session public sharing
is still restricted to published catalogs. This fixes the previous rule that
collapsed the session boundary to the public boundary whenever media was present.

`deploy/sglang-omni-gpu0.service` runs the same single-GPU model profile on physical
GPU0 / port 18004, with scoped caching enabled and external TTS on 40001.
It does not switch the D_video_call backend or restart GPU5.

Measure commit-to-action, commit-to-first-audio and digital-human application
separately, including P50/P95 under one and two sessions. Do not infer speedup from
theoretical KV estimates. Test startup failure/cancellation, third-session
rejection, slow performance/reply, late expression, duplicate and cancelled turns.
Restart only after active sessions drain; health alone does not prove there are
no idle connected sessions. The D consumer must be deployed together because old
consumers enforce expression-before-action ordering.
