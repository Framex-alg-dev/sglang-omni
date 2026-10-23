# SG: speculative Reply and committed Turn Body

This branch rebases the isolated Reply/Body change onto origin/dev/genesis2.0
at c8c6b26e. Its final history contains one new feature commit on that base.
The D counterpart uses branch
dev/genesis2.0_fanshide_reply_body_prefetch_20261023 in Framex-alg-dev/D_video_call.

- Body accepts identity/input revision fences, cancels queued API/model work,
  and takes priority over pending Performance work.
- Performance accepts the frozen performance_intent without redefining the
  host's required Body/expression obligations.
- Reply accepts regex or JSON schema and compiles a request-owned grammar
  matcher. Invalid/conflicting constraints fail instead of running unconstrained.
- runtime_workload_class=committed_reply|speculative_reply|tool_transition
  reaches shared admission. Committed replies and tool feedback precede queued
  speculative work. Cached grammar templates never share mutable matchers.

Use the existing action catalogs, token mappings, prompts, model versions and
deployment paths. The extraction adds no action catalog/model asset directories,
changes no action artifact defaults, and does not require D to copy SG's files.
The action service's optional expected-version/digest checks do not impose an
E67 model or a new version setting on callers.

Shared encoder media reuse is opt-in through SGLANG_OMNI_SHARED_MEDIA_CACHE_DOMAIN;
it never shares text/KV ownership across sessions, identities, Turns or input
revisions. Upstream encoder-cache fixes, task-classification changes and timeline
cleanup are retained by the rebase.

The matching TTS counterpart is 1f0194e1613293f7d8747b611be0ad11ec4d78f2
on codex/reply-body-prefetch-20261023 in Framex-alg-dev/CosyVoice3. It is one
integration commit above codex/sync-container-sources-20261022 at
5a56cbeda2c64ba82fc9c353ac01b7f94a038906; the target TTS branch needs that base
or equivalent implementations before cherry-picking. Existing session
reservations, continuous-text support and deployment configuration are retained.
The TTS guide documents its cancellation fixes and 82 passing service tests.

```bash
git fetch origin dev/genesis2.0_fanshide_reply_body_prefetch_20261023
git cherry-pick FETCH_HEAD
```

CPU tests use fake inference adapters. Existing model-release tests require the
deployment's external token-mapping data and E66/E67 prompt files. No GPU model,
live D/SG/TTS connection, latency measurement or service deployment is implied.

Validation after rebase: 213 focused CPU tests pass; two pre-existing frozen
E66/E67 prompt tests are deselected because those external prompts are absent.
The action-decision tests use temporary model-shaped token-mapping fixtures,
not a newly bundled runtime catalog. A broader catalog/gateway/classification
suite has 127 passes and 19 failures; the unchanged c8c6b26e worktree reproduces
all 19 failing tests. Compilation and git diff whitespace checks pass.
