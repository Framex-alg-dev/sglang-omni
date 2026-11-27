# Knowledge decision prompt workbenches

Current live runtime integration: Agent owns Knowledge access for Action,
Timeline Event, Router, and Reply. It freezes immutable releases and sends the
corresponding prompt snapshot (or Reply messages) with each model request. Omni
validates consumer, protocol, release, content checksums, and required documents;
it does not access Knowledge or fall back to a local live prompt. A missing or
invalid Agent snapshot rejects Action, Timeline Event, and Router v7 requests.
Logical model identifiers are observability metadata: a mismatch with the model
actually deployed by Omni emits `prompt.binding_drift` with both bindings but does
not block inference. Publishing or selecting a release therefore takes effect
only after the owning Agent reloads it. The hooks below remain authenticated
diagnostic tools and do not change this ownership boundary.

The running classifier and action services expose authenticated `/v1/prompt-debug/state`
and `/v1/prompt-debug/evaluate` hooks. Gateway exposes fixed-target management proxies at
`/v1/prompt-debug/{router|action}/{state|evaluate}` only when
`SGLANG_OMNI_PROMPT_DEBUG_TOKEN` is explicitly configured. Credentials stay server-side.
The proxy reuses configured classifier/body upstream addresses and service authorization.
It preserves a mounted classifier URL prefix, and never accepts an arbitrary destination.

State reports model version and action registry identities, but deliberately does
not expose or retain prompt text. Knowledge supplies the complete Agent-owned rule
set for each evaluation. Evaluation requires the exact service-state SHA-256, a
complete rule set, and a native payload.
Each trial uses an independent model/engine instance with the original registry and model
client. No process-wide constant or runtime file is modified. Preview captures the real
GenerateRequest before inference; run delegates to the existing model client and parser.
Default inference entry points remain. Performance now lives in serve/performance_control; embedded TTS and its text/buffer helpers live directly in serve. The old session Realtime entry, local Action prompt copies, and file overrides have retired.

Router uses v1.0 platform scopes, not character persona. A serving-model identifier
that differs from the snapshot's logical model binding is logged as drift and does
not reject the request; protocol or checksum drift still fails closed. E67 uses the
Agent-supplied system prompt and never reads a cached Omni copy. Character and session
values remain request-scoped trusted data rendered under that policy. Knowledge UI
documents these differences.
Trials select and validate actions only; they do not execute avatar output or tools.

Development systemd configuration uses a root-only `/etc/sglang-omni/prompt-debug.env`
and `sglang-omni-inference-gateway.service.d/zzz-prompt-debug.conf`. The drop-in must load
after the existing private-bind drop-in, which resets EnvironmentFile entries.
Restart the gateway after installing the env file and running daemon-reload. Do not print
or check service credentials into source control.

Relevant tests (with the actual E67 artifact root):

```bash
ACTION_ARTIFACT_ROOT=/data/xingmt/model_repo/action_prediction_model/assets/e67_v116 \
.venv/bin/python -m pytest tests/unit_test/serve/test_prompt_debug.py \
 tests/unit_test/serve/test_inference_gateway.py \
 tests/unit_test/serve/test_inference_gateway_speech.py -q
```

The formal Knowledge migration remains 18 publication units. Optional legacy Reply stays
archive-only and retirement-pending. These debug hooks do not activate runtime Knowledge
snapshots or authorize retiring existing compatibility routes.
