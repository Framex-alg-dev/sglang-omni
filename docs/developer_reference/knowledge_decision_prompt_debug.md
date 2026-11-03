# Knowledge decision prompt workbenches

Current development runtime integration (2026-11-03): Action E67, generic Action,
Performance, and Router v1.0 now load selected Knowledge bundles once at startup
through `sglang_omni/serve/prompt_runtime.py`. The separate D Reply bundle also
covers event replies. Publishing and selecting a version do not hot-reload rules.
Restart `sglang-omni-GPU0_1-dev.service` for Action/Performance and
`sglang-omni-gpu3-decision-services.service` for Router, then verify each debug
state response's `runtime.release_id`, `bundle_sha256`, and `loaded_at` against the
Knowledge deployment. Invalid or missing enabled bundles block startup. Use the
read-only token in the root-owned environment file; the non-secret template is
`deploy/knowledge-runtime-prompts.env.example`. Existing legacy profiles and
Timeline remain unchanged. Earlier debug-only restrictions below describe that
implementation stage.

The running classifier and action services expose authenticated `/v1/prompt-debug/state`
and `/v1/prompt-debug/evaluate` hooks. Gateway exposes fixed-target management proxies at
`/v1/prompt-debug/{router|action}/{state|evaluate}` only when
`SGLANG_OMNI_PROMPT_DEBUG_TOKEN` is explicitly configured. Credentials stay server-side.
The proxy reuses configured classifier/body upstream addresses and service authorization.
It preserves a mounted classifier URL prefix, and never accepts an arbitrary destination.

State reports actual loaded rules, model version, and action registry identities.
Evaluation requires its exact state SHA-256, a complete rule set, and a native payload.
Each trial uses an independent model/engine instance with the original registry and model
client. No process-wide constant or runtime file is modified. Preview captures the real
GenerateRequest before inference; run delegates to the existing model client and parser.
Default inference entry points and shared realtime/performance/TTS modules remain in place.

Router uses v1.0 platform scopes, not character persona. E67 uses its cached reference
body; it does not interpolate dynamic character/session prompts. The generic proactive
action branch does interpolate those inputs. Knowledge UI documents these differences.
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
