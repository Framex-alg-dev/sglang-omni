# Knowledge decision prompt workbenches

Current development runtime integration: Action E67, generic Action, and
Performance can load selected Knowledge bundles through
`sglang_omni/serve/prompt_runtime.py`. Router v1.0 is different: Agent loads and
freezes its immutable Knowledge release, then supplies that snapshot on every
turn-router v7 request. Omni validates the consumer, protocol binding, release,
and per-item checksums, but does not access Knowledge or hot-reload Router rules.
Router v6 remains a rollback contract and does not accept a snapshot. The separate
D Reply bundle also covers event replies. Publishing and selecting a version do
not hot-reload rules; restart the owning Agent after changing the selected Router
release. Invalid or missing enabled bundles block Agent startup. Existing Router
profiles other than v1.0 remain on the legacy contract. Earlier debug-only
restrictions below describe that implementation stage.

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
Default inference entry points remain. Performance now lives in serve/performance_control; embedded TTS and its text/buffer helpers live directly in serve. The old session Realtime entry and file overrides have retired.

Router uses v1.0 platform scopes, not character persona. A serving-model identifier
that differs from the snapshot's logical model binding is logged as drift and does
not reject the request; protocol or checksum drift still fails closed. E67 uses its cached reference
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
