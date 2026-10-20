# Turn Body early inference and release assets

D owns the time at which a committed Turn starts Body inference. The action
service accepts identity/input revision fences, validates the frozen model and
Prompt release, cancels obsolete jobs, and admits Body ahead of queued
presentation scoring. Turn-scoped media reuse is opt-in and never shares text
or KV state across identities.

The six E67 release files ship in
`sglang_omni/serve/action_decision/assets/e67_v116/`, including wheel installs.
The launcher uses that directory by default. Override
`SGLANG_OMNI_ACTION_ARTIFACT_ROOT` for a separately deployed release. D ships the
same six files in `backend/data/action_model_artifacts/e67_v116/`; their SHA256
values must agree. These files are model token mappings and Prompt assets, not
the larger product action catalog. Changing the artifact path does not change
the checkpoint loaded by the server.

Keep `SGLANG_OMNI_MODEL_VERSION` aligned with D's
`CHARACTER_ACTION_MODEL_VERSION`. Body requests reject a different model or
Prompt release before inference. Identity and input revision fences do not
become model instructions.

On D, `CHARACTER_TURN_BODY_PREFETCH_ENABLED=false` disables Commit-time Body
prefetch and retains on-demand Body inference once the reply plan is ready.
It is independent of `CHARACTER_PRE_VAD_MODE=active`, which retains early
Reply/TTS candidates. Changing either option requires restarting the D service.

This action change does not publish the independent TTS deployment directories
or reply grammar changes in the SG working tree.
