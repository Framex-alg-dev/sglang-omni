# Turn Body early inference

D starts Body inference as soon as a Turn is committed. Later reply stages
reuse that Turn's task. The action service accepts identity/input revision
fences, cancels obsolete API and model jobs, and admits Body ahead of queued
presentation scoring. Turn-scoped media reuse is opt-in and never shares text
or KV state across identities.

The existing action catalogs, token mappings, checkpoint, reference prompts,
and their configured paths remain owned by the existing deployment. This
change does not bundle an additional model release or change the launcher's
artifact defaults. D does not need copies of SG's model files to start Body
earlier. Optional expected-version/digest validation is independent of any
specific model release and applies only when supplied by a caller.

On D, CHARACTER_TURN_BODY_PREFETCH_ENABLED=false disables Commit-time Body
prefetch and retains on-demand Body inference once the reply plan is ready.
It is independent of the early Reply/TTS candidate mode.
