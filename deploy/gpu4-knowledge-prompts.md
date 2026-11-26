# GPU4 Knowledge Prompt startup snapshot

The actual GPU4 Python import tree is /data/xingmt/sglang-omni-event-partial30-1750-20261021, selected by deploy/partial30.env. The launcher, prompt_runtime, native profile/client injection and private timeline_prompt_debug are deployed in that tree and mirrored in this repository. Keep native preprocessing in the actual tree.

Install deploy/gpu4-knowledge-prompts.conf as a systemd drop-in for sglang-omni-gpu4-timeline.service. The shared root-readable /etc/sglang-omni/knowledge-runtime-prompts.env supplies tenant-scoped runtime credentials and KNOWLEDGE_PROMPT_TIMELINE_PARTIAL30_RELEASE=active. Never copy credentials into this repository. The API binds only the private interface 10.0.0.45:18008; D uses the same endpoint and server-held token for WebSocket sessions and private debug requests.

Publish and select sglang_omni.timeline.partial30 in Knowledge, then restart GPU4. One validated immutable profile is captured at startup, shared by all sessions and prefix prefill. Selection never changes a live profile. Invalid/missing bundles prevent startup; no fallback to external Prompt files when enabled. Keep the original external system_prompt.txt and optional compatibility endpoints for rollback.

Native event IDs, output validation, H/C windows, media alignment, cooldowns and retry budgets stay in code. The debug API uses a unique isolated session and the real prefill/detection path; it never dispatches avatar replies/actions. Details and evidence: Knowledge docs/creation_repairs_timeline_prompts_CN.md and prompt-baselines/creation-repairs-timeline-development-validation.json.
