---
name: omni-turn-router
description: Implement, review, test, or deploy the Qwen3-Omni model-1 turn router that derives direct/Brain1 or delegate/Brain2 from the v7 capability code, including its shared GPU3 runtime with Task Brain.
---

# Omni Turn Router

Treat the production implementation as the source of truth:

- Agent-owned prompt snapshot contract: `sglang_omni/serve/agent_prompt_snapshot.py`
- Router v1.0 request assembly: `sglang_omni/serve/task_classification/prompt_v10.py`
- Input/output contract: `sglang_omni/serve/task_classification/contracts.py`
- Strict output validation and replaceable model seam: `sglang_omni/serve/task_classification/pipeline.py`
- Shared classifier/Task-Brain deployment: `deploy/GPU3_DECISION_SERVICES.md`

Preserve these invariants:

- Accept exactly one current-turn input: keyboard text or one original user-audio item.
- Do not substitute ASR text for audio and do not send image/video to the router.
- Require the complete immutable Router prompt snapshot supplied by Agent; Omni
  must not load a local live prompt or access Knowledge directly.
- Accept model output only when it is exactly one capability code: `0`, `1`, `2`,
  or `3`; whitespace, explanations, and JSON are invalid model responses.
- Derive `direct`/`BRAIN1` when the MCP bit is clear and `delegate`/`BRAIN2` when
  it is set. Preserve the multimodal bit independently in the wire result.
- Keep model invocation behind `TaskClassificationModel` so a dedicated trained
  checkpoint can replace the prompted shared base model without changing callers.
- On GPU3, classifier and Task Brain share one loaded base model. Give Router
  requests bounded admission priority and bound Brain concurrency so Router work
  cannot be starved. Timeline remains isolated on GPU4.

When changing behavior, update the canonical production module, Agent/Omni byte-
identical protocol fixture, and interface tests together. Do not copy prompt text
into this skill or maintain a second route vocabulary here.
