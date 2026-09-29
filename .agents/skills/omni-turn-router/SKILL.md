---
name: omni-turn-router
description: Implement, review, test, or deploy the Qwen3-Omni model-1 turn router that selects direct/Brain1 or delegate/Brain2, including its shared GPU3 runtime with timeline observation.
---

# Omni Turn Router

Treat the production implementation as the source of truth:

- Prompt and decision rules: `sglang_omni/serve/task_classification/prompt.py`
- Input/output contract: `sglang_omni/serve/task_classification/contracts.py`
- Strict output validation and replaceable model seam: `sglang_omni/serve/task_classification/pipeline.py`
- Shared classifier/timeline deployment: `deploy/GPU3_DECISION_SERVICES.md`

Preserve these invariants:

- Accept exactly one current-turn input: keyboard text or one original user-audio item.
- Do not substitute ASR text for audio and do not send image/video to the router.
- Accept model output only when it is exactly `direct` or `delegate`; whitespace,
  explanations, capitalization changes, and JSON are invalid model responses.
- Map `direct` to `BRAIN1` and `delegate` to `BRAIN2` at the contract boundary.
- Keep model invocation behind `TaskClassificationModel` so a dedicated trained
  checkpoint can replace the prompted shared base model without changing callers.
- On GPU3, classifier and timeline share one loaded base model. Give router
  requests bounded admission priority and keep timeline work bounded by queue,
  bytes, interval, and sliding-window limits.

When changing behavior, update the canonical production module and its interface
tests together. Do not copy the prompt into this skill or maintain a second route
vocabulary here.
