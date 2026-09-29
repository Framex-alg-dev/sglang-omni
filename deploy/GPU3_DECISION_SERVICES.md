# GPU3 decision services

GPU3 runs one Qwen3-Omni pipeline and exposes two private logical services on
port 18006. It must not run a second classifier or Brain model process.

| Capability | Endpoint | Scheduling |
| --- | --- | --- |
| classifier / turn router | `POST /classifier/v1/task-classification` | short, high-priority `direct`/`delegate`/`cancel` decision |
| task Brain | `POST /brain/v1/chat/completions` | bounded concurrency 2, short structured decision |

The classifier accepts either committed text or one original audio item. It
uses the prompt contract from `sglang_omni/serve/task_classification/prompt.py`
and returns both the exact token (`direct`, `delegate`, or `cancel`) and its stable route
(`BRAIN1`, `BRAIN2`, or `CONTROL`). It does not use ASR text for audio and does not accept
images or video.

The Brain endpoint accepts only the non-streaming OpenAI subset used by D's
`task-decision-v1` adapter. It requires JSON output, rejects non-`none`
reasoning effort, caps request bytes, and limits Brain concurrency so incoming
classifier work can still receive the Thinker scheduler's turn-router priority.
Classifier and Brain have independent tokens, health endpoints and metrics,
while model weights and encoders are shared.

Timeline does not run in this process. It is isolated on GPU4/18008 because its
continuous sliding-window workload is not on the Turn decision critical path.

## Install without starting

1. Copy `config_gpu3_decision_services.example.yaml` to
   `config_gpu3_decision_services.yaml` and verify the base checkpoint path.
2. Copy `gpu3-decision-services.env.example` to
   `gpu3-decision-services.env`, replace both tokens, and run `chmod 600`.
3. Install `sglang-omni-gpu3-decision-services.service` as a systemd unit.
4. Before starting, verify physical GPU3 is free and port 18006 is unused.

This prompted shared-base deployment intentionally does not require a release
manifest. If classifier or Brain later receives its own checkpoint, run its
standalone service role and generate an immutable manifest with
`scripts/generate_model_release_manifest.py`; the model adapter seam and wire
contract do not need to change.
