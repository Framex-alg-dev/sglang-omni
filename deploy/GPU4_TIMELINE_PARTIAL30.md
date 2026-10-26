# Partial30 checkpoint-1750 FP8: main-repository integration

This is an opt-in GPU4 timeline profile. Adding these files does not switch or
restart the running service, and does not replace its weights or frequency policy.
The baseline is `origin/dev/genesis2.0` at `c8c6b26e`; the isolated source being
integrated is `/data/xingmt/sglang-omni-event-partial30-1750-20261021`.

## Code and deployment assets

- `sglang_omni/serve/timeline_detection/partial30.py`: frozen training request
  wrapper, actual frame timestamps, H/C ownership, synchronized audio alignment
  and the retained 30-event catalog.
- `direct_e50.py` and `client_model.py`: select `direct-partial30`, validate its
  native E-code output, preserve observation IDs/names and capture the exact
  audio actually submitted to inference. Existing V25/direct-E50 paths remain.
- `sglang_omni/preprocessing/video.py` and the Qwen preprocessor: opt-in training
  patch grid and prevention of a second HF video resize.
- `sglang_omni/models/qwen3_omni/public_prefix.py`: approve the exact published
  training system prompt for static cross-session prefix caching. Dynamic
  window text, audio and video do not become public prefix content.
- `config_gpu4_timeline_partial30.yaml`: same checkpoint and GPU pipeline
  settings as the isolated deployment.
- `gpu4-timeline-partial30-model-release.json`: unchanged, checksummed weight
  manifest. Its SHA256 is
  `8abdcfe7f2c1f3cf100f6e7c1cf3fe6203acd73f77e0080383aeb538449efc86`.
- `gpu4-timeline-partial30.env.example`: main-repository-only path overlay;
  contains no token and is not loaded by the active systemd unit.
- `prompts/partial30-checkpoint1750/`: original system prompt and event definitions.
  The raw system prompt, including its final newline, has SHA256
  `dae551e6bb93ab2d007f402bcc6157873e746c149aa4d1d46f8241e2198b3e17`.

The checkpoint stays outside Git at
`/data/xingmt/action_omni_remote/deploy/event_partial30_checkpoint1750_fp8_20261021/checkpoint`.
Do not copy weights into this repository or edit hardlinked shards in place.

## Preserved input/output contract

- Bearer-authenticated `WS /v1/timeline`, transport contract version 2.
- Video-driven windows advance every 1 second. C is the latest 3 seconds;
  H supplies up to 7 preceding seconds. Cold start can have no H.
- One continuous native H+C MP4, 2 FPS, C=6 frames, total 6-20 frames. The
  request's per-frame table retains actual capture timestamps; it does not
  fabricate uniformly spaced timestamps for irregular captures.
- Dynamic video preprocessing uses pixel bounds 131072-137625 and patch
  factor 32. HF must not resize the already prepared video again. Neither
  448x252 nor 420x224 is a fixed input resolution for this profile.
- Separate mono 16 kHz PCM16 WAV, cropped to the first actual input frame
  through the request end. Missing/incomplete aligned audio has no audio
  placeholder and is explicitly described as unavailable, never synthesized.
- Unmodified frozen system prompt and dynamic `<request_window>` wrapper;
  temperature=0, top_p=1, seed=42, max_tokens=128.
- Raw model output is `N` or comma-separated native E-codes, not JSON/P-codes.
  Independent E22/E38/E44 instances may repeat. Unsupported codes fail closed.
  Accepted events become transport `observation` JSON only after frequency gating.
- Retained IDs: E0,E2,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E19,E20,E21,E22,E23,
  E24,E32,E33,E34,E36,E37,E38,E44,E46,E47,E49,E50. No renumbering or thumbs-up/OK
  collapse. E49 is named `频繁看手表`.

Frequency is not part of the training prompt. The overlay intentionally does
not change the existing event cooldown, control cooldown, confirmation-window,
rearm-window or silent-event settings. Those remain owned by the base GPU4 env;
D_video_call separately owns UI hiding, reply/action admission and interruption.

The two `PARTIAL30_*` preprocessing settings apply to the entire service process.
Set them only for the dedicated Partial30 GPU4 service, not shared action/reply,
router or other video model workers. Without these settings their old defaults
remain unchanged. The newer main-branch router prompts and GPU-resident image
encoder cache have not been replaced by the isolated snapshot.

## Verification without restarting a service

From the repository root:

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/unit_test/serve/timeline_detection \
  tests/unit_test/qwen3_omni/test_timeline_e50_public_prefix.py \
  -q -p no:cacheprovider
```

The tests cover all 30 retained IDs, removed-code rejection, exact prompt bytes,
native-video/audio wrapper, cold start, static prefill, aligned artifact audio,
frequency gates, transport parsing, and preprocessing opt-in/default behavior.
They do not allocate GPU model weights or alter the active service.

Integration validation on 2026-10-26: the event, preprocessing, Qwen pipeline,
action runtime and prefix/cache regression selection passed 224 tests. Metadata
replay against the frozen `partial30_window_v3_20261020_r2_boundary_pairs`
datasets exactly matched all 834 dev and 768 test system/user input templates
(1602 total; 370 with audio and 1232 without). This validates input rendering,
not a new GPU deployment or per-event recognition accuracy.

## Later cutover (requires a separate approved restart)

1. Keep the isolated runtime, current `partial30.conf` and base GPU4 env for
   rollback. Do not replace the base env or its authentication token.
2. Prepare `gpu4-timeline-partial30.env` from the new `.env.example`. Adjust
   repository/checkpoint paths if deployed elsewhere, and regenerate the
   manifest with `scripts/generate_model_release_manifest.py` if weights change.
3. Change only `/etc/systemd/system/sglang-omni-gpu4-timeline.service.d/partial30.conf`
   to load the main-repository overlay after the existing base env:

   ```ini
   [Service]
   EnvironmentFile=/data/fanshide/sglang-omni-dev/deploy/gpu4-timeline-partial30.env
   ```

4. Once the GPU4-only restart is approved, reload systemd and restart
   `sglang-omni-gpu4-timeline.service`. Its venv, service name, port 18008 and
   `qwen3-omni-gpu4-timeline` alias stay unchanged. No D_video_call or other model
   service restart is needed for this source-path switch.
5. Verify authenticated health, raw E-code and downstream observation replay,
   1-second window cadence and cached static prefix tokens before acceptance.
6. To roll back, restore the previous `partial30.conf` pointing to the isolated
   env, reload systemd and restart only GPU4. Do not delete the isolated source
   until the main-repository deployment has passed acceptance.

This integration does not perform any of the cutover/restart steps above.
