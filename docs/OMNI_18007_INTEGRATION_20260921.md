# Omni 18007 integration - 2026-09-21

## Verified source (read-only)
- Pro5000 TCP 18007 listener: PID 657651, sgl-omni.
- systemd service: sglang-omni-gpu5.service; cwd /data/fanshide/sglang-omni.
- Source HEAD: 5def8b30faed880adb15367d296047ae075e8f87 (detached HEAD).
- Tracked source files clean. Untracked deploy/realtime-executors.env was NOT copied or read.
- This is the Omni source version, NOT digital-human rollback commit 31e560b.

## Integration workspace
- /data/luozhijie/sglang-omni-18007-20260921
- Branch: luozhijie/omni-18007-sync-20260921
- Original repository /data/luozhijie/sglang-omni and its eight dirty files remain unchanged.
- Base f81aab01; local working changes preserved in bfaeed02 before merging upstream.
- Git history retained, no reset/squash, no remote push.

## Conflicts resolved
- dev_model.py: upstream reserved action IDs (000/00/UNSUPPORTED), retain local fake avatar state.
- protocol/input.py: preserve avatar-analysis task cancellation and upstream collecting/processing cancellation handling.
- protocol/session_start.py: keep upstream action_locale and local avatar-state diagnostics together.
- Other local avatar-state analysis, protocol and launch-script changes retained.

## Validation (no real service started)
CPU tests with CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1, pytest cache disabled:
- tests/unit_test/qwen3_omni/test_multimodal_session.py
- tests/unit_test/serve/test_realtime_dev_model.py
- tests/unit_test/serve/test_turn_intent.py
- tests/unit_test/serve/test_prompt_localization.py

Merged: 452 passed / 15 failed. Clean upstream: 450 passed / 15 failed.
All 15 failing node IDs are identical; no added failures in this test selection.
Existing failures are not declared fixed. This is not real GPU/WS acceptance.
Modified local Python files parse successfully; original dirty diff exactly matches snapshot commit.
Logs are in /data/luozhijie/sglang-omni-18007[-baseline]-20260921-test.log.

## Next: independent service (not performed)
Prepare a dedicated Python environment or explicitly selected interpreter whose source import resolves
here; confirm compatible model paths, a free GPU, a free listening port, and TTS endpoint/voice.
The old local launcher is retained, but its defaults are not approved for a new deployment.
Do not invoke it blindly: review ports/GPU/model paths first. Do not reuse the occupied 18007.
Model inference, dependencies and performance must be verified separately before other teams rely on it.
No Tencent deployment server was contacted. No upstream file, environment, service or process was modified.

## Baseline failing tests
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_direct_background_calibration_shadow_does_not_change_selection
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_direct_same_batch_category_gate_filters_global_action_ranking[category_scores0-concrete_scores0-288-supported-top1]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_direct_same_batch_category_gate_filters_global_action_ranking[category_scores1-concrete_scores1-UNSUPPORTED-unsupported-top1]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_direct_same_batch_category_gate_filters_global_action_ranking[category_scores2-concrete_scores2-285-supported-top2_close]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_direct_visual_deictic_action_uses_scoped_candidates_and_three_frames
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_enforced_greeting_reaction_allows_wave_with_no_explicit_body
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-False-P200]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-False-P201]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-False-P301]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-True-P200]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-True-P201]
- FAILED tests/unit_test/qwen3_omni/test_multimodal_session.py::test_mixed_numeric_turn_preserves_reply_and_visual_channels[True-True-P301]
- FAILED tests/unit_test/serve/test_realtime_dev_model.py::test_fusion_reports_unavailable_child_preference_and_uses_safe_fallback
- FAILED tests/unit_test/serve/test_realtime_dev_model.py::test_session_realtime_runs_deterministic_turn[outputs1]
- FAILED tests/unit_test/serve/test_realtime_dev_model.py::test_session_realtime_runs_deterministic_turn[outputs2]
