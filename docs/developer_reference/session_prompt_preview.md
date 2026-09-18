# Session prompt preview

`session.start` accepts one optional boolean: `preview` (default `false`).
When true, session initialization disables this session's memory store. The
normal session path retains its configured memory. Non-boolean values fail
protocol validation.

D uses this mode for an independent, one-turn text preview, requests only
`text` and `action` outputs, and closes the connection afterward. D does not
allocate a video Worker or dispatch the returned action. This flag does not
implement action restrictions or replace prompt fields; the five editable text
fields already belong to the existing session.start protocol.

The only runtime changes in Omni are in `protocol/validation.py` and
`protocol/session_start.py`. No action-rule classifier or turn-pipeline changes
are part of this feature. See D's `docs/digital-human/character/technical/session_prompt_drafts.md`
for the seven HTTP interfaces, draft storage and product workflow.

Tests: `tests/unit_test/qwen3_omni/test_multimodal_session.py -k prompt_preview`
covers preview memory isolation, normal-session behavior and parameter types.
