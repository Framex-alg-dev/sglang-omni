from sglang_omni.serve.task_classification.prompt import SYSTEM_PROMPT


def test_emotional_support_is_not_inferred_as_a_music_request() -> None:
    assert '“我不开心怎么办？” -> {"route_token":"direct"' in SYSTEM_PROMPT
    assert '“来首让我开心的歌。” -> {"route_token":"delegate"' in SYSTEM_PROMPT
    assert "作为临时插问保留原任务" in SYSTEM_PROMPT


def test_media_controls_use_the_deterministic_control_route() -> None:
    assert '“暂停播放。” -> {"route_token":"control"' in SYSTEM_PROMPT
    assert '“别唱了，停止播放。” -> {"route_token":"control"' in SYSTEM_PROMPT
