# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.serve.realtime.output_capabilities import SessionOutputCapabilities


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["text"], ("text",)),
        (["text", "audio"], ("text", "audio")),
        (["action"], ("action",)),
        (["action", "text"], ("text", "action")),
        (["action", "audio", "text"], ("text", "audio", "action")),
    ],
)
def test_parse_allowed_outputs(raw: list[str], expected: tuple[str, ...]) -> None:
    capabilities = SessionOutputCapabilities.parse(raw)

    assert capabilities.outputs == expected
    assert capabilities.text_enabled is ("text" in expected)
    assert capabilities.audio_enabled is ("audio" in expected)
    assert capabilities.action_enabled is ("action" in expected)


@pytest.mark.parametrize(
    "raw",
    [[], ["audio"], ["audio", "action"], ["text", "text"], ["video"], [""], "text"],
)
def test_parse_rejects_invalid_outputs(raw: object) -> None:
    with pytest.raises(ValueError):
        SessionOutputCapabilities.parse(raw)
