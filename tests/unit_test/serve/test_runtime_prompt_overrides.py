from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from sglang_omni.serve.realtime.runtime_prompt_overrides import (
    prompt_slot_payloads,
    read_runtime_prompt_section,
    write_runtime_prompt,
)


DEFAULTS = {
    "reply_rules": "repository reply",
    "action_rules": "repository action",
    "proactive_reply_rules": "repository proactive reply",
    "proactive_action_rules": "repository proactive action",
    "user_image_reply_rules": "repository user-image reply",
    "user_image_action_rules": "repository user-image action",
}


def test_empty_runtime_file_displays_repository_default() -> None:
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        os.environ,
        {"SGLANG_OMNI_RUNTIME_PROMPT_DIR": directory},
    ):
        Path(directory, "action_rules.txt").write_text("   ", encoding="utf-8")
        item = next(
            item
            for item in prompt_slot_payloads(DEFAULTS)
            if item["key"] == "action_rules"
        )

    assert item["source"] == "repository_default"
    assert item["effective_content"] == "repository action"
    assert item["override_content"] == ""


def test_plain_and_sectioned_proactive_overrides() -> None:
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        os.environ,
        {"SGLANG_OMNI_RUNTIME_PROMPT_DIR": directory},
    ):
        write_runtime_prompt("proactive_action_rules", "所有主动场景统一规则")
        assert (
            read_runtime_prompt_section("proactive_action_rules", "session_enter")
            == "所有主动场景统一规则"
        )

        write_runtime_prompt(
            "proactive_action_rules",
            "[session_enter]\n[动作意图：首次问候]\n入场动作\n"
            "[idle_timeout]\n[动作意图：低打扰提醒]\n空闲动作",
        )
        assert (
            read_runtime_prompt_section("proactive_action_rules", "idle_timeout")
            == "[动作意图：低打扰提醒]\n空闲动作"
        )
        assert read_runtime_prompt_section("proactive_action_rules", "farewell") is None


def test_user_image_rules_use_the_shared_runtime_prompt_directory() -> None:
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        os.environ,
        {"SGLANG_OMNI_RUNTIME_PROMPT_DIR": directory},
    ):
        write_runtime_prompt("user_image_reply_rules", "custom image reply")
        write_runtime_prompt("user_image_action_rules", "custom image action")
        items = {
            item["key"]: item for item in prompt_slot_payloads(DEFAULTS)
        }

        assert Path(directory, "user_image_reply_rules.txt").read_text(
            encoding="utf-8"
        ) == "custom image reply"
        assert Path(directory, "user_image_action_rules.txt").read_text(
            encoding="utf-8"
        ) == "custom image action"
        assert items["user_image_reply_rules"]["display_name"] == (
            "用户图片理解与回复规则"
        )
        assert items["user_image_action_rules"]["display_name"] == (
            "用户图片理解与动作规则"
        )
        assert items["user_image_reply_rules"]["activation"] == "new_turn"
        assert items["user_image_action_rules"]["activation"] == "new_turn"
