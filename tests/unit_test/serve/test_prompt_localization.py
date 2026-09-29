"""Language isolation at the request boundary, with canonical action IDs intact."""

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.models.qwen3_omni.global_action_catalog import load_global_action_catalog
from sglang_omni.serve.openai_api import _register_runtime_prompt_overrides
from sglang_omni.serve.realtime.action.decision import category_gate_prompt
from sglang_omni.serve.realtime.performance.pipeline import _Choice
from sglang_omni.serve.realtime.proactive.policies import proactive_scene_policy
from sglang_omni.serve.realtime.runtime_prompt_overrides import (
    effective_runtime_prompt,
    read_runtime_prompt,
    write_runtime_prompt,
)
from tests.unit_test.qwen3_omni.test_multimodal_session import (
    FakeClient,
    FakeWebSocket,
    make_session,
)


CATALOG = Path(__file__).parents[3] / "sglang_omni/assets/character_limited_action_global_catalog.json"


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_RUNTIME_PROMPT_DIR", str(tmp_path))


@pytest.mark.parametrize("reply_language", ["zh", "en"])
@pytest.mark.parametrize("action_language", ["zh", "en"])
def test_reply_and_action_overrides_are_independent(reply_language, action_language):
    for language in ("zh", "en"):
        for key in ("reply_rules", "action_rules", "user_image_reply_rules", "user_image_action_rules"):
            write_runtime_prompt(key, f"{key}-{language}", language=language)
    session = make_session(FakeWebSocket(), FakeClient())
    session.language = reply_language
    session.action_language = action_language
    session.action_locale = "en-US" if action_language == "en" else "zh-CN"
    reply = session._reply_role_and_agency_system_prompt()
    assert f"reply_rules-{reply_language}" in reply
    assert f"reply_rules-{'en' if reply_language == 'zh' else 'zh'}" not in reply
    assert session._reply_user_camera_response_guard_part()["text"] == f"user_image_reply_rules-{reply_language}"
    action = session._build_session_action_profile_instruction("child", has_user_camera=True)
    assert f"action_rules-{action_language}" in action
    assert f"user_image_action_rules-{action_language}" in action
    assert f"action_rules-{'en' if action_language == 'zh' else 'zh'}" not in action


def test_missing_empty_and_updated_english_override_never_reads_chinese():
    write_runtime_prompt("reply_rules", "中文覆盖")
    assert read_runtime_prompt("reply_rules", language="zh-CN") == "中文覆盖"
    assert effective_runtime_prompt("reply_rules", "English default", language="en") == "English default"
    write_runtime_prompt("reply_rules", "English override", language="en-US")
    assert effective_runtime_prompt("reply_rules", "English default", language="en") == "English override"
    write_runtime_prompt("reply_rules", "  ", language="en")
    assert effective_runtime_prompt("reply_rules", "English default", language="en-US") == "English default"
    assert read_runtime_prompt("reply_rules") == "中文覆盖"


def test_proactive_sections_are_isolated_by_language():
    for language in ("zh", "en"):
        for kind in ("reply", "action"):
            write_runtime_prompt(f"proactive_{kind}_rules", f"[session_enter]\n{kind}-{language}", language=language)
    policy = proactive_scene_policy("session_enter")
    assert policy.reply_policy_zh == "reply-zh"
    assert policy.reply_policy_en == "reply-en"
    assert policy.default_action_guidance_zh == "action-zh"
    assert policy.default_action_guidance_en == "action-en"
    other = proactive_scene_policy("idle_timeout")
    assert other.reply_policy_en != "reply-en"
    assert not re.search(r"[\u4e00-\u9fff]", other.reply_policy_en)


def test_admin_locale_selects_defaults_and_only_updates_that_locale():
    app = FastAPI()
    _register_runtime_prompt_overrides(app, "local-test-key")
    with TestClient(app) as client:
        assert client.get("/admin/runtime-prompts").status_code == 401
        headers = {"Authorization": "Bearer local-test-key"}
        response = client.put("/admin/runtime-prompts/reply_rules?locale=en-US", json={"content": "English only"}, headers=headers)
        assert response.status_code == 200
        assert response.json()["filename"] == "reply_rules.en-US.txt"
        english = client.get("/admin/runtime-prompts?locale=en-US", headers=headers).json()["items"]
        chinese = client.get("/admin/runtime-prompts", headers=headers).json()["items"]
        assert english[0]["effective_content"] == "English only"
        assert all(item["locale"] == "en-US" for item in english)
        assert all(not re.search(r"[\u4e00-\u9fff]", item["default_content"]) for item in english)
        assert chinese[0]["source"] == "repository_default"
        assert chinese[0]["filename"] == "reply_rules.txt"
        assert client.get("/admin/runtime-prompts?locale=fr-FR", headers=headers).status_code == 422


def test_limited_catalog_has_complete_english_prompt_text_and_stable_canonical_fields():
    catalog = load_global_action_catalog(CATALOG)
    raw = json.loads(CATALOG.read_text())
    assert len(catalog.categories) == 25
    assert catalog.candidate_count == 117
    prompts = [catalog.category_system_prompt_for("en-US")]
    for category in raw["categories"]:
        parsed = catalog.category_by_id[category["category_id"]]
        assert parsed.source_label == parsed.label_for("zh-CN") == category["source_label"]
        assert parsed.short_definition == parsed.definition_for("zh-CN") == category["short_definition"]
        assert parsed.label_for("en-US") != parsed.source_label
        for child in category["children"]:
            candidate = catalog.candidate_by_id[child["candidate_id"]]
            assert candidate.action_id == child["action_id"]
            assert candidate.source_label == candidate.label_for("zh-CN") == child["source_label"]
            assert candidate.short_definition == candidate.definition_for("zh-CN") == child["short_definition"]
            assert candidate.label_for("en-US") != candidate.source_label
            assert not re.search(r"[\u4e00-\u9fff]", candidate.definition_for("en-US"))
        for origin in ("user", "proactive"):
            prompts.append(catalog.child_system_prompt_for("en-US", parsed.category_id, origin))
    for origin in ("user", "proactive"):
        prompts.append(catalog.action_system_prompt_for("en-US", origin))
    assert all(not re.search(r"[\u4e00-\u9fff]", text) for text in prompts)


@pytest.mark.parametrize("language,locale", [("zh", "zh-CN"), ("en", "en-US")])
def test_session_renderers_localize_without_changing_matching_labels(language, locale):
    catalog = load_global_action_catalog(CATALOG)
    session = make_session(FakeWebSocket(), FakeClient(), global_action_catalog=catalog)
    session.action_language, session.action_locale = language, locale
    categories, candidates = session._canonicalize_global_hierarchical_catalog([
        {"category_id": category.category_id, "children": [
            {"candidate_id": child.candidate_id, "action_id": child.action_id}
            for child in category.children
        ]} for category in catalog.categories
    ])
    session.categories, session.candidates = categories, candidates
    assert all(c.source_label == catalog.candidate_by_id[c.candidate_id].source_label for c in candidates)
    prompts = [category_gate_prompt(categories, english=language == "en")]
    prompts.append(session._render_child_system_prompt(categories, candidates))
    prompts.append(session._render_child_system_prompt(categories, candidates, definition_mode="visual"))
    prompts.append(session._performance_system_prompt({"P0": _Choice("none", None), "P1": _Choice("face", "156")}))
    if language == "en":
        assert all(not re.search(r"[\u4e00-\u9fff]", text) for text in prompts)
        assert "Serious expression" in prompts[-1]
    else:
        assert "严肃" in prompts[-1]


@pytest.mark.parametrize("value", [None, [], {"fr-FR": {"label": "x", "short_definition": "y"}}, {"en-US": {"label": "x"}}])
def test_invalid_localized_catalog_fields_are_rejected(tmp_path, value):
    raw = json.loads(CATALOG.read_text())
    raw["categories"][0]["children"][0]["prompt_text_by_locale"] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_global_action_catalog(path)
