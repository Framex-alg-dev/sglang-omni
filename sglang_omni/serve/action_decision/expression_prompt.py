"""Render expression catalog rows and request evidence for Agent-owned prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .speech_audio import USER_CONTEXT_AUDIO_ROLE

if TYPE_CHECKING:
    from .catalog import ActionEntry, ResolvedCatalog
    from .contracts import ActionDecisionRequest
    from .usage_scenarios import UsageScenarioCatalog


def _cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("|", "\\u007c")


def _compact_expression_row(entry: ActionEntry, catalog: ResolvedCatalog, usage: UsageScenarioCatalog) -> str:
    scene = usage.for_entry(entry, catalog)
    if scene is not None:
        by_id = catalog.by_candidate_id
        neighbors = [by_id[candidate_id].label for candidate_id in scene.confusable_candidate_ids
                     if candidate_id in by_id]
        boundary = "易混：" + ("、".join(neighbors) if neighbors else "相似表情")
        values = (entry.code, entry.label, entry.motion_definition,
                  scene.recall_summary, boundary)
    elif entry.candidate_id == "IF0":
        values = (entry.code, "保持当前表情", "不发起新表情",
                  "普通交流、依据不足或不宜切换时成立",
                  "不表示能力不支持")
    elif entry.candidate_id == "000":
        values = (entry.code, "不支持的具体表情", "不执行替代表情",
                  "仅用于明确具体目标越过本次目录",
                  "普通交流应选择保持当前表情")
    else:
        values = (entry.code, entry.label, entry.motion_definition or entry.definition,
                  "仅用于可信业务绑定", "不得作为普通自然反应")
    return " | ".join(_cell(v) for v in values)


def build_expression_user_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    speech = catalog.runtime_context.get("decision_mode") == "speech_window"
    lines = []
    if speech:
        if request.text:
            lines.extend(("[ORIGINAL_USER_CONTEXT]", request.text))
        elif any(item.evidence_role == USER_CONTEXT_AUDIO_ROLE for item in request.media):
            lines.extend(("[ORIGINAL_USER_CONTEXT]", "见已标记的 USER_CONTEXT_AUDIO"))
        if request.avatar_history_text is not None:
            lines.extend(("[AVATAR_HISTORY_TEXT]", request.avatar_history_text))
        if request.avatar_current_text is not None:
            lines.extend(("[AVATAR_CURRENT_TEXT]", request.avatar_current_text))
    elif request.text and not any(item.evidence_role == "user_audio" for item in request.media):
        lines.extend(("[USER_INPUT]", request.text))
    return "\n".join(lines)
