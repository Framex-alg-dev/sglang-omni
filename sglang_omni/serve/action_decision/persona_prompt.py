"""Join Agent-owned persona rules with request and catalog data."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .contracts import DecisionChannel
from .prompt_text import fields_text, quoted_text
from .speech_window import PERSONA_AUDIO_PROFILE
from .speech_audio import USER_CONTEXT_AUDIO_ROLE
from .usage_scenarios import UsageScenarioCatalog, load_usage_scenarios, load_body_usage_scenarios

if TYPE_CHECKING:
    from .catalog import ActionEntry, ResolvedCatalog
    from .contracts import ActionDecisionRequest


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _body_persona_description(description: str) -> str:
    """Keep character style while omitting redundant allowed-action lists."""
    try:
        persona = json.loads(description)
    except (ValueError, TypeError):
        return description
    allowed_action_fields = {"动作白名单", "允许动作", "允许的动作", "allowed_actions", "action_whitelist"}
    if not isinstance(persona, dict):
        return description
    return fields_text({key: value for key, value in persona.items() if key not in allowed_action_fields})


def _cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("|", "\\u007c")


def _compact_catalog_row(entry: ActionEntry, catalog: ResolvedCatalog, usage: UsageScenarioCatalog) -> str:
    """Render only fields that can change the model's semantic decision."""
    scenario = usage.for_entry(entry, catalog)
    if scenario is not None:
        by_id = catalog.by_candidate_id
        neighbors = [by_id[candidate_id].label for candidate_id in scenario.confusable_candidate_ids
                     if candidate_id in by_id]
        boundary = "≠" + ("、".join(neighbors) if neighbors else "相似动作")
        # The full 210-action catalog must fit both text and audio requests.
        # Keep the semantic core and remove illustrative quotes already covered
        # by training examples; subject/persona exclusions live in the shared
        # decision rules above.
        scene_text = re.sub(r"[，；,;](?:例如|如).*$", "", scenario.meaning)
        scene_text = scene_text.replace("适用于", "").replace("时选择", "").replace("不得选择", "排除")
        scene_text = scene_text.strip("；。 ")
        values = (entry.code, entry.label, entry.motion_definition,
                  scene_text, boundary)
    elif entry.candidate_id == "000":
        values = (entry.code, "不支持的具体动作", "不执行替代动作",
                  "仅用于明确具体目标越过本次目录",
                  "普通交流必须选择目录中的真实动作")
    else:
        values = (entry.code, entry.label, entry.motion_definition or entry.definition,
                  "仅用于可信业务绑定", "不得作为普通伴随动作")
    return " | ".join(_cell(value) for value in values)


def build_persona_system_prompt(
    request: ActionDecisionRequest,
    catalog: ResolvedCatalog,
    *,
    agent_system_prompt: str,
) -> str:
    """Render changing persona/catalog facts under Agent-owned policy text."""
    body = request.channel is DecisionChannel.BODY
    audio_body = body and catalog.runtime_context["prompt_profile"] == PERSONA_AUDIO_PROFILE
    usage = load_body_usage_scenarios() if audio_body else load_usage_scenarios()
    usage.validate_catalog(catalog)
    lines = [
        agent_system_prompt,
        "[TRUSTED_RUNTIME_CONTEXT]",
        _json({
            "active_channel": request.channel.value,
            "decision_mode": catalog.runtime_context["decision_mode"],
            "prompt_profile": catalog.runtime_context["prompt_profile"],
            "turn_origin": request.turn_origin,
            "language": request.language,
            "catalog_version": catalog.catalog_version,
            "mapping_sha256": catalog.mapping_hash,
        }),
        "[CHARACTER_PERSONA]",
        quoted_text(_body_persona_description(request.character_prompt))
        if body else quoted_text(request.character_prompt),
    ]
    if request.session_prompt:
        lines.extend(("[SESSION_CONTEXT]", quoted_text(request.session_prompt)))
    if body:
        lines.extend((
            "[ACTION_CATALOG]",
            "格式：code | 动作名 | 动作定义 | 主动表达场景及排除条件 | 易混边界",
        ))
        lines.extend(_compact_catalog_row(entry, catalog, usage) for entry in catalog.entries)
    else:
        from .expression_prompt import _compact_expression_row
        lines.extend((
            "[EXPRESSION_CATALOG]",
            "格式：code | 表情名 | 表情定义 | 主动表达场景及排除条件 | 易混边界",
        ))
        lines.extend(
            _compact_expression_row(entry, catalog, usage)
            for entry in catalog.entries
        )
    return "\n".join(lines)


def build_persona_user_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    if catalog.runtime_context.get("prompt_profile") == PERSONA_AUDIO_PROFILE:
        return _build_audio_user_prompt(request, catalog)
    context = dict(catalog.runtime_context)
    window = context.pop("speech_window", None)
    kinds = [item.kind for item in request.media]
    lines = ["[TRUSTED_DECISION_CONTEXT]", _json({
        **context, "channel": request.channel.value,
        "turn_origin": request.turn_origin, "language": request.language,
    })]
    if window is not None:
        lines.extend(("[AVATAR_SPEECH_WINDOW]", _json(window)))
    observation: dict[str, object] = {}
    if "audio" in kinds:
        observation["audio"] = "原始用户音频，直接理解语义与语气，无ASR副本"
    elif request.text:
        observation["typed_text"] = request.text
    if "image" in kinds:
        observation["images"] = "按时间顺序提供的用户侧画面，不是数字人画面"
    lines.extend(("[USER_OBSERVATION]", _json(observation), "只输出一个合法两-token code。"))
    return "\n".join(lines)


def _build_audio_user_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    if request.channel is DecisionChannel.EXPRESSION:
        from .expression_prompt import build_expression_user_prompt
        return build_expression_user_prompt(request, catalog)
    speech = catalog.runtime_context.get("decision_mode") == "speech_window"
    previous = catalog.runtime_context.get("last_dispatched_action")
    lines = []
    if isinstance(previous, dict) and previous.get("label") and previous.get("status") != "unknown":
        description = previous.get("motion_definition")
        action = f"{previous['label']}：{description}" if description else previous["label"]
        lines.extend(("[LAST_DISPATCHED_BODY_ACTION]", f"{action}\n状态：{previous.get('status')}"))
    if speech:
        if request.text:
            lines.extend(("[ORIGINAL_USER_CONTEXT]", request.text))
        elif any(item.evidence_role == USER_CONTEXT_AUDIO_ROLE for item in request.media):
            lines.extend(("[ORIGINAL_USER_CONTEXT]", "见已标记的 USER_CONTEXT_AUDIO"))
        if request.avatar_history_text is not None:
            lines.extend(("[AVATAR_HISTORY_TEXT]", request.avatar_history_text))
        if request.avatar_current_text is not None:
            lines.extend(("[AVATAR_CURRENT_TEXT]", request.avatar_current_text))
    else:
        if request.text and not any(item.evidence_role == "user_audio" for item in request.media):
            lines.extend(("[USER_INPUT]", request.text))
    return "\n".join(lines)
