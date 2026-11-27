"""Channel-aware Action Omni prompts."""

from __future__ import annotations

import json

from .catalog import ResolvedCatalog
from .contracts import ActionDecisionRequest, DecisionChannel
from .speech_window import PERSONA_PROFILES
from .persona_prompt import build_persona_system_prompt, build_persona_user_prompt


MAX_CHARACTER_PROMPT_CHARS = 32_000
E65_REFERENCE_PROFILE = "e65_step1512_fp8"
E66_REFERENCE_PROFILE = "e66_v116_step935_fp8"
E67_REFERENCE_PROFILE = "e67_v116_step695_fp8"
REFERENCE_PROFILES = frozenset({
    E65_REFERENCE_PROFILE,
    E66_REFERENCE_PROFILE,
    E67_REFERENCE_PROFILE,
})


def reference_prompt_profile(request: ActionDecisionRequest) -> str | None:
    profile = request.runtime_context.get("prompt_profile")
    return (
        profile
        if request.channel is DecisionChannel.BODY
        and isinstance(profile, str)
        and profile in REFERENCE_PROFILES
        else None
    )


def uses_reference_prompt(request: ActionDecisionRequest) -> bool:
    return reference_prompt_profile(request) is not None


def uses_e65_reference_prompt(request: ActionDecisionRequest) -> bool:
    """Backward-compatible predicate for callers that still name E65."""
    return reference_prompt_profile(request) == E65_REFERENCE_PROFILE

def build_system_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog, *, prompt_snapshot: dict[str, str] | None = None) -> str:
    if prompt_snapshot is None and request.prompt_snapshot is not None:
        prompt_snapshot = dict(request.prompt_snapshot.rules)
    return _build_system_prompt(request, catalog, prompt_snapshot=prompt_snapshot)


def _build_system_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog, *, prompt_snapshot: dict[str, str] | None = None) -> str:
    if (
        request.character_prompt
        and len(request.character_prompt) > MAX_CHARACTER_PROMPT_CHARS
    ):
        raise ValueError(
            f"character_prompt exceeds {MAX_CHARACTER_PROMPT_CHARS} characters"
        )
    if request.session_prompt and len(request.session_prompt) > 8192:
        raise ValueError("session_prompt exceeds 8192 characters")
    if any(
        value is not None and "\x00" in value
        for value in (request.character_prompt, request.session_prompt, request.text)
    ):
        raise ValueError("action decision text fields must not contain NUL")
    if prompt_snapshot is None or set(prompt_snapshot) != {"system_prompt"}:
        raise ValueError("action decision requires Agent system_prompt")
    profile = reference_prompt_profile(request)
    agent_snapshot = prompt_snapshot["system_prompt"]
    if profile is not None:
        base = agent_snapshot
        if not request.character_prompt and not request.session_prompt:
            return base
        context = [base, "[TRUSTED_CHARACTER_BEHAVIOR]",
            "以下角色行为规则约束当前身体动作的选择、克制度和禁止项。只决定身体动作通道；"
            "角色背景不是用户本轮的新请求。遵守本轮明确请求和宿主的动作候选／通道约束，"
            "普通伴随动作按角色行为规则选择，也可不做动作。仍只输出原协议要求的一个合法两-token code。"]
        if request.character_prompt:
            context.append(request.character_prompt)
        if request.session_prompt:
            context.extend(("[SESSION_POLICY]", request.session_prompt))
        return "\n\n".join(context)
    if request.runtime_context.get("prompt_profile") in PERSONA_PROFILES:
        return build_persona_system_prompt(
            request,
            catalog,
            agent_system_prompt=agent_snapshot,
        )
    return _build_agent_system_prompt(request, catalog, agent_snapshot)


def _build_agent_system_prompt(
    request: ActionDecisionRequest,
    catalog: ResolvedCatalog,
    system_prompt: str,
) -> str:
    """Combine Agent-owned policy with only request-scoped trusted data."""
    target = "body" if request.channel is DecisionChannel.BODY else "expression"
    marker = "[ACTION_CATALOG]" if request.channel is DecisionChannel.BODY else "[EXPRESSION_CATALOG]"
    lines = [
        system_prompt,
        "[TRUSTED_RUNTIME_CONTEXT]",
        json.dumps(
            {
                "active_channel": target,
                "turn_origin": request.turn_origin,
                "language": request.language,
                "catalog_version": catalog.catalog_version,
                "mapping_sha256": catalog.mapping_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    if request.character_prompt:
        lines.extend(("[CHARACTER_PERSONA]", request.character_prompt))
    if request.session_prompt:
        lines.extend(("[SESSION_CONTEXT]", request.session_prompt))
    lines.extend((
        marker,
        "格式：code | candidate_id | 名称 | 定义",
    ))
    lines.extend(
        f"{entry.code} | {entry.candidate_id} | {entry.label} | {entry.definition}"
        for entry in catalog.entries
    )
    return "\n".join(lines)


def build_user_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    prompt = _build_user_prompt(request, catalog)
    if request.requested_action is not None:
        from .requested_action import requested_action_user_section
        background = prompt.rsplit("\n[FINAL_TASK]\n", 1)[0].rstrip()
        catalog = "ACTION_CATALOG" if request.channel is DecisionChannel.BODY else "EXPRESSION_CATALOG"
        prompt = (requested_action_user_section(request.requested_action) +
                  "\n[CONVERSATION_BACKGROUND]\n" + background +
                  "\n[FINAL_TASK]\n按 USER_REQUESTED_ACTION 的名称和定义匹配下一刻明确目标。" +
                  f"从 {catalog} 中选择能真正完成该目标的一项；没有匹配项选择000。只输出所选项的 code。")
    return prompt


def _build_user_prompt(request: ActionDecisionRequest, catalog: ResolvedCatalog) -> str:
    media_kinds = [item.kind for item in request.media]
    if request.runtime_context.get("prompt_profile") in PERSONA_PROFILES:
        return build_persona_user_prompt(request, catalog)
    if uses_reference_prompt(request):
        image_count = sum(kind in {"image", "video"} for kind in media_kinds)
        has_audio = "audio" in media_kinds
        parts = []
        assistant = request.turn_origin == "assistant"
        if image_count:
            parts.append("user_video：按时间顺序提供的用户侧画面；不是数字人输出画面。")
        if has_audio:
            parts.append("assistant_audio：这是数字人本轮正在表达的回复音频片段。" if assistant else
                         "user_audio：这是本轮用户原始音频，请直接理解语义和语气。")
        if request.text and not has_audio:
            parts.append(f"assistant_reply_text：{request.text}" if assistant else f"typed_user_text：{request.text}")
        if assistant and request.reply_prefix:
            parts.append(f"previous_reply_context：{request.reply_prefix}")
        parts.extend(
            (
                "[RUNTIME_CONTEXT]",
                "source=trusted_orchestrator",
                f"turn_origin={request.turn_origin}",
                "trigger=assistant_reply_window" if assistant else "trigger=user_input",
                "现在只输出一个合法的两-token动作 code。",
            )
        )
        return "\n".join(parts)
    state = {
        "channel": request.channel.value,
        "turn_origin": request.turn_origin,
        "language": request.language,
        "modalities": media_kinds,
    }
    parts = ["[TURN_STATE]", json.dumps(state, ensure_ascii=False, separators=(",", ":"))]
    if catalog.runtime_context:
        parts.extend(
            (
                "[RUNTIME_CONTEXT]",
                "source=trusted_orchestrator",
                json.dumps(catalog.runtime_context, ensure_ascii=False, separators=(",", ":")),
            )
        )
    if request.text:
        parts.append("[USER_TEXT]\n" + request.text)
    if request.reply_prefix:
        parts.append("[REPLY_PREFIX]\n" + request.reply_prefix)
    if media_kinds:
        parts.append("按输入顺序结合本轮原始音频和画面判断，不要把用户画面当作数字人画面。")
    parts.append("现在只输出一个合法的两-token code。")
    return "\n".join(parts)
