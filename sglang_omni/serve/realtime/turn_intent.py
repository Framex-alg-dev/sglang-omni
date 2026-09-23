"""One bounded semantic parse shared by speech, body, face, and visual routing."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextlib import aclosing, suppress
from dataclasses import dataclass

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.realtime.protocol.common import IMAGE_ROLE_USER_CAMERA
from sglang_omni.utils.structured_logs import emit_structured_log

TURN_INTENT_TIMEOUT_SECONDS = 4.0
GENERAL_INTENT_GATE = "GENERAL"
VISUAL_GESTURE_ANSWER_GATE = "VISUAL_ANSWER"
VISUAL_GENERAL_ANSWER_GATE = "VIEW_ANSWER"
VISUAL_HAND_MODE_IMITATE = "imitate"
VISUAL_HAND_MODE_IDENTIFY_NUMBER = "identify_number"
VISUAL_HAND_MODE_IDENTIFY_GESTURE = "identify_gesture"
VISUAL_HAND_IDENTIFICATION_MODES = frozenset(
    {VISUAL_HAND_MODE_IDENTIFY_NUMBER, VISUAL_HAND_MODE_IDENTIFY_GESTURE}
)

_VISUAL_SCOPE_GATE_CHOICES = {
    "COPY_ACTION": ("body", "这个动作"),
    "COPY_HAND": ("body", "这个手势"),
    "COPY_FACE": ("face", "这个表情"),
    "COPY_HEAD": ("body", "这个头部动作"),
    "COPY_ARM": ("body", "这个手臂动作"),
    "COPY_UPPER_BODY": ("body", "这个肩膀或躯干动作"),
    "COPY_LEG": ("body", "这个腿部动作"),
    "COPY_BODY": ("body", "这个全身动作"),
    "COPY_POSE": ("body", "这个姿势"),
    "COPY_OBJECT": ("body", "这个物品交互"),
    "COPY_SCREEN": ("body", "这个屏幕交互"),
}
_VISUAL_SCOPE_GATE_RESULTS = frozenset(
    {
        GENERAL_INTENT_GATE,
        VISUAL_GESTURE_ANSWER_GATE,
        VISUAL_GENERAL_ANSWER_GATE,
        *_VISUAL_SCOPE_GATE_CHOICES,
    }
)
_MODEL_VISUAL_ROUTE_TO_INTERNAL = {
    "NO_CURRENT_VIEW": GENERAL_INTENT_GATE,
    "COPY_CURRENT_ACTION": "COPY_ACTION",
    "COPY_CURRENT_HAND": "COPY_HAND",
    "COPY_CURRENT_FACE": "COPY_FACE",
    "COPY_CURRENT_HEAD": "COPY_HEAD",
    "COPY_CURRENT_ARM": "COPY_ARM",
    "COPY_CURRENT_UPPER_BODY": "COPY_UPPER_BODY",
    "COPY_CURRENT_LEG": "COPY_LEG",
    "COPY_CURRENT_BODY": "COPY_BODY",
    "COPY_CURRENT_POSE": "COPY_POSE",
    "COPY_CURRENT_OBJECT": "COPY_OBJECT",
    "COPY_CURRENT_SCREEN": "COPY_SCREEN",
    "ANSWER_CURRENT_VIEW": VISUAL_GENERAL_ANSWER_GATE,
    "ANSWER_CURRENT_VIEW_WITH_GESTURE": VISUAL_GESTURE_ANSWER_GATE,
}
_VISUAL_ROUTE_PREFIX_RE = re.compile(
    r'"(?:visual_route|visual)"\s*:\s*"(?P<route>[A-Z0-9_]+)"'
)


@dataclass(frozen=True)
class EarlyBodyIntent:
    """A complete body channel extracted before the full intent JSON ends."""

    body_intent: str
    body_task: str = ""


_BODY_INTENT_PREFIX_RE = re.compile(
    r'"body_intent"\s*:\s*"(?P<mode>perform|prohibit|capability|none)"'
)
_BODY_TASK_PREFIX_RE = re.compile(r'"body_task"\s*:\s*')
_FACE_TASK_PREFIX_RE = re.compile(r'"face_task"\s*:\s*')


def _body_intent_from_partial_output(raw: str) -> EarlyBodyIntent | None:
    """Extract only complete JSON values; never guess from a partial string."""

    mode_match = _BODY_INTENT_PREFIX_RE.search(raw)
    if mode_match is None:
        return None
    mode = mode_match.group("mode")
    if mode not in {"perform", "prohibit"}:
        return EarlyBodyIntent(body_intent=mode)
    task_match = _BODY_TASK_PREFIX_RE.search(raw)
    if task_match is None:
        return None
    try:
        task, _ = json.JSONDecoder().raw_decode(raw[task_match.end() :].lstrip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(task, str) or not task.strip():
        return None
    return EarlyBodyIntent(body_intent=mode, body_task=task.strip())


def _string_from_partial_output(
    raw: str,
    prefix: re.Pattern[str],
) -> str | None:
    """Decode one complete JSON string without accepting a partial value."""

    match = prefix.search(raw)
    if match is None:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(raw[match.end() :].lstrip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _same_visual_target(value: str, canonical_target: str) -> bool:
    """Compare model-owned canonical targets without user-phrase matching."""

    return value.strip().rstrip("。.!！?") == canonical_target


def _reconcile_copy_route_target(
    visual_route: str,
    *,
    body_task: object,
    face_task: object,
) -> str:
    """Prevent a resolved action target from being overwritten by COPY_*.

    A COPY route is an unresolved-reference contract. The unified model must
    use the canonical target owned by that route (or omit it for compatibility).
    Any other non-empty target is already semantically resolved and therefore
    belongs to the non-visual action path.
    """

    choice = _VISUAL_SCOPE_GATE_CHOICES.get(visual_route)
    if choice is None:
        return visual_route
    channel, canonical_target = choice
    value = body_task if channel == "body" else face_task
    if not isinstance(value, str) or not value.strip():
        return visual_route
    if _same_visual_target(value, canonical_target):
        return visual_route
    return GENERAL_INTENT_GATE


def _visual_route_from_partial_output(
    raw: str,
    *,
    has_user_camera: bool,
) -> str | None:
    """Return a safe early visual decision emitted by unified intent.

    GENERAL and visual arithmetic can be released from the visual field alone.
    COPY_* must wait for its target field: otherwise a premature model route
    could start a camera request before a concrete action such as
    ``数字六手势`` appears later in the same JSON object.
    """

    match = _VISUAL_ROUTE_PREFIX_RE.search(raw)
    if match is None:
        return None
    route = _MODEL_VISUAL_ROUTE_TO_INTERNAL.get(
        match.group("route"), match.group("route")
    )
    if route not in _VISUAL_SCOPE_GATE_RESULTS:
        return None
    if route != GENERAL_INTENT_GATE and not has_user_camera:
        return ""
    choice = _VISUAL_SCOPE_GATE_CHOICES.get(route)
    if choice is not None:
        channel, canonical_target = choice
        if channel == "body":
            early_body = _body_intent_from_partial_output(raw)
            if early_body is None:
                return None
            if early_body.body_intent != "perform" or not early_body.body_task:
                return None
            target = early_body.body_task
        else:
            target = _string_from_partial_output(raw, _FACE_TASK_PREFIX_RE)
            if target is None:
                return None
        return route if _same_visual_target(target, canonical_target) else ""
    return "" if route == GENERAL_INTENT_GATE else route

SYSTEM = '''你是数字人意图解析器，理解中英音频或文本，不识别图片，输出JSON。

has_user_camera=true/false是服务端事实。只有true才允许需画面的visual；有图片不等于要求模仿。

字段必须按流式优先级输出：首字段visual，第二字段body_intent；perform/prohibit时第三字段必须是body_task；
存在face_task时紧随动作字段输出；然后依次输出speech、reaction，再输出其他非默认载荷。visual使用下文枚举；speech只能是verbatim/generated/none；
body_intent只能是perform/prohibit/capability/none；reaction只能是respond/none。
其他非默认载荷依次为：text、speech_independent_of_body、history、reaction_task、voice_tone、
voice_pace、visual_hand_mode、visual_answer_operation、visual_answer_output。空字符串、history=false、natural和normal不得输出。
body_task只能在perform/prohibit时输出，且必须早于speech，以便身体通道独立就绪。
visual_route只能是：NO_CURRENT_VIEW、COPY_CURRENT_ACTION、COPY_CURRENT_HAND、COPY_CURRENT_FACE、COPY_CURRENT_HEAD、COPY_CURRENT_ARM、COPY_CURRENT_UPPER_BODY、COPY_CURRENT_LEG、COPY_CURRENT_BODY、COPY_CURRENT_POSE、COPY_CURRENT_OBJECT、COPY_CURRENT_SCREEN、ANSWER_CURRENT_VIEW、ANSWER_CURRENT_VIEW_WITH_GESTURE。

visual_route先于其他字段按以下互斥顺序判定：
1. 仅当不看当前画面就无法确定动作目标时选COPY_CURRENT_*：“做这个动作”=COPY_CURRENT_ACTION；“做这个手势/比这个数字/照着我的手比”=COPY_CURRENT_HAND。COPY的body_task/face_task必须使用对应标准目标“这个动作/这个手势/这个表情”等。
2. 必须看画面才能答的“这是什么/手里是什么/什么颜色”=ANSWER_CURRENT_VIEW+none+generated；只回答不模仿。明确模仿才COPY_CURRENT_*。
3. 画面中两个指代数字的四则运算=ANSWER_CURRENT_VIEW_WITH_GESTURE。默认手势加语音；只用手势/不说话时仅手势。手势识别仍用COPY_CURRENT_HAND+identify_*。
4. 语言已明确动作或数字时选NO_CURRENT_VIEW，与有无摄像头无关。“个/一个”是量词，不是“这个”；“比个数字三/比一个三/比个三/给我比六/挥手/点头/比个心”均不看图片，数字动作输出标准名“数字N手势”。

最小对比：“比个数字三/Show number three”=NO_CURRENT_VIEW+数字三手势；“比这个数字/Copy this hand sign”=COPY_CURRENT_HAND+这个手势。“不要做这个动作/做这个动作是什么意思”不得转COPY。有画面时“这是什么/What is this”=ANSWER_CURRENT_VIEW；“这是什么手势/What gesture is this”用identify_gesture；“这是数字几/这是几/What number is this”用identify_number。“这个加这个等于多少”=ANSWER_CURRENT_VIEW_WITH_GESTURE；“一加二等于多少”=NO_CURRENT_VIEW。

COPY：HAND手/gesture，FACE脸，HEAD头，ARM臂，UPPER_BODY躯干，LEG腿，BODY全身，POSE姿态，OBJECT物品，SCREEN屏幕；否则ACTION。无相机选NO_CURRENT_VIEW并说明。

一致性：COPY_CURRENT_HAND必须带visual_hand_mode：模仿=imitate，问数字=identify_number，问手势=identify_gesture。identify_*必须speech=none、无text，并以visual_answer_output控制答案：明确只用手势/不要说话=gesture_only，未指定或手势加语音播报答案=gesture_and_speech。模仿与说话可以同时存在，仍为COPY；纯模仿speech=none。“做这个动作/手势”不得选NO_CURRENT_VIEW。视觉计算必须带visual_answer_operation：加/减/乘/除对应add/subtract/multiply/divide，减除保留顺序；visual_answer_output同上。仅语音回答选NO_CURRENT_VIEW并省略两字段；额外原样话术与答案播报独立。明确的表情、语气、语速必须保留，未指定时不得编造。

其他字段：
- speech：verbatim=明确要求朗读指定正文；generated=需要语言回应；none=不说话。text为正文或语言任务。
- body_intent：perform=执行明确身体动作；prohibit=禁止动作；capability=询问动作能力；none=未要求。perform/prohibit时用body_task写动作目标。
- speech_independent_of_body：仅当同时存在身体任务和语言任务时输出布尔值。即使身体动作无法执行，语言内容仍应单独说出时为true；只是对动作的应答、确认或承诺时为false。省略等同false。
- face_task写明确表情目标；history仅在必须查阅先前对话时输出true。
- reaction仅为respond/none；用户直接问候、道别、感谢或亲昵，且无明确或禁止动作时可respond并写reaction_task。
- body_intent不是none时reaction必须none。未指定的动作、表情不要编造。只要求动作时speech=none；动作请求的礼貌疑问形式不会自动产生语言回答。

先划分语言和动作通道。对任意数字X、Y，“说X比Y”中“说X”只进入text，“比Y”只进入body_task：text=X，body_task=数字Y手势；严禁把X同时作为动作数字。“说X比Y这几个字”才是完整朗读“X比Y”，不执行手势。引用的动作词不执行；text内否定不改变body_intent。数字动作使用“数字Y手势”标准名。generated的text保留原语言任务，不作答、不变换人称。

具体立即请求通常要执行：“能挥挥手吗”“可以比个心吗”是perform；真正询问能力或范围才回答：“你会挥手吗”“你支持哪些动作”是generated且body_intent=capability。

示例：
说二比三 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字三手势","speech":"verbatim","reaction":"none","text":"二","speech_independent_of_body":true}
说三比四 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字四手势","speech":"verbatim","reaction":"none","text":"三","speech_independent_of_body":true}
比个数字三 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字三手势","speech":"none","reaction":"none"}
比一个四 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字四手势","speech":"none","reaction":"none"}
比个一 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字一手势","speech":"none","reaction":"none"}
比个三 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字三手势","speech":"none","reaction":"none"}
比个四 -> {"visual":"NO_CURRENT_VIEW","body_intent":"perform","body_task":"数字四手势","speech":"none","reaction":"none"}
这是几 -> {"visual":"COPY_CURRENT_HAND","body_intent":"perform","body_task":"这个手势","speech":"none","reaction":"none","visual_hand_mode":"identify_number","visual_answer_output":"gesture_and_speech"}
这是什么手势 -> {"visual":"COPY_CURRENT_HAND","body_intent":"perform","body_task":"这个手势","speech":"none","reaction":"none","visual_hand_mode":"identify_gesture","visual_answer_output":"gesture_and_speech"}
这个加这个等于多少，用手势回答 -> {"visual":"ANSWER_CURRENT_VIEW_WITH_GESTURE","body_intent":"none","speech":"none","reaction":"none","visual_answer_operation":"add","visual_answer_output":"gesture_only"}
这是什么 -> {"visual":"ANSWER_CURRENT_VIEW","body_intent":"none","speech":"generated","reaction":"none"}

保留方向、范围、对象、否定及多个通道。用户自述事实是generated而非要求复述；询问用户先前提供的姓名、偏好或事实才输出history=true。visual、body_intent、speech、reaction四字段必须存在，其余字段仅在非默认时输出；输出最多128个token。'''

# Voice is selected by the same semantic pass, independently of face scoring.
VOICE_TONES = {
    "natural": "natural tone, emotion matching the reply",
    "warm": "warm and gentle tone",
    "cheerful": "cheerful lively tone",
    "serious": "restrained serious tone",
    "sad": "soft subdued sad tone",
    "surprised": "surprised tone with rising pitch",
    "angry": "firm displeased tone without shouting",
    "playful": "playful light tone",
    "questioning": "questioning tone with natural rising endings",
    "nervous": "nervous uneasy tone",
}
VOICE_PACES = {
    "normal": "medium pace",
    "slow": "slower pace",
    "fast": "slightly faster pace",
}
SYSTEM += (
    '\nvoice_tone只能为'
    + '/'.join(VOICE_TONES)
    + '，voice_pace只能为normal/slow/fast；默认natural和normal时省略。'
    '语气不改变speech类型或原样正文。'
)


@dataclass(frozen=True)
class TurnIntent:
    speech: str
    text: str
    body: str
    body_mode: str
    face: str
    history: bool
    reaction_mode: str = "none"
    reaction: str = ""
    body_intent: str = ""
    elapsed_ms: float = 0
    voice_tone: str = "natural"
    voice_pace: str = "normal"
    visual_answer_output: str = ""
    visual_answer_operation: str = ""
    visual_hand_mode: str = ""
    speech_independent_of_body: bool = False
    # Internal compatibility field consumed by the action pipeline. The model
    # emits ``visual_route``; GENERAL is normalized to an empty string here.
    visual_scope_gate: str = ""

    @classmethod
    def parse(cls, raw: str, elapsed_ms=0, *, has_user_camera=False):
        if len(raw) > 4096:
            raise ValueError("intent too long")
        data = json.loads(raw)
        optional = {
            "voice_tone",
            "voice_pace",
            "visual_answer_operation",
            "visual_answer_output",
            "visual_hand_mode",
            "speech_independent_of_body",
        }
        if not isinstance(data, dict):
            raise ValueError("invalid intent fields")

        body_intent = ""
        sparse_payload_fields = {
            "body_task",
            "text",
            "face_task",
            "history",
            "reaction_task",
            *optional,
        }
        if "route" in data:
            # Compatibility with intent responses generated by the previous
            # nested schema during rolling deploys or cached test fixtures.
            allowed = {
                "route",
                *sparse_payload_fields,
            }
            if set(data) - allowed:
                raise ValueError("invalid intent fields")
            route = data.get("route")
            route_fields = {"visual", "speech", "body_intent", "reaction"}
            if not isinstance(route, dict) or set(route) != route_fields:
                raise ValueError("invalid intent route fields")
            body_intent = route.get("body_intent")
            if body_intent not in {"none", "perform", "prohibit", "capability"}:
                raise ValueError("invalid body intent")
            speech = route.get("speech")
            if body_intent == "capability" and speech != "generated":
                raise ValueError("capability intent requires generated speech")
            normalized = {
                "visual_route": route.get("visual"),
                "speech": speech,
                "text": data.get("text", ""),
                "body_mode": (
                    body_intent if body_intent in {"perform", "prohibit"} else "none"
                ),
                "body": data.get("body_task", ""),
                "face": data.get("face_task", ""),
                "history": data.get("history", False),
                "reaction_mode": route.get("reaction"),
                "reaction": data.get("reaction_task", ""),
                "speech_independent_of_body": data.get(
                    "speech_independent_of_body", False
                ),
            }
            normalized.update({key: data[key] for key in optional if key in data})
            data = normalized
        elif "visual" in data or "body_intent" in data:
            # Streaming-first schema. Keeping the action fields flat lets the
            # body channel become authoritative before speech/reaction detail
            # has finished decoding.
            required_route_fields = {
                "visual",
                "body_intent",
                "speech",
                "reaction",
            }
            allowed = required_route_fields | sparse_payload_fields
            if (
                not required_route_fields <= set(data)
                or set(data) - allowed
            ):
                raise ValueError("invalid intent fields")
            body_intent = data.get("body_intent")
            if body_intent not in {
                "none",
                "perform",
                "prohibit",
                "capability",
            }:
                raise ValueError("invalid body intent")
            speech = data.get("speech")
            if body_intent == "capability" and speech != "generated":
                raise ValueError("capability intent requires generated speech")
            normalized = {
                "visual_route": data.get("visual"),
                "speech": speech,
                "text": data.get("text", ""),
                "body_mode": (
                    body_intent
                    if body_intent in {"perform", "prohibit"}
                    else "none"
                ),
                "body": data.get("body_task", ""),
                "face": data.get("face_task", ""),
                "history": data.get("history", False),
                "reaction_mode": data.get("reaction"),
                "reaction": data.get("reaction_task", ""),
                "speech_independent_of_body": data.get(
                    "speech_independent_of_body", False
                ),
            }
            normalized.update(
                {key: data[key] for key in optional if key in data}
            )
            data = normalized
        else:
            required = {
                "visual_route",
                "speech",
                "text",
                "body",
                "body_mode",
                "face",
                "history",
                "reaction_mode",
                "reaction",
            }
            model_visual_route = data.get("visual_route")
            legacy_visual_route = _MODEL_VISUAL_ROUTE_TO_INTERNAL.get(
                model_visual_route, model_visual_route
            )
            if legacy_visual_route == VISUAL_GESTURE_ANSWER_GATE:
                visual_required = {
                    "visual_route",
                    "visual_answer_operation",
                    "visual_answer_output",
                }
                allowed = required | optional
                if not visual_required <= set(data) or set(data) - allowed:
                    raise ValueError("invalid intent fields")
                for key, default in {
                    "speech": "none",
                    "text": "",
                    "body_mode": "none",
                    "body": "",
                    "face": "",
                    "history": False,
                    "reaction_mode": "none",
                    "reaction": "",
                }.items():
                    data.setdefault(key, default)
            elif not required <= set(data) or set(data) - required - optional:
                raise ValueError("invalid intent fields")
            body_intent = data.get("body_mode", "none")

        model_visual_route = data.pop("visual_route")
        visual_route = _MODEL_VISUAL_ROUTE_TO_INTERNAL.get(
            model_visual_route, model_visual_route
        )
        if visual_route not in _VISUAL_SCOPE_GATE_RESULTS:
            raise ValueError("invalid visual route")
        visual_route = _reconcile_copy_route_target(
            visual_route,
            body_task=data.get("body"),
            face_task=data.get("face"),
        )
        if visual_route != GENERAL_INTENT_GATE and not has_user_camera:
            raise ValueError("visual route requires a current user camera image")
        if (
            data.get("voice_tone", "natural") not in VOICE_TONES
            or data.get("voice_pace", "normal") not in VOICE_PACES
        ):
            raise ValueError("invalid voice plan")
        visual_answer_operation = data.get("visual_answer_operation", "")
        visual_answer_output = data.get("visual_answer_output", "")
        visual_hand_mode = data.get("visual_hand_mode", "")
        if not isinstance(visual_hand_mode, str):
            raise ValueError("invalid visual hand mode")
        if visual_route == VISUAL_GESTURE_ANSWER_GATE:
            if visual_hand_mode:
                raise ValueError("visual hand mode requires COPY_HAND")
            if visual_answer_operation not in {
                "add",
                "subtract",
                "multiply",
                "divide",
            }:
                raise ValueError("visual answer requires an explicit operation")
            if visual_answer_output not in {
                "gesture_only",
                "gesture_and_speech",
            }:
                raise ValueError("invalid visual answer output")
        elif visual_route == "COPY_HAND":
            if visual_hand_mode not in {
                "",
                VISUAL_HAND_MODE_IMITATE,
                *VISUAL_HAND_IDENTIFICATION_MODES,
            }:
                raise ValueError("invalid visual hand mode")
            if visual_hand_mode in VISUAL_HAND_IDENTIFICATION_MODES:
                if visual_answer_operation:
                    raise ValueError("visual hand identification cannot calculate")
                if visual_answer_output not in {
                    "gesture_only",
                    "gesture_and_speech",
                }:
                    raise ValueError(
                        "visual hand identification requires an answer output"
                    )
                if data["speech"] != "none" or data["text"]:
                    raise ValueError(
                        "visual hand identification owns the answer channel"
                    )
            elif visual_answer_output or visual_answer_operation:
                raise ValueError(
                    "visual hand imitation cannot use visual answer fields"
                )
            data["visual_hand_mode"] = (
                visual_hand_mode or VISUAL_HAND_MODE_IMITATE
            )
        elif visual_answer_output or visual_answer_operation or visual_hand_mode:
            raise ValueError("visual answer fields require visual answer route")
        else:
            data["visual_answer_operation"] = ""
            data["visual_answer_output"] = ""
            data["visual_hand_mode"] = ""
        if (
            data["speech"] not in {"verbatim", "generated", "none"}
            or type(data["history"]) is not bool
            or type(data.get("speech_independent_of_body", False)) is not bool
        ):
            raise ValueError("invalid intent types")
        if data["body_mode"] not in {"perform", "prohibit", "none"}:
            raise ValueError("invalid body mode")
        if data["reaction_mode"] not in {"respond", "none"}:
            raise ValueError("invalid reaction mode")
        for key in ["text", "body", "face", "reaction"]:
            if not isinstance(data[key], str) or len(data[key]) > 512:
                raise ValueError("invalid intent content")

        # A concrete non-visual body command has no spoken channel unless the
        # same semantic parse explicitly marks that speech as independent.
        # This fails closed when short audio such as ``比个三`` is mistakenly
        # copied into verbatim/generated text while its body target is already
        # resolved. Mixed commands such as ``说二比三`` remain intact because
        # their prompt contract sets speech_independent_of_body=true.
        if (
            visual_route == GENERAL_INTENT_GATE
            and data["body_mode"] == "perform"
            and data["speech"] != "none"
            and not data.get("speech_independent_of_body", False)
        ):
            data["speech"] = "none"
            data["text"] = ""

        visual_targets = {target for _, target in _VISUAL_SCOPE_GATE_CHOICES.values()}
        if visual_route == GENERAL_INTENT_GATE and (
            (
                data["body_mode"] == "perform"
                and data["body"] in visual_targets
            )
            or data["face"] in visual_targets
        ):
            raise ValueError("unresolved visual target requires a visual route")
        if (
            visual_route in _VISUAL_SCOPE_GATE_CHOICES
            and data["body_mode"] == "prohibit"
        ):
            raise ValueError("copy route conflicts with a prohibited action")

        # visual_route is authoritative. Canonical targets keep the downstream
        # action scorer independent of model wording while preserving an
        # explicitly requested second channel (for example copied face + body).
        if visual_route in _VISUAL_SCOPE_GATE_CHOICES:
            channel, target = _VISUAL_SCOPE_GATE_CHOICES[visual_route]
            data["reaction_mode"] = "none"
            data["reaction"] = ""
            if channel == "body":
                data["body_mode"] = "perform"
                data["body"] = target
                body_intent = "perform"
            else:
                data["face"] = target
        elif visual_route == VISUAL_GESTURE_ANSWER_GATE:
            if data["body_mode"] != "none" or data["body"]:
                raise ValueError(
                    "visual answer cannot combine with another body action"
                )
            if data["reaction_mode"] != "none" or data["reaction"]:
                raise ValueError(
                    "visual answer cannot combine with an implicit reaction"
                )
        elif visual_route == VISUAL_GENERAL_ANSWER_GATE:
            if data["speech"] != "generated":
                raise ValueError("current-view answer requires generated speech")
            if data["body_mode"] != "none" or data["body"] or data["face"]:
                raise ValueError("current-view answer cannot execute an action")
            if data["reaction_mode"] != "none" or data["reaction"]:
                raise ValueError("current-view answer cannot contain a reaction")

        if (data["body_mode"] == "none") != (not bool(data["body"])):
            raise ValueError("inconsistent body mode")
        if (data["reaction_mode"] == "none") != (not bool(data["reaction"])):
            raise ValueError("inconsistent reaction mode")
        if data["body_mode"] != "none" and data["reaction_mode"] != "none":
            raise ValueError("explicit body task cannot contain implicit reaction")
        if data["speech"] == "verbatim" and not data["text"].strip():
            raise ValueError("empty verbatim content")
        if data["speech"] == "none" and data["text"]:
            raise ValueError("silent intent contains speech")
        if (
            visual_route == VISUAL_GESTURE_ANSWER_GATE
            and data["speech"] == "generated"
        ):
            raise ValueError(
                "visual answer additional speech must be explicit verbatim text"
            )

        return cls(
            **data,
            body_intent=body_intent,
            elapsed_ms=elapsed_ms,
            visual_scope_gate=(
                "" if visual_route == GENERAL_INTENT_GATE else visual_route
            ),
        )

    def tts_instruction(self) -> str:
        return (
            f"{VOICE_TONES[self.voice_tone]}, {VOICE_PACES[self.voice_pace]}, "
            "clear articulation"
        )

    def speaks_visual_answer(self) -> bool:
        return (
            self.visual_scope_gate == VISUAL_GESTURE_ANSWER_GATE
            and self.visual_answer_output == "gesture_and_speech"
        )

    def identifies_visual_hand(self) -> bool:
        return (
            self.visual_scope_gate == "COPY_HAND"
            and self.visual_hand_mode in VISUAL_HAND_IDENTIFICATION_MODES
        )

    def speaks_visual_hand_answer(self) -> bool:
        return (
            self.identifies_visual_hand()
            and self.visual_answer_output == "gesture_and_speech"
        )

    def has_visual_public_speech(self) -> bool:
        """Return whether visual arithmetic must publish any spoken text."""

        return self.speaks_visual_answer() or (
            self.visual_scope_gate == VISUAL_GESTURE_ANSWER_GATE
            and self.speech == "verbatim"
            and bool(self.text.strip())
        )

    def visual_additional_speech(self) -> str:
        """Return an explicit extra utterance, never an internal task string."""

        if (
            self.visual_scope_gate == VISUAL_GESTURE_ANSWER_GATE
            and self.speech == "verbatim"
        ):
            return self.text.strip()
        return ""

    def action_context(self, original):
        # Data remains in user context. Catalog/state rules stay system authority.
        return json.dumps(
            {
                "original_text": original,
                "body_task": (
                    "禁止" + self.body
                    if self.body_mode == "prohibit"
                    else self.body
                ),
                "body_mode": self.body_mode,
                "body_intent": self.body_intent or self.body_mode,
                "face_task": self.face,
                "speech_task": (
                    original
                    if self.speech == "generated" and original
                    else self.text
                ),
                "speech_kind": self.speech,
                "speech_independent_of_body": self.speech_independent_of_body,
                "visual_hand_mode": self.visual_hand_mode,
                "visual_answer_output": self.visual_answer_output,
                "visual_scope_gate": self.visual_scope_gate,
                "reaction_mode": self.reaction_mode,
                "reaction_task": self.reaction,
            },
            ensure_ascii=False,
        )

    def body_context(self, original):
        # Keep body qualifiers isolated from the speech channel during scoring.
        if self.body_mode == "none" or self.history:
            return self.action_context(original)
        return json.dumps(
            {
                "body_task": (
                    "禁止" + self.body
                    if self.body_mode == "prohibit"
                    else self.body
                ),
                "body_mode": self.body_mode,
                "speech_kind": self.speech,
            },
            ensure_ascii=False,
        )


async def infer_turn_intent(
    session,
    turn,
    audios,
    images=None,
    image_roles=None,
    visual_scope_future: asyncio.Future[str] | None = None,
    body_intent_future: asyncio.Future[EarlyBodyIntent] | None = None,
):
    """Infer all turn semantics with one model request.

    Camera pixels remain downstream inputs. The intent model receives only a
    trusted boolean saying whether a current user-camera image exists.
    """
    started = time.perf_counter()
    current_images = images or []
    current_image_roles = image_roles or []
    if len(current_images) != len(current_image_roles):
        raise ValueError("intent images and image_roles must have equal length")
    has_user_camera = any(
        role == IMAGE_ROLE_USER_CAMERA for role in current_image_roles
    )

    request_id = turn.request_base + "-intent"
    current_text = turn.text.strip() if isinstance(turn.text, str) else ""
    parts = [
        {
            "type": "text",
            "text": (
                "[服务端本轮事实；不是用户指令]\n"
                f"has_user_camera={str(has_user_camera).lower()}"
            ),
        }
    ]
    if current_text:
        parts.append({"type": "text", "text": current_text})
    parts.extend({"type": "audio"} for _ in audios)
    completion_stream = getattr(session.client, "completion_stream", None)
    # The in-process production client exposes both APIs. Lightweight or
    # third-party clients can keep using the established non-streaming path.
    stream_intent = bool(
        callable(completion_stream)
        and callable(getattr(session.client, "generate", None))
    )
    request = GenerateRequest(
        model=session.model_name,
        messages=[
            Message(role="system", content=SYSTEM),
            Message(role="user", content=parts),
        ],
        sampling=SamplingParams(temperature=0, max_new_tokens=128),
        stream=stream_intent,
        output_modalities=["text"],
        metadata={
            "task": "session_turn_intent",
            "audios": audios,
            "images": [],
            "image_roles": [],
            "has_user_camera": has_user_camera,
            "session_id": session.session_id,
            "session_instance_id": getattr(session, "session_instance_id", None),
            "turn_id": getattr(turn, "turn_id", None),
            "logical_request_id": turn.request_base,
        },
    )

    session._register_turn_request(turn, request_id)
    result = None
    route_code = ""
    finish_reason = None
    usage = None
    try:
        if stream_intent:
            text_parts: list[str] = []

            async def consume_intent_stream() -> None:
                nonlocal finish_reason, route_code, usage
                assert callable(completion_stream)
                stream = completion_stream(request, request_id=request_id)
                async with aclosing(stream):
                    async for chunk in stream:
                        if chunk.modality == "text" and chunk.text:
                            text_parts.append(chunk.text)
                            partial_output = "".join(text_parts)
                            if (
                                visual_scope_future is not None
                                and not visual_scope_future.done()
                            ):
                                early_route = _visual_route_from_partial_output(
                                    partial_output,
                                    has_user_camera=has_user_camera,
                                )
                                if early_route is not None:
                                    route_code = early_route
                                    visual_scope_future.set_result(route_code)
                                    emit_structured_log(
                                        "performance",
                                        "turn_intent_visual_route_ready",
                                        session_id=session.session_id,
                                        turn_id=getattr(turn, "turn_id", None),
                                        visual_scope_gate=route_code,
                                        has_user_camera=has_user_camera,
                                        elapsed_ms=(
                                            time.perf_counter() - started
                                        )
                                        * 1000,
                                    )
                            if (
                                body_intent_future is not None
                                and not body_intent_future.done()
                            ):
                                early_body = _body_intent_from_partial_output(
                                    partial_output
                                )
                                if early_body is not None:
                                    body_intent_future.set_result(early_body)
                                    emit_structured_log(
                                        "performance",
                                        "turn_intent_body_ready",
                                        session_id=session.session_id,
                                        turn_id=getattr(turn, "turn_id", None),
                                        body_intent=early_body.body_intent,
                                        body_task=early_body.body_task,
                                        elapsed_ms=(
                                            time.perf_counter() - started
                                        ) * 1000,
                                    )
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                            if chunk.usage is not None:
                                usage = chunk.usage.to_dict()

            await asyncio.wait_for(
                consume_intent_stream(),
                timeout=TURN_INTENT_TIMEOUT_SECONDS,
            )
            raw_output = "".join(text_parts)
        else:
            result = await asyncio.wait_for(
                session.client.completion(request, request_id=request_id),
                timeout=TURN_INTENT_TIMEOUT_SECONDS,
            )
            raw_output = result.text or ""
            finish_reason = getattr(result, "finish_reason", None)
            usage = (
                result.usage.to_dict()
                if getattr(result, "usage", None) is not None
                else None
            )
        emit_structured_log(
            "diagnostic",
            "turn_intent_generation_completed",
            session_id=session.session_id,
            turn_id=getattr(turn, "turn_id", None),
            request_id=request_id,
            finish_reason=finish_reason,
            usage=usage,
            output_chars=len(raw_output),
            output_sha256=hashlib.sha256(raw_output.encode()).hexdigest(),
        )
        intent = TurnIntent.parse(
            raw_output,
            (time.perf_counter() - started) * 1000,
            has_user_camera=has_user_camera,
        )
        parsed_body = EarlyBodyIntent(intent.body_intent, intent.body)
        if body_intent_future is not None:
            if body_intent_future.done():
                with suppress(asyncio.CancelledError):
                    early_body = body_intent_future.result()
                    if early_body != parsed_body:
                        emit_structured_log(
                            "error",
                            "turn_intent_body_mismatch",
                            level="warning",
                            session_id=session.session_id,
                            turn_id=getattr(turn, "turn_id", None),
                            early_body_intent=early_body.body_intent,
                            early_body_task=early_body.body_task,
                            parsed_body_intent=parsed_body.body_intent,
                            parsed_body_task=parsed_body.body_task,
                        )
            else:
                body_intent_future.set_result(parsed_body)
        parsed_route_code = intent.visual_scope_gate
        if (
            visual_scope_future is not None
            and visual_scope_future.done()
            and route_code != parsed_route_code
        ):
            emit_structured_log(
                "error",
                "turn_intent_visual_route_mismatch",
                level="warning",
                session_id=session.session_id,
                turn_id=getattr(turn, "turn_id", None),
                early_visual_scope_gate=route_code,
                parsed_visual_scope_gate=parsed_route_code,
            )
        route_code = parsed_route_code
        emit_structured_log(
            "performance",
            "turn_intent_ready",
            session_id=session.session_id,
            turn_id=getattr(turn, "turn_id", None),
            speech_kind=intent.speech,
            has_body=bool(intent.body),
            has_face=bool(intent.face),
            visual_scope_gate=route_code,
            has_user_camera=has_user_camera,
            elapsed_ms=intent.elapsed_ms,
            unified_visual_route=True,
        )
        return intent
    except asyncio.CancelledError:
        if hasattr(session.client, "abort"):
            with suppress(Exception):
                await session.client.abort(request_id)
        raise
    except Exception as exc:
        raw_output = (
            "".join(text_parts)
            if stream_intent
            else (
                (getattr(result, "text", None) or "")
                if result is not None
                else None
            )
        )
        if not stream_intent:
            usage = (
                result.usage.to_dict()
                if result is not None
                and getattr(result, "usage", None) is not None
                else None
            )
        emit_structured_log(
            "error",
            "turn_intent_fallback",
            session_id=session.session_id,
            turn_id=getattr(turn, "turn_id", None),
            request_id=request_id,
            error_type=type(exc).__name__,
            validation_reason=str(exc) if isinstance(exc, ValueError) else None,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            finish_reason=finish_reason,
            usage=usage,
            output_chars=(len(raw_output) if raw_output is not None else None),
            output_sha256=(
                hashlib.sha256(raw_output.encode()).hexdigest()
                if raw_output is not None
                else None
            ),
            raw_output=(raw_output[:4096] if raw_output is not None else None),
            raw_output_truncated=(
                len(raw_output) > 4096 if raw_output is not None else False
            ),
        )
        if result is None and hasattr(session.client, "abort"):
            with suppress(Exception):
                await session.client.abort(request_id)
        # Parsing is advisory for speech, but action execution fails closed.
        return TurnIntent(
            speech="generated",
            text=current_text,
            body="",
            body_mode="none",
            face="",
            history=False,
            reaction_mode="none",
            reaction="",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        session._unregister_turn_request(turn, request_id)
        if visual_scope_future is not None and not visual_scope_future.done():
            visual_scope_future.set_result(route_code)
        if body_intent_future is not None and not body_intent_future.done():
            body_intent_future.set_result(EarlyBodyIntent("none"))
