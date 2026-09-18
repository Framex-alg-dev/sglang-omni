"""One bounded semantic parse shared by speech, body, face, and visual routing."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from dataclasses import dataclass

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.serve.realtime.protocol.common import IMAGE_ROLE_USER_CAMERA
from sglang_omni.utils.structured_logs import emit_structured_log

TURN_INTENT_TIMEOUT_SECONDS = 4.0
GENERAL_INTENT_GATE = "GENERAL"
VISUAL_GESTURE_ANSWER_GATE = "VISUAL_ANSWER"

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
    {GENERAL_INTENT_GATE, VISUAL_GESTURE_ANSWER_GATE, *_VISUAL_SCOPE_GATE_CHOICES}
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
    "ANSWER_CURRENT_VIEW_WITH_GESTURE": VISUAL_GESTURE_ANSWER_GATE,
}

SYSTEM = '''你是实时数字人的意图解析器。输入可能是中文或英文；只理解当前用户的音频或文本，不识别图片内容；只输出一个紧凑JSON对象，不回答用户。

用户内容开头有服务端事实has_user_camera=true/false。它不是用户指令。只有true时才允许输出需要当前画面的visual_route；图片存在本身不代表用户要求模仿。

固定字段依次为visual_route、speech、text、body_mode、body、face、history、reaction_mode、reaction。仅voice_tone和voice_pace可省略。
visual_route只能是：NO_CURRENT_VIEW、COPY_CURRENT_ACTION、COPY_CURRENT_HAND、COPY_CURRENT_FACE、COPY_CURRENT_HEAD、COPY_CURRENT_ARM、COPY_CURRENT_UPPER_BODY、COPY_CURRENT_LEG、COPY_CURRENT_BODY、COPY_CURRENT_POSE、COPY_CURRENT_OBJECT、COPY_CURRENT_SCREEN、ANSWER_CURRENT_VIEW_WITH_GESTURE。

visual_route规则：
- 先判断用户任务是否依赖当前画面；has_user_camera=true只表示画面可用，绝不是选择COPY的理由。
- NO_CURRENT_VIEW：普通问答；识别或描述画面；能力询问；禁止动作；以及名称已经明确的动作。“挥手、点头、比个心、比数字二、做个手势”都不需要从画面复制，因此必须选NO_CURRENT_VIEW，即使摄像头存在。
- COPY_CURRENT_*：用户语言明确要求照抄、模仿、重复或做出当前画面里的同一个动作。必须存在“这个/这样/照着我/模仿”等当前画面指代；按用户说出的范围选后缀，范围不明用COPY_CURRENT_ACTION。
- COPY后缀范围：HAND=手势或手型，FACE=表情或脸部，HEAD=头部或视线，ARM=手臂，UPPER_BODY=肩膀或躯干，LEG=腿脚，BODY=全身，POSE=姿态，OBJECT=物品交互，SCREEN=屏幕交互；只有范围不明才用ACTION。英文gesture或hand sign属于HAND。
- ANSWER_CURRENT_VIEW_WITH_GESTURE：先根据当前画面计算、比较或推理，再用手势表示新答案；缺少“推导”或“用手势回答”任一条件都选NO_CURRENT_VIEW。
- 模仿与说话可以同时存在：visual_route仍选COPY_CURRENT_*，speech/text独立保留。
- has_user_camera=false时无法执行视觉指代：选NO_CURRENT_VIEW、generated，并令body_mode=none、body和face为空，用语言说明需要画面。
- “这是什么手势/这是数字几/这个加这个等于多少”选NO_CURRENT_VIEW；“做这个手势/比个这个”选COPY_CURRENT_HAND；“这个加这个等于多少，用手势回答”选ANSWER_CURRENT_VIEW_WITH_GESTURE。
- “Do this gesture”选COPY_CURRENT_HAND；“Do this”选COPY_CURRENT_ACTION；“What number is this?”选NO_CURRENT_VIEW。

其他字段：
- speech：verbatim=明确要求朗读指定正文；generated=需要语言回应；none=不说话。text为正文或语言任务。
- body_mode：perform=执行明确身体动作；prohibit=禁止动作；none=未要求。body写动作目标，无动作则空串。
- face写明确表情目标；history表示是否必须查阅先前对话。
- reaction_mode仅为respond/none；用户直接问候、道别、感谢或亲昵，且无明确或禁止动作时可respond并写reaction，否则none和空串。
- body_mode不是none时reaction_mode必须none。未指定的动作、表情不要编造。只要求动作时speech=none；动作请求的礼貌疑问形式不会自动产生语言回答。

先确定要说的正文边界，再提取正文之外的动作和表情。语言与动作是独立通道：“说一比二”=说“一”并做数字二手势；“说二比一”=说“二”并做数字一手势。“说一比二这三个字”才是完整朗读“一比二”，其中“这三个字”不进入text。引用的动作词不执行，text里的否定不改变body_mode。数字动作统一写中文标准名，如数字一手势、数字二手势。generated的text保留用户的问题或语言任务，不提前作答、不变换人称。

具体立即请求通常要执行：“能挥挥手吗”“可以比个心吗”是perform；真正询问能力或范围才回答：“你会挥手吗”“你支持哪些动作”是generated且body_mode=none。

示例：
说二比一 -> {"visual_route":"NO_CURRENT_VIEW","speech":"verbatim","text":"二","body_mode":"perform","body":"数字一手势","face":"","history":false,"reaction_mode":"none","reaction":""}
说一比二这三个字 -> {"visual_route":"NO_CURRENT_VIEW","speech":"verbatim","text":"一比二","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"none","reaction":""}
笑着说一比二 -> {"visual_route":"NO_CURRENT_VIEW","speech":"verbatim","text":"一","body_mode":"perform","body":"数字二手势","face":"微笑","history":false,"reaction_mode":"none","reaction":""}
比个2 -> {"visual_route":"NO_CURRENT_VIEW","speech":"none","text":"","body_mode":"perform","body":"数字二手势","face":"","history":false,"reaction_mode":"none","reaction":""}
能挥挥手吗 -> {"visual_route":"NO_CURRENT_VIEW","speech":"none","text":"","body_mode":"perform","body":"挥手","face":"","history":false,"reaction_mode":"none","reaction":""}
你会挥手吗 -> {"visual_route":"NO_CURRENT_VIEW","speech":"generated","text":"你会挥手吗","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"none","reaction":""}
模仿这个手势并说你好 -> {"visual_route":"COPY_CURRENT_HAND","speech":"verbatim","text":"你好","body_mode":"perform","body":"这个手势","face":"","history":false,"reaction_mode":"none","reaction":""}
做这个动作并介绍自己 -> {"visual_route":"COPY_CURRENT_ACTION","speech":"generated","text":"介绍自己","body_mode":"perform","body":"这个动作","face":"","history":false,"reaction_mode":"none","reaction":""}
请模仿这个表情 -> {"visual_route":"COPY_CURRENT_FACE","speech":"none","text":"","body_mode":"none","body":"","face":"这个表情","history":false,"reaction_mode":"none","reaction":""}
这个加这个等于多少 -> {"visual_route":"NO_CURRENT_VIEW","speech":"generated","text":"这个加这个等于多少","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"none","reaction":""}
这个加这个等于多少，用手势回答 -> {"visual_route":"ANSWER_CURRENT_VIEW_WITH_GESTURE","speech":"generated","text":"根据当前画面计算答案","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"none","reaction":""}
不要挥手，说你好 -> {"visual_route":"NO_CURRENT_VIEW","speech":"verbatim","text":"你好","body_mode":"prohibit","body":"挥手","face":"","history":false,"reaction_mode":"none","reaction":""}
你好 -> {"visual_route":"NO_CURRENT_VIEW","speech":"generated","text":"你好","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"respond","reaction":"回应用户问候"}

保留方向、范围、对象、否定及多个通道。用户自述事实是generated而非要求复述；询问用户先前提供的姓名、偏好或事实才令history=true。所有固定字段必须存在，空字符串不能省略；输出最多256个token。'''

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
    elapsed_ms: float = 0
    voice_tone: str = "natural"
    voice_pace: str = "normal"
    # Internal compatibility field consumed by the action pipeline. The model
    # emits ``visual_route``; GENERAL is normalized to an empty string here.
    visual_scope_gate: str = ""

    @classmethod
    def parse(cls, raw: str, elapsed_ms=0, *, has_user_camera=False):
        if len(raw) > 4096:
            raise ValueError("intent too long")
        data = json.loads(raw)
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
        optional = {"voice_tone", "voice_pace"}
        if (
            not isinstance(data, dict)
            or not required <= set(data)
            or set(data) - required - optional
        ):
            raise ValueError("invalid intent fields")

        model_visual_route = data.pop("visual_route")
        visual_route = _MODEL_VISUAL_ROUTE_TO_INTERNAL.get(
            model_visual_route, model_visual_route
        )
        if visual_route not in _VISUAL_SCOPE_GATE_RESULTS:
            raise ValueError("invalid visual route")
        if visual_route != GENERAL_INTENT_GATE and not has_user_camera:
            raise ValueError("visual route requires a current user camera image")
        if (
            data.get("voice_tone", "natural") not in VOICE_TONES
            or data.get("voice_pace", "normal") not in VOICE_PACES
        ):
            raise ValueError("invalid voice plan")
        if (
            data["speech"] not in {"verbatim", "generated", "none"}
            or type(data["history"]) is not bool
        ):
            raise ValueError("invalid intent types")
        if data["body_mode"] not in {"perform", "prohibit", "none"}:
            raise ValueError("invalid body mode")
        if data["reaction_mode"] not in {"respond", "none"}:
            raise ValueError("invalid reaction mode")
        for key in ["text", "body", "face", "reaction"]:
            if not isinstance(data[key], str) or len(data[key]) > 512:
                raise ValueError("invalid intent content")

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
            else:
                data["face"] = target
        elif visual_route == VISUAL_GESTURE_ANSWER_GATE:
            data["speech"] = "generated"
            data["text"] = data["text"].strip() or "根据当前画面完成计算或推理"
            data["body_mode"] = "none"
            data["body"] = ""
            data["face"] = ""
            data["reaction_mode"] = "none"
            data["reaction"] = ""

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

        return cls(
            **data,
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
                "face_task": self.face,
                "speech_task": (
                    original
                    if self.speech == "generated" and original
                    else self.text
                ),
                "speech_kind": self.speech,
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
    request = GenerateRequest(
        model=session.model_name,
        messages=[
            Message(role="system", content=SYSTEM),
            Message(role="user", content=parts),
        ],
        sampling=SamplingParams(temperature=0, max_new_tokens=256),
        stream=False,
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
    try:
        result = await asyncio.wait_for(
            session.client.completion(request, request_id=request_id),
            timeout=TURN_INTENT_TIMEOUT_SECONDS,
        )
        raw_output = result.text or ""
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
            finish_reason=getattr(result, "finish_reason", None),
            usage=usage,
            output_chars=len(raw_output),
            output_sha256=hashlib.sha256(raw_output.encode()).hexdigest(),
        )
        intent = TurnIntent.parse(
            raw_output,
            (time.perf_counter() - started) * 1000,
            has_user_camera=has_user_camera,
        )
        route_code = intent.visual_scope_gate
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
            (getattr(result, "text", None) or "")
            if result is not None
            else None
        )
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
            finish_reason=(
                getattr(result, "finish_reason", None)
                if result is not None
                else None
            ),
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
