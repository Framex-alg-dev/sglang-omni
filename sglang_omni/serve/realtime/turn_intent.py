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
_VISUAL_ROUTE_PREFIX_RE = re.compile(
    r'"visual_route"\s*:\s*"(?P<route>[A-Z0-9_]+)"'
)
_PURE_DEICTIC_COPY_ROUTES = {
    # This is deliberately a narrow semantic consistency repair, not a second
    # intent classifier. It only covers complete, unambiguous imperatives that
    # the model transcribed into ``text`` while contradicting itself with
    # NO_CURRENT_VIEW/body_mode=none.
    "COPY_ACTION": frozenset(
        {
            "做这个动作",
            "模仿这个动作",
            "复刻这个动作",
            "重复这个动作",
            "照着我做",
            "do this",
            "copy this action",
            "imitate this action",
        }
    ),
    "COPY_HAND": frozenset(
        {
            "做这个手势",
            "比这个手势",
            "比这个数字",
            "做这个数字",
            "比个这个",
            "模仿这个手势",
            "复刻这个手势",
            "重复这个手势",
            "do this gesture",
            "make this gesture",
            "copy this gesture",
            "imitate this gesture",
        }
    ),
}
_PURE_TASK_TRAILING_PUNCTUATION_RE = re.compile(r"[\s。！？!?，,；;：:]+$")


def _repair_contradictory_copy_route(
    data: dict,
    visual_route: str,
    *,
    has_user_camera: bool,
) -> str:
    """Repair only an exact pure-copy imperative contradicted by its fields."""

    if (
        not has_user_camera
        or visual_route != GENERAL_INTENT_GATE
        or data.get("speech") != "generated"
        or data.get("body_mode") != "none"
        or data.get("body")
        or data.get("face")
    ):
        return visual_route
    task = data.get("text")
    if not isinstance(task, str):
        return visual_route
    normalized_task = _PURE_TASK_TRAILING_PUNCTUATION_RE.sub(
        "", task.strip().lower()
    )
    for repaired_route, phrases in _PURE_DEICTIC_COPY_ROUTES.items():
        if normalized_task in phrases:
            # A complete pure-action imperative must not leak into the spoken
            # reply channel after the visual route is repaired.
            data["speech"] = "none"
            data["text"] = ""
            return repaired_route
    return visual_route


def _visual_route_from_partial_output(
    raw: str,
    *,
    has_user_camera: bool,
) -> str | None:
    """Return the first complete visual route emitted by unified intent.

    This is only an early scheduling hint. The completed JSON still goes
    through ``TurnIntent.parse`` before any action can be published.
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
    return "" if route == GENERAL_INTENT_GATE else route

SYSTEM = '''你是数字人意图解析器。理解当前中英文音频或文本，不识别图片；只输出JSON。

has_user_camera=true/false是服务端事实。只有true才允许需要当前画面的visual_route；有图片不等于要求模仿。

首字段必须是visual_route。非视觉计算输出speech、text、body_mode、body、face、history、reaction_mode、reaction全部字段。
视觉计算只输出visual_route、visual_answer_operation、visual_answer_output；额外原样话术才追加speech=verbatim和text，明确表情/语气/语速才追加face/voice_tone/voice_pace，其余空字段省略。数字手势占用身体动作位；若另有身体动作，必须追加body_mode/body而不得丢弃。
visual_route只能是：NO_CURRENT_VIEW、COPY_CURRENT_ACTION、COPY_CURRENT_HAND、COPY_CURRENT_FACE、COPY_CURRENT_HEAD、COPY_CURRENT_ARM、COPY_CURRENT_UPPER_BODY、COPY_CURRENT_LEG、COPY_CURRENT_BODY、COPY_CURRENT_POSE、COPY_CURRENT_OBJECT、COPY_CURRENT_SCREEN、ANSWER_CURRENT_VIEW_WITH_GESTURE。

visual_route先于其他字段按以下互斥顺序判定：
1. 复现当前画面：执行词（做、比、模仿、复刻、重复、照着做；do/copy/imitate）和当前指代（这个、这样、照着我、和我一样；this/like me）同时出现，必须选COPY_CURRENT_*，不能把命令塞进speech/text后选NO_CURRENT_VIEW。“做这个动作/Do this”=COPY_CURRENT_ACTION；“做这个手势/比这个数字/比个这个/Do this gesture”=COPY_CURRENT_HAND。
2. 视觉四则运算：要求对当前画面中的两个指代数字做加、减、乘、除时，选ANSWER_CURRENT_VIEW_WITH_GESTURE。未指定回答方式时默认手势加语音；明确“用手势回答/只用手势/不要说话”时仅手势。
3. 其他情况选NO_CURRENT_VIEW：普通问答、画面识别或描述、能力询问、禁止动作、名称明确且不需照抄画面的动作。“挥手、点头、比个心、比数字二、做个手势”均属于此类。

最小对比：“做个手势”是任意手势，选NO_CURRENT_VIEW并提取动作；“做这个手势/比这个数字”指向画面，必须COPY_CURRENT_HAND。“比数字二”选NO_CURRENT_VIEW；“比个这个”必须COPY_CURRENT_HAND。“不要做这个动作”和“做这个动作是什么意思”不得转COPY。“这是什么手势/这是数字几”选NO_CURRENT_VIEW；“这个加这个等于多少”依赖当前画面，必须选ANSWER_CURRENT_VIEW_WITH_GESTURE；“一加二等于多少”不依赖当前画面，选NO_CURRENT_VIEW。

COPY范围：HAND=手势/手型，FACE=表情/脸，HEAD=头部/视线，ARM=手臂，UPPER_BODY=肩膀/躯干，LEG=腿脚，BODY=全身，POSE=姿态，OBJECT=物品交互，SCREEN=屏幕交互；范围不明才用ACTION。gesture/hand sign属于HAND。has_user_camera=false时选NO_CURRENT_VIEW、generated、body_mode=none，body/face为空并说明需要画面。

一致性：模仿与说话可以同时存在，visual_route仍为COPY；纯复现speech=none。“做这个动作/手势”不得选NO_CURRENT_VIEW。视觉计算必须带visual_answer_operation：加/减/乘/除对应add/subtract/multiply/divide，禁止缺省，减除保留顺序；必须带visual_answer_output：仅手势或不播报答案=gesture_only，未指定或手势加语音播报答案=gesture_and_speech；仅语音回答则选NO_CURRENT_VIEW并省略两字段。“并说出来”只由visual_answer_output表达；额外原样话术的speech/text与答案播报独立。明确的表情、语气、语速必须保留，未指定时不得编造。

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
能挥挥手吗 -> {"visual_route":"NO_CURRENT_VIEW","speech":"none","text":"","body_mode":"perform","body":"挥手","face":"","history":false,"reaction_mode":"none","reaction":""}
你会挥手吗 -> {"visual_route":"NO_CURRENT_VIEW","speech":"generated","text":"你会挥手吗","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"none","reaction":""}
模仿这个手势并说你好 -> {"visual_route":"COPY_CURRENT_HAND","speech":"verbatim","text":"你好","body_mode":"perform","body":"这个手势","face":"","history":false,"reaction_mode":"none","reaction":""}
做这个动作并介绍自己 -> {"visual_route":"COPY_CURRENT_ACTION","speech":"generated","text":"介绍自己","body_mode":"perform","body":"这个动作","face":"","history":false,"reaction_mode":"none","reaction":""}
这个加这个等于多少，用手势回答 -> {"visual_route":"ANSWER_CURRENT_VIEW_WITH_GESTURE","visual_answer_operation":"add","visual_answer_output":"gesture_only"}
这个除以这个等于多少 -> {"visual_route":"ANSWER_CURRENT_VIEW_WITH_GESTURE","visual_answer_operation":"divide","visual_answer_output":"gesture_and_speech"}
微笑着用手势回答这个加这个并说出来 -> {"visual_route":"ANSWER_CURRENT_VIEW_WITH_GESTURE","visual_answer_operation":"add","visual_answer_output":"gesture_and_speech","face":"微笑"}
用手势回答这个加这个，并说你好 -> {"visual_route":"ANSWER_CURRENT_VIEW_WITH_GESTURE","visual_answer_operation":"add","visual_answer_output":"gesture_only","speech":"verbatim","text":"你好"}
你好 -> {"visual_route":"NO_CURRENT_VIEW","speech":"generated","text":"你好","body_mode":"none","body":"","face":"","history":false,"reaction_mode":"respond","reaction":"回应用户问候"}

保留方向、范围、对象、否定及多个通道。用户自述事实是generated而非要求复述；询问用户先前提供的姓名、偏好或事实才令history=true。非视觉计算的固定字段必须存在；视觉计算必须是以上稀疏格式，不得输出未激活的空字段；输出最多256个token。'''

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
    visual_answer_operation: str = ""
    visual_answer_output: str = ""
    # Internal compatibility field consumed by the action pipeline. The model
    # emits ``visual_route``; GENERAL is normalized to an empty string here.
    visual_scope_gate: str = ""

    @classmethod
    def parse(cls, raw: str, elapsed_ms=0, *, has_user_camera=False):
        if len(raw) > 4096:
            raise ValueError("intent too long")
        data = json.loads(raw)
        if not isinstance(data, dict) or "visual_route" not in data:
            raise ValueError("invalid intent fields")

        model_visual_route = data["visual_route"]
        visual_route = _MODEL_VISUAL_ROUTE_TO_INTERNAL.get(
            model_visual_route, model_visual_route
        )
        if visual_route not in _VISUAL_SCOPE_GATE_RESULTS:
            raise ValueError("invalid visual route")

        full_required = {
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
        visual_required = {
            "visual_route",
            "visual_answer_operation",
            "visual_answer_output",
        }
        allowed = full_required | {
            "voice_tone",
            "voice_pace",
            "visual_answer_operation",
            "visual_answer_output",
        }
        required = (
            visual_required
            if visual_route == VISUAL_GESTURE_ANSWER_GATE
            else full_required
        )
        if not required <= set(data) or set(data) - allowed:
            raise ValueError("invalid intent fields")

        data.pop("visual_route")
        if visual_route == VISUAL_GESTURE_ANSWER_GATE:
            # The visual-arithmetic wire format is sparse. Missing common
            # fields mean that the user did not activate those channels; an
            # explicitly emitted value is preserved and validated below.
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
        if visual_route != GENERAL_INTENT_GATE and not has_user_camera:
            raise ValueError("visual route requires a current user camera image")
        if (
            data.get("voice_tone", "natural") not in VOICE_TONES
            or data.get("voice_pace", "normal") not in VOICE_PACES
        ):
            raise ValueError("invalid voice plan")
        visual_answer_operation = data.get("visual_answer_operation", "")
        visual_answer_output = data.get("visual_answer_output", "")
        if visual_route == VISUAL_GESTURE_ANSWER_GATE:
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
        elif visual_answer_output or visual_answer_operation:
            raise ValueError("visual answer fields require visual answer route")
        else:
            data["visual_answer_operation"] = ""
            data["visual_answer_output"] = ""
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

        visual_route = _repair_contradictory_copy_route(
            data,
            visual_route,
            has_user_camera=has_user_camera,
        )

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
            # The numeric gesture already owns the single body-action slot.
            # Reject a second body directive instead of silently overwriting
            # user intent. Face, voice and explicit speech remain independent.
            if data["body_mode"] != "none" or data["body"]:
                raise ValueError(
                    "visual answer cannot combine with another body action"
                )
            if data["reaction_mode"] != "none" or data["reaction"]:
                raise ValueError(
                    "visual answer cannot combine with an implicit reaction"
                )

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
        sampling=SamplingParams(temperature=0, max_new_tokens=256),
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
                            if (
                                visual_scope_future is not None
                                and not visual_scope_future.done()
                            ):
                                early_route = _visual_route_from_partial_output(
                                    "".join(text_parts),
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
