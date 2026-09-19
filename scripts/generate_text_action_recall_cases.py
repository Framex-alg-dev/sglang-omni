#!/usr/bin/env python3
"""Generate reviewable, text-only recall cases for the limited action catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang_omni.models.qwen3_omni.global_action_catalog import (
    load_global_action_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    ROOT / "sglang_omni/assets/character_limited_action_global_catalog.json"
)
DEFAULT_OUTPUT = (
    ROOT / "tests/unit_test/fixtures/realtime_text_action_recall_cases.json"
)


NATURAL_CASES = (
    ("idle-breathe", "请保持自然呼吸，安静待机。", "130"),
    ("head-shake", "连续摇摇头。", "135"),
    ("head-down", "把头低下来。", "137"),
    ("look-camera", "看着镜头。", "138"),
    ("look-down", "头别动，只把眼睛往下看。", "142"),
    ("eye-scan", "用眼睛左右扫视一下。", "143"),
    ("serious-face", "做一个严肃的表情。", "156"),
    ("surprised-face", "露出惊讶的表情。", "157"),
    ("afraid-face", "做个害怕的表情。", "159"),
    ("aggrieved-face", "做一个委屈的表情。", "160"),
    ("sad-face", "表现得悲伤一点。", "161"),
    ("confused-face", "露出疑惑的表情。", "162"),
    ("angry-face", "做出生气的表情。", "163"),
    ("pitiful-face", "低头咬唇抬眼，装一下可怜。", "164"),
    ("lean-forward", "身体往前倾一点。", "170"),
    ("nod-greeting", "轻轻点一下头表示致意。", "189"),
    ("raise-arm-side", "把一只手臂向侧面抬起来。", "194"),
    ("student-hand", "像学生回答问题那样举手。", "196"),
    ("arm-forward", "把一只手臂伸到前方。", "197"),
    ("arm-right", "把一只手臂伸向右边。", "199"),
    ("touch-head", "伸手摸摸自己的头。", "200"),
    ("one-hand-hip", "单手叉腰。", "201"),
    ("arms-crossed", "把两条手臂交叠起来。", "211"),
    ("extend-elbow", "把手肘伸直。", "220"),
    ("elbow-support", "用手肘支撑身体。", "223"),
    ("point-front", "用手指向你的正前方。", "225"),
    ("point-left", "帮我指一下左边。", "226"),
    ("point-up", "请指向上方。", "228"),
    ("point-down", "单手指一下下方。", "230"),
    ("point-self-one", "用一根手指指向自己。", "232"),
    ("point-self-two", "用双手的手指指向自己。", "233"),
    ("palms-self", "用双手手掌指向自己。", "234"),
    ("point-board", "指一下黑板。", "236"),
    ("palm-display", "单手摊开手掌做展示。", "237"),
    ("palm-receive", "手掌朝上做一个承接动作。", "242"),
    ("display-left-right", "手心朝上，从左到右展示一下。", "245"),
    ("finger-circle-down", "用手指朝下画一个圈。", "249"),
    ("finger-circle-front", "用手指在前方画一个圈。", "250"),
    ("finger-circle-up", "用手指朝上画一个圈。", "251"),
    ("flip-one-hand", "翻转一下单手手掌。", "253"),
    ("flip-two-hands", "双手同时翻掌。", "254"),
    ("hands-facing", "双手五指张开，掌心相对。", "255"),
    ("number-one", "比个数字一。", "258"),
    ("number-two", "用手表示数字二。", "259"),
    ("number-three", "给我比个三。", "260"),
    ("number-four", "用手比出数字四。", "261"),
    ("number-five", "比一个数字五的手势。", "262"),
    ("number-six", "给我比个六。", "263"),
    ("number-seven", "用手表示数字七。", "264"),
    ("number-eight", "比个数字八。", "265"),
    ("number-nine", "用手比出数字九。", "266"),
    ("letter-c", "单手比一个字母 C。", "268"),
    ("brackets", "用双手比一对括号。", "269"),
    ("thumb-up", "给我点个赞。", "270"),
    ("two-thumbs-up", "双手一起给我点赞。", "271"),
    ("victory-one", "单手比个耶。", "274"),
    ("victory-two", "双手比耶。", "275"),
    ("victory-face", "双手在脸旁比耶。", "277"),
    ("victory-wave", "比耶以后轻轻晃一晃手。", "278"),
    ("rabbit", "比个小兔子耳朵。", "279"),
    ("heart", "给我比个爱心。", "285"),
    ("flying-kiss", "给我一个飞吻吧。", "287"),
    ("wave", "挥挥手跟我打个招呼。", "288"),
    ("stop", "手掌向前推，表示停止。", "300"),
    ("cheer-fist", "握拳给我加油。", "305"),
    ("celebrate", "振臂欢呼一下。", "307"),
    ("beckon-one", "用一只手招呼我靠近。", "310"),
    ("beckon-two", "用双手招呼我过来。", "311"),
    ("salute", "向我敬个礼。", "316"),
    ("touch-nose", "摸一下鼻子。", "322"),
    ("touch-ear", "摸摸耳朵。", "323"),
    ("touch-behind-ear", "用手摸一下耳朵后面。", "324"),
    ("touch-forehead", "摸一下额头。", "326"),
    ("touch-face", "用一只手摸脸。", "335"),
    ("touch-face-two", "双手摸脸。", "336"),
    ("poke-cheek", "用一根手指戳一下脸颊。", "337"),
    ("fix-hair", "用手整理一下头发。", "344"),
    ("listen-ear", "手指向耳朵，示意我听。", "351"),
    ("listen-lean", "侧过耳朵认真听。", "352"),
    ("chin-hand", "单手托着脸颊。", "353"),
    ("chin-two-hands", "双手托腮。", "354"),
    ("reassure-chest", "拍拍胸口表示放心。", "355"),
    ("cover-chest", "双手捂住胸口。", "356"),
    ("one-fist", "单手握拳。", "367"),
    ("air-quotes", "用双手做一个双引号手势。", "369"),
    ("finger-flower", "用双手转一个花手。", "371"),
    ("finger-no", "伸出食指左右摇摆表示不行。", "375"),
    ("confident", "双手叉腰、挺胸抬头，表现得自信一点。", "438"),
    ("yawn", "单手捂嘴打个哈欠。", "449"),
    ("headache", "扶着头表现头疼。", "450"),
    ("facepalm", "单手捂脸表示无奈。", "455"),
    ("cover-mouth-laugh", "单手捂着嘴笑。", "462"),
    ("think-chin", "用手指轻抚下巴思考。", "465"),
    ("think-temple", "用一根手指抵住太阳穴思考。", "467"),
    ("cough", "捂嘴咳嗽一下。", "468"),
    ("tilt-shrug", "向左歪头并耸一下肩。", "472"),
    ("check-phone", "拿起手机看一下。", "578"),
    ("hand-chest", "单手轻抚胸口。", "624"),
    ("press-button", "按一下实体按钮。", "639"),
)


BOUNDARY_CASES = (
    ("say-score", "说二比一。", {"execute": False}),
    ("say-number-one", "请说数字一。", {"execute": False}),
    ("read-like", "请念出“点赞”这两个字。", {"execute": False}),
    ("capability-heart", "你会比心吗？", {"execute": False}),
    ("prohibit-like", "不要点赞。", {"execute": False}),
    ("prohibit-wave", "别挥手。", {"execute": False}),
    ("explicit-number-one", "比个数字一。", {"candidate_ids": ["258"], "execute": True}),
    ("explicit-like", "给我点个赞。", {"candidate_ids": ["270"], "execute": True}),
    ("explicit-heart", "双手比心。", {"candidate_ids": ["285"], "execute": True}),
    ("greeting", "你好呀。", {"candidate_ids": ["288"], "execute": True}),
    ("farewell", "拜拜。", {"candidate_ids": ["288"], "execute": True}),
    ("reaction-surprise", "我中奖了！", {"candidate_ids": ["157"], "execute": True}),
)


UNSUPPORTED_CASES = (
    "后空翻",
    "倒立行走",
    "原地劈叉",
    "连续转三个圈",
    "跳一段芭蕾舞",
    "用左手写字",
    "拿起水杯喝水",
    "弹一段钢琴",
    "蹲下来系鞋带",
    "双脚向前跳两步",
)


def build_cases(catalog_path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = load_global_action_catalog(catalog_path)
    action_by_id = {
        action["candidate_id"]: (category, action)
        for category in raw["categories"]
        for action in category["children"]
    }
    cases: list[dict] = []
    for candidate_id, (category, action) in action_by_id.items():
        cases.append(
            {
                "id": f"canonical-{candidate_id}",
                "group": "canonical_label",
                "text": f"请做出“{action['source_label']}”这个动作。",
                "expected": {
                    "candidate_ids": [candidate_id],
                    "execute": True,
                    "support_status": "supported",
                },
                "category_id": category["category_id"],
                "source_label": action["source_label"],
            }
        )
    for case_id, text, candidate_id in NATURAL_CASES:
        if candidate_id not in action_by_id:
            raise ValueError(f"unknown natural-case candidate_id: {candidate_id}")
        category, action = action_by_id[candidate_id]
        cases.append(
            {
                "id": f"natural-{case_id}",
                "group": "natural_request",
                "text": text,
                "expected": {
                    "candidate_ids": [candidate_id],
                    "execute": True,
                    "support_status": "supported",
                },
                "category_id": category["category_id"],
                "source_label": action["source_label"],
            }
        )
    for case_id, text, expected in BOUNDARY_CASES:
        cases.append(
            {
                "id": f"boundary-{case_id}",
                "group": "intent_boundary",
                "text": text,
                "expected": expected,
            }
        )
    for index, action_text in enumerate(UNSUPPORTED_CASES, start=1):
        cases.append(
            {
                "id": f"unsupported-{index:02d}",
                "group": "unsupported_request",
                "text": f"请{action_text}。",
                "expected": {
                    "candidate_ids": ["UNSUPPORTED"],
                    "execute": False,
                    "support_status": "unsupported",
                },
            }
        )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return (
        {
            "schema_version": 1,
            "description": "Text-only, isolated-session recall evaluation for the limited action catalog.",
            "catalog_path": str(catalog_path.relative_to(ROOT)),
            "catalog_hash": catalog.catalog_hash,
            "catalog_candidate_count": catalog.candidate_count,
            "groups": {
                "canonical_label": "One exact catalog-label command per action; measures catalog coverage.",
                "natural_request": "Human-authored paraphrases; primary natural-language accuracy set.",
                "intent_boundary": "Minimal pairs for speech-only, capability, prohibition, greeting, and reaction routing.",
                "unsupported_request": "Explicit actions absent from the limited catalog.",
            },
        },
        cases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metadata, cases = build_cases(args.catalog.resolve())
    payload = {**metadata, "case_count": len(cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = {
        group: sum(case["group"] == group for case in cases)
        for group in metadata["groups"]
    }
    print(json.dumps({"output": str(args.output), "cases": len(cases), "groups": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
