#!/usr/bin/env python3
"""Materialize the reviewed realtime hierarchical-action recall fixture.

The generated JSON is committed and consumed as a fixed oracle.  This script is
an authoring aid only: regenerating it after a catalog change requires review of
the resulting diff rather than automatic acceptance of new expectations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "sglang_omni/assets/character_action_global_catalog.json"
DEFAULT_OUTPUT = (
    ROOT / "tests/unit_test/fixtures/realtime_action_recall_cases.json"
)

CHANGED_CATEGORY_IDS = frozenset(
    {
        "13", "21", "22", "24", "25", "29", "30",
        "33", "34", "36", "37", "38", "45", "47",
        "48", "50", "51", "52", "53", "54", "55",
        "56", "57", "58", "59",
    }
)

# Natural requests concentrate on the boundaries changed in the candidate
# catalog.  Keep the expected concrete action explicit so the same corpus can
# measure category recall and end-to-end Child accuracy.
CURATED_CHANGED_CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "13": (
        ("先别比划，保持自然呼吸就好", "130"),
        ("在椅子上稍微挪一下坐稳", "132"),
        ("坐着轻轻左右转一下椅子", "133"),
    ),
    "21": (
        ("把上半身朝镜头凑近一点", "170"),
        ("身体往后靠一些", "171"),
        ("从画面边上探出半张脸看看", "172"),
    ),
    "22": (
        ("腰和上身一起向左边侧过去", "174"),
        ("脚别动，只把整个上身转向右侧", "177"),
        ("让上半身左右连续晃几下", "178"),
    ),
    "24": (
        ("轻轻点一下头向大家致意", "189"),
        ("双手放在身体两边，正式地深鞠一躬", "190"),
        ("双手交叠在小腹前，弯腰感谢观众", "191"),
    ),
    "25": (
        ("只把一边的胳膊平举到身体侧面", "194"),
        ("像课堂回答问题那样举起一只手", "196"),
        ("用一只手叉在腰上站好", "201"),
    ),
    "29": (
        ("伸出食指明确指向你身体左边", "226"),
        ("用手指点着自己的胸口", "232"),
        ("抬手指给大家看黑板上的位置", "236"),
    ),
    "30": (
        ("掌心朝上，从左往右扫过来介绍这一片区域", "245"),
        ("用食指在正前方完整画一个圆", "250"),
        ("两只空手掌心向上摊开，展示整个范围", "238"),
    ),
    "33": (
        ("说到重点时用一只手用力拍一下桌面", "293"),
        ("把一只手掌向前推，明确示意停下", "300"),
        ("双臂在胸前交叉成叉，表示拒绝", "302"),
    ),
    "34": (
        ("为刚才的表现拍几下手鼓励一下", "304"),
        ("双臂举过头顶用力挥舞欢呼", "307"),
        ("面向镜头伸出手掌来击个掌", "308"),
    ),
    "36": (
        ("双手在胸前合掌，低头表达感谢", "314"),
        ("一手握拳、另一手包住拳头行个礼", "315"),
        ("四指并拢抬到眉梢敬个礼", "316"),
    ),
    "37": (
        ("用指尖轻轻揉一揉自己的耳廓和耳垂", "323"),
        ("把散下来的头发整理拨到一边", "340"),
        ("手掌贴在耳边，偏着头仔细听", "352"),
    ),
    "38": (
        ("用拇指和中指摩擦弹一下，打个响指", "365"),
        ("把两只手的手指交错扣在一起", "366"),
        ("面前没有琴，单手在空中模拟弹奏", "372"),
    ),
    "45": (
        ("面前有钢琴，只用一只手按琴键演奏", "424"),
        ("抱着吉他，一只手按弦另一只手拨弦", "426"),
        ("双手撑地，让身体向侧面翻转一周", "421"),
    ),
    "47": (
        ("双手做成猫爪托着脸，歪头装可爱", "442"),
        ("吓得双手抱住头，身体也往后缩", "460"),
        ("视线移开，用一根手指慢慢抚过下巴思考", "465"),
    ),
    "48": (
        ("用食指贴着虚拟屏幕横向滑过去", "475"),
        ("在屏幕上用拇指和食指向外撑开放大内容", "476"),
        ("伸手点击前方界面里的虚拟按钮", "474"),
    ),
    "50": (
        ("手伸到桌面，把原本放着的东西抓起来", "486"),
        ("把手伸进口袋，取出里面的东西", "495"),
        ("伸手到上方架子，把东西拿下来", "498"),
    ),
    "51": (
        ("把已经拿在手里的东西抱紧在胸前", "503"),
        ("握住袋子的提手，让袋子垂在身边", "506"),
        ("双手从下面稳稳托着已经装好东西的托盘", "510"),
    ),
    "52": (
        ("把手里的商品靠近镜头，让我看清细节", "519"),
        ("转动手中物品，把正面和背面都展示出来", "521"),
        ("撕开包装，取出里面的东西后立刻向前展示", "524"),
    ),
    "53": (
        ("从对方手里用一只手接过递来的东西", "531"),
        ("先接过对方递来的物品，然后马上举起来展示", "535"),
        ("把手里的东西递给站在你左边的人", "527"),
    ),
    "54": (
        ("把手中物品向下放到桌面上", "537"),
        ("把好几个东西一个挨一个并排摆好", "553"),
        ("把这些物品一层一层叠起来", "554"),
    ),
    "55": (
        ("把已经拿着的东西绕到背后藏起来", "559"),
        ("把手中的小物件塞进口袋收好", "560"),
        ("手里本来就拿着东西，从画面外把它带进镜头", "565"),
    ),
    "56": (
        ("拿杯子喝一口水，我没有指定别的容器", "571"),
        ("拿起耳机，把它戴到耳朵上", "582"),
        ("捏住纸质书的一角，翻到下一页", "570"),
    ),
    "57": (
        ("耳机已经戴着了，用一只手把耳机位置调好", "625"),
        ("用手指摸一摸另一只手上戴着的戒指", "620"),
        ("把一只手插进衣兜里，保持放松姿势", "627"),
    ),
    "58": (
        ("只把盒子的盖子掀开，不要取出或展示里面的东西", "635"),
        ("把包上的拉链拉开，但不要从里面拿东西", "642"),
        ("双手抓住包装两端，把包装直接撕开", "644"),
    ),
    "59": (
        ("握住瓶身上下摇几下，把里面的液体混匀", "646"),
        ("拿着布在物体表面来回擦干净", "647"),
        ("把商品条码靠近扫描区域停一下识别", "648"),
    ),
}

SYSTEM_CASES = (
    ("reply_weather", "给我解释一下今天为什么这么热", "01"),
    ("reply_story", "讲一个很短的睡前故事", "01"),
    ("reply_translation", "把早上好翻译成英文", "01"),
    ("reply_knowledge", "海水为什么是咸的", "01"),
    ("reply_repeat", "只说一句欢迎回来", "01"),
    ("silent_wait", "先安静陪着我，不要说话，也不要做明显动作", "02"),
    ("silent_observe", "保持安静看着就好，不用回答", "02"),
    ("silent_no_gesture", "别出声，也别做表达性手势", "02"),
    ("silent_rest", "安静待一会儿，只要自然呼吸", "02"),
    ("silent_reset", "不用回复，恢复到自然安静的状态", "02"),
)

UNSUPPORTED_CASES = (
    ("restricted_dance", "现在跳一段现代舞", ("13", "14")),
    ("restricted_walk", "向前走到镜头旁边", ("18", "19")),
    ("restricted_drink", "拿杯子喝一口水", ("29", "30")),
    ("restricted_write", "拿笔在纸上写几个字", ("21", "22")),
    ("restricted_bow", "正式地鞠一躬", ("31", "32")),
    ("restricted_wave", "抬手向我挥手告别", ("15", "16")),
    ("restricted_pickup", "从桌上拿起那个盒子", ("51", "52")),
    ("restricted_screen", "在虚拟屏幕上向左滑动", ("53", "54")),
    ("restricted_applause", "为大家鼓掌庆祝", ("10", "11")),
    ("restricted_glasses", "把眼镜戴到脸上", ("45", "46")),
)


def _case(
    *,
    case_id: str,
    text: str,
    category_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...] = (),
    group: str,
    coverage_kind: str,
    source: str,
    changed_category: bool = False,
    allowed_category_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    expected_support = "unsupported" if category_ids == ("00",) else "supported"
    return {
        "id": case_id,
        "locale": "zh-CN",
        "input": {"text": text},
        "allowed_category_ids": list(allowed_category_ids),
        "expected": {
            "acceptable_category_ids": list(category_ids),
            "acceptable_candidate_ids": list(candidate_ids),
            "support_status": expected_support,
        },
        "group": group,
        "coverage_kind": coverage_kind,
        "source": source,
        "changed_category": changed_category,
    }


def build_cases(catalog: dict[str, object]) -> list[dict[str, object]]:
    categories = catalog.get("categories")
    if not isinstance(categories, list):
        raise ValueError("global catalog has no categories")
    membership_count = Counter(
        child["candidate_id"]
        for category in categories
        for child in category["children"]
    )
    cases: list[dict[str, object]] = []
    for category in categories:
        category_id = category["category_id"]
        if category_id in {"01", "02"}:
            continue
        children = [
            child
            for child in category["children"]
            if membership_count[child["candidate_id"]] == 1
        ] or list(category["children"])
        seeds = children[:3]
        while len(seeds) < 3:
            seeds.append(children[len(seeds) % len(children)])
        seed_prefixes = (
            "请按这个动作细节来做",
            "请完成下面描述的身体动作",
            "现在照这个执行细节做一次",
            "我希望看到这样的动作",
            "请把以下动作表现出来",
        )
        for index, child in enumerate(seeds, 1):
            detail = child["short_definition"].rstrip("。")
            prefix_index = (int(category_id[1:]) + index) % len(seed_prefixes)
            cases.append(
                _case(
                    case_id=f"seed-{category_id.lower()}-{index}",
                    text=f"{seed_prefixes[prefix_index]}：{detail}",
                    category_ids=(category_id,),
                    candidate_ids=(child["candidate_id"],),
                    group="catalog_definition_seed",
                    coverage_kind="normal",
                    source="catalog_definition_seed",
                    changed_category=category_id in CHANGED_CATEGORY_IDS,
                )
            )

    for category_id, entries in CURATED_CHANGED_CASES.items():
        for index, (text, candidate_id) in enumerate(entries, 1):
            cases.append(
                _case(
                    case_id=f"boundary-{category_id.lower()}-{index}",
                    text=text,
                    category_ids=(category_id,),
                    candidate_ids=(candidate_id,),
                    group="changed_category_boundary",
                    coverage_kind="boundary" if index < 3 else "adversarial",
                    source="reviewed_boundary",
                    changed_category=True,
                )
            )

    for case_id, text, category_id in SYSTEM_CASES:
        cases.append(
            _case(
                case_id=case_id,
                text=text,
                category_ids=(category_id,),
                group="system_route",
                coverage_kind="boundary",
                source="reviewed_system_route",
            )
        )
    for case_id, text, allowed_category_ids in UNSUPPORTED_CASES:
        cases.append(
            _case(
                case_id=case_id,
                text=text,
                category_ids=("00",),
                group="restricted_category_unsupported",
                coverage_kind="adversarial",
                source="reviewed_restricted_catalog",
                allowed_category_ids=allowed_category_ids,
            )
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    cases = build_cases(catalog)
    document = {
        "schema_version": 1,
        "description": (
            "Fixed, reviewable inputs for Category Top-1/Top-2 recall and "
            "end-to-end Child accuracy. Report metrics separately by group."
        ),
        "notes": [
            "catalog_definition_seed is broad coverage, not a natural-language quality claim.",
            "changed_category_boundary is the primary set for category-description A/B decisions.",
            "An empty acceptable_candidate_ids list means only category routing is judged.",
            "allowed_category_ids narrows the Session catalog for 00 rejection cases.",
            "The committed fixture is an oracle; regenerate only with human review of its diff.",
        ],
        "changed_category_ids": sorted(CHANGED_CATEGORY_IDS),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(cases),
                "groups": Counter(case["group"] for case in cases),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
