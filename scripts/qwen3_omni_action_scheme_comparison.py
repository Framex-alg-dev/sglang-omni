
#!/usr/bin/env python3
"""Run three action suffix scoring schemes on the same multimodal sessions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Any

from scripts.qwen3_omni_atomic_action_smoke import (
    ACTION_PREFIX,
    DEFAULT_BASE_URL,
    DEFAULT_CATALOG,
    DEFAULT_COMMON_ACTIONS_DIR,
    DEFAULT_CONTAINER_COMMON_ACTIONS_DIR,
    _chat_turn,
    _history_turn_content,
    _json_request,
    build_action_context_messages,
    build_catalog_candidates,
    build_common_action_scenarios,
    load_action_catalog,
)

DEFAULT_MODEL = os.environ.get("QWEN3_OMNI_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
NATURAL = "natural_suffix"
SHORT_ID = "candidate_list_short_id"
SOURCE_LABEL = "source_label_suffix"
SCHEMES = (NATURAL, SHORT_ID, SOURCE_LABEL)
PREFIX_HEAD = (
    "请根据以上完整对话和当前数字人状态，从候选动作列表中选择唯一一个最合适的动作。"
    "候选动作列表："
)
PREFIX_TAIL = (
    "。只允许输出一个 action_id，不要解释；没有明确动作指令时必须输出 none。"
    "下一步 action_id 是："
)


def compact_candidates(catalog: dict[str, dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    result = []
    for index, label in enumerate(labels, 1):
        item = catalog[label]
        definition = item.get("short_definition") or label
        result.append({
            "candidate_id": f"a{index:02d}",
            "suffix": f"a{index:02d}",
            "description": f"{label}：{definition}",
            "action_id": item["action_id"],
            "source_label": label,
            "short_definition": definition,
            "category_path": list(item.get("category_path") or []),
            "action_type": item.get("action_type", ""),
            "execution_binding": {},
        })
    result.append({
        "candidate_id": "none",
        "suffix": "none",
        "description": "不做动作：不需要做任何动作",
        "action_id": "no_action",
        "source_label": "no_action",
        "short_definition": "不需要做任何动作",
        "category_path": [],
        "action_type": "control",
        "execution_binding": {},
    })
    return result


def label_candidates(catalog: dict[str, dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    """Use each catalog source_label itself as the scored suffix."""
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")
    result = []
    for label in labels:
        item = catalog[label]
        definition = item.get("short_definition") or label
        result.append({
            "candidate_id": label,
            "suffix": label,
            "description": f"{label}：{definition}",
            "action_id": item["action_id"],
            "source_label": label,
            "short_definition": definition,
            "category_path": list(item.get("category_path") or []),
            "action_type": item.get("action_type", ""),
            "execution_binding": {},
        })
    result.append({
        "candidate_id": "no_action",
        "suffix": "no_action",
        "description": "不做动作：不需要做任何动作",
        "action_id": "no_action",
        "source_label": "no_action",
        "short_definition": "不需要做任何动作",
        "category_path": [],
        "action_type": "control",
        "execution_binding": {},
    })
    return result


def candidates_for(catalog: dict[str, dict[str, Any]], labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    return {
        NATURAL: build_catalog_candidates(catalog, labels),
        SHORT_ID: compact_candidates(catalog, labels),
        SOURCE_LABEL: label_candidates(catalog, labels),
    }


def make_prefix(scheme: str, candidates: list[dict[str, Any]]) -> str:
    if scheme in (NATURAL, SOURCE_LABEL):
        return ACTION_PREFIX
    mapping = "; ".join(
        f"{x['candidate_id']}={x['source_label']}（{x['short_definition']}）"
        for x in candidates
    )
    return f"{PREFIX_HEAD}{mapping}{PREFIX_TAIL}"


def make_history(base_url: str, model: str, scenario: dict[str, Any], timeout: float) -> dict[str, Any]:
    session_id = f"scheme-compare-{scenario['id']}"
    history: list[dict[str, Any]] = []
    audios: list[str] = []
    images: list[str] = []
    turns: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, turn in enumerate(scenario["turns"], 1):
        content = turn if isinstance(turn, str) else turn["content"]
        turn_audios = [] if isinstance(turn, str) else list(turn.get("audios", []))
        turn_images = [] if isinstance(turn, str) else list(turn.get("images", []))
        response, assistant = _chat_turn(
            base_url,
            model,
            [*history, {"role": "user", "content": content}],
            audios=[*audios, *turn_audios],
            images=[*images, *turn_images],
            request_id=f"{session_id}-chat-{index}",
            timeout=timeout,
        )
        turns.append({
            "turn_index": index,
            "user": content,
            "audios": turn_audios,
            "images": turn_images,
            "status_code": response["status_code"],
            "assistant": assistant,
            "response": response["body"],
        })
        if response["status_code"] != 200 or not assistant.strip():
            errors.append(f"chat turn {index} failed")
            break
        history.extend([
            {"role": "user", "content": _history_turn_content(content, turn_audios, turn_images)},
            {"role": "assistant", "content": assistant},
        ])
        audios.extend(turn_audios)
        images.extend(turn_images)
    action_history = history[:-1] if history and history[-1]["role"] == "assistant" else list(history)
    return {
        "session_id": session_id,
        "turns": turns,
        "history": action_history,
        "history_audios": audios,
        "history_images": images,
        "avatar_state": scenario.get(
            "avatar_state",
            {"pose": "seated", "gaze": "camera", "hands": "resting"},
        ),
        "errors": errors,
    }


def enrich(body: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {x["candidate_id"]: x for x in candidates}
    ranking = []
    for score in body.get("scores") or []:
        candidate = by_id.get(score.get("candidate_id"), {})
        tokens = []
        for token in score.get("token_scores") or []:
            logprob = float(token["logprob"])
            tokens.append({
                "token_id": int(token["token_id"]),
                "logprob": logprob,
                "probability": math.exp(logprob),
                "nll": -logprob,
            })
        total = math.fsum(x["logprob"] for x in tokens)
        ranking.append({
            "candidate_id": score.get("candidate_id"),
            "action_id": candidate.get("action_id"),
            "source_label": candidate.get("source_label"),
            "suffix": candidate.get("suffix"),
            "short_definition": candidate.get("short_definition"),
            "token_count": int(score.get("token_count", len(tokens))),
            "sum_logprob": total,
            "mean_logprob": float(score["mean_logprob"]),
            "mean_nll": float(score["mean_nll"]),
            "ppl": float(score["ppl"]),
            "raw_logit_available": False,
            "token_scores": tokens,
        })
    ranking.sort(key=lambda x: x["mean_logprob"], reverse=True)
    for rank, item in enumerate(ranking, 1):
        item["rank"] = rank
    top1 = ranking[0] if ranking else None
    top2 = ranking[1] if len(ranking) > 1 else None
    return {
        "prefix_cached": body.get("prefix_cached"),
        "stats": body.get("stats") or {},
        "ranking": ranking,
        "top1": top1,
        "top2": top2,
        "top1_margin_mean_logprob": (
            top1["mean_logprob"] - top2["mean_logprob"] if top1 and top2 else None
        ),
        "raw_response": body,
    }


def score(
    base_url: str,
    model: str,
    scenario: dict[str, Any],
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
    scheme: str,
    timeout: float,
) -> dict[str, Any]:
    prefix = make_prefix(scheme, candidates)
    prompt_messages = build_action_context_messages(
        context["history"],
        prefix,
        context["avatar_state"],
        audio_count=len(context["history_audios"]),
        image_count=len(context["history_images"]),
    )
    media = scenario.get("score_media", {})
    payload = {
        "request_id": f"{context['session_id']}-{scheme}-score",
        "model": model,
        "prefix": prefix,
        "language": "zh",
        "sample_rate": 16000,
        "micro_batch_size": 32,
        "session_id": context["session_id"],
        "history": context["history"],
        "history_audios": context["history_audios"],
        "history_images": context["history_images"],
        "audios": list(media.get("audios", [])),
        "images": list(media.get("images", [])),
        "avatar_state": context["avatar_state"],
        "candidates": [
            {
                "candidate_id": x["candidate_id"],
                "suffix": x["suffix"],
                "action_id": x["action_id"],
                "execution_binding": x.get("execution_binding", {}),
            }
            for x in candidates
        ],
    }
    response = _json_request(base_url, "/v1/action-scores", payload, timeout=timeout)
    result: dict[str, Any] = {
        "scheme": scheme,
        "prefix": prefix,
        "candidate_count": len(candidates),
        "candidate_map": [
            {
                "candidate_id": x["candidate_id"],
                "suffix": x["suffix"],
                "source_label": x["source_label"],
                "action_id": x["action_id"],
                "short_definition": x["short_definition"],
            }
            for x in candidates
        ],
        "prompt_messages": prompt_messages,
        "request_payload": payload,
        "status_code": response["status_code"],
        "response": response["body"],
    }
    if response["status_code"] == 200 and isinstance(response["body"], dict):
        result.update(enrich(response["body"], candidates))
    else:
        result.update({
            "prefix_cached": None,
            "stats": {},
            "ranking": [],
            "top1": None,
            "top2": None,
            "top1_margin_mean_logprob": None,
        })
    expected = scenario.get("expected_action")
    top1 = result.get("top1") or {}
    result["expected_action"] = expected
    result["top1_matches_expected"] = expected is None or top1.get("source_label") == expected
    return result


def run_one(base_url: str, model: str, scenario: dict[str, Any], by_scheme: dict[str, list[dict[str, Any]]], timeout: float) -> dict[str, Any]:
    context = make_history(base_url, model, scenario, timeout)
    result = {
        "scenario_id": scenario["id"],
        "expected_action": scenario.get("expected_action"),
        "audio_sample": scenario.get("audio_sample"),
        "session_id": context["session_id"],
        "turns": context["turns"],
        "history_for_scoring": context["history"],
        "history_audios": context["history_audios"],
        "history_images": context["history_images"],
        "avatar_state": context["avatar_state"],
        "chat_errors": context["errors"],
        "schemes": {},
    }
    if context["errors"]:
        return result
    for name, candidates in by_scheme.items():
        result["schemes"][name] = score(
            base_url, model, scenario, context, candidates, name, timeout
        )
    return result


def summary(results: list[dict[str, Any]], scheme: str) -> dict[str, Any]:
    rows = [r["schemes"].get(scheme) for r in results if r["schemes"].get(scheme)]
    good = [r for r in rows if r["status_code"] == 200]
    expected = [r for r in good if r.get("expected_action") is not None]
    matches = sum(r["top1_matches_expected"] for r in expected)
    ranked = [r for r in good if r.get("ranking")]
    margins = [r["top1_margin_mean_logprob"] for r in ranked if r.get("top1_margin_mean_logprob") is not None]
    return {
        "scenario_count": len(rows),
        "http_200_count": len(good),
        "expected_action_count": len(expected),
        "top1_exact_count": matches,
        "top1_exact_rate": matches / len(expected) if expected else None,
        "prefix_cached_count": sum(r.get("prefix_cached") is True for r in good),
        "all_candidates_have_token_scores": all(
            all(bool(x.get("token_scores")) for x in r.get("ranking", []))
            for r in good
        ),
        "prefix_chars": len(good[0].get("prefix", "")) if good else None,
        "avg_prefix_token_count": (
            sum((r.get("stats") or {}).get("prefix_token_count", 0) for r in good) / len(good)
            if good else None
        ),
        "avg_total_ms": (
            sum((r.get("stats") or {}).get("total_ms", 0.0) for r in good) / len(good)
            if good else None
        ),
        "avg_candidate_suffix_chars": (
            sum(
                sum(len(str(x.get("suffix") or "")) for x in r.get("ranking", [])) / len(r["ranking"])
                for r in ranked
            ) / len(ranked)
            if ranked else None
        ),
        "avg_candidate_suffix_tokens": (
            sum(
                sum(x.get("token_count", 0) for x in r.get("ranking", [])) / len(r["ranking"])
                for r in ranked
            ) / len(ranked)
            if ranked else None
        ),
        "avg_top1_margin_mean_logprob": sum(margins) / len(margins) if margins else None,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-Omni 动作评分方案对比报告",
        "",
        f"模型：{report['model']}",
        f"场景数：{report['scenario_count']}，每个动作场景均包含文本和 WAV",
        f"生成时间：{report['finished_at']}",
        "",
        "## 1. 方案定义",
        "",
        "方案 A：相同 prefix + 自然语言 suffix；suffix 为 catalog 的 short_definition。",
        "方案 B：相同 prefix + 动作候选列表 + 短 action_id suffix；候选为 a01 到 a20，none 表示 no_action。",
        "方案 C：与方案 A 使用相同 prefix，但 suffix 直接使用 catalog source_label 动作名称；no_action 使用 no_action。",
        "",
        "## 2. Logit、logprob、PPL 规则",
        "",
        "对于 suffix token 序列 y_1 到 y_T：",
        "",
        "logprob_t = log P(y_t | 完整 prefix, 之前的 suffix tokens)",
        "sum_logprob = sum(logprob_t)",
        "mean_logprob = sum_logprob / T",
        "mean_nll = -mean_logprob",
        "PPL = exp(mean_nll) = exp(-mean_logprob)",
        "",
        "当前接口返回归一化 token logprob，不返回 vocabulary raw logit。raw logit 只能由 logprob 加上同一位置的未知归一化常数得到，因此报告中的 raw_logit_available 为 false。每个 token 的 token_id、logprob、probability、nll 均保留。",
        "",
        "## 3. 汇总",
        "",
        "| 方案 | HTTP 200 | prefix_cached | Top-1 命中 | 命中率 | prefix 字符 | prefix token | suffix 字符/token | 平均耗时 ms | 平均 margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, title in ((NATURAL, "A natural suffix"), (SHORT_ID, "B candidate list + short ID"), (SOURCE_LABEL, "C source_label suffix")):
        r = report["summary"][name]
        lines.append(
            f"| {title} | {r['http_200_count']}/{r['scenario_count']} | "
            f"{r['prefix_cached_count']}/{r['http_200_count']} | "
            f"{r['top1_exact_count']}/{r['expected_action_count']} | "
            f"{r['top1_exact_rate']:.1%} | {r['prefix_chars']} | "
            f"{r['avg_prefix_token_count']:.1f} | "
            f"{r['avg_candidate_suffix_chars']:.2f}/{r['avg_candidate_suffix_tokens']:.2f} | "
            f"{r['avg_total_ms']:.1f} | {r['avg_top1_margin_mean_logprob']:.3f} |"
        )
    lines += [
        "",
        "## 4. 逐场景 Top-1",
        "",
        "| 场景 | 期望 | A Top-1/PPL | B Top-1/PPL | C Top-1/PPL |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        tops = [r["schemes"].get(name, {}).get("top1") or {} for name in SCHEMES]
        values = " | ".join(
            f"{top.get('source_label') or '-'} / {top.get('ppl', 0):.3f}"
            for top in tops
        )
        lines.append(
            f"| {r['scenario_id']} | {r.get('expected_action') or '无'} | {values} |"
        )
    lines += ["", "## 5. 具体例子", ""]
    for r in report["examples"]:
        lines += [
            f"### {r['scenario_id']}，期望：{r.get('expected_action')}",
            "",
            "同一个 session 的聊天轮次：",
            "",
            json.dumps(r["turns"], ensure_ascii=False, indent=2),
            "",
        ]
        for name in SCHEMES:
            data = r["schemes"].get(name, {})
            top = data.get("top1") or {}
            lines += [
                f"方案 {name} 的完整 prompt messages：",
                "",
                json.dumps(data.get("prompt_messages"), ensure_ascii=False, indent=2),
                "",
                f"Top-1={top.get('candidate_id')} / {top.get('source_label')}，"
                f"mean_logprob={top.get('mean_logprob')}，PPL={top.get('ppl')}，"
                f"token_count={top.get('token_count')}",
                "",
                "Top-5 token score 摘要：",
                "",
                json.dumps(data.get("ranking", [])[:5], ensure_ascii=False, indent=2),
                "",
            ]
    lines += [
        "## 6. 优化空间",
        "",
        "若 B 明显优于 A，说明长自然语言 suffix 的长度、词频和表达概率偏置是主要问题，候选列表加短 ID 更适合作为工程基线。",
        "若 B 仍失败但独立 action_id 分类正确，说明模型能理解动作，主要问题是 suffix logprob 校准。",
        "若 A、B 都失败，应考虑受约束分类头、候选校准器或结构化 action_id 输出，而不是继续增长 suffix 描述。",
        "两个方案的 prefix 和 suffix token 序列不同，绝对 PPL 不应跨方案直接比较；应比较各自候选集合内的排名、命中率和 margin。",
        "报告保留全部原始候选排名和 token 分数，不用阈值掩盖错误。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--common-actions-dir", default=DEFAULT_COMMON_ACTIONS_DIR)
    parser.add_argument("--container-common-actions-dir", default=DEFAULT_CONTAINER_COMMON_ACTIONS_DIR)
    parser.add_argument("--audio-variant", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output", default="results/qwen3_omni_action_scheme_comparison.json")
    parser.add_argument("--markdown-output", default="results/qwen3_omni_action_scheme_comparison.md")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    health = _json_request(args.base_url, "/health", timeout=args.timeout)
    models = _json_request(args.base_url, "/v1/models", timeout=args.timeout)
    model_ids = [x.get("id") for x in (models.get("body", {}).get("data") or [])]
    if health["status_code"] != 200 or args.model not in model_ids:
        print(json.dumps({"health": health, "models": models}, ensure_ascii=False, indent=2))
        return 2

    catalog = load_action_catalog(args.catalog)
    scenarios = build_common_action_scenarios(
        args.common_actions_dir,
        args.container_common_actions_dir,
        catalog,
        variant_index=args.audio_variant,
    )
    labels = [x["source_label"] for x in scenarios]
    by_scheme = candidates_for(catalog, labels)
    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "catalog_path": args.catalog,
        "common_actions_dir": args.common_actions_dir,
        "container_common_actions_dir": args.container_common_actions_dir,
        "audio_variant": args.audio_variant,
        "scenario_count": len(scenarios),
        "candidate_counts": {k: len(v) for k, v in by_scheme.items()},
        "scheme_definitions": {
            NATURAL: {
                "name": "相同 prefix + 自然语言 suffix",
                "prefix": make_prefix(NATURAL, by_scheme[NATURAL]),
                "suffix_rule": "catalog short_definition",
            },
            SHORT_ID: {
                "name": "相同 prefix + 候选列表 + 短 action_id suffix",
                "prefix": make_prefix(SHORT_ID, by_scheme[SHORT_ID]),
                "suffix_rule": "a01..a20 / none",
            },
            SOURCE_LABEL: {
                "name": "相同 prefix + source_label suffix",
                "prefix": make_prefix(SOURCE_LABEL, by_scheme[SOURCE_LABEL]),
                "suffix_rule": "catalog source_label / no_action",
            },
        },
        "preflight": {"health": health, "models": models},
        "results": [],
    }
    for index, scenario in enumerate(scenarios, 1):
        row = run_one(args.base_url, args.model, scenario, by_scheme, args.timeout)
        report["results"].append(row)
        tops = [row["schemes"].get(name, {}).get("top1") or {} for name in SCHEMES]
        labels_text = " ".join(
            f"{letter}={top.get('source_label')}"
            for letter, top in zip(("A", "B", "C"), tops, strict=True)
        )
        print(
            f"[{index:02d}/{len(scenarios)}] {scenario['id']} "
            f"expected={scenario.get('expected_action')} {labels_text}"
        )
    report["summary"] = {k: summary(report["results"], k) for k in SCHEMES}
    disagreements = [
        row for row in report["results"]
        if len({
            row["schemes"].get(name, {}).get("top1", {}).get("source_label")
            for name in SCHEMES
        }) > 1
    ]
    report["examples"] = []
    for row in report["results"][:1] + disagreements[:3]:
        if row not in report["examples"]:
            report["examples"].append(row)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["formula"] = {
        "token_logprob": "log P(token_t | prefix, previous_suffix_tokens)",
        "sum_logprob": "sum(token_logprob)",
        "mean_logprob": "sum_logprob / token_count",
        "mean_nll": "-mean_logprob",
        "ppl": "exp(mean_nll)",
        "raw_logit": "not returned by /v1/action-scores",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(f"JSON report: {output}")
    print(f"Markdown report: {markdown_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
