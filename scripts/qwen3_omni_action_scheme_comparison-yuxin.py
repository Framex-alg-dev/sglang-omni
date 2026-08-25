
#!/usr/bin/env python3
"""Evaluate source-label action suffix scoring on multimodal sessions."""

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
    DEFAULT_BASE_URL,
    DEFAULT_CATALOG,
    DEFAULT_COMMON_ACTIONS_DIR,
    DEFAULT_CONTAINER_COMMON_ACTIONS_DIR,
    _json_request,
    build_common_action_scenarios,
    load_action_catalog,
)

DEFAULT_MODEL = os.environ.get("QWEN3_OMNI_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
SOURCE_LABEL = "source_label_suffix"
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "你是一个动作序列判断助手，根据用户的输入（可能是文本、图像、语音之一，也可能是三种模态的混合），"
    "判断当下应该执行什么动作。你可见的动作有{actions}。\n"
    "特别的，你只需要输出动作就好，不需要回答用户的内容"
)


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


def load_system_prompt_template(template: str, template_file: str | None) -> str:
    if template_file:
        template = Path(template_file).read_text(encoding="utf-8")
    if "{actions}" not in template:
        raise ValueError("system prompt template must contain the {actions} placeholder")
    return template


def make_system_prompt(template: str, labels: list[str]) -> str:
    if len(labels) != len(set(labels)):
        raise ValueError("system prompt action labels must be unique")
    # Use catalog source_label values verbatim as the simplified action
    # descriptions; do not paraphrase or mutate catalog wording.
    actions = "、".join([*labels, "no_action"])
    return template.replace("{actions}", actions)


def make_scoring_history(
    history: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        *(dict(message) for message in history),
    ]


def make_prompt_messages(
    scoring_history: list[dict[str, Any]], instruction: str
) -> list[dict[str, Any]]:
    """Mirror the client: media first, then the current user text."""
    messages = [dict(message) for message in scoring_history]
    latest_user = messages[-1]
    content = latest_user.get("content")
    parts = [dict(part) for part in content] if isinstance(content, list) else []
    latest_user["content"] = [*parts, {"type": "text", "text": instruction}]
    return messages


def make_history(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build a single-turn scoring history from the final action request."""
    session_id = f"scheme-compare-{scenario['id']}"
    turn = scenario["turns"][-1]
    content = turn if isinstance(turn, str) else turn["content"]
    audios = [] if isinstance(turn, str) else list(turn.get("audios", []))
    images = [] if isinstance(turn, str) else list(turn.get("images", []))
    media_parts = [
        *({"type": "audio"} for _ in audios),
        *({"type": "image"} for _ in images),
    ]
    history = [
        {
            "role": "user",
            "content": media_parts,
        }
    ]
    return {
        "session_id": session_id,
        "turns": [
            {
                "turn_index": 1,
                "user": content,
                "audios": audios,
                "images": images,
            }
        ],
        "history": history,
        "instruction": str(content),
        "history_audios": audios,
        "history_images": images,
        "avatar_state": scenario.get(
            "avatar_state",
            {"pose": "seated", "gaze": "camera", "hands": "resting"},
        ),
        "errors": [],
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
    system_prompt: str,
    timeout: float,
) -> dict[str, Any]:
    prefix = context["instruction"]
    scoring_history = make_scoring_history(context["history"], system_prompt)
    prompt_messages = make_prompt_messages(scoring_history, prefix)
    media = scenario.get("score_media", {})
    payload = {
        "request_id": f"{context['session_id']}-{scheme}-score",
        "model": model,
        "prefix": prefix,
        "language": "zh",
        "sample_rate": 16000,
        "micro_batch_size": 32,
        "session_id": context["session_id"],
        "history": scoring_history,
        "history_audios": context["history_audios"],
        "history_images": context["history_images"],
        "audios": list(media.get("audios", [])),
        "images": list(media.get("images", [])),
        # The desired system message is already present in history. Keeping
        # this empty prevents the client from prepending a second state-only
        # system message.
        "avatar_state": {},
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
        "system_prompt": system_prompt,
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


def run_one(
    base_url: str,
    model: str,
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
    system_prompt: str,
    timeout: float,
) -> dict[str, Any]:
    context = make_history(scenario)
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
    result["schemes"][SOURCE_LABEL] = score(
        base_url,
        model,
        scenario,
        context,
        candidates,
        SOURCE_LABEL,
        system_prompt,
        timeout,
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
        "prefix_chars": (
            sum(len(r.get("prefix", "")) for r in good) / len(good)
            if good else None
        ),
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
        "# Qwen3-Omni Source Label 动作评分报告",
        "",
        f"模型：{report['model']}",
        f"场景数：{report['scenario_count']}，每个动作场景均包含文本和 WAV",
        f"生成时间：{report['finished_at']}",
        "",
        "## 1. 方案定义",
        "",
        f"System prompt：{report['system_prompt']}",
        "任务规则只存在于 system message；user 中多模态内容在前，原始用户文本在后。",
        "候选 suffix 使用 catalog source_label，并在动作后追加 <|im_end|> 参与评分。",
        "",
        "## 2. Logit、logprob、PPL 规则",
        "",
        "对于动作 token 序列 y_1 到 y_T，以及动作结束 token <|im_end|>：",
        "",
        "logprob_t = log P(y_t | 完整 prefix, 之前的 suffix tokens)",
        "sum_logprob = sum(logprob_t)",
        "mean_logprob = sum_logprob / (T + 1)",
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
    summary_row = report["summary"]
    lines.append(
        f"| source_label suffix | {summary_row['http_200_count']}/{summary_row['scenario_count']} | "
        f"{summary_row['prefix_cached_count']}/{summary_row['http_200_count']} | "
        f"{summary_row['top1_exact_count']}/{summary_row['expected_action_count']} | "
        f"{summary_row['top1_exact_rate']:.1%} | {summary_row['prefix_chars']} | "
        f"{summary_row['avg_prefix_token_count']:.1f} | "
        f"{summary_row['avg_candidate_suffix_chars']:.2f}/{summary_row['avg_candidate_suffix_tokens']:.2f} | "
        f"{summary_row['avg_total_ms']:.1f} | "
        f"{summary_row['avg_top1_margin_mean_logprob']:.3f} |"
    )
    lines += [
        "",
        "## 4. 逐场景 Top-1",
        "",
        "| 场景 | 期望 | Top-1/PPL |",
        "|---|---|---|",
    ]
    for r in report["results"]:
        top = r["schemes"].get(SOURCE_LABEL, {}).get("top1") or {}
        lines.append(
            f"| {r['scenario_id']} | {r.get('expected_action') or '无'} | "
            f"{top.get('source_label') or '-'} / {top.get('ppl', 0):.3f} |"
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
        data = r["schemes"].get(SOURCE_LABEL, {})
        top = data.get("top1") or {}
        lines += [
            "完整 prompt messages：",
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
        "重点优化 system prompt、source_label 表达、候选校准和 Top-1 margin。",
        "PPL 是 source_label 文本的续写难度，不是归一化后的动作分类概率。",
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
    parser.add_argument("--output", default="results/qwen3_omni_source_label_scoring-yuxin.json")
    parser.add_argument("--markdown-output", default="results/qwen3_omni_source_label_scoring-yuxin.md")
    system_prompt_group = parser.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system-prompt-template",
        default=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        help="System prompt template; must contain the {actions} placeholder.",
    )
    system_prompt_group.add_argument(
        "--system-prompt-file",
        help="UTF-8 file containing a system prompt template with {actions}.",
    )
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
    candidates = label_candidates(catalog, labels)
    system_prompt_template = load_system_prompt_template(
        args.system_prompt_template,
        args.system_prompt_file,
    )
    system_prompt = make_system_prompt(system_prompt_template, labels)
    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "model": args.model,
        "catalog_path": args.catalog,
        "common_actions_dir": args.common_actions_dir,
        "container_common_actions_dir": args.container_common_actions_dir,
        "audio_variant": args.audio_variant,
        "scenario_count": len(scenarios),
        "candidate_count": len(candidates),
        "system_prompt_template": system_prompt_template,
        "system_prompt": system_prompt,
        "scheme_definitions": {
            SOURCE_LABEL: {
                "name": "system prompt + source_label suffix",
                "prefix_rule": "current user text, appended after multimodal content",
                "suffix_rule": "catalog source_label / no_action",
            },
        },
        "preflight": {"health": health, "models": models},
        "results": [],
    }
    for index, scenario in enumerate(scenarios, 1):
        row = run_one(
            args.base_url,
            args.model,
            scenario,
            candidates,
            system_prompt,
            args.timeout,
        )
        report["results"].append(row)
        top = row["schemes"].get(SOURCE_LABEL, {}).get("top1") or {}
        print(
            f"[{index:02d}/{len(scenarios)}] {scenario['id']} "
            f"expected={scenario.get('expected_action')} top1={top.get('source_label')}"
        )
    report["summary"] = summary(report["results"], SOURCE_LABEL)
    mismatches = [
        row for row in report["results"]
        if not row["schemes"].get(SOURCE_LABEL, {}).get("top1_matches_expected", False)
    ]
    report["examples"] = []
    for row in report["results"][:1] + mismatches[:3]:
        if row not in report["examples"]:
            report["examples"].append(row)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["formula"] = {
        "token_logprob": "log P(token_t | prefix, previous_suffix_tokens)",
        "sum_logprob": "sum(action_token_logprob) + im_end_logprob",
        "mean_logprob": "sum_logprob / token_count",
        "mean_nll": "-mean_logprob",
        "ppl": "exp(mean_nll)",
        "terminal_token": "<|im_end|> is included in token_count and PPL",
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
