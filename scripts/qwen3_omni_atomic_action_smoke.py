#!/usr/bin/env python3
"""Live Qwen3-Omni atomic-action conversation smoke test.

The script deliberately keeps action selection separate from animation
playback: normal chat produces the assistant turn, then ``/v1/action-scores``
rankings are evaluated against the same session history.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = os.environ.get("QWEN3_OMNI_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.environ.get(
    "QWEN3_OMNI_MODEL", "Qwen3-Omni-30B-A3B-Instruct"
)
DEFAULT_AUDIO = "/data/models/hehy/D_human_train/examples/test/bench20_examples/none_4.wav"
DEFAULT_IMAGE = "/data/models/xingmt/wan_export_step651_speedtest/ref.png"
ACTION_PREFIX = (
    "请根据以上对话和当前数字人状态，选择唯一一个最合适的原子动作。"
    "只输出动作本身，不要解释；如果没有明确动作指令，就选择不做任何动作。"
    "下一步动作是："
)


def candidate(
    candidate_id: str,
    suffix: str,
    *,
    action_id: str | None = None,
    execution_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "suffix": suffix,
        "action_id": action_id or candidate_id,
        "execution_binding": execution_binding or {},
    }


def build_candidates() -> list[dict[str, Any]]:
    return [
        candidate("wave_left", "使用左手挥手", execution_binding={"body_side": "left"}),
        candidate("wave_right", "使用右手挥手", execution_binding={"body_side": "right"}),
        candidate("wave_both", "使用双手挥手", execution_binding={"body_side": "both"}),
        candidate("laugh", "大笑"),
        candidate("heart_both", "使用双手比心", execution_binding={"body_side": "both"}),
        candidate("thumbs_up_left", "使用左手点赞", execution_binding={"body_side": "left"}),
        candidate("thumbs_up_right", "使用右手点赞", execution_binding={"body_side": "right"}),
        candidate("number_1", "用右手比数字一", execution_binding={"body_side": "right", "number": "1"}),
        candidate("number_2", "用右手比数字二", execution_binding={"body_side": "right", "number": "2"}),
        candidate("number_3", "用右手比数字三", execution_binding={"body_side": "right", "number": "3"}),
        candidate("number_4", "用右手比数字四", execution_binding={"body_side": "right", "number": "4"}),
        candidate("number_5", "用右手比数字五", execution_binding={"body_side": "right", "number": "5"}),
        candidate("no_action", "不需要做任何动作", action_id="no_action"),
    ]


ACTION_SCENARIOS: list[dict[str, Any]] = [
    {"id": "wave-left", "expected_action": "wave_left", "turns": ["你好，我们来聊聊天吧。", "见到你很开心，请用左手向我挥手打招呼。"]},
    {"id": "wave-right", "expected_action": "wave_right", "turns": ["你好，我们来聊聊天吧。", "见到你很开心，请用右手向我挥手打招呼。"]},
    {"id": "wave-both", "expected_action": "wave_both", "turns": ["今天是值得庆祝的一天。", "太棒了，请用双手向我挥手庆祝。"]},
    {"id": "laugh", "expected_action": "laugh", "turns": ["我刚刚讲了一个笑话。", "这个笑话太逗了，请开心地大笑一下。"]},
    {"id": "heart-both", "expected_action": "heart_both", "turns": ["谢谢你一直陪我聊天。", "谢谢你的陪伴，请用双手给我比一个心。"]},
    {"id": "thumbs-left", "expected_action": "thumbs_up_left", "turns": ["我刚刚完成了任务。", "我这次做得好吗，请用左手给我点个赞。"]},
    {"id": "thumbs-right", "expected_action": "thumbs_up_right", "turns": ["我刚刚完成了任务。", "我这次做得好吗，请用右手给我点个赞。"]},
    {"id": "number-1", "expected_action": "number_1", "turns": ["我们来做一个简单的数字练习。", "请用右手比出数字一。"]},
    {"id": "number-2", "expected_action": "number_2", "turns": ["我们来做一个简单的数字练习。", "请用右手比出数字二。"]},
    {"id": "number-3", "expected_action": "number_3", "turns": ["我们来做一个简单的数字练习。", "请用右手比出数字三。"]},
    {"id": "number-4", "expected_action": "number_4", "turns": ["我们来做一个简单的数字练习。", "请用右手比出数字四。"]},
    {"id": "number-5", "expected_action": "number_5", "turns": ["我们来做一个简单的数字练习。", "请用右手比出数字五。"]},
    {"id": "no-action-explicit", "expected_action": "no_action", "turns": ["我想了解一下人工智能。", "请解释一下人工智能是什么，这一轮不需要做任何动作。"]},
    {"id": "no-action-neutral", "expected_action": "no_action", "turns": ["我们继续聊刚才的话题。", "请用两句话总结一下刚才的内容。"]},
]


def _json_request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return {"status_code": response.status, "body": json.loads(raw)}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return {"status_code": exc.code, "body": parsed}
    except (URLError, TimeoutError, OSError) as exc:
        return {"status_code": None, "body": {"error": str(exc)}}


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or "")


def _chat_turn(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    audios: list[str],
    images: list[str],
    request_id: str,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "modalities": ["text"],
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
        "request_id": request_id,
    }
    if audios:
        payload["audios"] = audios
    if images:
        payload["images"] = images
    result = _json_request(base_url, "/v1/chat/completions", payload, timeout=timeout)
    if result["status_code"] != 200:
        return result, ""
    choices = result["body"].get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        return result, ""
    return result, _message_text(choices[0]["message"])


def _run_scenario(
    base_url: str,
    model: str,
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    timeout: float,
) -> dict[str, Any]:
    session_id = f"atomic-actions-{scenario['id']}"
    history: list[dict[str, Any]] = []
    history_audios: list[str] = []
    history_images: list[str] = []
    turns: list[dict[str, Any]] = []
    errors: list[str] = []

    for turn_index, turn in enumerate(scenario["turns"], start=1):
        user_content = turn if isinstance(turn, str) else turn["content"]
        turn_audios = [] if isinstance(turn, str) else list(turn.get("audios", []))
        turn_images = [] if isinstance(turn, str) else list(turn.get("images", []))
        response, assistant_text = _chat_turn(
            base_url,
            model,
            [*history, {"role": "user", "content": user_content}],
            audios=[*history_audios, *turn_audios],
            images=[*history_images, *turn_images],
            request_id=f"{session_id}-chat-{turn_index}",
            timeout=timeout,
        )
        turn_result = {
            "turn_index": turn_index,
            "user": user_content,
            "status_code": response["status_code"],
            "assistant": assistant_text,
            "response": response["body"],
        }
        turns.append(turn_result)
        if response["status_code"] != 200:
            errors.append(f"chat turn {turn_index} returned {response['status_code']}")
            break
        if not assistant_text.strip():
            errors.append(f"chat turn {turn_index} returned empty assistant text")
            break
        history.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        history_audios.extend(turn_audios)
        history_images.extend(turn_images)

    score_media = scenario.get("score_media", {})
    current_audios = list(score_media.get("audios", []))
    current_images = list(score_media.get("images", []))
    score_result: dict[str, Any] = {
        "status_code": None,
        "response": None,
        "ranking": [],
        "selected_action": None,
        "execute": None,
        "top1_margin": None,
        "prefix_cached": None,
        "valid_token_scores": False,
    }
    if not errors:
        payload = {
            "request_id": f"{session_id}-score",
            "model": model,
            "prefix": ACTION_PREFIX,
            "language": "zh",
            "sample_rate": 16000,
            "micro_batch_size": 32,
            "session_id": session_id,
            "history": history,
            "history_audios": history_audios,
            "history_images": history_images,
            "audios": current_audios,
            "images": current_images,
            "avatar_state": scenario.get(
                "avatar_state",
                {"pose": "seated", "gaze": "camera", "hands": "resting"},
            ),
            "candidates": candidates,
        }
        response = _json_request(
            base_url, "/v1/action-scores", payload, timeout=timeout
        )
        score_result["status_code"] = response["status_code"]
        score_result["response"] = response["body"]
        if response["status_code"] != 200:
            errors.append(f"action score returned {response['status_code']}")
        else:
            body = response["body"]
            scores = body.get("scores") or []
            by_id = {item["candidate_id"]: item for item in candidates}
            ranked = sorted(
                scores,
                key=lambda item: float(item.get("mean_logprob", float("-inf"))),
                reverse=True,
            )
            score_result["ranking"] = [
                {
                    "rank": index,
                    "candidate_id": item.get("candidate_id"),
                    "action_id": by_id.get(item.get("candidate_id"), {}).get("action_id"),
                    "mean_logprob": item.get("mean_logprob"),
                    "ppl": item.get("ppl"),
                    "token_count": item.get("token_count"),
                }
                for index, item in enumerate(ranked, start=1)
            ]
            score_result["prefix_cached"] = body.get("prefix_cached")
            score_result["valid_token_scores"] = bool(scores) and all(
                isinstance(item.get("token_scores"), list) and bool(item["token_scores"])
                for item in scores
            )
            if not scores:
                errors.append("action score returned no scores")
            elif len(ranked) >= 2:
                score_result["top1_margin"] = float(ranked[0]["mean_logprob"]) - float(
                    ranked[1]["mean_logprob"]
                )
            if ranked:
                selected_candidate = ranked[0].get("candidate_id")
                score_result["selected_action"] = by_id.get(selected_candidate, {}).get(
                    "action_id", selected_candidate
                )
                score_result["execute"] = score_result["selected_action"] != "no_action"
            if body.get("prefix_cached") is not True:
                errors.append("prefix_cached was not true")
            if not score_result["valid_token_scores"]:
                errors.append("one or more candidates have empty token_scores")

    expected = scenario.get("expected_action")
    if expected is not None and score_result["selected_action"] != expected:
        errors.append(
            f"expected {expected}, selected {score_result['selected_action']}"
        )
    return {
        "scenario_id": scenario["id"],
        "session_id": session_id,
        "expected_action": expected,
        "turns": turns,
        "action_score": score_result,
        "passed": not errors,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--output",
        default="/tmp/qwen3_omni_atomic_actions_report.json",
        help="JSON report path",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only the named scenario ID; repeat for multiple IDs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates = build_candidates()
    candidate_ids = [item["candidate_id"] for item in candidates]
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "candidate_ids": candidate_ids,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight": {},
        "scenarios": [],
    }

    health = _json_request(args.base_url, "/health", timeout=args.timeout)
    models = _json_request(args.base_url, "/v1/models", timeout=args.timeout)
    model_body = models["body"] if isinstance(models["body"], dict) else {}
    model_ids = [item.get("id") for item in (model_body.get("data") or [])]
    report["preflight"] = {
        "health": health,
        "models": models,
        "health_ok": health["status_code"] == 200
        and health["body"].get("running") is True
        and "action_score" in health["body"].get("stages", []),
        "model_ok": args.model in model_ids,
    }
    if not report["preflight"]["health_ok"] or not report["preflight"]["model_ok"]:
        report["error"] = "preflight failed"
        _write_report(args.output, report)
        print(json.dumps(report["preflight"], ensure_ascii=False, indent=2))
        return 2

    scenarios = list(ACTION_SCENARIOS)
    scenarios.append(
        {
            "id": "multimodal-context",
            "expected_action": None,
            "turns": [
                {
                    "content": [
                        {"type": "audio"},
                        {"type": "image"},
                        {"type": "text", "text": "请结合这段声音和图片理解我现在的状态。"},
                    ],
                    "audios": [args.audio],
                    "images": [args.image],
                }
            ],
            "score_media": {"audios": [args.audio], "images": [args.image]},
            "avatar_state": {"pose": "seated", "gaze": "camera", "hands": "resting"},
        }
    )
    requested = set(args.only)
    available = {scenario["id"] for scenario in scenarios}
    unknown = requested - available
    if unknown:
        print(
            "Unknown scenario ID(s): " + ", ".join(sorted(unknown)) + ". "
            "Available: " + ", ".join(sorted(available)),
            file=sys.stderr,
        )
        return 2
    if requested:
        scenarios = [scenario for scenario in scenarios if scenario["id"] in requested]
    for scenario in scenarios:
        result = _run_scenario(
            args.base_url,
            args.model,
            scenario,
            candidates,
            timeout=args.timeout,
        )
        report["scenarios"].append(result)
        score = result["action_score"]
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['scenario_id']}: "
            f"expected={result['expected_action']} "
            f"selected={score['selected_action']} "
            f"margin={score['top1_margin']}"
        )
        for item in score["ranking"]:
            print(
                f"  #{item['rank']:02d} {item['action_id']} "
                f"mean_logprob={item['mean_logprob']:.6f} ppl={item['ppl']:.3f}"
            )
        for error in result["errors"]:
            print(f"  ERROR: {error}", file=sys.stderr)

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["passed"] = all(item["passed"] for item in report["scenarios"])
    _write_report(args.output, report)
    print(f"JSON report: {args.output}")
    return 0 if report["passed"] else 1


def _write_report(path: str, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
