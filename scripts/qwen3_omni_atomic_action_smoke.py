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
DEFAULT_COMMON_ACTIONS_DIR = str(
    Path(__file__).resolve().parents[1] / "tests/data/actions/common_actions_20"
)
DEFAULT_CONTAINER_COMMON_ACTIONS_DIR = "/opt/sglang-omni/tests/data/actions/common_actions_20"
DEFAULT_CATALOG = str(
    Path(__file__).resolve().parents[1] / "tests/data/actions/character_action_catalog.json"
)
ACTION_PREFIX = (
    "请根据以上对话和当前数字人状态，选择唯一一个最合适的原子动作。"
    "只输出动作本身，不要解释；如果没有明确动作指令，就选择不做任何动作。"
    "下一步动作是："
)


def load_action_catalog(path: str) -> dict[str, dict[str, Any]]:
    """Load the canonical catalog indexed by its human-readable source label."""
    catalog_path = Path(path)
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    actions = document.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"catalog {catalog_path} has no actions list")
    by_label: dict[str, dict[str, Any]] = {}
    for item in actions:
        source = item.get("source") or {}
        label = source.get("source_label")
        action_id = item.get("action_id")
        if not isinstance(label, str) or not label or not isinstance(action_id, str):
            continue
        if label in by_label:
            raise ValueError(f"catalog source_label is not unique: {label}")
        by_label[label] = item
    if not by_label:
        raise ValueError(f"catalog {catalog_path} contains no usable actions")
    return by_label


def _binding_suffix(short_definition: str, binding: dict[str, str]) -> str:
    side = binding.get("body_side")
    side_text = {"left": "左手", "right": "右手", "both": "双手"}.get(side)
    if side_text is None:
        return short_definition
    return f"{short_definition}，使用{side_text}执行"


def catalog_candidate(
    selection_id: str,
    catalog: dict[str, dict[str, Any]],
    source_label: str,
    *,
    execution_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    item = catalog.get(source_label)
    if item is None:
        raise KeyError(f"action source_label not found in catalog: {source_label}")
    short_definition = item.get("short_definition") or item.get("prompt") or source_label
    binding = dict(execution_binding or {})
    return {
        # This is a human-readable variant key used by the classifier. The
        # canonical runtime action ID is carried separately below.
        "candidate_id": selection_id,
        "suffix": _binding_suffix(short_definition, binding),
        "description": f"{source_label}：{short_definition}",
        "action_id": item["action_id"],
        "source_label": source_label,
        "short_definition": short_definition,
        "category_path": list(item.get("category_path") or []),
        "action_type": item.get("action_type", ""),
        "execution_binding": binding,
    }


def no_action_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "no_action",
        "suffix": "不需要做任何动作",
        "description": "不做动作：不需要做任何动作",
        "action_id": "no_action",
        "source_label": "no_action",
        "short_definition": "不需要做任何动作",
        "category_path": [],
        "action_type": "control",
        "execution_binding": {},
    }


def build_catalog_candidates(
    catalog: dict[str, dict[str, Any]],
    labels: list[str],
) -> list[dict[str, Any]]:
    """Build one candidate per selected catalog action plus no_action."""
    if len(labels) != len(set(labels)):
        raise ValueError("common action labels must be unique")
    return [
        *[catalog_candidate(label, catalog, label) for label in labels],
        no_action_candidate(),
    ]


def _manifest_audio_relative_path(audio_path: str) -> Path:
    prefix = "data/common_actions_20/"
    if not audio_path.startswith(prefix):
        raise ValueError(f"unexpected common_actions_20 audio path: {audio_path}")
    return Path(audio_path[len(prefix) :])


def build_common_action_scenarios(
    common_actions_dir: str,
    container_common_actions_dir: str,
    catalog: dict[str, dict[str, Any]],
    *,
    variant_index: int = 1,
) -> list[dict[str, Any]]:
    """Create 20 catalog-backed scenarios whose action turn has text + audio."""
    root = Path(common_actions_dir)
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    texts = json.loads((root / "texts.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    text_samples = {}
    for sample in texts.get("samples", []):
        expected = sample.get("expected_actions") or []
        metadata = sample.get("metadata") or {}
        if expected and metadata.get("variant_index") == variant_index:
            text_samples[expected[0]["id"]] = sample
    manifest_samples = {sample.get("id"): sample for sample in manifest.get("samples", [])}

    scenarios: list[dict[str, Any]] = []
    for action in selection.get("actions", []):
        label = action["label"]
        catalog_item = catalog.get(label)
        if catalog_item is None:
            raise KeyError(f"common action label not found in catalog: {label}")
        action_id = catalog_item["action_id"]
        text_sample = text_samples.get(action_id)
        if text_sample is None:
            raise KeyError(f"no text sample for catalog action: {action_id}")
        sample_id = text_sample["id"]
        manifest_sample = manifest_samples.get(sample_id)
        if manifest_sample is None:
            raise KeyError(f"no audio manifest sample for: {sample_id}")
        relative_audio = _manifest_audio_relative_path(manifest_sample["audio_path"])
        local_audio = root / relative_audio
        if not local_audio.is_file():
            raise FileNotFoundError(local_audio)
        container_audio = str(Path(container_common_actions_dir) / relative_audio)
        scenarios.append(
            {
                "id": f"common-{action_id}",
                "expected_action": label,
                "expected_catalog_action_id": action_id,
                "source_label": label,
                "turns": [
                    "你好，我们来聊聊天吧。",
                    {
                        "content": text_sample["text"],
                        "audios": [container_audio],
                    },
                ],
                "audio_sample": {
                    "sample_id": sample_id,
                    "local_path": str(local_audio),
                    "container_path": container_audio,
                    "variant_index": variant_index,
                },
                # The audio is already part of the current chat turn and is
                # carried into action scoring through history_audios.
                "score_media": {"audios": [], "images": []},
            }
        )
    if len(scenarios) != 20:
        raise ValueError(f"expected 20 common action scenarios, got {len(scenarios)}")
    return scenarios


def action_selector_prompt(
    candidates: list[dict[str, Any]], avatar_state: dict[str, Any]
) -> str:
    definitions = "；".join(
        f"{item['candidate_id']}={item['description']}" for item in candidates
    )
    candidate_ids = ", ".join(item["candidate_id"] for item in candidates)
    state = json.dumps(avatar_state, ensure_ascii=False, separators=(",", ":"))
    return (
        "你是数字人动作识别器。请根据以上完整对话和当前数字人状态，"
        "选择唯一一个最合适的动作候选。"
        f"当前数字人状态：{state}。"
        f"动作候选定义：{definitions}。"
        f"只允许输出以下一个 candidate_id：{candidate_ids}。"
        "没有明确动作指令时必须输出 no_action。只输出 candidate_id，不要解释。"
    )


def build_action_context_messages(
    history: list[dict[str, Any]],
    instruction: str,
    avatar_state: dict[str, Any],
    *,
    audio_count: int = 0,
    image_count: int = 0,
) -> list[dict[str, Any]]:
    """Mirror the server/client action-context ordering for reportability."""
    state = json.dumps(avatar_state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"当前数字人状态：{state}"},
        *(dict(message) for message in history),
    ]
    if messages and messages[-1].get("role") == "user":
        latest = messages.pop()
        content = latest.get("content")
        if isinstance(content, str) and not audio_count and not image_count:
            latest["content"] = f"{content}\n{instruction}"
        elif isinstance(content, list):
            existing_audio = sum(
                isinstance(part, dict) and part.get("type") == "audio"
                for part in content
            )
            existing_image = sum(
                isinstance(part, dict) and part.get("type") == "image"
                for part in content
            )
            latest["content"] = [
                *content,
                *(
                    {"type": "audio"}
                    for _ in range(max(audio_count - existing_audio, 0))
                ),
                *(
                    {"type": "image"}
                    for _ in range(max(image_count - existing_image, 0))
                ),
                {"type": "text", "text": instruction},
            ]
        else:
            latest["content"] = [
                {"type": "text", "text": str(content or "")},
                *( {"type": "audio"} for _ in range(audio_count) ),
                *( {"type": "image"} for _ in range(image_count) ),
                {"type": "text", "text": instruction},
            ]
        messages.append(latest)
    else:
        messages.append({"role": "user", "content": instruction})
    return messages


def extract_action_id(text: str, candidates: list[dict[str, Any]]) -> str | None:
    normalized = text.strip().strip("`* \n")
    candidate_ids = [item["candidate_id"] for item in candidates]
    if normalized in candidate_ids:
        return normalized
    for candidate_id in sorted(candidate_ids, key=len, reverse=True):
        if candidate_id in text:
            return candidate_id
    return None


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


def _history_turn_content(
    content: Any, audios: list[str], images: list[str]
) -> Any:
    """Persist media placeholders with a turn kept in action-score history."""
    if not audios and not images:
        return content
    if isinstance(content, list):
        parts = [dict(part) if isinstance(part, dict) else part for part in content]
    else:
        parts = [] if content is None else [{"type": "text", "text": str(content)}]
    existing_audio = sum(
        isinstance(part, dict) and part.get("type") == "audio" for part in parts
    )
    existing_image = sum(
        isinstance(part, dict) and part.get("type") == "image" for part in parts
    )
    parts.extend(
        {"type": "audio"}
        for _ in range(max(len(audios) - existing_audio, 0))
    )
    parts.extend(
        {"type": "image"}
        for _ in range(max(len(images) - existing_image, 0))
    )
    return parts


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


def resolved_action(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "action_id": candidate["action_id"],
        "source_label": candidate.get("source_label"),
        "short_definition": candidate.get("short_definition"),
        "execution_binding": dict(candidate.get("execution_binding") or {}),
        "category_path": list(candidate.get("category_path") or []),
        "action_type": candidate.get("action_type"),
    }


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
            "audios": turn_audios,
            "images": turn_images,
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
                {
                    "role": "user",
                    "content": _history_turn_content(
                        user_content, turn_audios, turn_images
                    ),
                },
                {"role": "assistant", "content": assistant_text},
            ]
        )
        history_audios.extend(turn_audios)
        history_images.extend(turn_images)

    # The chat history contains the assistant reply generated for the current
    # user request. It is useful for the transcript, but it must not sit
    # between the latest user intent and the action-selection instruction.
    action_history = (
        history[:-1]
        if history and history[-1].get("role") == "assistant"
        else list(history)
    )
    avatar_state = scenario.get(
        "avatar_state",
        {"pose": "seated", "gaze": "camera", "hands": "resting"},
    )
    action_context_messages = build_action_context_messages(
        action_history,
        ACTION_PREFIX,
        avatar_state,
        audio_count=len(history_audios),
        image_count=len(history_images),
    )

    score_media = scenario.get("score_media", {})
    current_audios = list(score_media.get("audios", []))
    current_images = list(score_media.get("images", []))
    score_result: dict[str, Any] = {
        "status_code": None,
        "response": None,
        "ranking": [],
        "selected_action": None,
        "selected_action_id": None,
        "selected_candidate_id": None,
        "resolved_action": None,
        "execute": None,
        "top1_margin": None,
        "prefix_cached": None,
        "valid_token_scores": False,
    }
    selection_result: dict[str, Any] = {
        "status_code": None,
        "response": None,
        "raw_text": "",
        "selected_action": None,
        "selected_action_id": None,
        "selected_candidate_id": None,
        "resolved_action": None,
        "execute": None,
        "valid_action_id": False,
    }
    if not errors:
        api_candidates = [
            {
                key: value
                for key, value in item.items()
                if key in {"candidate_id", "suffix", "action_id", "execution_binding"}
            }
            for item in candidates
        ]
        payload = {
            "request_id": f"{session_id}-score",
            "model": model,
            "prefix": ACTION_PREFIX,
            "language": "zh",
            "sample_rate": 16000,
            "micro_batch_size": 32,
            "session_id": session_id,
            "history": action_history,
            "history_audios": history_audios,
            "history_images": history_images,
            "audios": current_audios,
            "images": current_images,
            "avatar_state": avatar_state,
            "candidates": api_candidates,
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
                    "source_label": by_id.get(item.get("candidate_id"), {}).get("source_label"),
                    "execution_binding": by_id.get(item.get("candidate_id"), {}).get("execution_binding", {}),
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
                selected_item = by_id.get(selected_candidate)
                score_result["selected_candidate_id"] = selected_candidate
                score_result["selected_action"] = selected_candidate
                score_result["selected_action_id"] = (
                    selected_item.get("action_id") if selected_item else selected_candidate
                )
                score_result["resolved_action"] = resolved_action(selected_item)
                score_result["execute"] = selected_candidate != "no_action"
            if body.get("prefix_cached") is not True:
                errors.append("prefix_cached was not true")
            if not score_result["valid_token_scores"]:
                errors.append("one or more candidates have empty token_scores")

    by_id = {item["candidate_id"]: item for item in candidates}
    expected = scenario.get("expected_action")
    if not errors and history_images:
        selection_result["status"] = "skipped"
        selection_result["skip_reason"] = (
            "image-containing action_id classification is skipped after the single "
            "multimodal context pass; action score remains validated"
        )
        if expected is not None:
            errors.append("action_id classification is required for this multimodal scenario")
    elif not errors:
        selector_response, selector_text = _chat_turn(
            base_url,
            model,
            build_action_context_messages(
                action_history,
                action_selector_prompt(candidates, avatar_state),
                avatar_state,
                audio_count=len(history_audios),
                image_count=len(history_images),
            ),
            audios=history_audios,
            images=history_images,
            request_id=f"{session_id}-action-id",
            timeout=timeout,
        )
        selection_result["status_code"] = selector_response["status_code"]
        selection_result["response"] = selector_response["body"]
        selection_result["raw_text"] = selector_text
        selected_candidate = extract_action_id(selector_text, candidates)
        selected_item = by_id.get(selected_candidate) if selected_candidate else None
        selection_result["selected_candidate_id"] = selected_candidate
        selection_result["selected_action"] = selected_candidate
        selection_result["selected_action_id"] = (
            selected_item.get("action_id") if selected_item else None
        )
        selection_result["resolved_action"] = resolved_action(selected_item)
        selection_result["valid_action_id"] = selected_candidate is not None
        if selected_candidate is not None:
            selection_result["execute"] = selected_candidate != "no_action"
        if selector_response["status_code"] != 200:
            errors.append(
                f"action_id classification returned {selector_response['status_code']}"
            )
        elif not selection_result["valid_action_id"]:
            errors.append("action_id classification returned an unknown action_id")

    raw_score_matches = expected is None or score_result["selected_action"] == expected
    action_id_matches = expected is None or selection_result["selected_action"] == expected
    if expected is not None and not raw_score_matches:
        score_result["semantic_error"] = (
            f"expected {expected}, suffix score selected {score_result['selected_action']}"
        )
    if expected is not None and not action_id_matches:
        errors.append(
            f"expected {expected}, action_id classifier selected "
            f"{selection_result['selected_action']}"
        )
    return {
        "scenario_id": scenario["id"],
        "session_id": session_id,
        "expected_action": expected,
        "expected_action_catalog": resolved_action(by_id.get(expected)) if expected else None,
        "turns": turns,
        "action_context_messages": action_context_messages,
        "action_score": score_result,
        "action_id_selection": selection_result,
        "suffix_score_matches_expected": raw_score_matches,
        "passed": not errors,
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--common-actions-dir", default=DEFAULT_COMMON_ACTIONS_DIR)
    parser.add_argument(
        "--container-common-actions-dir",
        default=DEFAULT_CONTAINER_COMMON_ACTIONS_DIR,
        help="Path visible inside the server container for common_actions_20 audio",
    )
    parser.add_argument("--audio-variant", type=int, default=1, choices=(1, 2, 3))
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
    catalog = load_action_catalog(args.catalog)
    common_scenarios = build_common_action_scenarios(
        args.common_actions_dir,
        args.container_common_actions_dir,
        catalog,
        variant_index=args.audio_variant,
    )
    candidates = build_catalog_candidates(
        catalog, [scenario["source_label"] for scenario in common_scenarios]
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "catalog_path": args.catalog,
        "catalog_action_count": len(catalog),
        "common_actions_dir": args.common_actions_dir,
        "container_common_actions_dir": args.container_common_actions_dir,
        "audio_variant": args.audio_variant,
        "common_action_count": len(common_scenarios),
        "candidate_ids": candidate_ids,
        "candidate_catalog_actions": [
            {
                "candidate_id": item["candidate_id"],
                "action_id": item["action_id"],
                "source_label": item["source_label"],
                "execution_binding": item["execution_binding"],
            }
            for item in candidates
        ],
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

    scenarios = [
        *common_scenarios,
        {
            "id": "no-action-explicit",
            "expected_action": "no_action",
            "turns": [
                "我想了解一下人工智能。",
                "请解释一下人工智能是什么，这一轮不需要做任何动作。",
            ],
        },
        {
            "id": "no-action-neutral",
            "expected_action": "no_action",
            "turns": [
                "我们继续聊刚才的话题。",
                "请用两句话总结一下刚才的内容。",
            ],
        },
    ]
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
            # The media already belongs to the preceding multimodal chat turn
            # and is carried through history_audios/history_images. Sending it
            # again as current media duplicates the encoder input and can hang
            # the single-GPU service.
            "score_media": {"audios": [], "images": []},
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
        selection = result["action_id_selection"]
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['scenario_id']}: "
            f"expected={result['expected_action']} "
            f"candidate={selection['selected_action']} "
            f"catalog_action_id={selection.get('selected_action_id')} "
            f"suffix_candidate={score['selected_action']} "
            f"margin={score['top1_margin']}"
        )
        for item in score["ranking"]:
            print(
                f"  #{item['rank']:02d} {item['candidate_id']} "
                f"catalog_action_id={item['action_id']} "
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
