#!/usr/bin/env python3
"""Fit per-label prior offsets from single-token shadow-score JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _score_maps(path: Path) -> Iterable[dict[str, float]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        report = None
    if isinstance(report, dict) and isinstance(report.get("rows"), list):
        for row in report["rows"]:
            ranking = row.get("single_token_shadow_ranking_all")
            if not isinstance(ranking, list):
                continue
            scores = {
                (
                    "000"
                    if item.get("candidate_id") == "UNSUPPORTED"
                    else str(item.get("candidate_id"))
                ): float(item["mean_logprob"])
                for item in ranking
                if isinstance(item, dict)
                and item.get("candidate_id") is not None
                and isinstance(item.get("mean_logprob"), (int, float))
            }
            decision = row.get("single_token_shadow_metadata", {}).get(
                "decision", {}
            )
            groups = decision.get("groups", {}) if isinstance(decision, dict) else {}
            if isinstance(groups, dict):
                for group in groups.values():
                    group_scores = (
                        group.get("scores", {})
                        if isinstance(group, dict) else {}
                    )
                    if isinstance(group_scores, dict):
                        scores.update(
                            {
                                str(key): float(value)
                                for key, value in group_scores.items()
                                if isinstance(value, (int, float))
                            }
                        )
            if scores:
                yield scores
        return
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in _walk(document):
                scores = item.get("single_token_shadow_scores")
                if isinstance(scores, dict) and scores:
                    yield {
                        str(key): float(value)
                        for key, value in scores.items()
                        if isinstance(value, (int, float)) and math.isfinite(value)
                    }
                    break


def _group(candidate_id: str) -> str:
    for prefix in ("IB", "IF", "IR", "IV"):
        if candidate_id.startswith(prefix):
            return prefix
    return "action"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-abs-bias", type=float, default=5.0)
    parser.add_argument(
        "--group",
        action="append",
        choices=("action", "IB", "IF", "IR", "IV"),
        help="Calibrate only selected groups; may be repeated. Defaults to all.",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    manifest = json.loads(mapping_path.read_text(encoding="utf-8"))
    active_ids = [
        str(item["candidate_id"])
        for item in manifest["entries"]
        if item.get("active", True)
    ]
    previous_bias = {
        str(key): float(value)
        for key, value in (
            (manifest.get("calibration") or {}).get("score_bias") or {}
        ).items()
    }
    selected_groups = set(args.group or ("action", "IB", "IF", "IR", "IV"))
    calibrated_ids = [
        candidate_id
        for candidate_id in active_ids
        if _group(candidate_id) in selected_groups
    ]
    values: dict[str, list[float]] = defaultdict(list)
    record_count = 0
    for raw_path in args.logs:
        for scores in _score_maps(Path(raw_path)):
            record_count += 1
            for candidate_id in calibrated_ids:
                if candidate_id in scores:
                    values[candidate_id].append(
                        scores[candidate_id] - previous_bias.get(candidate_id, 0.0)
                    )
    insufficient = {
        candidate_id: len(values[candidate_id])
        for candidate_id in calibrated_ids
        if len(values[candidate_id]) < args.min_samples
    }
    if insufficient:
        raise SystemExit(
            "insufficient shadow samples: "
            + ", ".join(
                f"{key}={count}" for key, count in list(insufficient.items())[:12]
            )
        )

    means = {
        candidate_id: statistics.fmean(values[candidate_id])
        for candidate_id in calibrated_ids
    }
    group_centers = {
        group: statistics.fmean(
            mean for candidate_id, mean in means.items()
            if _group(candidate_id) == group
        )
        for group in {_group(candidate_id) for candidate_id in calibrated_ids}
    }
    fitted_bias = {
        candidate_id: round(
            max(
                -args.max_abs_bias,
                min(
                    args.max_abs_bias,
                    group_centers[_group(candidate_id)] - means[candidate_id],
                ),
            ),
            8,
        )
        for candidate_id in calibrated_ids
    }
    score_bias = {
        candidate_id: fitted_bias.get(
            candidate_id, previous_bias.get(candidate_id, 0.0)
        )
        for candidate_id in active_ids
    }
    manifest["calibration"] = {
        "version": args.version,
        "method": "group_centered_mean_logprob",
        "record_count": record_count,
        "min_samples": args.min_samples,
        "max_abs_bias": args.max_abs_bias,
        "groups": sorted(selected_groups),
        "score_bias": score_bias,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(target),
                "version": args.version,
                "record_count": record_count,
                "label_count": len(fitted_bias),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
