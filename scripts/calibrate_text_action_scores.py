#!/usr/bin/env python3
"""Offline background-baseline calibration for text action score reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION = ROOT / "reports/text_action_recall_18004_full_scores.json"
DEFAULT_OUTPUT = ROOT / "reports/text_action_recall_background_calibration.json"


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("rows"):
        raise ValueError(f"report has no rows: {path}")
    return report


def full_scores(row: dict[str, Any]) -> dict[str, float]:
    ranking = row.get("ranking_all")
    if not ranking:
        raise ValueError(
            f"row {row.get('id')!r} has no ranking_all; rerun evaluator with "
            "--save-all-scores"
        )
    scores: dict[str, float] = {}
    for item in ranking:
        candidate_id = str(item["candidate_id"])
        value = item.get("mean_logprob")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"row {row.get('id')!r} has invalid score for {candidate_id!r}"
            )
        scores[candidate_id] = float(value)
    return scores


def catalog_hashes(report: dict[str, Any]) -> set[str]:
    return {
        str(row["catalog_hash"])
        for row in report["rows"]
        if row.get("catalog_hash")
    }


def build_baseline(
    report: dict[str, Any],
    *,
    group: str | None,
    exclude_expected: bool,
) -> tuple[dict[str, float], dict[str, float], int]:
    values: dict[str, list[float]] = defaultdict(list)
    candidate_sets: set[frozenset[str]] = set()
    rows = [
        row
        for row in report["rows"]
        if (group is None or row.get("group") == group) and row.get("ranking_all")
    ]
    if not rows:
        raise ValueError(f"baseline has no scored rows for group={group!r}")
    for row in rows:
        scores = full_scores(row)
        candidate_sets.add(frozenset(scores))
        expected_ids = set(row.get("expected", {}).get("candidate_ids") or [])
        for candidate_id, value in scores.items():
            if exclude_expected and candidate_id in expected_ids:
                continue
            values[candidate_id].append(value)
    if len(candidate_sets) != 1:
        raise ValueError("neutral rows do not contain an identical candidate set")
    means = {candidate_id: statistics.fmean(items) for candidate_id, items in values.items()}
    stdevs = {
        candidate_id: statistics.stdev(items) if len(items) > 1 else 0.0
        for candidate_id, items in values.items()
    }
    return means, stdevs, len(rows)


def divide(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_alpha(
    rows: list[dict[str, Any]],
    baseline: dict[str, float],
    *,
    alpha: float,
    unsupported_alpha: float,
    median_baseline: float,
) -> dict[str, Any]:
    evaluated = 0
    top1 = 0
    top3 = 0
    top5 = 0
    supported_evaluated = 0
    supported_top1 = 0
    unsupported_expected = 0
    unsupported_top1 = 0
    unsupported_false_positive = 0
    error_selected: Counter[str] = Counter()
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    selections: list[dict[str, Any]] = []
    skipped_unscored = 0

    for row in rows:
        expected_ids = set(row.get("expected", {}).get("candidate_ids") or [])
        if not expected_ids:
            continue
        if not row.get("ranking_all"):
            skipped_unscored += 1
            continue
        raw_scores = full_scores(row)
        missing = set(raw_scores) - set(baseline)
        if missing:
            raise ValueError(
                f"baseline is missing {len(missing)} candidates, including "
                f"{sorted(missing)[:5]}"
            )
        def calibrated(candidate_id: str) -> float:
            candidate_alpha = (
                unsupported_alpha if candidate_id == "UNSUPPORTED" else alpha
            )
            return raw_scores[candidate_id] - candidate_alpha * (
                baseline[candidate_id] - median_baseline
            )

        ranked = sorted(
            raw_scores,
            key=calibrated,
            reverse=True,
        )
        selected = ranked[0]
        evaluated += 1
        top1_ok = selected in expected_ids
        top3_ok = bool(expected_ids.intersection(ranked[:3]))
        top5_ok = bool(expected_ids.intersection(ranked[:5]))
        top1 += top1_ok
        top3 += top3_ok
        top5 += top5_ok
        group = str(row.get("group"))
        group_counts[group]["evaluated"] += 1
        group_counts[group]["top1"] += top1_ok
        group_counts[group]["top3"] += top3_ok
        group_counts[group]["top5"] += top5_ok
        if "UNSUPPORTED" in expected_ids:
            unsupported_expected += 1
            unsupported_top1 += selected == "UNSUPPORTED"
        else:
            supported_evaluated += 1
            supported_top1 += top1_ok
            unsupported_false_positive += selected == "UNSUPPORTED"
        if not top1_ok:
            error_selected[selected] += 1
        selections.append(
            {
                "id": row.get("id"),
                "group": group,
                "text": row.get("text"),
                "expected": sorted(expected_ids),
                "selected": selected,
                "top1": top1_ok,
                "raw_score": raw_scores[selected],
                "baseline": baseline[selected],
                "calibrated_score": calibrated(selected),
                "expected_ranks": {
                    candidate_id: ranked.index(candidate_id) + 1
                    for candidate_id in expected_ids
                    if candidate_id in ranked
                },
            }
        )

    unsupported_predicted = unsupported_top1 + unsupported_false_positive
    return {
        "alpha": alpha,
        "unsupported_alpha": unsupported_alpha,
        "evaluated": evaluated,
        "skipped_unscored": skipped_unscored,
        "top1": {"passed": top1, "rate": divide(top1, evaluated)},
        "top3": {"passed": top3, "rate": divide(top3, evaluated)},
        "top5": {"passed": top5, "rate": divide(top5, evaluated)},
        "supported_top1": {
            "passed": supported_top1,
            "evaluated": supported_evaluated,
            "rate": divide(supported_top1, supported_evaluated),
        },
        "unsupported": {
            "true_positive": unsupported_top1,
            "expected": unsupported_expected,
            "false_positive": unsupported_false_positive,
            "predicted": unsupported_predicted,
            "precision": divide(unsupported_top1, unsupported_predicted),
            "recall": divide(unsupported_top1, unsupported_expected),
        },
        "groups": {
            group: {
                "evaluated": counts["evaluated"],
                "top1": {
                    "passed": counts["top1"],
                    "rate": divide(counts["top1"], counts["evaluated"]),
                },
                "top3": {
                    "passed": counts["top3"],
                    "rate": divide(counts["top3"], counts["evaluated"]),
                },
                "top5": {
                    "passed": counts["top5"],
                    "rate": divide(counts["top5"], counts["evaluated"]),
                },
            }
            for group, counts in sorted(group_counts.items())
        },
        "top_error_selections": [
            {"candidate_id": candidate_id, "count": count}
            for candidate_id, count in error_selected.most_common(15)
        ],
        "selections": selections,
    }


def parse_alphas(values: str) -> list[float]:
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed or any(value < 0.0 for value in parsed):
        raise argparse.ArgumentTypeError("alphas must be nonnegative comma-separated numbers")
    return parsed


def parse_alpha_pairs(values: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in values.split(","):
        if not item.strip():
            continue
        left, separator, right = item.partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(
                "alpha pairs must use action_alpha:unsupported_alpha"
            )
        pair = (float(left), float(right))
        if any(value < 0.0 for value in pair):
            raise argparse.ArgumentTypeError("alpha pairs must be nonnegative")
        pairs.append(pair)
    if not pairs:
        raise argparse.ArgumentTypeError("at least one alpha pair is required")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline-group",
        default="canonical_label",
        help="Use only this group to estimate the balanced background baseline.",
    )
    parser.add_argument(
        "--include-own-positive-in-baseline",
        action="store_true",
        help="Do not exclude a candidate's own positive row from its baseline.",
    )
    parser.add_argument(
        "--alphas", type=parse_alphas, default=parse_alphas("0,0.25,0.5,0.75,1")
    )
    parser.add_argument(
        "--alpha-pairs",
        type=parse_alpha_pairs,
        default=parse_alpha_pairs("0.55:0.15,0.6:0.2"),
        help="Additional action_alpha:unsupported_alpha experiments.",
    )
    args = parser.parse_args()

    evaluation_report = load_report(args.evaluation)
    baseline_report = (
        load_report(args.baseline) if args.baseline is not None else evaluation_report
    )
    baseline_hashes = catalog_hashes(baseline_report)
    evaluation_hashes = catalog_hashes(evaluation_report)
    if baseline_hashes != evaluation_hashes:
        raise ValueError(
            f"catalog hash mismatch: baseline={sorted(baseline_hashes)}, "
            f"evaluation={sorted(evaluation_hashes)}"
        )

    baseline, stdev, baseline_case_count = build_baseline(
        baseline_report,
        group=args.baseline_group,
        exclude_expected=not args.include_own_positive_in_baseline,
    )
    median_baseline = statistics.median(baseline.values())
    alpha_pairs = [(alpha, alpha) for alpha in args.alphas]
    alpha_pairs.extend(args.alpha_pairs)
    alpha_pairs = list(dict.fromkeys(alpha_pairs))
    experiments = [
        evaluate_alpha(
            evaluation_report["rows"],
            baseline,
            alpha=alpha,
            unsupported_alpha=unsupported_alpha,
            median_baseline=median_baseline,
        )
        for alpha, unsupported_alpha in alpha_pairs
    ]
    priors = sorted(
        (
            {
                "candidate_id": candidate_id,
                "mean_logprob": value,
                "stdev": stdev[candidate_id],
                "centered_bias": value - median_baseline,
            }
            for candidate_id, value in baseline.items()
        ),
        key=lambda item: item["mean_logprob"],
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "method": "balanced_background_mean_logprob_subtraction",
        "formula": (
            "calibrated_score = raw_mean_logprob - candidate_alpha * "
            "(background_mean_logprob - median_background_mean_logprob)"
        ),
        "warning": (
            "This baseline mixes identifier/token prior with cross-action semantic "
            "attractiveness; it is an offline experiment, not a production artifact."
        ),
        "baseline_report": str(args.baseline or args.evaluation),
        "baseline_group": args.baseline_group,
        "baseline_excludes_own_positive": not args.include_own_positive_in_baseline,
        "evaluation_report": str(args.evaluation),
        "catalog_hashes": sorted(baseline_hashes),
        "baseline_case_count": baseline_case_count,
        "candidate_count": len(baseline),
        "median_background_mean_logprob": median_baseline,
        "candidate_priors": priors,
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    concise = {
        "baseline_case_count": output["baseline_case_count"],
        "candidate_count": output["candidate_count"],
        "watched_priors": {
            candidate_id: next(
                (item for item in priors if item["candidate_id"] == candidate_id), None
            )
            for candidate_id in ("130", "135", "UNSUPPORTED")
        },
        "experiments": [
            {
                "alpha": item["alpha"],
                "unsupported_alpha": item["unsupported_alpha"],
                "top1": item["top1"],
                "top3": item["top3"],
                "supported_top1": item["supported_top1"],
                "unsupported": item["unsupported"],
                "top_error_selections": item["top_error_selections"][:5],
            }
            for item in experiments
        ],
        "output": str(args.output),
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
