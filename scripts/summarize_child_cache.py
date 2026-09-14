"""Summarize paired child_cache_request/result JSONL records, without raw prompts.

Usage: python scripts/summarize_child_cache.py logs/realtime/2026-09-12
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def summarize(paths):
    requests, results = {}, {}
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            for line in file.open():
                if '"child_cache_' not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # A live log may end with an incomplete line.
                key = (record.get("service_instance_id"), record.get("session_id"), record.get("request_id"))
                if record.get("event") == "child_cache_request":
                    requests[key] = record
                elif record.get("event") == "child_cache_result":
                    results[key] = record
    groups = defaultdict(list)
    for key, request in requests.items():
        group = (request["prefix_identity"], request.get("locale"), request.get("turn_origin"), bool(request["identity_seen_before"]))
        groups[group].append(results.get(key, {"status": "pending_or_missing"}))
    rows = []
    for (identity, locale, origin, seen), records in sorted(groups.items()):
        row = {"prefix_identity": identity, "locale": locale, "turn_origin": origin, "seen_before": seen, "requests": len(records)}
        row["status_counts"] = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
        for metric in ("elapsed_ms", "prefix_prefill_ms", "parent_computed_token_count", "parent_cache_hit_ratio"):
            values = sorted(r[metric] for r in records if r["status"] == "completed" and isinstance(r.get(metric), (int, float)))
            def percentile(p):
                n = (len(values)-1)*p
                lo = int(n)
                return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(n-lo)
            row[metric] = {"n": len(values), "mean": mean(values), "p50": percentile(.5), "p95": percentile(.95)} if values else {"n": 0}
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    print(json.dumps(summarize(parser.parse_args().paths), ensure_ascii=False, indent=2))
