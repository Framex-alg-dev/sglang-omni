#!/usr/bin/env python3
"""Generate the versioned one-token output-label map used by action scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from sglang_omni.models.qwen3_omni.action_token_mapping import (
    DEFAULT_ALLOCATION_SALT,
    CatalogEntity,
    allocate_single_token_ids,
    discover_single_token_ids,
    sha256_json,
    tokenizer_metadata,
)
from sglang_omni.models.qwen3_omni.global_action_catalog import (
    UNSUPPORTED_CHILD_SCORE_ID,
    load_global_action_catalog,
)
from sglang_omni.serve.realtime.action.decision import (
    BODY_LABELS,
    FACE_LABELS,
    REACTION_LABELS,
    VISUAL_LABELS,
)


def _logical_ids(catalog: object) -> list[str]:
    action_ids = sorted(
        catalog.candidate_by_id,
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    decisions = [
        *BODY_LABELS,
        *FACE_LABELS,
        *REACTION_LABELS,
        *VISUAL_LABELS,
    ]
    return [*action_ids, UNSUPPORTED_CHILD_SCORE_ID, *decisions]


def _build(args: argparse.Namespace) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    catalog = load_global_action_catalog(args.catalog)
    candidate_ids = _logical_ids(catalog)
    entities = [
        CatalogEntity(
            kind="action",
            stable_key=f"selection:{candidate_id}",
            source_id=candidate_id,
            source_label=candidate_id,
            short_definition=candidate_id,
        )
        for candidate_id in candidate_ids
    ]
    pool = discover_single_token_ids(
        tokenizer,
        pattern=r"[A-Z]{2}",
        min_token_id=args.min_token_id,
    )
    allocated = allocate_single_token_ids(
        entities,
        pool,
        allocation_salt=args.allocation_salt,
    )
    by_id = {
        row["source_id"]: row
        for row in allocated
        if row.get("active", True)
    }
    entries = [
        {
            "candidate_id": candidate_id,
            "short_id": by_id[candidate_id]["short_id"],
            "token_id": int(by_id[candidate_id]["token_id"]),
            "active": True,
        }
        for candidate_id in candidate_ids
    ]
    hash_rows = [
        {
            "candidate_id": row["candidate_id"],
            "short_id": row["short_id"],
            "token_id": row["token_id"],
        }
        for row in entries
    ]
    return {
        "schema_version": 1,
        "mapping_kind": "action_selection",
        "mapping_version": args.mapping_version,
        "catalog_hash": catalog.catalog_hash,
        "tokenizer": tokenizer_metadata(tokenizer, args.model_path),
        "policy": {
            "pattern": "[A-Z]{2}",
            "min_token_id": args.min_token_id,
            "allocation_salt": args.allocation_salt,
            "single_token_required": True,
        },
        "counts": {
            "actions": len(catalog.candidate_by_id),
            "unsupported": 1,
            "decisions": len(candidate_ids) - len(catalog.candidate_by_id) - 1,
            "total": len(entries),
            "pool_size": len(pool),
        },
        "assignment_sha256": sha256_json(hash_rows),
        "calibration": {
            "version": "uncalibrated",
            "score_bias": {},
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping-version", default="v1")
    parser.add_argument("--min-token-id", type=int, default=3000)
    parser.add_argument("--allocation-salt", default=DEFAULT_ALLOCATION_SALT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = _build(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    target = Path(args.output)
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"selection-token map is stale: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(target),
                "catalog_hash": payload["catalog_hash"],
                "assignment_sha256": payload["assignment_sha256"],
                "counts": payload["counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
