#!/usr/bin/env python3
"""Generate and validate one-token IDs for a canonical action catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglang_omni.models.qwen3_omni.action_token_mapping import (
    DEFAULT_ALLOCATION_SALT,
    DEFAULT_ID_PATTERN,
    allocate_single_token_ids,
    build_mapping_manifest,
    build_runtime_catalog,
    discover_single_token_ids,
    parse_canonical_action_catalog,
    validate_mapping_manifest,
)


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_tokenizer(
    model_path: str,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def _common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        required=True,
        help="Primary deployed Thinker model/tokenizer path.",
    )
    parser.add_argument(
        "--verify-model-path",
        action="append",
        default=[],
        help="Additional deployment tokenizer path; may be repeated.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow Hugging Face network resolution; local files are required by default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", required=True, help="Canonical actions JSON.")
    parser.add_argument("--category-depth", type=int, default=2)
    parser.add_argument(
        "--add-no-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Generate a stable mapping and a realtime action_candidates catalog.",
    )
    _common_model_args(generate)
    _catalog_args(generate)
    generate.add_argument("--mapping-output", required=True)
    generate.add_argument("--catalog-output", required=True)
    generate.add_argument(
        "--previous-mapping",
        help="Preserve all active and retired assignments from a previous manifest.",
    )
    generate.add_argument("--id-pattern", default=DEFAULT_ID_PATTERN)
    generate.add_argument(
        "--min-token-id",
        type=int,
        default=3000,
        help="Exclude the most frequent vocabulary IDs; 3000 fits the deployed Qwen catalog.",
    )
    generate.add_argument("--max-token-id", type=int)
    generate.add_argument("--allocation-salt", default=DEFAULT_ALLOCATION_SALT)

    validate = subparsers.add_parser(
        "validate",
        help="Verify an existing mapping against deployment tokenizer(s) and catalog.",
    )
    _common_model_args(validate)
    _catalog_args(validate)
    validate.add_argument("--mapping", required=True)
    validate.add_argument("--runtime-catalog")
    return parser


def _load_all_tokenizers(args: argparse.Namespace) -> list[tuple[str, Any]]:
    paths = [args.model_path, *args.verify_model_path]
    return [
        (
            path,
            _load_tokenizer(
                path,
                local_files_only=not args.allow_remote,
                trust_remote_code=args.trust_remote_code,
            ),
        )
        for path in paths
    ]


def _validate_runtime_catalog(
    generated: dict[str, Any],
    runtime_catalog_path: str,
) -> None:
    actual, _ = _read_json(runtime_catalog_path)
    if actual != generated:
        raise ValueError(
            "runtime catalog does not match the mapping and canonical source; regenerate it"
        )


def command_generate(args: argparse.Namespace) -> dict[str, Any]:
    source_document, source_sha256 = _read_json(args.catalog)
    parsed = parse_canonical_action_catalog(
        source_document,
        category_depth=args.category_depth,
        add_no_action=args.add_no_action,
    )
    tokenizers = _load_all_tokenizers(args)
    primary_path, primary = tokenizers[0]
    pool = discover_single_token_ids(
        primary,
        pattern=args.id_pattern,
        min_token_id=args.min_token_id,
        max_token_id=args.max_token_id,
    )
    previous = None
    if args.previous_mapping:
        previous, _ = _read_json(args.previous_mapping)
        validate_mapping_manifest(primary, previous)
    entries = allocate_single_token_ids(
        parsed.entities,
        pool,
        allocation_salt=args.allocation_salt,
        previous_mapping=previous,
    )
    manifest = build_mapping_manifest(
        parsed,
        entries,
        tokenizer=primary,
        model_path=primary_path,
        source_catalog_path=args.catalog,
        source_catalog_sha256=source_sha256,
        pattern=args.id_pattern,
        min_token_id=args.min_token_id,
        max_token_id=args.max_token_id,
        allocation_salt=args.allocation_salt,
        pool_size=len(pool),
        category_depth=args.category_depth,
        add_no_action=args.add_no_action,
    )
    validation = []
    for model_path, tokenizer in tokenizers:
        result = validate_mapping_manifest(tokenizer, manifest, parsed=parsed)
        validation.append({"model_path": model_path, **result})
    runtime_catalog = build_runtime_catalog(parsed, manifest)
    _write_json(args.mapping_output, manifest)
    _write_json(args.catalog_output, runtime_catalog)
    return {
        "status": "ok",
        "command": "generate",
        "mapping_output": args.mapping_output,
        "catalog_output": args.catalog_output,
        "counts": manifest["counts"],
        "assignment_sha256": manifest["assignment_sha256"],
        "action_candidates_sha256": runtime_catalog["action_candidates_sha256"],
        "validated_tokenizers": validation,
        "warnings": manifest["warnings"],
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    source_document, source_sha256 = _read_json(args.catalog)
    parsed = parse_canonical_action_catalog(
        source_document,
        category_depth=args.category_depth,
        add_no_action=args.add_no_action,
    )
    manifest, _ = _read_json(args.mapping)
    policy = manifest.get("policy", {})
    if policy.get("category_depth") != args.category_depth:
        raise ValueError("category_depth does not match mapping policy")
    if policy.get("add_no_action") is not args.add_no_action:
        raise ValueError("add_no_action does not match mapping policy")
    expected_source_sha256 = manifest.get("source_catalog", {}).get("sha256")
    if expected_source_sha256 != source_sha256:
        raise ValueError(
            "source catalog sha256 mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    validation = []
    for model_path, tokenizer in _load_all_tokenizers(args):
        result = validate_mapping_manifest(tokenizer, manifest, parsed=parsed)
        validation.append({"model_path": model_path, **result})
    runtime_catalog = build_runtime_catalog(parsed, manifest)
    if args.runtime_catalog:
        _validate_runtime_catalog(runtime_catalog, args.runtime_catalog)
    return {
        "status": "ok",
        "command": "validate",
        "mapping": args.mapping,
        "runtime_catalog": args.runtime_catalog,
        "validated_tokenizers": validation,
        "assignment_sha256": manifest["assignment_sha256"],
        "action_candidates_sha256": runtime_catalog["action_candidates_sha256"],
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "generate":
            result = command_generate(args)
        else:
            result = command_validate(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
