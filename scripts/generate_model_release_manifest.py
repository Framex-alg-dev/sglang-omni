#!/usr/bin/env python3
"""Generate a checksummed release manifest for a decision-model checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SERVICE_CONTRACTS = {
    "task-classification": {
        "input_modalities": ["text", "audio"],
        "output_contract": "turn-router.v1",
    },
    "timeline-detection": {
        "input_modalities": ["audio", "video"],
        "output_contract": "timeline-detection.v1",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    checkpoint: Path,
    service_role: str,
    model_id: str,
    base_model: str,
    output: Path,
) -> dict[str, object]:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    artifacts = []
    for path in sorted(checkpoint.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        resolved = path.resolve()
        if checkpoint not in resolved.parents:
            raise ValueError(f"checkpoint contains an external symlink: {path}")
        artifacts.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not artifacts:
        raise ValueError("checkpoint contains no regular artifacts")
    contract = SERVICE_CONTRACTS[service_role]
    return {
        "schema_version": 1,
        "service_role": service_role,
        "contract_version": 1,
        "model_id": model_id,
        "base_model": base_model,
        "checkpoint_path": str(checkpoint),
        "input_modalities": contract["input_modalities"],
        "output_contract": contract["output_contract"],
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--service-role",
        choices=tuple(SERVICE_CONTRACTS),
        required=True,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        checkpoint=args.checkpoint,
        service_role=args.service_role,
        model_id=args.model_id,
        base_model=args.base_model,
        output=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.write_bytes(encoded)
    print(f"manifest={args.output.expanduser().resolve()}")
    print(f"SGLANG_OMNI_MODEL_VERSION=sha256:{hashlib.sha256(encoded).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
