"""Fail-fast validation for independently deployed decision-model releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_CONTRACTS = {
    "task-classification": "turn-router.v1",
    "timeline-detection": "timeline-detection.v1",
}
_INPUT_MODALITIES = {
    "task-classification": frozenset({"text", "audio"}),
    "timeline-detection": frozenset({"audio", "video"}),
}


@dataclass(frozen=True)
class ModelReleaseManifest:
    path: Path
    service_role: str
    model_id: str
    checkpoint_path: Path
    manifest_version: str
    artifact_count: int


def validate_model_release_from_env(
    *,
    service_role: str,
    pipeline_config: object,
    model_name: str | None,
) -> ModelReleaseManifest:
    """Validate release identity and artifacts before allocating GPU memory."""
    manifest_value = os.environ.get("SGLANG_OMNI_MODEL_MANIFEST", "").strip()
    expected_version = os.environ.get("SGLANG_OMNI_MODEL_VERSION", "").strip()
    if not manifest_value:
        raise ValueError("SGLANG_OMNI_MODEL_MANIFEST is required for decision services")
    manifest_path = Path(manifest_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("decision-model release manifest does not exist")
    raw_bytes = manifest_path.read_bytes()
    actual_version = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    if expected_version != actual_version:
        raise ValueError("SGLANG_OMNI_MODEL_VERSION does not match the release manifest")
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision-model release manifest is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("decision-model release manifest must be an object")
    required = {
        "schema_version",
        "service_role",
        "contract_version",
        "model_id",
        "base_model",
        "checkpoint_path",
        "input_modalities",
        "output_contract",
        "artifacts",
    }
    if set(raw) != required:
        raise ValueError("decision-model release manifest fields do not match schema v1")
    if raw["schema_version"] != 1 or raw["contract_version"] != 1:
        raise ValueError("decision-model release manifest contract is unsupported")
    if raw["service_role"] != service_role:
        raise ValueError("decision-model release manifest has the wrong service role")
    expected_model_id = str(
        model_name or getattr(pipeline_config, "name", "")
    ).strip()
    if not expected_model_id or raw["model_id"] != expected_model_id:
        raise ValueError("decision-model release manifest has the wrong model id")
    if not isinstance(raw["base_model"], str) or not raw["base_model"].strip():
        raise ValueError("decision-model release manifest base_model is required")
    configured_checkpoint = _configured_checkpoint(pipeline_config)
    manifest_checkpoint = _absolute_path(raw["checkpoint_path"], "checkpoint_path")
    if manifest_checkpoint != configured_checkpoint:
        raise ValueError("release manifest checkpoint_path differs from pipeline config")
    if not configured_checkpoint.is_dir():
        raise ValueError("decision-model checkpoint directory does not exist")
    modalities = raw["input_modalities"]
    if (
        not isinstance(modalities, list)
        or not all(isinstance(item, str) for item in modalities)
        or frozenset(modalities) != _INPUT_MODALITIES[service_role]
    ):
        raise ValueError("decision-model release manifest input modalities are invalid")
    if raw["output_contract"] != _OUTPUT_CONTRACTS[service_role]:
        raise ValueError("decision-model release manifest output contract is invalid")
    verify_sha256 = os.environ.get(
        "SGLANG_OMNI_VERIFY_ARTIFACT_SHA256", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    artifact_count = _validate_artifacts(
        configured_checkpoint,
        raw["artifacts"],
        verify_sha256=verify_sha256,
    )
    return ModelReleaseManifest(
        path=manifest_path,
        service_role=service_role,
        model_id=expected_model_id,
        checkpoint_path=configured_checkpoint,
        manifest_version=actual_version,
        artifact_count=artifact_count,
    )


def _configured_checkpoint(pipeline_config: object) -> Path:
    value = getattr(pipeline_config, "model_path", "")
    return _absolute_path(value, "pipeline model_path")


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path.resolve()


def _validate_artifacts(
    checkpoint: Path,
    raw: Any,
    *,
    verify_sha256: bool,
) -> int:
    if not isinstance(raw, list) or not raw:
        raise ValueError("decision-model release manifest artifacts must be non-empty")
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("decision-model artifact fields do not match schema v1")
        relative = item["path"]
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("decision-model artifact path is required")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("decision-model artifact path must stay inside checkpoint")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise ValueError("decision-model artifact paths must be unique")
        seen.add(normalized)
        expected_size = item["size_bytes"]
        expected_sha256 = item["sha256"]
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError("decision-model artifact size_bytes must be non-negative")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise ValueError("decision-model artifact sha256 is invalid")
        artifact = (checkpoint / relative_path).resolve()
        if checkpoint not in artifact.parents or not artifact.is_file():
            raise ValueError("decision-model artifact is missing or outside checkpoint")
        if artifact.stat().st_size != expected_size:
            raise ValueError("decision-model artifact size does not match manifest")
        if verify_sha256 and _file_sha256(artifact) != expected_sha256:
            raise ValueError("decision-model artifact sha256 does not match manifest")
    actual: set[str] = set()
    for path in checkpoint.rglob("*"):
        if not path.is_file():
            continue
        if checkpoint not in path.resolve().parents:
            raise ValueError("decision-model checkpoint contains an external symlink")
        actual.add(path.relative_to(checkpoint).as_posix())
    if actual != seen:
        raise ValueError("decision-model checkpoint files differ from manifest")
    return len(seen)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
