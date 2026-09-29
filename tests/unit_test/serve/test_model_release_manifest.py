from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.generate_model_release_manifest import build_manifest
from sglang_omni.serve.model_release_manifest import (
    validate_model_release_from_env,
)


def _write_manifest(
    root: Path,
    *,
    service_role: str = "task-classification",
    model_id: str = "task-classification",
    artifact_sha256: str | None = None,
) -> tuple[Path, Path]:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    artifact = checkpoint / "weights.bin"
    artifact.write_bytes(b"approved weights")
    digest = artifact_sha256 or hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = root / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "service_role": service_role,
                "contract_version": 1,
                "model_id": model_id,
                "base_model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "checkpoint_path": str(checkpoint),
                "input_modalities": ["text", "audio"],
                "output_contract": "turn-router.v1",
                "artifacts": [
                    {
                        "path": "weights.bin",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": digest,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return checkpoint, manifest


def _configure(monkeypatch, manifest: Path) -> None:
    monkeypatch.setenv("SGLANG_OMNI_MODEL_MANIFEST", str(manifest))
    monkeypatch.setenv(
        "SGLANG_OMNI_MODEL_VERSION",
        "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def test_accepts_exact_manifest_and_checkpoint_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint, manifest = _write_manifest(tmp_path)
    _configure(monkeypatch, manifest)

    release = validate_model_release_from_env(
        service_role="task-classification",
        pipeline_config=SimpleNamespace(
            name="task-classification-dedicated",
            model_path=str(checkpoint),
        ),
        model_name="task-classification",
    )

    assert release.checkpoint_path == checkpoint
    assert release.artifact_count == 1
    assert release.manifest_version.startswith("sha256:")


def test_rejects_manifest_version_mismatch(tmp_path: Path, monkeypatch) -> None:
    checkpoint, manifest = _write_manifest(tmp_path)
    monkeypatch.setenv("SGLANG_OMNI_MODEL_MANIFEST", str(manifest))
    monkeypatch.setenv("SGLANG_OMNI_MODEL_VERSION", "sha256:" + "0" * 64)

    with pytest.raises(ValueError, match="does not match"):
        validate_model_release_from_env(
            service_role="task-classification",
            pipeline_config=SimpleNamespace(model_path=str(checkpoint)),
            model_name="task-classification",
        )


def test_rejects_artifact_checksum_drift_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint, manifest = _write_manifest(
        tmp_path,
        artifact_sha256="0" * 64,
    )
    _configure(monkeypatch, manifest)
    monkeypatch.delenv("SGLANG_OMNI_VERIFY_ARTIFACT_SHA256", raising=False)

    with pytest.raises(ValueError, match="sha256 does not match"):
        validate_model_release_from_env(
            service_role="task-classification",
            pipeline_config=SimpleNamespace(model_path=str(checkpoint)),
            model_name="task-classification",
        )


def test_rejects_wrong_service_role_before_start(tmp_path: Path, monkeypatch) -> None:
    checkpoint, manifest = _write_manifest(
        tmp_path,
        service_role="timeline-detection",
    )
    _configure(monkeypatch, manifest)

    with pytest.raises(ValueError, match="wrong service role"):
        validate_model_release_from_env(
            service_role="task-classification",
            pipeline_config=SimpleNamespace(model_path=str(checkpoint)),
            model_name="task-classification",
        )


def test_rejects_unlisted_checkpoint_files(tmp_path: Path, monkeypatch) -> None:
    checkpoint, manifest = _write_manifest(tmp_path)
    _configure(monkeypatch, manifest)
    (checkpoint / "unreviewed.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="files differ"):
        validate_model_release_from_env(
            service_role="task-classification",
            pipeline_config=SimpleNamespace(model_path=str(checkpoint)),
            model_name="task-classification",
        )


def test_generator_records_every_checkpoint_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "tokenizer").mkdir(parents=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    (checkpoint / "tokenizer" / "config.json").write_text(
        "{}",
        encoding="utf-8",
    )

    manifest = build_manifest(
        checkpoint=checkpoint,
        service_role="timeline-detection",
        model_id="timeline-detection",
        base_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        output=tmp_path / "release.json",
    )

    assert [item["path"] for item in manifest["artifacts"]] == [
        "tokenizer/config.json",
        "weights.bin",
    ]
    assert manifest["input_modalities"] == ["audio", "video"]


def test_generator_uses_turn_router_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")

    manifest = build_manifest(
        checkpoint=checkpoint,
        service_role="task-classification",
        model_id="turn-router",
        base_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        output=tmp_path / "release.json",
    )

    assert manifest["input_modalities"] == ["text", "audio"]
    assert manifest["output_contract"] == "turn-router.v1"
