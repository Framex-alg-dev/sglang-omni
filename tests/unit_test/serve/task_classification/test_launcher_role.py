from __future__ import annotations

import pytest

from sglang_omni.serve.launcher import _run_server


@pytest.mark.asyncio
async def test_classifier_role_requires_token_before_pipeline_start(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "task-classification")
    monkeypatch.delenv("SGLANG_OMNI_TASK_CLASSIFICATION_TOKEN", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_MODEL_VERSION", "sha256:123")

    with pytest.raises(ValueError, match="TASK_CLASSIFICATION_TOKEN"):
        await _run_server(object())


@pytest.mark.asyncio
async def test_unknown_service_role_fails_before_pipeline_start(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "unknown")

    with pytest.raises(ValueError, match="SERVICE_ROLE"):
        await _run_server(object())


@pytest.mark.asyncio
async def test_timeline_role_requires_token_before_pipeline_start(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "timeline-detection")
    monkeypatch.delenv("SGLANG_OMNI_TIMELINE_DETECTION_TOKEN", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_MODEL_VERSION", "sha256:123")

    with pytest.raises(ValueError, match="TIMELINE_DETECTION_TOKEN"):
        await _run_server(object())


@pytest.mark.asyncio
async def test_decision_services_require_classifier_and_brain_tokens(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "decision-services")
    monkeypatch.setenv("SGLANG_OMNI_TASK_CLASSIFICATION_TOKEN", "classifier")
    monkeypatch.delenv("SGLANG_OMNI_TASK_BRAIN_TOKEN", raising=False)
    monkeypatch.setenv("SGLANG_OMNI_MODEL_VERSION", "prompted-v1")

    with pytest.raises(ValueError, match="TASK_BRAIN_TOKEN"):
        await _run_server(object())


@pytest.mark.asyncio
async def test_decision_services_require_model_version(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_SERVICE_ROLE", "decision-services")
    monkeypatch.setenv("SGLANG_OMNI_TASK_CLASSIFICATION_TOKEN", "classifier")
    monkeypatch.setenv("SGLANG_OMNI_TASK_BRAIN_TOKEN", "brain")
    monkeypatch.delenv("SGLANG_OMNI_MODEL_VERSION", raising=False)

    with pytest.raises(ValueError, match="MODEL_VERSION"):
        await _run_server(object())
