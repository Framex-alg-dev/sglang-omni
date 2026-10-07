from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.serve.task_classification.pipeline import (
    TaskClassificationPipeline,
)
from sglang_omni.serve.task_classification.service import (
    create_decision_services_app,
    create_task_classification_app,
)


def _child(name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"name": name}

    return app


def test_mounts_both_services_on_one_api_process() -> None:
    app = create_decision_services_app(
        classifier_app=_child("classifier"),
        brain_app=_child("brain"),
    )

    with TestClient(app) as client:
        classifier = client.get("/classifier/health")
        brain = client.get("/brain/health")

    assert classifier.json() == {"name": "classifier"}
    assert brain.json() == {"name": "brain"}


def test_combined_service_runs_classifier_prefix_prewarm() -> None:
    class Model:
        model_id = "turn-router"
        model_version = "prompt-v1"

        def __init__(self) -> None:
            self.prewarm_calls = 0

        async def prewarm(self) -> None:
            self.prewarm_calls += 1

        async def classify(self, _request):
            raise AssertionError("classification is not expected")

    model = Model()
    classifier = create_task_classification_app(
        TaskClassificationPipeline(model),
        token="secret",
    )
    app = create_decision_services_app(
        classifier_app=classifier,
        brain_app=_child("brain"),
    )

    with TestClient(app):
        assert model.prewarm_calls == 1

    assert model.prewarm_calls == 1
