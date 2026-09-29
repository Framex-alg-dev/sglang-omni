from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.serve.task_classification.service import (
    create_decision_services_app,
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
