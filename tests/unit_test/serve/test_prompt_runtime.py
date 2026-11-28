import hashlib
import json
import httpx
import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.prompt_runtime import (
    SPECS,
    decode_bundle,
    load_runtime_bundle,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def bundle(profile="generic", release="release-1", marker="FIRST"):
    consumer, contract, protocol, keys = SPECS[profile]
    binding = {
        "assembly_contract": contract,
        "output_protocol": protocol,
        "model_id": "model",
        "model_version": "weights",
        "catalog_version": None,
        "mapping_version": None,
    }
    value = {
        "tenant_id": "tenant",
        "consumer": consumer,
        "release_id": release,
        "contract_version": contract,
        "binding": binding,
        "items": [
            {
                "key": key,
                "revision": 1,
                "content": marker + " " + key,
                "sha256": hashlib.sha256((marker + " " + key).encode()).hexdigest(),
            }
            for key in keys
        ],
    }
    return {**value, "bundle_sha256": digest(value)}


EXPECTED = {
    "model_id": "model",
    "model_version": "weights",
    "catalog_version": None,
    "mapping_version": None,
}


@pytest.mark.asyncio
async def test_selected_release_is_only_resolved_at_startup():
    current = bundle()
    calls = []

    def respond(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=current)

    env = {
        "KNOWLEDGE_PROMPT_ENABLED": "true",
        "KNOWLEDGE_PROMPT_URL": "http://knowledge",
        "KNOWLEDGE_PROMPT_TENANT_ID": "tenant",
        "KNOWLEDGE_PROMPT_RUNTIME_TOKEN": "secret",
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        loaded = await load_runtime_bundle(
            "generic", expected=EXPECTED, client=client, environment=env
        )
        current = bundle(release="release-2", marker="SECOND")
        assert (
            loaded.release_id == "release-1"
            and loaded.rules["system_rules"].startswith("FIRST")
            and len(calls) == 1
        )
        restarted = await load_runtime_bundle(
            "generic", expected=EXPECTED, client=client, environment=env
        )
        assert restarted.release_id == "release-2" and restarted.rules[
            "system_rules"
        ].startswith("SECOND")
        assert "secret" not in str(restarted.status())


@pytest.mark.parametrize("change", ["scope", "digest", "model", "item", "keys", "pin"])
def test_invalid_published_snapshot_is_rejected(change):
    value = bundle()
    selector = "active"
    if change == "scope":
        value["tenant_id"] = "wrong"
    if change == "model":
        value["binding"]["model_version"] = "wrong"
    if change == "item":
        value["items"][0]["content"] = "tampered"
    if change == "keys":
        value["items"].pop()
    if change == "pin":
        selector = "other"
    value["bundle_sha256"] = (
        digest({k: v for k, v in value.items() if k != "bundle_sha256"})
        if change != "digest"
        else "bad"
    )
    with pytest.raises(ValueError):
        decode_bundle(
            value,
            profile="generic",
            tenant="tenant",
            selector=selector,
            expected=EXPECTED,
        )


@pytest.mark.asyncio
async def test_unavailable_or_unselected_bundle_has_no_default_fallback():
    env = {
        "KNOWLEDGE_PROMPT_ENABLED": "true",
        "KNOWLEDGE_PROMPT_URL": "http://knowledge",
        "KNOWLEDGE_PROMPT_TENANT_ID": "tenant",
        "KNOWLEDGE_PROMPT_RUNTIME_TOKEN": "secret",
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await load_runtime_bundle(
                "e67", expected=EXPECTED, client=client, environment=env
            )


def test_action_uses_agent_snapshot_without_knowledge_io(monkeypatch):
    import sglang_omni.serve.prompt_runtime as runtime
    from sglang_omni.serve.action_decision.service import create_action_decision_app
    from tests.unit_test.serve.test_action_decision_service import (
        _config,
        _FakeClient,
        _raw_action_snapshot,
    )

    model = _FakeClient()

    async def load(*args, **kwargs):
        raise AssertionError("Action Knowledge I/O belongs to Agent")

    monkeypatch.setattr(runtime, "load_runtime_bundle", load)
    monkeypatch.setenv("KNOWLEDGE_PROMPT_ENABLED", "true")
    app = create_action_decision_app(model, config=_config())
    payload = {
        "request_id": "one",
        "session_id": "session",
        "turn_id": "one",
        "decision_point_id": "point",
        "channel": "body",
        "text": "挥手",
        "runtime_context": {"prompt_profile": "e67_v116_step695_fp8"},
        "prompt_snapshot": _raw_action_snapshot("AGENT-FIRST"),
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/action-decision",
            headers={"Authorization": "Bearer secret"},
            json=payload,
        )
        assert response.status_code == 200, response.text
        assert model.requests[-1][1].messages[0].content == "AGENT-FIRST"

        payload.update(
            request_id="two",
            turn_id="two",
            prompt_snapshot=_raw_action_snapshot("AGENT-SECOND"),
        )
        response = client.post(
            "/v1/action-decision",
            headers={"Authorization": "Bearer secret"},
            json=payload,
        )
        assert response.status_code == 200, response.text
        assert model.requests[-1][1].messages[0].content == "AGENT-SECOND"


def test_router_never_loads_knowledge_inside_omni(monkeypatch):
    import sglang_omni.serve.prompt_runtime as runtime
    from sglang_omni.serve.task_classification.client_model import (
        SglangClientTaskClassificationModel,
    )
    from sglang_omni.serve.task_classification.pipeline import (
        TaskClassificationPipeline,
    )
    from sglang_omni.serve.task_classification.service import (
        create_task_classification_app,
    )
    from tests.unit_test.serve.task_classification.test_service import _body, _prompt_snapshot
    from tests.unit_test.serve.test_prompt_debug import _FakeClient

    client_model = _FakeClient(code="0")

    async def load(*args, **kwargs):
        raise AssertionError("Router Knowledge I/O belongs to Agent")

    monkeypatch.setattr(runtime, "load_runtime_bundle", load)
    monkeypatch.setenv("KNOWLEDGE_PROMPT_ENABLED", "true")
    app = create_task_classification_app(
        TaskClassificationPipeline(
            SglangClientTaskClassificationModel(
                client_model, model_id="turn-router", model_version="prompt-v1"
            )
        ),
        token="secret",
    )
    with TestClient(app) as client:
        snapshot = _prompt_snapshot()
        r = client.post(
            "/v1/task-classification",
            headers={"Authorization": "Bearer secret"},
            content=_body("查天气", prompt_snapshot=snapshot),
        )
        assert r.status_code == 200, r.text
        assert client_model.requests[-1][1].messages[0].content == snapshot["system_prompt"]
