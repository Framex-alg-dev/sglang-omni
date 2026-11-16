import base64

import pytest
from fastapi.testclient import TestClient

from sglang_omni.serve.action_decision.service import create_action_decision_app
from tests.unit_test.serve.test_persona_audio_window import Client, audio
from tests.unit_test.serve.test_persona_window_prompt import config


AUTH = {"Authorization": "Bearer offline-test"}
PATH = "/v1/action-recall"
REQUEST_ID = "recall-websocket-1"


def _start(socket, payload):
    socket.send_json({
        "type": "request.start", "contract_version": 2,
        "request_id": REQUEST_ID, "payload": payload,
    })
    assert socket.receive_json() == {
        "type": "request.ready", "contract_version": 2, "request_id": REQUEST_ID,
    }


def _send_audio(socket, role):
    item = audio(role)
    socket.send_json({"type": "input.media", "request_id": REQUEST_ID, **item.header()})
    socket.send_bytes(item.payload)
    assert socket.receive_json() == {
        "type": "input.media.ack", "request_id": REQUEST_ID, **item.header(),
    }
    return item


def _commit(socket):
    socket.send_json({"type": "request.commit", "request_id": REQUEST_ID})
    return socket.receive_json()


@pytest.mark.parametrize(
    "channel,selected_code,candidate_ids",
    [("body", "_q_d", ["661", "237"]), ("expression", "_l_a", ["154", "161"])],
)
def test_binary_avatar_audio_matches_http_recall(channel, selected_code, candidate_ids):
    model = Client(selected_code)
    payload = {
        "mode": "avatar_speech", "channel": channel,
        "character_prompt": "沉稳、耐心的老师", "allowed_candidate_ids": candidate_ids,
        "user_input": {"text": "你会讲故事吗？"},
    }
    with TestClient(create_action_decision_app(model, config=config())) as client:
        with client.websocket_connect(PATH, headers=AUTH) as socket:
            _start(socket, payload)
            item = _send_audio(socket, "avatar_speech_audio")
            actual = _commit(socket)
        body = dict(payload, request_id=REQUEST_ID, avatar_current={
            "audio": {"data": base64.b64encode(item.payload).decode(), "format": "wav"},
        })
        expected = client.post("/v1/action-recall", json=body, headers=AUTH)
        assert expected.status_code == 200, expected.text
        assert actual == {
            "type": "response.completed", "request_id": REQUEST_ID,
            "response": expected.json(),
        }
        assert len(model.calls) == 1


def test_body_rejects_avatar_audio_in_user_reaction():
    model = Client("_q_d")
    with TestClient(create_action_decision_app(model, config=config())) as client:
        with client.websocket_connect(PATH, headers=AUTH) as socket:
            _start(socket, {
                "mode": "user_reaction", "channel": "body",
                "character_prompt": "沉稳、耐心的老师",
            })
            _send_audio(socket, "avatar_speech_audio")
            assert _commit(socket)["code"] == "invalid_request"
        assert not model.calls
