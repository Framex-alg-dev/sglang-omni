from types import SimpleNamespace
import pytest

from sglang_omni.serve.inference_gateway.app import _validate_action_stream_media, _validate_evidence_role
from sglang_omni.serve.streaming_request import _validate_evidence_role as validate_stream_media
from sglang_omni.serve.action_decision.service import _parse_request


def test_assistant_audio_retains_its_provenance_through_both_media_boundaries():
    media=SimpleNamespace(kind="audio", evidence_role="assistant_spoken", encoding="wav")
    _validate_evidence_role(media)
    validate_stream_media(media)
    _validate_action_stream_media("body",{"turn_origin":"assistant"},[media])


@pytest.mark.parametrize("stage,source", [("body","user"),("expression","assistant"),
    ("classifier","assistant"),("brain","assistant"),("reply","assistant")])
def test_assistant_audio_cannot_be_misrepresented_as_user_or_expression(stage,source):
    media=SimpleNamespace(kind="audio",evidence_role="assistant_spoken",encoding="wav")
    with pytest.raises(ValueError):
        _validate_action_stream_media(stage,{"turn_origin":source},[media])


def test_upstream_action_service_rechecks_source_even_without_gateway():
    media=SimpleNamespace(kind="audio",evidence_role="assistant_spoken",encoding="wav")
    payload={"channel":"body","request_id":"a","session_id":"s","turn_id":"t",
             "decision_point_id":"w","turn_origin":"assistant"}
    assert _parse_request(payload,media=(media,)).turn_origin=="assistant"
    payload["turn_origin"]="user"
    with pytest.raises(ValueError):
        _parse_request(payload,media=(media,))
