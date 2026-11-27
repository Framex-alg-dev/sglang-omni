"""Log identity follows a window through retry and dequeue without changing wire data."""
from dataclasses import replace
import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from framex_event_policy.windows import DetectionWindow, WindowEvent
from sglang_omni.serve.request_log_context import RequestLogContext
from sglang_omni.serve.timeline_detection.client_model import SglangClientTimelineDetectionModel
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile
from sglang_omni.serve.timeline_detection.service import create_timeline_detection_app
from sglang_omni.utils.framex_service_logs import (
    configure_service_logging, current_service_log_context, shutdown_service_logging,
)
from .test_client_model import Client, start, feed_first_window
from tests.unit_test.serve.test_agent_prompt_snapshot import _timeline_snapshot


@pytest.mark.asyncio
async def test_retry_and_queued_output_keep_one_window_operation(monkeypatch):
    class CapturingClient(Client):
        def __init__(self):
            super().__init__(["invalid", "N"])
            self.contexts = []
        async def completion(self, *args, **kwargs):
            self.contexts.append(current_service_log_context())
            return await super().completion(*args, **kwargs)
    client = CapturingClient()
    profile = DirectE50Profile(
        system_prompt="strict training system prompt",
        user_prompt="strict training user prompt",
        prompt_version="timeline-v3-test",
    )
    monkeypatch.setattr(
        DirectE50Profile,
        "from_environment",
        classmethod(lambda cls: profile),
    )
    request = replace(start(), root_session_id="canonical-call")
    model = SglangClientTimelineDetectionModel(client, request,
        model_version="log-contract-test", inference_interval_ms=1000)
    try:
        await model.start(request)
        await feed_first_window(model)
        assert (await model.next_observations()).events == ()
        first, retry = client.contexts
        result = model.output_log_context
        assert first.request_id == retry.request_id == result.request_id
        assert result.root_session_id == "canonical-call"
        assert first.turn_id == retry.turn_id == result.turn_id == "timeline:session-1:0:0:3000"
        assert client.calls[0][1] == "timeline:session-1:0:0:3000:1"
        assert client.calls[1][1] == "timeline:session-1:0:0:3000:2"
    finally:
        await model.close()


def test_raw_window_transport_logs_dequeued_identity_without_wire_metadata(tmp_path):
    class Model:
        model_id = "timeline-model"
        model_version = "test"
        output_log_context = RequestLogContext("req_0123456789abcdef", "call-root", "timeline:session-1:0:0:200")
        def __init__(self): self.outputs = asyncio.Queue()
        async def start(self, request): pass
        async def close(self): pass
        async def discontinuity(self, event): pass
        async def append(self, chunk):
            await self.outputs.put(DetectionWindow(session_id="session-1", identity_epoch=2,
                observer_epoch=0, stream_epoch=0, window_sequence=1,
                evidence_start_ms=100, evidence_end_ms=200, model_id=self.model_id,
                model_version=self.model_version, evidence_mode="audio_video", audio_status="complete",
                events=(WindowEvent(event_type="E5", summary="打哈欠"),)))
        async def next_observations(self): return await self.outputs.get()
    configure_service_logging("omni-event-worker", log_root=tmp_path, version="test", include_stdout=False)
    try:
        app = create_timeline_detection_app(lambda _: Model(), token="secret")
        with TestClient(app) as client, client.websocket_connect('/v1/timeline',
                headers={'Authorization':'Bearer secret'}) as ws:
            ws.send_json(dict(type="session.start", contract_version=3, session_id="session-1",
                root_session_id="call-root", request_id="incoming-start", identity_epoch=2,
                observer_epoch=0, stream_epoch=0, next_sequence=1, audio_format="pcm16/16000/mono",
                video_format="jpeg", model_id="timeline-model",
                prompt_snapshot=_timeline_snapshot()))
            assert ws.receive_json()['type'] == 'session.ready'
            ws.send_json(dict(type="media", session_id="session-1", identity_epoch=2, observer_epoch=0,
                stream_epoch=0, sequence=1, kind="video", start_ms=100, end_ms=200,
                encoding="jpeg", payload_bytes=3))
            ws.send_bytes(b'abc')
            results=[ws.receive_json(),ws.receive_json()]
            window=next(r for r in results if r['type']=='window.result')
            metadata = window.pop('log_context')
            assert metadata['request_id'] == 'req_0123456789abcdef'
            assert DetectionWindow.from_wire(window).events[0].event_type == 'E5'
            ws.send_json({'type':'session.close'})
            assert ws.receive_json()['type']=='session.closed'
    finally:
        shutdown_service_logging('omni-event-worker')
    records=[json.loads(line) for line in (tmp_path/'omni-event-worker/app.log').read_text().splitlines()]
    done=next(r for r in records if r.get('operation')=='timeline_window')
    assert done['request_id']=='req_0123456789abcdef'
    assert done['root_session_id']=='call-root'
    assert done['turn_id']==done['operation_id']=='timeline:session-1:0:0:200'
    assert next(r for r in records if r['event']=='ws.connected')['request_id']=='incoming-start'
