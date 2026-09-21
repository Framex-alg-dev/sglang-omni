import importlib.util
import json
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_pacing_is_start_to_start_and_does_not_catch_up(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import evaluate_action_priority_latency as benchmark
    now = [100.0]
    sleeps = []
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: now[0])

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(benchmark.asyncio, "sleep", sleep)
    assert await benchmark.pace_turn(None, 6) == (100.0, None)
    assert await benchmark.pace_turn(98, 6) == (104.0, 6.0)
    assert sleeps == [4.0]
    # A slow previous turn already exceeded the interval: no additional sleep
    # and no burst of catch-up requests.
    assert await benchmark.pace_turn(90, 6) == (104.0, 14.0)
    assert sleeps == [4.0]


def test_latency_summary_excludes_absent_audio_and_uses_nearest_rank(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("action_latency_test", scripts / "evaluate_action_priority_latency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.summarize([
        {"concurrency": 1, "wire_first_ms": {"turn.action.ready": 10, "response.audio.delta": 30}},
        {"concurrency": 1, "wire_first_ms": {"turn.action.ready": 20}},
    ])
    assert result["1"]["turn.action.ready"]["p50_ms"] == 15
    assert result["1"]["turn.action.ready"]["p95_ms"] == 20
    assert result["1"]["response.audio.delta"]["count"] == 1
    assert result["1"]["response.audio.delta"]["missing"] == 1
    assert result["2"]["turn.action.ready"]["p50_ms"] is None


@pytest.mark.asyncio
async def test_audio_image_turn_never_sends_transcript(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import evaluate_mixed_instructions as mixed
    monkeypatch.setattr(mixed, "load_audio", lambda path: b"\0\0" * 160)
    # No decoder is exercised by this wire-format test.
    image_path = tmp_path / "fixture.jpg"
    image_path.write_bytes(b"jpeg-fixture")

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        async def recv(self):
            last = self.sent[-1]
            kind = {"turn.start": "turn.started", "input.image.append": "input.ack",
                    "input.audio.append": "input.ack", "turn.commit": "turn.result"}[last["type"]]
            return json.dumps({"type": kind, "turn_id": last["turn_id"]})

    ws = Socket()
    await mixed.run_turn(ws, {"audio": "fixture.wav", "image": "fixture.jpg", "text": "DO NOT SEND"}, tmp_path, 2)
    assert [e["type"] for e in ws.sent] == ["turn.start", "input.image.append", "input.audio.append", "turn.commit"]
    assert ws.sent[1]["image_source"] == "user_camera"
    assert set(ws.sent[1]) == {"type", "turn_id", "seq", "image_source", "media_type", "data"}
