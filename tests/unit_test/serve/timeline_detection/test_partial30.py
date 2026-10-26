from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from sglang_omni.models.qwen3_omni.public_prefix import is_published_prompt
from sglang_omni.serve.timeline_detection.client_model import SglangClientTimelineDetectionModel, _WindowWork
from sglang_omni.serve.timeline_detection.direct_e50 import DirectE50Profile, build_direct_e50_request, build_direct_e50_prefill_request, parse_direct_e50_output
from sglang_omni.serve.timeline_detection.parser import ModelOutputError
from sglang_omni.serve.timeline_detection.partial30 import PARTIAL30_EVENT_IDS, PARTIAL30_EVENT_SUMMARIES, PARTIAL30_PROTOCOL
from sglang_omni.serve.timeline_detection.contracts import MediaKind, TimelineMediaChunk, TimelineSessionStart
from sglang_omni.serve.timeline_detection.window import AudioEvidenceStatus, EventDetectionWindow


ROOT = Path(__file__).resolve().parents[4]
PROMPTS = ROOT / "deploy/prompts/partial30-checkpoint1750"


def profile():
    return DirectE50Profile(system_prompt=(PROMPTS / "system_prompt.txt").read_text(),
                            user_prompt="", prompt_version="partial30-test",
                            max_tokens=128, video_max_pixels=137625,
                            protocol=PARTIAL30_PROTOCOL, confirmation_windows=1)


def window():
    output = io.BytesIO()
    # A distinct prefix lets the test prove audio is trimmed, not just relabeled.
    with wave.open(output, "wb") as target:
        target.setnchannels(1); target.setsampwidth(2); target.setframerate(16000)
        target.writeframes(b"\x01\x00" * 8000 + b"\x02\x00" * 104000)
    jpeg = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (320, 240), (48, 96, 144)).save(jpeg, format="JPEG")
    frames = tuple(TimelineMediaChunk(session_id="test-partial30", identity_epoch=1,
        stream_epoch=0, sequence=i + 1, kind=MediaKind.VIDEO,
        start_ms=(i + 1) * 500, end_ms=(i + 1) * 500 + 100,
        encoding="image/jpeg", payload=jpeg.getvalue()) for i in range(14))
    return EventDetectionWindow(start_ms=0, current_start_ms=4000, end_ms=7000,
        history_frames=frames[:8], current_frames=frames[8:],
        audio_wav=output.getvalue(), audio_start_ms=0,
        audio_status=AudioEvidenceStatus.COMPLETE)


def request(value):
    return build_direct_e50_request(profile=profile(), model_id="qwen3-omni-gpu4-timeline",
        session_id="test-partial30", window=value, video_mp4=b"native-video",
        retry=False, logical_request_id="test", contract_version=2,
        observer_epoch=0, stream_epoch=0)


def test_event_ids_and_names_match_frozen_training_catalog():
    events = json.loads((PROMPTS / "events.json").read_text())
    assert {row['code'] for row in events} == PARTIAL30_EVENT_IDS
    assert len(events) == len(PARTIAL30_EVENT_IDS) == 30
    for row in events:
        assert PARTIAL30_EVENT_SUMMARIES[row['code']] == row['name']


def test_profile_reads_exact_training_prompt_without_a_user_prompt(monkeypatch):
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_MODEL_PROTOCOL', PARTIAL30_PROTOCOL)
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_SYSTEM_PROMPT_PATH',
                       str(PROMPTS / 'system_prompt.txt'))
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_USER_PROMPT_PATH', '/not-a-prompt')
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_PROMPT_VERSION', 'partial30-test')
    monkeypatch.delenv('SGLANG_OMNI_TIMELINE_MAX_TOKENS', raising=False)

    value = DirectE50Profile.from_environment()

    assert value is not None
    assert value.protocol == PARTIAL30_PROTOCOL
    assert value.system_prompt == (PROMPTS / 'system_prompt.txt').read_text()
    assert value.system_prompt.endswith('\n')
    assert value.user_prompt == ''
    assert value.allowed_event_ids == PARTIAL30_EVENT_IDS
    assert value.max_tokens == 128
    assert value.video_max_pixels == 137625


@pytest.mark.parametrize('raw, expected', [
    ('N', []),
    ('E38,E38', ['E38', 'E38']),
    ('E0,E34', ['E0', 'E34']),
])
def test_partial30_preserves_neutral_and_multi_event_output(raw, expected):
    assert parse_direct_e50_output(raw, allowed_event_ids=PARTIAL30_EVENT_IDS) == expected


@pytest.mark.parametrize('code', sorted(PARTIAL30_EVENT_IDS))
def test_all_retained_ids_are_forwarded_unchanged(code):
    assert parse_direct_e50_output(code, allowed_event_ids=PARTIAL30_EVENT_IDS) == [code]


@pytest.mark.parametrize('code', ['E1', 'E3', 'E27', 'E40', 'E48', 'E51', 'P13', '{"event":"E20"}'])
def test_removed_codes_aliases_and_json_cannot_escape(code):
    with pytest.raises(ModelOutputError):
        parse_direct_e50_output(code, allowed_event_ids=PARTIAL30_EVENT_IDS)


def test_digit_codes_are_not_legacy_collapsed():
    assert parse_direct_e50_output('E13,E14', allowed_event_ids=PARTIAL30_EVENT_IDS) == ['E13', 'E14']


def test_payload_matches_frozen_wrapper_and_real_frame_timestamps():
    value = request(window())
    parts = value.messages[1].content
    rows = value.metadata['frame_manifest']
    assert rows[0] == 'F01|-6.500|H'
    assert rows[7] == 'F08|-3.000|H'
    assert rows[8] == 'F09|-2.500|C'
    assert rows[-1] == 'F14|0.000|C'
    assert [x['type'] for x in parts] == ['text', 'video', 'text', 'audio', 'text']
    assert parts[0]['text'].endswith('\n</request_window>\n')
    assert parts[2]['text'] == '\n【同步音频：从首个输入画面时刻-6.500秒至本轮末端0秒】\n'
    assert parts[-1]['text'] == '\n结合完整H+C确认事件成立并定位触发点，只输出触发点属于C的合法E码。触发点只在H、尚未达到条件或证据不足时输出N。'
    assert len(value.metadata['videos']) == len(value.metadata['audios']) == 1
    assert value.metadata['video_min_pixels'] == 131072
    assert value.metadata['video_max_pixels'] == 137625
    assert value.metadata['use_audio_in_video'] is False
    assert value.sampling.seed == 42 and value.max_tokens == 128
    audio = base64.b64decode(value.metadata['audios'][0].split(',', 1)[1])
    with wave.open(io.BytesIO(audio), 'rb') as source:
        assert source.getnframes() == 104000
        assert source.readframes(10) == b'\x02\x00' * 10


def test_missing_audio_uses_no_placeholder_or_silent_fabrication():
    value = request(replace(window(), audio_wav=None, audio_start_ms=None,
                            audio_status=AudioEvidenceStatus.MISSING))
    assert 'audios' not in value.metadata
    assert all(x['type'] != 'audio' for x in value.messages[1].content)
    assert '【本轮未提供有效音频】' in str(value.messages[1].content)


def test_static_prefill_keeps_media_private_and_approves_exact_prompt():
    value = build_direct_e50_prefill_request(profile=profile(), model_id='timeline-model',
        session_id='test', logical_request_id='prefill')
    assert value.metadata['videos'] == value.metadata['audios'] == []
    assert value.metadata['prefill_only'] is True
    assert value.sampling.seed == 42
    assert hashlib.sha256(profile().system_prompt.encode()).hexdigest() == 'dae551e6bb93ab2d007f402bcc6157873e746c149aa4d1d46f8241e2198b3e17'
    assert is_published_prompt(profile().system_prompt)
    assert not is_published_prompt(profile().system_prompt + '\nchanged')


@pytest.mark.asyncio
async def test_model_observations_preserve_control_and_thumbs_up_ids(monkeypatch, tmp_path):
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT', str(tmp_path))
    monkeypatch.setattr(DirectE50Profile, 'from_environment', classmethod(lambda cls: profile()))
    class Client:
        async def completion(self, request, *, request_id, **kwargs):
            return SimpleNamespace(text='E13,E14,E20,E21')
    start = TimelineSessionStart(session_id='test-partial30', identity_epoch=1,
        stream_epoch=0, audio_format='pcm_s16le/16000/1', video_format='image/jpeg',
        model_id='qwen3-omni-gpu4-timeline')
    model = SglangClientTimelineDetectionModel(Client(), start,
        model_version='partial30-checkpoint-1750-fp8-20261020', inference_interval_ms=1000)
    try:
        observations = await model._detect(_WindowWork(0, 0, window()))
        assert [event.event_type for event in observations] == ['E13', 'E14', 'E20', 'E21']
        assert [event.summary for event in observations] == ['点赞', 'OK手势', '嘘手势', '暂停手势']
    finally:
        await model.close()


@pytest.mark.parametrize('status', [
    AudioEvidenceStatus.MISSING, AudioEvidenceStatus.GAPPED,
    AudioEvidenceStatus.BEHIND,
])
def test_unavailable_audio_is_never_fabricated(status):
    value = request(replace(window(), audio_wav=None, audio_start_ms=None,
                            audio_status=status))
    assert 'audios' not in value.metadata
    assert [part['type'] for part in value.messages[1].content] == [
        'text', 'video', 'text', 'text',
    ]


def test_audio_that_starts_after_first_frame_falls_back_to_video_only():
    value = request(replace(window(), audio_start_ms=1000))
    assert 'audios' not in value.metadata
    assert value.metadata['audio_start_ms'] is None


def test_cold_start_has_six_current_frames_and_no_history():
    source = window()
    value = request(replace(source, start_ms=4000, history_frames=()))
    assert value.metadata['history_frame_count'] == 0
    assert value.metadata['current_frame_count'] == 6
    assert value.metadata['frame_manifest'][0] == 'F01|-2.500|C'
    assert value.metadata['frame_manifest'][-1] == 'F06|0.000|C'


@pytest.mark.asyncio
async def test_e49_observation_uses_partial30_name_and_saves_aligned_audio(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv('SGLANG_OMNI_TIMELINE_ARTIFACT_ROOT', str(tmp_path))
    monkeypatch.setattr(DirectE50Profile, 'from_environment',
                        classmethod(lambda cls: profile()))

    class Client:
        async def completion(self, request, *, request_id, **kwargs):
            return SimpleNamespace(text='E49')

    start = TimelineSessionStart(
        session_id='test-partial30', identity_epoch=1, stream_epoch=0,
        audio_format='pcm_s16le/16000/1', video_format='image/jpeg',
        model_id='qwen3-omni-gpu4-timeline',
    )
    model = SglangClientTimelineDetectionModel(
        Client(), start, model_version='partial30-test', inference_interval_ms=1000,
    )
    try:
        observations = await model._detect(_WindowWork(0, 0, window()))
        assert [event.summary for event in observations] == ['频繁看手表']
    finally:
        await model.close()

    audio_paths = list(tmp_path.rglob('*.wav'))
    assert len(audio_paths) == 1
    expected = base64.b64decode(request(window()).metadata['audios'][0].split(',', 1)[1])
    assert audio_paths[0].read_bytes() == expected
