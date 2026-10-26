"""Partial30 training preprocessing is opt-in and leaves legacy defaults intact."""

from __future__ import annotations

import asyncio
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import preprocessor as preprocessor_mod
from sglang_omni.preprocessing import video as video_mod
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.mark.parametrize('patch_factor', [None, 32])
def test_native_video_uses_default_or_opt_in_training_patch_grid(monkeypatch, patch_factor):
    if patch_factor is None:
        monkeypatch.delenv('PARTIAL30_VIDEO_PATCH_FACTOR', raising=False)
    else:
        monkeypatch.setenv('PARTIAL30_VIDEO_PATCH_FACTOR', str(patch_factor))
    video = torch.zeros((6, 3, 496, 864), dtype=torch.uint8)
    monkeypatch.setattr(video_mod.qwen_vision, 'get_video_reader_backend', lambda: 'test')
    monkeypatch.setitem(video_mod.qwen_vision.VIDEO_READER_BACKENDS,
                        'test', lambda element: (video, 2.0))

    actual, fps = video_mod.load_video_path(
        'test.mp4', fps=2, max_frames=20, min_pixels=131072, max_pixels=137625,
    )

    expected_height, expected_width = video_mod.qwen_vision.smart_resize(
        496, 864, factor=patch_factor or video_mod.qwen_vision.IMAGE_FACTOR,
        min_pixels=131072, max_pixels=137625,
    )
    assert actual.shape == (6, 3, expected_height, expected_width)
    assert actual.dtype == torch.float32
    assert fps == 2.0


@pytest.mark.parametrize('parity', [None, '0', '1'])
def test_hf_second_video_resize_is_disabled_only_when_explicitly_enabled(
    monkeypatch, parity,
):
    if parity is None:
        monkeypatch.delenv('PARTIAL30_TRAIN_INPUT_PARITY', raising=False)
    else:
        monkeypatch.setenv('PARTIAL30_TRAIN_INPUT_PARITY', parity)
    captured = {}

    class ProcessorReached(Exception):
        pass

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'rendered prompt'

        def __call__(self, **kwargs):
            captured.update(kwargs)
            raise ProcessorReached

    async def images(*args, **kwargs):
        return []

    async def videos(*args, **kwargs):
        return [torch.zeros((6, 3, 256, 480))], [2.0], None

    monkeypatch.setattr(preprocessor_mod, 'ensure_image_list_async', images)
    monkeypatch.setattr(preprocessor_mod, 'ensure_audio_list_async', images)
    monkeypatch.setattr(preprocessor_mod, 'ensure_video_list_async', videos)
    pre = object.__new__(preprocessor_mod.Qwen3OmniPreprocessor)
    pre.processor = Processor()
    for name in ('fps', 'max_frames', 'min_pixels', 'max_pixels', 'total_pixels'):
        setattr(pre, 'default_video_' + name, None)
    payload = StagePayload(
        request_id='partial30-preprocessing',
        request=OmniRequest(
            inputs={
                'messages': [{'role': 'user', 'content': [{'type': 'video'}]}],
                'videos': ['native-video.mp4'],
                'video_fps': 2.0,
                'video_max_frames': 20,
                'video_min_pixels': 131072,
                'video_max_pixels': 137625,
            },
            metadata={'task': 'timeline_detection'},
        ),
        data={},
    )
    with pytest.raises(ProcessorReached):
        asyncio.run(pre._call_impl(payload))

    kwargs = captured['videos_kwargs']
    assert kwargs['fps'] == 2.0
    assert kwargs['max_frames'] == 20
    assert kwargs['device'] == 'cpu'
    if parity == '1':
        assert kwargs['do_resize'] is False
    else:
        assert 'do_resize' not in kwargs
