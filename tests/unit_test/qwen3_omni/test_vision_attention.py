from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.qwen3_omni.components import vision_attention as va
from sglang_omni.models.qwen3_omni.components.vision_compat import Qwen3OmniMoeVisionEncoderCompat
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf


def test_default_and_explicit_backend(monkeypatch):
    cfg = hf.Qwen3OmniMoeVisionEncoderConfig(depth=1, hidden_size=16,
        intermediate_size=32, num_heads=2, out_hidden_size=16,
        num_position_embeddings=64, deepstack_visual_indexes=())
    monkeypatch.delenv("SGLANG_OMNI_VISION_ATTENTION_BACKEND", raising=False)
    assert Qwen3OmniMoeVisionEncoderCompat(cfg).blocks[0].attn._omni_rope_state['backend'] == 'fa4'
    monkeypatch.setenv("SGLANG_OMNI_VISION_ATTENTION_BACKEND", "sdpa")
    assert Qwen3OmniMoeVisionEncoderCompat(cfg).blocks[0].attn._omni_rope_state['backend'] == 'sdpa'
    monkeypatch.setenv("SGLANG_OMNI_VISION_ATTENTION_BACKEND", "typo")
    with pytest.raises(ValueError, match="must be fa4 or sdpa"):
        Qwen3OmniMoeVisionEncoderCompat(cfg)


def test_sdpa_does_not_import_fa4(monkeypatch):
    load = Mock(side_effect=AssertionError('unexpected import'))
    monkeypatch.setattr(va, '_load_fa4', load)
    fallback = Mock()
    assert va.select_vision_attention({'backend':'sdpa'}, fallback) is fallback
    load.assert_not_called()


def test_old_beta_rejected(monkeypatch):
    monkeypatch.setattr(va, 'version', lambda name: '4.0.0b19')
    with pytest.raises(ImportError, match='4.0.0b38'):
        va._load_fa4()


def test_missing_dependency_falls_back_once(monkeypatch, caplog):
    load = Mock(side_effect=ImportError('missing'))
    monkeypatch.setattr(va, '_load_fa4', load)
    state = {'backend':'fa4'};fallback = Mock()
    for _ in range(3):
        assert va.select_vision_attention(state, fallback) is fallback
    assert load.call_count == 1
    assert 'using SDPA' in caplog.text


@pytest.mark.parametrize('tuple_output', [False, True])
def test_layout_scale_and_cached_loader(monkeypatch, tuple_output):
    q = torch.randn(1, 2, 7, 8)
    result = q.transpose(1, 2).contiguous()
    fn = Mock(return_value=(result, None) if tuple_output else result)
    load = Mock(return_value=fn);monkeypatch.setattr(va, '_load_fa4', load)
    state = {'backend':'fa4'};fallback = Mock(side_effect=AssertionError('unexpected fallback'))
    for _ in range(2):
        attn = va.select_vision_attention(state, fallback)
        output, weights = attn(None, q, q, q, scaling=.125)
        assert output is result and weights is None
        assert fn.call_args.args[0].shape == (1, 7, 2, 8)
        assert torch.equal(fn.call_args.args[0], q.transpose(1, 2))
        assert fn.call_args.kwargs == {'softmax_scale':.125, 'causal':False}
    assert load.call_count == 1


def test_runtime_errors_are_not_hidden(monkeypatch):
    monkeypatch.setattr(va, '_load_fa4', lambda: Mock(side_effect=RuntimeError('CUDA error')))
    fn = va.select_vision_attention({'backend':'fa4'}, Mock())
    q = torch.randn(1, 2, 7, 8)
    with pytest.raises(RuntimeError, match='CUDA error'):
        fn(None, q, q, q)


@pytest.mark.parametrize('kwargs', [{'dropout':.1}, {'is_causal':True}, {'attention_mask':torch.ones(1)}])
def test_unsupported_attention_options_fall_back(monkeypatch, kwargs):
    kernel = Mock(side_effect=AssertionError('unexpected kernel'))
    monkeypatch.setattr(va, '_load_fa4', lambda: kernel)
    fallback = Mock(return_value=('fallback',None))
    fn = va.select_vision_attention({'backend':'fa4'}, fallback)
    q = torch.randn(1, 2, 7, 8)
    assert fn(None, q, q, q, **kwargs) == ('fallback',None)
    fallback.assert_called_once()
