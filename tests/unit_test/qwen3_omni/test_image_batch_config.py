from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni import stages
from sglang_omni.scheduling.messages import IncomingMessage


def fake_model(**kwargs):
    return SimpleNamespace(spatial_merge_size=2, out_hidden_size=2048,
                           deepstack_layers=3, visual_dtype_bytes=2)


@pytest.mark.parametrize("limit", [1, 2, 4, 32])
def test_configured_image_batch_limit(monkeypatch, limit):
    monkeypatch.setattr(stages, "Qwen3OmniImageEncoder", fake_model)
    scheduler = stages.create_image_encoder_executor("unused", max_batch_size=limit)
    assert scheduler._max_batch_size == limit
    assert scheduler._max_batch_wait_s == 0.05


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "4"])
def test_invalid_limit_rejected_before_model_load(monkeypatch, limit):
    def forbidden(**kwargs):
        raise AssertionError("model must not load for an invalid limit")
    monkeypatch.setattr(stages, "Qwen3OmniImageEncoder", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        stages.create_image_encoder_executor("unused", max_batch_size=limit)


def test_single_image_never_waits_or_consumes_next_request(monkeypatch):
    monkeypatch.setattr(stages, "Qwen3OmniImageEncoder", fake_model)
    scheduler = stages.create_image_encoder_executor("unused", max_batch_size=1)
    first = IncomingMessage("first", "new_request")
    second = IncomingMessage("second", "new_request")
    scheduler.inbox.put(second)
    assert scheduler._collect_batch(first) == [first]
    assert scheduler.inbox.get_nowait() is second


def test_default_batch_limit_unchanged(monkeypatch):
    monkeypatch.setattr(stages, "Qwen3OmniImageEncoder", fake_model)
    assert stages.create_image_encoder_executor("unused")._max_batch_size == 32
