# SPDX-License-Identifier: Apache-2.0
"""Rollout logprob recording uses sampler-computed logprobs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


def test_rollout_logprobs_record_sampler_values_not_raw_logits() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.tensor([[2.0, 1.0]])
    token_ids = torch.tensor([0])
    raw_logprob = torch.log_softmax(logits, dim=-1)[0, 0].item()
    sampler_logprob = torch.log_softmax(logits / 0.5, dim=-1)[0, 0].item()
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    request = SimpleNamespace(data=data)

    runner._record_rollout_logprobs(
        torch.tensor([sampler_logprob]), token_ids, [request]
    )

    assert len(data.output_token_logprobs) == 1
    assert data.output_token_logprobs[0][1] == 0
    assert math.isclose(data.output_token_logprobs[0][0], sampler_logprob, abs_tol=1e-4)
    assert not math.isclose(data.output_token_logprobs[0][0], raw_logprob, abs_tol=1e-4)


def test_rollout_logprobs_align_per_request_at_batch_2() -> None:
    runner = object.__new__(ModelRunner)
    data0 = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    data1 = SimpleNamespace(return_logprob=True, output_token_logprobs=[])
    requests = [SimpleNamespace(data=data0), SimpleNamespace(data=data1)]

    runner._record_rollout_logprobs(
        torch.tensor([-0.5, -1.5]), torch.tensor([11, 22]), requests
    )

    # each request must get ITS OWN row, not the other request's token/logprob
    assert data0.output_token_logprobs[0][1] == 11
    assert data1.output_token_logprobs[0][1] == 22
    assert math.isclose(data0.output_token_logprobs[0][0], -0.5, abs_tol=1e-4)
    assert math.isclose(data1.output_token_logprobs[0][0], -1.5, abs_tol=1e-4)


def test_rollout_logprobs_record_requested_top_tokens_from_same_forward() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(
        return_logprob=True,
        top_logprobs_num=2,
        output_token_logprobs=[],
        output_top_logprobs=[],
    )

    runner._record_rollout_logprobs(
        torch.tensor([-0.1]),
        torch.tensor([11]),
        [SimpleNamespace(data=data)],
        top_logprobs_values=torch.tensor([[-0.1, -0.7]]),
        top_logprobs_indices=torch.tensor([[11, 22]]),
    )

    assert data.output_token_logprobs == [[pytest.approx(-0.1), 11]]
    assert data.output_top_logprobs == [
        [[pytest.approx(-0.1), 11], [pytest.approx(-0.7), 22]]
    ]


def test_rollout_logprobs_raises_on_batch_size_mismatch() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    # more logprobs than requests => batching assumption broke => fail loud
    with pytest.raises(RuntimeError, match="batch-size mismatch"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5, -1.5]),
            torch.tensor([11, 22]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_on_malformed_sampler_shape() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    with pytest.raises(RuntimeError, match="Failed to convert"):
        runner._record_rollout_logprobs(
            [[-0.5, -0.7]],
            torch.tensor([11]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_when_sampler_omits_token_ids() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[])

    with pytest.raises(RuntimeError, match="next_token_ids"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5]),
            None,
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_enable_sampler_logprobs_initializes_missing_forward_batch_fields() -> None:
    forward_batch = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)

    ModelRunner._enable_sampler_logprobs(forward_batch, batch_size=2)

    assert forward_batch.return_logprob is True
    assert forward_batch.top_logprobs_nums == [0, 0]
    assert forward_batch.token_ids_logprobs == [None, None]


def test_enable_sampler_logprobs_preserves_requested_top_k_per_row() -> None:
    forward_batch = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)

    ModelRunner._enable_sampler_logprobs(
        forward_batch,
        batch_size=2,
        top_logprobs_nums=[2, 0],
    )

    assert forward_batch.top_logprobs_nums == [2, 0]


def test_record_rollout_logprobs_skips_without_return_flag() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=False, output_token_logprobs=[])

    runner._record_rollout_logprobs(
        torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
    )

    assert data.output_token_logprobs == []


def test_record_rollout_logprobs_requires_output_list() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True)

    with pytest.raises(AttributeError, match="output_token_logprobs"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
        )

    assert "output_token_logprobs" not in vars(data)


def test_record_rollout_logprobs_requires_return_flag() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(output_token_logprobs=[])

    with pytest.raises(AttributeError, match="return_logprob"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
        )


def test_sample_next_token_ids_requires_sampler_logprobs_when_requested() -> None:
    runner = object.__new__(ModelRunner)
    runner._apply_repetition_penalty = lambda *args: None
    runner._apply_codec_suppress_tokens = lambda *args: None
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda _logits_output, _forward_batch: torch.tensor([44])
        )
    )
    req = SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=None))
    data = SimpleNamespace(return_logprob=True, output_token_logprobs=[], req=req)
    request = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=None),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits_output = SimpleNamespace()

    with pytest.raises(RuntimeError, match="next_token_logprobs"):
        runner._sample_next_token_ids(
            logits_output,
            forward_batch,
            SimpleNamespace(),
            [request],
        )

    assert forward_batch.return_logprob is True
    assert data.output_token_logprobs == []
