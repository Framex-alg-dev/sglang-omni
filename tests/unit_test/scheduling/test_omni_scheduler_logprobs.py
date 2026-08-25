# SPDX-License-Identifier: Apache-2.0
"""Regression tests for idempotent SGLang logprob CPU conversion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.scheduling.omni_scheduler import _move_logprobs_to_cpu_compat


def test_logprob_cpu_transfer_accepts_mixed_tensor_and_list_rows() -> None:
    output = SimpleNamespace(
        next_token_logprobs=torch.tensor([-0.1]),
        input_token_logprobs=torch.tensor([-0.2, -0.3]),
        next_token_top_logprobs_val=[torch.tensor([-0.4]), [-0.5]],
        next_token_top_logprobs_idx=[torch.tensor([1]), [2]],
        next_token_token_ids_logprobs_val=[torch.tensor([-0.6]), [-0.7]],
    )

    _move_logprobs_to_cpu_compat(
        object(), batch=SimpleNamespace(return_logprob=True), logits_output=output
    )
    # A second pass reproduces the previously fatal already-converted path.
    _move_logprobs_to_cpu_compat(
        object(), batch=SimpleNamespace(return_logprob=True), logits_output=output
    )

    assert output.next_token_logprobs == pytest.approx([-0.1])
    assert output.input_token_logprobs == pytest.approx((-0.2, -0.3))
    assert output.next_token_top_logprobs_val[1] == [-0.5]
    assert output.next_token_top_logprobs_idx == [[1], [2]]
    assert output.next_token_token_ids_logprobs_val[1] == [-0.7]
