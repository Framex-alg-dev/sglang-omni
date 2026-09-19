# SPDX-License-Identifier: Apache-2.0
"""Regression tests for idempotent SGLang logprob CPU conversion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.scheduling.omni_scheduler import (
    OmniScheduler,
    _move_logprobs_to_cpu_compat,
)


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


def test_single_token_enforce_finishes_at_prefix_without_suffix_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    plan = {
        "candidate_ids": ["100", "101"],
        "micro_batch_size": 200,
        "scoring_mode": "single_token_enforce",
        "selection_token_ids": {"100": 10, "101": 11},
        "selection_score_bias": {"100": 0.0, "101": 0.25},
        "prefix_physical_prefill_chunk_count": 0,
        "prefix_chunk_timings": [],
    }
    parent = SimpleNamespace(action_scoring_plan=plan)
    data = SimpleNamespace(
        action_scoring_parent=parent,
        action_scoring_plan=plan,
        extra_model_outputs={
            "action_prefix_token_logprobs": {10: -0.5, 11: -1.0}
        },
        generation_steps=1,
    )
    completed = []
    monkeypatch.setattr(
        OmniScheduler,
        "_close_completed_request",
        lambda self, req: None,
    )
    monkeypatch.setattr(
        OmniScheduler,
        "_finish_action_scoring",
        lambda self, current_parent, current_plan: completed.append(
            (current_parent, current_plan)
        ),
    )
    monkeypatch.setattr(
        OmniScheduler,
        "_build_and_enqueue_action_candidate_batch",
        lambda *args, **kwargs: pytest.fail("suffix batch must not be built"),
    )

    scheduler._handle_action_prefix_terminal(SimpleNamespace(), data)

    assert completed == [(parent, plan)]
    assert plan["candidate_batches"] == []
    assert [score.candidate_id for score in plan["single_token_scores"]] == [
        "100",
        "101",
    ]
    assert [score.mean_logprob for score in plan["single_token_scores"]] == [
        pytest.approx(-0.5),
        pytest.approx(-0.75),
    ]


def test_single_token_enforce_never_starts_candidate_materializer() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    parent = SimpleNamespace(
        action_scoring_plan={
            "scoring_mode": "single_token_enforce",
            "candidate_ids": ["100", "101"],
            "micro_batch_size": 200,
        }
    )

    scheduler._start_action_candidate_materialization(parent)

    assert "candidate_materialize_submitted_at" not in parent.action_scoring_plan
