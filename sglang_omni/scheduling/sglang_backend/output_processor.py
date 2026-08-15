# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import logging

import torch

from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


logger = logging.getLogger(__name__)


def _to_cpu_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [_to_cpu_python(item) for item in value]
    if isinstance(value, tuple):
        return [_to_cpu_python(item) for item in value]
    return value


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
        should_emit_hidden: Callable[[Any], bool] | None = None,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model
        self._should_emit_hidden = should_emit_hidden

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        host_token_ids: torch.Tensor | None = None,
    ) -> dict[str, RequestOutput]:
        ids = host_token_ids
        if ids is None:
            ids = model_output.next_token_ids
        token_list = ids.tolist() if ids is not None else []

        hidden_extras_by_request: dict[int, dict[str, Any] | None] = {}
        if self._capture_hidden:
            should_emit_hidden_by_request = [
                self._should_emit_hidden_for_request(request)
                for request in scheduler_output.requests
            ]
            hidden_extras_by_request = self._build_hidden_extras_by_request(
                model_output,
                scheduler_output=scheduler_output,
                should_emit_hidden_by_request=should_emit_hidden_by_request,
            )

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = dict(hidden_extras_by_request.get(i) or {})
            extra.update(self._action_scoring_extra(
                model_output, scheduler_output, i, sched_req.data
            ))
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra or None,
            )
        return outputs

    def _action_scoring_extra(
        self, model_output: Any, scheduler_output: SchedulerOutput, index: int, data: Any
    ) -> dict[str, Any]:
        role = getattr(data, "action_scoring_role", None)
        if role is None or model_output.logits_output is None:
            return {}
        logits_output = model_output.logits_output
        if role == "prefix":
            values = getattr(logits_output, "input_token_ids_logprobs_val", None)
            token_ids = getattr(logits_output, "input_token_ids_logprobs_idx", None)
            if values is not None and token_ids is not None:
                values = _to_cpu_python(values)
                token_ids = _to_cpu_python(token_ids)
                if values and isinstance(values[0], list):
                    values = values[index]
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[index]
                # SGLang returns one row per prefill position; a fully cached
                # prefix has exactly one requested position (the dummy token).
                # Keep the final row so chunked-prefill bookkeeping cannot mix
                # earlier prefix positions into the first suffix distribution.
                if values and isinstance(values[0], list):
                    values = values[-1]
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[-1]
                if values and token_ids:
                    return {
                        "action_prefix_token_logprobs": {
                            int(token_id): float(value)
                            for token_id, value in zip(token_ids, values, strict=True)
                        }
                    }

            # Some SGLang prefill-only paths return the next-token distribution
            # but omit input_token_ids_logprobs_* even though the prefix request
            # asks for selected token probabilities. Reuse the selected-token
            # next-token fields when available.
            values = getattr(logits_output, "next_token_token_ids_logprobs_val", None)
            token_ids = getattr(logits_output, "next_token_token_ids_logprobs_idx", None)
            if values is not None and token_ids is not None:
                values = _to_cpu_python(values)
                token_ids = _to_cpu_python(token_ids)
                if values and isinstance(values[0], list):
                    values = values[index]
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[index]
                if values and token_ids:
                    logger.warning(
                        "action scoring prefix logprob fallback source=next_token_selected "
                        "request_id=%s",
                        getattr(getattr(data, "req", None), "rid", None),
                    )
                    return {
                        "action_prefix_token_logprobs": {
                            int(token_id): float(value)
                            for token_id, value in zip(token_ids, values, strict=True)
                        }
                    }

            # Last resort: derive the requested first-suffix-token logprobs
            # from next_token_logits. This path is only used when SGLang did
            # not materialize either selected-token logprob representation.
            logits = getattr(logits_output, "next_token_logits", None)
            req = getattr(data, "req", None)
            req_logprob = getattr(req, "logprob", None)
            requested_ids = getattr(req_logprob, "token_ids_logprob", None)
            if requested_ids is None:
                plan = getattr(data, "action_scoring_plan", None) or {}
                requested_ids = sorted(
                    {
                        int(suffix[0])
                        for suffix in plan.get("candidate_suffix_ids", {}).values()
                        if suffix
                    }
                )
            if logits is not None and requested_ids:
                if hasattr(logits, "detach"):
                    logits = logits.detach().float()
                    if logits.ndim == 2:
                        logits = logits[index]
                    elif logits.ndim != 1:
                        logits = None
                    if logits is not None:
                        logprobs = torch.log_softmax(logits, dim=-1)
                        logger.warning(
                            "action scoring prefix logprob fallback source=next_token_logits "
                            "request_id=%s requested_token_count=%d",
                            getattr(getattr(data, "req", None), "rid", None),
                            len(requested_ids),
                        )
                        return {
                            "action_prefix_token_logprobs": {
                                int(token_id): float(logprobs[int(token_id)].item())
                                for token_id in requested_ids
                            }
                        }

            raise RuntimeError("action scoring prefix selected-token logprobs are missing")
        if role != "candidate":
            return {}
        values = getattr(logits_output, "input_token_logprobs", None)
        if values is None:
            raise RuntimeError("action scoring candidate forward returned no input token logprobs")
        values = list(_to_cpu_python(values))
        batch = scheduler_output.batch_data
        lengths = list(getattr(batch, "extend_lens", []) or [])
        starts = list(getattr(batch, "extend_logprob_start_lens", []) or [])
        if len(lengths) != len(scheduler_output.requests):
            lengths = [len(getattr(req, "origin_input_ids", [])) for req in batch.reqs]
            starts = [0] * len(lengths)
        offset = sum(max(length - start, 0) for length, start in zip(lengths[:index], starts[:index], strict=True))
        count = max(lengths[index] - starts[index] - 1, 0)
        return {"action_candidate_input_token_logprobs": values[offset : offset + count]}

    def _should_emit_hidden_for_request(self, request: Any) -> bool:
        if self._should_emit_hidden is None:
            return True
        return self._should_emit_hidden(request)

    def _build_hidden_extras_by_request(
        self,
        model_output: Any,
        *,
        scheduler_output: SchedulerOutput,
        should_emit_hidden_by_request: list[bool],
    ) -> dict[int, dict[str, Any] | None]:
        request_indexes = [
            i
            for i, should_emit in enumerate(should_emit_hidden_by_request)
            if should_emit
        ]

        if self._model is not None and self._capture_hidden_layers:
            captured_aux_hidden_states = self._model._captured_aux_hidden_states
            if captured_aux_hidden_states is not None:
                self._model._captured_aux_hidden_states = None
                if not request_indexes:
                    return {}
                stream_hidden_states = self._extract_stream_hidden_states(model_output)
                return {
                    request_index: self._build_aux_hidden_extra(
                        captured_aux_hidden_states,
                        request_index=request_index,
                        scheduler_output=scheduler_output,
                        stream_hidden_states=stream_hidden_states,
                    )
                    for request_index in request_indexes
                }

        if not request_indexes:
            return {}

        logits_output = model_output.logits_output
        if logits_output is None:
            return {}
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            return {}

        if isinstance(raw_hidden, dict):
            return {
                request_index: self._build_dict_hidden_extra(
                    raw_hidden,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for request_index in request_indexes
            }
        elif isinstance(raw_hidden, torch.Tensor):
            return {
                request_index: {
                    "hidden_states": self._slice_per_request_tensor(
                        raw_hidden,
                        request_index=request_index,
                        scheduler_output=scheduler_output,
                    )
                }
                for request_index in request_indexes
            }
        return {}

    def _build_aux_hidden_extra(
        self,
        aux_hidden_states: Sequence[torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
        stream_hidden_states: torch.Tensor | None,
    ) -> dict[str, Any]:
        per_request_hidden = {}
        for layer_id, tensor in zip(
            self._capture_hidden_layers or [],
            aux_hidden_states,
        ):
            key = "embed" if layer_id == 0 else layer_id
            per_request_hidden[key] = self._slice_per_request_tensor(
                tensor,
                request_index=request_index,
                scheduler_output=scheduler_output,
            ).clone()

        extra: dict[str, Any] = {"hidden_states": per_request_hidden}
        if stream_hidden_states is not None:
            extra["stream_hidden_states"] = self._slice_per_request_tensor(
                stream_hidden_states,
                request_index=request_index,
                scheduler_output=scheduler_output,
            ).clone()
        return extra

    def _build_dict_hidden_extra(
        self,
        hidden_states: dict[Any, torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, Any]:
        return {
            "hidden_states": {
                key: self._slice_per_request_tensor(
                    tensor,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for key, tensor in hidden_states.items()
            }
        }

    def _extract_stream_hidden_states(self, model_output: Any) -> torch.Tensor | None:
        logits_output = model_output.logits_output
        if logits_output is None:
            return None
        raw_hidden = logits_output.hidden_states
        return raw_hidden if isinstance(raw_hidden, torch.Tensor) else None

    @staticmethod
    def _slice_per_request_tensor(
        tensor: torch.Tensor,
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor

        requests = scheduler_output.requests
        if len(requests) == 1:
            return tensor[0] if tensor.ndim >= 2 else tensor

        batch_data = scheduler_output.batch_data
        reqs = batch_data.reqs
        num_requests = len(reqs)
        if tensor.shape[0] == num_requests:
            return tensor[request_index]

        lengths = [req.extend_range.length for req in reqs]
        total_tokens = sum(lengths)
        if tensor.shape[0] == total_tokens:
            start = sum(lengths[:request_index])
            end = start + lengths[request_index]
            return tensor[start:end]

        return tensor
