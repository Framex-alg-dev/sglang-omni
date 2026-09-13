"""Bounded, process-local observations of Child prefixes (not a KV cache)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import time
import uuid
from typing import Any


@lru_cache(maxsize=128)
def prompt_sha256(text: str) -> str:
    """Reuse exact-text digests without changing any cache namespace."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_hash(value: Any) -> str:
    return prompt_sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@dataclass
class _Seen:
    count: int
    first_seen: float
    last_seen: float
    request_id: str
    status: str = "submitted"


class ChildCacheObserver:
    def __init__(self, capacity: int = 1024, ttl_seconds: float = 3600) -> None:
        if capacity < 1 or ttl_seconds < 0:
            raise ValueError("observation capacity must be positive and TTL non-negative")
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.epoch = uuid.uuid4().hex
        self._seen: OrderedDict[str, _Seen] = OrderedDict()

    def submitted(self, request: Any, category_ids: list[str], locale: str) -> dict[str, Any]:
        now = time.monotonic()
        while self._seen and now - next(iter(self._seen.values())).last_seen >= self.ttl_seconds:
            self._seen.popitem(last=False)
        candidates = [c.candidate_id for c in request.candidates]
        fields = {
            "selected_category_ids": list(category_ids),
            "candidate_ids_ordered": candidates,
            "candidate_count": len(candidates),
            "candidate_set_hash": identity_hash(sorted(set(candidates))),
            "candidate_order_hash": identity_hash(candidates),
            "static_prompt_sha256": prompt_sha256(request.system_prompt or ""),
            "session_instruction_sha256": prompt_sha256(request.session_instruction or ""),
            "prefix_cache_namespace": request.prefix_cache_namespace,
            "locale": locale,
            "language": request.language,
            "turn_origin": request.turn_origin,
            "model": request.model,
            "tokenizer_identity": None,  # Not supplied by the scoring client.
            "cache_static_system_only": request.cache_static_system_only,
            "process_epoch": self.epoch,
            "seen_scope": "process_window",
            "seen_capacity": self.capacity,
            "seen_ttl_seconds": self.ttl_seconds,
        }
        prefix_identity = identity_hash([
            request.model, request.prefix_cache_namespace, request.language,
            fields["static_prompt_sha256"], fields["session_instruction_sha256"],
            request.cache_static_system_only,
        ])
        previous = self._seen.pop(prefix_identity, None)
        fields.update(
            prefix_identity=prefix_identity,
            scoring_identity=identity_hash([
                prefix_identity, category_ids,
                [(c.candidate_id, c.suffix) for c in request.candidates],
                request.suffix_tokenization_mode, request.micro_batch_size,
            ]),
            identity_seen_before=previous is not None,
            seen_count_before=previous.count if previous else 0,
            first_seen_monotonic=previous.first_seen if previous else now,
            last_seen_monotonic=previous.last_seen if previous else None,
            previous_status=previous.status if previous else None,
        )
        self._seen[prefix_identity] = _Seen(
            (previous.count if previous else 0) + 1,
            previous.first_seen if previous else now, now, request.request_id,
        )
        while len(self._seen) > self.capacity:
            self._seen.popitem(last=False)
        return fields

    def finished(self, identity: str, request_id: str, status: str, stats: dict[str, Any] | None) -> dict[str, Any]:
        if identity in self._seen and self._seen[identity].request_id == request_id:
            self._seen[identity].status = status
        stats = stats or {}
        keys = (
            "parent_radix_cached_token_count", "parent_computed_token_count",
            "prefix_token_count", "parent_cache_hit_ratio",
            "candidate_prefix_recompute_tokens", "reusable_boundary_token_count",
            "action_slot_wait_ms", "preprocessing_ms", "prefix_prefill_ms",
            "suffix_batch_ms", "suffix_batch_queue_wait_ms",
        )
        return {"status": status, **{k: stats.get(k) for k in keys}}


child_cache_observer = ChildCacheObserver()
