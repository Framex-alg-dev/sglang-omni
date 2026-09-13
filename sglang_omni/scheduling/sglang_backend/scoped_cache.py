"""Radix scopes without model-visible tokens or duplicate KV ownership.

Only request builders may construct scopes. Unknown modes retain upstream
isolation. All allocation, reference locks and node eviction remain upstream.
"""

from dataclasses import dataclass, replace
import json
import os
import time

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

MARKER = "omni-scopes-v1:"


@dataclass(frozen=True)
class PrefixScopes:
    family: str
    public_end: int
    session_end: int
    session: str
    request: str

    def encode(self):
        return MARKER + json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))

    def scope(self, position):
        if position < self.public_end:
            return (self.family, "public")
        if position < self.session_end:
            return (self.family, "session", self.session)
        return (self.family, "request", self.session, self.request)


class ScopedRadixKey(RadixKey):
    __slots__ = ("scopes", "offset")

    def __init__(self, tokens, scopes, offset=0, limit=None):
        super().__init__(tokens, extra_key=MARKER + scopes.family, limit=limit)
        self.scopes, self.offset = scopes, offset

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            index = slice(index, index + 1)
        start, stop, step = index.indices(len(self))
        if step != 1:
            raise ValueError("scope slices must have unit stride")
        return ScopedRadixKey(self.token_ids[start:stop], self.scopes, self.offset + start)

    def child_key(self, page_size=1):
        return (self.scopes.scope(self.offset), super().child_key(page_size))

    def match(self, other, page_size=1):
        if not isinstance(other, ScopedRadixKey) or self.extra_key != other.extra_key:
            return 0
        token_count = super().match(other, page_size=1)
        position = 0
        while position < token_count:
            if self.scopes.scope(self.offset + position) != other.scopes.scope(other.offset + position):
                break
            boundaries = [token_count]
            for key in (self, other):
                boundaries.extend(end - key.offset for end in (key.scopes.public_end, key.scopes.session_end)
                                  if end - key.offset > position)
            position = min(boundaries)
        return position // page_size * page_size


class ScopedRadixCache(RadixCache):
    def __init__(self, params):
        self.session_budget_tokens = int(os.environ.get("SGLANG_OMNI_SESSION_KV_BUDGET_TOKENS", "0"))
        if self.session_budget_tokens < 0:
            raise ValueError("session KV budget must be nonnegative")
        super().__init__(params)

    @staticmethod
    def _private_owner(key):
        if isinstance(key, ScopedRadixKey):
            return key.scopes.session if key.offset >= key.scopes.public_end else None
        extra = getattr(key, "extra_key", None)
        return extra.rsplit(":owner:", 1)[1] if isinstance(extra, str) and ":owner:" in extra else None

    def release_session(self, session):
        if not isinstance(session, str) or not session:
            raise ValueError("session instance is required")
        if not hasattr(self, "_closing_sessions"):
            self._closing_sessions = {}
        self._closing_sessions[session] = time.monotonic() + 300
        return self.reap_closed_sessions(force=True)

    def reap_closed_sessions(self, force=False):
        closing = getattr(self, "_closing_sessions", {})
        now = time.monotonic()
        if (not closing and not self.session_budget_tokens) or (not force and now < getattr(self, "_next_reap", 0)):
            return 0
        self._next_reap = now + 1
        # Split public/private boundaries before pruning leaves. Never free a
        # referenced node; late request terminalization is handled next pass.
        stack = list(self.root_node.children.values())
        usage = {}
        while stack:
            node = stack.pop()
            key = node.key
            if isinstance(key, ScopedRadixKey) and (key.scopes.session in closing or self.session_budget_tokens):
                cut = key.scopes.public_end - key.offset
                if 0 < cut < len(key):
                    self._split_node(key, node, cut)
            owner = self._private_owner(node.key)
            if owner is not None:
                usage[owner] = usage.get(owner, 0) + len(node.value)
            stack.extend(node.children.values())
        freed = 0
        pending = sorted(self.evictable_leaves, key=lambda node: node.last_access_time, reverse=True)
        while pending:
            node = pending.pop()
            key = node.key
            if node is self.root_node or node.lock_ref or node.children:
                continue
            owner = self._private_owner(key)
            over_budget = self.session_budget_tokens and usage.get(owner, 0) > self.session_budget_tokens
            if owner is None or (owner not in closing and not over_budget):
                continue
            self.token_to_kv_pool_allocator.free(node.value)
            freed += len(node.value)
            usage[owner] -= len(node.value)
            parent = node.parent
            self._delete_leaf(node)
            self._record_remove_event(node)
            pending.append(parent)
        self._closing_sessions = {key: expiry for key, expiry in closing.items() if expiry > now}
        return freed

    def _scope_key(self, key):
        if isinstance(key, ScopedRadixKey):
            return key
        if self.is_eagle or key.is_bigram or not isinstance(key.extra_key, str) or not key.extra_key.startswith(MARKER):
            return key
        try:
            scopes = PrefixScopes(**json.loads(key.extra_key[len(MARKER):]))
            if not (isinstance(scopes.public_end, int) and isinstance(scopes.session_end, int)
                    and 0 <= scopes.public_end <= scopes.session_end
                    and all(isinstance(value, str) and value for value in (scopes.family, scopes.session, scopes.request))):
                return key
            scopes = replace(scopes, public_end=scopes.public_end // self.page_size * self.page_size,
                             session_end=scopes.session_end // self.page_size * self.page_size)
            return ScopedRadixKey(key.token_ids, scopes, limit=key.limit)
        except (TypeError, ValueError):
            return key

    def match_prefix(self, params):
        return super().match_prefix(replace(params, key=self._scope_key(params.key)))

    def insert(self, params):
        return super().insert(replace(params, key=self._scope_key(params.key)))
