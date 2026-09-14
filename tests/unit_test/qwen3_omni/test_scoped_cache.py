from array import array
import pytest
import torch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams, EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_omni.scheduling.sglang_backend.scoped_cache import PrefixScopes, ScopedRadixCache


@pytest.mark.parametrize('page', [1, 2, 4])
def test_public_session_request_boundaries_and_single_ownership(page):
    class Allocator:
        device = 'cpu'
        freed = []
        def free(self, values):
            self.freed.extend(values.tolist())
    allocator = Allocator()
    cache = ScopedRadixCache(CacheInitParams(False, None, allocator, page))
    tokens = array('q', range(1, 17))
    def key(session, request, public=5):
        return RadixKey(tokens, PrefixScopes('model', public, 10, session, request).encode())
    cache.insert(InsertParams(key=key('a','1'), value=torch.arange(16)))
    def match(session, request, public=5):
        return cache.match_prefix(MatchPrefixParams(key('a' if session is None else session, request, public)))
    assert len(match('b','1').device_indices) == 5 // page * page
    assert len(match('a','2').device_indices) == 10 // page * page
    assert len(match('a','1').device_indices) == 16
    # Insert values allocated for the unmatched suffix only; prefix addresses
    # come from the existing tree, never a second owner in another tree.
    common = 5 // page * page
    cache.insert(InsertParams(key=key('b','1'), value=torch.cat([torch.arange(common), torch.arange(100+common,116)])))
    b = match('b','1')
    assert b.device_indices[:common].tolist() == list(range(common))
    cache.inc_lock_ref(b.last_device_node)
    cache.evict(EvictParams(num_tokens=1000))
    assert not set(b.device_indices.tolist()) & set(allocator.freed)
    cache.dec_lock_ref(b.last_device_node)
    cache.evict(EvictParams(num_tokens=1000))
    assert len(allocator.freed) == len(set(allocator.freed)) == 32-common


def test_different_public_boundaries_do_not_share_private_tokens():
    cache = ScopedRadixCache(CacheInitParams(False, None, None, 1))
    tokens = array('q', range(10))
    a = RadixKey(tokens, PrefixScopes('model', 3, 8, 'a', '1').encode())
    b = RadixKey(tokens, PrefixScopes('model', 5, 8, 'b', '1').encode())
    cache.insert(InsertParams(key=a))
    assert len(cache.match_prefix(MatchPrefixParams(b)).device_indices) == 3


def test_close_frees_private_leaves_after_unlock_but_retains_public():
    class Allocator:
        device = 'cpu'
        freed = []
        def free(self, value):
            self.freed.extend(value.tolist())
    allocator = Allocator()
    cache = ScopedRadixCache(CacheInitParams(False, None, allocator, 1))
    def key(owner):
        return RadixKey(array('q', range(10)), PrefixScopes('model', 3, 8, owner, 'turn').encode())
    cache.insert(InsertParams(key=key('a'), value=torch.arange(10)))
    locked = cache.match_prefix(MatchPrefixParams(key('a')))
    cache.inc_lock_ref(locked.last_device_node)
    assert cache.release_session('a') == 0
    assert len(cache.match_prefix(MatchPrefixParams(key('b'))).device_indices) == 3
    cache.dec_lock_ref(locked.last_device_node)
    assert cache.reap_closed_sessions(force=True) == 7
    assert allocator.freed == list(range(3,10))
    assert len(cache.match_prefix(MatchPrefixParams(key('b'))).device_indices) == 3
    assert cache.reap_closed_sessions(force=True) == 0


def test_private_budget_never_evicts_public_or_locked_values(monkeypatch):
    monkeypatch.setenv('SGLANG_OMNI_SESSION_KV_BUDGET_TOKENS', '4')
    class Allocator:
        device = 'cpu'
        freed = []
        def free(self, value):
            self.freed.extend(value.tolist())
    allocator = Allocator()
    cache = ScopedRadixCache(CacheInitParams(False, None, allocator, 1))
    key = RadixKey(array('q', range(10)), PrefixScopes('model',3,8,'a','1').encode())
    cache.insert(InsertParams(key=key, value=torch.arange(10)))
    locked=cache.match_prefix(MatchPrefixParams(key))
    cache.inc_lock_ref(locked.last_device_node)
    assert cache.reap_closed_sessions(force=True) == 0
    cache.dec_lock_ref(locked.last_device_node)
    assert cache.reap_closed_sessions(force=True) == 7
    assert allocator.freed == list(range(3,10))
    assert len(cache.match_prefix(MatchPrefixParams(key)).device_indices) == 3


def test_multimodal_scope_shares_only_verified_ordinary_position_prefix():
    from sglang_omni.models.qwen3_omni.request_builders import _verified_scope_boundaries
    metadata = {'public_prefix_token_count': 3, 'scope_has_media': True}
    positions = torch.arange(12).repeat(3, 1)
    positions[:, 3:] += 100
    assert _verified_scope_boundaries(metadata, positions, 8) == (3, 3)
    positions[1, 2] = 90
    assert _verified_scope_boundaries(metadata, positions, 8) is None
    assert _verified_scope_boundaries(metadata, None, 8) is None
    assert _verified_scope_boundaries(metadata, torch.zeros(3, 2), 8) is None
    metadata['scope_has_media'] = False
    assert _verified_scope_boundaries(metadata, torch.arange(12).repeat(3, 1), 8) == (3, 8)
