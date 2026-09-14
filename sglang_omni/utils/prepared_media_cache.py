"""Session-scoped, one-shot prepared media with bounded retention."""

from collections import OrderedDict
import threading
import time


def media_size(value):
    if isinstance(value, dict):
        return sum(media_size(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(media_size(item) for item in value)
    if isinstance(value, (bytes, bytearray, str)):
        return len(value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if hasattr(value, "nbytes"):
        return value.nbytes
    if hasattr(value, "size") and hasattr(value, "getbands"):
        return value.size[0] * value.size[1] * len(value.getbands())
    return 0


class PreparedMediaCache:
    def __init__(self, *, max_entries=32, max_bytes=256 << 20,
                 per_session_bytes=32 << 20, ttl_seconds=30, clock=time.monotonic):
        if min(max_entries, max_bytes, per_session_bytes, ttl_seconds) <= 0:
            raise ValueError("media cache budgets must be positive")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.per_session_bytes, self.ttl_seconds = per_session_bytes, ttl_seconds
        self._clock = clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self.current_bytes = 0
        self._closed = {}

    def _remove(self, key):
        row = self._entries.pop(key, None)
        if row:
            self.current_bytes -= row[1]
        return row

    def _expire(self):
        now = self._clock()
        self._closed = {owner: expiry for owner, expiry in self._closed.items() if expiry > now}
        for key, row in list(self._entries.items()):
            if row[2] <= now:
                self._remove(key)

    def pop(self, key):
        with self._lock:
            self._expire()
            row = self._remove(key)
            return row[0] if row else None

    def put(self, key, value):
        size = media_size(value)
        with self._lock:
            self._expire()
            self._remove(key)
            if key[0] in self._closed:
                return
            if size > min(self.max_bytes, self.per_session_bytes):
                return
            self._entries[key] = (value, size, self._clock() + self.ttl_seconds)
            self.current_bytes += size
            owner = key[0]
            while sum(row[1] for item, row in self._entries.items() if item[0] == owner) > self.per_session_bytes:
                self._remove(next(item for item in self._entries if item[0] == owner))
            while self.current_bytes > self.max_bytes or len(self._entries) > self.max_entries:
                self._remove(next(iter(self._entries)))

    def release_session(self, owner):
        with self._lock:
            self._expire()
            self._closed[owner] = self._clock() + max(self.ttl_seconds, 300)
            for key in list(self._entries):
                if key[0] == owner:
                    self._remove(key)
