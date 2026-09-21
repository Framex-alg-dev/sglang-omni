from sglang_omni.serve.realtime.action.routing import IntentShortcutBackoff


def test_shortcut_failure_expires_and_is_session_local():
    first, second = IntentShortcutBackoff(), IntentShortcutBackoff()
    key = ("260", "数字三手势")
    first.reject(key, 10)
    assert first.blocked(key, 39.9)
    assert not first.blocked(("260", "另一个任务"), 11)
    assert not second.blocked(key, 11)
    assert not first.blocked(key, 40)


def test_shortcut_backoff_is_bounded():
    cache = IntentShortcutBackoff(capacity=2)
    for index in range(3):
        cache.reject((str(index), "task"), 0)
    assert len(cache.failures) == 2
    assert not cache.blocked(("0", "task"), 1)
