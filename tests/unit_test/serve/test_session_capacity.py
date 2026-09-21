"""Admission is per live owner, including initialization and cleanup."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime.multimodal_session import MultimodalSessionManager


class Socket:
    def __init__(self):
        self.client_state = self.application_state = WebSocketState.CONNECTED
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.code = None
        self.headers = {}

    async def receive(self):
        return await self.incoming.get()

    async def send_text(self, text):
        await self.outgoing.put(json.loads(text))

    async def close(self, code=1000):
        self.code = code
        self.application_state = WebSocketState.DISCONNECTED

    def event(self, payload):
        self.incoming.put_nowait({'type': 'websocket.receive', 'text': json.dumps(payload)})

    async def result(self):
        return await asyncio.wait_for(self.outgoing.get(), 5)


def start(sid, **fields):
    return {'type': 'session.start', 'protocol_version': 1, 'session_id': sid,
            'outputs': ['text'], 'locale': 'zh-CN', **fields}


@pytest_asyncio.fixture
async def runtime(monkeypatch):
    monkeypatch.setenv('SGLANG_OMNI_SESSION_MEMORY_ENABLED', '0')
    client = SimpleNamespace(release_session_cache=AsyncMock())
    manager = MultimodalSessionManager(client=client, model_name='test')
    tasks = []

    def open_socket():
        ws = Socket()
        session = manager.create(ws)
        task = asyncio.create_task(session.run())
        tasks.append(task)
        return ws, session, task

    yield manager, client, open_socket
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await manager.close()


@pytest.mark.asyncio
async def test_eight_starts_only_two_admitted_and_duplicate_keeps_error(runtime):
    manager, client, open_socket = runtime
    sockets = [open_socket() for _ in range(8)]
    for i, (ws, _, _) in enumerate(sockets):
        ws.event(start(f's{i}'))
    results = await asyncio.gather(*(ws.result() for ws, _, _ in sockets))
    assert [r['type'] for r in results[:2]] == ['session.started'] * 2
    for i, result in enumerate(results[2:], 2):
        assert result == {'type': 'error', 'session_id': f's{i}', 'error': {
            'type': 'server_error', 'code': 'session_busy',
            'message': 'Maximum concurrent sessions reached (2). Please retry later.'}}
        await sockets[i][2]
        assert sockets[i][0].code == 1013
    assert len(manager.sessions) == 2
    snapshot = manager.load_snapshot()
    assert snapshot['max_session_count'] == 2
    assert snapshot['available_session_count'] == 0
    client.release_session_cache.assert_not_awaited()
    ws, _, task = open_socket()
    ws.event(start('s0'))
    result = await ws.result()
    assert result['error']['code'] != 'session_busy'
    assert 'already active' in result['error']['message']
    assert not task.done()
    # Release is owner-checked and idempotent.
    manager.release('s0', sockets[4][1])
    assert len(manager.sessions) == 2
    sockets[0][0].event({'type': 'session.close'})
    assert (await sockets[0][0].result())['type'] == 'session.closed'
    await sockets[0][2]
    assert manager.load_snapshot()['available_session_count'] == 1
    manager.release('s0', sockets[0][1])
    ws.event(start('replacement'))
    assert (await ws.result())['type'] == 'session.started'
    assert len(manager.sessions) == 2


@pytest.mark.asyncio
async def test_unstarted_and_invalid_connections_do_not_reserve(runtime):
    manager, _, open_socket = runtime
    sockets = [open_socket() for _ in range(6)]
    assert not manager.sessions
    sockets[0][0].event(start('invalid', outputs=['unknown']))
    assert (await sockets[0][0].result())['type'] == 'error'
    assert not manager.sessions
    for i, (ws, _, _) in enumerate(sockets[:2]):
        ws.event(start(str(i)))
        assert (await ws.result())['type'] == 'session.started'
    assert len(manager.sessions) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_initialization", [False, True])
async def test_initializing_sessions_reserve_before_knowledge_io(runtime, cancel_initialization):
    manager, _, open_socket = runtime
    entered = asyncio.Queue()
    unblock = asyncio.Event()

    async def resolve(**kwargs):
        entered.put_nowait(kwargs['session_id'])
        await unblock.wait()
        raise ValueError('initialization failed')

    controller = SimpleNamespace(resolve_session=AsyncMock(side_effect=resolve))
    sockets = [open_socket() for _ in range(3)]
    for i, (ws, session, _) in enumerate(sockets):
        session.knowledge_controller = controller
        ws.event(start(f'k{i}', knowledge={'binding_id': 'test', 'required': True}))
        if i < 2:
            assert await asyncio.wait_for(entered.get(), 5) == f'k{i}'
    assert len(manager.sessions) == 2
    assert not any(s.started for _, s, _ in sockets)
    assert (await sockets[2][0].result())['error']['code'] == 'session_busy'
    await sockets[2][2]
    assert controller.resolve_session.await_count == 2
    if cancel_initialization:
        for _, _, task in sockets[:2]:
            task.cancel()
        await asyncio.gather(*(task for _, _, task in sockets[:2]), return_exceptions=True)
    else:
        unblock.set()
        for ws, _, task in sockets[:2]:
            assert (await ws.result())['type'] == 'error'
            await task
    assert all(ws.code == 1000 for ws, _, _ in sockets[:2])
    assert not manager.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_kind', ['disconnect', 'cancel'])
@pytest.mark.parametrize('failure', ['none', 'cancel_turn', 'memory', 'tts', 'cache'])
async def test_cleanup_failures_release_owner_and_attempt_remaining_layers(runtime, exit_kind, failure):
    manager, client, open_socket = runtime
    ws, session, task = open_socket()
    ws.event(start('cleanup'))
    assert (await ws.result())['type'] == 'session.started'
    cancel = AsyncMock(side_effect=RuntimeError('cleanup') if failure == 'cancel_turn' else None)
    memory = AsyncMock(side_effect=RuntimeError('cleanup') if failure == 'memory' else None)
    tts = AsyncMock(side_effect=RuntimeError('cleanup') if failure == 'tts' else None)
    session._cancel_active_turn = cancel
    session._shutdown_session_memory = memory
    session.embedded_tts = SimpleNamespace(close=tts)
    if failure == 'cache':
        client.release_session_cache.side_effect = RuntimeError('cleanup')
    if exit_kind == 'cancel':
        task.cancel()
    else:
        ws.incoming.put_nowait({'type': 'websocket.disconnect'})
    await asyncio.gather(task, return_exceptions=True)
    assert not manager.sessions
    cancel.assert_awaited_once()
    memory.assert_awaited_once()
    tts.assert_awaited_once()
    client.release_session_cache.assert_awaited_once_with(session.session_instance_id)
    assert ws.code == 1000


@pytest.mark.asyncio
async def test_cleaning_owner_holds_slot_until_cleanup_finishes(runtime):
    manager, client, open_socket = runtime
    sockets = [open_socket() for _ in range(2)]
    for i, (ws, _, _) in enumerate(sockets):
        ws.event(start(str(i)))
        assert (await ws.result())['type'] == 'session.started'
    entered, unblock = asyncio.Event(), asyncio.Event()

    async def release_cache(owner):
        entered.set()
        await unblock.wait()

    client.release_session_cache.side_effect = release_cache
    sockets[0][0].incoming.put_nowait({'type': 'websocket.disconnect'})
    await asyncio.wait_for(entered.wait(), 5)
    ws, _, task = open_socket()
    ws.event(start('during-cleanup'))
    assert (await ws.result())['error']['code'] == 'session_busy'
    await task
    assert len(manager.sessions) == 2
    unblock.set()
    await sockets[0][2]
    assert len(manager.sessions) == 1
