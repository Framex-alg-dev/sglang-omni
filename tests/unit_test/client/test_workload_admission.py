import asyncio
from unittest.mock import patch

from sglang_omni.client.client import _PriorityAdmissionGate
from sglang_omni.utils.turn_media_scope import turn_media_scope


def test_committed_reply_and_tool_feedback_precede_queued_speculative_reply():
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateChunk, GenerateRequest

    async def run():
        client = Client.__new__(Client)
        gate = _PriorityAdmissionGate(1)
        client._action_scoring_gate = gate
        order = []

        async def generate(request, request_id=None):
            order.append(request_id)
            yield GenerateChunk(request_id=request_id, modality="text", text="ok")

        client._generate_unadmitted = generate

        async def consume(role):
            request = GenerateRequest(metadata={"runtime_workload_class": role})
            async for _ in client.generate(request, request_id=role):
                pass

        await gate.acquire(0, "busy")
        tasks = []
        for role in ("speculative_reply", "committed_reply", "tool_transition"):
            tasks.append(asyncio.create_task(consume(role)))
            await asyncio.sleep(0)
        await gate.release()
        await asyncio.gather(*tasks)
        assert order == ["tool_transition", "committed_reply", "speculative_reply"]
        assert gate.snapshot()["active_total"] == 0
        assert not gate._waiters

    asyncio.run(run())


def test_body_precedes_queued_performance_and_cancelled_queue_is_removed():
    async def run():
        gate = _PriorityAdmissionGate(1)
        await gate.acquire(1, 'running-p')
        order = []
        async def work(priority, name):
            async with gate.slot(priority, name):
                order.append(name)
        stale = asyncio.create_task(work(1, 'expired-p'))
        performance = asyncio.create_task(work(1, 'next-p'))
        body = asyncio.create_task(work(-100, 'body-other-session'))
        await asyncio.sleep(0)
        stale.cancel()
        await asyncio.gather(stale, return_exceptions=True)
        await gate.release()
        await asyncio.gather(performance, body)
        assert order == ['body-other-session', 'next-p']
        assert gate._active == 0 and not gate._waiters
    asyncio.run(run())


def test_waiting_route_cannot_age_past_a_new_body_request():
    async def run():
        clock = [1000.0]
        with patch('sglang_omni.client.client.time.monotonic', side_effect=lambda: clock[0]):
            gate = _PriorityAdmissionGate(1)
            await gate.acquire(1, 'running-p')
            order = []
            async def work(priority, name):
                async with gate.slot(priority, name):
                    order.append(name)
            route = asyncio.create_task(work(-50, 'old-route'))
            await asyncio.sleep(0)
            clock[0] += 100
            body = asyncio.create_task(work(-100, 'body'))
            await asyncio.sleep(0)
            await gate.release()
            await asyncio.gather(route, body)
            assert order == ['body', 'old-route']
            assert gate.snapshot()['active_total'] == 0
    asyncio.run(run())


def test_encoder_domain_is_opt_in_and_separates_identity_turn_revision():
    with patch.dict('os.environ', {'SGLANG_OMNI_SHARED_MEDIA_CACHE_DOMAIN': ''}):
        assert turn_media_scope('s', 't', 1, 0) is None
    with patch.dict('os.environ', {'SGLANG_OMNI_SHARED_MEDIA_CACHE_DOMAIN': 'release-sha:preprocess-v1'}):
        baseline = turn_media_scope('s', 't', 1, 0)
        assert baseline == turn_media_scope('s', 't', 1, 0)
        assert all(baseline != turn_media_scope(*args) for args in [
            ('other', 't', 1, 0), ('s', 'other', 1, 0), ('s', 't', 2, 0), ('s', 't', 1, 1)])
        assert turn_media_scope('s', 't', None, 0) is None
