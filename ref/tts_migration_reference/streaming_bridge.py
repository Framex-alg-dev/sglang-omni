"""Bridge incremental multimodal text output into streaming TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from .contracts import SpeechAudioSink, SpeechSynthesisRequest, SpeechSynthesisResult, SpeechSynthesizer


class StreamingSpeechMode(str, Enum):
    REUSED = "reused"
    TEXT_CHANGED = "text_changed"
    PROVIDER_RETRY = "provider_retry"


@dataclass(frozen=True)
class StreamingSpeechResult:
    speech: SpeechSynthesisResult | None
    mode: StreamingSpeechMode
    failure: BaseException | None = None


class StreamingSpeech:
    """Start TTS on deltas, but withhold PCM until final text is accepted."""

    def __init__(self, *, request: SpeechSynthesisRequest,
                 synthesizer: SpeechSynthesizer) -> None:
        self.request = request
        self._synthesizer = synthesizer
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._text_parts: list[str] = []
        self._input_finished = self._claimed = False
        self._task = asyncio.create_task(self._run(), name=f"streaming-speech:{request.turn_id}")

    async def push_text(self, delta: str) -> None:
        if self._input_finished:
            raise RuntimeError("streaming speech text input is already finished")
        if delta:
            self._text_parts.append(delta)
            await self._text_queue.put(delta)

    async def finish_input(self) -> None:
        if not self._input_finished:
            self._input_finished = True
            await self._text_queue.put(None)

    async def deliver(self, final_request: SpeechSynthesisRequest, *,
                      audio_sink: SpeechAudioSink) -> StreamingSpeechResult:
        if self._claimed:
            raise RuntimeError("streaming speech can only be delivered once")
        self._claimed = True
        await self.finish_input()
        if not self._matches(final_request):
            await self.cancel()
            return StreamingSpeechResult(None, StreamingSpeechMode.TEXT_CHANGED)
        emitted_chunks = 0
        while True:
            pcm = await self._audio_queue.get()
            if pcm is None:
                break
            await audio_sink(pcm)
            emitted_chunks += 1
        try:
            result = await self._task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if emitted_chunks:
                raise
            return StreamingSpeechResult(None, StreamingSpeechMode.PROVIDER_RETRY, exc)
        return StreamingSpeechResult(result, StreamingSpeechMode.REUSED)

    async def cancel(self) -> None:
        if not self._task.done():
            self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> SpeechSynthesisResult:
        async def direct_chunks():
            while True:
                chunk = await self._text_queue.get()
                if chunk is None:
                    return
                yield chunk

        async def buffered_chunks():
            max_chars = self.request.input_chunk_max_chars
            if max_chars is None:
                raise RuntimeError("buffered streaming speech has no chunk size")
            interval_seconds = self.request.input_chunk_interval_ms / 1000.0
            pending, input_finished, emitted = "", False, False
            while True:
                if not pending and not input_finished:
                    delta = await self._text_queue.get()
                    if delta is None:
                        input_finished = True
                    else:
                        pending += delta
                if not pending:
                    if input_finished:
                        return
                    continue
                if emitted and interval_seconds > 0:
                    await asyncio.sleep(interval_seconds)
                chunk, pending = pending[:max_chars], pending[max_chars:]
                emitted = True
                yield chunk

        text_chunks = buffered_chunks() if self.request.input_chunking_mode == "buffered" else direct_chunks()

        async def buffer_audio(pcm: bytes) -> None:
            if pcm:
                await self._audio_queue.put(bytes(pcm))

        try:
            return await self._synthesizer.synthesize_streaming(
                self.request, text_chunks=text_chunks, audio_sink=buffer_audio)
        finally:
            await self._audio_queue.put(None)

    def _matches(self, final_request: SpeechSynthesisRequest) -> bool:
        return ("".join(self._text_parts).strip() == final_request.text.strip()
                and self.request.voice == final_request.voice
                and self.request.language == final_request.language)
