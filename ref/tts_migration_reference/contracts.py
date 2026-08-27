"""Provider-neutral interface copied from the current backend TTS path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Protocol


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    turn_id: str
    text: str
    voice: str
    language: str = "Auto"
    input_chunking_mode: str = "direct"
    input_chunk_max_chars: int | None = None
    input_chunk_interval_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("speech synthesis turn_id is required")
        if self.text and not self.text.strip():
            raise ValueError("speech synthesis text is required")
        if not self.voice.strip():
            raise ValueError("speech synthesis voice is required")
        if self.input_chunking_mode not in {"direct", "buffered"}:
            raise ValueError("speech synthesis input_chunking_mode is invalid")
        if self.input_chunking_mode == "buffered" and (
            self.input_chunk_max_chars is None or self.input_chunk_max_chars < 1
        ):
            raise ValueError("buffered synthesis requires a positive chunk size")
        if self.input_chunk_interval_ms < 0:
            raise ValueError("speech synthesis chunk interval must not be negative")


@dataclass(frozen=True)
class SpeechSynthesisResult:
    audio_bytes: int
    chunk_count: int
    provider_response_id: str | None = None


SpeechAudioSink = Callable[[bytes], Awaitable[None]]


class SpeechSynthesizer(Protocol):
    async def synthesize(
        self, request: SpeechSynthesisRequest, *, audio_sink: SpeechAudioSink
    ) -> SpeechSynthesisResult:
        """Stream 24 kHz mono PCM16 for one complete text request."""

    async def synthesize_streaming(
        self,
        request: SpeechSynthesisRequest,
        *,
        text_chunks: AsyncIterator[str],
        audio_sink: SpeechAudioSink,
    ) -> SpeechSynthesisResult:
        """Consume ordered text deltas and stream 24 kHz mono PCM16."""
