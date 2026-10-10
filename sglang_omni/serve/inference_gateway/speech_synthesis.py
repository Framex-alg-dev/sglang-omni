"""Whole-text speech synthesis for the inference-session gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlsplit

import httpx


AudioSink = Callable[[bytes], Awaitable[None]]


class GatewaySpeechError(RuntimeError):
    def __init__(self, message: str, *, phase: str, retryable: bool) -> None:
        super().__init__(message)
        self.phase = phase
        self.retryable = retryable


@dataclass(frozen=True)
class GatewaySpeechConfig:
    endpoint: str
    timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    frame_bytes: int = 12_000
    max_audio_bytes: int = 32 * 1024 * 1024
    seed: int = 0

    def __post_init__(self) -> None:
        endpoint = urlsplit(self.endpoint)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("gateway speech endpoint must be absolute HTTP(S)")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("gateway speech timeouts must be positive")
        if self.frame_bytes <= 0 or self.frame_bytes % 2:
            raise ValueError("gateway speech frame_bytes must be a positive PCM16 size")
        if self.max_audio_bytes <= 0:
            raise ValueError("gateway speech max_audio_bytes must be positive")
        if self.seed < 0:
            raise ValueError("gateway speech seed must be non-negative")


@dataclass(frozen=True)
class GatewaySpeechResult:
    audio_bytes: int
    chunk_count: int
    provider_response_id: str | None


def validate_first_chunk_ms(value: object) -> int | None:
    if value is not None and (type(value) is not int or not 40 <= value <= 1000):
        raise ValueError("first_chunk_ms must be an integer between 40 and 1000")
    return value


class _PcmFramer:
    def __init__(self, frame_bytes: int, *, first_frame_bytes: int | None = None) -> None:
        self._frame_bytes = frame_bytes
        self._next_frame_bytes = first_frame_bytes or frame_bytes
        self._pending = bytearray()

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self._pending.extend(chunk)
        frames: list[bytes] = []
        while len(self._pending) >= self._next_frame_bytes:
            frames.append(bytes(self._pending[: self._next_frame_bytes]))
            del self._pending[: self._next_frame_bytes]
            self._next_frame_bytes = self._frame_bytes
        return tuple(frames)

    def finish(self) -> bytes | None:
        if not self._pending:
            return None
        tail = bytes(self._pending)
        self._pending.clear()
        return tail


class GatewaySpeechSynthesizer:
    """Render one final reply with exactly one upstream synthesis request."""

    def __init__(
        self,
        config: GatewaySpeechConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.timeout_seconds,
                connect=config.connect_timeout_seconds,
            ),
            trust_env=False,
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def synthesize(
        self,
        *,
        text: str,
        voice: str,
        instruction: str,
        audio_sink: AudioSink,
        first_chunk_ms: int | None = None,
    ) -> GatewaySpeechResult:
        if not text.strip() or not voice.strip():
            raise ValueError("gateway speech text and voice are required")
        first_chunk_ms = validate_first_chunk_ms(first_chunk_ms)
        payload: dict[str, object] = {
            "text": text,
            "speaker_id": voice,
            "seed": self._config.seed,
        }
        if instruction.strip():
            payload["instruct"] = instruction.strip()
        if first_chunk_ms is not None:
            payload["first_chunk_ms"] = first_chunk_ms
        # 24 kHz mono PCM16: 48 bytes/ms. Only the first frame changes.
        framer = _PcmFramer(
            self._config.frame_bytes,
            first_frame_bytes=(48 * first_chunk_ms if first_chunk_ms is not None else None),
        )
        received_bytes = 0
        emitted_bytes = 0
        chunk_count = 0
        provider_response_id: str | None = None

        async def emit(frame: bytes) -> None:
            nonlocal emitted_bytes, chunk_count
            await audio_sink(frame)
            emitted_bytes += len(frame)
            chunk_count += 1

        try:
            async with self._client.stream(
                "POST",
                self._config.endpoint,
                json=payload,
            ) as response:
                provider_response_id = response.headers.get("x-request-id") or None
                if response.status_code != 200:
                    raise GatewaySpeechError(
                        "speech provider rejected the request",
                        phase="provider",
                        retryable=response.status_code >= 500,
                    )
                if (
                    first_chunk_ms is not None
                    and response.headers.get("x-tts-first-chunk-ms") != str(first_chunk_ms)
                ):
                    raise GatewaySpeechError(
                        "speech provider did not acknowledge first_chunk_ms",
                        phase="protocol",
                        retryable=False,
                    )
                sample_rate = response.headers.get("x-audio-sample-rate")
                if sample_rate not in (None, "24000"):
                    raise GatewaySpeechError(
                        "speech provider returned an unsupported sample rate",
                        phase="protocol",
                        retryable=False,
                    )
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    received_bytes += len(chunk)
                    if received_bytes > self._config.max_audio_bytes:
                        raise GatewaySpeechError(
                            "speech provider exceeded the audio budget",
                            phase="protocol",
                            retryable=False,
                        )
                    for frame in framer.feed(chunk):
                        await emit(frame)
                tail = framer.finish()
                if tail is not None:
                    await emit(tail)
        except asyncio.CancelledError:
            raise
        except GatewaySpeechError:
            raise
        except httpx.TimeoutException as exc:
            raise GatewaySpeechError(
                "speech provider timed out",
                phase="timeout",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise GatewaySpeechError(
                "speech provider transport failed",
                phase="transport",
                retryable=True,
            ) from exc

        if received_bytes == 0 or emitted_bytes != received_bytes:
            raise GatewaySpeechError(
                "speech provider completed without valid PCM",
                phase="protocol",
                retryable=False,
            )
        return GatewaySpeechResult(
            audio_bytes=emitted_bytes,
            chunk_count=chunk_count,
            provider_response_id=provider_response_id,
        )
