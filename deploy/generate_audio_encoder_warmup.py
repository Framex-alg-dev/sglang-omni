#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the deterministic Qwen3-Omni audio-encoder warmup asset."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
DURATION_S = 24
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "sglang_omni"
    / "assets"
    / "audio_encoder_warmup.wav"
)


def _fade(segment: np.ndarray, duration_s: float = 0.03) -> np.ndarray:
    fade_samples = min(int(SAMPLE_RATE * duration_s), len(segment) // 2)
    if fade_samples <= 0:
        return segment
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float64)
    segment[:fade_samples] *= ramp
    segment[-fade_samples:] *= ramp[::-1]
    return segment


def _chirp(t: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    duration = max(float(t[-1] + 1.0 / SAMPLE_RATE), 1.0 / SAMPLE_RATE)
    slope = (end_hz - start_hz) / duration
    phase = 2.0 * np.pi * (start_hz * t + 0.5 * slope * t * t)
    return np.sin(phase)


def build_waveform() -> np.ndarray:
    rng = np.random.default_rng(20260822)
    waveform = np.zeros(SAMPLE_RATE * DURATION_S, dtype=np.float64)

    def put(start_s: int, end_s: int, values: np.ndarray) -> None:
        start = start_s * SAMPLE_RATE
        end = end_s * SAMPLE_RATE
        waveform[start:end] = _fade(values[: end - start].copy())

    # Near-silence exercises low-energy log-mel values without being exactly zero.
    put(0, 2, rng.normal(0.0, 0.001, 2 * SAMPLE_RATE))

    t = np.arange(4 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    put(2, 6, 0.24 * _chirp(t, 55.0, 900.0))

    # Speech-like fundamentals, harmonics and formant bands with amplitude motion.
    phase = 2.0 * np.pi * (135.0 * t + 18.0 * np.sin(2.0 * np.pi * 0.7 * t))
    envelope = 0.45 + 0.35 * np.sin(2.0 * np.pi * 2.3 * t) ** 2
    voiced = envelope * (
        0.20 * np.sin(phase)
        + 0.10 * np.sin(2.0 * phase)
        + 0.07 * np.sin(2.0 * np.pi * 720.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 1_450.0 * t)
        + 0.035 * np.sin(2.0 * np.pi * 2_800.0 * t)
    )
    put(6, 10, voiced)

    # Wide-band sweep reaches most useful bins below the 8 kHz Nyquist limit.
    put(10, 14, 0.18 * _chirp(t, 180.0, 7_200.0))

    # Mixed white/colored noise broadens spectral and dynamic-range coverage.
    white = rng.normal(0.0, 1.0, 4 * SAMPLE_RATE)
    colored = np.convolve(white, np.ones(17) / 17.0, mode="same")
    noise_envelope = np.linspace(0.02, 0.25, len(t))
    put(14, 18, noise_envelope * (0.55 * white + 0.75 * colored))

    # Transients and gated tone bursts exercise sharp onsets and short activity.
    transients = np.zeros(3 * SAMPLE_RATE, dtype=np.float64)
    for index, onset in enumerate(np.arange(0.08, 2.95, 0.16)):
        start = int(onset * SAMPLE_RATE)
        length = min(int(0.045 * SAMPLE_RATE), len(transients) - start)
        local_t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
        decay = np.exp(-local_t * (38.0 + index % 5))
        transients[start : start + length] += (
            0.34 * decay * np.sin(2.0 * np.pi * (220.0 + index * 95.0) * local_t)
        )
    put(18, 21, transients)

    t = np.arange(3 * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    mixed = (
        0.12 * _chirp(t, 90.0, 5_800.0)
        + 0.10 * np.sin(2.0 * np.pi * 210.0 * t)
        + rng.normal(0.0, 0.025, len(t))
    )
    mixed *= 0.25 + 0.75 * np.sin(2.0 * np.pi * 1.6 * t) ** 2
    put(21, 24, mixed)

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform *= 0.72 / peak
    return np.clip(waveform, -1.0, 1.0)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.rint(build_waveform() * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_wav(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
