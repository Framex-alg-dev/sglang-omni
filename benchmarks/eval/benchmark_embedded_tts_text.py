"""Replay fixed text/delta timing through the real embedded TTS adapter.

Does not restart omni or modify the gateway. The measured start is synthesis
invocation, NOT server turn.commit. Model inference is deliberately excluded.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
import wave
from pathlib import Path

from sglang_omni.serve.realtime.embedded_tts import EmbeddedTTSConfig, EmbeddedTTSConnection
from sglang_omni.utils.structured_logs import shutdown_structured_log_writer

CASES = [
    ("english", ["  What's", " new?\r", "\n", "  We", "'ll explain", " it clearly.  "]),
    ("chinese", ["  你好，", "\n\n", " 今天一起", "了解人工智能。  "]),
    ("short", [" ", "好的", "。\r\n"]),
]


async def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SGLANG_OMNI_REALTIME_LOG_DIR", str(out / "logs"))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    results = []
    modes = args.modes.split(",")
    for concurrency in args.concurrency:
        connections = {
            mode: [EmbeddedTTSConnection(EmbeddedTTSConfig(
                url=args.url, voice=args.voice,
                normalize_text_whitespace=mode != "raw",
                text_buffer_enabled=mode == "buffered",
                text_coalesce=mode == "batched",
                text_append_target_chars=512 if mode == "batched" else 0,
                log_text_payloads=args.log_text,
            ), session_id=f"tts-text-{out.name}-{concurrency}-{mode}-{side}")
            for side in range(concurrency)] for mode in modes
        }
        try:
            for rep in range(args.repeats):
                case_id, chunks = CASES[rep % len(CASES)]
                # Rotate mode order to reduce warmup/time-of-run bias.
                for mode in modes[rep % len(modes):] + modes[:rep % len(modes)]:
                    async def one(side):
                        started = time.monotonic()
                        pcm = bytearray()
                        packets = []
                        tid = f"{concurrency}-{mode}-{rep}-{side}"
                        async def source():
                            for chunk in chunks:
                                await asyncio.sleep(args.delta_ms / 1000)
                                yield chunk
                        async def sink(data):
                            if data:
                                packets.append({"ms": (time.monotonic()-started)*1000, "bytes": len(data)})
                                pcm.extend(data)
                        row = dict(turn_id=tid, concurrency=concurrency, mode=mode, rep=rep,
                                   side=side, case_id=case_id, input_chunks=chunks,
                                   start_unix_ns=time.time_ns())
                        try:
                            await connections[mode][side].synthesize_streaming(
                                turn_id=tid, text_chunks=source(), audio_sink=sink,
                                instruct="Read clearly at a natural conversational pace.",
                            )
                            row["completed"] = True
                        except Exception as exc:
                            row.update(completed=False, error_type=type(exc).__name__, error=str(exc))
                        row.update(total_ms=(time.monotonic()-started)*1000,
                                   first_audio_ms=packets[0]["ms"] if packets else None,
                                   packets=packets, audio_seconds=len(pcm)/48000,
                                   audio_sha256=hashlib.sha256(pcm).hexdigest())
                        # Supply deficit estimate if playback began on first packet;
                        # not a measurement of an actual browser's playback buffer.
                        play_end = packets[0]["ms"] if packets else 0
                        deficit = 0.0
                        for packet in packets:
                            deficit += max(0, packet["ms"]-play_end)
                            play_end = max(play_end, packet["ms"]) + packet["bytes"]/48
                        row["supply_deficit_ms"] = deficit
                        if pcm:
                            with wave.open(str(out / (tid + ".wav")), "wb") as wav:
                                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                                wav.writeframes(pcm)
                        results.append(row)
                        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
                        print(tid, row["completed"], round(row["first_audio_ms"] or 0), flush=True)
                    await asyncio.gather(*(one(side) for side in range(concurrency)))
                    await asyncio.sleep(args.gap)
        finally:
            await asyncio.gather(*(c.close() for cs in connections.values() for c in cs))
    summary = {}
    for c in args.concurrency:
        for mode in modes:
            rows = [r for r in results if r["concurrency"] == c and r["mode"] == mode]
            values = sorted(r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None)
            summary[f"{c}/{mode}"] = dict(
                n=len(rows), completed=sum(r["completed"] for r in rows),
                first_audio_p50_ms=statistics.median(values) if values else None,
                first_audio_p95_ms=values[math.ceil(.95*len(values))-1] if len(values)>=20 else None,
                supply_deficit_p50_ms=statistics.median(r["supply_deficit_ms"] for r in rows),
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "COMPLETE").write_text(str(len(results)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:40001/api-ws/v1/realtime")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--modes", default="raw,normalized,batched,buffered")
    parser.add_argument("--delta-ms", type=float, default=15)
    parser.add_argument("--gap", type=float, default=3)
    parser.add_argument("--log-text", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(c not in (1, 2) for c in args.concurrency):
        parser.error("positive repeats and concurrency 1 or 2 required")
    if args.delta_ms < 0 or args.gap < 0:
        parser.error("delta-ms and gap must be nonnegative")
    if not set(args.modes.split(",")) <= {"raw", "normalized", "batched", "buffered"}:
        parser.error("unknown mode")
    try:
        asyncio.run(run(args))
    finally:
        shutdown_structured_log_writer()
