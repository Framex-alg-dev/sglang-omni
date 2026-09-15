#!/usr/bin/env python3
"""Build deterministic image and spoken-audio inputs for the 24-turn journey."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly

from sglang_omni.serve.realtime.embedded_tts import (
    EmbeddedTTSConfig,
    EmbeddedTTSConnection,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "benchmarks/eval/realtime_multimodal_user_journey_cases.json"
DEFAULT_OUTPUT = ROOT / "reports/realtime_multimodal_user_journey/assets"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int = 32,
    *,
    fill: str = "#172033",
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=_font(size, bold=bold), fill=fill)


def _canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1024, 768), "#eef2f6")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, 996, 104), 18, fill="#ffffff")
    _label(draw, (58, 46), title, 34, bold=True)
    return image, draw


def _draw_desk(scene: str) -> Image.Image:
    image, draw = _canvas("Camera view: workshop desk")
    draw.rectangle((30, 360, 994, 740), fill="#caa77e")
    draw.rounded_rectangle((650, 400, 950, 650), 18, fill="#252b36")
    draw.rectangle((690, 435, 910, 585), fill="#9bc5e8")
    _label(draw, (748, 490), "LAPTOP", 28, bold=True)
    if scene == "desk_start":
        draw.rounded_rectangle((360, 455, 600, 645), 12, fill="#2674c7")
        _label(draw, (390, 520), "BLUE", 28, fill="white", bold=True)
        _label(draw, (383, 560), "NOTEBOOK", 25, fill="white", bold=True)
        draw.ellipse((90, 430, 255, 485), fill="#d94343", outline="#792020", width=4)
        draw.rounded_rectangle((100, 455, 245, 625), 22, fill="#d94343", outline="#792020", width=4)
        draw.arc((215, 475, 305, 580), -85, 85, fill="#792020", width=18)
        _label(draw, (124, 520), "RED", 29, fill="white", bold=True)
        _label(draw, (119, 558), "MUG", 29, fill="white", bold=True)
    else:
        draw.rounded_rectangle((315, 455, 555, 645), 12, fill="#2674c7")
        _label(draw, (345, 520), "BLUE", 28, fill="white", bold=True)
        _label(draw, (338, 560), "NOTEBOOK", 25, fill="white", bold=True)
        draw.ellipse((105, 500, 220, 615), fill="#6eaa58")
        draw.rectangle((145, 600, 180, 705), fill="#6a472d")
    return image


def _draw_schedule(room: str) -> Image.Image:
    image, draw = _canvas("Workshop schedule")
    draw.rounded_rectangle((120, 145, 904, 680), 22, fill="white", outline="#27354d", width=5)
    _label(draw, (210, 200), "AI LITERACY WORKSHOP", 45, bold=True)
    draw.line((180, 285, 840, 285), fill="#aab4c3", width=3)
    _label(draw, (220, 345), "TIME", 31, fill="#58657a", bold=True)
    _label(draw, (530, 330), "14:30", 62, fill="#144b8b", bold=True)
    _label(draw, (220, 485), "ROOM", 31, fill="#58657a", bold=True)
    _label(draw, (530, 465), f"ROOM {room}", 58, fill="#8b2e2e", bold=True)
    return image


def _draw_task_board() -> Image.Image:
    image, draw = _canvas("Preparation board")
    cards = [
        ((90, 175, 930, 310), "CHECK AUDIO", "HIGH", "#ffd868", "#a73333"),
        ((90, 350, 930, 485), "PRINT BADGES", "DONE", "#a9dfaa", "#276d35"),
        ((90, 525, 930, 660), "BUY CABLES", "OPEN", "#b9d7f4", "#285780"),
    ]
    for box, task, state, color, state_color in cards:
        draw.rounded_rectangle(box, 18, fill=color)
        _label(draw, (box[0] + 38, box[1] + 40), task, 38, bold=True)
        _label(draw, (box[2] - 180, box[1] + 42), state, 31, fill=state_color, bold=True)
    return image


def _draw_supplies(count: int) -> Image.Image:
    image, draw = _canvas("Camera view: supply table")
    draw.rectangle((30, 340, 994, 740), fill="#caa77e")
    positions = [(155, 420), (410, 420), (665, 420)]
    for index, (x, y) in enumerate(positions[:count], 1):
        draw.rectangle((x, y, x + 205, y + 210), fill="#d6a85e", outline="#6d4c25", width=5)
        draw.line((x, y + 65, x + 205, y + 65), fill="#6d4c25", width=4)
        _label(draw, (x + 48, y + 108), f"BOX {index}", 29, bold=True)
    return image


def _draw_budget() -> Image.Image:
    image, draw = _canvas("Workshop budget")
    draw.rounded_rectangle((120, 150, 904, 680), 20, fill="white", outline="#33415c", width=4)
    rows = [("PROJECTOR", "680"), ("CABLES", "120"), ("BADGES", "90")]
    _label(draw, (190, 190), "ITEM", 31, fill="#59677d", bold=True)
    _label(draw, (690, 190), "COST", 31, fill="#59677d", bold=True)
    for index, (item, amount) in enumerate(rows):
        y = 285 + index * 120
        draw.line((170, y - 25, 850, y - 25), fill="#d4d9e1", width=2)
        _label(draw, (190, y), item, 37, bold=True)
        _label(draw, (685, y), f"CNY {amount}", 37, fill="#164f86", bold=True)
    return image


def _draw_claim() -> Image.Image:
    image, draw = _canvas("Draft presentation slide")
    draw.rounded_rectangle((85, 145, 939, 685), 20, fill="white", outline="#33415c", width=4)
    _label(draw, (150, 205), "AI SAVES 50% OF", 52, fill="#174b87", bold=True)
    _label(draw, (150, 275), "TEACHERS' TIME", 52, fill="#174b87", bold=True)
    draw.rounded_rectangle((145, 410, 875, 605), 18, fill="#fff0d0")
    _label(draw, (195, 455), "PILOT: 5 VOLUNTEERS", 36, bold=True)
    _label(draw, (195, 525), "NO PUBLISHED BASELINE", 36, fill="#a23838", bold=True)
    return image


def _draw_badge() -> Image.Image:
    image, draw = _canvas("Camera view: updated host badge")
    draw.rounded_rectangle((285, 160, 739, 670), 28, fill="white", outline="#244c7c", width=6)
    draw.rectangle((285, 160, 739, 270), fill="#244c7c")
    _label(draw, (392, 194), "HOST", 45, fill="white", bold=True)
    _label(draw, (360, 355), "MAYA", 76, fill="#182b45", bold=True)
    _label(draw, (335, 490), "AI WORKSHOP", 33, fill="#5b6879", bold=True)
    return image


def _draw_spill(clean: bool) -> Image.Image:
    image, draw = _canvas("Camera view: equipment table")
    draw.rectangle((30, 350, 994, 740), fill="#caa77e")
    draw.rounded_rectangle((650, 470, 930, 610), 15, fill="#eeeeee", outline="#3f4650", width=4)
    for x in (700, 790, 880):
        draw.ellipse((x, 510, x + 24, 534), fill="#30343a")
        draw.rectangle((x + 7, 548, x + 17, 568), fill="#30343a")
    _label(draw, (680, 630), "POWER STRIP", 24, bold=True)
    if clean:
        draw.rounded_rectangle((125, 445, 475, 615), 24, fill="#e6f5e5", outline="#4f8e55", width=5)
        _label(draw, (192, 495), "AREA DRY", 38, fill="#34703b", bold=True)
        _label(draw, (167, 550), "PLUG REMOVED", 27, fill="#34703b", bold=True)
    else:
        draw.ellipse((180, 470, 600, 660), fill="#6ec6e8", outline="#237fa4", width=5)
        draw.line((650, 540, 600, 565), fill="#d22f2f", width=8)
        _label(draw, (225, 535), "WATER", 42, fill="#145b77", bold=True)
    return image


def _draw_checklist(final_summary: bool) -> Image.Image:
    image, draw = _canvas("Final workshop check")
    if final_summary:
        draw.rounded_rectangle((55, 140, 420, 670), 18, fill="white")
        _label(draw, (120, 185), "MAYA", 48, bold=True)
        _label(draw, (103, 270), "ROOM C", 40, fill="#8b2e2e", bold=True)
        _label(draw, (120, 345), "14:30", 52, fill="#144b8b", bold=True)
        x0 = 465
    else:
        x0 = 135
    draw.rounded_rectangle((x0, 140, 965 if final_summary else 890, 680), 18, fill="white")
    _label(draw, (x0 + 55, 185), "CHECKLIST", 40, bold=True)
    rows = [
        ("CHECK AUDIO", True),
        ("PRINT BADGES", True),
        ("BUY CABLES", True),
        ("PRACTICE OPENING", False),
    ]
    for index, (item, done) in enumerate(rows):
        y = 285 + index * 88
        draw.rectangle((x0 + 65, y, x0 + 105, y + 40), outline="#26364d", width=4)
        if done:
            draw.line((x0 + 72, y + 22, x0 + 83, y + 34), fill="#31833e", width=6)
            draw.line((x0 + 82, y + 34, x0 + 101, y + 7), fill="#31833e", width=6)
        _label(draw, (x0 + 135, y - 3), item, 29, bold=not done)
    return image


def _render_scene(scene: str) -> Image.Image:
    if scene in {"desk_start", "neutral_desk"}:
        return _draw_desk(scene)
    if scene == "schedule_a":
        return _draw_schedule("A")
    if scene == "schedule_b":
        return _draw_schedule("C")
    if scene == "task_board":
        return _draw_task_board()
    if scene == "supplies_three":
        return _draw_supplies(3)
    if scene == "supplies_two":
        return _draw_supplies(2)
    if scene == "budget":
        return _draw_budget()
    if scene == "claim_slide":
        return _draw_claim()
    if scene == "badge_maya":
        return _draw_badge()
    if scene == "spill":
        return _draw_spill(False)
    if scene == "spill_clean":
        return _draw_spill(True)
    if scene == "final_checklist":
        return _draw_checklist(False)
    if scene == "final_summary":
        return _draw_checklist(True)
    raise ValueError(f"unknown scene: {scene}")


def _load_turns(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    turns = payload.get("turns")
    if not isinstance(turns, list) or len(turns) < 20:
        raise ValueError("journey must contain at least 20 turns")
    if len({str(turn.get("id")) for turn in turns}) != len(turns):
        raise ValueError("turn ids must be unique")
    return turns


def _prepare_images(turns: list[dict[str, Any]], output: Path) -> None:
    for scene in sorted({str(turn["scene"]) for turn in turns}):
        target = output / f"{scene}.jpg"
        _render_scene(scene).save(target, format="JPEG", quality=94, optimize=True)


async def _prepare_audio(
    turns: list[dict[str, Any]],
    output: Path,
    *,
    tts_url: str,
    voice: str,
) -> None:
    connection = EmbeddedTTSConnection(
        EmbeddedTTSConfig(
            url=tts_url,
            voice=voice,
            turn_timeout_seconds=120,
        ),
        session_id="realtime-multimodal-user-journey-fixtures",
    )
    try:
        for turn in turns:
            target = output / f"turn_{turn['id']}.wav"
            if target.exists():
                continue
            pcm_24k = bytearray()

            async def chunks():
                yield str(turn["prompt"])

            async def sink(chunk: bytes) -> None:
                pcm_24k.extend(chunk)

            await connection.synthesize_streaming(
                turn_id=f"fixture-{turn['id']}",
                text_chunks=chunks(),
                audio_sink=sink,
                instruct="Read naturally as a real user speaking in a video call.",
            )
            samples = np.frombuffer(pcm_24k, dtype="<i2").astype(np.float32)
            samples_16k = np.clip(
                resample_poly(samples, 2, 3), -32768, 32767
            ).astype("<i2")
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples_16k.tobytes())
            print(f"audio {turn['id']}: {len(samples_16k) / 16000:.2f}s", flush=True)
    finally:
        await connection.close()


def _write_manifest(turns: list[dict[str, Any]], output: Path) -> None:
    assets: list[dict[str, Any]] = []
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        item: dict[str, Any] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path.suffix == ".wav":
            with wave.open(str(path)) as wav:
                item["sample_rate"] = wav.getframerate()
                item["channels"] = wav.getnchannels()
                item["seconds"] = round(wav.getnframes() / wav.getframerate(), 3)
        assets.append(item)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "turn_count": len(turns),
                "every_turn_has_audio": all((output / f"turn_{turn['id']}.wav").exists() for turn in turns),
                "every_turn_has_image": all((output / f"{turn['scene']}.jpg").exists() for turn in turns),
                "assets": assets,
            },
            indent=2,
        )
    )


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tts-url", default="ws://127.0.0.1:40001/api-ws/v1/realtime")
    parser.add_argument("--voice", default="spk_691b97a24dcc")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()

    turns = _load_turns(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_images(turns, args.output_dir)
    if not args.images_only:
        await _prepare_audio(
            turns,
            args.output_dir,
            tts_url=args.tts_url,
            voice=args.voice,
        )
    _write_manifest(turns, args.output_dir)
    print(f"prepared {len(turns)} turns in {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(_main())
