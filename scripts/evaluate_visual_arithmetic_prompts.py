#!/usr/bin/env python3
"""Replay saved user-camera turns against the visual arithmetic prompt."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from sglang_omni.serve.realtime.reply.prompts import ReplyPromptComponent


class _ChinesePrompt(ReplyPromptComponent):
    @staticmethod
    def _prompt(*, zh: str, en: str) -> str:
        del en
        return zh


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18010")
    return parser.parse_args()


def _request(endpoint: str, prompt: str, images: list[str]) -> tuple[str, float]:
    payload = {
        "model": "qwen3-omni-single-gpu",
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[当前用户摄像头图片]紧随其后的一张或多张图片来自"
                            "本轮用户摄像头，按采集顺序排列，只表示用户及其周围"
                            "环境，不表示当前角色自身的外观、姿势、动作或状态。"
                        ),
                    },
                    *({"type": "image"} for _ in images),
                ],
            },
        ],
        "images": images,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 8,
        "stop": ["\n"],
        "modalities": ["text"],
    }
    encoded = json.dumps(payload).encode()
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result["choices"][0]["message"]["content"], elapsed_ms


def main() -> None:
    args = _parse_args()
    manifest_path = args.session / "user_camera_frames.json"
    manifest = json.loads(manifest_path.read_text())
    prompt = _ChinesePrompt()._visual_arithmetic_operand_output_part()["text"]
    for turn in manifest["turns"]:
        images = [
            str(args.session / frame["file"])
            for frame in turn.get("frames", [])
        ]
        if not images:
            continue
        output, elapsed_ms = _request(args.endpoint, prompt, images)
        print(
            json.dumps(
                {
                    "turn_id": turn["turn_id"],
                    "image_count": len(images),
                    "output": output,
                    "elapsed_ms": round(elapsed_ms, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
