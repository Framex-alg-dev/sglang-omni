from __future__ import annotations

from io import BytesIO

import msgpack
import pytest
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.models.qwen3_omni.action_scoring import (
    ActionScoreCandidate,
    ActionSuffixScoreRequest,
    validate_action_suffix_request,
)
from sglang_omni.preprocessing.image import (
    compute_image_cache_key,
    ensure_image_list_async,
    is_prepared_image_wire,
    prepare_image_bytes_for_wire,
)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (3, 2), color=(17, 34, 51))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_prepared_image_wire_round_trips_exact_rgb_pixels() -> None:
    payload = prepare_image_bytes_for_wire(_png_bytes())

    assert is_prepared_image_wire(payload)
    assert payload["width"] == 3
    assert payload["height"] == 2
    assert len(payload["pixel_bytes"]) == 18

    images = await ensure_image_list_async([payload])
    assert len(images) == 1
    assert images[0].mode == "RGB"
    assert images[0].size == (3, 2)
    assert images[0].tobytes() == payload["pixel_bytes"]


def test_prepared_image_wire_has_stable_encoder_cache_key() -> None:
    payload = prepare_image_bytes_for_wire(_png_bytes())

    first = compute_image_cache_key([payload])
    second = compute_image_cache_key([dict(payload)])

    assert first is not None
    assert first == second


def test_prepared_image_wire_rejects_inconsistent_dimensions() -> None:
    payload = prepare_image_bytes_for_wire(_png_bytes())
    payload["width"] = 4

    assert not is_prepared_image_wire(payload)


def test_prepared_image_wire_is_msgpack_safe() -> None:
    payload = prepare_image_bytes_for_wire(_png_bytes())

    encoded = msgpack.packb(payload, use_bin_type=True)
    decoded = msgpack.unpackb(encoded, raw=False)

    assert is_prepared_image_wire(decoded)
    assert decoded["pixel_bytes"] == payload["pixel_bytes"]


def test_action_request_accepts_and_serializes_prepared_image() -> None:
    payload = prepare_image_bytes_for_wire(_png_bytes())
    request = ActionSuffixScoreRequest(
        request_id="prepared-image-request",
        model="qwen3-omni",
        prefix="请选择动作：",
        language="zh",
        candidates=[
            ActionScoreCandidate(
                candidate_id="A1",
                suffix="A1",
                action_id="wave",
            )
        ],
        audios=[],
        images=[payload],
        sample_rate=16000,
        suffix_tokenization_mode="short_id",
    )

    validate_action_suffix_request(request)
    omni_request = Client._build_action_scoring_request(request)
    encoded = msgpack.packb(omni_request.to_dict(), use_bin_type=True)

    assert encoded
