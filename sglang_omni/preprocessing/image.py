# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic image preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key


PREPARED_IMAGE_WIRE_TYPE = "sglang_omni.prepared_image.rgb.v1"


def load_image_path(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB."""
    return Image.open(path).convert("RGB")


class ImageMediaIO(MediaIO[Image.Image]):
    """MediaIO implementation for image files."""

    def __init__(self, *, image_mode: str = "RGB", **kwargs) -> None:
        """Initialize ImageMediaIO.

        Args:
            image_mode: Target image mode (default: "RGB").
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
        """
        super().__init__()
        self.image_mode = image_mode
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> Image.Image:
        """Load image from raw bytes."""
        try:
            return Image.open(BytesIO(data)).convert(self.image_mode)
        except UnidentifiedImageError as e:
            raise ValueError(f"Failed to identify image: {e}") from e

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> Image.Image:
        """Load image from base64-encoded data."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> Image.Image:
        """Load image from a local file path."""
        try:
            return Image.open(filepath).convert(self.image_mode)
        except UnidentifiedImageError as e:
            raise ValueError(f"Failed to identify image: {e}") from e


def is_prepared_image_wire(value: Any) -> bool:
    """Return whether ``value`` is a valid msgpack-safe prepared RGB image."""
    if not isinstance(value, dict) or value.get("_type") != PREPARED_IMAGE_WIRE_TYPE:
        return False
    width = value.get("width")
    height = value.get("height")
    pixels = value.get("pixel_bytes")
    source_sha256 = value.get("source_sha256")
    pixel_sha256 = value.get("pixel_sha256")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
        or not isinstance(pixels, bytes)
        or len(pixels) != width * height * 3
    ):
        return False
    return all(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        for digest in (source_sha256, pixel_sha256)
    )


def prepare_image_bytes_for_wire(data: bytes) -> dict[str, Any]:
    """Decode compressed image bytes before inference into a wire envelope."""
    image = ImageMediaIO(image_mode="RGB").load_bytes(data)
    image.load()
    pixels = image.tobytes()
    return {
        "_type": PREPARED_IMAGE_WIRE_TYPE,
        "width": int(image.width),
        "height": int(image.height),
        "pixel_bytes": pixels,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
    }


def load_prepared_image_wire(value: dict[str, Any]) -> Image.Image:
    """Reconstruct a PIL image from a validated prepared-image envelope."""
    if not is_prepared_image_wire(value):
        raise ValueError("invalid prepared image wire payload")
    return Image.frombytes(
        "RGB",
        (int(value["width"]), int(value["height"])),
        value["pixel_bytes"],
    )


def compute_image_cache_key(images: Any) -> str | None:
    """Compute cache key from raw image inputs (paths, URLs, PIL Images).

    This should be called BEFORE ensure_image_list() to capture original
    paths/URLs which are much cheaper to hash than pixel data.
    """
    return compute_media_cache_key(images, prefix="image")


async def ensure_image_list_async(
    images: Any,
    *,
    image_mode: str = "RGB",
    media_connector: Any | None = None,
) -> list[Any]:
    """Asynchronously normalize image inputs into a list.

    Args:
        images: Image input(s) - can be a path, URL, PIL Image, or list.
        image_mode: Target image mode (default: "RGB").
        media_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.

    Returns:
        List of normalized PIL Images.
    """
    if images is None:
        return []
    items = images if isinstance(images, list) else [images]

    # Import here to avoid circular dependency
    if media_connector is None:
        from .resource_connector import get_global_resource_connector

        media_connector = get_global_resource_connector()

    # Collect coroutines for URL items
    coroutines: list[asyncio.Task[Any] | None] = []
    url_indices: list[int] = []
    normalized: list[Any] = []

    # First pass: identify URL items and create coroutines
    for idx, item in enumerate(items):
        if is_prepared_image_wire(item):
            normalized.append(load_prepared_image_wire(item))
            continue
        if isinstance(item, (str, Path)):
            if _is_url(item):
                # Create coroutine for async URL fetching
                coro = media_connector.fetch_image_async(
                    str(item), image_mode=image_mode
                )
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder
            else:
                normalized.append(load_image_path(item))
        else:
            # Already processed (PIL Image, etc.)
            normalized.append(item)

    # Wait for all URL fetches to complete
    if coroutines:
        results = await asyncio.gather(*coroutines)
        # Fill in the results at the correct indices
        for url_idx, result in zip(url_indices, results):
            normalized[url_idx] = result

    return normalized


def build_image_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard image tensors from HF processor outputs."""
    return {
        "pixel_values": hf_inputs.get("pixel_values"),
        "image_grid_thw": hf_inputs.get("image_grid_thw"),
    }
