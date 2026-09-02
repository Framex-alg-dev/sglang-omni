# SPDX-License-Identifier: Apache-2.0
"""Install repository-owned SGLang kernel configs for RTX PRO 5000 Blackwell."""

from __future__ import annotations

import filecmp
import importlib.util
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEVICE_MARKER = "device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell"
_DEVICE_NAME = "NVIDIA RTX PRO 5000 72GB Blackwell"


def _repository_asset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "deploy" / "kernel_configs" / "blackwell"


def _installed_sglang_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate the installed sglang package")
    return Path(spec.origin).resolve().parent


def _copy_if_changed(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and filecmp.cmp(source, destination, shallow=False):
        return False
    shutil.copy2(source, destination)
    return True


def install_blackwell_kernel_configs(
    *,
    asset_root: Path | None = None,
    sglang_root: Path | None = None,
    triton_version: str | None = None,
) -> tuple[Path, ...]:
    """Install tuned dense-FP8 and MoE JSON files into the active SGLang.

    The CLI calls this before launching pipeline processes, so editable local
    development and the editable production Docker install use the same files.
    Returning only changed destinations makes the operation idempotent.
    """

    assets = asset_root or _repository_asset_root()
    if not assets.is_dir():
        raise RuntimeError(f"Blackwell kernel config directory is missing: {assets}")

    root = sglang_root or _installed_sglang_root()
    if triton_version is None:
        import triton

        triton_version = triton.__version__

    dense_dir = root / "kernels" / "ops" / "quantization" / "configs"
    moe_dir = (
        root
        / "srt"
        / "layers"
        / "moe"
        / "moe_runner"
        / "triton_utils"
        / "configs"
        / f"triton_{triton_version.replace('.', '_')}"
    )

    sources = sorted(assets.glob("*.json"))
    if not sources or any(_DEVICE_MARKER not in source.name for source in sources):
        raise RuntimeError(f"invalid or empty Blackwell kernel config set: {assets}")

    changed: list[Path] = []
    for source in sources:
        destination_dir = moe_dir if source.name.startswith("E=") else dense_dir
        destination = destination_dir / source.name
        if _copy_if_changed(source, destination):
            changed.append(destination)

    if changed:
        logger.info(
            "Installed %d RTX PRO 5000 Blackwell kernel configs into SGLang",
            len(changed),
        )
    return tuple(changed)


def install_blackwell_kernel_configs_if_available() -> tuple[Path, ...]:
    """Install configs only when the tuned GPU is visible to this process."""

    import torch

    if not torch.cuda.is_available():
        return ()
    visible_devices = {
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    }
    if _DEVICE_NAME not in visible_devices:
        return ()
    return install_blackwell_kernel_configs()
