# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from sglang_omni.utils.blackwell_kernel_configs import (
    install_blackwell_kernel_configs,
    install_blackwell_kernel_configs_if_available,
)


def test_install_blackwell_kernel_configs(tmp_path: Path) -> None:
    assets = Path(__file__).resolve().parents[3] / "deploy" / "kernel_configs" / "blackwell"
    sglang_root = tmp_path / "sglang"

    changed = install_blackwell_kernel_configs(
        asset_root=assets,
        sglang_root=sglang_root,
        triton_version="3.6.0",
    )

    assert len(changed) == 5
    assert len(list((sglang_root / "kernels/ops/quantization/configs").glob("N=*.json"))) == 2
    assert len(
        list(
            (
                sglang_root
                / "srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0"
            ).glob("E=*.json")
        )
    ) == 3
    assert (
        install_blackwell_kernel_configs(
            asset_root=assets,
            sglang_root=sglang_root,
            triton_version="3.6.0",
        )
        == ()
    )


def test_skip_blackwell_kernel_configs_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert install_blackwell_kernel_configs_if_available() == ()
