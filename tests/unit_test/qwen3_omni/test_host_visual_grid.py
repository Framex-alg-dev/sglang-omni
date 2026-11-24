import pytest
import torch

from sglang_omni.models.qwen3_omni.merge import _host_visual_grid, build_thinker_inputs
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState


def test_missing_metadata_and_cpu_encoder_fallback():
    grid = torch.tensor([[1, 4, 6]])
    assert _host_visual_grid(None, grid) is grid
    assert _host_visual_grid(grid, None) is grid
    assert _host_visual_grid(None, None) is None
    assert _host_visual_grid(grid, grid.clone()) is grid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("rows", [1, 4])
def test_merge_preserves_features_and_reuses_host_grid(modality, rows):
    grid = torch.tensor([[1, 4, 6]] * rows)
    device_grid = grid.cuda()
    features = torch.randn(rows * 6, 8, device="cuda", dtype=torch.bfloat16)
    deepstack = [features.clone() for _ in range(3)]
    state = Qwen3OmniPipelineState(mm_inputs={modality: {f"{modality}_grid_thw": grid}})
    encoded = {"image_encoder": {
        f"{modality}_grid_thw": device_grid,
        f"{modality}_embeds": features,
        f"deepstack_visual_embeds_{modality}": deepstack,
    }}
    result = build_thinker_inputs(state, encoded)["model_inputs"]
    assert result[f"{modality}_grid_thw"] is grid
    assert torch.equal(result[f"{modality}_grid_thw"], device_grid.cpu())
    assert result[f"{modality}_embeds"] is features
    assert result["deepstack_visual_embeds"] is deepstack
    assert _host_visual_grid(device_grid, None) is device_grid
    assert _host_visual_grid(device_grid, grid[:0]) is device_grid
    assert _host_visual_grid(device_grid, device_grid) is device_grid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("images", [1, 4])
def test_mrope_positions_are_bitwise_identical(images):
    from types import SimpleNamespace

    from sglang_omni.models.qwen3_omni.request_builders import _compute_mrope_positions

    cfg = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
        vision_start_token_id=1000, vision_end_token_id=1001,
        image_token_id=1002, video_token_id=1003,
        audio_token_id=1004, audio_start_token_id=1005,
        position_id_per_seconds=25,
    )
    grid = torch.tensor([[1, 4, 6]] * images)
    tokens = [42]
    for _ in range(images):
        tokens += [cfg.vision_start_token_id] + [cfg.image_token_id] * 6
        tokens += [cfg.vision_end_token_id, 43]
    ids = torch.tensor(tokens)
    before = _compute_mrope_positions(ids, {"image_grid_thw": grid.cuda()}, cfg)
    after = _compute_mrope_positions(ids, {"image_grid_thw": grid}, cfg)
    assert all(torch.equal(x, y) for x, y in zip(before, after))
