"""Check the opt-in override without importing the GPU scheduler stack."""
import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


def factory():
    path = Path(__file__).resolve().parents[3] / "sglang_omni/models/qwen3_omni/stages.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "create_decode_executor")
    torch = Mock()
    scheduler = object()
    create = Mock(return_value=scheduler)
    namespace = {"torch": torch, "logger": Mock(),
                 "create_streaming_detokenize_scheduler": create}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["create_decode_executor"], torch, create, scheduler


def test_default_preserves_process_threads():
    fn, torch, create, scheduler = factory()
    assert fn("model") is scheduler
    create.assert_called_once_with("model")
    torch.set_num_threads.assert_not_called()


def test_override_applied_after_scheduler_initialization():
    fn, torch, create, scheduler = factory()
    order = []
    create.side_effect = lambda _: order.append("scheduler") or scheduler
    torch.set_num_threads.side_effect = lambda n: order.append(n)
    assert fn("model", cpu_num_threads=8) is scheduler
    assert order == ["scheduler", 8]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_override_fails_before_initialization(value):
    fn, torch, create, _ = factory()
    with pytest.raises(ValueError, match="positive integer"):
        fn("model", cpu_num_threads=value)
    create.assert_not_called()
    torch.set_num_threads.assert_not_called()
