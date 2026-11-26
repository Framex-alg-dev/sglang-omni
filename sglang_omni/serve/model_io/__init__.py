"""Non-blocking model input/output archival for Omni services."""

from .recorder import ModelIORecorder, get_model_io_recorder

__all__ = ["ModelIORecorder", "get_model_io_recorder"]
