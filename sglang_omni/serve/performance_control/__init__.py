"""Per-turn facial-expression and internal speech-style inference."""

from .fusion import PerformanceFusionResult, fuse_performance_decision
from .models import PerformanceDecision
from .pipeline import PerformancePipeline

__all__ = [
    "PerformanceDecision",
    "PerformanceFusionResult",
    "PerformancePipeline",
    "fuse_performance_decision",
]
