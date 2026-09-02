"""Architecture contracts for the package-based realtime implementation."""

from pathlib import Path

from sglang_omni.serve.realtime.action import ActionPipeline
from sglang_omni.serve.realtime.action.pipeline import ActionScoringPipeline
from sglang_omni.serve.realtime.action.category import ActionCategoryComponent
from sglang_omni.serve.realtime.action.candidate import ActionCandidateComponent
from sglang_omni.serve.realtime.memory import (
    SessionMemoryScheduler,
    SessionMemoryStore,
    build_memory_extraction_request,
)
from sglang_omni.serve.realtime.memory.extractor import (
    build_memory_extraction_request as package_build_memory_extraction_request,
)
from sglang_omni.serve.realtime.memory.controller import SessionMemoryController
from sglang_omni.serve.realtime.memory.scheduler import (
    SessionMemoryScheduler as PackageSessionMemoryScheduler,
)
from sglang_omni.serve.realtime.memory.store import (
    SessionMemoryStore as PackageSessionMemoryStore,
)
from sglang_omni.serve.realtime.multimodal import MultimodalSession
from sglang_omni.serve.realtime.protocol.input import TurnInputComponent
from sglang_omni.serve.realtime.protocol.validation import ProtocolComponent
from sglang_omni.serve.realtime.reply import ReplyPipeline
from sglang_omni.serve.realtime.reply.routing import ReplyRoutingComponent
from sglang_omni.serve.realtime.reply.tts import ReplyTTSComponent
from sglang_omni.serve.realtime.session_memory import (
    SessionMemoryScheduler as FacadeSessionMemoryScheduler,
    SessionMemoryStore as FacadeSessionMemoryStore,
    build_memory_extraction_request as facade_build_memory_extraction_request,
)
from sglang_omni.serve.realtime.turn_pipeline import TurnPipeline

REALTIME_DIR = Path(__file__).parents[3] / "sglang_omni" / "serve" / "realtime"


def test_session_memory_facades_preserve_public_identity() -> None:
    assert SessionMemoryStore is PackageSessionMemoryStore
    assert SessionMemoryScheduler is PackageSessionMemoryScheduler
    assert build_memory_extraction_request is package_build_memory_extraction_request
    assert FacadeSessionMemoryStore is PackageSessionMemoryStore
    assert FacadeSessionMemoryScheduler is PackageSessionMemoryScheduler
    assert (
        facade_build_memory_extraction_request
        is package_build_memory_extraction_request
    )


def test_multimodal_session_uses_explicit_component_composition() -> None:
    assert MultimodalSession.__mro__ == (MultimodalSession, object)
    assert MultimodalSession.__components__ == (
        TurnPipeline,
        ProtocolComponent,
        SessionMemoryController,
        ReplyPipeline,
        ActionPipeline,
    )


def test_nested_pipelines_are_composed_without_mixin_inheritance() -> None:
    assert ProtocolComponent.__mro__ == (ProtocolComponent, object)
    assert ReplyPipeline.__mro__ == (ReplyPipeline, object)
    assert ActionPipeline.__mro__ == (ActionPipeline, object)
    assert TurnInputComponent in ProtocolComponent.__components__
    assert ReplyRoutingComponent in ReplyPipeline.__components__
    assert ActionScoringPipeline in ActionPipeline.__components__
    assert ActionCategoryComponent in ActionScoringPipeline.__components__
    assert ActionCandidateComponent in ActionScoringPipeline.__components__


def test_high_risk_methods_are_owned_by_focused_components() -> None:
    assert MultimodalSession.handle_turn_commit is TurnPipeline.handle_turn_commit
    assert (
        MultimodalSession._score_action_hierarchical
        is ActionCategoryComponent._score_action_hierarchical
    )
    assert (
        MultimodalSession._classify_reply_history_requirement
        is ReplyRoutingComponent._classify_reply_history_requirement
    )
    assert MultimodalSession._start_reply_tts is ReplyTTSComponent._start_reply_tts


def test_public_facades_stay_small_and_target_packages_exist() -> None:
    required = {
        "protocol/models.py",
        "protocol/validation.py",
        "protocol/events.py",
        "reply/pipeline.py",
        "reply/provisional.py",
        "reply/tts.py",
        "reply/routing.py",
        "reply/prompts.py",
        "reply/history.py",
        "action/models.py",
        "action/pipeline.py",
        "action/category.py",
        "action/candidate.py",
        "action/context.py",
        "action/prompts.py",
        "memory/models.py",
        "memory/config.py",
        "memory/store.py",
        "memory/extractor.py",
        "memory/scheduler.py",
        "memory/controller.py",
        "multimodal_session.py",
    }
    assert all((REALTIME_DIR / relative).is_file() for relative in required)
    assert len((REALTIME_DIR / "multimodal.py").read_text().splitlines()) <= 50
    assert len((REALTIME_DIR / "session_memory.py").read_text().splitlines()) <= 50


def test_first_stage_flat_modules_do_not_return() -> None:
    transitional = {
        "multimodal_action.py",
        "multimodal_action_scoring.py",
        "multimodal_common.py",
        "multimodal_memory.py",
        "multimodal_protocol.py",
        "multimodal_reply.py",
        "multimodal_reply_generation.py",
        "multimodal_reply_routing.py",
        "multimodal_reply_stream.py",
        "multimodal_turn.py",
        "multimodal_turn_input.py",
        "multimodal_types.py",
        "session_memory_extraction.py",
        "session_memory_models.py",
        "session_memory_scheduler.py",
        "session_memory_store.py",
    }
    assert not any((REALTIME_DIR / name).exists() for name in transitional)
