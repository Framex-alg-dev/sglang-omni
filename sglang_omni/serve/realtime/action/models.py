"""Action-domain data contracts.

The canonical dataclasses remain protocol-visible because they are serialized
on the wire; this module gives action code a domain-local import surface.
"""

from sglang_omni.serve.realtime.protocol.models import (
    ActionHistoryTurn,
    ExecutedActionRecord,
    SessionActionCandidate,
    SessionActionCategory,
    SessionActionProfile,
)

__all__ = [
    "ActionHistoryTurn",
    "ExecutedActionRecord",
    "SessionActionCandidate",
    "SessionActionCategory",
    "SessionActionProfile",
]

