from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class KnowledgeEntityHint:
    type: str
    external_id: str
    display_name: str | None = None

    def as_dict(self) -> dict[str, str]:
        value = {"type": self.type, "external_id": self.external_id}
        if self.display_name:
            value["display_name"] = self.display_name
        return value


@dataclass(frozen=True, slots=True)
class KnowledgeBinding:
    binding_id: str
    required: bool
    tenant_id: str
    snapshot_id: str
    state_token: str
    binding_revision: int | None = None
    status: Literal["ready", "degraded"] = "ready"
    mode: Literal["retrieval", "provided_context"] = "retrieval"


@dataclass(frozen=True, slots=True)
class ProvidedEntitySnapshot:
    snapshot_id: str
    revision: int
    current_entity_id: str
    current_entity_text: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeEvidence:
    evidence_id: str
    source_type: str
    source_id: str
    title: str
    content: str
    authority: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    decision: Literal["SKIP", "RETRIEVE", "CLARIFY", "DEGRADED"]
    reason: str
    result_id: str
    state_token: str
    snapshot_id: str
    capabilities: tuple[str, ...] = ()
    evidence: tuple[KnowledgeEvidence, ...] = ()
    clarification_reason: str | None = None
    degraded_code: str | None = None
    elapsed_ms: float = 0.0

    @property
    def should_inject(self) -> bool:
        return self.decision in {"RETRIEVE", "CLARIFY", "DEGRADED"}


@dataclass(frozen=True, slots=True)
class PreparedKnowledgeTurn:
    request_id: str
    session_id: str
    turn_id: str
    snapshot_id: str
    state_token: str
    preparation_id: str | None
    payload: dict[str, Any]
    started_at: float
    deadline_at: float
    error_code: str | None = None
