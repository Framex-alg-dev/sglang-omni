"""V25 frozen events exposed through the current 50-event E-code contract."""

from __future__ import annotations

from dataclasses import dataclass

from .v25_contract import R9_CONTRACTS, R9_TEMPORAL_COMPONENTS


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    model_code: str
    summary: str
    rule: str


_V25_EVENTS = tuple(
    EventSpec(
        event_id=f"E{int(contract.code[1:])}",
        model_code=contract.code,
        summary=contract.name,
        rule=(
            f"成立:{contract.evidence}|"
            f"本轮触发:{contract.output_when.replace('落入A', '落入C')}|"
            f"排除:{contract.exclude}"
        ),
    )
    for contract in R9_CONTRACTS
    if contract.code != "P27"
)

# V25/R10 emits every cough/sneeze and every watch check as an occurrence.
# The service compatibility layer derives the current R12 aggregate events.
EVENTS = (*_V25_EVENTS, EventSpec(
    event_id="E50",
    model_code="P50",
    summary="多次咳嗽或喷嚏",
    rule="最近10秒内至少两次独立完整发作，第二次完成点进入C时触发",
))

EVENT_IDS = tuple(event.event_id for event in EVENTS)
EVENT_ID_SET = frozenset(EVENT_IDS)
EVENT_SUMMARIES = {event.event_id: event.summary for event in EVENTS}
P_TO_EVENT = {event.model_code: event.event_id for event in EVENTS}
# V25 cannot reliably distinguish its thumbs-up and OK outputs. Keep the
# frozen model prompt untouched and collapse both model codes at this adapter
# boundary. E13 remains in the current 50-event catalog for a future model.
P_TO_EVENT["P13"] = "E14"
OCCURRENCE_EVENT_IDS = frozenset({"E8", "E22", "E38", "E44", "E49"})

if len(EVENTS) != 50 or len(EVENT_IDS) != len(set(EVENT_IDS)):
    raise RuntimeError("timeline catalog must contain 50 unique E events")
if "E27" in EVENT_IDS:
    raise RuntimeError("retired compatibility event E27 must stay disabled")
if not set(P_TO_EVENT.values()) <= EVENT_ID_SET:
    raise RuntimeError("V25 compatibility map points outside the 50-event catalog")


def model_event_table(alias_by_p: dict[str, str]) -> str:
    return "\n".join(
        (
            f"{alias_by_p[event.model_code]}|"
            f"{'+'.join(R9_TEMPORAL_COMPONENTS[event.model_code])}|"
            f"{event.summary}|{event.rule}"
        )
        for event in _V25_EVENTS
    )
