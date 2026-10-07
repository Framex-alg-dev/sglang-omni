"""Closed coarse decisions for low-latency turn-router suffix scoring."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.models.qwen3_omni.action_scoring import ActionScoreCandidate


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    candidate_id: str
    route_token: str
    output_directive: str
    task_directive: str
    media_directive: str
    response_locale: str

    @property
    def requires_cue(self) -> bool:
        return self.route_token == "delegate"

    def prompt_definition(self) -> str:
        cue = "cue_required" if self.requires_cue else "cue_null"
        return "|".join(
            (
                self.route_token,
                self.output_directive,
                self.task_directive,
                self.media_directive,
                self.response_locale,
                cue,
            )
        )

    def scoring_candidate(self) -> ActionScoreCandidate:
        return ActionScoreCandidate(
            candidate_id=self.candidate_id,
            suffix=self.candidate_id,
        )


def _catalog() -> tuple[RouteCandidate, ...]:
    coarse: list[tuple[str, str, str, str]] = [
        ("direct", "keep", "keep", "none"),
        ("delegate", "keep", "keep", "none"),
        ("control", "suppress_reply", "keep", "none"),
        ("control", "stop_current", "keep", "none"),
        ("control", "suppress_reply", "cancel_current", "none"),
        ("control", "suppress_reply", "cancel_all", "none"),
        ("control", "keep", "keep", "stop"),
        ("control", "keep", "keep", "pause"),
        ("control", "keep", "keep", "resume"),
        ("control", "suppress_reply", "keep", "stop"),
    ]
    candidates: list[RouteCandidate] = []
    for locale in ("zh-CN", "en-US"):
        for route_token, output, task, media in coarse:
            candidates.append(
                RouteCandidate(
                    candidate_id=f"R{len(candidates) + 1:03d}",
                    route_token=route_token,
                    output_directive=output,
                    task_directive=task,
                    media_directive=media,
                    response_locale=locale,
                )
            )
    return tuple(candidates)


ROUTE_CANDIDATES = _catalog()
ROUTE_CANDIDATE_BY_ID = {
    candidate.candidate_id: candidate for candidate in ROUTE_CANDIDATES
}
SCORING_CANDIDATE_CATALOG = "\n".join(
    f"{candidate.candidate_id}={candidate.prompt_definition()}"
    for candidate in ROUTE_CANDIDATES
)


__all__ = [
    "ROUTE_CANDIDATES",
    "ROUTE_CANDIDATE_BY_ID",
    "SCORING_CANDIDATE_CATALOG",
    "RouteCandidate",
]
