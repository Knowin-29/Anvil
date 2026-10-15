"""Reference implementation of the Anvil contract-v2 routing protocol.

This module validates record structure and implements the deterministic
semantic-first-fault traversal. It intentionally does not parse model output,
score images, or create observations; those components must be supplied by a
future experiment runner and recorded with their own revisions and hashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Real
from typing import Dict, Mapping, Optional, Tuple


class ContractError(ValueError):
    """Raised when a contract-v2 record violates an invariant."""


class RecordType(str, Enum):
    ATOMIC = "atomic"
    LINKED = "linked"


class Stage(str, Enum):
    G = "G"
    S = "S"
    A = "A"


class RouteState(str, Enum):
    ATOMIC_DECLARED = "atomic_declared"
    STRUCTURAL_INVALID = "structural_invalid"
    SEMANTIC_FAILURE = "semantic_failure"
    INDETERMINATE_LOW_CONFIDENCE = "indeterminate_low_confidence"
    EVALUATION_UNAVAILABLE = "evaluation_unavailable"
    ALL_PASS = "all_pass"


FORM_TO_STAGE: Mapping[str, Stage] = {
    "bbox": Stage.G,
    "detection": Stage.G,
    "mask": Stage.G,
    "point": Stage.G,
    "relative_depth": Stage.S,
    "affordance": Stage.A,
    "gripper_dof": Stage.A,
    "action_plan": Stage.A,
}
STAGE_ORDER: Tuple[Stage, ...] = (Stage.G, Stage.S, Stage.A)


def stage_for_form(form: str) -> Stage:
    try:
        return FORM_TO_STAGE[form]
    except KeyError as exc:
        raise ContractError(f"unknown gsa_form: {form!r}") from exc


@dataclass(frozen=True)
class StageSpan:
    """Half-open character offsets in the reference serialization, not tokens."""

    start: int
    end: int

    def validate(self, text_length: int) -> None:
        if type(self.start) is not int or type(self.end) is not int:
            raise ContractError("stage span bounds must be integers")
        if self.start < 0 or self.end <= self.start or self.end > text_length:
            raise ContractError(
                f"invalid stage span [{self.start}, {self.end}) for text length {text_length}"
            )


@dataclass(frozen=True)
class RolloutTokenSpan:
    """Half-open token offsets supplied by the sampled-rollout parser."""

    start: int
    end: int

    def validate(self, token_count: int) -> None:
        if any(type(value) is not int for value in (self.start, self.end, token_count)):
            raise ContractError("rollout token bounds and count must be integers")
        if token_count < 0 or not 0 <= self.start <= self.end <= token_count:
            raise ContractError("rollout token span is outside the sampled sequence")


@dataclass(frozen=True)
class DependencyEdge:
    upstream: Stage
    downstream: Stage
    consumed_field: str


@dataclass(frozen=True)
class StageEvaluation:
    stage: Stage
    structural_valid: bool
    semantic_score: Optional[float]
    tau_route: float
    delta: float

    def state(self) -> str:
        # Invalid scorer configuration is not evidence of a student failure.
        for name, value in (("tau_route", self.tau_route), ("delta", self.delta)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ContractError(f"{name} must be a finite real number")
        if not 0 <= self.tau_route <= 1:
            raise ContractError("tau_route must be in [0, 1]")
        if self.delta < 0:
            raise ContractError("delta must be non-negative")
        if type(self.structural_valid) is not bool:
            raise ContractError("structural_valid must be a boolean")
        if self.semantic_score is not None:
            score = self.semantic_score
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                raise ContractError("semantic score must be a finite real number")
            if not 0 <= score <= 1:
                raise ContractError("semantic score must be in [0, 1]")
        if not self.structural_valid:
            return RouteState.STRUCTURAL_INVALID.value
        if self.semantic_score is None:
            return RouteState.EVALUATION_UNAVAILABLE.value
        if self.semantic_score < self.tau_route - self.delta:
            return RouteState.SEMANTIC_FAILURE.value
        if self.semantic_score >= self.tau_route + self.delta:
            return "pass"
        return RouteState.INDETERMINATE_LOW_CONFIDENCE.value


@dataclass(frozen=True)
class ContractRecord:
    record_type: RecordType
    canonical_uid: str
    image_hash: str
    task_id: str
    entity_namespace: str
    serialized_text: str
    active_stages: Tuple[Stage, ...]
    stage_forms: Mapping[Stage, str]
    stage_spans: Mapping[Stage, StageSpan] = field(default_factory=dict)
    declared_form: Optional[str] = None
    edges: Tuple[DependencyEdge, ...] = ()
    route_eligible: bool = True

    def validate(self) -> None:
        if not self.canonical_uid or not self.image_hash or not self.task_id:
            raise ContractError("canonical_uid, image_hash, and task_id are required")
        if not self.entity_namespace:
            raise ContractError("entity_namespace is required")
        if not isinstance(self.serialized_text, str):
            raise ContractError("serialized_text must be a string")
        if not self.active_stages:
            raise ContractError("active_stages cannot be empty")
        if len(set(self.active_stages)) != len(self.active_stages):
            raise ContractError("active_stages must be unique")
        if any(stage not in STAGE_ORDER for stage in self.active_stages):
            raise ContractError("active_stages contains an unknown stage")
        if set(self.stage_forms) != set(self.active_stages):
            raise ContractError("stage_forms must cover exactly active_stages")
        for stage, form in self.stage_forms.items():
            if stage_for_form(form) is not stage:
                raise ContractError(f"form {form!r} is not assigned to stage {stage.value}")

        spans = self.stage_spans
        if self.record_type is RecordType.ATOMIC:
            if not self.declared_form:
                raise ContractError("atomic records require declared_form")
            declared_stage = stage_for_form(self.declared_form)
            if self.active_stages != (declared_stage,):
                raise ContractError("atomic active_stages must contain the declared stage only")
            if self.stage_forms[declared_stage] != self.declared_form:
                raise ContractError("atomic declared_form disagrees with stage_forms")
            if self.edges:
                raise ContractError("atomic records cannot declare dependency edges")
            return

        if self.record_type is not RecordType.LINKED:
            raise ContractError(f"unknown record_type: {self.record_type!r}")
        if self.declared_form is not None:
            raise ContractError("linked records must not use declared_form")
        if set(spans) != set(self.active_stages):
            raise ContractError("linked records require one stage span per active stage")
        for span in spans.values():
            span.validate(len(self.serialized_text))
        ordered_spans = sorted((span.start, span.end, stage) for stage, span in spans.items())
        for (_, previous_end, previous_stage), (current_start, _, current_stage) in zip(
            ordered_spans, ordered_spans[1:]
        ):
            if current_start < previous_end:
                raise ContractError(
                    f"stage spans overlap: {previous_stage.value} and {current_stage.value}"
                )

        for edge in self.edges:
            if edge.upstream not in self.active_stages or edge.downstream not in self.active_stages:
                raise ContractError("dependency edge endpoint is not an active stage")
            if edge.upstream is edge.downstream:
                raise ContractError("dependency edges cannot be self-loops")
            if not edge.consumed_field.strip():
                raise ContractError("every dependency edge needs a consumed field")
        self.topological_order()

    def predecessors(self, stage: Stage) -> Tuple[Stage, ...]:
        return tuple(edge.upstream for edge in self.edges if edge.downstream is stage)

    def topological_order(self) -> Tuple[Stage, ...]:
        active = set(self.active_stages)
        incoming: Dict[Stage, int] = {stage: 0 for stage in active}
        outgoing: Dict[Stage, list[Stage]] = {stage: [] for stage in active}
        for edge in self.edges:
            incoming[edge.downstream] += 1
            outgoing[edge.upstream].append(edge.downstream)

        ready = [stage for stage in STAGE_ORDER if stage in active and incoming[stage] == 0]
        order = []
        while ready:
            stage = ready.pop(0)
            order.append(stage)
            for downstream in sorted(outgoing[stage], key=STAGE_ORDER.index):
                incoming[downstream] -= 1
                if incoming[downstream] == 0:
                    ready.append(downstream)
            ready.sort(key=STAGE_ORDER.index)
        if len(order) != len(active):
            raise ContractError("dependency graph contains a cycle")
        return tuple(order)


@dataclass(frozen=True)
class RouteDecision:
    state: str
    selected_stage: Optional[Stage]
    mtd_action: str
    reason: str


def _decision_for_state(state: str, stage: Stage, reason: str) -> RouteDecision:
    if state in (RouteState.STRUCTURAL_INVALID.value, RouteState.SEMANTIC_FAILURE.value):
        return RouteDecision(state, stage, "resolve_rollout_span", reason)
    if state in (RouteState.INDETERMINATE_LOW_CONFIDENCE.value, RouteState.EVALUATION_UNAVAILABLE.value):
        return RouteDecision(state, None, "skip_mtd_retain_sft", reason)
    raise ContractError(f"cannot route unexpected state {state!r}")


def route(
    record: ContractRecord,
    evaluations: Mapping[Stage, StageEvaluation] | None = None,
) -> RouteDecision:
    """Localize a route; allocate_mtd must resolve its sampled token span next."""

    record.validate()
    if not record.route_eligible:
        return RouteDecision(
            RouteState.INDETERMINATE_LOW_CONFIDENCE.value,
            None,
            "skip_mtd_retain_sft",
            "record_not_route_eligible",
        )
    if record.record_type is RecordType.ATOMIC:
        return RouteDecision(
            RouteState.ATOMIC_DECLARED.value,
            stage_for_form(record.declared_form or ""),
            "resolve_rollout_span",
            "atomic_declared_family",
        )
    evaluations = evaluations or {}
    for stage in record.topological_order():
        evaluation = evaluations.get(stage)
        if evaluation is None:
            return _decision_for_state(
                RouteState.EVALUATION_UNAVAILABLE.value,
                stage,
                "missing_stage_evaluation",
            )
        if evaluation.stage is not stage:
            raise ContractError(f"evaluation key {stage.value} disagrees with payload stage")
        state = evaluation.state()
        if state != "pass":
            return _decision_for_state(state, stage, f"first_nonpassing_stage:{stage.value}")
    return RouteDecision(
        RouteState.ALL_PASS.value,
        None,
        "skip_mtd_retain_sft",
        "all_active_stages_pass",
    )


def allocate_mtd(
    decision: RouteDecision,
    sampled_span: Optional[RolloutTokenSpan],
    rollout_token_count: int,
    *,
    teacher_qualified: Optional[bool],
) -> RouteDecision:
    """Require qualification and a validated, nonempty sampled token span.

    A reference character span cannot be used here. Missing model delimiters
    require an explicit structural-failure evaluation and a parser-provided
    suffix span; a missing evaluator result never creates that evidence.
    The qualification flag is supplied by external, hash-bound qualification
    integration. This flag check does not establish the underlying evidence.
    """
    if teacher_qualified is not None and type(teacher_qualified) is not bool:
        raise ContractError("teacher_qualified must be a boolean or None")
    if decision.mtd_action not in ("resolve_rollout_span", "apply_mtd"):
        return decision
    if decision.selected_stage is None:
        raise ContractError("a pending MTD allocation requires a selected stage")
    if teacher_qualified is not True:
        reason = "teacher_not_qualified" if teacher_qualified is False else "teacher_qualification_unavailable"
        return replace(decision, mtd_action="skip_mtd_retain_sft", reason=reason)
    if sampled_span is None:
        return replace(decision, mtd_action="skip_mtd_retain_sft", reason="missing_rollout_span")
    if not isinstance(sampled_span, RolloutTokenSpan):
        raise ContractError("MTD requires sampled token offsets, not reference character offsets")
    sampled_span.validate(rollout_token_count)
    if sampled_span.start == sampled_span.end:
        return replace(decision, mtd_action="skip_mtd_retain_sft", reason="empty_rollout_span")
    return replace(decision, mtd_action="apply_mtd")


def replay_signature(decision: RouteDecision) -> Tuple[str, Optional[str], str, str]:
    """Return a stable tuple suitable for deterministic replay tests and logs."""

    return (
        decision.state,
        decision.selected_stage.value if decision.selected_stage else None,
        decision.mtd_action,
        decision.reason,
    )
