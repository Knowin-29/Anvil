"""Protocol-only selector diagnostics over a shared sampled-context ledger.

Unlike the production route, diagnostics may inspect a descendant after a
semantic failure. They never substitute reference upstream context. Supplied
scores must describe structurally interpretable outputs in the original sampled
context; this module does not parse, score, train, or establish causal blame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from contract_v2 import (
    ContractError,
    ContractRecord,
    RecordType,
    RouteDecision,
    RouteState,
    Stage,
    StageEvaluation,
    route,
)


CENSORED = "censored"
DECISIVE_STATES = ("pass", RouteState.SEMANTIC_FAILURE.value)


class SelectorRule(str, Enum):
    FIRST_SEMANTIC_FAULT = "first_semantic_fault"
    LAST_SEMANTIC_FAULT = "last_semantic_fault"
    LARGEST_NORMALIZED_DEFICIT = "largest_normalized_deficit"
    ALL_FAILED_STAGE_MIXTURE = "all_failed_stage_mixture"


@dataclass(frozen=True)
class LedgerEntry:
    stage: Stage
    state: str
    semantic_score: Optional[float]
    normalized_deficit: Optional[float]
    censored_by: Tuple[Stage, ...] = ()


@dataclass(frozen=True)
class DiagnosticLedger:
    canonical_uid: str
    entries: Tuple[LedgerEntry, ...]
    production_decision: RouteDecision
    record_route_eligible: bool

    @property
    def comparison_exclusion_reason(self) -> Optional[str]:
        if not self.record_route_eligible:
            return "record_not_route_eligible"
        if any(entry.state not in DECISIVE_STATES for entry in self.entries):
            return "not_all_stages_decisive_and_structurally_valid"
        if self.production_decision.state != RouteState.SEMANTIC_FAILURE.value:
            return "no_production_semantic_fault"
        return None

    @property
    def comparison_eligible(self) -> bool:
        return self.comparison_exclusion_reason is None

    @property
    def semantic_failures(self) -> Tuple[LedgerEntry, ...]:
        return tuple(
            entry for entry in self.entries
            if entry.state == RouteState.SEMANTIC_FAILURE.value
        )


@dataclass(frozen=True)
class SelectorDecision:
    rule: SelectorRule
    selected_stages: Tuple[Stage, ...]
    stage_weights: Tuple[float, ...]
    reason: str


def build_diagnostic_ledger(
    record: ContractRecord,
    evaluations: Mapping[Stage, StageEvaluation] | None = None,
) -> DiagnosticLedger:
    """Read stage evaluations under declared dependencies and retain censorship.

    Pass and decisive semantic failure permit diagnostic descendant inspection.
    Structural invalidity, uncertainty, absent scores, and prior censorship do
    not. Independent branches are still inspected. The unchanged production
    decision is retained separately, and only fully decisive, structurally valid
    linked traces with a production semantic fault enter selector comparisons.
    """

    record.validate()
    if record.record_type is not RecordType.LINKED:
        raise ContractError("selector diagnostics require a linked record")
    evaluations = evaluations if evaluations is not None else {}
    if set(evaluations) - set(record.active_stages):
        raise ContractError("diagnostic evaluations contain an inactive stage")

    entries = []
    by_stage = {}
    for stage in record.topological_order():
        # Censor before reading a score that requires unresolved sampled context.
        blocked = tuple(
            predecessor for predecessor in record.predecessors(stage)
            if by_stage[predecessor].state not in DECISIVE_STATES
        )
        if blocked:
            entry = LedgerEntry(stage, CENSORED, None, None, blocked)
        else:
            evaluation = evaluations.get(stage)
            if evaluation is None:
                entry = LedgerEntry(
                    stage, RouteState.EVALUATION_UNAVAILABLE.value, None, None
                )
            else:
                if evaluation.stage is not stage:
                    raise ContractError(
                        f"evaluation key {stage.value} disagrees with payload stage"
                    )
                state = evaluation.state()
                score = evaluation.semantic_score if evaluation.structural_valid else None
                # Scores already share [0, 1]; no division by the threshold occurs.
                deficit = (
                    evaluation.tau_route - score
                    if state == RouteState.SEMANTIC_FAILURE.value else None
                )
                entry = LedgerEntry(stage, state, score, deficit)
        entries.append(entry)
        by_stage[stage] = entry

    return DiagnosticLedger(
        record.canonical_uid,
        tuple(entries),
        route(record, evaluations),
        record.route_eligible,
    )


def select(ledger: DiagnosticLedger, rule: SelectorRule) -> SelectorDecision:
    """Select from the identical decisive population; return no training action.

    Largest deficit is tau_route minus score on the normalized [0, 1] score
    domain, with the first topological stage winning a tie. The mixture uses
    equal stage weights; token allocation and teacher qualification are outside
    this diagnostic module and must not be inferred from these weights.
    """

    if not isinstance(rule, SelectorRule):
        raise ContractError("selector rule must be a SelectorRule")
    exclusion_reason = ledger.comparison_exclusion_reason
    if exclusion_reason is not None:
        return SelectorDecision(rule, (), (), exclusion_reason)
    failures = ledger.semantic_failures
    if not failures:
        raise ContractError("a comparison-eligible ledger requires a semantic failure")

    if rule is SelectorRule.FIRST_SEMANTIC_FAULT:
        chosen = (failures[0],)
    elif rule is SelectorRule.LAST_SEMANTIC_FAULT:
        chosen = (failures[-1],)
    elif rule is SelectorRule.LARGEST_NORMALIZED_DEFICIT:
        chosen = (max(failures, key=lambda entry: entry.normalized_deficit),)
    else:
        chosen = failures
    return SelectorDecision(
        rule,
        tuple(entry.stage for entry in chosen),
        (1.0 / len(chosen),) * len(chosen),
        "shared_decisive_sampled_context_ledger",
    )
