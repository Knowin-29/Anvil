"""Pure FirstFault routing shared by evidence scoring and training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real


class RouteKind(str, Enum):
    """Publication and training-facing FirstFault decisions."""

    ATTRIBUTED = "Attributed"
    LOCAL = "Local"
    INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True)
class FirstFaultRoute:
    """A fail-closed route decision and its machine-readable reason."""

    kind: RouteKind
    stage: str | None
    reason: str

    @property
    def eligible(self) -> bool:
        return self.kind in {RouteKind.ATTRIBUTED, RouteKind.LOCAL}


def _inconclusive(reason: str) -> FirstFaultRoute:
    return FirstFaultRoute(RouteKind.INCONCLUSIVE, None, reason)


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact_bool_mapping(
    value: object, expected: Sequence[str]
) -> dict[str, bool] | None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        return None
    if any(type(value[key]) is not bool for key in expected):
        return None
    return {key: value[key] for key in expected}


def _exact_score_mapping(
    value: object, expected: Sequence[str]
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        return None
    parsed = {key: _finite_score(value[key]) for key in expected}
    if any(score is None for score in parsed.values()):
        return None
    return {key: float(parsed[key]) for key in expected}


def _exact_identity_mapping(
    value: object, expected: Sequence[str]
) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        return None
    if any(not _is_sha256(value[key]) for key in expected):
        return None
    return {key: str(value[key]) for key in expected}


def response_scores_consistent(
    scores: Mapping[str, float], response_identities: Mapping[str, str]
) -> bool:
    """Return whether identical responses have identical bound semantic scores."""

    if set(scores) != set(response_identities):
        return False
    seen: dict[str, float] = {}
    for key in scores:
        score = _finite_score(scores[key])
        identity = response_identities[key]
        if score is None or not _is_sha256(identity):
            return False
        previous = seen.setdefault(identity, score)
        if previous != score:
            return False
    return True


def confirm_state_use(
    *,
    repaired_target_score: float,
    absent_target_score: float,
    corrupted_target_score: float,
    minimum_effect: float,
    previous_response_identity: str,
    repaired_response_identity: str,
) -> bool:
    """Apply the frozen control threshold and require an observable repair change."""

    repaired = _finite_score(repaired_target_score)
    absent = _finite_score(absent_target_score)
    corrupted = _finite_score(corrupted_target_score)
    minimum = _finite_score(minimum_effect)
    if (
        repaired is None
        or absent is None
        or corrupted is None
        or minimum is None
        or not _is_sha256(previous_response_identity)
        or not _is_sha256(repaired_response_identity)
    ):
        return False
    sensitivity = repaired - 0.5 * (absent + corrupted)
    corrupted_effect = repaired - corrupted
    return (
        repaired_response_identity != previous_response_identity
        and sensitivity > minimum
        and corrupted_effect > minimum
    )


def route_first_fault(
    *,
    stage_order: Sequence[str],
    local_correct: Mapping[str, bool],
    predicted_target_score: float,
    repair_target_scores: Mapping[str, float],
    predicted_target_response_identity: str,
    repair_target_response_identities: Mapping[str, str],
    state_use_confirmed: Mapping[str, bool],
    probes_valid: bool,
    hit_token_cap: bool,
    correct_threshold: float,
    monotonic_tolerance: float,
) -> FirstFaultRoute:
    """Route one failed downstream chain under the strict three-state rule."""

    if isinstance(stage_order, (str, bytes)):
        return _inconclusive("unsupported_or_invalid_stage_order")
    stages = tuple(stage_order)
    if stages not in {("G", "S"), ("G", "S", "A")}:
        return _inconclusive("unsupported_or_invalid_stage_order")
    if type(probes_valid) is not bool or not probes_valid:
        return _inconclusive("invalid_or_incomplete_probes")
    if type(hit_token_cap) is not bool:
        return _inconclusive("invalid_probe_health_metadata")
    if hit_token_cap:
        return _inconclusive("capped_probe")

    threshold = _finite_score(correct_threshold)
    tolerance = _finite_score(monotonic_tolerance)
    if threshold is None or threshold <= 0.0 or tolerance is None:
        return _inconclusive("invalid_threshold_configuration")

    upstream = stages[:-1]
    target = stages[-1]
    local = _exact_bool_mapping(local_correct, stages)
    repairs = _exact_score_mapping(repair_target_scores, upstream)
    repair_identities = _exact_identity_mapping(
        repair_target_response_identities, upstream
    )
    state_use = _exact_bool_mapping(state_use_confirmed, upstream)
    predicted = _finite_score(predicted_target_score)
    if (
        local is None
        or repairs is None
        or repair_identities is None
        or state_use is None
        or predicted is None
        or not _is_sha256(predicted_target_response_identity)
    ):
        return _inconclusive("invalid_or_incomplete_evidence")

    trajectory_scores = {
        "predicted": predicted,
        **{stage: repairs[stage] for stage in upstream},
    }
    trajectory_identities = {
        "predicted": predicted_target_response_identity,
        **{stage: repair_identities[stage] for stage in upstream},
    }
    if not response_scores_consistent(trajectory_scores, trajectory_identities):
        return _inconclusive("response_score_inconsistency")

    trajectory = (predicted, *(repairs[stage] for stage in upstream))
    if any(
        later + tolerance < earlier
        for earlier, later in zip(trajectory, trajectory[1:])
    ):
        return _inconclusive("oracle_degradation")

    predicted_ok = predicted >= threshold
    oracle_ok = repairs[upstream[-1]] >= threshold
    if local[target] != oracle_ok:
        return _inconclusive("oracle_target_and_local_label_disagree")
    if predicted_ok:
        return _inconclusive("predicted_target_correct")

    upstream_faults = [stage for stage in upstream if not local[stage]]
    if len(upstream_faults) > 1:
        return _inconclusive("multiple_unresolved_faults")
    if not upstream_faults:
        if not oracle_ok:
            return FirstFaultRoute(
                RouteKind.LOCAL, target, "target_fails_under_oracle_context"
            )
        return _inconclusive("unisolated_failure")

    candidate = upstream_faults[0]
    if not oracle_ok:
        return _inconclusive("multiple_unresolved_faults")
    if not state_use[candidate]:
        return _inconclusive("context_insensitive_repair")

    candidate_index = upstream.index(candidate)
    if any(repairs[stage] >= threshold for stage in upstream[:candidate_index]):
        return _inconclusive("repair_precedes_local_fault")
    if repairs[candidate] < threshold:
        return _inconclusive("candidate_repair_does_not_fix_target")
    previous_identity = (
        predicted_target_response_identity
        if candidate_index == 0
        else repair_identities[upstream[candidate_index - 1]]
    )
    if repair_identities[candidate] == previous_identity:
        return _inconclusive("repair_did_not_change_target_response")
    return FirstFaultRoute(
        RouteKind.ATTRIBUTED, candidate, "monotonic_upstream_repair"
    )


__all__ = [
    "FirstFaultRoute",
    "RouteKind",
    "confirm_state_use",
    "response_scores_consistent",
    "route_first_fault",
]
