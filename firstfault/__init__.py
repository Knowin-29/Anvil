"""Dependency-free FirstFault routing contract."""

from .routing import (
    FirstFaultRoute,
    RouteKind,
    confirm_state_use,
    response_scores_consistent,
    route_first_fault,
)

__all__ = [
    "FirstFaultRoute",
    "RouteKind",
    "confirm_state_use",
    "response_scores_consistent",
    "route_first_fault",
]
