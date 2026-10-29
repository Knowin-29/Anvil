"""Routing and loss utilities for GSA multi-teacher distillation."""

from .firstfault_core import (
    FirstFaultRoute,
    RouteKind,
    advantage_group_ids,
    anvil_selected_span_mtd_loss,
    anvil_stage_sft_loss,
    confirm_state_use,
    dependency_advantage,
    grouped_generalized_jsd_loss,
    grouped_reverse_kl_loss,
    grouped_token_mean,
    route_first_fault,
    response_scores_consistent,
    token_generalized_jsd,
    token_reverse_kl,
    within_stage_weights,
)
from .routing import GSA_FAMILIES, FamilyBalancedBatchSampler, dataset_families

__all__ = [
    "FirstFaultRoute",
    "GSA_FAMILIES",
    "FamilyBalancedBatchSampler",
    "RouteKind",
    "advantage_group_ids",
    "anvil_selected_span_mtd_loss",
    "anvil_stage_sft_loss",
    "confirm_state_use",
    "dataset_families",
    "dependency_advantage",
    "grouped_generalized_jsd_loss",
    "grouped_reverse_kl_loss",
    "grouped_token_mean",
    "route_first_fault",
    "response_scores_consistent",
    "token_generalized_jsd",
    "token_reverse_kl",
    "within_stage_weights",
]
