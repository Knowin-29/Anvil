import unittest
from dataclasses import replace

from contract_v2 import (
    ContractError,
    ContractRecord,
    DependencyEdge,
    RecordType,
    RolloutTokenSpan,
    RouteState,
    Stage,
    StageEvaluation,
    StageSpan,
    allocate_mtd,
    replay_signature,
    route,
)


TEXT = "<GROUND>g</GROUND><RELATE>r</RELATE><ACT>a</ACT>"


def linked_record(*, edges=None, route_eligible=True):
    return ContractRecord(
        record_type=RecordType.LINKED,
        canonical_uid="scene-001",
        image_hash="sha256:image",
        task_id="task-001",
        entity_namespace="entity-001",
        serialized_text=TEXT,
        active_stages=(Stage.G, Stage.S, Stage.A),
        stage_forms={Stage.G: "bbox", Stage.S: "relative_depth", Stage.A: "action_plan"},
        stage_spans={
            Stage.G: StageSpan(0, TEXT.index("<RELATE>")),
            Stage.S: StageSpan(TEXT.index("<RELATE>"), TEXT.index("<ACT>")),
            Stage.A: StageSpan(TEXT.index("<ACT>"), len(TEXT)),
        },
        edges=tuple(
            edges
            if edges is not None
            else (
                DependencyEdge(Stage.G, Stage.S, "entity_id"),
                DependencyEdge(Stage.S, Stage.A, "relative_depth"),
            )
        ),
        route_eligible=route_eligible,
    )


def evaluation(stage, score, *, structural_valid=True, tau=0.5, delta=0.05):
    return StageEvaluation(stage, structural_valid, score, tau, delta)


class ContractV2Tests(unittest.TestCase):
    def test_atomic_route_is_declared_and_content_independent(self):
        record = ContractRecord(
            record_type=RecordType.ATOMIC,
            canonical_uid="atomic-001",
            image_hash="sha256:image",
            task_id="task-atomic",
            entity_namespace="entity-001",
            serialized_text="<GROUND>g</GROUND>",
            active_stages=(Stage.G,),
            stage_forms={Stage.G: "point"},
            declared_form="point",
        )
        first = route(record, {Stage.G: evaluation(Stage.G, 0.0)})
        second = route(record, {Stage.G: evaluation(Stage.G, 1.0)})
        self.assertEqual(first.state, RouteState.ATOMIC_DECLARED.value)
        self.assertEqual(first.selected_stage, Stage.G)
        self.assertEqual(replay_signature(first), replay_signature(second))

    def test_linked_spans_must_not_overlap(self):
        record = linked_record()
        bad = dict(record.stage_spans)
        bad[Stage.S] = StageSpan(15, 31)
        with self.assertRaises(ContractError):
            replace(record, stage_spans=bad).validate()

    def test_upstream_structural_failure_short_circuits(self):
        decision = route(
            linked_record(),
            {
                Stage.G: evaluation(Stage.G, 1.0, structural_valid=False),
                Stage.S: evaluation(Stage.S, 0.0),
                Stage.A: evaluation(Stage.A, 1.0),
            },
        )
        self.assertEqual(decision.state, RouteState.STRUCTURAL_INVALID.value)
        self.assertEqual(decision.selected_stage, Stage.G)
        self.assertEqual(decision.mtd_action, "resolve_rollout_span")

    def test_semantic_first_fault_selects_relation(self):
        decision = route(
            linked_record(),
            {
                Stage.G: evaluation(Stage.G, 0.9),
                Stage.S: evaluation(Stage.S, 0.2),
                Stage.A: evaluation(Stage.A, 0.0),
            },
        )
        self.assertEqual(decision.state, RouteState.SEMANTIC_FAILURE.value)
        self.assertEqual(decision.selected_stage, Stage.S)

    def test_indeterminate_upstream_skips_mtd_and_blocks_downstream(self):
        decision = route(
            linked_record(),
            {
                Stage.G: evaluation(Stage.G, 0.52),
                Stage.S: evaluation(Stage.S, 0.0),
                Stage.A: evaluation(Stage.A, 1.0),
            },
        )
        self.assertEqual(decision.state, RouteState.INDETERMINATE_LOW_CONFIDENCE.value)
        self.assertIsNone(decision.selected_stage)
        self.assertEqual(decision.mtd_action, "skip_mtd_retain_sft")

    def test_all_pass_skips_mtd(self):
        decision = route(
            linked_record(),
            {stage: evaluation(stage, 0.9) for stage in (Stage.G, Stage.S, Stage.A)},
        )
        self.assertEqual(decision.state, RouteState.ALL_PASS.value)
        self.assertIsNone(decision.selected_stage)

    def test_edges_need_consumed_fields_and_acyclic_graph(self):
        with self.assertRaises(ContractError):
            linked_record(
                edges=(DependencyEdge(Stage.G, Stage.S, ""),)
            ).validate()
        with self.assertRaises(ContractError):
            linked_record(
                edges=(
                    DependencyEdge(Stage.G, Stage.S, "entity_id"),
                    DependencyEdge(Stage.S, Stage.G, "depth"),
                )
            ).validate()

    def test_missing_stage_evaluation_abstains_without_blame(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.9)})
        self.assertEqual(decision.state, RouteState.EVALUATION_UNAVAILABLE.value)
        self.assertIsNone(decision.selected_stage)
        self.assertEqual(decision.mtd_action, "skip_mtd_retain_sft")

    def test_non_eligible_linked_trace_is_explicit_abstention(self):
        decision = route(linked_record(route_eligible=False), {})
        self.assertEqual(decision.state, RouteState.INDETERMINATE_LOW_CONFIDENCE.value)
        self.assertEqual(decision.reason, "record_not_route_eligible")
        self.assertIsNone(decision.selected_stage)

    def test_reference_character_spans_cover_complete_stages(self):
        record = linked_record()
        for stage, tag in ((Stage.G, "GROUND"), (Stage.S, "RELATE"), (Stage.A, "ACT")):
            span = record.stage_spans[stage]
            self.assertEqual(TEXT[span.start:span.end], f"<{tag}>{stage.value.lower() if stage != Stage.S else 'r'}</{tag}>")

    def test_invalid_configuration_is_rejected_even_without_a_score(self):
        for structural_valid in (False, True):
            for tau, delta in ((float("nan"), .05), (1.1, .05), (.5, -.1), (.5, float("inf"))):
                with self.subTest(structural_valid=structural_valid, tau=tau, delta=delta):
                    with self.assertRaises(ContractError):
                        evaluation(Stage.G, None, structural_valid=structural_valid, tau=tau, delta=delta).state()

    def test_scores_are_normalized_finite_numbers(self):
        for score in (-.1, 1.1, float("nan"), float("inf"), True, "0.9"):
            with self.subTest(score=score):
                with self.assertRaises(ContractError):
                    evaluation(Stage.G, score).state()

    def test_semantic_boundary_conditions(self):
        self.assertEqual(evaluation(Stage.G, .45).state(), RouteState.INDETERMINATE_LOW_CONFIDENCE.value)
        self.assertEqual(evaluation(Stage.G, .55).state(), "pass")
        self.assertEqual(evaluation(Stage.G, .5, delta=0).state(), "pass")
        self.assertEqual(evaluation(Stage.G, None).state(), RouteState.EVALUATION_UNAVAILABLE.value)

    def test_mtd_allocation_requires_sampled_nonempty_token_span(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)})
        self.assertEqual(allocate_mtd(decision, RolloutTokenSpan(2, 6), 9, teacher_qualified=True).mtd_action, "apply_mtd")
        for span in (None, RolloutTokenSpan(2, 2)):
            with self.subTest(span=span):
                self.assertEqual(allocate_mtd(decision, span, 9, teacher_qualified=True).mtd_action, "skip_mtd_retain_sft")
        for span in (StageSpan(0, 3), RolloutTokenSpan(-1, 3), RolloutTokenSpan(2, 10), RolloutTokenSpan(4, 2)):
            with self.subTest(span=span):
                with self.assertRaises(ContractError):
                    allocate_mtd(decision, span, 9, teacher_qualified=True)

    def test_teacher_qualification_is_a_required_keyword_only_input(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)})
        with self.assertRaises(TypeError):
            allocate_mtd(decision, RolloutTokenSpan(0, 3), 3)
        with self.assertRaises(TypeError):
            allocate_mtd(decision, RolloutTokenSpan(0, 3), 3, True)

    def test_unqualified_and_unavailable_teachers_have_distinct_skip_reasons(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)})
        for qualified, reason in (
            (False, "teacher_not_qualified"),
            (None, "teacher_qualification_unavailable"),
        ):
            with self.subTest(qualified=qualified):
                allocation = allocate_mtd(
                    decision, RolloutTokenSpan(0, 3), 3, teacher_qualified=qualified
                )
                self.assertEqual(allocation.mtd_action, "skip_mtd_retain_sft")
                self.assertEqual(allocation.reason, reason)
                self.assertEqual(allocation.selected_stage, decision.selected_stage)
                self.assertEqual(allocation.state, decision.state)

    def test_invalid_qualification_flag_is_rejected_even_for_abstention(self):
        decisions = (
            route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)}),
            route(linked_record(), {}),
        )
        for decision in decisions:
            for qualified in (0, 1, "true", [], 0.0):
                with self.subTest(action=decision.mtd_action, qualified=qualified):
                    with self.assertRaises(ContractError):
                        allocate_mtd(
                            decision, RolloutTokenSpan(0, 3), 3, teacher_qualified=qualified
                        )

    def test_finalized_allocation_cannot_bypass_qualification(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)})
        allocation = allocate_mtd(
            decision, RolloutTokenSpan(0, 3), 3, teacher_qualified=True
        )
        for qualified in (False, None):
            with self.subTest(qualified=qualified):
                self.assertEqual(
                    allocate_mtd(
                        allocation, RolloutTokenSpan(0, 3), 3, teacher_qualified=qualified
                    ).mtd_action,
                    "skip_mtd_retain_sft",
                )

    def test_abstention_cannot_be_promoted_by_providing_a_span(self):
        decision = route(linked_record(), {})
        self.assertEqual(allocate_mtd(decision, RolloutTokenSpan(0, 3), 3, teacher_qualified=True), decision)

    def test_finalized_allocation_revalidates_the_current_span(self):
        decision = route(linked_record(), {Stage.G: evaluation(Stage.G, 0.1)})
        allocated = allocate_mtd(decision, RolloutTokenSpan(0, 3), 3, teacher_qualified=True)
        self.assertEqual(
            allocate_mtd(allocated, None, 3, teacher_qualified=True).mtd_action,
            "skip_mtd_retain_sft",
        )
        with self.assertRaises(ContractError):
            allocate_mtd(allocated, RolloutTokenSpan(0, 4), 3, teacher_qualified=True)

    def test_atomic_record_respects_eligibility_and_sampled_span(self):
        record = ContractRecord(RecordType.ATOMIC, "uid", "hash", "task", "entities", "",
                                (Stage.G,), {Stage.G: "point"}, declared_form="point")
        self.assertEqual(allocate_mtd(route(record), RolloutTokenSpan(0, 0), 0, teacher_qualified=True).mtd_action, "skip_mtd_retain_sft")
        self.assertEqual(route(replace(record, route_eligible=False)).mtd_action, "skip_mtd_retain_sft")


if __name__ == "__main__":
    unittest.main()
