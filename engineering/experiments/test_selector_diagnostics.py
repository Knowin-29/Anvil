import unittest
from dataclasses import replace

from contract_v2 import (
    ContractError,
    ContractRecord,
    DependencyEdge,
    RecordType,
    RouteState,
    Stage,
    StageEvaluation,
    StageSpan,
)
from selector_diagnostics import (
    CENSORED,
    SelectorRule,
    build_diagnostic_ledger,
    select,
)


def linked_record(edges=None):
    text = "<GROUND>g</GROUND><RELATE>r</RELATE><ACT>a</ACT>"
    return ContractRecord(
        record_type=RecordType.LINKED,
        canonical_uid="synthetic-selector-fixture",
        image_hash="synthetic-image",
        task_id="synthetic-task",
        entity_namespace="synthetic-entities",
        serialized_text=text,
        active_stages=(Stage.G, Stage.S, Stage.A),
        stage_forms={Stage.G: "bbox", Stage.S: "relative_depth", Stage.A: "action_plan"},
        stage_spans={
            Stage.G: StageSpan(0, 18),
            Stage.S: StageSpan(18, 36),
            Stage.A: StageSpan(36, len(text)),
        },
        edges=tuple(edges) if edges is not None else (
            DependencyEdge(Stage.G, Stage.S, "entity_id"),
            DependencyEdge(Stage.S, Stage.A, "relative_depth"),
        ),
    )


def evaluation(stage, score, *, valid=True, tau=0.5, delta=0.05):
    return StageEvaluation(stage, valid, score, tau, delta)


class SelectorDiagnosticsTests(unittest.TestCase):
    def test_two_faults_make_shared_selectors_non_degenerate(self):
        ledger = build_diagnostic_ledger(linked_record(), {
            Stage.G: evaluation(Stage.G, 0.9),
            Stage.S: evaluation(Stage.S, 0.3),
            Stage.A: evaluation(Stage.A, 0.1),
        })
        self.assertTrue(ledger.comparison_eligible)
        self.assertEqual(ledger.production_decision.selected_stage, Stage.S)
        self.assertEqual(ledger.production_decision.mtd_action, "resolve_rollout_span")
        expected = {
            SelectorRule.FIRST_SEMANTIC_FAULT: (Stage.S,),
            SelectorRule.LAST_SEMANTIC_FAULT: (Stage.A,),
            SelectorRule.LARGEST_NORMALIZED_DEFICIT: (Stage.A,),
            SelectorRule.ALL_FAILED_STAGE_MIXTURE: (Stage.S, Stage.A),
        }
        for rule, stages in expected.items():
            with self.subTest(rule=rule):
                decision = select(ledger, rule)
                self.assertEqual(decision.selected_stages, stages)
                self.assertAlmostEqual(sum(decision.stage_weights), 1.0)
        self.assertEqual(
            select(ledger, SelectorRule.ALL_FAILED_STAGE_MIXTURE).stage_weights,
            (0.5, 0.5),
        )

    def test_normalized_domain_deficit_does_not_divide_by_threshold(self):
        ledger = build_diagnostic_ledger(linked_record(), {
            Stage.G: evaluation(Stage.G, 0.9),
            Stage.S: evaluation(Stage.S, 0.2, tau=0.8),
            Stage.A: evaluation(Stage.A, 0.0, tau=0.4),
        })
        self.assertAlmostEqual(ledger.semantic_failures[0].normalized_deficit, 0.6)
        self.assertAlmostEqual(ledger.semantic_failures[1].normalized_deficit, 0.4)
        self.assertEqual(
            select(ledger, SelectorRule.LARGEST_NORMALIZED_DEFICIT).selected_stages,
            (Stage.S,),
        )

    def test_equal_deficits_use_topological_first_tie_break(self):
        ledger = build_diagnostic_ledger(linked_record(), {
            Stage.G: evaluation(Stage.G, 0.9),
            Stage.S: evaluation(Stage.S, 0.1),
            Stage.A: evaluation(Stage.A, 0.1),
        })
        self.assertEqual(
            select(ledger, SelectorRule.LARGEST_NORMALIZED_DEFICIT).selected_stages,
            (Stage.S,),
        )

    def test_structural_invalid_child_censors_descendants_before_reading_score(self):
        class GuardedEvaluations(dict):
            def get(self, key, default=None):
                if key is Stage.A:
                    raise AssertionError("censored descendant score was read")
                return super().get(key, default)

        ledger = build_diagnostic_ledger(linked_record(), GuardedEvaluations({
            Stage.G: evaluation(Stage.G, 0.9),
            Stage.S: evaluation(Stage.S, None, valid=False),
        }))
        self.assertEqual(ledger.entries[1].state, RouteState.STRUCTURAL_INVALID.value)
        self.assertEqual(ledger.entries[2].state, CENSORED)
        self.assertEqual(ledger.entries[2].censored_by, (Stage.S,))
        self.assertFalse(ledger.comparison_eligible)
        self.assertEqual(ledger.production_decision.selected_stage, Stage.S)
        for rule in SelectorRule:
            self.assertEqual(select(ledger, rule).selected_stages, ())

    def test_absent_and_indeterminate_predecessors_censor_transitively(self):
        for first in (None, evaluation(Stage.G, None), evaluation(Stage.G, 0.5)):
            with self.subTest(first=first):
                evaluations = {
                    Stage.S: evaluation(Stage.S, 0.1),
                    Stage.A: evaluation(Stage.A, 0.1),
                }
                if first is not None:
                    evaluations[Stage.G] = first
                ledger = build_diagnostic_ledger(linked_record(), evaluations)
                self.assertEqual(ledger.entries[1].state, CENSORED)
                self.assertEqual(ledger.entries[2].state, CENSORED)
                self.assertIsNone(ledger.entries[2].semantic_score)
                self.assertFalse(ledger.comparison_eligible)

    def test_independent_branch_remains_visible_but_cannot_change_population(self):
        ledger = build_diagnostic_ledger(
            linked_record(edges=(DependencyEdge(Stage.G, Stage.S, "entity_id"),)),
            {
                Stage.G: evaluation(Stage.G, 0.5),
                Stage.S: evaluation(Stage.S, 0.0),
                Stage.A: evaluation(Stage.A, 0.0),
            },
        )
        self.assertEqual(ledger.entries[1].state, CENSORED)
        self.assertEqual(ledger.entries[2].state, RouteState.SEMANTIC_FAILURE.value)
        self.assertIsNone(ledger.production_decision.selected_stage)
        self.assertFalse(ledger.comparison_eligible)
        self.assertEqual(select(ledger, SelectorRule.FIRST_SEMANTIC_FAULT).selected_stages, ())

    def test_join_requires_every_predecessor_to_be_decisive_and_valid(self):
        ledger = build_diagnostic_ledger(
            linked_record(edges=(
                DependencyEdge(Stage.G, Stage.A, "entity_id"),
                DependencyEdge(Stage.S, Stage.A, "relative_depth"),
            )),
            {
                Stage.G: evaluation(Stage.G, 0.1),
                Stage.S: evaluation(Stage.S, None),
                Stage.A: evaluation(Stage.A, 0.0),
            },
        )
        self.assertEqual(ledger.entries[2].state, CENSORED)
        self.assertEqual(ledger.entries[2].censored_by, (Stage.S,))
        self.assertFalse(ledger.comparison_eligible)

    def test_all_pass_and_ineligible_records_produce_no_diagnostic_selection(self):
        for scores, eligible in (((0.9, 0.9, 0.9), True), ((0.1, 0.1, 0.1), False)):
            with self.subTest(scores=scores, eligible=eligible):
                record = replace(linked_record(), route_eligible=eligible)
                ledger = build_diagnostic_ledger(record, {
                    stage: evaluation(stage, score)
                    for stage, score in zip(record.active_stages, scores)
                })
                self.assertFalse(ledger.comparison_eligible)
                for rule in SelectorRule:
                    decision = select(ledger, rule)
                    self.assertEqual(decision.selected_stages, ())
                    self.assertEqual(decision.stage_weights, ())

    def test_readable_invalid_scores_and_stage_identity_are_rejected(self):
        for supplied in (evaluation(Stage.G, 2.0), evaluation(Stage.S, 0.1)):
            with self.subTest(supplied=supplied):
                with self.assertRaises(ContractError):
                    build_diagnostic_ledger(linked_record(), {Stage.G: supplied})

    def test_zero_threshold_is_valid_and_never_requires_division(self):
        ledger = build_diagnostic_ledger(linked_record(), {
            Stage.G: evaluation(Stage.G, 0.0, tau=0.0, delta=0.0),
            Stage.S: evaluation(Stage.S, 0.1),
            Stage.A: evaluation(Stage.A, 0.9),
        })
        self.assertEqual(ledger.entries[0].state, "pass")
        self.assertEqual(
            select(ledger, SelectorRule.LARGEST_NORMALIZED_DEFICIT).selected_stages,
            (Stage.S,),
        )

    def test_atomic_records_are_not_a_dynamic_selector_diagnostic(self):
        record = ContractRecord(
            RecordType.ATOMIC, "uid", "hash", "task", "entities", "point",
            (Stage.G,), {Stage.G: "point"}, declared_form="point",
        )
        with self.assertRaises(ContractError):
            build_diagnostic_ledger(record, {})


if __name__ == "__main__":
    unittest.main()
