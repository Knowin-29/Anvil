import json
import os
import tempfile

from apps.data_flywheel.ontology.schema import (
    BaseAffordance,
    ContainerProfile,
    GovernanceMetadata,
    ObjectClass,
    ObjectSubtype,
)
from apps.data_flywheel.ontology.store import OntologyStore
from apps.data_flywheel.review.observation import CandidateCluster
from apps.data_flywheel.review.promotion import PromotionAction, apply_review_action
from apps.data_flywheel.review.workbench import render_store_markdown


def test_create_class_uses_new_affordance_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OntologyStore()
        store_path = os.path.join(tmpdir, "store.json")
        store.save(store_path)
        clusters_path = os.path.join(tmpdir, "clusters.jsonl")
        cluster = CandidateCluster(
            cluster_id="c1",
            representative_label="杯子",
            normalized_label="杯子",
            labels={"杯子": 2},
            observation_ids=["o1"],
            sample_images=["a.jpg"],
            sample_crops=[],
            count=2,
            avg_confidence=0.9,
        )
        with open(clusters_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(cluster.to_dict(), ensure_ascii=False) + "\n")
        audit_path = os.path.join(tmpdir, "audit.jsonl")

        apply_review_action(
            store_path,
            clusters_path,
            audit_path,
            PromotionAction(cluster_id="c1", action_type="create_class", new_name="杯子"),
        )

        loaded = OntologyStore.load(store_path)
        cup = loaded.get_class("杯子")
        assert cup is not None
        assert cup.affordance.operation is None
        assert not hasattr(cup.affordance, "can_pick")


def test_render_store_markdown_shows_subtype_counts_and_placement_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OntologyStore()
        store.add_class(
            ObjectClass(
                class_id="pen_holder",
                name="笔筒",
                affordance=BaseAffordance(
                    operation="non_operable",
                    is_container=True,
                    container_placeability="placeable",
                ),
                container_profile=ContainerProfile(container_id=3),
                placement_policy="leave_in_place",
                metadata=GovernanceMetadata(status="approved"),
                alias_auto_approve=True,
            )
        )
        store.add_subtype(
            ObjectSubtype(
                subtype_id="desk_pen_holder",
                class_id="pen_holder",
                name="桌面笔筒",
            )
        )
        store_path = os.path.join(tmpdir, "store.json")
        store.save(store_path)

        markdown = render_store_markdown(store_path)

        assert "笔筒" in markdown
        assert "leave_in_place" in markdown
        assert "桌面笔筒" in markdown
