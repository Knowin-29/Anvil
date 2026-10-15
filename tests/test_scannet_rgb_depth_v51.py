from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import scannet_rgb_depth_v51 as depth  # noqa: E402


RELEASE_MANIFEST = depth.DEFAULT_RELEASE / "release.manifest.json"


def _synthetic_pairs() -> list[depth.PairExample]:
    return [
        depth.PairExample(
            eval_id=f"gsa-sdepth-v51-{index:020x}",
            source_identity=f"private-{index}",
            scene_id=f"scene{index:04d}_00",
            source_image=Path(f"/private/{index}.png"),
            image_sha256=f"{index + 1:064x}",
            entity_a=depth.Entity("chair", (100, 100, 300, 500)),
            entity_b=depth.Entity("table", (500, 100, 800, 600)),
            nearer_choice="A" if index % 2 == 0 else "B",
        )
        for index in range(4)
    ]


def test_production_release_is_exactly_bound_balanced_and_blinded() -> None:
    release = depth.validate_request_release(RELEASE_MANIFEST, verify_images=True)
    assert len(release.source.pairs) == 40
    assert release.manifest["formal_source"]["human_audit_status"] == "FORMAL_PASS"
    assert release.manifest["training_materialization"]["eligible"] is False
    assert release.manifest["ab_assignment"]["label_balance"] == {"A": 20, "B": 20}
    assert set(release.requests) == {depth.TRACK_G, depth.TRACK_S_ORACLE}
    assert all(len(rows) == 40 for rows in release.requests.values())
    public = json.dumps(release.requests, ensure_ascii=False).casefold()
    for forbidden in (
        "[instance",
        "scannet-sdepth",
        "source_sample_id",
        "sensor_measurements",
        "relative_depth",
        '"rank"',
        "nearer_choice",
    ):
        assert forbidden not in public
    assert depth._SCENE_ID.search(public) is None
    for rows in release.requests.values():
        for row in rows:
            assert Path(row["images"][0]).name == f"{row['eval_id']}.png"


def test_public_request_validator_rejects_private_identity_and_target() -> None:
    release = depth.validate_request_release(RELEASE_MANIFEST, verify_images=False)
    clean = release.requests[depth.TRACK_G][0]
    leaked_scene = copy.deepcopy(clean)
    leaked_scene["pair"]["A"]["name"] = "scene0653_00"
    with pytest.raises(depth.ContractError, match="private source identity leaked"):
        depth.validate_public_request(leaked_scene, expected_track=depth.TRACK_G)
    leaked_target = copy.deepcopy(clean)
    leaked_target["target"] = "A"
    with pytest.raises(depth.ContractError, match="schema drift"):
        depth.validate_public_request(leaked_target, expected_track=depth.TRACK_G)


@pytest.mark.parametrize(
    ("response", "status"),
    [
        (
            '{"A":{"bbox_2d":[1,2,100,200]},"B":{"bbox_2d":[300,4,900,800]}}',
            "parsed_strict_json",
        ),
        ('{"A":{"bbox_2d":[1,2,100,200]}}', "grounding_schema"),
        (
            '{"A":{"bbox_2d":[1,2,100,200]},"B":{"bbox_2d":[300,4,900,800]},"answer":"A"}',
            "grounding_schema",
        ),
        ("<think>unfinished", "unclosed_reasoning"),
        ("not-json", "no_json_object"),
    ],
)
def test_grounding_parser_is_strict(response: str, status: str) -> None:
    parsed = depth.parse_grounding_response(response)
    assert parsed.status == status
    assert parsed.valid is (status == "parsed_strict_json")


def test_invalid_g_gets_zero_and_paired_bootstrap_is_deterministic() -> None:
    pairs = _synthetic_pairs()
    good = '{"A":{"bbox_2d":[100,100,300,500]},"B":{"bbox_2d":[500,100,800,600]}}'
    wrong = '{"A":{"bbox_2d":[0,0,50,50]},"B":{"bbox_2d":[50,50,100,100]}}'
    g_rows = [
        {"eval_id": pairs[0].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": good, "hit_token_cap": False},
        {"eval_id": pairs[2].eval_id, "response": "invalid", "hit_token_cap": False},
        {"eval_id": pairs[3].eval_id, "response": wrong, "hit_token_cap": False},
    ]
    oracle = [
        {"eval_id": pair.eval_id, "response": pair.nearer_choice, "hit_token_cap": False}
        for pair in pairs
    ]
    predicted = [
        {"eval_id": pairs[0].eval_id, "response": "A", "hit_token_cap": False},
        {"eval_id": pairs[1].eval_id, "response": "A", "hit_token_cap": False},
        {"eval_id": pairs[3].eval_id, "response": "B", "hit_token_cap": False},
    ]
    first = depth.score_paired_predictions(pairs, g_rows, oracle, predicted)
    second = depth.score_paired_predictions(pairs, g_rows, oracle, predicted)
    assert first["metrics"]["g_both_box_iou_at_0_5"] == 0.5
    assert first["metrics"]["s_oracle_accuracy"] == 1.0
    assert first["metrics"]["s_predicted_context_end_to_end_accuracy"] == 0.25
    assert first["metrics"]["s_accuracy_given_g_success"] == 0.5
    assert first["metrics"]["dependency_gap_oracle_minus_predicted_e2e"] == 0.75
    assert first["paired_bootstrap"] == second["paired_bootstrap"]
    invalid = first["per_item_transitions"][2]
    assert invalid["g_parse_valid"] is False
    assert invalid["s_pred_request_emitted"] is False
    assert invalid["s_pred_end_to_end_correct"] is False


def test_incomplete_or_capped_prediction_fails_closed() -> None:
    pairs = _synthetic_pairs()
    good = '{"A":{"bbox_2d":[100,100,300,500]},"B":{"bbox_2d":[500,100,800,600]}}'
    g_rows = [
        {"eval_id": pair.eval_id, "response": good, "hit_token_cap": False}
        for pair in pairs
    ]
    oracle = [
        {"eval_id": pair.eval_id, "response": pair.nearer_choice, "hit_token_cap": False}
        for pair in pairs
    ]
    predicted = copy.deepcopy(oracle)
    with pytest.raises(depth.ContractError, match="cover every formal scene"):
        depth.score_paired_predictions(pairs, g_rows[:-1], oracle, predicted)
    g_rows[0]["hit_token_cap"] = True
    with pytest.raises(depth.ContractError, match="token cap"):
        depth.score_paired_predictions(pairs, g_rows, oracle, predicted)


def test_evaluator_cli_dry_run_is_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "eval/scannet_rgb_depth_v51.py"), "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY_RUN_PASS" in result.stdout
    assert "Traceback" not in result.stderr
