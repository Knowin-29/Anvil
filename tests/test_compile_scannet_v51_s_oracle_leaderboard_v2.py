from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from eval import compile_scannet_v51_s_oracle_leaderboard as five  # noqa: E402
from eval import compile_scannet_v51_s_oracle_leaderboard_v2 as seven  # noqa: E402
import test_compile_scannet_v51_s_oracle_leaderboard as five_fixtures  # noqa: E402


EXTRA_KEYS = ("internvl3_5_8b", "llava_onevision_qwen2_7b")


def _add_extra_result(
    tmp_path: Path,
    source_result: Path,
    key: str,
    wrong_indices: set[int],
) -> seven.ModelSpec:
    report = copy.deepcopy(json.loads(source_result.read_text(encoding="utf-8")))
    model_dir = tmp_path / "models" / key
    config = five_fixtures._binding(
        five_fixtures._write(
            model_dir / "config.json", f'{{"model_type":"{key}"}}\n'.encode()
        )
    )
    weight = five_fixtures._binding(
        five_fixtures._write(
            model_dir / "model.safetensors", f"weights-{key}".encode()
        ),
        mtime=True,
    )
    inventory = five_fixtures._identity(
        {
            "path": str(model_dir.resolve()),
            "architecture": f"{key}ForConditionalGeneration",
            "model_type": key,
            "source_kind": "standalone_merged_weights",
            "metadata_files": {"config.json": config},
            "weight_files": [weight],
            "weight_bytes_fully_sha256_hashed": True,
        }
    )
    binding_payload = five_fixtures._identity(
        {"role": "evaluated", "model": inventory}
    )
    binding_path = five_fixtures._write(
        tmp_path / key / "model_binding.json",
        (json.dumps(binding_payload, sort_keys=True) + "\n").encode(),
    )
    binding = five_fixtures._binding(binding_path)

    provenance = report["provenance"]
    bundle = provenance["s_oracle_bundle"]
    bundle["model_binding"] = binding
    bundle["model_binding_identity_sha256"] = binding_payload["identity_sha256"]
    bundle["model_inventory_identity_sha256"] = inventory["identity_sha256"]
    bundle["identity_sha256"] = five.canonical_hash(
        {name: value for name, value in bundle.items() if name != "identity_sha256"}
    )
    provenance["model_binding"] = binding
    provenance["model_binding_identity_sha256"] = binding_payload["identity_sha256"]
    provenance["model_inventory"] = inventory
    provenance["identity_sha256"] = five.canonical_hash(
        {
            name: value
            for name, value in provenance.items()
            if name != "identity_sha256"
        }
    )

    for index, item in enumerate(report["per_item"]):
        item["correct"] = index not in wrong_indices
    correct = sum(int(item["correct"]) for item in report["per_item"])
    a_correct = sum(int(item["correct"]) for item in report["per_item"][:20])
    b_correct = sum(int(item["correct"]) for item in report["per_item"][20:])
    report["metrics"] = five_fixtures._metric_block(correct, 40)
    report["ab_strata"] = {
        "A": five_fixtures._metric_block(a_correct, 20),
        "B": five_fixtures._metric_block(b_correct, 20),
    }
    report["confidence_intervals"] = {
        "accuracy_scene_bootstrap": five._bootstrap_ci(
            [int(item["correct"]) for item in report["per_item"]], paired=False
        ),
        "accuracy_binomial": five._wilson_ci(correct, 40),
    }
    report["identity_sha256"] = five.canonical_hash(
        {name: value for name, value in report.items() if name != "identity_sha256"}
    )
    result_path = five_fixtures._write(
        tmp_path / key / "result.json",
        (json.dumps(report, sort_keys=True) + "\n").encode(),
    )
    return seven.ModelSpec(
        key=key,
        display_name=key,
        role="fixture_cross_family",
        result_path=result_path,
        expected_model_inventory_identity_sha256=inventory["identity_sha256"],
    )


def _write_anchor(path: Path, report: dict) -> seven.HistoricalAnchorSpec:
    five._atomic_json(path, report)
    return seven.HistoricalAnchorSpec(
        path=path,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_identity_sha256=report["identity_sha256"],
    )


def _make_seven_fixture(
    tmp_path: Path,
) -> tuple[tuple[seven.ModelSpec, ...], seven.HistoricalAnchorSpec, dict[str, Path]]:
    five_specs, paths = five_fixtures._make_fixture(tmp_path / "five")
    historical = five.compile_leaderboard(five_specs)
    anchor = _write_anchor(tmp_path / "historical_five.json", historical)
    extras = (
        _add_extra_result(
            tmp_path, paths["qwen3_vl_8b"], EXTRA_KEYS[0], {0, 20}
        ),
        _add_extra_result(
            tmp_path,
            paths["qwen3_vl_8b"],
            EXTRA_KEYS[1],
            {0, 1, 2, 20, 21, 22},
        ),
    )
    return tuple(five_specs) + extras, anchor, paths


def test_compiles_seven_and_preserves_historical_five_projection(
    tmp_path: Path,
) -> None:
    specs, anchor, _ = _make_seven_fixture(tmp_path)
    first = seven.compile_leaderboard(specs, anchor)
    second = seven.compile_leaderboard(specs, anchor)
    assert first == second
    unsigned = {key: value for key, value in first.items() if key != "identity_sha256"}
    assert first["identity_sha256"] == five.canonical_hash(unsigned)
    assert first["status"] == seven.LEADERBOARD_STATUS
    assert first["trust"]["leaderboard_rows"] == 7
    assert first["trust"]["historical_five_rows_preserved"] is True
    assert first["lineage"]["historical_five_row_projection_match"] is True
    assert first["live_rehash"]["explicit_result_files_rehashed"] == 7
    rows = {row["model_key"]: row for row in first["leaderboard"]}
    assert rows[EXTRA_KEYS[0]]["correct"] == 38
    assert rows[EXTRA_KEYS[0]]["accuracy"] == 0.95
    assert rows[EXTRA_KEYS[1]]["correct"] == 34
    assert rows[EXTRA_KEYS[1]]["accuracy"] == 0.85


def test_requires_exact_pinned_seven_key_set_before_reading_anchor(
    tmp_path: Path,
) -> None:
    specs, anchor, _ = _make_seven_fixture(tmp_path)
    with pytest.raises(seven.LeaderboardError, match="exactly seven"):
        seven.compile_leaderboard(specs[:-1], anchor)
    duplicate = list(specs)
    duplicate[-1] = seven.ModelSpec(
        key=duplicate[-2].key,
        display_name=duplicate[-1].display_name,
        role=duplicate[-1].role,
        result_path=duplicate[-1].result_path,
        expected_model_inventory_identity_sha256=(
            duplicate[-1].expected_model_inventory_identity_sha256
        ),
    )
    with pytest.raises(seven.LeaderboardError, match="pinned seven-model key set"):
        seven.compile_leaderboard(duplicate, anchor)


def test_rejects_reanchored_historical_row_projection_drift(tmp_path: Path) -> None:
    specs, anchor, _ = _make_seven_fixture(tmp_path)
    historical = json.loads(anchor.path.read_text(encoding="utf-8"))
    historical["leaderboard"][0]["display_name"] = "drifted historical label"
    historical["identity_sha256"] = five.canonical_hash(
        {
            key: value
            for key, value in historical.items()
            if key != "identity_sha256"
        }
    )
    drifted_anchor = _write_anchor(tmp_path / "historical_drifted.json", historical)
    with pytest.raises(seven.LeaderboardError, match="historical_row_projection"):
        seven.compile_leaderboard(specs, drifted_anchor)


def test_rejects_extra_result_trust_drift(tmp_path: Path) -> None:
    specs, anchor, _ = _make_seven_fixture(tmp_path)
    path = specs[-1].result_path
    report = json.loads(path.read_text(encoding="utf-8"))
    report["trust"]["score_available"] = False
    report["identity_sha256"] = five.canonical_hash(
        {key: value for key, value in report.items() if key != "identity_sha256"}
    )
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(seven.LeaderboardError, match=r"llava_onevision_qwen2_7b\.trust drift"):
        seven.compile_leaderboard(specs, anchor)
