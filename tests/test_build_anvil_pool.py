"""Small source-bound pool fixtures; no benchmark or production-quality claim."""
import copy
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest

from tools import build_anvil_pool as pool
from tools.arxiv_source_scale import fingerprint, sha


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


@pytest.fixture
def pool_input(tmp_path):
    dense = tmp_path / "dense"; dense.mkdir()
    eligible_rows = []; eligible_images = []; inventory = {"state": "inventory_complete_not_frozen_benchmark", "inputs": {}}
    patterns = {"coco_train2017": [(1, True, 2), (2, True, 2), (3, False, 8)],
                "pixmo_points": [(4, True, 8), (5, False, 12)]}
    for source, examples in patterns.items():
        folder = dense / source; folder.mkdir(); by_split = {"train": [], "evaluation_candidate": []}; infos = []
        for number, eligible, amount in examples:
            image = tmp_path / f"image{number}.png"
            Image.fromarray(np.random.default_rng(number).integers(0, 256, (32, 36, 3), dtype=np.uint8)).save(image)
            fp = fingerprint(image); group = fp["decoded_rgb_sha256"]
            info = {**fp, "path": str(image), "source": source, "group_id": group, "split": "train"}
            infos.append(info)
            if eligible: eligible_images.append(group)
            forms = ["point"] if source == "pixmo_points" else sorted(pool.DENSE_FORMS)
            for form in forms:
                for i in range(amount):
                    uid = hashlib.sha256(f"{source}|{number}|{form}|{i}".encode()).hexdigest()
                    row = {"id": uid, "images": [str(image)], "messages": [{"role": "user", "content": "source fixture"},
                           {"role": "assistant", "content": "{}"}],
                           "anvil_meta": {"group_id": group, "source": source, "split": "train",
                                          "native_form": form, "schema": "source-fixture-v1",
                                          "verification": "source_annotation_and_deterministic_geometry"}}
                    by_split["train"].append(row)
                    if eligible: eligible_rows.append((uid, group))
        write_lines(folder / "images.jsonl", infos)
        for name, rows in by_split.items(): write_lines(folder / (name + ".jsonl"), rows)
        manifest = {"files": {name: {"sha256": sha(folder / name)} for name in
                    ["images.jsonl", "train.jsonl", "evaluation_candidate.jsonl"]}}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        inventory["inputs"][source] = {"manifest_sha256": sha(folder / "manifest.json"),
            **{name: {"sha256": sha(folder / (name + ".jsonl"))} for name in by_split}}
    index = tmp_path / "eligible.sqlite"
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE images(group_id TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE records(id TEXT PRIMARY KEY,group_id TEXT)")
        db.executemany("INSERT INTO images VALUES(?)", [(key,) for key in eligible_images])
        db.executemany("INSERT INTO records VALUES(?,?)", eligible_rows)
    inventory_path = tmp_path / "inventory.json"; inventory_path.write_text(json.dumps(inventory))
    native_specs = []
    for number, split, form in [(10, "train", "affordance"), (11, "benchmark", "dof")]:
        image = tmp_path / f"native{number}.png"
        Image.fromarray(np.random.default_rng(number).integers(0, 256, (32, 36, 3), dtype=np.uint8)).save(image)
        row = {"id": f"native-{number}", "images": [str(image)],
               "anvil_meta": {"source": "fixture-native", "native_form": form,
                              "native_variant": f"fixture-{form}-v1",
                              "group_id": f"scene-{number}",
                              "image_sha256": sha(image), "split": "train" if split == "train" else "validation",
                              "training_approved": split == "train"}}
        folder = tmp_path / f"native-{number}"; folder.mkdir()
        path = folder / "records.jsonl"; write_lines(path, [row])
        manifest = folder / "manifest.json"; manifest.write_text(json.dumps({"records_sha256": sha(path)}))
        native_specs.append([f"native-{form}", split, str(path), str(manifest)])
    return Namespace(dense_root=dense, eligible_index=index, eligible_inventory=inventory_path,
                     native_source=native_specs, pending_source=["action-source"], output=tmp_path / "output",
                     train_records=45, benchmark_records=20, images=7, seed="fixture-fixed-seed")


@pytest.mark.parametrize("total,weights,caps,floors", [
    (9, {"a": 1, "b": 2}, {"a": 9, "b": 9}, {}),
    (9, {"a": 1, "b": 2}, {"a": 2, "b": 9}, {}),
    (9, {"a": 1, "b": 2}, {"a": 9, "b": 9}, {"a": 6}),
    (0, {"a": 1, "b": 2}, {"a": 9, "b": 9}, {}),
])
def test_capped_hamilton_exact_and_deterministic(total, weights, caps, floors):
    result = pool.apportion(total, weights, caps, floors)
    assert sum(result.values()) == total
    assert all(floors.get(k, 0) <= result[k] <= caps[k] for k in weights)
    assert result == pool.apportion(total, weights, caps, floors)


def test_capped_hamilton_cannot_invent_capacity():
    with pytest.raises(ValueError, match="capacity"):
        pool.apportion(9, {"a": 1}, {"a": 2})


def test_qa_group_budget_uses_real_counts_and_is_order_independent():
    images = {"a": {"qa_count": 10}, "b": {"qa_count": 2}, "c": {"qa_count": 10}}
    result = pool.qa_image_quotas(images, 17, "fixture")
    assert sum(result.values()) == 17 and len(result) == 2
    assert result == pool.qa_image_quotas(dict(reversed(list(images.items()))), 17, "fixture")
    with pytest.raises(ValueError, match="capacity"):
        pool.qa_image_quotas(images, 23, "fixture")


@pytest.fixture
def refinement_input(pool_input):
    base = pool.build(pool_input)
    root = pool_input.output.parent
    qa = root / "qa"; qa.mkdir()
    image = root / "qa.png"
    Image.fromarray(np.random.default_rng(101).integers(0, 256, (32, 36, 3), dtype=np.uint8)).save(image)
    fp = fingerprint(image)
    info = {**fp, "path": str(image), "split": "train", "qa_count": 4, "group_id": "CLEVR:train:1"}
    write_lines(qa / "images.jsonl", [info])
    records = [{"id": f"clevr-qa:{i}", "images": [str(image)],
        "messages": [{"role": "user", "content": "<image>Source fixture?"}, {"role": "assistant", "content": "yes"}],
        "anvil_meta": {"record_type": "qa_replay", "schema": "anvil-clevr-official-qa-replay-v1",
            "source": "CLEVR", "split": "train", "training_approved": True,
            "group_id": info["group_id"], "image_sha256": fp["sha256"], "decoded_rgb_sha256": fp["decoded_rgb_sha256"]}}
        for i in range(4)]
    write_lines(qa / "train_candidates.jsonl", records)
    manifest = {"schema": "anvil-clevr-official-qa-replay-v1", "state": "official_QA_matched_local_image_candidates",
                "counts": {"train_candidate_qa": 4},
                "protections": [{"kind": "fixed_pool_benchmark", "sha256": sha(pool_input.output / "manifest.json"),
                                 "index_sha256": base["selection_sqlite_sha256"]}],
                "files": {n: sha(qa / n) for n in ("images.jsonl", "train_candidates.jsonl")}}
    (qa / "manifest.json").write_text(json.dumps(manifest))
    return Namespace(base_index=pool_input.output / "selection.sqlite", base_manifest=pool_input.output / "manifest.json",
                     qa_root=qa, qa_records=4, seed="fixture-refinement", native_source=[], output=root / "refined")


def test_refinement_preserves_benchmark_and_adds_exact_qa_without_count_inflation(refinement_input):
    args = refinement_input
    old_sha = sha(args.base_index)
    result = pool.merge_replay(args)
    assert result["qa_replay_count"] == 4 and result["qa_replay_missing"] == 0
    assert result["observed"]["selected_records"] == 65
    assert result["observed"]["unique_images"] == 7
    assert result["all_selected_offsets_replayed"] == 65
    assert result["recipe_counts_complete"] and not result["paper_pool_complete"]
    assert sha(args.base_index) == old_sha
    with sqlite3.connect(args.base_index) as old, sqlite3.connect(args.output / "selection.sqlite") as new:
        assert pool.selection_digest(old, "benchmark") == pool.selection_digest(new, "benchmark")
        assert new.execute("SELECT count(*) FROM selection WHERE form='QA' AND split='benchmark'").fetchone()[0] == 0


def test_refinement_rejects_wrong_qa_benchmark_protection(refinement_input):
    args = refinement_input
    p = args.qa_root / "manifest.json"; m = json.loads(p.read_text())
    m["protections"][0]["index_sha256"] = "0" * 64; p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="exact benchmark"):
        pool.merge_replay(args)
    assert not args.output.exists()


def test_refinement_rejects_source_drift_and_existing_output(refinement_input):
    args = refinement_input
    pool.merge_replay(args)
    before = sha(args.output / "selection.sqlite")
    with pytest.raises(ValueError, match="new immutable"):
        pool.merge_replay(args)
    assert sha(args.output / "selection.sqlite") == before
    args.output = args.output.parent / "next"
    with (args.qa_root / "train_candidates.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="QA source SHA256"):
        pool.merge_replay(args)


def test_refinement_cannot_move_new_native_validation_into_train(refinement_input):
    args = refinement_input
    args.native_source = [["fixture", "benchmark", "unused", "unused"]]
    with pytest.raises(ValueError, match="alter benchmark"):
        pool.merge_replay(args)


@pytest.mark.parametrize("reuse_native_image", [True, False])
def test_refinement_adds_native_train_while_preserving_existing_native(refinement_input, reuse_native_image):
    args = refinement_input
    root = args.output.parent / "new-native"; root.mkdir()
    image = root / "image.png"
    Image.fromarray(np.random.default_rng(102).integers(0, 256, (32, 36, 3), dtype=np.uint8)).save(image)
    if reuse_native_image:
        with sqlite3.connect(args.base_index) as db:
            image = Path(db.execute("SELECT image_path FROM selection WHERE split='train' AND mandatory=1 LIMIT 1").fetchone()[0])
    row = {"id": "new-native-action", "images": [str(image)],
           "anvil_meta": {"source": "fixture-action", "native_form": "action_cot", "native_variant": "fixture-action-v1",
                          "split": "train", "training_approved": True, "group_id": "fixture-action:scene-1",
                          "image_sha256": sha(image)}}
    write_lines(root / "train.jsonl", [row])
    (root / "manifest.json").write_text(json.dumps({"files": {"train.jsonl": sha(root / "train.jsonl")}}))
    args.native_source = [["fixture-action", "train", str(root / "train.jsonl"), str(root / "manifest.json")]]
    if not reuse_native_image:
        # This deliberately tiny seven-image pool cannot hold both new image groups
        # and retain enough dense rows. Production capacity must not be fabricated.
        with pytest.raises(ValueError, match="Insufficient stratum capacity"):
            pool.merge_replay(args)
        assert not (args.output / "manifest.json").exists()
        return
    result = pool.merge_replay(args)
    assert result["selection"]["new_native_rows"] == 1
    assert result["selection"]["mandatory_train_rows"] == 2
    with sqlite3.connect(args.output / "selection.sqlite") as db:
        assert db.execute("SELECT split,mandatory FROM selection WHERE uid='new-native-action'").fetchone() == ("train", 1)


@pytest.fixture
def benchmark_extension_input(refinement_input):
    old = refinement_input; parent = pool.merge_replay(old)
    root = old.output.parent / "native-test"; root.mkdir()
    with sqlite3.connect(old.output / "selection.sqlite") as db:
        image = db.execute("SELECT image_path FROM selection WHERE split='benchmark' LIMIT 1").fetchone()[0]
    row = {"id": "official-test-fixture", "images": [image], "anvil_meta": {
        "source": "ShareRobot-Bench", "native_form": "affordance", "native_variant": "anvil-affordance-region-v1",
        "group_id": "ShareRobot-Bench:image:fixture", "split": "test", "training_approved": False,
        "benchmark_eligible": True, "image_sha256": sha(image)}}
    write_lines(root / "evaluation.jsonl", [row])
    manifest = {"schema": "anvil-sharerobot-official-test-v1", "state": "official_test_subset_exported",
        "training_approved": False, "source_split": "test", "records": 1,
        "files": {"evaluation.jsonl": sha(root / "evaluation.jsonl")}, "protected_training": {
            "pool_index_sha256": parent["selection_sqlite_sha256"], "pool_manifest_sha256": sha(old.output / "manifest.json")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return Namespace(base_index=old.output / "selection.sqlite", base_manifest=old.output / "manifest.json",
                     native_records=root / "evaluation.jsonl", native_manifest=root / "manifest.json",
                     output=old.output.parent / "benchmark-extended", seed="fixture-benchmark")


def test_benchmark_extension_versions_test_and_keeps_train_exact(benchmark_extension_input):
    args = benchmark_extension_input
    before = sha(args.base_index)
    result = pool.extend_benchmark(args)
    assert result["observed"]["selected_records"] == 65
    assert result["all_selected_offsets_replayed"] == 65
    assert result["benchmark_version_change"]["new_native_records"] == 1
    assert result["benchmark_version_change"]["old_scores_transferable"] is False
    assert not result["paper_pool_complete"] and not result["benchmark_scored"]
    assert before == sha(args.base_index)
    with sqlite3.connect(args.base_index) as old, sqlite3.connect(args.output / "selection.sqlite") as new:
        assert pool.selection_digest(old, "train") == pool.selection_digest(new, "train")
        assert pool.selection_digest(old, "benchmark") != pool.selection_digest(new, "benchmark")
        assert new.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0] == 7
        assert new.execute("SELECT split,form FROM selection WHERE uid='official-test-fixture'").fetchone() == ("benchmark", "affordance")
    with pytest.raises(ValueError, match="immutable"): pool.extend_benchmark(args)


@pytest.mark.parametrize("fault", ["wrong_pool", "scored", "source_drift", "train_collision", "train_label"])
def test_benchmark_extension_rejects_unbound_or_trained_source(benchmark_extension_input, fault):
    args = benchmark_extension_input
    native = json.loads(args.native_manifest.read_text())
    if fault == "wrong_pool": native["protected_training"]["pool_index_sha256"] = "0" * 64
    if fault == "scored":
        parent = json.loads(args.base_manifest.read_text()); parent["benchmark_scored"] = True
        args.base_manifest.write_text(json.dumps(parent))
    if fault == "source_drift":
        with args.native_records.open("a") as stream: stream.write("{}\n")
    if fault in {"train_collision", "train_label"}:
        row = json.loads(args.native_records.read_text())
        if fault == "train_collision":
            with sqlite3.connect(args.base_index) as db:
                image = db.execute("SELECT image_path FROM selection WHERE split='train' LIMIT 1").fetchone()[0]
            row["images"] = [image]; row["anvil_meta"]["image_sha256"] = sha(image)
        else: row["anvil_meta"]["split"] = "train"
        write_lines(args.native_records, [row]); native["files"]["evaluation.jsonl"] = sha(args.native_records)
    args.native_manifest.write_text(json.dumps(native))
    with pytest.raises(ValueError): pool.extend_benchmark(args)
    assert not args.output.exists()


def test_new_benchmark_images_can_only_rebalance_dense_train_with_opt_in(pool_input):
    pool_input.train_records = 12
    refinement = refinement_input.__wrapped__(pool_input)
    # Share a QA/native training image in this seven-image fixture so there is
    # still a nonmandatory dense image to replace without losing all capacity.
    with sqlite3.connect(refinement.base_index) as db:
        shared = db.execute("SELECT image_path FROM selection WHERE split='train' AND mandatory=1").fetchone()[0]
    fp = fingerprint(shared); qa = refinement.qa_root
    info = json.loads((qa / "images.jsonl").read_text()); info.update(fp, path=shared)
    write_lines(qa / "images.jsonl", [info])
    records = [json.loads(line) for line in (qa / "train_candidates.jsonl").read_text().splitlines()]
    for row in records:
        row["images"] = [shared]
        row["anvil_meta"].update(image_sha256=fp["sha256"], decoded_rgb_sha256=fp["decoded_rgb_sha256"])
    write_lines(qa / "train_candidates.jsonl", records)
    m = json.loads((qa / "manifest.json").read_text())
    m["files"] = {name: sha(qa / name) for name in m["files"]}
    (qa / "manifest.json").write_text(json.dumps(m))
    args = benchmark_extension_input.__wrapped__(refinement)
    image = args.output.parent / "independent-test.png"
    Image.fromarray(np.random.default_rng(902).integers(0, 256, (32, 36, 3), dtype=np.uint8)).save(image)
    row = json.loads(args.native_records.read_text()); row["images"] = [str(image)]
    row["anvil_meta"]["image_sha256"] = sha(image)
    write_lines(args.native_records, [row])
    native = json.loads(args.native_manifest.read_text())
    native["files"]["evaluation.jsonl"] = sha(args.native_records)
    args.native_manifest.write_text(json.dumps(native))
    args.rebalance_dense_training = True
    result = pool.extend_benchmark(args)
    assert result["observed"]["rows_by_split"] == {"benchmark": 20, "train": 12}
    assert result["benchmark_version_change"]["new_image_groups"] == 1
    assert result["benchmark_version_change"]["removed_groups_split"] == "train"
    assert result["training_version_change"]["dense_rebalance_enabled"]
    assert result["training_version_change"]["native_and_qa_unchanged"]
    assert not result["training_version_change"]["all_rows_unchanged"]
    with sqlite3.connect(args.base_index) as old, sqlite3.connect(args.output / "selection.sqlite") as new:
        frozen = "SELECT uid FROM selection WHERE split='train' AND (mandatory=1 OR form='QA') ORDER BY uid"
        assert old.execute(frozen).fetchall() == new.execute(frozen).fetchall()
        assert new.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0] == 7


def test_native_cannot_change_split():
    row = {"id": "n", "images": ["/source/image.png"], "anvil_meta": {
        "source": "s", "native_form": "dof", "native_variant": "se3-v1",
        "group_id": "scene", "split": "train", "training_approved": True, "image_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="conflicts"):
        pool.load_native_row(row, "benchmark")


def test_native_training_requires_explicit_source_approval():
    row = {"id": "n", "images": ["/source/image.png"], "anvil_meta": {
        "source": "s", "native_form": "dof", "native_variant": "se3-v1",
        "group_id": "scene", "split": "train", "image_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="training approval"):
        pool.load_native_row(row, "train")


def test_full_fixture_build_exact_counts_and_replay(pool_input):
    result = pool.build(pool_input)
    assert result["qa_replay_count"] == 0 and result["qa_replay_missing"] == 464000
    assert result["paper_pool_complete"] is False
    assert result["observed"] == {"rows_by_split": {"benchmark": 20, "train": 45},
                                  "selected_records": 65, "unique_images": 7}
    assert result["pending_sources"] == ["action-source"]
    assert result["state"] == "count_complete_partial_source_pool"
    assert not result["public_release_eligible"] and not result["release_complete"]
    with sqlite3.connect(pool_input.output / "selection.sqlite") as db:
        selected = list(db.execute("SELECT uid,source_path,byte_offset,byte_length FROM selection"))
        assert len({r[0] for r in selected}) == 65
        assert db.execute("SELECT count(*) FROM selection WHERE mandatory=1").fetchone()[0] == 2
        assert db.execute("SELECT count(DISTINCT image_id) FROM selection").fetchone()[0] == 7
        for uid, path, offset, length in selected:
            with Path(path).open("rb") as stream:
                stream.seek(offset); row = json.loads(stream.read(length))
            assert row["id"] == uid
    # No output image directory, no fabricated/copied samples.
    assert {p.suffix for p in pool_input.output.iterdir()} <= {".sqlite", ".json"}


def test_same_seed_same_selected_identity(pool_input):
    first = pool.build(pool_input)
    pool_input.output = pool_input.output.with_name("output-second")
    second = pool.build(pool_input)
    assert first["logical_selected_rows_sha256"] == second["logical_selected_rows_sha256"]


def test_changed_source_bytes_fail_closed(pool_input):
    path = pool_input.dense_root / "coco_train2017" / "train.jsonl"
    with path.open("a") as handle: handle.write(" " + "\n")
    with pytest.raises(ValueError):
        pool.build(pool_input)
    assert not (pool_input.output / "manifest.json").exists()


def test_native_budget_never_overrides_frozen_images(pool_input):
    pool_input.images = 1
    with pytest.raises(ValueError, match="image budget"):
        pool.build(pool_input)


def test_insufficient_rows_cannot_report_completion(pool_input):
    pool_input.train_records = 10000
    with pytest.raises(ValueError, match="shortfall"):
        pool.build(pool_input)
    assert not (pool_input.output / "manifest.json").exists()


def test_existing_output_is_not_overwritten(pool_input):
    pool_input.output.mkdir(); sentinel = pool_input.output / "keep"; sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="immutable"):
        pool.build(pool_input)
    assert sentinel.read_text() == "preserve"


def test_changed_native_image_fails_closed(pool_input):
    path = Path(pool_input.native_source[0][2])
    row = json.loads(path.read_text())
    Image.new("RGB", (36, 32), (1, 2, 3)).save(row["images"][0])
    with pytest.raises(ValueError, match="image bytes changed"):
        pool.build(pool_input)
    assert not (pool_input.output / "manifest.json").exists()


def test_native_physical_scene_cannot_cross_splits(pool_input):
    path = Path(pool_input.native_source[1][2])
    row = json.loads(path.read_text())
    row["anvil_meta"]["group_id"] = "scene-10"
    write_lines(path, [row])
    Path(pool_input.native_source[1][3]).write_text(json.dumps({"records_sha256": sha(path)}))
    with pytest.raises(ValueError, match="physical scene spans"):
        pool.build(pool_input)
    assert not (pool_input.output / "manifest.json").exists()
