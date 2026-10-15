from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "quick_scannet_rgb_depth_v52_unique_referent_audit",
    TOOLS / "quick_scannet_rgb_depth_v52_unique_referent_audit.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_native_candidate_sample_excludes_prior_scenes_and_private_fields(tmp_path: Path) -> None:
    from compile_machine_preaudit import prepare_native_candidates, sha256

    rows = []
    for index in range(8):
        image = tmp_path / f"image{index}.png"
        image.write_bytes(f"unique image {index}".encode())
        rows.append({
            "sample_id": f"sample{index}", "gsa_form": "point", "images": [str(image)],
            "gsa_payload": [{"label": "chair", "point_2d": [50, 50]}],
            "gsa_meta": {"image_sha256": sha256(image), "scene_group": f"scene{index // 2}",
                         "private_sensor_depth": 42},
            "messages": [{"role": "user", "content": "Point to the chair"},
                         {"role": "assistant", "content": "HIDDEN_ASSISTANT"}],
            "gsa_context": {"private_gate": True, "spatial": {"depth_order": [{"rank": 1}]}},
        })
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, rows)
    prior = tmp_path / "prior.jsonl"
    _write_jsonl(prior, [{"sample_id": "prior", "image_path": rows[0]["images"][0],
                          "image_sha256": rows[0]["gsa_meta"]["image_sha256"]}])
    output = tmp_path / "public.jsonl"
    report = prepare_native_candidates(source, output, per_form=6, seed="fixed", exclude=[prior])
    assert report["excluded_scene_groups"] == 1 and report["eligible_rows"] == 6
    public = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["sample_id"] for row in public} == {f"sample{i}" for i in range(2, 8)}
    assert all(row["spatial_context"] == {} for row in public)
    assert not any(key in output.read_text() for key in ("private_sensor", "private_gate", "HIDDEN_ASSISTANT"))
    with pytest.raises(ValueError, match="existing"):
        prepare_native_candidates(source, output, per_form=6, seed="fixed", exclude=[prior])
    with pytest.raises(ValueError, match="insufficient"):
        prepare_native_candidates(source, tmp_path / "short.jsonl", per_form=7, seed="fixed", exclude=[prior])
    _write_jsonl(source, list(reversed(rows)))
    reordered = tmp_path / "reordered.jsonl"
    prepare_native_candidates(source, reordered, per_form=3, seed="fixed", exclude=[prior])
    _write_jsonl(source, rows)
    original = tmp_path / "original.jsonl"
    prepare_native_candidates(source, original, per_form=3, seed="fixed", exclude=[prior])
    assert original.read_bytes() == reordered.read_bytes()
    filtered = tmp_path / "filtered.jsonl"
    filtered_report = prepare_native_candidates(source, filtered, per_form=3, seed="fixed", exclude=[prior], selected_forms=["point"])
    assert filtered.read_bytes() == original.read_bytes()
    assert filtered_report["source_sha256"] == sha256(source)
    with pytest.raises(ValueError, match="present in the source"):
        prepare_native_candidates(source, tmp_path / "missing_form.jsonl", per_form=3, seed="fixed", exclude=[], selected_forms=["mask"])
    rows[2]["gsa_meta"]["image_sha256"] = "drift"
    _write_jsonl(source, rows)
    with pytest.raises(ValueError, match="binding drift"):
        prepare_native_candidates(source, tmp_path / "drift.jsonl", per_form=6, seed="fixed", exclude=[prior])


def _make_pack(tmp_path: Path) -> tuple[Path, list[dict]]:
    pack = tmp_path / "pack"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"review-image")
    image_sha = MODULE.file_sha256(image)
    labels: list[dict] = []
    selected: list[dict] = []
    for index in range(MODULE.NEW_TARGET):
        sample_id = f"v52-new-{index:02d}"
        nearer = "A" if index % 2 == 0 else "B"
        label = {
            "sample_id": sample_id,
            "image_path": str(image),
            "lineage": "new_v52",
            "review_required_in_v52": True,
            "public_interface": {
                "grounding_targets": [
                    {
                        "slot": "A",
                        "category": "chair",
                        "referring_expression": "the only visible chair",
                    },
                    {
                        "slot": "B",
                        "category": "table",
                        "referring_expression": "the only visible table",
                    },
                ]
            },
            "audit_targets": {
                "grounding": [
                    {
                        "slot": "A",
                        "bbox": [10, 20, 300, 400],
                        "source_instance_id": 91,
                        "source_label": "chair",
                    },
                    {
                        "slot": "B",
                        "bbox": [500, 100, 900, 700],
                        "source_instance_id": 37,
                        "source_label": "table",
                    },
                ],
                "spatial": {
                    "nearer_slot": nearer,
                    "farther_slot": "B" if nearer == "A" else "A",
                },
            },
            "source_payload": {"private": True},
            "rgb_sensor_gate": {"depth_gap_m": 1.25},
            "referent_gate": {"instance_projection": "private"},
        }
        labels.append(label)
        selected.append(
            {
                "sample_id": sample_id,
                "lineage": "new_v52",
                "image_path": str(image),
                "image_sha256": image_sha,
                "label_sha256": MODULE.canonical_hash(label),
            }
        )
    selected.extend(
        {"sample_id": f"v52-reused-{index}", "lineage": "reused_v51"}
        for index in range(MODULE.FORMAL_TARGET - MODULE.NEW_TARGET)
    )

    bindings: dict[str, dict] = {}
    for side in ("a", "b"):
        path = pack / f"annotator_{side}/labels/GS_dependency/unique_pair.jsonl"
        _write_jsonl(path, labels)
        bindings[f"reviewer_{side}_labels"] = {
            "path": str(path.relative_to(pack)),
            "bytes": path.stat().st_size,
            "sha256": MODULE.file_sha256(path),
        }
        (pack / f"annotator_{side}/annotations").mkdir(parents=True)
    manifest = {
        "protocol_version": MODULE.PACK_PROTOCOL,
        "audit_contract": MODULE.AUDIT_CONTRACT,
        "stage": "formal_pending_review",
        "records": MODULE.FORMAL_TARGET,
        "selected": selected,
        "provenance": bindings,
    }
    manifest["integrity"] = {
        "manifest_payload_sha256": MODULE.canonical_hash(manifest)
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pack, labels


def test_v52_case_is_reviewable_without_private_sensor_or_provenance_fields(
    tmp_path: Path,
) -> None:
    pack, _ = _make_pack(tmp_path)
    store = MODULE.V52QuickAuditStore(pack, "a")
    case = store.case(0)
    assert store.stats()["expected_decisions"] == MODULE.NEW_TARGET
    assert case["forms"][0]["form"] == "depth"
    assert len(case["forms"][0]["gsa_context"]["groundings"]) == 2
    assert [row["rank"] for row in case["forms"][0]["payload"]] == [1, 2]
    serialized = json.dumps(case, sort_keys=True)
    for private_field in (
        "sample_id",
        "image_name",
        "source_bucket",
        "scene_type",
        "sensor_measurements",
        "sensor_provenance",
        "source_payload",
        "source_instance_id",
        "source_label",
        "rgb_sensor_gate",
        "referent_gate",
        "depth_gap_m",
        "nearer_slot",
    ):
        assert private_field not in serialized
    assert set(case["forms"][0]) == {
        "form",
        "depth_scope",
        "payload",
        "gsa_context",
        "previous",
    }
    assert not store.annotations_path.exists()


def test_v52_submit_requires_concrete_negative_and_preserves_frozen_snapshot(
    tmp_path: Path,
) -> None:
    pack, labels = _make_pack(tmp_path)
    store = MODULE.V52QuickAuditStore(pack, "b")
    with pytest.raises(ValueError, match="concrete comment"):
        store.submit(
            0,
            [{"form": "depth", "correct": False, "error_code": "ambiguous_depth"}],
        )
    result = store.submit(
        0,
        [
            {
                "form": "depth",
                "correct": False,
                "error_code": "ambiguous_depth",
                "comment": "The two surfaces have no visible occlusion cue.",
            }
        ],
    )
    assert result["saved"] == 1
    row = json.loads(store.annotations_path.read_text(encoding="utf-8").strip())
    assert row["label_snapshot"] == labels[0]
    assert row["annotator_key"] == "quick-audit-v52-b"
    assert row["error_code"] == "ambiguous_depth"
    assert MODULE.V52QuickAuditStore(pack, "b").stats()["decisions"] == 1


def test_offline_human_review_embeds_only_public_bound_payload(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit

    pack, labels = _make_pack(tmp_path)
    machine = {
        "reviewer": "machine-a", "supported": False, "error_code": "identity_ambiguous",
        "concise_reason": "Repeated chairs </script><script>bad()</script>", "confidence": 0.8,
    }
    queue = [{"sample_id": label["sample_id"], "reviewer_a": machine,
              "reviewer_b": {**machine, "reviewer": "machine-b"}} for label in labels]
    report = {"provenance": {"reviewer_a": {"sha256": "a" * 64},
                             "reviewer_b": {"sha256": "b" * 64}}}
    html = preaudit.render_human_html(pack, report, queue)
    serialized = html.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
    payload = json.loads(serialized)
    assert len(payload["cases"]) == 33
    assert payload["pack_sha256"] == MODULE.file_sha256(pack / "manifest.json")
    assert len({case["id"] for case in payload["cases"]}) == 33
    assert payload["cases"][0]["image"].startswith("data:image/jpeg;base64,")
    assert payload["cases"][0]["machine"][0]["concise_reason"] == machine["concise_reason"]
    for private in (str(tmp_path), "v52-new-", "source_instance_id", "rgb_sensor_gate",
                    "depth_gap_m", "source_payload", "label_snapshot", "</script>"):
        assert private not in serialized
    assert "independent_human_agreement_eligible:false" in html
    assert "may_populate_paper_metrics:false" in html
    assert not (pack / "annotator_a/annotations/annotations.jsonl").exists()
    (tmp_path / "image.jpg").write_bytes(b"tampered-image")
    with pytest.raises(ValueError, match="binding drift"):
        preaudit.render_human_html(pack, report, queue)


def test_flywheel_offline_review_excludes_verifier_and_detects_image_drift(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit

    raw = tmp_path / "raw"
    raw.mkdir()
    image_path = tmp_path / "private-scene.png"
    image_path.write_bytes(b"bound-test-image")
    image_sha = preaudit.sha256(image_path)
    source = {
        "form": "point", "image_path": str(image_path), "image_sha256": image_sha,
        "payload": [{"label": "mug", "point_2d": [200, 300]}],
        "gsa_context": {"groundings": [{"object_class": "mug", "bbox": [100, 100, 400, 500],
                                         "private_sensor_value": "secret"}]},
        "_verification": {"verdict": "verified", "confidence": 1.0},
    }
    (raw / "point.jsonl").write_text(json.dumps(source) + "\n")
    public = tmp_path / "public.jsonl"
    assert preaudit.prepare_flywheel_inputs(raw, public) == 1
    content = public.read_text()
    assert "_verification" not in content and "secret" not in content
    with pytest.raises(ValueError, match="refusing existing"):
        preaudit.prepare_flywheel_inputs(raw, public)
    machine = {"reviewer": "agent", "supported": True, "error_code": None,
               "concise_reason": "point is on mug", "confidence": 0.9}
    queue = [{"sample_id": image_sha + ":point", "reviewer_a": machine, "reviewer_b": machine}]
    report = {"provenance": {"reviewer_a": {"sha256": "a" * 64},
                             "reviewer_b": {"sha256": "b" * 64}}}
    html = preaudit.render_public_human_html(public, report, queue)
    serialized = html.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
    payload = json.loads(serialized)
    assert payload["cases"][0]["form"] == "point"
    assert payload["cases"][0]["proposal"] == source["payload"]
    assert str(tmp_path) not in serialized and "private-scene" not in serialized
    assert "_verification" not in serialized
    html_path = tmp_path / "issued.html"
    html_path.write_text(html)
    case = payload["cases"][0]
    returned = {
        "schema": payload["schema"], "pack_sha256": payload["pack_sha256"],
        "machine_stream_sha256": payload["machine_stream_sha256"],
        "reviewer": "human-fixture", "machine_aided_review": True,
        "independent_human_agreement_eligible": False, "may_populate_paper_metrics": False,
        "status": "PARTIAL_HUMAN_REVIEW", "records": [{
            "case_id": case["id"], "public_sha256": case["public_sha256"], "complete": False,
        }],
    }
    exported = tmp_path / "returned.json"
    exported.write_text(json.dumps(returned))
    assert preaudit.validate_human_export(exported, html_path)["pending"] == 1
    returned["status"] = "HUMAN_REVIEW_PENDING_VERIFICATION"
    returned["records"][0].update({
        "complete": True, "supported": False, "error_code": "point_off_object",
        "comment": "The contact falls outside the visible object.",
        "reviewed_at_utc": "2026-09-11T00:00:00Z", "machine_suggestion_viewed": True,
        "machine_seen_before_valid_decision": True,
    })
    exported.write_text(json.dumps(returned))
    validated = preaudit.validate_human_export(exported, html_path)
    assert validated["complete"] == validated["unsupported"] == 1
    assert validated["machine_seen_before_valid_decision"] == 1
    assert validated["may_populate_paper_metrics"] is False
    for field, value, match in (
        ("public_sha256", "wrong", "binding drift"),
        ("case_id", "wrong", "unknown or duplicate"),
        ("complete", False, "completion flag"),
        ("supported", "false", "completion flag"),
        ("comment", " ", "completion flag"),
        ("reviewed_at_utc", "2026-09-11T00:00:00", "time zone"),
        ("machine_suggestion_viewed", False, "viewing history"),
    ):
        tampered = json.loads(json.dumps(returned))
        tampered["records"][0][field] = value
        exported.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match=match):
            preaudit.validate_human_export(exported, html_path)
    returned["machine_stream_sha256"]["reviewer_a"] = "another-round"
    exported.write_text(json.dumps(returned))
    with pytest.raises(ValueError, match="binding drift"):
        preaudit.validate_human_export(exported, html_path)
    image_path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="binding drift"):
        preaudit.render_public_human_html(public, report, queue)


def test_public_flywheel_input_rejects_unreviewed_payload_fields(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit

    raw = tmp_path / "raw"
    raw.mkdir()
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"test-image")
    source = {"form": "depth", "image_path": str(image_path),
              "image_sha256": preaudit.sha256(image_path),
              "payload": [{"label": "mug", "rank": 1, "sensor_depth_m": 3.5}]}
    (raw / "depth.jsonl").write_text(json.dumps(source) + "\n")
    public = tmp_path / "public.jsonl"
    with pytest.raises(ValueError, match="unreviewed payload fields"):
        preaudit.prepare_flywheel_inputs(raw, public)
    assert not public.exists()


def test_review_context_excludes_current_answer_and_binds_visible_instruction(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit

    context = {"spatial": {
        "depth_order": [{"label": "mug", "rank": 1, "sensor_depth_m": 5}],
        "affordance": [{"label": "handle", "affordance": "grasp", "private": "secret"}],
        "dof": [{"label": "mug", "wrist_deg": 90}],
    }}
    assert preaudit.visible_upstream_context("depth", context) == {}
    assert preaudit.visible_upstream_context("affordance", context) == {
        "depth_order": [{"label": "mug", "rank": 1}]}
    assert set(preaudit.visible_upstream_context("dof", context)) == {"depth_order", "affordance"}
    assert "secret" not in json.dumps(preaudit.visible_upstream_context("action_cot", context))
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    public = tmp_path / "public.jsonl"
    row = {"sample_id": "x", "image_path": str(image), "image_sha256": preaudit.sha256(image),
           "form": "affordance", "groundings": [], "payload": [],
           "spatial_context": preaudit.visible_upstream_context("affordance", context)}
    report = {"provenance": {"reviewer_a": {"sha256": "a"}, "reviewer_b": {"sha256": "b"}}}
    queue = [{"sample_id": "x", "reviewer_a": {}, "reviewer_b": {}}]

    def render_case():
        public.write_text(json.dumps(row) + "\n")
        html = preaudit.render_public_human_html(public, report, queue)
        return json.loads(html.split('<script id="data" type="application/json">')[1]
                          .split('</script>')[0])["cases"][0]

    initial = render_case()
    row["instruction"] = "Task <script>unsafe</script>"
    updated = render_case()
    assert updated["instruction"] == row["instruction"]
    assert initial["public_sha256"] != updated["public_sha256"]
    row["spatial_context"]["depth_order"][0]["rank"] = 2
    assert render_case()["public_sha256"] != updated["public_sha256"]


def test_native_prediction_review_joins_by_prompt_and_never_exposes_reference(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit

    image = tmp_path / "image.png"
    image.write_bytes(b"bound-image")
    source, predictions, public = (tmp_path / name for name in ("source.jsonl", "pred.jsonl", "public.jsonl"))
    rows, inferred = [], []
    for number in range(2):
        prompt = f"Rank visible objects, case {number}."
        reference = "<RELATE>PRIVATE_REFERENCE_" + str(number)
        messages = [{"role": "user", "content": prompt}]
        rows.append({"sample_id": str(number), "images": [str(image)],
                     "messages": messages + [{"role": "assistant", "content": reference}],
                     "gsa_family": "S", "gsa_form": "depth", "gsa_context": {},
                     "gsa_meta": {"image_sha256": preaudit.sha256(image)}})
        inferred.append({"images": [{"path": str(image), "bytes": None}],
                         "messages": [{**messages[0], "loss": None}], "labels": reference,
                         "response": '<think>\n</think>\n<RELATE>[{"label":"object","rank":1}]'})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    predictions.write_text("".join(json.dumps(row) + "\n" for row in reversed(inferred)))
    assert preaudit.prepare_native_predictions(source, [("S", predictions)], public) == 2
    assert "PRIVATE_REFERENCE" not in public.read_text()
    assert "loss" not in public.read_text()
    inferred[0]["messages"][0]["content"] = "wrong prompt"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in inferred))
    with pytest.raises(ValueError, match="every source prompt/image"):
        preaudit.prepare_native_predictions(source, [("S", predictions)], tmp_path / "bad.jsonl")
    assert not (tmp_path / "bad.jsonl").exists()


def test_native_grounding_review_hides_reference_boxes_outside_model_input(tmp_path: Path) -> None:
    import compile_machine_preaudit as preaudit
    image = tmp_path / "image.png"
    image.write_bytes(b"bound image")
    source, predicted, public = (tmp_path / name for name in ("source.jsonl", "predicted.jsonl", "public.jsonl"))
    messages = [{"role": "user", "content": "Locate the named mug."}]
    reference = '<GROUND>[{"label":"mug","bbox_2d":[789,789,989,989]}]'
    context = {"groundings": [{"object_class": "PRIVATE_GROUNDING", "bbox": [789, 789, 989, 989]}]}
    row = {"sample_id": "one", "images": [str(image)], "gsa_form": "bbox", "gsa_family": "G",
           "messages": messages + [{"role": "assistant", "content": reference}], "gsa_context": context,
           "gsa_meta": {"image_sha256": preaudit.sha256(image)}}
    response = {"images": [str(image)], "messages": messages, "labels": reference,
                "response": '<GROUND>[{"label":"mug","bbox_2d":[100,100,200,200]}]'}
    source.write_text(json.dumps(row) + "\n")
    predicted.write_text(json.dumps(response) + "\n")
    assert preaudit.prepare_native_predictions(source, [("G", predicted)], public) == 1
    text = public.read_text()
    assert "PRIVATE_GROUNDING" not in text and "789" not in text and "989" not in text
    assert json.loads(text)["groundings"] == []
    for form in ("bbox", "detection", "point", "mask"):
        assert preaudit.visible_native_groundings(form, context) == []
    assert preaudit.visible_native_groundings("depth", context) == context["groundings"]


def test_machine_agreement_gate_is_strict_and_never_admits_human_metrics(tmp_path: Path) -> None:
    import argparse
    import compile_machine_preaudit as preaudit

    def stream(reviewer, disagreements):
        return [{"sample_id": str(i), "reviewer": reviewer, "supported": i >= disagreements,
                 "error_code": "wrong_label" if i < disagreements else None,
                 "concise_reason": "synthetic gate fixture", "confidence": 0.9,
                 "reviewed_at_utc": "2026-09-11T00:00:00+00:00"} for i in range(20)]

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(row) + "\n" for row in stream("a", 0)))
    args = argparse.Namespace(reviewer_a=a, reviewer_b=b, expected_count=20,
                              pack=None, public_inputs=None, min_agreement=0.9, min_confidence=0.8)
    for differences, passes in ((2, False), (1, True)):
        b.write_text("".join(json.dumps(row) + "\n" for row in stream("b", differences)))
        report, queue = preaudit.compile_streams(args)
        assert report["gate_passed"] is passes
        assert report["may_populate_formal_human_metrics"] is False
        assert len(queue) == 20 and all(row["requires_human_review"] for row in queue)


def test_calibration_preserves_first_pass_denominator_and_carried_votes(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import compile_machine_preaudit as preaudit

    image = tmp_path / "image.png"
    image.write_bytes(b"bound image fixture")
    public = tmp_path / "public.jsonl"
    _write_jsonl(public, [{"sample_id": str(i), "form": form, "image_path": str(image),
                           "image_sha256": preaudit.sha256(image)}
                          for i, form in enumerate(("dof", "point"))])
    streams = [[{"sample_id": str(i), "reviewer": side, "supported": True,
                 "error_code": None, "concise_reason": "synthetic calibration fixture",
                 "confidence": 0.9, "reviewed_at_utc": "2026-09-11T00:00:00+00:00"}
                for i in range(2)] for side in ("a", "b")]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, rows in zip((a, b), streams):
        _write_jsonl(path, rows)
    args = argparse.Namespace(reviewer_a=a, reviewer_b=b, expected_count=2, pack=None,
                              public_inputs=public, min_agreement=0.9, min_confidence=0.8)
    original, _ = preaudit.compile_streams(args)
    previous = tmp_path / "report.json"
    previous.write_text(json.dumps(original))
    args.previous_report, args.rereviewed_form = previous, "dof"
    args.reviewer_a, args.reviewer_b = tmp_path / "a2.jsonl", tmp_path / "b2.jsonl"
    for path, rows in zip((args.reviewer_a, args.reviewer_b), streams):
        rows[0]["reviewed_at_utc"] = "2026-09-12T00:00:00+00:00"
        _write_jsonl(path, rows)
    report, _ = preaudit.compile_streams(args)
    assert report["calibration"]["carried_records_per_reviewer"] == 1
    assert report["calibration"]["rereviewed_records_per_reviewer"] == 1
    assert report["calibration"]["fresh_full_sample_independent_pass"] is False
    assert report["provenance"]["previous_report"]["sha256"] == preaudit.sha256(previous)
    streams[0][1]["confidence"] = 0.99
    _write_jsonl(args.reviewer_a, streams[0])
    with pytest.raises(ValueError, match="carried record"):
        preaudit.compile_streams(args)
    streams[0][1]["confidence"] = 0.9
    _write_jsonl(args.reviewer_a, streams[0])
    a.write_text(a.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance drift"):
        preaudit.compile_streams(args)
    monkeypatch.setattr(sys, "argv", ["preaudit", "--reviewer-a", str(a), "--reviewer-b", str(b),
                                     "--report", str(previous), "--human-queue", str(tmp_path / "queue.jsonl")])
    with pytest.raises(ValueError, match="overwrite issued review evidence"):
        preaudit.main()
    assert json.loads(previous.read_text()) == original
