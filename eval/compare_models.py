#!/usr/bin/env python3
"""Re-score recorded predictions and report base vs SFT honestly.

Three things the per-run JSON cannot show on its own:

1. **Scale rule** (audit F1/F2). The same dumps are scored under the old `auto` rule and
   under the prompt-driven rule, so the change attributable to fixing the scale
   interpretation is separated from the change attributable to the model.
2. **Paired subset** (audit F3). The two models truncate at different rates and an
   unparsed answer scores 0, so the full-set gap partly measures the token budget. The
   subset where BOTH models produced a parseable answer is reported alongside.
3. **Referent split**. Point benchmarks mix "point at an object" and "point at empty
   space"; only the former matches what the training corpus taught. The registry's
   `referent_by_group` is used to break the score out.

No inference: this replays `save/logs/preds_<tag>_<bench>.jsonl`.

    python eval/compare_models.py
    python eval/compare_models.py --benchmarks where2place refspatial_bench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gsa_eval.parsing import parse_points  # noqa: E402
from eval.gsa_eval.registry import BENCHMARKS, get_spec  # noqa: E402
from eval.gsa_eval.scorers import score_point_in_mask  # noqa: E402


def load_dump(tag: str, bench: str) -> dict[str, str]:
    p = ROOT / "save" / "logs" / f"preds_{tag}_{bench}.jsonl"
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["uid"])] = row.get("response") or ""
    return out


def samples_for(bench: str) -> dict:
    spec = get_spec(bench)
    scratch = ROOT / ".cache" / "eval_assets"
    return {str(s.uid): s for s in spec.loader(scratch)}


class _FakeSpec:
    """Same spec with the scale rule overridden, to isolate that one variable."""

    def __init__(self, spec, prompt_coord_scale):
        self.referent_type = spec.referent_type
        self.referent_by_group = spec.referent_by_group
        self.prompt_coord_scale = prompt_coord_scale


def score_subset(spec, by_uid: dict, responses: dict[str, str],
                 uids: list[str]) -> tuple[float, dict, int]:
    samples = [by_uid[u] for u in uids if u in by_uid]
    texts = [responses.get(u, "") for u in uids if u in by_uid]
    return score_point_in_mask(samples, texts, spec=spec)


def run_point_bench(bench: str, tags: list[str]) -> dict:
    spec = get_spec(bench)
    by_uid = samples_for(bench)
    dumps = {t: load_dump(t, bench) for t in tags}
    missing = [t for t, d in dumps.items() if not d]
    if missing:
        return {"benchmark": bench, "error": f"no dump for {missing}"}

    # uids present in every dump, so both columns describe the same questions
    common = sorted(set.intersection(*(set(d) for d in dumps.values())) & set(by_uid))
    parsed_ok = {t: {u for u in common if parse_points(dumps[t][u])} for t in tags}
    both_parsed = sorted(set.intersection(*(parsed_ok[t] for t in tags)))

    res: dict = {"benchmark": bench, "n_common": len(common),
                 "referent_type": spec.referent_type,
                 "referent_by_group": spec.referent_by_group,
                 "prompt_cardinality": spec.prompt_cardinality,
                 "n_both_parsed": len(both_parsed), "per_model": {}}
    old_spec = _FakeSpec(spec, "")          # old auto/corpus-vote path
    new_spec = _FakeSpec(spec, spec.prompt_coord_scale or "unit")

    for t in tags:
        s_old, d_old, f_old = score_subset(old_spec, by_uid, dumps[t], common)
        s_new, d_new, f_new = score_subset(new_spec, by_uid, dumps[t], common)
        s_pair, d_pair, _ = score_subset(new_spec, by_uid, dumps[t], both_parsed)
        res["per_model"][t] = {
            "score_auto_old": s_old,
            "score_prompt_driven": s_new,
            "delta_from_scale_fix": round(s_new - s_old, 2),
            "n_reinterpreted": d_new["coord_scale"]["n_reinterpreted_vs_auto"],
            "scale_used": d_new["coord_scale"]["used"],
            "auto_would_have_said": d_new["coord_scale"]["auto_would_have_said"],
            "parse_failures": f_new,
            "parse_success_rate": round(100.0 * len(parsed_ok[t]) / len(common), 2),
            "points_per_question": d_new["points_per_question"],
            "score_paired_subset": s_pair,
            "per_group": d_new["per_group"],
            "by_referent_type": d_new.get("by_referent_type", {}),
        }
    if len(tags) == 2:
        a, b = tags
        pa, pb = res["per_model"][a], res["per_model"][b]
        res["gap"] = {
            f"{b}_minus_{a}_auto_old": round(pb["score_auto_old"] - pa["score_auto_old"], 2),
            f"{b}_minus_{a}_prompt_driven": round(
                pb["score_prompt_driven"] - pa["score_prompt_driven"], 2),
            f"{b}_minus_{a}_paired_subset": round(
                pb["score_paired_subset"] - pa["score_paired_subset"], 2),
        }
        # Aligned comparison: only the object-referring groups, which are the ones asking
        # the question the training corpus actually taught.
        if spec.referent_by_group:
            obj = [g for g, r in spec.referent_by_group.items() if r == "object"]
            if obj:
                res["object_referring_only"] = {
                    "groups": obj,
                    **{t: {g: res["per_model"][t]["per_group"].get(g)
                           for g in obj} for t in tags}}
    return res


NATIVE_METRICS = {
    "bbox": "bbox_iou", "detection": "primary", "mask": "mask_iou", "point": "hit_at_0_1",
    "depth": "mean_per_item_pair_accuracy", "affordance": "single_reference_affordance_score",
    "dof": "single_reference_dof_score", "action_cot": "target_point_score",
}


def native_cluster_interval(rows: list[dict], *, forms: tuple[str, ...] | None = None,
                            replicates: int = 10000, seed: int = 20260913) -> dict:
    """Exploratory paired cluster bootstrap at fixed model checkpoints.

    Draw each scene/image once per replicate and carry all of its forms together.
    Recompute the equal-form macro inside every draw; never resample form rows as
    if four outputs on one image were four independent observations.
    """
    import numpy as np

    if not rows or type(replicates) is not int or replicates < 100:
        raise ValueError("cluster interval requires rows and at least 100 replicates")
    forms = tuple(sorted({row["form"] for row in rows})) if forms is None else forms
    if not forms or len(set(forms)) != len(forms):
        raise ValueError("cluster interval requires unique nonempty forms")
    selected = [row for row in rows if row["form"] in forms]
    if set(forms) != {row["form"] for row in selected}:
        raise ValueError("cluster interval is missing a requested form")
    clusters = sorted({row["cluster_id"] for row in selected})
    cluster_index = {key: i for i, key in enumerate(clusters)}
    form_index = {key: i for i, key in enumerate(forms)}
    sums = np.zeros((len(clusters), len(forms)))
    counts = np.zeros_like(sums)
    for row in selected:
        delta = row["delta_pp"]
        if isinstance(delta, bool) or not math.isfinite(delta) or not -100 <= delta <= 100:
            raise ValueError("invalid paired score difference")
        i, j = cluster_index[row["cluster_id"]], form_index[row["form"]]
        sums[i, j] += delta
        counts[i, j] += 1
    result = {"method": "paired_cluster_percentile_bootstrap_equal_form_macro",
              "cluster_unit": "explicit_source_scene_group_else_image_sha256",
              "n_clusters": len(clusters), "n_rows": len(selected), "forms": list(forms),
              "image_fallback_clusters": sum(key.startswith("image:") for key in clusters),
              "observed_delta_pp": float((sums.sum(0) / counts.sum(0)).mean()),
              "confidence_level": 0.95, "replicates": replicates, "seed": seed,
              "ci_pp": None, "lower_bound_gt_zero": None,
              "training_randomness_covered": False, "multiplicity_adjusted": False,
              "result_blind_preregistered": False, "publication_inference_allowed": False}
    if len(clusters) < 2:
        return {**result, "status": "INSUFFICIENT_CLUSTERS"}
    draws = np.random.default_rng(seed).integers(len(clusters), size=(replicates, len(clusters)))
    draw_counts = counts[draws].sum(1)
    missing = int((draw_counts == 0).any(1).sum())
    result["replicates_missing_form_coverage"] = missing
    if missing:
        # Omitting or repeatedly redrawing these replicates would condition the
        # bootstrap on form coverage and silently change the declared design.
        return {**result, "status": "RESAMPLES_MISSING_FORM_COVERAGE"}
    boot = (sums[draws].sum(1) / draw_counts).mean(1)
    lower, upper = np.quantile(boot, [0.025, 0.975], method="linear")
    return {**result, "status": "EXPLORATORY_FIXED_CHECKPOINT_INTERVAL",
            "ci_pp": [float(lower), float(upper)], "lower_bound_gt_zero": bool(lower > 0)}


def native_paired_difference(source_a: list[dict], model_a: dict, source_b: list[dict], model_b: dict) -> dict:
    """Compare complete native targets; retain failed parents and single-reference limits."""
    def records(source, model):
        by_id = {row["sample_id"]: row for row in source}
        if len(by_id) != len(source) or len(model["items"]) != len(source):
            raise ValueError("native comparison source/item coverage mismatch")
        indexed, image_clusters = {}, {}
        for item in model["items"]:
            row = by_id[item["sample_id"]]
            identity = (row.get("gsa_eval_original_sample_id", row["sample_id"]),
                        row["gsa_meta"]["image_sha256"], row["gsa_form"])
            if identity in indexed or item["form"] != row["gsa_form"]:
                raise ValueError("duplicate or inconsistent native comparison identity")
            form, metric = item["form"], NATIVE_METRICS[item["form"]]
            if form in ("bbox", "detection", "mask", "point"):
                score = item["g_reference_scores"][metric]
            elif form == "depth":
                pairs = item["depth_pairs"]
                score = pairs["correct"] / pairs["total"] if pairs["total"] else 0.0
            else:
                score = item["a_reference_diagnostics"]["scores"][metric]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("native comparison score outside [0,1]")
            if item.get("generation_status") == "BLOCKED_UPSTREAM" and score != 0:
                raise ValueError("blocked native case received nonzero score")
            meta = row["gsa_meta"]
            scene = meta.get("scene_group")
            if scene is not None and (not isinstance(scene, str) or not scene.strip() or not meta.get("source")):
                raise ValueError("native cluster scene identity requires a source and nonempty group")
            # 'scene' is often a broad category such as 'embodied_home'; it is
            # not a physical-scene identifier and cannot be a resampling unit.
            cluster = ("scene:" + json.dumps([meta["source"], scene], separators=(",", ":"))
                       if scene is not None else "image:" + identity[1])
            if image_clusters.setdefault(identity[1], cluster) != cluster:
                raise ValueError("same native image has inconsistent cluster metadata")
            indexed[identity] = (float(score), row["gsa_payload"], cluster)
        return indexed

    left, right = records(source_a, model_a), records(source_b, model_b)
    if set(left) != set(right) or any(left[key][1:] != right[key][1:] for key in left):
        raise ValueError("native paired target/image coverage or reference payload drift")
    forms, rows = {}, []
    for identity in sorted(left):
        uid, image_hash, form = identity
        rows.append({"original_sample_id": uid, "image_sha256": image_hash, "form": form,
                     "cluster_id": left[identity][2],
                     "left": left[identity][0], "right": right[identity][0],
                     "delta_pp": 100 * (right[identity][0] - left[identity][0])})
    for form in sorted({row["form"] for row in rows}):
        selected = [row for row in rows if row["form"] == form]
        n = len(selected)
        forms[form] = {"metric": NATIVE_METRICS[form], "n": n,
                       "n_independent_images": len({row["image_sha256"] for row in selected}),
                       "left_percent": 100 * sum(row["left"] for row in selected) / n,
                       "right_percent": 100 * sum(row["right"] for row in selected) / n,
                       "delta_pp": sum(row["delta_pp"] for row in selected) / n,
                       "improved": sum(row["delta_pp"] > 1e-9 for row in selected),
                       "regressed": sum(row["delta_pp"] < -1e-9 for row in selected),
                       "unchanged": sum(abs(row["delta_pp"]) <= 1e-9 for row in selected),
                       "paired_cluster_interval": native_cluster_interval(selected)}
    g_forms = ("bbox", "detection", "mask", "point")
    g_macro = ({key: sum(forms[form][key] for form in g_forms) / 4
                for key in ("left_percent", "right_percent", "delta_pp")}
               if all(form in forms for form in g_forms) else None)
    if g_macro is not None:
        g_macro["paired_cluster_interval"] = native_cluster_interval(rows, forms=g_forms)
    return {"per_form": forms, "native_G_reference_macro": g_macro, "items": rows,
            "A_family_semantic_macro": None, "action_plan_semantic_score": None,
            "significance": None, "publication_inference_allowed": False}


def native_depth_context_availability(source: list[dict]) -> dict:
    """Describe reference entity availability without selecting or repairing targets."""
    from eval.g_family_qualification import _label

    items = []
    for row in source:
        if row["gsa_form"] != "depth":
            continue
        reference = [_label(item["label"]) for item in row["gsa_payload"]]
        context = row.get("gsa_provided_context", row["gsa_context"])
        provided = [_label(item.get("object_class")) for item in (context or {}).get("groundings", [])]
        unique = bool(provided) and all(provided) and len(set(provided)) == len(provided)
        items.append({"original_sample_id": row.get("gsa_eval_original_sample_id", row["sample_id"]),
                      "reference_labels": reference, "provided_labels": provided,
                      "missing_reference_labels": sorted(set(reference) - set(provided)),
                      "extra_provided_labels": sorted(set(provided) - set(reference)),
                      "provided_labels_unique_and_nonempty": bool(unique),
                      "exact_entity_set_available": bool(unique and set(reference) == set(provided)),
                      "blocked_upstream": bool(row.get("gsa_eval_blocked_upstream"))})
    return {"n": len(items), "exact_entity_set_available": sum(x["exact_entity_set_available"] for x in items),
            "items": items, "changes_scoring_denominator": False,
            "interpretation": "The full reference score includes upstream entity loss; it is not conditional S reasoning accuracy."}


def native_specialist_joint_comparisons(source_rows: dict, models: dict, roles: dict) -> dict:
    """Compare each specialist with the joint model from the same training arm.

    Refresh versus one's own initialization does not establish specialization.
    This audit exposes the paper's comparator and the still-unmet budget and
    evidence conditions without manufacturing a passing eligibility manifest.
    """
    results = {}
    for family in ("G", "S", "A"):
        for arm in ("initial", "control", "feedback"):
            specialist, joint = family + "_" + arm, "joint_" + arm
            if specialist not in models or joint not in models:
                continue
            comparisons = {variant: native_paired_difference(rows, models[joint][variant],
                                                             rows, models[specialist][variant])
                           for variant, rows in source_rows.items()}
            intervals = {}
            for variant, comparison in comparisons.items():
                score = (comparison["native_G_reference_macro"] if family == "G"
                         else comparison["per_form"].get("depth") if family == "S" else None)
                intervals[variant] = score.get("paired_cluster_interval") if score else None
            specialist_budget = roles.get(family, {}).get("budget", {}).get(arm, {})
            joint_budget = roles.get("joint", {}).get("budget", {}).get(arm, {})
            left_tokens = joint_budget.get("num_input_tokens_seen")
            right_tokens = specialist_budget.get("num_input_tokens_seen")
            token_counts_available = all(type(x) is int and x > 0 for x in (left_tokens, right_tokens))
            environments = [budget.get("environment", {}) for budget in (joint_budget, specialist_budget)]
            lora_keys = ("RANK", "ALPHA", "DROPOUT")
            lora_settings_match = (all(all(key in env for key in lora_keys) for env in environments)
                                   and all(environments[0][key] == environments[1][key] for key in lora_keys))
            required = {"oracle", "student_cascade", "corrupted"}
            lower_bounds_positive = (set(intervals) == required and
                                     all(x is not None and x["lower_bound_gt_zero"] is True
                                         for x in intervals.values()))
            results[specialist + "_minus_" + joint] = {
                "specialist": specialist, "comparator": joint, "family": family,
                "contexts": comparisons, "family_intervals": intervals,
                "all_three_exploratory_lower_bounds_positive": lower_bounds_positive,
                "refresh_budget_observation": {
                    "joint_input_tokens": left_tokens, "specialist_input_tokens": right_tokens,
                    "token_counts_available": token_counts_available,
                    "exact_input_token_counts_equal": token_counts_available and left_tokens == right_tokens,
                    "specialist_minus_joint_input_tokens": right_tokens - left_tokens if token_counts_available else None,
                    "lora_hyperparameters_match": lora_settings_match,
                    "trainable_parameter_count_match_verified": False,
                    "includes_initial_training_budget": False},
                "teacher_eligible": False, "publication_claim_eligible": False,
                "limitations": ["Fixed-checkpoint exploratory intervals do not cover training randomness or multiplicity.",
                                "Refresh token counts alone do not establish matched complete Teacher training budgets.",
                                "Image-cluster intervals cannot establish independence between unknown physical scenes.",
                                "A family semantic aggregate is unavailable; target-point geometry is not plan semantics.",
                                "S full-reference scores include upstream entity loss under changed contexts."]}
    return results


def native_code_snapshot(record: dict, snapshots: tuple[Path, ...]) -> Path:
    """Resolve changed source only to explicitly supplied, byte-identical code."""
    original = Path(record["path"])
    if original.suffix not in {".py", ".sh"}:
        raise ValueError("code snapshot cannot replace data, predictions or weights")
    for candidate in snapshots:
        if (candidate.is_file() and candidate.name == original.name and
                hashlib.sha256(candidate.read_bytes()).hexdigest() == record["sha256"]):
            return candidate.resolve()
    raise ValueError(f"native comparison artifact binding drift: {original.name}; exact source snapshot required")


def compile_native_refresh(refresh_root: Path, context_root: Path | None = None,
                           code_snapshots: tuple[Path, ...] = ()) -> dict:
    """Snapshot only completed, hash-bound runs; incomplete arms stay explicitly pending."""
    from eval.g_family_qualification import native_diagnostic, load_jsonl

    evidence, hashes, code_resolutions = {}, {}, {}

    def sha(path):
        path = Path(path).resolve()
        if path not in hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path] = digest.hexdigest()
        return hashes[path]

    def binding(record, *, code=False):
        path = Path(record["path"]).resolve()
        observed = sha(path) if path.is_file() else None
        if observed != record["sha256"]:
            if not code:
                raise ValueError(f"native comparison artifact binding drift: {path.name}")
            snapshot = native_code_snapshot(record, code_snapshots)
            code_resolutions[str(path)] = {"observed_current_sha256": observed,
                                          "snapshot_path": str(snapshot), "bound_sha256": record["sha256"]}
            path = snapshot
        evidence[str(path)] = record["sha256"]
        return path

    def read_receipt(path):
        raw = path.read_bytes()
        evidence[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def diagnostic(source, predictions):
        evidence[str(source)] = sha(source)
        for _, path in predictions:
            evidence[str(path)] = sha(path)
        return native_diagnostic(source, predictions)

    roles = {}
    for role in ("G", "S", "A", "joint"):
        completed, scores, receipts, states = {}, {}, {}, {}
        for arm, directory in (("control", role + "_control_r1"), ("feedback", role + "_feedback")):
            run = refresh_root / directory
            receipt_path = run / "launch.json"
            if not receipt_path.is_file():
                states[arm] = "MISSING_RECEIPT"
                continue
            receipt = read_receipt(receipt_path)
            states[arm] = receipt["status"]
            if receipt["status"] != "COMPLETED_DIAGNOSTIC":
                continue
            if receipt["role"] != role or receipt["arm"] != arm or not receipt["completed"]["selected_adapter_finite"]:
                raise ValueError("native refresh role/arm/completion mismatch")
            for name, record in receipt["files"].items():
                binding(record, code=name in {"driver", "evaluator", "contracts", "validators", "launcher"})
            for record in receipt["completed"]["persistent_selected_adapter"]["files"].values():
                binding(record)
            for record in receipt["initial_adapter"]["files"].values():
                binding(record)
            old_diagnostic = run / "qualification_diagnostic.json"
            prediction = run / "qualification_predictions.jsonl"
            binding({"path": str(old_diagnostic), "sha256": receipt["evaluation"]["diagnostic_sha256"]})
            binding({"path": str(prediction), "sha256": receipt["evaluation"]["predictions_sha256"]})
            saved = json.loads(old_diagnostic.read_text())
            source = binding(saved["source"])
            initial_prediction = binding(saved["models"]["initial"]["prediction"])
            current = diagnostic(source, [(arm, prediction), ("initial", initial_prediction)])
            completed[arm] = source
            scores[arm] = current["models"][arm]
            if "initial" in scores and scores["initial"] != current["models"]["initial"]:
                raise ValueError("paired arms use different initial model predictions")
            scores["initial"] = current["models"]["initial"]
            receipts[arm] = receipt
        comparisons, budget = {}, {}
        for arm, source in completed.items():
            rows = load_jsonl(source, "native comparison source")
            comparisons[arm + "_minus_initial"] = native_paired_difference(rows, scores["initial"], rows, scores[arm])
            receipt = receipts[arm]
            budget[arm] = {"steps": receipt["completed"]["steps"],
                           "num_input_tokens_seen": receipt["completed"]["num_input_tokens_seen"],
                           "effective_batch": receipt["effective_batch"], "train_rows": receipt["train_rows"],
                           "initial_weights_sha256": receipt["initial_adapter"]["files"]["selected_adapter_weights"]["sha256"],
                           "environment": {key: receipt["environment"][key] for key in
                                           ("SEED", "LR", "BS", "GA", "RANK", "ALPHA", "DROPOUT", "IMAGE_MAX_TOKEN_NUM")}}
        if set(completed) == {"control", "feedback"}:
            if completed["control"] != completed["feedback"] or sha(completed["control"]) != sha(completed["feedback"]):
                raise ValueError("paired refresh qualification source mismatch")
            for key in ("steps", "effective_batch", "initial_weights_sha256", "environment"):
                if budget["control"][key] != budget["feedback"][key]:
                    raise ValueError(f"paired refresh budget/initialization mismatch: {key}")
            rows = load_jsonl(completed["control"], "native comparison source")
            comparisons["feedback_minus_control"] = native_paired_difference(rows, scores["control"], rows, scores["feedback"])
        roles[role] = {"receipt_statuses": states, "budget": budget, "token_matched_experiment": False,
                       "oracle_models": {key: {"per_form": value["per_form"], "output_health": value["output_health"]}
                                         for key, value in scores.items()}, "comparisons": comparisons}

    contexts = {"status": "NOT_REQUESTED", "workers": {}, "comparisons": {}}
    if context_root:
        pack_path = context_root / "contexts/manifest.json"
        contexts["status"] = "CONTEXT_PACK_PENDING"
        if pack_path.is_file():
            manifest = read_receipt(pack_path)
            if manifest.get("status") != "COMPLETE":
                raise ValueError("native context manifest is not complete")
            for record in manifest["files"].values():
                binding(record)
            sources = {variant: context_root / f"contexts/{variant}.jsonl"
                       for variant in ("oracle", "student_cascade", "corrupted")}
            source_rows = {variant: load_jsonl(path, "native context source") for variant, path in sources.items()}
            models = {}
            for role in ("G", "S", "A", "joint"):
                for arm in ("control", "feedback"):
                    name = role + "_" + arm
                    path = context_root / name / "launch.json"
                    if not path.is_file():
                        contexts["workers"][name] = "MISSING_RECEIPT"
                        continue
                    receipt = read_receipt(path)
                    contexts["workers"][name] = receipt["status"]
                    if receipt["status"] != "COMPLETED_CONTEXT_DIAGNOSTICS":
                        continue
                    expected_candidates = {name, role + "_initial"} if arm == "control" else {name}
                    if receipt["role"] != role or receipt["arm"] != arm or set(receipt["candidates"]) != expected_candidates:
                        raise ValueError("context worker role/arm/candidate identity mismatch")
                    if receipt["context_manifest_sha256"] != sha(pack_path):
                        raise ValueError("context worker used a different context manifest")
                    for key, record in receipt["files"].items():
                        binding(record, code=key in {"driver", "evaluator", "contracts", "validators", "launcher"})
                    for model, adapter in receipt["candidates"].items():
                        for record in adapter["files"].values():
                            binding(record)
                        models[model] = {}
                        for variant, source in sources.items():
                            folder = path.parent / model
                            saved = read_receipt(folder / (variant + "_diagnostic.json"))
                            prediction = binding(saved["models"][model]["prediction"])
                            if binding(saved["source"]) != source.resolve():
                                raise ValueError("context diagnostic source differs from shared pack")
                            models[model][variant] = diagnostic(source, [(model, prediction)])["models"][model]
            contexts["status"] = "COMPLETE" if len(models) == 12 else "PARTIAL"
            contexts["variants"] = manifest["variants"]
            contexts["depth_reference_entity_availability"] = {
                variant: native_depth_context_availability(rows) for variant, rows in source_rows.items()}
            contexts["models"] = {name: {variant: {"per_form": score["per_form"], "output_health": score["output_health"],
                                          "provided_context_contracts": {
                                              form: {"n": sum(x["form"] == form for x in score["items"]),
                                                     "passed": sum(x["form"] == form and x.get("provided_context_contract_pass", False)
                                                                   for x in score["items"])}
                                              for form in {x["form"] for x in score["items"]}}}
                                          for variant, score in variants.items()} for name, variants in models.items()}
            for name, variants in models.items():
                contexts["comparisons"][name] = {variant + "_minus_oracle": native_paired_difference(
                    source_rows["oracle"], variants["oracle"], source_rows[variant], variants[variant])
                    for variant in ("student_cascade", "corrupted")}
            for role in ("G", "S", "A", "joint"):
                for left, right in (("initial", "control"), ("initial", "feedback"), ("control", "feedback")):
                    first, second = role + "_" + left, role + "_" + right
                    if first in models and second in models:
                        contexts["comparisons"][role + "_" + right + "_minus_" + left] = {
                            variant: native_paired_difference(source_rows[variant], models[first][variant],
                                                               source_rows[variant], models[second][variant])
                            for variant in sources}
            contexts["specialist_joint_comparisons"] = native_specialist_joint_comparisons(source_rows, models, roles)
    return {"schema": "anvil-native-refresh-comparison-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "roles": roles, "contexts": contexts, "evidence_sha256": evidence,
            "historical_code_resolutions": code_resolutions,
            "comparison_code_sha256": sha(Path(__file__)),
            "evaluator_sha256": sha(ROOT / "eval/g_family_qualification.py"),
            "teacher_eligible": False, "publication_claim_eligible": False,
            "limitations": ["One seed and a small historical qualification pilot; no global Teacher or SOTA claim.",
                            "Equal update/example budgets do not imply equal token budgets.",
                            "G uses bbox IoU, detection item F1, mask IoU and point hit rate, not full benchmark mAP.",
                            "A geometry uses one historical reference; alternative grasps and plan semantics are incomplete.",
                            "Current-stage targets remain hidden; upstream failures stay in every paired denominator.",
                            "Intervals resample explicit scene groups or image clusters at fixed checkpoints; no training-seed inference.",
                            "Exploratory intervals were added after observing these pilot scores; no preregistered significance claim.",
                            "Receipt status is an artifact observation, not a live-process health check."]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+",
                    default=["where2place", "refspatial_bench"])
    ap.add_argument("--tags", nargs="+", default=["base", "sft_9b_lora"])
    ap.add_argument("--out", default=str(ROOT / "eval_results" / "compare_models.json"))
    ap.add_argument("--native-refresh-root", type=Path)
    ap.add_argument("--native-context-root", type=Path)
    ap.add_argument("--native-code-snapshot", type=Path, action="append", default=[],
                    help="Exact original code file for a receipt whose working source has since changed; repeat as needed")
    a = ap.parse_args()

    if a.native_refresh_root:
        out = Path(a.out)
        if out.exists():
            ap.error("native comparison requires a fresh output")
        report = compile_native_refresh(a.native_refresh_root.resolve(),
                                        a.native_context_root.resolve() if a.native_context_root else None,
                                        tuple(a.native_code_snapshot))
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"roles": {role: value["receipt_statuses"] for role, value in report["roles"].items()},
                          "contexts": report["contexts"]["status"], "publication_claim_eligible": False}, indent=2))
        return 0

    report = {"tags": a.tags, "benchmarks": {}}
    for b in a.benchmarks:
        if b not in BENCHMARKS:
            print(f"unknown benchmark {b}", file=sys.stderr)
            return 2
        if get_spec(b).scorer is not score_point_in_mask:
            print(f"skip {b}: not a point-in-mask benchmark")
            continue
        print(f"=== {b} ===", flush=True)
        r = run_point_bench(b, a.tags)
        report["benchmarks"][b] = r
        print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
