"""Reconcile original corpus, supplemental QA, and the paper score denominator.

Read-only audit: no training data, predictions, scorer, or manuscript is rewritten.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT.parents[2] / "arxiv/native_training_export_20261012_v4_r2"
PROJECT = Path("/Knowin/demo/weilinruan/Anvil")
COLLECT = Path("/Knowin/demo/weilinruan/GSA_collect")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    export = json.loads((EXPORT / "manifest.json").read_text())
    db = sqlite3.connect("file:" + export["pool_index"] + "?mode=ro", uri=True)
    selected_images = {h for (h,) in db.execute(
        "SELECT DISTINCT i.decoded_sha256 FROM images i JOIN records r USING(image_id) JOIN selected s USING(rid)")}
    assert len(selected_images) == 160000 and export["records"] == 5220000
    qa_path = EXPORT / "train_QA.jsonl"
    assert sha(qa_path) == export["files"][qa_path.name]["sha256"]
    qa_sources = Counter()
    for line in qa_path.open():
        row = json.loads(line)
        assert row["gsa_meta"]["split"] == "train"
        qa_sources[row["gsa_meta"]["binding"]["source"]] += 1
    assert sum(qa_sources.values()) == 464000
    replay_path = ROOT / "retention_replay/manifest.json"
    replay = json.loads(replay_path.read_text())
    extra_images = set()
    extra_sources = {}
    for name, count in replay["additional_training_records"].items():
        file = replay_path.parent / (name + ".jsonl")
        assert sha(file) == replay["files_sha256"][name]
        rows = [json.loads(line) for line in file.open()]
        assert len(rows) == count
        images = {r["gsa_meta"]["binding"]["decoded_rgb_sha256"] for r in rows}
        extra_sources[name] = dict(records=count, distinct_images=len(images),
                                  images_in_original_corpus=len(images & selected_images))
        extra_images.update(images)
    sources = PROJECT / "02_Paper/arxiv/tables/scene_source.tex"
    extra_count = sum(x["records"] for x in extra_sources.values())
    scope = dict(schema="anvil-original-and-supplement-scale-audit-v1",
        original_full_corpus=dict(records=5800000, images=len(selected_images)),
        original_sft=dict(records=export["records"], images=export["unique_images"],
                          spatial=4756000, QA=464000, qa_sources=dict(qa_sources)),
        original_spatial_holdout=580000,
        supplemental_qa=dict(records=extra_count, sources=extra_sources,
                             distinct_images=len(extra_images),
                             new_images=len(extra_images - selected_images)),
        original_plus_supplement_available_union=dict(training_records=5220000 + extra_count,
            full_corpus_records=5800000 + extra_count,
            full_corpus_images=len(selected_images | extra_images)),
        unique_exposed_training_records="Compute from each run's exposure logs; pool size is not actual coverage",
        report_source_mixture_matched=False,
        discrepancy="Paper lists multiple QA source families; original 464000 QA rows are all CLEVR. Supplemental AI2D/ScienceQA/TextVQA are separately sourced and do not establish that mixture.",
        original_data_modified=False, heldout_answers_used_for_training=False,
        bindings={str(p): sha(p) for p in [EXPORT / "manifest.json", replay_path, sources, Path(__file__)]})
    (ROOT / "data_scale_reconciliation.json").write_text(json.dumps(scope, indent=2) + "\n")

    comparison_path = ROOT / "benchmark_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    registry = COLLECT / "experiments/result_registry.yaml"
    historical_path = COLLECT / "eval_results/qwen3_5_9b_base_full/refspatial_bench.json"
    historical = json.loads(historical_path.read_text())
    assert historical["detail"]["per_group"]["location"]["n"] == 100
    aligned = {}
    for role, results in comparison["results"].items():
        values = {name: entry["score"] for name, entry in results["public"].items()}
        ref = results["public"]["refspatial_bench"]
        assert ref["n"] == 277 and ref["detail"]["per_group"]["location"]["n"] == 100
        values["refspatial_bench"] = ref["detail"]["per_group"]["location"]["mean_in_mask"]
        aligned[role] = dict(scores=values, macro=sum(values.values()) / 7,
            refspatial_full277=ref["score"],
            full277_macro_preserved=results["public_macro"])
    audit = dict(schema="anvil-paper-denominator-audit-v1", results=aligned,
        refspatial_paper_value_path="detail.per_group.location.mean_in_mask",
        refspatial_paper_n=100, refspatial_full_diagnostic_n=277,
        subgroup_selected_from_prior_registry=True,
        historical_base_location=historical["detail"]["per_group"]["location"]["mean_in_mask"],
        paper_targets=comparison["paper_main_table_target"],
        exact_paper_protocol_equivalence_established=False,
        remaining_differences=["Historical registry batch16 vs diagnostic batch8",
            "Recheck image preprocessing, model revision and inference engine against historical receipts",
            "Released README states main-table final checkpoint is not yet bound; seven target cells are not a verified single-model receipt"],
        predictions_or_scorer_modified=False,
        bindings={str(p): sha(p) for p in [comparison_path, registry, historical_path,
            PROJECT / "01_Anvil_arxiv/README.md", PROJECT / "02_Paper/arxiv/tables/eval_results.tex", Path(__file__)]})
    (ROOT / "paper_protocol_reconciliation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(dict(data_union=scope["original_plus_supplement_available_union"],
                         original_qa_sources=dict(qa_sources), aligned=aligned), indent=2))


if __name__ == "__main__":
    main()
