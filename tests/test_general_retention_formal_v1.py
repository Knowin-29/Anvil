from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import general_retention_formal_v3 as formal  # noqa: E402


def _identity(value: dict) -> dict:
    value = dict(value)
    value["identity_sha256"] = formal.content_hash(value)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _jpeg_bytes(colour: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 6), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _fake_model(root: Path, name: str, weights: bytes) -> Path:
    model = root / name
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["FixtureVLForConditionalGeneration"],
                "model_type": "fixture_vl",
                "vision_config": {"model_type": "fixture_vision"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(weights)
    (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return model


def _fake_nvidia_smi(root: Path) -> Path:
    script = root / "nvidia-smi-fixture"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *--query-gpu=*)
    printf '%s\n' \
      '0, NVIDIA L20, 46068, GPU-fixture-0' \
      '1, NVIDIA L20, 46068, GPU-fixture-1' \
      '2, NVIDIA L20, 46068, GPU-fixture-2' \
      '3, NVIDIA L20, 46068, GPU-fixture-3'
    ;;
  *--query-compute-apps=*) ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _failing_nvidia_smi(root: Path) -> Path:
    script = root / "nvidia-smi-failing"
    script.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def _malformed_process_nvidia_smi(root: Path) -> Path:
    script = root / "nvidia-smi-malformed-process"
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *--query-gpu=*)
    printf '%s\n' \
      '0, NVIDIA L20, 46068, GPU-fixture-0' \
      '1, NVIDIA L20, 46068, GPU-fixture-1' \
      '2, NVIDIA L20, 46068, GPU-fixture-2' \
      '3, NVIDIA L20, 46068, GPU-fixture-3'
    ;;
  *--query-compute-apps=*) printf '%s\n' 'malformed-row-without-columns' ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fixture_protocol(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    samples: list[dict] = []
    adapter_images: list[dict] = []
    crosswalk_datasets: list[dict] = []
    dataset_entries: list[dict] = []
    colours = {
        "ai2d_test": (210, 30, 30),
        "scienceqa_test": (30, 180, 30),
        "textvqa_val": (30, 30, 210),
    }
    specifications = {
        "ai2d_test": ("multiple_choice_accuracy", 32),
        "scienceqa_test": ("multiple_choice_accuracy", 32),
        "textvqa_val": ("official_vqa_soft_accuracy", 64),
    }
    for dataset in formal.DATASET_ORDER:
        raw = _jpeg_bytes(colours[dataset])
        image_sha = hashlib.sha256(raw).hexdigest()
        image_path = source_dir / f"{dataset}.jpg"
        image_path.write_bytes(raw)
        primary_image_id = f"{dataset}-image"
        adapter_image_id = (
            f"ai2d/images/{primary_image_id}.png"
            if dataset == "ai2d_test"
            else primary_image_id
        )
        adapter_images.append(
            {
                "dataset": dataset,
                "adapter_image_id": adapter_image_id,
                "primary_image_id": primary_image_id,
                "primary_image_sha256": image_sha,
                "sha256": image_sha,
                "path": str(image_path.resolve()),
                "mapping_method": "exact_bytes",
            }
        )
        tsv = source_dir / f"{dataset}.tsv"
        if dataset == "ai2d_test":
            fieldnames = ["index", "question", "A", "B", "answer", "abcLabel", "image_path", "image"]
        elif dataset == "scienceqa_test":
            fieldnames = ["index", "question", "answer", "hint", "A", "B", "image"]
        else:
            fieldnames = ["index", "image", "question", "answer"]
        row_bindings: list[dict] = []
        with tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            first_index = ""
            for offset in range(4):
                index = str(1000 * (formal.DATASET_ORDER.index(dataset) + 1) + offset)
                if offset == 0:
                    first_index = index
                uid = f"{dataset}-uid-{offset}"
                question = f"{dataset} fixture question {offset}?"
                if dataset == "textvqa_val":
                    answer: str | list[str] = ["cat"] * 10
                    choices = None
                    image_value = base64.b64encode(raw).decode() if offset == 0 else first_index
                    writer.writerow(
                        {
                            "index": index,
                            "image": image_value,
                            "question": question,
                            "answer": repr(answer),
                        }
                    )
                else:
                    answer = "A"
                    choices = ["correct", "wrong"]
                    row = {
                        "index": index,
                        "question": question,
                        "answer": answer,
                        "A": choices[0],
                        "B": choices[1],
                        "image": base64.b64encode(raw).decode(),
                    }
                    if dataset == "ai2d_test":
                        row.update(
                            {
                                "abcLabel": "false",
                                "image_path": adapter_image_id,
                            }
                        )
                    else:
                        row["hint"] = "fixture hint"
                    writer.writerow(row)
                primary = {
                    "dataset": dataset,
                    "adapter_index": index,
                    "uid": uid,
                    "image_id": primary_image_id,
                    "question": question,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "answer": answer,
                    "choices": choices,
                    "source_image_sha256": image_sha,
                }
                if dataset == "ai2d_test":
                    primary.update(
                        {
                            "expected_adapter_image_path": adapter_image_id,
                            "abc_label": False,
                            "upstream_is_abc": False,
                        }
                    )
                samples.append(primary)
                row_bindings.append(
                    {
                        "adapter_index": index,
                        "primary_uid": uid,
                        "primary_image_id": primary_image_id,
                        "adapter_image_id": adapter_image_id,
                        "adapter_image_sha256": image_sha,
                        "primary_image_sha256": image_sha,
                        "mapping_method": "exact_bytes",
                        "primary_abc_label": primary.get("abc_label"),
                        "upstream_is_abc": primary.get("upstream_is_abc"),
                        "image_reference_index": (
                            first_index if dataset == "textvqa_val" and offset > 0 else None
                        ),
                    }
                )
        digest = hashlib.sha256()
        for row in row_bindings:
            digest.update(formal.canonical_json(row).encode("utf-8") + b"\n")
        crosswalk_datasets.append(
            {
                "dataset": dataset,
                "rows": 4,
                "unique_adapter_images": 1,
                "adapter_image_reference_rows": 3 if dataset == "textvqa_val" else 0,
                "row_mapping_sha256": digest.hexdigest(),
            }
        )
        metric, max_tokens = specifications[dataset]
        dataset_entries.append(
            {
                "id": dataset,
                "display_name": dataset,
                "split": "fixture",
                "expected_n": 4,
                "tsv_path": str(tsv.resolve()),
                "md5": formal.file_hash(tsv, "md5"),
                "sha256": formal.file_hash(tsv),
                "metric": metric,
                "parser": "fixture",
                "max_tokens": max_tokens,
            }
        )

    sample_path = source_dir / "samples.jsonl"
    adapter_path = source_dir / "adapter_images.jsonl"
    _write_jsonl(sample_path, samples)
    _write_jsonl(adapter_path, adapter_images)
    crosswalk = _identity(
        {
            "schema_version": 1,
            "suite_id": formal.SUITE_ID,
            "state": "PASS",
            "datasets": crosswalk_datasets,
        }
    )
    crosswalk_path = source_dir / "crosswalk.json"
    _write_json(crosswalk_path, crosswalk)
    source = _identity(
        {
            "schema_version": 1,
            "suite_id": formal.SUITE_ID,
            "state": "SOURCE_AND_ADAPTER_FROZEN",
            "total_samples": len(samples),
            "total_unique_adapter_images": len(adapter_images),
            "sample_manifest_sha256": formal.file_hash(sample_path),
            "adapter_image_manifest_sha256": formal.file_hash(adapter_path),
            "crosswalk_identity_sha256": crosswalk["identity_sha256"],
        }
    )
    source_path = source_dir / "source_manifest.json"
    _write_json(source_path, source)
    leakage = _identity(
        {
            "schema_version": 1,
            "suite_id": formal.SUITE_ID,
            "state": "PASS",
            "training_fingerprints_ok": 1,
            "training_unique_image_paths": 1,
            "training_fingerprint_errors": 0,
            "preflight_errors": [],
            "exact_image_collisions": 0,
            "transformation_normalized_exact_image_collisions": 0,
            "dataset_qualified_source_id_collisions": 0,
            "near_duplicate_image_collisions_unresolved": 0,
            "image_question_collisions": 0,
            "source_manifest": {
                "path": str(source_path.resolve()),
                "file_sha256": formal.file_hash(source_path),
                "identity_sha256": source["identity_sha256"],
            },
        }
    )
    leakage_path = source_dir / "leakage.json"
    _write_json(leakage_path, leakage)
    vqa_reference = ROOT / "data/general_retention_v1/sources/vlmevalkit/vqa_eval.py"
    config = {
        "schema_version": 1,
        "protocol_id": formal.PROTOCOL,
        "suite_id": formal.SUITE_ID,
        "test_fixture": True,
        "source_evidence": {
            "source_manifest_path": str(source_path.resolve()),
            "source_manifest_identity_sha256": source["identity_sha256"],
            "sample_manifest_path": str(sample_path.resolve()),
            "adapter_image_manifest_path": str(adapter_path.resolve()),
            "crosswalk_path": str(crosswalk_path.resolve()),
            "crosswalk_identity_sha256": crosswalk["identity_sha256"],
            "leakage_report_path": str(leakage_path.resolve()),
            "leakage_identity_sha256": leakage["identity_sha256"],
            "vqa_reference_path": str(vqa_reference.resolve()),
            "vqa_reference_sha256": formal.file_hash(vqa_reference),
        },
        "generation": {
            "engine": "swift_transformers_engine",
            "logical_shards": 4,
            "shard_rule": "stable_runtime_index_stride_4",
            "hardware": "exactly_4_unique_physical_L20_at_least_44_GiB",
            "batch_size_per_gpu": 16,
            "dataset_batch_sizes": {
                "ai2d_test": 16,
                "scienceqa_test": 16,
                "textvqa_val": 1,
            },
            "image_max_token_num": 4096,
            "oom_policy": "clear_cache_then_bisect_to_singleton_or_fail",
            "temperature": 0.0,
            "thinking": False,
            "device_map_per_worker": "cuda:0",
        },
        "health_admission": {
            "malformed_response_credit": 0.0,
            "max_malformed_percent": 5.0,
            "max_truncation_percent": 1.0,
            "raw_scores_always_reported_after_complete_protocol_valid_run": True,
            "general_avg_requires_all_cells_admitted": True,
        },
        "aggregate": {
            "name": "General Avg",
            "rule": "unweighted_macro_average_of_three_benchmark_percentages",
            "no_partial_aggregate": True,
        },
        "datasets": dataset_entries,
    }
    config_path = tmp_path / "fixture_protocol.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    base_model = _fake_model(tmp_path, "base-model", b"base fixture weights")
    e1_model = _fake_model(tmp_path, "e1-model", b"e1 fixture weights")
    return config_path, base_model, e1_model


def _bound_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config, base_model, e1_model = _fixture_protocol(tmp_path)
    hardware = formal.query_l20_hardware(
        ["0", "1", "2", "3"], str(_fake_nvidia_smi(tmp_path)), require_idle=True
    )
    runtime_index = tmp_path / "run" / "runtime.jsonl"
    receipt_path = tmp_path / "run" / "receipt.json"
    formal.atomic_json(
        receipt_path,
        formal.build_input_receipt(
            config,
            runtime_index,
            hardware_inventory=hardware,
            allow_test_fixture=True,
        ),
    )
    base_binding = tmp_path / "run" / "base.json"
    e1_binding = tmp_path / "run" / "e1.json"
    formal.atomic_json(
        base_binding,
        formal.bind_model(
            "base", base_model, receipt_path, allow_test_fixture=True
        ),
    )
    formal.atomic_json(
        e1_binding,
        formal.bind_model("e1", e1_model, receipt_path, allow_test_fixture=True),
    )
    return receipt_path, base_binding, e1_binding


def _complete_fake_role(
    tmp_path: Path, receipt: Path, binding: Path, role: str
) -> tuple[list[Path], Path, dict]:
    shard_paths: list[Path] = []

    def fake(public: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        assert all(set(item) == {"messages", "images"} for item in public)
        serialised = json.dumps(public)
        assert "source_image_sha256" not in serialised
        assert '"answer"' not in serialised
        response = "cat" if max_tokens == 64 else "A"
        return [(response, False) for _ in public]

    for shard in range(4):
        path = tmp_path / "run" / role / f"shard_{shard}.jsonl"
        formal.infer_shard(
            receipt_path=receipt,
            binding_path=binding,
            shard=shard,
            num_shards=4,
            batch_size=16,
            out=path,
            infer_public=fake,
            allow_test_fixture=True,
        )
        generated = formal.load_jsonl(path, "generated fixture shard")
        assert all(row["max_batch_size_per_gpu"] == 16 for row in generated)
        assert all(row["image_max_token_num"] == 4096 for row in generated)
        assert all(row["oom_policy"] == formal.OOM_POLICY for row in generated)
        assert all(
            row["batch_size_per_gpu"]
            == formal.DATASET_BATCH_SIZES[row["dataset"]]
            for row in generated
        )
        shard_paths.append(path)
    result_path = tmp_path / "run" / f"{role}-result.json"
    result = formal.score_predictions(
        receipt_path=receipt,
        binding_path=binding,
        shard_paths=shard_paths,
        out=result_path,
        allow_test_fixture=True,
    )
    return shard_paths, result_path, result


@pytest.mark.parametrize("text, expected", [
    ("A", "A"), ("answer: B", "B"), ("cat", "A"),
    ("A. cat", "A"), ("(B) dog", "B"), ("B) dog", "B"),
    ("B: DOG.", "B"), ("<think>\n\n</think>\nD. 405,410 km", "D"),
    ("A. dog", None), ("Z. cat", None), ("A. cat or B. dog", None),
    ("A. cat because it has ears", None), ("<think>A. cat", None),
    ("A. ca", None), ("A. category", None), ("B. cat", None),
])
def test_mc_compat_accepts_exact_presentation_only(text: str, expected: str | None) -> None:
    from eval.general_retention_mc_compat import parse_multiple_choice
    assert parse_multiple_choice(text, ["cat", "dog", "horse", "405,410 km"])[0] == expected


def test_mc_compat_is_audited_posthoc_and_preserves_failed_original(tmp_path: Path) -> None:
    from eval.general_retention_mc_compat import audit_pair
    receipt, base_binding, e1_binding = _bound_fixture(tmp_path)
    _, base_path, _ = _complete_fake_role(tmp_path, receipt, base_binding, "base")
    shards = []
    for shard in range(4):
        path = tmp_path / "compat" / f"shard_{shard}.jsonl"
        formal.infer_shard(
            receipt_path=receipt, binding_path=e1_binding, shard=shard,
            num_shards=4, batch_size=16, out=path,
            infer_public=lambda public, cap: [("cat" if cap == 64 else "A. correct", False) for _ in public],
            allow_test_fixture=True,
        )
        shards.append(path)
    candidate_path = tmp_path / "compat" / "e1_result.json"
    original = formal.score_predictions(receipt_path=receipt, binding_path=e1_binding,
                                        shard_paths=shards, out=candidate_path, allow_test_fixture=True)
    assert original["status"] == "SCORED_HEALTH_NOT_ADMITTED"
    before = (base_path.read_bytes(), candidate_path.read_bytes())
    audited = audit_pair(base_path, candidate_path, allow_test_fixture=True)
    assert audited["may_populate_formal_paper_metrics"] is False
    assert audited["candidate"]["original_status"] == "SCORED_HEALTH_NOT_ADMITTED"
    assert audited["candidate"]["changed_count"] == 8
    assert audited["compat_raw_retention_percent"] == pytest.approx(100)
    assert before == (base_path.read_bytes(), candidate_path.read_bytes())
    with shards[0].open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(formal.FormalProtocolError, match="drift"):
        audit_pair(base_path, candidate_path, allow_test_fixture=True)


def test_reference_aware_loader_fake_engine_and_paired_scorer(tmp_path: Path) -> None:
    receipt, base_binding, e1_binding = _bound_fixture(tmp_path)
    receipt_value = formal.verify_input_receipt(
        receipt, deep_sources=True, allow_test_fixture=True
    )
    text_receipt = next(
        item for item in receipt_value["sources"]["datasets"] if item["id"] == "textvqa_val"
    )
    assert text_receipt["rows"] == 4
    assert text_receipt["image_reference_rows"] == 3

    _, base_result_path, base = _complete_fake_role(
        tmp_path, receipt, base_binding, "base"
    )
    _, e1_result_path, e1 = _complete_fake_role(tmp_path, receipt, e1_binding, "e1")
    for result in (base, e1):
        assert result["status"] == "FORMAL_SCORE_ADMITTED"
        assert result["general_avg_percent"] == pytest.approx(100.0)
        assert all(cell["malformed_count"] == 0 for cell in result["benchmarks"].values())
        assert all(cell["health_admitted"] for cell in result["benchmarks"].values())
    paired_path = tmp_path / "run" / "paired.json"
    paired = formal.compare_results(
        base_result_path,
        e1_result_path,
        paired_path,
        allow_test_fixture=True,
    )
    assert paired["status"] == "PAIRED_GENERAL_RETENTION_ADMITTED"
    assert formal.verify_comparison(
        paired_path, require_admitted=True, allow_test_fixture=True
    ) == paired
    assert paired["raw_general_avg_delta_points"] == pytest.approx(0.0)
    assert all(cell["paired_regressed"] == 0 for cell in paired["benchmarks"].values())

    forged = json.loads(base_result_path.read_text(encoding="utf-8"))
    forged["benchmarks"]["ai2d_test"]["raw_score_percent"] = 99.0
    forged.pop("identity_sha256")
    forged["identity_sha256"] = formal.content_hash(forged)
    forged_path = tmp_path / "run" / "forged-self-consistent-result.json"
    _write_json(forged_path, forged)
    with pytest.raises(formal.FormalProtocolError, match="do not recompute"):
        formal._verify_result(forged_path, allow_test_fixture=True)


def test_malformed_is_zero_health_is_separate_and_shards_fail_closed(tmp_path: Path) -> None:
    receipt, base_binding, _ = _bound_fixture(tmp_path)

    def malformed(public: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        del max_tokens
        return [("<think>unfinished", False) for _ in public]

    shards = []
    for shard in range(4):
        path = tmp_path / "bad" / f"shard_{shard}.jsonl"
        formal.infer_shard(
            receipt_path=receipt,
            binding_path=base_binding,
            shard=shard,
            num_shards=4,
            batch_size=16,
            out=path,
            infer_public=malformed,
            allow_test_fixture=True,
        )
        shards.append(path)
    result = formal.score_predictions(
        receipt_path=receipt,
        binding_path=base_binding,
        shard_paths=shards,
        out=tmp_path / "bad" / "result.json",
        allow_test_fixture=True,
    )
    assert result["status"] == "SCORED_HEALTH_NOT_ADMITTED"
    assert result["raw_general_avg_percent"] == 0.0
    assert result["general_avg_percent"] is None
    assert all(cell["malformed_response_credit"] == 0.0 for cell in result["benchmarks"].values())

    rows = formal.load_jsonl(shards[0], "fixture shard")
    legacy = [dict(row) for row in rows]
    legacy[0]["protocol"] = "general-retention-formal-v2"
    legacy_path = tmp_path / "bad" / "legacy-v2-shard.jsonl"
    _write_jsonl(legacy_path, legacy)
    with pytest.raises(formal.FormalProtocolError, match="provenance/order drift"):
        formal.verify_shard(
            receipt_path=receipt,
            binding_path=base_binding,
            shard_path=legacy_path,
            expected_shard=0,
            allow_test_fixture=True,
        )

    rows[0]["global_uid"] = "tampered"
    shards[0].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    with pytest.raises(formal.FormalProtocolError, match="provenance/order drift"):
        formal.verify_shard(
            receipt_path=receipt,
            binding_path=base_binding,
            shard_path=shards[0],
            expected_shard=0,
            allow_test_fixture=True,
        )


def test_launcher_is_exact_four_l20_and_preflight_only_is_cpu() -> None:
    launcher = ROOT / "scripts/run_general_retention_dp.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = launcher.read_text(encoding="utf-8")
    runner_text = (ROOT / "eval/general_retention_formal_v3.py").read_text(encoding="utf-8")
    assert "count != 4" in text
    assert '"L20" not in gpu["name"].upper()' in runner_text
    assert "MIG" in runner_text
    assert "four unique physical devices" in runner_text
    assert "BATCH=16" in text
    assert "TextVQA batch1" in text
    assert "IMAGE_MAX_TOKEN_NUM=4096" in text
    assert 'RUN_STAGE="${RUN_STAGE:-paired}"' in text
    assert 'if [[ "$RUN_STAGE" == base ]]' in text
    assert "--require-admitted" in text
    assert "PREFLIGHT_ONLY" in text
    assert "no CUDA context or model process was started" in text
    assert "CUDA_VISIBLE_DEVICES=\"${GPU_IDS[$shard]}\"" in text
    assert "--gpu-selector" in text
    assert "--live-hardware --require-idle-hardware" in text
    assert "test_fixture configs are forbidden by the formal API" in runner_text


def test_cuda_oom_bisects_preserves_order_and_singleton_fails(monkeypatch) -> None:
    class FakeCudaOOM(RuntimeError):
        pass

    cleaned: list[bool] = []
    monkeypatch.setattr(formal, "_clear_cuda_after_oom", lambda: cleaned.append(True))

    class SplitEngine:
        def infer(self, requests, _config):
            if len(requests) > 1:
                raise FakeCudaOOM("CUDA out of memory")
            return [f"ok:{requests[0]}"]

    requests = ["r0", "r1", "r2", "r3"]
    result = formal._adaptive_engine_infer(
        SplitEngine(), requests, ["u0", "u1", "u2", "u3"], object(),
        initial_batch_size=4,
    )
    assert result == ["ok:r0", "ok:r1", "ok:r2", "ok:r3"]
    assert cleaned

    class SingletonOOMEngine:
        def infer(self, _requests, _config):
            raise FakeCudaOOM("CUDA out of memory")

    with pytest.raises(formal.FormalProtocolError, match="no sample was skipped: fatal-uid"):
        formal._adaptive_engine_infer(
            SingletonOOMEngine(), ["one"], ["fatal-uid"], object(),
            initial_batch_size=1,
        )


def test_base_wait_wrapper_is_fresh_fail_closed_and_base_only(tmp_path: Path) -> None:
    wrapper = ROOT / "scripts/queue_general_retention_base_l20_4gpu.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(wrapper)], text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = wrapper.read_text(encoding="utf-8")
    assert "hardware-status" in text
    assert "HARDWARE_QUERY_FAILED" in text
    assert '[[ ! -e "$OUT_BASE" ]]' in text
    assert '[[ ! -e "$QUEUE_DIR" ]]' in text
    assert 'setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_STAGE=base' in text
    assert 'kill -TERM -- "-$CURRENT_PGID"' in text
    assert 'grep -Fq "BASE_ONLY_COMPLETE:"' in text
    assert '"$OUT/results/base_vs_e1.json"' in text
    assert '"$OUT/predictions/e1"' in text
    assert "--require-admitted" in text
    assert "E1 not queued" in text
    assert "process_group_alive" in text
    assert "HARDWARE_RACE_REQUEUE" in text
    assert "RUNTIME_HARDWARE_INTRUSION_REQUEUE" in text
    assert "FOREIGN_COMPUTE_CONTEXTS" in text
    assert 'pathlib.Path(f"/proc/{pid}/stat")' in text
    assert 'OUT="${OUT_BASE}.retry${attempt}"' in text

    common_env = {
        "GSA_ROOT": str(ROOT),
        "RUN_TAG": "fixture_idle",
        "OUT": str(tmp_path / "out-idle"),
        "QUEUE_ROOT": str(tmp_path / "queues-idle"),
        "QUEUE_LOCK": str(tmp_path / "idle.lock"),
        "QUEUE_PREFLIGHT_ONLY": "1",
        "POLL_SECONDS": "0",
        "STABILITY_SECONDS": "0",
        "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(tmp_path)),
    }
    idle = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, **common_env},
        text=True,
        capture_output=True,
        check=False,
    )
    assert idle.returncode == 0, idle.stderr
    assert "QUEUE_PREFLIGHT_READY" in idle.stdout
    assert not (tmp_path / "out-idle").exists()

    failing_env = {
        **common_env,
        "RUN_TAG": "fixture_query_error",
        "OUT": str(tmp_path / "out-error"),
        "QUEUE_ROOT": str(tmp_path / "queues-error"),
        "QUEUE_LOCK": str(tmp_path / "error.lock"),
        "NVIDIA_SMI_BIN": str(_failing_nvidia_smi(tmp_path)),
    }
    failed = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, **failing_env},
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 4
    assert "HARDWARE_QUERY_FAILED" in failed.stdout
    assert not (tmp_path / "out-error").exists()

    with pytest.raises(
        formal.FormalProtocolError, match="unexpected nvidia-smi compute-process row"
    ):
        formal.query_l20_hardware(
            ["0", "1", "2", "3"],
            str(_malformed_process_nvidia_smi(tmp_path)),
            require_idle=False,
        )


def test_base_wait_wrapper_records_stubbed_admitted_base_success(tmp_path: Path) -> None:
    wrapper = ROOT / "scripts/queue_general_retention_base_l20_4gpu.sh"
    project = tmp_path / "project"
    (project / "eval").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / ".env").write_text("\n", encoding="utf-8")
    (project / "eval/general_retention_formal_v1.py").write_text("\n", encoding="utf-8")

    fake_python = project / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c ]]; then
  payload=$(cat)
  [[ "$payload" == *HARDWARE_IDLE* ]] && echo HARDWARE_IDLE && exit 0
  [[ "$payload" == *HARDWARE_BUSY* ]] && echo HARDWARE_BUSY && exit 0
  exit 9
fi
case "${2:-}" in
  hardware-status) printf '%s\\n' '{"status":"HARDWARE_IDLE"}' ;;
  verify-result) printf '%s\\n' '{"status":"FORMAL_RESULT_VERIFIED"}' ;;
  *) exit 9 ;;
esac
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_launcher = project / "scripts/run_general_retention_dp.sh"
    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[[ "${RUN_STAGE:-}" == base ]]
mkdir -p "$OUT/results"
printf '{}\\n' > "$OUT/results/base.json"
echo "BASE_ONLY_COMPLETE: admitted fixture base; E1 was not queued or started"
""",
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)

    out = project / "fresh-out"
    queue_root = project / "queue"
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": str(fake_python),
            "RUN_TAG": "stubbed_admitted_base",
            "OUT": str(out),
            "QUEUE_ROOT": str(queue_root),
            "QUEUE_LOCK": str(project / "queue.lock"),
            "POLL_SECONDS": "0",
            "STABILITY_SECONDS": "0",
            "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(tmp_path)),
            "MONITOR_PYTHON_BIN": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "base\tSUCCEEDED\t0" in completed.stdout
    assert "base admitted; E1 not queued" in completed.stdout
    assert (out / "results/base.json").is_file()
    assert not (out / "results/e1.json").exists()
    assert not (out / "results/base_vs_e1.json").exists()

    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
nohup sleep 60 >/dev/null 2>&1 &
printf '%s\\n' "$!" > "$OUT/leaked.pid"
mkdir -p "$OUT/results"
printf '{}\\n' > "$OUT/results/base.json"
echo "BASE_ONLY_COMPLETE: launcher exited while a child remained"
""",
        encoding="utf-8",
    )
    leaked_out = project / "leaked-child-out"
    leaked = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": str(fake_python),
            "RUN_TAG": "stubbed_leaked_child",
            "OUT": str(leaked_out),
            "QUEUE_ROOT": str(queue_root),
            "QUEUE_LOCK": str(project / "queue.lock"),
            "POLL_SECONDS": "0",
            "STABILITY_SECONDS": "0",
            "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(tmp_path)),
            "MONITOR_PYTHON_BIN": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert leaked.returncode == 6
    leaked_pid = int((leaked_out / "leaked.pid").read_text(encoding="utf-8"))
    for _ in range(20):
        try:
            os.kill(leaked_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(leaked_pid, 9)
        pytest.fail(f"wrapper left child process alive: {leaked_pid}")

    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$OUT" != *.retry1 ]]; then
  echo "FATAL: selected GPUs have compute processes: fixture race" >&2
  exit 2
fi
mkdir -p "$OUT/results"
printf '{}\\n' > "$OUT/results/base.json"
echo "BASE_ONLY_COMPLETE: admitted after a fresh hardware-race retry"
""",
        encoding="utf-8",
    )
    race_out = project / "hardware-race-out"
    retried = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": str(fake_python),
            "RUN_TAG": "stubbed_hardware_race",
            "OUT": str(race_out),
            "QUEUE_ROOT": str(queue_root),
            "QUEUE_LOCK": str(project / "queue.lock"),
            "POLL_SECONDS": "0",
            "STABILITY_SECONDS": "0",
            "NVIDIA_SMI_BIN": str(_fake_nvidia_smi(tmp_path)),
            "MONITOR_PYTHON_BIN": sys.executable,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert retried.returncode == 0, retried.stderr
    assert "HARDWARE_RACE_REQUEUE" in retried.stdout
    assert race_out.is_dir()
    assert not (race_out / "results/base.json").exists()
    assert (project / "hardware-race-out.retry1/results/base.json").is_file()


def test_base_wait_wrapper_requeues_runtime_foreign_gpu_context(tmp_path: Path) -> None:
    wrapper = ROOT / "scripts/queue_general_retention_base_l20_4gpu.sh"
    project = tmp_path / "project"
    (project / "eval").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / ".env").write_text("\n", encoding="utf-8")
    (project / "eval/general_retention_formal_v1.py").write_text(
        "\n", encoding="utf-8"
    )

    fake_python = project / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c ]]; then
  payload=$(cat)
  [[ "$payload" == *HARDWARE_IDLE* ]] && echo HARDWARE_IDLE && exit 0
  exit 9
fi
case "${2:-}" in
  hardware-status) printf '%s\\n' '{"status":"HARDWARE_IDLE"}' ;;
  verify-result) printf '%s\\n' '{"status":"FORMAL_RESULT_VERIFIED"}' ;;
  *) exit 9 ;;
esac
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_launcher = project / "scripts/run_general_retention_dp.sh"
    fake_launcher.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$OUT" != *.retry1 ]]; then
  sleep 30
  exit 8
fi
mkdir -p "$OUT/results"
printf '{}\\n' > "$OUT/results/base.json"
echo "BASE_ONLY_COMPLETE: admitted after runtime intrusion retry"
""",
        encoding="utf-8",
    )
    fake_launcher.chmod(0o755)

    marker = project / "foreign-context-emitted"
    fake_nvidia = project / "nvidia-smi-runtime-intrusion"
    fake_nvidia.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ! -e "$RUNTIME_MONITOR_MARKER" ]]; then
  : > "$RUNTIME_MONITOR_MARKER"
  printf '%s\\n' 'GPU-external, 999999999, [Not Found], 1024'
fi
""",
        encoding="utf-8",
    )
    fake_nvidia.chmod(0o755)

    out = project / "runtime-race-out"
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env={
            **os.environ,
            "GSA_ROOT": str(project),
            "PYTHON_BIN": str(fake_python),
            "MONITOR_PYTHON_BIN": sys.executable,
            "NVIDIA_SMI_BIN": str(fake_nvidia),
            "RUNTIME_MONITOR_MARKER": str(marker),
            "RUN_TAG": "stubbed_runtime_intrusion",
            "OUT": str(out),
            "QUEUE_ROOT": str(project / "queue"),
            "QUEUE_LOCK": str(project / "queue.lock"),
            "POLL_SECONDS": "0",
            "STABILITY_SECONDS": "0",
            "RUNTIME_MONITOR_SECONDS": "1",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "RUNTIME_HARDWARE_INTRUSION_REQUEUE" in completed.stdout
    reason = project / "queue/stubbed_runtime_intrusion/runtime_monitor_attempt_0.tsv"
    assert reason.read_text(encoding="utf-8") == (
        "FOREIGN_COMPUTE_CONTEXTS\ttotal=1\tforeign=1\n"
    )
    assert out.is_dir()
    assert not (out / "results/base.json").exists()
    assert (project / "runtime-race-out.retry1/results/base.json").is_file()


def test_official_textvqa_fractional_parity_and_normalization() -> None:
    reference = ROOT / "data/general_retention_v1/sources/vlmevalkit/vqa_eval.py"
    digit_article = formal._reference_digit_article(reference)
    expected = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9, 4: 1.0}
    for matching, score in expected.items():
        answers = ["yes"] * matching + ["no"] * (10 - matching)
        assert formal._vqa_soft_credit("yes", answers, digit_article) == pytest.approx(score)
    assert formal._vqa_normalize("The two cats!", digit_article) == "2 cats"
    assert formal._vqa_normalize("dont", digit_article) == "don't"
    assert formal._vqa_normalize("1,000.", digit_article) == "1000"


def test_fixture_gate_forward_reference_and_content_only_model_identity(tmp_path: Path) -> None:
    config_path, _, _ = _fixture_protocol(tmp_path)
    hardware = formal.query_l20_hardware(
        ["0", "1", "2", "3"], str(_fake_nvidia_smi(tmp_path)), require_idle=True
    )
    with pytest.raises(formal.FormalProtocolError, match="test_fixture configs are forbidden"):
        formal.build_input_receipt(
            config_path,
            tmp_path / "forbidden-runtime.jsonl",
            hardware_inventory=hardware,
        )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    text_entry = next(row for row in config["datasets"] if row["id"] == "textvqa_val")
    tsv = Path(text_entry["tsv_path"])
    with tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    rows[1]["image"] = rows[2]["index"]
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    text_entry["md5"] = formal.file_hash(tsv, "md5")
    text_entry["sha256"] = formal.file_hash(tsv)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        formal.FormalProtocolError,
        match="(?:differs from|absent from) adapter manifest",
    ):
        formal.build_input_receipt(
            config_path,
            tmp_path / "forward-runtime.jsonl",
            hardware_inventory=hardware,
            allow_test_fixture=True,
        )

    first = _fake_model(tmp_path, "same-content-a", b"identical weights")
    second = _fake_model(tmp_path, "same-content-b", b"identical weights")
    first_inventory = formal.model_inventory(first)
    second_inventory = formal.model_inventory(second)
    assert first_inventory["identity_sha256"] != second_inventory["identity_sha256"]
    assert (
        first_inventory["checkpoint_content_identity_sha256"]
        == second_inventory["checkpoint_content_identity_sha256"]
    )


def test_runtime_image_drift_is_rejected_before_fake_or_real_engine(tmp_path: Path) -> None:
    receipt, base_binding, _ = _bound_fixture(tmp_path)
    receipt_value = formal.verify_input_receipt(
        receipt, allow_test_fixture=True
    )
    runtime = formal.load_jsonl(
        Path(receipt_value["runtime_index"]["path"]), "fixture runtime"
    )
    image = Path(runtime[0]["image_path"])
    image.write_bytes(b"post-preflight drift")
    called = False

    def must_not_run(public: list[dict], max_tokens: int) -> list[tuple[str, bool]]:
        nonlocal called
        called = True
        return [("A", False) for _ in public]

    with pytest.raises(formal.FormalProtocolError, match="runtime image drift"):
        formal.infer_shard(
            receipt_path=receipt,
            binding_path=base_binding,
            shard=0,
            num_shards=4,
            batch_size=16,
            out=tmp_path / "must-not-exist.jsonl",
            infer_public=must_not_run,
            allow_test_fixture=True,
        )
    assert called is False
