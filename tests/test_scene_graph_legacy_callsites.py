import json
import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from apps.data_flywheel.config import FlywheelConfig
from apps.data_flywheel.pipeline import run_pipeline
from apps.data_flywheel.scene_understanding import builder as builder_module


def test_pipeline_uses_new_build_scene_graph(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "scene.jpg"
    Image.new("RGB", (96, 64), color="white").save(image_path)

    output_root = tmp_path / "output"
    calls = []
    real_build_scene_graph = builder_module.build_scene_graph

    def tracking_build_scene_graph(*args, **kwargs):
        calls.append(
            {
                "image_id": kwargs["image_id"],
                "image_size": kwargs["image_size"],
                "skip_phase_b": kwargs.get("skip_phase_b"),
            }
        )
        return real_build_scene_graph(*args, **kwargs)

    monkeypatch.setattr(builder_module, "build_scene_graph", tracking_build_scene_graph)

    def fake_run_grounding(*, image_caption_pairs, output_jsonl, **kwargs):
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        records = []
        for image_file, _caption in image_caption_pairs:
            records.append(
                {
                    "image": image_file,
                    "width": 96,
                    "height": 64,
                    "phrases": [
                        {
                            "phrase": "book",
                            "boxes": [
                                {"bbox": [8, 16, 36, 40], "confidence": 0.95},
                                {"bbox": [52, 16, 84, 40], "confidence": 0.93},
                            ],
                        }
                    ],
                }
            )
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"count": len(records), "quality_stats": {}}

    def fake_generate_spatial_qa(*, output_jsonl, **kwargs):
        Path(output_jsonl).write_text("", encoding="utf-8")
        return {"count": 0, "type_counts": {}}

    def fake_export_dataset(*, output_jsonl, **kwargs):
        Path(output_jsonl).write_text("", encoding="utf-8")
        return {"total": 0, "stats": {}, "output_jsonl": output_jsonl}

    grounding_module = ModuleType("apps.data_flywheel.grounding")
    grounding_module.run_grounding = fake_run_grounding
    monkeypatch.setitem(sys.modules, "apps.data_flywheel.grounding", grounding_module)

    spatial_qa_module = ModuleType("apps.data_flywheel.spatial_qa")
    spatial_qa_module.generate_spatial_qa = fake_generate_spatial_qa
    monkeypatch.setitem(sys.modules, "apps.data_flywheel.spatial_qa", spatial_qa_module)

    exporter_module = ModuleType("apps.data_flywheel.exporter")
    exporter_module.export_dataset = fake_export_dataset
    monkeypatch.setitem(sys.modules, "apps.data_flywheel.exporter", exporter_module)

    cfg = FlywheelConfig(
        image_dirs=[str(image_dir)],
        output_root=str(output_root),
        rounds=1,
        max_images_per_round=1,
        use_api=True,
        enable_augment=False,
        enable_caption=False,
        enable_grounding=True,
        enable_referring=False,
        enable_pointing=False,
        enable_eval=False,
        enable_hard_mining=False,
        enable_scene_graph=True,
        enable_depth_estimation=False,
        enable_ontology_bootstrap=False,
        relative_paths=False,
    )

    report = run_pipeline(cfg)

    assert report["rounds"] == 1
    assert calls == [
        {
            "image_id": str(image_path),
            "image_size": (96, 64),
            "skip_phase_b": False,
        }
    ]

    scene_graph_jsonl = output_root / "round_0" / "scene_graph.jsonl"
    lines = scene_graph_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["image"] == str(image_path)
    assert len(record["objects"]) == 2
    assert any(relation["predicate"] == "left-of" for relation in record["relations"])
