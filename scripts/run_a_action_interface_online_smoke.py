"""Run a minimal online smoke test for A_action_interface.

This script intentionally avoids the full GSA runner. It checks only:
- VLM endpoint/key are usable.
- AActionInterfacePipeline can produce valid pick/push interface labels.

The script loads .env from the repo root if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.data_flywheel.gsa.a_action_interface_pipeline import AActionInterfacePipeline
from apps.data_flywheel.gsa.a_pipeline_schemas import validate_action_interface
from apps.data_flywheel.gsa.visual_evidence import GsaEvidenceBundle, GsaObjectRef
from apps.data_flywheel.workers.vlm_client import VLMClient


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="tests/fixtures/desk_sample.jpg")
    parser.add_argument("--object-name", default="red block")
    parser.add_argument("--object-id", default="red_block_0")
    parser.add_argument("--bbox", nargs=4, type=float, default=[8.0, 34.0, 55.0, 83.0])
    parser.add_argument(
        "--base-url",
        required=True,
    )
    parser.add_argument("--model", default="MLLM")
    return parser.parse_args()


def main() -> int:
    args = _args()
    _load_env(ROOT / ".env")

    api_key = os.environ.get("VLM_API_KEY")
    if not api_key:
        print("ONLINE_SMOKE_FAILED: VLM_API_KEY is not set")
        return 2

    vlm = VLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=90,
        max_retries=0,
    )
    bundle = GsaEvidenceBundle(
        sample_id="online_smoke_a_action_interface",
        image_path=args.image,
        scene_family="tabletop_tidy",
        scenario_id="SC.2",
        objects=[
            GsaObjectRef(
                object_id=args.object_id,
                object_name=args.object_name.replace(" ", "_"),
                raw_object_name=args.object_name,
                bbox=args.bbox,
                source_provenance="vlm",
            )
        ],
    )

    labels = AActionInterfacePipeline(vlm).run(bundle)
    result = []
    for label in labels:
        validation = validate_action_interface(label)
        d = label.to_dict()
        result.append(
            {
                "action_type": d.get("action_type"),
                "subtype": d.get("subtype"),
                "source_provenance": d.get("source_provenance"),
                "verifier_status": d.get("verifier_status"),
                "grasp_point_2d": d.get("grasp_point_2d"),
                "jaw_axis_2d": d.get("jaw_axis_2d"),
                "push_contact_point_2d": d.get("push_contact_point_2d"),
                "push_direction_2d": d.get("push_direction_2d"),
                "valid": validation.valid,
                "errors": validation.errors,
                "missing_fields": validation.missing_fields,
            }
        )

    print("ONLINE_A_ACTION_INTERFACE_SMOKE_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not labels or not all(item["valid"] for item in result):
        return 1
    if not any(item["source_provenance"] == "vlm" for item in result):
        print("ONLINE_SMOKE_WARNING: all labels used heuristic fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
