"""Offline probe for the A_action_interface v0 plan.

This module only reads external datasets. It does not download data, invoke VLM
services, unzip full archives, or mutate NAS / teammate worktrees.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..a_action_interface_contract import (
    ACTION_INTERFACE_SCHEMA_VERSION,
    LABEL_STATUS_CANDIDATE,
    action_interface_schema,
    verify_action_interface_candidate,
)
from ..taxonomy import SourceProvenance, VerifierStatus
from .paco_lvis_adapter import load_paco_annotations


WORKTREE_ROOT = Path(__file__).resolve().parents[4]
NAS_ROOT = Path("/path/to/datasets")
BRIDGE_ROOT = Path("/path/to/environment/backup_420/bridgedatav2")


@dataclass
class OfflineCandidate:
    """A candidate row for action-interface planning, not a stable label."""

    source_dataset: str
    sample_id: str
    candidate_kind: str
    action_type: str
    label_type: str
    source_provenance: str
    verifier_status: str
    stable_training_label: bool
    schema_version: str = ACTION_INTERFACE_SCHEMA_VERSION
    label_status: str = LABEL_STATUS_CANDIDATE
    source_fields: List[str] = field(default_factory=list)
    target_interface_fields: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    prompt: Optional[str] = None
    image_path: Optional[str] = None
    image_ref: Optional[str] = None
    video_ref: Optional[str] = None
    frame_refs: List[str] = field(default_factory=list)
    object_id: Optional[str] = None
    object_class: Optional[str] = None
    bbox: Optional[List[float]] = None
    point_2d: Optional[List[float]] = None
    grasp_point_2d: Optional[List[float]] = None
    jaw_axis_2d: Optional[List[float]] = None
    push_contact_point_2d: Optional[List[float]] = None
    push_direction_2d: Optional[List[float]] = None
    place_region_2d: Optional[Dict[str, Any]] = None
    coordinate_frame: Optional[str] = None
    region_2d: Optional[Dict[str, Any]] = None
    promotion_eligible: bool = False
    promotion_blockers: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}


@dataclass
class DatasetProbe:
    source_dataset: str
    priority: str
    recommended_use: str
    status: str
    fields_available: List[str]
    risk_flags: List[str]
    records: List[OfflineCandidate] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["records"] = [r.to_dict() for r in self.records]
        return d


def run_offline_probe(max_samples: int = 20) -> Dict[str, Any]:
    """Run all read-only probes and return a serializable report."""
    probes = [
        probe_paco_lvis(max_samples=max_samples),
        probe_molmo2_vp(max_samples=max_samples),
        probe_fsd_dataset(max_samples=max_samples),
        probe_bridge_data(max_samples=max_samples),
        probe_deferred_sources(),
    ]
    _finalize_probe_records(probes)
    records = [r for p in probes for r in p.records]
    return {
        "title": "A_action_interface offline probe v1",
        "schema_version": ACTION_INTERFACE_SCHEMA_VERSION,
        "max_samples_per_dataset": max_samples,
        "write_boundary": str(WORKTREE_ROOT),
        "datasets": [p.to_dict() for p in probes],
        "candidate_count": len(records),
        "stable_training_label_count": sum(1 for r in records if r.stable_training_label),
        "label_status_counts": _count_values(records, "label_status"),
        "verifier_status_counts": _count_values(records, "verifier_status"),
        "promotion_eligible_count": sum(1 for r in records if r.promotion_eligible),
        "verification_error_count": sum(
            len(r.verification.get("errors", [])) for r in records
        ),
    }


def probe_paco_lvis(max_samples: int = 20) -> DatasetProbe:
    ds_path = NAS_ROOT / "001_done_P1_A0_paco-lvis"
    data = load_paco_annotations(ds_path, split="val") if ds_path.exists() else None
    if not data:
        return DatasetProbe(
            source_dataset="PACO-LVIS",
            priority="P1 A0",
            recommended_use="defer until PACO JSON is readable",
            status="unavailable",
            fields_available=[],
            risk_flags=["paco_json_unreadable"],
            notes="Expected raw/paco_lvis_v1.zip with paco_lvis_v1_val.json.",
        )

    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat for cat in data.get("categories", [])}
    records: List[OfflineCandidate] = []
    for ann in data.get("annotations", [])[:max_samples]:
        image = images.get(ann.get("image_id"), {})
        category = categories.get(ann.get("category_id"), {})
        bbox = _xywh_to_xyxy(ann.get("bbox"))
        records.append(
            OfflineCandidate(
                source_dataset="PACO-LVIS",
                sample_id=f"paco_lvis_{ann.get('id')}",
                candidate_kind="region_support",
                action_type="pick",
                label_type="region",
                source_provenance=SourceProvenance.PUBLIC_DATASET.value,
                verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                stable_training_label=False,
                source_fields=[
                    "image_id",
                    "file_name",
                    "bbox",
                    "segmentation",
                    "category_id",
                    "attribute_ids",
                ],
                target_interface_fields=["affordance_region", "place_region_2d_support"],
                risk_flags=[
                    "not_native_grasp_label",
                    "region_support_only",
                    "image_inside_coco_zip_or_external_ref",
                ],
                image_ref=image.get("file_name") or image.get("coco_url"),
                object_id=f"paco_{ann.get('id')}",
                object_class=category.get("name", "unknown"),
                bbox=bbox,
                region_2d=_bbox_polygon(bbox),
                notes="Use as G_context / affordance-region support, not as grasp-point truth.",
            )
        )

    return DatasetProbe(
        source_dataset="PACO-LVIS",
        priority="P1 A0",
        recommended_use="bbox/mask/region support for A_action_interface probes",
        status="readable_annotation_only",
        fields_available=[
            "image_ref",
            "object_class",
            "bbox",
            "segmentation",
            "attribute_ids",
        ],
        risk_flags=[
            "no_native_grasp_point_2d",
            "no_jaw_axis_2d",
            "no_push_direction_2d",
            "no_place_region_2d",
        ],
        records=records,
        notes="PACO validates region plumbing but cannot produce stable action-interface labels by itself.",
    )


def probe_molmo2_vp(max_samples: int = 20) -> DatasetProbe:
    parquet_path = (
        NAS_ROOT
        / "038_done_P0_molmo2-vp"
        / "raw/hf/allenai__Molmo2-VideoPoint/spatial reference/train-00000-of-00001.parquet"
    )
    if not parquet_path.exists():
        return DatasetProbe(
            source_dataset="Molmo2-VP",
            priority="P0 Eval",
            recommended_use="defer; parquet file missing",
            status="unavailable",
            fields_available=[],
            risk_flags=["parquet_missing"],
        )
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return DatasetProbe(
            source_dataset="Molmo2-VP",
            priority="P0 Eval",
            recommended_use="point-coordinate schema audit after pyarrow is available",
            status="dependency_missing",
            fields_available=["parquet_exists"],
            risk_flags=["pyarrow_missing"],
            notes="Use /path/to/environment/env/rex/bin/python in this environment.",
        )

    table = pq.ParquetFile(parquet_path).read_row_group(0).slice(0, max_samples)
    records: List[OfflineCandidate] = []
    for idx, row in enumerate(table.to_pylist()):
        point = _first_molmo_point(row.get("points"))
        records.append(
            OfflineCandidate(
                source_dataset="Molmo2-VP",
                sample_id=f"molmo2_vp_{idx}",
                candidate_kind="point_coordinate_support",
                action_type="pick",
                label_type="point",
                source_provenance=SourceProvenance.PUBLIC_DATASET.value,
                verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                stable_training_label=False,
                source_fields=[
                    "video_id",
                    "question",
                    "label",
                    "points",
                    "raw_frames",
                    "raw_timestamps",
                    "clip_start",
                    "clip_end",
                ],
                target_interface_fields=["point_2d_schema_support"],
                risk_flags=[
                    "not_robot_action_point",
                    "coordinate_frame_percent_like_unconfirmed",
                    "video_context_required",
                ],
                prompt=row.get("question"),
                video_ref=row.get("video_id"),
                point_2d=point,
                coordinate_frame="dataset_points_percent_like",
                notes="Use for point parsing and range checks, not stable grasp labels.",
            )
        )

    return DatasetProbe(
        source_dataset="Molmo2-VP",
        priority="P0 Eval",
        recommended_use="point coordinate parsing and verifier tests",
        status="readable_metadata_and_generated_video_assets",
        fields_available=[
            "video_id",
            "question",
            "label",
            "points",
            "raw_frames",
            "raw_timestamps",
        ],
        risk_flags=[
            "not_action_interface_truth",
            "video_point_eval_not_training_source",
        ],
        records=records,
        notes="Good for proving point-field handling without claiming manipulation supervision.",
    )


def probe_fsd_dataset(max_samples: int = 20) -> DatasetProbe:
    ds_root = NAS_ROOT / "004_done_P1_A0_fsd-dataset/raw/IffYuan__FSD-Dataset"
    archive_specs = [
        (ds_root / "openx_without_video.tar.gz", "openx_without_video"),
        (ds_root / "droid_wo_video.tar.gz", "droid_wo_video"),
    ]
    records: List[OfflineCandidate] = []
    missing = []
    for archive_path, kind in archive_specs:
        if not archive_path.exists():
            missing.append(str(archive_path))
            continue
        records.extend(_read_fsd_archive(archive_path, kind, max_samples - len(records)))
        if len(records) >= max_samples:
            break

    status = "readable_task_frames" if records else "unavailable"
    return DatasetProbe(
        source_dataset="FSD-dataset",
        priority="P1 A0",
        recommended_use="robot task frame candidates for later teacher labeling",
        status=status,
        fields_available=[
            "start_frame",
            "middle_frame",
            "end_frame",
            "lang_or_current_task",
            "metadata",
        ],
        risk_flags=[
            "no_native_grasp_point_2d",
            "no_native_push_direction_2d",
            "teacher_or_human_needed_for_geometry",
        ] + (["expected_archives_missing"] if missing else []),
        records=records,
        notes="Frame triplets are useful for candidates; any action-interface label generated later must be VLM/human provenance and review-gated.",
    )


def probe_bridge_data(max_samples: int = 20) -> DatasetProbe:
    data_root = BRIDGE_ROOT / "data"
    image_list = data_root / "images.txt"
    prompt_list = data_root / "prompt.txt"
    video_list = data_root / "videos.txt"
    if not (image_list.exists() and prompt_list.exists()):
        return DatasetProbe(
            source_dataset="BridgeDataV2",
            priority="external worktree reference",
            recommended_use="defer; BridgeDataV2 lists missing",
            status="unavailable",
            fields_available=[],
            risk_flags=["bridge_lists_missing"],
        )

    records: List[OfflineCandidate] = []
    image_rows = _read_lines(image_list)
    prompt_rows = _read_lines(prompt_list)
    video_rows = _read_lines(video_list) if video_list.exists() else []
    for idx, image_rel in enumerate(image_rows[:max_samples]):
        image_path = data_root / image_rel
        records.append(
            OfflineCandidate(
                source_dataset="BridgeDataV2",
                sample_id=Path(image_rel).stem,
                candidate_kind="robot_pick_place_candidate",
                action_type=_infer_action_type(prompt_rows[idx] if idx < len(prompt_rows) else ""),
                label_type="point_or_region",
                source_provenance=SourceProvenance.PUBLIC_DATASET.value,
                verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                stable_training_label=False,
                source_fields=["image_path", "video_path", "prompt"],
                target_interface_fields=[
                    "grasp_point_2d_candidate",
                    "push_contact_point_2d_candidate",
                    "place_region_2d_candidate",
                ],
                risk_flags=[
                    "no_native_2d_action_geometry",
                    "requires_teacher_or_human_label",
                    "read_only_external_worktree",
                ],
                prompt=prompt_rows[idx] if idx < len(prompt_rows) else "",
                image_path=str(image_path),
                video_ref=video_rows[idx] if idx < len(video_rows) else None,
                notes="Best scene source for later VLM/human action-interface labeling.",
            )
        )

    return DatasetProbe(
        source_dataset="BridgeDataV2",
        priority="high value auxiliary source",
        recommended_use="main source for later robot action-interface teacher labeling",
        status="readable_image_prompt_video_lists",
        fields_available=["image_path", "video_ref", "language_task"],
        risk_flags=[
            "outside_contributor_worktree_read_only",
            "no_native_action_point_labels",
        ],
        records=records,
        notes="Use as read-only source; write derived manifests only inside contributor worktree.",
    )


def probe_deferred_sources() -> DatasetProbe:
    return DatasetProbe(
        source_dataset="Deferred sources",
        priority="blocked",
        recommended_use="risk documentation only",
        status="not_used_in_v0_probe",
        fields_available=[],
        risk_flags=[
            "VLM-3R_missing_mp4_media",
            "PACO_EGO4D_missing_frames",
            "PixMo_Points_url_only_no_local_images",
            "TaskGrasp_AGD20K_access_pending",
        ],
        records=[],
        notes="Do not include these in the first stable probe until media/access issues are resolved.",
    )


def write_probe_outputs(report: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Write report artifacts, enforcing contributor worktree as the write boundary."""
    output_dir = output_dir.resolve()
    root = WORKTREE_ROOT.resolve()
    if output_dir != root and root not in output_dir.parents:
        raise ValueError(f"Output path must stay under {root}: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "dataset_summary.json"
    manifest_path = output_dir / "candidate_manifest.jsonl"
    report_path = output_dir / "a_action_interface_probe_report.md"
    schema_path = output_dir / "a_action_interface_schema_v1.json"
    preview_path = output_dir / "candidate_preview.html"

    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(action_interface_schema(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        for ds in report["datasets"]:
            for record in ds.get("records", []):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    report_path.write_text(render_markdown_report(report), encoding="utf-8")
    preview_path.write_text(render_html_preview(report), encoding="utf-8")
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "report": report_path,
        "schema": schema_path,
        "preview": preview_path,
    }


def render_markdown_report(report: Dict[str, Any]) -> str:
    rows = []
    field_rows = []
    risk_rows = []
    status_rows = []
    sample_rows = []
    for ds in report["datasets"]:
        rows.append(
            "| {source_dataset} | {priority} | {status} | {recommended_use} | {count} |".format(
                source_dataset=ds["source_dataset"],
                priority=ds["priority"],
                status=ds["status"],
                recommended_use=ds["recommended_use"],
                count=len(ds.get("records", [])),
            )
        )
        field_rows.append(
            "| {source_dataset} | {fields} |".format(
                source_dataset=ds["source_dataset"],
                fields=", ".join(ds.get("fields_available", [])) or "-",
            )
        )
        target_fields = sorted({
            field_name
            for record in ds.get("records", [])
            for field_name in record.get("target_interface_fields", [])
        })
        status_counts = _count_values_from_dicts(ds.get("records", []), "label_status")
        status_rows.append(
            "| {source_dataset} | {target_fields} | {status_counts} |".format(
                source_dataset=ds["source_dataset"],
                target_fields=", ".join(target_fields) or "-",
                status_counts=", ".join(f"{k}:{v}" for k, v in sorted(status_counts.items())) or "-",
            )
        )
        risk_rows.append(
            "| {source_dataset} | {risks} |".format(
                source_dataset=ds["source_dataset"],
                risks=", ".join(ds.get("risk_flags", [])) or "-",
            )
        )
        for record in ds.get("records", [])[:3]:
            sample_rows.append(
                "| {dataset} | {sample_id} | {action_type} | {label_status} | {blockers} | {prompt} |".format(
                    dataset=ds["source_dataset"],
                    sample_id=record.get("sample_id", "-"),
                    action_type=record.get("action_type", "-"),
                    label_status=record.get("label_status", "-"),
                    blockers=", ".join(record.get("promotion_blockers", [])[:4]) or "-",
                    prompt=_md_cell((record.get("prompt") or record.get("notes") or "")[:120]),
                )
            )

    return "\n".join([
        "# A_action_interface Offline Probe Report",
        "",
        f"- Write boundary: `{report['write_boundary']}`",
        f"- Schema version: `{report['schema_version']}`",
        f"- Max samples per dataset: `{report['max_samples_per_dataset']}`",
        f"- Candidate records: `{report['candidate_count']}`",
        f"- Stable training labels: `{report['stable_training_label_count']}`",
        f"- Promotion eligible records: `{report['promotion_eligible_count']}`",
        f"- Verification errors: `{report['verification_error_count']}`",
        f"- Label status counts: `{report['label_status_counts']}`",
        f"- Verifier status counts: `{report['verifier_status_counts']}`",
        "",
        "## Dataset Priority",
        "",
        "| Dataset | Priority | Probe status | Recommended use | Records |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## Field Mapping",
        "",
        "| Dataset | Available fields |",
        "|---|---|",
        *field_rows,
        "",
        "## Interface Field Support",
        "",
        "| Dataset | Target interface fields | Label status counts |",
        "|---|---|---|",
        *status_rows,
        "",
        "## Risks",
        "",
        "| Dataset | Risk flags |",
        "|---|---|",
        *risk_rows,
        "",
        "## Candidate Preview",
        "",
        "| Dataset | Sample | Action | Label status | Promotion blockers | Prompt/notes |",
        "|---|---|---|---|---|---|",
        *sample_rows,
        "",
        "## A_action_interface v1 Contract",
        "",
        "- `pick`: requires `grasp_point_2d` and `jaw_axis_2d`.",
        "- `push`: requires `push_contact_point_2d` and `push_direction_2d`.",
        "- `place`: requires `place_region_2d`.",
        "- Offline probe records default to `label_status=candidate` and `stable_training_label=false`.",
        "- VLM teacher output must use `label_status=teacher_generated` until reviewed.",
        "- Training use requires `human_verified` or `stable_training_label` status plus valid geometry.",
        "",
        "## Sub-agent Split",
        "",
        "- Agent A: compare legacy `A_action` with `A_action_interface` and list mixed fields to avoid.",
        "- Agent B: audit data completeness and media risk for candidate datasets.",
        "- Agent C: validate schema, coordinate ranges, provenance, and verifier status.",
        "- Agent D: build read-only adapters and small candidate manifests.",
        "- Agent E: consolidate report and next implementation tasks.",
        "",
    ]) + "\n"


def render_html_preview(report: Dict[str, Any]) -> str:
    """Render a compact review preview for candidate records."""
    cards = []
    for ds in report["datasets"]:
        for record in ds.get("records", [])[:10]:
            geometry = {
                key: record.get(key)
                for key in (
                    "bbox",
                    "point_2d",
                    "grasp_point_2d",
                    "jaw_axis_2d",
                    "push_contact_point_2d",
                    "push_direction_2d",
                    "region_2d",
                    "place_region_2d",
                )
                if record.get(key) is not None
            }
            media = record.get("image_path") or record.get("image_ref") or record.get("video_ref") or "-"
            cards.append(
                f"""
                <section class="card">
                  <div class="topline"><b>{_html(ds["source_dataset"])}</b><span>{_html(record.get("label_status", "-"))}</span></div>
                  <div class="sample">{_html(record.get("sample_id", "-"))} · {_html(record.get("action_type", "-"))}</div>
                  <div class="media">{_html(media)}</div>
                  {_render_svg_geometry(record)}
                  <pre>{_html(json.dumps(geometry, ensure_ascii=False, indent=2))}</pre>
                  <div class="risk">{_html(", ".join(record.get("promotion_blockers", [])) or "no blockers")}</div>
                  <p>{_html(record.get("prompt") or record.get("notes") or "")}</p>
                </section>
                """
            )
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>A_action_interface candidate preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f8fafc; }}
    h1 {{ font-size: 22px; margin-bottom: 4px; }}
    .summary {{ color: #52606d; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; }}
    .topline {{ display: flex; justify-content: space-between; gap: 12px; font-size: 14px; }}
    .sample {{ color: #334e68; margin-top: 6px; font-size: 13px; }}
    .media {{ color: #627d98; margin: 8px 0; font-size: 12px; word-break: break-all; }}
    .sketch {{ width: 100%; height: 160px; background: #f0f4f8; border: 1px solid #d9e2ec; border-radius: 4px; margin: 8px 0; }}
    .empty {{ display: flex; align-items: center; justify-content: center; color: #829ab1; font-size: 12px; }}
    pre {{ background: #102a43; color: #d9e2ec; padding: 8px; border-radius: 4px; overflow: auto; font-size: 12px; }}
    .risk {{ color: #9f1239; font-size: 12px; margin-bottom: 8px; }}
    p {{ font-size: 13px; line-height: 1.35; }}
  </style>
</head>
<body>
  <h1>A_action_interface Candidate Preview</h1>
  <div class="summary">Schema: {schema} · candidates: {candidates} · stable labels: {stable}</div>
  <div class="grid">{cards}</div>
</body>
</html>
""".format(
        schema=_html(str(report.get("schema_version", "-"))),
        candidates=int(report.get("candidate_count", 0)),
        stable=int(report.get("stable_training_label_count", 0)),
        cards="\n".join(cards),
    )


def _xywh_to_xyxy(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _bbox_polygon(bbox: Optional[List[float]]) -> Optional[Dict[str, Any]]:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "type": "polygon",
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def _first_molmo_point(points: Any) -> Optional[List[float]]:
    if not points:
        return None
    for group in points:
        if group:
            first = group[0]
            if isinstance(first, dict) and {"x", "y"} <= first.keys():
                return [float(first["x"]), float(first["y"])]
    return None


def _read_fsd_archive(archive_path: Path, kind: str, max_samples: int) -> List[OfflineCandidate]:
    if max_samples <= 0:
        return []
    episodes: Dict[str, Dict[str, Any]] = {}
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf:
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            episode_key = "/".join(parts[:2])
            filename = parts[-1]
            if filename not in {"start_frame.jpg", "middle_frame.jpg", "end_frame.jpg", "lang.txt"} and not filename.startswith("metadata_"):
                continue
            episode = episodes.setdefault(episode_key, {"frames": []})
            if filename.endswith(".jpg"):
                episode["frames"].append(f"{archive_path}::{member.name}")
            elif filename == "lang.txt":
                episode["prompt"] = _read_tar_text(tf, member)
            elif filename.startswith("metadata_") and filename.endswith(".json"):
                metadata = json.loads(_read_tar_text(tf, member) or "{}")
                episode["prompt"] = metadata.get("current_task") or episode.get("prompt")
                episode["metadata_keys"] = sorted(metadata.keys())[:12]
            complete = [
                key for key, value in episodes.items()
                if value.get("prompt") and len(value.get("frames", [])) >= 3
            ]
            if len(complete) >= max_samples:
                break

    records: List[OfflineCandidate] = []
    for idx, (episode_key, episode) in enumerate(episodes.items()):
        if idx >= max_samples:
            break
        if not episode.get("prompt"):
            continue
        records.append(
            OfflineCandidate(
                source_dataset="FSD-dataset",
                sample_id=f"{kind}_{episode_key.split('/')[-1]}",
                candidate_kind="robot_frame_task_candidate",
                action_type=_infer_action_type(episode.get("prompt", "")),
                label_type="point_or_region",
                source_provenance=SourceProvenance.PUBLIC_DATASET.value,
                verifier_status=VerifierStatus.REVIEW_REQUIRED.value,
                stable_training_label=False,
                source_fields=["frame_members", "language_task", "metadata"],
                target_interface_fields=[
                    "grasp_point_2d_candidate",
                    "push_contact_point_2d_candidate",
                    "place_region_2d_candidate",
                ],
                risk_flags=[
                    "no_native_2d_action_geometry",
                    "archive_member_not_extracted",
                    "requires_teacher_or_human_label",
                ],
                prompt=episode.get("prompt", ""),
                frame_refs=sorted(episode.get("frames", [])),
                notes=f"{kind} frame triplet candidate.",
            )
        )
    return records


def _read_tar_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    f = tf.extractfile(member)
    if not f:
        return ""
    return f.read().decode("utf-8", errors="replace").strip()


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _infer_action_type(prompt: str) -> str:
    text = prompt.lower()
    if any(word in text for word in ("push", "slide")):
        return "push"
    if any(word in text for word in ("place", "put", "move", "insert")):
        return "place"
    if any(word in text for word in ("flip", "rotate", "turn")):
        return "reorient"
    return "pick"


def _render_svg_geometry(record: Dict[str, Any]) -> str:
    points = _geometry_points(record)
    if not points:
        return '<div class="sketch empty">no 2D geometry in candidate</div>'

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max((x_max - x_min), (y_max - y_min), 1.0) * 0.15
    view_x = x_min - pad
    view_y = y_min - pad
    view_w = max(x_max - x_min + pad * 2, 1.0)
    view_h = max(y_max - y_min + pad * 2, 1.0)

    elems = []
    bbox = record.get("bbox")
    if _is_point_list(bbox, 4):
        x1, y1, x2, y2 = bbox
        elems.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" '
            'fill="rgba(56, 189, 248, 0.12)" stroke="#0284c7" stroke-width="2" />'
        )

    for field_name, color in (("region_2d", "#16a34a"), ("place_region_2d", "#9333ea")):
        region = record.get(field_name)
        region_points = region.get("points") if isinstance(region, dict) else None
        if isinstance(region_points, list) and len(region_points) >= 3:
            pts = " ".join(f"{pt[0]},{pt[1]}" for pt in region_points if _is_point_list(pt, 2))
            elems.append(
                f'<polygon points="{pts}" fill="rgba(34, 197, 94, 0.12)" '
                f'stroke="{color}" stroke-width="2" />'
            )

    for field_name, color in (
        ("point_2d", "#eab308"),
        ("grasp_point_2d", "#dc2626"),
        ("push_contact_point_2d", "#f97316"),
    ):
        point = record.get(field_name)
        if _is_point_list(point, 2):
            elems.append(f'<circle cx="{point[0]}" cy="{point[1]}" r="{max(view_w, view_h) * 0.02}" fill="{color}" />')

    _append_vector(elems, record.get("grasp_point_2d"), record.get("jaw_axis_2d"), "#dc2626", view_w, view_h)
    _append_vector(
        elems,
        record.get("push_contact_point_2d"),
        record.get("push_direction_2d"),
        "#f97316",
        view_w,
        view_h,
    )

    return (
        f'<svg class="sketch" viewBox="{view_x} {view_y} {view_w} {view_h}" '
        'preserveAspectRatio="xMidYMid meet">'
        + "".join(elems)
        + "</svg>"
    )


def _append_vector(
    elems: List[str],
    point: Any,
    vector: Any,
    color: str,
    view_w: float,
    view_h: float,
) -> None:
    if not (_is_point_list(point, 2) and _is_point_list(vector, 2)):
        return
    scale = max(view_w, view_h) * 0.15
    x2 = point[0] + vector[0] * scale
    y2 = point[1] + vector[1] * scale
    elems.append(
        f'<line x1="{point[0]}" y1="{point[1]}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="2" />'
    )


def _geometry_points(record: Dict[str, Any]) -> List[List[float]]:
    points: List[List[float]] = []
    bbox = record.get("bbox")
    if _is_point_list(bbox, 4):
        x1, y1, x2, y2 = bbox
        points.extend([[x1, y1], [x2, y2]])
    for field_name in ("point_2d", "grasp_point_2d", "push_contact_point_2d"):
        point = record.get(field_name)
        if _is_point_list(point, 2):
            points.append([float(point[0]), float(point[1])])
    for field_name in ("region_2d", "place_region_2d"):
        region = record.get(field_name)
        region_points = region.get("points") if isinstance(region, dict) else None
        if isinstance(region_points, list):
            points.extend(
                [float(point[0]), float(point[1])]
                for point in region_points
                if _is_point_list(point, 2)
            )
    return points


def _is_point_list(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) for item in value)
    )


def _finalize_probe_records(probes: List[DatasetProbe]) -> None:
    for probe in probes:
        for record in probe.records:
            data = record.to_dict()
            verification = verify_action_interface_candidate(data)
            record.verification = verification.to_dict()
            record.promotion_eligible = verification.promotion_eligible
            record.promotion_blockers = verification.promotion_blockers


def _count_values(records: List[OfflineCandidate], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = getattr(record, field_name)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_values_from_dicts(records: List[Dict[str, Any]], field_name: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = record.get(field_name, "-")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
