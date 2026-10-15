"""候选 Observation 与 CandidateCluster。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import json
import os
from typing import Dict, List, Optional

from PIL import Image

from apps.data_flywheel.ontology.matcher import resolve_observation_name
from apps.data_flywheel.ontology.schema import normalize_name, slugify_identifier
from apps.data_flywheel.ontology.store import OntologyStore


@dataclass
class Observation:
    observation_id: str
    image: str
    raw_label: str
    normalized_label: str
    bbox: List[float]
    confidence: float
    width: int = 0
    height: int = 0
    crop_path: str = ""
    source: str = "grounding"
    co_occurrence_labels: List[str] = field(default_factory=list)
    matched_class_id: Optional[str] = None
    matched_subtype_id: Optional[str] = None
    match_source: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Observation":
        return cls(**data)


@dataclass
class CandidateCluster:
    cluster_id: str
    representative_label: str
    normalized_label: str
    labels: Dict[str, int]
    observation_ids: List[str]
    sample_images: List[str]
    sample_crops: List[str]
    count: int
    avg_confidence: float
    suggested_class_id: Optional[str] = None
    suggested_subtype_id: Optional[str] = None
    status: str = "needs_review"
    promoted_aliases: List[str] = field(default_factory=list)
    promoted_class_id: str = ""
    promoted_subtype_id: str = ""
    created_target_kind: str = ""
    created_target_id: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CandidateCluster":
        return cls(**data)


def _save_crop(image_path: str, bbox: List[float], crop_dir: str, stem: str) -> str:
    if not crop_dir or not image_path or not os.path.exists(image_path):
        return ""
    os.makedirs(crop_dir, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size
        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(x1 + 1, min(width, int(bbox[2])))
        y2 = max(y1 + 1, min(height, int(bbox[3])))
        crop = image.crop((x1, y1, x2, y2))
        out_path = os.path.join(crop_dir, f"{stem}.png")
        crop.save(out_path)
        return out_path


def collect_observations_from_grounding(
    grounding_jsonl: str,
    ontology_store: Optional[OntologyStore] = None,
    output_jsonl: Optional[str] = None,
    crop_dir: Optional[str] = None,
) -> List[Observation]:
    observations: List[Observation] = []
    with open(grounding_jsonl, "r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            image_path = item.get("image", "")
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            all_labels = [phrase.get("phrase", "") for phrase in item.get("phrases", [])]
            for phrase_idx, phrase in enumerate(item.get("phrases", [])):
                raw_label = phrase.get("phrase", "")
                context_labels = [label for label in all_labels if label != raw_label]
                for box_idx, box in enumerate(phrase.get("boxes", [])):
                    obs_id = f"{slugify_identifier(os.path.basename(image_path) or str(line_idx))}-{phrase_idx}-{box_idx}"
                    crop_path = _save_crop(image_path, box.get("bbox", [0, 0, 0, 0]), crop_dir or "", obs_id)
                    matched_class_id = None
                    matched_subtype_id = None
                    if ontology_store is not None:
                        resolution = resolve_observation_name(raw_label, ontology_store)
                        matched_class_id = resolution.object_class_id
                        matched_subtype_id = resolution.object_subtype_id
                        if resolution.matched_alias:
                            match_source = "alias"
                        elif resolution.object_subtype_id:
                            match_source = "subtype_name"
                        elif resolution.object_class_id:
                            match_source = "class_name"
                        else:
                            match_source = ""
                    else:
                        match_source = ""
                    observations.append(
                        Observation(
                            observation_id=obs_id,
                            image=image_path,
                            raw_label=raw_label,
                            normalized_label=normalize_name(raw_label),
                            bbox=list(box.get("bbox", [])),
                            confidence=float(box.get("confidence", box.get("score", 1.0))),
                            width=width,
                            height=height,
                            crop_path=crop_path,
                            co_occurrence_labels=context_labels,
                            matched_class_id=matched_class_id,
                            matched_subtype_id=matched_subtype_id,
                            match_source=match_source,
                        )
                    )
    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in observations:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return observations


def load_observations(path: str) -> List[Observation]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [Observation.from_dict(json.loads(line)) for line in f if line.strip()]


def _find_unknown_bucket(label: str, buckets: List[str], threshold: float = 0.92) -> Optional[str]:
    for key in buckets:
        if SequenceMatcher(None, label, key).ratio() >= threshold:
            return key
    return None


def cluster_observations(
    observations: List[Observation],
    output_jsonl: Optional[str] = None,
) -> List[CandidateCluster]:
    grouped: Dict[str, List[Observation]] = {}
    unknown_keys: List[str] = []
    for obs in observations:
        if obs.matched_subtype_id:
            group_key = f"subtype:{obs.matched_subtype_id}"
        elif obs.matched_class_id:
            group_key = f"class:{obs.matched_class_id}"
        else:
            fuzzy = _find_unknown_bucket(obs.normalized_label, unknown_keys)
            if fuzzy:
                group_key = f"unknown:{fuzzy}"
            else:
                unknown_keys.append(obs.normalized_label)
                group_key = f"unknown:{obs.normalized_label}"
        grouped.setdefault(group_key, []).append(obs)

    clusters: List[CandidateCluster] = []
    for group_key, items in grouped.items():
        labels: Dict[str, int] = {}
        sample_images: List[str] = []
        sample_crops: List[str] = []
        total_confidence = 0.0
        for item in items:
            labels[item.raw_label] = labels.get(item.raw_label, 0) + 1
            total_confidence += item.confidence
            if item.image and item.image not in sample_images and len(sample_images) < 5:
                sample_images.append(item.image)
            if item.crop_path and item.crop_path not in sample_crops and len(sample_crops) < 5:
                sample_crops.append(item.crop_path)
        representative_label = max(labels.items(), key=lambda pair: pair[1])[0]
        first = items[0]
        cluster = CandidateCluster(
            cluster_id=slugify_identifier(group_key),
            representative_label=representative_label,
            normalized_label=first.normalized_label,
            labels=labels,
            observation_ids=[item.observation_id for item in items],
            sample_images=sample_images,
            sample_crops=sample_crops,
            count=len(items),
            avg_confidence=round(total_confidence / max(1, len(items)), 4),
            suggested_class_id=first.matched_class_id,
            suggested_subtype_id=first.matched_subtype_id,
        )
        clusters.append(cluster)

    clusters.sort(key=lambda item: (-item.count, item.representative_label))
    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in clusters:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return clusters


def load_candidate_clusters(path: str) -> List[CandidateCluster]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [CandidateCluster.from_dict(json.loads(line)) for line in f if line.strip()]


def save_candidate_clusters(path: str, clusters: List[CandidateCluster]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cluster in clusters:
            f.write(json.dumps(cluster.to_dict(), ensure_ascii=False) + "\n")
