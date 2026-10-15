"""Source-label adapters for the versioned Anvil-Gen scene manifest.

SpatialSense's official boxes are [y0,y1,x0,x1], NOT COCO xywh. Labels include
negative relationships. Neither negative labels nor official test splits may
be dropped or silently converted into positive training relations.
"""
from __future__ import annotations

from trains.flywheel.alignment_schema import Scene

SPATIALSENSE_PREDICATES = {
    "above": ("direction", "above", "image"),
    "under": ("direction", "below", "image"),
    "to the left of": ("direction", "left_of", "image"),
    "to the right of": ("direction", "right_of", "image"),
    "behind": ("relative_depth", "farther", "camera"),
    "in front of": ("relative_depth", "nearer", "camera"),
    "in": ("containment", "inside", "world"),
    "on": ("support", "supported_by", "world"),
    "next to": ("distance", "near", "world"),
}


def spatialsense_scenes(rows, image_index, *, training_approved=False):
    """Resolve only approved local images and externally established scene groups.

    image_index maps original URL to {image, group_id}. The group_id is the
    shared physical-scene/parent-image identity across reused corpora, not a new
    dataset-specific ID invented here. Unknown predicates/splits fail explicitly.
    """
    splits = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
    for row in rows:
        url = row["url"]
        if url not in image_index:
            raise ValueError("SpatialSense image missing from approved local index")
        local = image_index[url]
        width, height = row["width"], row["height"]
        if width <= 0 or height <= 0 or row["split"] not in splits:
            raise ValueError("invalid SpatialSense dimensions or split")
        entities, entity_keys, evidence, tasks = [], {}, [], []
        for index, annotation in enumerate(row["annotations"]):
            predicate = annotation["predicate"]
            if predicate not in SPATIALSENSE_PREDICATES or type(annotation["label"]) is not bool:
                raise ValueError("unmapped SpatialSense predicate or nonboolean label")
            refs = []
            for role in ("subject", "object"):
                value = annotation[role]
                y0, y1, x0, x1 = value["bbox"]
                bbox = [1000*x0/width, 1000*y0/height, 1000*x1/width, 1000*y1/height]
                key = (value["name"], tuple(bbox))
                if key not in entity_keys:
                    entity_id = f"entity_{len(entities)}"
                    entity_keys[key] = entity_id
                    entities.append({"entity_id": entity_id, "label": value["name"], "bbox": bbox})
                refs.append(entity_keys[key])
            if refs[0] == refs[1]:
                raise ValueError("SpatialSense self relationship")
            subtype, canonical, frame = SPATIALSENSE_PREDICATES[predicate]
            evidence_id = f"annotation_{index}"
            evidence.append({"evidence_id": evidence_id, "kind": "source_annotation", "entity_ids": refs,
                "description": f"Source predicate={predicate}; subject={refs[0]}; object={refs[1]}; "
                               f"truth={annotation['label']}; frame={frame}.",
                "relation": {"task": subtype, "subject_id": refs[0], "object_id": refs[1],
                             "predicate": canonical, "truth": annotation["label"], "frame": frame}})
            tasks.append({"task_id": f"relation_{index}", "form": "spatial", "spatial_task": subtype,
                          "requested_predicate": canonical, "entity_ids": refs,
                          "question": f"Is {refs[0]} {predicate} {refs[1]}? Return the relation and its truth value."})
        if not tasks:
            raise ValueError("SpatialSense image has no relationships")
        tasks.insert(0, {"task_id": "ground_anchor", "form": "bbox", "entity_ids": [entities[0]["entity_id"]],
                         "question": f"Locate {entities[0]['label']}."})
        yield Scene.model_validate({"source": "SpatialSense", "scene_id": url,
                                    "group_id": local["group_id"], "images": [local["image"]],
                                    "source_split": splits[row["split"]], "training_approved": training_approved,
                                    "entities": entities, "evidence": evidence, "tasks": tasks})
