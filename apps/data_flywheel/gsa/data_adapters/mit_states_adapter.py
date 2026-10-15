"""MIT-States Adapter — Convert MIT-States annotations to A_obj_state labels.

MIT-States contains 63K images annotated with (adjective, noun) pairs
representing object states like (open, door), (broken, cup), etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ..a_pipeline_schemas import AObjStateLabel, validate_obj_state
from ..taxonomy import (
    AObjStateSubtype,
    Bucket,
    SourceProvenance,
    VerifierStatus,
)

logger = logging.getLogger("gsa.data_adapters.mit_states")

ADJ_TO_STATE_TYPE = {
    "open": "open_close",
    "closed": "open_close",
    "broken": "damage",
    "cracked": "damage",
    "dented": "damage",
    "full": "occupancy",
    "empty": "occupancy",
    "clean": "cleanliness",
    "dirty": "cleanliness",
    "upright": "pose",
    "tilted": "pose",
    "bent": "pose",
    "on": "on_off",
    "off": "on_off",
}

ADJ_TO_STATE_VALUE = {
    "open": "open",
    "closed": "closed",
    "broken": "broken",
    "cracked": "cracked",
    "dented": "dented",
    "full": "full",
    "empty": "empty",
    "clean": "clean",
    "dirty": "dirty",
    "upright": "upright",
    "tilted": "tilted",
    "bent": "lying",
    "on": "on",
    "off": "off",
}


def discover_mit_states_samples(dataset_path: Path, max_samples: int = 1000) -> List[dict]:
    """Discover image samples from MIT-States extracted directory."""
    samples = []
    extracted = dataset_path / "extracted"
    raw = dataset_path / "raw"

    search_dirs = [extracted, raw]
    for base in search_dirs:
        if not base.exists():
            continue
        for adj_dir in sorted(base.rglob("*")):
            if not adj_dir.is_dir():
                continue
            parts = adj_dir.name.split("_")
            if len(parts) < 2:
                continue
            adj = parts[0].lower()
            noun = "_".join(parts[1:]).lower()
            if adj not in ADJ_TO_STATE_TYPE:
                continue
            for img in sorted(adj_dir.glob("*.jpg"))[:5]:
                samples.append({
                    "image_path": str(img),
                    "adjective": adj,
                    "noun": noun,
                })
                if len(samples) >= max_samples:
                    return samples
    return samples


def convert_to_obj_state_labels(
    samples: List[dict],
) -> List[AObjStateLabel]:
    """Convert MIT-States samples to AObjStateLabel instances."""
    labels = []
    for i, sample in enumerate(samples):
        adj = sample["adjective"]
        noun = sample["noun"]

        state_type = ADJ_TO_STATE_TYPE.get(adj)
        state_value = ADJ_TO_STATE_VALUE.get(adj)
        if not state_type or not state_value:
            continue

        subtype_map = {
            "pose": AObjStateSubtype.POSE,
            "open_close": AObjStateSubtype.OPEN_CLOSE,
            "occupancy": AObjStateSubtype.OCCUPANCY,
            "cleanliness": AObjStateSubtype.CLEANLINESS,
            "damage": AObjStateSubtype.DAMAGE,
            "on_off": AObjStateSubtype.ON_OFF,
        }
        subtype = subtype_map.get(state_type, AObjStateSubtype.OTHER)

        label = AObjStateLabel(
            sample_id=f"mit_states_{i}",
            object_id=f"mit_{noun}_{i}",
            object_class=noun,
            state_type=state_type,
            state_value=state_value,
            subtype=subtype.value,
            source_provenance=SourceProvenance.PUBLIC_DATASET.value,
            verifier_status=VerifierStatus.AUTO_VERIFIABLE.value,
        )

        result = validate_obj_state(label)
        if result.valid:
            labels.append(label)

    logger.info(f"Converted {len(labels)} MIT-States samples to A_obj_state")
    return labels
