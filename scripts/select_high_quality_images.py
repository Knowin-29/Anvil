"""Curated image selection for high-quality GSA flywheel.

Strategy: Select images that BEST exercise each capability dimension:
- G (Grounding): Multi-object indoor scenes with diverse objects
- S (Spatial): Scenes with clear spatial relationships (stacking, occlusion, containment)
- A_identity_attr: Objects with distinct visual attributes (color, material, shape)
- A_obj_state: Objects with observable states (open/closed, upright/fallen, on/off)
- A_action_interface: Robot manipulation scenes with graspable objects

Sources ranked by quality for each dimension:
1. RoboAfford (100 images) - BEST for A_action + affordance
2. Benchmark indoor (17 images) - BEST for G + S (real multi-object scenes)
3. BridgeDataV2 (25K) - BEST for A_robot_state + A_action_interface
4. appliance_datasets (10K) - BEST for A_obj_state (appliance states)
5. Test fixtures (11) - BEST for S (controlled spatial relations)
"""

import os
import random
from pathlib import Path
from glob import glob

random.seed(2026)

selected = []

# === Source 1: RoboAfford (affordance scenes with objects) ===
# Best for: G (multi-object), A_action_interface (grasp/push affordance)
roboafford_images = sorted(glob("/path/to/workspace/foundation/contributor/数据集调研/RoboAfford/data/RoboAfford-code/images/*.jpg"))
if roboafford_images:
    # Take 15 diverse images
    step = max(1, len(roboafford_images) // 15)
    selected.extend(roboafford_images[::step][:15])
    print(f"RoboAfford: {min(15, len(roboafford_images))} images (affordance scenes)")

# === Source 2: Real indoor benchmark scenes ===
# Best for: G (grounding), S (spatial), A_obj_state (real object states)
benchmark = sorted(glob("/path/to/datasets/datasets/benchmark_eval_all/test_little_data/*.jpg") +
                   glob("/path/to/datasets/datasets/benchmark_eval_all/test_little_data/*.png"))
if benchmark:
    selected.extend(benchmark[:15])
    print(f"Benchmark indoor: {min(15, len(benchmark))} images (real scenes)")

# === Source 3: BridgeDataV2 (robot tabletop manipulation) ===
# Best for: A_robot_state, A_action_interface, G (tabletop objects)
bridge = sorted(glob("/path/to/environment/backup_420/bridgedatav2/data/images/*.png"))
if bridge:
    # Sample diverse episodes (every 500th image for variety)
    sampled = bridge[::500][:15]
    selected.extend(sampled)
    print(f"BridgeDataV2: {len(sampled)} images (robot manipulation)")

# === Source 4: Appliance/household (appliance_datasets) ===
# Best for: A_obj_state (on/off, open/closed), A_identity_attr (materials)
appliance = sorted(glob("/path/to/datasets/datasets/appliance_datasets/**/*.png", recursive=True))
if appliance:
    # Prefer images with "door", "knob", "button", "screen" in name (state-rich)
    state_rich = [f for f in appliance if any(k in f.lower() for k in ("door", "knob", "button", "screen", "handle"))]
    if len(state_rich) >= 10:
        random.shuffle(state_rich)
        selected.extend(state_rich[:10])
    else:
        random.shuffle(appliance)
        selected.extend(appliance[:10])
    print(f"Appliance: 10 images (state-rich household)")

# === Source 5: Synthetic fixtures (controlled spatial) ===
# Best for: S (precise spatial relations), validation
fixtures = sorted(glob("tests/fixtures/images/*.png"))
selected.extend(fixtures)
print(f"Fixtures: {len(fixtures)} images (controlled spatial)")

# Deduplicate and verify existence
final = []
seen = set()
for img in selected:
    if img not in seen and os.path.exists(img):
        final.append(img)
        seen.add(img)

print(f"\nTotal: {len(final)} curated images")
print(f"  RoboAfford (affordance): {len([x for x in final if 'RoboAfford' in x])}")
print(f"  Benchmark (indoor): {len([x for x in final if 'benchmark' in x])}")
print(f"  BridgeDataV2 (robot): {len([x for x in final if 'bridgedatav2' in x])}")
print(f"  Appliance (states): {len([x for x in final if 'appliance_datasets' in x])}")
print(f"  Fixtures (spatial): {len([x for x in final if 'fixtures' in x])}")

# Save
output_path = "data/input/high_quality_50.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    for img in final:
        f.write(img + "\n")
print(f"\nSaved to {output_path}")
