"""按 GSA 各能力维度精选 50 条高质量数据源（快速版）。"""

import os
import random
import json
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "input" / "dimension_sources"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-collected image lists (non-recursive, fast)
def list_images_fast(directory, max_depth=1, max_count=200):
    """List images without deep recursion, skip __MACOSX."""
    images = []
    if not os.path.exists(directory):
        return images
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        depth = root.replace(directory, "").count(os.sep)
        if depth > max_depth:
            dirs.clear()
            continue
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")) and not f.startswith("._"):
                images.append(os.path.join(root, f))
                if len(images) >= max_count:
                    return images
    return images


# Data source paths
BENCHMARK_IMAGES = "/path/to/datasets/datasets/benchmark_eval_all/scripts/api_logs/images"
WASHING_MACHINE = "/path/to/datasets/datasets/washing_machine_eval_data"
anvil_WASHING = "/path/to/datasets/datasets/appliance_datasets/washing_machine_1029/processed_washing_machine_data_all"
BRIDGE_DATA = "/path/to/environment/backup_420/bridgedatav2/data/images"
MIT_STATES = "/path/to/datasets/013_done_P1_A1_mit-states/raw/extracted/release_dataset/images"
TEST_FIXTURES = str(PROJECT_ROOT / "tests" / "fixtures" / "images")


def select_g_grounding():
    """G维度: 多物体桌面/室内场景。"""
    candidates = []
    # Test fixtures (curated tabletop scenes with multiple objects)
    candidates.extend(list_images_fast(TEST_FIXTURES, max_depth=0, max_count=50))
    # Benchmark indoor scenes
    bench = list_images_fast(BENCHMARK_IMAGES, max_depth=0, max_count=200)
    random.shuffle(bench)
    candidates.extend(bench[:40])
    return candidates[:50]


def select_s_spatial():
    """S维度: 有明确空间关系的多物体场景。"""
    candidates = []
    # Test fixtures with spatial keywords
    fixtures = list_images_fast(TEST_FIXTURES, max_depth=0, max_count=50)
    spatial_kw = ["stack", "block", "occlu", "cascade", "multiple", "reference", "pick_place"]
    spatial = [p for p in fixtures if any(k in os.path.basename(p).lower() for k in spatial_kw)]
    candidates.extend(spatial)
    # Benchmark scenes (multi-object)
    bench = list_images_fast(BENCHMARK_IMAGES, max_depth=0, max_count=200)
    random.shuffle(bench)
    candidates.extend(bench[:40])
    # Washing machine panels (spatial layout of knobs)
    wm = list_images_fast(anvil_WASHING, max_depth=1, max_count=100)
    random.shuffle(wm)
    candidates.extend(wm[:10])
    return candidates[:50]


def select_a_attribute():
    """A_attribute维度: 有丰富属性的物体。"""
    candidates = []
    # MIT-States: objects with attribute annotations
    mit = list_images_fast(MIT_STATES, max_depth=2, max_count=200)
    random.shuffle(mit)
    candidates.extend(mit[:40])
    # Benchmark images
    bench = list_images_fast(BENCHMARK_IMAGES, max_depth=0, max_count=100)
    random.shuffle(bench)
    candidates.extend(bench[:10])
    return candidates[:50]


def select_a_object_state():
    """A_object_state维度: 有明确状态变化的物体。"""
    candidates = []
    # Washing machine knobs/buttons (on/off states)
    wm = list_images_fast(WASHING_MACHINE, max_depth=2, max_count=200)
    knob_kw = ["knob", "button", "switch", "dial", "panel", "status", "bbox"]
    knob_imgs = [p for p in wm if any(k in os.path.basename(p).lower() for k in knob_kw)]
    random.shuffle(knob_imgs)
    candidates.extend(knob_imgs[:25])
    # anvil washing machine
    kw = list_images_fast(anvil_WASHING, max_depth=1, max_count=100)
    random.shuffle(kw)
    candidates.extend(kw[:15])
    # MIT-States (state changes)
    mit = list_images_fast(MIT_STATES, max_depth=2, max_count=200)
    random.shuffle(mit)
    candidates.extend(mit[:10])
    return candidates[:50]


def select_a_action_interface():
    """A_action_interface维度: 机器人操作场景。"""
    candidates = []
    # BridgeDataV2: robot tabletop manipulation
    bridge = list_images_fast(BRIDGE_DATA, max_depth=0, max_count=200)
    random.shuffle(bridge)
    candidates.extend(bridge[:40])
    # Test fixtures
    fixtures = list_images_fast(TEST_FIXTURES, max_depth=0, max_count=50)
    action_kw = ["pick", "place", "tabletop"]
    action = [p for p in fixtures if any(k in os.path.basename(p).lower() for k in action_kw)]
    candidates.extend(action)
    # Fill with benchmark
    bench = list_images_fast(BENCHMARK_IMAGES, max_depth=0, max_count=50)
    random.shuffle(bench)
    candidates.extend(bench[:10])
    return candidates[:50]


def select_a_precondition():
    """A_precondition维度: 有遮挡/阻碍关系的场景。"""
    candidates = []
    # Test fixtures with blocking/occlusion
    fixtures = list_images_fast(TEST_FIXTURES, max_depth=0, max_count=50)
    block_kw = ["block", "occlu", "cascade", "non_operable", "multiple"]
    blocked = [p for p in fixtures if any(k in os.path.basename(p).lower() for k in block_kw)]
    candidates.extend(blocked)
    # Benchmark scenes (cluttered)
    bench = list_images_fast(BENCHMARK_IMAGES, max_depth=0, max_count=200)
    random.shuffle(bench)
    candidates.extend(bench[:35])
    # Bridge data (robot scenes with obstacles)
    bridge = list_images_fast(BRIDGE_DATA, max_depth=0, max_count=100)
    random.shuffle(bridge)
    candidates.extend(bridge[:10])
    return candidates[:50]


def main():
    dimensions = {
        "G_grounding": select_g_grounding,
        "S_spatial": select_s_spatial,
        "A_attribute": select_a_attribute,
        "A_object_state": select_a_object_state,
        "A_action_interface": select_a_action_interface,
        "A_precondition": select_a_precondition,
    }

    manifest = {}
    all_images = set()

    for dim_name, selector_fn in dimensions.items():
        images = selector_fn()
        valid_images = [p for p in images if os.path.exists(p)]
        if len(valid_images) < 50:
            print(f"  WARNING: {dim_name} only has {len(valid_images)} valid images")

        list_path = OUTPUT_DIR / f"{dim_name}.txt"
        with open(list_path, "w") as f:
            for img in valid_images:
                f.write(img + "\n")

        manifest[dim_name] = {
            "count": len(valid_images),
            "list_file": str(list_path),
        }
        all_images.update(valid_images)
        print(f"  {dim_name}: {len(valid_images)} images")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    combined_path = OUTPUT_DIR / "all_dimensions_combined.txt"
    with open(combined_path, "w") as f:
        for img in sorted(all_images):
            f.write(img + "\n")

    print(f"\nTotal unique images: {len(all_images)}")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
