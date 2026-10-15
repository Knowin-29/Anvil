"""End-to-end test: verify all VLM Workers output on 2 images."""
import json, os, sys, time, glob
from apps.data_flywheel.workers.vlm_client import VLMClient
from apps.data_flywheel.workers.grounding_worker import GroundingWorker
from apps.data_flywheel.workers.attribute_worker import AttributeWorker
from apps.data_flywheel.workers.spatial_worker import SpatialWorker
from apps.data_flywheel.workers.reasoning_worker import ReasoningWorker
from apps.data_flywheel.workers.planning_worker import PlanningWorker
from apps.data_flywheel.workers.instruction_worker import InstructionWorker
from apps.data_flywheel.workers.hallucination_worker import HallucinationWorker
from apps.data_flywheel.workers.multiturn_worker import MultiturnWorker
from apps.data_flywheel.workers.affordance_worker import AffordanceWorker
from apps.data_flywheel.sources.base import Sample
from PIL import Image

vlm = VLMClient(base_url="http://localhost:8000", model="qwen2.5-vl-32b")

data_root = "/path/to/datasets/datasets/VLM_Dataset_Root/01_Raw_Archive"
images = glob.glob(data_root + "/**/*.png", recursive=True)[:5]
if not images:
    images = glob.glob(data_root + "/**/*.jpg", recursive=True)[:5]

print(f"Testing with {len(images)} images available, using 2")
for img in images[:2]:
    print(f"  {os.path.basename(img)}")

workers = {
    "G": GroundingWorker(vlm_client=vlm),
    "A": AttributeWorker(vlm_client=vlm),
    "S": SpatialWorker(vlm_client=vlm),
    "R": ReasoningWorker(vlm_client=vlm),
    "P": PlanningWorker(vlm_client=vlm),
    "F": InstructionWorker(vlm_client=vlm),
    "H": HallucinationWorker(vlm_client=vlm),
    "E": MultiturnWorker(vlm_client=vlm),
    "Aff": AffordanceWorker(vlm_client=vlm),
}

total_results = {}
start = time.time()

for img_path in images[:2]:
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"{'='*60}")
    
    pil_img = Image.open(img_path)
    sample = Sample(
        sample_id=f"test_{os.path.basename(img_path)[:8]}",
        source_type="image_dir",
        frames=[pil_img],
        metadata={"image_path": img_path}
    )
    
    print("\n[G] Grounding...")
    g_anns = workers["G"].process(sample)
    grounding_data = []
    for ann in g_anns:
        c = ann.content
        grounding_data.append({"name": c.get("name", ""), "bbox": c.get("bbox", [])})
    print(f"  Detected {len(g_anns)} objects: {[d['name'] for d in grounding_data]}")
    total_results.setdefault("G", []).extend(g_anns)
    
    for tag in ["A", "S", "R", "P", "F", "H", "E", "Aff"]:
        print(f"\n[{tag}] {workers[tag].__class__.__name__}...")
        try:
            anns = workers[tag].process(sample, grounding_data=grounding_data)
            sub_tags = set(a.sub_tag for a in anns)
            total_results.setdefault(tag, []).extend(anns)
            print(f"  Generated {len(anns)} annotations")
            print(f"  Sub-tags: {sub_tags}")
            if anns:
                print(f"  Content keys: {list(anns[0].content.keys())}")
                print(f"  Difficulty: [{min(a.difficulty for a in anns)}, {max(a.difficulty for a in anns)}]")
                print(f"  Confidence: [{min(a.confidence for a in anns):.3f}, {max(a.confidence for a in anns):.3f}]")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

elapsed = time.time() - start
print(f"\n\n{'='*60}")
print(f"TEST SUMMARY (elapsed: {elapsed:.1f}s)")
print(f"{'='*60}")
for tag, anns in total_results.items():
    sub_tags = set(a.sub_tag for a in anns)
    print(f"  [{tag}] {len(anns)} annotations | sub_tags={sub_tags}")

total = sum(len(a) for a in total_results.values())
all_vlm = all(a.provenance == "vlm" for anns in total_results.values() for a in anns)
print(f"\nTotal annotations: {total}")
print(f"All provenance='vlm': {all_vlm}")
print(f"PASS" if total > 0 and all_vlm else "FAIL")
