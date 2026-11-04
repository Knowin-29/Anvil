---
pretty_name: Anvil-Ben
language:
- en
task_categories:
- visual-question-answering
tags:
- embodied-ai
- spatial-reasoning
- visual-grounding
- anvil
configs:
- config_name: clevr_qa_replay
  data_files:
  - split: train
    path: records/train/QA-*.jsonl.gz
---

# Anvil-Ben

### Spatial intelligence starts with structured supervision.

**5.77M downloadable annotations · 7 spatial forms · source-linked targets**

[Paper](https://arxiv.org/abs/2610.21882) · [Project](https://knowin-29.github.io/Anvil.github.io/) · [Code](https://github.com/Knowin-29/Anvil) · [Model & checkpoints](https://huggingface.co/Knowin-29/Anvil-MOPD-9B)

Anvil-Ben brings visual grounding, spatial relations and compositional QA into a common training interface. Native targets, source references and explicit train/evaluation splits make it practical to build spatial specialists and train a unified vision-language model.

## At a glance

| Component | Downloadable records | Full project corpus |
|---|---:|---:|
| Spatial training | 4,729,834 | 4,756,000 |
| CLEVR QA training replay | 464,000 | 464,000 |
| Spatial evaluation | 572,299 | 580,000 |
| **Total** | **5,766,133** | **5,800,000** |

The full project uses **5.22M training records** and a **580K evaluation partition**, with an index of **160K canonical images**. This downloadable edition provides COCO- and PixMo-derived spatial annotations plus CLEVR QA replay. Images are reconstructed from separately obtained upstream datasets. The full corpus also includes 33,867 ALFRED, PRISM and ShareRobot records retained in the project package.

## What you can build

- **Visual grounding:** boxes, multi-object detection, polygon masks and points.
- **Spatial reasoning:** directions, relation graphs and image-space metric comparisons.
- **Compositional QA:** 464K CLEVR examples for auxiliary training replay.

Each spatial record carries its native form, prompt, reference and source provenance. Evaluation annotations have a dedicated split for scoring; use the training split for model fitting and feedback.

## Quick start

```python
from huggingface_hub import snapshot_download

root = snapshot_download("Knowin-29/Anvil-Ben", repo_type="dataset")
```

Load the CLEVR replay directly with Datasets:

```python
from datasets import load_dataset

qa = load_dataset("Knowin-29/Anvil-Ben", "clevr_qa_replay", split="train")
```

Read a native spatial shard:

```python
import gzip, json
from pathlib import Path

shard = next((Path(root) / "spatial/train").glob("bbox-*.jsonl.gz"))
with gzip.open(shard, "rt") as stream:
    record = json.loads(next(stream))
```

## Repository guide

| Directory | Contents |
|---|---|
| `spatial/train/` | Spatial training annotations, grouped by native form |
| `spatial/benchmark/` | Held-out spatial annotations |
| `records/train/` | CLEVR QA replay |
| `scripts/` | Image reconstruction tools |
| `inventory/` | Source inventories and image hashes |
| `docs/` | Attribution and release details |
| `release_manifest.json` | Annotation counts and SHA-256 checksums |

## Connect your images

After obtaining the upstream images, run from the downloaded repository:

```bash
python scripts/rebuild_clevr_images.py --clevr-root /data/CLEVR_v1.0 --output /data/anvil-clevr
python scripts/rebuild_spatial_images.py --source coco_train2017 \
  --source-root /data/coco/train2017 --output /data/anvil-images
python scripts/rebuild_spatial_images.py --source pixmo_points \
  --source-root /data/pixmo/images --output /data/anvil-images
```

The tools match source images to the released paths by content hash. CLEVR reconstruction checks all 464K replay records and 46,400 referenced images.

## Sources and terms

COCO and CLEVR annotations follow **CC BY 4.0**; PixMo-derived annotations follow **ODC-BY 1.0**. Image rights follow their respective sources. See [attribution and component terms](docs/ATTRIBUTION.md) for creators, transformations and upstream links.

Anvil connects **Anvil-Gen** data production, **Anvil-Ben** structured supervision and **Anvil-MOPD** specialist-to-generalist learning. Explore the [source repository](https://github.com/Knowin-29/Anvil) to build on the pipeline.
