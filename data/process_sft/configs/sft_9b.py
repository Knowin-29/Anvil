# Rex-Omni SFT Configuration for EmbodiedForge Data
# Base model: Qwen3.5-9B (auto-detected in train_adaptive.py)

from dataset import ConcatDataset, DataCollatorForSupervisedDataset, GroundingTSVDataset
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import GROUNDING_SINGLE_REGION_STAGE_XYXY

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "/path/to/environment/hf_models/Qwen/Qwen3.5-9B"

TSV_BASE = "/path/to/datasets/dataflywheel/output/rex_omni_tsv"

embodiedforge_grounding = dict(
    type=GroundingTSVDataset,
    img_tsv_file=f"{TSV_BASE}/embodiedforge_grounding.images.tsv",
    ann_tsv_file=f"{TSV_BASE}/embodiedforge_grounding.annotations.tsv",
    ann_lineidx_file=f"{TSV_BASE}/embodiedforge_grounding.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="embodiedforge_grounding",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[embodiedforge_grounding],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
