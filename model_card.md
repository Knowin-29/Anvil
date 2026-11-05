# Anvil-MOPD-9B

### Spatial perception, reasoning and action in one 9B model.

**11 native spatial forms · 5.22M-record SFT ancestry · specialist-to-generalist MOPD**

[Paper](https://arxiv.org/abs/2610.21882) · [Project](https://knowin-29.github.io/Anvil.github.io/) · [Code](https://github.com/Knowin-29/Anvil) · [Data](https://huggingface.co/datasets/Knowin-29/Anvil-Ben)

Anvil connects scene-aligned spatial supervision with specialist knowledge transfer. Built on Qwen3.5-9B, it produces structured outputs for visual grounding, spatial relations and action-oriented tasks. MOPD transfers specialist feedback on the student's own rollouts into a unified model.

## Release highlights

- **Ready-to-load BF16 weights** at the repository root, with the final LoRA adapter in `adapter/`.
- **Visual geometry checkpoint 7168**, continuing coordinate MOPD with 458,752 repair-stage sample exposures.
- **Five featured training stages**, each with adapter weights and complete continuation state.
- **Full release evaluation:** 580,000 native spatial examples, 7,366 public benchmark examples and 10,105 general-ability examples.
- **72.27% public benchmark mean** across the seven benchmark scores below.

## Results

| Benchmark | Released model (%) | Paper reference (%) |
|---|---:|---:|
| CV-Bench | 88.44 | 88.21 |
| EmbSpatial | 81.29 | 82.34 |
| RefSpatial · Location100 | 60.38 | 59.00 |
| Where2Place | 71.47 | 68.00 |
| BLINK · Relative Depth | 82.26 | 91.13 |
| ERQA | 47.74 | 50.52 |
| VABench · Point | 74.33 | 80.67 |

[Native geometry and general-ability results](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/blob/main/docs/evaluation/README.md) · [Evaluation report](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/blob/main/docs/evaluation/evaluation_report.json) · [Evaluation protocol](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/blob/main/docs/evaluation/evaluation_spec.json)

## Get the model

Download the inference package:

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "Knowin-29/Anvil-MOPD-9B",
    ignore_patterns=["checkpoints/**", "docs/**"],
)
```

Load with Transformers `Qwen3_5ForConditionalGeneration` and `AutoProcessor`, using BF16 and thinking disabled. Evaluation uses up to 1,536 image tokens. The packaged `anvil/anvil_interface.py` handles native DoF prompts and pose decoding: call `prepare_native_prompt("dof", prompt)` before generation and `decode_native_response("dof", text, prompt)` afterward. The pose representation is `rotation128`; other native forms use their original prompts and responses.

## Featured checkpoints

| Stage | Original stage identifier |
|---|---|
| [Initial adaptation](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/tree/main/checkpoints/initial-adaptation) | `ancestor_repair/checkpoint-2048` |
| [Balanced MOPD](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/tree/main/checkpoints/balanced-mopd) | `balanced_mopd/checkpoint-512` |
| [Format-enhanced MOPD](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/tree/main/checkpoints/format-mopd) | `format_mopd/checkpoint-256` |
| [Coordinate MOPD](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/tree/main/checkpoints/coordinate-mopd) | `coordinate_mopd/checkpoint-256` |
| [Visual geometry adaptation](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/tree/main/checkpoints/visual-geometry) | `visual_repair/checkpoint-7168` |

Use [the checkpoint guide](https://huggingface.co/Knowin-29/Anvil-MOPD-9B/blob/main/checkpoints/README.md) to download an individual stage. Additional saved stages are organized in `checkpoints/archive/`.

## Repository guide

| Path | Contents |
|---|---|
| Root model files | Final merged BF16 model, processor and tokenizer |
| `adapter/` | Selected visual geometry LoRA adapter |
| `checkpoints/` | Featured stages and training archive |
| `anvil/` | Native output interface |
| `docs/evaluation/` | Measured scores and evaluation settings |
| `docs/provenance/` | Training lineage and release records |
| `release_manifest.json` | File checksums and source revision |

Anvil's training design uses 5.22M SFT records and a 580K native evaluation split. Explore the [Anvil-Ben data page](https://huggingface.co/datasets/Knowin-29/Anvil-Ben) for downloadable annotations and image reconstruction tools.
