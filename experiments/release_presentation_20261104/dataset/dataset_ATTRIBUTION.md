# Anvil-Ben — sources and attribution


There is **no blanket MIT data license**. Component terms remain separate.

### CLEVR QA replay — CC BY 4.0

The [official CLEVR project](https://cs.stanford.edu/people/jcjohns/clevr/) releases
its data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Original creators: Justin Johnson, Bharath Hariharan, Laurens van der Maaten,
Li Fei-Fei, C. Lawrence Zitnick and Ross Girshick.
*CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual
Reasoning*, CVPR 2017, https://arxiv.org/abs/1612.06890 .
Changes: subset selection, chat wrapping, portable paths and provenance fields.
Original question/answer content is retained. No endorsement is implied.

### COCO-derived spatial annotations — CC BY 4.0

Annotations retain the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
terms of the **COCO Consortium**. The [official terms](https://cocodataset.org/#termsofuse)
distinguish annotations from the individual photographs' rights.

Original source: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick.
*Microsoft COCO: Common Objects in Context*, ECCV 2014,
https://arxiv.org/abs/1405.0312 .
Changes: image selection and split assignment, geometric transformations of
source boxes/masks/points into spatial targets, prompts and chat packaging.
No endorsement by the COCO Consortium is implied. No image pixels are
redistributed; this annotation license grants no rights to those photographs.

### PixMo-derived pointing annotations — ODC-BY 1.0

Contains information from **[PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points)**,
created by **Allen Institute for AI (Ai2)** and made available under the
**[ODC Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/)**.
The derived database retains that license and upstream notices.

Changes: subset selection, coordinate-format conversion, point/count prompts,
scene/image grouping and chat packaging. See *Molmo and PixMo: Open Weights and
Open Data for State-of-the-Art Vision-Language Models*,
https://arxiv.org/abs/2409.17146 .
Follow Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).
Linked photographs retain their own rights and are not included.

### Release boundary

The original construction provenance retains `public_release_eligible: false`.
That field records the aggregate's construction-time status. This staged release
covers only the explicitly listed annotation subsets; it does not retroactively
clear the full aggregate or the image bundle. `release_manifest.json` binds
the released files, counts, terms and checksums. Unreleased components retain
their respective upstream rights.

Reconstruction scripts are project-owned MIT software; the full notice is
included in `rebuild_clevr_images.py`. Their license does not relicense data.

