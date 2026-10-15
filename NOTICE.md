# License and release boundary

Project-owned Anvil code is licensed under MIT; see LICENSE. The inherited
Rex-Embodied MIT copyright notice is retained. Third-party code, dependencies,
datasets and pretrained weights retain their original licenses: MIT here does
not grant redistribution rights to those assets. Existing in-file copyright
and attribution notices must not be removed.

This tree consolidates Rex-Embodied, GSA_Forge and GSA_collect code,
including robot-state and action-interface worktrees. Release archives exclude
Git history. Development history must not be pushed as a public release: old
commits may retain private configuration removed from the current files.
The public source map records component lineage; the detailed worktree/hash
audit is retained privately by the project owner.

Raw images, credentials, private endpoints, full service logs and model weights
are not part of this source release. Dataset and checkpoint access conditions
are indexed in registry/. The repository name and MIT grant do not imply that
the project has already been pushed or that all GPU pipelines were rerun after
integration. See README for the tested scope and optional dependencies.

## Dataset-source terms checked on 2026-10-12

Provider declarations and media rights are separate. The table records evidence,
not a blanket grant to redistribute the assembled candidate. Fixed source-card
revisions and hashes are in `registry/datasets.json`.

| Source | Provider declaration | Remaining distribution conditions |
| --- | --- | --- |
| [COCO](https://cocodataset.org/#termsofuse) | Annotations: CC BY 4.0; images have separate owners. | Preserve each image's source/license and establish creator attribution; do not apply Anvil's MIT license to images. |
| [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | ODC-BY-1.0 database; images supplied as external URLs. | Carry database attribution and license link; establish individual linked-image rights separately. |
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) | All data: CC BY 4.0. | Credit creators, link license/source, and describe selection/reformatting; retain upstream notices. |
| [PRISM](https://huggingface.co/datasets/allenai/PRISM) | Dataset card: MIT; scenes use ShapeNet-Sem and ACRONYM assets. | Preserve provider terms and resolve applicable underlying-asset conditions for the intended media distribution. |
| [ALFRED](https://github.com/askforalfred/alfred/tree/f91f4c0c96c7a29f33d0557f86b0a21035379b3b) | Repository: MIT, copyright 2019 ALFRED. | Preserve the notice; confirm the coverage of externally hosted rendered images before mirroring. |
| [ShareRobot](https://huggingface.co/datasets/BAAI/ShareRobot) | Checked card has no aggregate license declaration and combines multiple datasets. | Obtain an applicable annotation grant and map component-image licenses; missing metadata is not permission. |
| [ShareRobot-Bench](https://huggingface.co/datasets/BAAI/ShareRobot-Bench) | Dataset card: Apache-2.0. | Preserve provider notices and review the source-image/component terms; the card alone does not resolve the entire chain. |

The [ODC-BY terms](https://opendatacommons.org/licenses/by/1-0/) do not grant
copyright over individual database contents. The
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require attribution,
a license link and indication of modifications. A future CLEVR-derived data
package must credit Justin Johnson, Bharath Hariharan, Laurens van der Maaten,
Fei-Fei Li, C. Lawrence Zitnick and Ross Girshick, and identify its selection and
message-format changes. Source-image bytes in the current candidate are unchanged.

The bound COCO inventory maps all 81,897 selected encoded images to official
license IDs. It preserves all matches for 11 images with multiple upstream IDs,
including differing license entries. Of the selected images, 56,574 have at least
one NonCommercial entry; this count is not a legal resolution of ambiguous grants.
Official image URLs/license IDs do not by themselves establish complete creator
attribution. The candidate remains unpublished; no images were removed or
relicensed to produce these counts.
