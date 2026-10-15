"""Drift guard: keep the independent G_grounding validation module in sync with production.

scripts/evaluate_grounding_generator.py is intentionally isolated from the live
GSA pipeline (scripts/generator_service.py): it re-implements the generation
logic instead of importing it, and it surfaces every tunable through
configs/grounding_verifier_rules.yaml. That isolation is the whole point of the
module, but it means the two copies can silently drift apart.

These tests import BOTH copies and assert that the parts which are meant to be a
*faithful reproduction* (not a tuning surface) stay byte-for-byte equivalent:

  * derive_source   - pure path-based heuristic, must match exactly
  * parse_json      - JSON extraction from VLM output, must match exactly
  * denormalize_bbox- coordinate-scaling + geometry gate, must match exactly when
                      the module is run with production's default area band
  * detection prompt- a documented tuning surface, so this test pins it to the
                      production prompt MODULO the explicitly allowed deltas below;
                      any *other* change to the production prompt fails the test
                      and forces a conscious review.

If production changes, these tests fail on purpose. Either port the change into
the independent module / config, or extend the explicitly-allowed delta list.
"""

import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_PATH = REPO_ROOT / "configs" / "grounding_verifier_rules.yaml"
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_script_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_script_module("evaluate_grounding_generator")
prod = _load_script_module("generator_service")


# ----------------------------------------------------------------------------
# Explicitly-allowed, reviewed deltas between production and the tuning config.
# Each entry is (production_substring, config_substring) plus a why. Anything not
# listed here MUST be identical, otherwise the relevant test fails.
# ----------------------------------------------------------------------------
ALLOWED_PROMPT_DELTAS = [
    (
        # The config prompt loosens the wording to agree with the actual geometry
        # gate (bbox_filters.max_area_ratio == 0.95), instead of the stale 70%
        # text left in production. Behaviorally the real cap is enforced by
        # denormalize_bbox, so this is a wording fix, not a logic change.
        "Each bbox MUST be smaller than 70% of image area.",
        "Each bbox MUST be smaller than 95% of image area.",
    ),
]


def _apply_allowed_deltas(production_text: str) -> str:
    out = production_text
    for prod_sub, cfg_sub in ALLOWED_PROMPT_DELTAS:
        assert prod_sub in out, (
            f"Allowed-delta source substring no longer present in production prompt: "
            f"{prod_sub!r}. The production prompt changed; review ALLOWED_PROMPT_DELTAS."
        )
        out = out.replace(prod_sub, cfg_sub)
    return out


# ----------------------------------------------------------------------------
# derive_source: pure reproduction, must match exactly.
# ----------------------------------------------------------------------------

def test_derive_source_matches_production():
    paths = [
        "/data/bridgedatav2/episode_000001.png",
        "/x/bridge/foo.jpg",
        "/x/erqa/embodied/y.png",
        "/x/embodied_reasoner/z.png",
        "/data/vlm_dataset_root/raw_benchmark/a.jpg",
        "/data/appliance_datasets/data_instruction/washing/b.png",
        "/sets/mit-states/c.jpg",
        "/sets/mit_states/c.jpg",
        "/sets/changeit/object-states/d.jpg",
        "/sets/spatialsense/e.jpg",
        "/sets/where2place/f.jpg",
        "/repo/tests/fixtures/images/scene.png",
        "/completely/unknown/path.png",
        "",
    ]
    for p in paths:
        assert gen.derive_source(p) == prod.derive_source(p), p


# ----------------------------------------------------------------------------
# parse_json: pure reproduction, must match exactly.
# ----------------------------------------------------------------------------

def test_parse_json_matches_production():
    samples = [
        '[{"object_name": "cup"}]',
        '<think>reasoning here</think>\n[{"object_name": "cup"}]',
        '```json\n[{"a": 1}]\n```',
        '```\n{"b": 2}\n```',
        'garbage before [{"x": 1}] garbage after',
        'prefix {"only": "object"} suffix',
        'totally not json at all',
        '',
        '{"match": true, "actual_object": "plate", "confidence": 0.8}',
        '<think>...</think>```json\n{"match": false}\n```',
    ]
    for s in samples:
        assert gen.parse_json(s) == prod.VLMClient._parse_json(s), repr(s)


# ----------------------------------------------------------------------------
# denormalize_bbox: both implementations use normalized coordinates regardless
# of image size. Preserve geometry gates and the sub-1000px regression below.
# ----------------------------------------------------------------------------

def test_denormalize_bbox_matches_production_outside_the_fix():
    # (bbox, img_w, img_h) — cases that do NOT hit the sub-1000px normalized path.
    cases = [
        ([0.1, 0.1, 0.5, 0.5], 1000, 1000),     # 0-1 fractional (first branch, identical)
        ([100, 100, 500, 500], 2000, 2000),     # 0-1000 on a >1000px image (both scale)
        ([10, 10, 200, 200], 1280, 960),        # 0-1000 on a >1000px image (both scale)
        ([0, 0, 1000, 1000], 1000, 1000),       # full image -> dropped by both
        ([500, 500, 100, 100], 1000, 1000),     # degenerate -> dropped by both
        ([0, 0, 5, 5], 1000, 1000),             # tiny -> dropped by both
        ([-10, 0, 100, 100], 1000, 1000),       # negative -> else/pixel -> dropped by both
        ([0, 0, 2000, 100], 1000, 1000),        # coord > 1000 -> else/pixel -> dropped by both
        ([1, 2, 3], 1000, 1000),                # wrong length -> None for both
        ([], 1000, 1000),                        # empty -> None for both
    ]
    for bbox, w, h in cases:
        expected = prod.denormalize_bbox(list(bbox), w, h)
        actual = gen.denormalize_bbox(list(bbox), w, h, max_area_ratio=0.95, min_area_ratio=0.001)
        assert actual == expected, (bbox, w, h, actual, expected)


def test_denormalize_bbox_preserves_subkilopixel_normalized_coordinates():
    # A 0-1000 normalized box must not be mistaken for pixels on a small image.
    bbox, w, h = [200, 300, 800, 900], 640, 480
    assert prod.denormalize_bbox(list(bbox), w, h) == [128, 144, 512, 432]
    fixed = gen.denormalize_bbox(list(bbox), w, h, max_area_ratio=0.95, min_area_ratio=0.001)
    # 200*640/1000=128, 300*480/1000=144, 800*640/1000=512, 900*480/1000=432
    assert fixed == [128, 144, 512, 432]


# ----------------------------------------------------------------------------
# build_label: G_grounding schema parity for the fields both copies emit.
# ----------------------------------------------------------------------------

def test_build_label_g_grounding_schema_parity():
    item = {"object_name": "red_cup", "bbox": [100, 100, 300, 300], "confidence": 0.9, "scene_type": "kitchen"}
    config = gen.load_grounding_generator_config(CONFIG_PATH)

    mod_label = gen.build_label(item, "/data/fixtures/x.jpg", 1000, 1000, "sid", 0, config, scene_type="kitchen")
    prod_label = prod.build_label(item, "G_grounding", "/data/fixtures/x.jpg", 1000, 1000, "sid", 0)

    # Compare the production-defined fields (the module adds none beyond these).
    shared_keys = [
        "pipeline", "source", "dataset", "sample_id", "image_path", "image_size",
        "bucket", "dimension", "object_id", "confidence", "scene_type",
        "object_class", "bbox",
    ]
    for key in shared_keys:
        assert mod_label[key] == prod_label[key], (key, mod_label.get(key), prod_label.get(key))


# ----------------------------------------------------------------------------
# Detection prompt: pinned to production modulo explicitly-allowed deltas.
# ----------------------------------------------------------------------------

def test_detection_prompt_matches_production_modulo_allowed_deltas():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg_prompt = config["grounding_generator"]["detection"]["prompt"].strip()
    prod_prompt = prod.PROMPTS["G_grounding"].strip()

    expected = _apply_allowed_deltas(prod_prompt)
    assert cfg_prompt == expected, (
        "Detection prompt drifted from production beyond the allowed deltas.\n"
        "Either port the production change into configs/grounding_verifier_rules.yaml "
        "or extend ALLOWED_PROMPT_DELTAS after review."
    )


# ----------------------------------------------------------------------------
# Cross-check prompt: the production verify_bbox wording must remain reproducible
# by the config template so the two stay semantically aligned.
# ----------------------------------------------------------------------------

def test_cross_check_prompt_template_matches_production_wording():
    config = gen.load_grounding_generator_config(CONFIG_PATH)
    template = config["cross_check"]["prompt"]

    obj, bbox = "red_cup", [100, 100, 300, 300]
    rendered = gen.render_cross_check_prompt(template, obj, bbox).strip()

    # Production builds the prompt inline; reconstruct its exact text.
    prod_prompt = (
        f'I claim there is a "{obj}" in the region '
        f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}] (normalized 0-1000 coords). '
        f'Look at that region carefully. '
        f'Is my claim correct? Is "{obj}" really there? '
        f'Reply ONLY with JSON: '
        f'{{"match": true/false, "actual_object": "what you actually see there", "confidence": 0.0-1.0}}'
    ).strip()

    assert rendered == prod_prompt, (
        "Cross-check prompt drifted from production verify_bbox wording.\n"
        f"module:\n{rendered}\nproduction:\n{prod_prompt}"
    )
