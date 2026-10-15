from __future__ import annotations

import io

import numpy as np
from PIL import Image

from tools.freeze_general_retention_v1_sources import (
    _distance_summary,
    _ai2d_adapter_index,
    _ai2d_uses_labeled_derivative,
    _float_summary,
    _image_features,
    _paired_threshold_sensitivity,
    _pixel_gate_sensitivity,
    canonical_hash,
)


def test_perceptual_identity_survives_png_to_jpeg_transport() -> None:
    image = Image.new("RGB", (96, 64), "white")
    for x in range(8, 88):
        image.putpixel((x, 20 + (x % 7)), (20, 60, 180))
    png = io.BytesIO()
    jpeg = io.BytesIO()
    image.save(png, format="PNG")
    image.save(jpeg, format="JPEG", quality=75)
    source = _image_features(png.getvalue())
    adapter = _image_features(jpeg.getvalue())
    assert source["sha256"] != adapter["sha256"]
    assert (source["width"], source["height"]) == (adapter["width"], adapter["height"])
    assert (int(source["phash64"], 16) ^ int(adapter["phash64"], 16)).bit_count() <= 4
    assert (int(source["dhash64"], 16) ^ int(adapter["dhash64"], 16)).bit_count() <= 4


def test_decoded_rgb_identity_ignores_lossless_container() -> None:
    image = Image.new("RGB", (32, 24), (10, 80, 170))
    png = io.BytesIO()
    bmp = io.BytesIO()
    image.save(png, format="PNG")
    image.save(bmp, format="BMP")
    png_features = _image_features(png.getvalue())
    bmp_features = _image_features(bmp.getvalue())
    assert png_features["sha256"] != bmp_features["sha256"]
    assert png_features["decoded_rgb_sha256"] == bmp_features["decoded_rgb_sha256"]


def test_pillow_default_jpeg_transform_is_exactly_fingerprinted() -> None:
    image = Image.new("RGB", (48, 35), (30, 90, 150))
    png = io.BytesIO()
    jpeg = io.BytesIO()
    image.save(png, format="PNG")
    image.save(jpeg, format="JPEG")
    source = _image_features(png.getvalue())
    transported = _image_features(jpeg.getvalue())
    assert source["pillow_default_jpeg_sha256"] == transported["sha256"]


def test_high_bit_depth_png_can_be_fingerprinted() -> None:
    image = Image.fromarray(np.full((12, 16), 4096, dtype=np.uint16))
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    features = _image_features(payload.getvalue())
    assert features["width"] == 16
    assert len(features["pillow_default_jpeg_sha256"]) == 64


def test_canonical_hash_is_order_independent_for_mappings() -> None:
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_distance_summary_uses_deterministic_nearest_ranks() -> None:
    assert _distance_summary([]) is None
    assert _distance_summary([9, 1, 5, 3, 7]) == {
        "n": 5,
        "min": 1,
        "p50": 5,
        "p90": 9,
        "p95": 9,
        "p99": 9,
        "max": 9,
    }


def test_float_summary_and_pixel_gate_sensitivity() -> None:
    assert _float_summary([0.03, 0.01, 0.02])["p50"] == 0.02
    gates = [
        {"max_normalized_rgb_mae": 0.01, "min_grayscale_correlation": 0.99},
        {"max_normalized_rgb_mae": 0.02, "min_grayscale_correlation": 0.98},
    ]
    assert _pixel_gate_sensitivity([0.005, 0.015], [0.995, 0.985], gates) == [
        {**gates[0], "matched": 1, "total": 2},
        {**gates[1], "matched": 2, "total": 2},
    ]


def test_paired_threshold_sensitivity_is_descriptive() -> None:
    assert _paired_threshold_sensitivity([0, 3, 9], [0, 4, 8], (0, 4, 8)) == [
        {"max_phash64_hamming": 0, "max_dhash64_hamming": 0, "matched": 1, "total": 3},
        {"max_phash64_hamming": 4, "max_dhash64_hamming": 4, "matched": 2, "total": 3},
        {"max_phash64_hamming": 8, "max_dhash64_hamming": 8, "matched": 2, "total": 3},
    ]


def test_ai2d_adapter_index_matches_upstream_integer_encoding() -> None:
    assert _ai2d_adapter_index("0", 0) == "0"
    assert _ai2d_adapter_index("60", 0) == "6000"
    assert _ai2d_adapter_index("2669", 1) == "266901"


def test_ai2d_upstream_is_abc_rule_distinguishes_raw_and_labeled_paths() -> None:
    assert not _ai2d_uses_labeled_derivative(
        True,
        ["405,140 km", "410,405 km", "401,510 km", "405,410 km"],
        "What is apogee?",
    )
    assert _ai2d_uses_labeled_derivative(
        True,
        ["Sun", "Earth", "None of the above", "Moon"],
        "F is labeled what?",
    )
    assert not _ai2d_uses_labeled_derivative(False, ["A", "B", "C", "D"], "Where?")
