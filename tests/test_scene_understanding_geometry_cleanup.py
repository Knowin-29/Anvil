import re
from pathlib import Path

from apps.data_flywheel import geometry

_HEURISTIC_PATTERN = re.compile(
    r"is_pickable|is_container_by_name|NON_PICKABLE|CONTAINER_CATEGORIES|is_pushable|is_stackable\("
)


def test_geometry_no_heuristic_categories():
    src = Path("apps/data_flywheel/geometry.py").read_text()
    matches = _HEURISTIC_PATTERN.findall(src)
    assert matches == [], f"Found heuristic residue: {matches}"


def test_geometry_retains_pure_helper_functions():
    assert geometry.bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) > 0
    assert geometry.bbox_contains([0, 0, 20, 20], [5, 5, 10, 10]) is True
    assert geometry.is_inside_3d([0, 0, 0], [1, 1, 1], [0, 0, 0], [4, 4, 4]) is True
