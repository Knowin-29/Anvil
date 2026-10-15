from PIL import Image

from apps.data_flywheel.scene_understanding.enrich.color import (
    extract_color_attributes,
    hsv_to_color_name,
)


def _solid_image(color, size=(40, 40)):
    return Image.new("RGB", size, color)


def test_extract_red_mask_returns_red():
    image = _solid_image((220, 20, 20))
    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "red"


def test_extract_white_mask_returns_white():
    image = _solid_image((245, 245, 245))
    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "white"


def test_extract_black_mask_returns_black():
    image = _solid_image((10, 10, 10))
    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "black"


def test_extract_gray_mask_returns_gray():
    image = _solid_image((130, 130, 130))
    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "gray"


def test_extract_brown_mask_returns_brown():
    image = _solid_image((140, 90, 40))
    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "brown"


def test_secondary_color_if_ratio_gt_threshold():
    image = Image.new("RGB", (40, 40))
    for x in range(40):
        for y in range(40):
            image.putpixel((x, y), (230, 20, 20) if x < 28 else (20, 20, 230))

    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_primary"] == "red"
    assert result["color_secondary"] == "blue"


def test_no_secondary_if_ratio_below_threshold():
    image = Image.new("RGB", (40, 40))
    for x in range(40):
        for y in range(40):
            image.putpixel((x, y), (230, 20, 20) if x < 35 else (20, 20, 230))

    result = extract_color_attributes(image, [[0, 0], [39, 0], [39, 39], [0, 39]])

    assert result["color_secondary"] is None


def test_empty_polygon_returns_none():
    image = _solid_image((220, 20, 20))

    result = extract_color_attributes(image, [])

    assert result == {
        "color_primary": None,
        "color_primary_hsv": None,
        "color_secondary": None,
    }


def test_hsv_to_color_name_uses_opencv_range():
    assert hsv_to_color_name([0, 255, 255]) == "red"
    assert hsv_to_color_name([120, 255, 255]) == "blue"
