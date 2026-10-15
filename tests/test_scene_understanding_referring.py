from PIL import Image

from apps.data_flywheel.scene_understanding.enrich.referring import (
    generate_referring_expressions,
)
from apps.data_flywheel.types import SceneNode


def _image():
    return Image.new("RGB", (16, 16), color="white")


def _node(**overrides):
    base = {
        "id": "node-1",
        "class_name": "chair",
        "bbox": [1, 2, 3, 4],
        "hierarchical_name": "the red chair",
    }
    base.update(overrides)
    return SceneNode(**base)


def test_referring_with_mock_vlm():
    class FakeVLM:
        def __init__(self):
            self.prompts = []

        def ask(self, image, question, max_tokens=512):
            self.prompts.append((image.size, question, max_tokens))
            return (
                '{"en": "the red chair", '
                '"zh": "红椅子", '
                '"en_contextual": "the chair near the table"}'
            )

    node = _node()
    other = _node(id="node-2", class_name="table", hierarchical_name="the table")
    vlm = FakeVLM()

    expressions = generate_referring_expressions(node, [node, other], _image(), vlm)

    assert expressions == ["the red chair", "红椅子", "the chair near the table"]
    assert vlm.prompts[0][0] == (16, 16)
    assert vlm.prompts[0][2] == 512
    assert "the red chair" in vlm.prompts[0][1]
    assert "the table" in vlm.prompts[0][1]
    assert "[1, 2, 3, 4]" in vlm.prompts[0][1]


def test_referring_vlm_failure_fallback():
    class BrokenVLM:
        def ask(self, image, question, max_tokens=512):
            raise RuntimeError("boom")

    node = _node(hierarchical_name="the wooden chair")

    assert generate_referring_expressions(node, [node], _image(), BrokenVLM()) == [
        "the wooden chair"
    ]


def test_referring_deduplication():
    class FakeVLM:
        def ask(self, image, question, max_tokens=512):
            return '{"en": "Chair", "zh": "chair", "en_contextual": "  CHAIR  "}'

    node = _node(hierarchical_name="the chair")

    assert generate_referring_expressions(node, [node], _image(), FakeVLM()) == ["Chair"]


def test_referring_skip_when_vlm_none():
    node = _node()

    assert generate_referring_expressions(node, [node], _image(), None) == []
