from apps.data_flywheel.scene_understanding.fusion import fuse_edges
from apps.data_flywheel.types import SceneEdge


def _edge(subject: str, object: str, predicate: str, source: str, confidence: float, frame: str = "untyped") -> SceneEdge:
    return SceneEdge(
        subject=subject,
        object=object,
        predicate=predicate,
        source=source,
        confidence=confidence,
        frame=frame,
    )


def test_fusion_both_agree_higher_confidence():
    edges_2d = [_edge("a", "b", "left-of", "2d", 0.4)]
    edges_3d = [_edge("a", "b", "left-of", "3d", 0.7)]

    fused = fuse_edges(edges_2d, edges_3d)

    assert fused == [
        SceneEdge(
            subject="a",
            object="b",
            predicate="left-of",
            source="2d+3d",
            confidence=1.0,
            frame="multi",
        )
    ]


def test_fusion_conflict_higher_confidence_wins():
    edges_2d = [_edge("a", "b", "left-of", "2d", 0.3)]
    edges_3d = [_edge("a", "b", "above", "3d", 0.8)]

    fused = fuse_edges(edges_2d, edges_3d, strategy="higher_confidence")

    assert fused == [
        SceneEdge(
            subject="a",
            object="b",
            predicate="above",
            source="3d",
            confidence=0.8,
            frame="untyped",
        )
    ]


def test_fusion_only_2d_when_no_depth():
    edges_2d = [
        _edge("a", "b", "near", "2d", 0.8),
        _edge("b", "c", "left-of", "2d", 0.9),
    ]

    fused = fuse_edges(edges_2d, [])

    assert fused == edges_2d


def test_fusion_only_3d_when_2d_fails():
    edges_3d = [
        _edge("a", "b", "inside", "3d", 0.9),
        _edge("c", "d", "stacked-on", "3d", 0.85),
    ]

    fused = fuse_edges([], edges_3d)

    assert fused == edges_3d
