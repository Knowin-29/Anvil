from __future__ import annotations

import glob
import json



def test_all_11_scene_fixtures_parse_as_valid_scenegraph():
    fixtures = sorted(glob.glob('tests/fixtures/scenes/scene_*.json'))
    assert len(fixtures) == 11, f'Expected 11 fixtures, got {len(fixtures)}'

    for path in fixtures:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)

        assert 'image_id' in data
        assert 'nodes' in data
        assert 'edges' in data
        assert 'blocking_dag' in data
        assert 'ternary_relations' in data

        if data['image_id'] == 'scene_11_tabletop_pick_place':
            assert 'support_relations' in data
            assert data['support_relations']
            assert 'placement_regions' in data
            assert data['placement_regions']

        for node in data['nodes']:
            assert 'id' in node
            assert 'class_name' in node
            assert 'bbox' in node
            assert len(node['bbox']) == 4
