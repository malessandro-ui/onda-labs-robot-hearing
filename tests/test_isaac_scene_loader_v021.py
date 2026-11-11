"""Unit tests that run without Isaac Sim; native USD tests in separate file."""
import math
import pytest

from onda_hearing.simulators.isaac_scene_loader import (
    SceneLoaderConfig, AcousticMaterialProfile, IsaacSceneLoader,
    category_from_path, distance_point_aabb, demo_materials, scene_identifier,
)


def test_categories():
    assert category_from_path('/Root/SM_floor27/SM_floor02') == 'floor'
    assert category_from_path('/Root/SM_WallA_InnerCorner/Section0') == 'wall'
    assert category_from_path('/Root/SM_RackShelf_4480/Section2') == 'rack_shelf'
    assert category_from_path('/Root/forklift/S_ForkliftBody') == 'forklift'
    assert category_from_path('/Root/WoodCrate') == 'pallet'
    assert category_from_path('/Root/Cube123') == 'other'


def test_point_box_distance():
    assert distance_point_aabb((-1, 3.6, 1.5), (-4, 0.6, 0), (2, 6.6, 0)) == pytest.approx(1.5)
    assert distance_point_aabb((0, 0, 0), (-1, -1, -1), (1, 1, 1)) == 0
    assert distance_point_aabb((3, 4, 0), (-1, -1, -1), (1, 1, 1)) == pytest.approx(math.sqrt(13))
    with pytest.raises(ValueError): distance_point_aabb((0, 0, 0), (1, 0, 0), (0, 1, 1))


def test_materials_are_explicitly_unmeasured():
    mapping = demo_materials()
    assert 'UNCALIBRATED' in mapping['floor'].transmission.provenance
    assert 'UNCALIBRATED' in mapping['wall'].reflection.provenance
    with pytest.raises(ValueError): IsaacSceneLoader(SceneLoaderConfig())
    assert IsaacSceneLoader(SceneLoaderConfig(allow_uncalibrated_demo_materials=True))


@pytest.mark.parametrize('kw', [
    {'max_reflectors':25}, {'max_surfaces':0}, {'max_triangles_total':100},
    {'region_radius_m':0}, {'categories':('invented',)},
    {'region_center_m':(float('nan'),0,0)},
])
def test_config_rejects_invalid(kw):
    with pytest.raises(ValueError): SceneLoaderConfig(**kw)


def test_scene_id_changes_with_identifier():
    class Layer:
        def __init__(self, name): self.identifier=name
    class FakeStage:
        def __init__(self, name): self.layer=Layer(name)
        def GetRootLayer(self): return self.layer
    assert scene_identifier(FakeStage('/tmp/a.usd')) == scene_identifier(FakeStage('/tmp/a.usd'))
    assert scene_identifier(FakeStage('/tmp/a.usd')) != scene_identifier(FakeStage('/tmp/b.usd'))
