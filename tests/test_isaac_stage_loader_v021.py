"""Native USD geometry test; skipped where pxr is unavailable."""
import pytest
pxr = pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, Gf
from onda_hearing.simulators.isaac_scene_loader import IsaacSceneLoader, SceneLoaderConfig


def test_minimal_usd_floor_import():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    prim=UsdGeom.Mesh.Define(stage, '/Root/SM_floor01')
    prim.CreatePointsAttr([Gf.Vec3f(-2,-2,0),Gf.Vec3f(2,-2,0),
                          Gf.Vec3f(2,2,0),Gf.Vec3f(-2,2,0)])
    prim.CreateFaceVertexCountsAttr([4])
    prim.CreateFaceVertexIndicesAttr([0,1,2,3])
    config=SceneLoaderConfig(region_center_m=(0,0,1),region_radius_m=5,
                             allow_uncalibrated_demo_materials=True,
                             max_reflectors=1)
    result=IsaacSceneLoader(config).build(stage)
    assert len(result.geometry.surfaces)==1
    assert len(result.reflectors)==1
    assert result.public_summary()['triangles']==2
    assert result.metadata['meters_per_unit']==1.0
