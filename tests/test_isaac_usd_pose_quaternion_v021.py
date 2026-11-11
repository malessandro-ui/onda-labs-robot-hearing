"""USD world-pose regression tests; most run without Kit/pxr installed."""
from __future__ import annotations
import math
import sys
import types
import numpy as np
import pytest
from onda_hearing.acoustic_scene.adapters import IsaacUSDSceneSnapshot


class FakeQuat:
    def __init__(self, xyzw): self.xyzw = xyzw
    def GetImaginary(self): return self.xyzw[:3]
    def GetReal(self): return self.xyzw[3]


class FakeMatrix:
    def __init__(self, translation, basis, raw_quat, cleaned=None, det=None):
        self.translation = translation
        self.basis = basis
        self.raw_quat = raw_quat
        self.cleaned = cleaned
        self.det = float(np.linalg.det(np.asarray(basis))) if det is None else det
    def ExtractTranslation(self): return self.translation
    def ExtractRotationQuat(self): return FakeQuat(self.raw_quat)
    def RemoveScaleShear(self): return self.cleaned or self
    def GetDeterminant3(self): return self.det
    def __getitem__(self, i): return self.basis[i]


class FakeStage:
    def __init__(self, matrix, valid=True):
        self.matrix = matrix
        self.valid = valid
    def GetPrimAtPath(self, path):
        return types.SimpleNamespace(IsValid=lambda: self.valid)


@pytest.fixture
def pxr_stub(monkeypatch):
    package = types.ModuleType('pxr')
    class TimeCode:
        @staticmethod
        def Default(): return None
    class XformCache:
        def __init__(self, t): pass
        def GetLocalToWorldTransform(self, prim): return teststage[0].matrix
    teststage = [None]
    package.Usd = types.SimpleNamespace(TimeCode=TimeCode)
    package.UsdGeom = types.SimpleNamespace(XformCache=XformCache, GetStageMetersPerUnit=lambda stage: 0.01)
    monkeypatch.setitem(sys.modules, 'pxr', package)
    monkeypatch.setitem(sys.modules, 'pxr.Usd', package.Usd)
    monkeypatch.setitem(sys.modules, 'pxr.UsdGeom', package.UsdGeom)
    return teststage


def test_scaled_usd_matrix_cleans_rotation_before_quaternion(pxr_stub):
    angle = math.pi / 3
    c, s = math.cos(angle), math.sin(angle)
    pure = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    world = [[2*c, 2*s, 0], [-3*s, 3*c, 0], [0, 0, 4]]
    correct = (0, 0, math.sin(angle/2), math.cos(angle/2))
    clean_matrix = FakeMatrix((100, 200, 300), pure, correct)
    matrix = FakeMatrix((100, 200, 300), world, (0, 0, 1.5, 1.5), clean_matrix)
    stage = FakeStage(matrix)
    pxr_stub[0] = stage
    got = IsaacUSDSceneSnapshot(stage).capture('/Root/forklift/S_ForkliftBody')
    np.testing.assert_allclose(got.xyz, (1, 2, 3), atol=1e-12)
    np.testing.assert_allclose(got.quat_xyzw, correct, atol=1e-12)


def test_nonunit_after_scale_removal_is_normalized(pxr_stub):
    clean = FakeMatrix((0, 0, 0), np.eye(3), (0, 0, 0, 1.001))
    stage = FakeStage(FakeMatrix((0, 0, 0), np.eye(3), (0, 0, 0, 2), clean))
    pxr_stub[0] = stage
    pose = IsaacUSDSceneSnapshot(stage).capture('/Root/A')
    assert pose.quat_xyzw == (0, 0, 0, 1)


def test_degenerate_world_scale_rejected(pxr_stub):
    stage = FakeStage(FakeMatrix((0, 0, 0), np.zeros((3, 3)), (0, 0, 0, 1)))
    pxr_stub[0] = stage
    with pytest.raises(ValueError, match='degenerate USD world transform'):
        IsaacUSDSceneSnapshot(stage).capture('/Root/A')


def test_failed_scale_removal_rejected(pxr_stub):
    bad = [[2,0,0],[0,1,0],[0,0,1]]
    stage = FakeStage(FakeMatrix((0, 0, 0), bad, (0, 0, 0, 1)))
    pxr_stub[0] = stage
    with pytest.raises(ValueError, match='rotation could not be isolated'):
        IsaacUSDSceneSnapshot(stage).capture('/Root/A')


def test_missing_prim_rejected(pxr_stub):
    stage = FakeStage(FakeMatrix((0, 0, 0), np.eye(3), (0, 0, 0, 1)), valid=False)
    pxr_stub[0] = stage
    with pytest.raises(KeyError):
        IsaacUSDSceneSnapshot(stage).capture('/Root/missing')


def test_true_usd_scaled_transform_if_pxr_available():
    pxr = pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, '/Root/Forklift')
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate((1.5, 2.5, 3.5))
    api.SetRotate((0, 0, 60), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale((2, 3, 4))
    pose = IsaacUSDSceneSnapshot(stage).capture('/Root/Forklift')
    np.testing.assert_allclose(pose.xyz, (1.5, 2.5, 3.5), atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(pose.quat_xyzw), 1, atol=1e-8)
    # A unit quaternion alone is not enough: validate the intended z-rotation.
    np.testing.assert_allclose(abs(pose.quat_xyzw[2]), 0.5, atol=1e-6)
    np.testing.assert_allclose(abs(pose.quat_xyzw[3]), math.sqrt(3)/2, atol=1e-6)
