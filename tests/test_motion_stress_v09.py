import math
import sys
from pathlib import Path
import numpy as np
import pytest
from onda_hearing.physics.propagation import PoseTrajectory
from onda_hearing.spatial.motion_robustness import ClockAlignment, perturb_poses


def poses():
    return PoseTrajectory(np.array([0., .1, .2]),np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]))


def test_clock_alignment_correct_direction():
    raw=PoseTrajectory(np.array([-.01,.09,.19]),poses().positions_m)
    shifted=ClockAlignment(.01,'capture','robot').align(raw,audio_clock_id='capture',pose_clock_id='robot')
    assert np.allclose(shifted.times_s,poses().times_s)


def test_clock_alignment_rejects_wrong_clocks():
    with pytest.raises(ValueError,match='clock'):
        ClockAlignment(.01,'capture','robot').align(poses(),audio_clock_id='sim',pose_clock_id='robot')


def test_invalid_offsets_rejected():
    with pytest.raises(ValueError):ClockAlignment(float('nan'),'capture','robot')
    with pytest.raises(ValueError):ClockAlignment(0.,'capture','')


def test_jitter_deterministic_and_does_not_modify_input():
    before=poses()
    a=perturb_poses(before,seed=123,yaw_std_deg=2,position_std_m=.01)
    b=perturb_poses(before,seed=123,yaw_std_deg=2,position_std_m=.01)
    assert np.array_equal(a.rotations,b.rotations)
    assert np.array_equal(a.positions_m,b.positions_m)
    assert np.allclose(before.positions_m,[[0,0,0],[1,0,0],[2,0,0]])


def test_no_noise_exact_identity():
    before=poses()
    out=perturb_poses(before,seed=0)
    assert np.array_equal(before.positions_m,out.positions_m)
    assert np.array_equal(before.rotations,out.rotations)


def test_invalid_noise_rejected():
    with pytest.raises(ValueError):perturb_poses(poses(),seed=1,yaw_std_deg=-1)
    with pytest.raises(ValueError):perturb_poses(poses(),seed=-2)


def test_rotation_remains_orthonormal():
    out=perturb_poses(poses(),seed=4,yaw_std_deg=10)
    for R in out.rotations:
        assert np.allclose(R.T@R,np.eye(3),atol=1e-12)
        assert math.isclose(np.linalg.det(R),1.,abs_tol=1e-12)


def test_small_paired_benchmark(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
    from motion_stress_v09 import run
    report=run(tmp_path,limit=1)
    assert (tmp_path/'mixtures'/'000.npz').exists()
    assert (tmp_path/'results.csv').exists()
    assert report['overall']['default_v08']['cases']==1
    assert len(report['overall'])==4
