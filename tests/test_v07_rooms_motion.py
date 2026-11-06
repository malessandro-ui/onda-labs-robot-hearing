from __future__ import annotations
import numpy as np
import pytest
from onda_hearing.rooms import RoomMaterial,ShoeboxRoom,MeasuredRIR,image_source_rir
from onda_hearing.physics.propagation import PoseTrajectory
from onda_hearing.motion import TimedBearing,to_world,angular_error_deg

FS=16000

def room(a=.5):return ShoeboxRoom((6,5,3),tuple(RoomMaterial((a,)*7) for _ in range(6)))

def test_room_validation():
    with pytest.raises(ValueError): RoomMaterial((.3,)*6)
    with pytest.raises(ValueError): RoomMaterial((1.5,)*7)
    with pytest.raises(ValueError): ShoeboxRoom((0,4,2),tuple(RoomMaterial((.5,)*7) for _ in range(6)))

def test_direct_only_delay_and_inverse_distance():
    s=np.array([[1.,1.,1.]]);m=np.array([[2.,1.,1.],[3.,1.,1.]])
    h=image_source_rir(room(),s,m,sample_rate_hz=FS,max_order=0)
    # Expected first direct arrivals and inverse-distance amplitude, integral of sinc.
    assert abs(np.argmax(np.abs(h[:,0,0]))-FS/343)<1
    assert abs(np.argmax(np.abs(h[:,0,1]))-2*FS/343)<1
    assert np.sum(h[:,0,0]**2)**.5 > 1.8*np.sum(h[:,0,1]**2)**.5

def test_frequency_dependent_reflections():
    s=np.array([[1.3,1.1,1.2]]);m=np.array([[2.2,1.7,1.1]])
    no=image_source_rir(room(1.),s,m,max_order=1)
    direct=image_source_rir(room(1.),s,m,max_order=0)
    assert np.allclose(no,direct,atol=1e-10)
    reflected=image_source_rir(room(0.),s,m,max_order=1)
    assert np.linalg.norm(reflected-direct)>0.1
    mixed=ShoeboxRoom((6,5,3),tuple(RoomMaterial((.01,.01,.01,.2,.5,.9,.99)) for _ in range(6)))
    spectral=image_source_rir(mixed,s,m,max_order=1)
    assert not np.allclose(spectral,reflected)

def test_measured_rir_roundtrip_and_order(tmp_path):
    h=np.zeros((30,2,3));h[5,:,:]=.5
    r=MeasuredRIR(h,FS,('a','b','c'),('s0','s1'),'synthetic fixture')
    p=tmp_path/'rir.npz';r.save(p);other=MeasuredRIR.load(p)
    np.testing.assert_array_equal(other.rir,h)
    x=np.zeros((4,2));x[0]=1
    y=other.render(x,sample_rate_hz=FS,source_ids=('s0','s1'),microphone_names=('a','b','c'))
    assert y.shape==(33,3)
    assert np.allclose(y[5],1.)
    with pytest.raises(ValueError,match='order'):other.render(x,sample_rate_hz=FS,source_ids=('s1','s0'),microphone_names=('a','b','c'))
    with pytest.raises(ValueError,match='sample rate'):other.render(x,sample_rate_hz=24000,source_ids=('s0','s1'),microphone_names=('a','b','c'))

def test_rotation_world_stability():
    t=np.linspace(0,1,101);a=t*np.pi
    rotations=np.stack([np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1.]]) for z in a])
    p=PoseTrajectory(t,np.zeros((len(t),3)),rotations)
    for ti in np.linspace(0,1,21):
        local=np.deg2rad(35)-np.pi*ti
        obs=TimedBearing(local,float(ti),'array','sim')
        world=to_world(obs,p,pose_clock_id='sim',array_frame_id='array',max_pose_gap_s=.01)
        assert angular_error_deg(35,np.rad2deg(world.azimuth_rad))<1e-7

def test_frame_clock_and_staleness_rejected():
    p=PoseTrajectory([0.,1.],np.zeros((2,3)))
    b=TimedBearing(0.,.5,'array','capture')
    with pytest.raises(ValueError,match='clock'):to_world(b,p,pose_clock_id='sim',array_frame_id='array')
    with pytest.raises(ValueError,match='frame'):to_world(b,p,pose_clock_id='capture',array_frame_id='other')
    with pytest.raises(ValueError,match='far'):to_world(b,p,pose_clock_id='capture',array_frame_id='array')
    with pytest.raises(ValueError,match='extrapolation'):to_world(TimedBearing(0.,2.,'array','capture'),p,pose_clock_id='capture',array_frame_id='array')
