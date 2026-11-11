from dataclasses import replace
import json
import numpy as np
import pytest
from onda_hearing.acoustic_scene.types import Pose3
from onda_hearing.simulators.isaac_microphone_array_v021b import (
    MicrophoneRig,MicrophoneRigStore,PRESETS,MARKER_PATH)


def test_presets_and_custom_layouts():
    for key,coords in PRESETS.items():
        mic=MicrophoneRig(local_positions_m=coords)
        array,world,pose=mic.capture()
        assert len(array.local_positions_m)==len(coords)
        assert len(world)==len(coords)
        assert pose.xyz==(-1.,3.6,1.5)
    coords=tuple((.05*np.cos(i),.05*np.sin(i),0) for i in range(16))
    assert len(MicrophoneRig(local_positions_m=coords).capture()[1])==16

@pytest.mark.parametrize('points', [((0,0,0),)*3,((0,0,0),(1,0,0)),
                                       ((0,0,0),(1,0,0),(float('nan'),0,0))])
def test_reject_bad_points(points):
    with pytest.raises(ValueError):MicrophoneRig(local_positions_m=points)


def test_pose_and_rotation_of_attached_rig():
    # Parent frame rotated 90 degrees around +Z. Mic local X becomes world +Y.
    q=(0.,0.,2**-.5,2**-.5)
    mic=MicrophoneRig(center_m=(1.,0,0),anchor_prim_path='/Root/Robot',
        local_positions_m=((0,0,0),(.08,0,0),(0,.08,0)))
    array,world,pose=mic.capture(lambda path:Pose3((2.,3.,1.),q))
    assert pose.xyz==pytest.approx((2.,4.,1.))
    assert world[1]==pytest.approx([2.,4.08,1.],abs=1e-6)
    assert world[2]==pytest.approx([1.92,4.,1.],abs=1e-6)


def test_persistence_scene_scoping_and_rollback(tmp_path,monkeypatch):
    manifest=tmp_path/'mics.json'
    store=MicrophoneRigStore(manifest,'stage-A')
    rig=store.set_rig({'preset':'hexagon6','center_m':(1.,2.,3.)})
    assert len(rig.local_positions_m)==6
    assert manifest.exists()
    assert MicrophoneRigStore(manifest,'stage-B').load()['reason']=='scene_id_mismatch'
    new=MicrophoneRigStore(manifest,'stage-A')
    assert new.load()['loaded'] is True
    assert len(new.rig.local_positions_m)==6
    before=new.rig
    def fail():raise OSError('disk full')
    monkeypatch.setattr(new,'save',fail)
    with pytest.raises(OSError):new.set_rig({'center_m':(4.,5.,6.)})
    assert new.rig==before


def test_marker_persists_and_can_be_reconstituted(tmp_path):
    s=MicrophoneRigStore(tmp_path/'p.json','scene')
    s.set_rig({'anchor_prim_path':MARKER_PATH,'center_m':(0,0,0),
               'marker_world_position_m':(7.,8.,1.5)})
    restored=MicrophoneRigStore(tmp_path/'p.json','scene')
    assert restored.load()['loaded']
    arr,points,pose=restored.rig.capture(lambda _:Pose3((7.,8.,1.5)))
    assert len(points)==4 and pose.xyz==(7.,8.,1.5)
