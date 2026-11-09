"""Native MuJoCo object-bound acoustic scene smoke. Requires `mujoco` optional.

Creates a tiny MJCF scene, animates a source-body using mocap, samples world
transforms into a PoseTimeline, then renders 4-channel PCM. This tests actual
world pose bindings; it is NOT a trainable robot policy or acoustic calibration.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from onda_hearing.acoustic_scene import (
    AudioAsset, Pose3,Binding,Emitter,Schedule,MicArray,RendererConfig,
    AcousticSceneRenderer,PoseTimeline,MuJoCoPoseSnapshot,load_wav)

MJCF='''<mujoco model="onda_v017_attached_sound">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_head" pos="0 0 0"><geom type="sphere" size="0.08"/></body>
    <body name="phone" mocap="true" pos="2 0 0"><geom type="box" size="0.07 0.1 0.03"/></body>
  </worldbody>
</mujoco>'''

def main():
    import mujoco
    p=argparse.ArgumentParser()
    p.add_argument('--wav',required=True)
    args=p.parse_args()
    asset=load_wav(args.wav,'phone')
    if asset.sample_rate!=16000:raise ValueError('need 16k mono WAV')
    model=mujoco.MjModel.from_xml_string(MJCF)
    data=mujoco.MjData(model)
    snap=MuJoCoPoseSnapshot(model,data)
    timeline=PoseTimeline()
    mics=MicArray(Binding(object_path='robot_head'),
        ((.05,.03,0),(-.045,.035,0),(-.05,-.038,0),(.042,-.047,0)))
    render=AcousticSceneRenderer(RendererConfig(),mics,
        [Emitter('phone','phone',Binding(object_path='phone'),Schedule(loop=True))],
        {'phone':asset},timeline)
    arrays=[];index=0;chunks=50
    def capture():
        for name in ('robot_head','phone'):
            timeline.append(name,float(data.time),snap.capture('body',name))
    mujoco.mj_forward(model,data)
    capture()
    for chunk in range(chunks):
        n=512
        until=(index+n)/16000
        while data.time<until+0.003:
            data.mocap_pos[0]=[2.,np.sin(data.time*2),0.]
            mujoco.mj_step(model,data)
            capture()
        pcm=render.render(index,n)
        arrays.append(pcm);index+=n
    out=np.concatenate(arrays)
    assert out.shape==(chunks*512,4) and np.isfinite(out).all()
    print('Mujoco moving phone PCM PASS',out.shape,render.telemetry())

if __name__=='__main__':main()
