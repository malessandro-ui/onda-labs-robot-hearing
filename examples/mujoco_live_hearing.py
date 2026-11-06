"""Optional MuJoCo demo: real live mjData site poses -> Onda microphone PCM.

Run: pip install -e '.[mujoco]' && python examples/mujoco_live_hearing.py
Outputs: mujoco_hearing.wav (4 channels, 16 kHz). Does not require a GUI.
"""
from __future__ import annotations
from pathlib import Path
import math
import wave
import numpy as np

from onda_hearing.contracts import MicrophoneArray
from onda_hearing.runtime import HearingRuntime, ScheduledSource
from onda_hearing.simulators import MuJoCoSceneAdapter

XML = '''<mujoco model="onda_hearing_demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="head" pos="0 0 1">
      <joint name="head_yaw" type="hinge" axis="0 0 1" damping="1"/>
      <geom type="sphere" size="0.09" mass="1" rgba="0.2 0.6 0.8 1"/>
      <site name="head_array" pos="0.08 0 0" size="0.01"/>
    </body>
    <site name="ringing_phone" pos="1.4 0.8 1.0" size="0.02"/>
    <site name="knock" pos="-1.2 0.6 1.2" size="0.02"/>
  </worldbody>
  <actuator><position name="head_servo" joint="head_yaw" kp="20"/></actuator>
</mujoco>'''


def main():
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit('Install optional simulator dependency: pip install -e ".[mujoco]"') from exc
    model=mujoco.MjModel.from_xml_string(XML)
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    adapter=MuJoCoSceneAdapter(model,data,array_site='head_array',
                               source_sites={'phone':'ringing_phone','knock':'knock'})
    fs=16000
    t=np.arange(fs*2)/fs
    ring=(.018*np.sin(2*np.pi*820*t)*np.sin(2*np.pi*3*t)**2).astype(np.float32)
    rng=np.random.default_rng(17)
    knock=np.zeros_like(ring);knock[0:1000]=.015*rng.normal(size=1000)*np.exp(-np.arange(1000)/150)
    array=MicrophoneArray.from_positions(
        [[.04,.04,0],[-.04,.04,0],[-.04,-.04,0],[.04,-.04,0]],
        array_id='head_array',frame_id='robot/head',geometry_id='square80mm')
    hearing=HearingRuntime(array,(ScheduledSource('phone',ring,0),ScheduledSource('knock',knock,250_000_000)),
                           sample_rate_hz=fs)
    hearing.advance(adapter.snapshot(0))
    chunks=[];frames=0
    for step in range(500):
        data.ctrl[0]=.7*math.sin(2*math.pi*0.45*step*model.opt.timestep)
        mujoco.mj_step(model,data)
        if (step+1)%10==0:
            timestamp_ns=round(float(data.time)*1e9)
            result=hearing.advance(adapter.snapshot(timestamp_ns))
            if result is not None:
                chunks.append(result.audio.pcm)
                frames+=1
    pcm=np.concatenate(chunks,axis=0)
    assert pcm.shape==(fs,4),pcm.shape
    output=Path('mujoco_hearing.wav')
    with wave.open(str(output),'wb') as wf:
        wf.setnchannels(pcm.shape[1]);wf.setsampwidth(2);wf.setframerate(fs)
        wf.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
    print(f'Generated {output} shape={pcm.shape} chunks={frames}; robot head rotated while phone and knock sources remained fixed')
    print('This tests audio synthesis and pose integration, NOT correct localization or policy behavior.')

if __name__=='__main__': main()
