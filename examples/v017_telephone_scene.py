"""Onda v0.17 real integration example: WAV -> scene -> v0.16 PCM -> brain.

Run with a legal 16 kHz mono WAV. Requires Onda Sound Brain checkpoint, ordered
527-label mapping, trusted official BEATs.py. No simulator or sound card needed.
"""
from __future__ import annotations
import argparse
from time import monotonic_ns, sleep
import numpy as np

from onda_hearing.acoustic_scene import (
    Pose3,Binding,Schedule,Emitter,MicArray,AudioAsset,load_wav,
    RendererConfig,AcousticSceneRenderer,StaticPoseResolver)
from onda_hearing.semantics import OndaSoundBrain,AsyncSemanticHearing,PolicyEncoder,UnifiedHearing
from onda_hearing.streaming.core import StreamConfig,StreamingHearing,GCCPHATSRP,Pose,PCMBuffer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wav',required=True,help='Your own licensed mono 16kHz WAV file')
    p.add_argument('--checkpoint',help='Optional override for ONDA_BEATS_CHECKPOINT')
    p.add_argument('--labels',help='Ordered AudioSet label file; or ONDA_BEATS_LABELS')
    p.add_argument('--beats-dir',help='Official trusted BEATs.py directory; or ONDA_BEATS_MODULE_DIR')
    p.add_argument('--device',default=None)
    p.add_argument('--duration',type=float,default=4.0)
    args=p.parse_args()
    sound=load_wav(args.wav,'ring')
    if sound.sample_rate!=16000:raise SystemExit('convert your WAV to 16kHz mono PCM before use')
    mics=np.array([[.05,.03,0],[-.045,.035,0],[-.05,-.038,0],[.042,-.047,0]],dtype=float)
    scene=AcousticSceneRenderer(RendererConfig(),
        MicArray(Binding(object_path='/World/RobotHead'),tuple(map(tuple,mics))),
        [Emitter('phone','ring',Binding(object_path='/World/Desk/Phone'),
                 Schedule(onset_s=.35,loop=True))],
        {'ring':sound},
        StaticPoseResolver({'/World/RobotHead':Pose3((0,0,0)),
                            '/World/Desk/Phone':Pose3((2,1,0))}))
    # Trusted weights only. No downloaded files, no silent mock classification.
    brain=OndaSoundBrain(checkpoint_path=args.checkpoint,labels_path=args.labels,
                         beats_module_dir=args.beats_dir,device=args.device,
                         allow_unsafe_checkpoint=True)
    start_ns=monotonic_ns()
    stream=StreamingHearing(StreamConfig(),start_ns,GCCPHATSRP(mics))
    semantic_buffer=PCMBuffer(StreamConfig(channels=4,window_samples=32000,hop_samples=8000),start_ns)
    semantic=AsyncSemanticHearing(semantic_buffer,brain,window_samples=32000,hop_samples=8000)
    hearing=UnifiedHearing(stream,semantic,PolicyEncoder(tuple(brain.labels)))
    try:
        hearing.push_pose(Pose(start_ns,(0.,0.,0.),(0.,0.,0.,1.)))
        cursor=0;chunk=512;total=int(args.duration*16000)
        while cursor<total:
            n=min(chunk,total-cursor)
            pcm=scene.render(cursor,n)
            # Pose uses the SAME source clock as the stream. No implicit wall-time conversion.
            end_ns=start_ns+(cursor+n)*1_000_000_000//16000
            hearing.push_pose(Pose(end_ns,(0.,0.,0.),(0.,0.,0.,1.)))
            hearing.push_audio(pcm,cursor)
            cursor+=n
            sleep(n/16000)  # wall-paced integration demo only
        sleep(.3)
        features,spatial,events,metrics=hearing.poll(start_ns+cursor*1_000_000_000//16000)
        print('Onda Sound Brain:',brain.name)
        print('Spatial:',spatial)
        print('Top semantic events:',events.events,'status:',events.reason)
        print('Policy feature shape:',features.shape)
        print('Telemetry:',metrics)
    finally:
        if not hearing.close():print('WARNING: hearing worker did not exit within timeout')

if __name__=='__main__':main()
