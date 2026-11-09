#!/usr/bin/env python3
"""Explicit validation probe for a trusted local Onda Sound Brain checkpoint.

Not an accuracy benchmark: check model load, probability range, AudioSet labels,
and repeatable inference on a provided mono 16 kHz WAV.
"""
import argparse
import json
from onda_hearing.acoustic_scene import load_wav
from onda_hearing.semantics import OndaSoundBrain


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wav',required=True,help='16k mono WAV supplied by user')
    p.add_argument('--checkpoint')
    p.add_argument('--labels')
    p.add_argument('--beats-dir')
    p.add_argument('--device')
    args=p.parse_args()
    a=load_wav(args.wav)
    if a.sample_rate!=16000:raise SystemExit('BEATs requires 16k mono; convert explicitly')
    brain=OndaSoundBrain(checkpoint_path=args.checkpoint,labels_path=args.labels,
                        beats_module_dir=args.beats_dir,device=args.device,
                        allow_unsafe_checkpoint=True)
    results=brain.classify(a.samples,16000)
    print(json.dumps({'brain':brain.name,'device':brain.device,
        'checkpoint':str(brain.checkpoint),'samples':len(a.samples),
        'sample_rate':a.sample_rate,'events':[{'label':e.label,'score':e.score} for e in results]},indent=2))

if __name__=='__main__':main()
