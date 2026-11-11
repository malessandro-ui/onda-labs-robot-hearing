#!/usr/bin/env python3
"""Onda v0.18: preliminary clean-audio semantic baseline."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from onda_hearing.acoustic_scene import load_wav
from onda_hearing.semantics import OndaSoundBrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]

    assert len(items) == 20

    brain = OndaSoundBrain(
        checkpoint_path=os.environ["ONDA_BEATS_CHECKPOINT"],
        labels_path=os.environ["ONDA_BEATS_LABELS"],
        beats_module_dir=os.environ["ONDA_BEATS_MODULE_DIR"],
        device=args.device,
        top_k=527,
        min_score=0.0,
        allow_unsafe_checkpoint=True,
    )

    target_labels = {
        "telephone_ringing": ["Ringtone"],
        "trumpet": ["Trumpet"],
    }

    results = []

    for item in items:
        asset = load_wav(item["pcm16_path"])
        start = round(item["start_s"] * 16000)
        segment = asset.samples[start:start + 48000]

        if len(segment) != 48000:
            raise ValueError(f"Invalid window: {item['clip_id']}")

        if not np.isfinite(segment).all():
            raise ValueError(f"Non-finite audio: {item['clip_id']}")

        t0 = time.perf_counter()
        events = brain.classify(segment, 16000)

        if args.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert len(events) == 527

        scores = {e.label: e.score for e in events}
        ranks = {e.label: i for i, e in enumerate(events, 1)}

        targets = target_labels[item["category"]]

        result = {
            "clip_id": item["clip_id"],
            "category": item["category"],
            "source_labels": item["ground_truth_labels"],
            "window_start_s": item["start_s"],
            "window_end_s": item["end_s"],
            "rms": float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))),
            "inference_ms": elapsed_ms,
            "top_5": [
                {"label": e.label, "score": e.score}
                for e in events[:5]
            ],
            "target_scores": {
                label: scores.get(label)
                for label in targets
            },
            "target_ranks": {
                label: ranks.get(label)
                for label in targets
            },
            "all_scores": scores,
        }

        results.append(result)

        label = targets[0]
        print(
            f"{item['clip_id']:>7} "
            f"{item['category']:<20} "
            f"rank={str(ranks.get(label)):>4} "
            f"score={scores.get(label, 0):.4f} "
            f"top1={events[0].label}"
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps({
        "suite": "Onda v0.18 clean semantic baseline",
        "status": "completed",
        "checkpoint": str(brain.checkpoint),
        "device": brain.device,
        "clip_count": len(results),
        "window_duration_s": 3.0,
        "results": results,
    }, indent=2))

    print("\nONDA V018 CLEAN BASELINE: COMPLETE")
    print("Recordings:", len(results))
    print("Report:", output)


if __name__ == "__main__":
    main()
