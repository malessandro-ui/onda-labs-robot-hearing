#!/usr/bin/env python3
"""Onda v0.18 paired BEATs benchmark using the unchanged v0.17 renderer.

Pilot: original / 1 m direct / 1 m RMS-matched direct.
Extended: adds 3 m, 5 m, a *synthetic* echo FIR, calibrated noise and combo.
Source windows stay fixed across all conditions; propagation delay is compensated.
Research diagnostics, NOT independently verified semantic accuracy results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from onda_hearing.acoustic_scene import (
    AcousticSceneRenderer, AudioAsset, Binding, Emitter, MicArray,
    Pose3, RendererConfig, Schedule, load_wav,
)
from onda_hearing.semantics import OndaSoundBrain

SR = 16000
WINDOW = 3 * SR
MIC_POSITIONS = ((.05, .03, 0.), (-.045, .035, 0.),
                 (-.05, -.038, 0.), (.042, -.047, 0.))
TARGET = {"telephone_ringing": "Ringtone", "trumpet": "Trumpet"}
BASE_CONDITIONS = ("clean", "direct_1m", "gainmatched_1m")
EXT_CONDITIONS = BASE_CONDITIONS + (
    "direct_3m", "direct_5m", "echo_1m", "snr20_1m", "snr10_1m",
    "snr0_1m", "combined_5m_snr10",
)


def rms(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(y*y)))


def toy_echo_fir(channels: int) -> np.ndarray:
    """Documented synthetic reflections, not a measured or physical room RIR."""
    taps = np.zeros((int(.115 * SR) + 1, channels), dtype=np.float32)
    taps[0] = 1.
    for t_s, amplitude in ((.023, .24), (.055, .15), (.108, .08)):
        taps[round(t_s * SR)] = amplitude
    return taps


def render_aligned(source: np.ndarray, *, distance_m: float,
                   channel: int = 0, rir: np.ndarray | None = None,
                   noise_std: float = 0., seed: int = 2026):
    """Render 4-channel PCM sequentially and return one delay-aligned 3s mic.

    No circular shifts or reference-audio padding; last received source sample
    is retained by rendering beyond the end of the emitted asset.
    """
    if source.shape != (WINDOW,) or not np.isfinite(source).all():
        raise ValueError("expected finite 3-second mono source")
    if channel not in range(len(MIC_POSITIONS)):
        raise ValueError("invalid microphone channel")
    if not (0.1 <= distance_m <= 50.):
        raise ValueError("distance out of range")
    asset = AudioAsset(asset_id="source", samples=np.asarray(source, dtype=np.float32), sample_rate=SR)
    mics = MicArray(Binding(world=Pose3((0., 0., 0.))), MIC_POSITIONS)
    emitter = Emitter("emitter", "source", Binding(world=Pose3((distance_m, 0., 0.))),
                      Schedule(onset_s=0., loop=False), directivity="omni")
    cfg = RendererConfig(sample_rate=SR, noise_std=float(noise_std), seed=seed,
                         rir_taps=rir, clip_output=False)
    renderer = AcousticSceneRenderer(cfg, mics, [emitter], {"source": asset})
    separation = float(np.linalg.norm(np.asarray((distance_m, 0., 0.)) -
                                      np.asarray(MIC_POSITIONS[channel])))
    delay_samples = int(round(separation * SR / cfg.sound_speed_m_s))
    rir_tail = (len(rir)-1) if rir is not None else 0
    count = WINDOW + delay_samples + rir_tail + 8
    chunks = []
    cursor = 0
    while cursor < count:
        n = min(8192, count-cursor)  # renderer hard maximum: 2 seconds
        chunks.append(renderer.render(cursor, n))
        cursor += n
    all_channels = np.concatenate(chunks, axis=0)
    assert all_channels.shape == (count, len(MIC_POSITIONS))
    aligned = all_channels[delay_samples:delay_samples+WINDOW, channel].copy()
    assert aligned.shape == source.shape and np.isfinite(aligned).all()
    return aligned, {
        "distance_m": distance_m, "channel": channel, "delay_samples": delay_samples,
        "delay_ms": 1000*separation/cfg.sound_speed_m_s,
        "renderer": renderer.telemetry(),
    }


def matched_level(signal: np.ndarray, reference: np.ndarray):
    r_sig = rms(signal)
    r_ref = rms(reference)
    if r_sig < 1.e-10:
        raise ValueError("rendered signal is silent; cannot RMS-match")
    gain = r_ref/r_sig
    return (np.asarray(signal, dtype=np.float64)*gain).astype(np.float32), float(gain)


def add_condition(source: np.ndarray, condition: str, *, channel: int, seed: int):
    if condition == "clean":
        return source.copy(), {"description": "unaltered PCM16 source window"}
    distance = 5. if condition in {"direct_5m", "combined_5m_snr10"} else (
        3. if condition == "direct_3m" else 1.)
    rir = toy_echo_fir(len(MIC_POSITIONS)) if condition in {"echo_1m", "combined_5m_snr10"} else None
    base, meta = render_aligned(source, distance_m=distance, channel=channel, rir=rir, seed=seed)
    meta["rir_kind"] = "synthetic_3_echoes" if rir is not None else "none"
    meta["nominal_snr_db"] = None
    meta["measured_snr_db"] = None
    if condition == "gainmatched_1m":
        matched, gain = matched_level(base, source)
        meta["rms_match_gain"] = gain
        return matched, meta
    snr = {"snr20_1m": 20., "snr10_1m": 10., "snr0_1m": 0.,
           "combined_5m_snr10": 10.}.get(condition)
    if snr is not None:
        sigma = rms(base) / (10**(snr/20.))
        noisy, noisy_meta = render_aligned(source, distance_m=distance,
                                          channel=channel, rir=rir,
                                          noise_std=sigma, seed=seed)
        noise = noisy.astype(np.float64)-base.astype(np.float64)
        actual = 20*math.log10(rms(base)/rms(noise)) if rms(noise)>0 else float("inf")
        meta.update({"nominal_snr_db": snr, "measured_snr_db": actual,
                     "noise_std": sigma, "noise_renderer": noisy_meta["renderer"]})
        return noisy, meta
    return base, meta


def infer(brain, data: np.ndarray, target: str):
    import torch
    if brain.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    events = brain.classify(data, SR)
    if brain.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)*1000.
    if len(events) != 527:
        raise ValueError(f"expected 527 scores but received {len(events)}")
    scores = {event.label: float(event.score) for event in events}
    ranks = {event.label: index for index, event in enumerate(events, 1)}
    if target not in scores:
        raise ValueError(f"target label {target!r} not found in checkpoint vocabulary")
    return {"target_label": target, "target_score": scores[target],
            "target_rank": ranks[target], "top1": events[0].label,
            "top5": [{"label": e.label, "score": float(e.score)} for e in events[:5]],
            "scores": scores, "inference_ms": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", nargs="*", default=["2323", "26197", "59738", "68060"])
    parser.add_argument("--profile", choices=["pilot", "extended"], default="pilot")
    parser.add_argument("--mic-channel", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mic_channel not in range(4):
        parser.error("--mic-channel must be 0, 1, 2, or 3")
    manifest_path = Path(args.manifest)
    items = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len({str(x["clip_id"]) for x in items}) != len(items):
        raise ValueError("duplicate clip IDs in manifest")
    wanted = set(map(str, args.ids))
    if wanted:
        items = [x for x in items if str(x["clip_id"]) in wanted]
        missing = wanted-{str(x["clip_id"]) for x in items}
        if missing:
            raise ValueError(f"unknown clip IDs: {sorted(missing)}")
    if not items:
        raise ValueError("no selected clips")
    conditions = BASE_CONDITIONS if args.profile == "pilot" else EXT_CONDITIONS
    print(f"Loading Onda Sound Brain once on {args.device}; {len(items)} clips, {len(conditions)} conditions", flush=True)
    brain = OndaSoundBrain(
        checkpoint_path=os.environ["ONDA_BEATS_CHECKPOINT"],
        labels_path=os.environ["ONDA_BEATS_LABELS"],
        beats_module_dir=os.environ["ONDA_BEATS_MODULE_DIR"],
        device=args.device, top_k=527, min_score=0., allow_unsafe_checkpoint=True)
    results = []
    for item in items:
        clip_id = str(item["clip_id"])
        category = item["category"]
        if category not in TARGET:
            raise ValueError(f"unsupported category {category}")
        asset = load_wav(item["pcm16_path"])
        if asset.sample_rate != SR:
            raise ValueError(f"{clip_id} is not 16kHz")
        start = round(float(item["start_s"])*SR)
        if start < 0:
            raise ValueError("negative source window start")
        source = np.asarray(asset.samples[start:start+WINDOW], dtype=np.float32)
        if source.shape != (WINDOW,):
            raise ValueError(f"{clip_id} does not have an entire 3s selected window")
        if not np.isfinite(source).all():
            raise ValueError(f"nonfinite source samples: {clip_id}")
        source_hash = hashlib.sha256(source.tobytes()).hexdigest()

        # Pre-render all conditions. Apply one common gain per clip
        # when necessary to keep every condition within safe range.
        prepared = [
            (condition, *add_condition(
                source, condition,
                channel=args.mic_channel, seed=args.seed
            ))
            for condition in conditions
        ]

        max_pre_guard_peak = max(
            float(np.max(np.abs(data)))
            for _, data, _ in prepared
        )

        level_guard_gain = (
            min(1.0, 0.99 / max_pre_guard_peak)
            if max_pre_guard_peak else 1.0
        )

        if level_guard_gain < 1.0:
            print(
                f"{clip_id}: common headroom gain="
                f"{level_guard_gain:.6f} "
                f"(peak before={max_pre_guard_peak:.4f})",
                flush=True
            )

        for condition, raw_data, meta in prepared:
            data = (
                raw_data.astype(np.float64) * level_guard_gain
            ).astype(np.float32)

            peak = float(np.max(np.abs(data)))

            if not np.isfinite(data).all() or peak > 1.00001:
                raise ValueError(
                    f"{clip_id}/{condition}: "
                    f"invalid post-guard peak {peak:.6f}"
                )

            meta["level_guard_gain"] = level_guard_gain
            meta["max_pre_guard_peak_clip"] = max_pre_guard_peak

            pred = infer(brain, data, TARGET[category])
            row = {"clip_id": clip_id, "category": category,
                   "source_ground_truth_labels": item.get("ground_truth_labels", []),
                   "window_start_s": float(item["start_s"]),
                   "window_end_s": float(item["end_s"]),
                   "review_status": item.get("review_status", "pending"),
                   "source_sha256_float32": source_hash, "condition": condition,
                   "rms": rms(data), "peak_abs": peak,
                   "level_guard_gain": level_guard_gain,
                   "metadata": meta, **pred}
            results.append(row)
            print(f"{clip_id:>7} {condition:<21} target={pred['target_score']:.4f} "
                  f"rank={pred['target_rank']:>3} top1={pred['top1']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"suite": "Onda v0.18 paired acoustic-semantics pilot",
               "status": "completed; source windows require independent listening review",
               "device": brain.device, "checkpoint": str(brain.checkpoint),
               "checkpoint_sha256": hashlib.sha256(Path(brain.checkpoint).read_bytes()).hexdigest(),
               "manifest": str(manifest_path), "sample_rate": SR,
               "window_samples": WINDOW, "seed": args.seed, "mic_channel": args.mic_channel,
               "microphone_positions_m": MIC_POSITIONS,
               "acoustic_model": "Onda v0.17 analytic direct path; synthetic FIR only in extended conditions",
               "conditions": conditions, "rows": results}
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("clip_id", "category", "condition", "target_label",
                                               "target_score", "target_rank", "top1", "rms",
                                               "peak_abs", "level_guard_gain", "inference_ms", "window_start_s"))
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print("\nONDA V018 PAIRED BENCHMARK: COMPLETE")
    print(f"Rows: {len(results)}; clips: {len(items)}; conditions: {len(conditions)}")
    print("JSON:", out)
    print("CSV:", csv_path)
    print("NOTE: these are diagnostic predictions, not independently human-reviewed accuracy metrics")


if __name__ == "__main__":
    main()
