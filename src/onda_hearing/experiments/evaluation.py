"""Evaluation-only records: source truth must never feed the control policy."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

from onda_hearing.control import wrap_degrees


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    time_s: float
    head_yaw_deg: float
    command_yaw_deg: float
    acoustic_bearing_world_deg: float | None
    acoustic_score: float | None
    control_reason: str
    acoustic_reason: str
    ground_truth_world_deg: float | None
    ground_truth_alignment_error_deg: float | None


def evaluate_episode(rows: list[EvaluationRow], *, threshold_deg: float = 10.,
                     requires_detection: bool = True) -> dict:
    if not rows or not math.isfinite(threshold_deg) or threshold_deg <= 0:
        raise ValueError('nonempty episode and valid threshold required')
    final = rows[-1]
    observed = [r for r in rows if r.acoustic_bearing_world_deg is not None]
    if final.ground_truth_alignment_error_deg is None:
        max_motion = max(abs(wrap_degrees(r.head_yaw_deg - rows[0].head_yaw_deg)) for r in rows)
        return {'status': 'PASS' if max_motion <= 2 else 'FAIL', 'mode': 'silence',
                'maximum_uncommanded_motion_deg': max_motion,
                'acoustic_observation_count': len(observed),
                'final_head_yaw_deg': final.head_yaw_deg}
    final_error = abs(final.ground_truth_alignment_error_deg)
    passed = final_error <= threshold_deg and (not requires_detection or bool(observed))
    return {'status': 'PASS' if passed else 'FAIL', 'mode': 'sound',
            'final_alignment_error_deg': final_error,
            'final_head_yaw_deg': final.head_yaw_deg,
            'ground_truth_world_deg': final.ground_truth_world_deg,
            'acoustic_observation_count': len(observed),
            'alignment_threshold_deg': threshold_deg}


def save_episode(directory: str | Path, rows: list[EvaluationRow], summary: dict) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'telemetry.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    (path / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
