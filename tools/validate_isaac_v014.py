#!/usr/bin/env python3
"""Reproduce Onda v0.14 native Isaac Sim acceptance tests."""

import csv
import json
import math
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "benchmark_results" / f"v014_native_validation_{STAMP}"
OUT.mkdir(parents=True, exist_ok=False)

GENERATOR = ROOT / "examples/isaac_make_demo_stage.py"
EXPERIMENT = ROOT / "examples/isaac_sim_hear_and_turn.py"

CASES = [
    ("plus38", 38.0, "sound"),
    ("minus38", -38.0, "sound"),
    ("silence", 38.0, "silence"),
]

GENERATE_CODE = """
import runpy
import sys
from isaacsim import SimulationApp

output, bearing = sys.argv[1:3]
sys.argv = [
    "isaac_make_demo_stage.py",
    "--output", output,
    "--source-bearing", bearing,
]

app = SimulationApp({"headless": True})
try:
    runpy.run_path(
        "examples/isaac_make_demo_stage.py",
        run_name="__main__",
    )
finally:
    app.close()
"""

def execute(command, log_path):
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            log.write("\nONDA VALIDATOR: Process timed out.\n")
            return 124

def validate(case_dir, mode):
    summary_path = case_dir / "summary.json"
    telemetry_path = case_dir / "telemetry.csv"
    audio_path = case_dir / "microphones.wav"

    if not all(p.is_file() for p in
               (summary_path, telemetry_path, audio_path)):
        raise RuntimeError("Missing output artifacts")

    summary = json.loads(summary_path.read_text())

    with telemetry_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()

    if not rows:
        raise RuntimeError("Telemetry has no observations")

    if channels != 4 or sample_rate != 16000 or frames != 38400:
        raise RuntimeError("Unexpected microphone WAV format")

    if summary.get("status") != "PASS":
        raise RuntimeError("SDK acceptance gate returned FAIL")

    if summary.get("source_truth_exposed_to_controller") is not False:
        raise RuntimeError("Ground-truth isolation check failed")

    if mode == "sound":
        error = summary.get("final_alignment_error_deg")
        if (error is None or not math.isfinite(error)
                or error > 10 or error < 0):
            raise RuntimeError("Alignment exceeds 10 degrees")

        if summary.get("accepted_decision_count", 0) <= 0:
            raise RuntimeError("No acoustic decisions accepted")

    else:
        if summary.get("accepted_decision_count") != 0:
            raise RuntimeError("Silence produced accepted decisions")

        if summary.get("max_head_movement_deg", float("inf")) > 2:
            raise RuntimeError("Head moved during silence")

    return summary, len(rows)

results = []

for name, bearing, mode in CASES:
    print(f"\nONDA: Running {name}", flush=True)

    case_dir = OUT / name
    case_dir.mkdir()

    usd = OUT / f"{name}.usd"

    generation_exit = execute(
        [sys.executable, "-c", GENERATE_CODE,
         str(usd), str(bearing)],
        case_dir / "stage_generation.log",
    )

    record = {
        "case": name,
        "bearing_deg": bearing,
        "mode": mode,
        "stage_exit_code": generation_exit,
    }

    if generation_exit == 0 and usd.is_file():
        experiment_exit = execute(
            [
                sys.executable, "-u", str(EXPERIMENT),
                "--usd", str(usd),
                "--robot-prim", "/World/Robot",
                "--array-prim",
                "/World/Robot/Head/MicrophoneArray",
                "--head-joint", "head_yaw",
                "--source-prim", "/World/Beacon",
                "--duration", "2.4",
                "--mode", mode,
                "--headless", "--strict",
                "--output-dir", str(case_dir),
            ],
            case_dir / "experiment.log",
        )

        record["experiment_exit_code"] = experiment_exit

        try:
            if experiment_exit != 0:
                raise RuntimeError(
                    f"Experiment exited with {experiment_exit}"
                )

            summary, observations = validate(case_dir, mode)
            record.update(summary)
            record["observation_count"] = observations
            record["validated"] = True

        except Exception as exc:
            record["validated"] = False
            record["validation_error"] = str(exc)
    else:
        record["validated"] = False
        record["validation_error"] = "Stage generation failed"

    results.append(record)
    print(
        f"ONDA: {name} = "
        f"{'PASS' if record['validated'] else 'FAIL'}",
        flush=True,
    )

passed = sum(bool(r["validated"]) for r in results)

report = {
    "sdk_version": "0.14.0a0",
    "simulator": "Isaac Sim",
    "timestamp_utc": STAMP,
    "passed": passed,
    "total": len(results),
    "status": "PASS" if passed == len(results) else "FAIL",
    "cases": results,
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

lines = [
    "# Onda v0.14 Native Isaac Sim Validation",
    "",
    f"Overall: **{report['status']}** "
    f"({passed}/{len(results)} passed)",
    "",
    "| Case | Result | Final yaw | Alignment error |",
    "|---|---|---:|---:|",
]

for r in results:
    yaw = r.get("final_head_yaw_deg")
    error = r.get("final_alignment_error_deg")
    yaw_text = f"{yaw:.3f}°" if isinstance(yaw, (int, float)) else "—"
    error_text = (
        f"{error:.3f}°"
        if isinstance(error, (int, float))
        else "—"
    )
    lines.append(
        f"| {r['case']} | "
        f"{'PASS' if r['validated'] else 'FAIL'} | "
        f"{yaw_text} | {error_text} |"
    )

(OUT / "validation_report.md").write_text(
    "\n".join(lines) + "\n"
)

print("\n" + "\n".join(lines), flush=True)
print(f"\nResults: {OUT}", flush=True)

sys.exit(0 if passed == len(results) else 1)
