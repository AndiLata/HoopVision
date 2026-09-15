#!/usr/bin/env python3
"""Run the pipeline on the synthetic clip and score it against truth.json.

This is the regression harness. Change a threshold in events.py and run this
to see, in numbers, what it cost you.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.detect import ReplayDetector
from hoopvision.metrics import summarise
from hoopvision.pipeline import Pipeline, PipelineConfig
from hoopvision.pose import ReplayPose


def run(sample_dir: str = "sample"):
    d = Path(sample_dir)
    truth = json.loads((d / "truth.json").read_text())
    raw = json.loads((d / "detections.json").read_text())

    pipe = Pipeline(ReplayDetector(d / "detections.json"),
                    ReplayPose(d / "poses.json"),
                    PipelineConfig(fps=truth["fps"]))
    for i in range(max(int(k) for k in raw["frames"]) + 1):
        pipe.step(i, None)
    return pipe, truth


def main() -> int:
    pipe, truth = run(sys.argv[1] if len(sys.argv) > 1 else "sample")
    got, want = pipe.records, truth["shots"]

    print(f"shots: detected {len(got)}, expected {len(want)}")
    if len(got) != len(want):
        print("FAIL: shot count mismatch")
        return 1

    rows, failures = [], 0
    checks = [
        ("outcome",            "outcome",              None,  None),
        ("release frame",      "release_frame",        2.0,   "frames"),
        ("release speed",      "release_speed_ms",     0.25,  "m/s"),
        ("release angle",      "release_angle_deg",    2.0,   "deg"),
        ("entry angle",        "entry_angle_deg",      1.5,   "deg"),
        ("miss offset",        "miss_offset_m",        0.02,  "m"),
        ("dribbles",           "dribbles_before",      0,     "count"),
        ("elbow at release",   "elbow_at_release_deg", 4.0,   "deg"),
        ("elbow set point",    "elbow_set_point_deg",  4.0,   "deg"),
    ]
    truth_key = {
        "release_frame": "release_frame", "release_speed_ms": "release_speed_ms",
        "release_angle_deg": "release_angle_deg", "entry_angle_deg": "entry_angle_deg",
        "miss_offset_m": "miss_offset_m", "outcome": "outcome",
        "dribbles_before": "dribbles", "elbow_at_release_deg": "elbow_at_release_deg",
        "elbow_set_point_deg": "elbow_set_point_deg",
    }

    for i, (g, w) in enumerate(zip(got, want), 1):
        for name, attr, tol, unit in checks:
            gv = getattr(g, attr)
            wv = w[truth_key[attr]]
            if tol is None:
                ok = gv == wv
                err = "" if ok else f"{gv} != {wv}"
            elif gv is None:
                ok, err = False, "missing"
            else:
                err_val = abs(float(gv) - float(wv))
                ok = err_val <= tol
                err = f"{err_val:.3f} {unit}"
            if not ok:
                failures += 1
            rows.append((i, name, gv, wv, "ok" if ok else "FAIL", err))

    w1 = max(len(r[1]) for r in rows) + 2
    print(f"\n{'#':<3}{'metric':<{w1}}{'got':>12}{'want':>12}{'':>6}  error")
    print("-" * (35 + w1))
    for i, name, gv, wv, status, err in rows:
        gs = f"{gv:.2f}" if isinstance(gv, float) else str(gv)
        ws = f"{wv:.2f}" if isinstance(wv, float) else str(wv)
        print(f"{i:<3}{name:<{w1}}{gs:>12}{ws:>12}{status:>6}  {err}")

    print("\nsession summary")
    for k, v in summarise(pipe.records).items():
        print(f"  {k:<22}{v if not isinstance(v, float) else round(v, 2)}")

    print(f"\n{failures} failing checks out of {len(rows)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
