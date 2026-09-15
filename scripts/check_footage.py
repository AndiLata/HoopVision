#!/usr/bin/env python3
"""Answer one question about a clip: is it worth labelling?

    python scripts/check_footage.py --video clips/test.mov --rim rim.json

Run this on a 60-second test clip BEFORE committing four hours to labelling.
It reports frame rate, how big the ball actually is in pixels, how often it is
found, and where it gets lost - then says plainly whether to re-shoot.

It uses zero-shot COCO detection, which is weaker than the model you will
eventually train. That is deliberate: a clip that is trackable with COCO is
certainly trackable after fine-tuning, and the failures COCO exposes
(backlit sky, ball too small, camera too far) are camera problems that no
amount of labelling will fix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.bootstrap import CocoBallDetector, FixedRim, analyse_detections, verdicts

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--rim", help="rim.json from mark_rim.py (optional for this check)")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=1200,
                    help="cap the sample; 10 s at 120 fps is plenty to judge a setup")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rim = FixedRim.load(args.rim) if args.rim else FixedRim(0, 0, 10, 4)
    det = CocoBallDetector(rim, args.weights, args.device, args.conf, args.imgsz)

    hits, idx = [], 0
    while idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        d = det.detect(idx, frame)
        if d.ball is not None:
            hits.append((idx, d.ball.radius_px * 2, d.ball.cy))
        else:
            hits.append((idx, None, None))
        idx += 1
        if idx % 120 == 0:
            print(f"  ...{idx} frames", end="\r", flush=True)
    cap.release()

    r = analyse_detections(hits, idx, fps, width, height)
    print(" " * 40, end="\r")
    print(f"\n{Path(args.video).name}")
    print(f"{width}x{height}, {fps:.0f} fps, {r.duration_s:.1f} s sampled\n")

    worst = "ok"
    for level, headline, advice in verdicts(r):
        tag = {"ok": "  ok  ", "warn": " warn ", "fail": " FAIL "}[level]
        print(f"[{tag}] {headline}")
        print(f"         {advice}\n")
        if level == "fail":
            worst = "fail"
        elif level == "warn" and worst != "fail":
            worst = "warn"

    if worst == "fail":
        print("VERDICT: re-shoot before labelling. Fix the camera problems above first -")
        print("         labelling cannot recover information the footage never captured.")
        return 2
    if worst == "warn":
        print("VERDICT: usable. Go ahead and label, and expect fine-tuning to close")
        print("         the detection-rate gap.")
        return 0
    print("VERDICT: good footage. Label it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
