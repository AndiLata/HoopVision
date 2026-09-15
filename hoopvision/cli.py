"""Command line entry point.

    # no model needed - runs on the synthetic clip
    python -m hoopvision.cli --replay sample/ --video sample/clip.mp4 \
        --out out/dashboard.mp4 --csv out/shots.csv

    # real footage
    python -m hoopvision.cli --video clips/session01.mov --weights runs/best.pt \
        --out out/session01.mp4 --csv out/session01.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detect import ReplayDetector, YoloDetector
from .geometry import CourtHomography
from .metrics import summarise
from .pipeline import Pipeline, PipelineConfig
from .pose import ReplayPose, YoloPose


def main() -> int:
    ap = argparse.ArgumentParser(prog="hoopvision")
    ap.add_argument("--video", help="input clip")
    ap.add_argument("--replay", help="directory with detections.json + poses.json")
    ap.add_argument("--weights", help="fine-tuned detector weights (best.pt)")
    ap.add_argument("--bootstrap", action="store_true",
                    help="no trained model: COCO 'sports ball' + a rim marked once. "
                         "Requires --rim. Indicative numbers only.")
    ap.add_argument("--rim", help="rim.json from mark_rim.py (bootstrap mode)")
    ap.add_argument("--pose-weights", default="yolo26n-pose.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--homography", help="court homography .txt from calibrate_court.py")
    ap.add_argument("--out", help="annotated dashboard mp4")
    ap.add_argument("--csv", help="per-shot csv")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    if not (args.replay or args.weights or args.bootstrap):
        ap.error("pick one: --replay (synthetic clip), --bootstrap (no model yet), "
                 "or --weights (fine-tuned model)")
    if args.bootstrap and not args.rim:
        ap.error("--bootstrap needs --rim: run scripts/mark_rim.py first")

    if args.replay:
        d = Path(args.replay)
        detector = ReplayDetector(d / "detections.json")
        pose = ReplayPose(d / "poses.json")
        meta = json.loads((d / "detections.json").read_text()).get("meta", {})
        fps = meta.get("fps", args.fps)
    elif args.bootstrap:
        from .bootstrap import CocoBallDetector, FixedRim

        detector = CocoBallDetector(FixedRim.load(args.rim), device=args.device)
        pose = YoloPose(args.pose_weights, device=args.device)
        fps = args.fps
        print("bootstrap mode: zero-shot COCO ball detection with a fixed rim.\n"
              "Good enough to see the pipeline work on your footage; not good "
              "enough to quote. Run check_footage.py to see how much to trust it.\n")
    else:
        detector = YoloDetector(args.weights, device=args.device)
        pose = YoloPose(args.pose_weights, device=args.device)
        fps = args.fps

    homog = CourtHomography.load(args.homography) if args.homography else None
    pipe = Pipeline(detector, pose, PipelineConfig(fps=fps, out_width=args.width), homog)

    if not args.video:
        ap.error("--video is required (the frames to read)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    result = pipe.run_video(args.video, args.out, args.csv)

    if result.camera_moved:
        print("WARNING: the rim drifted in frame - the camera moved. "
              "Scale and court mapping are unreliable for this clip.")

    print(f"{result.frames_processed} frames, {len(result.records)} attempts")
    for k, v in result.session.items():
        print(f"  {k:<22}{v if not isinstance(v, float) else round(v, 2)}")
    if args.csv:
        print(f"\nper-shot csv -> {args.csv}")
    if args.out:
        print(f"dashboard    -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
