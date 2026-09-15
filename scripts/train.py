#!/usr/bin/env python3
"""Fine-tune the ball + rim detector on Apple Silicon.

    python scripts/train.py --data datasets/hoopvision/data.yaml

Defaults are tuned for this specific problem, not for COCO. The two that
matter most are imgsz and the success bar printed at the end.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml exported from Roboflow")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="NOT 640. A ball 20 m out is ~8 px across at 640.")
    ap.add_argument("--batch", type=int, default=8, help="drop to 4 on memory pressure")
    ap.add_argument("--device", default="mps", help="'mps' on Apple Silicon, 0 on a Colab T4")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="hoopvision")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=args.name,
        cache=False,              # 960 px frames will not fit in RAM
        # The camera is fixed and level, so heavy rotation teaches a
        # variation that never occurs and costs capacity needed for
        # small objects. Blur, however, is real in this data.
        degrees=5.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("""
Before you trust this model, check three things:

  rim  mAP50                 > 0.95   (static and easy - a miss means label noise)
  ball mAP50                 > 0.85
  ball recall, IN-FLIGHT     > 0.90   <- not in the metrics table

That last one is the one that matters and the only way to check it is to
watch a held-out clip frame by frame. A model with excellent mAP that drops
the ball for three frames at the apex will ruin every trajectory fit, and
mAP will never tell you, because those three frames are a rounding error in
a dataset where most frames have an easy, slow-moving ball.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
