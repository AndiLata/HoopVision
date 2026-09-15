#!/usr/bin/env python3
"""Click four court points to build the image -> court homography.

    python scripts/calibrate_court.py --video clips/session01.mov --out court.txt

Four clicks, once per camera setup. Deliberately manual: outdoor court lines
are faded, repainted off-spec or simply missing, and an auto-detector fails
silently where a human clicking four points does not.

Default correspondence points are the free-throw lane, in metres, with the
origin directly under the centre of the rim and +y running out toward
half court. Override with --court if you use different landmarks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.geometry import CourtHomography

# FIBA/NBA lane is 4.88 m wide; the free-throw line is 5.79 m from the
# backboard, which puts it 4.60 m out from the rim centre.
DEFAULT_COURT = [
    (-2.44, 0.00),   # near lane corner at the baseline
    (2.44, 0.00),    # far lane corner at the baseline
    (2.44, 4.60),    # far end of the free-throw line
    (-2.44, 4.60),   # near end of the free-throw line
]
LABELS = ["baseline / lane corner (left)", "baseline / lane corner (right)",
          "free-throw line (right)", "free-throw line (left)"]


def grab_frame(video: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index}")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="court.txt")
    ap.add_argument("--court", help="8 comma-separated metres: x1,y1,...,x4,y4")
    args = ap.parse_args()

    court = DEFAULT_COURT
    if args.court:
        v = [float(x) for x in args.court.split(",")]
        if len(v) != 8:
            ap.error("--court needs 8 numbers")
        court = list(zip(v[0::2], v[1::2]))

    frame = grab_frame(args.video, args.frame)
    picked: list[tuple[float, float]] = []
    win = "click 4 court points  (u = undo, enter = done)"

    def draw():
        img = frame.copy()
        for i, (x, y) in enumerate(picked):
            cv2.circle(img, (int(x), int(y)), 6, (39, 83, 188), -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (int(x) + 10, int(y) - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (39, 83, 188), 2, cv2.LINE_AA)
        if len(picked) < 4:
            cv2.putText(img, f"{len(picked) + 1}. {LABELS[len(picked)]}", (16, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append((float(x), float(y)))
            draw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(picked) == 4:
            break
        if k == ord("u") and picked:
            picked.pop()
            draw()
        if k == 27:
            cv2.destroyAllWindows()
            print("cancelled")
            return 1
    cv2.destroyAllWindows()

    h = CourtHomography.from_points(picked, court)
    h.save(args.out)

    # Sanity check: round-trip the clicked points and report the error in cm.
    errs = [np.hypot(*(np.array(h.to_court(*p)) - np.array(c)))
            for p, c in zip(picked, court)]
    print(f"homography -> {args.out}")
    print(f"round-trip error: max {max(errs) * 100:.1f} cm, mean {np.mean(errs) * 100:.1f} cm")
    if max(errs) > 0.15:
        print("WARNING: that is high. Re-click - a point is probably off, "
              "or the four points are close to collinear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
