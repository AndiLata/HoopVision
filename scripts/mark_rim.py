#!/usr/bin/env python3
"""Click the rim's two inner edges. That's the whole calibration.

    python scripts/mark_rim.py --video clips/session01.mov --out rim.json

The camera is on a tripod, so the rim is a constant. Marking it once per
setup removes it from the detection problem entirely - which means the only
class you ever have to label is `ball`.

Click the LEFT inner edge of the ring, then the RIGHT inner edge. Inner, not
outer: the 0.4572 m regulation figure is the inner diameter, and that is what
the scale calculation uses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopvision.bootstrap import FixedRim
from hoopvision.constants import BALL_DIAMETER_M, RIM_INNER_DIAMETER_M

PROMPTS = ["click the LEFT inner edge of the ring",
           "click the RIGHT inner edge of the ring"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="rim.json")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"cannot read frame {args.frame}")
        return 1

    picked: list[tuple[float, float]] = []
    win = "mark the rim  (u = undo, enter = done, esc = cancel)"

    def draw():
        img = frame.copy()
        for x, y in picked:
            cv2.drawMarker(img, (int(x), int(y)), (39, 83, 188),
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        if len(picked) == 2:
            cv2.line(img, (int(picked[0][0]), int(picked[0][1])),
                     (int(picked[1][0]), int(picked[1][1])), (39, 83, 188), 2, cv2.LINE_AA)
        msg = PROMPTS[len(picked)] if len(picked) < 2 else "enter to save"
        cv2.putText(img, msg, (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, msg, (16, 34), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 2:
            picked.append((float(x), float(y)))
            draw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(picked) == 2:
            break
        if k == ord("u") and picked:
            picked.pop()
            draw()
        if k == 27:
            cv2.destroyAllWindows()
            print("cancelled")
            return 1
    cv2.destroyAllWindows()

    rim = FixedRim.from_two_clicks(picked[0], picked[1])
    rim.save(args.out)

    width_px = rim.x2 - rim.x1
    m_per_px = RIM_INNER_DIAMETER_M / width_px
    print(f"rim -> {args.out}   ({width_px:.0f} px wide)")
    print(f"scale at the rim plane: {m_per_px * 100:.2f} cm per pixel")
    print(f"a basketball at that depth should be about "
          f"{BALL_DIAMETER_M / m_per_px:.0f} px across")
    if width_px < 30:
        print("\nWARNING: that rim is very small in frame. Everything downstream "
              "inherits this measurement, so a couple of pixels of click error "
              "becomes several percent of every distance. Move closer if you can.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
