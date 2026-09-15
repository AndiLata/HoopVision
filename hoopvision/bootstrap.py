"""Run on real footage before you have a trained detector.

Two observations collapse most of the "you need a model first" barrier:

1. **The rim never moves.** The camera is on a tripod, so the rim is a
   constant in the scene. It needs one click, not a detector, and certainly
   not 600 labelled frames. Half the labelling budget in Stage 1 exists only
   because the naive pipeline asks a model to re-find a stationary object
   sixty times a second.

2. **COCO already has a `sports ball` class.** It was not trained on a
   basketball at 15 px against a bright sky and it will drop frames that a
   fine-tuned model would catch. But it is free, it is available right now,
   and it is enough to answer the only question that matters before you
   commit to labelling: *does my camera setup produce trackable footage?*

Bootstrap mode is the Stage 0 answer to "is this clip usable", not a
substitute for Stage 2. Treat its numbers as indicative and never as final -
`check_footage.py` will tell you how much to trust them for a given clip.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detect import Detection, FrameDetections

# COCO-80 index for `sports ball`. Matching on the name rather than the
# index, since class order is a model-card detail and not a promise.
COCO_BALL_NAMES = {"sports ball"}


# --------------------------------------------------------------------------
# the rim: one click, not a model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedRim:
    """A rim marked once for a camera setup and reused for every frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_detection(self) -> Detection:
        return Detection("rim", self.x1, self.y1, self.x2, self.y2, conf=1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"rim": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}}, indent=2))

    @staticmethod
    def load(path: str | Path) -> "FixedRim":
        d = json.loads(Path(path).read_text())["rim"]
        return FixedRim(d["x1"], d["y1"], d["x2"], d["y2"])

    @staticmethod
    def from_two_clicks(left: Tuple[float, float], right: Tuple[float, float],
                        thickness_px: float = 6.0) -> "FixedRim":
        """Built from the ring's left and right inner edges.

        Two clicks rather than a drag: the ring's inner edges are the two
        points you can actually see and place precisely, and they are exactly
        what the scale calculation needs (the 0.4572 m inner diameter).
        """
        y = (left[1] + right[1]) / 2
        x1, x2 = sorted((left[0], right[0]))
        return FixedRim(x1, y - thickness_px / 2, x2, y + thickness_px / 2)


class CocoBallDetector:
    """Zero-shot ball detection + a fixed rim.

    Confidence is deliberately low (0.15) and imgsz deliberately high: COCO's
    sports-ball class is weak on a small, fast, motion-blurred basketball, so
    the useful trade is to accept marginal detections and let the tracker's
    gravity gate throw out the false ones. That gate is why a low threshold
    is safe here and would not be if the tracker were a generic one.
    """

    def __init__(self, rim: FixedRim, weights: str = "yolo26n.pt",
                 device: str = "mps", conf: float = 0.15, imgsz: int = 1280):
        from ultralytics import YOLO  # noqa: PLC0415 - lazy on purpose

        self.model = YOLO(weights)
        self.rim = rim
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.names = self.model.names

    def detect(self, frame_idx: int, image) -> FrameDetections:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        best: Optional[Detection] = None
        for box in res.boxes:
            name = self.names[int(box.cls)]
            if name not in COCO_BALL_NAMES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            d = Detection("ball", x1, y1, x2, y2, float(box.conf))
            if best is None or d.conf > best.conf:
                best = d
        return FrameDetections(frame_idx, ball=best, rim=self.rim.as_detection())


# --------------------------------------------------------------------------
# is this clip usable?
# --------------------------------------------------------------------------

@dataclass
class FootageReport:
    frames: int
    fps: float
    width: int
    height: int
    ball_frames: int
    median_ball_px: Optional[float]
    longest_gap_frames: int
    upper_half_rate: Optional[float]

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def detection_rate(self) -> float:
        return self.ball_frames / self.frames if self.frames else 0.0

    @property
    def longest_gap_ms(self) -> float:
        return 1000.0 * self.longest_gap_frames / self.fps if self.fps else 0.0


def analyse_detections(hits: Sequence[Tuple[int, Optional[float], Optional[float]]],
                       frames: int, fps: float, width: int, height: int) -> FootageReport:
    """Summarise per-frame ball detections.

    `hits` is (frame_idx, diameter_px or None, cy or None) - kept as plain
    tuples so this is testable without a model or a video.
    """
    seen = [(f, d, cy) for f, d, cy in hits if d is not None]
    diams = [d for _, d, _ in seen]

    longest = cur = 0
    present = {f for f, _, _ in seen}
    for f in range(frames):
        cur = 0 if f in present else cur + 1
        longest = max(longest, cur)

    upper = [1 for _, _, cy in seen if cy is not None and cy < height / 2]
    upper_total = sum(1 for _, _, cy in seen if cy is not None)

    return FootageReport(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        ball_frames=len(seen),
        median_ball_px=float(np.median(diams)) if diams else None,
        longest_gap_frames=longest,
        upper_half_rate=(len(upper) / upper_total) if upper_total else None,
    )


def verdicts(r: FootageReport) -> List[Tuple[str, str, str]]:
    """(level, headline, what to do). Levels: ok | warn | fail."""
    out: List[Tuple[str, str, str]] = []

    if r.fps >= 110:
        out.append(("ok", f"{r.fps:.0f} fps", "Enough to interpolate the rim crossing tightly."))
    elif r.fps >= 55:
        out.append(("warn", f"{r.fps:.0f} fps",
                    "Workable, but one sample gap spans ~31% of the ball's clearance "
                    "through the rim. Expect some make/miss calls to be coin flips."))
    else:
        out.append(("fail", f"{r.fps:.0f} fps",
                    "Too slow. One sample gap spans ~61% of the ball's clearance, so the "
                    "make/miss call is dominated by interpolation error. Re-shoot in "
                    "Slo-mo mode at 120 fps."))

    if r.median_ball_px is None:
        out.append(("fail", "No ball detected at all",
                    "Check the rim is in frame and the ball is not backlit. If the clip "
                    "looks fine to you, the ball is probably too small - move closer or "
                    "raise --imgsz."))
    elif r.median_ball_px >= 20:
        out.append(("ok", f"ball ~{r.median_ball_px:.0f} px across",
                    "Comfortably above the size where detection degrades."))
    elif r.median_ball_px >= 12:
        out.append(("warn", f"ball ~{r.median_ball_px:.0f} px across",
                    "Small. A fine-tuned model will handle it; COCO will struggle. "
                    "Moving 1-2 m closer is cheaper than more labelling."))
    else:
        out.append(("fail", f"ball ~{r.median_ball_px:.0f} px across",
                    "Below the useful floor. Move the camera closer, or shoot at a higher "
                    "resolution and raise --imgsz."))

    rate = r.detection_rate
    if rate >= 0.85:
        out.append(("ok", f"ball found in {rate * 100:.0f}% of frames", "Solid."))
    elif rate >= 0.5:
        out.append(("warn", f"ball found in {rate * 100:.0f}% of frames",
                    "Expected for zero-shot COCO. Fine-tuning is what closes this gap - "
                    "this number is the argument for Stage 1, not a reason to abandon the clip."))
    else:
        out.append(("fail", f"ball found in only {rate * 100:.0f}% of frames",
                    "Too sparse to track. Usually lighting: shoot with the sun behind "
                    "the camera, never behind the rim."))

    if r.longest_gap_frames <= 4:
        out.append(("ok", f"longest gap {r.longest_gap_frames} frames",
                    "Within the range the tracker interpolates."))
    else:
        out.append(("warn", f"longest gap {r.longest_gap_frames} frames "
                            f"({r.longest_gap_ms:.0f} ms)",
                    "Longer than the tracker will interpolate, so any shot spanning that "
                    "gap is dropped rather than guessed at. Check whether it happens at "
                    "the apex - that is the sky-background case."))

    if r.upper_half_rate is not None and r.upper_half_rate < 0.15:
        out.append(("warn", "almost no detections in the upper half of frame",
                    "The ball is being lost in flight, which is exactly where the metrics "
                    "come from. Classic backlit-sky signature."))
    return out
