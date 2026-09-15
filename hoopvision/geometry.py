"""Pixels to metres, and image to court.

Two rulers, deliberately kept separate:

  RimScale    fixed, exact, correct only at the rim's depth plane.
  BallScale   per-frame, depth-corrected, noisier.

Use RimScale for anything measured at the rim (entry angle, crossing
position) and BallScale for anything measured out where the shooter is
(release speed, release height). Mixing them up is the single most common
way single-camera speed numbers come out 20-40% high.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .constants import BALL_DIAMETER_M, RIM_INNER_DIAMETER_M


@dataclass(frozen=True)
class RimScale:
    """Metres per pixel at the rim plane, from the regulation ring width."""

    m_per_px: float
    rim_cx: float
    rim_cy: float
    rim_width_px: float

    @staticmethod
    def from_rim_box(x1: float, y1: float, x2: float, y2: float) -> "RimScale":
        w = float(x2 - x1)
        if w <= 0:
            raise ValueError("rim box has non-positive width")
        return RimScale(
            m_per_px=RIM_INNER_DIAMETER_M / w,
            rim_cx=(x1 + x2) / 2,
            rim_cy=(y1 + y2) / 2,
            rim_width_px=w,
        )

    def px_to_m(self, px: float) -> float:
        return px * self.m_per_px

    def m_to_px(self, m: float) -> float:
        return m / self.m_per_px


class BallScale:
    """Depth-corrected scale from the ball's own apparent diameter.

    A size-7 ball is 0.2395 m across no matter where it is in the frame, so
    its pixel diameter is a direct readout of depth. Detected radius is
    noisy (motion blur inflates the box, occlusion shrinks it), hence the
    median over a short window.
    """

    def __init__(self, window: int = 5, ball_diameter_m: float = BALL_DIAMETER_M):
        self.window = window
        self.ball_diameter_m = ball_diameter_m
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, radius_px: Optional[float]) -> None:
        if radius_px and radius_px > 0:
            self._buf.append(float(radius_px))

    @property
    def ready(self) -> bool:
        return len(self._buf) > 0

    @property
    def m_per_px(self) -> Optional[float]:
        if not self._buf:
            return None
        median_diameter_px = float(np.median(self._buf)) * 2
        return self.ball_diameter_m / median_diameter_px

    def px_to_m(self, px: float) -> Optional[float]:
        s = self.m_per_px
        return None if s is None else px * s


def rim_stable(boxes: Sequence[Tuple[float, float, float, float]], tol_px: float = 3.0) -> bool:
    """True if the rim box never drifts more than tol_px.

    A moving rim means a moving camera, which invalidates the homography and
    both rulers. Catching it here and aborting beats shipping wrong numbers.
    """
    if len(boxes) < 2:
        return True
    arr = np.asarray(boxes, dtype=float)
    centres = np.column_stack([(arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2])
    return bool(np.max(np.abs(centres - centres[0])) <= tol_px)


class CourtHomography:
    """Image plane -> court plane, from four clicked correspondences.

    Four clicks beat line auto-detection on an outdoor court every time:
    the lines are faded, repainted off-spec, or simply absent, and a failed
    auto-detector fails silently while a human clicking four points does not.
    """

    def __init__(self, H: np.ndarray):
        self.H = np.asarray(H, dtype=float).reshape(3, 3)

    @staticmethod
    def from_points(image_pts: Sequence[Tuple[float, float]],
                    court_pts: Sequence[Tuple[float, float]]) -> "CourtHomography":
        if len(image_pts) != 4 or len(court_pts) != 4:
            raise ValueError("need exactly 4 point correspondences")
        import cv2  # noqa: PLC0415

        H, _ = cv2.findHomography(
            np.asarray(image_pts, dtype=np.float32),
            np.asarray(court_pts, dtype=np.float32),
        )
        if H is None:
            raise ValueError("homography solve failed - are the 4 points collinear?")
        return CourtHomography(H)

    def to_court(self, x: float, y: float) -> Tuple[float, float]:
        v = self.H @ np.array([x, y, 1.0])
        if abs(v[2]) < 1e-9:
            raise ValueError("point maps to the horizon")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def save(self, path) -> None:
        np.savetxt(path, self.H)

    @staticmethod
    def load(path) -> "CourtHomography":
        return CourtHomography(np.loadtxt(path))
