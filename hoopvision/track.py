"""Ball tracking by physics, not by association.

ByteTrack is right for people - many objects, unpredictable motion, identity
matters. For the ball it is the wrong tool: there is one object and it obeys
a law you already know. Gating on a ballistic prediction rejects the head,
the distant second ball, and the shoe that briefly looks orange, using far
less machinery.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from .detect import Detection
from .trajectory import Sample


class BallTracker:
    def __init__(
        self,
        fps: float,
        gate_px: float = 90.0,
        radius_tol: float = 0.30,
        max_coast_frames: int = 6,
    ):
        self.fps = fps
        self.gate_px = gate_px
        self.radius_tol = radius_tol
        self.max_coast_frames = max_coast_frames

        self.samples: List[Sample] = []
        self._radii: deque[float] = deque(maxlen=15)
        self._missed = 0
        self.rejected = 0

    # -- prediction ------------------------------------------------------
    def _predict(self) -> Optional[tuple]:
        """Constant-acceleration extrapolation from the last three samples."""
        n = len(self.samples)
        if n < 2:
            return None
        if n == 2:
            a, b = self.samples[-2], self.samples[-1]
            dt = max(b.frame - a.frame, 1)
            return b.x + (b.x - a.x) / dt, b.y + (b.y - a.y) / dt
        a, b, c = self.samples[-3], self.samples[-2], self.samples[-1]
        t = np.array([a.frame, b.frame, c.frame], dtype=float)
        px = np.polyfit(t, np.array([a.x, b.x, c.x]), 2)
        py = np.polyfit(t, np.array([a.y, b.y, c.y]), 2)
        nxt = c.frame + 1
        return float(np.polyval(px, nxt)), float(np.polyval(py, nxt))

    def _radius_ok(self, r: float) -> bool:
        if not self._radii:
            return True
        med = float(np.median(self._radii))
        if med <= 0:
            return True
        return abs(r - med) / med <= self.radius_tol

    # -- main ------------------------------------------------------------
    def update(self, frame_idx: int, ball: Optional[Detection]) -> Optional[Sample]:
        if ball is None:
            self._missed += 1
            if self._missed > self.max_coast_frames:
                self.reset_soft()
            return None

        r = ball.radius_px
        if not self._radius_ok(r):
            self.rejected += 1
            return None

        pred = self._predict()
        if pred is not None:
            dist = float(np.hypot(ball.cx - pred[0], ball.cy - pred[1]))
            # The gate widens while coasting: a longer gap means more room for
            # the ball to have legitimately travelled.
            if dist > self.gate_px * (1 + self._missed):
                self.rejected += 1
                self.reset_soft()
                self._missed = 0
                s = Sample(frame_idx, ball.cx, ball.cy, r)
                self.samples.append(s)
                self._radii.append(r)
                return s

        self._missed = 0
        s = Sample(frame_idx, ball.cx, ball.cy, r)
        self.samples.append(s)
        self._radii.append(r)
        return s

    def reset_soft(self) -> None:
        """Break the track without discarding history the metrics still need."""
        self.samples = self.samples[-1:] if self.samples else []

    def recent(self, n: int) -> List[Sample]:
        return self.samples[-n:]
