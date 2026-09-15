"""Ballistic fitting and the rim-plane crossing test.

Image coordinates throughout: +y points DOWN. So an ascending ball has
negative vy, and gravity is a positive y acceleration.

One property used repeatedly: scale is isotropic, so any *angle* computed
from pixel velocities equals the angle in metres. Entry angle and release
angle therefore need no calibration at all. Only speeds and heights do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .constants import G


@dataclass(frozen=True)
class Sample:
    frame: int
    x: float          # px
    y: float          # px, +down
    radius: float     # px


@dataclass(frozen=True)
class Crossing:
    """Where and how the ball met the rim plane."""

    x_px: float
    frame: float          # fractional - the crossing is between samples
    vx_px_s: float
    vy_px_s: float

    @property
    def entry_angle_deg(self) -> float:
        """Angle below horizontal at the rim plane. Scale-free.

        A flat line drive is near 0 deg; a dropped-in rainbow approaches 90.
        The coaching literature's sweet spot is 43-47.
        """
        return float(np.degrees(np.arctan2(abs(self.vy_px_s), abs(self.vx_px_s))))


def linear_velocity(samples: Sequence[Sample], fps: float) -> Tuple[float, float]:
    """Least-squares velocity in px/s over a short window.

    Least squares over 3-7 samples rather than a two-point difference: the
    detector's box jitters by a pixel or two every frame, and differencing
    amplifies exactly that.
    """
    if len(samples) < 2:
        raise ValueError("need >= 2 samples for a velocity")
    t = np.array([s.frame for s in samples], dtype=float) / fps
    x = np.array([s.x for s in samples], dtype=float)
    y = np.array([s.y for s in samples], dtype=float)
    t = t - t.mean()
    denom = float(np.sum(t * t))
    if denom == 0:
        raise ValueError("degenerate time window")
    return float(np.sum(t * x) / denom), float(np.sum(t * y) / denom)


def fit_parabola_y(samples: Sequence[Sample], fps: float) -> Tuple[float, float, float]:
    """Fit y(t) = a t^2 + b t + c in pixels, t in seconds from the first sample."""
    if len(samples) < 3:
        raise ValueError("need >= 3 samples to fit a parabola")
    t0 = samples[0].frame / fps
    t = np.array([s.frame / fps - t0 for s in samples], dtype=float)
    y = np.array([s.y for s in samples], dtype=float)
    a, b, c = np.polyfit(t, y, 2)
    return float(a), float(b), float(c)


def scale_from_gravity(samples: Sequence[Sample], fps: float) -> Optional[float]:
    """A third ruler: metres per pixel, derived from gravity itself.

    A free ball accelerates at exactly 9.80665 m/s^2. Fit its pixel
    trajectory, and the quadratic coefficient is (g/2) / m_per_px. So the
    scene tells you its own scale with no reference object at all.

    Mostly useful as a *cross-check*: if this disagrees with the ball-diameter
    scale by more than ~20%, the track is contaminated (wrong object, missed
    frames, or the ball was still in contact with a hand).
    """
    try:
        a, _, _ = fit_parabola_y(samples, fps)
    except ValueError:
        return None
    if a <= 1e-6:
        return None  # not in free flight, or ascending-only noise
    return (G / 2) / a


def apex_index(samples: Sequence[Sample]) -> Optional[int]:
    """Index of the highest point on the track (smallest y, since y is down)."""
    if not samples:
        return None
    return int(np.argmin([s.y for s in samples]))


def crossing_at_y(samples: Sequence[Sample], y_target: float, fps: float) -> Optional[Crossing]:
    """Interpolate the descending crossing of a horizontal line.

    This is the whole point of the two-zone method: the ball does NOT have to
    be sampled at y_target, or anywhere near it. Take the last sample above
    the line and the first below it and interpolate between them.

    Linear interpolation is correct here even though the path is a parabola -
    across one sample gap near the rim the curvature contributes well under a
    millimetre.
    """
    above: Optional[Sample] = None
    for i in range(len(samples) - 1):
        s0, s1 = samples[i], samples[i + 1]
        if s0.y <= y_target < s1.y:       # crossing downward
            above = s0
            below = s1
            span = below.y - above.y
            if span <= 0:
                continue
            f = (y_target - above.y) / span
            x_px = above.x + f * (below.x - above.x)
            frame = above.frame + f * (below.frame - above.frame)
            window = samples[max(0, i - 2): i + 3]
            try:
                vx, vy = linear_velocity(window, fps)
            except ValueError:
                dt = (below.frame - above.frame) / fps
                vx = (below.x - above.x) / dt
                vy = span / dt
            return Crossing(x_px=x_px, frame=frame, vx_px_s=vx, vy_px_s=vy)
    return None


def is_monotonic_descent(samples: Sequence[Sample], start_idx: int, n: int = 6,
                         tol_px: float = 2.0) -> bool:
    """Did the ball keep falling for n samples after the crossing?

    This is what separates a soft make from a rattle that pops back up. A
    small tolerance absorbs detector jitter without admitting a real bounce.
    """
    seg = samples[start_idx: start_idx + n + 1]
    if len(seg) < 2:
        return False
    return all(seg[i + 1].y >= seg[i].y - tol_px for i in range(len(seg) - 1))


def refine_release_frame(samples: Sequence[Sample], candidate: int, fps: float,
                         settle: int = 8, fit_len: int = 32,
                         tol_radii: float = 0.5) -> int:
    """Find the true release by asking when the ball stopped being pushed.

    While the ball is in the hand the arm accelerates it; once it leaves,
    only gravity acts. So: fit a parabola to samples that are definitely in
    free flight, extrapolate it backwards, and the release is the last frame
    the observed position still agrees with that fit.

    This matters more than it sounds. A release frame even five frames late
    at 120 fps reads the ball after gravity has already taken ~0.4 m/s off
    it, which shows up as a release speed several percent low and a release
    angle a couple of degrees flat - on every single shot, in the same
    direction, so it never looks like noise.
    """
    by_frame = {s.frame: s for s in samples}
    flight = [s for s in samples if candidate + settle <= s.frame <= candidate + settle + fit_len]
    if len(flight) < 5:
        return candidate

    t0 = flight[0].frame / fps
    t = np.array([s.frame / fps - t0 for s in flight])
    px = np.polyfit(t, np.array([s.x for s in flight]), 1)
    py = np.polyfit(t, np.array([s.y for s in flight]), 2)

    tol = tol_radii * float(np.median([s.radius for s in flight]))
    release = candidate
    for f in range(candidate + settle, candidate - 40, -1):
        s = by_frame.get(f)
        if s is None:
            continue
        tt = f / fps - t0
        dx = s.x - float(np.polyval(px, tt))
        dy = s.y - float(np.polyval(py, tt))
        if float(np.hypot(dx, dy)) > tol:
            break
        release = f
    return release


def resample_track(samples: Sequence[Sample], max_gap: int = 4) -> List[Sample]:
    """Fill short detection gaps by linear interpolation; refuse long ones.

    Four frames at 120 fps is 33 ms - a ball cannot do anything surprising in
    that time, so interpolating is safe. Beyond that it can hit the rim, a
    hand, or another ball, and inventing a straight line through that is how
    you get a confident wrong answer.
    """
    if not samples:
        return []
    out: List[Sample] = [samples[0]]
    for prev, cur in zip(samples, samples[1:]):
        gap = cur.frame - prev.frame
        if 1 < gap <= max_gap + 1:
            for k in range(1, gap):
                f = k / gap
                out.append(Sample(
                    frame=prev.frame + k,
                    x=prev.x + f * (cur.x - prev.x),
                    y=prev.y + f * (cur.y - prev.y),
                    radius=prev.radius + f * (cur.radius - prev.radius),
                ))
        out.append(cur)
    return out
