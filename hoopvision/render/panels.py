"""The dashboard panels.

Panel set, per the brief:
  tiles       field goals | shooting arm | dribbles | entry angle
  wide        shot trajectory - the actual parabola, with ghosts of prior shots
  right       arm angle trace through the release
  right       shot chart

Every panel takes plain data and returns an image, so each one can be
rendered and eyeballed on its own without running the pipeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..constants import OPTIMAL_ENTRY_ANGLE_DEG
from .theme import FONT, THEME, Theme, label, panel, small, text_width, value


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------

def tile(w: int, h: int, head: str, big: str, subs: Sequence[str] = (),
         big_color=None, unit: Optional[str] = None, t: Theme = THEME) -> np.ndarray:
    """Label, one big number, up to three supporting lines.

    `unit` is drawn small beside the number rather than baked into it -
    OpenCV's Hershey fonts are ASCII only, so there is no degree glyph to
    bake in even if we wanted one.
    """
    img = panel(w, h, t)
    label(img, head, 12, 22, t, 1.0)
    by = int(h * 0.58)
    value(img, big, 12, by, t, scale=1.7, color=big_color, thickness=2)
    if unit:
        small(img, unit, 12 + text_width(big, FONT, 1.7, 2) + 8, by, t, 1.0)
    y = by + 20
    for s in subs:
        small(img, s, 12, y, t, 0.85)
        y += 15
    return img


def fg_tile(w: int, h: int, makes: int, attempts: int, t: Theme = THEME) -> np.ndarray:
    pct = (100.0 * makes / attempts) if attempts else None
    subs = [f"fg {pct:.0f}%"] if pct is not None else ["fg --"]
    colour = t.good if pct is not None and pct >= 50 else t.ink
    return tile(w, h, "field goals", f"{makes} / {attempts}", subs, colour, None, t)


def arm_tile(w: int, h: int, at_release: Optional[float], set_point: Optional[float],
             session_sd: Optional[float], t: Theme = THEME) -> np.ndarray:
    big = f"{at_release:.0f}" if at_release is not None else "--"
    subs = [
        f"set point: {set_point:.0f} deg" if set_point is not None else "set point: --",
        f"session sd: {session_sd:.1f} deg" if session_sd is not None else "session sd: --",
    ]
    return tile(w, h, "elbow at release", big, subs, None,
                "deg" if at_release is not None else None, t)


def dribble_tile(w: int, h: int, count: Optional[int], mean_count: Optional[float],
                 to_release_s: Optional[float], t: Theme = THEME) -> np.ndarray:
    big = str(count) if count is not None else "--"
    subs = [
        f"session avg: {mean_count:.1f}" if mean_count is not None else "session avg: --",
        f"to release: {to_release_s:.2f} s" if to_release_s is not None else "to release: --",
    ]
    return tile(w, h, "dribbles before shot", big, subs, None, None, t)


def entry_tile(w: int, h: int, entry: Optional[float], mean_entry: Optional[float],
               in_band_pct: Optional[float], t: Theme = THEME) -> np.ndarray:
    lo, hi = OPTIMAL_ENTRY_ANGLE_DEG
    big = f"{entry:.0f}" if entry is not None else "--"
    colour = t.good if entry is not None and lo <= entry <= hi else t.ink
    subs = [
        f"optimal {lo:.0f}-{hi:.0f} deg",
        f"in band: {in_band_pct:.0f}%" if in_band_pct is not None else "in band: --",
        f"mean: {mean_entry:.0f} deg" if mean_entry is not None else "mean: --",
    ]
    return tile(w, h, "entry angle", big, subs, colour,
                "deg" if entry is not None else None, t)


# --------------------------------------------------------------------------
# shot trajectory
# --------------------------------------------------------------------------

def _fit(pts: np.ndarray, w: int, h: int, pad: int,
         bounds: Optional[Tuple[float, float, float, float]] = None):
    """Return a function mapping source px -> panel px, preserving aspect."""
    if bounds is None:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    else:
        x0, y0, x1, y1 = bounds
    sw, sh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    s = min((w - 2 * pad) / sw, (h - 2 * pad) / sh)
    ox = pad + ((w - 2 * pad) - sw * s) / 2
    oy = pad + ((h - 2 * pad) - sh * s) / 2

    def to_panel(x: float, y: float) -> Tuple[int, int]:
        return int(round(ox + (x - x0) * s)), int(round(oy + (y - y0) * s))

    return to_panel, s


def trajectory_panel(
    w: int, h: int,
    current: Sequence[Tuple[float, float]],
    ghosts: Sequence[Sequence[Tuple[float, float]]] = (),
    rim: Optional[Tuple[float, float, float]] = None,   # cx, cy, width_px
    floor_y: Optional[float] = None,
    made: Optional[bool] = None,
    t: Theme = THEME,
) -> np.ndarray:
    """The shot arc, in the camera's own image plane.

    Plotting in image pixels rather than reprojecting to a court side-view is
    deliberate: the arc you see here is exactly the arc the tracker saw, so a
    bad track looks bad instead of being smoothed into a plausible parabola.
    """
    img = panel(w, h, t)
    label(img, "shot trajectory", 12, 22, t, 1.0)

    allpts: List[Tuple[float, float]] = list(current)
    for g in ghosts:
        allpts.extend(g)
    if rim is not None:
        allpts.extend([(rim[0] - rim[2], rim[1]), (rim[0] + rim[2], rim[1])])
    if floor_y is not None and allpts:
        allpts.append((allpts[0][0], floor_y))
    if len(allpts) < 2:
        small(img, "waiting for a shot", 12, h // 2, t, 1.0)
        return img

    arr = np.asarray(allpts, dtype=float)
    to_panel, _ = _fit(arr, w, h - 30, pad=16)

    def shift(p):
        x, y = to_panel(*p)
        return x, y + 26

    if floor_y is not None:
        _, fy = shift((arr[0][0], floor_y))
        fy = min(fy, h - 6)
        cv2.line(img, (8, fy), (w - 8, fy), t.grid, 1, cv2.LINE_AA)
        small(img, "floor", 10, max(fy - 5, 34), t, 0.8, t.ink_soft)

    for g in ghosts:
        if len(g) < 2:
            continue
        pts = np.array([shift(p) for p in g], dtype=np.int32)
        cv2.polylines(img, [pts], False, t.ghost, 1, cv2.LINE_AA)

    if rim is not None:
        cx, cy, rw = rim
        lx, ly = shift((cx - rw / 2, cy))
        rx, ry = shift((cx + rw / 2, cy))
        cv2.line(img, (lx, ly), (rx, ry), t.ink, 3, cv2.LINE_AA)
        small(img, "rim", rx + 6, ry + 4, t, 0.8)

    if len(current) >= 2:
        colour = t.good if made else (t.bad if made is False else t.accent)
        pts = np.array([shift(p) for p in current], dtype=np.int32)
        cv2.polylines(img, [pts], False, colour, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[0]), 4, colour, -1, cv2.LINE_AA)
        apex_i = int(np.argmin([p[1] for p in current]))
        ax, ay = shift(current[apex_i])
        cv2.circle(img, (ax, ay), 3, colour, 1, cv2.LINE_AA)
        small(img, "apex", ax + 6, ay - 4, t, 0.8, colour)
    return img


# --------------------------------------------------------------------------
# arm angle
# --------------------------------------------------------------------------

def arm_angle_panel(
    w: int, h: int,
    frames: Sequence[int],
    angles: Sequence[float],
    release_frame: Optional[int] = None,
    t: Theme = THEME,
) -> np.ndarray:
    """Elbow angle through the run-up to release.

    Read it as a shape, not a number: a clean shot dips to a consistent set
    point and extends smoothly. A jagged trace is a rushed or contested one.
    """
    img = panel(w, h, t)
    label(img, "shooting arm angle", 12, 22, t, 1.0)
    if len(angles) < 2:
        small(img, "no pose", 12, h // 2, t, 1.0)
        return img

    lo, hi = 40.0, 180.0
    left, right, top, bot = 34, w - 12, 34, h - 20

    for deg in (60, 90, 120, 150, 180):
        y = int(bot - (deg - lo) / (hi - lo) * (bot - top))
        cv2.line(img, (left, y), (right, y), t.grid, 1)
        small(img, f"{deg}", 6, y + 4, t, 0.7, t.grid)

    n = len(angles)
    pts = []
    for i, a in enumerate(angles):
        x = int(left + i / max(n - 1, 1) * (right - left))
        y = int(bot - (min(max(a, lo), hi) - lo) / (hi - lo) * (bot - top))
        pts.append((x, y))
    cv2.polylines(img, [np.array(pts, np.int32)], False, t.accent, 2, cv2.LINE_AA)

    i_min = int(np.argmin(angles))
    cv2.circle(img, pts[i_min], 4, t.warn, -1, cv2.LINE_AA)
    small(img, f"set {angles[i_min]:.0f}", max(left, pts[i_min][0] - 18),
          min(pts[i_min][1] + 15, bot - 2), t, 0.8, t.warn)

    cv2.circle(img, pts[-1], 4, t.ink, -1, cv2.LINE_AA)
    rel_txt = f"rel {angles[-1]:.0f}"
    small(img, rel_txt, max(left, right - 52), max(top + 10, pts[-1][1] - 8), t, 0.8, t.ink)
    small(img, "run-up to release (deg)", left - 24, h - 5, t, 0.7, t.ink_soft)
    return img


# --------------------------------------------------------------------------
# shot chart
# --------------------------------------------------------------------------

def shot_chart_panel(
    w: int, h: int,
    shots: Sequence[Tuple[float, float, bool, int]],   # court x, y, made, attempt
    t: Theme = THEME,
) -> np.ndarray:
    """Makes and misses on the floor. Falls back to a message when no
    homography has been calibrated - which is the common case outdoors."""
    img = panel(w, h, t)
    label(img, "shot chart", 12, 22, t, 1.0)
    if not shots:
        small(img, "no court calibration", 12, h // 2, t, 0.9)
        small(img, "run scripts/calibrate_court.py", 12, h // 2 + 15, t, 0.8)
        return img

    arr = np.array([[s[0], s[1]] for s in shots], dtype=float)
    pad = 24
    x0, x1 = arr[:, 0].min() - 1, arr[:, 0].max() + 1
    y0, y1 = arr[:, 1].min() - 1, arr[:, 1].max() + 1
    to_panel, _ = _fit(arr, w, h - 30, pad, bounds=(x0, y0, x1, y1))

    for cx, cy, made, idx in shots:
        px, py = to_panel(cx, cy)
        py += 26
        colour = t.good if made else t.bad
        cv2.circle(img, (px, py), 7, colour, 1, cv2.LINE_AA)
        small(img, str(idx), px - 4, py + 4, t, 0.7, colour)
    return img
