"""Composes the frame: video pane on top, tile row, panel row.

Same structure as the reel that started this, with the panel set the shot
data actually supports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pose import SKELETON, Pose
from .panels import (
    arm_angle_panel,
    arm_tile,
    dribble_tile,
    entry_tile,
    fg_tile,
    shot_chart_panel,
    trajectory_panel,
)
from .theme import THEME, Theme, small


def annotate_video(
    frame: np.ndarray,
    pose: Optional[Pose] = None,
    ball_trail: Sequence[Tuple[float, float]] = (),
    rim_box: Optional[Tuple[float, float, float, float]] = None,
    badge: Optional[str] = None,
    t: Theme = THEME,
) -> np.ndarray:
    img = frame.copy()

    if rim_box is not None:
        x1, y1, x2, y2 = (int(v) for v in rim_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), t.accent, 2)

    # Fading trail - most recent is brightest.
    n = len(ball_trail)
    for i, (x, y) in enumerate(ball_trail):
        a = (i + 1) / max(n, 1)
        colour = tuple(int(c * a + 255 * (1 - a)) for c in t.accent)
        cv2.circle(img, (int(x), int(y)), max(2, int(3 * a) + 1), colour, -1, cv2.LINE_AA)

    if pose is not None:
        for i, j in SKELETON:
            a, b = pose.pts[i], pose.pts[j]
            if a[2] >= 0.3 and b[2] >= 0.3:
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (255, 255, 255), 2, cv2.LINE_AA)
        for x, y, c in pose.pts:
            if c >= 0.3:
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    if badge:
        small(img, badge, 12, 22, t, 1.2, (255, 255, 255))
    return img


def compose(
    video: np.ndarray,
    *,
    makes: int,
    attempts: int,
    arm_at_release: Optional[float],
    arm_set_point: Optional[float],
    arm_sd: Optional[float],
    dribbles: Optional[int],
    dribbles_mean: Optional[float],
    to_release_s: Optional[float],
    entry: Optional[float],
    entry_mean: Optional[float],
    entry_in_band: Optional[float],
    current_arc: Sequence[Tuple[float, float]],
    ghost_arcs: Sequence[Sequence[Tuple[float, float]]],
    rim: Optional[Tuple[float, float, float]],
    floor_y: Optional[float],
    made: Optional[bool],
    arm_frames: Sequence[int],
    arm_angles: Sequence[float],
    chart: Sequence[Tuple[float, float, bool, int]],
    width: int = 1280,
    t: Theme = THEME,
) -> np.ndarray:
    vh = int(video.shape[0] * width / video.shape[1])
    vid = cv2.resize(video, (width, vh))

    tile_h = 96
    panel_h = 210
    tw = width // 4
    tiles = [
        fg_tile(tw, tile_h, makes, attempts, t),
        arm_tile(tw, tile_h, arm_at_release, arm_set_point, arm_sd, t),
        dribble_tile(tw, tile_h, dribbles, dribbles_mean, to_release_s, t),
        entry_tile(width - 3 * tw, tile_h, entry, entry_mean, entry_in_band, t),
    ]
    tile_row = np.hstack(tiles)

    pw1 = int(width * 0.5)
    pw2 = int(width * 0.26)
    pw3 = width - pw1 - pw2
    panels = [
        trajectory_panel(pw1, panel_h, current_arc, ghost_arcs, rim, floor_y, made, t),
        arm_angle_panel(pw2, panel_h, arm_frames, arm_angles, None, t),
        shot_chart_panel(pw3, panel_h, chart, t),
    ]
    panel_row = np.hstack(panels)

    out = np.vstack([vid, tile_row, panel_row])
    return out
