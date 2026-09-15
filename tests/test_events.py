"""Make/miss boundary behaviour and dribble counting."""
import math

import numpy as np
import pytest

from hoopvision.constants import (
    BALL_DIAMETER_M,
    FITS_THROUGH_HALF_WIDTH_M,
    G,
    RIM_INNER_DIAMETER_M,
)
from hoopvision.events import Outcome, ShotDetector
from hoopvision.form import count_dribbles
from hoopvision.geometry import RimScale
from hoopvision.pose import Pose
from hoopvision.trajectory import Sample

FPS = 120.0
S = 110.0                      # px per metre
FLOOR = 600.0
RIM_X_M, RIM_Y_M = 4.18, 3.048
RIM_W_PX = RIM_INNER_DIAMETER_M * S


def rim_scale():
    cx = 300 + RIM_X_M * S
    cy = FLOOR - RIM_Y_M * S
    return RimScale.from_rim_box(cx - RIM_W_PX / 2, cy - 4, cx + RIM_W_PX / 2, cy + 4)


def shot_track(lateral_error_m: float, theta=56.0, y0=2.20, n_extra=40):
    """Ball released at x=0 aimed to cross the rim plane offset by the error."""
    target = RIM_X_M + lateral_error_m
    dy = RIM_Y_M - y0
    th = math.radians(theta)
    v0 = math.sqrt(G * target ** 2 / (2 * math.cos(th) ** 2 * (math.tan(th) * target - dy)))
    vx, vy0 = v0 * math.cos(th), v0 * math.sin(th)
    n = int((target / vx) * FPS) + n_extra
    out = []
    for f in range(n):
        t = f / FPS
        x, y = vx * t, y0 + vy0 * t - 0.5 * G * t * t
        out.append(Sample(f, 300 + x * S, FLOOR - y * S, BALL_DIAMETER_M / 2 * S))
    return out


def run(track, wrist_frames=6):
    """Drive the state machine: a few in-hand frames, then the flight."""
    rs = rim_scale()
    sd = ShotDetector(FPS, rs)
    first = track[0]
    hip_y = FLOOR - 0.95 * S
    for i in range(wrist_frames):
        s = Sample(-wrist_frames + i, first.x, first.y, first.radius)
        sd.update(s.frame, s, wrists=[(s.x, s.y + 2)], hip_y=hip_y)
    event = None
    for s in track:
        e = sd.update(s.frame, s, wrists=[(first.x, first.y + 2)], hip_y=hip_y)
        if e is not None:
            event = e
            break
    return event


def test_dead_centre_is_a_make():
    e = run(shot_track(0.0))
    assert e and e.outcome is Outcome.MAKE


def test_just_inside_the_clearance_is_a_make():
    e = run(shot_track(FITS_THROUGH_HALF_WIDTH_M - 0.01))
    assert e and e.outcome is Outcome.MAKE


def test_just_outside_the_clearance_is_a_miss():
    e = run(shot_track(FITS_THROUGH_HALF_WIDTH_M + 0.01))
    assert e and e.outcome is Outcome.MISS


def test_miss_offset_sign_gives_the_direction():
    right = run(shot_track(0.22))
    left = run(shot_track(-0.22))
    assert right.miss_offset_m > 0 and left.miss_offset_m < 0


def test_make_miss_call_survives_a_30fps_sample_rate():
    """Decimating to 30 fps must not flip the call."""
    for err in (0.0, 0.05, -0.19, 0.22):
        dense = shot_track(err)
        sparse = [Sample(i, s.x, s.y, s.radius)
                  for i, s in enumerate(dense[::4])]
        assert run(dense).outcome is run(sparse, wrist_frames=4).outcome


# ---------------------------------------------------------------- dribbles

def bounce_track(n_bounces: int, h=0.35, start_frame=0):
    half = math.sqrt(2 * h / G)
    per = int(round(2 * half * FPS))
    out, f = [], start_frame
    for _ in range(n_bounces):
        for k in range(per):
            t = k / FPS
            dt = t if t <= half else (2 * half - t)
            y = max(h - 0.5 * G * dt * dt, 0.0) + BALL_DIAMETER_M / 2
            out.append(Sample(f, 300 + 0.35 * S, FLOOR - y * S, BALL_DIAMETER_M / 2 * S))
            f += 1
    return out


def test_counts_bounces():
    for n in (0, 1, 3, 5):
        track = bounce_track(n)
        got, _ = count_dribbles(track, {}, 0, 10 ** 6, fps=FPS, floor_y=FLOOR - 0.06 * S)
        assert got == n, f"expected {n}, got {got}"


def test_a_midair_pickup_is_not_a_dribble():
    """The ball descends, stops at waist height and rises again - a gather,
    not a bounce. Only the floor gate tells them apart."""
    track = []
    for f in range(60):
        y = 1.8 - 1.3 * (f / 59)
        track.append(Sample(f, 300 + 0.35 * S, FLOOR - y * S, BALL_DIAMETER_M / 2 * S))
    for f in range(60, 120):
        y = 0.5 + 0.9 * ((f - 60) / 59)
        track.append(Sample(f, 300 + 0.35 * S, FLOOR - y * S, BALL_DIAMETER_M / 2 * S))
    got, _ = count_dribbles(track, {}, 0, 10 ** 6, fps=FPS, floor_y=FLOOR - 0.06 * S)
    assert got == 0


def test_dribble_count_is_frame_rate_independent():
    dense = bounce_track(3)
    sparse = [Sample(i, s.x, s.y, s.radius) for i, s in enumerate(dense[::4])]
    a, _ = count_dribbles(dense, {}, 0, 10 ** 6, fps=FPS, floor_y=FLOOR - 0.06 * S)
    b, _ = count_dribbles(sparse, {}, 0, 10 ** 6, fps=FPS / 4, floor_y=FLOOR - 0.06 * S)
    assert a == b == 3
