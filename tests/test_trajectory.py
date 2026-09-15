"""The parts where a quiet bug produces plausible-looking wrong numbers."""
import math

import numpy as np
import pytest

from hoopvision.constants import BALL_DIAMETER_M, G, RIM_INNER_DIAMETER_M
from hoopvision.geometry import BallScale, RimScale
from hoopvision.trajectory import (
    Sample,
    crossing_at_y,
    is_monotonic_descent,
    refine_release_frame,
    resample_track,
    scale_from_gravity,
)

FPS = 120.0


def ballistic(v0, theta_deg, fps, n, scale_px_per_m=110.0, x0=0.0, y0=2.2, floor_px=600.0):
    """Ground-truth ballistic track in image pixels."""
    th = math.radians(theta_deg)
    out = []
    for f in range(n):
        t = f / fps
        x = x0 + v0 * math.cos(th) * t
        y = y0 + v0 * math.sin(th) * t - 0.5 * G * t * t
        out.append(Sample(f, 300 + x * scale_px_per_m, floor_px - y * scale_px_per_m, 13.2))
    return out


def test_crossing_found_when_ball_is_never_sampled_at_the_rim():
    """The whole point of the two-zone method.

    Sample a real trajectory at 30 fps - so sparsely that no sample lands
    anywhere near the rim plane - and the crossing must still be recovered.
    """
    rim_y = 600 - 3.048 * 110
    dense = ballistic(7.16, 56.0, 120.0, 200)
    truth = crossing_at_y(dense, rim_y, 120.0)
    assert truth is not None
    m_per_px = 1.0 / 110.0
    ball_radius_px = BALL_DIAMETER_M / 2 * 110

    for phase in range(4):                                  # every 30 fps alignment
        sparse = [Sample(i, s.x, s.y, s.radius)
                  for i, s in enumerate(dense[phase::4])]
        got = crossing_at_y(sparse, rim_y, 30.0)
        assert got is not None, f"no crossing recovered at phase {phase}"
        # right to within a centimetre regardless of where the samples land
        assert abs(got.x_px - truth.x_px) * m_per_px < 0.01


def test_crossing_survives_a_dropped_detection_at_the_rim():
    """Delete every sample that overlaps the rim plane and try again.

    This is the property that matters: a rule phrased as "was the ball ever
    inside the hoop?" has nothing to work with here, while interpolating
    between the samples that straddle the plane is unaffected.
    """
    rim_y = 600 - 3.048 * 110
    dense = ballistic(7.16, 56.0, 120.0, 200)
    truth = crossing_at_y(dense, rim_y, 120.0)
    ball_radius_px = BALL_DIAMETER_M / 2 * 110

    gapped = [s for s in dense if abs(s.y - rim_y) > ball_radius_px]
    assert len(gapped) < len(dense)                      # we really did remove some
    got = crossing_at_y(gapped, rim_y, 120.0)
    assert got is not None
    assert abs(got.x_px - truth.x_px) / 110.0 < 0.01


def test_entry_angle_is_scale_free():
    a = ballistic(7.16, 56.0, FPS, 200, scale_px_per_m=110.0)
    b = ballistic(7.16, 56.0, FPS, 200, scale_px_per_m=260.0)
    rim_a = 600 - 3.048 * 110
    rim_b = 600 - 3.048 * 260
    ca, cb = crossing_at_y(a, rim_a, FPS), crossing_at_y(b, rim_b, FPS)
    assert ca and cb
    assert abs(ca.entry_angle_deg - cb.entry_angle_deg) < 0.2


def test_gravity_recovers_the_scale():
    """A free ball tells you metres per pixel with no reference object."""
    for s_px in (80.0, 110.0, 240.0):
        track = ballistic(7.0, 55.0, FPS, 120, scale_px_per_m=s_px)
        est = scale_from_gravity(track, FPS)
        assert est is not None
        assert abs(est - 1.0 / s_px) / (1.0 / s_px) < 0.02


def test_refine_release_walks_back_to_the_true_release():
    """The lift is not ballistic, so the refit can see where it started.

    Before release the arm carries the ball up the outside of the shoulder
    at roughly constant speed. Back-extrapolating the free-flight parabola
    into that stretch disagrees with where the ball actually was - in x as
    much as in y - and the last frame that still agrees is the release.
    """
    flight = ballistic(7.16, 56.0, FPS, 60)
    x0, y0 = flight[0].x, flight[0].y
    lift_vy_px = 4.37 * 110 / FPS          # ~constant-speed lift, px per frame
    held = []
    for k in range(10, 0, -1):
        u = k / 22.0
        held.append(Sample(-k,
                           x0 + 0.22 * math.sin(math.pi * u) * 110,   # rides outside
                           y0 + lift_vy_px * k,
                           13.2))
    track = held + flight

    # The state machine's first guess is always late - it cannot know the ball
    # has gone until it has visibly separated from the hand.
    got = refine_release_frame(track, candidate=5, fps=FPS)
    assert abs(got - 0) <= 2, f"refined to {got}, expected within 2 frames of 0"
    assert got < 5, "refinement did not improve on the late candidate"


def test_monotonic_descent_rejects_a_rattle():
    down = [Sample(i, 100, 200 + i * 5, 13) for i in range(8)]
    assert is_monotonic_descent(down, 0, 6)
    rattle = down[:3] + [Sample(3 + i, 100, 215 - i * 6, 13) for i in range(5)]
    assert not is_monotonic_descent(rattle, 0, 6)


def test_resample_fills_short_gaps_and_refuses_long_ones():
    a = [Sample(0, 0, 0, 10), Sample(3, 30, 30, 10)]          # 2-frame gap
    assert [s.frame for s in resample_track(a, max_gap=4)] == [0, 1, 2, 3]
    b = [Sample(0, 0, 0, 10), Sample(9, 90, 90, 10)]          # 8-frame gap
    assert [s.frame for s in resample_track(b, max_gap=4)] == [0, 9]


def test_ball_scale_is_robust_to_a_single_blown_up_box():
    bs = BallScale(window=5)
    for r in (13.0, 13.2, 13.1, 40.0, 13.0):                 # one motion-blur outlier
        bs.update(r)
    assert abs(bs.m_per_px - BALL_DIAMETER_M / 26.2) / (BALL_DIAMETER_M / 26.2) < 0.03


def test_rim_scale_from_regulation_width():
    rs = RimScale.from_rim_box(100, 50, 150, 58)
    assert rs.rim_width_px == 50
    assert abs(rs.px_to_m(50) - RIM_INNER_DIAMETER_M) < 1e-9
