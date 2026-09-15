"""Shooting form from pose, and dribble counting from ball motion.

Two things live here because they share an input: the sequence of poses and
ball samples in the run-up to a release.

Arm angle is the headline. It is measured across the whole approach to the
release, not just at the release frame, because the useful number is the
*set point* - how deep the elbow was cocked - and how far it travelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pose import Pose, joint_angle
from .trajectory import Sample


# --------------------------------------------------------------------------
# Shooting arm
# --------------------------------------------------------------------------

def shooting_side(pose: Pose, ball: Sample) -> str:
    """Which arm is doing the work: whichever wrist is nearer the ball."""
    best, side = float("inf"), "right"
    for s in ("left", "right"):
        w = pose.get(f"{s}_wrist")
        if w is None:
            continue
        d = float(np.hypot(ball.x - w[0], ball.y - w[1]))
        if d < best:
            best, side = d, s
    return side


def elbow_angle(pose: Pose, side: str) -> Optional[float]:
    """Shoulder-elbow-wrist. ~90 deg is a well-set shooting pocket;
    ~170 deg is full extension at the top of the follow-through."""
    return joint_angle(pose.get(f"{side}_shoulder"),
                       pose.get(f"{side}_elbow"),
                       pose.get(f"{side}_wrist"))


def knee_angle(pose: Pose, side: str) -> Optional[float]:
    """Hip-knee-ankle. The dip: smaller is a deeper load."""
    return joint_angle(pose.get(f"{side}_hip"),
                       pose.get(f"{side}_knee"),
                       pose.get(f"{side}_ankle"))


@dataclass
class ArmTrace:
    """The shooting arm through the run-up to release."""

    side: str
    frames: List[int]
    elbow_deg: List[float]

    @property
    def at_release(self) -> Optional[float]:
        return self.elbow_deg[-1] if self.elbow_deg else None

    @property
    def set_point_deg(self) -> Optional[float]:
        """Deepest cock of the elbow before it extends - the shooting pocket."""
        return min(self.elbow_deg) if self.elbow_deg else None

    @property
    def extension_deg(self) -> Optional[float]:
        """How far the elbow travelled from set point to release."""
        if len(self.elbow_deg) < 2:
            return None
        return float(max(self.elbow_deg) - min(self.elbow_deg))


def trace_shooting_arm(
    poses: Dict[int, Pose],
    ball_by_frame: Dict[int, Sample],
    release_frame: int,
    lookback: int = 45,
) -> Optional[ArmTrace]:
    """Elbow angle over the frames leading into the release."""
    ref_pose = poses.get(release_frame)
    ref_ball = ball_by_frame.get(release_frame)
    if ref_pose is None or ref_ball is None:
        return None
    side = shooting_side(ref_pose, ref_ball)

    frames, angles = [], []
    for f in range(release_frame - lookback, release_frame + 1):
        p = poses.get(f)
        if p is None:
            continue
        a = elbow_angle(p, side)
        if a is not None:
            frames.append(f)
            angles.append(a)
    if not angles:
        return None
    return ArmTrace(side=side, frames=frames, elbow_deg=angles)


def follow_through_held(
    poses: Dict[int, Pose],
    release_frame: int,
    side: str,
    frames: int = 20,
    min_fraction: float = 0.7,
) -> Optional[bool]:
    """Did the wrist stay above the shoulder after release?

    Cheap proxy for holding the follow-through, and one of the few form cues
    a single camera reads reliably regardless of angle.
    """
    checked = held = 0
    for f in range(release_frame, release_frame + frames + 1):
        p = poses.get(f)
        if p is None:
            continue
        w, sh = p.get(f"{side}_wrist"), p.get(f"{side}_shoulder")
        if w is None or sh is None:
            continue
        checked += 1
        if w[1] < sh[1]:            # y is down, so above means smaller
            held += 1
    if checked == 0:
        return None
    return (held / checked) >= min_fraction


def dip_knee_angle(poses: Dict[int, Pose], release_frame: int, side: str,
                   lookback: int = 45) -> Optional[float]:
    """Deepest knee flexion in the run-up - how much leg went into the shot."""
    vals = []
    for f in range(release_frame - lookback, release_frame + 1):
        p = poses.get(f)
        if p is None:
            continue
        a = knee_angle(p, side)
        if a is not None:
            vals.append(a)
    return min(vals) if vals else None


# --------------------------------------------------------------------------
# Dribbles
# --------------------------------------------------------------------------

def count_dribbles(
    samples: Sequence[Sample],
    poses: Dict[int, Pose],
    start_frame: int,
    end_frame: int,
    fps: float = 120.0,
    floor_y: Optional[float] = None,
    min_bounce_diams: float = 0.8,
    min_gap_s: float = 0.20,
    rebound_window_s: float = 0.25,
    floor_tol_diams: float = 1.5,
) -> Tuple[int, List[int]]:
    """Count bounces between two frames.

    A dribble is a local *maximum* in image y (y points down, so that is the
    bottom of the bounce) that happens below the hip and is followed by the
    ball climbing back up by a real margin.

    Every threshold here is expressed in seconds or ball diameters rather
    than frames or pixels, because the same clip shot at 30 and at 240 fps
    has to produce the same dribble count, and a ball at the far end of the
    court is half the pixel size of one at the near end.
    """
    seg = [s for s in samples if start_frame <= s.frame <= end_frame]
    if len(seg) < 5:
        return 0, []

    look = max(int(rebound_window_s * fps), 3)
    half = max(int(0.05 * fps), 2)
    min_gap = max(int(min_gap_s * fps), 2)

    bounces: List[int] = []
    last = -10**9
    for i in range(half, len(seg) - half):
        cur = seg[i]
        window = seg[max(0, i - half): i + half + 1]
        if cur.y < max(s.y for s in window) - 1e-9:
            continue                       # not the bottom of the arc
        ahead = seg[i + 1: i + 1 + look]
        if not ahead:
            continue
        rebound_px = cur.y - min(s.y for s in ahead)
        if rebound_px < min_bounce_diams * max(cur.radius * 2, 1e-6):
            continue                       # detector jitter, not a bounce
        # A dribble touches the floor. Anything that reverses direction in
        # mid-air is a pickup off a rebound, a gather, or a pass being
        # caught - all of which otherwise look exactly like a bounce.
        if floor_y is not None:
            if cur.y < floor_y - floor_tol_diams * max(cur.radius * 2, 1e-6):
                continue
        else:
            p = poses.get(cur.frame)
            if p is not None:
                hip = p.hip_y()
                if hip is not None and cur.y < hip:
                    continue               # above the waist: not a dribble
        if cur.frame - last < min_gap:
            continue
        bounces.append(cur.frame)
        last = cur.frame
    return len(bounces), bounces


def time_to_release_s(bounce_frames: Sequence[int], release_frame: int, fps: float) -> Optional[float]:
    """Last dribble to release. Quickness, and directly coachable."""
    prior = [f for f in bounce_frames if f <= release_frame]
    if not prior:
        return None
    return (release_frame - prior[-1]) / fps
