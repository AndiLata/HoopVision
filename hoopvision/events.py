"""The shot-event state machine, and the make/miss call.

Deliberately not a classifier. Every transition is a threshold on a quantity
already computed, which means every rejected "shot" can be explained in one
sentence - and when the thing mislabels a pass as an attempt, you can see
exactly which threshold let it through.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .constants import BALL_DIAMETER_M, FITS_THROUGH_HALF_WIDTH_M
from .geometry import RimScale
from .trajectory import (
    Crossing,
    Sample,
    apex_index,
    crossing_at_y,
    is_monotonic_descent,
    linear_velocity,
    refine_release_frame,
)


class State(Enum):
    IDLE = "idle"
    LOADED = "loaded"
    FLIGHT = "flight"


class Outcome(Enum):
    MAKE = "make"
    MISS = "miss"
    UNRESOLVED = "unresolved"     # airball, rebound, ball left frame


@dataclass
class ShotEvent:
    attempt: int
    release_frame: int
    resolve_frame: Optional[int]
    outcome: Outcome
    samples: List[Sample] = field(default_factory=list)
    crossing: Optional[Crossing] = None
    miss_offset_m: Optional[float] = None    # signed: + is right of centre
    rejected_reason: Optional[str] = None


class ShotDetector:
    """Incremental. Feed it one frame at a time; it yields completed shots.

    Streaming rather than batch so the same code path serves a live demo and
    an offline render.
    """

    def __init__(
        self,
        fps: float,
        rim: RimScale,
        loaded_frames: int = 3,
        load_dist_diams: float = 1.5,
        release_dist_diams: float = 2.5,
        timeout_s: float = 3.0,
        descent_confirm: int = 6,
        max_offset_rim_widths: float = 3.0,
    ):
        self.fps = fps
        self.rim = rim
        self.loaded_frames = loaded_frames
        self.load_dist_diams = load_dist_diams
        self.release_dist_diams = release_dist_diams
        self.timeout_frames = int(timeout_s * fps)
        self.descent_confirm = descent_confirm
        self.max_offset_rim_widths = max_offset_rim_widths

        self.state = State.IDLE
        self.attempts = 0
        self._near_count = 0
        self._buffer: List[Sample] = []
        self._release_frame: Optional[int] = None
        self._prev: Optional[Sample] = None
        # (sample, was_in_hand) for the last ~1s, used to backtrack the release
        self._history: deque = deque(maxlen=int(fps))

    # -- helpers ---------------------------------------------------------
    def _ball_diameter_px(self, s: Sample) -> float:
        return max(s.radius * 2, 1e-6)

    def _wrist_distance(self, s: Sample, wrists: Optional[List[Tuple[float, float]]]) -> Optional[float]:
        if not wrists:
            return None
        return min(float(np.hypot(s.x - wx, s.y - wy)) for wx, wy in wrists)

    def _ascending(self, s: Sample) -> bool:
        if self._prev is None:
            return False
        return s.y < self._prev.y          # y is down, so smaller y is higher

    # -- main ------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        sample: Optional[Sample],
        wrists: Optional[List[Tuple[float, float]]] = None,
        hip_y: Optional[float] = None,
    ) -> Optional[ShotEvent]:
        """Feed one frame. `hip_y` gates dribbles out of the release test:
        a ball that leaves the hand below the waist is a bounce, not a shot."""
        event: Optional[ShotEvent] = None

        if self.state is State.FLIGHT:
            if sample is not None:
                self._buffer.append(sample)
            if self._release_frame is not None and \
                    frame_idx - self._release_frame > self.timeout_frames:
                event = self._resolve(frame_idx, timed_out=True)
            else:
                crossing = crossing_at_y(self._buffer, self.rim.rim_cy, self.fps)
                if crossing is not None and self._enough_after_crossing(crossing):
                    event = self._resolve(frame_idx, timed_out=False)
            self._prev = sample or self._prev
            return event

        if sample is None:
            self._prev = None
            return None

        d = self._wrist_distance(sample, wrists)
        diam = self._ball_diameter_px(sample)
        near = d is not None and d <= self.load_dist_diams * diam
        self._history.append((sample, near))

        if self.state is State.IDLE:
            in_hand = near or (d is None and not self._ascending(sample))
            self._near_count = self._near_count + 1 if in_hand else 0
            if self._near_count >= self.loaded_frames:
                self.state = State.LOADED

        elif self.state is State.LOADED:
            separated = d is None or d > self.release_dist_diams * diam
            above_waist = hip_y is None or sample.y < hip_y
            if separated and self._ascending(sample) and above_waist:
                self.state = State.FLIGHT
                self._release_frame = self._backtrack_release(sample)
                # Keep a little pre-release history so the ballistic refit
                # below has something to walk back into.
                keep = self._release_frame - 15
                self._buffer = [s for s, _ in self._history if s.frame >= keep] or [sample]
                self._near_count = 0

        self._prev = sample
        return event

    def _backtrack_release(self, current: Sample) -> int:
        """The true release is the last frame the ball was still in the hand.

        By the time separation exceeds 2.5 diameters the ball has been gone
        for several frames, and reading release speed from there would report
        the ball already decelerating under gravity.
        """
        for s, was_near in reversed(self._history):
            if was_near:
                return s.frame
        return self._history[0][0].frame if self._history else current.frame

    def _enough_after_crossing(self, crossing: Crossing) -> bool:
        after = [s for s in self._buffer if s.frame > crossing.frame]
        return len(after) >= self.descent_confirm

    # -- resolution ------------------------------------------------------
    def _resolve(self, frame_idx: int, timed_out: bool) -> ShotEvent:
        self.attempts += 1
        samples = list(self._buffer)
        if self._release_frame is not None:
            self._release_frame = refine_release_frame(
                samples, self._release_frame, self.fps)
        crossing = crossing_at_y(samples, self.rim.rim_cy, self.fps)

        outcome = Outcome.UNRESOLVED
        offset_m: Optional[float] = None
        reason: Optional[str] = None

        if crossing is None:
            reason = "timed out with no rim-plane crossing (airball or rebound)"
        else:
            offset_px = crossing.x_px - self.rim.rim_cx
            offset_m = self.rim.px_to_m(offset_px)

            if abs(offset_px) > self.max_offset_rim_widths * self.rim.rim_width_px:
                outcome = Outcome.UNRESOLVED
                reason = "crossing too far from the rim to be an attempt at it"
            else:
                idx = next((i for i, s in enumerate(samples) if s.frame >= crossing.frame),
                           len(samples) - 1)
                fits = abs(offset_m) <= FITS_THROUGH_HALF_WIDTH_M
                kept_falling = is_monotonic_descent(samples, idx, self.descent_confirm)
                if fits and kept_falling:
                    outcome = Outcome.MAKE
                else:
                    outcome = Outcome.MISS
                    reason = ("rattled out - crossing was inside but the ball came back up"
                              if fits else "crossing outside the ball's clearance")

        # An attempt must have actually gone up at the rim.
        ai = apex_index(samples)
        if ai is not None and samples[ai].y > self.rim.rim_cy:
            outcome = Outcome.UNRESOLVED
            reason = "apex never reached rim height - not a shot"

        event = ShotEvent(
            attempt=self.attempts,
            release_frame=self._release_frame if self._release_frame is not None else frame_idx,
            resolve_frame=int(crossing.frame) if crossing else frame_idx,
            outcome=outcome,
            samples=samples,
            crossing=crossing,
            miss_offset_m=offset_m,
            rejected_reason=reason,
        )
        if outcome is Outcome.UNRESOLVED:
            self.attempts -= 1          # do not count non-attempts

        self.state = State.IDLE
        self._buffer = []
        self._release_frame = None
        self._near_count = 0
        return event


def release_kinematics(samples: List[Sample], release_frame: int, fps: float,
                       m_per_px: float, floor_y_px: Optional[float] = None) -> dict:
    """Speed, angle and height at the moment the ball leaves the hand.

    The angle needs no scale (isotropic), the speed and height do - and they
    should use the BALL ruler, not the rim ruler, because the shooter is
    nowhere near the rim's depth plane.
    """
    window = [s for s in samples if release_frame <= s.frame <= release_frame + 5]
    if len(window) < 2:
        return {}
    vx, vy = linear_velocity(window, fps)
    speed_px_s = float(np.hypot(vx, vy))
    out = {
        "release_speed_ms": speed_px_s * m_per_px,
        "release_angle_deg": float(np.degrees(np.arctan2(-vy, abs(vx)))),
    }
    if floor_y_px is not None:
        out["release_height_m"] = (floor_y_px - window[0].y) * m_per_px
    return out
