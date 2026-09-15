"""Per-shot records and session aggregates.

The session block leads with standard deviations rather than means on
purpose. FG% needs hundreds of attempts to move out of its own noise;
release-angle sigma responds after about thirty. If you want a number that
tells you whether today's session did anything, it is a sigma.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

from .constants import OPTIMAL_ENTRY_ANGLE_DEG
from .events import Outcome, ShotEvent
from .trajectory import Sample


@dataclass
class ShotRecord:
    attempt: int
    outcome: str
    release_frame: int
    resolve_frame: Optional[int] = None

    # flight
    entry_angle_deg: Optional[float] = None
    release_angle_deg: Optional[float] = None
    release_speed_ms: Optional[float] = None
    release_height_m: Optional[float] = None
    apex_height_m: Optional[float] = None
    miss_offset_m: Optional[float] = None

    # form
    arm_side: Optional[str] = None
    elbow_at_release_deg: Optional[float] = None
    elbow_set_point_deg: Optional[float] = None
    elbow_extension_deg: Optional[float] = None
    knee_dip_deg: Optional[float] = None
    follow_through_held: Optional[bool] = None

    # approach
    dribbles_before: Optional[int] = None
    last_dribble_to_release_s: Optional[float] = None

    # placement
    court_x_m: Optional[float] = None
    court_y_m: Optional[float] = None
    distance_m: Optional[float] = None

    # the arc itself, kept for the trajectory panel
    arc_px: List[tuple] = field(default_factory=list)

    @property
    def made(self) -> bool:
        return self.outcome == Outcome.MAKE.value


def _vals(records: List[ShotRecord], attr: str) -> List[float]:
    return [getattr(r, attr) for r in records if getattr(r, attr) is not None]


@dataclass
class SessionStats:
    attempts: int = 0
    makes: int = 0

    @property
    def fg_pct(self) -> Optional[float]:
        return 100.0 * self.makes / self.attempts if self.attempts else None


def summarise(records: List[ShotRecord]) -> Dict[str, Optional[float]]:
    """Session-level numbers, sigmas first."""
    scored = [r for r in records if r.outcome in (Outcome.MAKE.value, Outcome.MISS.value)]
    makes = sum(1 for r in scored if r.made)

    entry = _vals(scored, "entry_angle_deg")
    rel_ang = _vals(scored, "release_angle_deg")
    rel_spd = _vals(scored, "release_speed_ms")
    elbow = _vals(scored, "elbow_at_release_deg")
    dribbles = _vals(scored, "dribbles_before")

    lo, hi = OPTIMAL_ENTRY_ANGLE_DEG
    in_band = [a for a in entry if lo <= a <= hi]

    misses = [r.miss_offset_m for r in scored
              if not r.made and r.miss_offset_m is not None]

    def sd(v: List[float]) -> Optional[float]:
        return float(pstdev(v)) if len(v) > 1 else None

    def mn(v: List[float]) -> Optional[float]:
        return float(mean(v)) if v else None

    return {
        "attempts": len(scored),
        "makes": makes,
        "fg_pct": 100.0 * makes / len(scored) if scored else None,
        # consistency - the point of the whole exercise
        "release_angle_sd": sd(rel_ang),
        "release_speed_sd": sd(rel_spd),
        "entry_angle_sd": sd(entry),
        "elbow_at_release_sd": sd(elbow),
        # central tendency
        "entry_angle_mean": mn(entry),
        "release_angle_mean": mn(rel_ang),
        "release_speed_mean": mn(rel_spd),
        "elbow_at_release_mean": mn(elbow),
        "dribbles_mean": mn(dribbles),
        "entry_in_band_pct": 100.0 * len(in_band) / len(entry) if entry else None,
        # directional bias: + means the ball tends right of centre
        "miss_bias_m": mn(misses),
    }


def to_csv(records: List[ShotRecord], path: str | Path) -> None:
    cols = [f.name for f in fields(ShotRecord) if f.name != "arc_px"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = {k: v for k, v in asdict(r).items() if k in cols}
            w.writerow(row)


def build_record(event: ShotEvent, rim_scale, ball_m_per_px: Optional[float],
                 floor_y_px: Optional[float]) -> ShotRecord:
    """Assemble the flight half of a record. Form and approach are filled in
    by the pipeline, which is the only place that holds the pose history."""
    rec = ShotRecord(
        attempt=event.attempt,
        outcome=event.outcome.value,
        release_frame=event.release_frame,
        resolve_frame=event.resolve_frame,
        miss_offset_m=event.miss_offset_m,
        arc_px=[(s.frame, s.x, s.y) for s in event.samples],
    )
    if event.crossing is not None:
        rec.entry_angle_deg = event.crossing.entry_angle_deg
    if event.samples and floor_y_px is not None and ball_m_per_px is not None:
        apex_y = min(s.y for s in event.samples)
        rec.apex_height_m = (floor_y_px - apex_y) * ball_m_per_px
    return rec


def dominant_sample(samples: List[Sample]) -> Optional[Sample]:
    return samples[0] if samples else None
