"""End-to-end orchestration.

Runs with ReplayDetector on the synthetic clip (no model, no torch) or with
YoloDetector on real footage. Same code path either way, which is the point:
every stage downstream of detection is testable before you label a frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detect import Detector, FrameDetections
from .events import Outcome, ShotDetector, ShotEvent, release_kinematics
from .form import (
    count_dribbles,
    dip_knee_angle,
    follow_through_held,
    time_to_release_s,
    trace_shooting_arm,
)
from .geometry import BallScale, CourtHomography, RimScale, rim_stable
from .metrics import ShotRecord, build_record, summarise, to_csv
from .pose import Pose, PoseEstimator
from .render.dashboard import annotate_video, compose
from .track import BallTracker
from .trajectory import Sample


@dataclass
class PipelineConfig:
    fps: float = 120.0
    trail_len: int = 24
    out_width: int = 1280
    rim_lock_frames: int = 30
    rim_drift_tol_px: float = 3.0
    max_ghost_arcs: int = 6


@dataclass
class Result:
    records: List[ShotRecord] = field(default_factory=list)
    session: Dict[str, Optional[float]] = field(default_factory=dict)
    frames_processed: int = 0
    camera_moved: bool = False


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        pose_estimator: Optional[PoseEstimator] = None,
        config: Optional[PipelineConfig] = None,
        homography: Optional[CourtHomography] = None,
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.cfg = config or PipelineConfig()
        self.homography = homography

        self.rim: Optional[RimScale] = None
        self.ball_scale = BallScale()
        self.shots: Optional[ShotDetector] = None
        self.tracker = BallTracker(self.cfg.fps)

        self.all_samples: List[Sample] = []
        self.poses: Dict[int, Pose] = {}
        self.records: List[ShotRecord] = []
        self._rim_boxes: List[Tuple[float, float, float, float]] = []
        self._last_resolve_frame = 0
        self._floor_y: Optional[float] = None

    # -- per frame -------------------------------------------------------
    def _lock_rim(self, det: FrameDetections) -> None:
        if det.rim is None:
            return
        box = (det.rim.x1, det.rim.y1, det.rim.x2, det.rim.y2)
        self._rim_boxes.append(box)
        if self.rim is None and len(self._rim_boxes) >= self.cfg.rim_lock_frames:
            med = np.median(np.asarray(self._rim_boxes), axis=0)
            self.rim = RimScale.from_rim_box(*med)
            self.shots = ShotDetector(self.cfg.fps, self.rim)

    def _finish_shot(self, event: ShotEvent) -> Optional[ShotRecord]:
        if event.outcome is Outcome.UNRESOLVED:
            self._last_resolve_frame = event.resolve_frame or self._last_resolve_frame
            return None

        rec = build_record(event, self.rim, self.ball_scale.m_per_px, self._floor_y)

        m_per_px = self.ball_scale.m_per_px
        if m_per_px:
            kin = release_kinematics(event.samples, event.release_frame,
                                     self.cfg.fps, m_per_px, self._floor_y)
            rec.release_speed_ms = kin.get("release_speed_ms")
            rec.release_angle_deg = kin.get("release_angle_deg")
            rec.release_height_m = kin.get("release_height_m")

        ball_by_frame = {s.frame: s for s in self.all_samples}
        trace = trace_shooting_arm(self.poses, ball_by_frame, event.release_frame)
        if trace is not None:
            rec.arm_side = trace.side
            rec.elbow_at_release_deg = trace.at_release
            rec.elbow_set_point_deg = trace.set_point_deg
            rec.elbow_extension_deg = trace.extension_deg
            rec.knee_dip_deg = dip_knee_angle(self.poses, event.release_frame, trace.side)
            rec.follow_through_held = follow_through_held(
                self.poses, event.release_frame, trace.side)

        n, bounce_frames = count_dribbles(
            self.all_samples, self.poses, self._last_resolve_frame,
            event.release_frame, fps=self.cfg.fps, floor_y=self._floor_y)
        rec.dribbles_before = n
        rec.last_dribble_to_release_s = time_to_release_s(
            bounce_frames, event.release_frame, self.cfg.fps)

        if self.homography is not None and self.poses.get(event.release_frame):
            p = self.poses[event.release_frame]
            fy = p.floor_y()
            la, ra = p.get("left_ankle"), p.get("right_ankle")
            xs = [q[0] for q in (la, ra) if q]
            if fy is not None and xs:
                cx, cy = self.homography.to_court(float(np.mean(xs)), fy)
                rec.court_x_m, rec.court_y_m = cx, cy
                rec.distance_m = float(np.hypot(cx, cy))

        self._last_resolve_frame = event.resolve_frame or event.release_frame
        self.records.append(rec)
        return rec

    def step(self, frame_idx: int, image) -> Tuple[Optional[FrameDetections], Optional[ShotRecord]]:
        det = self.detector.detect(frame_idx, image)
        self._lock_rim(det)

        pose = self.pose_estimator.estimate(frame_idx, image) if self.pose_estimator else None
        if pose is not None:
            self.poses[frame_idx] = pose
            fy = pose.floor_y()
            if fy is not None:
                self._floor_y = fy

        sample = self.tracker.update(frame_idx, det.ball)
        if sample is not None:
            self.all_samples.append(sample)
            self.ball_scale.update(sample.radius)

        rec = None
        if self.shots is not None:
            wrists = pose.wrists() if pose else None
            hip_y = pose.hip_y() if pose else None
            event = self.shots.update(frame_idx, sample, wrists, hip_y)
            if event is not None:
                rec = self._finish_shot(event)
        return det, rec

    # -- rendering -------------------------------------------------------
    def render_frame(self, image, frame_idx: int, det: Optional[FrameDetections]) -> np.ndarray:
        trail = [(s.x, s.y) for s in self.all_samples[-self.cfg.trail_len:]]
        rim_box = None
        if det and det.rim:
            rim_box = (det.rim.x1, det.rim.y1, det.rim.x2, det.rim.y2)
        vid = annotate_video(image, self.poses.get(frame_idx), trail, rim_box,
                             badge=f"frame {frame_idx}")

        sess = summarise(self.records)
        last = self.records[-1] if self.records else None
        ghosts = [[(x, y) for _, x, y in r.arc_px]
                  for r in self.records[-self.cfg.max_ghost_arcs - 1:-1]]
        current = [(x, y) for _, x, y in last.arc_px] if last else []

        arm_frames: Sequence[int] = []
        arm_angles: Sequence[float] = []
        if last is not None:
            ball_by_frame = {s.frame: s for s in self.all_samples}
            tr = trace_shooting_arm(self.poses, ball_by_frame, last.release_frame)
            if tr:
                arm_frames, arm_angles = tr.frames, tr.elbow_deg

        chart = [(r.court_x_m, r.court_y_m, r.made, r.attempt)
                 for r in self.records if r.court_x_m is not None]

        rim_tuple = (self.rim.rim_cx, self.rim.rim_cy, self.rim.rim_width_px) if self.rim else None
        return compose(
            vid,
            makes=sum(1 for r in self.records if r.made),
            attempts=len(self.records),
            arm_at_release=last.elbow_at_release_deg if last else None,
            arm_set_point=last.elbow_set_point_deg if last else None,
            arm_sd=sess.get("elbow_at_release_sd"),
            dribbles=last.dribbles_before if last else None,
            dribbles_mean=sess.get("dribbles_mean"),
            to_release_s=last.last_dribble_to_release_s if last else None,
            entry=last.entry_angle_deg if last else None,
            entry_mean=sess.get("entry_angle_mean"),
            entry_in_band=sess.get("entry_in_band_pct"),
            current_arc=current,
            ghost_arcs=ghosts,
            rim=rim_tuple,
            floor_y=self._floor_y,
            made=last.made if last else None,
            arm_frames=arm_frames,
            arm_angles=arm_angles,
            chart=chart,
            width=self.cfg.out_width,
        )

    # -- driver ----------------------------------------------------------
    def run_video(self, src: str | Path, out_path: Optional[str | Path] = None,
                  csv_path: Optional[str | Path] = None) -> Result:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg.fps
        self.cfg.fps = fps

        writer = None
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                det, _ = self.step(idx, frame)
                if out_path:
                    canvas = self.render_frame(frame, idx, det)
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(canvas)
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        moved = not rim_stable(self._rim_boxes, self.cfg.rim_drift_tol_px)
        if csv_path:
            to_csv(self.records, csv_path)
        return Result(self.records, summarise(self.records), idx, moved)
