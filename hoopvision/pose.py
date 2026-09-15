"""Pose keypoints, COCO-17 convention.

Nothing here needs labelling - the pretrained pose model handles people out
of the box. That is why `person` is deliberately not one of the classes you
annotate in Stage 1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

# COCO-17 keypoint order
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


@dataclass
class Pose:
    """17 keypoints as (x, y, confidence)."""

    pts: np.ndarray          # (17, 3)

    def get(self, name: str, min_conf: float = 0.3) -> Optional[Tuple[float, float]]:
        i = KP[name]
        x, y, c = self.pts[i]
        return (float(x), float(y)) if c >= min_conf else None

    def wrists(self, min_conf: float = 0.3) -> List[Tuple[float, float]]:
        return [p for p in (self.get("left_wrist", min_conf),
                            self.get("right_wrist", min_conf)) if p]

    def floor_y(self, min_conf: float = 0.3) -> Optional[float]:
        """Ankle midpoint - the best available stand-in for the floor plane."""
        a = self.get("left_ankle", min_conf)
        b = self.get("right_ankle", min_conf)
        ys = [p[1] for p in (a, b) if p]
        return float(np.mean(ys)) if ys else None

    def hip_y(self, min_conf: float = 0.3) -> Optional[float]:
        ys = [p[1] for p in (self.get("left_hip", min_conf),
                             self.get("right_hip", min_conf)) if p]
        return float(np.mean(ys)) if ys else None

    def to_list(self) -> list:
        return self.pts.tolist()

    @staticmethod
    def from_list(v) -> "Pose":
        return Pose(np.asarray(v, dtype=float).reshape(17, 3))


class PoseEstimator(Protocol):
    def estimate(self, frame_idx: int, image) -> Optional[Pose]: ...


class ReplayPose:
    """Poses from a JSON sidecar - the synthetic clip's ground truth."""

    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text())
        self._frames: Dict[int, Pose] = {
            int(k): Pose.from_list(v) for k, v in raw["poses"].items()
        }

    def estimate(self, frame_idx: int, image=None) -> Optional[Pose]:
        return self._frames.get(frame_idx)


class YoloPose:
    """Pretrained pose model. Lazy import so tests run without torch."""

    def __init__(self, weights: str = "yolo26n-pose.pt", device: str = "mps", conf: float = 0.4):
        from ultralytics import YOLO  # noqa: PLC0415

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def estimate(self, frame_idx: int, image) -> Optional[Pose]:
        res = self.model.predict(image, device=self.device, conf=self.conf, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            return None
        # Largest person box - the shooter, on a clip filmed for this purpose.
        areas = [(b.xywh[0][2] * b.xywh[0][3]).item() for b in res.boxes]
        i = int(np.argmax(areas))
        kp = res.keypoints.data[i].cpu().numpy()
        return Pose(kp.reshape(17, 3))


def joint_angle(a: Optional[Tuple[float, float]],
                b: Optional[Tuple[float, float]],
                c: Optional[Tuple[float, float]]) -> Optional[float]:
    """Interior angle at b, in degrees. Returns None if any point is missing.

    Scale-free and rotation-free, so it needs no calibration - which is what
    makes joint angles the most trustworthy pose-derived metric available
    from a single camera.
    """
    if a is None or b is None or c is None:
        return None
    v1 = np.array(a, dtype=float) - np.array(b, dtype=float)
    v2 = np.array(c, dtype=float) - np.array(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
